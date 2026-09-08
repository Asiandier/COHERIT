"""Independent dense oracles for the matrix-free information correction."""
import importlib
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg as sla

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
REML = importlib.import_module(f"{ROOT.name}.reml")
INFO = importlib.import_module(f"{ROOT.name}.sparse_information")
SPARSE = importlib.import_module(f"{ROOT.name}.run_sparse_reml_pipeline")


def fixture():
    rng = np.random.default_rng(2617)
    n = 36
    x = rng.normal(size=(n, 5)).astype(np.float32)
    c = np.column_stack([np.ones(n), rng.normal(size=n)]).astype(np.float32)
    kernels = []
    for m in (18, 24):
        a = rng.normal(size=(n, m))
        kernels.append(a @ a.T/m)
    g = x @ np.array([.2, -.1, .15, .07, -.08])
    residual = rng.normal(size=n) + .2*x[:, 0]
    return x, c, kernels, g, residual


def dense(theta, kernels, c, residual, z):
    n = len(residual)
    matrices = [*kernels, np.eye(n)]
    v = sum(a*b for a, b in zip(theta, matrices))
    vi = np.linalg.inv(v)
    hc = vi @ c
    cg = c.T @ hc
    gamma = np.linalg.solve(cg, c.T @ vi @ residual)
    p = vi - hc @ np.linalg.solve(cg, hc.T)
    base_det = np.linalg.slogdet(v)[1] + np.linalg.slogdet(cg)[1]
    pr = p @ residual
    if z.shape[1]:
        gram = z.T @ p @ z
        sigma = z @ np.linalg.solve(gram, z.T)
        ps = p-p @ sigma @ p
        info_det = np.linalg.slogdet(gram)[1]
    else:
        sigma, ps, info_det = np.zeros_like(p), p, 0.
    ll = -.5*(base_det+info_det+residual @ pr)/n
    score = np.array([.5*(pr @ d @ pr-np.trace(ps @ d))/n for d in matrices])
    return ll, score, np.trace(sigma), gamma, p, ps


def context(theta, x, c, kernels, residual, info):
    n = len(residual)
    # Exact trace probes; SLQ is independently replaced by a dense logdet below.
    probes = jnp.eye(n)*np.sqrt(n)
    kj = [jnp.asarray(k, dtype=jnp.float32) for k in kernels]
    rhs = jnp.column_stack([c, residual, probes])
    ctx = REML.REMLContext(
        n=n, G=len(kernels), E=1,
        K_mvs=tuple(lambda b, k=k: k @ b for k in kj),
        weighted_hv=None, stacked_kv=None,
        diag_stack=jnp.stack([jnp.diag(k) for k in kj]),
        residual_diag_stack=None, xmat=jnp.asarray(c), y=jnp.asarray(residual, dtype=jnp.float32),
        rhs_const=rhs, y_col=c.shape[1], rand_stop=rhs.shape[1],
        n_XyZ_cols=rhs.shape[1], R_rand=n,
        kvrand_stack=jnp.stack([k @ probes for k in kj]), precond_conf=None,
        mean_information=info,
    )
    return ctx


def evaluate(ctx, theta):
    return REML._eval_once(
        ctx, jnp.asarray(theta, dtype=jnp.float32), jnp.zeros_like(ctx.rhs_const),
        key_slq=jax.random.PRNGKey(15), minq_tol=1e-5, maxiter=200,
        precond_eps=1e-7, slq_samples=36, slq_m=36,
    )


def exact_logdet_and_score(ctx, hv, theta, *args, **kwargs):
    identity = jnp.eye(ctx.n)
    covariance = np.asarray(hv(identity), dtype=float)
    inverse = np.linalg.inv(covariance)
    matrices = [np.asarray(operator(identity), dtype=float) for operator in ctx.K_mvs]
    matrices.append(np.eye(ctx.n))
    return (jnp.asarray(np.linalg.slogdet(covariance)[1]),
            jnp.asarray([np.trace(inverse @ d) for d in matrices]))


def test_consistent_score_with_warm_pcg_solutions():
    """A loose solve used to create a false likelihood plateau on warm trials."""
    x, c, kernels, _, residual = fixture()
    theta = jnp.asarray([.27, .21, .52], dtype=jnp.float32)
    ctx = context(theta, x, c, kernels, residual, None)
    def trial(t, warm=None):
        return REML._eval_once(
            ctx, t, jnp.zeros_like(ctx.rhs_const) if warm is None else warm,
            key_slq=jax.random.PRNGKey(15), minq_tol=1e-5, maxiter=200,
            precond_eps=1e-7, slq_samples=8, slq_m=10,
            warm_ready=warm is not None, compute_traces=False,
        )
    with jax.default_matmul_precision("highest"):
        center = trial(theta)
        differences = []
        for delta in jnp.eye(3)*.001:
            plus, minus = trial(theta+delta, center[4]), trial(theta-delta, center[4])
            differences.append(float((plus[0]-minus[0])/.002))
        np.testing.assert_allclose(center[1], differences, rtol=.005, atol=5e-4)


@pytest.mark.parametrize("kind", ["regular", "duplicate", "empty", "zero_component"])
def test_corrected_objective_score_trace_and_h2_match_dense(monkeypatch, kind):
    x, c, kernels, g, residual = fixture()
    if kind == "duplicate":
        x = np.column_stack([x, x[:, 0], 2*x[:, 1]])
    if kind == "empty":
        x = x[:, :0]
        g = np.zeros_like(g)
    theta = np.array([0.0 if kind == "zero_component" else .27, .21, .52])
    info = INFO.SparseMeanInformation(x, c, batch_size=2)
    ctx = context(theta, x, c, kernels, residual, info)
    monkeypatch.setattr(REML, "_multi_slq_logdet_and_score", exact_logdet_and_score)
    ll, score, ai, *_ = evaluate(ctx, theta)
    expected, gradient, trace, gamma, p, ps = dense(theta, kernels, c, residual, info.basis)
    np.testing.assert_allclose(ll, expected, atol=8e-6, rtol=2e-5)
    np.testing.assert_allclose(score, gradient, atol=2e-5, rtol=2e-4)
    np.testing.assert_allclose(info._state["trace"], trace, atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(info._state["w"] @ info._state["w"].T, p-ps, atol=8e-6)
    assert np.linalg.eigvalsh(np.asarray(ai.mat))[0] >= -1e-6
    # It is essential NOT to replace the Lasso residual's P by P_[C,Z].
    if x.shape[1]:
        assert abs(residual @ (p-ps) @ residual) > .1
    v = sum(a*b for a, b in zip(theta, [*kernels, np.eye(len(g))]))
    state, r, gh = info.statistics(theta.astype(np.float32), lambda b:jnp.asarray(v) @ b,
                                  None, residual, tol=1e-5, maxiter=200)
    np.testing.assert_allclose(gh, gamma, atol=2e-5)
    bg = sum(a*np.trace(k)/len(g) for a, k in zip(theta[:-1], kernels))
    h2, q = SPARSE._outer_coherit_h2_from_fitted_sparse_mean(
        g, r, background_genetic_variance=bg, residual_variance=theta[-1],
        mean_uncertainty_trace=state["trace"],
    )
    q_ref = (g @ g + 2*g @ (residual-c @ gamma)-trace)/len(g)
    assert q == pytest.approx(q_ref, abs=2e-6)
    assert h2 == pytest.approx((q_ref+bg)/(q_ref+bg+theta[-1]), abs=2e-6)


def test_corrected_score_is_derivative_and_empty_span_is_ordinary(monkeypatch):
    x, c, kernels, g, residual = fixture()
    theta = np.array([.27, .21, .52])
    info = INFO.SparseMeanInformation(x, c)
    ctx = context(theta, x, c, kernels, residual, info)
    monkeypatch.setattr(REML, "_multi_slq_logdet_and_score", exact_logdet_and_score)
    ll, score, *_ = evaluate(ctx, theta)
    delta = np.eye(3)*.002
    fd = [(float(evaluate(ctx, theta+d)[0])-float(evaluate(ctx, theta-d)[0]))/.004
          for d in delta]
    np.testing.assert_allclose(score, fd, atol=1e-4, rtol=.003)
    ctx.mean_information = INFO.SparseMeanInformation(x[:, :0], c)
    empty = evaluate(ctx, theta)
    ctx.mean_information = None
    ordinary = evaluate(ctx, theta)
    np.testing.assert_array_equal(empty[0], ordinary[0])
    np.testing.assert_array_equal(empty[1], ordinary[1])
    np.testing.assert_array_equal(empty[2].mat, ordinary[2].mat)


def test_basis_invariance_cache_invalidation_and_covariate_shift():
    x, c, kernels, g, residual = fixture()
    theta = np.array([.27, .21, .52], dtype=np.float32)
    rng = np.random.default_rng(777)
    change = rng.normal(size=(5, 5)) + 3*np.eye(5)
    values = []
    for active in (x, x @ change):
        info = INFO.SparseMeanInformation(active, c, batch_size=2)
        v = sum(a*b for a,b in zip(theta, [*kernels, np.eye(len(g))]))
        hv = lambda b: jnp.asarray(v) @ b
        state, r, _ = info.statistics(theta, hv, None, residual, tol=1e-5, maxiter=200)
        repeated = info.evaluate(theta, hv, None, None, tol=1e-5, maxiter=200)
        assert repeated is state
        _, shifted, _ = info.statistics(theta, hv, None, residual+c @ [1., -2.],
                                        tol=1e-5, maxiter=200)
        np.testing.assert_allclose(r, shifted, atol=5e-5)
        values.append([state["trace"], state["logdet"]])
        other = theta + np.array([.02, -.01, .03], dtype=np.float32)
        other_v = sum(a*b for a,b in zip(other, [*kernels, np.eye(len(g))]))
        new, _, _ = info.statistics(other, lambda b: jnp.asarray(other_v) @ b,
                                    None, residual, tol=1e-5, maxiter=200)
        assert new is not state
        assert not np.isclose(new["trace"], state["trace"])
    np.testing.assert_allclose(values[0], values[1], atol=3e-5)


def test_reject_unidentified_mean_and_unconverged_solve():
    x, c, *_ = fixture()
    with pytest.raises(ValueError, match="intersects"):
        INFO.SparseMeanInformation(np.column_stack([x, c[:, 0]]), c)
    with pytest.raises(ValueError, match="contrast dimension"):
        INFO.SparseMeanInformation(np.eye(36), c)
    rng = np.random.default_rng(92)
    matrix = jnp.diag(jnp.linspace(.01, 3., 36))
    with pytest.raises(FloatingPointError, match="PCG failed"):
        INFO.solve_columns(lambda b: matrix @ b, rng.normal(size=(36, 3)), None,
                           tol=1e-5, maxiter=1)


def test_information_warm_hit_uses_only_pcgs_own_true_residual_check():
    rhs = np.arange(24, dtype=np.float32).reshape(8, 3)
    calls = []

    def hv(value):
        calls.append(value.shape)
        return value

    result = INFO.solve_columns(hv, rhs, None, tol=1e-5, maxiter=100,
                                batch_size=2, warm=rhs)
    np.testing.assert_array_equal(result, rhs)
    assert calls == [(8, 2), (8, 1)]


def test_information_statistics_reuse_and_invalidate_response_and_covariance(monkeypatch):
    x, c, kernels, g, residual = fixture()
    theta = np.array([.27, .21, .52], dtype=np.float32)
    info = INFO.SparseMeanInformation(x, c)
    original = INFO.solve_columns
    columns = []

    def counted(hv, rhs, *args, **kwargs):
        columns.append(rhs.shape[1])
        return original(hv, rhs, *args, **kwargs)

    monkeypatch.setattr(INFO, "solve_columns", counted)

    def statistics(t, r):
        v = sum(a*b for a, b in zip(t, [*kernels, np.eye(len(r))]))
        return info.statistics(t, lambda b:jnp.asarray(v, dtype=jnp.float32) @ b,
                               None, r, tol=1e-5, maxiter=200)

    first = statistics(theta, residual)
    assert columns == [3, 5]
    repeated = statistics(theta, residual.copy())
    assert columns == [3, 5]  # No new solves at all for the same paired state.
    np.testing.assert_array_equal(first[1], repeated[1])
    shifted = residual + c @ np.array([1., -.2])
    result = statistics(theta, shifted)
    assert columns == [3, 5, 1]  # Changed alpha/response does not invalidate C.
    np.testing.assert_allclose(result[1], first[1], atol=3e-5)
    other = theta + np.array([.02, -.01, .03], dtype=np.float32)
    result = statistics(other, shifted)
    assert columns == [3, 5, 1, 3, 5]
    expected = dense(other, kernels, c, shifted, info.basis)
    np.testing.assert_allclose(result[2], expected[3], atol=3e-5)
    assert result[0]["trace"] == pytest.approx(expected[2], rel=3e-5)


def test_statistics_reuse_covar_solve_supplied_by_reml(monkeypatch):
    x, c, kernels, _, residual = fixture()
    theta = np.array([.27, .21, .52], dtype=np.float32)
    v = sum(a*b for a, b in zip(theta, [*kernels, np.eye(len(residual))]))
    hv = lambda b: jnp.asarray(v, dtype=jnp.float32) @ b
    info = INFO.SparseMeanInformation(x, c)
    info.evaluate(theta, hv, None, np.linalg.solve(v, c), tol=1e-5, maxiter=200)
    original = INFO.solve_columns
    columns = []

    def counted(hv, rhs, *args, **kwargs):
        columns.append(rhs.shape[1])
        return original(hv, rhs, *args, **kwargs)

    monkeypatch.setattr(INFO, "solve_columns", counted)
    info.statistics(theta, hv, None, residual, tol=1e-5, maxiter=200)
    assert columns == [1]


def test_response_cache_keeps_the_tolerance_of_the_actual_solve(monkeypatch):
    x, c, kernels, _, residual = fixture()
    theta = np.array([.27, .21, .52], dtype=np.float32)
    info = INFO.SparseMeanInformation(x, c)
    original = INFO.solve_columns
    columns = []

    def counted(hv, rhs, *args, **kwargs):
        columns.append(rhs.shape[1])
        return original(hv, rhs, *args, **kwargs)

    def operator(t):
        v = sum(a*b for a, b in zip(t, [*kernels, np.eye(len(residual))]))
        return lambda b:jnp.asarray(v, dtype=jnp.float32) @ b, np.linalg.solve(v, c)

    monkeypatch.setattr(INFO, "solve_columns", counted)
    hv, hc = operator(theta)
    info.statistics(theta, hv, None, residual, tol=1e-6, maxiter=200)
    other = theta + np.array([.02, -.01, .03], dtype=np.float32)
    other_hv, other_hc = operator(other)
    info.evaluate(other, other_hv, None, other_hc, tol=1e-4, maxiter=200)

    # Moving back to theta rebuilds the state and solves the response at the
    # requested looser tolerance; an older tight response cache cannot label
    # this new solve as tight merely because theta and response match again.
    info.statistics(theta, hv, None, residual, tol=1e-4, maxiter=200)
    info.evaluate(theta, hv, None, hc, tol=1e-6, maxiter=200)
    before = len(columns)
    info.statistics(theta, hv, None, residual, tol=1e-6, maxiter=200)
    assert columns[before:] == [1]
    before = len(columns)
    info.statistics(theta, hv, None, residual, tol=1e-4, maxiter=200)
    assert len(columns) == before


def test_information_trace_cache_is_scoped_to_factor_and_operators():
    x, c, kernels, _, residual = fixture()
    theta = np.array([.27, .21, .52], dtype=np.float32)
    v = sum(a*b for a, b in zip(theta, [*kernels, np.eye(len(residual))]))
    info = INFO.SparseMeanInformation(x, c)
    state = info.evaluate(theta, lambda b:jnp.asarray(v, dtype=jnp.float32) @ b,
                          None, np.linalg.solve(v, c), tol=1e-5, maxiter=200)
    calls = []

    def kernel(b):
        calls.append(b.shape)
        return b

    first = info.score_trace_correction(state, [kernel])
    np.testing.assert_array_equal(info.score_trace_correction(state, [kernel]), first)
    assert len(calls) == 1
    changed = info.score_trace_correction(state, [lambda b:2*kernel(b)])
    assert len(calls) == 2
    np.testing.assert_allclose(changed[0], 2*first[0])
