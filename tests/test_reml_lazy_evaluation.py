"""Strict line search skips unused work without changing its numerical target."""
import importlib
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
REML = importlib.import_module(f"{ROOT.name}.reml")
SLQ = importlib.import_module(f"{ROOT.name}.slq")
INFO = importlib.import_module(f"{ROOT.name}.sparse_information")


def _problem(groups=2):
    rng = np.random.default_rng(684)
    n = 24
    kernels = []
    for k in range(groups):
        x = rng.normal(size=(n, 12+5*k)).astype(np.float32)
        kernels.append(jnp.asarray(x @ x.T/x.shape[1] + .1*np.eye(n)))
    return dict(
        y=jnp.asarray(rng.normal(size=n), dtype=jnp.float32),
        K_mvs=tuple(lambda b, k=k: k @ b for k in kernels),
        diag_list=tuple(jnp.diag(k) for k in kernels), covar=jnp.ones((n, 1)),
        n_rand_vec=7, maxiter=100, minq_iter=1, slq_samples=4, slq_m=8,
        pcg_tol=1e-5, verbose=False, response_is_standardized=True,
    )


@pytest.mark.parametrize("groups", [1, 2])
def test_strict_has_no_hutchinson_work_or_probe_rhs(monkeypatch, groups):
    def unused(*args, **kwargs):
        pytest.fail("strict REML must not compute Hutchinson traces")
    monkeypatch.setattr(REML, "_compute_traces_from_pcg", unused)
    monkeypatch.setattr(REML, "_compute_score_traces", unused)
    solve = REML.pcg_solve
    widths = []
    def observed(hv, rhs, **kwargs):
        widths.append(rhs.shape[1])
        return solve(hv, rhs, **kwargs)
    monkeypatch.setattr(REML, "pcg_solve", observed)
    cache = REML.REMLProbeCache()
    args = _problem(groups)
    first = REML.fit_reml(**args, probe_cache=cache)
    signature = cache.signature
    second = REML.fit_reml(**{**args, "n_rand_vec": 17}, probe_cache=cache)
    assert set(widths) <= {2, groups+1}
    assert cache.kvrand_stack is None
    assert cache.signature == signature
    np.testing.assert_array_equal(first[0], second[0])


def test_smile_keeps_taylor_probes_and_mode_switch_invalidates_cache(monkeypatch):
    calls = {"traces": 0, "score": 0}
    for name, key in (("_compute_traces_from_pcg", "traces"), ("_compute_score_traces", "score")):
        original = getattr(REML, name)
        def observed(*args, _key=key, _original=original, **kwargs):
            calls[_key] += 1
            return _original(*args, **kwargs)
        monkeypatch.setattr(REML, name, observed)
    cache = REML.REMLProbeCache()
    args = _problem()
    REML.fit_reml(**args, probe_cache=cache)
    assert calls == {"traces": 0, "score": 0}
    REML.fit_reml(**args, optimizer="smile_scoring", taylor_threshold=1e10, probe_cache=cache)
    assert calls == {"traces": 2, "score": 1}
    assert cache.kvrand_stack.shape == (2, 24, 7)
    REML.fit_reml(**args, probe_cache=cache)
    assert calls == {"traces": 2, "score": 1}
    assert cache.kvrand_stack is None


@pytest.mark.parametrize("budget", [1, 50000, 1 << 20])
@pytest.mark.parametrize("accepted", [False, True])
def test_slq_gate_preserves_value_and_never_repeats_forward_products(budget, accepted):
    calls = {"forward": 0, "reverse": 0}
    matrix = jnp.diag(jnp.linspace(.3, 2., 24))
    def mv(b):
        calls["forward"] += 1
        return matrix @ b
    def pullback(left, right):
        calls["reverse"] += 1
        product = matrix @ left
        return product, jnp.asarray([jnp.sum(product*right)])
    args = dict(nsamples=6, m=8, workspace_bytes=budget)
    expected = SLQ.logdet_value_and_grad(mv, pullback, 24, jax.random.PRNGKey(54), **args)
    eager_counts = calls.copy()
    calls.update(forward=0, reverse=0)
    actual = SLQ.logdet_value_and_grad(
        mv, pullback, 24, jax.random.PRNGKey(54), **args,
        accept_value=lambda value: accepted,
    )
    np.testing.assert_array_equal(actual[0], expected[0])
    assert calls["forward"] == eager_counts["forward"]
    if accepted:
        np.testing.assert_array_equal(actual[1], expected[1])
        assert calls["reverse"] == eager_counts["reverse"]
    else:
        assert actual[1] is None
        assert calls["reverse"] == (eager_counts["reverse"] if budget == 1 else 0)


@pytest.mark.parametrize("groups,with_information", [(1, False), (1, True), (2, False), (2, True)])
def test_rejected_trial_skips_ai_and_mean_score_and_accepted_reuses_solves(
    monkeypatch, groups, with_information,
):
    args = _problem(groups)
    theta = jnp.asarray([.2]*groups+[.6])
    y, c = args["y"], args["covar"]
    n = len(y)
    info = INFO.SparseMeanInformation(np.random.default_rng(85).normal(size=(n, 3)), c) if with_information else None
    rhs = jnp.column_stack([c, y])
    ctx = REML.REMLContext(
        n=n, G=groups, E=1, K_mvs=args["K_mvs"], weighted_hv=None, stacked_kv=None,
        diag_stack=jnp.stack(args["diag_list"]), residual_diag_stack=None,
        xmat=c, y=y, rhs_const=rhs, y_col=1, rand_stop=2, n_XyZ_cols=2,
        R_rand=0, kvrand_stack=None, precond_conf=None, mean_information=info,
    )
    if groups == 1:
        ctx.affine_slq_cache = REML._build_affine_slq_cache(
            args["K_mvs"][0], n, jax.random.PRNGKey(28), nsamples=4, m=8,
        )
    calls = {"solves": 0, "mean_score": 0, "reverse": 0}
    for module, name, key in ((REML, "pcg_solve", "solves"), (SLQ, "_step_pullback", "reverse")):
        original = getattr(module, name)
        def observed(*a, _original=original, _key=key, **kw):
            calls[_key] += 1
            return _original(*a, **kw)
        monkeypatch.setattr(module, name, observed)
    if info is not None:
        original_score = info.score_trace_correction
        def score(*a, **kw):
            calls["mean_score"] += 1
            return original_score(*a, **kw)
        monkeypatch.setattr(info, "score_trace_correction", score)
    def evaluate(floor=None):
        return REML._eval_once(
            ctx, theta, jnp.zeros_like(rhs), key_slq=jax.random.PRNGKey(28),
            minq_tol=1e-5, maxiter=100, precond_eps=1e-6, slq_samples=4, slq_m=8,
            compute_traces=False, min_loglik=floor,
        )
    eager = evaluate()
    calls.update(solves=0, mean_score=0, reverse=0)
    rejected = evaluate(eager[0]+1)
    np.testing.assert_array_equal(rejected[0], eager[0])
    assert rejected[1] is None and rejected[2] is None
    assert calls == {"solves": 1, "mean_score": 0, "reverse": 0}
    calls.update(solves=0, mean_score=0, reverse=0)
    accepted = evaluate(eager[0])
    assert calls["solves"] == 2
    assert calls["mean_score"] == int(with_information)
    assert calls["reverse"] == (8 if groups > 1 else 0)
    for idx in (0, 1, 4, 5, 8):
        np.testing.assert_array_equal(accepted[idx], eager[idx])
    np.testing.assert_array_equal(accepted[2].mat, eager[2].mat)


def test_lazy_backtracking_matches_eager_trajectory(monkeypatch):
    args = {**_problem(), "minq_iter": 3, "max_linesearch_trials": 8}
    original_direction = REML._projected_fisher_direction
    def overshoot(*a, **kw):
        direction, mask = original_direction(*a, **kw)
        return 8*direction, mask
    monkeypatch.setattr(REML, "_projected_fisher_direction", overshoot)
    lazy = REML.fit_reml(**args)
    assert any(row["line_search_trials"] > 1 for row in lazy[1])
    original_eval = REML._eval_once
    def eager(*a, **kw):
        kw.pop("min_loglik", None)
        return original_eval(*a, **kw)
    monkeypatch.setattr(REML, "_eval_once", eager)
    full = REML.fit_reml(**args)
    np.testing.assert_array_equal(lazy[0], full[0])
    assert len(lazy[1]) == len(full[1])
    for first, second in zip(lazy[1], full[1]):
        for field in ("loglik", "params", "accepted", "line_search_trials", "stop_reason"):
            np.testing.assert_equal(first.get(field), second.get(field))
