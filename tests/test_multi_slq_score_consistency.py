"""Independent finite-difference checks of the same matrix-free SLQ target."""
import importlib
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
SLQ = importlib.import_module(f"{ROOT.name}.slq")
REML = importlib.import_module(f"{ROOT.name}.reml")
INFO = importlib.import_module(f"{ROOT.name}.sparse_information")
PRECOND = importlib.import_module(f"{ROOT.name}.precond")


def problem(kind):
    rng = np.random.default_rng(904)
    n = 24
    kernels = []
    for width in (13, 21, 37):
        raw = rng.normal(size=(n, width))
        kernels.append(raw@raw.T/width)
    if kind == "scalar":
        kernels = [np.eye(n), 2*np.eye(n), 3*np.eye(n)]
    elif kind == "clustered":
        kernels = [np.eye(n)+1e-4*k for k in kernels]
    theta = np.array([.21, .16, .28, .57])
    if kind == "zero":
        theta[:3] = 0
    elif kind == "ill_conditioned":
        theta[:] = [.8, .01, .01, .002]
    return np.stack([*kernels, np.eye(n)]), theta


def oracle(directions, theta, depth, budget=64*1024**2, value_only=False):
    matrix = jnp.einsum('k,kij->ij', theta, directions, precision=jax.lax.Precision.HIGHEST)
    def mv(v):
        assert not isinstance(v, jax.core.Tracer)
        return jnp.matmul(matrix, v, precision=jax.lax.Precision.HIGHEST)
    def pullback(left, right):
        assert not isinstance(right, jax.core.Tracer)
        products = jnp.matmul(directions, left, precision=jax.lax.Precision.HIGHEST)
        return (jnp.einsum('k,kns->ns', theta, products, precision=jax.lax.Precision.HIGHEST),
                jnp.einsum('ns,kns->k', right, products, precision=jax.lax.Precision.HIGHEST))
    return SLQ.logdet_value_and_grad(mv, None if value_only else pullback,
                                     matrix.shape[0], jax.random.PRNGKey(73),
                                     nsamples=4, m=depth, workspace_bytes=budget,
                                     dtype=theta.dtype)


@pytest.mark.parametrize("kind", ["regular", "scalar", "clustered", "zero", "ill_conditioned"])
@pytest.mark.parametrize("depth", [1, 10, 30])
def test_local_slq_derivative_matches_its_value(kind, depth):
    matrices, theta = problem(kind)
    with jax.experimental.enable_x64():
        directions = jnp.asarray(matrices)
        t = jnp.asarray(theta)
        value, gradient = oracle(directions, t, depth)
        delta = 1e-6 if kind == "ill_conditioned" else 2e-5
        finite = []
        for k in range(len(t)):
            offset = np.eye(len(t))[k]*delta
            if t[k] == 0:
                finite.append(float((oracle(directions, t+offset, depth)[0]-value)/delta))
            else:
                finite.append(float((oracle(directions, t+offset, depth)[0]
                                     -oracle(directions, t-offset, depth)[0])/(2*delta)))
        np.testing.assert_allclose(gradient, finite, rtol=2e-3, atol=3e-3)


@pytest.mark.parametrize("kind", ["regular", "scalar", "clustered", "zero", "ill_conditioned"])
def test_float32_full_depth_is_finite_and_matches_dense_probe_oracle(kind):
    matrices, theta = problem(kind)
    directions = jnp.asarray(matrices, dtype=jnp.float32)
    t = jnp.asarray(theta, dtype=jnp.float32)
    value, gradient = oracle(directions, t, 50)
    matrix = np.einsum('k,kij->ij', theta, matrices)
    vals, vecs = np.linalg.eigh(matrix)
    keys = jax.random.split(jax.random.PRNGKey(73), 4)
    z = np.asarray(jax.vmap(lambda key:jax.random.rademacher(key, (24,), dtype=jnp.int32))(keys)).T
    log_matrix = (vecs*np.log(vals))@vecs.T
    expected = np.einsum('ns,nm,ms->', z, log_matrix, z)/4
    assert np.isfinite(np.asarray(gradient)).all()
    np.testing.assert_allclose(value, expected, rtol=1e-3, atol=1e-3)
    gap = vals[:, None]-vals[None, :]
    divided = np.divide(np.log(vals[:, None])-np.log(vals[None, :]), gap,
                        out=1/np.broadcast_to(vals, gap.shape).copy(), where=np.abs(gap)>1e-12)
    coefficients = vecs.T @ z
    grad_matrix = vecs @ (divided*(coefficients @ coefficients.T/4)) @ vecs.T
    expected_gradient = np.einsum('ij,kij->k', grad_matrix, matrices)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=2e-3, atol=2e-3)
    value_only = oracle(directions, t, 50, value_only=True)
    np.testing.assert_array_equal(value_only[0], value)
    assert value_only[1] is None


def test_probe_batching_preserves_the_same_target_and_gradient():
    directions, theta = problem("regular")
    with jax.experimental.enable_x64():
        directions, theta = jnp.asarray(directions), jnp.asarray(theta)
        full = oracle(directions, theta, 10)
        single = oracle(directions, theta, 10, budget=1)
        np.testing.assert_allclose(single[0], full[0], atol=1e-10)
        np.testing.assert_allclose(single[1], full[1], atol=1e-9)


@pytest.mark.parametrize("groups,optimizer,requested,residual_diags,expected", [
    (2, "strict", .005, False, 1e-5),
    (2, "strict", 1e-6, False, 1e-6),
    (1, "strict", .005, False, .005),
    (1, "strict", .005, True, 1e-5),
    (2, "smile_scoring", .005, False, .005),
])
def test_solve_accuracy_for_consistent_strict_slq(
    monkeypatch, groups, optimizer, requested, residual_diags, expected,
):
    original = REML._eval_once
    tolerances = []
    def observed(*args, **kwargs):
        tolerances.append(kwargs["minq_tol"])
        return original(*args, **kwargs)
    monkeypatch.setattr(REML, "_eval_once", observed)
    matrices, _ = problem("regular")
    matrices = jnp.asarray(matrices[:groups], dtype=jnp.float32)
    REML.fit_reml(
        y=jnp.asarray(np.random.default_rng(41).normal(size=24), dtype=jnp.float32),
        K_mvs=tuple(lambda b, k=k: jnp.matmul(k, b, precision=jax.lax.Precision.HIGHEST)
                    for k in matrices),
        diag_list=tuple(jnp.diag(k) for k in matrices), covar=None,
        n_rand_vec=3, maxiter=100, minq_iter=0, slq_samples=3, slq_m=8,
        pcg_tol=requested, optimizer=optimizer, verbose=False,
        residual_diag_list=[jnp.linspace(.5, 1.5, 24)] if residual_diags else None,
    )
    assert tolerances == [expected]


@pytest.mark.parametrize("with_covar,with_information,with_reference,multiple_residual,zero_component", [
    (False, False, False, False, False),
    (True, False, False, False, False),
    (True, True, False, False, False),
    (True, True, True, False, False),
    (True, True, True, True, False),
    (True, True, True, False, True),
])
def test_integrated_reml_score_matches_its_objective(
    with_covar, with_information, with_reference, multiple_residual, zero_component,
):
    """Do not mock SLQ: exercise C, selected mean, whitening and residual terms."""
    with jax.default_matmul_precision("highest"):
        matrices, theta = problem("regular")
        n = len(matrices[0])
        rng = np.random.default_rng(91)
        kernels = jnp.asarray(matrices[:3], dtype=jnp.float32)
        operators = tuple(lambda b, k=k: k @ b for k in kernels)
        residual_diags = None
        if multiple_residual:
            residual_diags = jnp.asarray(np.stack([np.ones(n), np.linspace(.3, 1.5, n)]), dtype=jnp.float32)
            theta = np.r_[theta, .16]
        if zero_component:
            theta[0] = 0
        theta = jnp.asarray(theta, dtype=jnp.float32)
        covar = (jnp.asarray(np.column_stack([np.ones(n), rng.normal(size=n)]), dtype=jnp.float32)
                 if with_covar else None)
        y = jnp.asarray(rng.normal(size=n), dtype=jnp.float32)
        probes = jnp.asarray(rng.choice([-1., 1.], size=(n, 4)), dtype=jnp.float32)
        columns = 0 if covar is None else covar.shape[1]
        rhs = jnp.column_stack([y, probes] if covar is None else [covar, y, probes])
        ctx = REML.REMLContext(
            n=n, G=3, E=2 if multiple_residual else 1, K_mvs=operators,
            weighted_hv=None, stacked_kv=None, diag_stack=jnp.diagonal(kernels, axis1=1, axis2=2),
            residual_diag_stack=residual_diags, xmat=covar, y=y, rhs_const=rhs,
            y_col=columns, rand_stop=rhs.shape[1], n_XyZ_cols=rhs.shape[1], R_rand=4,
            kvrand_stack=jnp.stack([op(probes) for op in operators]), precond_conf=None,
            mean_information=INFO.SparseMeanInformation(rng.normal(size=(n, 3)), covar)
                             if with_information else None,
        )
        if with_reference:
            basis = jnp.asarray(np.linalg.qr(rng.normal(size=(n, 4)))[0], dtype=jnp.float32)
            eigvals = jnp.asarray([1.1, 1.3, 1.7, 2.1], dtype=jnp.float32)
            ctx.slq_reference_runtime = PRECOND.ProjectedCoreRuntime(
                U=basis, total_rank=4, d=jnp.asarray(.9), d_inv=jnp.asarray(1/.9),
                chol=jnp.diag(jnp.sqrt(eigvals)), eigvals=eigvals, eigvecs=jnp.eye(4),
            )
        def evaluate(t, **kwargs):
            return REML._eval_once(
                ctx, t, jnp.zeros_like(rhs), key_slq=jax.random.PRNGKey(83),
                minq_tol=2e-6, maxiter=200, precond_eps=1e-7,
                slq_samples=4, slq_m=10, slq_mode="projected_core_residual",
                compute_traces=False,
                **kwargs,
            )
        value, gradient, information, *_ = evaluate(theta)
        accepted = evaluate(theta, min_loglik=value)
        rejected = evaluate(theta, min_loglik=value+.1)
        np.testing.assert_array_equal(accepted[0], value)
        np.testing.assert_array_equal(rejected[0], value)
        np.testing.assert_array_equal(accepted[1], gradient)
        np.testing.assert_array_equal(accepted[2].mat, information.mat)
        assert rejected[1] is None and rejected[2] is None
        fd = []
        for k in range(len(theta)):
            delta = jnp.eye(len(theta), dtype=theta.dtype)[k]*.001
            if theta[k] == 0:
                fd.append(float((evaluate(theta+delta)[0]-value)/.001))
            else:
                fd.append(float((evaluate(theta+delta)[0]-evaluate(theta-delta)[0])/.002))
        np.testing.assert_allclose(gradient, fd, atol=6e-4, rtol=5e-3)
        if not zero_component:
            direction = np.linalg.solve(np.asarray(information.mat)+np.eye(len(theta))*information.ridge,
                                        np.asarray(gradient))
            assert float(gradient @ direction) > 0
            step = min(.002, .05*float(np.min(np.asarray(theta)/np.maximum(abs(direction), 1e-9))))
            assert float(evaluate(theta+step*direction)[0]) > float(value)
