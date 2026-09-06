"""PCG accuracy at different RHS scales and with finite-precision residual gaps."""
import importlib
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
PCG = importlib.import_module(f"{ROOT.name}.pcg")


def _true_relative(hv, x, b):
    # Use the supplied operator, then accumulate norms in float64 on the host.
    residual = np.asarray(b - hv(x), dtype=np.float64)
    rhs = np.asarray(b, dtype=np.float64)
    denominator = np.linalg.norm(rhs, axis=0)
    return float(np.max(
        np.linalg.norm(residual, axis=0) / np.where(denominator > 0, denominator, 1)
    ))


@pytest.mark.parametrize("check_every", [1, 2, 5])
@pytest.mark.parametrize("preconditioned", [False, True])
def test_diagonal_solve_handles_zero_and_extreme_rhs_scales(check_every, preconditioned):
    scales = jnp.asarray([0, 1e-30, 1e-8, 1, 1e30], dtype=jnp.float32)
    b = jnp.arange(1, 17, dtype=jnp.float32)[:, None] * scales
    calls = []

    def hv(value):
        calls.append(None)
        return 2 * value

    preconditioner = (lambda value: value / 2) if preconditioned else None
    x, relative, iterations = PCG.pcg_solve(
        hv, b, M=preconditioner, tol=1e-6, maxiter=20, check_every=check_every,
    )
    assert len(calls) == iterations + 2  # Initial residual, steps, final residual.
    np.testing.assert_allclose(np.asarray(x[:, 1:] / scales[1:]),
                               np.broadcast_to(np.arange(1, 17)[:, None] / 2, (16, 4)),
                               rtol=1e-6)
    np.testing.assert_array_equal(x[:, 0], 0)
    assert float(relative) == pytest.approx(_true_relative(hv, x, b), abs=1e-8)
    assert float(relative) <= 1e-6
    assert iterations <= check_every


@pytest.mark.parametrize("preconditioned", [False, True])
def test_spd_solution_with_scaled_rhs_and_warm_start_matches_dense(preconditioned):
    rng = np.random.default_rng(237)
    root = rng.normal(size=(24, 24))
    h = jnp.asarray(root @ root.T / 24 + np.eye(24), dtype=jnp.float32)
    rhs = rng.normal(size=(24, 3)).astype(np.float32)
    scales = jnp.asarray([1e-20, 1, 1e20], dtype=jnp.float32)
    b = jnp.asarray(rhs) * scales
    x0 = 0.1 * b
    # Match the production GRM operators' precision; GPU DEFAULT matmul can
    # introduce errors larger than this solve's tolerance in the operator itself.
    hv = lambda value: jnp.matmul(h, value, precision=jax.lax.Precision.HIGHEST)
    preconditioner = (lambda value: value / jnp.diag(h)[:, None]) if preconditioned else None
    x, relative, _ = PCG.pcg_solve(hv, b, M=preconditioner, X0=x0, tol=1e-6, maxiter=100)
    expected = np.linalg.solve(np.asarray(h, dtype=np.float64), rhs)
    np.testing.assert_allclose(np.asarray(x / scales), expected, atol=2e-6, rtol=2e-5)
    assert float(relative) <= 1e-6
    assert float(relative) == pytest.approx(_true_relative(hv, x, b), rel=2e-6, abs=1e-12)


def test_vector_input_and_converged_warm_start_are_preserved():
    diagonal = jnp.arange(1, 9, dtype=jnp.float32)
    hv = lambda value: diagonal * value
    b = jnp.arange(8, dtype=jnp.float32) * 1e-20
    x0 = b / diagonal
    x, relative, iterations = PCG.pcg_solve(hv, b, X0=x0, tol=1e-6)
    assert x.shape == b.shape
    np.testing.assert_array_equal(x, x0)
    assert iterations == 0
    assert float(relative) == pytest.approx(_true_relative(hv, x, b), abs=1e-8)


def test_zero_rhs_ignores_nonzero_warm_start_and_identity_may_alias():
    b = jnp.asarray([[0, 1], [0, -2], [0, 3]], dtype=jnp.float32)
    x, relative, _ = PCG.pcg_solve(lambda value: value, b, X0=jnp.ones_like(b), tol=1e-6)
    np.testing.assert_array_equal(x, b)
    assert float(relative) == 0


@pytest.mark.parametrize("maxiter", [1, 314])
def test_ill_conditioned_return_reports_true_residual_even_at_budget(maxiter):
    rng = np.random.default_rng(12)
    basis = np.linalg.qr(rng.normal(size=(60, 60)))[0]
    h = jnp.asarray((basis * np.geomspace(1, 1e4, 60)) @ basis.T, dtype=jnp.float32)
    b = jnp.asarray(rng.normal(size=(60, 2)), dtype=jnp.float32)
    hv = lambda value: jnp.matmul(h, value, precision=jax.lax.Precision.HIGHEST)
    x, relative, iterations = PCG.pcg_solve(hv, b, tol=1e-5, maxiter=maxiter)
    assert iterations <= maxiter
    assert float(relative) == pytest.approx(_true_relative(hv, x, b), rel=2e-6, abs=1e-10)


@pytest.mark.parametrize("maxiter", [1, 100])
def test_optimistic_recurrence_is_verified_and_restarted_within_budget(monkeypatch, maxiter):
    original = PCG._pcg_state_update
    calls = {"updates": 0, "operator": 0}

    def lose_residual_once(*args):
        x, residual, breakdown = original(*args)
        calls["updates"] += 1
        if calls["updates"] == 1:
            # Emulate an optimistic recursive residual independently of the
            # returned solution. Real residual gaps depend on backend rounding.
            residual = jnp.zeros_like(residual)
        return x, residual, breakdown

    diagonal = jnp.linspace(1, 4, 20)[:, None]
    b = jnp.ones((20, 1))

    def hv(value):
        calls["operator"] += 1
        return diagonal * value

    monkeypatch.setattr(PCG, "_pcg_state_update", lose_residual_once)
    x, relative, iterations = PCG.pcg_solve(hv, b, tol=1e-6, maxiter=maxiter, check_every=1)
    solver_operator_calls = calls["operator"]
    assert float(relative) == pytest.approx(_true_relative(hv, x, b), rel=2e-6, abs=1e-10)
    assert iterations <= maxiter
    if maxiter == 1:
        assert float(relative) > 1e-6
    else:
        assert solver_operator_calls > iterations + 2  # Initial, restart, final checks.
        assert float(relative) <= 1e-6
        np.testing.assert_allclose(x, b / diagonal, rtol=3e-6)


def test_zero_iteration_budget_returns_verified_initial_residual():
    b = jnp.ones((5, 2)) * 1e-20
    x, relative, iterations = PCG.pcg_solve(lambda value: 2 * value, b, maxiter=0)
    np.testing.assert_array_equal(x, 0)
    assert iterations == 0
    assert float(relative) == pytest.approx(1)


@pytest.mark.parametrize("factor", [0, -1, np.nan])
def test_invalid_operator_cannot_be_reported_as_converged(factor):
    with pytest.raises(FloatingPointError):
        PCG.pcg_solve(lambda value: factor * value, jnp.ones((8, 1)), maxiter=4)
