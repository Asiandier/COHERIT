from __future__ import annotations

import importlib
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT = os.path.dirname(REPO_ROOT)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

jax.config.update("jax_platform_name", "cpu")

PKG = os.path.basename(REPO_ROOT)
REML = importlib.import_module(f"{PKG}.reml")


def _dense_affine_context(*, with_fixed_effects: bool):
    rng = np.random.default_rng(20260723)
    n = 10
    raw = rng.normal(size=(n, 14))
    kernel_np = raw @ raw.T / raw.shape[1] + 0.08 * np.eye(n)
    kernel = jnp.asarray(kernel_np, dtype=jnp.float32)
    y = jnp.asarray(rng.normal(size=n), dtype=jnp.float32)
    xmat = (
        jnp.asarray(
            np.column_stack([np.ones(n), rng.normal(size=n)]),
            dtype=jnp.float32,
        )
        if with_fixed_effects
        else None
    )
    probes = jnp.asarray(
        rng.choice([-1.0, 1.0], size=(n, 6)), dtype=jnp.float32
    )

    rhs_parts = []
    x_cols = 0 if xmat is None else int(xmat.shape[1])
    if xmat is not None:
        rhs_parts.append(xmat)
    rhs_parts.extend([y[:, None], probes])
    rhs_const = jnp.concatenate(rhs_parts, axis=1)

    cache = REML._build_affine_slq_cache(
        lambda value: kernel @ value,
        n,
        jax.random.PRNGKey(991),
        nsamples=5,
        m=n,
    )
    context = REML.REMLContext(
        n=n,
        G=1,
        E=1,
        K_mvs=(lambda value: kernel @ value,),
        weighted_hv=None,
        stacked_kv=None,
        diag_stack=jnp.diag(kernel)[None, :],
        residual_diag_stack=None,
        xmat=xmat,
        y=y,
        rhs_const=rhs_const,
        y_col=x_cols,
        rand_stop=x_cols + 1 + int(probes.shape[1]),
        n_XyZ_cols=int(rhs_const.shape[1]),
        R_rand=int(probes.shape[1]),
        precond_conf=None,
        kvrand_stack=(kernel @ probes)[None, :, :],
        affine_slq_cache=cache,
    )
    return context


def _evaluate(context, theta):
    theta = jnp.asarray(theta, dtype=jnp.float32)
    warm = jnp.zeros(context.rhs_const.shape, dtype=jnp.float32)
    warm_ai = jnp.zeros((context.n, 2), dtype=jnp.float32)
    return REML._eval_once(
        context,
        theta,
        warm,
        warm_ai=warm_ai,
        key_slq=jax.random.PRNGKey(991),
        minq_tol=2e-6,
        maxiter=100,
        precond_eps=1e-7,
        slq_samples=5,
        slq_m=context.n,
        slq_mode="raw",
        warm_ready=False,
        warm_ai_ready=False,
        compute_traces=False,
    )


@pytest.mark.parametrize("with_fixed_effects", [False, True])
def test_affine_slq_score_matches_finite_difference_of_returned_reml_objective(
    with_fixed_effects,
):
    context = _dense_affine_context(with_fixed_effects=with_fixed_effects)
    theta = np.asarray([0.37, 0.74], dtype=np.float32)
    ll, grad, *_ = _evaluate(context, theta)
    assert np.isfinite(float(ll))

    eps = 2e-3
    finite_difference = []
    for component in range(2):
        offset = np.zeros(2, dtype=np.float32)
        offset[component] = eps
        ll_plus = float(_evaluate(context, theta + offset)[0])
        ll_minus = float(_evaluate(context, theta - offset)[0])
        finite_difference.append((ll_plus - ll_minus) / (2.0 * eps))

    np.testing.assert_allclose(
        np.asarray(grad),
        np.asarray(finite_difference),
        atol=7e-4,
        rtol=7e-3,
    )


def test_affine_slq_ai_direction_is_an_ascent_direction_with_fixed_effects():
    context = _dense_affine_context(with_fixed_effects=True)
    theta = np.asarray([0.22, 0.91], dtype=np.float32)
    ll, grad, average_info, *_ = _evaluate(context, theta)
    ai = np.asarray(average_info.mat, dtype=np.float64)
    direction = np.linalg.solve(
        0.5 * (ai + ai.T) + float(average_info.ridge) * np.eye(ai.shape[0]),
        np.asarray(grad, dtype=np.float64),
    )
    gradient = np.asarray(grad)

    assert float(gradient @ direction) > 0.0
    improvements = []
    for alpha in (0.25, 0.125, 0.0625, 0.03125, 0.015625):
        candidate = theta + alpha * direction
        if np.all(candidate > 0.0):
            improvements.append(float(_evaluate(context, candidate)[0] - ll))
    assert improvements
    assert max(improvements) > 0.0


def test_affine_slq_derivatives_remain_finite_at_zero_genetic_variance():
    context = _dense_affine_context(with_fixed_effects=False)
    cache = context.affine_slq_cache
    logdet, derivatives = REML._affine_slq_logdet(
        cache,
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32),
        return_derivatives=True,
    )
    assert np.isfinite(float(logdet))
    assert np.all(np.isfinite(np.asarray(derivatives)))

    eps = 2e-3
    forward = (
        float(REML._affine_slq_logdet(cache, eps, 1.0)) - float(logdet)
    ) / eps
    np.testing.assert_allclose(
        float(derivatives[0]), forward, atol=2e-2, rtol=2e-2
    )
    np.testing.assert_allclose(
        float(derivatives[1]), context.n, atol=2e-4, rtol=2e-5
    )
