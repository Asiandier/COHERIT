"""
Block preconditioned conjugate gradient (multi-RHS).

Performance notes
-----------------
*  `Hv(P)` calls `kv()` which is the dominant cost per iteration.
*  Convergence checks force a GPU→CPU synchronisation,
   draining the entire async dispatch pipeline. This is done every
   `check_every` iterations instead of every iteration to keep the GPU busy.
*  The state-update and direction-update steps are individually JIT-compiled
   so that XLA fuses the elementwise arithmetic into efficient kernels.
*  A true residual is checked before returning. Residual-gap restarts share
   the original iteration budget; verification adds an operator application.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp

Array = jnp.ndarray


def _identity(v: Array) -> Array:
    return v


@jax.jit
def _pcg_state_update(
    X: Array,
    R: Array,
    P: Array,
    rs: Array,
    AP: Array,
    active: Array,
) -> Tuple[Array, Array, Array]:
    """Compute (X_new, R_new) given AP = H @ P."""
    # AP may alias P for an identity operator, so it must not be donated.
    denom = jnp.sum(P * AP, axis=0)
    valid = (rs > 0) & (denom > 0) & jnp.isfinite(rs) & jnp.isfinite(denom)
    moving = active & valid
    alpha = jnp.where(moving, rs / jnp.where(moving, denom, 1), 0)
    X_new = X + P * alpha
    R_new = R - AP * alpha
    return X_new, R_new, jnp.any(active & ~valid)


@jax.jit
def _pcg_direction_update(
    P: Array,
    R_new: Array,
    Z_new: Array,
    rs_old: Array,
    threshold: Array,
) -> Tuple[Array, Array, Array, Array]:
    """Update search direction and compute new residual norm."""
    rs_new    = jnp.sum(R_new * Z_new, axis=0)
    rnorm_new = jnp.linalg.norm(R_new, axis=0)
    active = rnorm_new > threshold
    beta = jnp.where(active, rs_new / jnp.where(rs_old > 0, rs_old, 1), 0)
    P_new = jnp.where(active, Z_new + P * beta, 0)
    return P_new, jnp.where(active, rs_new, 0), rnorm_new, active


def pcg_solve(
    Hv: Callable[[Array], Array],
    B: Array,
    M: Optional[Callable[[Array], Array]] = None,
    tol: float = 1e-2,
    maxiter: int = 200,
    X0: Optional[Array] = None,
    check_every: int = 2,
) -> Tuple[Array, Array, int]:
    """
    Solve H X = B with block PCG.

    Parameters
    ----------
    check_every : int
        Check recursive convergence every this many iterations. Reducing sync
        frequency from every-iteration to every-N keeps the GPU dispatch
        pipeline full.  The solver may overshoot convergence by at most
        ``check_every - 1`` iterations (cheap compared to sync cost).

    Each RHS is scaled independently while iterating, without changing H or M.
    Zero RHS columns have the exact solution zero, including with an X0.

    Returns ``(X, max_rel_res, iters)``. ``max_rel_res`` is computed from
    ``B - Hv(X)`` for the returned X, not the recursively updated residual.
    If a recursive convergence check fails this verification, PCG restarts
    from the true residual within the remaining ``maxiter`` budget. Exhausting
    that budget returns the actual residual even when it exceeds ``tol``;
    callers remain responsible for accepting or rejecting an inexact solve.
    """
    if M is None:
        M = _identity
    check_every = max(1, int(check_every))
    # Power-of-two scaling avoids norm under/overflow without rounding the
    # rescaled solution when converting between iteration and caller units.
    magnitude = jnp.max(jnp.abs(B), axis=0)
    nonzero = magnitude > 0
    _, exponent = jnp.frexp(magnitude)
    rhs_scale = jnp.where(
        nonzero, jnp.ldexp(jnp.ones_like(magnitude), exponent - 1), 1
    )
    b_norm = jnp.where(nonzero, jnp.linalg.norm(B / rhs_scale, axis=0), 1)
    threshold = tol * b_norm
    solution = jnp.zeros_like(B) if X0 is None else jnp.where(nonzero, X0, 0)

    def true_residual(value):
        residual = (B - Hv(value)) / rhs_scale
        norms = jnp.linalg.norm(residual, axis=0)
        relative = jnp.max(norms / b_norm)
        relative_host = float(jax.device_get(relative))
        if not math.isfinite(relative_host):
            raise FloatingPointError("Non-finite true residual in PCG.")
        return residual, norms, relative, relative_host

    # Preserve a converged warm start exactly; its residual is already verified.
    R, rnorm, res_final, relative_host = true_residual(solution)
    if relative_host <= tol or maxiter <= 0:
        return solution, res_final, 0

    X = solution / rhs_scale
    active = rnorm > threshold
    Z = M(R)
    P = jnp.where(active, Z, 0)
    rs = jnp.where(active, jnp.sum(R * Z, axis=0), 0)
    del Z

    k = 0
    breakdown = jnp.asarray(False)
    while k < maxiter:
        AP   = Hv(P)                                      # ← dominant cost
        X, R, invalid_step = _pcg_state_update(X, R, P, rs, AP, active)
        breakdown = breakdown | invalid_step
        del AP
        Z    = M(R)
        P, rs, rnorm, active = _pcg_direction_update(P, R, Z, rs, threshold)
        del Z
        k += 1

        # Periodic convergence check (GPU→CPU sync).
        if k % check_every == 0 or k == maxiter:
            recursive_relative = float(jax.device_get(jnp.where(
                breakdown, jnp.inf, jnp.max(rnorm / b_norm)
            )))
            if not math.isfinite(recursive_relative):
                raise FloatingPointError(
                    "PCG encountered nonpositive curvature or a non-finite recurrence."
                )
            if recursive_relative <= tol or k == maxiter:
                solution = X * rhs_scale
                R, rnorm, res_final, relative_host = true_residual(solution)
                if relative_host <= tol or k == maxiter:
                    return solution, res_final, k
                # Re-anchor both the solution and search directions to the
                # verified state. The iteration counter is never reset.
                X = solution / rhs_scale
                active = rnorm > threshold
                Z = M(R)
                P = jnp.where(active, Z, 0)
                rs = jnp.where(active, jnp.sum(R * Z, axis=0), 0)
                del Z


__all__ = ["pcg_solve"]
