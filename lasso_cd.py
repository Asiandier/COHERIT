"""
Sparse weighted LASSO utilities for large-scale REML pipelines.

This module provides:
- Coordinate-descent LASSO solver on a precomputed Gram system.
- Complete lambda-path construction for validation selection.
- Frozen lambda-ratio selection for the final train+validation refit.
- Weighted sparse-effect fitting with unpenalized covariates and
  penalized SNP effects.

Objective (given H^{-1}):
    min_{b_c, b_s} 0.5 * (y - C b_c - Z b_s)^T H^{-1} (y - C b_c - Z b_s)
                  + lambda * ||b_s||_1
where C is unpenalized covariates and Z is penalized SNP matrix.

Performance notes:
- The CD inner loop is Numba-JIT compiled (>50× faster than pure Python).
- Gram matrices are kept column-major because coordinate updates scan columns.
- Complete coefficient paths reuse one batched Gram product for external starts.
- SPD solves are delegated to pipeline_common.solve_spd.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla
from numba import njit

logger = logging.getLogger(__name__)

from .pipeline_common import solve_spd


@dataclass
class LassoPathConfig:
    lam_min_ratio: float = 0.05
    n_lambda: int = 60
    max_cd_iter: int = 2000
    cd_tol: float = 1e-6
    active_set_period: int = 5
    kkt_abs_tol: float = 1e-4
    kkt_rel_tol: float = 1e-4
    fixed_lam_ratio: float | None = None
    verbose: bool = False


@dataclass(frozen=True)
class _FactorizedLinearSystem:
    factor: np.ndarray
    lower: bool
    use_cholesky: bool
    piv: np.ndarray | None = None


def _factor_linear_system(mat: np.ndarray) -> _FactorizedLinearSystem:
    A = np.asarray(mat, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Linear solve factorization expects a square matrix.")
    if A.shape[0] == 0:
        return _FactorizedLinearSystem(
            factor=np.empty((0, 0), dtype=np.float64),
            lower=True,
            use_cholesky=True,
            piv=None,
        )
    try:
        factor, lower = sla.cho_factor(A, lower=True, check_finite=False)
        return _FactorizedLinearSystem(
            factor=factor,
            lower=bool(lower),
            use_cholesky=True,
            piv=None,
        )
    except np.linalg.LinAlgError:
        factor, piv = sla.lu_factor(A, check_finite=False)
        return _FactorizedLinearSystem(
            factor=factor,
            lower=False,
            use_cholesky=False,
            piv=piv,
        )


def _solve_factorized_system(system: _FactorizedLinearSystem, rhs: np.ndarray) -> np.ndarray:
    B = np.asarray(rhs, dtype=np.float64)
    if system.factor.shape[0] == 0:
        return np.zeros_like(B, dtype=np.float64)
    if system.use_cholesky:
        return sla.cho_solve((system.factor, system.lower), B, check_finite=False)
    if system.piv is None:
        raise RuntimeError("LU factorization is missing pivot metadata.")
    return sla.lu_solve((system.factor, system.piv), B, check_finite=False)


def make_lambda_sequence(lam_max: float, lam_min_ratio: float, n_lambda: int) -> np.ndarray:
    lam_max = float(lam_max)
    lam_min_ratio = float(lam_min_ratio)
    n_lambda = int(n_lambda)
    if not math.isfinite(lam_max) or lam_max < 0.0:
        raise ValueError("lam_max must be finite and nonnegative.")
    if (
        not math.isfinite(lam_min_ratio)
        or not 0.0 < lam_min_ratio <= 1.0
    ):
        raise ValueError("lam_min_ratio must lie in (0, 1].")
    if n_lambda < 1:
        raise ValueError("n_lambda must be >= 1.")
    if lam_max == 0.0:
        return np.array([0.0], dtype=np.float64)
    if n_lambda == 1:
        return np.array([lam_max], dtype=np.float64)

    lam_min = lam_max * lam_min_ratio
    if lam_min == 0.0:
        raise ValueError("lam_max * lam_min_ratio underflows to zero.")
    # Construct the grid relative to lambda_max, then pin both endpoints.
    # exp(log(lambda_max)) is not generally bit-identical to lambda_max; the
    # former implementation could therefore report a first ratio slightly
    # above one while treating that point as the exact lambda-max zero model.
    sequence = lam_max * np.exp(
        np.linspace(0.0, math.log(lam_min_ratio), n_lambda)
    )
    sequence[0] = lam_max
    sequence[-1] = lam_min
    return sequence.astype(np.float64, copy=False)


# ---------------------------------------------------------------------------
# Numba-JIT coordinate descent kernel
# ---------------------------------------------------------------------------

@njit(cache=True, nogil=True)
def _cd_epoch(Q: np.ndarray, q: np.ndarray, diag: np.ndarray,
              beta: np.ndarray, Qb: np.ndarray, lam: float) -> float:
    """
    One full CD sweep over all coordinates. Returns max |delta|.

    Fused inner loop: soft-threshold + Qb update in compiled code.
    ~50-100× faster than the equivalent Python loop.
    """
    k = beta.shape[0]
    max_delta = 0.0
    for j in range(k):
        q_j_tilde = q[j] - (Qb[j] - diag[j] * beta[j])
        # inline soft-threshold
        if q_j_tilde > lam:
            beta_new = (q_j_tilde - lam) / diag[j]
        elif q_j_tilde < -lam:
            beta_new = (q_j_tilde + lam) / diag[j]
        else:
            beta_new = 0.0
        delta = beta_new - beta[j]
        if delta != 0.0:
            beta[j] = beta_new
            # Qb += Q[:, j] * delta — column update
            for i in range(k):
                Qb[i] += Q[i, j] * delta
            ad = abs(delta)
            if ad > max_delta:
                max_delta = ad
    return max_delta


@njit(cache=True, nogil=True)
def _cd_active_epoch(Q: np.ndarray, q: np.ndarray, diag: np.ndarray,
                     beta: np.ndarray, Qb: np.ndarray, lam: float,
                     active: np.ndarray) -> float:
    """
    CD sweep over active set only — much faster when most coordinates are zero.
    """
    n_active = active.shape[0]
    max_delta = 0.0
    for idx in range(n_active):
        j = active[idx]
        q_j_tilde = q[j] - (Qb[j] - diag[j] * beta[j])
        if q_j_tilde > lam:
            beta_new = (q_j_tilde - lam) / diag[j]
        elif q_j_tilde < -lam:
            beta_new = (q_j_tilde + lam) / diag[j]
        else:
            beta_new = 0.0
        delta = beta_new - beta[j]
        if delta != 0.0:
            beta[j] = beta_new
            for idx_i in range(n_active):
                i = active[idx_i]
                Qb[i] += Q[i, j] * delta
            ad = abs(delta)
            if ad > max_delta:
                max_delta = ad
    return max_delta


@njit(cache=True, nogil=True)
def _cd_active_full_qb_epoch(
    Q: np.ndarray,
    q: np.ndarray,
    diag: np.ndarray,
    beta: np.ndarray,
    Qb: np.ndarray,
    lam: float,
    active: np.ndarray,
) -> float:
    """Active-coordinate sweep that keeps every entry of ``Qb`` current.

    Once the active set is moderately dense, scanning complete column-major
    Gram columns is faster than gathering the active rows of those columns.
    Keeping inactive Qb entries current also removes the dense reconstruction
    otherwise required before the next full sweep.
    """
    k = beta.shape[0]
    max_delta = 0.0
    for idx in range(active.shape[0]):
        j = active[idx]
        q_j_tilde = q[j] - (Qb[j] - diag[j] * beta[j])
        if q_j_tilde > lam:
            beta_new = (q_j_tilde - lam) / diag[j]
        elif q_j_tilde < -lam:
            beta_new = (q_j_tilde + lam) / diag[j]
        else:
            beta_new = 0.0
        delta = beta_new - beta[j]
        if delta != 0.0:
            beta[j] = beta_new
            for i in range(k):
                Qb[i] += Q[i, j] * delta
            ad = abs(delta)
            if ad > max_delta:
                max_delta = ad
    return max_delta


@njit(cache=True, nogil=True)
def _score_kkt_errors(
    q: np.ndarray,
    Qb: np.ndarray,
    beta: np.ndarray,
    lam: float,
) -> tuple[bool, float, float]:
    """Single-pass, allocation-free active/inactive KKT errors."""
    max_active_error = 0.0
    max_inactive_excess = 0.0
    for index in range(beta.size):
        q_value = q[index]
        Qb_value = Qb[index]
        beta_value = beta[index]
        if not (
            np.isfinite(q_value)
            and np.isfinite(Qb_value)
            and np.isfinite(beta_value)
        ):
            return False, np.inf, np.inf
        score = q_value - Qb_value
        if beta_value != 0.0:
            sign = 1.0 if beta_value > 0.0 else -1.0
            error = abs(score - lam * sign)
            if error > max_active_error:
                max_active_error = error
        else:
            excess = abs(score) - lam
            if excess > max_inactive_excess:
                max_inactive_excess = excess
    return True, max_active_error, max(max_inactive_excess, 0.0)


def _score_kkt_diagnostics(
    *,
    q: np.ndarray,
    Qb: np.ndarray,
    beta: np.ndarray,
    lam: float,
    abs_tol: float,
    rel_tol: float,
) -> tuple[bool, float, float, float]:
    """Return the complete active/inactive score-KKT certificate."""
    lam_f = float(lam)
    tolerance = max(
        float(abs_tol),
        float(rel_tol) * max(1.0, abs(lam_f)),
    )
    q_arr = np.asarray(q, dtype=np.float64)
    Qb_arr = np.asarray(Qb, dtype=np.float64)
    beta_arr = np.asarray(beta, dtype=np.float64)
    if (
        q_arr.ndim != 1
        or Qb_arr.shape != q_arr.shape
        or beta_arr.shape != q_arr.shape
        or not np.isfinite(lam_f)
        or lam_f < 0.0
    ):
        return False, tolerance, float("inf"), float("inf")

    finite, max_active_error, max_inactive_excess = _score_kkt_errors(
        q_arr,
        Qb_arr,
        beta_arr,
        lam_f,
    )
    passed = bool(
        finite
        and max_active_error <= tolerance
        and max_inactive_excess <= tolerance
    )
    return passed, tolerance, max_active_error, max_inactive_excess


def solve_lasso_cd_gram(
    Q: np.ndarray,
    q: np.ndarray,
    lam: float,
    *,
    beta0: np.ndarray | None = None,
    _Qb0: np.ndarray | None = None,
    max_iter: int = 2000,
    tol: float = 1e-6,
    active_set_period: int = 5,
    kkt_abs_tol: float | None = None,
    kkt_rel_tol: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, int, bool]:
    """
    Coordinate descent for
        min_b 0.5 b^T Q b - q^T b + lam ||b||_1,
    where Q is symmetric PSD and diag(Q) > 0.

    Uses Numba-JIT inner loop with active-set acceleration:
    alternates between full sweeps and active-set-only sweeps.  ``converged``
    is determined solely by the explicit active/inactive score-KKT equations.
    The coefficient-update tolerance only schedules an earlier full sweep; it
    is not a convergence condition because its scale depends on the Gram
    diagonal and column correlation.
    """
    # Coordinate updates repeatedly scan Q[:, j].  Keeping the Gram matrix in
    # Fortran order makes those column reads contiguous.  ``asfortranarray`` is
    # a no-op when solve_lasso_path has already prepared the shared matrix, so
    # the complete lambda path pays for at most one layout conversion.
    Q = np.asfortranarray(Q, dtype=np.float64)
    q = np.ascontiguousarray(q.reshape(-1), dtype=np.float64)
    k = q.size

    if Q.shape != (k, k):
        raise ValueError("Q shape mismatch.")

    if beta0 is None:
        beta = np.zeros(k, dtype=np.float64)
    else:
        beta = np.ascontiguousarray(beta0.reshape(-1), dtype=np.float64).copy()
        if beta.size != k:
            raise ValueError("beta0 shape mismatch.")

    diag = np.diag(Q).copy()
    min_diag = float(np.min(diag)) if diag.size > 0 else 1.0
    if min_diag <= 0.0:
        raise ValueError("Q diagonal must be strictly positive for coordinate descent.")

    if _Qb0 is None:
        Qb = Q @ beta
    else:
        Qb = np.ascontiguousarray(
            np.asarray(_Qb0, dtype=np.float64).reshape(-1)
        ).copy()
        if Qb.size != k or not np.all(np.isfinite(Qb)):
            raise ValueError("_Qb0 shape/values are invalid.")
    converged = False
    qb_full_stale = False

    max_iter = max(int(max_iter), 1)
    kkt_abs = float(tol) if kkt_abs_tol is None else float(kkt_abs_tol)
    kkt_rel = float(kkt_rel_tol)
    if not np.isfinite(float(lam)) or float(lam) < 0.0:
        raise ValueError("lambda must be finite and nonnegative.")
    if (
        not np.isfinite(kkt_abs)
        or not np.isfinite(kkt_rel)
        or kkt_abs < 0.0
        or kkt_rel < 0.0
    ):
        raise ValueError("KKT tolerances must be nonnegative.")
    # Active-set strategy: alternate cheaper active-only sweeps with periodic
    # full sweeps.  A cheap incremental-score check filters which full sweeps
    # need the decisive exact-matvec KKT certificate.
    active_set_period = max(int(active_set_period), 1)
    it = 0
    active = np.empty((0,), dtype=np.int64)

    for it_idx in range(1, max_iter + 1):
        it = it_idx
        completed_full_sweep = False
        if it_idx == 1 or it_idx % active_set_period == 0:
            # Full sweep
            if qb_full_stale:
                Qb = Q @ beta
                qb_full_stale = False
            max_delta = _cd_epoch(Q, q, diag, beta, Qb, lam)
            completed_full_sweep = True
            active = np.flatnonzero(beta != 0.0).astype(np.int64)
        else:
            # Active-set sweep
            if active.size == 0:
                # All zero — check full sweep to see if any should activate
                if qb_full_stale:
                    Qb = Q @ beta
                    qb_full_stale = False
                max_delta = _cd_epoch(Q, q, diag, beta, Qb, lam)
                completed_full_sweep = True
                active = np.flatnonzero(beta != 0.0).astype(np.int64)
            elif not qb_full_stale and 4 * active.size >= k:
                # At roughly one-quarter density, contiguous full-column Qb
                # updates become cheaper than gathered active-row updates on
                # the publication-scale Gram matrices.  This is purely a
                # memory-layout choice; the coordinate sequence is unchanged.
                max_delta = _cd_active_full_qb_epoch(
                    Q, q, diag, beta, Qb, lam, active
                )
            else:
                max_delta = _cd_active_epoch(Q, q, diag, beta, Qb, lam, active)
                if max_delta > 0.0:
                    qb_full_stale = True

        if max_delta <= tol and not completed_full_sweep:
            # A small active-set update triggers an immediate full sweep.  It
            # is only an efficiency heuristic; the score KKT equations below,
            # not the coefficient delta, decide convergence.
            if qb_full_stale:
                Qb = Q @ beta
                qb_full_stale = False
            max_delta = _cd_epoch(Q, q, diag, beta, Qb, lam)
            it += 1
            completed_full_sweep = True
            active = np.flatnonzero(beta != 0.0).astype(np.int64)

        if completed_full_sweep:
            # Use the incrementally maintained score as a cheap filter.  Only
            # a point that may be converged pays for the exact dense matvec;
            # accumulated roundoff is therefore never allowed to decide the
            # final score-KKT certificate.
            provisional_kkt, _, _, _ = _score_kkt_diagnostics(
                q=q,
                Qb=Qb,
                beta=beta,
                lam=float(lam),
                abs_tol=kkt_abs,
                rel_tol=kkt_rel,
            )
            if provisional_kkt or max_delta <= tol:
                Qb = Q @ beta
                qb_full_stale = False
                kkt_passed, _, _, _ = _score_kkt_diagnostics(
                    q=q,
                    Qb=Qb,
                    beta=beta,
                    lam=float(lam),
                    abs_tol=kkt_abs,
                    rel_tol=kkt_rel,
                )
                if kkt_passed:
                    converged = True
                    break

    if not converged:
        # ``max_iter`` may end immediately after an active-only sweep.  Rebuild
        # the exact score and apply the same decisive certificate once more;
        # do not report a KKT solution as failed merely because no filtered
        # exact check remained in the iteration budget.
        Qb = Q @ beta
        qb_full_stale = False
        converged, _, _, _ = _score_kkt_diagnostics(
            q=q,
            Qb=Qb,
            beta=beta,
            lam=float(lam),
            abs_tol=kkt_abs,
            rel_tol=kkt_rel,
        )
    return beta, Qb, it, converged


def solve_lasso_path(
    Q: np.ndarray,
    q: np.ndarray,
    yHy: float,
    cfg: LassoPathConfig,
    beta_path0: np.ndarray | None = None,
) -> dict:
    """
    Solve the complete requested lambda path.

    Validation selection is deliberately performed by the caller because it
    requires held-out genotypes and phenotypes.  When ``fixed_lam_ratio`` is
    supplied, this function solves only lambda-max and the exact frozen-ratio
    target required by the final train+validation refit.

    Args:
        Q, q: weighted quadratic system.
        yHy: constant term for the quadratic RSS represented by Q and q.
            This is y^T H^{-1} y without unpenalized covariates, and the
            profiled constant after projecting out covariates otherwise.
        beta_path0: Optional coefficient path from a nearby problem, aligned
            row-for-row with the requested lambda-ratio grid.  Every supplied
            row is only a warm start; the usual score-KKT certificate still
            determines convergence and acceptance.
    """
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    Q = np.asfortranarray(Q, dtype=np.float64)
    if Q.shape != (q.size, q.size):
        raise ValueError("Q/q shape mismatch.")

    fixed_lam_ratio = cfg.fixed_lam_ratio
    if fixed_lam_ratio is not None:
        if (
            not math.isfinite(float(fixed_lam_ratio))
            or not 0.0 < float(fixed_lam_ratio) <= 1.0
        ):
            raise ValueError(
                "fixed_lam_ratio must lie in (0, 1]."
            )

    lam_max = float(np.max(np.abs(q))) if q.size > 0 else 0.0
    if fixed_lam_ratio is None:
        lam_seq = make_lambda_sequence(lam_max, cfg.lam_min_ratio, cfg.n_lambda)
        if lam_max > 0.0:
            lam_ratio_seq = lam_seq / lam_max
            lam_ratio_seq[0] = 1.0
            if lam_ratio_seq.size > 1:
                lam_ratio_seq[-1] = float(cfg.lam_min_ratio)
        else:
            lam_ratio_seq = np.ones(lam_seq.size, dtype=np.float64)
        path_role = "complete_validation_grid"
    else:
        # The final train+validation refit has already frozen the ratio.  Its
        # convex target depends only on lambda_max and the requested target;
        # solving the other validation-grid points cannot change that target
        # solution.  Keep lambda_max as a deterministic zero-model warm start
        # and solve exactly one additional point (or one point when ratio=1).
        target_lam = lam_max * float(fixed_lam_ratio)
        lam_seq = (
            np.asarray([lam_max], dtype=np.float64)
            if lam_max <= 0.0 or float(fixed_lam_ratio) == 1.0
            else np.asarray([lam_max, target_lam], dtype=np.float64)
        )
        lam_ratio_seq = (
            np.asarray(
                [
                    float(fixed_lam_ratio)
                    if lam_max <= 0.0
                    else 1.0
                ],
                dtype=np.float64,
            )
            if lam_seq.size == 1
            else np.asarray([1.0, float(fixed_lam_ratio)], dtype=np.float64)
        )
        path_role = "frozen_ratio_target_only"

    external_beta_path = None
    external_Qb_path = None
    if beta_path0 is not None:
        external_beta_path = np.asarray(beta_path0, dtype=np.float64)
        expected_shape = (int(lam_seq.size), int(q.size))
        if external_beta_path.shape != expected_shape:
            raise ValueError(
                "beta_path0 shape mismatch: expected "
                f"{expected_shape}, got {external_beta_path.shape}."
            )
        if not np.all(np.isfinite(external_beta_path)):
            raise ValueError("beta_path0 must contain only finite values.")
        # One level-3 BLAS operation is substantially cheaper than one Python-
        # dispatched matrix-vector product for every lambda row.
        external_Qb_path = np.ascontiguousarray(
            (Q @ external_beta_path.T).T,
            dtype=np.float64,
        )

    rss0 = float(yHy)
    beta_warm = np.zeros_like(q)
    Qb_warm = np.zeros_like(q)
    path = []
    beta_path: list[np.ndarray] = []
    external_warm_rows_used = 0

    for i, (lam, lam_ratio) in enumerate(zip(lam_seq, lam_ratio_seq)):
        # lambda_max has the exact all-zero solution.  For lower path points,
        # a same-ratio solution mapped from the preceding candidate/outer fit
        # is normally much closer than restarting from the adjacent solution
        # on a newly enlarged candidate set.
        beta_start = beta_warm
        Qb_start = Qb_warm
        if (
            external_beta_path is not None
            and external_Qb_path is not None
            and i > 0
        ):
            external_beta = external_beta_path[i]
            external_Qb = external_Qb_path[i]
            sequential_kkt = _score_kkt_diagnostics(
                q=q,
                Qb=Qb_warm,
                beta=beta_warm,
                lam=float(lam),
                abs_tol=float(cfg.kkt_abs_tol),
                rel_tol=float(cfg.kkt_rel_tol),
            )
            external_kkt = _score_kkt_diagnostics(
                q=q,
                Qb=external_Qb,
                beta=external_beta,
                lam=float(lam),
                abs_tol=float(cfg.kkt_abs_tol),
                rel_tol=float(cfg.kkt_rel_tol),
            )
            sequential_error = max(sequential_kkt[2], sequential_kkt[3])
            external_error = max(external_kkt[2], external_kkt[3])
            if external_error < sequential_error:
                beta_start = external_beta
                Qb_start = external_Qb
                external_warm_rows_used += 1
        if i == 0:
            # By construction lambda_max=max(abs(q)), so the exact first path
            # solution is beta=0.  Avoid entering coordinate descent merely to
            # rediscover that deterministic KKT point.
            beta = np.zeros_like(q)
            Qb = np.zeros_like(q)
            n_iter = 0
            converged = True
        else:
            beta, Qb, n_iter, converged = solve_lasso_cd_gram(
                Q,
                q,
                float(lam),
                beta0=beta_start,
                _Qb0=Qb_start,
                max_iter=cfg.max_cd_iter,
                tol=cfg.cd_tol,
                active_set_period=cfg.active_set_period,
                kkt_abs_tol=cfg.kkt_abs_tol,
                kkt_rel_tol=cfg.kkt_rel_tol,
            )
        beta_warm = beta
        Qb_warm = Qb

        rss = max(float(rss0 - 2.0 * (beta @ q) + (beta @ Qb)), 0.0)
        k = int(np.count_nonzero(beta))
        (
            kkt_passed,
            kkt_tolerance,
            max_active_kkt_error,
            max_inactive_kkt_excess,
        ) = _score_kkt_diagnostics(
            q=q,
            Qb=Qb,
            beta=beta,
            lam=float(lam),
            abs_tol=float(cfg.kkt_abs_tol),
            rel_tol=float(cfg.kkt_rel_tol),
        )
        path.append(
            {
                "lam": float(lam),
                "lam_ratio": float(lam_ratio),
                "k": k,
                "rss": rss,
                "cd_iter": int(n_iter),
                "converged": bool(converged),
                "kkt_passed": kkt_passed,
                "kkt_tolerance": kkt_tolerance,
                "max_active_kkt_error": max_active_kkt_error,
                "max_inactive_kkt_excess": max_inactive_kkt_excess,
            }
        )
        beta_path.append(beta.copy())

        # Never silently truncate the requested path at an unsolved point.
        # Validation selection must see the same complete, certified grid in
        # every alpha/theta iteration.
        if not (converged and kkt_passed):
            raise RuntimeError(
                "Lasso path failed score-KKT convergence at "
                f"index={i}, lambda={float(lam):.8e}, cd_iter={int(n_iter)}, "
                f"active_error={max_active_kkt_error:.8e}, "
                f"inactive_excess={max_inactive_kkt_excess:.8e}, "
                f"tolerance={kkt_tolerance:.8e}."
            )

        if cfg.verbose and (i == 0 or i == len(lam_seq) - 1 or (i + 1) % 10 == 0):
            logger.info(
                "[lasso_path] %03d/%03d lam=%.3e k=%4d rss=%.4e "
                "cd_iter=%d conv=%s",
                i + 1, len(lam_seq), lam, k, rss, n_iter, converged,
            )

    beta_path_array = np.stack(beta_path, axis=0).astype(
        np.float64, copy=False
    )
    result = {
        "path": path,
        "beta_path": beta_path_array,
        "lam_max": float(lam_max),
        "selected_index": None,
        "selection_method": None,
        "selected_lam_ratio": None,
        "path_role": path_role,
        "requested_n_lambda": int(cfg.n_lambda),
        "external_beta_path_warm_start_provided": bool(
            external_beta_path is not None
        ),
        "external_beta_path_warm_start_used": bool(
            external_warm_rows_used > 0
        ),
        "external_beta_path_warm_start_rows_used": int(
            external_warm_rows_used
        ),
    }
    if fixed_lam_ratio is not None:
        selected_index = len(path) - 1
        beta_selected = beta_path_array[selected_index].copy()
        result.update(
            {
                "lam": float(path[selected_index]["lam"]),
                "beta": beta_selected,
                "active_idx": np.flatnonzero(beta_selected != 0.0).astype(
                    np.int64
                ),
                "selected_index": int(selected_index),
                "selection_method": "fixed_lam_ratio",
                "selected_lam_ratio": float(fixed_lam_ratio),
            }
        )
    return result


def compute_projected_hinv_vector(
    covar: np.ndarray | None,
    Hinv_covar: np.ndarray | None,
    Hinv_target: np.ndarray,
    ridge: float = 1e-6,
) -> np.ndarray:
    """
    Compute P_C * target in H^{-1} metric:
        P_C target = H^{-1} target - H^{-1}C (C^T H^{-1}C)^{-1} C^T H^{-1} target.
    """
    t = np.asarray(Hinv_target, dtype=np.float64).reshape(-1)

    if covar is None or Hinv_covar is None or Hinv_covar.size == 0:
        return t

    C = np.asarray(covar, dtype=np.float64)
    HC = np.asarray(Hinv_covar, dtype=np.float64)
    if C.ndim != 2 or HC.ndim != 2:
        raise ValueError("covar/Hinv_covar must be 2D.")

    if C.shape != HC.shape:
        raise ValueError("covar and Hinv_covar shape mismatch.")
    if C.shape[0] != t.size:
        raise ValueError("Hinv_target length mismatch with covariates.")

    A = C.T @ HC
    if A.size > 0:
        A = 0.5 * (A + A.T)
        A = A + float(ridge) * np.eye(A.shape[0], dtype=np.float64)
    rhs = C.T @ t
    coef = solve_spd(A, rhs)
    return t - HC @ coef


def fit_weighted_lasso_with_covariates(
    y: np.ndarray,
    covar: np.ndarray | None,
    geno: np.ndarray,
    Hinv_y: np.ndarray,
    Hinv_covar: np.ndarray | None,
    Hinv_geno: np.ndarray,
    cfg: LassoPathConfig,
    ridge: float = 1e-6,
    beta_snp_path0: np.ndarray | None = None,
) -> dict:
    """
    Weighted sparse fitting with unpenalized covariates and penalized SNP effects.

    Args:
        y, covar, geno: design matrices in sample order.
        Hinv_*: PCG solves under current variance components.
        beta_snp_path0: Optional same-grid SNP coefficient path used only as
            a warm start for coordinate descent.
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    Z = np.asarray(geno, dtype=np.float64)
    Hy = np.asarray(Hinv_y, dtype=np.float64).reshape(-1)
    HZ = np.asarray(Hinv_geno, dtype=np.float64)

    if Z.ndim != 2:
        raise ValueError("geno must be a 2D matrix.")
    if HZ.shape != Z.shape:
        raise ValueError("geno and Hinv_geno shape mismatch.")
    if y.size != Z.shape[0] or Hy.size != y.size:
        raise ValueError("y/Hinv_y size mismatch with geno rows.")

    n = y.size
    k = Z.shape[1]

    # Compute Gram products in float64 to avoid float32 accumulation error.
    # For n=50k, k=256, float32 matmul can lose ~3 digits of precision.
    yHy = float(y @ Hy)
    profile_yHy = yHy
    gZy = Z.T @ Hy
    GZZ = Z.T @ HZ
    GCC = None
    GCZ = None
    gCy = None

    if covar is None or (covar.size == 0):
        Q = 0.5 * (GZZ + GZZ.T)
        q = gZy
        beta_cov = np.empty((0,), dtype=np.float64)
    else:
        C = np.asarray(covar, dtype=np.float64)
        HC = np.asarray(Hinv_covar, dtype=np.float64)
        if C.ndim != 2 or HC.ndim != 2:
            raise ValueError("covar/Hinv_covar must be 2D.")
        if C.shape != HC.shape:
            raise ValueError("covar and Hinv_covar shape mismatch.")
        if C.shape[0] != n:
            raise ValueError("covar row count mismatch with y.")

        GCC = C.T @ HC
        GCC = 0.5 * (GCC + GCC.T)
        GCC = GCC + float(ridge) * np.eye(GCC.shape[0], dtype=np.float64)

        GCZ = C.T @ HZ
        gCy = C.T @ Hy

        gcc_factor = _factor_linear_system(GCC)
        Ainv_gCy = _solve_factorized_system(gcc_factor, gCy)
        Ainv_GCZ = _solve_factorized_system(gcc_factor, GCZ)

        profile_yHy = float(yHy - gCy @ Ainv_gCy)
        Q = GZZ - GCZ.T @ Ainv_GCZ
        Q = 0.5 * (Q + Q.T)
        q = gZy - GCZ.T @ Ainv_gCy

        beta_cov = np.zeros(C.shape[1], dtype=np.float64)

    if k > 0:
        d = np.diag(Q).copy()
        need = d < 1e-8
        if np.any(need):
            Q[np.diag_indices(k)] += (1e-8 - d) * need

    lasso = solve_lasso_path(
        Q=Q,
        q=q,
        yHy=profile_yHy,
        cfg=cfg,
        beta_path0=beta_snp_path0,
    )

    beta_snp_path = np.asarray(lasso["beta_path"], dtype=np.float64)

    if covar is not None and covar.size > 0:
        if GCC is None or GCZ is None or gCy is None:
            raise RuntimeError("Internal error: covariate normal equations were not built.")
        beta_cov_path = np.ascontiguousarray(
            _solve_factorized_system(
                gcc_factor,
                gCy[:, None] - GCZ @ beta_snp_path.T,
            ).T,
            dtype=np.float64,
        )
    else:
        beta_cov_path = np.empty(
            (beta_snp_path.shape[0], 0), dtype=np.float64
        )

    result = {
        "path": lasso["path"],
        "beta_snp_path": beta_snp_path,
        "beta_cov_path": beta_cov_path,
        "lam_max": float(lasso["lam_max"]),
        "selected_index": lasso["selected_index"],
        "selection_method": lasso["selection_method"],
        "selected_lam_ratio": lasso["selected_lam_ratio"],
        "path_role": lasso["path_role"],
        "requested_n_lambda": lasso["requested_n_lambda"],
        "external_beta_path_warm_start_used": lasso[
            "external_beta_path_warm_start_used"
        ],
        "external_beta_path_warm_start_rows_used": lasso[
            "external_beta_path_warm_start_rows_used"
        ],
    }
    if lasso["selected_index"] is not None:
        selected_index = int(lasso["selected_index"])
        beta_snp = beta_snp_path[selected_index].copy()
        beta_cov = beta_cov_path[selected_index].copy()
        result.update(
            {
                "beta_cov": beta_cov,
                "beta_snp": beta_snp,
                "active_idx": np.flatnonzero(beta_snp != 0.0).astype(
                    np.int64
                ),
                "lam": float(lasso["lam"]),
            }
        )
    return result


__all__ = [
    "LassoPathConfig",
    "make_lambda_sequence",
    "solve_lasso_cd_gram",
    "solve_lasso_path",
    "compute_projected_hinv_vector",
    "fit_weighted_lasso_with_covariates",
]
