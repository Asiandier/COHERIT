"""Streamed REML covariance-score diagnostics used by COHERIT-CovTree."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .covtree import (
    CovTreeCandidate,
    bootstrap_max_score_statistics,
    trace_orthogonal_contrasts,
)
from .geno_stream import _ensure_on_device
from .kv_impl import _device_put_block, _zxb_multi_one_call_jit
from .pcg import pcg_solve


@dataclass
class REMLProjector:
    """Apply the restricted inverse ``P`` for a fixed fitted covariance."""

    fitter: object
    ops: object
    theta: np.ndarray
    covar: np.ndarray | None
    pcg_tol: float
    max_pcg_iters: int

    def __post_init__(self) -> None:
        theta = np.asarray(self.theta, dtype=np.float64).reshape(-1)
        n_components = len(self.ops.K_mvs)
        if theta.shape != (n_components + 1,):
            raise ValueError("theta must contain K genetic values and one residual.")
        self.theta = theta
        self._theta_g = jnp.asarray(theta[:-1], dtype=jnp.float32)
        self._theta_e = jnp.asarray(theta[-1], dtype=jnp.float32)
        self._hv = self.fitter._make_hv(
            self.ops, self._theta_g, self._theta_e
        )
        self._precond = self.fitter._make_effect_precond(
            self.ops, self._theta_g, self._theta_e
        )
        n_samples = int(self.fitter.streamers[0].n)
        if self.covar is None:
            self._covar = np.empty((n_samples, 0), dtype=np.float32)
        else:
            self._covar = np.asarray(self.covar, dtype=np.float32)
            if self._covar.ndim != 2 or self._covar.shape[0] != n_samples:
                raise ValueError("covar must have one row per fitted sample.")
        self._vinv_c = None
        self._covar_gram_inverse = None
        self.solve_diagnostics: list[dict[str, float | int | str]] = []
        if self._covar.shape[1] > 0:
            vinv_c = self._solve(self._covar, stage="covariate_projection")
            gram = self._covar.astype(np.float64).T @ vinv_c.astype(np.float64)
            self._covar_gram_inverse = np.linalg.pinv(
                0.5 * (gram + gram.T), rcond=1e-10, hermitian=True
            )
            self._vinv_c = vinv_c

    def _solve(self, rhs: np.ndarray, *, stage: str) -> np.ndarray:
        rhs_np = np.asarray(rhs, dtype=np.float32)
        squeeze = rhs_np.ndim == 1
        rhs_matrix = rhs_np[:, None] if squeeze else rhs_np
        solution, reported_residual, iterations = pcg_solve(
            self._hv,
            jnp.asarray(rhs_matrix, dtype=jnp.float32),
            M=self._precond,
            tol=float(self.pcg_tol),
            maxiter=int(self.max_pcg_iters),
        )
        solution_np = np.asarray(jax.device_get(solution), dtype=np.float32)
        residual = np.asarray(
            jax.device_get(
                self._hv(jnp.asarray(solution, dtype=jnp.float32))
                - jnp.asarray(rhs_matrix, dtype=jnp.float32)
            ),
            dtype=np.float64,
        )
        denominator = np.maximum(
            np.linalg.norm(rhs_matrix.astype(np.float64), axis=0),
            np.finfo(float).tiny,
        )
        true_relative = float(
            np.max(np.linalg.norm(residual, axis=0) / denominator)
        )
        reported = float(np.max(np.asarray(jax.device_get(reported_residual))))
        iteration_count = int(np.max(np.asarray(jax.device_get(iterations))))
        if (
            not np.isfinite(reported)
            or not np.isfinite(true_relative)
            or reported > max(5.0 * float(self.pcg_tol), 5e-5)
            or true_relative > max(5.0 * float(self.pcg_tol), 5e-5)
        ):
            raise RuntimeError(
                f"CovTree PCG failed at {stage}: reported={reported:.3g}, "
                f"true={true_relative:.3g}, iterations={iteration_count}."
            )
        self.solve_diagnostics.append(
            {
                "stage": stage,
                "rhs_columns": int(rhs_matrix.shape[1]),
                "reported_relative_residual": reported,
                "true_relative_residual": true_relative,
                "iterations": iteration_count,
            }
        )
        return solution_np[:, 0] if squeeze else solution_np

    def apply(self, rhs: np.ndarray, *, stage: str) -> np.ndarray:
        """Return ``P rhs`` with the fixed-effect projection included."""
        rhs_np = np.asarray(rhs, dtype=np.float32)
        squeeze = rhs_np.ndim == 1
        rhs_matrix = rhs_np[:, None] if squeeze else rhs_np
        projected = self._solve(rhs_matrix, stage=stage)
        if self._vinv_c is not None:
            correction_coef = self._covar_gram_inverse @ (
                self._covar.astype(np.float64).T
                @ projected.astype(np.float64)
            )
            projected = projected - self._vinv_c @ correction_coef.astype(np.float32)
        return projected[:, 0] if squeeze else projected

    def fit_covariates(self, target: np.ndarray) -> np.ndarray:
        """Return GLS coefficients for the projector's fixed-effect design."""
        response = np.asarray(target, dtype=np.float32).reshape(-1)
        if self._covar.shape[1] == 0:
            return np.empty((0,), dtype=np.float64)
        vinv_response = self._solve(response, stage="fixed_effect_gls")
        return self._covar_gram_inverse @ (
            self._covar.astype(np.float64).T
            @ vinv_response.astype(np.float64)
        )


def sample_partitioned_null_residuals(
    streamer,
    *,
    theta: np.ndarray,
    n_draws: int,
    seed: int,
) -> np.ndarray:
    """Draw ``N(0, sum(theta_g K_g) + theta_e I)`` without materializing K."""
    if not bool(getattr(streamer, "has_component_partition", False)):
        raise ValueError("CovTree bootstrap requires a component-partitioned streamer.")
    theta_values = np.asarray(theta, dtype=np.float64).reshape(-1)
    if theta_values.shape != (int(streamer.n_components) + 1,):
        raise ValueError("theta does not match the partitioned streamer.")
    if (
        not np.all(np.isfinite(theta_values))
        or np.any(theta_values[:-1] < 0.0)
        or theta_values[-1] <= 0.0
    ):
        raise ValueError("theta contains invalid covariance values.")
    draws = int(n_draws)
    if draws < 3:
        raise ValueError("n_draws must be >= 3.")

    rng = np.random.default_rng(int(seed))
    dev = streamer.dev
    result = jnp.zeros((int(streamer.n), draws), dtype=jnp.float32)
    missing = jnp.asarray(np.uint8(streamer._missing_val), dtype=jnp.uint8)
    if streamer._n_calls > 0:
        streamer._prepare_kv_pass()
        g_dev_next = _device_put_block(streamer._pop_cached(0), dev)
        for call_idx in range(streamer._n_calls):
            g_dev_cur = g_dev_next
            if call_idx + 1 < streamer._n_calls:
                g_dev_next = _device_put_block(
                    streamer._pop_cached(call_idx + 1), dev
                )
            component_index = int(streamer._call_component_ids[call_idx])
            effective_m = float(streamer._component_eff_m_host[component_index])
            scale = (
                np.sqrt(theta_values[component_index] / effective_m)
                if effective_m > 0.0 and theta_values[component_index] > 0.0
                else 0.0
            )
            width = int(streamer._call_true_widths[call_idx])
            coefficients = np.zeros(
                (streamer._max_unpack_width, draws), dtype=np.float32
            )
            coefficients[:width] = (
                rng.standard_normal((width, draws), dtype=np.float32)
                * np.float32(scale)
            )
            result = result + _zxb_multi_one_call_jit(
                g_dev_cur,
                streamer._true_widths_dev[call_idx],
                streamer._means_by_call[call_idx],
                streamer._inv_by_call[call_idx],
                jax.device_put(jnp.asarray(coefficients), dev),
                missing,
            )
            del g_dev_cur
    residual_noise = rng.standard_normal(
        (int(streamer.n), draws), dtype=np.float32
    ) * np.float32(np.sqrt(theta_values[-1]))
    result = result + jax.device_put(jnp.asarray(residual_noise), dev)
    return np.asarray(jax.device_get(result), dtype=np.float32)


def _source_cache_maps(grm_index) -> tuple[np.ndarray, np.ndarray]:
    cache_indices = np.arange(grm_index.m_total, dtype=np.int64)
    cache_to_source = np.asarray(
        grm_index.source_variant_indices(cache_indices), dtype=np.int64
    )
    if not np.array_equal(
        np.sort(cache_to_source),
        np.arange(grm_index.m_total, dtype=np.int64),
    ):
        raise ValueError("CovTree requires a one-to-one source/cache marker map.")
    source_to_cache = np.empty_like(cache_to_source)
    source_to_cache[cache_to_source] = cache_indices
    return cache_to_source, source_to_cache


def _candidate_contrast_definitions(
    streamer,
    grm_index,
    candidates: Sequence[CovTreeCandidate],
) -> tuple[list[dict[str, object]], list[tuple[int, int]], list[dict[str, object]]]:
    _, source_to_cache = _source_cache_maps(grm_index)
    inv_sd = np.asarray(streamer._inv_sds_host, dtype=np.float64)
    valid = inv_sd > 0.0
    if streamer._count_host is None:
        trace_mass = np.full(int(streamer.m), int(streamer.n), dtype=np.float64)
    else:
        trace_mass = np.asarray(streamer._count_host, dtype=np.float64)

    definitions: list[dict[str, object]] = []
    slices: list[tuple[int, int]] = []
    metadata: list[dict[str, object]] = []
    atom_cursor = 0
    for candidate in candidates:
        child_cache = [
            source_to_cache[np.asarray(child, dtype=np.int64)]
            for child in candidate.children
        ]
        child_effective = np.asarray(
            [np.count_nonzero(valid[indices]) for indices in child_cache],
            dtype=np.float64,
        )
        if np.any(child_effective <= 0.0):
            raise ValueError(f"Candidate {candidate.name} contains an empty effective child.")
        child_trace = np.asarray(
            [
                np.sum(trace_mass[indices][valid[indices]], dtype=np.float64)
                / (float(streamer.n) * child_effective[child_index])
                for child_index, indices in enumerate(child_cache)
            ],
            dtype=np.float64,
        )
        contrasts = trace_orthogonal_contrasts(child_trace)
        start = atom_cursor
        definitions.append(
            {
                "child_cache_indices": child_cache,
                "child_effective_markers": child_effective,
                "contrasts": contrasts,
            }
        )
        atom_cursor += int(contrasts.shape[1])
        slices.append((start, atom_cursor))
        metadata.append(
            {
                "name": candidate.name,
                "parent_index": int(candidate.parent_index),
                "parent_name": candidate.parent_name,
                "split_kind": candidate.split_kind,
                "child_sizes": [int(child.size) for child in candidate.children],
                "child_effective_markers": child_effective.astype(int).tolist(),
                "child_trace_atoms": child_trace.tolist(),
                "contrast_coefficients": contrasts.tolist(),
                "diagnostics": candidate.diagnostics,
            }
        )
    return definitions, slices, metadata


def evaluate_covtree_candidates(
    *,
    fitter,
    ops,
    grm_index,
    candidates: Sequence[CovTreeCandidate],
    theta: np.ndarray,
    covar: np.ndarray | None,
    residual: np.ndarray,
    bootstrap_draws: int,
    seed: int,
    alpha: float,
    rank_rtol: float,
    pcg_tol: float,
    max_pcg_iters: int,
) -> tuple[dict[str, object], CovTreeCandidate | None, dict[str, np.ndarray]]:
    """Evaluate candidates and return inference, the split, and marker scores."""
    if len(fitter.streamers) != 1:
        raise ValueError("CovTree currently requires one dense genotype source.")
    streamer = fitter.streamers[0]
    residual = np.asarray(residual, dtype=np.float32).reshape(-1)
    if residual.shape != (int(streamer.n),) or not np.all(np.isfinite(residual)):
        raise ValueError("residual is malformed.")
    significance = float(alpha)
    if not np.isfinite(significance) or not 0.0 < significance < 1.0:
        raise ValueError("alpha must lie in (0, 1).")
    candidate_definitions, relative_slices, candidate_metadata = (
        _candidate_contrast_definitions(
            streamer, grm_index, candidates
        )
    )

    bootstrap_residual = sample_partitioned_null_residuals(
        streamer,
        theta=theta,
        n_draws=int(bootstrap_draws),
        seed=int(seed),
    )
    projector = REMLProjector(
        fitter=fitter,
        ops=ops,
        theta=np.asarray(theta, dtype=np.float64),
        covar=covar,
        pcg_tol=float(pcg_tol),
        max_pcg_iters=int(max_pcg_iters),
    )
    projected = projector.apply(
        np.concatenate([residual[:, None], bootstrap_residual], axis=1),
        stage="observed_and_parametric_bootstrap",
    )
    projected_dev = _ensure_on_device(
        jnp.asarray(projected, dtype=jnp.float32), streamer.dev
    )
    marker_projection = np.asarray(
        jax.device_get(grm_index.xtv_all(projected_dev, normalize=False)),
        dtype=np.float32,
    )
    if marker_projection.shape != (grm_index.m_total, projected.shape[1]):
        raise RuntimeError("CovTree marker projection has an unexpected shape.")
    if marker_projection.shape[1] != int(bootstrap_draws) + 1:
        raise RuntimeError("CovTree marker projection/bootstrap count mismatch.")
    projected_numerator = np.asarray(marker_projection[:, 0], dtype=np.float64)
    projected_information = np.einsum(
        "ij,ij->i",
        marker_projection[:, 1:],
        marker_projection[:, 1:],
        dtype=np.float64,
        optimize=True,
    ) / float(bootstrap_draws)
    covariance_score = 0.5 * (
        np.square(projected_numerator) - projected_information
    )
    if (
        not np.all(np.isfinite(projected_information))
        or np.any(projected_information < 0.0)
        or not np.all(np.isfinite(covariance_score))
    ):
        raise RuntimeError("CovTree bootstrap marker scores are invalid.")
    marker_scores = {
        "covariance_score": covariance_score.astype(np.float32),
        "projected_score_numerator": projected_numerator.astype(np.float32),
        "projected_information_diagonal": projected_information.astype(np.float32),
    }
    valid = np.asarray(streamer._inv_sds_host, dtype=np.float64) > 0.0
    atom_count = sum(
        int(definition["contrasts"].shape[1])
        for definition in candidate_definitions
    )
    quadratics = np.empty((atom_count, projected.shape[1]), dtype=np.float64)
    atom_cursor = 0
    for definition in candidate_definitions:
        child_quadratics = []
        for indices, effective in zip(
            definition["child_cache_indices"],
            definition["child_effective_markers"],
            strict=True,
        ):
            child_valid_indices = indices[valid[indices]]
            values = marker_projection[child_valid_indices]
            child_quadratics.append(
                np.einsum(
                    "ij,ij->j", values, values, dtype=np.float64, optimize=True
                )
                / float(effective)
            )
        child_matrix = np.stack(child_quadratics, axis=0)
        contrast_quadratics = (
            np.asarray(definition["contrasts"], dtype=np.float64).T
            @ child_matrix
        )
        stop = atom_cursor + int(contrast_quadratics.shape[0])
        quadratics[atom_cursor:stop] = contrast_quadratics
        atom_cursor = stop

    inference = bootstrap_max_score_statistics(
        observed_quadratics=quadratics[:, 0],
        bootstrap_quadratics=quadratics[:, 1:],
        candidate_slices=relative_slices,
        rank_rtol=float(rank_rtol),
    )
    for metadata, result in zip(
        candidate_metadata, inference["candidates"], strict=True
    ):
        metadata.update(result)

    best_index = inference["best_candidate_index"]
    adjusted_p = inference["max_score_adjusted_p"]
    accepted = bool(
        best_index is not None
        and adjusted_p is not None
        and float(adjusted_p) <= significance
    )
    selected = candidates[int(best_index)] if accepted else None
    summary = {
        "method": "covariance_contrast_parametric_max_bootstrap",
        "bootstrap_generation": "streamed_gaussian_from_fitted_partitioned_covariance",
        "trace_estimator": "parametric_bootstrap_mean_quadratic",
        "information_estimator": "parametric_bootstrap_score_covariance",
        "alpha": significance,
        "accepted": accepted,
        "selected_candidate_index": int(best_index) if accepted else None,
        "best_candidate_index": int(best_index) if best_index is not None else None,
        "best_candidate_name": (
            candidates[int(best_index)].name if best_index is not None else None
        ),
        "max_score_adjusted_p": adjusted_p,
        "max_score_adjusted_p_mc_se": inference[
            "max_score_adjusted_p_mc_se"
        ],
        "bootstrap_draws": int(bootstrap_draws),
        "rank_rtol": float(rank_rtol),
        "candidate_count": len(candidates),
        "candidate_contrast_count": int(atom_count),
        "quadratic_backend": "single_Xt_Pe_pass_then_marker_group_reduction",
        "candidates": candidate_metadata,
        "pcg": projector.solve_diagnostics,
    }
    return summary, selected, marker_scores


__all__ = [
    "REMLProjector",
    "evaluate_covtree_candidates",
    "sample_partitioned_null_residuals",
]
