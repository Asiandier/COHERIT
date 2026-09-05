"""Adaptive LD-rank covariance splitting for sparse COHERIT.

This module contains only the selected K=1 -> fixed-alpha LD-CUSUM -> endpoint
refit algorithm.  Candidate boundaries are deterministic balanced cuts of one
global LD-score ordering.  At each layer an efficient REML score is adjusted
for the current GRM components and residual variance, and a parametric
bootstrap calibrates the maximum over all still-available boundaries.

The user-facing orchestration lives in :mod:`GPU_REML.run_sparse_pipeline`.
The small command-line interface here is internal: separate processes keep GPU
memory bounded between the validation fit, score, and covariance-only refit.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Sequence

from .runtime_env import configure_runtime_env

configure_runtime_env()

import jax
import jax.numpy as jnp
import numpy as np

from .component_spec import load_component_specs
from .data_utils import load_pheno_covar_aligned_with_transform
from .geno_source import PgenGenoSource
from .kv_impl import _device_put_block, _zxb_multi_one_call_jit
from .pcg import pcg_solve
from .pipeline_common import (
    cleanup_path,
    compute_sample_mask,
    make_nonbed_input_fam,
    read_keep_ids,
    resolve_cpu_threads,
    run_planner,
    setup_gpu,
)
from .reml_model import FitConfig, InfinitesimalREMLFitter, standardize_response
from .run_sparse_reml_pipeline import (
    MultiGRMIndex,
    _accepted_reml_theta,
    _validate_component_partition,
)
from .variant_io import iter_variant_records_for_prefix


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def atomic_json(path: str | Path, value: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)


def load_ld_rank(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        order = np.asarray(payload["source_order"], dtype=np.int64).reshape(-1)
        positions = np.asarray(
            payload["boundary_positions"], dtype=np.int64
        ).reshape(-1)
        marker_count = int(np.asarray(payload["marker_count"]).reshape(()))
    if (
        order.size != marker_count
        or not np.array_equal(np.sort(order), np.arange(marker_count))
        or positions.size < 1
        or np.any(positions <= 0)
        or np.any(positions >= marker_count)
        or np.any(np.diff(positions) <= 0)
    ):
        raise ValueError(f"Invalid LD-rank artifact: {path}")
    return order, positions


def load_component_groups(path: str | Path, marker_count: int) -> list[np.ndarray]:
    groups = [
        np.asarray(spec.variant_indices, dtype=np.int64).reshape(-1)
        for spec in load_component_specs(str(path))
    ]
    return _validate_component_partition(groups, n_markers=int(marker_count))


def write_component_spec(
    path: str | Path,
    groups: Sequence[np.ndarray],
    *,
    provenance: dict[str, Any],
) -> None:
    output = Path(path)
    normalized = [np.sort(np.asarray(group, dtype=np.int64)) for group in groups]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}.npz")
    annotations = [
        json.dumps(
            {
                "feature": "ld_score_rank",
                "interval_index": index,
                "markers": int(group.size),
            },
            sort_keys=True,
        )
        for index, group in enumerate(normalized)
    ]
    provenance_values = [json.dumps(provenance, sort_keys=True)] * len(normalized)
    np.savez_compressed(
        temporary,
        **{f"arr_{index}": group for index, group in enumerate(normalized)},
        component_names=np.asarray(
            [f"ld_interval_{index:03d}" for index in range(len(normalized))]
        ),
        component_annotations_json=np.asarray(annotations),
        component_provenance_json=np.asarray(provenance_values),
    )
    os.replace(temporary, output)


def write_root_component_spec(
    path: str | Path,
    *,
    marker_count: int,
    rank_path: str | Path,
) -> None:
    if marker_count < 2:
        raise ValueError("Adaptive covariance splitting requires at least two SNPs.")
    write_component_spec(
        path,
        [np.arange(marker_count, dtype=np.int64)],
        provenance={
            "algorithm": "adaptive_ld_cusum",
            "stage": "root_single_grm",
            "ld_rank_artifact": str(Path(rank_path).resolve()),
        },
    )


def _rank_intervals(
    groups: Sequence[np.ndarray], source_order: np.ndarray
) -> list[tuple[int, int]]:
    marker_count = int(source_order.size)
    source_to_rank = np.empty(marker_count, dtype=np.int64)
    source_to_rank[source_order] = np.arange(marker_count, dtype=np.int64)
    intervals: list[tuple[int, int]] = []
    for group in groups:
        ranks = np.sort(source_to_rank[np.asarray(group, dtype=np.int64)])
        start = int(ranks[0])
        stop = int(ranks[-1]) + 1
        if ranks.size != stop - start or not np.array_equal(
            ranks, np.arange(start, stop, dtype=np.int64)
        ):
            raise ValueError("Current adaptive components are not LD-rank intervals.")
        intervals.append((start, stop))
    if sorted(intervals) != intervals:
        raise ValueError("Adaptive components are not ordered by LD rank.")
    if intervals[0][0] != 0 or intervals[-1][1] != marker_count:
        raise ValueError("Adaptive components do not cover the LD-rank root.")
    for left, right in zip(intervals[:-1], intervals[1:], strict=True):
        if left[1] != right[0]:
            raise ValueError("Adaptive LD-rank intervals are not contiguous.")
    return intervals


def existing_boundary_positions(
    groups: Sequence[np.ndarray], source_order: np.ndarray
) -> np.ndarray:
    intervals = _rank_intervals(groups, source_order)
    return np.asarray([stop for _start, stop in intervals[:-1]], dtype=np.int64)


def add_ld_boundary(
    *,
    parent_spec: str | Path,
    parent_theta: Sequence[float],
    rank_path: str | Path,
    boundary_position: int,
    output_spec: str | Path,
) -> dict[str, Any]:
    source_order, available_positions = load_ld_rank(rank_path)
    marker_count = int(source_order.size)
    groups = load_component_groups(parent_spec, marker_count)
    theta = np.asarray(parent_theta, dtype=np.float64).reshape(-1)
    if theta.shape != (len(groups) + 1,):
        raise ValueError("Parent variance components do not align with its partition.")
    position = int(boundary_position)
    if position not in set(available_positions.tolist()):
        raise ValueError("Selected boundary is absent from the balanced LD grid.")
    current = set(existing_boundary_positions(groups, source_order).tolist())
    if position in current:
        raise ValueError("Selected LD boundary is already present.")

    prefix = np.zeros(marker_count, dtype=bool)
    prefix[source_order[:position]] = True
    split_index = -1
    children: tuple[np.ndarray, np.ndarray] | None = None
    for index, group in enumerate(groups):
        low = group[prefix[group]]
        high = group[~prefix[group]]
        if low.size and high.size:
            if children is not None:
                raise RuntimeError("One LD boundary split multiple components.")
            split_index = index
            children = (np.sort(low), np.sort(high))
    if children is None:
        raise ValueError("Selected boundary does not refine the current partition.")

    low, high = children
    parent_size = float(groups[split_index].size)
    low_fraction = float(low.size / parent_size)
    new_groups = groups[:split_index] + [low, high] + groups[split_index + 1 :]
    new_theta = np.concatenate(
        [
            theta[:split_index],
            np.asarray(
                [
                    theta[split_index] * low_fraction,
                    theta[split_index] * (1.0 - low_fraction),
                ]
            ),
            theta[split_index + 1 :],
        ]
    )
    if not np.isclose(np.sum(new_theta[:-1]), np.sum(theta[:-1])):
        raise RuntimeError("Variance-preserving child initialization failed.")
    write_component_spec(
        output_spec,
        new_groups,
        provenance={
            "algorithm": "adaptive_ld_cusum",
            "parent_spec": str(Path(parent_spec).resolve()),
            "ld_rank_artifact": str(Path(rank_path).resolve()),
            "added_boundary_position": position,
        },
    )
    return {
        "component_spec": str(Path(output_spec).resolve()),
        "split_parent_component": int(split_index),
        "boundary_position": position,
        "low_fraction": low_fraction,
        "component_sizes": [int(group.size) for group in new_groups],
        "variance_components_init": new_theta.tolist(),
    }


def efficient_score_statistics(
    quadratics: np.ndarray,
    *,
    nuisance_count: int,
    boundary_positions: np.ndarray,
) -> tuple[list[dict[str, float | int]], dict[str, Any]]:
    """Nuisance-adjust candidate scores and bootstrap the global maximum."""
    values = np.asarray(quadratics, dtype=np.float64)
    positions = np.asarray(boundary_positions, dtype=np.int64).reshape(-1)
    if values.ndim != 2 or values.shape[1] < 20:
        raise ValueError("Score calibration requires observed data plus >=19 draws.")
    if values.shape[0] != nuisance_count + positions.size:
        raise ValueError("Quadratic rows do not match nuisance and candidate counts.")
    if nuisance_count < 1 or not np.all(np.isfinite(values)):
        raise ValueError("Score calibration requires finite quadratics and nuisance rows.")
    bootstrap_mean = np.mean(values[:, 1:], axis=1)
    scores = 0.5 * (values - bootstrap_mean[:, None])
    centered = scores[:, 1:] - np.mean(scores[:, 1:], axis=1, keepdims=True)
    denominator = float(centered.shape[1] - 1)
    nuisance_centered = centered[:nuisance_count]
    nuisance_information = nuisance_centered @ nuisance_centered.T / denominator
    nuisance_information = 0.5 * (nuisance_information + nuisance_information.T)
    nuisance_inverse = np.linalg.pinv(
        nuisance_information, rcond=1e-8, hermitian=True
    )

    rows: list[dict[str, float | int]] = []
    statistic_rows: list[np.ndarray] = []
    for offset, boundary in enumerate(positions.tolist()):
        index = nuisance_count + offset
        candidate_centered = centered[index]
        cross = candidate_centered @ nuisance_centered.T / denominator
        raw_information = float(candidate_centered @ candidate_centered / denominator)
        information = float(
            raw_information
            - cross @ nuisance_inverse @ cross.T
        )
        # A contrast in the nuisance span is not testable. The relative
        # roundoff guard also removes tiny positive Schur complements caused
        # by cancellation; it is not a statistical significance threshold.
        information_roundoff = (
            64.0 * np.finfo(np.float64).eps * max(1, nuisance_count)
            * raw_information
        )
        if not np.isfinite(information):
            raise FloatingPointError("Non-finite efficient score information.")
        if information <= information_roundoff:
            continue
        efficient = np.asarray(
            scores[index] - cross @ nuisance_inverse @ scores[:nuisance_count],
            dtype=np.float64,
        )
        statistics = np.square(efficient) / information
        if not np.all(np.isfinite(statistics)):
            raise FloatingPointError("Non-finite efficient score statistic.")
        statistic_rows.append(statistics)
        rows.append(
            {
                "boundary_position": int(boundary),
                "raw_score": float(scores[index, 0]),
                "efficient_score": float(efficient[0]),
                "efficient_information": information,
                "score_statistic": float(statistics[0]),
            }
        )
    rows.sort(key=lambda row: float(row["score_statistic"]), reverse=True)
    # The empty family must stop splitting, never acquire the minimum Monte
    # Carlo p-value through comparisons against NaN.
    maximum = (
        np.max(np.stack(statistic_rows, axis=0), axis=0)
        if statistic_rows else np.zeros(values.shape[1], dtype=np.float64)
    )
    observed_maximum = float(maximum[0])
    bootstrap_maximum = np.asarray(maximum[1:], dtype=np.float64)
    global_p = float(
        (1 + np.count_nonzero(bootstrap_maximum >= observed_maximum))
        / (bootstrap_maximum.size + 1)
    )
    diagnostics = {
        "testable_candidate_count": len(rows),
        "global_sup_score": observed_maximum,
        "global_sup_score_p_value": global_p,
        "bootstrap_max_quantiles": {
            str(probability): float(np.quantile(bootstrap_maximum, probability))
            for probability in (0.5, 0.9, 0.95, 0.99)
        },
        "nuisance_score": scores[:nuisance_count, 0].tolist(),
        "nuisance_information_eigenvalues": np.linalg.eigvalsh(
            nuisance_information
        ).tolist(),
    }
    return rows, diagnostics


@dataclass
class REMLProjector:
    fitter: Any
    ops: Any
    theta: np.ndarray
    covar: np.ndarray | None
    pcg_tol: float
    max_pcg_iters: int

    def __post_init__(self) -> None:
        theta = np.asarray(self.theta, dtype=np.float64).reshape(-1)
        if theta.shape != (len(self.ops.K_mvs) + 1,):
            raise ValueError("Projector theta does not align with covariance operators.")
        self.theta = theta
        theta_device = jnp.asarray(theta, dtype=jnp.float32)
        self._hv = self.fitter._make_hv(
            self.ops, theta_device[:-1], theta_device[-1]
        )
        self._preconditioner = self.fitter._make_effect_precond(
            self.ops, theta_device[:-1], theta_device[-1]
        )
        n_samples = int(self.fitter.streamers[0].n)
        self._covar = (
            np.empty((n_samples, 0), dtype=np.float32)
            if self.covar is None
            else np.asarray(self.covar, dtype=np.float32)
        )
        if self._covar.ndim != 2 or self._covar.shape[0] != n_samples:
            raise ValueError("Covariates do not align with the fitted samples.")
        self._vinv_c: np.ndarray | None = None
        self._gram_inverse: np.ndarray | None = None
        self.solve_diagnostics: list[dict[str, float | int | str]] = []
        if self._covar.shape[1]:
            self._vinv_c = self._solve(self._covar, stage="covariate_projection")
            gram = self._covar.astype(np.float64).T @ self._vinv_c.astype(np.float64)
            self._gram_inverse = np.linalg.pinv(
                0.5 * (gram + gram.T), rcond=1e-10, hermitian=True
            )

    def _solve(self, rhs: np.ndarray, *, stage: str) -> np.ndarray:
        array = np.asarray(rhs, dtype=np.float32)
        squeeze = array.ndim == 1
        matrix = array[:, None] if squeeze else array
        solution, reported, iterations = pcg_solve(
            self._hv,
            jnp.asarray(matrix, dtype=jnp.float32),
            M=self._preconditioner,
            tol=float(self.pcg_tol),
            maxiter=int(self.max_pcg_iters),
        )
        host = np.asarray(jax.device_get(solution), dtype=np.float32)
        residual = np.asarray(
            jax.device_get(self._hv(solution) - jnp.asarray(matrix)),
            dtype=np.float64,
        )
        denominator = np.maximum(
            np.linalg.norm(matrix.astype(np.float64), axis=0),
            np.finfo(float).tiny,
        )
        true_relative = float(np.max(np.linalg.norm(residual, axis=0) / denominator))
        reported_relative = float(np.max(np.asarray(jax.device_get(reported))))
        iteration_count = int(np.max(np.asarray(jax.device_get(iterations))))
        limit = max(5.0 * float(self.pcg_tol), 5e-5)
        if (
            not np.isfinite(reported_relative)
            or not np.isfinite(true_relative)
            or reported_relative > limit
            or true_relative > limit
        ):
            raise RuntimeError(
                f"Adaptive score PCG failed at {stage}: reported={reported_relative:.3g}, "
                f"true={true_relative:.3g}."
            )
        self.solve_diagnostics.append(
            {
                "stage": stage,
                "rhs_columns": int(matrix.shape[1]),
                "reported_relative_residual": reported_relative,
                "true_relative_residual": true_relative,
                "iterations": iteration_count,
            }
        )
        return host[:, 0] if squeeze else host

    def apply(self, rhs: np.ndarray, *, stage: str) -> np.ndarray:
        array = np.asarray(rhs, dtype=np.float32)
        squeeze = array.ndim == 1
        matrix = array[:, None] if squeeze else array
        projected = self._solve(matrix, stage=stage)
        if self._vinv_c is not None and self._gram_inverse is not None:
            coefficients = self._gram_inverse @ (
                self._covar.astype(np.float64).T @ projected.astype(np.float64)
            )
            projected = projected - self._vinv_c @ coefficients.astype(np.float32)
        return projected[:, 0] if squeeze else projected

def sample_partitioned_null_residuals(
    streamer: Any,
    *,
    theta: np.ndarray,
    n_draws: int,
    seed: int,
) -> np.ndarray:
    if not bool(getattr(streamer, "has_component_partition", False)):
        raise ValueError("Adaptive bootstrap requires a partitioned streamer.")
    values = np.asarray(theta, dtype=np.float64).reshape(-1)
    if values.shape != (int(streamer.n_components) + 1,):
        raise ValueError("Bootstrap theta does not align with the component partition.")
    if np.any(values[:-1] < 0.0) or values[-1] <= 0.0:
        raise ValueError("Bootstrap variance components are invalid.")
    draws = int(n_draws)
    if draws < 19:
        raise ValueError("Adaptive bootstrap requires at least 19 draws.")

    rng = np.random.default_rng(int(seed))
    result = jnp.zeros((int(streamer.n), draws), dtype=jnp.float32)
    missing = jnp.asarray(np.uint8(streamer._missing_val), dtype=jnp.uint8)
    if streamer._n_calls > 0:
        streamer._prepare_kv_pass()
        next_block = _device_put_block(streamer._pop_cached(0), streamer.dev)
        for call_index in range(streamer._n_calls):
            current_block = next_block
            if call_index + 1 < streamer._n_calls:
                next_block = _device_put_block(
                    streamer._pop_cached(call_index + 1), streamer.dev
                )
            component = int(streamer._call_component_ids[call_index])
            effective_m = float(streamer._component_eff_m_host[component])
            scale = (
                np.sqrt(values[component] / effective_m)
                if effective_m > 0.0 and values[component] > 0.0
                else 0.0
            )
            width = int(streamer._call_true_widths[call_index])
            coefficients = np.zeros(
                (streamer._max_unpack_width, draws), dtype=np.float32
            )
            coefficients[:width] = rng.standard_normal(
                (width, draws), dtype=np.float32
            ) * np.float32(scale)
            result = result + _zxb_multi_one_call_jit(
                current_block,
                streamer._true_widths_dev[call_index],
                streamer._means_by_call[call_index],
                streamer._inv_by_call[call_index],
                jax.device_put(jnp.asarray(coefficients), streamer.dev),
                missing,
            )
            del current_block
    noise = rng.standard_normal(
        (int(streamer.n), draws), dtype=np.float32
    ) * np.float32(np.sqrt(values[-1]))
    result = result + jax.device_put(jnp.asarray(noise), streamer.dev)
    return np.asarray(jax.device_get(result), dtype=np.float32)


@dataclass
class AnalysisContext:
    fitter: InfinitesimalREMLFitter
    grm_index: MultiGRMIndex
    y: np.ndarray
    covar: np.ndarray
    groups: list[np.ndarray]
    temporary_fam: str | None

    def close(self) -> None:
        self.fitter.close()
        if self.temporary_fam:
            cleanup_path(self.temporary_fam)


def _standardize(y: np.ndarray) -> np.ndarray:
    standardized, _mean, _scale = standardize_response(
        jnp.asarray(np.asarray(y, dtype=np.float32), dtype=jnp.float32)
    )
    return np.asarray(jax.device_get(standardized), dtype=np.float32)


def build_analysis_context(args: argparse.Namespace) -> AnalysisContext:
    genotype_format = "bed" if args.bed_prefix else "pgen"
    prefix = args.bed_prefix or args.pgen_prefix
    temporary_fam: str | None = None
    fam_path = prefix + ".fam"
    if genotype_format == "pgen":
        temporary_fam = make_nonbed_input_fam(pgen_prefix=prefix)
        fam_path = temporary_fam
    keep_ids = read_keep_ids(args.keep_path)
    y, covar, kept_ids, _dropped, _transform = (
        load_pheno_covar_aligned_with_transform(
            fam_path=fam_path,
            pheno_path=args.pheno_txt,
            covar_path=args.covar_txt or None,
            add_intercept=True,
            keep_ids=keep_ids,
        )
    )
    y = _standardize(y)
    covar = np.asarray(covar, dtype=np.float32)
    marker_count_source = sum(
        1 for _ in iter_variant_records_for_prefix(prefix, genotype_format)
    )
    groups = load_component_groups(args.component_spec, marker_count_source)
    sample_mask = compute_sample_mask(fam_path, kept_ids)
    sources = None
    if genotype_format == "pgen":
        sources = [PgenGenoSource(prefix, sample_mask=sample_mask)]
        sample_mask = None
        if int(sources[0].m) != marker_count_source:
            raise ValueError("PGEN and PVAR variant counts differ.")
    if marker_count_source != sum(group.size for group in groups):
        raise ValueError("Component spec marker count differs from genotype source.")

    gpu_name, _gpu_total, gpu_free = setup_gpu()
    cpu_threads, _source = resolve_cpu_threads(args.cpu_threads or None)
    gpu_budget = (
        float(args.gpu_budget_gib) * 1024**3
        if float(args.gpu_budget_gib) > 0.0
        else None
    )
    plan = run_planner(
        n_samples=int(y.size),
        p_list=[marker_count_source],
        n_grm=len(groups),
        component_block_sizes=[int(group.size) for group in groups],
        gpu_free=gpu_free,
        gpu_budget=gpu_budget,
        n_covar=int(covar.shape[1]),
        n_rand_vec=int(args.n_rand_vec),
        slq_samples=int(args.slq_samples),
        gpu_name=gpu_name,
        source_format=genotype_format,
        arbitrary_component_partition=True,
        ring_depth=(int(args.ring_depth) if int(args.ring_depth) > 0 else None),
        requested_call_width=(int(args.call_width) if int(args.call_width) > 0 else None),
    )
    config = dict(
        device=args.device,
        sample_mask=sample_mask,
        component_variant_indices=groups,
        call_width=plan.call_width,
        cpu_threads=cpu_threads,
        keep_host_stats=True,
        gpu_budget_bytes=(
            gpu_budget
            if gpu_budget is not None
            else float(plan.gpu_budget_gib) * 1024**3
        ),
        ring_depth=plan.ring_depth,
        n_rand_vec=int(args.n_rand_vec),
        minq_iter=int(args.minq_iter),
        slq_samples=int(args.slq_samples),
        slq_m=int(args.slq_m),
        precond_rank=plan.precond_rank,
        reml_pcg_tol=float(args.pcg_tol),
        response_is_standardized=True,
        unit_variance_components=True,
        max_pcg_iters=int(args.max_pcg_iters),
        pcg_ridge=float(args.pcg_ridge),
        capture_reml_diagnostics=True,
        strict_max_linesearch_trials=int(args.reml_max_linesearch_trials),
        verbose=bool(args.verbose),
    )
    fitter = InfinitesimalREMLFitter(
        FitConfig(
            sources=sources,
            **config,
        )
        if sources is not None
        else FitConfig(bed_prefix=prefix, **config)
    )
    return AnalysisContext(
        fitter=fitter,
        grm_index=MultiGRMIndex(
            fitter.streamers, component_variant_indices=groups
        ),
        y=y,
        covar=covar,
        groups=groups,
        temporary_fam=temporary_fam,
    )


def _load_sparse_mean(
    state_path: str | Path,
    grm_index: MultiGRMIndex,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(state_path, allow_pickle=False) as state:
        source_markers = np.asarray(state["marker_indices"], dtype=np.int64)
        beta = np.asarray(state["selected_beta_snp"], dtype=np.float64)
    cache_markers = grm_index.cache_variant_indices(source_markers)
    sparse_mean = (
        grm_index.extract_standardized_columns(cache_markers).astype(np.float64)
        @ beta
    )
    return source_markers, beta, sparse_mean


def run_score(args: argparse.Namespace) -> None:
    context = build_analysis_context(args)
    try:
        summary = read_json(args.summary_path)
        theta = np.asarray(summary["var_components_lasso_ml"], dtype=np.float64)
        if theta.shape != (len(context.groups) + 1,):
            raise ValueError("Score parent theta does not align with its partition.")
        ops = context.fitter._assemble_reml_operators()
        context.fitter._ensure_projected_core_precond_ready(
            ops, var_components_init=jnp.asarray(theta, dtype=jnp.float32)
        )
        projector = REMLProjector(
            fitter=context.fitter,
            ops=ops,
            theta=theta,
            covar=context.covar,
            pcg_tol=float(args.pcg_tol),
            max_pcg_iters=int(args.max_pcg_iters),
        )
        _markers, _beta, sparse_mean = _load_sparse_mean(
            args.state_path, context.grm_index
        )
        # P_theta already profiles every column of C because P_theta C = 0.
        # Subtracting a separately fitted C gamma before applying the same
        # projector is algebraically redundant and adds an unnecessary PCG
        # solve.  Keep only the frozen sparse mean outside the projector.
        working_response = (
            context.y.astype(np.float64) - sparse_mean
        ).astype(np.float32)
        bootstrap = sample_partitioned_null_residuals(
            context.fitter.streamers[0],
            theta=theta,
            n_draws=int(args.bootstrap_draws),
            seed=int(args.bootstrap_seed),
        )
        projected = projector.apply(
            np.concatenate([working_response[:, None], bootstrap], axis=1),
            stage="adaptive_ld_observed_and_bootstrap",
        )
        marker_projection = np.asarray(
            context.grm_index.xtv_all(
                jnp.asarray(projected, dtype=jnp.float32), normalize=False
            ),
            dtype=np.float64,
        )
        nuisance = [
            np.mean(
                np.square(marker_projection[int(start) : int(stop)]), axis=0
            )
            for start, stop in zip(
                context.grm_index.offsets[:-1],
                context.grm_index.offsets[1:],
                strict=True,
            )
        ]
        nuisance.append(np.einsum("ij,ij->j", projected, projected, optimize=True))

        source_order, grid_positions = load_ld_rank(args.rank_path)
        current_positions = existing_boundary_positions(context.groups, source_order)
        candidates = np.setdiff1d(
            grid_positions, current_positions, assume_unique=True
        )
        if not candidates.size:
            payload = {
                "schema_version": 1,
                "accepted": False,
                "stop_reason": "no_available_boundary",
                "current_k": len(context.groups),
                "candidates": [],
                "diagnostics": {"global_sup_score_p_value": 1.0},
            }
            atomic_json(args.out, payload)
            return

        cache_order = context.grm_index.cache_variant_indices(source_order)
        grid_edges = np.concatenate([[0], grid_positions, [source_order.size]])
        bin_sums: list[np.ndarray] = []
        bin_sizes: list[int] = []
        for start, stop in zip(grid_edges[:-1], grid_edges[1:], strict=True):
            indices = cache_order[int(start) : int(stop)]
            bin_sums.append(np.sum(np.square(marker_projection[indices]), axis=0))
            bin_sizes.append(int(indices.size))
        cumulative_sums = np.cumsum(np.stack(bin_sums, axis=0), axis=0)
        cumulative_sizes = np.cumsum(np.asarray(bin_sizes, dtype=np.int64))
        total_sum = cumulative_sums[-1]
        total_size = int(cumulative_sizes[-1])
        grid_index_by_position = {
            int(position): index
            for index, position in enumerate(grid_positions.tolist())
        }
        candidate_quadratics: list[np.ndarray] = []
        for position in candidates.tolist():
            boundary_index = grid_index_by_position[int(position)]
            low_size = int(cumulative_sizes[boundary_index])
            high_size = total_size - low_size
            low = cumulative_sums[boundary_index] / low_size
            high = (total_sum - cumulative_sums[boundary_index]) / high_size
            candidate_quadratics.append(low - high)
        quadratics = np.stack(nuisance + candidate_quadratics, axis=0)
        rows, diagnostics = efficient_score_statistics(
            quadratics,
            nuisance_count=len(nuisance),
            boundary_positions=candidates,
        )
        alpha = float(args.split_alpha)
        accepted = bool(rows and diagnostics["global_sup_score_p_value"] <= alpha)
        payload = {
            "schema_version": 1,
            "method": "parametric_bootstrap_efficient_reml_ld_cusum",
            "criterion": "global_sup_score_p_value_le_split_alpha",
            "split_alpha": alpha,
            "accepted": accepted,
            "stop_reason": (
                "split_signal" if accepted else
                "global_score_rejected" if rows else "no_testable_contrast"
            ),
            "current_k": len(context.groups),
            "selected_candidate": rows[0] if rows else None,
            "candidates": rows,
            "diagnostics": diagnostics,
            "bootstrap_draws": int(args.bootstrap_draws),
            "bootstrap_seed": int(args.bootstrap_seed),
            "current_boundaries": current_positions.tolist(),
            "available_boundary_count": int(candidates.size),
            "pcg": projector.solve_diagnostics,
        }
        atomic_json(args.out, payload)
    finally:
        context.close()


def run_frozen_fit(args: argparse.Namespace) -> None:
    context = build_analysis_context(args)
    try:
        parent = read_json(args.parent_summary)
        _markers, _beta, sparse_mean = _load_sparse_mean(
            args.state_path, context.grm_index
        )
        residual = context.y.astype(np.float64) - sparse_mean
        theta_init = np.asarray(
            json.loads(args.theta_init_json), dtype=np.float64
        ).reshape(-1)
        if theta_init.shape != (len(context.groups) + 1,):
            raise ValueError("Frozen-fit theta initializer has the wrong length.")
        fit = context.fitter.fit_infinitesimal(
            jnp.asarray(residual, dtype=jnp.float32),
            jnp.asarray(context.covar, dtype=jnp.float32),
            h2_init=float(np.sum(theta_init[:-1]) / np.sum(theta_init)),
            var_components_init=jnp.asarray(theta_init, dtype=jnp.float32),
        )
        theta, covariance_stop_reason = _accepted_reml_theta(
            fit,
            expected_components=len(context.groups) + 1,
            stage="adaptive frozen-alpha covariance refit",
        )
        q_sparse = float(parent["q_chive"])
        genetic = q_sparse + float(np.sum(theta[:-1]))
        h2 = float(genetic / (genetic + float(theta[-1])))
        history = list(fit.history)
        payload = {
            "schema_version": 1,
            "method": "fixed_sparse_mean_covariance_reml",
            "n_grms": len(context.groups),
            "n_snps_total": int(context.grm_index.m_total),
            "component_spec": str(Path(args.component_spec).resolve()),
            "component_sizes": [int(group.size) for group in context.groups],
            "parent_summary": str(Path(args.parent_summary).resolve()),
            "parent_q_sparse_held_fixed": q_sparse,
            "q_chive": q_sparse,
            "theta_initial": theta_init.tolist(),
            "theta": theta.tolist(),
            "var_components_lasso_ml": theta.tolist(),
            "h2": h2,
            "support_size": int(parent["support_size"]),
            "restricted_loglik_per_sample": float(fit.final_loglik),
            "stop_reason": covariance_stop_reason,
            "history": history,
        }
        atomic_json(args.out, payload)
    finally:
        context.close()


def _add_common_fit_arguments(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--bed-prefix", default="")
    source.add_argument("--pgen-prefix", default="")
    parser.add_argument("--pheno-txt", required=True)
    parser.add_argument("--covar-txt", default="")
    parser.add_argument("--keep-path", required=True)
    parser.add_argument("--component-spec", required=True)
    parser.add_argument("--state-path", required=True)
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--gpu-budget-gib", type=float, default=0.0)
    parser.add_argument("--cpu-threads", type=int, default=0)
    parser.add_argument("--call-width", type=int, default=0)
    parser.add_argument("--ring-depth", type=int, default=0)
    parser.add_argument("--n-rand-vec", type=int, default=100)
    parser.add_argument("--slq-samples", type=int, default=100)
    parser.add_argument("--slq-m", type=int, default=50)
    parser.add_argument("--minq-iter", type=int, default=50)
    parser.add_argument("--pcg-tol", type=float, default=5e-3)
    parser.add_argument("--pcg-ridge", type=float, default=1e-6)
    parser.add_argument("--max-pcg-iters", type=int, default=400)
    parser.add_argument("--reml-max-linesearch-trials", type=int, default=8)
    parser.add_argument("--verbose", action="store_true")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    score = commands.add_parser("score", help="Evaluate the next LD-CUSUM split.")
    _add_common_fit_arguments(score)
    score.add_argument("--summary-path", required=True)
    score.add_argument("--rank-path", required=True)
    score.add_argument("--bootstrap-draws", type=int, default=199)
    score.add_argument("--bootstrap-seed", type=int, default=20260831)
    score.add_argument("--split-alpha", type=float, default=0.05)
    score.add_argument("--out", required=True)

    frozen = commands.add_parser(
        "frozen-fit", help="Refit covariance after one accepted split."
    )
    _add_common_fit_arguments(frozen)
    frozen.add_argument("--parent-summary", required=True)
    frozen.add_argument("--theta-init-json", required=True)
    frozen.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if (
        args.cpu_threads < 0
        or args.gpu_budget_gib < 0.0
        or args.call_width < 0
        or args.ring_depth < 0
    ):
        parser.error("Threads, GPU budget, call width, and ring depth must be nonnegative.")
    if args.command == "score":
        if args.bootstrap_draws < 19:
            parser.error("--bootstrap-draws must be at least 19.")
        if not 0.0 < args.split_alpha < 1.0:
            parser.error("--split-alpha must lie in (0, 1).")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "score":
        run_score(args)
    elif args.command == "frozen-fit":
        run_frozen_fit(args)
    else:  # pragma: no cover - argparse enforces the subcommand.
        raise RuntimeError(args.command)


if __name__ == "__main__":
    main()


__all__ = [
    "add_ld_boundary",
    "efficient_score_statistics",
    "existing_boundary_positions",
    "load_ld_rank",
    "write_root_component_spec",
]
