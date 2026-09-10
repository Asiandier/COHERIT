"""Adaptive LD-rank covariance splitting for sparse COHERIT.

This module contains only the selected K=1 -> fixed-alpha LD-CUSUM -> endpoint
refit algorithm.  Candidate boundaries are deterministic balanced cuts of one
global LD-score ordering. At each layer independent trace probes fit and
evaluate covariance-score directions. A common Gaussian quadratic reference
jointly calibrates the maximum absolute standardized score across all cuts.

The user-facing orchestration lives in :mod:`GPU_REML.run_sparse_pipeline`.
The small command-line interface here is internal: separate processes keep GPU
memory bounded between the validation fit, score, and covariance-only refit.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Sequence

from .runtime_env import configure_runtime_env

configure_runtime_env()

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as sla

# Preserve the sparse workflow's precision without importing its CLI module.
jax.config.update(
    "jax_default_matmul_precision",
    os.environ.get("GPU_REML_MATMUL_PRECISION", "highest"),
)

from .component_spec import load_component_specs
from .data_utils import load_pheno_covar_aligned_with_transform
from .geno_source import PgenGenoSource
from .pcg import pcg_solve
from .kv_impl import _device_put_block, _zxb_multi_one_call_jit
from .pipeline_common import (
    cleanup_path,
    compute_sample_mask,
    genetic_variance,
    make_nonbed_input_fam,
    read_keep_ids,
    resolve_cpu_threads,
    run_planner,
    setup_gpu,
)
from .reml_model import FitConfig, InfinitesimalREMLFitter, standardize_response
from .sparse_core import (
    MultiGRMIndex,
    accepted_reml_theta as _accepted_reml_theta,
    sparse_dense_h2 as _sparse_dense_h2,
    validate_component_partition as _validate_component_partition,
)
from .sparse_information import SparseMeanInformation
from .io_utils import atomic_json, read_json
from .variant_io import iter_variant_records_for_prefix
from .score_process import (
    TracePrecisionError,
    TraceSketch,
    calibrate_boundary_maximum,
    fit_nuisance_projection,
    score_process_moments,
)

logger = logging.getLogger(__name__)

# Bound float32 dot-product reduction length even when the runtime planner
# selects a very wide packed block. This is kernel geometry, not a model knob.
_FACTOR_MAX_REDUCTION_WIDTH = 65536


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


@dataclass
class REMLProjector:
    fitter: Any
    ops: Any
    theta: np.ndarray
    covar: np.ndarray | None
    pcg_tol: float
    max_pcg_iters: int
    mean_information: SparseMeanInformation | None = None

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
        self.mean_state = None
        self._mean_w = np.empty((n_samples, 0), dtype=np.float64)
        self._reference_null_basis = None
        if self.mean_information is not None:
            self.mean_state = self.mean_information.evaluate(
                self.theta, self._hv, self._preconditioner, self._vinv_c,
                tol=self.pcg_tol, maxiter=self.max_pcg_iters,
            )
            self._mean_w = self.mean_state["w"]
            if self.mean_information.rank:
                basis = self.mean_information.basis
                covar_remainder = self._covar-basis @ (basis.T @ self._covar)
                covar_basis = sla.qr(covar_remainder, mode="economic", check_finite=False)[0]
                self._reference_null_basis = np.column_stack([basis, covar_basis])

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
        # PCG's returned residual is already checked against the actual Hx.
        reported_relative = float(np.max(np.asarray(jax.device_get(reported))))
        true_relative = reported_relative
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

    def apply_reference(self, rhs: np.ndarray, *, stage: str) -> np.ndarray:
        """Apply P_S=P-P Sigma_mu P for the local Gaussian reference only.

        The observed offset residual uses apply(), i.e. P r. Replacing it by
        P_S r would instead test the unpenalized expanded-mean REML model.
        """
        array = np.asarray(rhs, dtype=np.float64)
        null_basis = self._reference_null_basis
        if null_basis is not None:
            # P_S annihilates [C,Q_S] on both sides. Enforce this exact identity
            # so cancellation of P and W W' cannot create spurious core modes.
            array = array-null_basis @ (null_basis.T @ array)
        selected_projection = self._mean_w @ (self._mean_w.T @ array)
        result = np.asarray(self.apply(array, stage=stage), dtype=np.float64) - selected_projection
        if null_basis is not None:
            result = result-null_basis @ (null_basis.T @ result)
        return result

@dataclass
class BoundaryContractions:
    """Apply all covariance directions through shared LD-bin prefix sums."""

    cache_order: np.ndarray
    grid_positions: np.ndarray
    candidates: np.ndarray
    intervals: list[tuple[int, int]]
    component_normalizers: np.ndarray

    def __post_init__(self) -> None:
        self.edges = np.r_[0, self.grid_positions, self.cache_order.size]
        if (not np.all(np.isin(self.candidates, self.grid_positions))
                or not np.all(np.isin(self.intervals, self.edges))):
            raise ValueError("Candidates and component endpoints must align with the LD-rank grid.")
        self.candidate_bins = np.searchsorted(self.grid_positions, self.candidates)
        self.component_bins = [
            (int(np.searchsorted(self.edges, start)), int(np.searchsorted(self.edges, stop)))
            for start, stop in self.intervals
        ]

    def apply(self, left, right, marker_left, marker_right) -> np.ndarray:
        bins = np.stack([
            np.sum(
                marker_left[self.cache_order[start:stop]].astype(np.float64)
                * marker_right[self.cache_order[start:stop]].astype(np.float64), axis=0,
            )
            for start, stop in zip(self.edges[:-1], self.edges[1:], strict=True)
        ])
        nuisance = [
            np.sum(bins[start:stop], axis=0) / normalizer
            if normalizer > 0 else np.zeros(bins.shape[1])
            for (start, stop), normalizer in zip(
                self.component_bins, self.component_normalizers, strict=True
            )
        ]
        nuisance.append(np.einsum(
            "ij,ij->j", np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
        ))
        prefix = np.cumsum(bins, axis=0)
        low = prefix[self.candidate_bins]
        high = prefix[-1] - low
        contrasts = (low / self.candidates[:, None]
                     - high / (self.cache_order.size - self.candidates[:, None]))
        return np.concatenate([np.stack(nuisance), contrasts], axis=0)

    def bilinear(self, left, right, marker_left, marker_right, *, out=None):
        """Full shared bilinear products with one running LD prefix."""
        shape = (len(self.intervals) + 1 + len(self.candidates), left.shape[1], right.shape[1])
        result = np.empty(shape, dtype=np.float64) if out is None else out
        if result.shape != shape:
            raise ValueError("Bilinear output shape does not match directions and vectors.")
        width = max(1, min(16384, 32 * 1024**2 // (8 * max(1, left.shape[1] + right.shape[1]))))

        def marker_product(start, stop):
            value = np.zeros(shape[1:], dtype=np.float64)
            for offset in range(start, stop, width):
                indices = self.cache_order[offset:min(stop, offset + width)]
                value += (np.asarray(marker_left[indices], dtype=np.float64).T
                          @ np.asarray(marker_right[indices], dtype=np.float64))
            return value

        total = np.zeros(shape[1:], dtype=np.float64)
        for g, ((start, stop), normalizer) in enumerate(zip(
            self.intervals, self.component_normalizers, strict=True
        )):
            value = marker_product(start, stop)
            total += value
            result[g] = value / normalizer if normalizer > 0 else 0
        identity = np.zeros(shape[1:], dtype=np.float64)
        for offset in range(0, len(left), width):
            identity += (np.asarray(left[offset:offset + width], dtype=np.float64).T
                         @ np.asarray(right[offset:offset + width], dtype=np.float64))
        result[len(self.intervals)] = identity
        prefix = np.zeros_like(total)
        candidate = 0
        for start, stop in zip(self.edges[:-1], self.edges[1:], strict=True):
            prefix += marker_product(start, stop)
            if candidate < len(self.candidates) and stop == self.candidates[candidate]:
                result[len(self.intervals) + 1 + candidate] = (
                    prefix / stop - (total - prefix) / (self.cache_order.size - stop)
                )
                candidate += 1
        if candidate != len(self.candidates):
            raise RuntimeError("Not all score candidates were visited.")
        return result


class ScoreWorkspace:
    """Bound the lifetime of scratch arrays; only current probe work stays live."""

    def __init__(self, directory: str | Path):
        Path(directory).mkdir(parents=True, exist_ok=True)
        self._temporary = tempfile.TemporaryDirectory(prefix=".score-", dir=directory)
        self._arrays: dict[int, np.ndarray] = {}
        self._next_file = 0
        self.live_bytes = 0
        self.peak_bytes = 0

    def ensure_capacity(self, additional_bytes, *, stage="allocation"):
        # mmap creation can reserve a sparse file without consuming disk yet.
        # Account for those outstanding writes as well as the next allocation.
        unwritten = sum(max(0, value.nbytes - Path(value.filename).stat().st_blocks * 512)
                        for value in self._arrays.values() if isinstance(value, np.memmap))
        available = shutil.disk_usage(self._temporary.name).free - unwritten
        if int(additional_bytes) > available:
            raise RuntimeError(
                f"Insufficient score scratch space for {stage}: need "
                f"{int(additional_bytes)/1024**3:.2f} GiB beyond live arrays, "
                f"{max(0, available)/1024**3:.2f} GiB available. "
                "Use an output filesystem with more free scratch space."
            )

    def _register(self, value):
        self._arrays[id(value)] = value
        self.live_bytes += value.nbytes
        self.peak_bytes = max(self.peak_bytes, self.live_bytes)
        return value

    def allocate(self, shape, *, dtype=np.float64):
        if np.prod(shape, dtype=np.int64) * np.dtype(dtype).itemsize < 8 * 1024**2:
            return self._register(np.empty(shape, dtype=dtype))
        self.ensure_capacity(int(np.prod(shape)) * np.dtype(dtype).itemsize)
        value = np.memmap(
            Path(self._temporary.name) / f"{self._next_file}.bin",
            mode="w+", dtype=dtype, shape=shape,
        )
        self._next_file += 1
        return self._register(value)

    def release(self, *arrays):
        """Release owned arrays, including the ndarray views used by TraceSketch."""
        for value in arrays:
            while isinstance(value, np.ndarray) and id(value) not in self._arrays:
                value = value.base
            owned = self._arrays.pop(id(value), None)
            if owned is None:
                continue
            self.live_bytes -= owned.nbytes
            if isinstance(owned, np.memmap):
                path = Path(owned.filename)
                owned._mmap.close()
                path.unlink()

    def grow_rows(self, value, rows):
        """Append probe capacity without regenerating or copying a mapped prefix.

        Call only after releasing all views of the old probe cache. Its layout
        is row-major (probe, sample/marker), so file extension preserves rows.
        """
        if id(value) not in self._arrays or not value.flags.c_contiguous:
            raise ValueError("Only owned contiguous probe caches can grow.")
        if rows <= value.shape[0]:
            return value
        shape = (int(rows), *value.shape[1:])
        size = int(np.prod(shape)) * value.dtype.itemsize
        if not isinstance(value, np.memmap):
            expanded = self.allocate(shape, dtype=value.dtype)
            expanded[:len(value)] = value
            self.release(value)
            return expanded
        self.ensure_capacity(size - value.nbytes, stage="probe cache growth")
        path, dtype, old_size = Path(value.filename), value.dtype, value.nbytes
        value._mmap.close()
        with path.open("r+b") as handle:
            handle.truncate(size)
        expanded = np.memmap(path, mode="r+", dtype=dtype, shape=shape)
        self._arrays.pop(id(value))
        self.live_bytes -= old_size
        return self._register(expanded)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        try:
            self.release(*list(self._arrays.values()))
        finally:
            self._temporary.cleanup()


def _score_rhs_width(projector, index):
    """Reserve a bounded part of the existing runtime budget for probe RHSs.

    The genotype block geometry is still selected by the normal planner. The
    additional budget covers the GPU X'B output, its host copy and PCG vectors;
    it must not charge the full genotype matrix once per probe.
    """
    cfg = getattr(getattr(projector, "fitter", None), "cfg", None)
    total = getattr(cfg, "gpu_budget_bytes", None)
    budget = min(1024**3, float(total) / 8) if total else 256 * 1024**2
    per_column = 12 * index.m_total + 64 * projector._covar.shape[0]
    capacity = max(1, min(256, int(budget) // max(1, per_column)))
    return 1 << (capacity.bit_length() - 1)


def _factor_rhs(index, theta, streams):
    """Covariance-factor Rademacher probes with exactly the GRM m_eff weights.

    Each probe has its own stream, so changing RHS or genotype block widths
    preserves its random signs. Reuse the packed prediction kernel: transfer
    and decode each cached block once for the whole batch, not on the CPU.
    """
    n = index.streamer.n
    normalizers = np.asarray(index.streamer._component_eff_m_host, dtype=np.float64)
    weights = np.sqrt(np.divide(theta[:-1], normalizers, out=np.zeros_like(normalizers),
                                where=normalizers > 0))
    st = index.streamer
    st._prepare_kv_pass()
    result = jax.device_put(jnp.zeros((n, len(streams)), dtype=jnp.float32), st.dev)
    miss = jnp.asarray(st._missing_val, dtype=jnp.uint8)
    next_block = _device_put_block(st._pop_cached(0), st.dev) if st._n_calls else None
    for call in range(st._n_calls):
        block = next_block
        if call + 1 < st._n_calls:
            next_block = _device_put_block(st._pop_cached(call + 1), st.dev)
        call_width = int(st._call_true_widths[call])
        component = int(st._call_component_ids[call])
        for offset in range(0, call_width, _FACTOR_MAX_REDUCTION_WIDTH):
            width = min(_FACTOR_MAX_REDUCTION_WIDTH, call_width - offset)
            packed = block[:, offset//4:(offset + width + 3)//4]
            padded = int(packed.shape[1]) * 4
            coefficients = np.zeros((padded, len(streams)), dtype=np.float32)
            for column, rng in enumerate(streams):
                signs = 2 * rng.integers(0, 2, size=width, dtype=np.int32) - 1
                coefficients[:width, column] = weights[component] * signs
            result = result + _zxb_multi_one_call_jit(
                packed, jnp.asarray(width, dtype=jnp.int32),
                st._means_by_call[call, offset:offset+padded],
                st._inv_by_call[call, offset:offset+padded],
                jax.device_put(coefficients, st.dev), miss,
            )
        del block
    residual = np.column_stack([
        2 * rng.integers(0, 2, size=n, dtype=np.int32) - 1 for rng in streams
    ]).astype(np.float32)
    return np.asarray(result + jnp.asarray(np.sqrt(theta[-1]) * residual, dtype=jnp.float32))


def _build_score_core(projector, index, contractions, rank, rng, workspace):
    """Build H=P_S T with T'P_S T=I from the root-GRM sketch."""
    n = projector._covar.shape[0]
    rank = min(int(rank), n, index.m_total)
    width = _score_rhs_width(projector, index)
    sketch = workspace.allocate((n, rank))
    # Preserve the established random sketch independently of compute batching.
    # This small input is generated once; the large genotype passes below use
    # the runtime-budgeted RHS width instead of the random stream's tile width.
    draw_width = max(1, min(16, 32 * 1024**2 // (8 * (n + index.m_total))))
    omega = np.empty((n, rank), dtype=np.float32)
    for offset in range(0, rank, draw_width):
        stop = min(rank, offset + draw_width)
        omega[:, offset:stop] = rng.standard_normal((n, stop - offset), dtype=np.float32)
    for offset in range(0, rank, width):
        stop = min(rank, offset + width)
        sketch[:, offset:stop] = np.asarray(index.streamer.kv(jnp.asarray(omega[:, offset:stop])))
    del omega
    basis, singular, _ = np.linalg.svd(sketch, full_matrices=False)
    workspace.release(sketch)
    del sketch
    basis = basis[:, singular > 1e-7 * np.max(singular, initial=0.0)]
    projected = workspace.allocate(basis.shape)
    for offset in range(0, basis.shape[1], width):
        projected[:, offset:offset + width] = projector.apply_reference(
            basis[:, offset:offset + width], stage="adaptive_ld_common_core",
        )
    gram = basis.T @ projected
    values, rotation = np.linalg.eigh(0.5 * (gram + gram.T))
    tolerance = 1e-7 * np.max(values, initial=0.0)
    if np.min(values, initial=0.0) < -max(tolerance, np.finfo(float).eps):
        raise TracePrecisionError("Common-core P Gram is indefinite beyond roundoff.")
    keep = values > tolerance
    transform = rotation[:, keep] / np.sqrt(values[keep])
    h = projected @ transform
    workspace.release(projected)
    del projected
    marker_h = workspace.allocate((index.m_total, h.shape[1]), dtype=np.float32)
    for offset in range(0, h.shape[1], width):
        marker_h[:, offset:offset + width] = index.xtv_all(
            jnp.asarray(h[:, offset:offset + width], dtype=jnp.float32), normalize=False,
            dtype=np.float32,
        )
    core = workspace.allocate((len(contractions.intervals) + 1 + len(contractions.candidates),
                               h.shape[1], h.shape[1]))
    contractions.bilinear(h, h, marker_h, marker_h, out=core)
    return h, marker_h, core


def _fill_probe_cache(projector, index, h, seed, samples, markers, start, stop, *, stage):
    width = _score_rhs_width(projector, index)
    for offset in range(start, stop, width):
        end = min(stop, offset + width)
        streams = [np.random.default_rng(child) for child in seed.spawn(end - offset)]
        rhs = _factor_rhs(index, projector.theta, streams)
        projected = projector.apply_reference(rhs, stage=stage).astype(np.float64)
        b = projected - h @ (h.T @ rhs)
        samples[offset:end] = b.T
        markers[offset:end] = index.xtv_all(
            jnp.asarray(b, dtype=jnp.float32), normalize=False, dtype=np.float32,
        ).T


def _probe_sketch(contractions, h, marker_h, core, samples, markers, workspace):
    count = len(samples)
    ql = len(core)
    b, marker_b = samples.T, markers.T
    cross = workspace.allocate((ql, h.shape[1], count))
    bulk = workspace.allocate((ql, count, count))
    contractions.bilinear(h, b, marker_h, marker_b, out=cross)
    contractions.bilinear(b, b, marker_b, marker_b, out=bulk)
    diagonal = np.diagonal(bulk, axis1=1, axis2=2).copy()
    index = np.arange(count)
    bulk[:, index, index] = 0
    return TraceSketch(core, cross, bulk, diagonal)


def _estimate_score_process(args, projector, grm_index, contractions, response, *, workspace):
    q = len(contractions.intervals) + 1
    projected = projector.apply(response[:, None], stage="adaptive_ld_observed")
    marker = np.asarray(grm_index.xtv_all(jnp.asarray(projected), normalize=False))
    quadratics = contractions.apply(projected, projected, marker, marker)[:, 0]
    # Keep the derivative of the fitted offset objective ell_C unchanged.
    # The rank-adjusted working reference is v*~N(0,P_S), giving score
    # covariance tr(P_S D_i P_S D_j)/2. This is not the exact distribution of
    # an adaptively fitted Lasso residual; no extra shrinkage-centering term
    # is subtracted from the likelihood score used to propose a split.
    core_seed, pilot_seed, evaluation_seed, reference_seed = np.random.SeedSequence(
        args.score_trace_seed
    ).spawn(4)
    h, marker_h, core = _build_score_core(
        projector, grm_index, contractions, args.score_core_rank, np.random.default_rng(core_seed), workspace,
    )
    limit = int(args.score_trace_max_probes)
    pilot_count = min(limit, max(int(args.score_trace_probes), 4 * q))
    n, m = response.size, grm_index.m_total
    directions, rank = len(core), h.shape[1]
    pilot_bytes = (4 * pilot_count * (n + m)
                   + 8 * directions * (rank * pilot_count + pilot_count**2))
    workspace.ensure_capacity(pilot_bytes, stage=f"{pilot_count}-probe pilot")
    logger.info("Adaptive score: %d RHSs per batch; pilot scratch %.2f GiB",
                _score_rhs_width(projector, grm_index), pilot_bytes / 1024**3)
    pilot_samples = workspace.allocate((pilot_count, n), dtype=np.float32)
    pilot_markers = workspace.allocate((pilot_count, m), dtype=np.float32)
    _fill_probe_cache(projector, grm_index, h, pilot_seed, pilot_samples, pilot_markers,
                      0, pilot_count, stage="adaptive_ld_trace_pilot")
    pilot = _probe_sketch(contractions, h, marker_h, core, pilot_samples, pilot_markers, workspace)
    coefficients, nuisance_eigenvalues = fit_nuisance_projection(pilot, q)
    workspace.release(pilot.cross, pilot.bulk, pilot_samples, pilot_markers)
    del pilot, pilot_samples, pilot_markers
    target, count = int(args.score_trace_probes), 0
    samples = workspace.allocate((0, n), dtype=np.float32)
    markers = workspace.allocate((0, m), dtype=np.float32)
    while True:
        # Reserve a whole tier before spending time generating it. Earlier
        # tiers and the pilot have been released, not left mapped until exit.
        tier_bytes = (4 * (target - count) * (n + m)
                      + 8 * directions * (rank * target + target**2)
                      + 4 * len(contractions.candidates) * (rank + target)**2)
        workspace.ensure_capacity(tier_bytes, stage=f"{target}-probe score tier")
        samples = workspace.grow_rows(samples, target)
        markers = workspace.grow_rows(markers, target)
        _fill_probe_cache(projector, grm_index, h, evaluation_seed, samples, markers,
                          count, target, stage="adaptive_ld_trace_evaluation")
        count = target
        evaluation = _probe_sketch(contractions, h, marker_h, core,
                                   samples[:count], markers[:count], workspace)
        reference = workspace.allocate(
            (len(contractions.candidates), h.shape[1] + count, h.shape[1] + count),
            dtype=np.float32,
        )
        process = score_process_moments(
            quadratics, evaluation=evaluation, coefficients=coefficients,
            boundary_positions=contractions.candidates, reference_out=reference,
        )
        workspace.release(evaluation.cross, evaluation.bulk)
        del evaluation
        process.diagnostics.update(
            pilot_probes=pilot_count,
            nuisance_information_eigenvalues=nuisance_eigenvalues.tolist(),
            reference_seed=int(reference_seed.generate_state(1)[0]),
            mean_information_rank=(projector.mean_information.rank
                                   if projector.mean_information is not None else 0),
            reference_covariance="P_S = P - P Sigma_mu P",
            reference_mean="zero: rank-adjusted Gaussian working reference",
            reference_scope="local_selected_space_zero_mean_working_reference_not_post_selection_exact",
        )
        error = max(process.diagnostics["max_score_trace_standard_error"],
                    process.diagnostics["max_information_relative_standard_error"] / 2)
        logger.info("Adaptive score: core rank %d, %d pilot, %d evaluation probes; standard error %.3g",
                    h.shape[1], pilot_count, count, error)
        if error <= args.score_trace_tol:
            workspace.release(samples, markers, marker_h, core)
            process.diagnostics.update(probe_rhs_width=_score_rhs_width(projector, grm_index),
                                       scratch_peak_bytes=workspace.peak_bytes)
            return process
        workspace.release(reference)
        del reference, process
        if count >= limit:
            raise TracePrecisionError(
                f"Score trace standard error {error:.3g} exceeds {args.score_trace_tol:.3g}; "
                f"increase --score-trace-max-probes (current {limit})."
            )
        target = min(limit, 2 * target)

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
        slq_m=int(args.slq_m),
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
    covar: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, SparseMeanInformation]:
    with np.load(state_path, allow_pickle=False) as state:
        source_markers = np.asarray(state["marker_indices"], dtype=np.int64)
        beta = np.asarray(state["selected_beta_snp"], dtype=np.float64)
    active = beta != 0.0
    source_markers, beta = source_markers[active], beta[active]
    cache_markers = grm_index.cache_variant_indices(source_markers)
    design = grm_index.extract_standardized_columns(cache_markers).astype(np.float64)
    return source_markers, beta, design @ beta, SparseMeanInformation(design, covar)


def run_score(args: argparse.Namespace) -> None:
    context = build_analysis_context(args)
    try:
        summary = read_json(args.summary_path)
        theta = np.asarray(summary["var_components_lasso_ml"], dtype=np.float64)
        if theta.shape != (len(context.groups) + 1,):
            raise ValueError("Score parent theta does not align with its partition.")
        if not np.all(np.isfinite(theta)) or np.any(theta[:-1] < 0) or theta[-1] <= 0:
            raise ValueError("Score parent variance components are invalid.")
        source_order, grid_positions = load_ld_rank(args.rank_path)
        intervals = _rank_intervals(context.groups, source_order)
        current_positions = existing_boundary_positions(context.groups, source_order)
        candidates = np.setdiff1d(grid_positions, current_positions, assume_unique=True)
        parent_indices = np.searchsorted([stop for _start, stop in intervals], candidates)
        positive_parent = theta[parent_indices] > 0
        excluded = candidates[~positive_parent]
        candidates = candidates[positive_parent]
        base = {
            "schema_version": 5,
            "method": "joint_quadratic_information_corrected_ld_cusum",
            "criterion": "global_p_value_le_split_alpha",
            "split_alpha": float(args.split_alpha),
            "current_k": len(context.groups),
            "current_boundaries": current_positions.tolist(),
            "available_boundary_count": int(candidates.size),
            "zero_parent_boundaries_excluded": excluded.tolist(),
            "score_trace_seed": int(args.score_trace_seed),
        }
        if not candidates.size:
            atomic_json(args.out, {
                **base, "accepted": False,
                "stop_reason": "no_positive_parent_variance" if excluded.size else "no_available_boundary",
                "selected_candidate": None, "candidates": [],
                "diagnostics": {"global_p_value": 1.0, "calibration_method": "empty_family",
                                "testable_candidate_count": 0, "reference_samples": 0},
            })
            return
        ops = context.fitter._assemble_reml_operators()
        context.fitter._ensure_projected_core_precond_ready(
            ops, var_components_init=jnp.asarray(theta, dtype=jnp.float32)
        )
        _markers, _beta, sparse_mean, mean_information = _load_sparse_mean(
            args.state_path, context.grm_index, context.covar,
        )
        projector = REMLProjector(
            fitter=context.fitter, ops=ops, theta=theta, covar=context.covar,
            pcg_tol=min(float(args.pcg_tol), 1e-5, float(args.score_trace_tol) / 10),
            max_pcg_iters=int(args.max_pcg_iters),
            mean_information=mean_information,
        )
        # P already profiles C; retain the same frozen sparse mean and response
        # scale as the covariance fit. No simulated phenotype enters this stage.
        response = (context.y.astype(np.float64) - sparse_mean).astype(np.float32)
        contractions = BoundaryContractions(
            cache_order=context.grm_index.cache_variant_indices(source_order),
            grid_positions=grid_positions, candidates=candidates, intervals=intervals,
            component_normalizers=np.asarray(
                context.fitter.streamers[0]._component_eff_m_host, dtype=np.float64
            ),
        )
        with jax.default_device(context.grm_index.streamer.dev), ScoreWorkspace(Path(args.out).parent) as workspace:
            process = _estimate_score_process(
                args, projector, context.grm_index, contractions, response, workspace=workspace,
            )
            rows, diagnostics = calibrate_boundary_maximum(
                process, alpha=float(args.split_alpha), samples=int(args.score_reference_samples),
                seed=process.diagnostics["reference_seed"],
            )
        accepted = bool(diagnostics["accepted"])
        stop_reason = (
            "split_signal" if accepted else "no_testable_contrast" if not rows else
            "global_score_not_significant"
        )
        atomic_json(args.out, {
            **base, "accepted": accepted, "stop_reason": stop_reason,
            "selected_candidate": rows[0] if rows else None,
            "candidates": rows, "diagnostics": diagnostics,
            "pcg": projector.solve_diagnostics,
        })
    finally:
        context.close()


def run_frozen_fit(args: argparse.Namespace) -> None:
    context = build_analysis_context(args)
    try:
        markers, _beta, sparse_mean, mean_information = _load_sparse_mean(
            args.state_path, context.grm_index, context.covar,
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
            var_components_init=jnp.asarray(theta_init, dtype=jnp.float32),
            mean_information=mean_information,
        )
        theta, covariance_stop_reason = _accepted_reml_theta(
            fit,
            expected_components=len(context.groups) + 1,
            stage="adaptive frozen-alpha covariance refit",
        )
        ops = context.fitter._assemble_reml_operators()
        theta_device = jnp.asarray(theta, dtype=jnp.float32)
        hv = context.fitter._make_hv(ops, theta_device[:-1], theta_device[-1])
        precond = context.fitter._make_effect_precond(ops, theta_device[:-1], theta_device[-1])
        state, calibrated_residual, gamma = mean_information.statistics(
            theta, hv, precond, residual,
            tol=min(float(args.pcg_tol), 1e-5), maxiter=int(args.max_pcg_iters),
        )
        n = sparse_mean.size
        term1 = float(sparse_mean @ sparse_mean / n)
        term2 = float(2 * sparse_mean @ calibrated_residual / n)
        uncertainty = float(state["trace"] / n)
        q_sparse = term1 + term2 - uncertainty
        genetic_trace_atoms = np.asarray(
            jax.device_get(fit.genetic_trace_atoms), dtype=np.float64
        )
        h2 = _sparse_dense_h2(
            q_sparse,
            genetic_variance(theta[:-1], genetic_trace_atoms),
            float(theta[-1]),
        )
        history = list(fit.history)
        payload = {
            "schema_version": 2,
            "method": "fixed_sparse_mean_information_corrected_reml",
            "n_grms": len(context.groups),
            "n_snps_total": int(context.grm_index.m_total),
            "component_spec": str(Path(args.component_spec).resolve()),
            "component_sizes": [int(group.size) for group in context.groups],
            "parent_summary": str(Path(args.parent_summary).resolve()),
            "q_chive": q_sparse,
            "q_chive_term1": term1,
            "q_chive_term2": term2,
            "mean_uncertainty_trace_per_n": uncertainty,
            "mean_information_rank": mean_information.rank,
            "mean_information_logdet": float(state["logdet"]),
            "beta_cov": gamma.tolist(),
            "theta_initial": theta_init.tolist(),
            "theta": theta.tolist(),
            "var_components_lasso_ml": theta.tolist(),
            "grm_variance_scale": "trace_weighted",
            "genetic_trace_atoms": genetic_trace_atoms.tolist(),
            "h2": h2,
            "support_size": int(markers.size),
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
    score.add_argument("--score-core-rank", type=int, default=64)
    score.add_argument("--score-reference-samples", type=int, default=16383)
    score.add_argument("--score-trace-probes", type=int, default=512)
    score.add_argument("--score-trace-max-probes", type=int, default=4096)
    score.add_argument("--score-trace-tol", type=float, default=0.05)
    score.add_argument("--score-trace-seed", type=int, default=20260831)
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
        if args.score_core_rank < 1:
            parser.error("--score-core-rank must be positive.")
        if args.score_reference_samples < 255:
            parser.error("--score-reference-samples must be at least 255.")
        if 1 / (args.score_reference_samples + 1) > args.split_alpha:
            parser.error("--score-reference-samples cannot resolve --split-alpha.")
        if args.score_trace_probes < 32:
            parser.error("--score-trace-probes must be at least 32.")
        if args.score_trace_max_probes < args.score_trace_probes:
            parser.error("--score-trace-max-probes must be >= --score-trace-probes.")
        if not 0 < args.score_trace_tol <= 0.25:
            parser.error("--score-trace-tol must lie in (0, 0.25].")
        if args.score_trace_seed < 0:
            parser.error("--score-trace-seed must be nonnegative.")
        if not 0.0 < args.split_alpha < 1.0:
            parser.error("--split-alpha must lie in (0, 1).")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
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
    "existing_boundary_positions",
    "load_ld_rank",
    "write_root_component_spec",
]
