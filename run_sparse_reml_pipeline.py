#!/usr/bin/env python3
"""
Sparse REML + LASSO pipeline.

Performance notes (vs. previous version):
  • KKT-certified candidates persist across alpha/theta outer iterations and
    are unioned with the refreshed signal screen instead of being discarded
    down to the active support.
  • Small KKT-violation batches receive an adaptive near-threshold buffer,
    avoiding repeated complete-path solves for one-digit follow-up violations.
  • Coefficient paths are remapped across every candidate expansion that
    retains the old basis; each lambda uses the remapped row only when its KKT
    residual improves on the ordinary descending-lambda warm start.
  • Column-major Gram storage, density-adaptive Qβ updates, and filtered exact
    KKT matvecs accelerate coordinate descent without changing its certificate.
  • Warm-start dictionary: maps SNP index → previous PCG solution column.
    When the candidate set overlaps between iterations (common), the warm
    starts for those columns are reused — dramatically reducing PCG iterations
    for the Z portion.
  • Gram matrix inputs kept in float64 throughout to avoid precision loss.
  • Pre-built JAX arrays for y and covar avoid repeated device_put.
  • Shared planner: uses the same closed-form GPU-width rule as pure REML.
"""
from __future__ import annotations

import argparse
import atexit
import dataclasses
import importlib
import json
import logging
import math
import os
import sys
import time
from datetime import datetime

repo_root = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(repo_root)
if parent not in sys.path:
    sys.path.insert(0, parent)
pkg_name = os.path.basename(repo_root)
_runtime_mod = importlib.import_module(f"{pkg_name}.runtime_env")
_runtime_mod.configure_runtime_env()

import jax
import jax.numpy as jnp
import numpy as np
from bed_reader import open_bed

# Highest FP32 accumulation is the robust default; users can explicitly select
# a faster hardware-specific mode after validating it on their GPU.
jax.config.update(
    "jax_default_matmul_precision",
    os.environ.get("GPU_REML_MATMUL_PRECISION", "highest"),
)
logger = logging.getLogger(__name__)

_inf_mod = importlib.import_module(f"{pkg_name}.reml_model")
_data_mod = importlib.import_module(f"{pkg_name}.data_utils")
_lasso_mod = importlib.import_module(f"{pkg_name}.lasso_cd")
_pcg_mod = importlib.import_module(f"{pkg_name}.pcg")
_information_mod = importlib.import_module(f"{pkg_name}.sparse_information")
_precond_mod = importlib.import_module(f"{pkg_name}.precond")
_common_mod = importlib.import_module(f"{pkg_name}.pipeline_common")
_io_utils_mod = importlib.import_module(f"{pkg_name}.io_utils")
_component_spec_mod = importlib.import_module(f"{pkg_name}.component_spec")
_effect_io_mod = importlib.import_module(f"{pkg_name}.effect_io")
_variant_io_mod = importlib.import_module(f"{pkg_name}.variant_io")
_sparse_prediction_mod = importlib.import_module(
    f"{pkg_name}.sparse_prediction"
)
_sparsity_selection_mod = importlib.import_module(
    f"{pkg_name}.sparsity_selection"
)

InfinitesimalREMLFitter = _inf_mod.InfinitesimalREMLFitter
FitConfig = _inf_mod.FitConfig
standardize_response = _inf_mod.standardize_response
load_pheno_covar_aligned_with_transform = (
    _data_mod.load_pheno_covar_aligned_with_transform
)
load_covar_aligned = _data_mod.load_covar_aligned
LassoPathConfig = _lasso_mod.LassoPathConfig
make_lambda_sequence = _lasso_mod.make_lambda_sequence
compute_projected_hinv_vector = _lasso_mod.compute_projected_hinv_vector
fit_weighted_lasso_with_covariates = _lasso_mod.fit_weighted_lasso_with_covariates
pcg_solve = _pcg_mod.pcg_solve
SparseMeanInformation = _information_mod.SparseMeanInformation
load_component_specs = _component_spec_mod.load_component_specs
write_sparse_effect_outputs = _effect_io_mod.write_sparse_effect_outputs
iter_variant_records_for_prefix = _variant_io_mod.iter_variant_records_for_prefix
predict_sparse_branch = _sparse_prediction_mod.predict_sparse_branch
predict_sparse_path_partitioned = (
    _sparse_prediction_mod.predict_sparse_path_partitioned
)
predict_sparse_path_single_grm = (
    _sparse_prediction_mod.predict_sparse_path_single_grm
)
write_sparse_prediction_outputs = (
    _sparse_prediction_mod.write_sparse_prediction_outputs
)
write_sparse_prediction_status = (
    _sparse_prediction_mod.write_sparse_prediction_status
)
remove_sparse_prediction_outputs = (
    _sparse_prediction_mod.remove_sparse_prediction_outputs
)
evaluate_prediction_path = _sparsity_selection_mod.evaluate_prediction_path
merge_path_diagnostics = _sparsity_selection_mod.merge_path_diagnostics
read_phenotype_aligned = _sparsity_selection_mod.read_phenotype_aligned
write_selection_outputs = _sparsity_selection_mod.write_selection_outputs

_source_mod = importlib.import_module(f"{pkg_name}.geno_source")
PgenGenoSource = _source_mod.PgenGenoSource

# Shared pipeline utilities (eliminates duplication with run_reml_pipeline.py)
env = _common_mod.env
query_gpu = _common_mod.query_gpu
read_keep_ids = _common_mod.read_keep_ids
setup_gpu = _common_mod.setup_gpu
run_planner = _common_mod.run_planner
print_planner_info = _common_mod.print_planner_info
log_runtime_gpu_memory = _common_mod.log_runtime_gpu_memory
ensure_parent_dir = _io_utils_mod.ensure_parent_dir
solve_spd = _common_mod.solve_spd
cleanup_path = _common_mod.cleanup_path
fam_order_mismatch = _common_mod.fam_order_mismatch
make_nonbed_input_fam = _common_mod.make_nonbed_input_fam
compute_sample_mask = _common_mod.compute_sample_mask
write_keep_file = _common_mod.write_keep_file
resolve_cpu_threads = _common_mod.resolve_cpu_threads
genetic_variance = _common_mod.genetic_variance
effective_preconditioner_rank_contract = (
    _common_mod.effective_preconditioner_rank_contract
)

def _bed_count(path: str, attr: str) -> int:
    bed = open_bed(path)
    try:
        return int(getattr(bed, attr))
    finally:
        close = getattr(bed, "close", None)
        if close is not None:
            close()


def _load_component_variant_indices(path: str) -> list[np.ndarray]:
    """Load source-variant memberships in declared component order."""
    return [
        np.asarray(spec.variant_indices, dtype=np.int64).reshape(-1)
        for spec in load_component_specs(path)
    ]


def _validate_component_partition(
    groups: list[np.ndarray], *, n_markers: int
) -> list[np.ndarray]:
    """Require a nonempty, disjoint, exhaustive source-marker partition."""
    if not groups:
        return []
    normalized: list[np.ndarray] = []
    for component_idx, group in enumerate(groups):
        values = np.asarray(group, dtype=np.int64).reshape(-1)
        if values.size == 0:
            raise ValueError(
                f"Component {component_idx} is empty; every GRM must contain SNPs."
            )
        unique = np.unique(values)
        if unique.size != values.size:
            raise ValueError(f"Component {component_idx} contains duplicate SNPs.")
        if np.any((unique < 0) | (unique >= int(n_markers))):
            raise ValueError(
                f"Component {component_idx} contains an index outside "
                f"[0, {int(n_markers)})."
            )
        normalized.append(unique)
    assigned = np.concatenate(normalized)
    if assigned.size != int(n_markers) or np.unique(assigned).size != int(
        n_markers
    ):
        raise ValueError(
            "--component-spec must assign every source SNP exactly once "
            "across mutually exclusive components."
        )
    return normalized


def _load_lasso_warm_state(
    path: str,
    *,
    grm_index=None,
    n_markers: int | None = None,
    target_lam_ratio: float | None = None,
    target_path_lam_ratios: np.ndarray | None = None,
) -> dict[str, object]:
    """Map a certified sparse state into the current GRM cache order.

    A frozen-ratio refit consumes the selected alpha only. A validation fit
    may additionally reuse a previously certified path prefix, but every row
    is re-optimized and globally KKT-certified under the new covariance.
    """
    if (target_lam_ratio is None) == (target_path_lam_ratios is None):
        raise ValueError(
            "Specify exactly one of target_lam_ratio and "
            "target_path_lam_ratios."
        )
    target_ratio = (
        _canonical_fixed_lam_ratio(float(target_lam_ratio))
        if target_lam_ratio is not None
        else None
    )
    target_path_ratios = None
    if target_path_lam_ratios is not None:
        target_path_ratios = np.asarray(
            target_path_lam_ratios, dtype=np.float64
        ).reshape(-1)
        if (
            target_path_ratios.size < 1
            or not np.all(np.isfinite(target_path_ratios))
            or np.any(target_path_ratios <= 0.0)
            or np.any(target_path_ratios > 1.0)
        ):
            raise ValueError("Target Lasso path ratios are invalid.")
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "lasso_warm_state_schema_version",
            "marker_indices",
            "selected_beta_snp",
            "selected_lam_ratio",
        }
        if not required.issubset(payload.files):
            raise ValueError("Lasso warm-state artifact is incomplete.")
        schema_version = int(
            np.asarray(payload["lasso_warm_state_schema_version"]).reshape(())
        )
        marker_indices = np.asarray(
            payload["marker_indices"], dtype=np.int64
        ).reshape(-1)
        selected_beta = np.asarray(
            payload["selected_beta_snp"], dtype=np.float64
        ).reshape(-1)
        selected_ratio = _canonical_fixed_lam_ratio(
            float(np.asarray(payload["selected_lam_ratio"]).reshape(()))
        )
        path_marker_indices = None
        beta_path_source = None
        saved_path_ratios = None
        if schema_version == 3:
            path_required = {
                "path_marker_indices",
                "beta_snp_path",
                "path_lam_ratios",
            }
            if not path_required.issubset(payload.files):
                raise ValueError(
                    "Schema-3 Lasso warm-state path is incomplete."
                )
            path_marker_indices = np.asarray(
                payload["path_marker_indices"], dtype=np.int64
            ).reshape(-1)
            beta_path_source = np.asarray(
                payload["beta_snp_path"], dtype=np.float64
            )
            saved_path_ratios = np.asarray(
                payload["path_lam_ratios"], dtype=np.float64
            ).reshape(-1)

    if schema_version not in {1, 2, 3}:
        raise ValueError(
            f"Unsupported Lasso warm-state schema: {schema_version}."
        )
    marker_count = (
        int(grm_index.m_total)
        if grm_index is not None
        else int(n_markers) if n_markers is not None else -1
    )
    if marker_count < 0:
        raise ValueError("A GRM index or marker count is required.")
    if (
        marker_indices.size != selected_beta.size
        or np.unique(marker_indices).size != marker_indices.size
        or np.any(
            (marker_indices < 0)
            | (marker_indices >= marker_count)
        )
        or not np.all(np.isfinite(selected_beta))
    ):
        raise ValueError("Lasso warm-state marker coefficients are invalid.")
    if target_ratio is not None and not math.isclose(
        selected_ratio, target_ratio, rel_tol=1e-10, abs_tol=1e-12
    ):
        raise ValueError(
            "Lasso warm state does not match the frozen lambda ratio."
        )

    candidate_unsorted = (
        marker_indices
        if schema_version == 1 or grm_index is None
        else grm_index.cache_variant_indices(marker_indices)
    )
    order = np.argsort(candidate_unsorted)
    candidate = candidate_unsorted[order]
    selected_beta = selected_beta[order]
    reuse_mode = "selected_alpha_for_fixed_ratio"
    if target_ratio is not None:
        beta_path = (
            selected_beta.reshape(1, -1)
            if math.isclose(target_ratio, 1.0, rel_tol=0.0, abs_tol=1e-12)
            else np.vstack([np.zeros_like(selected_beta), selected_beta])
        )
    else:
        # Old artifacts, and schema-3 artifacts from a different lambda grid,
        # still provide a safe selected-alpha initialization. Coordinate
        # descent and global KKT checks determine the actual new solutions.
        beta_path = np.repeat(
            selected_beta.reshape(1, -1),
            int(target_path_ratios.size),
            axis=0,
        )
        if math.isclose(
            float(target_path_ratios[0]), 1.0, rel_tol=0.0, abs_tol=1e-12
        ):
            beta_path[0, :] = 0.0
        reuse_mode = "selected_alpha_broadcast_for_validation"

        if schema_version == 3:
            if (
                path_marker_indices is None
                or beta_path_source is None
                or saved_path_ratios is None
                or beta_path_source.ndim != 2
                or beta_path_source.shape
                != (saved_path_ratios.size, path_marker_indices.size)
                or np.unique(path_marker_indices).size
                != path_marker_indices.size
                or np.any(
                    (path_marker_indices < 0)
                    | (path_marker_indices >= marker_count)
                )
                or not np.all(np.isfinite(beta_path_source))
                or not np.all(np.isfinite(saved_path_ratios))
                or saved_path_ratios.size < 1
                or saved_path_ratios.size > target_path_ratios.size
            ):
                raise ValueError("Schema-3 Lasso warm-state path is invalid.")
            ratios_match = np.allclose(
                saved_path_ratios,
                target_path_ratios[: saved_path_ratios.size],
                rtol=1e-10,
                atol=1e-12,
            )
            if ratios_match:
                path_candidate_unsorted = (
                    path_marker_indices
                    if grm_index is None
                    else grm_index.cache_variant_indices(path_marker_indices)
                )
                path_order = np.argsort(path_candidate_unsorted)
                candidate = path_candidate_unsorted[path_order]
                beta_path = beta_path_source[:, path_order]
                reuse_mode = "certified_path_prefix_for_validation"

    return {
        "candidate": candidate,
        "support": candidate_unsorted[order].copy(),
        "beta_snp_path": beta_path,
        "selected_lam_ratio": selected_ratio,
        "coordinate_system": (
            "single_grm_marker_index"
            if schema_version == 1
            else "source_variant_index"
        ),
        "reuse_mode": reuse_mode,
    }


def _write_lasso_warm_state(
    path: str,
    *,
    grm_index=None,
    candidate: np.ndarray,
    support: np.ndarray,
    selected_beta_snp: np.ndarray,
    selected_lam_ratio: float,
    beta_snp_path: np.ndarray | None = None,
    path_lam_ratios: np.ndarray | None = None,
) -> dict[str, object]:
    """Persist selected alpha plus an optional certified path prefix."""
    candidate_indices = np.asarray(candidate, dtype=np.int64).reshape(-1)
    support_indices = np.sort(
        np.asarray(support, dtype=np.int64).reshape(-1)
    )
    selected_beta = np.asarray(
        selected_beta_snp, dtype=np.float64
    ).reshape(-1)
    selected_ratio = _canonical_fixed_lam_ratio(
        float(selected_lam_ratio)
    )
    if (
        selected_beta.shape != candidate_indices.shape
        or np.unique(candidate_indices).size != candidate_indices.size
        or np.unique(support_indices).size != support_indices.size
        or not np.all(np.isfinite(selected_beta))
    ):
        raise ValueError("Cannot emit an invalid Lasso warm state.")

    candidate_position = {
        int(marker): int(position)
        for position, marker in enumerate(candidate_indices.tolist())
    }
    try:
        support_positions = np.asarray(
            [candidate_position[int(marker)] for marker in support_indices],
            dtype=np.int64,
        )
    except KeyError as exc:
        raise ValueError(
            "Lasso warm-state support is not contained in the candidate set."
        ) from exc
    support_beta = selected_beta[support_positions]
    source_support = (
        support_indices.copy()
        if grm_index is None
        else grm_index.source_variant_indices(support_indices)
    )
    source_order = np.argsort(source_support)
    source_support = source_support[source_order]
    support_beta = support_beta[source_order]

    full_beta_path = (
        selected_beta.reshape(1, -1)
        if beta_snp_path is None
        else np.asarray(beta_snp_path, dtype=np.float64)
    )
    full_path_ratios = (
        np.asarray([selected_ratio], dtype=np.float64)
        if path_lam_ratios is None
        else np.asarray(path_lam_ratios, dtype=np.float64).reshape(-1)
    )
    if (
        full_beta_path.ndim != 2
        or full_beta_path.shape
        != (full_path_ratios.size, candidate_indices.size)
        or full_path_ratios.size < 1
        or not np.all(np.isfinite(full_beta_path))
        or not np.all(np.isfinite(full_path_ratios))
        or np.any(full_path_ratios <= 0.0)
        or np.any(full_path_ratios > 1.0)
    ):
        raise ValueError("Cannot emit an invalid Lasso path warm state.")
    source_candidate = (
        candidate_indices.copy()
        if grm_index is None
        else grm_index.source_variant_indices(candidate_indices)
    )
    path_source_order = np.argsort(source_candidate)
    source_candidate = source_candidate[path_source_order]
    full_beta_path = full_beta_path[:, path_source_order]

    ensure_parent_dir(path)
    temporary = f"{path}.tmp.{os.getpid()}"
    with open(temporary, "wb") as handle:
        np.savez(
            handle,
            lasso_warm_state_schema_version=np.asarray(
                3, dtype=np.int64
            ),
            marker_indices=source_support,
            selected_beta_snp=support_beta.astype(np.float32),
            selected_lam_ratio=np.asarray(selected_ratio, dtype=np.float64),
            path_marker_indices=source_candidate,
            beta_snp_path=full_beta_path.astype(np.float32),
            path_lam_ratios=full_path_ratios,
        )
    os.replace(temporary, path)
    return {
        "status": "emitted",
        "path": os.path.abspath(path),
        "marker_count": int(support_indices.size),
        "path_marker_count": int(candidate_indices.size),
        "path_rows": int(full_path_ratios.size),
        "selected_lam_ratio": selected_ratio,
        "coordinate_system": (
            "single_grm_marker_index"
            if grm_index is None
            else "source_variant_index"
        ),
        "reuse_contract": "cross_partition_validation_or_fixed_refit",
    }

# ---------------------------------------------------------------------------
# Sparse marker-coordinate index
# ---------------------------------------------------------------------------

class MultiGRMIndex:
    """Map the sparse cache order to component-local and source coordinates.

    A component-partitioned source is physically cached as the concatenation
    of its components.  Sparse optimization works in that cache order, while
    BIM/PVAR reporting must use original source-variant rows.  This class is
    the single authority for translating between the two coordinate systems.
    """

    def __init__(self, streamers, component_variant_indices=None):
        self.streamers = tuple(streamers)
        if len(self.streamers) != 1:
            raise ValueError(
                "Sparse fitting accepts exactly one physical genotype source."
            )
        self.streamer = self.streamers[0]
        self._partitioned_single_streamer = component_variant_indices is not None
        self._source_variant_indices: np.ndarray | None = None

        if self._partitioned_single_streamer:
            streamer = self.streamer
            if not bool(getattr(streamer, "has_component_partition", False)):
                raise ValueError(
                    "component_variant_indices were supplied, but the genotype "
                    "streamer is not component-partitioned."
                )
            requested_groups = [
                np.asarray(group, dtype=np.int64).reshape(-1)
                for group in component_variant_indices
            ]
            self.n_grm = int(streamer.n_components)
            component_offsets = np.asarray(
                streamer._component_snp_offsets, dtype=np.int64
            ).reshape(-1)
            if component_offsets.shape != (self.n_grm + 1,):
                raise ValueError("Invalid component offsets in partitioned streamer.")
            self.m_per_grm = np.diff(component_offsets)
            cache_to_source = np.asarray(
                streamer._cache_to_source_variant_indices, dtype=np.int64
            ).reshape(-1)
            if cache_to_source.size != int(streamer.m):
                raise ValueError(
                    "Partitioned streamer's cache-to-source SNP map has the "
                    "wrong length."
                )
            if not np.array_equal(
                np.sort(cache_to_source),
                np.arange(int(streamer.m), dtype=np.int64),
            ):
                raise ValueError(
                    "Sparse component partitions must cover every source SNP "
                    "exactly once."
                )
            if len(requested_groups) != self.n_grm:
                raise ValueError(
                    "Component count mismatch between component spec and "
                    "genotype streamer."
                )
            for component_idx, requested in enumerate(requested_groups):
                start = int(component_offsets[component_idx])
                stop = int(component_offsets[component_idx + 1])
                actual = cache_to_source[start:stop]
                if not np.array_equal(actual, np.unique(requested)):
                    raise ValueError(
                        "Component SNP mapping mismatch between component spec "
                        f"and genotype streamer for component {component_idx}."
                    )
            self._source_variant_indices = cache_to_source.copy()
        else:
            self.n_grm = 1
            self.m_per_grm = np.asarray([int(self.streamer.m)], dtype=np.int64)

        self.offsets = np.zeros(self.n_grm + 1, dtype=np.int64)
        np.cumsum(self.m_per_grm, out=self.offsets[1:])
        self.m_total = int(self.offsets[-1])

    def _validated_global_indices(self, global_idx: np.ndarray) -> np.ndarray:
        idx = np.asarray(global_idx, dtype=np.int64)
        if idx.ndim != 1:
            raise ValueError("Global SNP indices must be one-dimensional.")
        if np.any((idx < 0) | (idx >= self.m_total)):
            raise IndexError(
                f"Global SNP indices must lie in [0, {self.m_total})."
            )
        return idx

    def global_to_local(
        self, global_idx: np.ndarray
    ) -> list[tuple[int, np.ndarray, np.ndarray]]:
        idx = self._validated_global_indices(global_idx)
        grm_ids = np.searchsorted(self.offsets[1:], idx, side="right")
        grm_ids = np.clip(grm_ids, 0, self.n_grm - 1)
        groups: list[tuple[int, np.ndarray, np.ndarray]] = []
        for grm_idx in range(self.n_grm):
            positions = np.flatnonzero(grm_ids == grm_idx)
            if positions.size == 0:
                continue
            local = idx[positions] - int(self.offsets[grm_idx])
            groups.append((grm_idx, local, positions))
        return groups

    def xtv_all(self, u_jax: jnp.ndarray, normalize: bool = False, *, dtype=np.float64) -> np.ndarray:
        return np.asarray(
            self.streamer.xtv(u_jax, normalize=normalize),
            dtype=dtype,
        )

    def extract_standardized_columns(
        self, global_idx: np.ndarray
    ) -> np.ndarray:
        idx = self._validated_global_indices(global_idx)
        return self.streamer.extract_standardized_columns(idx)

    def source_variant_indices(self, global_idx: np.ndarray) -> np.ndarray:
        idx = self._validated_global_indices(global_idx)
        if self._source_variant_indices is None:
            return idx.copy()
        return self._source_variant_indices[idx].copy()

    def cache_variant_indices(self, source_idx: np.ndarray) -> np.ndarray:
        """Translate original source rows to the current concatenated cache."""
        source = np.asarray(source_idx, dtype=np.int64).reshape(-1)
        if self._source_variant_indices is None:
            return self._validated_global_indices(source).copy()
        if np.any((source < 0) | (source >= self.m_total)):
            raise IndexError(
                f"Source SNP indices must lie in [0, {self.m_total})."
            )
        source_to_cache = np.empty(self.m_total, dtype=np.int64)
        source_to_cache[self._source_variant_indices] = np.arange(
            self.m_total, dtype=np.int64
        )
        return source_to_cache[source]

    def lookup_bim_rows(
        self, bed_prefixes: list[str], global_idx: np.ndarray
    ) -> dict[int, tuple[str, str, str, str, str, str]]:
        idx = self._validated_global_indices(global_idx)
        result: dict[int, tuple[str, str, str, str, str, str]] = {}
        source_idx = self.source_variant_indices(idx)
        source_rows = _lookup_bim_rows(bed_prefixes[0] + ".bim", source_idx)
        for global_snp, source_snp in zip(idx.tolist(), source_idx.tolist()):
            if int(source_snp) in source_rows:
                result[int(global_snp)] = source_rows[int(source_snp)]
        return result


class SingleGRMIndex(MultiGRMIndex):
    """Compatibility wrapper for callers that explicitly require K=1."""

    def __init__(self, streamers):
        if len(streamers) != 1:
            raise ValueError(
                "The sparse pipeline supports exactly one whole-genome GRM."
            )
        streamer = streamers[0]
        if bool(getattr(streamer, "has_component_partition", False)) or int(
            getattr(streamer, "n_components", 1)
        ) != 1:
            raise ValueError(
                "The single-GRM index does not accept component partitions."
            )
        super().__init__(streamers)

    def lookup_bim_rows(
        self, bed_prefix: str, marker_idx: np.ndarray
    ) -> dict[int, tuple[str, str, str, str, str, str]]:
        return super().lookup_bim_rows([bed_prefix], marker_idx)

def _lookup_bim_rows(bim_path: str, snp_indices: np.ndarray) -> dict[int, tuple[str, str, str, str, str, str]]:
    idx = np.asarray(snp_indices, dtype=np.int64)
    if idx.size == 0:
        return {}
    need = set(int(i) for i in idx.tolist())
    rows: dict[int, tuple[str, str, str, str, str, str]] = {}
    with open(bim_path, "r") as f:
        for i, line in enumerate(f):
            if i not in need:
                continue
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            rows[i] = (parts[0], parts[1], parts[2], parts[3], parts[4], parts[5])
            if len(rows) == len(need):
                break
    return rows


def _lookup_pvar_rows(
    pvar_path: str, snp_indices: np.ndarray,
) -> dict[int, tuple[str, str, str, str, str, str]]:
    """Look up variant annotation from a .pvar file for given indices."""
    idx = np.asarray(snp_indices, dtype=np.int64)
    if idx.size == 0:
        return {}
    need = set(int(i) for i in idx.tolist())
    rows: dict[int, tuple[str, str, str, str, str, str]] = {}
    data_line = 0
    with open(pvar_path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            if data_line not in need:
                data_line += 1
                continue
            parts = line.strip().split("\t")
            if len(parts) < 5:
                parts = line.strip().split()
            chrom = parts[0] if len(parts) >= 1 else "NA"
            pos = parts[1] if len(parts) >= 2 else "NA"
            snp_id = parts[2] if len(parts) >= 3 else f"SNP_{data_line}"
            a1 = parts[3] if len(parts) >= 4 else "NA"
            a2 = parts[4] if len(parts) >= 5 else "NA"
            rows[data_line] = (chrom, snp_id, "0", pos, a1, a2)
            if len(rows) == len(need):
                break
            data_line += 1
    return rows


@dataclasses.dataclass
class _PredictionFitContext:
    """Prediction genotype/covariate state shared by iterative validation."""

    fitter: object
    grm_index: MultiGRMIndex
    covar: np.ndarray | None
    sample_ids: list[str]
    dropped_ids: list[str]
    _close_callback: object
    _closed: bool = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        atexit.unregister(self._close_callback)
        self._close_callback()


def _build_prediction_fit_context(
    *,
    args,
    training_fitter,
    prediction_bed_list: list[str],
    prediction_pgen_prefix: str,
    covar_transform,
    call_width: int,
    cpu_threads: int,
    gpu_budget_bytes: float,
    ring_depth: int,
    component_variant_indices: list[np.ndarray],
    prediction_keep_path: str | None = None,
) -> _PredictionFitContext:
    """Build prediction state using training-only genotype standardization."""
    if len(training_fitter.streamers) != 1:
        raise RuntimeError("Sparse prediction requires exactly one genotype source.")
    training_streamer = training_fitter.streamers[0]
    if (
        training_streamer._means_host is None
        or training_streamer._inv_sds_host is None
    ):
        raise RuntimeError(
            "Sparse prediction requires retained training SNP "
            "standardization statistics."
        )
    standardization_overrides = [
        (training_streamer._means_host, training_streamer._inv_sds_host)
    ]

    if prediction_pgen_prefix:
        prediction_fam_path = make_nonbed_input_fam(
            pgen_prefix=prediction_pgen_prefix
        )
        atexit.register(cleanup_path, prediction_fam_path)
    else:
        prediction_fam_path = prediction_bed_list[0] + ".fam"

    keep_path = (
        args.prediction_keep_path
        if prediction_keep_path is None
        else prediction_keep_path
    )
    requested_prediction_ids = None
    if keep_path:
        if not os.path.exists(keep_path):
            raise SystemExit(
                "--prediction-keep-path does not exist: "
                f"{keep_path}"
            )
        requested_prediction_ids = read_keep_ids(keep_path)
    prediction_covar, prediction_ids, prediction_dropped = load_covar_aligned(
        prediction_fam_path,
        args.prediction_covar_txt or None,
        transform=covar_transform,
        keep_ids=requested_prediction_ids,
    )
    logger.info(
        "[prediction] loaded %s samples; dropped %s",
        len(prediction_ids),
        len(prediction_dropped),
    )

    prediction_sources = None
    prediction_sample_mask = None
    if prediction_pgen_prefix:
        prediction_sample_mask = compute_sample_mask(
            prediction_fam_path, prediction_ids
        )
        prediction_sources = [
            PgenGenoSource(
                prediction_pgen_prefix,
                sample_mask=prediction_sample_mask,
            )
        ]
        prediction_sample_mask = None
    else:
        n_prediction_bed = _bed_count(
            prediction_bed_list[0] + ".bed", "iid_count"
        )
        if n_prediction_bed != len(prediction_ids):
            prediction_sample_mask = compute_sample_mask(
                prediction_fam_path, prediction_ids
            )

    prediction_cfg_kwargs = dict(
        device=args.device,
        sample_mask=prediction_sample_mask,
        standardization_overrides=standardization_overrides,
        component_variant_indices=component_variant_indices or None,
        call_width=call_width,
        keep_host_stats=True,
        cpu_threads=cpu_threads,
        gpu_budget_bytes=gpu_budget_bytes,
        ring_depth=ring_depth,
        n_rand_vec=args.n_rand_vec,
        minq_iter=args.minq_iter,
        slq_samples=args.slq_samples,
        slq_m=args.slq_m,
        precond_rank=0,
        max_pcg_iters=args.max_pcg_iters,
        pcg_ridge=args.pcg_ridge,
        response_is_standardized=True,
        verbose=args.verbose,
    )
    if prediction_sources is not None:
        prediction_fitter = InfinitesimalREMLFitter(
            FitConfig(sources=prediction_sources, **prediction_cfg_kwargs)
        )
    else:
        prediction_fitter = InfinitesimalREMLFitter(
            FitConfig(
                bed_prefix=prediction_bed_list,
                **prediction_cfg_kwargs,
            )
        )
    close_callback = prediction_fitter.close
    atexit.register(close_callback)
    try:
        prediction_grm_index = MultiGRMIndex(
            prediction_fitter.streamers,
            component_variant_indices=component_variant_indices or None,
        )
    except Exception:
        atexit.unregister(close_callback)
        close_callback()
        raise

    return _PredictionFitContext(
        fitter=prediction_fitter,
        grm_index=prediction_grm_index,
        covar=prediction_covar,
        sample_ids=list(prediction_ids),
        dropped_ids=list(prediction_dropped),
        _close_callback=close_callback,
    )


def _emit_lasso_prediction(
    *,
    out_prefix: str,
    keep_path: str,
    prediction_context: _PredictionFitContext,
    prediction_bed_list: list[str],
    prediction_pgen_prefix: str,
    input_phenotype_mean: float,
    input_phenotype_standard_deviation: float,
    training_fitter,
    y_train: np.ndarray,
    train_covar: np.ndarray | None,
    train_support: np.ndarray,
    support_indices: np.ndarray,
    beta_cov: np.ndarray,
    beta_active: np.ndarray,
    theta: np.ndarray,
    pcg_tol: float,
    max_pcg_iters: int,
    background_effects=None,
) -> dict[str, object]:
    """Predict one cohort from an already fitted alpha/theta pair."""
    request_metadata = {
        "estimator_mode": "coherit",
        "genotype_standardization_source": "training_samples_only",
        "covariate_transform_source": "training_samples_only",
        "prediction_keep_path": keep_path or None,
        "prediction_genotype": {
            "format": "pgen" if prediction_pgen_prefix else "bed",
            "prefixes": (
                [prediction_pgen_prefix]
                if prediction_pgen_prefix
                else prediction_bed_list
            ),
        },
        "input_phenotype_standardization": {
            "mean": float(input_phenotype_mean),
            "standard_deviation": float(input_phenotype_standard_deviation),
        },
    }
    write_sparse_prediction_status(
        out_prefix=out_prefix,
        status="preparing",
        metadata={
            **request_metadata,
            "branch_outputs_emitted": False,
        },
    )
    prediction_ids = prediction_context.sample_ids
    try:
        prediction_support = (
            prediction_context.grm_index.extract_standardized_columns(
                support_indices
            ).astype(np.float32, copy=False)
        )
        lasso_prediction = predict_sparse_branch(
            name="lasso",
            fitter=training_fitter,
            test_fitter=prediction_context.fitter,
            y_train=y_train,
            train_covar=train_covar,
            test_covar=prediction_context.covar,
            train_active_geno=train_support,
            test_active_geno=prediction_support,
            beta_cov=beta_cov,
            beta_active=beta_active,
            theta=theta,
            pcg_tol=pcg_tol,
            max_pcg_iters=max_pcg_iters,
            background_effects=background_effects,
        )
    finally:
        prediction_context.close()

    branch_metadata = {
        "lasso": {
            "estimator_valid": True,
            "output_emitted": True,
            "invalid_reasons": [],
            "mean_estimator": "final_weighted_lasso",
            "covariance_estimator": "lasso_covariate_contrast_reml",
            "theta": np.asarray(theta, dtype=np.float64).tolist(),
            "residual": "y-X_beta_cov_lasso-Z_support_beta_lasso",
            "support_size": int(np.asarray(support_indices).size),
            "pcg_rel_res": lasso_prediction.pcg_rel_res,
            "pcg_iters": lasso_prediction.pcg_iters,
        },
    }
    prediction_paths = write_sparse_prediction_outputs(
        out_prefix=out_prefix,
        sample_ids=prediction_ids,
        lasso=lasso_prediction,
        metadata={
            **request_metadata,
            "branch_outputs_emitted": True,
            "emitted_branches": ["lasso"],
            "branches": branch_metadata,
        },
    )
    return {
        "requested": True,
        "status": "emitted",
        "n_samples": len(prediction_ids),
        "emitted_branches": ["lasso"],
        "paths": prediction_paths,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run sparse REML + LASSO pipeline on real genotype data.")
    # Exactly one genotype source must be supplied. It may be partitioned into
    # disjoint GRM components by --component-spec.
    p.add_argument("--bed-prefix", default=env("BED_PREFIX", ""),
                   help="One PLINK1 BED file prefix (no extension).")
    p.add_argument("--pgen-prefix", default=env("PGEN_PREFIX", ""),
                   help="PLINK2 PGEN file prefix (direct read, no conversion needed)")
    p.add_argument(
        "--component-spec",
        default=env("COMPONENT_SPEC", ""),
        help=(
            "Optional JSON/NPZ component spec partitioning every source SNP "
            "into one disjoint GRM. Omit for one whole-genome GRM."
        ),
    )
    p.add_argument(
        "--variance-components-init",
        default="",
        help=(
            "Optional warm start: JSON array with one value per GRM followed "
            "by residual variance."
        ),
    )
    p.add_argument(
        "--lasso-warm-state-in",
        default="",
        help=(
            "Optional sparse-state warm start. Validation mode re-optimizes "
            "and globally KKT-certifies every reused path point."
        ),
    )
    p.add_argument(
        "--lasso-warm-state-out",
        default="",
        help=(
            "Write the final globally KKT-certified sparse candidate/alpha "
            "path state. This can warm-start later validation or frozen-"
            "lambda fitting under a different GRM partition."
        ),
    )
    p.add_argument("--pheno-txt", default=env("PHENO_TXT", ""))
    p.add_argument("--covar-txt", default=env("COVAR_TXT", ""))
    p.add_argument(
        "--prediction-bed-prefix",
        default=env("PREDICTION_BED_PREFIX", ""),
        help="One prediction BED prefix.",
    )
    p.add_argument(
        "--prediction-pgen-prefix",
        default=env("PREDICTION_PGEN_PREFIX", ""),
        help="Prediction PGEN prefix.",
    )
    p.add_argument(
        "--prediction-covar-txt",
        default=env("PREDICTION_COVAR_TXT", ""),
        help=(
            "Prediction covariates transformed with training-set parameters."
        ),
    )
    p.add_argument(
        "--prediction-keep-path",
        default=env("PREDICTION_KEEP_PATH", ""),
        help="Optional IID keep file selecting prediction samples.",
    )
    p.add_argument(
        "--compute-effects",
        action="store_true",
        help=(
            "After the final alpha/theta fit, write nuisance, background, "
            "sparse-SNP, and total per-SNP effect sizes."
        ),
    )
    p.add_argument("--keep-path", default=env("KEEP_PATH", ""))
    p.add_argument("--keep-out", default=env("KEEP_OUT", ""))
    p.add_argument("--dropped-out", default=env("DROPPED_OUT", ""))
    p.add_argument("--out-prefix", default=env("OUT_PREFIX", "sparse_reml"))
    p.add_argument("--device", default=env("DEVICE", "gpu"))
    p.add_argument(
        "--cpu-threads",
        type=int,
        default=int(env("CPU_THREADS", "0")),
        help="CPU threads for source/build work (0 = auto-detect).",
    )
    p.add_argument("--call-width", type=int, default=0,
                   help="Call width w (0 = auto from planner)")
    p.add_argument(
        "--gpu-budget-gib",
        dest="gpu_budget_gib",
        type=float,
        default=float(env("GPU_BUDGET_GIB", "0")),
        help=(
            "Planner budget for active GPU allocations in GiB "
            "(0 = use 85%% of current free memory). JAX allocator reservation shown by nvidia-smi may "
            "be higher."
        ),
    )
    p.add_argument("--ring-depth", type=int, default=int(env("RING_DEPTH", "0")),
                   help="Pinned ring buffer depth (0 = auto, default 32)")
    p.add_argument("--n-rand-vec", type=int, default=100)
    p.add_argument("--slq-samples", type=int, default=100)
    p.add_argument("--slq-m", type=int, default=int(env("SLQ_M", "50")))
    p.add_argument("--minq-iter", type=int, default=int(env("MINQ_ITER", "50")))
    p.add_argument(
        "--export-ai",
        action="store_true",
        help=(
            "Export the last completed covariate-contrast REML average-"
            "information matrix, score, and restricted log-likelihood in "
            "the summary."
        ),
    )
    p.add_argument(
        "--reml-max-linesearch-trials",
        type=int,
        default=int(env("REML_MAX_LINESEARCH_TRIALS", "8")),
        help=(
            "Maximum strict REML backtracking trials. Eight trials permit "
            "steps through 1/128 while preserving monotone acceptance."
        ),
    )
    p.add_argument("--pcg-tol", type=float, default=float(env("PCG_TOL", "5e-3")))
    p.add_argument("--pcg-ridge", type=float, default=float(env("PCG_RIDGE", "1e-6")))
    p.add_argument("--max-pcg-iters", type=int, default=int(env("MAX_PCG_ITERS", "400")))
    p.add_argument("--outer-max", type=int, default=20)
    p.add_argument("--screen-topk", type=int, default=2000)
    p.add_argument("--candidate-k", type=int, default=256)
    p.add_argument(
        "--h2-abs-tol",
        type=float,
        default=1e-2,
        help=(
            "Absolute tolerance for the change in the primary COHERIT h2 "
            "estimate between consecutive outer updates."
        ),
    )
    p.add_argument(
        "--effect-rel-tol",
        type=float,
        default=5e-2,
        help=(
            "Relative tolerance for the change in the fitted fixed mean. "
            "This replaces exact selected-support equality."
        ),
    )
    p.add_argument("--lasso-lam-min-ratio", type=float, default=0.05)
    p.add_argument("--lasso-n-lambda", type=int, default=60)
    p.add_argument(
        "--lasso-fixed-lam-ratio",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    p.add_argument("--lasso-cd-max-iter", type=int, default=2000)
    p.add_argument("--lasso-cd-tol", type=float, default=1e-6)
    p.add_argument("--lasso-active-set-period", type=int, default=5)
    p.add_argument("--lasso-ridge", type=float, default=1e-6)
    p.add_argument(
        "--sparsity-validation-pheno-txt",
        default="",
        help=(
            "Phenotype for the prediction samples used to evaluate the full "
            "Lasso lambda path."
        ),
    )
    p.add_argument(
        "--sparsity-validation-out",
        default="",
        help=(
            "JSON audit of the lambda selected by validation predictive R2 "
            "inside every alpha/theta outer iteration."
        ),
    )
    p.add_argument(
        "--validation-early-stopping-lag",
        type=int,
        default=5,
        help=(
            "Stop the descending exact lambda path when the best validation "
            "predictive R2 in the latest lag points is below an earlier best. "
            "Only full-marker-KKT-certified points count."
        ),
    )
    p.add_argument("--proj-ridge", type=float, default=1e-6)
    p.add_argument(
        "--kkt-tol",
        type=float,
        default=None,
        help=(
            "Absolute LASSO KKT tolerance. The effective value cannot be "
            "smaller than max(1e-4, 2*--pcg-tol)."
        ),
    )
    p.add_argument(
        "--kkt-rel-tol",
        type=float,
        default=None,
        help=(
            "Relative LASSO KKT tolerance. The effective value cannot be "
            "smaller than max(1e-4, 2*--pcg-tol)."
        ),
    )
    p.add_argument(
        "--kkt-add-topk",
        type=int,
        default=256,
        help=(
            "Nominal outside-marker expansion batch per KKT round. A strict-"
            "violator overflow of at most one tenth is absorbed in the same round; "
            "otherwise the strongest strict violators are added in batches."
        ),
    )
    p.add_argument(
        "--kkt-max-rounds",
        type=int,
        default=30,
        help="Maximum candidate-expansion rounds used to certify global KKT optimality.",
    )
    p.add_argument(
        "--kkt-max-candidate",
        type=int,
        default=0,
        help="Optional hard cap for KKT-expanded candidate set size (0 = no explicit cap).",
    )
    p.add_argument(
        "--candidate-pcg-rhs-batch-size",
        type=int,
        default=1024,
        help=(
            "Maximum genotype RHS columns per candidate Hinv[Z] block-PCG "
            "solve. Bounds streamed-GRM X.T@V memory during large working-set "
            "expansion."
        ),
    )
    p.add_argument(
        "--basil-marker-batch-size",
        type=int,
        default=1000,
        help=(
            "Computational marker batch used by exact BASIL lambda-path "
            "rollout. This changes memory/runtime, not the certified solution."
        ),
    )
    p.add_argument(
        "--basil-lambda-block-size",
        type=int,
        default=10,
        help=(
            "Number of unresolved lambda points fitted per BASIL block. "
            "This changes memory/runtime, not the certified solution."
        ),
    )
    p.add_argument(
        "--basil-max-iterations",
        type=int,
        default=200,
        help="Maximum exact-prefix/enlargement iterations for one BASIL path.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        default=env("VERBOSE", "").strip().lower() in {"1", "true", "yes", "on"},
    )
    args = p.parse_args()
    # The Gram equations and the full-marker score both use PCG solutions.
    # Requiring a certificate far below that solve precision creates false
    # failures, so both KKT checks use one compatible numerical tolerance.
    kkt_floor = max(1e-4, 2.0 * float(args.pcg_tol))
    args.kkt_tol = max(
        kkt_floor,
        float(args.kkt_tol) if args.kkt_tol is not None else kkt_floor,
    )
    args.kkt_rel_tol = max(
        kkt_floor,
        float(args.kkt_rel_tol)
        if args.kkt_rel_tol is not None
        else kkt_floor,
    )
    return args


def _max_rel_change(new_v: np.ndarray, old_v: np.ndarray) -> float:
    denom = np.maximum(np.abs(old_v), 1e-6)
    return float(np.max(np.abs(new_v - old_v) / denom))


def _canonical_fixed_lam_ratio(value: float, *, atol: float = 1e-12) -> float:
    """Preserve the exact lambda-max endpoint across JSON round trips."""
    ratio = float(value)
    if abs(ratio - 1.0) <= float(atol):
        return 1.0
    return ratio


def _relative_fitted_mean_change(
    current: np.ndarray,
    previous: np.ndarray | None,
    phenotype: np.ndarray,
) -> float:
    """Relative change in the complete fitted fixed mean.

    This criterion is stable under equivalent support swaps among correlated
    variants, unlike exact selected-support equality.
    """
    if previous is None:
        return float("inf")
    current_arr = np.asarray(current, dtype=np.float64).reshape(-1)
    previous_arr = np.asarray(previous, dtype=np.float64).reshape(-1)
    phenotype_arr = np.asarray(phenotype, dtype=np.float64).reshape(-1)
    if current_arr.shape != previous_arr.shape or current_arr.shape != phenotype_arr.shape:
        return float("inf")
    denom = max(
        float(np.linalg.norm(current_arr)),
        float(np.linalg.norm(previous_arr)),
        1e-8 * float(np.linalg.norm(phenotype_arr)),
        np.finfo(np.float64).tiny,
    )
    return float(np.linalg.norm(current_arr - previous_arr) / denom)


def _heritability_converged(
    current_h2: float,
    previous_h2: float | None,
    *,
    abs_tol: float = 1e-2,
) -> tuple[bool, float]:
    """Check absolute change in the primary COHERIT heritability estimate."""
    if previous_h2 is None:
        return False, float("inf")
    current = float(current_h2)
    previous = float(previous_h2)
    change = abs(current - previous)
    tolerance = float(abs_tol)
    at_boundary = bool(
        np.isclose(change, tolerance, rtol=1e-12, atol=1e-15)
    )
    return bool(
        np.isfinite(change) and (change <= tolerance or at_boundary)
    ), float(change)


def _alignment_action(
    *,
    provisional_convergence: bool,
    h2_stable: bool,
    effect_stable: bool,
    outer: int,
    outer_max: int,
) -> str:
    """Certify the returned iterate, or reuse its Lasso in another update."""
    if provisional_convergence and h2_stable and effect_stable:
        return "converged"
    return "outer_max" if outer >= outer_max else "continue"


def _accepted_reml_theta(
    fit_result,
    *,
    expected_components: int,
    stage: str,
) -> tuple[np.ndarray, str]:
    """Return a valid, converged REML state.

    ``fit_reml`` returns the last accepted parameter vector when all
    line-search candidates are downhill. ``ll_down`` is an accepted terminal
    state: its rejected trial is discarded, but earlier accepted updates in
    the same covariance block are retained.
    """
    theta = np.asarray(fit_result.var_components, dtype=np.float64).reshape(-1)
    history = list(fit_result.history)
    stop_reason = (
        str(history[-1].get("stop_reason", "")) if history else ""
    )
    valid_theta = (
        theta.shape == (int(expected_components),)
        and np.all(np.isfinite(theta))
        and np.all(theta[:-1] >= 0.0)
        and theta[-1] > 0.0
    )
    last_history = history[-1] if history else {}
    candidate_accepted = bool(last_history.get("accepted", False))
    converged = bool(last_history.get("converged", False))
    usable_history = bool(
        history
        and converged
        and (
            (
                stop_reason in {"rel_dll", "scoring_step"}
                and candidate_accepted
            )
            or (stop_reason == "ll_down" and not candidate_accepted)
        )
    )
    if not valid_theta or not usable_history:
        raise RuntimeError(
            f"{stage} did not return a usable REML state: "
            f"theta={theta.tolist()}, history_rows={len(history)}, "
            f"stop_reason={stop_reason!r}, "
            f"last_accepted="
            f"{bool(history[-1].get('accepted', False)) if history else False}, "
            f"converged={converged}, "
            f"dll_true={last_history.get('dll_true')}, "
            f"max_rel_dp={last_history.get('max_rel_dp')}."
        )
    return theta, stop_reason


def _fit_covariate_contrast_residual_reml(
    fitter,
    residual: np.ndarray,
    theta_init: np.ndarray,
    *,
    covar: np.ndarray | None,
    h2_init: float,
    mean_information=None,
):
    """Profile the full nuisance design in the sparse variance-component block.

    The supplied residual may already subtract a fitted nuisance score.  This
    does not change the restricted likelihood because ``P_C C = 0``; passing
    the complete design ``C`` here is what makes the update equivalent to
    profiling the nuisance coefficients at every candidate covariance.

    The sparse pipeline standardizes the phenotype once at input.  This REML
    block therefore consumes the residual on that same scale and performs no
    response rescaling.
    """
    residual = np.asarray(residual, dtype=np.float32).reshape(-1)
    theta = np.asarray(theta_init, dtype=np.float32).reshape(-1)
    if residual.size < 2:
        raise ValueError("Covariate-contrast REML requires at least two samples.")
    if covar is None:
        nuisance_design = np.ones((residual.size, 1), dtype=np.float32)
    else:
        nuisance_design = np.asarray(covar, dtype=np.float32)
        if nuisance_design.ndim == 1:
            nuisance_design = nuisance_design[:, None]
        if (
            nuisance_design.ndim != 2
            or nuisance_design.shape[0] != residual.size
            or nuisance_design.shape[1] == 0
        ):
            raise ValueError(
                "Covariate design must have shape (n_samples, n_covariates)."
            )
        if not np.all(np.isfinite(nuisance_design)):
            raise ValueError("Covariate design contains non-finite values.")
    return fitter.fit_infinitesimal(
        jnp.asarray(residual, dtype=jnp.float32),
        jnp.asarray(nuisance_design, dtype=jnp.float32),
        h2_init=float(h2_init),
        var_components_init=jnp.asarray(theta, dtype=jnp.float32),
        **({"mean_information": mean_information} if mean_information is not None else {}),
    )


def _require_pcg_converged(
    rel_res,
    *,
    tol: float,
    iters: int,
    maxiter: int,
    stage: str,
) -> float:
    """Reject sparse-pipeline statistics built from an unconverged PCG solve."""
    rel = float(np.asarray(jax.device_get(rel_res)))
    if (not np.isfinite(rel)) or rel > float(tol) * 1.05:
        raise RuntimeError(
            f"{stage} PCG did not converge: relative residual={rel:.3e}, "
            f"tolerance={float(tol):.3e}, iterations={int(iters)}/{int(maxiter)}."
        )
    return rel


def _true_pcg_relative_residual(hv, rhs, solution) -> float:
    """Recompute ``||B-HX||/||B||`` instead of trusting PCG recurrence state."""
    rhs_arr = jnp.asarray(rhs)
    solution_arr = jnp.asarray(solution)
    true_residual = rhs_arr - hv(solution_arr)
    denominator = jnp.linalg.norm(rhs_arr, axis=0) + 1e-12
    relative = jnp.max(
        jnp.linalg.norm(true_residual, axis=0) / denominator
    )
    return float(np.asarray(jax.device_get(relative)))


def _solve_hinv_columns_batched(
    *,
    hv,
    precond,
    rhs: np.ndarray,
    warm_start: np.ndarray | None,
    tol: float,
    maxiter: int,
    batch_size: int,
    stage: str,
) -> tuple[np.ndarray, dict[str, object]]:
    """Solve ``H X = B`` in bounded RHS batches.

    A streamed GRM matvec forms an ``X.T @ V`` intermediate whose memory is
    proportional to the number of right-hand sides.  Candidate expansion can
    therefore make one monolithic Hinv[Z] solve much larger than the Lasso
    Gram matrix itself.  Fixed-size batches bound that temporary while still
    using block PCG inside every batch.
    """
    rhs_np = np.asarray(rhs, dtype=np.float32)
    if rhs_np.ndim != 2:
        raise ValueError("Batched PCG right-hand side must be a matrix.")
    n_rows, n_columns = rhs_np.shape
    batch = int(batch_size)
    if batch < 1:
        raise ValueError("Batched PCG size must be positive.")
    warm_np = None
    if warm_start is not None:
        warm_np = np.asarray(warm_start, dtype=np.float32)
        if warm_np.shape != rhs_np.shape:
            raise ValueError("Batched PCG warm start does not align with RHS.")
        if not np.all(np.isfinite(warm_np)):
            raise ValueError("Batched PCG warm start must be finite.")
    if not np.all(np.isfinite(rhs_np)):
        raise ValueError("Batched PCG right-hand side must be finite.")
    if n_columns == 0:
        return np.empty((n_rows, 0), dtype=np.float32), {
            "batch_size": batch,
            "n_batches": 0,
            "n_columns": 0,
            "max_reported_relative_residual": 0.0,
            "max_true_relative_residual": 0.0,
            "max_iterations": 0,
            "total_batch_iterations": 0,
        }

    solution_np = np.empty_like(rhs_np)
    max_reported = 0.0
    max_true = 0.0
    max_iterations = 0
    total_iterations = 0
    n_batches = 0
    for start in range(0, n_columns, batch):
        stop = min(start + batch, n_columns)
        rhs_batch = jnp.asarray(rhs_np[:, start:stop], dtype=jnp.float32)
        warm_batch = (
            jnp.asarray(warm_np[:, start:stop], dtype=jnp.float32)
            if warm_np is not None
            else None
        )
        solution, reported_residual, iterations = pcg_solve(
            hv,
            rhs_batch,
            M=precond,
            tol=float(tol),
            maxiter=int(maxiter),
            X0=warm_batch,
        )
        reported = _require_pcg_converged(
            reported_residual,
            tol=float(tol),
            iters=int(iterations),
            maxiter=int(maxiter),
            stage=f"{stage} columns {start}:{stop}",
        )
        true_residual = _true_pcg_relative_residual(
            hv, rhs_batch, solution
        )
        if not np.isfinite(true_residual):
            raise RuntimeError(
                f"{stage} columns {start}:{stop} produced a non-finite "
                "true residual."
            )
        solution_np[:, start:stop] = np.asarray(
            jax.device_get(solution), dtype=np.float32
        )
        max_reported = max(max_reported, float(reported))
        max_true = max(max_true, float(true_residual))
        max_iterations = max(max_iterations, int(iterations))
        total_iterations += int(iterations)
        n_batches += 1

    return solution_np, {
        "batch_size": batch,
        "n_batches": int(n_batches),
        "n_columns": int(n_columns),
        "max_reported_relative_residual": float(max_reported),
        "max_true_relative_residual": float(max_true),
        "max_iterations": int(max_iterations),
        "total_batch_iterations": int(total_iterations),
    }


def _chive_q_hat_given_active(
    z_active: np.ndarray,
    y: np.ndarray,
    beta_active: np.ndarray,
) -> tuple[float, float, float]:
    """
    CHIVE single-sample estimator on the active SNP set:
        Q = (1/n)||Z_S b_S||^2 + (2/n) b_S^T Z_S^T (y - Z_S b_S)
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    beta = np.asarray(beta_active, dtype=np.float64).reshape(-1)
    if z_active.size == 0 or beta.size == 0:
        return 0.0, 0.0, 0.0

    Zs = np.asarray(z_active, dtype=np.float64)
    if Zs.ndim != 2 or Zs.shape[0] != y.size or Zs.shape[1] != beta.size:
        raise ValueError("CHIVE input shape mismatch.")

    n = float(y.size)
    g = Zs @ beta
    r = y - g
    term1 = float(g @ g / n)
    term2 = float(2.0 * (beta @ (Zs.T @ r)) / n)
    return term1 + term2, term1, term2


def _standardize_phenotype_at_input(
    y: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Standardize the phenotype once at the sparse-pipeline input boundary."""
    y_standardized, y_mean, y_scale = standardize_response(
        jnp.asarray(np.asarray(y, dtype=np.float32).reshape(-1), dtype=jnp.float32)
    )
    standardized_host, mean_host, scale_host = jax.device_get(
        (y_standardized, y_mean, y_scale)
    )
    return (
        np.asarray(standardized_host, dtype=np.float32),
        float(mean_host),
        float(scale_host),
    )


def _sparse_dense_h2(
    q_sparse: float,
    background_genetic_variance: float,
    residual_variance: float,
) -> float:
    """Combine sparse, dense-background, and residual variance on one scale."""
    genetic = float(q_sparse) + float(background_genetic_variance)
    residual = float(residual_variance)
    denominator = genetic + residual
    if (
        not np.isfinite(genetic)
        or not np.isfinite(residual)
        or not np.isfinite(denominator)
        or denominator <= 0.0
    ):
        return float("nan")
    return genetic / denominator


def _outer_coherit_h2_from_fitted_sparse_mean(
    sparse_mean: np.ndarray,
    residual: np.ndarray,
    *,
    background_genetic_variance: float,
    residual_variance: float,
    mean_uncertainty_trace: float = 0.0,
) -> tuple[float, float]:
    """Return current COHERIT h2 and calibrated sparse variance.

    This is the same calibrated sparse-variance functional used by the final
    primary estimator, expressed through the already available fitted sparse
    mean ``g`` and residual ``r``:

        q_sparse = (g'g + 2 g'r - tr(Sigma_g)) / n.

    Using these vectors avoids another genotype matrix product inside the
    outer convergence check.
    """
    sparse_arr = np.asarray(sparse_mean, dtype=np.float64).reshape(-1)
    residual_arr = np.asarray(residual, dtype=np.float64).reshape(-1)
    if sparse_arr.shape != residual_arr.shape or sparse_arr.size == 0:
        raise ValueError(
            "Sparse fitted mean and residual must be non-empty aligned vectors."
        )
    n_samples = float(sparse_arr.size)
    q_sparse = float(
        (
            sparse_arr @ sparse_arr
            + 2.0 * (sparse_arr @ residual_arr)
            - float(mean_uncertainty_trace)
        )
        / n_samples
    )
    h2 = _sparse_dense_h2(
        q_sparse,
        background_genetic_variance,
        residual_variance,
    )
    return float(h2), float(q_sparse)


def _finite_float_or_none(value: float) -> float | None:
    """Return a JSON-safe finite scalar, or ``None`` when unavailable."""
    value_f = float(value)
    return value_f if np.isfinite(value_f) else None


def _json_safe_value(value):
    """Recursively replace non-finite numeric diagnostics by JSON ``null``."""
    if isinstance(value, dict):
        return {
            str(key): _json_safe_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _select_converged_validation_path_index(
    path_rows: list[dict],
    metrics: list[dict],
) -> int:
    """Select predictive R2 only among full-genome KKT path solutions."""
    if len(path_rows) != len(metrics) or not path_rows:
        raise ValueError("Lasso path and validation metrics must align.")
    eligible = [
        index
        for index, (row, metric) in enumerate(zip(path_rows, metrics))
        if bool(row.get("converged", False))
        and bool(row.get("kkt_passed", False))
        and bool(row.get("global_kkt_passed", False))
        and metric.get("predictive_r2") is not None
        and np.isfinite(float(metric["predictive_r2"]))
    ]
    if not eligible:
        raise RuntimeError(
            "No converged full-genome-KKT Lasso path point has finite "
            "validation predictive R2."
        )
    return max(
        eligible,
        key=lambda index: (
            float(metrics[index]["predictive_r2"]),
            -int(path_rows[index]["k"]),
            float(path_rows[index]["lam_ratio"]),
        ),
    )


def _materialize_validation_selected_lasso(
    lasso_path: dict,
    metrics: list[dict],
    *,
    selected_index: int,
    path_prediction_pcg_res: float,
    path_prediction_pcg_iters: int,
) -> tuple[dict, dict]:
    """Make one validation-selected path row the alpha used downstream."""
    path_rows = list(lasso_path["path"])
    index = int(selected_index)
    if not 0 <= index < len(path_rows) or len(metrics) != len(path_rows):
        raise ValueError("Validation-selected path index is out of range.")
    beta_snp_path = np.asarray(lasso_path["beta_snp_path"], dtype=np.float64)
    beta_cov_path = np.asarray(lasso_path["beta_cov_path"], dtype=np.float64)
    if beta_snp_path.shape[0] != len(path_rows) or beta_cov_path.shape[0] != len(
        path_rows
    ):
        raise ValueError("Lasso coefficient paths do not align with diagnostics.")

    selected_row = path_rows[index]
    if not (
        bool(selected_row.get("converged", False))
        and bool(selected_row.get("kkt_passed", False))
        and bool(selected_row.get("global_kkt_passed", False))
    ):
        raise RuntimeError(
            "Validation-selected Lasso path point is not full-genome-KKT "
            "converged."
        )
    beta_snp = beta_snp_path[index].copy()
    beta_cov = beta_cov_path[index].copy()
    active_idx = np.flatnonzero(beta_snp != 0.0).astype(np.int64)
    selected_metric = dict(metrics[index])
    merged_path = merge_path_diagnostics(path_rows, metrics)
    selection_record = {
        "selection_metric": (
            "predictive_r2_one_minus_sse_over_sst_total_phenotype_prediction"
        ),
        "selected": {
            "path_index": index,
            "lam": float(selected_row["lam"]),
            "lam_ratio": float(selected_row["lam_ratio"]),
            "support_size": int(active_idx.size),
            **selected_metric,
        },
        "path_prediction_pcg_true_res": float(path_prediction_pcg_res),
        "path_prediction_pcg_iters": int(path_prediction_pcg_iters),
        "path": merged_path,
    }

    selected = dict(lasso_path)
    selected.update(
        {
            "beta_cov": beta_cov,
            "beta_snp": beta_snp,
            "active_idx": active_idx,
            "lam": float(selected_row["lam"]),
            "selected_index": index,
            "selection_method": "validation_predictive_r2",
            "selected_lam_ratio": float(selected_row["lam_ratio"]),
            "validation_selection": selection_record,
        }
    )
    return selected, selection_record


def _validation_path_early_stopping_decision(
    metrics: list[dict],
    *,
    stopping_lag: int,
) -> dict[str, object]:
    """Apply predictive-R2 early stopping to an exact path prefix.

    For a lag of ``L``, stop when the best metric before the most recent
    ``L`` certified models is strictly better than every metric in that recent
    window.  Equal/flat leading metrics therefore cannot stop the path before
    an actual earlier peak has appeared.
    """
    lag = int(stopping_lag)
    if lag < 1:
        raise ValueError("Validation early-stopping lag must be positive.")
    values = np.asarray(
        [metric.get("predictive_r2") for metric in metrics],
        dtype=np.float64,
    )
    if values.size > 0 and not np.all(np.isfinite(values)):
        raise ValueError(
            "Validation early stopping requires finite predictive R2 at "
            "every exact point."
        )
    best_index = int(np.argmax(values)) if values.size > 0 else None
    best_r2 = float(values[best_index]) if best_index is not None else None
    if values.size <= lag:
        return {
            "stopped": False,
            "stopping_lag": lag,
            "n_evaluated": int(values.size),
            "best_path_index": best_index,
            "best_predictive_r2": best_r2,
            "earlier_max": None,
            "recent_max": None,
            "reason": "insufficient_certified_points",
        }
    earlier_max = float(np.max(values[:-lag]))
    recent_max = float(np.max(values[-lag:]))
    stopped = bool(earlier_max > recent_max)
    return {
        "stopped": stopped,
        "stopping_lag": lag,
        "n_evaluated": int(values.size),
        "best_path_index": best_index,
        "best_predictive_r2": best_r2,
        "earlier_max": earlier_max,
        "recent_max": recent_max,
        "reason": (
            "earlier_peak_exceeds_latest_window"
            if stopped
            else "latest_window_still_reaches_global_prefix_max"
        ),
    }


def _evaluate_lasso_path_on_validation(
    *,
    args,
    fitter,
    prediction_context: _PredictionFitContext,
    y_train: np.ndarray,
    train_covar: np.ndarray | None,
    train_candidate: np.ndarray,
    candidate: np.ndarray,
    beta_cov_path: np.ndarray,
    beta_snp_path: np.ndarray,
    theta: np.ndarray,
    validation_outcome: np.ndarray,
    hinv_residual_path: np.ndarray | None = None,
    training_score_path: np.ndarray | None = None,
) -> dict[str, object]:
    """Evaluate one already-certified coefficient block on validation data."""
    validation_candidate = (
        prediction_context.grm_index.extract_standardized_columns(candidate)
        .astype(np.float32, copy=False)
    )
    path_predictor = (
        predict_sparse_path_partitioned
        if prediction_context.grm_index.n_grm > 1
        else predict_sparse_path_single_grm
    )
    path_prediction = path_predictor(
        fitter=fitter,
        test_fitter=prediction_context.fitter,
        y_train=y_train,
        train_covar=train_covar,
        test_covar=prediction_context.covar,
        train_candidate_geno=train_candidate,
        test_candidate_geno=validation_candidate,
        beta_cov_path=beta_cov_path,
        beta_candidate_path=beta_snp_path,
        theta=theta,
        pcg_tol=float(args.pcg_tol),
        max_pcg_iters=int(args.max_pcg_iters),
        hinv_residual_path=hinv_residual_path,
        training_score_path=training_score_path,
    )
    return {
        "metrics": evaluate_prediction_path(
            path_prediction.phenotype_prediction,
            validation_outcome,
        ),
        "pcg_rel_res": float(path_prediction.pcg_rel_res),
        "pcg_iters": int(path_prediction.pcg_iters),
    }


def _select_lasso_by_validation_prediction(
    *,
    args,
    fitter,
    prediction_context: _PredictionFitContext,
    y_train: np.ndarray,
    train_covar: np.ndarray | None,
    train_candidate: np.ndarray,
    candidate: np.ndarray,
    lasso_path: dict,
    theta: np.ndarray,
    validation_outcome: np.ndarray,
) -> tuple[dict, dict]:
    """Evaluate the complete path and return the validation-selected alpha."""
    evaluated = _evaluate_lasso_path_on_validation(
        args=args,
        fitter=fitter,
        prediction_context=prediction_context,
        y_train=y_train,
        train_covar=train_covar,
        train_candidate=train_candidate,
        candidate=candidate,
        beta_cov_path=lasso_path["beta_cov_path"],
        beta_snp_path=lasso_path["beta_snp_path"],
        theta=theta,
        validation_outcome=validation_outcome,
    )
    metrics = list(evaluated["metrics"])
    selected_index = _select_converged_validation_path_index(
        list(lasso_path["path"]), metrics
    )
    return _materialize_validation_selected_lasso(
        lasso_path,
        metrics,
        selected_index=selected_index,
        path_prediction_pcg_res=float(evaluated["pcg_rel_res"]),
        path_prediction_pcg_iters=int(evaluated["pcg_iters"]),
    )


def _write_iterative_validation_output(
    *,
    output_path: str,
    phenotype_path: str,
    validation_outcome: np.ndarray,
    selection_trace: list[dict[str, object]],
    final_lasso: dict,
    final_candidate: np.ndarray,
    grm_index: MultiGRMIndex,
    theta: np.ndarray,
    outer_converged: bool,
    outer_stop_reason: str,
) -> dict[str, object]:
    """Write the audit proving validation selection occurred in every round."""
    if not selection_trace:
        raise RuntimeError("Iterative validation selection trace is empty.")
    final_selection = final_lasso.get("validation_selection")
    if not isinstance(final_selection, dict):
        raise RuntimeError("Final Lasso lacks iterative validation diagnostics.")
    candidate = np.asarray(final_candidate, dtype=np.int64).reshape(-1)
    active_local = np.asarray(final_lasso["active_idx"], dtype=np.int64).reshape(-1)
    support = candidate[active_local]
    selected = {
        **dict(final_selection["selected"]),
        "lam_max": float(final_lasso["lam_max"]),
        "support_indices": support.tolist(),
        "support_source_indices": (
            grm_index.source_variant_indices(support).tolist()
        ),
    }
    payload = {
        "schema_version": 4,
        "selection_role": "inside_every_alpha_theta_outer_iteration",
        "selection_metric": (
            "predictive_r2_one_minus_sse_over_sst_total_phenotype_prediction"
        ),
        "validation_phenotype_path": os.path.abspath(phenotype_path),
        "n_validation_samples": int(
            np.asarray(validation_outcome).reshape(-1).size
        ),
        "outer_converged": bool(outer_converged),
        "outer_stop_reason": str(outer_stop_reason),
        "theta_final": np.asarray(theta, dtype=np.float64).tolist(),
        "n_path_selections": int(len(selection_trace)),
        "outer_selection_trace": selection_trace,
        "final_selected": selected,
        "path": list(final_selection["path"]),
    }
    output_paths = write_selection_outputs(
        output_path, _json_safe_value(payload)
    )
    return {
        "requested": True,
        "status": "emitted",
        "selection_role": payload["selection_role"],
        "selection_metric": payload["selection_metric"],
        "selected_lam_ratio": float(selected["lam_ratio"]),
        "selected_support_size": int(selected["support_size"]),
        "validation_predictive_r2": float(
            selected["predictive_r2"]
        ),
        "n_path_selections": int(len(selection_trace)),
        "outputs": output_paths,
    }


def _lasso_kkt_certificate_from_scores(
    *,
    score: np.ndarray,
    beta: np.ndarray,
    lam: float,
    abs_tol: float,
    rel_tol: float,
) -> dict[str, float | bool]:
    """Check all active and inactive Lasso KKT equations from score values."""
    score_arr = np.asarray(score, dtype=np.float64).reshape(-1)
    beta_arr = np.asarray(beta, dtype=np.float64).reshape(-1)
    lam_f = float(lam)
    tolerance = max(
        float(abs_tol),
        float(rel_tol) * max(1.0, abs(lam_f)),
    )
    if (
        score_arr.shape != beta_arr.shape
        or not np.all(np.isfinite(score_arr))
        or not np.all(np.isfinite(beta_arr))
        or not np.isfinite(lam_f)
        or lam_f < 0.0
    ):
        return {
            "passed": False,
            "tolerance": tolerance,
            "max_active_error": float("inf"),
            "max_inactive_excess": float("inf"),
        }

    active = beta_arr != 0.0
    if np.any(active):
        max_active_error = float(
            np.max(
                np.abs(
                    score_arr[active]
                    - lam_f * np.sign(beta_arr[active])
                )
            )
        )
    else:
        max_active_error = 0.0
    if np.any(~active):
        max_inactive_excess = float(
            max(np.max(np.abs(score_arr[~active])) - lam_f, 0.0)
        )
    else:
        max_inactive_excess = 0.0
    return {
        "passed": bool(
            max_active_error <= tolerance
            and max_inactive_excess <= tolerance
        ),
        "tolerance": tolerance,
        "max_active_error": max_active_error,
        "max_inactive_excess": max_inactive_excess,
    }


def _coherit_estimator_guard(
    *,
    alpha_theta_pair_certified: bool,
    lasso_quadratics_available: bool,
    h2_chive: float,
) -> dict[str, object]:
    """Validate the sole COHERIT estimator without baseline substitution."""
    lasso_outputs_finite = bool(np.isfinite(float(h2_chive)))

    lasso_reasons: list[str] = []
    if not alpha_theta_pair_certified:
        lasso_reasons.append("penalized_alpha_theta_pair_not_certified")
    if not lasso_quadratics_available:
        lasso_reasons.append("lasso_quadratic_unavailable")
    if not lasso_outputs_finite:
        lasso_reasons.append("nonfinite_lasso_estimator")

    lasso_branch_valid = bool(
        alpha_theta_pair_certified
        and lasso_quadratics_available
        and lasso_outputs_finite
    )
    return {
        "lasso_branch_valid": lasso_branch_valid,
        "lasso_outputs_finite": lasso_outputs_finite,
        "lasso_branch_invalid_reasons": lasso_reasons,
    }


def _sparse_output_contract() -> dict[str, object]:
    """Return the fixed output contract for the sole COHERIT mode."""
    return {
        "sparse_output_schema_version": 11,
        "estimator_mode": "coherit",
        "computed_estimators": ["h2_chive"],
        "selected_snp_columns": [
            "snp_index",
            "source_snp_index",
            "grm",
            "chr",
            "snp_id",
            "cm",
            "bp",
            "a1",
            "a2",
            "beta_lasso",
        ],
    }


def _lasso_residual(
    y: np.ndarray,
    covar: np.ndarray | None,
    geno: np.ndarray,
    beta_cov: np.ndarray,
    beta_snp: np.ndarray,
) -> np.ndarray:
    """Residual for the current weighted LASSO solution."""
    resid = np.asarray(y, dtype=np.float64).reshape(-1).copy()
    if covar is not None and covar.size > 0 and beta_cov.size > 0:
        resid -= np.asarray(covar, dtype=np.float64) @ np.asarray(beta_cov, dtype=np.float64).reshape(-1)
    if geno.size > 0 and beta_snp.size > 0:
        resid -= np.asarray(geno, dtype=np.float64) @ np.asarray(beta_snp, dtype=np.float64).reshape(-1)
    return resid


def _outside_kkt_violators(
    *,
    score_abs: np.ndarray,
    candidate: np.ndarray,
    lam: float,
    abs_tol: float,
    rel_tol: float,
) -> tuple[np.ndarray, float, float]:
    """
    Return outside-candidate SNPs violating inactive LASSO KKT conditions.

    The inactive full-genome condition is |x_j^T H^{-1} r| <= lambda for
    every SNP not included in the candidate LASSO system.
    """
    scores = np.asarray(score_abs, dtype=np.float64).reshape(-1)
    cand = np.asarray(candidate, dtype=np.int64).reshape(-1)
    lam_f = float(lam)
    tol = max(float(abs_tol), float(rel_tol) * max(1.0, abs(lam_f)))
    threshold = lam_f + tol

    outside = np.ones(scores.size, dtype=bool)
    if cand.size > 0:
        outside[cand] = False
    outside_scores = scores[outside]
    max_outside = float(np.max(outside_scores)) if outside_scores.size > 0 else 0.0
    viol_mask = outside & (scores > threshold)
    violators = np.flatnonzero(viol_mask).astype(np.int64)
    return violators, max_outside, threshold


def _candidate_lasso_kkt_from_scores(
    *,
    score: np.ndarray,
    candidate: np.ndarray,
    beta_candidate: np.ndarray,
    lam: float,
    abs_tol: float,
    rel_tol: float,
) -> dict[str, object]:
    """Return signed full/candidate certificates and outside violations."""
    score_arr = np.asarray(score, dtype=np.float64).reshape(-1)
    candidate_arr = np.asarray(candidate, dtype=np.int64).reshape(-1)
    beta_arr = np.asarray(beta_candidate, dtype=np.float64).reshape(-1)
    lam_f = float(lam)
    abs_tol_f = float(abs_tol)
    rel_tol_f = float(rel_tol)
    if (
        not np.all(np.isfinite(score_arr))
        or not np.all(np.isfinite(beta_arr))
        or not np.isfinite(lam_f)
        or lam_f < 0.0
        or not np.isfinite(abs_tol_f)
        or not np.isfinite(rel_tol_f)
        or abs_tol_f < 0.0
        or rel_tol_f < 0.0
    ):
        raise ValueError("Full-p KKT inputs must be finite and valid.")
    if candidate_arr.size != beta_arr.size:
        raise ValueError("Candidate and coefficient sizes do not match.")
    if (
        candidate_arr.size > 0
        and (
            np.any(candidate_arr < 0)
            or np.any(candidate_arr >= score_arr.size)
            or np.unique(candidate_arr).size != candidate_arr.size
        )
    ):
        raise ValueError("Candidate indices are invalid for the score vector.")

    beta_global = np.zeros(score_arr.size, dtype=np.float64)
    beta_global[candidate_arr] = beta_arr
    full_certificate = _lasso_kkt_certificate_from_scores(
        score=score_arr,
        beta=beta_global,
        lam=lam,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
    )
    candidate_certificate = _lasso_kkt_certificate_from_scores(
        score=score_arr[candidate_arr],
        beta=beta_arr,
        lam=lam,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
    )
    outside_violators, max_outside, threshold = _outside_kkt_violators(
        score_abs=np.abs(score_arr),
        candidate=candidate_arr,
        lam=lam,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
    )
    return {
        "full_certificate": full_certificate,
        "candidate_certificate": candidate_certificate,
        "outside_violators": outside_violators,
        "max_outside_score": max_outside,
        "threshold": threshold,
    }


def _build_hinv_lasso_residual_path(
    *,
    hinv_y: np.ndarray,
    hinv_covar: np.ndarray | None,
    hinv_geno: np.ndarray,
    beta_cov_path: np.ndarray,
    beta_snp_path: np.ndarray,
) -> np.ndarray:
    """Build H^-1 residual for a complete Lasso path with batched GEMMs.

    The candidate Gram system already used the supplied H^-1 y,
    H^-1 C and H^-1 Z solves. Recombining those same solutions keeps
    the outside-marker KKT score consistent with that convex objective and
    avoids one PCG solve per lambda.
    """
    hy = np.asarray(hinv_y, dtype=np.float32).reshape(-1)
    hz = np.asarray(hinv_geno, dtype=np.float32)
    beta_z = np.asarray(beta_snp_path, dtype=np.float32)
    beta_c = np.asarray(beta_cov_path, dtype=np.float32)
    if hz.ndim != 2 or hz.shape[0] != hy.size:
        raise ValueError("Hinv genotype path basis has an invalid shape.")
    if beta_z.ndim != 2 or beta_z.shape[1] != hz.shape[1]:
        raise ValueError("SNP coefficient path does not align with Hinv genotype.")
    if beta_c.ndim != 2 or beta_c.shape[0] != beta_z.shape[0]:
        raise ValueError("Covariate and SNP coefficient paths do not align.")
    if not (
        np.all(np.isfinite(hy))
        and np.all(np.isfinite(hz))
        and np.all(np.isfinite(beta_z))
        and np.all(np.isfinite(beta_c))
    ):
        raise ValueError("Lasso path residual inputs must be finite.")

    n_path = int(beta_z.shape[0])
    residual_path = np.repeat(hy[:, None], n_path, axis=1)
    if beta_c.shape[1] > 0:
        if hinv_covar is None:
            raise ValueError("Covariate coefficients require Hinv covariates.")
        hc = np.asarray(hinv_covar, dtype=np.float32)
        if hc.shape != (hy.size, beta_c.shape[1]):
            raise ValueError("Hinv covariates do not align with coefficient path.")
        if not np.all(np.isfinite(hc)):
            raise ValueError("Hinv covariates must be finite.")
        residual_path -= hc @ np.ascontiguousarray(beta_c.T)
    elif hinv_covar is not None:
        hc = np.asarray(hinv_covar)
        if hc.ndim != 2 or hc.shape[0] != hy.size:
            raise ValueError("Hinv covariates have an invalid shape.")
    if hz.shape[1] > 0:
        residual_path -= hz @ np.ascontiguousarray(beta_z.T)
    if not np.all(np.isfinite(residual_path)):
        raise RuntimeError("Batched Hinv residual path is non-finite.")
    return np.asarray(residual_path, dtype=np.float32)


def _certify_complete_lasso_path_kkt_from_scores(
    *,
    score_path: np.ndarray,
    candidate: np.ndarray,
    beta_candidate_path: np.ndarray,
    path_rows: list[dict],
    abs_tol: float,
    rel_tol: float,
) -> dict[str, object]:
    """Certify outside-marker KKT conditions for every Lasso path point.

    Candidate-set score KKT is checked by the coordinate-descent solver.
    This routine checks every marker that was fixed to zero outside that
    candidate, returns the union of strict violators, and annotates every path
    row before validation is allowed to compare their predictions.
    """
    scores = np.asarray(score_path, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.int64).reshape(-1)
    beta_path = np.asarray(beta_candidate_path, dtype=np.float64)
    rows = [dict(row) for row in path_rows]
    if scores.ndim == 1:
        scores = scores[:, None]
    if scores.ndim != 2 or beta_path.ndim != 2:
        raise ValueError("Path KKT scores and coefficients must be matrices.")
    if scores.shape[1] != len(rows) or beta_path.shape != (
        len(rows),
        cand.size,
    ):
        raise ValueError("Path KKT arrays do not align with path diagnostics.")
    if (
        cand.size > 0
        and (
            np.any(cand < 0)
            or np.any(cand >= scores.shape[0])
            or np.unique(cand).size != cand.size
        )
    ):
        raise ValueError("Candidate indices are invalid for path KKT scores.")
    if not np.all(np.isfinite(scores)) or not np.all(np.isfinite(beta_path)):
        raise ValueError("Path KKT scores and coefficients must be finite.")

    outside = np.ones(scores.shape[0], dtype=bool)
    outside[cand] = False
    union_mask = np.zeros(scores.shape[0], dtype=bool)
    priority = np.zeros(scores.shape[0], dtype=np.float64)
    row_summaries: list[dict[str, object]] = []
    max_excess = 0.0
    violating_path_points = 0
    for path_index, row in enumerate(rows):
        if not (
            bool(row.get("converged", False))
            and bool(row.get("kkt_passed", False))
        ):
            raise RuntimeError(
                "Complete-path global KKT certification requires every "
                "candidate path point to be score-KKT converged."
            )
        lam = float(row["lam"])
        violators, max_outside, threshold = _outside_kkt_violators(
            score_abs=np.abs(scores[:, path_index]),
            candidate=cand,
            lam=lam,
            abs_tol=float(abs_tol),
            rel_tol=float(rel_tol),
        )
        if violators.size > 0:
            violating_path_points += 1
            union_mask[violators] = True
        if np.any(outside):
            normalized = (
                np.abs(scores[:, path_index])
                / max(float(threshold), np.finfo(np.float64).tiny)
            )
            priority[outside] = np.maximum(
                priority[outside], normalized[outside]
            )
        outside_excess = max(float(max_outside) - float(threshold), 0.0)
        max_excess = max(max_excess, outside_excess)
        passed = bool(violators.size == 0)
        row.update(
            {
                "global_kkt_scope": "all_markers",
                "global_kkt_passed": passed,
                "global_kkt_max_outside_score": float(max_outside),
                "global_kkt_threshold": float(threshold),
                "global_kkt_n_outside_violators": int(violators.size),
            }
        )
        row_summaries.append(
            {
                "path_index": int(path_index),
                "lam": lam,
                "lam_ratio": float(row["lam_ratio"]),
                "support_size": int(row["k"]),
                "passed": passed,
                "max_outside_score": float(max_outside),
                "threshold": float(threshold),
                "n_outside_violators": int(violators.size),
            }
        )

    union_violators = np.flatnonzero(union_mask).astype(np.int64)
    return {
        "passed": bool(union_violators.size == 0),
        "n_path_points": int(len(rows)),
        "n_violating_path_points": int(violating_path_points),
        "n_union_violators": int(union_violators.size),
        "max_outside_excess_over_threshold": float(max_excess),
        "outside_violators": union_violators,
        "priority_score": priority,
        "path_rows": rows,
        "row_summaries": row_summaries,
        "score_backend": "batched_hinv_path_then_single_full_marker_xtv",
    }


def _top_scored_markers_outside(
    *,
    score_abs: np.ndarray,
    excluded: np.ndarray,
    count: int,
) -> np.ndarray:
    """Return the strongest marker scores outside a working set."""
    scores = np.asarray(score_abs, dtype=np.float64).reshape(-1)
    blocked = np.asarray(excluded, dtype=np.int64).reshape(-1)
    requested = int(count)
    if requested < 0:
        raise ValueError("BASIL marker batch size must be nonnegative.")
    if requested == 0:
        return np.empty((0,), dtype=np.int64)
    if not np.all(np.isfinite(scores)):
        raise ValueError("BASIL screening scores must be finite.")
    eligible = np.ones(scores.size, dtype=bool)
    if blocked.size > 0:
        if (
            np.any(blocked < 0)
            or np.any(blocked >= scores.size)
            or np.unique(blocked).size != blocked.size
        ):
            raise ValueError("BASIL excluded marker indices are invalid.")
        eligible[blocked] = False
    pool = np.flatnonzero(eligible)
    take = min(requested, int(pool.size))
    if take == 0:
        return np.empty((0,), dtype=np.int64)
    if take == pool.size:
        selected = pool
    else:
        local = np.argpartition(scores[pool], -take)[-take:]
        selected = pool[local]
    selected = np.asarray(selected, dtype=np.int64)
    selected.sort()
    return selected


def _fit_complete_weighted_lasso_path_basil(
    *,
    args,
    grm_index: MultiGRMIndex,
    hv,
    precond,
    y: np.ndarray,
    covar: np.ndarray | None,
    hinv_y: np.ndarray,
    hinv_covar: np.ndarray | None,
    initial_score_abs: np.ndarray,
    path_cfg: LassoPathConfig,
    previous_candidate: np.ndarray,
    previous_beta_path: np.ndarray | None,
    previous_hinv_z: dict[int, np.ndarray],
    outer: int,
    sparse_path_performance: dict[str, object],
    validation_evaluator=None,
    validation_stopping_lag: int = 5,
) -> dict[str, object]:
    """Compute an exact global path with weighted BASIL rollout.

    This is the weighted, matrix-free analogue of snpnet's Batch Screening
    Iterative Lasso.  It solves only a short unresolved lambda block on an
    in-memory strong set, checks every block residual in one full-marker
    ``X.T @ V`` pass, accepts the longest KKT-certified prefix, and then
    refreshes the strong set from the last exact residual.  Screen-only
    markers are discarded after progress; true ever-active markers persist.
    """
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    score_abs = np.asarray(initial_score_abs, dtype=np.float64).reshape(-1)
    if score_abs.shape != (int(grm_index.m_total),):
        raise ValueError("BASIL initial score does not cover every marker.")
    if not np.all(np.isfinite(score_abs)) or np.any(score_abs < 0.0):
        raise ValueError("BASIL initial marker scores must be finite/nonnegative.")
    if path_cfg.fixed_lam_ratio is not None:
        raise ValueError("BASIL path rollout cannot use a fixed lambda ratio.")
    if validation_evaluator is None:
        raise ValueError("BASIL validation path requires an incremental evaluator.")
    if int(validation_stopping_lag) < 1:
        raise ValueError("BASIL validation stopping lag must be positive.")

    lam_max = float(np.max(score_abs)) if score_abs.size > 0 else 0.0
    lam_sequence = make_lambda_sequence(
        lam_max,
        float(path_cfg.lam_min_ratio),
        int(path_cfg.n_lambda),
    )
    n_path = int(lam_sequence.size)
    if n_path < 1:
        raise RuntimeError("BASIL received an empty lambda path.")

    previous = np.asarray(previous_candidate, dtype=np.int64).reshape(-1)
    previous_path = None
    if previous_beta_path is not None:
        candidate_previous_path = np.asarray(
            previous_beta_path, dtype=np.float64
        )
        if (
            candidate_previous_path.ndim == 2
            and candidate_previous_path.shape[1] == previous.size
            and 1 <= candidate_previous_path.shape[0] <= n_path
        ):
            previous_path = candidate_previous_path
        else:
            previous_path = None

    base_marker_batch = min(
        int(grm_index.m_total),
        max(
            int(args.candidate_k),
            min(int(args.screen_topk), int(args.basil_marker_batch_size)),
        ),
    )
    marker_batch = base_marker_batch
    marker_batch_increment = int(args.kkt_add_topk)
    lambda_block_size = min(int(args.basil_lambda_block_size), n_path)
    current_lambda_block_size = lambda_block_size
    max_iterations = int(args.basil_max_iterations)

    next_path_index = 0
    priority_score = score_abs.copy()
    ever_active = np.empty((0,), dtype=np.int64)
    stalled_markers = np.empty((0,), dtype=np.int64)
    outer_hinv_z: dict[int, np.ndarray] = {}
    solution_markers: list[np.ndarray | None] = [None] * n_path
    solution_coefficients: list[np.ndarray | None] = [None] * n_path
    solution_covariates: list[np.ndarray | None] = [None] * n_path
    certified_rows: list[dict | None] = [None] * n_path
    validation_metrics: list[dict | None] = [None] * n_path
    validation_trace: list[dict[str, object]] = []
    validation_early_stopping = (
        _validation_path_early_stopping_decision(
            [], stopping_lag=int(validation_stopping_lag)
        )
    )
    validation_early_stopped = False
    validation_pcg_max_residual = 0.0
    validation_pcg_max_iterations = 0
    total_validation_seconds = 0.0
    basil_trace: list[dict[str, object]] = []
    total_path_seconds = 0.0
    total_kkt_seconds = 0.0
    total_validation_warm_rows = 0
    any_external_path_warm_start = False
    aggregate_pcg = {
        "batch_size": int(args.candidate_pcg_rhs_batch_size),
        "n_batches": 0,
        "n_columns": 0,
        "max_reported_relative_residual": 0.0,
        "max_true_relative_residual": 0.0,
        "max_iterations": 0,
        "total_batch_iterations": 0,
        "warm_start_columns": 0,
        "reused_within_outer_columns": 0,
    }

    for basil_iteration in range(1, max_iterations + 1):
        if next_path_index >= n_path:
            break
        requested_lambda_block_size = int(current_lambda_block_size)
        block_stop = min(
            next_path_index + requested_lambda_block_size, n_path
        )
        unresolved_indices = np.arange(
            next_path_index, block_stop, dtype=np.int64
        )
        if next_path_index > 0:
            fit_indices = np.concatenate(
                [
                    np.asarray([next_path_index - 1], dtype=np.int64),
                    unresolved_indices,
                ]
            )
            boundary_offset = 1
        else:
            fit_indices = unresolved_indices
            boundary_offset = 0

        prior_block_active = np.empty((0,), dtype=np.int64)
        if previous_path is not None and previous.size > 0:
            prior_valid_indices = fit_indices[
                fit_indices < previous_path.shape[0]
            ]
            if prior_valid_indices.size > 0:
                prior_mask = np.any(
                    previous_path[prior_valid_indices, :] != 0.0, axis=0
                )
                prior_block_active = previous[prior_mask]

        ever_active_size_before = int(ever_active.size)
        requested_marker_batch = int(marker_batch)
        retained = np.unique(
            np.concatenate(
                [ever_active, stalled_markers, prior_block_active]
            )
        ).astype(np.int64)
        # glmnet's sequential strong rule predicts coordinates that can enter
        # at the next lambda from the last exact residual.  It is deliberately
        # only a screen: the subsequent all-marker KKT pass is the certificate.
        strong_rule_threshold = None
        strong_rule_markers = np.empty((0,), dtype=np.int64)
        if next_path_index > 0:
            previous_lambda = float(lam_sequence[next_path_index - 1])
            next_lambda = float(lam_sequence[next_path_index])
            strong_rule_threshold = max(
                2.0 * next_lambda - previous_lambda, 0.0
            )
            strong_rule_markers = np.flatnonzero(
                priority_score >= strong_rule_threshold
            ).astype(np.int64)
            if retained.size > 0:
                strong_rule_markers = np.setdiff1d(
                    strong_rule_markers, retained, assume_unique=True
                )
        # In p >> n genotype data the raw sequential rule can admit tens of
        # thousands of correlated markers at small lambda.  BASIL's bounded
        # top-M rollout is the appropriate out-of-core guard: strong-rule
        # markers receive their natural score priority, but never enlarge the
        # computational batch.  A failed global KKT scan triggers the ordinary
        # exact enlargement step below.
        new_markers = _top_scored_markers_outside(
            score_abs=priority_score,
            excluded=retained,
            count=marker_batch,
        )
        selected_strong_rule_markers = int(
            np.intersect1d(
                new_markers, strong_rule_markers, assume_unique=True
            ).size
        )
        candidate = np.unique(
            np.concatenate([retained, new_markers])
        ).astype(np.int64)
        candidate.sort()
        if candidate.size == 0:
            raise RuntimeError("BASIL could not construct a nonempty strong set.")
        max_candidate = int(args.kkt_max_candidate)
        if max_candidate > 0 and candidate.size > max_candidate:
            raise RuntimeError(
                "BASIL strong set exceeds --kkt-max-candidate "
                f"({candidate.size} > {max_candidate})."
            )

        Z_candidate = grm_index.extract_standardized_columns(candidate).astype(
            np.float32, copy=False
        )
        missing_positions = np.asarray(
            [
                column
                for column, marker in enumerate(candidate.tolist())
                if int(marker) not in outer_hinv_z
            ],
            dtype=np.int64,
        )
        reused_columns = int(candidate.size - missing_positions.size)
        pcg_diagnostic = {
            "batch_size": int(args.candidate_pcg_rhs_batch_size),
            "n_batches": 0,
            "n_columns": 0,
            "max_reported_relative_residual": 0.0,
            "max_true_relative_residual": 0.0,
            "max_iterations": 0,
            "total_batch_iterations": 0,
            "warm_start_columns": 0,
            "reused_within_outer_columns": reused_columns,
        }
        if missing_positions.size > 0:
            missing_markers = candidate[missing_positions]
            missing_rhs = np.ascontiguousarray(
                Z_candidate[:, missing_positions], dtype=np.float32
            )
            missing_warm = np.zeros_like(missing_rhs)
            warm_hits = 0
            for column, marker in enumerate(missing_markers.tolist()):
                previous_solution = previous_hinv_z.get(int(marker))
                if previous_solution is not None:
                    missing_warm[:, column] = previous_solution
                    warm_hits += 1
            candidate_pcg_started = time.perf_counter()
            solved_missing, solved_diagnostic = _solve_hinv_columns_batched(
                hv=hv,
                precond=precond,
                rhs=missing_rhs,
                warm_start=missing_warm if warm_hits > 0 else None,
                tol=float(args.pcg_tol),
                maxiter=int(args.max_pcg_iters),
                batch_size=int(args.candidate_pcg_rhs_batch_size),
                stage=(
                    f"outer {outer} BASIL iteration {basil_iteration} candidate"
                ),
            )
            sparse_path_performance["candidate_hinv_pcg_seconds"] = (
                sparse_path_performance.get("candidate_hinv_pcg_seconds", 0.0)
                + time.perf_counter() - candidate_pcg_started
            )
            pcg_diagnostic.update(solved_diagnostic)
            pcg_diagnostic["warm_start_columns"] = int(warm_hits)
            pcg_diagnostic["reused_within_outer_columns"] = reused_columns
            for column, marker in enumerate(missing_markers.tolist()):
                outer_hinv_z[int(marker)] = solved_missing[:, column]

        Hinv_Z_candidate = np.empty_like(Z_candidate, dtype=np.float32)
        for column, marker in enumerate(candidate.tolist()):
            Hinv_Z_candidate[:, column] = outer_hinv_z[int(marker)]

        for key in ("n_batches", "n_columns", "total_batch_iterations"):
            aggregate_pcg[key] += int(pcg_diagnostic[key])
        for key in ("warm_start_columns", "reused_within_outer_columns"):
            aggregate_pcg[key] += int(pcg_diagnostic[key])
        for key in (
            "max_reported_relative_residual",
            "max_true_relative_residual",
        ):
            aggregate_pcg[key] = max(
                float(aggregate_pcg[key]), float(pcg_diagnostic[key])
            )
        aggregate_pcg["max_iterations"] = max(
            int(aggregate_pcg["max_iterations"]),
            int(pcg_diagnostic["max_iterations"]),
        )
        sparse_path_performance["candidate_hinv_pcg_batches"] += int(
            pcg_diagnostic["n_batches"]
        )
        sparse_path_performance[
            "candidate_hinv_pcg_columns_solved"
        ] += int(pcg_diagnostic["n_columns"])
        sparse_path_performance[
            "candidate_hinv_columns_reused"
        ] += reused_columns
        sparse_path_performance[
            "candidate_hinv_pcg_total_batch_iterations"
        ] += int(pcg_diagnostic["total_batch_iterations"])

        beta_path0 = None
        warm_columns = 0
        if previous_path is not None:
            mapped_previous, warm_columns = _remap_lasso_beta_path(
                previous_candidate=previous,
                previous_beta_path=previous_path,
                candidate=candidate,
            )
            if mapped_previous is not None:
                beta_path0 = np.zeros(
                    (fit_indices.size, candidate.size), dtype=np.float64
                )
                valid_rows = fit_indices < mapped_previous.shape[0]
                beta_path0[valid_rows, :] = mapped_previous[
                    fit_indices[valid_rows], :
                ]
        if boundary_offset:
            boundary_markers = solution_markers[next_path_index - 1]
            boundary_beta = solution_coefficients[next_path_index - 1]
            if boundary_markers is None or boundary_beta is None:
                raise RuntimeError("BASIL boundary solution is unavailable.")
            if beta_path0 is None:
                beta_path0 = np.zeros(
                    (fit_indices.size, candidate.size), dtype=np.float64
                )
            common, boundary_pos, candidate_pos = np.intersect1d(
                np.asarray(boundary_markers, dtype=np.int64),
                candidate,
                assume_unique=True,
                return_indices=True,
            )
            if common.size != np.asarray(boundary_markers).size:
                raise RuntimeError(
                    "BASIL strong set omitted an active boundary marker."
                )
            beta_path0[0, :] = 0.0
            beta_path0[0, candidate_pos] = np.asarray(
                boundary_beta, dtype=np.float64
            )[boundary_pos]

        path_started = time.perf_counter()
        block_fit = fit_weighted_lasso_with_covariates(
            y=y_arr,
            covar=covar,
            geno=Z_candidate,
            Hinv_y=hinv_y,
            Hinv_covar=hinv_covar,
            Hinv_geno=Hinv_Z_candidate,
            cfg=path_cfg,
            ridge=float(args.lasso_ridge),
            beta_snp_path0=beta_path0,
            lambda_sequence=lam_sequence[fit_indices],
            lambda_max_reference=lam_max,
        )
        path_seconds = float(time.perf_counter() - path_started)
        total_path_seconds += path_seconds
        sparse_path_performance["lasso_path_solves"] += 1
        sparse_path_performance["lasso_path_solve_seconds"] += path_seconds
        sparse_path_performance["lasso_cd_iterations"] += int(
            sum(int(row["cd_iter"]) for row in block_fit["path"])
        )
        warm_rows_used = int(
            block_fit["external_beta_path_warm_start_rows_used"]
        )
        total_validation_warm_rows += warm_rows_used
        any_external_path_warm_start = bool(
            any_external_path_warm_start
            or block_fit["external_beta_path_warm_start_used"]
        )
        sparse_path_performance[
            "lasso_path_warm_start_rows_used"
        ] += warm_rows_used

        kkt_started = time.perf_counter()
        hinv_residual_path = _build_hinv_lasso_residual_path(
            hinv_y=hinv_y,
            hinv_covar=hinv_covar,
            hinv_geno=Hinv_Z_candidate,
            beta_cov_path=block_fit["beta_cov_path"],
            beta_snp_path=block_fit["beta_snp_path"],
        )
        path_score_signed = np.asarray(
            grm_index.xtv_all(
                jnp.asarray(hinv_residual_path, dtype=jnp.float32),
                normalize=False,
            ),
            dtype=np.float64,
        )
        block_kkt = _certify_complete_lasso_path_kkt_from_scores(
            score_path=path_score_signed,
            candidate=candidate,
            beta_candidate_path=block_fit["beta_snp_path"],
            path_rows=list(block_fit["path"]),
            abs_tol=float(args.kkt_tol),
            rel_tol=float(args.kkt_rel_tol),
        )
        kkt_seconds = float(time.perf_counter() - kkt_started)
        total_kkt_seconds += kkt_seconds
        sparse_path_performance["lasso_path_global_kkt_passes"] += 1
        sparse_path_performance["lasso_path_global_kkt_seconds"] += (
            kkt_seconds
        )
        sparse_path_performance[
            "lasso_path_global_kkt_points_checked"
        ] += int(block_kkt["n_path_points"])
        block_rows = list(block_kkt["path_rows"])
        if boundary_offset and not bool(block_rows[0]["global_kkt_passed"]):
            raise RuntimeError(
                "Previously certified BASIL boundary failed after basis remap."
            )

        n_new_valid = 0
        for local_index in range(boundary_offset, len(block_rows)):
            if not bool(block_rows[local_index]["global_kkt_passed"]):
                break
            n_new_valid += 1

        accepted_local = range(
            boundary_offset, boundary_offset + n_new_valid
        )
        newly_active_parts: list[np.ndarray] = []
        for local_index in accepted_local:
            global_path_index = int(fit_indices[local_index])
            beta_local = np.asarray(
                block_fit["beta_snp_path"][local_index], dtype=np.float64
            )
            active_local = np.flatnonzero(beta_local != 0.0).astype(
                np.int64
            )
            active_markers = candidate[active_local]
            solution_markers[global_path_index] = active_markers.copy()
            solution_coefficients[global_path_index] = beta_local[
                active_local
            ].copy()
            solution_covariates[global_path_index] = np.asarray(
                block_fit["beta_cov_path"][local_index], dtype=np.float64
            ).copy()
            certified_row = dict(block_rows[local_index])
            certified_row.update(
                {
                    "basil_iteration": int(basil_iteration),
                    "basil_strong_set_size": int(candidate.size),
                }
            )
            certified_rows[global_path_index] = certified_row
            if active_markers.size > 0:
                newly_active_parts.append(active_markers)

        validation_batch_record = None
        if n_new_valid > 0:
            if newly_active_parts:
                ever_active = np.unique(
                    np.concatenate([ever_active, *newly_active_parts])
                ).astype(np.int64)
            last_local = boundary_offset + n_new_valid - 1
            priority_score = np.abs(path_score_signed[:, last_local])
            next_path_index += n_new_valid
            stalled_markers = np.empty((0,), dtype=np.int64)
            marker_batch = base_marker_batch
            if n_new_valid < unresolved_indices.size:
                # Do not repeatedly solve the rest of a block after its exact
                # prefix has shown that the current screen supports fewer
                # points.  Dense low-lambda tails naturally settle at one
                # lambda per KKT scan.
                current_lambda_block_size = max(1, int(n_new_valid))
            decision = "advance_exact_prefix"

            validation_started = time.perf_counter()
            validation_block = validation_evaluator(
                candidate=candidate,
                train_candidate=Z_candidate,
                beta_cov_path=np.asarray(
                    block_fit["beta_cov_path"][
                        boundary_offset : boundary_offset + n_new_valid
                    ],
                    dtype=np.float64,
                ),
                beta_snp_path=np.asarray(
                    block_fit["beta_snp_path"][
                        boundary_offset : boundary_offset + n_new_valid
                    ],
                    dtype=np.float64,
                ),
                hinv_residual_path=hinv_residual_path[
                    :, boundary_offset : boundary_offset + n_new_valid
                ],
                training_score_path=path_score_signed[
                    :, boundary_offset : boundary_offset + n_new_valid
                ],
            )
            validation_seconds = float(
                time.perf_counter() - validation_started
            )
            total_validation_seconds += validation_seconds
            sparse_path_performance[
                "lasso_validation_selection_seconds"
            ] += validation_seconds
            block_metrics = list(validation_block["metrics"])
            if len(block_metrics) != n_new_valid:
                raise RuntimeError(
                    "Incremental validation metrics do not align with the "
                    "new exact BASIL prefix."
                )
            first_new_index = next_path_index - n_new_valid
            for metric_offset, metric in enumerate(block_metrics):
                global_metric_index = first_new_index + metric_offset
                metric_global = dict(metric)
                metric_global["path_index"] = int(global_metric_index)
                validation_metrics[global_metric_index] = metric_global
            validation_pcg_max_residual = max(
                validation_pcg_max_residual,
                float(validation_block["pcg_rel_res"]),
            )
            validation_pcg_max_iterations = max(
                validation_pcg_max_iterations,
                int(validation_block["pcg_iters"]),
            )
            validation_prefix = [
                dict(value)
                for value in validation_metrics[:next_path_index]
                if value is not None
            ]
            if len(validation_prefix) != next_path_index:
                raise RuntimeError(
                    "Incremental validation path has a missing exact prefix row."
                )
            validation_early_stopping = (
                _validation_path_early_stopping_decision(
                    validation_prefix,
                    stopping_lag=int(validation_stopping_lag),
                )
            )
            validation_early_stopped = bool(
                validation_early_stopping["stopped"]
            )
            validation_batch_record = {
                "path_indices": list(
                    range(first_new_index, next_path_index)
                ),
                "metrics": block_metrics,
                "pcg_rel_res": float(validation_block["pcg_rel_res"]),
                "pcg_iters": int(validation_block["pcg_iters"]),
                "seconds": validation_seconds,
                "early_stopping": dict(validation_early_stopping),
            }
            validation_trace.append(validation_batch_record)
            if validation_early_stopped:
                decision = "early_stop_after_exact_validation_peak"
        else:
            stalled_markers = np.unique(
                np.concatenate([stalled_markers, new_markers])
            ).astype(np.int64)
            marker_batch += marker_batch_increment
            current_lambda_block_size = 1
            decision = "enlarge_strong_set"

        basil_trace.append(
            {
                "iteration": int(basil_iteration),
                "fit_path_indices": fit_indices.tolist(),
                "requested_lambda_block_size": requested_lambda_block_size,
                "first_unresolved_index": int(
                    fit_indices[boundary_offset]
                ),
                "strong_set_size": int(candidate.size),
                "ever_active_size_before": ever_active_size_before,
                "retained_marker_size_before": int(retained.size),
                "prior_outer_block_active_size": int(
                    prior_block_active.size
                ),
                "new_marker_batch_size": int(new_markers.size),
                "sequential_strong_rule_threshold": (
                    float(strong_rule_threshold)
                    if strong_rule_threshold is not None
                    else None
                ),
                "sequential_strong_rule_candidates": int(
                    strong_rule_markers.size
                ),
                "sequential_strong_rule_selected_in_batch": (
                    selected_strong_rule_markers
                ),
                "score_batch_markers": int(new_markers.size),
                "configured_marker_batch_size": requested_marker_batch,
                "n_new_valid": int(n_new_valid),
                "next_path_index": int(next_path_index),
                "n_fit_points": int(len(block_rows)),
                "n_violating_fit_points": int(
                    block_kkt["n_violating_path_points"]
                ),
                "n_union_violators": int(
                    block_kkt["n_union_violators"]
                ),
                "path_solve_seconds": path_seconds,
                "global_kkt_seconds": kkt_seconds,
                "path_warm_start_common_columns": int(warm_columns),
                "path_warm_start_rows_used": warm_rows_used,
                "candidate_pcg": dict(pcg_diagnostic),
                "validation_batch": validation_batch_record,
                "decision": decision,
            }
        )
        logger.info(
            "[outer %s BASIL %s] unresolved=%s:%s strong=%s ever=%s "
            "new=%s exact_advance=%s next=%s/%s path_sec=%.1f "
            "kkt_sec=%.1f validation_best=%.8f early_stop=%s decision=%s",
            outer,
            basil_iteration,
            int(fit_indices[boundary_offset]),
            int(fit_indices[-1]),
            int(candidate.size),
            int(ever_active.size),
            int(new_markers.size),
            int(n_new_valid),
            int(next_path_index),
            n_path,
            path_seconds,
            kkt_seconds,
            float(
                validation_early_stopping[
                    "best_predictive_r2"
                ]
            )
            if validation_early_stopping[
                "best_predictive_r2"
            ] is not None
            else float("nan"),
            validation_early_stopped,
            decision,
        )
        if validation_early_stopped:
            break

    if next_path_index != n_path and not validation_early_stopped:
        raise RuntimeError(
            "BASIL failed to certify the complete global lambda path within "
            f"{max_iterations} iterations (certified {next_path_index}/{n_path})."
        )
    evaluated_n_path = int(next_path_index)
    if any(value is None for value in certified_rows[:evaluated_n_path]):
        raise RuntimeError("BASIL complete path has missing diagnostics.")

    final_candidate = np.asarray(ever_active, dtype=np.int64)
    if final_candidate.size == 0:
        final_candidate = _top_scored_markers_outside(
            score_abs=score_abs,
            excluded=np.empty((0,), dtype=np.int64),
            count=1,
        )
    final_candidate.sort()
    beta_snp_path = np.zeros(
        (evaluated_n_path, final_candidate.size), dtype=np.float64
    )
    n_covar = 0 if covar is None else int(np.asarray(covar).shape[1])
    beta_cov_path = np.zeros((evaluated_n_path, n_covar), dtype=np.float64)
    for path_index in range(evaluated_n_path):
        markers = np.asarray(solution_markers[path_index], dtype=np.int64)
        coefficients = np.asarray(
            solution_coefficients[path_index], dtype=np.float64
        )
        if markers.size > 0:
            positions = np.searchsorted(final_candidate, markers)
            if not np.array_equal(final_candidate[positions], markers):
                raise RuntimeError(
                    "BASIL ever-active set omitted a certified coefficient."
                )
            beta_snp_path[path_index, positions] = coefficients
        beta_cov_path[path_index] = np.asarray(
            solution_covariates[path_index], dtype=np.float64
        )

    Z_final = grm_index.extract_standardized_columns(final_candidate).astype(
        np.float32, copy=False
    )
    Hinv_Z_final = np.empty_like(Z_final, dtype=np.float32)
    for column, marker in enumerate(final_candidate.tolist()):
        if int(marker) not in outer_hinv_z:
            raise RuntimeError("BASIL final Hinv[Z] cache is incomplete.")
        Hinv_Z_final[:, column] = outer_hinv_z[int(marker)]

    lasso_path = {
        "path": [
            dict(value) for value in certified_rows[:evaluated_n_path]
        ],
        "beta_snp_path": beta_snp_path,
        "beta_cov_path": beta_cov_path,
        "lam_max": lam_max,
        "selected_index": None,
        "selection_method": None,
        "selected_lam_ratio": None,
        "path_role": (
            "early_stopped_kkt_certified_validation_prefix_weighted_basil"
            if validation_early_stopped
            else "complete_validation_grid_weighted_basil"
        ),
        "requested_n_lambda": n_path,
        "evaluated_n_lambda": evaluated_n_path,
        "validation_metrics": [
            dict(value) for value in validation_metrics[:evaluated_n_path]
        ],
        "validation_early_stopping": dict(validation_early_stopping),
        "external_beta_path_warm_start_used": bool(
            any_external_path_warm_start
        ),
        "external_beta_path_warm_start_rows_used": int(
            total_validation_warm_rows
        ),
        "basil": {
            "algorithm": "weighted_batch_screening_iterative_lasso",
            "n_iterations": int(len(basil_trace)),
            "base_marker_batch_size": int(base_marker_batch),
            "lambda_block_size": int(lambda_block_size),
            "ever_active_size": int(final_candidate.size),
            "path_solve_seconds": float(total_path_seconds),
            "global_kkt_seconds": float(total_kkt_seconds),
            "validation_seconds": float(total_validation_seconds),
            "validation_trace": validation_trace,
            "trace": basil_trace,
        },
    }
    final_hinv_z = {
        int(marker): outer_hinv_z[int(marker)]
        for marker in final_candidate.tolist()
    }
    return {
        "lasso_path": lasso_path,
        "candidate": final_candidate,
        "geno": Z_final,
        "hinv_geno": Hinv_Z_final,
        "hinv_z_dict": final_hinv_z,
        "kkt_trace": basil_trace,
        "candidate_pcg": aggregate_pcg,
        "global_kkt": {
            "passed": True,
            "n_path_points": evaluated_n_path,
            "n_violating_path_points": 0,
            "n_union_violators": 0,
            "max_outside_excess_over_threshold": 0.0,
            "score_backend": (
                "weighted_basil_batched_hinv_blocks_then_full_marker_xtv"
            ),
        },
        "validation": {
            "metrics": [
                dict(value)
                for value in validation_metrics[:evaluated_n_path]
            ],
            "pcg_max_rel_res": float(validation_pcg_max_residual),
            "pcg_max_iters": int(validation_pcg_max_iterations),
            "seconds": float(total_validation_seconds),
            "early_stopping": dict(validation_early_stopping),
            "trace": validation_trace,
        },
    }


def _build_lasso_candidate(
    *,
    previous_support: np.ndarray,
    screened_indices: np.ndarray,
    candidate_target: int,
    previous_candidate: np.ndarray | None = None,
) -> np.ndarray:
    """Reuse the certified candidate and add the current screened seed.

    On the first outer iteration, ``candidate_target`` is the minimum seed
    size after retaining any supplied support.  Once a complete candidate has
    passed the outside-marker KKT check, later outer iterations keep that
    candidate and union it with the first ``candidate_target`` entries from
    the current score screen.  This avoids rebuilding the same candidate by
    repeated full-path KKT refinements after every covariance update.
    """
    target = int(candidate_target)
    if target <= 0:
        raise ValueError("candidate_target must be > 0.")

    keep_list: list[int] = []
    seen: set[int] = set()
    cached = (
        np.empty((0,), dtype=np.int64)
        if previous_candidate is None
        else np.asarray(previous_candidate, dtype=np.int64).reshape(-1)
    )
    for value in cached:
        index = int(value)
        if index not in seen:
            keep_list.append(index)
            seen.add(index)
    for value in np.asarray(previous_support, dtype=np.int64).reshape(-1):
        index = int(value)
        if index not in seen:
            keep_list.append(index)
            seen.add(index)
    screened = np.asarray(screened_indices, dtype=np.int64).reshape(-1)
    screen_limit = min(target, int(screened.size))
    for value in screened[:screen_limit]:
        # With no reusable candidate, the target is the desired total seed
        # size.  With a reusable candidate, it is the size of the fresh score
        # seed to union with that candidate.
        if cached.size == 0 and len(keep_list) >= target:
            break
        index = int(value)
        if index not in seen:
            keep_list.append(index)
            seen.add(index)
    return np.asarray(sorted(keep_list), dtype=np.int64)


def _remap_lasso_beta_path(
    *,
    previous_candidate: np.ndarray,
    previous_beta_path: np.ndarray | None,
    candidate: np.ndarray,
) -> tuple[np.ndarray | None, int]:
    """Map a prior coefficient path into the current global-SNP basis."""
    if previous_beta_path is None:
        return None, 0
    previous = np.asarray(previous_candidate, dtype=np.int64).reshape(-1)
    current = np.asarray(candidate, dtype=np.int64).reshape(-1)
    beta_path = np.asarray(previous_beta_path, dtype=np.float64)
    if beta_path.ndim != 2 or beta_path.shape[1] != previous.size:
        raise ValueError(
            "Previous Lasso coefficient path does not align with its candidate."
        )
    if beta_path.shape[0] < 1 or not np.all(np.isfinite(beta_path)):
        raise ValueError(
            "Previous Lasso coefficient path must be finite and non-empty."
        )
    if previous.size == 0 or current.size == 0:
        return None, 0
    common, previous_pos, current_pos = np.intersect1d(
        previous,
        current,
        assume_unique=True,
        return_indices=True,
    )
    if common.size == 0:
        return None, 0
    mapped = np.zeros(
        (int(beta_path.shape[0]), int(current.size)), dtype=np.float64
    )
    mapped[:, current_pos] = beta_path[:, previous_pos]
    return mapped, int(common.size)


def _allow_mapped_lasso_path_warm_start(
    *,
    previous_size: int,
    current_size: int,
    common_size: int,
) -> bool:
    """Reuse any nonempty overlap of two marker bases as a warm start.

    The mapped coefficients are only initial values.  ``solve_lasso_path``
    compares them with the ordinary descending-lambda start row by row and
    still requires the same complete score-KKT certificate.  It is therefore
    safe to drop coordinates that are zero along the entire old path and map
    the remaining rows into a pruned/expanded working set.
    """
    previous = int(previous_size)
    current = int(current_size)
    common = int(common_size)
    return bool(
        previous >= 1
        and current >= 1
        and common >= 1
        and common <= min(previous, current)
    )


def _complete_path_active_union(
    *,
    candidate: np.ndarray,
    beta_path: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return candidate markers active at one or more lambda path points."""
    candidate_arr = np.asarray(candidate, dtype=np.int64).reshape(-1)
    beta_arr = np.asarray(beta_path, dtype=np.float64)
    if beta_arr.ndim != 2 or beta_arr.shape[1] != candidate_arr.size:
        raise ValueError("Complete Lasso path does not align with candidate.")
    if not np.all(np.isfinite(beta_arr)):
        raise ValueError("Complete Lasso path coefficients must be finite.")
    keep_local = np.flatnonzero(np.any(beta_arr != 0.0, axis=0)).astype(
        np.int64
    )
    # Preserve a nonempty numerical basis for the downstream Lasso warm-state
    # schema in the degenerate all-zero path.  No pruning is needed there.
    if keep_local.size == 0:
        keep_local = np.arange(candidate_arr.size, dtype=np.int64)
    return candidate_arr[keep_local], keep_local


def _buffered_kkt_expansion_indices(
    *,
    score_abs: np.ndarray,
    candidate: np.ndarray,
    violators: np.ndarray,
    max_add: int,
) -> dict[str, object]:
    """Choose strict KKT violators plus a small near-threshold look-ahead.

    The configured ``max_add`` remains the hard per-round budget.  When all
    strict violators fit, the batch is enlarged adaptively to at least the
    square root of that budget and at most twice the strict-violator count.
    This catches the common cascade of one-digit follow-up violations without
    inflating every candidate by a full ``max_add`` block.
    """
    scores = np.asarray(score_abs, dtype=np.float64).reshape(-1)
    current = np.asarray(candidate, dtype=np.int64).reshape(-1)
    strict = np.unique(
        np.asarray(violators, dtype=np.int64).reshape(-1)
    )
    budget = int(max_add)
    if budget < 1:
        raise ValueError("max_add must be >= 1.")
    if not np.all(np.isfinite(scores)) or np.any(scores < 0.0):
        raise ValueError("score_abs must contain finite nonnegative values.")
    for name, indices in (("candidate", current), ("violators", strict)):
        if np.any(indices < 0) or np.any(indices >= scores.size):
            raise ValueError(f"{name} contains an out-of-range marker index.")
    if strict.size == 0:
        return {
            "add_indices": np.empty((0,), dtype=np.int64),
            "n_strict_added": 0,
            "n_buffered_added": 0,
        }
    if np.intersect1d(current, strict, assume_unique=True).size > 0:
        raise ValueError("KKT violators must lie outside the candidate set.")

    n_strict_added = min(int(strict.size), budget)
    strict_order = np.argsort(scores[strict], kind="stable")[-n_strict_added:]
    selected_strict = strict[strict_order]

    target_size = n_strict_added
    if strict.size <= budget:
        target_size = min(
            budget,
            max(
                int(strict.size) * 2,
                int(np.ceil(np.sqrt(float(budget)))),
            ),
        )
    n_buffer = max(int(target_size) - n_strict_added, 0)
    buffered = np.empty((0,), dtype=np.int64)
    if n_buffer > 0:
        eligible = np.ones(scores.size, dtype=bool)
        eligible[current] = False
        eligible[strict] = False
        pool = np.flatnonzero(eligible)
        n_buffer = min(n_buffer, int(pool.size))
        if n_buffer > 0:
            if n_buffer == pool.size:
                buffered = pool
            else:
                local = np.argpartition(scores[pool], -n_buffer)[-n_buffer:]
                buffered = pool[local]

    add_indices = np.unique(
        np.concatenate([selected_strict, buffered])
    ).astype(np.int64)
    add_indices.sort()
    return {
        "add_indices": add_indices,
        "n_strict_added": int(selected_strict.size),
        "n_buffered_added": int(buffered.size),
    }


def _kkt_expansion_budget(n_violators: int, nominal_budget: int) -> int:
    """Absorb a small strict-violator overflow to avoid a nearly empty round."""
    count = int(n_violators)
    nominal = int(nominal_budget)
    if count < 0 or nominal < 1:
        raise ValueError("KKT violator count/budget is invalid.")
    spillover = nominal // 10
    if nominal < count <= nominal + spillover:
        return count
    return nominal


def _complete_path_kkt_expansion_budget(
    n_violators: int,
    nominal_budget: int,
    current_candidate_size: int,
) -> int:
    """Grow a complete-path working set geometrically when violations abound.

    A low-lambda path point can initially violate KKT at hundreds of thousands
    of markers.  Reusing the whole coefficient path makes a moderately larger
    expansion cheap relative to repeating PCG and Gram construction in fixed
    256-column increments.  The increment grows with the working set but is
    capped at 1024 so scores are refreshed before admitting another block.
    """
    count = int(n_violators)
    current = int(current_candidate_size)
    if current < 1:
        raise ValueError("Complete-path candidate size must be positive.")
    nominal = _kkt_expansion_budget(count, int(nominal_budget))
    return min(count, max(nominal, min(current, 1024)))


def _parse_variance_components_init(
    value: str,
    *,
    n_grm: int,
) -> np.ndarray:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(
            "--variance-components-init must be a JSON array."
        ) from error
    theta = np.asarray(parsed, dtype=np.float64).reshape(-1)
    expected = int(n_grm) + 1
    if theta.shape != (expected,):
        raise ValueError(
            "--variance-components-init must contain one value per GRM "
            f"followed by residual variance; expected {expected}, "
            f"got {int(theta.size)}."
        )
    if (
        not np.all(np.isfinite(theta))
        or np.any(theta[:-1] < 0.0)
        or theta[-1] <= 0.0
    ):
        raise ValueError(
            "Initial genetic variance components must be nonnegative and "
            "the residual component must be positive."
        )
    return theta


def main() -> None:
    args = parse_args()
    if int(args.screen_topk) < int(args.candidate_k):
        raise SystemExit("screen-topk must be >= candidate-k.")
    if int(args.outer_max) < 1:
        raise SystemExit("outer-max must be >= 1.")
    if int(args.minq_iter) < 1:
        raise SystemExit("minq-iter must be >= 1 for both variance blocks.")
    if int(args.reml_max_linesearch_trials) < 1:
        raise SystemExit("reml-max-linesearch-trials must be >= 1.")
    if float(args.h2_abs_tol) <= 0.0:
        raise SystemExit("h2-abs-tol must be > 0.")
    if float(args.effect_rel_tol) <= 0.0:
        raise SystemExit("effect-rel-tol must be > 0.")
    if int(args.kkt_max_rounds) < 1:
        raise SystemExit("kkt-max-rounds must be >= 1.")
    if int(args.kkt_add_topk) < 1:
        raise SystemExit("kkt-add-topk must be >= 1.")
    if int(args.candidate_pcg_rhs_batch_size) < 1:
        raise SystemExit("candidate-pcg-rhs-batch-size must be >= 1.")
    if int(args.basil_marker_batch_size) < 1:
        raise SystemExit("basil-marker-batch-size must be >= 1.")
    if int(args.basil_lambda_block_size) < 1:
        raise SystemExit("basil-lambda-block-size must be >= 1.")
    if int(args.basil_max_iterations) < 1:
        raise SystemExit("basil-max-iterations must be >= 1.")
    if int(args.validation_early_stopping_lag) < 1:
        raise SystemExit("validation-early-stopping-lag must be >= 1.")
    if (
        not np.isfinite(float(args.kkt_tol))
        or not np.isfinite(float(args.kkt_rel_tol))
        or float(args.kkt_tol) < 0.0
        or float(args.kkt_rel_tol) < 0.0
    ):
        raise SystemExit("kkt tolerances must be finite and nonnegative.")
    if (
        not np.isfinite(float(args.pcg_tol))
        or float(args.pcg_tol) <= 0.0
    ):
        raise SystemExit("pcg-tol must be finite and > 0.")
    for state_flag, state_path in (
        ("--lasso-warm-state-in", args.lasso_warm_state_in),
        ("--lasso-warm-state-out", args.lasso_warm_state_out),
    ):
        if state_path and not state_path.lower().endswith(".npz"):
            raise SystemExit(f"{state_flag} must end in .npz.")
    if args.lasso_warm_state_in and not os.path.isfile(
        args.lasso_warm_state_in
    ):
        raise SystemExit(
            "--lasso-warm-state-in does not exist: "
            f"{args.lasso_warm_state_in}"
        )
    if (
        not np.isfinite(float(args.lasso_lam_min_ratio))
        or not 0.0 < float(args.lasso_lam_min_ratio) <= 1.0
    ):
        raise SystemExit("lasso-lam-min-ratio must lie in (0, 1].")
    if int(args.lasso_n_lambda) < 1:
        raise SystemExit("lasso-n-lambda must be >= 1.")
    fixed_ratio_refit = args.lasso_fixed_lam_ratio is not None
    iterative_validation_selection = not fixed_ratio_refit
    if fixed_ratio_refit:
        args.lasso_fixed_lam_ratio = _canonical_fixed_lam_ratio(
            float(args.lasso_fixed_lam_ratio)
        )
        if (
            not np.isfinite(float(args.lasso_fixed_lam_ratio))
            or not 0.0 < float(args.lasso_fixed_lam_ratio) <= 1.0
        ):
            raise SystemExit(
                "--lasso-fixed-lam-ratio must lie in (0, 1]."
            )
        # Lambda / lambda_max can round just below the path endpoint. Allow
        # relative roundoff without changing the frozen, selected ratio.
        if (
            float(args.lasso_fixed_lam_ratio) < float(args.lasso_lam_min_ratio)
            and not math.isclose(
                float(args.lasso_fixed_lam_ratio),
                float(args.lasso_lam_min_ratio),
                rel_tol=1e-12,
                abs_tol=0.0,
            )
        ):
            raise SystemExit(
                "lasso-fixed-lam-ratio must be at least "
                "--lasso-lam-min-ratio so it lies on the fitted path."
            )
    supplied_theta_init = args.variance_components_init.strip()
    if fixed_ratio_refit and not (
        args.lasso_warm_state_in and supplied_theta_init
    ):
        raise SystemExit(
            "The frozen-ratio fit is an internal final-refit stage and "
            "requires both its validation-selected Lasso warm state and "
            "variance-component warm start. Use gpu-reml-sparse."
        )

    sparsity_validation_requested = bool(
        args.sparsity_validation_pheno_txt
        or args.sparsity_validation_out
    )
    if bool(args.sparsity_validation_pheno_txt) != bool(
        args.sparsity_validation_out
    ):
        raise SystemExit(
            "--sparsity-validation-pheno-txt and "
            "--sparsity-validation-out must be supplied together."
    )
    if iterative_validation_selection and not sparsity_validation_requested:
        raise SystemExit(
            "Sparse fitting requires both --sparsity-validation-pheno-txt "
            "and --sparsity-validation-out so validation predictive R2 can select "
            "lambda inside every outer iteration. For a final refit, supply "
            "--lasso-fixed-lam-ratio instead."
        )
    if fixed_ratio_refit and sparsity_validation_requested:
        raise SystemExit(
            "Validation selection inputs and --lasso-fixed-lam-ratio are "
            "mutually exclusive stages."
        )
    if iterative_validation_selection:
        if not os.path.exists(args.sparsity_validation_pheno_txt):
            raise SystemExit(
                "--sparsity-validation-pheno-txt does not exist: "
                f"{args.sparsity_validation_pheno_txt}"
            )
        if not args.sparsity_validation_out.lower().endswith(".json"):
            raise SystemExit("--sparsity-validation-out must end in .json.")

    logger.info("[INFO] sparse pipeline start @ %s", datetime.now().isoformat(timespec='seconds'))
    t0 = time.time()

    bed_prefix = args.bed_prefix.strip()
    if "," in bed_prefix:
        raise SystemExit(
            "Sparse fitting accepts one genotype source; supply one BED prefix."
        )
    bed_list = [bed_prefix] if bed_prefix else []
    pgen_prefix = args.pgen_prefix.strip()
    component_spec_source = args.component_spec.strip()
    try:
        component_variant_indices = (
            _load_component_variant_indices(component_spec_source)
            if component_spec_source
            else []
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Invalid --component-spec: {exc}") from exc
    prediction_bed_prefix = args.prediction_bed_prefix.strip()
    if "," in prediction_bed_prefix:
        raise SystemExit(
            "Sparse prediction accepts one genotype source; supply one BED prefix."
        )
    prediction_bed_list = (
        [prediction_bed_prefix] if prediction_bed_prefix else []
    )
    prediction_pgen_prefix = args.prediction_pgen_prefix.strip()
    prediction_active = bool(
        prediction_bed_list or prediction_pgen_prefix
    )
    _n_formats = sum(bool(x) for x in [bed_list, pgen_prefix])
    if _n_formats == 0:
        raise SystemExit(
            "No genotype input specified. "
            "Use --bed-prefix or --pgen-prefix."
        )
    if _n_formats > 1:
        raise SystemExit(
            "Specify only one of --bed-prefix / --pgen-prefix."
        )
    if prediction_active and sum(
        bool(value)
        for value in (prediction_bed_list, prediction_pgen_prefix)
    ) != 1:
        raise SystemExit(
            "Prediction requires exactly one of --prediction-bed-prefix or "
            "--prediction-pgen-prefix."
        )
    if (
        args.prediction_covar_txt or args.prediction_keep_path
    ) and not prediction_active:
        raise SystemExit(
            "Prediction covariate/keep inputs require a prediction BED or "
            "PGEN prefix."
        )
    if sparsity_validation_requested and not prediction_active:
        raise SystemExit(
            "Sparsity validation requires a prediction BED or PGEN prefix."
        )
    if not args.pheno_txt:
        raise SystemExit("--pheno-txt is required.")

    temp_paths: list[str] = []

    # ---- Sample alignment (BED or PGEN FAM) ----------------
    if pgen_prefix:
        fam_path = make_nonbed_input_fam(pgen_prefix=pgen_prefix)
        temp_paths.append(fam_path)
    else:
        fam_path = bed_list[0] + ".fam"

    for path in temp_paths:
        atexit.register(cleanup_path, path)

    keep_ids = None
    if args.keep_path and os.path.exists(args.keep_path):
        keep_ids = read_keep_ids(args.keep_path)

    # Use the single GRM's FAM as the sample-order reference.
    (
        y_np,
        covar_np,
        fam_keep,
        dropped,
        covar_transform,
    ) = load_pheno_covar_aligned_with_transform(
        fam_path=fam_path,
        pheno_path=args.pheno_txt,
        covar_path=args.covar_txt or None,
        add_intercept=True,
        keep_ids=keep_ids,
    )
    y_np, input_phenotype_mean, input_phenotype_standard_deviation = (
        _standardize_phenotype_at_input(y_np)
    )
    if covar_np is not None:
        covar_np = covar_np.astype(np.float32, copy=False)

    logger.info("Loaded %s samples; dropped %s", y_np.shape[0], len(dropped))

    # ---- PGEN direct read or BED sample-mask ---------------------------------
    sources = None
    sample_mask = None
    if pgen_prefix:
        sample_mask = compute_sample_mask(fam_path, fam_keep)
        sources = [PgenGenoSource(pgen_prefix, sample_mask=sample_mask)]
        logger.info("[INFO] Direct read: PgenGenoSource "
              "n_source=%s m=%s "
              "n_keep=%s", sources[0]._n_full, sources[0].m, sources[0].n)
        sample_mask = None  # handled inside source
    else:
        n_bed = _bed_count(bed_list[0] + ".bed", "iid_count")
        if n_bed != len(fam_keep):
            sample_mask = compute_sample_mask(fam_path, fam_keep)
            logger.info(
                "[INFO] BED path: using sample_mask "
                "(n_bed=%s -> n_keep=%s) "
                "instead of writing a subset BED", n_bed, len(fam_keep)
            )

    if dropped:
        logger.info("Filtered out %s samples missing pheno/covar.", len(dropped))
    logger.info("Using %s samples after alignment.", len(fam_keep))

    out_prefix = args.out_prefix.strip() or "sparse_reml"
    ensure_parent_dir(out_prefix)
    if args.keep_out:
        ensure_parent_dir(args.keep_out)
        with open(args.keep_out, "w") as f:
            for iid in fam_keep:
                f.write(f"{iid} {iid}\n")
        logger.info("Wrote aligned keep list to %s", args.keep_out)
    if args.dropped_out and dropped:
        ensure_parent_dir(args.dropped_out)
        with open(args.dropped_out, "w") as f:
            for iid in dropped:
                f.write(f"{iid}\n")
        logger.info("Wrote dropped IDs to %s", args.dropped_out)

    gpu_name, gpu_total, gpu_free = setup_gpu()
    n_covar = int(covar_np.shape[1]) if covar_np is not None else 0
    cpu_threads, cpu_threads_src = resolve_cpu_threads(args.cpu_threads or None)
    if sources is not None:
        p_list = [src.m for src in sources]
    else:
        p_list = [_bed_count(pref + ".bed", "sid_count") for pref in bed_list]
    try:
        component_variant_indices = _validate_component_partition(
            component_variant_indices,
            n_markers=int(p_list[0]),
        )
    except ValueError as exc:
        raise SystemExit(f"Invalid --component-spec: {exc}") from exc
    planned_n_grm = (
        len(component_variant_indices) if component_variant_indices else 1
    )
    plan = run_planner(
        n_samples=y_np.shape[0], p_list=p_list,
        n_grm=planned_n_grm,
        component_block_sizes=(
            [int(group.size) for group in component_variant_indices]
            if component_variant_indices
            else None
        ),
        gpu_free=gpu_free,
        gpu_budget=(args.gpu_budget_gib * 1024**3) if args.gpu_budget_gib > 0 else None,
        n_covar=n_covar,
        n_rand_vec=args.n_rand_vec,
        slq_samples=args.slq_samples,
        gpu_name=gpu_name,
        ring_depth=args.ring_depth if args.ring_depth > 0 else None,
        source_format=(
            "bed"
            if bed_list
            else "pgen"
            if pgen_prefix
            else None
        ),
        arbitrary_component_partition=bool(component_variant_indices),
        requested_call_width=(args.call_width if args.call_width > 0 else None),
    )
    call_width = plan.call_width
    gpu_budget_bytes = (
        float(args.gpu_budget_gib) * 1024**3
        if args.gpu_budget_gib > 0
        else float(plan.gpu_budget_gib) * 1024**3
    )
    print_planner_info(
        plan, gpu_name, gpu_free, call_width,
    )
    logger.info(
        "GPU params: gpu=%s, free=%s GiB, "
        "call_width=%s, "
        "gpu_budget_gib=%s, "
        "n_rand_vec=%s, precond_rank=%s, "
        "slq_samples=%s, slq_m=%s, "
        "minq_iter=%s, pcg_tol=%s, "
        "max_pcg_iters=%s",
        gpu_name, gpu_free/1024**3 if gpu_free else 'unk',
        call_width,
        args.gpu_budget_gib if args.gpu_budget_gib > 0 else 'auto',
        args.n_rand_vec, plan.precond_rank,
        args.slq_samples, args.slq_m,
        args.minq_iter, args.pcg_tol,
        args.max_pcg_iters,
    )
    logger.info("[INFO] cpu_threads=%s (source=%s)", cpu_threads, cpu_threads_src)
    logger.info("jax devices: %s", jax.devices())
    if component_variant_indices:
        logger.info(
            "[INFO] component partition: spec=%s n_grm=%s sizes=%s",
            component_spec_source,
            planned_n_grm,
            [int(group.size) for group in component_variant_indices],
        )
    if sources is not None:
        fit_cfg = FitConfig(
            sources=sources, sample_mask=sample_mask, device=args.device,
            component_variant_indices=component_variant_indices or None,
            call_width=call_width,
            cpu_threads=cpu_threads,
            keep_host_stats=True,
            gpu_budget_bytes=gpu_budget_bytes,
            ring_depth=plan.ring_depth,
            n_rand_vec=args.n_rand_vec, minq_iter=args.minq_iter,
            slq_samples=args.slq_samples, slq_m=args.slq_m,
            precond_rank=plan.precond_rank,
            reml_pcg_tol=args.pcg_tol,
            effect_pcg_tol=args.pcg_tol,
            response_is_standardized=True,
            capture_reml_diagnostics=bool(args.export_ai),
            cache_reml_setup=True,
            strict_max_linesearch_trials=args.reml_max_linesearch_trials,
            max_pcg_iters=args.max_pcg_iters, pcg_ridge=args.pcg_ridge,
            verbose=args.verbose,
        )
    else:
        fit_cfg = FitConfig(
            bed_prefix=bed_list, device=args.device,
            sample_mask=sample_mask,
            component_variant_indices=component_variant_indices or None,
            call_width=call_width,
            cpu_threads=cpu_threads,
            keep_host_stats=True,
            gpu_budget_bytes=gpu_budget_bytes,
            ring_depth=plan.ring_depth,
            n_rand_vec=args.n_rand_vec, minq_iter=args.minq_iter,
            slq_samples=args.slq_samples, slq_m=args.slq_m,
            precond_rank=plan.precond_rank,
            reml_pcg_tol=args.pcg_tol,
            effect_pcg_tol=args.pcg_tol,
            response_is_standardized=True,
            capture_reml_diagnostics=bool(args.export_ai),
            cache_reml_setup=True,
            strict_max_linesearch_trials=args.reml_max_linesearch_trials,
            max_pcg_iters=args.max_pcg_iters, pcg_ridge=args.pcg_ridge,
            verbose=args.verbose,
        )
    logger.info(
        "[INFO] streamer config: "
        "call_width=%s keep_host_stats=%s",
        call_width, fit_cfg.keep_host_stats,
    )

    logger.info("[INFO] build fitter @ %s", datetime.now().isoformat(timespec='seconds'))
    fitter = InfinitesimalREMLFitter(fit_cfg)
    close_fitter = fitter.close
    atexit.register(close_fitter)
    ops = fitter._assemble_reml_operators()
    grm_index = MultiGRMIndex(
        fitter.streamers,
        component_variant_indices=component_variant_indices or None,
    )
    logger.info(
        "[INFO] sparse GRM: K=%s markers_per_grm=%s total=%s",
        grm_index.n_grm,
        grm_index.m_per_grm.tolist(),
        grm_index.m_total,
    )

    y_jax = jnp.asarray(y_np, dtype=jnp.float32)
    n_grm = len(ops.K_mvs)
    if n_grm != grm_index.n_grm:
        raise RuntimeError(
            "Sparse covariance/operator mismatch: "
            f"{n_grm} operators for {grm_index.n_grm} components."
        )

    # These atoms belong to this fitter's sample set and GRM partition.
    # Theta remains a coefficient of the original K in every covariance solve.
    genetic_trace_atoms = np.asarray(
        jax.device_get(fitter._projected_core_diag_atoms(ops.diag_list)),
        dtype=np.float64,
    )

    def _background_genetic_variance(theta_values: np.ndarray) -> float:
        theta_arr = np.asarray(theta_values, dtype=np.float64).reshape(-1)
        return genetic_variance(theta_arr[:n_grm], genetic_trace_atoms)

    def _background_h2(theta_values: np.ndarray) -> float:
        return _sparse_dense_h2(
            0.0, _background_genetic_variance(theta_values), float(theta_values[-1])
        )

    # Preserve the coefficient initialization; only variance summaries use atoms.
    h2_init_default = 0.5
    if supplied_theta_init:
        theta = _parse_variance_components_init(
            supplied_theta_init,
            n_grm=n_grm,
        )
        theta_init_source = "command_line_json"
    else:
        theta_g0 = np.full(
            n_grm,
            h2_init_default / float(n_grm),
            dtype=np.float64,
        )
        theta_e0 = np.array([1.0 - h2_init_default], dtype=np.float64)
        theta = np.concatenate([theta_g0, theta_e0], axis=0)
        theta_init_source = "equal_kernel_coefficients"
    theta_initial = theta.copy()
    fitter._ensure_projected_core_precond_ready(
        ops,
        var_components_init=jnp.asarray(theta, dtype=jnp.float32),
    )
    logger.info(
        "[INFO] init theta (%s) @ %s: %s",
        theta_init_source,
        datetime.now().isoformat(timespec='seconds'),
        theta.tolist(),
    )

    path_cfg = LassoPathConfig(
        lam_min_ratio=args.lasso_lam_min_ratio,
        n_lambda=args.lasso_n_lambda,
        max_cd_iter=args.lasso_cd_max_iter,
        cd_tol=args.lasso_cd_tol,
        active_set_period=args.lasso_active_set_period,
        kkt_abs_tol=args.kkt_tol,
        kkt_rel_tol=args.kkt_rel_tol,
        fixed_lam_ratio=args.lasso_fixed_lam_ratio,
        verbose=args.verbose,
    )

    if iterative_validation_selection:
        logger.info(
            "[validation alpha] full-marker-KKT Lasso prefix will be evaluated "
            "incrementally with validation early stopping inside every "
            "alpha/theta outer iteration."
        )

    prediction_context: _PredictionFitContext | None = None
    validation_outcome: np.ndarray | None = None
    iterative_validation_trace: list[dict[str, object]] = []
    if iterative_validation_selection:
        prediction_context = _build_prediction_fit_context(
            args=args,
            training_fitter=fitter,
            prediction_bed_list=prediction_bed_list,
            prediction_pgen_prefix=prediction_pgen_prefix,
            covar_transform=covar_transform,
            call_width=call_width,
            cpu_threads=cpu_threads,
            gpu_budget_bytes=gpu_budget_bytes,
            ring_depth=plan.ring_depth,
            component_variant_indices=component_variant_indices,
        )
        validation_outcome = read_phenotype_aligned(
            args.sparsity_validation_pheno_txt,
            prediction_context.sample_ids,
        )
        validation_outcome = (
            np.asarray(validation_outcome, dtype=np.float64)
            - float(input_phenotype_mean)
        ) / float(input_phenotype_standard_deviation)
        logger.info(
            "[validation alpha] aligned validation phenotype for %s samples.",
            int(validation_outcome.size),
        )

    support = np.array([], dtype=np.int64)
    candidate_cache = np.array([], dtype=np.int64)
    previous_fixed_mean = None
    previous_outer_h2: float | None = None
    history: list[dict] = []

    warm_screen = None
    warm_z_dict: dict[int, np.ndarray] = {}
    warm_lasso_candidate = np.array([], dtype=np.int64)
    warm_lasso_beta_path: np.ndarray | None = None
    sparse_path_performance = {
        "candidate_reuse_across_outer": True,
        "buffered_kkt_expansion": True,
        "kkt_small_overflow_absorption": True,
        "lasso_path_coefficient_warm_start": True,
        "lasso_path_mapped_basis_warm_start": True,
        "lasso_column_major_gram": True,
        "lasso_density_adaptive_qb_updates": True,
        "lasso_filtered_exact_kkt_matvec": True,
        "lasso_batched_external_path_products": True,
        "lasso_batched_covariate_path_solves": True,
        "lasso_complete_path_global_kkt": bool(
            iterative_validation_selection
        ),
        "lasso_complete_path_global_kkt_backend": (
            "weighted_basil_short_blocks_with_full_marker_xtv"
            if iterative_validation_selection
            else None
        ),
        "lasso_validation_path_solver": (
            "weighted_batch_screening_iterative_lasso"
            if iterative_validation_selection
            else None
        ),
        "validation_path_early_stopping": bool(
            iterative_validation_selection
        ),
        "validation_path_early_stopping_lag": (
            int(args.validation_early_stopping_lag)
            if iterative_validation_selection
            else None
        ),
        "basil_marker_batch_size": (
            int(args.basil_marker_batch_size)
            if iterative_validation_selection
            else None
        ),
        "basil_lambda_block_size": (
            int(args.basil_lambda_block_size)
            if iterative_validation_selection
            else None
        ),
        "basil_max_iterations": (
            int(args.basil_max_iterations)
            if iterative_validation_selection
            else None
        ),
        "candidate_hinv_columns_reused_within_outer": True,
        "candidate_hinv_pcg_rhs_batch_size": int(
            args.candidate_pcg_rhs_batch_size
        ),
        "candidate_hinv_pcg_batches": 0,
        "candidate_hinv_pcg_columns_solved": 0,
        "candidate_hinv_columns_reused": 0,
        "candidate_hinv_pcg_total_batch_iterations": 0,
        "lasso_path_solves": 0,
        "lasso_path_solve_seconds": 0.0,
        "lasso_validation_selection_seconds": 0.0,
        "lasso_path_global_kkt_passes": 0,
        "lasso_path_global_kkt_seconds": 0.0,
        "lasso_path_global_kkt_points_checked": 0,
        "lasso_cd_iterations": 0,
        "lasso_path_warm_start_rows_used": 0,
        "outer_start_candidate_columns_reused": 0,
        "outer_start_screen_columns_added": 0,
        "kkt_strict_violators_added": 0,
        "kkt_buffered_markers_added": 0,
    }

    final_candidate = np.array([], dtype=np.int64)
    final_lasso = None
    theta_lasso = theta.copy()
    n_samples = y_np.shape[0]
    variance_blocks_completed = 0
    outer_converged = False
    outer_stop_reason = "outer_max"
    final_alignment_pending = False
    provisional_convergence = False
    final_alignment_completed = False
    final_pair_available = False
    final_pair_source = "unavailable"
    final_alignment_warning = None
    last_aligned_pair = None
    last_covariance_reml_diagnostics = None
    final_information = None
    information_markers = None
    mean_information = None
    lasso_reml_stop_reason = ""
    penalized_failure_reason = None

    # ---- Precompute loop-invariant B_screen = [y | covar] on device --------
    screen_parts = [y_np[:, None]]
    if covar_np is not None:
        screen_parts.append(covar_np)
    B_screen_np = np.concatenate(screen_parts, axis=1).astype(np.float32, copy=False)
    B_screen_dev = jnp.asarray(B_screen_np, dtype=jnp.float32)
    n_screen = B_screen_np.shape[1]
    lasso_warm_state_in_summary = None
    if args.lasso_warm_state_in:
        warm_load_args = (
            {
                "target_path_lam_ratios": make_lambda_sequence(
                    1.0,
                    float(args.lasso_lam_min_ratio),
                    int(args.lasso_n_lambda),
                )
            }
            if iterative_validation_selection
            else {"target_lam_ratio": float(args.lasso_fixed_lam_ratio)}
        )
        warm_state = _load_lasso_warm_state(
            args.lasso_warm_state_in,
            grm_index=grm_index,
            **warm_load_args,
        )
        state_candidate = np.asarray(
            warm_state["candidate"], dtype=np.int64
        )
        support = np.asarray(warm_state["support"], dtype=np.int64)
        candidate_cache = state_candidate.copy()
        warm_lasso_candidate = state_candidate.copy()
        warm_lasso_beta_path = np.asarray(
            warm_state["beta_snp_path"], dtype=np.float64
        )
        lasso_warm_state_in_summary = {
            "status": "loaded",
            "path": os.path.abspath(args.lasso_warm_state_in),
            "marker_count": int(state_candidate.size),
            "lambda_rows": int(warm_lasso_beta_path.shape[0]),
            "selected_lam_ratio": float(
                warm_state["selected_lam_ratio"]
            ),
            "coordinate_system": str(warm_state["coordinate_system"]),
            "reuse_mode": str(warm_state["reuse_mode"]),
        }
        sparse_path_performance["cross_fit_marker_state_reused"] = True
        sparse_path_performance["cross_fit_warm_marker_count"] = int(
            state_candidate.size
        )
        logger.info(
            "[Lasso warm] loaded selection state: markers=%s",
            int(state_candidate.size),
        )
    else:
        sparse_path_performance["cross_fit_marker_state_reused"] = False
        sparse_path_performance["cross_fit_warm_marker_count"] = 0

    final_kkt_certificate = {
        "passed": False,
        "tolerance": float("nan"),
        "max_active_error": float("inf"),
        "max_inactive_excess": float("inf"),
        "decision_precision": "ordinary_pcg",
        "method": (
            "weighted_basil_complete_path_global_kkt"
            if iterative_validation_selection
            else "candidate_gram_plus_outside_marker_score"
        ),
    }
    outer = 0
    while outer < int(args.outer_max) or final_alignment_pending:
        final_alignment = bool(final_alignment_pending)
        final_alignment_pending = False
        if not final_alignment:
            outer += 1
        iter_t0 = time.time()
        theta_g = jnp.asarray(theta[:-1], dtype=jnp.float32)
        theta_e = jnp.asarray(theta[-1], dtype=jnp.float32)
        hv = fitter._make_hv(ops, theta_g, theta_e)
        precond = fitter._make_effect_precond(ops, theta_g, theta_e)

        # ---- Step 1: screen PCG (y + covar) ----
        x0_screen = warm_screen if (
            warm_screen is not None and warm_screen.shape == (n_samples, n_screen)
        ) else None
        sol_screen, res_screen, it_screen = pcg_solve(
            hv, B_screen_dev,
            M=precond, tol=args.pcg_tol, maxiter=args.max_pcg_iters,
            X0=x0_screen,
        )
        _require_pcg_converged(
            res_screen,
            tol=args.pcg_tol,
            iters=it_screen,
            maxiter=args.max_pcg_iters,
            stage=f"outer {outer} screening",
        )
        warm_screen = sol_screen

        # ---- Step 2: xtv screening score ----
        Hinv_y_np = np.asarray(sol_screen[:, 0], dtype=np.float64)
        Hinv_covar_np = None
        if covar_np is not None and covar_np.shape[1] > 0:
            Hinv_covar_np = np.asarray(sol_screen[:, 1:n_screen], dtype=np.float64)

        u = compute_projected_hinv_vector(
            covar=covar_np, Hinv_covar=Hinv_covar_np,
            Hinv_target=Hinv_y_np, ridge=args.proj_ridge,
        )

        score = np.abs(grm_index.xtv_all(
            jnp.asarray(u, dtype=jnp.float32), normalize=False,
        ))
        topk = min(int(args.screen_topk), score.size)
        if topk <= 0:
            raise RuntimeError("screen_topk must be > 0.")

        top_idx_unsorted = np.argpartition(score, -topk)[-topk:]
        top_idx = top_idx_unsorted[np.argsort(score[top_idx_unsorted])[::-1]]

        candidate_target = min(int(args.candidate_k), score.size)
        if candidate_target <= 0:
            raise RuntimeError("candidate_k must be > 0.")

        candidate_seed = top_idx[:candidate_target]

        # Keep the complete KKT-certified candidate from the preceding outer
        # update, not merely its active support.  Unioning the current score
        # seed still admits markers whose ranking changed with the covariance.
        candidate = _build_lasso_candidate(
            previous_support=support,
            screened_indices=candidate_seed,
            candidate_target=candidate_target,
            previous_candidate=candidate_cache,
        )
        max_candidate = int(args.kkt_max_candidate)
        if max_candidate > 0 and candidate.size > max_candidate:
            retained = np.unique(
                np.concatenate([candidate_cache, support])
            ).astype(np.int64)
            if retained.size > max_candidate:
                raise RuntimeError(
                    "Previously certified candidate exceeds "
                    f"--kkt-max-candidate ({retained.size} > {max_candidate})."
                )
            retained_set = set(retained.tolist())
            available = max_candidate - int(retained.size)
            fresh = [
                int(index)
                for index in candidate_seed.tolist()
                if int(index) not in retained_set
            ][:available]
            candidate = np.unique(
                np.concatenate(
                    [retained, np.asarray(fresh, dtype=np.int64)]
                )
            ).astype(np.int64)

        reused_candidate_columns = int(
            np.intersect1d(
                candidate, candidate_cache, assume_unique=True
            ).size
        )
        screen_columns_added = int(
            np.setdiff1d(
                candidate, candidate_cache, assume_unique=True
            ).size
        )
        sparse_path_performance[
            "outer_start_candidate_columns_reused"
        ] += reused_candidate_columns
        sparse_path_performance[
            "outer_start_screen_columns_added"
        ] += screen_columns_added
        logger.info(
            "[outer %s%s] candidate seed: reused=%s fresh=%s total=%s",
            outer,
            " final" if final_alignment else "",
            reused_candidate_columns,
            screen_columns_added,
            int(candidate.size),
        )

        # ---- Step 3/4: candidate LASSO with global KKT refinement ----------
        kkt_trace: list[dict] = []
        Z_cand = np.empty((n_samples, 0), dtype=np.float32)
        sol_z_np = np.empty((n_samples, 0), dtype=np.float32)
        res_all = np.asarray(0.0, dtype=np.float32)
        it_all = 0
        lasso = None
        active_local = np.array([], dtype=np.int64)
        support_new = np.array([], dtype=np.int64)
        certified_kkt = False
        penalized_block_failure = None
        # H and therefore Hinv[Z_j] stay fixed throughout candidate expansion
        # inside this outer update.  Cache certified columns here so an
        # expansion solves only its newly admitted markers.
        outer_hinv_z_dict: dict[int, np.ndarray] = {}

        accepted_kkt_record = None
        Hinv_y_for_path = np.asarray(sol_screen[:, 0], dtype=np.float64)
        Hinv_covar_for_path = None
        if covar_np is not None and covar_np.shape[1] > 0:
            Hinv_covar_for_path = np.asarray(
                sol_screen[:, 1:n_screen], dtype=np.float64
            )

        # Validation selection must compare exact solutions of the *global*
        # weighted-Lasso problem at every lambda.  Use a BASIL rollout rather
        # than growing one shared candidate until it happens to solve the whole
        # path: short lambda blocks retain only ever-active markers, add a score
        # batch, and receive one all-marker KKT scan before their certified
        # prefix can enter the validation comparison.
        if iterative_validation_selection:
            try:
                if prediction_context is None or validation_outcome is None:
                    raise RuntimeError(
                        "Iterative validation selection context is unavailable."
                    )

                def evaluate_basil_validation_block(
                    *,
                    candidate,
                    train_candidate,
                    beta_cov_path,
                    beta_snp_path,
                    hinv_residual_path,
                    training_score_path,
                ):
                    return _evaluate_lasso_path_on_validation(
                        args=args,
                        fitter=fitter,
                        prediction_context=prediction_context,
                        y_train=y_np,
                        train_covar=covar_np,
                        train_candidate=train_candidate,
                        candidate=candidate,
                        beta_cov_path=beta_cov_path,
                        beta_snp_path=beta_snp_path,
                        theta=theta,
                        validation_outcome=validation_outcome,
                        hinv_residual_path=hinv_residual_path,
                        training_score_path=training_score_path,
                    )

                basil_result = _fit_complete_weighted_lasso_path_basil(
                    args=args,
                    grm_index=grm_index,
                    hv=hv,
                    precond=precond,
                    y=y_np,
                    covar=covar_np,
                    hinv_y=Hinv_y_for_path,
                    hinv_covar=Hinv_covar_for_path,
                    initial_score_abs=np.asarray(score, dtype=np.float64),
                    path_cfg=path_cfg,
                    previous_candidate=warm_lasso_candidate,
                    previous_beta_path=warm_lasso_beta_path,
                    previous_hinv_z=warm_z_dict,
                    outer=outer,
                    sparse_path_performance=sparse_path_performance,
                    validation_evaluator=evaluate_basil_validation_block,
                    validation_stopping_lag=int(
                        args.validation_early_stopping_lag
                    ),
                )
                lasso_path = basil_result["lasso_path"]
                candidate = np.asarray(
                    basil_result["candidate"], dtype=np.int64
                )
                Z_cand = np.asarray(basil_result["geno"], dtype=np.float32)
                sol_z_np = np.asarray(
                    basil_result["hinv_geno"], dtype=np.float32
                )
                warm_z_dict = dict(basil_result["hinv_z_dict"])
                kkt_trace = list(basil_result["kkt_trace"])
                candidate_pcg = dict(basil_result["candidate_pcg"])
                path_global_kkt = dict(basil_result["global_kkt"])
                validation_bundle = dict(basil_result["validation"])
                validation_metrics = list(validation_bundle["metrics"])
                selected_index = _select_converged_validation_path_index(
                    list(lasso_path["path"]), validation_metrics
                )
                lasso, validation_record = (
                    _materialize_validation_selected_lasso(
                        lasso_path,
                        validation_metrics,
                        selected_index=selected_index,
                        path_prediction_pcg_res=float(
                            validation_bundle["pcg_max_rel_res"]
                        ),
                        path_prediction_pcg_iters=int(
                            validation_bundle["pcg_max_iters"]
                        ),
                    )
                )
                validation_record["early_stopping"] = dict(
                    validation_bundle["early_stopping"]
                )
                validation_record["incremental_batches"] = list(
                    validation_bundle["trace"]
                )
                lasso["validation_selection"] = validation_record
                validation_selection_seconds = float(
                    validation_bundle["seconds"]
                )

                warm_lasso_candidate = candidate.copy()
                warm_lasso_beta_path = np.asarray(
                    lasso["beta_snp_path"], dtype=np.float64
                ).copy()
                active_local = np.asarray(lasso["active_idx"], dtype=np.int64)
                support_new = np.sort(candidate[active_local])
                best_path = dict(lasso["path"][int(lasso["selected_index"])])
                accepted_kkt_record = {
                    "passed": True,
                    "tolerance": float(best_path["kkt_tolerance"]),
                    "max_active_error": float(
                        best_path["max_active_kkt_error"]
                    ),
                    "max_inactive_excess": float(
                        max(
                            float(best_path["max_inactive_kkt_excess"]),
                            float(
                                best_path.get(
                                    "global_kkt_max_outside_score", 0.0
                                )
                            )
                            - float(best_path["lam"]),
                            0.0,
                        )
                    ),
                    "method": "weighted_basil_complete_path_global_kkt",
                    "decision_precision": "ordinary_pcg",
                    "pcg_tol": float(args.pcg_tol),
                    "candidate_path_kkt_passed": bool(
                        best_path["kkt_passed"]
                    ),
                    "complete_path_global_kkt_passed": True,
                    "complete_path_global_kkt_points": int(
                        path_global_kkt["n_path_points"]
                    ),
                    "max_outside_score": float(
                        best_path.get("global_kkt_max_outside_score", 0.0)
                    ),
                    "n_outside_violators": 0,
                }
                certified_kkt = True
                theta_lasso = theta.copy()

                trace_record = {
                    "outer": int(outer),
                    "stage": (
                        "final_covariance_lasso"
                        if final_alignment
                        else "outer_update"
                    ),
                    "final_alignment": bool(final_alignment),
                    "path_solver": "weighted_basil",
                    "candidate_size": int(candidate.size),
                    "lasso_path_warm_start_used": bool(
                        lasso[
                            "external_beta_path_warm_start_used"
                        ]
                    ),
                    "lasso_path_warm_start_rows_used": int(
                        lasso[
                            "external_beta_path_warm_start_rows_used"
                        ]
                    ),
                    "lasso_path_solve_seconds": float(
                        lasso["basil"]["path_solve_seconds"]
                    ),
                    "path_global_kkt_seconds": float(
                        lasso["basil"]["global_kkt_seconds"]
                    ),
                    "validation_selection_seconds": (
                        validation_selection_seconds
                    ),
                    "theta": np.asarray(
                        theta, dtype=np.float64
                    ).tolist(),
                    **validation_record,
                }
                iterative_validation_trace.append(trace_record)
                logger.info(
                    "[outer %s validation alpha/BASIL] ever_active=%s "
                    "ratio=%.6g active=%s validation_predictive_R2=%.8f",
                    outer,
                    int(candidate.size),
                    float(lasso["selected_lam_ratio"]),
                    int(active_local.size),
                    float(
                        validation_record["selected"][
                            "predictive_r2"
                        ]
                    ),
                )
            except (
                FloatingPointError,
                RuntimeError,
                ValueError,
                np.linalg.LinAlgError,
            ) as error:
                penalized_block_failure = (
                    "Weighted BASIL Lasso path failed: " f"{error}"
                )
            # The fixed-ratio branch below retains its selected-lambda KKT
            # refinement.  The validation branch has already certified every
            # path point and therefore has no second refinement loop.
            max_kkt_rounds = 0
        else:
            max_kkt_rounds = int(args.kkt_max_rounds)

        for kkt_round in range(1, max_kkt_rounds + 1):
            path_pcg_tol = float(args.pcg_tol)
            # Extract the current design, but solve Hinv[Z] only for columns
            # not already certified under this outer iteration's covariance.
            Z_cand = grm_index.extract_standardized_columns(candidate).astype(
                np.float32, copy=False
            )
            missing_positions = np.asarray(
                [
                    column
                    for column, marker in enumerate(candidate.tolist())
                    if int(marker) not in outer_hinv_z_dict
                ],
                dtype=np.int64,
            )
            reused_hinv_columns = int(candidate.size - missing_positions.size)
            candidate_pcg = {
                "batch_size": int(args.candidate_pcg_rhs_batch_size),
                "n_batches": 0,
                "n_columns": 0,
                "max_reported_relative_residual": 0.0,
                "max_true_relative_residual": 0.0,
                "max_iterations": 0,
                "total_batch_iterations": 0,
                "warm_start_columns": 0,
                "reused_within_outer_columns": reused_hinv_columns,
            }

            try:
                if missing_positions.size > 0:
                    missing_markers = candidate[missing_positions]
                    missing_rhs = np.ascontiguousarray(
                        Z_cand[:, missing_positions], dtype=np.float32
                    )
                    missing_warm = np.zeros_like(missing_rhs)
                    warm_hits = 0
                    for column, marker in enumerate(
                        missing_markers.tolist()
                    ):
                        previous = warm_z_dict.get(int(marker))
                        if previous is not None:
                            missing_warm[:, column] = previous
                            warm_hits += 1
                    candidate_pcg_started = time.perf_counter()
                    solved_missing, solve_diagnostic = (
                        _solve_hinv_columns_batched(
                            hv=hv,
                            precond=precond,
                            rhs=missing_rhs,
                            warm_start=(
                                missing_warm if warm_hits > 0 else None
                            ),
                            tol=path_pcg_tol,
                            maxiter=int(args.max_pcg_iters),
                            batch_size=int(
                                args.candidate_pcg_rhs_batch_size
                            ),
                            stage=(
                                f"outer {outer} KKT round {kkt_round} "
                                "candidate"
                            ),
                        )
                    )
                    candidate_pcg.update(solve_diagnostic)
                    sparse_path_performance["candidate_hinv_pcg_seconds"] = (
                        sparse_path_performance.get("candidate_hinv_pcg_seconds", 0.0)
                        + time.perf_counter() - candidate_pcg_started
                    )
                    candidate_pcg["warm_start_columns"] = int(warm_hits)
                    candidate_pcg["reused_within_outer_columns"] = int(
                        reused_hinv_columns
                    )
                    for column, marker in enumerate(
                        missing_markers.tolist()
                    ):
                        outer_hinv_z_dict[int(marker)] = solved_missing[
                            :, column
                        ]
                sol_z_np = np.empty_like(Z_cand, dtype=np.float32)
                for column, marker in enumerate(candidate.tolist()):
                    sol_z_np[:, column] = outer_hinv_z_dict[int(marker)]
                res_all = np.asarray(
                    candidate_pcg["max_reported_relative_residual"],
                    dtype=np.float32,
                )
                candidate_true_res = float(
                    candidate_pcg["max_true_relative_residual"]
                )
                it_all = int(candidate_pcg["max_iterations"])
                sparse_path_performance[
                    "candidate_hinv_pcg_batches"
                ] += int(candidate_pcg["n_batches"])
                sparse_path_performance[
                    "candidate_hinv_pcg_columns_solved"
                ] += int(candidate_pcg["n_columns"])
                sparse_path_performance[
                    "candidate_hinv_columns_reused"
                ] += reused_hinv_columns
                sparse_path_performance[
                    "candidate_hinv_pcg_total_batch_iterations"
                ] += int(candidate_pcg["total_batch_iterations"])
                _require_pcg_converged(
                    res_all,
                    tol=path_pcg_tol,
                    iters=it_all,
                    maxiter=args.max_pcg_iters,
                    stage=f"outer {outer} KKT round {kkt_round} candidate",
                )
            except (FloatingPointError, RuntimeError, ValueError) as error:
                kkt_trace.append(
                    {
                        "round": int(kkt_round),
                        "candidate_size": int(candidate.size),
                        "path_pcg_tol": path_pcg_tol,
                        "decision": "candidate_linear_solve_failed",
                        "error": str(error),
                    }
                )
                penalized_block_failure = (
                    "Candidate Hinv[Z] PCG failed as a linear-solve "
                    f"error: {error}"
                )
                break

            # Retain only the current monotone candidate as a warm start for
            # the next covariance update and optional final-refit state.
            warm_z_dict = {
                int(marker): outer_hinv_z_dict[int(marker)]
                for marker in candidate.tolist()
            }

            beta_snp_path0, warm_lasso_columns = _remap_lasso_beta_path(
                previous_candidate=warm_lasso_candidate,
                previous_beta_path=warm_lasso_beta_path,
                candidate=candidate,
            )
            # A candidate expansion retains every old column.  Remap the prior
            # complete path into that enlarged basis; the solver chooses the
            # better initial point independently at each lambda and preserves
            # the same complete score-KKT acceptance rule.
            if not _allow_mapped_lasso_path_warm_start(
                previous_size=int(warm_lasso_candidate.size),
                current_size=int(candidate.size),
                common_size=warm_lasso_columns,
            ):
                beta_snp_path0 = None
                warm_lasso_columns = 0

            lasso_path_started = time.perf_counter()
            lasso_path_solve_seconds = float("nan")
            validation_selection_seconds = 0.0
            path_global_kkt = None
            path_global_kkt_seconds = 0.0
            try:
                try:
                    lasso = fit_weighted_lasso_with_covariates(
                        y=y_np,
                        covar=covar_np,
                        geno=Z_cand,
                        Hinv_y=Hinv_y_for_path,
                        Hinv_covar=Hinv_covar_for_path,
                        Hinv_geno=sol_z_np,
                        cfg=path_cfg,
                        ridge=args.lasso_ridge,
                        beta_snp_path0=beta_snp_path0,
                    )
                except ValueError as warm_error:
                    # A zero-score path has one row regardless of the
                    # requested grid.  If the adjacent fit crosses that
                    # degenerate case, discard only the optional warm start.
                    if (
                        beta_snp_path0 is None
                        or "beta_path0 shape mismatch" not in str(warm_error)
                    ):
                        raise
                    beta_snp_path0 = None
                    warm_lasso_columns = 0
                    lasso = fit_weighted_lasso_with_covariates(
                        y=y_np,
                        covar=covar_np,
                        geno=Z_cand,
                        Hinv_y=Hinv_y_for_path,
                        Hinv_covar=Hinv_covar_for_path,
                        Hinv_geno=sol_z_np,
                        cfg=path_cfg,
                        ridge=args.lasso_ridge,
                    )
                lasso_path_solve_seconds = float(
                    time.perf_counter() - lasso_path_started
                )
                sparse_path_performance["lasso_path_solves"] += 1
                sparse_path_performance["lasso_path_solve_seconds"] += (
                    lasso_path_solve_seconds
                )
                sparse_path_performance["lasso_cd_iterations"] += int(
                    sum(int(row["cd_iter"]) for row in lasso["path"])
                )
                sparse_path_performance[
                    "lasso_path_warm_start_rows_used"
                ] += int(
                    lasso["external_beta_path_warm_start_rows_used"]
                )
                warm_lasso_candidate = candidate.copy()
                warm_lasso_beta_path = np.asarray(
                    lasso["beta_snp_path"], dtype=np.float64
                ).copy()
                if iterative_validation_selection:
                    if prediction_context is None or validation_outcome is None:
                        raise RuntimeError(
                            "Iterative validation selection context is unavailable."
                        )
                    # Validation may compare path points only after every one
                    # is a full-genome Lasso solution.  Recombine the PCG
                    # solves already used by the candidate Gram objective,
                    # then obtain all-marker scores for every lambda in one
                    # streamed X'V pass.  No per-lambda PCG loop is needed.
                    path_kkt_started = time.perf_counter()
                    hinv_residual_path = _build_hinv_lasso_residual_path(
                        hinv_y=Hinv_y_for_path,
                        hinv_covar=Hinv_covar_for_path,
                        hinv_geno=sol_z_np,
                        beta_cov_path=lasso["beta_cov_path"],
                        beta_snp_path=lasso["beta_snp_path"],
                    )
                    path_score_signed = np.asarray(
                        grm_index.xtv_all(
                            jnp.asarray(
                                hinv_residual_path, dtype=jnp.float32
                            ),
                            normalize=False,
                        ),
                        dtype=np.float64,
                    )
                    path_global_kkt = (
                        _certify_complete_lasso_path_kkt_from_scores(
                            score_path=path_score_signed,
                            candidate=candidate,
                            beta_candidate_path=lasso["beta_snp_path"],
                            path_rows=list(lasso["path"]),
                            abs_tol=float(args.kkt_tol),
                            rel_tol=float(args.kkt_rel_tol),
                        )
                    )
                    lasso["path"] = list(path_global_kkt["path_rows"])
                    path_global_kkt_seconds = float(
                        time.perf_counter() - path_kkt_started
                    )
                    sparse_path_performance[
                        "lasso_path_global_kkt_passes"
                    ] += 1
                    sparse_path_performance[
                        "lasso_path_global_kkt_seconds"
                    ] += path_global_kkt_seconds
                    sparse_path_performance[
                        "lasso_path_global_kkt_points_checked"
                    ] += int(path_global_kkt["n_path_points"])

                    path_violators = np.asarray(
                        path_global_kkt["outside_violators"],
                        dtype=np.int64,
                    )
                    if path_violators.size > 0:
                        n_path_viol = int(path_violators.size)
                        expansion_budget = (
                            _complete_path_kkt_expansion_budget(
                                n_path_viol,
                                int(args.kkt_add_topk),
                                int(candidate.size),
                            )
                        )
                        max_candidate = int(args.kkt_max_candidate)
                        candidate_limit_error = None
                        if max_candidate > 0:
                            available = max_candidate - int(candidate.size)
                            strict_required = min(
                                expansion_budget, n_path_viol
                            )
                            if available < strict_required:
                                candidate_limit_error = (
                                    "Complete-path KKT refinement cannot add "
                                    "the required union of strict violators "
                                    "without exceeding --kkt-max-candidate "
                                    f"({candidate.size} + {strict_required} "
                                    f"> {max_candidate})."
                                )
                            else:
                                expansion_budget = min(
                                    expansion_budget, available
                                )
                        expansion = {
                            "add_indices": np.empty(
                                (0,), dtype=np.int64
                            ),
                            "n_strict_added": 0,
                            "n_buffered_added": 0,
                        }
                        if candidate_limit_error is None:
                            expansion = _buffered_kkt_expansion_indices(
                                score_abs=np.asarray(
                                    path_global_kkt["priority_score"],
                                    dtype=np.float64,
                                ),
                                candidate=candidate,
                                violators=path_violators,
                                max_add=expansion_budget,
                            )
                        action = (
                            "candidate_limit_reached"
                            if candidate_limit_error is not None
                            else "expand_candidate"
                        )
                        kkt_trace.append(
                            {
                                "round": int(kkt_round),
                                "candidate_size": int(candidate.size),
                                "support_size": int(
                                    max(
                                        int(row["k"])
                                        for row in lasso["path"]
                                    )
                                ),
                                "lambda": None,
                                "lambda_ratio": None,
                                "selection_method": (
                                    "pending_complete_path_global_kkt"
                                ),
                                "validation_predictive_r2": None,
                                "n_violators": n_path_viol,
                                "n_expansion_strict_added": int(
                                    expansion["n_strict_added"]
                                ),
                                "n_expansion_buffered_added": int(
                                    expansion["n_buffered_added"]
                                ),
                                "lasso_path_warm_start_columns": int(
                                    warm_lasso_columns
                                ),
                                "lasso_path_warm_start_used": bool(
                                    lasso[
                                        "external_beta_path_warm_start_used"
                                    ]
                                ),
                                "lasso_path_warm_start_rows_used": int(
                                    lasso[
                                        "external_beta_path_warm_start_rows_used"
                                    ]
                                ),
                                "lasso_path_solve_seconds": (
                                    lasso_path_solve_seconds
                                ),
                                "path_global_kkt_seconds": (
                                    path_global_kkt_seconds
                                ),
                                "validation_selection_seconds": 0.0,
                                "decision": action,
                                "path_pcg_tol": path_pcg_tol,
                                "candidate_pcg_reported_res": float(
                                    np.asarray(res_all)
                                ),
                                "candidate_pcg_true_res": float(
                                    candidate_true_res
                                ),
                                "candidate_pcg_iters": int(it_all),
                                "candidate_pcg": dict(candidate_pcg),
                                "complete_path_global_kkt": {
                                    "passed": False,
                                    "n_path_points": int(
                                        path_global_kkt["n_path_points"]
                                    ),
                                    "n_violating_path_points": int(
                                        path_global_kkt[
                                            "n_violating_path_points"
                                        ]
                                    ),
                                    "n_union_violators": n_path_viol,
                                    "max_outside_excess_over_threshold": (
                                        float(
                                            path_global_kkt[
                                                "max_outside_excess_over_threshold"
                                            ]
                                        )
                                    ),
                                    "score_backend": str(
                                        path_global_kkt["score_backend"]
                                    ),
                                },
                            }
                        )
                        logger.info(
                            "[outer %s kkt %s path] cand=%s lambdas=%s "
                            "violating_lambdas=%s union_violators=%s "
                            "add=%s buffer=%s warm_cols=%s path_sec=%.1f "
                            "global_kkt_sec=%.1f decision=%s",
                            outer,
                            kkt_round,
                            int(candidate.size),
                            int(path_global_kkt["n_path_points"]),
                            int(
                                path_global_kkt[
                                    "n_violating_path_points"
                                ]
                            ),
                            n_path_viol,
                            int(expansion["n_strict_added"]),
                            int(expansion["n_buffered_added"]),
                            int(warm_lasso_columns),
                            lasso_path_solve_seconds,
                            path_global_kkt_seconds,
                            action,
                        )
                        if candidate_limit_error is not None:
                            penalized_block_failure = (
                                candidate_limit_error
                            )
                            break
                        add_idx = np.asarray(
                            expansion["add_indices"], dtype=np.int64
                        )
                        if add_idx.size == 0:
                            penalized_block_failure = (
                                "Complete-path KKT refinement produced an "
                                "empty expansion despite "
                                f"{n_path_viol} union violators."
                            )
                            break
                        sparse_path_performance[
                            "kkt_strict_violators_added"
                        ] += int(expansion["n_strict_added"])
                        sparse_path_performance[
                            "kkt_buffered_markers_added"
                        ] += int(expansion["n_buffered_added"])
                        candidate = np.unique(
                            np.concatenate([candidate, add_idx])
                        ).astype(np.int64)
                        candidate.sort()
                        if (
                            max_candidate > 0
                            and candidate.size > max_candidate
                        ):
                            penalized_block_failure = (
                                "Complete-path KKT refinement exceeded "
                                "--kkt-max-candidate "
                                f"({candidate.size} > {max_candidate})."
                            )
                            break
                        # The next round remaps this complete alpha path onto
                        # the expanded basis and uses it as a same-lambda warm
                        # start.  Validation is intentionally deferred until
                        # every path point passes the all-marker certificate.
                        continue

                    validation_started = time.perf_counter()
                    lasso, validation_record = (
                        _select_lasso_by_validation_prediction(
                            args=args,
                            fitter=fitter,
                            prediction_context=prediction_context,
                            y_train=y_np,
                            train_covar=covar_np,
                            train_candidate=Z_cand,
                            candidate=candidate,
                            lasso_path=lasso,
                            theta=theta,
                            validation_outcome=validation_outcome,
                        )
                    )
                    validation_selection_seconds = float(
                        time.perf_counter() - validation_started
                    )
                    sparse_path_performance[
                        "lasso_validation_selection_seconds"
                    ] += validation_selection_seconds
                    trace_record = {
                        "outer": int(outer),
                        "stage": (
                            "final_covariance_lasso"
                            if final_alignment
                            else "outer_update"
                        ),
                        "final_alignment": bool(final_alignment),
                        "kkt_round": int(kkt_round),
                        "candidate_size": int(candidate.size),
                        "lasso_path_warm_start_columns": int(
                            warm_lasso_columns
                        ),
                        "lasso_path_warm_start_used": bool(
                            lasso[
                                "external_beta_path_warm_start_used"
                            ]
                        ),
                        "lasso_path_warm_start_rows_used": int(
                            lasso[
                                "external_beta_path_warm_start_rows_used"
                            ]
                        ),
                        "lasso_path_solve_seconds": lasso_path_solve_seconds,
                        "validation_selection_seconds": (
                            validation_selection_seconds
                        ),
                        "theta": np.asarray(
                            theta, dtype=np.float64
                        ).tolist(),
                        **validation_record,
                    }
                    iterative_validation_trace.append(trace_record)
                    logger.info(
                        "[outer %s kkt %s validation alpha] cand=%s "
                        "ratio=%.6g active=%s validation_predictive_R2=%.8f",
                        outer,
                        kkt_round,
                        int(candidate.size),
                        float(lasso["selected_lam_ratio"]),
                        int(np.asarray(lasso["active_idx"]).size),
                        float(
                            validation_record["selected"][
                                "predictive_r2"
                            ]
                        ),
                    )
            except (
                FloatingPointError,
                RuntimeError,
                ValueError,
                np.linalg.LinAlgError,
            ) as error:
                penalized_block_failure = (
                    "Weighted-LASSO solve failed: " f"{error}"
                )
                break
            theta_lasso = theta.copy()

            best_path = min(
                lasso["path"],
                key=lambda row: abs(
                    float(row["lam"]) - float(lasso["lam"])
                ),
            )
            if not (
                bool(best_path.get("converged", False))
                and bool(best_path.get("kkt_passed", False))
                and (
                    not iterative_validation_selection
                    or bool(best_path.get("global_kkt_passed", False))
                )
            ):
                penalized_block_failure = (
                    "Selected LASSO solution did not receive complete "
                    "candidate/full-genome KKT convergence. Increase "
                    "--lasso-cd-max-iter or inspect its score certificate."
                )
                break

            active_local = np.asarray(
                lasso["active_idx"], dtype=np.int64
            )
            support_new = np.sort(candidate[active_local])
            beta_cov = np.asarray(
                lasso.get("beta_cov", np.empty((0,))), dtype=np.float64
            )
            beta_snp = np.asarray(
                lasso["beta_snp"], dtype=np.float64
            )
            resid_lasso = _lasso_residual(
                y=y_np,
                covar=covar_np,
                geno=Z_cand,
                beta_cov=beta_cov,
                beta_snp=beta_snp,
            )
            residual_rhs = jnp.asarray(
                resid_lasso[:, None], dtype=jnp.float32
            )
            try:
                sol_resid, res_kkt, it_kkt = pcg_solve(
                    hv,
                    residual_rhs,
                    M=precond,
                    tol=path_pcg_tol,
                    maxiter=args.max_pcg_iters,
                )
                true_res_kkt = _true_pcg_relative_residual(
                    hv, residual_rhs, sol_resid
                )
                if not np.isfinite(true_res_kkt):
                    raise RuntimeError(
                        "KKT score PCG produced a non-finite true residual."
                    )
                _require_pcg_converged(
                    res_kkt,
                    tol=path_pcg_tol,
                    iters=it_kkt,
                    maxiter=args.max_pcg_iters,
                    stage=f"outer {outer} KKT round {kkt_round} residual",
                )
            except (FloatingPointError, RuntimeError, ValueError) as error:
                kkt_trace.append(
                    {
                        "round": int(kkt_round),
                        "candidate_size": int(candidate.size),
                        "support_size": int(support_new.size),
                        "path_pcg_tol": path_pcg_tol,
                        "decision": "score_linear_solve_failed",
                        "error": str(error),
                    }
                )
                penalized_block_failure = (
                    "KKT score PCG failed as a linear-solve error: "
                    f"{error}"
                )
                break

            score_signed = np.asarray(
                grm_index.xtv_all(sol_resid[:, 0], normalize=False),
                dtype=np.float64,
            )
            kkt_parts = _candidate_lasso_kkt_from_scores(
                score=score_signed,
                candidate=candidate,
                beta_candidate=beta_snp,
                lam=float(lasso["lam"]),
                abs_tol=float(args.kkt_tol),
                rel_tol=float(args.kkt_rel_tol),
            )
            violators = np.asarray(
                kkt_parts["outside_violators"], dtype=np.int64
            )
            max_outside_score = float(
                kkt_parts["max_outside_score"]
            )
            kkt_threshold = float(kkt_parts["threshold"])
            score_kkt_decision = np.abs(score_signed)
            n_viol = int(violators.size)
            internal_kkt_passed = bool(best_path.get("kkt_passed", False))
            internal_active_error = float(
                best_path.get("max_active_kkt_error", float("inf"))
            )
            internal_inactive_excess = float(
                best_path.get("max_inactive_kkt_excess", float("inf"))
            )
            outside_inactive_excess = max(
                max_outside_score - float(lasso["lam"]), 0.0
            )
            action = "accept" if n_viol == 0 else "expand_candidate"
            expansion = {
                "add_indices": np.empty((0,), dtype=np.int64),
                "n_strict_added": 0,
                "n_buffered_added": 0,
            }
            candidate_limit_error = None
            if n_viol > 0:
                nominal_expansion_budget = int(args.kkt_add_topk)
                expansion_budget = _kkt_expansion_budget(
                    n_viol,
                    nominal_expansion_budget,
                )
                max_candidate = int(args.kkt_max_candidate)
                if max_candidate > 0:
                    available = max_candidate - int(candidate.size)
                    strict_required = min(expansion_budget, n_viol)
                    if available < strict_required:
                        candidate_limit_error = (
                            "KKT refinement cannot add the required strict "
                            "violators without exceeding --kkt-max-candidate "
                            f"({candidate.size} + {strict_required} > "
                            f"{max_candidate})."
                        )
                        action = "candidate_limit_reached"
                    else:
                        expansion_budget = min(expansion_budget, available)
                if candidate_limit_error is None:
                    expansion = _buffered_kkt_expansion_indices(
                        score_abs=score_kkt_decision,
                        candidate=candidate,
                        violators=violators,
                        max_add=expansion_budget,
                    )
            current_record = {
                "passed": bool(action == "accept"),
                "tolerance": float(best_path.get("kkt_tolerance", kkt_threshold - float(lasso["lam"]))),
                "max_active_error": internal_active_error,
                "max_inactive_excess": max(
                    internal_inactive_excess, outside_inactive_excess
                ),
                "method": (
                    "complete_path_batched_global_kkt_plus_selected_direct_score"
                    if iterative_validation_selection
                    else "candidate_gram_plus_outside_marker_score"
                ),
                "decision_precision": "ordinary_pcg",
                "pcg_tol": path_pcg_tol,
                "pcg_reported_res": float(np.asarray(res_kkt)),
                "pcg_true_res": float(true_res_kkt),
                "pcg_iters": int(it_kkt),
                "max_outside_score": max_outside_score,
                "n_outside_violators": n_viol,
                "candidate_path_kkt_passed": internal_kkt_passed,
                "complete_path_global_kkt_passed": (
                    bool(path_global_kkt["passed"])
                    if path_global_kkt is not None
                    else None
                ),
                "complete_path_global_kkt_points": (
                    int(path_global_kkt["n_path_points"])
                    if path_global_kkt is not None
                    else None
                ),
                # This independent score is diagnostic only: finite-PCG
                # differences on candidate coordinates do not reject an
                # otherwise solved Lasso block.
                "direct_score_candidate_diagnostic": dict(
                    kkt_parts["candidate_certificate"]
                ),
            }
            kkt_trace.append(
                {
                    "round": int(kkt_round),
                    "candidate_size": int(candidate.size),
                    "support_size": int(support_new.size),
                    "lambda": float(lasso["lam"]),
                    "lambda_ratio": float(lasso["selected_lam_ratio"]),
                    "selection_method": str(lasso["selection_method"]),
                    "validation_predictive_r2": (
                        float(
                            lasso["validation_selection"]["selected"][
                                "predictive_r2"
                            ]
                        )
                        if lasso.get("validation_selection") is not None
                        else None
                    ),
                    "threshold": float(kkt_threshold),
                    "max_outside_score": max_outside_score,
                    "n_violators": n_viol,
                    "n_expansion_strict_added": int(
                        expansion["n_strict_added"]
                    ),
                    "n_expansion_buffered_added": int(
                        expansion["n_buffered_added"]
                    ),
                    "lasso_path_warm_start_columns": int(
                        warm_lasso_columns
                    ),
                    "lasso_path_warm_start_used": bool(
                        lasso["external_beta_path_warm_start_used"]
                    ),
                    "lasso_path_warm_start_rows_used": int(
                        lasso[
                            "external_beta_path_warm_start_rows_used"
                        ]
                    ),
                    "lasso_path_solve_seconds": lasso_path_solve_seconds,
                    "path_global_kkt_seconds": path_global_kkt_seconds,
                    "validation_selection_seconds": (
                        validation_selection_seconds
                    ),
                    "decision": action,
                    "path_pcg_tol": path_pcg_tol,
                    "candidate_pcg_reported_res": float(np.asarray(res_all)),
                    "candidate_pcg_true_res": float(candidate_true_res),
                    "candidate_pcg_iters": int(it_all),
                    "candidate_pcg": dict(candidate_pcg),
                    "complete_path_global_kkt": (
                        {
                            "passed": bool(path_global_kkt["passed"]),
                            "n_path_points": int(
                                path_global_kkt["n_path_points"]
                            ),
                            "n_violating_path_points": int(
                                path_global_kkt[
                                    "n_violating_path_points"
                                ]
                            ),
                            "n_union_violators": int(
                                path_global_kkt["n_union_violators"]
                            ),
                            "max_outside_excess_over_threshold": float(
                                path_global_kkt[
                                    "max_outside_excess_over_threshold"
                                ]
                            ),
                            "score_backend": str(
                                path_global_kkt["score_backend"]
                            ),
                        }
                        if path_global_kkt is not None
                        else None
                    ),
                    "kkt": dict(current_record),
                }
            )
            logger.info(
                "[outer %s kkt %s] cand=%s active=%s lam=%.3e "
                "max_outside=%.3e threshold=%.3e violators=%s "
                "add=%s buffer=%s warm_cols=%s path_sec=%.1f "
                "validation_sec=%.1f decision=%s",
                outer,
                kkt_round,
                int(candidate.size),
                int(support_new.size),
                float(lasso["lam"]),
                max_outside_score,
                kkt_threshold,
                n_viol,
                int(expansion["n_strict_added"]),
                int(expansion["n_buffered_added"]),
                int(warm_lasso_columns),
                lasso_path_solve_seconds,
                validation_selection_seconds,
                action,
            )

            if action == "accept":
                certified_kkt = True
                accepted_kkt_record = current_record
                break
            if candidate_limit_error is not None:
                penalized_block_failure = candidate_limit_error
                break
            add_idx = np.asarray(
                expansion["add_indices"], dtype=np.int64
            )
            if add_idx.size == 0:
                penalized_block_failure = (
                    "KKT refinement produced an empty expansion despite "
                    f"{n_viol} outside-candidate violators."
                )
                break
            sparse_path_performance[
                "kkt_strict_violators_added"
            ] += int(expansion["n_strict_added"])
            sparse_path_performance[
                "kkt_buffered_markers_added"
            ] += int(expansion["n_buffered_added"])
            candidate = np.unique(
                np.concatenate([candidate, add_idx])
            ).astype(np.int64)
            candidate.sort()

            max_candidate = int(args.kkt_max_candidate)
            if max_candidate > 0 and candidate.size > max_candidate:
                penalized_block_failure = (
                    "KKT refinement exceeded --kkt-max-candidate "
                    f"({candidate.size} > {max_candidate})."
                )
                break

        if (
            penalized_block_failure is None
            and not certified_kkt
        ):
            penalized_block_failure = (
                "Failed to certify global LASSO KKT optimality within "
                f"{int(args.kkt_max_rounds)} refinement rounds. "
                "Increase --kkt-max-rounds/--kkt-add-topk or inspect the KKT trace."
            )

        if penalized_block_failure is not None:
            if final_alignment and last_aligned_pair is not None:
                final_candidate = last_aligned_pair["candidate"]
                final_lasso = last_aligned_pair["lasso"]
                support = last_aligned_pair["support"]
                theta = last_aligned_pair["theta"]
                theta_lasso = theta.copy()
                final_kkt_certificate = last_aligned_pair["kkt"]
                final_information = last_aligned_pair["mean_information"]
                final_pair_available = True
                final_pair_source = "last_complete_pair"
                final_alignment_warning = str(penalized_block_failure)
                outer_converged = False
                outer_stop_reason = "final_alignment_failed"
                history.append(
                    {
                        "outer": outer,
                        "stage": "final_covariance_lasso_fallback",
                        "theta": theta.tolist(),
                        "support_size": int(support.size),
                        "kkt_certified": True,
                        "final_alignment": True,
                        "final_alignment_attempt_kkt_certified": False,
                        "variance_update": "not_run_final_alignment",
                        "warning": final_alignment_warning,
                    }
                )
                logger.warning(
                    "[WARN] final covariance Lasso update was unavailable; "
                    "returning the most recent complete alpha/theta pair: %s",
                    final_alignment_warning,
                )
                break
            penalized_failure_reason = str(penalized_block_failure)
            outer_stop_reason = "penalized_block_failed"
            final_candidate = candidate
            final_lasso = lasso
            theta_lasso = theta.copy()
            support = support_new
            history.append(
                {
                    "outer": outer,
                    "theta": theta.tolist(),
                    "support_size": int(support_new.size),
                    "lam": (
                        float(lasso["lam"])
                        if lasso is not None and lasso.get("lam") is not None
                        else None
                    ),
                    "kkt_certified": False,
                    "kkt_trace": kkt_trace,
                    "final_alignment": final_alignment,
                    "variance_update": "not_run",
                    "failure": penalized_failure_reason,
                }
            )
            logger.warning(
                "[WARN] penalized block rejected at outer=%s: %s",
                outer,
                penalized_failure_reason,
            )
            break

        if lasso is None:
            raise RuntimeError("Internal error: LASSO refinement loop did not run.")

        # Only a candidate that passed the full outside-marker KKT check is
        # carried into the next covariance update.
        candidate_cache = candidate.copy()

        if args.verbose:
            logger.info(
                "[outer %s] lasso_select: method=%s candidate=%s "
                "k_selected=%s kkt_certified=%s",
                outer,
                str(lasso["selection_method"]),
                int(candidate.size),
                int(active_local.size),
                bool(certified_kkt),
            )

        # ---- Step 5: covariate-contrast REML variance block ---------------
        # Hold the sparse genetic score fixed and maximize the restricted
        # likelihood after projecting out the complete nuisance design C.
        # Although the response below already subtracts C beta_cov, passing C
        # remains essential: P_C C = 0 makes this exactly equivalent to
        # profiling beta_cov at every candidate covariance, as in the paper.
        beta_cov_current = np.asarray(
            lasso.get("beta_cov", np.empty((0,))), dtype=np.float64
        )
        beta_snp_current = np.asarray(lasso["beta_snp"], dtype=np.float64)
        residual = _lasso_residual(
            y=y_np,
            covar=covar_np,
            geno=Z_cand,
            beta_cov=beta_cov_current,
            beta_snp=beta_snp_current,
        )
        fixed_mean_current = np.asarray(y_np, dtype=np.float64) - residual
        sparse_mean_current = fixed_mean_current.copy()
        if covar_np is not None and covar_np.size and beta_cov_current.size:
            sparse_mean_current -= np.asarray(covar_np, dtype=np.float64) @ beta_cov_current
        information_started = time.perf_counter()
        active_markers = candidate[active_local]
        if information_markers is None or not np.array_equal(active_markers, information_markers):
            mean_information = SparseMeanInformation(
                Z_cand[:, active_local], covar_np,
                hinv_active=sol_z_np[:, active_local],
                batch_size=int(args.candidate_pcg_rhs_batch_size),
            )
            information_markers = active_markers.copy()
        mean_state, residual, beta_cov_current = mean_information.statistics(
            theta, hv, precond, np.asarray(y_np, dtype=np.float64)-sparse_mean_current,
            tol=min(float(args.pcg_tol), 1e-5), maxiter=int(args.max_pcg_iters),
        )
        lasso["beta_cov"] = beta_cov_current
        fixed_mean_current = sparse_mean_current + (
            np.asarray(covar_np) @ beta_cov_current if covar_np is not None else 0.
        )
        sparse_path_performance["mean_information_seconds"] = (
            sparse_path_performance.get("mean_information_seconds", 0.)
            + time.perf_counter()-information_started
        )
        selected_h2, outer_q_sparse = _outer_coherit_h2_from_fitted_sparse_mean(
            sparse_mean_current,
            residual,
            background_genetic_variance=_background_genetic_variance(theta),
            residual_variance=float(theta[-1]),
            mean_uncertainty_trace=mean_state["trace"],
        )
        selected_validation_r2 = (
            float(lasso["validation_selection"]["selected"]["predictive_r2"])
            if lasso.get("validation_selection") is not None else None
        )
        # Every score-bearing record describes alpha and its actual covariance.
        if iterative_validation_selection:
            iterative_validation_trace[-1].update(
                coherit_h2=float(selected_h2), q_sparse=float(outer_q_sparse)
            )

        # Alignment can change alpha and h2. Only the returned state may be
        # labelled converged; a failed check reuses this Lasso for the next
        # covariance update instead of solving the same path a second time.
        if final_alignment:
            alignment_effect_rel = _relative_fitted_mean_change(
                fixed_mean_current, previous_fixed_mean, y_np
            )
            alignment_h2_stable, alignment_h2_change = _heritability_converged(
                selected_h2, previous_outer_h2, abs_tol=float(args.h2_abs_tol)
            )
            alignment_effect_stable = bool(
                np.isfinite(alignment_effect_rel)
                and alignment_effect_rel <= float(args.effect_rel_tol)
            )
            alignment_action = _alignment_action(
                provisional_convergence=provisional_convergence,
                h2_stable=alignment_h2_stable,
                effect_stable=alignment_effect_stable,
                outer=outer,
                outer_max=int(args.outer_max),
            )
            history.append(
                {
                    "outer": outer,
                    "stage": (
                        "covariance_alignment_check" if alignment_action == "continue"
                        else "final_covariance_lasso"
                    ),
                    "theta": theta.tolist(),
                    "coherit_h2": float(selected_h2),
                    "q_sparse": float(outer_q_sparse),
                    "mean_uncertainty_trace_per_n": mean_state["trace"]/n_samples,
                    "mean_information_rank": mean_information.rank,
                    "h2_abs_change": float(alignment_h2_change),
                    "h2_stable": bool(alignment_h2_stable),
                    "effect_rel": float(alignment_effect_rel),
                    "effect_stable": bool(alignment_effect_stable),
                    "alignment_action": alignment_action,
                    "support_size": int(support_new.size),
                    "lam": float(lasso["lam"]),
                    "lam_ratio": float(lasso["selected_lam_ratio"]),
                    "lambda_selection_method": str(
                        lasso["selection_method"]
                    ),
                    "validation_predictive_r2": selected_validation_r2,
                    "kkt_certified": True,
                    "kkt_trace": kkt_trace,
                    "final_alignment": True,
                    "variance_update": "not_run_final_alignment",
                }
            )
            logger.info(
                "[alignment after outer %s] h2=%.6f h2_change=%.3e "
                "effect_rel=%.3e action=%s",
                outer, selected_h2, alignment_h2_change,
                alignment_effect_rel, alignment_action,
            )
            if alignment_action != "continue":
                support = support_new
                final_candidate = candidate
                final_information = mean_information
                final_lasso = lasso
                final_alignment_completed = True
                final_pair_available = True
                final_pair_source = "final_covariance_lasso"
                final_kkt_certificate = dict(accepted_kkt_record)
                outer_converged = alignment_action == "converged"
                outer_stop_reason = alignment_action
                break
            outer += 1
            final_alignment = False
            provisional_convergence = False
            if iterative_validation_selection:
                iterative_validation_trace[-1].update(
                    outer=int(outer), stage="outer_update", final_alignment=False
                )

        last_aligned_pair = {
            "candidate": candidate.copy(), "lasso": lasso,
            "support": support_new.copy(), "theta": theta.copy(),
            "kkt": dict(accepted_kkt_record),
            "mean_information": mean_information,
        }

        covariance_started = time.perf_counter()
        try:
            ml_res = _fit_covariate_contrast_residual_reml(
                fitter,
                residual,
                theta,
                covar=covar_np,
                h2_init=_background_h2(theta),
                mean_information=mean_information,
            )
            theta_new, lasso_reml_stop_reason = _accepted_reml_theta(
                ml_res,
                expected_components=n_grm + 1,
                stage=f"outer {outer} Lasso covariate-contrast REML block",
            )
            if args.export_ai:
                if (
                    ml_res.final_ai is None
                    or ml_res.final_grad is None
                    or ml_res.final_loglik is None
                ):
                    raise RuntimeError(
                        "--export-ai requested but the sparse covariance REML "
                        "block did not capture final diagnostics."
                    )
                last_covariance_reml_diagnostics = {
                    "theta": theta_new.tolist(),
                    "score": np.asarray(
                        jax.device_get(ml_res.final_grad), dtype=np.float64
                    ).tolist(),
                    "average_information_per_sample": np.asarray(
                        jax.device_get(ml_res.final_ai), dtype=np.float64
                    ).tolist(),
                    "restricted_loglik_per_sample": float(
                        ml_res.final_loglik
                    ),
                    "stop_reason": str(lasso_reml_stop_reason),
                }
            # ``ll_down`` rejects its last trial, preserving the last accepted
            # theta, including any earlier accepted steps in this block.
            variance_blocks_completed += 1
            sparse_path_performance["covariance_update_seconds"] = (
                sparse_path_performance.get("covariance_update_seconds", 0.0)
                + time.perf_counter() - covariance_started
            )
        except (FloatingPointError, RuntimeError, ValueError) as error:
            penalized_failure_reason = str(error)
            outer_stop_reason = "covariate_contrast_reml_failed"
            support = support_new
            final_candidate = candidate
            final_lasso = lasso
            history.append(
                {
                    "outer": outer,
                    "theta": theta.tolist(),
                    "support_size": int(support_new.size),
                    "lam": float(lasso["lam"]),
                    "kkt_certified": bool(certified_kkt),
                    "kkt_trace": kkt_trace,
                    "final_alignment": False,
                    "variance_update": "failed",
                    "failure": penalized_failure_reason,
                }
            )
            logger.warning(
                "[WARN] covariate-contrast REML block rejected at outer=%s: %s",
                outer,
                penalized_failure_reason,
            )
            break
        if args.verbose:
            logger.info(
                "[outer %s] covariate_contrast_reml_init_theta=%s stop=%s",
                outer,
                theta.tolist(),
                lasso_reml_stop_reason,
            )

        # ---- Convergence checks ------------------------------------------
        effect_rel = _relative_fitted_mean_change(
            fixed_mean_current,
            previous_fixed_mean,
            y_np,
        )
        effect_stable = bool(
            np.isfinite(effect_rel)
            and effect_rel <= float(args.effect_rel_tol)
        )
        hv_after = fitter._make_hv(
            ops, jnp.asarray(theta_new[:-1], dtype=jnp.float32),
            jnp.asarray(theta_new[-1], dtype=jnp.float32),
        )
        precond_after = fitter._make_effect_precond(
            ops, jnp.asarray(theta_new[:-1], dtype=jnp.float32),
            jnp.asarray(theta_new[-1], dtype=jnp.float32),
        )
        mean_after, residual_after, _gamma_after = mean_information.statistics(
            theta_new, hv_after, precond_after,
            np.asarray(y_np, dtype=np.float64)-sparse_mean_current,
            tol=min(float(args.pcg_tol), 1e-5), maxiter=int(args.max_pcg_iters),
        )
        outer_h2, outer_q_after = (
            _outer_coherit_h2_from_fitted_sparse_mean(
                sparse_mean_current,
                residual_after,
                background_genetic_variance=(
                    _background_genetic_variance(theta_new)
                ),
                residual_variance=float(theta_new[-1]),
                mean_uncertainty_trace=mean_after["trace"],
            )
        )
        h2_stable, h2_abs_change = _heritability_converged(
            outer_h2,
            previous_outer_h2,
            abs_tol=float(args.h2_abs_tol),
        )

        history.append({
            "outer": outer,
            "stage": "outer_update",
            "pcg_screen_iters": int(it_screen),
            "pcg_screen_res": float(np.asarray(res_screen)),
            "pcg_all_iters": int(it_all),
            "pcg_all_res": float(np.asarray(res_all)),
            "theta": theta.tolist(),
            "theta_after_variance_update": theta_new.tolist(),
            "support_size": int(support_new.size),
            "effect_rel": float(effect_rel),
            "effect_stable": bool(effect_stable),
            "coherit_h2": float(selected_h2),
            "coherit_h2_after_variance_update": float(outer_h2),
            "q_sparse": float(outer_q_sparse),
            "q_sparse_after_variance_update": float(outer_q_after),
            "mean_uncertainty_trace_per_n": mean_state["trace"]/n_samples,
            "mean_uncertainty_trace_per_n_after_variance_update": mean_after["trace"]/n_samples,
            "mean_information_rank": mean_information.rank,
            "variance_update_h2_abs_change": float(h2_abs_change),
            "variance_update_h2_stable": bool(h2_stable),
            "lam": float(lasso["lam"]),
            "lam_ratio": float(lasso["selected_lam_ratio"]),
            "lambda_selection_method": str(lasso["selection_method"]),
            "validation_predictive_r2": selected_validation_r2,
            "kkt_certified": bool(certified_kkt),
            "kkt_trace": kkt_trace,
            "final_alignment": False,
            "variance_update": "sparse_mean_information_corrected_reml",
            "variance_stop_reason": lasso_reml_stop_reason,
            "variance_step_rejected": bool(
                lasso_reml_stop_reason == "ll_down"
            ),
        })

        logger.info(
            "[outer %s] pcg_screen=%s pcg_all=%s cand=%s active=%s "
            "kkt_rounds=%s lam=%.3e validation_predictive_R2=%s h2=%.6f "
            "h2_after_variance_update=%.6f variance_update_h2_change=%.3e "
            "effect_rel=%.3e iter_time=%.1fs",
            outer,
            int(it_screen),
            int(it_all),
            int(candidate.size),
            int(support_new.size),
            len(kkt_trace),
            float(lasso["lam"]),
            (
                "%.8f"
                % float(
                    lasso["validation_selection"]["selected"][
                        "predictive_r2"
                    ]
                )
                if lasso.get("validation_selection") is not None
                else "fixed"
            ),
            selected_h2,
            outer_h2,
            h2_abs_change,
            effect_rel,
            time.time() - iter_t0,
        )

        theta = theta_new
        support = support_new
        final_candidate = candidate
        final_lasso = lasso
        previous_fixed_mean = fixed_mean_current
        previous_outer_h2 = float(outer_h2)

        if h2_stable and effect_stable:
            provisional_convergence = True
            final_alignment_pending = True
            logger.info(
                "[INFO] provisional convergence at update %s; checking "
                "the covariance-aligned %s-selected Lasso.",
                outer,
                str(lasso["selection_method"]),
            )
        elif outer >= int(args.outer_max):
            outer_stop_reason = "outer_max"
            final_alignment_pending = True
            logger.warning(
                "[WARN] outer iteration reached the limit (%s) before the "
                "change tolerances; returning the finite iterate after one "
                "final covariance-aligned %s-selected Lasso.",
                int(args.outer_max),
                str(lasso["selection_method"]),
            )
    # Freeze the primary COHERIT branch.  The final Lasso and theta now come
    # from the same covariance, with convergence checked on this returned pair.
    theta_lasso_ml = np.asarray(theta, dtype=np.float64).copy()
    lasso_ml_outer_converged = bool(outer_converged)
    theta_lasso_to_lasso_ml_rel = _max_rel_change(
        theta_lasso_ml, theta_lasso
    )
    last_round_kkt_certified = bool(
        history and history[-1].get("kkt_certified", False)
    )
    finite_valid_theta = bool(
        theta_lasso_ml.shape == (n_grm + 1,)
        and np.all(np.isfinite(theta_lasso_ml))
        and np.all(theta_lasso_ml[:-1] >= 0.0)
        and theta_lasso_ml[-1] > 0.0
    )
    alpha_theta_pair_usable = bool(
        final_pair_available
        and final_lasso is not None
        and last_round_kkt_certified
        and bool(final_kkt_certificate["passed"])
        and finite_valid_theta
        and penalized_failure_reason is None
    )
    alpha_theta_fixed_point_coherent = bool(
        lasso_ml_outer_converged and alpha_theta_pair_usable
    )
    outer_convergence_warning = (
        None
        if lasso_ml_outer_converged or not alpha_theta_pair_usable
        else (
            "outer_max_reached_before_change_tolerances"
            if outer_stop_reason == "outer_max" else outer_stop_reason
        )
    )

    # ---- Output results ----
    out_dir = os.path.dirname(out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    unavailable = float("nan")
    background_genetic_variance = _background_genetic_variance(theta_lasso_ml)
    theta_e_lasso_ml = float(theta_lasso_ml[-1])
    q_chive = unavailable
    q_chive_term1 = unavailable
    q_chive_term2 = unavailable
    q_mean_uncertainty = unavailable
    beta_cov_lasso = np.empty((0,), dtype=np.float64)
    beta_lasso_active = np.empty((0,), dtype=np.float64)
    lasso_quadratics_available = bool(
        alpha_theta_pair_usable
        and final_lasso is not None
        and penalized_failure_reason is None
    )
    Z_support = np.empty((n_samples, 0), dtype=np.float32)

    if final_lasso is not None:
        beta_snp_final = np.asarray(final_lasso["beta_snp"], dtype=np.float64)
        beta_cov_lasso = np.asarray(
            final_lasso.get("beta_cov", np.empty((0,))),
            dtype=np.float64,
        ).reshape(-1)
        try:
            final_active_local = np.asarray(
                final_lasso["active_idx"], dtype=np.int64
            ).reshape(-1)
            final_active_support = np.sort(
                final_candidate[final_active_local]
            )
            if not np.array_equal(final_active_support, support):
                lasso_quadratics_available = False
                if penalized_failure_reason is None:
                    penalized_failure_reason = (
                        "Final Lasso active set does not match the exported "
                        "support."
                    )
                alpha_theta_fixed_point_coherent = False
                alpha_theta_pair_usable = False
        except (IndexError, KeyError) as error:
            lasso_quadratics_available = False
            if penalized_failure_reason is None:
                penalized_failure_reason = (
                    "Final Lasso active-set validation failed: "
                    f"{error}"
                )
            alpha_theta_fixed_point_coherent = False
            alpha_theta_pair_usable = False
        if support.size > 0:
            Z_support = (
                grm_index.extract_standardized_columns(support)
                .astype(np.float32, copy=False)
            )
            cand_pos = {
                int(snp): i for i, snp in enumerate(final_candidate.tolist())
            }
            try:
                beta_lasso_active = np.asarray(
                    [
                        beta_snp_final[cand_pos[int(snp)]]
                        for snp in support.tolist()
                    ],
                    dtype=np.float64,
                )
            except (IndexError, KeyError) as error:
                lasso_quadratics_available = False
                alpha_theta_fixed_point_coherent = False
                alpha_theta_pair_usable = False
                penalized_failure_reason = (
                    "Final Lasso support/coefficient mapping failed: "
                    f"{error}"
                )

        if lasso_quadratics_available:
            if final_information is None:
                final_information = SparseMeanInformation(Z_support, covar_np)
            final_hv = fitter._make_hv(
                ops, jnp.asarray(theta_lasso_ml[:-1], dtype=jnp.float32),
                jnp.asarray(theta_lasso_ml[-1], dtype=jnp.float32),
            )
            final_precond = fitter._make_effect_precond(
                ops, jnp.asarray(theta_lasso_ml[:-1], dtype=jnp.float32),
                jnp.asarray(theta_lasso_ml[-1], dtype=jnp.float32),
            )
            final_g = np.asarray(Z_support, dtype=np.float64) @ beta_lasso_active
            final_mean_state, _final_residual, beta_cov_lasso = final_information.statistics(
                theta_lasso_ml, final_hv, final_precond,
                np.asarray(y_np, dtype=np.float64)-final_g,
                tol=min(float(args.pcg_tol), 1e-5), maxiter=int(args.max_pcg_iters),
            )
            q_mean_uncertainty = final_mean_state["trace"]/n_samples
            y_chive = np.asarray(y_np, dtype=np.float64)
            if (
                covar_np is not None
                and covar_np.size > 0
                and beta_cov_lasso.size > 0
            ):
                y_chive -= (
                    np.asarray(covar_np, dtype=np.float64)
                    @ beta_cov_lasso
                )
            q_chive, q_chive_term1, q_chive_term2 = (
                _chive_q_hat_given_active(
                    Z_support,
                    y_chive,
                    beta_lasso_active,
                )
            )
            q_chive -= q_mean_uncertainty

    # The sole COHERIT estimate uses the calibrated Lasso quadratic and the
    # covariate-contrast REML covariance from the same penalized branch.
    h2_chive = _sparse_dense_h2(
        q_chive,
        background_genetic_variance,
        theta_e_lasso_ml,
    )
    branch_guards = _coherit_estimator_guard(
        alpha_theta_pair_certified=alpha_theta_pair_usable,
        lasso_quadratics_available=lasso_quadratics_available,
        h2_chive=h2_chive,
    )
    lasso_branch_valid = bool(branch_guards["lasso_branch_valid"])
    sparse_outputs_finite = bool(
        branch_guards["lasso_outputs_finite"]
    )
    sparse_fit_rejection_reasons = list(
        branch_guards["lasso_branch_invalid_reasons"]
    )
    if not lasso_branch_valid:
        logger.warning(
            "[WARN] COHERIT estimator unavailable: lasso_valid=%s "
            "reasons=%s. No ordinary-REML value will replace it.",
            lasso_branch_valid,
            ",".join(sparse_fit_rejection_reasons),
        )

    h2_chive_guarded = float(h2_chive) if lasso_branch_valid else unavailable
    h2 = h2_chive_guarded
    primary_h2_method = (
        "information_corrected_sparse_reml" if lasso_branch_valid else "unavailable"
    )

    lasso_warm_state_out_summary = None
    if args.lasso_warm_state_out:
        if (
            not alpha_theta_pair_usable
            or final_lasso is None
        ):
            raise RuntimeError(
                "Lasso warm-state output requires a valid final "
                "covariance-aligned Lasso pair."
            )
        lasso_warm_state_out_summary = _write_lasso_warm_state(
            args.lasso_warm_state_out,
            grm_index=(grm_index if component_variant_indices else None),
            candidate=final_candidate,
            support=support,
            selected_beta_snp=np.asarray(
                final_lasso["beta_snp"], dtype=np.float64
            ),
            selected_lam_ratio=float(final_lasso["selected_lam_ratio"]),
            beta_snp_path=np.asarray(
                final_lasso["beta_snp_path"], dtype=np.float64
            ),
            path_lam_ratios=np.asarray(
                [row["lam_ratio"] for row in final_lasso["path"]],
                dtype=np.float64,
            ),
        )
        logger.info(
            "[Lasso warm] emitted selection state -> %s",
            args.lasso_warm_state_out,
        )

    print(f"[RESULT] var_components_lasso_ml={theta_lasso_ml.tolist()}")
    print(f"[RESULT] h2={h2:.6f} (primary={primary_h2_method})")
    print(f"[RESULT] h2_chive={h2_chive:.6f} (penalized LASSO calibration)")
    print(f"[RESULT] support_size={int(support.size)}")
    # Preserve the candidate/outside, PCG-compatible KKT diagnostic. Non-finite
    # placeholders from an unavailable check become JSON null.
    final_kkt_certificate_summary = _json_safe_value(final_kkt_certificate)

    # One background-effect solve serves both effect-size output and every
    # prediction cohort.  The response has already had the fitted nuisance and
    # sparse means removed, so this is exactly the branch-matched BLUP used by
    # sparse prediction; no second PCG solve is needed.
    background_effects = None
    sparse_effect_summary: dict[str, object] = {
        "requested": bool(args.compute_effects),
        "status": "not_requested",
    }
    background_effects_requested = bool(args.compute_effects or prediction_active)
    if lasso_branch_valid and background_effects_requested:
        background_residual = np.asarray(y_np, dtype=np.float64).copy()
        if covar_np is not None and beta_cov_lasso.size:
            background_residual -= (
                np.asarray(covar_np, dtype=np.float64) @ beta_cov_lasso
            )
        if Z_support.shape[1]:
            background_residual -= (
                np.asarray(Z_support, dtype=np.float64) @ beta_lasso_active
            )
        background_effects = fitter.estimate_effects(
            jnp.asarray(background_residual, dtype=jnp.float32),
            var_components=jnp.asarray(theta_lasso_ml, dtype=jnp.float32),
            covar=None,
        )

    if args.compute_effects:
        if not lasso_branch_valid or background_effects is None:
            raise RuntimeError(
                "--compute-effects requires a valid final sparse COHERIT branch."
            )
        effect_component_specs = (
            load_component_specs(component_spec_source)
            if component_spec_source
            else []
        )
        effect_paths = write_sparse_effect_outputs(
            out_prefix=out_prefix,
            background_effects=background_effects,
            nuisance_fixed_effects=beta_cov_lasso,
            sample_ids=fam_keep,
            variant_records=iter_variant_records_for_prefix(
                pgen_prefix if pgen_prefix else bed_list[0],
                "pgen" if pgen_prefix else "bed",
            ),
            support_source_variant_indices=grm_index.source_variant_indices(
                support
            ),
            sparse_effects=beta_lasso_active,
            component_source_variant_indices=(
                [np.asarray(group, dtype=np.int64) for group in component_variant_indices]
                if component_variant_indices
                else None
            ),
            component_names=(
                [str(spec.name) for spec in effect_component_specs]
                if effect_component_specs
                else None
            ),
            component_annotations=(
                [spec.annotation for spec in effect_component_specs]
                if effect_component_specs
                else None
            ),
            component_provenance=(
                [spec.provenance for spec in effect_component_specs]
                if effect_component_specs
                else None
            ),
        )
        sparse_effect_summary = {
            "requested": True,
            "status": "emitted",
            "paths": effect_paths,
        }

    prediction_summary = {
        "requested": bool(prediction_active),
        "status": "not_requested",
    }
    if iterative_validation_selection and lasso_branch_valid:
        if validation_outcome is None or final_lasso is None:
            raise RuntimeError(
                "Iterative validation selection completed without a final model."
            )
        sparsity_validation_summary = _write_iterative_validation_output(
            output_path=args.sparsity_validation_out,
            phenotype_path=args.sparsity_validation_pheno_txt,
            validation_outcome=validation_outcome,
            selection_trace=iterative_validation_trace,
            final_lasso=final_lasso,
            final_candidate=final_candidate,
            grm_index=grm_index,
            theta=theta_lasso_ml,
            outer_converged=lasso_ml_outer_converged,
            outer_stop_reason=outer_stop_reason,
        )
    else:
        sparsity_validation_summary = {
            "requested": bool(sparsity_validation_requested),
            "status": (
                "pending" if sparsity_validation_requested else "not_requested"
            ),
        }
    if not prediction_active:
        # A reused output prefix must not retain a prediction table from an
        # earlier run when the current run did not request one.
        remove_sparse_prediction_outputs(out_prefix)
    if prediction_active:
        emitted_branches = ["lasso"] if lasso_branch_valid else []
        prediction_request_metadata = {
            "estimator_mode": "coherit",
            "genotype_standardization_source": "training_samples_only",
            "covariate_transform_source": "training_samples_only",
            "prediction_keep_path": args.prediction_keep_path or None,
            "prediction_genotype": {
                "format": "pgen" if prediction_pgen_prefix else "bed",
                "prefixes": (
                    [prediction_pgen_prefix]
                    if prediction_pgen_prefix
                    else prediction_bed_list
                ),
            },
            "input_phenotype_standardization": {
                "mean": float(input_phenotype_mean),
                "standard_deviation": float(
                    input_phenotype_standard_deviation
                ),
            },
        }
        if not emitted_branches:
            metadata_path = write_sparse_prediction_status(
                out_prefix=out_prefix,
                status="not_emitted_no_valid_branch",
                metadata={
                    **prediction_request_metadata,
                    "reason": "no_valid_sparse_prediction_branch",
                    "lasso_branch_valid": lasso_branch_valid,
                    "sparse_fit_rejection_reasons": list(
                        sparse_fit_rejection_reasons
                    ),
                    "branch_outputs_emitted": False,
                    "emitted_branches": [],
                    "branches": {
                        "lasso": {
                            "estimator_valid": False,
                            "output_emitted": False,
                            "invalid_reasons": list(
                                branch_guards[
                                    "lasso_branch_invalid_reasons"
                                ]
                            ),
                        },
                    },
                },
            )
            prediction_summary = {
                "requested": True,
                "status": "not_emitted_no_valid_branch",
                "emitted_branches": [],
                "metadata_path": metadata_path,
            }
            if sparsity_validation_requested:
                raise RuntimeError(
                    "Sparsity validation cannot proceed because the final "
                    "Lasso branch is invalid."
                )
        else:
            if prediction_context is None:
                prediction_context = _build_prediction_fit_context(
                    args=args,
                    training_fitter=fitter,
                    prediction_bed_list=prediction_bed_list,
                    prediction_pgen_prefix=prediction_pgen_prefix,
                    covar_transform=covar_transform,
                    call_width=call_width,
                    cpu_threads=cpu_threads,
                    gpu_budget_bytes=gpu_budget_bytes,
                    ring_depth=plan.ring_depth,
                    component_variant_indices=component_variant_indices,
                )
            prediction_summary = _emit_lasso_prediction(
                out_prefix=out_prefix,
                keep_path=args.prediction_keep_path,
                prediction_context=prediction_context,
                prediction_bed_list=prediction_bed_list,
                prediction_pgen_prefix=prediction_pgen_prefix,
                input_phenotype_mean=input_phenotype_mean,
                input_phenotype_standard_deviation=(
                    input_phenotype_standard_deviation
                ),
                training_fitter=fitter,
                y_train=y_np,
                train_covar=covar_np,
                train_support=Z_support,
                support_indices=support,
                beta_cov=beta_cov_lasso,
                beta_active=beta_lasso_active,
                theta=theta_lasso_ml,
                pcg_tol=args.pcg_tol,
                max_pcg_iters=args.max_pcg_iters,
                background_effects=background_effects,
            )

    output_contract = _sparse_output_contract()
    evaluated_path_kkt_certified = bool(
        final_lasso is not None
        and final_lasso.get("path")
        and (
            all(
                bool(row.get("global_kkt_passed", False))
                for row in final_lasso["path"]
            )
            if iterative_validation_selection
            else bool(final_kkt_certificate["passed"])
        )
    )
    summary = {
        # Schema 7 uses one analysis scale after input standardization and
        # removes the former raw/standardized duplicate fields.
        "sparse_output_schema_version": output_contract[
            "sparse_output_schema_version"
        ],
        "estimator_mode": output_contract["estimator_mode"],
        "computed_estimators": output_contract["computed_estimators"],
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_sec": float(time.time() - t0),
        "n_samples": int(y_np.shape[0]),
        "n_covar": int(covar_np.shape[1]) if covar_np is not None else 0,
        "n_snps_total": grm_index.m_total,
        "n_grms": grm_index.n_grm,
        "m_per_grm": grm_index.m_per_grm.tolist(),
        **effective_preconditioner_rank_contract(fit_cfg, fitter),
        "component_spec": component_spec_source or None,
        "component_partition_mode": (
            "source_variant_index" if component_variant_indices else None
        ),
        "sparse_grm_mode": (
            "component_partitioned_multi_grm"
            if component_variant_indices
            else "single_whole_genome_grm"
        ),
        "grm_variance_scale": "trace_weighted",
        "genetic_trace_atoms": genetic_trace_atoms.tolist(),
        "lambda_selection_method": (
            str(final_lasso["selection_method"])
            if final_lasso is not None
            else None
        ),
        "lasso_path_complete": bool(
            final_lasso is not None
            and len(final_lasso["path"]) == int(args.lasso_n_lambda)
        ),
        "lasso_path_role": (
            str(final_lasso["path_role"])
            if final_lasso is not None
            else None
        ),
        "lasso_path_points_solved": (
            int(len(final_lasso["path"]))
            if final_lasso is not None
            else 0
        ),
        "lasso_path_points_requested": int(args.lasso_n_lambda),
        "lasso_validation_early_stopping": (
            dict(final_lasso.get("validation_early_stopping", {}))
            if final_lasso is not None
            and iterative_validation_selection
            else None
        ),
        "sparse_path_performance_optimizations": dict(
            sparse_path_performance
        ),
        "validation_selection_inside_outer_loop": bool(
            iterative_validation_selection
        ),
        "lasso_fixed_lam_ratio_requested": args.lasso_fixed_lam_ratio,
        "lasso_selected_lam_ratio": (
            float(final_lasso["selected_lam_ratio"])
            if final_lasso is not None
            else None
        ),
        "lasso_selected_validation_r2": (
            float(
                final_lasso["validation_selection"]["selected"][
                    "predictive_r2"
                ]
            )
            if final_lasso is not None
            and final_lasso.get("validation_selection") is not None
            else None
        ),
        "var_components_lasso_ml": theta_lasso_ml.tolist(),
        "variance_component_branch_mapping": {
            "h2_chive": "var_components_lasso_ml",
        },
        "var_components_at_lasso": theta_lasso.tolist(),
        "input_phenotype_standardization": {
            "mean": float(input_phenotype_mean),
            "standard_deviation": float(
                input_phenotype_standard_deviation
            ),
        },
        "primary_h2_method": primary_h2_method,
        "lasso_branch_valid": lasso_branch_valid,
        "lasso_branch_invalid_reasons": list(
            branch_guards["lasso_branch_invalid_reasons"]
        ),
        "lasso_outputs_finite": bool(
            branch_guards["lasso_outputs_finite"]
        ),
        "sparse_outputs_finite": sparse_outputs_finite,
        "sparse_fit_rejection_reasons": sparse_fit_rejection_reasons,
        "lasso_ml_outer_converged": lasso_ml_outer_converged,
        "outer_iterations": int(variance_blocks_completed),
        "outer_max": int(args.outer_max),
        "pcg_tol": float(args.pcg_tol),
        "kkt_abs_tol_effective": float(args.kkt_tol),
        "kkt_rel_tol_effective": float(args.kkt_rel_tol),
        "h2_abs_tol": float(args.h2_abs_tol),
        "effect_rel_tol": float(args.effect_rel_tol),
        "outer_stop_reason": outer_stop_reason,
        "outer_convergence_warning": outer_convergence_warning,
        "penalized_failure_reason": penalized_failure_reason,
        "lasso_variance_update": "sparse_mean_information_corrected_reml",
        "lasso_variance_contrast": "orthogonal_to_complete_nuisance_design",
        "lasso_variance_analysis_dimension": int(
            y_np.shape[0] - n_covar
        ),
        "reml_max_linesearch_trials": int(
            args.reml_max_linesearch_trials
        ),
        "alpha_theta_fixed_point_coherent": alpha_theta_fixed_point_coherent,
        "alpha_theta_pair_usable": alpha_theta_pair_usable,
        "final_covariance_lasso_completed": final_alignment_completed,
        "final_pair_source": final_pair_source,
        "final_alignment_warning": final_alignment_warning,
        "final_kkt_certificate": final_kkt_certificate_summary,
        "theta_lasso_to_lasso_ml_rel_change": (
            theta_lasso_to_lasso_ml_rel
        ),
        "h2_background_lasso_ml": _background_h2(theta_lasso_ml),
        "h2_chive": _finite_float_or_none(h2_chive),
        "h2_chive_guarded": h2_chive_guarded,
        "h2": h2,
        "q_chive": _finite_float_or_none(q_chive),
        "q_chive_components": {
            "term1_g2_over_n": _finite_float_or_none(q_chive_term1),
            "term2_cross": _finite_float_or_none(q_chive_term2),
            "term3_mean_uncertainty_subtracted": _finite_float_or_none(q_mean_uncertainty),
        },
        "mean_information_rank": None if final_information is None else final_information.rank,
        "support_size": int(support.size),
        "support_indices": support.tolist(),
        "support_source_indices": grm_index.source_variant_indices(support).tolist(),
        "kkt_certification_scope": (
            "all_evaluated_path_points"
            if iterative_validation_selection
            else "fixed_lambda_target"
        ),
        "kkt_certificate_definition": (
            "candidate_gram_plus_outside_marker_score"
        ),
        "evaluated_lasso_path_kkt_certified": (
            evaluated_path_kkt_certified
        ),
        "kkt_certified": bool(
            history and bool(history[-1].get("kkt_certified", False))
        ),
        "final_kkt_certified": bool(final_kkt_certificate["passed"]),
        "outer_history": history,
        "sparse_effects": sparse_effect_summary,
        "sparse_prediction": prediction_summary,
        "sparsity_validation": sparsity_validation_summary,
    }

    if args.export_ai:
        if last_covariance_reml_diagnostics is None:
            raise RuntimeError(
                "--export-ai requested but no covariance REML block completed."
            )
        summary["covariance_reml_diagnostics"] = (
            last_covariance_reml_diagnostics
        )

    if supplied_theta_init:
        summary["variance_components_initial"] = theta_initial.tolist()
        summary["variance_components_init_source"] = theta_init_source
    if lasso_warm_state_in_summary is not None:
        summary["lasso_warm_state_in"] = lasso_warm_state_in_summary
    if lasso_warm_state_out_summary is not None:
        summary["lasso_warm_state_out"] = lasso_warm_state_out_summary

    with open(out_prefix + ".summary.json", "w") as f:
        json.dump(
            _json_safe_value(summary),
            f,
            indent=2,
            allow_nan=False,
        )
    with open(out_prefix + ".history.json", "w") as f:
        json.dump(
            _json_safe_value(history),
            f,
            indent=2,
            allow_nan=False,
        )

    beta_map: dict[int, float] = {}
    if final_lasso is not None and final_candidate.size > 0:
        beta_snp = np.asarray(final_lasso["beta_snp"], dtype=np.float64)
        for snp_idx, beta_val in zip(final_candidate.tolist(), beta_snp.tolist()):
            if beta_val != 0.0:
                beta_map[int(snp_idx)] = float(beta_val)
    if sources is None:
        bim_rows = grm_index.lookup_bim_rows(bed_list, support)
    elif support.size > 0 and pgen_prefix:
        source_support = grm_index.source_variant_indices(support)
        source_rows = _lookup_pvar_rows(pgen_prefix + ".pvar", source_support)
        bim_rows = {}
        for global_snp, source_snp in zip(support.tolist(), source_support.tolist()):
            if int(source_snp) in source_rows:
                bim_rows[int(global_snp)] = source_rows[int(source_snp)]
    else:
        bim_rows = {}
    source_index_map = {
        int(global_snp): int(source_snp)
        for global_snp, source_snp in zip(
            support.tolist(), grm_index.source_variant_indices(support).tolist()
        )
    }
    # Build global → grm_index map for output
    _snp_grm_map: dict[int, int] = {}
    if support.size > 0:
        for g, _local, _positions in grm_index.global_to_local(support):
            for pos in _positions:
                _snp_grm_map[int(support[pos])] = g

    with open(out_prefix + ".selected_snps.tsv", "w") as f:
        f.write("\t".join(output_contract["selected_snp_columns"]) + "\n")
        for snp_idx in support.tolist():
            chr_, snp_id, cm, bp, a1, a2 = bim_rows.get(
                int(snp_idx),
                ("NA", f"SNP_{int(snp_idx)}", "NA", "NA", "NA", "NA"),
            )
            grm_id = _snp_grm_map.get(int(snp_idx), -1)
            source_snp_idx = source_index_map.get(int(snp_idx), int(snp_idx))
            beta_val = beta_map.get(int(snp_idx), 0.0)
            row = (
                f"{int(snp_idx)}\t{source_snp_idx}\t{grm_id}\t{chr_}\t"
                f"{snp_id}\t{cm}\t{bp}\t{a1}\t{a2}\t{beta_val:.8e}"
            )
            f.write(row + "\n")

    logger.info("[INFO] done @ %s elapsed=%.1fs", datetime.now().isoformat(timespec='seconds'), time.time() - t0)
    logger.info("[INFO] summary -> %s.summary.json", out_prefix)
    logger.info("[INFO] support -> %s.selected_snps.tsv", out_prefix)
    log_runtime_gpu_memory(plan)
    close_fitter()
    atexit.unregister(close_fitter)


if __name__ == "__main__":
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(message)s"))
    for _name in ("GPU_REML_v6", __name__):
        _lg = logging.getLogger(_name)
        _lg.addHandler(_h)
        _lg.setLevel(logging.INFO)
    main()
