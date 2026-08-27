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
import csv
import dataclasses
import importlib
import itertools
import json
import logging
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
import scipy.linalg as sla
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
_precond_mod = importlib.import_module(f"{pkg_name}.precond")
_common_mod = importlib.import_module(f"{pkg_name}.pipeline_common")
_io_utils_mod = importlib.import_module(f"{pkg_name}.io_utils")
_component_spec_mod = importlib.import_module(f"{pkg_name}.component_spec")
_adaptive_partition_mod = importlib.import_module(
    f"{pkg_name}.adaptive_partition"
)
_covtree_mod = importlib.import_module(f"{pkg_name}.covtree")
_covariance_score_mod = importlib.import_module(
    f"{pkg_name}.covariance_score"
)
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
compute_projected_hinv_vector = _lasso_mod.compute_projected_hinv_vector
fit_weighted_lasso_with_covariates = _lasso_mod.fit_weighted_lasso_with_covariates
pcg_solve = _pcg_mod.pcg_solve
load_component_specs = _component_spec_mod.load_component_specs
AdaptiveComponent = _adaptive_partition_mod.AdaptiveComponent
write_component_spec = _adaptive_partition_mod.write_component_spec
generate_covtree_candidates = _covtree_mod.generate_covtree_candidates
replace_covtree_parent = _covtree_mod.replace_parent
selective_covariance_warm_start = (
    _covtree_mod.selective_covariance_warm_start
)
evaluate_covtree_candidates = (
    _covariance_score_mod.evaluate_covtree_candidates
)
predict_sparse_branch = _sparse_prediction_mod.predict_sparse_branch
predict_sparse_path_partitioned = (
    _sparse_prediction_mod.predict_sparse_path_partitioned
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

def _bed_count(path: str, attr: str) -> int:
    bed = open_bed(path)
    try:
        return int(getattr(bed, attr))
    finally:
        close = getattr(bed, "close", None)
        if close is not None:
            close()


def _load_component_variant_indices(path: str) -> list[np.ndarray]:
    return [
        np.asarray(spec.variant_indices, dtype=np.int64).reshape(-1)
        for spec in load_component_specs(path)
    ]


def _load_ld_score_in_bim_order(path: str, bim_path: str) -> np.ndarray:
    """Load individual-marker LD scores after proving exact BIM-ID alignment."""
    values: list[float] = []
    with open(path, encoding="utf-8", newline="") as score_handle:
        reader = csv.DictReader(score_handle, delimiter="\t")
        if not reader.fieldnames or not {"ID", "ld_score"}.issubset(
            reader.fieldnames
        ):
            raise ValueError("CovTree LD-score table must contain ID and ld_score.")
        with open(bim_path, encoding="utf-8") as bim_handle:
            sentinel = object()
            for row_index, pair in enumerate(
                itertools.zip_longest(reader, bim_handle, fillvalue=sentinel),
                start=1,
            ):
                score_row, bim_line = pair
                if score_row is sentinel or bim_line is sentinel:
                    raise ValueError("CovTree LD-score and BIM row counts differ.")
                fields = str(bim_line).split()
                if len(fields) < 2 or str(score_row["ID"]) != fields[1]:
                    raise ValueError(
                        "CovTree LD-score/BIM order mismatch at row "
                        f"{row_index}."
                    )
                value = float(score_row["ld_score"])
                if not np.isfinite(value) or value < 0.0:
                    raise ValueError(
                        f"Invalid CovTree LD score at row {row_index}."
                    )
                values.append(value)
    result = np.asarray(values, dtype=np.float64)
    if result.size == 0:
        raise ValueError("CovTree LD-score table is empty.")
    return result


def _covtree_components_from_spec(path: str) -> list[AdaptiveComponent]:
    specs = load_component_specs(path)
    return [
        AdaptiveComponent(
            name=str(spec.name),
            variant_indices=np.asarray(spec.variant_indices, dtype=np.int64),
            annotation=dict(spec.annotation or {}),
        )
        for spec in specs
    ]


def _load_sparse_numerical_state(
    path: str,
    *,
    grm_index: "MultiGRMIndex",
    n_samples: int,
    n_lambda: int,
) -> dict[str, object]:
    """Load a parent sparse state and remap source SNPs to current cache order."""
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "source_candidate_indices",
            "source_support_indices",
            "beta_snp_path",
            "screen_solution",
            "z_solution",
            "fixed_mean",
        }
        if not required.issubset(payload.files):
            raise ValueError("Sparse state artifact is incomplete.")
        source_candidate = np.asarray(
            payload["source_candidate_indices"], dtype=np.int64
        ).reshape(-1)
        source_support = np.asarray(
            payload["source_support_indices"], dtype=np.int64
        ).reshape(-1)
        beta_path = np.asarray(payload["beta_snp_path"], dtype=np.float64)
        screen_solution = np.asarray(payload["screen_solution"], dtype=np.float32)
        z_solution = np.asarray(payload["z_solution"], dtype=np.float32)
        fixed_mean = np.asarray(payload["fixed_mean"], dtype=np.float64).reshape(-1)
    if (
        np.unique(source_candidate).size != source_candidate.size
        or np.unique(source_support).size != source_support.size
        or np.setdiff1d(source_support, source_candidate).size > 0
    ):
        raise ValueError("Sparse state candidate/support source indices are invalid.")
    if beta_path.shape != (int(n_lambda), int(source_candidate.size)):
        raise ValueError(
            "Sparse state beta path shape mismatch: "
            f"{beta_path.shape} != {(int(n_lambda), int(source_candidate.size))}."
        )
    if z_solution.shape != (int(n_samples), int(source_candidate.size)):
        raise ValueError("Sparse state Hinv[Z] matrix has the wrong shape.")
    if screen_solution.ndim != 2 or screen_solution.shape[0] != int(n_samples):
        raise ValueError("Sparse state screening solution has the wrong shape.")
    if fixed_mean.shape != (int(n_samples),):
        raise ValueError("Sparse state fixed mean has the wrong shape.")
    arrays_to_check = (beta_path, screen_solution, z_solution, fixed_mean)
    if not all(np.all(np.isfinite(value)) for value in arrays_to_check):
        raise ValueError("Sparse numerical state contains non-finite values.")

    cache = np.arange(grm_index.m_total, dtype=np.int64)
    cache_to_source = grm_index.source_variant_indices(cache)
    source_to_cache = np.empty(grm_index.m_total, dtype=np.int64)
    source_to_cache[cache_to_source] = cache
    if np.any(
        (source_candidate < 0) | (source_candidate >= grm_index.m_total)
    ):
        raise ValueError("Sparse state contains an out-of-range source marker.")
    candidate = source_to_cache[source_candidate]
    support = source_to_cache[source_support]
    return {
        "candidate": candidate,
        "support": np.sort(support),
        "beta_snp_path": beta_path,
        "screen_solution": jnp.asarray(screen_solution, dtype=jnp.float32),
        "z_solution": z_solution,
        "fixed_mean": fixed_mean,
    }


def _write_sparse_numerical_state(
    path: str,
    *,
    grm_index: "MultiGRMIndex",
    candidate: np.ndarray,
    support: np.ndarray,
    beta_snp_path: np.ndarray,
    screen_solution,
    warm_z_dict: dict[int, np.ndarray],
    fixed_mean: np.ndarray,
) -> dict[str, object]:
    """Persist the exact state reusable under a covariance-preserving split."""
    candidate_indices = np.asarray(candidate, dtype=np.int64).reshape(-1)
    support_indices = np.asarray(support, dtype=np.int64).reshape(-1)
    beta_path = np.asarray(beta_snp_path, dtype=np.float32)
    screen = np.asarray(jax.device_get(screen_solution), dtype=np.float32)
    mean = np.asarray(fixed_mean, dtype=np.float32).reshape(-1)
    if beta_path.ndim != 2 or beta_path.shape[1] != candidate_indices.size:
        raise ValueError("Cannot emit sparse state: beta path/candidate mismatch.")
    missing_z = [
        int(index) for index in candidate_indices if int(index) not in warm_z_dict
    ]
    if missing_z:
        raise ValueError(
            "Cannot emit sparse state: Hinv[Z] columns are incomplete."
        )
    z_solution = np.column_stack(
        [warm_z_dict[int(index)] for index in candidate_indices]
    ).astype(np.float32, copy=False)
    source_candidate = grm_index.source_variant_indices(candidate_indices)
    source_support = grm_index.source_variant_indices(support_indices)
    ensure_parent_dir(path)
    temporary = f"{path}.tmp.{os.getpid()}"
    with open(temporary, "wb") as handle:
        np.savez(
            handle,
            source_candidate_indices=source_candidate,
            source_support_indices=source_support,
            beta_snp_path=beta_path,
            screen_solution=screen,
            z_solution=z_solution,
            fixed_mean=mean,
        )
    os.replace(temporary, path)
    return {
        "status": "emitted",
        "path": os.path.abspath(path),
        "candidate_size": int(candidate_indices.size),
        "support_size": int(support_indices.size),
        "lambda_rows": int(beta_path.shape[0]),
        "screen_rhs_columns": int(screen.shape[1]),
        "z_solution_shape": list(z_solution.shape),
        "coordinate_system": "source_marker_index",
        "reuse_contract": "covariance_preserving_split_only",
    }


def _normalized_design_is_well_conditioned(
    design: np.ndarray,
    *,
    relative_tol: float,
) -> bool:
    """Match the normalized-Gram rank criterion used by REML."""
    if design.shape[1] == 0:
        return True
    norms = np.linalg.norm(design, axis=0)
    if not np.all(np.isfinite(norms)) or np.any(norms <= 0.0):
        return False
    normalized = design / norms
    gram = normalized.T @ normalized
    eigvals = np.linalg.eigvalsh(0.5 * (gram + gram.T))
    return bool(
        eigvals[0]
        > float(relative_tol) * max(float(eigvals[-1]), 1.0)
    )


def _merge_independent_fixed_effects(
    covar: np.ndarray | None,
    active_geno: np.ndarray,
    *,
    relative_tol: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray]:
    """Append a numerically independent subset of active SNP fixed effects.

    LASSO may select perfectly linked SNPs. Their fixed-effect columns span the
    same space, but passing every duplicate to REML makes ``X'V^-1X`` singular.
    Pivoted QR finds a stable spanning subset while always retaining the base
    covariates. The returned indices refer to columns of ``active_geno``.
    """
    active = np.asarray(active_geno, dtype=np.float64)
    if active.ndim != 2:
        raise ValueError("active_geno must be a two-dimensional matrix.")
    n_samples = int(active.shape[0])

    if covar is None:
        base = np.empty((n_samples, 0), dtype=np.float64)
    else:
        base = np.asarray(covar, dtype=np.float64)
        if base.ndim != 2 or int(base.shape[0]) != n_samples:
            raise ValueError("covar and active_geno must have matching rows.")
    if not np.all(np.isfinite(base)) or not np.all(np.isfinite(active)):
        raise ValueError("Fixed-effect columns must contain only finite values.")
    if not _normalized_design_is_well_conditioned(
        base, relative_tol=relative_tol
    ):
        raise ValueError(
            "covar is rank-deficient or numerically collinear; "
            "remove redundant fixed-effect columns."
        )

    active_norms = np.linalg.norm(active, axis=0)
    eligible = np.flatnonzero(np.isfinite(active_norms) & (active_norms > 0.0))
    if eligible.size == 0:
        return np.asarray(base, dtype=np.float32), np.empty((0,), dtype=np.int64)

    active_normalized = active[:, eligible] / active_norms[eligible]
    if base.shape[1] > 0:
        base_normalized = base / np.linalg.norm(base, axis=0)
        q_base, _ = sla.qr(
            base_normalized,
            mode="economic",
            check_finite=False,
        )
        active_residual = active_normalized - q_base @ (q_base.T @ active_normalized)
    else:
        active_residual = active_normalized

    _, r_active, piv = sla.qr(
        active_residual,
        mode="economic",
        pivoting=True,
        check_finite=False,
    )
    diag = np.abs(np.diag(r_active))
    if diag.size == 0:
        selected_order = np.empty((0,), dtype=np.int64)
    else:
        qr_tol = np.sqrt(float(relative_tol)) * max(float(diag[0]), 1.0)
        rank = int(np.count_nonzero(diag > qr_tol))
        selected_order = eligible[np.asarray(piv[:rank], dtype=np.int64)]

    while selected_order.size > 0:
        trial = np.concatenate([base, active[:, selected_order]], axis=1)
        if _normalized_design_is_well_conditioned(
            trial, relative_tol=relative_tol
        ):
            break
        selected_order = selected_order[:-1]

    selected = np.sort(selected_order)
    merged = np.concatenate([base, active[:, selected]], axis=1)
    return np.asarray(merged, dtype=np.float32), selected


# ---------------------------------------------------------------------------
# Multi-GRM helpers
# ---------------------------------------------------------------------------

class MultiGRMIndex:
    """
    Maps global SNP indices to (grm_index, local_snp_index) pairs.

    With G GRMs having m_0, m_1, … SNPs, the global index space is
    [0, m_0) for GRM 0, [m_0, m_0+m_1) for GRM 1, etc.
    For a component-partitioned single source, this is the streamer's
    canonical component-concatenated cache order.  Source BIM/PVAR indices are
    obtained explicitly through :meth:`source_variant_indices`.
    """

    def __init__(self, streamers, call_plan=(), component_variant_indices=None):
        self.streamers = streamers
        self.call_plan = tuple(call_plan)
        self._partitioned_single_streamer = (
            component_variant_indices is not None
            and len(streamers) == 1
        )
        self._source_variant_indices = None
        if self._partitioned_single_streamer:
            if len(streamers) != 1:
                raise ValueError(
                    "Single-source component partitioning requires exactly one streamer."
                )
            streamer = streamers[0]
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
                    "Partitioned streamer's cache-to-source SNP map has the wrong length."
                )
            if len(requested_groups) != self.n_grm:
                raise ValueError(
                    "Component count mismatch between component spec and genotype streamer."
                )
            for component_idx, requested in enumerate(requested_groups):
                start = int(component_offsets[component_idx])
                stop = int(component_offsets[component_idx + 1])
                actual = cache_to_source[start:stop]
                # The streamer canonicalizes each component to increasing
                # source order.  Validate the requested membership, then use
                # that canonical map as the sole coordinate source for sparse
                # output, BIM/PVAR lookup and prediction auditing.
                expected = np.unique(requested)
                if not np.array_equal(actual, expected):
                    raise ValueError(
                        "Component SNP mapping mismatch between component spec "
                        f"and genotype streamer for component {component_idx}."
                    )
            self._source_variant_indices = cache_to_source.copy()
        else:
            self.n_grm = len(streamers)
            self.m_per_grm = np.array([st.m for st in streamers], dtype=np.int64)
        self.offsets = np.zeros(self.n_grm + 1, dtype=np.int64)
        np.cumsum(self.m_per_grm, out=self.offsets[1:])
        self.m_total = int(self.offsets[-1])

    def _validated_global_indices(self, global_idx: np.ndarray) -> np.ndarray:
        gidx = np.asarray(global_idx, dtype=np.int64)
        if gidx.ndim != 1:
            raise ValueError("global_idx must be one-dimensional.")
        if np.any((gidx < 0) | (gidx >= self.m_total)):
            raise IndexError(
                f"Global SNP indices must lie in [0, {self.m_total})."
            )
        return gidx

    def global_to_local(
        self, global_idx: np.ndarray
    ) -> list[tuple[int, np.ndarray, np.ndarray]]:
        """
        Convert global SNP indices to per-GRM groups.

        Returns list of (grm_idx, local_indices, positions_in_input) tuples,
        where positions_in_input are the positions in the original global_idx
        array so results can be assembled back.
        """
        gidx = self._validated_global_indices(global_idx)
        grm_ids = np.searchsorted(self.offsets[1:], gidx, side="right")
        grm_ids = np.clip(grm_ids, 0, self.n_grm - 1)
        groups: list[tuple[int, np.ndarray, np.ndarray]] = []
        for g in range(self.n_grm):
            mask = grm_ids == g
            if not np.any(mask):
                continue
            positions = np.flatnonzero(mask)
            local = gidx[positions] - int(self.offsets[g])
            groups.append((g, local, positions))
        return groups

    def xtv_all(self, u_jax: jnp.ndarray, normalize: bool = False) -> np.ndarray:
        """
        Compute X^T u across all GRMs, returning a global (m_total,) score.
        """
        if self._partitioned_single_streamer:
            return np.asarray(
                self.streamers[0].xtv(u_jax, normalize=normalize),
                dtype=np.float64,
            )
        if self.n_grm > 1 and self.call_plan:
            from .kv_impl import xtv_impl_multi_streamed_concat

            for st in self.streamers:
                st._prepare_kv_pass()
            return np.asarray(
                xtv_impl_multi_streamed_concat(
                    u_jax,
                    self.streamers,
                    self.call_plan,
                    missing_val=int(self.streamers[0]._missing_val),
                    normalize=normalize,
                ),
                dtype=np.float64,
            )

        scores = None
        for g, st in enumerate(self.streamers):
            off = int(self.offsets[g])
            block = np.asarray(
                st.xtv(u_jax, normalize=normalize), dtype=np.float64
            )
            if scores is None:
                scores = np.zeros(
                    (self.m_total, *block.shape[1:]),
                    dtype=np.float64,
                )
            scores[off : off + st.m, ...] = block
        if scores is None:
            raise RuntimeError("xtv_all requires at least one genotype streamer.")
        return scores

    def extract_standardized_columns(
        self, global_idx: np.ndarray
    ) -> np.ndarray:
        """
        Extract standardized genotype columns for global SNP indices.
        Dispatches to the correct streamer for each GRM and assembles
        columns in the original order.
        """
        gidx = self._validated_global_indices(global_idx)
        if self._partitioned_single_streamer:
            return self.streamers[0].extract_standardized_columns(gidx)
        n = self.streamers[0].n
        out = np.empty((n, gidx.size), dtype=np.float32)
        for g, local, positions in self.global_to_local(gidx):
            cols = self.streamers[g].extract_standardized_columns(local)
            out[:, positions] = cols
        return out

    def source_variant_indices(self, global_idx: np.ndarray) -> np.ndarray:
        gidx = self._validated_global_indices(global_idx)
        if self._source_variant_indices is None:
            return gidx.copy()
        return np.asarray(self._source_variant_indices[gidx], dtype=np.int64)

    def lookup_bim_rows(
        self, bed_prefixes: list[str], global_idx: np.ndarray
    ) -> dict[int, tuple[str, str, str, str, str, str]]:
        """
        Look up BIM info for global SNP indices, dispatching to the
        correct .bim file for each GRM.
        """
        result: dict[int, tuple[str, str, str, str, str, str]] = {}
        if self._partitioned_single_streamer:
            source_idx = self.source_variant_indices(global_idx)
            source_rows = _lookup_bim_rows(bed_prefixes[0] + ".bim", source_idx)
            for global_snp, src_idx in zip(global_idx.tolist(), source_idx.tolist()):
                if int(src_idx) in source_rows:
                    result[int(global_snp)] = source_rows[int(src_idx)]
            return result
        for g, local, positions in self.global_to_local(global_idx):
            bim_path = bed_prefixes[g] + ".bim"
            local_rows = _lookup_bim_rows(bim_path, local)
            for pos, loc_idx in zip(positions, local):
                global_snp = int(global_idx[pos])
                if int(loc_idx) in local_rows:
                    result[global_snp] = local_rows[int(loc_idx)]
        return result


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
    component_variant_indices: list[np.ndarray],
    prediction_bed_list: list[str],
    prediction_pgen_prefix: str,
    covar_transform,
    call_width: int,
    cpu_threads: int,
    gpu_budget_bytes: float,
    ring_depth: int,
) -> _PredictionFitContext:
    """Build prediction state using training-only genotype standardization."""
    standardization_overrides = []
    for streamer in training_fitter.streamers:
        if streamer._means_host is None or streamer._inv_sds_host is None:
            raise RuntimeError(
                "Sparse prediction requires retained training SNP "
                "standardization statistics."
            )
        standardization_overrides.append(
            (streamer._means_host, streamer._inv_sds_host)
        )

    if prediction_pgen_prefix:
        prediction_fam_path = make_nonbed_input_fam(
            pgen_prefix=prediction_pgen_prefix
        )
        atexit.register(cleanup_path, prediction_fam_path)
    else:
        prediction_fam_path = prediction_bed_list[0] + ".fam"

    requested_prediction_ids = None
    if args.prediction_keep_path:
        if not os.path.exists(args.prediction_keep_path):
            raise SystemExit(
                "--prediction-keep-path does not exist: "
                f"{args.prediction_keep_path}"
            )
        requested_prediction_ids = read_keep_ids(args.prediction_keep_path)
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
        component_variant_indices=component_variant_indices or None,
        standardization_overrides=standardization_overrides,
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
            call_plan=prediction_fitter._multi_call_plan,
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run sparse REML + LASSO pipeline on real genotype data.")
    # Genotype input — exactly one of the two groups must be supplied
    p.add_argument("--bed-prefix", default=env("BED_PREFIX", ""),
                   help="PLINK1 BED file prefix (no extension); comma-separated for multiple GRMs")
    p.add_argument("--pgen-prefix", default=env("PGEN_PREFIX", ""),
                   help="PLINK2 PGEN file prefix (direct read, no conversion needed)")
    p.add_argument(
        "--component-spec",
        default=env("COMPONENT_SPEC", ""),
        help="Structured component spec (.json or .npz) defining SNP-ID/index GRM partitions.",
    )
    p.add_argument(
        "--variance-components-init",
        default="",
        help=(
            "Optional JSON array with one initial value per GRM followed by "
            "the residual variance. Used by the Adaptive COHERIT warm start."
        ),
    )
    p.add_argument(
        "--sparse-state-in",
        default="",
        help=(
            "Optional NPZ sparse numerical state from a covariance-preserving "
            "parent layer. Source-marker coordinates are remapped to this partition."
        ),
    )
    p.add_argument(
        "--sparse-state-out",
        default="",
        help="Optional NPZ output used to warm-start the next CovTree layer.",
    )
    p.add_argument(
        "--marker-score-out",
        default="",
        help=(
            "Optional .npz output for covariance-standardized residual marker "
            "scores used by Adaptive COHERIT."
        ),
    )
    p.add_argument(
        "--marker-score-probes",
        type=int,
        default=32,
        help="Rademacher probes for the marker-information diagonal estimate.",
    )
    p.add_argument(
        "--marker-score-seed",
        type=int,
        default=0,
        help="Deterministic Rademacher seed for --marker-score-out.",
    )
    p.add_argument(
        "--marker-score-min-validation-r2",
        type=float,
        default=None,
        help=(
            "Optional Adaptive-K optimization: emit marker scores only when "
            "the final validation-selected model reaches this R2. A declining "
            "K layer can then stop without paying for unused score probes."
        ),
    )
    p.add_argument(
        "--covtree-diagnostic-out",
        default="",
        help=(
            "Optional JSON output for nuisance-adjusted REML covariance split "
            "tests under the converged fixed-K model."
        ),
    )
    p.add_argument(
        "--covtree-split-spec-out",
        default="",
        help=(
            "Optional NPZ component spec for the single split accepted by "
            "--covtree-diagnostic-out. No file is written when the layer stops."
        ),
    )
    p.add_argument(
        "--covtree-ld-score",
        default="",
        help="BIM-aligned TSV containing ID and individual-marker ld_score.",
    )
    p.add_argument("--covtree-bootstrap-draws", type=int, default=199)
    p.add_argument("--covtree-bootstrap-seed", type=int, default=0)
    p.add_argument("--covtree-alpha", type=float, default=0.05)
    p.add_argument("--covtree-rank-rtol", type=float, default=1e-7)
    p.add_argument("--covtree-min-child-markers", type=int, default=16)
    p.add_argument("--covtree-max-univariate-depth", type=int, default=2)
    p.add_argument("--covtree-parent-theta-abs-min", type=float, default=1e-6)
    p.add_argument("--covtree-parent-theta-rel-min", type=float, default=1e-4)
    p.add_argument("--pheno-txt", default=env("PHENO_TXT", ""))
    p.add_argument("--covar-txt", default=env("COVAR_TXT", ""))
    p.add_argument(
        "--prediction-bed-prefix",
        default=env("PREDICTION_BED_PREFIX", ""),
        help="Prediction BED prefix (comma-separated for multiple GRMs).",
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
    p.add_argument(
        "--compare-four-estimators",
        action="store_true",
        help=(
            "Opt in to the secondary four-estimator comparison. This runs "
            "the selected-support REML--GLS refit and emits the uncorrected "
            "Lasso plug-in, selected-span plug-in, and trace-corrected "
            "selected-span estimates in addition to the primary COHERIT "
            "estimate. By default only the COHERIT estimator and Lasso "
            "prediction branch are produced."
        ),
    )
    p.add_argument("--screen-topk", type=int, default=2000)
    p.add_argument("--candidate-k", type=int, default=256)
    p.add_argument("--vc-rel-tol", type=float, default=1e-2)
    p.add_argument(
        "--effect-rel-tol",
        type=float,
        default=1e-2,
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
        help=(
            "Frozen lambda/lambda_max ratio for the final train+validation "
            "refit. When omitted, validation phenotype/output inputs are "
            "required and validation R2 selects lambda inside every outer "
            "iteration."
        ),
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
            "Lasso lambda path. Never supply the held-out test phenotype."
        ),
    )
    p.add_argument(
        "--sparsity-validation-out",
        default="",
        help=(
            "JSON audit of the lambda selected by validation R2 inside every "
            "alpha/theta outer iteration."
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
        default=20,
        help="Maximum candidate-expansion rounds used to certify global KKT optimality.",
    )
    p.add_argument(
        "--kkt-max-candidate",
        type=int,
        default=0,
        help="Optional hard cap for KKT-expanded candidate set size (0 = no explicit cap).",
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


def _variance_components_converged(
    new_v: np.ndarray,
    old_v: np.ndarray,
    *,
    rel_tol: float,
    abs_tol: float = 1e-4,
) -> tuple[bool, float]:
    """Mixed absolute/relative convergence check for variance components.

    A purely relative check is unstable when a component is close to zero.
    The returned diagnostic is the largest componentwise change divided by
    its allowed mixed-tolerance bound; values at most one pass.
    """
    new_arr = np.asarray(new_v, dtype=np.float64).reshape(-1)
    old_arr = np.asarray(old_v, dtype=np.float64).reshape(-1)
    if new_arr.shape != old_arr.shape or new_arr.size == 0:
        return False, float("inf")
    scale = max(
        float(np.sum(np.abs(new_arr))),
        float(np.sum(np.abs(old_arr))),
        1.0,
    )
    allowed = (
        float(abs_tol) * scale
        + float(rel_tol) * np.maximum(np.abs(new_arr), np.abs(old_arr))
    )
    ratio = float(np.max(np.abs(new_arr - old_arr) / allowed))
    return bool(np.isfinite(ratio) and ratio <= 1.0), ratio


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


def _accepted_reml_theta(
    fit_result,
    *,
    expected_components: int,
    stage: str,
) -> tuple[np.ndarray, str]:
    """Return a valid, converged REML state.

    ``fit_reml`` returns the last accepted parameter vector when all
    line-search candidates are downhill.  ``ll_down`` therefore represents a
    converged no-update state: the current candidate is rejected while the
    previous accepted parameter vector is retained.
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
    residual_standardized: np.ndarray,
    theta_init: np.ndarray,
    *,
    covar: np.ndarray | None,
    h2_init: float,
):
    """Profile the full nuisance design in the sparse variance-component block.

    The supplied residual may already subtract a fitted nuisance score.  This
    does not change the restricted likelihood because ``P_C C = 0``; passing
    the complete design ``C`` here is what makes the update equivalent to
    profiling the nuisance coefficients at every candidate covariance.

    Core REML standardizes its response internally.  The Lasso residual already
    has units of the globally standardized phenotype, so estimates on the
    internal unit-residual scale are mapped back before they are combined with
    sparse quadratic terms.
    """
    residual = np.asarray(residual_standardized, dtype=np.float32).reshape(-1)
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
    _, residual_scale = _phenotype_standardization_stats(residual)
    variance_scale = float(residual_scale) ** 2
    fit_result = fitter.fit_infinitesimal(
        jnp.asarray(residual, dtype=jnp.float32),
        jnp.asarray(nuisance_design, dtype=jnp.float32),
        h2_init=float(h2_init),
        var_components_init=jnp.asarray(
            theta / variance_scale, dtype=jnp.float32
        ),
    )
    fit_result.var_components = (
        jnp.asarray(fit_result.var_components) * variance_scale
    )
    if fit_result.rep_var_components is not None:
        fit_result.rep_var_components = (
            jnp.asarray(fit_result.rep_var_components) * variance_scale
        )
    if fit_result.monte_carlo_se_var is not None:
        fit_result.monte_carlo_se_var = (
            jnp.asarray(fit_result.monte_carlo_se_var) * variance_scale
        )
    if fit_result.final_grad is not None:
        fit_result.final_grad = (
            jnp.asarray(fit_result.final_grad) / variance_scale
        )
    if fit_result.final_ai is not None:
        fit_result.final_ai = (
            jnp.asarray(fit_result.final_ai) / (variance_scale**2)
        )
    if fit_result.diagnostics is not None:
        diagnostics = dict(fit_result.diagnostics)
        if diagnostics.get("theta") is not None:
            diagnostics["theta"] = (
                jnp.asarray(diagnostics["theta"]) * variance_scale
            )
        if diagnostics.get("grad") is not None:
            diagnostics["grad"] = (
                jnp.asarray(diagnostics["grad"]) / variance_scale
            )
        if diagnostics.get("ai") is not None:
            diagnostics["ai"] = (
                jnp.asarray(diagnostics["ai"]) / (variance_scale**2)
            )
        fit_result.diagnostics = diagnostics

    history = []
    for source_row in fit_result.history:
        row = dict(source_row)
        if row.get("params") is not None:
            row["params"] = (
                np.asarray(row["params"], dtype=np.float64) * variance_scale
            ).tolist()
        if row.get("step_norm") is not None:
            row["step_norm"] = float(row["step_norm"]) * variance_scale
        if row.get("grad_norm") is not None:
            row["grad_norm"] = float(row["grad_norm"]) / variance_scale
        row["variance_scale_to_standardized_phenotype"] = variance_scale
        history.append(row)
    fit_result.history = history
    return fit_result


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


def _phenotype_standardization_stats(y: np.ndarray) -> tuple[float, float]:
    """Return the exact mean/scale convention used internally by ``fit_reml``."""
    _y_std, y_mean, y_scale = standardize_response(
        jnp.asarray(np.asarray(y, dtype=np.float32).reshape(-1), dtype=jnp.float32)
    )
    mean_host, scale_host = jax.device_get((y_mean, y_scale))
    return float(mean_host), float(scale_host)


def _quadratic_variance_to_reml_scale(q_raw: float, y_scale: float) -> float:
    """Convert a phenotype-variance quantity to fit_reml's standardized-y scale."""
    scale = float(y_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("y_scale must be positive and finite.")
    return float(q_raw) / (scale * scale)


def _sparse_dense_h2(
    q_sparse_standardized: float,
    background_genetic_variance: float,
    residual_variance: float,
) -> float:
    """Combine sparse, dense-background, and residual variance on one scale."""
    genetic = float(q_sparse_standardized) + float(background_genetic_variance)
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


def _finite_float_or_none(value: float) -> float | None:
    """Return a JSON-safe finite scalar, or ``None`` when unavailable."""
    value_f = float(value)
    return value_f if np.isfinite(value_f) else None


def _validation_allows_marker_score(
    observed_r2: float | None,
    minimum_r2: float | None,
) -> bool:
    """Skip score probes only when Adaptive K will stop at this layer."""
    if minimum_r2 is None:
        return True
    if observed_r2 is None:
        raise ValueError("Conditional marker scoring requires validation R2.")
    observed = float(observed_r2)
    minimum = float(minimum_r2)
    if not np.isfinite(observed) or not np.isfinite(minimum):
        raise ValueError("Marker-score validation thresholds must be finite.")
    return observed >= minimum


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
    """Select validation R2 only among converged, candidate-KKT path points."""
    if len(path_rows) != len(metrics) or not path_rows:
        raise ValueError("Lasso path and validation metrics must align.")
    eligible = [
        index
        for index, (row, metric) in enumerate(zip(path_rows, metrics))
        if bool(row.get("converged", False))
        and bool(row.get("kkt_passed", False))
        and metric.get("correlation_squared") is not None
        and np.isfinite(float(metric["correlation_squared"]))
    ]
    if not eligible:
        raise RuntimeError(
            "No converged candidate-KKT Lasso path point has finite validation R2."
        )
    return max(
        eligible,
        key=lambda index: (
            float(metrics[index]["correlation_squared"]),
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
    ):
        raise RuntimeError(
            "Validation-selected Lasso path point is not candidate-KKT converged."
        )
    beta_snp = beta_snp_path[index].copy()
    beta_cov = beta_cov_path[index].copy()
    active_idx = np.flatnonzero(beta_snp != 0.0).astype(np.int64)
    selected_metric = dict(metrics[index])
    merged_path = merge_path_diagnostics(path_rows, metrics)
    selection_record = {
        "selection_metric": (
            "squared_pearson_correlation_total_phenotype_prediction"
        ),
        "selected": {
            "path_index": index,
            "lam": float(selected_row["lam"]),
            "lam_ratio": float(selected_row["lam_ratio"]),
            "support_size": int(active_idx.size),
            **selected_metric,
        },
        "path_prediction_pcg_reported_res": float(path_prediction_pcg_res),
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
            "selection_method": "validation_r2",
            "selected_lam_ratio": float(selected_row["lam_ratio"]),
            "validation_selection": selection_record,
        }
    )
    return selected, selection_record


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
    theta_standardized: np.ndarray,
    phenotype_scale: float,
    validation_outcome: np.ndarray,
) -> tuple[dict, dict]:
    """Evaluate the complete path and return the validation-selected alpha."""
    validation_candidate = (
        prediction_context.grm_index.extract_standardized_columns(candidate)
        .astype(np.float32, copy=False)
    )
    path_prediction = predict_sparse_path_partitioned(
        fitter=fitter,
        test_fitter=prediction_context.fitter,
        y_train_raw=y_train,
        train_covar=train_covar,
        test_covar=prediction_context.covar,
        train_candidate_geno=train_candidate,
        test_candidate_geno=validation_candidate,
        beta_cov_path_raw=lasso_path["beta_cov_path"],
        beta_candidate_path_raw=lasso_path["beta_snp_path"],
        theta_standardized=theta_standardized,
        phenotype_scale=float(phenotype_scale),
        pcg_tol=float(args.pcg_tol),
        max_pcg_iters=int(args.max_pcg_iters),
    )
    metrics = evaluate_prediction_path(
        path_prediction.phenotype_prediction_raw,
        validation_outcome,
    )
    selected_index = _select_converged_validation_path_index(
        list(lasso_path["path"]), metrics
    )
    return _materialize_validation_selected_lasso(
        lasso_path,
        metrics,
        selected_index=selected_index,
        path_prediction_pcg_res=float(path_prediction.pcg_rel_res),
        path_prediction_pcg_iters=int(path_prediction.pcg_iters),
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
    theta_standardized: np.ndarray,
    phenotype_scale: float,
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
        "schema_version": 2,
        "selection_role": "inside_every_alpha_theta_outer_iteration",
        "selection_metric": (
            "squared_pearson_correlation_total_phenotype_prediction"
        ),
        "test_phenotype_used": False,
        "validation_phenotype_path": os.path.abspath(phenotype_path),
        "n_validation_samples": int(
            np.asarray(validation_outcome).reshape(-1).size
        ),
        "phenotype_scale": float(phenotype_scale),
        "outer_converged": bool(outer_converged),
        "outer_stop_reason": str(outer_stop_reason),
        "theta_standardized_final": np.asarray(
            theta_standardized, dtype=np.float64
        ).tolist(),
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
        "validation_correlation_squared": float(
            selected["correlation_squared"]
        ),
        "n_path_selections": int(len(selection_trace)),
        "outputs": output_paths,
    }


def _expand_selected_basis_coefficients(
    *,
    support_size: int,
    basis_positions: np.ndarray,
    basis_coefficients: np.ndarray,
) -> np.ndarray:
    """Map one full-rank selected-span coefficient back to the full support."""
    size = int(support_size)
    positions = np.asarray(basis_positions, dtype=np.int64).reshape(-1)
    coefficients = np.asarray(
        basis_coefficients, dtype=np.float64
    ).reshape(-1)
    if size < 0 or positions.size != coefficients.size:
        raise ValueError("Selected-span basis coefficient shape mismatch.")
    if (
        positions.size > 0
        and (
            np.any(positions < 0)
            or np.any(positions >= size)
            or np.unique(positions).size != positions.size
        )
    ):
        raise ValueError("Selected-span basis positions are invalid.")
    expanded = np.zeros(size, dtype=np.float64)
    expanded[positions] = coefficients
    return expanded


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


def _four_estimator_h2_from_branches(
    *,
    q_lasso_plugin_standardized: float,
    q_lasso_calibrated_standardized: float,
    q_selected_span_plugin_standardized: float,
    q_selected_span_trace_standardized: float,
    lasso_ml_background_variance: float,
    lasso_ml_residual_variance: float,
    selected_span_reml_background_variance: float,
    selected_span_reml_residual_variance: float,
) -> dict[str, float]:
    """Map the four sparse quadratics to their two variance branches."""
    return {
        "h2_lasso_plugin": _sparse_dense_h2(
            q_lasso_plugin_standardized,
            lasso_ml_background_variance,
            lasso_ml_residual_variance,
        ),
        "h2_chive": _sparse_dense_h2(
            q_lasso_calibrated_standardized,
            lasso_ml_background_variance,
            lasso_ml_residual_variance,
        ),
        "h2_ss_gls_plugin": _sparse_dense_h2(
            q_selected_span_plugin_standardized,
            selected_span_reml_background_variance,
            selected_span_reml_residual_variance,
        ),
        "h2_ss_gls_df_corrected": _sparse_dense_h2(
            q_selected_span_trace_standardized,
            selected_span_reml_background_variance,
            selected_span_reml_residual_variance,
        ),
    }


def _sparse_estimator_branch_guards(
    *,
    comparison_enabled: bool,
    alpha_theta_pair_certified: bool,
    lasso_quadratics_available: bool,
    selected_span_refit_ok: bool,
    lasso_estimator_values: np.ndarray,
    selected_support_estimator_values: np.ndarray,
) -> dict[str, object]:
    """Validate the Lasso and selected-support estimator branches separately.

    The primary COHERIT branch is validated independently.  When comparison
    mode is enabled, failure of the downstream selected-support REML--GLS
    refit must not erase a valid COHERIT estimate.  No ordinary-REML value is
    substituted into either branch.
    """
    lasso_values = np.asarray(
        lasso_estimator_values, dtype=np.float64
    ).reshape(-1)
    selected_values = np.asarray(
        selected_support_estimator_values, dtype=np.float64
    ).reshape(-1)
    expected_lasso_outputs = 2 if comparison_enabled else 1
    lasso_outputs_finite = bool(
        lasso_values.size == expected_lasso_outputs
        and np.all(np.isfinite(lasso_values))
    )
    selected_outputs_finite = bool(
        selected_values.size == 2 and np.all(np.isfinite(selected_values))
    )

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

    selected_reasons: list[str] = []
    selected_support_branch_valid = False
    if comparison_enabled:
        if not lasso_branch_valid:
            selected_reasons.append("lasso_support_branch_not_valid")
        if not selected_span_refit_ok:
            selected_reasons.append("selected_support_reml_gls_unavailable")
        if not selected_outputs_finite:
            selected_reasons.append("nonfinite_selected_support_estimator")
        selected_support_branch_valid = bool(
            lasso_branch_valid
            and selected_span_refit_ok
            and selected_outputs_finite
        )
    all_four_valid = bool(
        comparison_enabled
        and lasso_branch_valid
        and selected_support_branch_valid
    )
    all_requested_valid = bool(
        lasso_branch_valid
        and (not comparison_enabled or selected_support_branch_valid)
    )
    all_requested_outputs_finite = bool(
        lasso_outputs_finite
        and (not comparison_enabled or selected_outputs_finite)
    )
    combined_reasons = list(lasso_reasons) + list(selected_reasons)
    return {
        "lasso_branch_valid": lasso_branch_valid,
        "lasso_outputs_finite": lasso_outputs_finite,
        "lasso_branch_invalid_reasons": lasso_reasons,
        "selected_support_refit_branch_valid": (
            selected_support_branch_valid
        ),
        "selected_support_outputs_finite": selected_outputs_finite,
        "selected_support_refit_branch_invalid_reasons": selected_reasons,
        "all_four_estimators_valid": all_four_valid,
        "all_four_outputs_finite": bool(
            comparison_enabled
            and lasso_outputs_finite
            and selected_outputs_finite
        ),
        "all_requested_estimators_valid": all_requested_valid,
        "all_requested_outputs_finite": all_requested_outputs_finite,
        "combined_invalid_reasons": combined_reasons,
    }


def _sparse_prediction_branch_names(
    *,
    comparison_enabled: bool,
    lasso_branch_valid: bool,
    selected_support_refit_branch_valid: bool,
) -> list[str]:
    """Return independently available prediction branches in output order."""
    if (
        comparison_enabled
        and selected_support_refit_branch_valid
        and not lasso_branch_valid
    ):
        raise ValueError(
            "A selected-support prediction requires a valid Lasso branch."
        )
    names = ["lasso"] if lasso_branch_valid else []
    if comparison_enabled and selected_support_refit_branch_valid:
        names.append("selected_span")
    return names


def _sparse_output_contract(comparison_enabled: bool) -> dict[str, object]:
    """Return the mode-labelled sparse output contract for one run."""
    if comparison_enabled:
        return {
            "sparse_output_schema_version": 6,
            "estimator_mode": "four_estimator_comparison",
            "computed_estimators": [
                "h2_lasso_plugin",
                "h2_chive",
                "h2_ss_gls_plugin",
                "h2_ss_gls_df_corrected",
            ],
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
                "beta_gls_reml",
                "selected_span_basis",
            ],
        }
    return {
        "sparse_output_schema_version": 6,
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


def _selected_span_gls_quadratics(
    *,
    y: np.ndarray,
    covar: np.ndarray | None,
    z_active: np.ndarray,
    Hinv_y: np.ndarray,
    Hinv_covar: np.ndarray | None,
    Hinv_z_active: np.ndarray,
    phenotype_scale: float,
) -> dict[str, object]:
    """Construct raw and trace-corrected quadratics after selected-span GLS.

    The covariance operator represented by the ``Hinv_*`` arguments is on the
    internally standardized phenotype scale.  GLS coefficients and the
    squared fitted score are returned on the raw phenotype scale; the
    fixed-span estimation-noise trace correction is reported on both scales.
    Output field names use ``df`` for this analytic trace correction.
    """
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    z_arr = np.asarray(z_active, dtype=np.float64)
    hy = np.asarray(Hinv_y, dtype=np.float64).reshape(-1)
    hz = np.asarray(Hinv_z_active, dtype=np.float64)
    if (
        z_arr.ndim != 2
        or hz.shape != z_arr.shape
        or z_arr.shape[0] != y_arr.size
        or hy.size != y_arr.size
    ):
        raise ValueError("Selected-span GLS inputs have incompatible shapes.")

    merged, active_basis_idx = _merge_independent_fixed_effects(covar, z_arr)
    if covar is None:
        c_arr = np.empty((y_arr.size, 0), dtype=np.float64)
        hc = np.empty((y_arr.size, 0), dtype=np.float64)
    else:
        c_arr = np.asarray(covar, dtype=np.float64)
        hc = np.asarray(Hinv_covar, dtype=np.float64)
        if c_arr.ndim != 2 or hc.shape != c_arr.shape:
            raise ValueError("Covariate GLS inputs have incompatible shapes.")

    z_basis = z_arr[:, active_basis_idx]
    hz_basis = hz[:, active_basis_idx]
    hfixed = np.concatenate([hc, hz_basis], axis=1)
    fixed = np.asarray(merged, dtype=np.float64)
    if fixed.shape != hfixed.shape:
        raise RuntimeError("Selected-span basis and inverse-covariance image disagree.")

    if fixed.shape[1] == 0:
        return {
            "active_basis_idx": active_basis_idx,
            "beta_cov": np.empty((0,), dtype=np.float64),
            "beta_active_basis": np.empty((0,), dtype=np.float64),
            "q_plugin_raw": 0.0,
            "q_plugin_standardized": 0.0,
            "df_correction_raw": 0.0,
            "df_correction_standardized": 0.0,
            "q_df_corrected_raw": 0.0,
            "q_df_corrected_standardized": 0.0,
        }

    gram = fixed.T @ hfixed
    gram = 0.5 * (gram + gram.T)
    try:
        gram_inv = sla.inv(gram, check_finite=False)
    except np.linalg.LinAlgError:
        gram_inv = sla.pinvh(gram, rtol=1e-10, check_finite=False)
    gram_inv = 0.5 * (gram_inv + gram_inv.T)
    coef = gram_inv @ (fixed.T @ hy)
    p_c = int(c_arr.shape[1])
    beta_cov = np.asarray(coef[:p_c], dtype=np.float64)
    beta_active = np.asarray(coef[p_c:], dtype=np.float64)
    fitted_sparse = z_basis @ beta_active
    n = float(y_arr.size)
    q_plugin_raw = float(fitted_sparse @ fitted_sparse / n)

    sparse_gram = z_basis.T @ z_basis / n
    covariance_active_standardized = gram_inv[p_c:, p_c:]
    df_standardized = float(
        np.trace(sparse_gram @ covariance_active_standardized)
    )
    scale = float(phenotype_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("phenotype_scale must be positive and finite.")
    scale_sq = scale * scale
    q_plugin_standardized = q_plugin_raw / scale_sq
    df_raw = df_standardized * scale_sq

    return {
        "active_basis_idx": active_basis_idx,
        "beta_cov": beta_cov,
        "beta_active_basis": beta_active,
        "q_plugin_raw": q_plugin_raw,
        "q_plugin_standardized": q_plugin_standardized,
        "df_correction_raw": df_raw,
        "df_correction_standardized": df_standardized,
        "q_df_corrected_raw": q_plugin_raw - df_raw,
        "q_df_corrected_standardized": q_plugin_standardized - df_standardized,
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


def _partitioned_lasso_kkt_from_scores(
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


def _allow_monotone_lasso_path_warm_start(
    *,
    previous_size: int,
    current_size: int,
    common_size: int,
) -> bool:
    """Reuse a mapped path whenever the old marker basis is fully retained.

    The mapped coefficients are only initial values.  ``solve_lasso_path``
    compares them with the ordinary descending-lambda start row by row and
    still requires the same complete score-KKT certificate, so candidate-set
    growth does not need an arbitrary size cutoff.
    """
    previous = int(previous_size)
    current = int(current_size)
    common = int(common_size)
    return bool(previous >= 1 and current >= previous and common == previous)


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
    if theta.shape != (int(n_grm) + 1,):
        raise ValueError(
            "--variance-components-init must contain one value per GRM "
            f"followed by residual variance; expected {int(n_grm) + 1}, "
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


def _compute_adaptive_marker_scores(
    *,
    output_path: str,
    residual_raw: np.ndarray,
    covar: np.ndarray | None,
    phenotype_scale: float,
    fitter,
    ops,
    grm_index: MultiGRMIndex,
    theta: np.ndarray,
    n_probes: int,
    seed: int,
    pcg_tol: float,
    max_pcg_iters: int,
) -> dict[str, object]:
    """Write legacy association and signed REML covariance marker scores.

    Both information diagonals are estimated without materializing X or V.
    For a sample-space Rademacher probe q,
    E[(X' q) * (X' V^-1 q)] = diag(X' V^-1 X).
    Replacing ``V^-1 q`` by ``P q`` gives the fixed-effect-adjusted quantity
    needed by ``u_j = 0.5 * ((x_j' P e)^2 - x_j' P x_j)``.
    """
    if int(n_probes) < 1:
        raise ValueError("marker-score-probes must be >= 1.")
    if not output_path.lower().endswith(".npz"):
        raise ValueError("--marker-score-out must end in .npz.")
    residual = np.asarray(residual_raw, dtype=np.float32).reshape(-1)
    n_samples = int(fitter.streamers[0].n)
    if residual.shape != (n_samples,) or not np.all(np.isfinite(residual)):
        raise ValueError("Adaptive marker-score residual is malformed.")
    scale = float(phenotype_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("phenotype_scale must be finite and positive.")
    if covar is None:
        fixed_design = np.empty((n_samples, 0), dtype=np.float32)
    else:
        fixed_design = np.asarray(covar, dtype=np.float32)
        if fixed_design.ndim != 2 or fixed_design.shape[0] != n_samples:
            raise ValueError("Adaptive marker-score covariates are malformed.")

    theta_values = np.asarray(theta, dtype=np.float64).reshape(-1)
    if theta_values.shape != (grm_index.n_grm + 1,):
        raise ValueError("Adaptive marker-score theta has the wrong length.")
    theta_g = jnp.asarray(theta_values[:-1], dtype=jnp.float32)
    theta_e = jnp.asarray(theta_values[-1], dtype=jnp.float32)
    hv = fitter._make_hv(ops, theta_g, theta_e)
    precond = fitter._make_effect_precond(ops, theta_g, theta_e)

    rng = np.random.default_rng(int(seed))
    probes = rng.integers(
        0,
        2,
        size=(n_samples, int(n_probes)),
        dtype=np.int8,
    ).astype(np.float32)
    probes *= 2.0
    probes -= 1.0
    rhs = np.concatenate(
        [residual[:, None], probes, fixed_design], axis=1
    ).astype(
        np.float32,
        copy=False,
    )
    rhs_dev = jnp.asarray(rhs, dtype=jnp.float32)
    solution, reported_residual, iterations = pcg_solve(
        hv,
        rhs_dev,
        M=precond,
        tol=float(pcg_tol),
        maxiter=int(max_pcg_iters),
    )
    reported = _require_pcg_converged(
        reported_residual,
        tol=float(pcg_tol),
        iters=iterations,
        maxiter=int(max_pcg_iters),
        stage="adaptive marker score",
    )
    true_residual = _true_pcg_relative_residual(hv, rhs_dev, solution)
    if not np.isfinite(true_residual):
        raise RuntimeError("Adaptive marker-score PCG true residual is non-finite.")

    solution_np = np.asarray(jax.device_get(solution), dtype=np.float32)
    core_solution = solution_np[:, : int(n_probes) + 1]
    if fixed_design.shape[1] > 0:
        vinv_c = solution_np[:, int(n_probes) + 1 :]
        covar_gram = (
            fixed_design.astype(np.float64).T @ vinv_c.astype(np.float64)
        )
        covar_gram_inverse = np.linalg.pinv(
            0.5 * (covar_gram + covar_gram.T),
            rcond=1e-10,
            hermitian=True,
        )
        projection_coefficients = covar_gram_inverse @ (
            fixed_design.astype(np.float64).T
            @ core_solution.astype(np.float64)
        )
        projected_solution = core_solution - vinv_c @ projection_coefficients.astype(
            np.float32
        )
    else:
        projected_solution = core_solution

    projected_solution[:, 0] /= np.float32(scale)
    xt_solution = np.asarray(
        grm_index.xtv_all(
            jnp.asarray(core_solution, dtype=jnp.float32), normalize=False
        ),
        dtype=np.float64,
    )
    xt_projected_solution = np.asarray(
        grm_index.xtv_all(
            jnp.asarray(projected_solution, dtype=jnp.float32),
            normalize=False,
        ),
        dtype=np.float64,
    )
    xt_probe = np.asarray(
        grm_index.xtv_all(
            jnp.asarray(probes, dtype=jnp.float32),
            normalize=False,
        ),
        dtype=np.float64,
    )
    expected_shape = (grm_index.m_total, int(n_probes) + 1)
    if xt_solution.shape != expected_shape:
        raise RuntimeError(
            "Adaptive marker-score X'V^-1 RHS has the wrong shape: "
            f"{xt_solution.shape} != {expected_shape}."
        )
    if xt_probe.shape != (grm_index.m_total, int(n_probes)):
        raise RuntimeError("Adaptive marker-score X'probe has the wrong shape.")
    if xt_projected_solution.shape != expected_shape:
        raise RuntimeError(
            "Adaptive marker-score X'P RHS has the wrong shape: "
            f"{xt_projected_solution.shape} != {expected_shape}."
        )

    numerator = xt_solution[:, 0]
    information = np.mean(
        xt_probe * xt_solution[:, 1:],
        axis=1,
        dtype=np.float64,
    )
    positive = information[np.isfinite(information) & (information > 0.0)]
    if positive.size == 0:
        raise RuntimeError(
            "Hutchinson marker-information estimate has no positive entries."
        )
    information_floor = max(
        float(np.median(positive)) * 1e-6,
        float(np.finfo(np.float32).tiny),
    )
    clipped = ~np.isfinite(information) | (information <= information_floor)
    information_safe = np.where(clipped, information_floor, information)
    signal_score = np.abs(numerator) / np.sqrt(information_safe)
    if not np.all(np.isfinite(signal_score)):
        raise RuntimeError("Adaptive marker score contains non-finite values.")

    projected_numerator = xt_projected_solution[:, 0]
    projected_information = np.mean(
        xt_probe * xt_projected_solution[:, 1:],
        axis=1,
        dtype=np.float64,
    )
    projected_positive = projected_information[
        np.isfinite(projected_information) & (projected_information > 0.0)
    ]
    if projected_positive.size == 0:
        raise RuntimeError(
            "Hutchinson projected marker-information estimate has no positive entries."
        )
    projected_information_floor = max(
        float(np.median(projected_positive)) * 1e-6,
        float(np.finfo(np.float32).tiny),
    )
    projected_clipped = (
        ~np.isfinite(projected_information)
        | (projected_information <= projected_information_floor)
    )
    projected_information_safe = np.where(
        projected_clipped,
        projected_information_floor,
        projected_information,
    )
    covariance_score = 0.5 * (
        np.square(projected_numerator) - projected_information_safe
    )
    if not np.all(np.isfinite(covariance_score)):
        raise RuntimeError("Signed covariance marker score contains non-finite values.")

    global_indices = np.arange(grm_index.m_total, dtype=np.int64)
    source_indices = grm_index.source_variant_indices(global_indices)
    source_order = np.argsort(source_indices, kind="stable")
    if not np.array_equal(
        source_indices[source_order],
        np.arange(grm_index.m_total, dtype=np.int64),
    ):
        raise RuntimeError(
            "Adaptive marker scores require a one-to-one source variant order."
        )
    component_global = np.empty(grm_index.m_total, dtype=np.int32)
    for component_index in range(grm_index.n_grm):
        component_global[
            int(grm_index.offsets[component_index]) :
            int(grm_index.offsets[component_index + 1])
        ] = int(component_index)

    ensure_parent_dir(output_path)
    temporary = f"{output_path}.tmp.{os.getpid()}"
    with open(temporary, "wb") as handle:
        np.savez_compressed(
            handle,
            source_variant_index=source_indices[source_order],
            parent_component_index=component_global[source_order],
            signal_score=signal_score[source_order].astype(np.float32),
            score_numerator=numerator[source_order].astype(np.float32),
            information_diagonal=information_safe[source_order].astype(
                np.float32
            ),
            covariance_score=covariance_score[source_order].astype(np.float32),
            projected_score_numerator=projected_numerator[source_order].astype(
                np.float32
            ),
            projected_information_diagonal=projected_information_safe[
                source_order
            ].astype(np.float32),
        )
    os.replace(temporary, output_path)
    return {
        "requested": True,
        "status": "emitted",
        "path": os.path.abspath(output_path),
        "definition": "abs(x_t_Vinv_residual)/sqrt(x_t_Vinv_x)",
        "covariance_score_definition": (
            "0.5*((x_t_P_residual_standardized)^2-x_t_P_x)"
        ),
        "fixed_effect_projection": "P_includes_all_unpenalized_covariates",
        "residual_scale_for_covariance_score": "standardized_phenotype",
        "information_diagonal_estimator": (
            "sample_space_rademacher_hutchinson"
        ),
        "n_markers": int(grm_index.m_total),
        "n_probes": int(n_probes),
        "seed": int(seed),
        "information_floor": float(information_floor),
        "information_clipped_count": int(np.count_nonzero(clipped)),
        "information_clipped_fraction": float(np.mean(clipped)),
        "projected_information_floor": float(projected_information_floor),
        "projected_information_clipped_count": int(
            np.count_nonzero(projected_clipped)
        ),
        "projected_information_clipped_fraction": float(
            np.mean(projected_clipped)
        ),
        "pcg_reported_relative_residual": float(reported),
        "pcg_true_relative_residual": float(true_residual),
        "pcg_iterations": int(iterations),
    }


def _write_covtree_bootstrap_marker_scores(
    *,
    output_path: str,
    grm_index: MultiGRMIndex,
    marker_scores: dict[str, np.ndarray],
    bootstrap_draws: int,
    seed: int,
) -> tuple[dict[str, object], np.ndarray]:
    """Write CovTree's already-computed marker diagnostics in source order."""
    required = {
        "covariance_score",
        "projected_score_numerator",
        "projected_information_diagonal",
    }
    if not required.issubset(marker_scores):
        raise ValueError("CovTree bootstrap marker-score payload is incomplete.")
    cache_arrays = {
        name: np.asarray(marker_scores[name], dtype=np.float32).reshape(-1)
        for name in required
    }
    if any(value.shape != (grm_index.m_total,) for value in cache_arrays.values()):
        raise ValueError("CovTree bootstrap marker-score length mismatch.")
    if not all(np.all(np.isfinite(value)) for value in cache_arrays.values()):
        raise ValueError("CovTree bootstrap marker scores contain non-finite values.")

    cache_indices = np.arange(grm_index.m_total, dtype=np.int64)
    cache_to_source = grm_index.source_variant_indices(cache_indices)
    source_order = np.argsort(cache_to_source, kind="stable")
    if not np.array_equal(
        cache_to_source[source_order],
        np.arange(grm_index.m_total, dtype=np.int64),
    ):
        raise RuntimeError("CovTree marker scores require a one-to-one source order.")
    component_cache = np.empty(grm_index.m_total, dtype=np.int32)
    for component_index in range(grm_index.n_grm):
        component_cache[
            int(grm_index.offsets[component_index]) :
            int(grm_index.offsets[component_index + 1])
        ] = int(component_index)

    ensure_parent_dir(output_path)
    temporary = f"{output_path}.tmp.{os.getpid()}"
    with open(temporary, "wb") as handle:
        np.savez_compressed(
            handle,
            source_variant_index=cache_to_source[source_order],
            parent_component_index=component_cache[source_order],
            covariance_score=cache_arrays["covariance_score"][source_order],
            projected_score_numerator=cache_arrays[
                "projected_score_numerator"
            ][source_order],
            projected_information_diagonal=cache_arrays[
                "projected_information_diagonal"
            ][source_order],
        )
    os.replace(temporary, output_path)
    covariance_source = cache_arrays["covariance_score"][source_order]
    return (
        {
            "requested": True,
            "status": "emitted",
            "path": os.path.abspath(output_path),
            "definition": "0.5*((x_t_P_e)^2-x_t_P_x)",
            "fixed_effect_projection": "P_includes_all_unpenalized_covariates",
            "residual_scale": "standardized_phenotype",
            "information_diagonal_estimator": (
                "fitted_null_parametric_bootstrap_mean_square"
            ),
            "quadratic_backend": "reused_covtree_Xt_Pe_bootstrap_pass",
            "n_markers": int(grm_index.m_total),
            "bootstrap_draws": int(bootstrap_draws),
            "seed": int(seed),
        },
        covariance_source,
    )


def _run_covtree_diagnostic(
    *,
    args: argparse.Namespace,
    fitter,
    ops,
    grm_index: MultiGRMIndex,
    component_spec_path: str,
    bed_prefix: str,
    marker_score_path: str,
    residual_raw: np.ndarray,
    covar: np.ndarray | None,
    phenotype_scale: float,
    theta: np.ndarray,
) -> dict[str, object]:
    """Test all genotype-only splits and optionally emit one accepted spec."""
    components = _covtree_components_from_spec(component_spec_path)
    ld_score = _load_ld_score_in_bim_order(
        args.covtree_ld_score,
        bed_prefix + ".bim",
    )
    if ld_score.shape != (grm_index.m_total,):
        raise RuntimeError("CovTree LD-score length does not match the genotype panel.")
    streamer = fitter.streamers[0]
    means_cache = np.asarray(streamer._means_host, dtype=np.float64)
    cache_indices = np.arange(grm_index.m_total, dtype=np.int64)
    cache_to_source = grm_index.source_variant_indices(cache_indices)
    means_source = np.empty(grm_index.m_total, dtype=np.float64)
    means_source[cache_to_source] = means_cache
    allele_frequency = np.clip(0.5 * means_source, 0.0, 1.0)
    heterozygosity = 2.0 * allele_frequency * (1.0 - allele_frequency)
    positive_heterozygosity = heterozygosity[heterozygosity > 0.0]
    if positive_heterozygosity.size == 0:
        raise RuntimeError("CovTree could not estimate any positive heterozygosity.")
    heterozygosity_floor = max(
        float(np.min(positive_heterozygosity)) * 0.5,
        float(np.finfo(np.float32).tiny),
    )
    heterozygosity_floored = heterozygosity <= 0.0
    heterozygosity = np.where(
        heterozygosity_floored, heterozygosity_floor, heterozygosity
    )

    candidates, candidate_rejections = generate_covtree_candidates(
        components,
        ld_score=ld_score,
        heterozygosity=heterozygosity,
        min_child_markers=int(args.covtree_min_child_markers),
        max_univariate_depth=int(
            getattr(args, "covtree_max_univariate_depth", 2)
        ),
    )
    theta_values = np.asarray(theta, dtype=np.float64).reshape(-1)
    parent_threshold = max(
        float(args.covtree_parent_theta_abs_min),
        float(args.covtree_parent_theta_rel_min)
        * max(float(np.sum(theta_values[:-1])), np.finfo(float).tiny),
    )
    eligible_candidates = []
    bootstrap_rank_capacity = max(
        int(args.covtree_bootstrap_draws) - 1 - (len(components) + 1),
        0,
    )
    for candidate in candidates:
        parent_theta = float(theta_values[candidate.parent_index])
        if candidate.degrees_of_freedom > bootstrap_rank_capacity:
            candidate_rejections.append(
                {
                    "parent_index": int(candidate.parent_index),
                    "parent_name": candidate.parent_name,
                    "split_kind": candidate.split_kind,
                    "reason": "insufficient_bootstrap_information_rank_capacity",
                    "candidate_degrees_of_freedom": int(
                        candidate.degrees_of_freedom
                    ),
                    "bootstrap_rank_capacity": bootstrap_rank_capacity,
                }
            )
        elif parent_theta <= parent_threshold:
            candidate_rejections.append(
                {
                    "parent_index": int(candidate.parent_index),
                    "parent_name": candidate.parent_name,
                    "split_kind": candidate.split_kind,
                    "reason": "parent_variance_at_boundary",
                    "parent_theta": parent_theta,
                    "threshold": parent_threshold,
                }
            )
        else:
            eligible_candidates.append(candidate)

    residual_standardized = np.asarray(residual_raw, dtype=np.float32) / np.float32(
        phenotype_scale
    )
    selected_candidate = None
    marker_score_summary: dict[str, object] = {
        "requested": True,
        "status": "not_emitted_no_eligible_candidate",
        "path": None,
    }
    if eligible_candidates:
        inference, selected_candidate, bootstrap_marker_scores = (
            evaluate_covtree_candidates(
                fitter=fitter,
                ops=ops,
                grm_index=grm_index,
                candidates=eligible_candidates,
                theta=theta_values,
                covar=covar,
                residual_standardized=residual_standardized,
                bootstrap_draws=int(args.covtree_bootstrap_draws),
                seed=int(args.covtree_bootstrap_seed),
                alpha=float(args.covtree_alpha),
                rank_rtol=float(args.covtree_rank_rtol),
                pcg_tol=float(args.pcg_tol),
                max_pcg_iters=int(args.max_pcg_iters),
            )
        )
        marker_score_summary, signed_marker = _write_covtree_bootstrap_marker_scores(
            output_path=marker_score_path,
            grm_index=grm_index,
            marker_scores=bootstrap_marker_scores,
            bootstrap_draws=int(args.covtree_bootstrap_draws),
            seed=int(args.covtree_bootstrap_seed),
        )
        for candidate, metadata in zip(
            eligible_candidates, inference["candidates"], strict=True
        ):
            signed_child_means = np.asarray(
                [float(np.mean(signed_marker[child])) for child in candidate.children]
            )
            contrasts = np.asarray(
                metadata["contrast_coefficients"], dtype=np.float64
            )
            raw_signed_contrasts = signed_child_means @ contrasts
            metadata.update(
                {
                    "signed_marker_child_means": signed_child_means.tolist(),
                    "raw_signed_marker_contrast_scores": raw_signed_contrasts.tolist(),
                    "raw_signed_marker_contrast_norm": float(
                        np.linalg.norm(raw_signed_contrasts)
                    ),
                }
            )
    else:
        inference = {
            "method": "nuisance_adjusted_reml_score_parametric_max_bootstrap",
            "accepted": False,
            "best_candidate_index": None,
            "selected_candidate_index": None,
            "best_candidate_name": None,
            "max_score_adjusted_p": None,
            "max_score_adjusted_p_mc_se": None,
            "bootstrap_draws": 0,
            "candidate_count": 0,
            "candidate_contrast_count": 0,
            "candidates": [],
            "pcg": [],
        }

    split_spec_path = None
    warm_start = None
    warm_start_weighting = None
    if selected_candidate is not None:
        selected_metadata = inference["candidates"][
            int(inference["selected_candidate_index"])
        ]
        warm_start = selective_covariance_warm_start(
            theta_values,
            components,
            selected_candidate,
            child_effective_markers=selected_metadata[
                "child_effective_markers"
            ],
        )
        warm_start_weighting = "effective_marker_count"
        if args.covtree_split_spec_out:
            updated_components = replace_covtree_parent(
                components, selected_candidate
            )
            split_spec_path = str(
                write_component_spec(
                    args.covtree_split_spec_out,
                    updated_components,
                    provenance={
                        "algorithm": "coherit_covtree_v1",
                        "source_component_spec": os.path.abspath(
                            component_spec_path
                        ),
                        "selected_candidate": selected_candidate.name,
                        "max_score_adjusted_p": inference[
                            "max_score_adjusted_p"
                        ],
                    },
                )
            )

    diagnostic = {
        **inference,
        "status": "complete",
        "current_k": len(components),
        "next_k": (
            len(components) - 1 + len(selected_candidate.children)
            if selected_candidate is not None
            else len(components)
        ),
        "candidate_generation": (
            "within_parent_recursive_exact_2means_ld_maf_and_ld_by_maf"
        ),
        "candidate_features": {
            "ld": "log1p_individual_ld_score",
            "maf": "log_2p1mp_from_training_genotype_mean",
            "phenotype_independent": True,
        },
        "candidate_rejections": candidate_rejections,
        "parent_theta_boundary_threshold": parent_threshold,
        "bootstrap_rank_capacity": bootstrap_rank_capacity,
        "heterozygosity_floor": heterozygosity_floor,
        "heterozygosity_floored_count": int(
            np.count_nonzero(heterozygosity_floored)
        ),
        "signed_marker_score_path": marker_score_summary["path"],
        "marker_score": marker_score_summary,
        "selected_split_spec": split_spec_path,
        "covariance_preserving_warm_start": (
            warm_start.tolist() if warm_start is not None else None
        ),
        "covariance_preserving_warm_start_weighting": warm_start_weighting,
        "stopping_reason": (
            None
            if selected_candidate is not None
            else (
                "no_eligible_candidate"
                if not eligible_candidates
                else "max_score_not_significant"
            )
        ),
    }
    output_path = args.covtree_diagnostic_out
    ensure_parent_dir(output_path)
    temporary = f"{output_path}.tmp.{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(
            _json_safe_value(diagnostic),
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
    os.replace(temporary, output_path)
    return {
        "status": "complete",
        "path": os.path.abspath(output_path),
        "accepted": bool(selected_candidate is not None),
        "selected_candidate": (
            selected_candidate.name if selected_candidate is not None else None
        ),
        "max_score_adjusted_p": inference["max_score_adjusted_p"],
        "current_k": len(components),
        "next_k": diagnostic["next_k"],
        "selected_split_spec": split_spec_path,
        "covariance_preserving_warm_start": diagnostic[
            "covariance_preserving_warm_start"
        ],
        "stopping_reason": diagnostic["stopping_reason"],
        "marker_score": marker_score_summary,
    }


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
    if float(args.vc_rel_tol) <= 0.0:
        raise SystemExit("vc-rel-tol must be > 0.")
    if float(args.effect_rel_tol) <= 0.0:
        raise SystemExit("effect-rel-tol must be > 0.")
    if int(args.kkt_max_rounds) < 1:
        raise SystemExit("kkt-max-rounds must be >= 1.")
    if int(args.kkt_add_topk) < 1:
        raise SystemExit("kkt-add-topk must be >= 1.")
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
    if args.marker_score_out and int(args.marker_score_probes) < 1:
        raise SystemExit("marker-score-probes must be >= 1.")
    for state_flag, state_path in (
        ("--sparse-state-in", args.sparse_state_in),
        ("--sparse-state-out", args.sparse_state_out),
    ):
        if state_path and not state_path.lower().endswith(".npz"):
            raise SystemExit(f"{state_flag} must end in .npz.")
    if args.sparse_state_in and not os.path.isfile(args.sparse_state_in):
        raise SystemExit(
            f"--sparse-state-in does not exist: {args.sparse_state_in}"
        )
    if args.marker_score_min_validation_r2 is not None:
        threshold = float(args.marker_score_min_validation_r2)
        if not np.isfinite(threshold):
            raise SystemExit("marker-score-min-validation-r2 must be finite.")
        if not args.marker_score_out:
            raise SystemExit(
                "--marker-score-min-validation-r2 requires --marker-score-out."
            )
    covtree_requested = bool(args.covtree_diagnostic_out)
    if args.covtree_split_spec_out and not covtree_requested:
        raise SystemExit(
            "--covtree-split-spec-out requires --covtree-diagnostic-out."
        )
    if covtree_requested:
        if not args.marker_score_out:
            raise SystemExit(
                "--covtree-diagnostic-out requires --marker-score-out so the "
                "signed marker diagnostic is auditable."
            )
        if not args.covtree_ld_score:
            raise SystemExit(
                "--covtree-diagnostic-out requires --covtree-ld-score."
            )
        if args.marker_score_min_validation_r2 is not None:
            raise SystemExit(
                "CovTree covariance testing cannot be skipped by a prediction-R2 "
                "threshold; remove --marker-score-min-validation-r2."
            )
        if int(args.covtree_bootstrap_draws) < 19:
            raise SystemExit("covtree-bootstrap-draws must be >= 19.")
        if int(args.covtree_min_child_markers) < 1:
            raise SystemExit("covtree-min-child-markers must be >= 1.")
        if not 1 <= int(args.covtree_max_univariate_depth) <= 10:
            raise SystemExit("covtree-max-univariate-depth must lie in 1..10.")
        for name in (
            "covtree_rank_rtol",
            "covtree_parent_theta_abs_min",
            "covtree_parent_theta_rel_min",
        ):
            value = float(getattr(args, name))
            if not np.isfinite(value) or value < 0.0:
                raise SystemExit(
                    f"{name.replace('_', '-')} must be finite and nonnegative."
                )
        if float(args.covtree_rank_rtol) <= 0.0:
            raise SystemExit("covtree-rank-rtol must be > 0.")
        if (
            not np.isfinite(float(args.covtree_alpha))
            or not 0.0 < float(args.covtree_alpha) < 1.0
        ):
            raise SystemExit("covtree-alpha must lie in (0, 1).")
    if (
        not np.isfinite(float(args.lasso_lam_min_ratio))
        or not 0.0 < float(args.lasso_lam_min_ratio) <= 1.0
    ):
        raise SystemExit("lasso-lam-min-ratio must lie in (0, 1].")
    if int(args.lasso_n_lambda) < 1:
        raise SystemExit("lasso-n-lambda must be >= 1.")
    fixed_ratio_refit = args.lasso_fixed_lam_ratio is not None
    iterative_validation_selection = not fixed_ratio_refit
    if (
        args.marker_score_min_validation_r2 is not None
        and not iterative_validation_selection
    ):
        raise SystemExit(
            "Conditional marker-score emission requires validation-lambda selection."
        )
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
        if float(args.lasso_fixed_lam_ratio) < float(
            args.lasso_lam_min_ratio
        ):
            raise SystemExit(
                "lasso-fixed-lam-ratio must be at least "
                "--lasso-lam-min-ratio so it lies on the fitted path."
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
            "and --sparsity-validation-out so validation R2 can select "
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

    bed_list = [b.strip() for b in args.bed_prefix.split(",") if b.strip()]
    pgen_prefix = args.pgen_prefix.strip()
    component_spec_path = args.component_spec.strip()
    component_spec_source = component_spec_path
    component_variant_indices = (
        _load_component_variant_indices(component_spec_source)
        if component_spec_source
        else []
    )
    prediction_bed_list = [
        value.strip()
        for value in args.prediction_bed_prefix.split(",")
        if value.strip()
    ]
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
    if sparsity_validation_requested and not component_variant_indices:
        raise SystemExit(
            "Sparsity validation requires --component-spec, including for "
            "a single K=1 component."
        )
    if component_variant_indices:
        if len(bed_list) > 1:
            raise SystemExit("single-source component partitioning cannot be combined with multiple BED prefixes.")
        if not (len(bed_list) == 1 or pgen_prefix):
            raise SystemExit("single-source component partitioning requires exactly one genotype input.")
    if covtree_requested:
        if not component_variant_indices:
            raise SystemExit(
                "CovTree diagnostics require --component-spec, including at K=1."
            )
        if len(bed_list) != 1 or pgen_prefix:
            raise SystemExit(
                "CovTree diagnostics currently require one PLINK1 BED source."
            )
        if not os.path.isfile(args.covtree_ld_score):
            raise SystemExit(
                f"CovTree LD-score file does not exist: {args.covtree_ld_score}"
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

    # Use first GRM's FAM as the reference for sample alignment.
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
    y_np = y_np.astype(np.float32, copy=False)
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
    plan = run_planner(
        n_samples=y_np.shape[0], p_list=p_list,
        n_grm=(
            len(component_variant_indices)
            if component_variant_indices
            else len(p_list)
        ),
        component_block_sizes=(
            [int(len(group)) for group in component_variant_indices]
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
            "[INFO] single-source SNP-ID component partition enabled: "
            "component_spec=%s n_components=%s block_sizes=%s",
            component_spec_source,
            len(component_variant_indices),
            [int(len(group)) for group in component_variant_indices],
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
        call_plan=fitter._multi_call_plan,
        component_variant_indices=component_variant_indices or None,
    )
    logger.info(
        "[INFO] multi-GRM: n_grm=%s "
        "m_per_grm=%s m_total=%s",
        grm_index.n_grm, grm_index.m_per_grm.tolist(), grm_index.m_total,
    )

    y_jax = jnp.asarray(y_np, dtype=jnp.float32)
    phenotype_mean, phenotype_scale = _phenotype_standardization_stats(y_np)
    n_grm = len(ops.K_mvs)
    genetic_trace_atoms = np.asarray(
        jax.device_get(fitter._projected_core_diag_atoms(ops.diag_list)),
        dtype=np.float64,
    )
    if (
        genetic_trace_atoms.shape != (n_grm,)
        or not np.all(np.isfinite(genetic_trace_atoms))
        or not np.all(genetic_trace_atoms >= 0.0)
    ):
        raise RuntimeError("Invalid genetic trace atoms for sparse REML initialization.")

    def _trace_weighted_h2(theta_values: np.ndarray) -> float:
        theta_arr = np.asarray(theta_values, dtype=np.float64).reshape(-1)
        genetic_var = float(np.dot(theta_arr[:n_grm], genetic_trace_atoms))
        residual_var = float(theta_arr[n_grm])
        return genetic_var / max(genetic_var + residual_var, 1e-8)

    def _trace_weighted_genetic_var(theta_values: np.ndarray) -> float:
        theta_arr = np.asarray(theta_values, dtype=np.float64).reshape(-1)
        return float(np.dot(theta_arr[:n_grm], genetic_trace_atoms))

    # Match fit_reml's trace-calibrated default initialization unless an
    # adaptive parent fit supplies an exactly covariance-preserving child init.
    h2_init_default = 0.5
    trace_sum = float(np.sum(genetic_trace_atoms))
    if trace_sum <= 0.0:
        raise RuntimeError("Sparse REML requires at least one positive-trace GRM.")
    supplied_theta_init = args.variance_components_init.strip()
    if supplied_theta_init:
        theta = _parse_variance_components_init(
            supplied_theta_init,
            n_grm=n_grm,
        )
        theta_init_source = "command_line_json"
    else:
        theta_g0 = np.where(
            genetic_trace_atoms > 0.0,
            h2_init_default / trace_sum,
            0.0,
        )
        theta_e0 = np.array([1.0 - h2_init_default], dtype=np.float64)
        theta = np.concatenate([theta_g0, theta_e0], axis=0)
        theta_init_source = "trace_calibrated_default"
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
            "[validation alpha] complete Lasso path will be evaluated inside "
            "every alpha/theta outer iteration."
        )

    prediction_context: _PredictionFitContext | None = None
    validation_outcome: np.ndarray | None = None
    iterative_validation_trace: list[dict[str, object]] = []
    if iterative_validation_selection:
        prediction_context = _build_prediction_fit_context(
            args=args,
            training_fitter=fitter,
            component_variant_indices=component_variant_indices,
            prediction_bed_list=prediction_bed_list,
            prediction_pgen_prefix=prediction_pgen_prefix,
            covar_transform=covar_transform,
            call_width=call_width,
            cpu_threads=cpu_threads,
            gpu_budget_bytes=gpu_budget_bytes,
            ring_depth=plan.ring_depth,
        )
        validation_outcome = read_phenotype_aligned(
            args.sparsity_validation_pheno_txt,
            prediction_context.sample_ids,
        )
        logger.info(
            "[validation alpha] aligned validation phenotype for %s samples.",
            int(validation_outcome.size),
        )

    support = np.array([], dtype=np.int64)
    candidate_cache = np.array([], dtype=np.int64)
    previous_fixed_mean = None
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
        "lasso_path_monotone_basis_warm_start": True,
        "lasso_column_major_gram": True,
        "lasso_density_adaptive_qb_updates": True,
        "lasso_filtered_exact_kkt_matvec": True,
        "lasso_batched_external_path_products": True,
        "lasso_batched_covariate_path_solves": True,
        "lasso_path_solves": 0,
        "lasso_path_solve_seconds": 0.0,
        "lasso_validation_selection_seconds": 0.0,
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
    final_alignment_completed = False
    final_pair_available = False
    final_pair_source = "unavailable"
    final_alignment_warning = None
    last_aligned_pair = None
    lasso_reml_stop_reason = ""
    penalized_failure_reason = None

    # ---- Precompute loop-invariant B_screen = [y | covar] on device --------
    screen_parts = [y_np[:, None]]
    if covar_np is not None:
        screen_parts.append(covar_np)
    B_screen_np = np.concatenate(screen_parts, axis=1).astype(np.float32, copy=False)
    B_screen_dev = jnp.asarray(B_screen_np, dtype=jnp.float32)
    n_screen = B_screen_np.shape[1]
    sparse_state_in_summary = None
    if args.sparse_state_in:
        sparse_state = _load_sparse_numerical_state(
            args.sparse_state_in,
            grm_index=grm_index,
            n_samples=n_samples,
            n_lambda=int(args.lasso_n_lambda),
        )
        state_screen = sparse_state["screen_solution"]
        if state_screen.shape != (n_samples, n_screen):
            raise ValueError(
                "Sparse state screening RHS count differs from the current design."
            )
        state_candidate = np.asarray(
            sparse_state["candidate"], dtype=np.int64
        )
        support = np.asarray(sparse_state["support"], dtype=np.int64)
        candidate_cache = state_candidate.copy()
        warm_lasso_candidate = state_candidate.copy()
        warm_lasso_beta_path = np.asarray(
            sparse_state["beta_snp_path"], dtype=np.float64
        )
        warm_screen = state_screen
        state_z = np.asarray(sparse_state["z_solution"], dtype=np.float32)
        warm_z_dict = {
            int(marker): state_z[:, column]
            for column, marker in enumerate(state_candidate.tolist())
        }
        previous_fixed_mean = np.asarray(
            sparse_state["fixed_mean"], dtype=np.float64
        )
        sparse_state_in_summary = {
            "status": "loaded",
            "path": os.path.abspath(args.sparse_state_in),
            "candidate_size": int(state_candidate.size),
            "support_size": int(support.size),
            "lambda_rows": int(warm_lasso_beta_path.shape[0]),
            "screen_rhs_columns": int(n_screen),
            "coordinate_remap": "source_to_current_component_cache",
        }
        sparse_path_performance["covtree_parent_state_loaded"] = True
        sparse_path_performance["covtree_parent_candidate_columns"] = int(
            state_candidate.size
        )
        logger.info(
            "[covtree warm] loaded parent sparse state: candidate=%s support=%s",
            int(state_candidate.size),
            int(support.size),
        )
    else:
        sparse_path_performance["covtree_parent_state_loaded"] = False

    covariance_settling_summary = None
    if sparse_state_in_summary is not None:
        settling_residual = (
            np.asarray(y_np, dtype=np.float64) - previous_fixed_mean
        ) / float(phenotype_scale)
        settling_result = _fit_covariate_contrast_residual_reml(
            fitter,
            settling_residual,
            theta,
            covar=covar_np,
            h2_init=_trace_weighted_h2(theta),
        )
        settled_theta, settling_stop_reason = _accepted_reml_theta(
            settling_result,
            expected_components=n_grm + 1,
            stage="CovTree covariance-only settling",
        )
        covariance_settling_summary = {
            "alpha_frozen": True,
            "theta_input": np.asarray(theta, dtype=np.float64).tolist(),
            "theta_output": np.asarray(settled_theta, dtype=np.float64).tolist(),
            "relative_change": float(
                _max_rel_change(settled_theta, theta)
            ),
            "stop_reason": settling_stop_reason,
        }
        theta = np.asarray(settled_theta, dtype=np.float64)
        sparse_path_performance["covtree_covariance_only_settling"] = True
        logger.info(
            "[covtree warm] covariance-only settling stop=%s theta=%s",
            settling_stop_reason,
            theta.tolist(),
        )
    else:
        sparse_path_performance["covtree_covariance_only_settling"] = False

    # Kept for output-schema compatibility.  It now records the single KKT
    # check from the final covariance-aligned Lasso update; it is not a second
    # independent acceptance gate.
    returned_covariance_kkt = {
        "passed": False,
        "tolerance": float("nan"),
        "max_active_error": float("inf"),
        "max_inactive_excess": float("inf"),
        "decision_precision": "ordinary_pcg",
        "method": "candidate_gram_plus_outside_marker_score",
    }
    returned_covariance_kkt_error = None
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

        max_kkt_rounds = int(args.kkt_max_rounds)
        accepted_kkt_record = None
        for kkt_round in range(1, max_kkt_rounds + 1):
            path_pcg_tol = float(args.pcg_tol)
            Hinv_y_for_path = np.asarray(
                sol_screen[:, 0], dtype=np.float64
            )
            Hinv_covar_for_path = None
            if covar_np is not None and covar_np.shape[1] > 0:
                Hinv_covar_for_path = np.asarray(
                    sol_screen[:, 1:n_screen], dtype=np.float64
                )
            # Z_cand PCG with dictionary warm-start as the candidate expands.
            Z_cand = grm_index.extract_standardized_columns(candidate).astype(
                np.float32, copy=False
            )
            B_z = jnp.asarray(Z_cand, dtype=jnp.float32)
            x0_z = None
            if warm_z_dict:
                x0_arr = np.zeros(
                    (n_samples, candidate.size), dtype=np.float32
                )
                hit = 0
                for j, snp_idx in enumerate(candidate.tolist()):
                    snp_i = int(snp_idx)
                    if snp_i in warm_z_dict:
                        x0_arr[:, j] = warm_z_dict[snp_i]
                        hit += 1
                if hit > 0:
                    x0_z = jnp.asarray(x0_arr, dtype=jnp.float32)
                    if args.verbose:
                        logger.info(
                            "[outer %s kkt %s] Z warm-start: %s/%s columns reused",
                            outer, kkt_round, hit, candidate.size,
                        )

            try:
                sol_z, res_all, it_all = pcg_solve(
                    hv,
                    B_z,
                    M=precond,
                    tol=path_pcg_tol,
                    maxiter=args.max_pcg_iters,
                    X0=x0_z,
                )
                candidate_true_res = _true_pcg_relative_residual(
                    hv, B_z, sol_z
                )
                if not np.isfinite(candidate_true_res):
                    raise RuntimeError(
                        "Candidate PCG produced a non-finite true residual."
                    )
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

            sol_z_np = np.asarray(sol_z, dtype=np.float32)
            warm_z_dict = {
                int(snp_idx): sol_z_np[:, j]
                for j, snp_idx in enumerate(candidate.tolist())
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
            if not _allow_monotone_lasso_path_warm_start(
                previous_size=int(warm_lasso_candidate.size),
                current_size=int(candidate.size),
                common_size=warm_lasso_columns,
            ):
                beta_snp_path0 = None
                warm_lasso_columns = 0

            lasso_path_started = time.perf_counter()
            lasso_path_solve_seconds = float("nan")
            validation_selection_seconds = 0.0
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
                            theta_standardized=theta,
                            phenotype_scale=phenotype_scale,
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
                        "theta_standardized": np.asarray(
                            theta, dtype=np.float64
                        ).tolist(),
                        **validation_record,
                    }
                    iterative_validation_trace.append(trace_record)
                    logger.info(
                        "[outer %s kkt %s validation alpha] cand=%s "
                        "ratio=%.6g active=%s validation_R2=%.8f",
                        outer,
                        kkt_round,
                        int(candidate.size),
                        float(lasso["selected_lam_ratio"]),
                        int(np.asarray(lasso["active_idx"]).size),
                        float(
                            validation_record["selected"][
                                "correlation_squared"
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
            ):
                penalized_block_failure = (
                    "Selected LASSO solution did not converge; KKT "
                    "optimality cannot be certified. Increase "
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
            partitioned = _partitioned_lasso_kkt_from_scores(
                score=score_signed,
                candidate=candidate,
                beta_candidate=beta_snp,
                lam=float(lasso["lam"]),
                abs_tol=float(args.kkt_tol),
                rel_tol=float(args.kkt_rel_tol),
            )
            violators = np.asarray(
                partitioned["outside_violators"], dtype=np.int64
            )
            max_outside_score = float(
                partitioned["max_outside_score"]
            )
            kkt_threshold = float(partitioned["threshold"])
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
                "method": "candidate_gram_plus_outside_marker_score",
                "decision_precision": "ordinary_pcg",
                "pcg_tol": path_pcg_tol,
                "pcg_reported_res": float(np.asarray(res_kkt)),
                "pcg_true_res": float(true_res_kkt),
                "pcg_iters": int(it_kkt),
                "max_outside_score": max_outside_score,
                "n_outside_violators": n_viol,
                "candidate_path_kkt_passed": internal_kkt_passed,
                # This independent score is diagnostic only: finite-PCG
                # differences on candidate coordinates do not reject an
                # otherwise solved Lasso block.
                "direct_score_candidate_diagnostic": dict(
                    partitioned["candidate_certificate"]
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
                    "validation_correlation_squared": (
                        float(
                            lasso["validation_selection"]["selected"][
                                "correlation_squared"
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
                    "validation_selection_seconds": (
                        validation_selection_seconds
                    ),
                    "decision": action,
                    "path_pcg_tol": path_pcg_tol,
                    "candidate_pcg_reported_res": float(np.asarray(res_all)),
                    "candidate_pcg_true_res": float(candidate_true_res),
                    "candidate_pcg_iters": int(it_all),
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
                returned_covariance_kkt = last_aligned_pair["kkt"]
                final_pair_available = True
                final_pair_source = "last_complete_pair"
                final_alignment_warning = str(penalized_block_failure)
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
                        if lasso is not None
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
        support_same = bool(np.array_equal(support_new, support))

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
        residual_raw = _lasso_residual(
            y=y_np,
            covar=covar_np,
            geno=Z_cand,
            beta_cov=beta_cov_current,
            beta_snp=beta_snp_current,
        )
        fixed_mean_current = np.asarray(y_np, dtype=np.float64) - residual_raw

        if not final_alignment:
            last_aligned_pair = {
                "candidate": candidate.copy(),
                "lasso": lasso,
                "support": support_new.copy(),
                "theta": theta.copy(),
                "kkt": dict(accepted_kkt_record),
            }

        # Exactly one final selected-Lasso update aligns alpha with the
        # covariance returned by the outer loop. No variance update follows it.
        if final_alignment:
            support = support_new
            final_candidate = candidate
            final_lasso = lasso
            final_alignment_completed = True
            final_pair_available = True
            final_pair_source = "final_covariance_lasso"
            returned_covariance_kkt = dict(accepted_kkt_record)
            history.append(
                {
                    "outer": outer,
                    "stage": "final_covariance_lasso",
                    "theta": theta.tolist(),
                    "support_size": int(support_new.size),
                    "support_same": support_same,
                    "lam": float(lasso["lam"]),
                    "lam_ratio": float(lasso["selected_lam_ratio"]),
                    "lambda_selection_method": str(
                        lasso["selection_method"]
                    ),
                    "validation_correlation_squared": (
                        float(
                            lasso["validation_selection"]["selected"][
                                "correlation_squared"
                            ]
                        )
                        if lasso.get("validation_selection") is not None
                        else None
                    ),
                    "kkt_certified": True,
                    "kkt_trace": kkt_trace,
                    "final_alignment": True,
                    "variance_update": "not_run_final_alignment",
                }
            )
            logger.info(
                "[INFO] final covariance-aligned %s-selected Lasso completed "
                "after %s outer variance updates.",
                str(lasso["selection_method"]),
                outer,
            )
            break

        residual_standardized = residual_raw / float(phenotype_scale)
        try:
            ml_res = _fit_covariate_contrast_residual_reml(
                fitter,
                residual_standardized,
                theta,
                covar=covar_np,
                h2_init=_trace_weighted_h2(theta),
            )
            theta_new, lasso_reml_stop_reason = _accepted_reml_theta(
                ml_res,
                expected_components=n_grm + 1,
                stage=f"outer {outer} Lasso covariate-contrast REML block",
            )
            # ``ll_down`` is a converged no-update block: its downhill
            # candidate is rejected and the previous theta is retained.
            variance_blocks_completed += 1
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
                    "support_same": support_same,
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
        vc_rel = _max_rel_change(theta_new, theta)
        vc_stable, vc_change_ratio = _variance_components_converged(
            theta_new,
            theta,
            rel_tol=float(args.vc_rel_tol),
        )
        effect_rel = _relative_fitted_mean_change(
            fixed_mean_current,
            previous_fixed_mean,
            y_np,
        )
        effect_stable = bool(
            np.isfinite(effect_rel)
            and effect_rel <= float(args.effect_rel_tol)
        )

        history.append({
            "outer": outer,
            "stage": "outer_update",
            "pcg_screen_iters": int(it_screen),
            "pcg_screen_res": float(np.asarray(res_screen)),
            "pcg_all_iters": int(it_all),
            "pcg_all_res": float(np.asarray(res_all)),
            "theta": theta_new.tolist(),
            "support_size": int(support_new.size),
            "support_same": support_same,
            "vc_rel": float(vc_rel),
            "vc_change_ratio": float(vc_change_ratio),
            "vc_stable": bool(vc_stable),
            "effect_rel": float(effect_rel),
            "effect_stable": bool(effect_stable),
            "lam": float(lasso["lam"]),
            "lam_ratio": float(lasso["selected_lam_ratio"]),
            "lambda_selection_method": str(lasso["selection_method"]),
            "validation_correlation_squared": (
                float(
                    lasso["validation_selection"]["selected"][
                        "correlation_squared"
                    ]
                )
                if lasso.get("validation_selection") is not None
                else None
            ),
            "kkt_certified": bool(certified_kkt),
            "kkt_trace": kkt_trace,
            "final_alignment": False,
            "variance_update": (
                "covariate_contrast_residual_reml_no_update"
                if lasso_reml_stop_reason == "ll_down"
                else "covariate_contrast_residual_reml"
            ),
            "variance_stop_reason": lasso_reml_stop_reason,
            "variance_step_rejected": bool(
                lasso_reml_stop_reason == "ll_down"
            ),
        })

        logger.info(
            "[outer %s] pcg_screen=%s pcg_all=%s cand=%s active=%s "
            "kkt_rounds=%s lam=%.3e validation_R2=%s vc_ratio=%.3e "
            "effect_rel=%.3e support_same=%s iter_time=%.1fs",
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
                        "correlation_squared"
                    ]
                )
                if lasso.get("validation_selection") is not None
                else "fixed"
            ),
            vc_change_ratio,
            effect_rel,
            support_same,
            time.time() - iter_t0,
        )

        theta = theta_new
        support = support_new
        final_candidate = candidate
        final_lasso = lasso
        previous_fixed_mean = fixed_mean_current

        stable_candidate = bool(
            variance_blocks_completed >= 2
            and vc_stable
            and effect_stable
        )
        if stable_candidate:
            outer_converged = True
            outer_stop_reason = "converged"
            final_alignment_pending = True
            logger.info(
                "[INFO] outer convergence reached at update %s; running one "
                "final covariance-aligned %s-selected Lasso.",
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
    # from the same covariance.  A selected-support refit is an explicit,
    # downstream comparison and is never part of the default estimator.
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
        and bool(returned_covariance_kkt["passed"])
        and finite_valid_theta
        and penalized_failure_reason is None
    )
    alpha_theta_fixed_point_coherent = bool(
        lasso_ml_outer_converged and alpha_theta_pair_usable
    )
    outer_convergence_warning = (
        None
        if lasso_ml_outer_converged or not alpha_theta_pair_usable
        else "outer_max_reached_before_change_tolerances"
    )

    comparison_enabled = bool(args.compare_four_estimators)

    # ---- Optional selected-support REML refit ----------------------------
    Z_selected_for_reml = np.empty((n_samples, 0), dtype=np.float32)
    X_selected_span = covar_np
    selected_span_basis_local = np.empty((0,), dtype=np.int64)
    selected_span_reml_iterations = 0
    selected_span_reml_stop_reason = ""
    selected_span_reml_history = []
    selected_span_refit_ok = False
    selected_span_refit_error = None
    theta_selected_span_reml = theta_lasso_ml.copy()
    if comparison_enabled and alpha_theta_pair_usable:
        try:
            Z_selected_for_reml = (
                grm_index.extract_standardized_columns(support)
                .astype(np.float32, copy=False)
            )
            if support.size > 0:
                X_selected_span, selected_span_basis_local = (
                    _merge_independent_fixed_effects(
                        X_selected_span,
                        Z_selected_for_reml,
                    )
                )
                if selected_span_basis_local.size != support.size:
                    logger.info(
                        "[refit] selected-span REML retained %s/%s active SNP "
                        "columns after removing numerical dependencies.",
                        int(selected_span_basis_local.size),
                        int(support.size),
                    )
            selected_span_reml = fitter.fit_infinitesimal(
                y_jax,
                (
                    jnp.asarray(X_selected_span, dtype=jnp.float32)
                    if X_selected_span is not None
                    else None
                ),
                h2_init=_trace_weighted_h2(theta_lasso_ml),
                var_components_init=jnp.asarray(
                    theta_lasso_ml, dtype=jnp.float32
                ),
            )
            selected_span_reml_history = list(selected_span_reml.history)
            (
                theta_selected_span_reml,
                selected_span_reml_stop_reason,
            ) = _accepted_reml_theta(
                selected_span_reml,
                expected_components=n_grm + 1,
                stage="selected-span REML refit",
            )
            selected_span_reml_iterations = len(
                selected_span_reml.history
            )
            selected_span_refit_ok = True
        except (FloatingPointError, RuntimeError, ValueError) as error:
            selected_span_refit_error = str(error)
            logger.warning(
                "[WARN] selected-support REML refit is unavailable; "
                "estimators 3 and 4 will be null while a valid Lasso branch "
                "remains unchanged: %s",
                selected_span_refit_error,
            )
    elif comparison_enabled:
        selected_span_refit_error = (
            "skipped because the penalized-ML branch was not accepted"
        )

    # ---- Output results ----
    out_dir = os.path.dirname(out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    unavailable = float("nan")
    theta_lasso_ml_sum = _trace_weighted_genetic_var(theta_lasso_ml)
    theta_e_lasso_ml = float(theta_lasso_ml[-1])
    theta_final_sum = (
        _trace_weighted_genetic_var(theta_selected_span_reml)
        if selected_span_refit_ok
        else unavailable
    )
    theta_e_final = (
        float(theta_selected_span_reml[-1])
        if selected_span_refit_ok
        else unavailable
    )
    q_chive = unavailable
    q_chive_standardized = unavailable
    q_chive_term1 = unavailable
    q_chive_term2 = unavailable
    q_chive_term1_standardized = unavailable
    q_chive_term2_standardized = unavailable
    q_ss_gls_plugin_raw = unavailable
    q_ss_gls_plugin_standardized = unavailable
    q_ss_gls_df_corrected_raw = unavailable
    q_ss_gls_df_corrected_standardized = unavailable
    ss_gls_df_correction_raw = unavailable
    ss_gls_df_correction_standardized = unavailable
    ss_gls_basis_size = 0
    beta_cov_lasso = np.empty((0,), dtype=np.float64)
    beta_cov_gls = np.empty((0,), dtype=np.float64)
    beta_gls_active = np.empty((0,), dtype=np.float64)
    beta_lasso_active = np.empty((0,), dtype=np.float64)
    selected_span_basis_positions = np.empty((0,), dtype=np.int64)
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
                selected_span_refit_ok = False
                selected_span_refit_error = (
                    "invalidated because the final Lasso active set does not "
                    "match the exported support"
                )
        except (IndexError, KeyError) as error:
            lasso_quadratics_available = False
            if penalized_failure_reason is None:
                penalized_failure_reason = (
                    "Final Lasso active-set validation failed: "
                    f"{error}"
                )
            alpha_theta_fixed_point_coherent = False
            alpha_theta_pair_usable = False
            selected_span_refit_ok = False
            selected_span_refit_error = (
                "invalidated by final Lasso active-set validation failure"
            )
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
            q_chive_standardized = _quadratic_variance_to_reml_scale(
                q_chive, phenotype_scale
            )
            q_chive_term1_standardized = (
                _quadratic_variance_to_reml_scale(
                    q_chive_term1, phenotype_scale
                )
            )
            q_chive_term2_standardized = (
                _quadratic_variance_to_reml_scale(
                    q_chive_term2, phenotype_scale
                )
            )

    # Comparison estimators 3 and 4 and exported refit coefficients use this
    # one selected-span REML--GLS solution.  The independent basis is mapped
    # back to the selected support with zero coefficients for numerically
    # dependent marker columns.
    if selected_span_refit_ok:
        q_ss_gls_plugin_raw = 0.0
        q_ss_gls_plugin_standardized = 0.0
        q_ss_gls_df_corrected_raw = 0.0
        q_ss_gls_df_corrected_standardized = 0.0
        ss_gls_df_correction_raw = 0.0
        ss_gls_df_correction_standardized = 0.0
        beta_gls_active = np.zeros(support.size, dtype=np.float64)

    if selected_span_refit_ok:
        try:
            theta_g = jnp.asarray(
                theta_selected_span_reml[:-1], dtype=jnp.float32
            )
            theta_e = jnp.asarray(
                theta_selected_span_reml[-1], dtype=jnp.float32
            )
            hv_final = fitter._make_hv(ops, theta_g, theta_e)
            precond_final = fitter._make_effect_precond(
                ops, theta_g, theta_e
            )

            solve_parts = [y_np[:, None]]
            n_covar = 0
            if covar_np is not None:
                solve_parts.append(covar_np)
                n_covar = int(covar_np.shape[1])
            solve_parts.append(Z_support)
            B_final = np.concatenate(
                solve_parts, axis=1
            ).astype(np.float32, copy=False)
            sol_final, res_final, it_final = pcg_solve(
                hv_final,
                jnp.asarray(B_final, dtype=jnp.float32),
                M=precond_final,
                tol=args.pcg_tol,
                maxiter=args.max_pcg_iters,
            )
            _require_pcg_converged(
                res_final,
                tol=args.pcg_tol,
                iters=it_final,
                maxiter=args.max_pcg_iters,
                stage="final selected-span REML--GLS",
            )
            sol_final_np = np.asarray(sol_final, dtype=np.float64)
            Hinv_y_final = sol_final_np[:, 0]
            Hinv_covar_final = None
            if n_covar > 0:
                Hinv_covar_final = sol_final_np[:, 1 : 1 + n_covar]
            Hinv_Z_support = sol_final_np[:, 1 + n_covar :]

            ss_gls = _selected_span_gls_quadratics(
                y=y_np,
                covar=covar_np,
                z_active=Z_support,
                Hinv_y=Hinv_y_final,
                Hinv_covar=Hinv_covar_final,
                Hinv_z_active=Hinv_Z_support,
                phenotype_scale=phenotype_scale,
            )
            selected_span_basis_positions = np.asarray(
                ss_gls["active_basis_idx"], dtype=np.int64
            )
            if not np.array_equal(
                selected_span_basis_positions,
                selected_span_basis_local,
            ):
                raise RuntimeError(
                    "Selected-span REML and GLS retained different marker "
                    "bases."
                )
            q_ss_gls_plugin_raw = float(ss_gls["q_plugin_raw"])
            q_ss_gls_plugin_standardized = float(
                ss_gls["q_plugin_standardized"]
            )
            q_ss_gls_df_corrected_raw = float(
                ss_gls["q_df_corrected_raw"]
            )
            q_ss_gls_df_corrected_standardized = float(
                ss_gls["q_df_corrected_standardized"]
            )
            ss_gls_df_correction_raw = float(
                ss_gls["df_correction_raw"]
            )
            ss_gls_df_correction_standardized = float(
                ss_gls["df_correction_standardized"]
            )
            ss_gls_basis_size = int(
                selected_span_basis_positions.size
            )

            beta_cov_gls = np.asarray(
                ss_gls["beta_cov"], dtype=np.float64
            ).reshape(-1)
            beta_gls_active = _expand_selected_basis_coefficients(
                support_size=int(support.size),
                basis_positions=selected_span_basis_positions,
                basis_coefficients=np.asarray(
                    ss_gls["beta_active_basis"], dtype=np.float64
                ),
            )
        except (FloatingPointError, RuntimeError, ValueError) as error:
            selected_span_refit_ok = False
            selected_span_refit_error = (
                "Selected-span coefficient recovery failed: "
                f"{error}"
            )
            logger.warning(
                "[WARN] %s; estimators 3 and 4 will be null while a valid "
                "Lasso branch remains unchanged.",
                selected_span_refit_error,
            )
            theta_final_sum = unavailable
            theta_e_final = unavailable
            q_ss_gls_plugin_raw = unavailable
            q_ss_gls_plugin_standardized = unavailable
            q_ss_gls_df_corrected_raw = unavailable
            q_ss_gls_df_corrected_standardized = unavailable
            ss_gls_df_correction_raw = unavailable
            ss_gls_df_correction_standardized = unavailable
            beta_cov_gls = np.empty((0,), dtype=np.float64)
            beta_gls_active = np.empty((0,), dtype=np.float64)
            selected_span_basis_positions = np.empty(
                (0,), dtype=np.int64
            )
            ss_gls_basis_size = 0

    # The primary COHERIT estimate always uses the calibrated Lasso quadratic
    # and the covariate-contrast REML covariance from the same penalized
    # branch. The other three estimators are constructed only in explicit
    # comparison mode.
    h2_chive = _sparse_dense_h2(
        q_chive_standardized,
        theta_lasso_ml_sum,
        theta_e_lasso_ml,
    )
    h2_lasso_plugin = unavailable
    h2_ss_gls_plugin = unavailable
    h2_ss_gls_df_corrected = unavailable
    if comparison_enabled:
        four_h2 = _four_estimator_h2_from_branches(
            q_lasso_plugin_standardized=q_chive_term1_standardized,
            q_lasso_calibrated_standardized=q_chive_standardized,
            q_selected_span_plugin_standardized=q_ss_gls_plugin_standardized,
            q_selected_span_trace_standardized=(
                q_ss_gls_df_corrected_standardized
            ),
            lasso_ml_background_variance=theta_lasso_ml_sum,
            lasso_ml_residual_variance=theta_e_lasso_ml,
            selected_span_reml_background_variance=theta_final_sum,
            selected_span_reml_residual_variance=theta_e_final,
        )
        h2_lasso_plugin = four_h2["h2_lasso_plugin"]
        h2_chive = four_h2["h2_chive"]
        h2_ss_gls_plugin = four_h2["h2_ss_gls_plugin"]
        h2_ss_gls_df_corrected = four_h2["h2_ss_gls_df_corrected"]
    branch_guards = _sparse_estimator_branch_guards(
        comparison_enabled=comparison_enabled,
        alpha_theta_pair_certified=alpha_theta_pair_usable,
        lasso_quadratics_available=lasso_quadratics_available,
        selected_span_refit_ok=selected_span_refit_ok,
        lasso_estimator_values=np.asarray(
            [h2_lasso_plugin, h2_chive]
            if comparison_enabled
            else [h2_chive],
            dtype=np.float64,
        ),
        selected_support_estimator_values=np.asarray(
            [h2_ss_gls_plugin, h2_ss_gls_df_corrected],
            dtype=np.float64,
        ),
    )
    lasso_branch_valid = bool(branch_guards["lasso_branch_valid"])
    selected_support_refit_branch_valid = bool(
        branch_guards["selected_support_refit_branch_valid"]
    )
    all_requested_estimators_valid = bool(
        branch_guards["all_requested_estimators_valid"]
    )
    sparse_outputs_finite = bool(
        branch_guards["all_requested_outputs_finite"]
    )
    sparse_fit_rejection_reasons = list(
        branch_guards["combined_invalid_reasons"]
    )
    if not all_requested_estimators_valid:
        if comparison_enabled:
            logger.warning(
                "[WARN] requested sparse estimator branches incomplete: "
                "lasso_valid=%s selected_support_refit_valid=%s reasons=%s. "
                "No ordinary-REML value will replace a sparse estimator.",
                lasso_branch_valid,
                selected_support_refit_branch_valid,
                ",".join(sparse_fit_rejection_reasons),
            )
        else:
            logger.warning(
                "[WARN] COHERIT estimator unavailable: lasso_valid=%s "
                "reasons=%s. No ordinary-REML value will replace it.",
                lasso_branch_valid,
                ",".join(sparse_fit_rejection_reasons),
            )

    h2_lasso_plugin_guarded = (
        float(h2_lasso_plugin)
        if comparison_enabled and lasso_branch_valid
        else unavailable
    )
    h2_chive_guarded = float(h2_chive) if lasso_branch_valid else unavailable
    h2_ss_gls_plugin_guarded = (
        float(h2_ss_gls_plugin)
        if selected_support_refit_branch_valid
        else unavailable
    )
    h2_ss_gls_df_guarded = (
        float(h2_ss_gls_df_corrected)
        if selected_support_refit_branch_valid
        else unavailable
    )
    h2 = h2_chive_guarded
    primary_h2_method = (
        "penalized_reml_lasso_chive" if lasso_branch_valid else "unavailable"
    )

    h2_background_selected_span_reml = (
        _trace_weighted_h2(theta_selected_span_reml)
        if selected_support_refit_branch_valid
        else unavailable
    )

    adaptive_marker_score_summary = None
    covtree_diagnostic_summary = None
    marker_score_output = args.marker_score_out.strip()
    if marker_score_output and not covtree_requested:
        if not lasso_branch_valid:
            raise RuntimeError(
                "Adaptive marker scores require a valid covariance-aligned "
                "COHERIT Lasso branch."
            )
        marker_score_threshold = args.marker_score_min_validation_r2
        observed_validation_r2 = None
        if marker_score_threshold is not None:
            validation_selection = (
                final_lasso.get("validation_selection")
                if final_lasso is not None
                else None
            )
            if not isinstance(validation_selection, dict):
                raise RuntimeError(
                    "Conditional marker scoring requires final validation selection."
                )
            observed_validation_r2 = float(
                validation_selection["selected"]["correlation_squared"]
            )

        if not _validation_allows_marker_score(
            observed_validation_r2,
            marker_score_threshold,
        ):
            adaptive_marker_score_summary = {
                "status": "skipped_validation_decline",
                "path": os.path.abspath(marker_score_output),
                "observed_validation_r2": observed_validation_r2,
                "required_min_validation_r2": float(marker_score_threshold),
                "reason": "adaptive_k_layer_will_not_be_split",
            }
            logger.info(
                "[adaptive] skipped unused marker scores: validation_R2=%.8f "
                "< required %.8f",
                observed_validation_r2,
                float(marker_score_threshold),
            )
        else:
            marker_score_residual = _lasso_residual(
                y=y_np,
                covar=covar_np,
                geno=Z_support,
                beta_cov=beta_cov_lasso,
                beta_snp=beta_lasso_active,
            )
            adaptive_marker_score_summary = _compute_adaptive_marker_scores(
                output_path=marker_score_output,
                residual_raw=marker_score_residual,
                covar=covar_np,
                phenotype_scale=phenotype_scale,
                fitter=fitter,
                ops=ops,
                grm_index=grm_index,
                theta=theta_lasso_ml,
                n_probes=int(args.marker_score_probes),
                seed=int(args.marker_score_seed),
                pcg_tol=float(args.pcg_tol),
                max_pcg_iters=int(args.max_pcg_iters),
            )
            logger.info(
                "[adaptive] marker scores -> %s (probes=%s clipped=%s)",
                marker_score_output,
                int(args.marker_score_probes),
                adaptive_marker_score_summary[
                    "information_clipped_count"
                ],
            )

    if covtree_requested:
        marker_score_residual = _lasso_residual(
            y=y_np,
            covar=covar_np,
            geno=Z_support,
            beta_cov=beta_cov_lasso,
            beta_snp=beta_lasso_active,
        )
        covtree_diagnostic_summary = _run_covtree_diagnostic(
            args=args,
            fitter=fitter,
            ops=ops,
            grm_index=grm_index,
            component_spec_path=component_spec_source,
            bed_prefix=bed_list[0],
            marker_score_path=marker_score_output,
            residual_raw=marker_score_residual,
            covar=covar_np,
            phenotype_scale=phenotype_scale,
            theta=theta_lasso_ml,
        )
        adaptive_marker_score_summary = covtree_diagnostic_summary["marker_score"]
        logger.info(
            "[covtree] accepted=%s candidate=%s adjusted_p=%s next_K=%s",
            covtree_diagnostic_summary["accepted"],
            covtree_diagnostic_summary["selected_candidate"],
            covtree_diagnostic_summary["max_score_adjusted_p"],
            covtree_diagnostic_summary["next_k"],
        )

    sparse_state_out_summary = None
    if args.sparse_state_out:
        next_split_needed = not covtree_requested or bool(
            covtree_diagnostic_summary["accepted"]
        )
        if not next_split_needed:
            sparse_state_out_summary = {
                "status": "not_emitted_covtree_stopped",
                "path": None,
            }
        else:
            if (
                not alpha_theta_pair_usable
                or final_candidate.size == 0
                or warm_lasso_beta_path is None
                or warm_screen is None
            ):
                raise RuntimeError(
                    "Sparse numerical state output requires a valid final "
                    "covariance-aligned Lasso pair."
                )
            sparse_state_out_summary = _write_sparse_numerical_state(
                args.sparse_state_out,
                grm_index=grm_index,
                candidate=final_candidate,
                support=support,
                beta_snp_path=warm_lasso_beta_path,
                screen_solution=warm_screen,
                warm_z_dict=warm_z_dict,
                fixed_mean=fixed_mean_current,
            )
            logger.info(
                "[covtree warm] sparse numerical state -> %s",
                args.sparse_state_out,
            )

    print(f"[RESULT] var_components_lasso_ml={theta_lasso_ml.tolist()}")
    print(f"[RESULT] h2={h2:.6f} (primary={primary_h2_method})")
    print(f"[RESULT] h2_chive={h2_chive:.6f} (penalized LASSO calibration)")
    if comparison_enabled:
        print(
            "[RESULT] var_components_selected_span_reml="
            f"{theta_selected_span_reml.tolist() if selected_span_refit_ok else None}"
        )
        print(
            "[RESULT] h2_background_selected_span_reml="
            f"{h2_background_selected_span_reml:.6f}"
        )
        print(
            f"[RESULT] h2_lasso_plugin={h2_lasso_plugin:.6f} "
            "(uncorrected penalized-LASSO plug-in)"
        )
        print(
            f"[RESULT] h2_ss_gls_plugin={h2_ss_gls_plugin:.6f} "
            "(selected-span GLS plug-in)"
        )
        print(
            f"[RESULT] h2_ss_gls_df_corrected={h2_ss_gls_df_corrected:.6f} "
            "(trace-corrected selected-span GLS)"
        )
    print(f"[RESULT] support_size={int(support.size)}")

    theta_lasso_ml_to_selected_span_rel = (
        _max_rel_change(theta_selected_span_reml, theta_lasso_ml)
        if selected_span_refit_ok
        else None
    )
    selected_span_basis_support_indices = (
        support[selected_span_basis_positions].tolist()
        if selected_span_refit_ok
        else []
    )
    # Preserve the partitioned, PCG-compatible KKT diagnostic. Non-finite
    # placeholders from an unavailable check become JSON null.
    returned_covariance_kkt_summary = _json_safe_value(
        returned_covariance_kkt
    )

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
            theta_standardized=theta_lasso_ml,
            phenotype_scale=phenotype_scale,
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
        # earlier comparison run when the current run did not request one.
        remove_sparse_prediction_outputs(out_prefix)
    if prediction_active:
        emitted_branches = _sparse_prediction_branch_names(
            comparison_enabled=comparison_enabled,
            lasso_branch_valid=lasso_branch_valid,
            selected_support_refit_branch_valid=(
                selected_support_refit_branch_valid
            ),
        )
        prediction_request_metadata = {
            "estimator_mode": (
                "four_estimator_comparison"
                if comparison_enabled
                else "coherit"
            ),
            "test_phenotype_used": False,
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
            "scale_metadata": {
                "variance_components": (
                    "standardized_phenotype_variance"
                ),
                "fixed_effect_coefficients": (
                    "raw_phenotype_units_per_training_transformed_design_unit"
                ),
                "fixed_snp_score": "raw_phenotype_units",
                "background_blup": "raw_phenotype_units",
                "genetic_score": (
                    "raw_phenotype_units; "
                    "fixed_snp_score_plus_background_blup"
                ),
                "phenotype_prediction": (
                    "raw_phenotype_units; "
                    "nuisance_fixed_score_plus_genetic_score"
                ),
                "phenotype_scale": float(phenotype_scale),
                "phenotype_mean": float(phenotype_mean),
            },
        }
        if not emitted_branches:
            unavailable_branch_metadata = {
                "lasso_branch_valid": lasso_branch_valid,
            }
            if comparison_enabled:
                unavailable_branch_metadata[
                    "selected_support_refit_branch_valid"
                ] = selected_support_refit_branch_valid
            metadata_path = write_sparse_prediction_status(
                out_prefix=out_prefix,
                status="not_emitted_no_valid_branch",
                metadata={
                    **prediction_request_metadata,
                    "reason": "no_valid_sparse_prediction_branch",
                    **unavailable_branch_metadata,
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
                        **(
                            {
                                "selected_span": {
                                    "estimator_valid": False,
                                    "output_emitted": False,
                                    "invalid_reasons": list(
                                        branch_guards[
                                            "selected_support_refit_branch_invalid_reasons"
                                        ]
                                    ),
                                }
                            }
                            if comparison_enabled
                            else {}
                        ),
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
            write_sparse_prediction_status(
                out_prefix=out_prefix,
                status="preparing",
                metadata={
                    **prediction_request_metadata,
                    "branch_outputs_emitted": False,
                },
            )
            if prediction_context is None:
                prediction_context = _build_prediction_fit_context(
                    args=args,
                    training_fitter=fitter,
                    component_variant_indices=component_variant_indices,
                    prediction_bed_list=prediction_bed_list,
                    prediction_pgen_prefix=prediction_pgen_prefix,
                    covar_transform=covar_transform,
                    call_width=call_width,
                    cpu_threads=cpu_threads,
                    gpu_budget_bytes=gpu_budget_bytes,
                    ring_depth=plan.ring_depth,
                )
            prediction_fitter = prediction_context.fitter
            prediction_grm_index = prediction_context.grm_index
            prediction_covar = prediction_context.covar
            prediction_ids = prediction_context.sample_ids
            try:
                prediction_support = (
                    prediction_grm_index.extract_standardized_columns(
                        support
                    ).astype(np.float32, copy=False)
                )
                lasso_prediction = predict_sparse_branch(
                    name="lasso",
                    fitter=fitter,
                    test_fitter=prediction_fitter,
                    y_train_raw=y_np,
                    train_covar=covar_np,
                    test_covar=prediction_covar,
                    train_active_geno=Z_support,
                    test_active_geno=prediction_support,
                    beta_cov_raw=beta_cov_lasso,
                    beta_active_raw=beta_lasso_active,
                    theta_standardized=theta_lasso_ml,
                    phenotype_scale=phenotype_scale,
                    pcg_tol=args.pcg_tol,
                    max_pcg_iters=args.max_pcg_iters,
                )
                selected_span_prediction = None
                if "selected_span" in emitted_branches:
                    selected_span_prediction = predict_sparse_branch(
                        name="selected_span",
                        fitter=fitter,
                        test_fitter=prediction_fitter,
                        y_train_raw=y_np,
                        train_covar=covar_np,
                        test_covar=prediction_covar,
                        train_active_geno=Z_support,
                        test_active_geno=prediction_support,
                        beta_cov_raw=beta_cov_gls,
                        beta_active_raw=beta_gls_active,
                        theta_standardized=theta_selected_span_reml,
                        phenotype_scale=phenotype_scale,
                        pcg_tol=args.pcg_tol,
                        max_pcg_iters=args.max_pcg_iters,
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
                    "theta_standardized": theta_lasso_ml.tolist(),
                    "residual": (
                        "(y-X_beta_cov_lasso-Z_support_beta_lasso)"
                        "/phenotype_scale"
                    ),
                    "support_size": int(support.size),
                    "pcg_rel_res": lasso_prediction.pcg_rel_res,
                    "pcg_iters": lasso_prediction.pcg_iters,
                },
            }
            if comparison_enabled:
                branch_metadata["selected_span"] = {
                    "estimator_valid": bool(
                        selected_support_refit_branch_valid
                    ),
                    "output_emitted": bool(
                        selected_support_refit_branch_valid
                    ),
                    "invalid_reasons": list(
                        branch_guards[
                            "selected_support_refit_branch_invalid_reasons"
                        ]
                    ),
                }
            if "selected_span" in emitted_branches:
                assert selected_span_prediction is not None
                branch_metadata["selected_span"].update({
                    "mean_estimator": "selected_span_gls",
                    "covariance_estimator": "selected_span_reml",
                    "theta_standardized": (
                        theta_selected_span_reml.tolist()
                    ),
                    "residual": (
                        "(y-X_beta_cov_gls-Z_support_beta_gls)"
                        "/phenotype_scale"
                    ),
                    "support_size": int(support.size),
                    "independent_basis_size": int(ss_gls_basis_size),
                    "pcg_rel_res": (
                        selected_span_prediction.pcg_rel_res
                    ),
                    "pcg_iters": selected_span_prediction.pcg_iters,
                })
            prediction_paths = write_sparse_prediction_outputs(
                out_prefix=out_prefix,
                sample_ids=prediction_ids,
                lasso=lasso_prediction,
                selected_span=selected_span_prediction,
                metadata={
                    **prediction_request_metadata,
                    "branch_outputs_emitted": True,
                    "emitted_branches": emitted_branches,
                    "branches": branch_metadata,
                },
            )
            prediction_summary = {
                "requested": True,
                "status": "emitted",
                "n_samples": len(prediction_ids),
                "emitted_branches": emitted_branches,
                "paths": prediction_paths,
            }

    output_contract = _sparse_output_contract(comparison_enabled)
    summary = {
        # Schema 6 fixes validation R2 as the training-stage lambda selector,
        # reserves fixed-lambda-ratio selection for the final refit, and
        # removes the former information-criterion fields.
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
        "genetic_trace_atoms": genetic_trace_atoms.tolist(),
        "lambda_selection_method": (
            str(final_lasso["selection_method"])
            if final_lasso is not None
            else None
        ),
        "lasso_path_complete": True,
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
                    "correlation_squared"
                ]
            )
            if final_lasso is not None
            and final_lasso.get("validation_selection") is not None
            else None
        ),
        "component_spec": component_spec_source or None,
        "component_partition_mode": (
            "snp_id" if component_variant_indices else "input_prefix"
        ),
        "var_components_lasso_ml": theta_lasso_ml.tolist(),
        "variance_component_branch_mapping": {
            "h2_chive": "var_components_lasso_ml",
        },
        "var_components_at_lasso": theta_lasso.tolist(),
        "phenotype_mean": phenotype_mean,
        "phenotype_scale": phenotype_scale,
        "variance_component_scale": "standardized_phenotype",
        "primary_h2_method": primary_h2_method,
        "all_requested_estimators_valid": all_requested_estimators_valid,
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
        "vc_rel_tol": float(args.vc_rel_tol),
        "effect_rel_tol": float(args.effect_rel_tol),
        "outer_stop_reason": outer_stop_reason,
        "outer_convergence_warning": outer_convergence_warning,
        "penalized_failure_reason": penalized_failure_reason,
        "lasso_variance_update": "covariate_contrast_residual_reml",
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
        "returned_covariance_kkt": returned_covariance_kkt_summary,
        "returned_covariance_kkt_error": returned_covariance_kkt_error,
        "theta_lasso_to_lasso_ml_rel_change": (
            theta_lasso_to_lasso_ml_rel
        ),
        "h2_background_lasso_ml": _trace_weighted_h2(theta_lasso_ml),
        "h2_chive": _finite_float_or_none(h2_chive),
        "h2_chive_guarded": h2_chive_guarded,
        "h2": h2,
        "q_chive_raw": _finite_float_or_none(q_chive),
        "q_chive_standardized": _finite_float_or_none(
            q_chive_standardized
        ),
        "q_chive_components": {
            "term1_g2_over_n": _finite_float_or_none(q_chive_term1),
            "term2_cross": _finite_float_or_none(q_chive_term2),
            "scale": "raw_phenotype_variance",
        },
        "q_chive_components_standardized": {
            "term1_g2_over_n": _finite_float_or_none(
                q_chive_term1_standardized
            ),
            "term2_cross": _finite_float_or_none(
                q_chive_term2_standardized
            ),
            "scale": "standardized_phenotype_variance",
        },
        "support_size": int(support.size),
        "support_indices": support.tolist(),
        "support_source_indices": grm_index.source_variant_indices(support).tolist(),
        # Candidate expansion certifies the selected lambda in each outer
        # round. It does not certify every unselected point on the path.
        "kkt_certification_scope": "selected_lambda_only",
        "kkt_certificate_definition": (
            "candidate_gram_plus_outside_marker_score"
        ),
        "lasso_path_globally_kkt_certified": False,
        "kkt_certified": bool(
            history and bool(history[-1].get("kkt_certified", False))
        ),
        "returned_covariance_kkt_certified": bool(
            returned_covariance_kkt["passed"]
        ),
        "outer_history": history,
        "sparse_prediction": prediction_summary,
        "sparsity_validation": sparsity_validation_summary,
    }

    if supplied_theta_init:
        summary["variance_components_initial"] = theta_initial.tolist()
        summary["variance_components_init_source"] = theta_init_source
    if adaptive_marker_score_summary is not None:
        summary["adaptive_marker_score"] = adaptive_marker_score_summary
    if covtree_diagnostic_summary is not None:
        summary["covtree_diagnostic"] = covtree_diagnostic_summary
    if sparse_state_in_summary is not None:
        summary["sparse_state_in"] = sparse_state_in_summary
    if covariance_settling_summary is not None:
        summary["covtree_covariance_only_settling"] = (
            covariance_settling_summary
        )
    if sparse_state_out_summary is not None:
        summary["sparse_state_out"] = sparse_state_out_summary

    if comparison_enabled:
        summary.update({
            "var_components_selected_span_reml": (
                theta_selected_span_reml.tolist()
                if selected_span_refit_ok
                else None
            ),
            "variance_component_branch_mapping": {
                "h2_lasso_plugin": "var_components_lasso_ml",
                "h2_chive": "var_components_lasso_ml",
                "h2_ss_gls_plugin": "var_components_selected_span_reml",
                "h2_ss_gls_df_corrected": "var_components_selected_span_reml",
            },
            "all_sparse_branches_valid": bool(
                branch_guards["all_four_estimators_valid"]
            ),
            "selected_support_refit_branch_valid": (
                selected_support_refit_branch_valid
            ),
            "selected_support_refit_branch_invalid_reasons": list(
                branch_guards[
                    "selected_support_refit_branch_invalid_reasons"
                ]
            ),
            "selected_support_outputs_finite": bool(
                branch_guards["selected_support_outputs_finite"]
            ),
            "selected_span_reml_iterations": selected_span_reml_iterations,
            "selected_span_reml_history": selected_span_reml_history,
            "selected_span_reml_stop_reason": (
                selected_span_reml_stop_reason or None
            ),
            "selected_span_reml_converged": bool(
                selected_span_reml_stop_reason
                in {"rel_dll", "scoring_step", "ll_down"}
            ),
            "selected_span_refit_ok": selected_span_refit_ok,
            "selected_span_refit_error": selected_span_refit_error,
            "selected_span_basis_size": ss_gls_basis_size,
            "selected_span_basis_support_positions": (
                selected_span_basis_positions.tolist()
            ),
            "selected_span_basis_support_indices": (
                selected_span_basis_support_indices
            ),
            "theta_lasso_ml_to_selected_span_reml_rel_change": (
                theta_lasso_ml_to_selected_span_rel
            ),
            "h2_background_selected_span_reml": _finite_float_or_none(
                h2_background_selected_span_reml
            ),
            "h2_lasso_plugin": _finite_float_or_none(h2_lasso_plugin),
            "h2_lasso_plugin_guarded": h2_lasso_plugin_guarded,
            "h2_lasso_plugin_role": (
                "estimator_1_uncorrected_lasso_ml_plugin"
            ),
            "h2_ss_gls_plugin": _finite_float_or_none(h2_ss_gls_plugin),
            "h2_ss_gls_plugin_guarded": h2_ss_gls_plugin_guarded,
            "h2_ss_gls_df_corrected": _finite_float_or_none(
                h2_ss_gls_df_corrected
            ),
            "h2_ss_gls_df_guarded": h2_ss_gls_df_guarded,
            "h2_ss_gls_role": (
                "estimators_3_and_4_selected_support_reml_gls"
            ),
            "q_lasso_plugin_raw": _finite_float_or_none(q_chive_term1),
            "q_lasso_plugin_standardized": _finite_float_or_none(
                q_chive_term1_standardized
            ),
            "q_ss_gls_plugin_raw": _finite_float_or_none(
                q_ss_gls_plugin_raw
            ),
            "q_ss_gls_plugin_standardized": _finite_float_or_none(
                q_ss_gls_plugin_standardized
            ),
            "q_ss_gls_df_corrected_raw": _finite_float_or_none(
                q_ss_gls_df_corrected_raw
            ),
            "q_ss_gls_df_corrected_standardized": _finite_float_or_none(
                q_ss_gls_df_corrected_standardized
            ),
            "ss_gls_df_correction_raw": _finite_float_or_none(
                ss_gls_df_correction_raw
            ),
            "ss_gls_df_correction_standardized": _finite_float_or_none(
                ss_gls_df_correction_standardized
            ),
            "ss_gls_basis_size": ss_gls_basis_size,
        })

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
    beta_reml_map: dict[int, float] = {}
    if final_lasso is not None and final_candidate.size > 0:
        beta_snp = np.asarray(final_lasso["beta_snp"], dtype=np.float64)
        for snp_idx, beta_val in zip(final_candidate.tolist(), beta_snp.tolist()):
            if beta_val != 0.0:
                beta_map[int(snp_idx)] = float(beta_val)
    if support.size > 0 and beta_gls_active.size == support.size:
        for snp_idx, beta_val in zip(support.tolist(), beta_gls_active.tolist()):
            beta_reml_map[int(snp_idx)] = float(beta_val)

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

    selected_span_basis_set = set(
        int(v) for v in selected_span_basis_support_indices
    )
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
            if comparison_enabled:
                beta_reml = (
                    beta_reml_map.get(int(snp_idx), 0.0)
                    if selected_span_refit_ok
                    else float("nan")
                )
                basis_member = int(int(snp_idx) in selected_span_basis_set)
                row += f"\t{beta_reml:.8e}\t{basis_member}"
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
