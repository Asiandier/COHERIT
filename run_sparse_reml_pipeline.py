#!/usr/bin/env python3
"""
Sparse REML + LASSO pipeline.

Performance notes (vs. previous version):
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
import importlib
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
_sparse_prediction_mod = importlib.import_module(
    f"{pkg_name}.sparse_prediction"
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
predict_sparse_branch = _sparse_prediction_mod.predict_sparse_branch
write_sparse_prediction_outputs = (
    _sparse_prediction_mod.write_sparse_prediction_outputs
)
write_sparse_prediction_status = (
    _sparse_prediction_mod.write_sparse_prediction_status
)

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

LASSO_EBIC_ES_PATIENCE_FIXED = 10
TERMINAL_KKT_CORRECTION_ROUNDS_FIXED = 1


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
            groups = [
                np.asarray(group, dtype=np.int64).reshape(-1)
                for group in component_variant_indices
            ]
            self.n_grm = len(groups)
            self.m_per_grm = np.array([group.size for group in groups], dtype=np.int64)
            if groups:
                self._source_variant_indices = np.concatenate(groups, axis=0)
            else:
                self._source_variant_indices = np.empty((0,), dtype=np.int64)
        else:
            self.n_grm = len(streamers)
            self.m_per_grm = np.array([st.m for st in streamers], dtype=np.int64)
        self.offsets = np.zeros(self.n_grm + 1, dtype=np.int64)
        np.cumsum(self.m_per_grm, out=self.offsets[1:])
        self.m_total = int(self.offsets[-1])

    def global_to_local(
        self, global_idx: np.ndarray
    ) -> list[tuple[int, np.ndarray, np.ndarray]]:
        """
        Convert global SNP indices to per-GRM groups.

        Returns list of (grm_idx, local_indices, positions_in_input) tuples,
        where positions_in_input are the positions in the original global_idx
        array so results can be assembled back.
        """
        gidx = np.asarray(global_idx, dtype=np.int64)
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

        scores = np.zeros(self.m_total, dtype=np.float64)
        for g, st in enumerate(self.streamers):
            off = int(self.offsets[g])
            block = np.asarray(
                st.xtv(u_jax, normalize=normalize), dtype=np.float64
            )
            scores[off : off + st.m] = block
        return scores

    def extract_standardized_columns(
        self, global_idx: np.ndarray
    ) -> np.ndarray:
        """
        Extract standardized genotype columns for global SNP indices.
        Dispatches to the correct streamer for each GRM and assembles
        columns in the original order.
        """
        gidx = np.asarray(global_idx, dtype=np.int64)
        if self._partitioned_single_streamer:
            return self.streamers[0].extract_standardized_columns(gidx)
        n = self.streamers[0].n
        out = np.empty((n, gidx.size), dtype=np.float32)
        for g, local, positions in self.global_to_local(gidx):
            cols = self.streamers[g].extract_standardized_columns(local)
            out[:, positions] = cols
        return out

    def source_variant_indices(self, global_idx: np.ndarray) -> np.ndarray:
        gidx = np.asarray(global_idx, dtype=np.int64)
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
    p.add_argument(
        "--kkt-pcg-tol",
        type=float,
        default=(
            float(env("KKT_PCG_TOL", ""))
            if env("KKT_PCG_TOL", "").strip()
            else None
        ),
        help=(
            "PCG tolerance used only to recheck a failed coarse KKT "
            "certificate. By default it is the smaller of the KKT tolerance "
            "scale and --pcg-tol/50. The ordinary outer solves keep using "
            "--pcg-tol."
        ),
    )
    p.add_argument("--pcg-ridge", type=float, default=float(env("PCG_RIDGE", "1e-6")))
    p.add_argument("--max-pcg-iters", type=int, default=int(env("MAX_PCG_ITERS", "400")))
    p.add_argument("--outer-max", type=int, default=6)
    p.add_argument("--screen-topk", type=int, default=2000)
    p.add_argument("--candidate-k", type=int, default=256)
    p.add_argument("--vc-rel-tol", type=float, default=1e-2)
    p.add_argument("--support-stable-rounds", type=int, default=1)
    p.add_argument("--lasso-lam-min-ratio", type=float, default=0.05)
    p.add_argument("--lasso-n-lambda", type=int, default=60)
    p.add_argument("--lasso-ebic-gamma", type=float, default=0.5)
    p.add_argument("--lasso-ebic-early-stop", action="store_true")
    p.add_argument("--no-lasso-ebic-early-stop", dest="lasso_ebic_early_stop", action="store_false")
    p.set_defaults(lasso_ebic_early_stop=True)
    p.add_argument("--lasso-ebic-es-min-delta", type=float, default=0.0)
    p.add_argument("--lasso-cd-max-iter", type=int, default=2000)
    p.add_argument("--lasso-cd-tol", type=float, default=1e-6)
    p.add_argument("--lasso-active-set-period", type=int, default=5)
    p.add_argument("--lasso-ridge", type=float, default=1e-6)
    p.add_argument("--proj-ridge", type=float, default=1e-6)
    p.add_argument(
        "--ebic-p-mode",
        choices=["candidate", "full"],
        default="full",
        help=(
            "Model-space size used by the EBIC combinatorial penalty. "
            "The default 'full' matches genome-wide KKT certification; "
            "'candidate' is retained only for screened-EBIC sensitivity analyses."
        ),
    )
    p.add_argument(
        "--kkt-tol",
        type=float,
        default=1e-4,
        help="Absolute tolerance for global inactive-SNP LASSO KKT checks.",
    )
    p.add_argument(
        "--kkt-rel-tol",
        type=float,
        default=1e-4,
        help="Relative tolerance, multiplied by max(1, lambda), for global KKT checks.",
    )
    p.add_argument(
        "--kkt-add-topk",
        type=int,
        default=256,
        help="Maximum number of outside-candidate KKT violators added per refinement round.",
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
    if args.kkt_pcg_tol is None:
        positive_certificate_tolerances = [
            value
            for value in (
                float(args.kkt_tol),
                float(args.kkt_rel_tol),
            )
            if value > 0.0
        ]
        certificate_scale = (
            min(positive_certificate_tolerances)
            if positive_certificate_tolerances
            else 1e-6
        )
        args.kkt_pcg_tol = min(
            float(args.pcg_tol) / 50.0, certificate_scale
        )
    return args


def _max_rel_change(new_v: np.ndarray, old_v: np.ndarray) -> float:
    denom = np.maximum(np.abs(old_v), 1e-6)
    return float(np.max(np.abs(new_v - old_v) / denom))


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


def _intercept_contrast_fixed_effect(n_samples: int) -> np.ndarray:
    """Return the nuisance column whose REML contrasts remove the intercept."""
    n_samples = int(n_samples)
    if n_samples < 2:
        raise ValueError("Intercept contrasts require at least two samples.")
    return np.ones((n_samples, 1), dtype=np.float32)


def _fit_intercept_contrast_residual_ml(
    fitter,
    residual_standardized: np.ndarray,
    theta_init: np.ndarray,
    *,
    h2_init: float,
):
    """Fit residual covariance and return it on the standardized-phenotype scale.

    Core REML always standardizes its response. The Lasso residual already has
    units of the globally standardized phenotype, so the unit-residual-scale
    variance estimates are multiplied by the residual variance before they are
    combined with sparse quadratic terms.
    """
    residual = np.asarray(residual_standardized, dtype=np.float32).reshape(-1)
    theta = np.asarray(theta_init, dtype=np.float32).reshape(-1)
    _, residual_scale = _phenotype_standardization_stats(residual)
    variance_scale = float(residual_scale) ** 2
    fit_result = fitter.fit_infinitesimal(
        jnp.asarray(residual, dtype=jnp.float32),
        jnp.asarray(
            _intercept_contrast_fixed_effect(residual.size),
            dtype=jnp.float32,
        ),
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


def _terminal_sparse_pair_may_stop(
    *,
    terminal_verification: bool,
    stable_candidate: bool,
    returned_covariance_kkt: dict[str, float | bool],
) -> bool:
    """Require all terminal fixed-point and full-p KKT certificates."""
    return bool(
        terminal_verification
        and stable_candidate
        and returned_covariance_kkt.get("passed", False)
    )


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
    alpha_theta_pair_certified: bool,
    lasso_quadratics_available: bool,
    selected_span_refit_ok: bool,
    lasso_estimator_values: np.ndarray,
    selected_support_estimator_values: np.ndarray,
) -> dict[str, object]:
    """Validate the Lasso and selected-support estimator branches separately.

    The selected-support REML--GLS refit is downstream of support selection,
    but its numerical failure must not erase valid Lasso plug-in and CHIVE
    estimates.  No ordinary-REML value is substituted into either branch.
    """
    lasso_values = np.asarray(
        lasso_estimator_values, dtype=np.float64
    ).reshape(-1)
    selected_values = np.asarray(
        selected_support_estimator_values, dtype=np.float64
    ).reshape(-1)
    lasso_outputs_finite = bool(
        lasso_values.size == 2 and np.all(np.isfinite(lasso_values))
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
        lasso_branch_valid and selected_support_branch_valid
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
            lasso_outputs_finite and selected_outputs_finite
        ),
        "combined_invalid_reasons": combined_reasons,
    }


def _sparse_prediction_branch_names(
    *,
    lasso_branch_valid: bool,
    selected_support_refit_branch_valid: bool,
) -> list[str]:
    """Return independently available prediction branches in output order."""
    if selected_support_refit_branch_valid and not lasso_branch_valid:
        raise ValueError(
            "A selected-support prediction requires a valid Lasso branch."
        )
    names = ["lasso"] if lasso_branch_valid else []
    if selected_support_refit_branch_valid:
        names.append("selected_span")
    return names


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


def _strict_kkt_refinement_action(
    *,
    full_certificate_passed: bool,
    candidate_certificate_passed: bool,
    outside_violator_count: int,
) -> str:
    """Choose the only mathematically useful response to a strict scan."""
    if bool(full_certificate_passed):
        return "accept"
    if not bool(candidate_certificate_passed):
        return "rerun_path_strict"
    if int(outside_violator_count) > 0:
        return "expand_candidate_strict"
    return "inconsistent_full_certificate"


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
) -> np.ndarray:
    """Keep the complete previous support, then fill from the new screen.

    ``candidate_target`` is a minimum seed size, not a cap.  In particular,
    a valid Lasso support is never truncated merely because it is larger than
    the initial screening target.
    """
    target = int(candidate_target)
    if target <= 0:
        raise ValueError("candidate_target must be > 0.")

    keep_list: list[int] = []
    seen: set[int] = set()
    for value in np.asarray(previous_support, dtype=np.int64).reshape(-1):
        index = int(value)
        if index not in seen:
            keep_list.append(index)
            seen.add(index)
    for value in np.asarray(screened_indices, dtype=np.int64).reshape(-1):
        if len(keep_list) >= target:
            break
        index = int(value)
        if index not in seen:
            keep_list.append(index)
            seen.add(index)
    return np.asarray(sorted(keep_list), dtype=np.int64)


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
    if int(args.support_stable_rounds) < 1:
        raise SystemExit("support-stable-rounds must be >= 1.")
    if float(args.vc_rel_tol) <= 0.0:
        raise SystemExit("vc-rel-tol must be > 0.")
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
        or not np.isfinite(float(args.kkt_pcg_tol))
        or float(args.pcg_tol) <= 0.0
        or float(args.kkt_pcg_tol) <= 0.0
    ):
        raise SystemExit("PCG tolerances must be finite and > 0.")
    if float(args.kkt_pcg_tol) >= float(args.pcg_tol):
        raise SystemExit("kkt-pcg-tol must be < pcg-tol.")

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
    if component_variant_indices:
        if len(bed_list) > 1:
            raise SystemExit("single-source component partitioning cannot be combined with multiple BED prefixes.")
        if not (len(bed_list) == 1 or pgen_prefix):
            raise SystemExit("single-source component partitioning requires exactly one genotype input.")
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

    # Match fit_reml's trace-calibrated default initialization.
    h2_init_default = 0.5
    trace_sum = float(np.sum(genetic_trace_atoms))
    if trace_sum <= 0.0:
        raise RuntimeError("Sparse REML requires at least one positive-trace GRM.")
    theta_g0 = np.where(
        genetic_trace_atoms > 0.0,
        h2_init_default / trace_sum,
        0.0,
    )
    theta_e0 = np.array([1.0 - h2_init_default], dtype=np.float64)
    theta = np.concatenate([theta_g0, theta_e0], axis=0)
    fitter._ensure_projected_core_precond_ready(
        ops,
        var_components_init=jnp.asarray(theta, dtype=jnp.float32),
    )
    logger.info(
        "[INFO] init theta (fit_reml default) @ %s: "
        "%s",
        datetime.now().isoformat(timespec='seconds'), theta.tolist(),
    )

    path_cfg = LassoPathConfig(
        lam_min_ratio=args.lasso_lam_min_ratio, n_lambda=args.lasso_n_lambda,
        ebic_gamma=args.lasso_ebic_gamma, max_cd_iter=args.lasso_cd_max_iter,
        ebic_early_stop=args.lasso_ebic_early_stop,
        ebic_early_stop_patience=LASSO_EBIC_ES_PATIENCE_FIXED,
        ebic_early_stop_min_delta=args.lasso_ebic_es_min_delta,
        cd_tol=args.lasso_cd_tol, active_set_period=args.lasso_active_set_period,
        kkt_abs_tol=args.kkt_tol, kkt_rel_tol=args.kkt_rel_tol,
        verbose=args.verbose,
    )

    support = np.array([], dtype=np.int64)
    stable_rounds = 0
    history: list[dict] = []

    warm_screen = None
    warm_z_dict: dict[int, np.ndarray] = {}

    final_candidate = np.array([], dtype=np.int64)
    final_lasso = None
    theta_lasso = theta.copy()
    n_samples = y_np.shape[0]
    variance_blocks_completed = 0
    outer_converged = False
    outer_stop_reason = "outer_max"
    verification_pending = False
    terminal_kkt_corrections_used = 0
    lasso_ml_stop_reason = ""
    penalized_failure_reason = None

    # ---- Precompute loop-invariant B_screen = [y | covar] on device --------
    screen_parts = [y_np[:, None]]
    if covar_np is not None:
        screen_parts.append(covar_np)
    B_screen_np = np.concatenate(screen_parts, axis=1).astype(np.float32, copy=False)
    B_screen_dev = jnp.asarray(B_screen_np, dtype=jnp.float32)
    n_screen = B_screen_np.shape[1]

    def _certify_returned_lasso_pair(
        theta_values: np.ndarray,
        candidate_indices: np.ndarray,
        lasso_fit: dict,
        *,
        stage: str,
    ) -> tuple[dict[str, object], str | None]:
        """Two-stage full-p KKT certificate at the supplied covariance."""
        failed = {
            "passed": False,
            "tolerance": float("nan"),
            "max_active_error": float("inf"),
            "max_inactive_excess": float("inf"),
            "strict_recheck_triggered": False,
            "decision_precision": "unavailable",
            "coarse": None,
            "strict": None,
        }
        try:
            candidate_arr = np.asarray(
                candidate_indices, dtype=np.int64
            ).reshape(-1)
            beta_snp = np.asarray(
                lasso_fit["beta_snp"], dtype=np.float64
            ).reshape(-1)
            if beta_snp.size != candidate_arr.size:
                raise RuntimeError(
                    "Final Lasso coefficient/candidate sizes do not match."
                )
            z_candidate = grm_index.extract_standardized_columns(
                candidate_arr
            ).astype(np.float32, copy=False)
            residual = _lasso_residual(
                y=y_np,
                covar=covar_np,
                geno=z_candidate,
                beta_cov=np.asarray(
                    lasso_fit.get("beta_cov", np.empty((0,))),
                    dtype=np.float64,
                ),
                beta_snp=beta_snp,
            )
            theta_arr = np.asarray(
                theta_values, dtype=np.float64
            ).reshape(-1)
            hv_returned = fitter._make_hv(
                ops,
                jnp.asarray(theta_arr[:-1], dtype=jnp.float32),
                jnp.asarray(theta_arr[-1], dtype=jnp.float32),
            )
            precond_returned = fitter._make_effect_precond(
                ops,
                jnp.asarray(theta_arr[:-1], dtype=jnp.float32),
                jnp.asarray(theta_arr[-1], dtype=jnp.float32),
            )
            coarse_record = {
                "pcg_tol": float(args.pcg_tol),
                "pcg_iters": None,
                "pcg_reported_res": None,
                "pcg_true_res": None,
                "certificate": None,
                "error": None,
            }
            failed["coarse"] = coarse_record
            residual_rhs = jnp.asarray(
                residual[:, None], dtype=jnp.float32
            )
            sol, rel_res, iters = pcg_solve(
                hv_returned,
                residual_rhs,
                M=precond_returned,
                tol=args.pcg_tol,
                maxiter=args.max_pcg_iters,
            )
            coarse_record["pcg_iters"] = int(iters)
            coarse_record["pcg_reported_res"] = float(
                np.asarray(rel_res)
            )
            coarse_true_res = _true_pcg_relative_residual(
                hv_returned, residual_rhs, sol
            )
            coarse_record["pcg_true_res"] = float(coarse_true_res)
            _require_pcg_converged(
                coarse_true_res,
                tol=args.pcg_tol,
                iters=iters,
                maxiter=args.max_pcg_iters,
                stage=stage,
            )
            score = grm_index.xtv_all(sol[:, 0], normalize=False)
            beta_global = np.zeros(
                grm_index.m_total, dtype=np.float64
            )
            beta_global[candidate_arr] = beta_snp
            coarse_certificate = _lasso_kkt_certificate_from_scores(
                score=score,
                beta=beta_global,
                lam=float(lasso_fit["lam"]),
                abs_tol=float(args.kkt_tol),
                rel_tol=float(args.kkt_rel_tol),
            )
            coarse_record["certificate"] = dict(coarse_certificate)
            if bool(coarse_certificate["passed"]):
                return (
                    {
                        **coarse_certificate,
                        "strict_recheck_triggered": False,
                        "decision_precision": "coarse",
                        "coarse": coarse_record,
                        "strict": None,
                    },
                    None,
                )

            # A failed coarse certificate is not final.  Warm-start a stricter
            # residual solve from the coarse solution and recompute all-marker
            # signed scores.  The Lasso path/Gram solve itself is unchanged.
            failed["strict_recheck_triggered"] = True
            strict_record = {
                "pcg_tol": float(args.kkt_pcg_tol),
                "pcg_iters": None,
                "pcg_reported_res": None,
                "pcg_true_res": None,
                "certificate": None,
                "error": None,
            }
            failed["strict"] = strict_record
            try:
                sol_strict, rel_res_strict, iters_strict = pcg_solve(
                    hv_returned,
                    residual_rhs,
                    M=precond_returned,
                    tol=args.kkt_pcg_tol,
                    maxiter=args.max_pcg_iters,
                    X0=sol,
                )
                strict_record["pcg_iters"] = int(iters_strict)
                strict_record["pcg_reported_res"] = float(
                    np.asarray(rel_res_strict)
                )
                strict_true_res = _true_pcg_relative_residual(
                    hv_returned, residual_rhs, sol_strict
                )
                strict_record["pcg_true_res"] = float(strict_true_res)
                _require_pcg_converged(
                    strict_true_res,
                    tol=args.kkt_pcg_tol,
                    iters=iters_strict,
                    maxiter=args.max_pcg_iters,
                    stage=f"{stage} strict PCG recheck",
                )
                strict_score = grm_index.xtv_all(
                    sol_strict[:, 0], normalize=False
                )
                strict_certificate = _lasso_kkt_certificate_from_scores(
                    score=strict_score,
                    beta=beta_global,
                    lam=float(lasso_fit["lam"]),
                    abs_tol=float(args.kkt_tol),
                    rel_tol=float(args.kkt_rel_tol),
                )
                strict_record["certificate"] = dict(strict_certificate)
                return (
                    {
                        **strict_certificate,
                        "strict_recheck_triggered": True,
                        "decision_precision": "strict",
                        "coarse": coarse_record,
                        "strict": strict_record,
                    },
                    None,
                )
            except (FloatingPointError, RuntimeError, ValueError) as error:
                strict_record["error"] = str(error)
                failed["decision_precision"] = "strict_failed"
                return failed, str(error)
        except (FloatingPointError, RuntimeError, ValueError) as error:
            if isinstance(failed.get("coarse"), dict):
                failed["coarse"]["error"] = str(error)
            failed["decision_precision"] = "coarse_failed"
            return failed, str(error)

    returned_covariance_kkt = {
        "passed": False,
        "tolerance": float("nan"),
        "max_active_error": float("inf"),
        "max_inactive_excess": float("inf"),
        "strict_recheck_triggered": False,
        "decision_precision": "unavailable",
        "coarse": None,
        "strict": None,
    }
    returned_covariance_kkt_error = None

    outer = 0
    while outer < int(args.outer_max) or verification_pending:
        outer += 1
        terminal_verification = bool(verification_pending)
        verification_pending = False
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

        # ``candidate_target`` controls only the new screen seed.  The entire
        # previously selected support is retained even when it is larger.
        candidate = _build_lasso_candidate(
            previous_support=support,
            screened_indices=candidate_seed,
            candidate_target=candidate_target,
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
        strict_kkt_locked = False
        strict_screen_solution = None
        strict_screen_record = None
        for kkt_round in range(1, max_kkt_rounds + 1):
            strict_at_round_start = bool(strict_kkt_locked)
            path_pcg_tol = (
                float(args.kkt_pcg_tol)
                if strict_at_round_start
                else float(args.pcg_tol)
            )

            # Once a strict score confirms any KKT violation, every remaining
            # refinement round at this theta uses one coherent strict inverse
            # for H^{-1}[y,C,Z].  We never combine a strict residual score with
            # a newly rebuilt coarse Gram/path.
            if strict_at_round_start and strict_screen_solution is None:
                strict_screen_record = {
                    "pcg_tol": float(args.kkt_pcg_tol),
                    "pcg_reported_res": None,
                    "pcg_true_res": None,
                    "pcg_iters": None,
                    "error": None,
                }
                try:
                    (
                        strict_screen_solution,
                        strict_screen_reported_res,
                        strict_screen_iters,
                    ) = pcg_solve(
                        hv,
                        B_screen_dev,
                        M=precond,
                        tol=args.kkt_pcg_tol,
                        maxiter=args.max_pcg_iters,
                        X0=sol_screen,
                    )
                    strict_screen_record["pcg_reported_res"] = float(
                        np.asarray(strict_screen_reported_res)
                    )
                    strict_screen_record["pcg_iters"] = int(
                        strict_screen_iters
                    )
                    strict_screen_true_res = _true_pcg_relative_residual(
                        hv, B_screen_dev, strict_screen_solution
                    )
                    strict_screen_record["pcg_true_res"] = float(
                        strict_screen_true_res
                    )
                    _require_pcg_converged(
                        strict_screen_true_res,
                        tol=args.kkt_pcg_tol,
                        iters=strict_screen_iters,
                        maxiter=args.max_pcg_iters,
                        stage=(
                            f"outer {outer} KKT round {kkt_round} "
                            "strict Hinv[y,C] rebuild"
                        ),
                    )
                except (FloatingPointError, RuntimeError, ValueError) as error:
                    strict_screen_record["error"] = str(error)
                    kkt_trace.append(
                        {
                            "round": int(kkt_round),
                            "candidate_size": int(candidate.size),
                            "strict_mode_locked_at_round_start": True,
                            "decision": "strict_linear_solve_failed",
                            "strict_screen": strict_screen_record,
                        }
                    )
                    penalized_block_failure = (
                        "Strict PCG Hinv[y,C] rebuild failed as a "
                        f"linear-solve certification error: {error}"
                    )
                    break

            screen_solution_for_path = (
                strict_screen_solution
                if strict_at_round_start
                else sol_screen
            )
            Hinv_y_for_path = np.asarray(
                screen_solution_for_path[:, 0], dtype=np.float64
            )
            Hinv_covar_for_path = None
            if covar_np is not None and covar_np.shape[1] > 0:
                Hinv_covar_for_path = np.asarray(
                    screen_solution_for_path[:, 1:n_screen],
                    dtype=np.float64,
                )

            # Z_cand PCG with dictionary warm-start. Candidate may grow, and
            # after strict locking the coarse dictionary is merely X0 for the
            # strict solve; all returned columns are then replaced by strict
            # solutions.
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
                candidate_true_res = (
                    _true_pcg_relative_residual(hv, B_z, sol_z)
                    if strict_at_round_start
                    else float(np.asarray(res_all))
                )
                _require_pcg_converged(
                    candidate_true_res,
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
                        "strict_mode_locked_at_round_start": strict_at_round_start,
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

            p_for_ebic = (
                int(candidate.size)
                if args.ebic_p_mode == "candidate"
                else grm_index.m_total
            )
            try:
                lasso = fit_weighted_lasso_with_covariates(
                    y=y_np,
                    covar=covar_np,
                    geno=Z_cand,
                    Hinv_y=Hinv_y_for_path,
                    Hinv_covar=Hinv_covar_for_path,
                    Hinv_geno=sol_z_np,
                    p_total=p_for_ebic,
                    cfg=path_cfg,
                    ridge=args.lasso_ridge,
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
            if not bool(best_path.get("converged", False)):
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
                _require_pcg_converged(
                    true_res_kkt,
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
                        "strict_mode_locked_at_round_start": strict_at_round_start,
                        "path_pcg_tol": path_pcg_tol,
                        "decision": "score_linear_solve_failed",
                        "error": str(error),
                    }
                )
                penalized_block_failure = (
                    "KKT score PCG failed as a linear-solve certification "
                    f"error: {error}"
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
            current_record = {
                "pcg_tol": path_pcg_tol,
                "pcg_reported_res": float(np.asarray(res_kkt)),
                "pcg_true_res": float(true_res_kkt),
                "pcg_iters": int(it_kkt),
                "threshold": kkt_threshold,
                "max_outside_score": max_outside_score,
                "n_outside_violators": int(violators.size),
                "candidate_certificate": dict(
                    partitioned["candidate_certificate"]
                ),
                "full_certificate": dict(
                    partitioned["full_certificate"]
                ),
                "error": None,
            }

            coarse_record = None
            strict_record = None
            strict_recheck_triggered = False
            if strict_at_round_start:
                strict_record = current_record
                decision_precision = "strict_locked"
            else:
                coarse_record = current_record
                decision_precision = "coarse"

            # The first-stage decision is the signed full-p certificate, not
            # merely the outside-candidate scan.  A discrepancy on an active
            # or candidate-inactive coordinate also triggers the strict
            # recheck because the direct residual solve and the Gram solve are
            # independent finite-tolerance PCG calculations.
            coarse_full_passed = bool(
                partitioned["full_certificate"]["passed"]
            )
            if not strict_at_round_start and coarse_full_passed:
                action = "accept"
            elif not strict_at_round_start:
                strict_recheck_triggered = True
                strict_kkt_locked = True
                strict_record = {
                    "pcg_tol": float(args.kkt_pcg_tol),
                    "pcg_reported_res": None,
                    "pcg_true_res": None,
                    "pcg_iters": None,
                    "threshold": None,
                    "max_outside_score": None,
                    "n_outside_violators": None,
                    "candidate_certificate": None,
                    "full_certificate": None,
                    "error": None,
                }
                try:
                    (
                        sol_resid_strict,
                        res_kkt_strict,
                        it_kkt_strict,
                    ) = pcg_solve(
                        hv,
                        residual_rhs,
                        M=precond,
                        tol=args.kkt_pcg_tol,
                        maxiter=args.max_pcg_iters,
                        X0=sol_resid,
                    )
                    strict_record["pcg_reported_res"] = float(
                        np.asarray(res_kkt_strict)
                    )
                    strict_record["pcg_iters"] = int(it_kkt_strict)
                    strict_true_res = _true_pcg_relative_residual(
                        hv, residual_rhs, sol_resid_strict
                    )
                    strict_record["pcg_true_res"] = float(strict_true_res)
                    _require_pcg_converged(
                        strict_true_res,
                        tol=args.kkt_pcg_tol,
                        iters=it_kkt_strict,
                        maxiter=args.max_pcg_iters,
                        stage=(
                            f"outer {outer} KKT round {kkt_round} "
                            "strict residual recheck"
                        ),
                    )
                    strict_score_signed = np.asarray(
                        grm_index.xtv_all(
                            sol_resid_strict[:, 0], normalize=False
                        ),
                        dtype=np.float64,
                    )
                    partitioned = _partitioned_lasso_kkt_from_scores(
                        score=strict_score_signed,
                        candidate=candidate,
                        beta_candidate=beta_snp,
                        lam=float(lasso["lam"]),
                        abs_tol=float(args.kkt_tol),
                        rel_tol=float(args.kkt_rel_tol),
                    )
                    violators = np.asarray(
                        partitioned["outside_violators"],
                        dtype=np.int64,
                    )
                    max_outside_score = float(
                        partitioned["max_outside_score"]
                    )
                    kkt_threshold = float(partitioned["threshold"])
                    score_kkt_decision = np.abs(strict_score_signed)
                    strict_record.update(
                        {
                            "threshold": kkt_threshold,
                            "max_outside_score": max_outside_score,
                            "n_outside_violators": int(violators.size),
                            "candidate_certificate": dict(
                                partitioned["candidate_certificate"]
                            ),
                            "full_certificate": dict(
                                partitioned["full_certificate"]
                            ),
                        }
                    )
                    decision_precision = "strict_recheck"
                except (FloatingPointError, RuntimeError, ValueError) as error:
                    strict_record["error"] = str(error)
                    kkt_trace.append(
                        {
                            "round": int(kkt_round),
                            "candidate_size": int(candidate.size),
                            "support_size": int(support_new.size),
                            "lambda": float(lasso["lam"]),
                            "strict_mode_locked_at_round_start": False,
                            "strict_recheck_triggered": True,
                            "decision_precision": "strict_failed",
                            "decision": "strict_linear_solve_failed",
                            "path_pcg_tol": path_pcg_tol,
                            "coarse": coarse_record,
                            "strict": strict_record,
                        }
                    )
                    penalized_block_failure = (
                        "Strict PCG KKT recheck failed as a linear-solve "
                        f"certification error: {error}"
                    )
                    break
                action = _strict_kkt_refinement_action(
                    full_certificate_passed=bool(
                        partitioned["full_certificate"]["passed"]
                    ),
                    candidate_certificate_passed=bool(
                        partitioned["candidate_certificate"]["passed"]
                    ),
                    outside_violator_count=int(violators.size),
                )
            else:
                action = _strict_kkt_refinement_action(
                    full_certificate_passed=bool(
                        partitioned["full_certificate"]["passed"]
                    ),
                    candidate_certificate_passed=bool(
                        partitioned["candidate_certificate"]["passed"]
                    ),
                    outside_violator_count=int(violators.size),
                )
                if action == "rerun_path_strict":
                    # The path was already rebuilt from strict H^{-1}[y,C,Z]
                    # at this theta.  Repeating the identical strict solve
                    # cannot repair a remaining disagreement between its Gram
                    # equations and an independent strict residual solve.
                    action = "strict_candidate_inconsistency"

            n_viol = int(violators.size)
            kkt_trace.append(
                {
                    "round": int(kkt_round),
                    "candidate_size": int(candidate.size),
                    "support_size": int(support_new.size),
                    "lambda": float(lasso["lam"]),
                    "threshold": float(kkt_threshold),
                    "max_outside_score": float(max_outside_score),
                    "n_violators": n_viol,
                    "strict_mode_locked_at_round_start": strict_at_round_start,
                    "strict_mode_locked_after_round": bool(strict_kkt_locked),
                    "strict_recheck_triggered": strict_recheck_triggered,
                    "decision_precision": decision_precision,
                    "decision": action,
                    "path_pcg_tol": path_pcg_tol,
                    "candidate_pcg_reported_res": float(
                        np.asarray(res_all)
                    ),
                    "candidate_pcg_true_res": float(candidate_true_res),
                    "candidate_pcg_iters": int(it_all),
                    "strict_screen": strict_screen_record,
                    "coarse": coarse_record,
                    "strict": strict_record,
                }
            )
            logger.info(
                "[outer %s kkt %s] cand=%s active=%s lam=%.3e "
                "max_outside=%.3e threshold=%.3e violators=%s "
                "precision=%s decision=%s strict_locked=%s",
                outer,
                kkt_round,
                int(candidate.size),
                int(support_new.size),
                float(lasso["lam"]),
                max_outside_score,
                kkt_threshold,
                n_viol,
                decision_precision,
                action,
                strict_kkt_locked,
            )

            if action == "accept":
                certified_kkt = True
                break
            if action == "strict_candidate_inconsistency":
                penalized_block_failure = (
                    "Strict PCG Lasso path and strict direct residual score "
                    "disagree on a candidate coordinate; global KKT cannot "
                    "be certified at the requested tolerance."
                )
                break
            if action == "inconsistent_full_certificate":
                penalized_block_failure = (
                    "The strict full-p KKT certificate failed without an "
                    "identified candidate-coordinate or outside-coordinate "
                    "violation; numerical certification is inconsistent."
                )
                break
            if action == "rerun_path_strict":
                # The strict score disagrees on a coordinate already present
                # in the candidate. Candidate expansion cannot repair that;
                # the next round rebuilds the complete strict Gram/path.
                continue

            n_add = min(int(args.kkt_add_topk), n_viol)
            add_idx = violators[
                np.argsort(score_kkt_decision[violators])[-n_add:]
            ]
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
                    "best_ebic": (
                        float(lasso["best_ebic"])
                        if lasso is not None
                        else None
                    ),
                    "kkt_certified": False,
                    "kkt_trace": kkt_trace,
                    "verification": terminal_verification,
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

        support_same = bool(np.array_equal(support_new, support))

        if args.verbose:
            logger.info(
                "[outer %s] lasso_select: p_mode=%s "
                "candidate=%s k_selected=%s kkt_certified=%s",
                outer, args.ebic_p_mode, int(candidate.size), int(active_local.size),
                bool(certified_kkt),
            )

        # ---- Step 5: contrast residual-ML variance block ------------------
        # Hold the complete Lasso mean fixed and maximize the Gaussian
        # likelihood in the subspace orthogonal to the intercept, exactly as
        # in the manuscript's m=n-1 contrast coordinates.  Passing a constant
        # fixed-effect column to fit_infinitesimal is algebraically equivalent
        # to that contrast likelihood.  Omitting it would retain a zero-energy
        # constant mode of the centered GRM and spuriously drive residual
        # variance toward its numerical floor.
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
        residual_standardized = residual_raw / float(phenotype_scale)
        try:
            ml_res = _fit_intercept_contrast_residual_ml(
                fitter,
                residual_standardized,
                theta,
                h2_init=_trace_weighted_h2(theta),
            )
            theta_new, lasso_ml_stop_reason = _accepted_reml_theta(
                ml_res,
                expected_components=n_grm + 1,
                stage=f"outer {outer} Lasso residual-ML block",
            )
            # ``ll_down`` is a converged no-update block: its downhill
            # candidate is rejected and the previous theta is retained.
            variance_blocks_completed += 1
        except (FloatingPointError, RuntimeError, ValueError) as error:
            penalized_failure_reason = str(error)
            outer_stop_reason = "residual_ml_failed"
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
                    "best_ebic": float(lasso["best_ebic"]),
                    "kkt_certified": bool(certified_kkt),
                    "kkt_trace": kkt_trace,
                    "verification": terminal_verification,
                    "variance_update": "failed",
                    "failure": penalized_failure_reason,
                }
            )
            logger.warning(
                "[WARN] residual-ML block rejected at outer=%s: %s",
                outer,
                penalized_failure_reason,
            )
            break
        if args.verbose:
            logger.info(
                "[outer %s] residual_ml_init_theta=%s stop=%s",
                outer,
                theta.tolist(),
                lasso_ml_stop_reason,
            )

        # ---- Convergence checks ----
        vc_rel = _max_rel_change(theta_new, theta)
        if support_same and vc_rel < float(args.vc_rel_tol):
            stable_rounds += 1
        else:
            stable_rounds = 0

        history.append({
            "outer": outer,
            "pcg_screen_iters": int(it_screen),
            "pcg_screen_res": float(np.asarray(res_screen)),
            "pcg_all_iters": int(it_all),
            "pcg_all_res": float(np.asarray(res_all)),
            "theta": theta_new.tolist(),
            "support_size": int(support_new.size),
            "support_same": support_same,
            "vc_rel": float(vc_rel),
            "lam": float(lasso["lam"]),
            "best_ebic": float(lasso["best_ebic"]),
            "kkt_certified": bool(certified_kkt),
            "kkt_trace": kkt_trace,
            "verification": terminal_verification,
            "variance_update": (
                "intercept_contrast_residual_ml_no_update"
                if lasso_ml_stop_reason == "ll_down"
                else "intercept_contrast_residual_ml"
            ),
            "variance_stop_reason": lasso_ml_stop_reason,
            "variance_step_rejected": bool(
                lasso_ml_stop_reason == "ll_down"
            ),
        })

        logger.info(
            "[outer %s] pcg_screen=%s pcg_all=%s "
            "cand=%s active=%s kkt_rounds=%s certified=%s "
            "lam=%.3e ebic=%.4e "
            "vc_rel=%.3e support_same=%s "
            "iter_time=%.1fs",
            outer, int(it_screen), int(it_all),
            int(candidate.size), int(support_new.size), len(kkt_trace), bool(certified_kkt),
            float(lasso['lam']), float(lasso['best_ebic']),
            vc_rel, support_same,
            time.time() - iter_t0,
        )

        theta = theta_new
        support = support_new
        final_candidate = candidate
        final_lasso = lasso

        stable_candidate = bool(
            variance_blocks_completed >= 2
            and stable_rounds >= int(args.support_stable_rounds)
        )
        if terminal_verification and stable_candidate:
            (
                returned_covariance_kkt,
                returned_covariance_kkt_error,
            ) = _certify_returned_lasso_pair(
                theta_new,
                candidate,
                lasso,
                stage=(
                    f"outer {outer} returned-covariance full-p KKT "
                    "verification"
                ),
            )
            history[-1]["returned_covariance_kkt"] = dict(
                returned_covariance_kkt
            )
            history[-1]["returned_covariance_kkt_error"] = (
                returned_covariance_kkt_error
            )
            logger.info(
                "[outer %s returned covariance KKT] lambda=%.6e "
                "tolerance=%.6e max_active_error=%.6e "
                "max_inactive_excess=%.6e passed=%s "
                "precision=%s strict_recheck=%s error=%s",
                outer,
                float(lasso["lam"]),
                float(returned_covariance_kkt["tolerance"]),
                float(returned_covariance_kkt["max_active_error"]),
                float(returned_covariance_kkt["max_inactive_excess"]),
                bool(returned_covariance_kkt["passed"]),
                returned_covariance_kkt["decision_precision"],
                bool(returned_covariance_kkt["strict_recheck_triggered"]),
                returned_covariance_kkt_error,
            )
            if _terminal_sparse_pair_may_stop(
                terminal_verification=terminal_verification,
                stable_candidate=stable_candidate,
                returned_covariance_kkt=returned_covariance_kkt,
            ):
                logger.info(
                    "[INFO] stop at outer=%s: terminal Lasso/KKT, "
                    "residual-ML, and returned-covariance full-p KKT "
                    "verification passed.",
                    outer,
                )
                outer_converged = True
                outer_stop_reason = "terminal_verification_passed"
                break
            outer_stop_reason = "returned_covariance_kkt_failed"
            if (
                terminal_kkt_corrections_used
                < TERMINAL_KKT_CORRECTION_ROUNDS_FIXED
            ):
                terminal_kkt_corrections_used += 1
                verification_pending = True
                history[-1]["terminal_kkt_correction_scheduled"] = True
                history[-1]["terminal_kkt_correction_index"] = int(
                    terminal_kkt_corrections_used
                )
                logger.warning(
                    "[WARN] terminal returned-covariance full-p KKT failed "
                    "at outer=%s; scheduling bounded correction round %s/%s "
                    "from the returned covariance.",
                    outer,
                    terminal_kkt_corrections_used,
                    TERMINAL_KKT_CORRECTION_ROUNDS_FIXED,
                )
            else:
                history[-1]["terminal_kkt_correction_scheduled"] = False
                logger.warning(
                    "[WARN] terminal returned-covariance full-p KKT failed "
                    "at outer=%s and the bounded correction budget is "
                    "exhausted; continuing only if the ordinary outer budget "
                    "remains.",
                    outer,
                )
            continue
        if terminal_verification:
            outer_stop_reason = "terminal_verification_failed"
            logger.warning(
                "[WARN] terminal verification changed support or variance "
                "beyond tolerance at outer=%s; continuing if budget remains.",
                outer,
            )
        elif stable_candidate:
            verification_pending = True
            logger.info(
                "[INFO] outer=%s reached preliminary stability; scheduling "
                "one complete Lasso/KKT plus residual-ML verification at the "
                "returned covariance.",
                outer,
            )

    # Freeze the penalized-ML branch before the downstream refit.  In
    # particular, the selected-span REML result below is not fed back into the
    # weighted Lasso or its residual-ML variance block.
    theta_lasso_ml = np.asarray(theta, dtype=np.float64).copy()
    lasso_ml_outer_converged = bool(outer_converged)
    theta_lasso_to_lasso_ml_rel = _max_rel_change(
        theta_lasso_ml, theta_lasso
    )

    # The terminal outer iteration solves the adaptive Lasso at the covariance
    # entering that iteration and then updates the residual-ML covariance.
    # Recompute the full-p KKT scores once at the returned covariance so the
    # accepted pair is certified on the same finite-tolerance scale.
    if (
        lasso_ml_outer_converged
        and final_lasso is not None
        and not bool(returned_covariance_kkt["passed"])
    ):
        (
            returned_covariance_kkt,
            returned_covariance_kkt_error,
        ) = _certify_returned_lasso_pair(
            theta_lasso_ml,
            final_candidate,
            final_lasso,
            stage="returned-covariance full-p KKT verification",
        )

    if args.verbose and final_lasso is not None:
        logger.info(
            "[returned covariance KKT] lambda=%.6e tolerance=%.6e "
            "max_active_error=%.6e max_inactive_excess=%.6e passed=%s "
            "precision=%s strict_recheck=%s theta_lasso_to_ml_rel=%.6e "
            "error=%s",
            float(final_lasso["lam"]),
            float(returned_covariance_kkt["tolerance"]),
            float(returned_covariance_kkt["max_active_error"]),
            float(returned_covariance_kkt["max_inactive_excess"]),
            bool(returned_covariance_kkt["passed"]),
            returned_covariance_kkt["decision_precision"],
            bool(returned_covariance_kkt["strict_recheck_triggered"]),
            float(theta_lasso_to_lasso_ml_rel),
            returned_covariance_kkt_error,
        )

    last_round_kkt_certified = bool(
        history and history[-1].get("kkt_certified", False)
    )
    alpha_theta_fixed_point_coherent = bool(
        lasso_ml_outer_converged
        and last_round_kkt_certified
        and bool(returned_covariance_kkt["passed"])
        and theta_lasso_to_lasso_ml_rel < float(args.vc_rel_tol)
        and penalized_failure_reason is None
    )

    # ---- One selected-span REML refit -------------------------------------
    Z_selected_for_reml = np.empty(
        (n_samples, 0), dtype=np.float32
    )
    X_selected_span = covar_np
    selected_span_basis_local = np.empty((0,), dtype=np.int64)
    selected_span_reml_iterations = 0
    selected_span_reml_stop_reason = ""
    selected_span_reml_history = []
    selected_span_refit_ok = False
    selected_span_refit_error = None
    theta_selected_span_reml = theta_lasso_ml.copy()
    if alpha_theta_fixed_point_coherent:
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
    else:
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
    q_chive_post_gls_raw = unavailable
    q_chive_post_gls_standardized = unavailable
    q_chive_term1 = unavailable
    q_chive_term2 = unavailable
    q_chive_term1_standardized = unavailable
    q_chive_term2_standardized = unavailable
    q_chive_post_gls_term1_raw = unavailable
    q_chive_post_gls_term2_raw = unavailable
    q_chive_post_gls_term1_standardized = unavailable
    q_chive_post_gls_term2_standardized = unavailable
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
        final_lasso is not None and penalized_failure_reason is None
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

    # Estimators 3 and 4, the post-GLS diagnostic, and exported refit
    # coefficients all use this one selected-span REML--GLS solution.  The
    # independent basis is mapped back to the selected support with zero
    # coefficients for numerically dependent marker columns.
    if selected_span_refit_ok:
        q_ss_gls_plugin_raw = 0.0
        q_ss_gls_plugin_standardized = 0.0
        q_ss_gls_df_corrected_raw = 0.0
        q_ss_gls_df_corrected_standardized = 0.0
        ss_gls_df_correction_raw = 0.0
        ss_gls_df_correction_standardized = 0.0
        q_chive_post_gls_raw = 0.0
        q_chive_post_gls_term1_raw = 0.0
        q_chive_post_gls_term2_raw = 0.0
        q_chive_post_gls_standardized = 0.0
        q_chive_post_gls_term1_standardized = 0.0
        q_chive_post_gls_term2_standardized = 0.0
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
            y_chive_reml = np.asarray(y_np, dtype=np.float64)
            if (
                covar_np is not None
                and covar_np.size > 0
                and beta_cov_gls.size > 0
            ):
                y_chive_reml -= (
                    np.asarray(covar_np, dtype=np.float64)
                    @ beta_cov_gls
                )
            (
                q_chive_post_gls_raw,
                q_chive_post_gls_term1_raw,
                q_chive_post_gls_term2_raw,
            ) = _chive_q_hat_given_active(
                Z_support,
                y_chive_reml,
                beta_gls_active,
            )
            q_chive_post_gls_standardized = (
                _quadratic_variance_to_reml_scale(
                    q_chive_post_gls_raw, phenotype_scale
                )
            )
            q_chive_post_gls_term1_standardized = (
                _quadratic_variance_to_reml_scale(
                    q_chive_post_gls_term1_raw, phenotype_scale
                )
            )
            q_chive_post_gls_term2_standardized = (
                _quadratic_variance_to_reml_scale(
                    q_chive_post_gls_term2_raw, phenotype_scale
                )
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
            q_chive_post_gls_raw = unavailable
            q_chive_post_gls_term1_raw = unavailable
            q_chive_post_gls_term2_raw = unavailable
            q_chive_post_gls_standardized = unavailable
            q_chive_post_gls_term1_standardized = unavailable
            q_chive_post_gls_term2_standardized = unavailable
            beta_cov_gls = np.empty((0,), dtype=np.float64)
            beta_gls_active = np.empty((0,), dtype=np.float64)
            selected_span_basis_positions = np.empty(
                (0,), dtype=np.int64
            )
            ss_gls_basis_size = 0

    # The two Lasso-row estimators use the residual-ML covariance from the
    # penalized branch.  The two selected-span estimators above use the
    # independent REML refit covariance.  ``theta_lasso`` is retained only as
    # the covariance input to the last KKT-certified sparse solve.
    h2_chive_at_lasso_theta = (
        _sparse_dense_h2(
            q_chive_standardized,
            _trace_weighted_genetic_var(theta_lasso),
            float(theta_lasso[-1]),
        )
        if lasso_quadratics_available
        else unavailable
    )
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
    h2_chive_post_gls = _sparse_dense_h2(
        q_chive_post_gls_standardized,
        theta_final_sum,
        theta_e_final,
    )
    branch_guards = _sparse_estimator_branch_guards(
        alpha_theta_pair_certified=(
            alpha_theta_fixed_point_coherent
        ),
        lasso_quadratics_available=lasso_quadratics_available,
        selected_span_refit_ok=selected_span_refit_ok,
        lasso_estimator_values=np.asarray(
            [h2_lasso_plugin, h2_chive], dtype=np.float64
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
    all_sparse_branches_valid = bool(
        branch_guards["all_four_estimators_valid"]
    )
    sparse_outputs_finite = bool(
        branch_guards["all_four_outputs_finite"]
    )
    sparse_fit_rejection_reasons = list(
        branch_guards["combined_invalid_reasons"]
    )
    if not all_sparse_branches_valid:
        logger.warning(
            "[WARN] sparse estimator branches incomplete: "
            "lasso_valid=%s selected_support_refit_valid=%s reasons=%s. "
            "No ordinary-REML value will replace a sparse estimator.",
            lasso_branch_valid,
            selected_support_refit_branch_valid,
            ",".join(sparse_fit_rejection_reasons),
        )

    h2_lasso_plugin_guarded = (
        float(h2_lasso_plugin) if lasso_branch_valid else unavailable
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
        "penalized_ml_lasso_chive" if lasso_branch_valid else "unavailable"
    )

    h2_background_selected_span_reml = (
        _trace_weighted_h2(theta_selected_span_reml)
        if selected_support_refit_branch_valid
        else unavailable
    )

    print(f"[RESULT] var_components_lasso_ml={theta_lasso_ml.tolist()}")
    print(
        "[RESULT] var_components_selected_span_reml="
        f"{theta_selected_span_reml.tolist() if selected_span_refit_ok else None}"
    )
    print(f"[RESULT] h2={h2:.6f} (primary={primary_h2_method})")
    print(
        "[RESULT] h2_background_selected_span_reml="
        f"{h2_background_selected_span_reml:.6f}"
    )
    print(
        f"[RESULT] h2_lasso_plugin={h2_lasso_plugin:.6f} "
        "(uncorrected penalized-LASSO plug-in)"
    )
    print(f"[RESULT] h2_chive={h2_chive:.6f} (penalized LASSO calibration)")
    print(
        f"[RESULT] h2_chive_post_gls={h2_chive_post_gls:.6f} "
        "(diagnostic only)"
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
    # Preserve both PCG stages in the machine-readable summary.  Non-finite
    # placeholders from an unavailable/failed certificate become JSON null.
    returned_covariance_kkt_summary = _json_safe_value(
        returned_covariance_kkt
    )

    prediction_summary = {
        "requested": bool(prediction_active),
        "status": "not_requested",
    }
    if prediction_active:
        emitted_branches = _sparse_prediction_branch_names(
            lasso_branch_valid=lasso_branch_valid,
            selected_support_refit_branch_valid=(
                selected_support_refit_branch_valid
            ),
        )
        prediction_request_metadata = {
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
            metadata_path = write_sparse_prediction_status(
                out_prefix=out_prefix,
                status="not_emitted_no_valid_branch",
                metadata={
                    **prediction_request_metadata,
                    "reason": "no_valid_sparse_prediction_branch",
                    "lasso_branch_valid": lasso_branch_valid,
                    "selected_support_refit_branch_valid": (
                        selected_support_refit_branch_valid
                    ),
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
                        "selected_span": {
                            "estimator_valid": False,
                            "output_emitted": False,
                            "invalid_reasons": list(
                                branch_guards[
                                    "selected_support_refit_branch_invalid_reasons"
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
        else:
            write_sparse_prediction_status(
                out_prefix=out_prefix,
                status="preparing",
                metadata={
                    **prediction_request_metadata,
                    "branch_outputs_emitted": False,
                },
            )
            standardization_overrides = []
            for streamer in fitter.streamers:
                if (
                    streamer._means_host is None
                    or streamer._inv_sds_host is None
                ):
                    raise RuntimeError(
                        "Sparse prediction requires retained training SNP "
                        "standardization statistics."
                    )
                standardization_overrides.append(
                    (streamer._means_host, streamer._inv_sds_host)
                )

            prediction_temp_paths: list[str] = []
            if prediction_pgen_prefix:
                prediction_fam_path = make_nonbed_input_fam(
                    pgen_prefix=prediction_pgen_prefix
                )
                prediction_temp_paths.append(prediction_fam_path)
            else:
                prediction_fam_path = prediction_bed_list[0] + ".fam"
            for path in prediction_temp_paths:
                atexit.register(cleanup_path, path)

            requested_prediction_ids = None
            if args.prediction_keep_path:
                if not os.path.exists(args.prediction_keep_path):
                    raise SystemExit(
                        "--prediction-keep-path does not exist: "
                        f"{args.prediction_keep_path}"
                    )
                requested_prediction_ids = read_keep_ids(
                    args.prediction_keep_path
                )
            (
                prediction_covar,
                prediction_ids,
                prediction_dropped,
            ) = load_covar_aligned(
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
                component_variant_indices=(
                    component_variant_indices or None
                ),
                standardization_overrides=standardization_overrides,
                call_width=call_width,
                # Needed only to extract the selected fixed-SNP columns; the
                # retained values are supplied training overrides, not moments
                # estimated from prediction samples.
                keep_host_stats=True,
                cpu_threads=cpu_threads,
                gpu_budget_bytes=gpu_budget_bytes,
                ring_depth=plan.ring_depth,
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
                    FitConfig(
                        sources=prediction_sources,
                        **prediction_cfg_kwargs,
                    )
                )
            else:
                prediction_fitter = InfinitesimalREMLFitter(
                    FitConfig(
                        bed_prefix=prediction_bed_list,
                        **prediction_cfg_kwargs,
                    )
                )
            try:
                prediction_grm_index = MultiGRMIndex(
                    prediction_fitter.streamers,
                    call_plan=prediction_fitter._multi_call_plan,
                    component_variant_indices=(
                        component_variant_indices or None
                    ),
                )
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
                prediction_fitter.close()

            branch_metadata = {
                "lasso": {
                    "estimator_valid": True,
                    "output_emitted": True,
                    "invalid_reasons": [],
                    "mean_estimator": "final_weighted_lasso",
                    "covariance_estimator": "lasso_residual_ml",
                    "theta_standardized": theta_lasso_ml.tolist(),
                    "residual": (
                        "(y-X_beta_cov_lasso-Z_support_beta_lasso)"
                        "/phenotype_scale"
                    ),
                    "support_size": int(support.size),
                    "pcg_rel_res": lasso_prediction.pcg_rel_res,
                    "pcg_iters": lasso_prediction.pcg_iters,
                },
                "selected_span": {
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
                },
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

    summary = {
        "sparse_output_schema_version": 4,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_sec": float(time.time() - t0),
        "n_samples": int(y_np.shape[0]),
        "n_covar": int(covar_np.shape[1]) if covar_np is not None else 0,
        "n_snps_total": grm_index.m_total,
        "n_grms": grm_index.n_grm,
        "m_per_grm": grm_index.m_per_grm.tolist(),
        "genetic_trace_atoms": genetic_trace_atoms.tolist(),
        "ebic_p_mode": args.ebic_p_mode,
        "ebic_model_space_size": (
            grm_index.m_total
            if args.ebic_p_mode == "full"
            else int(final_candidate.size)
        ),
        "component_spec": component_spec_source or None,
        "component_partition_mode": (
            "snp_id" if component_variant_indices else "input_prefix"
        ),
        "var_components_lasso_ml": theta_lasso_ml.tolist(),
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
        "var_components_at_lasso": theta_lasso.tolist(),
        "phenotype_mean": phenotype_mean,
        "phenotype_scale": phenotype_scale,
        "variance_component_scale": "standardized_phenotype",
        "primary_h2_method": primary_h2_method,
        "all_sparse_branches_valid": all_sparse_branches_valid,
        "lasso_branch_valid": lasso_branch_valid,
        "lasso_branch_invalid_reasons": list(
            branch_guards["lasso_branch_invalid_reasons"]
        ),
        "lasso_outputs_finite": bool(
            branch_guards["lasso_outputs_finite"]
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
        "sparse_outputs_finite": sparse_outputs_finite,
        "sparse_fit_rejection_reasons": sparse_fit_rejection_reasons,
        "lasso_ml_outer_converged": lasso_ml_outer_converged,
        "outer_stop_reason": outer_stop_reason,
        "penalized_failure_reason": penalized_failure_reason,
        "lasso_variance_update": "intercept_contrast_residual_ml",
        "lasso_variance_contrast": "orthogonal_to_intercept",
        "lasso_variance_analysis_dimension": int(y_np.shape[0] - 1),
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
        "reml_max_linesearch_trials": int(
            args.reml_max_linesearch_trials
        ),
        "selected_span_basis_size": ss_gls_basis_size,
        "selected_span_basis_support_positions": (
            selected_span_basis_positions.tolist()
        ),
        "selected_span_basis_support_indices": (
            selected_span_basis_support_indices
        ),
        "alpha_theta_fixed_point_coherent": alpha_theta_fixed_point_coherent,
        "returned_covariance_kkt": returned_covariance_kkt_summary,
        "returned_covariance_kkt_error": returned_covariance_kkt_error,
        "theta_lasso_to_lasso_ml_rel_change": (
            theta_lasso_to_lasso_ml_rel
        ),
        "theta_lasso_ml_to_selected_span_reml_rel_change": (
            theta_lasso_ml_to_selected_span_rel
        ),
        "h2_background_lasso_ml": _trace_weighted_h2(theta_lasso_ml),
        "h2_background_selected_span_reml": _finite_float_or_none(
            h2_background_selected_span_reml
        ),
        "h2_lasso_plugin": _finite_float_or_none(h2_lasso_plugin),
        "h2_lasso_plugin_guarded": h2_lasso_plugin_guarded,
        "h2_lasso_plugin_role": "estimator_1_uncorrected_lasso_ml_plugin",
        "h2_chive": _finite_float_or_none(h2_chive),
        "h2_chive_guarded": h2_chive_guarded,
        "h2_chive_at_lasso_theta": _finite_float_or_none(
            h2_chive_at_lasso_theta
        ),
        "h2_chive_post_gls": _finite_float_or_none(h2_chive_post_gls),
        "h2_chive_post_gls_role": "diagnostic_only",
        "h2_ss_gls_plugin": _finite_float_or_none(h2_ss_gls_plugin),
        "h2_ss_gls_plugin_guarded": h2_ss_gls_plugin_guarded,
        "h2_ss_gls_df_corrected": _finite_float_or_none(
            h2_ss_gls_df_corrected
        ),
        "h2_ss_gls_df_guarded": h2_ss_gls_df_guarded,
        "h2_ss_gls_role": "estimators_3_and_4_selected_support_reml_gls",
        "h2": h2,
        "q_lasso_plugin_raw": _finite_float_or_none(q_chive_term1),
        "q_lasso_plugin_standardized": _finite_float_or_none(
            q_chive_term1_standardized
        ),
        "q_chive_raw": _finite_float_or_none(q_chive),
        "q_chive_standardized": _finite_float_or_none(
            q_chive_standardized
        ),
        "q_chive_post_gls_raw": _finite_float_or_none(q_chive_post_gls_raw),
        "q_chive_post_gls_standardized": _finite_float_or_none(
            q_chive_post_gls_standardized
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
        "q_chive_post_gls_components_raw": {
            "term1_g2_over_n": _finite_float_or_none(
                q_chive_post_gls_term1_raw
            ),
            "term2_cross": _finite_float_or_none(
                q_chive_post_gls_term2_raw
            ),
            "scale": "raw_phenotype_variance",
        },
        "q_chive_post_gls_components_standardized": {
            "term1_g2_over_n": _finite_float_or_none(
                q_chive_post_gls_term1_standardized
            ),
            "term2_cross": _finite_float_or_none(
                q_chive_post_gls_term2_standardized
            ),
            "scale": "standardized_phenotype_variance",
        },
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
        "support_size": int(support.size),
        "support_indices": support.tolist(),
        "support_source_indices": grm_index.source_variant_indices(support).tolist(),
        # Candidate expansion certifies the EBIC-selected lambda in each outer
        # round.  It does not certify every unselected point on the lambda path.
        "kkt_certification_scope": "selected_lambda_only",
        "ebic_path_globally_kkt_certified": False,
        "kkt_certified": bool(
            history and bool(history[-1].get("kkt_certified", False))
        ),
        "returned_covariance_kkt_certified": bool(
            returned_covariance_kkt["passed"]
        ),
        "outer_history": history,
        "sparse_prediction": prediction_summary,
    }

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
        f.write(
            "snp_index\tsource_snp_index\tgrm\tchr\tsnp_id\tcm\tbp"
            "\ta1\ta2\tbeta_lasso\tbeta_gls_reml"
            "\tselected_span_basis\n"
        )
        for snp_idx in support.tolist():
            chr_, snp_id, cm, bp, a1, a2 = bim_rows.get(
                int(snp_idx),
                ("NA", f"SNP_{int(snp_idx)}", "NA", "NA", "NA", "NA"),
            )
            grm_id = _snp_grm_map.get(int(snp_idx), -1)
            source_snp_idx = source_index_map.get(int(snp_idx), int(snp_idx))
            beta_val = beta_map.get(int(snp_idx), 0.0)
            beta_reml = (
                beta_reml_map.get(int(snp_idx), 0.0)
                if selected_span_refit_ok
                else float("nan")
            )
            basis_member = int(int(snp_idx) in selected_span_basis_set)
            f.write(
                f"{int(snp_idx)}\t{source_snp_idx}\t{grm_id}\t{chr_}\t{snp_id}\t{cm}\t{bp}\t{a1}\t{a2}\t"
                f"{beta_val:.8e}\t{beta_reml:.8e}\t{basis_member}\n"
            )

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
