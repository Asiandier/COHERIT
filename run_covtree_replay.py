#!/usr/bin/env python3
"""Replay CovTree scores from a completed sparse fit without rerunning LASSO."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Sequence

import jax
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent
PARENT = REPO_ROOT.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))
PKG = REPO_ROOT.name

from GPU_REML import run_sparse_reml_pipeline as sparse  # noqa: E402
from GPU_REML.reml_model import FitConfig, InfinitesimalREMLFitter  # noqa: E402


logger = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-dir", required=True)
    parser.add_argument("--ld-score", required=True)
    parser.add_argument("--diagnostic-out", required=True)
    parser.add_argument("--marker-score-out", required=True)
    parser.add_argument("--split-spec-out", default="")
    parser.add_argument("--bootstrap-draws", type=int, default=199)
    parser.add_argument("--bootstrap-seed", type=int, default=20260827)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--rank-rtol", type=float, default=1e-7)
    parser.add_argument("--min-child-markers", type=int, default=16)
    parser.add_argument("--max-univariate-depth", type=int, default=2)
    parser.add_argument("--parent-theta-abs-min", type=float, default=1e-6)
    parser.add_argument("--parent-theta-rel-min", type=float, default=1e-4)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _completed_fit_paths(fit_dir: Path) -> tuple[Path, Path, Path]:
    summary = fit_dir / "coherit.summary.json"
    selected = fit_dir / "coherit.selected_snps.tsv"
    trajectory = fit_dir / "h2_trajectory.json"
    for path in (summary, selected, trajectory):
        if not path.is_file():
            raise FileNotFoundError(path)
    return summary, selected, trajectory


def _parse_pipeline_args(arguments: Sequence[str]) -> argparse.Namespace:
    saved = sys.argv
    try:
        sys.argv = [str(REPO_ROOT / "run_sparse_reml_pipeline.py"), *arguments]
        return sparse.parse_args()
    finally:
        sys.argv = saved


def _selected_effects(path: Path) -> tuple[np.ndarray, np.ndarray]:
    indices: list[int] = []
    effects: list[float] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or not {"snp_index", "beta_lasso"}.issubset(
            reader.fieldnames
        ):
            raise ValueError(f"Selected-SNP table has an incompatible schema: {path}")
        for row in reader:
            indices.append(int(row["snp_index"]))
            effects.append(float(row["beta_lasso"]))
    index_array = np.asarray(indices, dtype=np.int64)
    effect_array = np.asarray(effects, dtype=np.float64)
    if (
        index_array.size != np.unique(index_array).size
        or not np.all(np.isfinite(effect_array))
    ):
        raise ValueError("Selected SNP indices/effects are invalid.")
    return index_array, effect_array


def _build_fitter(original: argparse.Namespace):
    bed_list = [value.strip() for value in original.bed_prefix.split(",") if value.strip()]
    if len(bed_list) != 1 or original.pgen_prefix:
        raise ValueError("CovTree replay currently requires one PLINK1 BED source.")
    component_indices = sparse._load_component_variant_indices(original.component_spec)
    if not component_indices:
        raise ValueError("CovTree replay requires the completed fit's component spec.")
    fam_path = bed_list[0] + ".fam"
    keep_ids = (
        sparse.read_keep_ids(original.keep_path)
        if original.keep_path and os.path.exists(original.keep_path)
        else None
    )
    y, covar, fam_keep, _, _ = sparse.load_pheno_covar_aligned_with_transform(
        fam_path=fam_path,
        pheno_path=original.pheno_txt,
        covar_path=original.covar_txt or None,
        add_intercept=True,
        keep_ids=keep_ids,
    )
    y = np.asarray(y, dtype=np.float32)
    covar = None if covar is None else np.asarray(covar, dtype=np.float32)
    n_bed = sparse._bed_count(bed_list[0] + ".bed", "iid_count")
    sample_mask = (
        sparse.compute_sample_mask(fam_path, fam_keep)
        if n_bed != len(fam_keep)
        else None
    )

    gpu_name, _, gpu_free = sparse.setup_gpu()
    cpu_threads, _ = sparse.resolve_cpu_threads(original.cpu_threads or None)
    marker_count = sparse._bed_count(bed_list[0] + ".bed", "sid_count")
    plan = sparse.run_planner(
        n_samples=y.shape[0],
        p_list=[marker_count],
        n_grm=len(component_indices),
        component_block_sizes=[len(group) for group in component_indices],
        gpu_free=gpu_free,
        gpu_budget=(
            original.gpu_budget_gib * 1024**3
            if original.gpu_budget_gib > 0
            else None
        ),
        n_covar=0 if covar is None else covar.shape[1],
        n_rand_vec=original.n_rand_vec,
        slq_samples=original.slq_samples,
        gpu_name=gpu_name,
        ring_depth=original.ring_depth if original.ring_depth > 0 else None,
        source_format="bed",
        arbitrary_component_partition=True,
        requested_call_width=(original.call_width if original.call_width > 0 else None),
    )
    gpu_budget_bytes = (
        float(original.gpu_budget_gib) * 1024**3
        if original.gpu_budget_gib > 0
        else float(plan.gpu_budget_gib) * 1024**3
    )
    config = FitConfig(
        bed_prefix=bed_list,
        device=original.device,
        sample_mask=sample_mask,
        component_variant_indices=component_indices,
        call_width=plan.call_width,
        cpu_threads=cpu_threads,
        keep_host_stats=True,
        gpu_budget_bytes=gpu_budget_bytes,
        ring_depth=plan.ring_depth,
        n_rand_vec=original.n_rand_vec,
        minq_iter=original.minq_iter,
        slq_samples=original.slq_samples,
        slq_m=original.slq_m,
        precond_rank=plan.precond_rank,
        reml_pcg_tol=original.pcg_tol,
        strict_max_linesearch_trials=original.reml_max_linesearch_trials,
        max_pcg_iters=original.max_pcg_iters,
        pcg_ridge=original.pcg_ridge,
        verbose=original.verbose,
    )
    fitter = InfinitesimalREMLFitter(config)
    ops = fitter._assemble_reml_operators()
    grm_index = sparse.MultiGRMIndex(
        fitter.streamers,
        call_plan=fitter._multi_call_plan,
        component_variant_indices=component_indices,
    )
    return fitter, ops, grm_index, y, covar, bed_list[0]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    fit_dir = Path(args.fit_dir).expanduser().resolve(strict=True)
    summary_path, selected_path, trajectory_path = _completed_fit_paths(fit_dir)
    completed = _load_json(summary_path)
    trajectory = _load_json(trajectory_path)
    pipeline_args = trajectory.get("pipeline_args")
    if not isinstance(pipeline_args, list) or not all(
        isinstance(value, str) for value in pipeline_args
    ):
        raise ValueError("h2_trajectory.json lacks the original pipeline_args.")
    original = _parse_pipeline_args(pipeline_args)
    original.verbose = bool(args.verbose)

    fitter, ops, grm_index, y, covar, bed_prefix = _build_fitter(original)
    try:
        theta = np.asarray(completed["var_components_lasso_ml"], dtype=np.float64)
        support, beta = _selected_effects(selected_path)
        expected_support = np.asarray(completed["support_indices"], dtype=np.int64)
        if not np.array_equal(support, expected_support):
            raise ValueError("Selected-SNP table does not match the completed summary.")
        active = grm_index.extract_standardized_columns(support).astype(
            np.float32, copy=False
        )
        # The restricted projector used by CovTree annihilates the unpenalized
        # covariate span, so no separate GLS fit is needed for replay.
        residual = y - active @ beta.astype(np.float32)

        marker_path = str(Path(args.marker_score_out).expanduser().resolve())
        replay_args = argparse.Namespace(
            covtree_ld_score=str(Path(args.ld_score).expanduser().resolve(strict=True)),
            covtree_min_child_markers=int(args.min_child_markers),
            covtree_max_univariate_depth=int(args.max_univariate_depth),
            covtree_parent_theta_abs_min=float(args.parent_theta_abs_min),
            covtree_parent_theta_rel_min=float(args.parent_theta_rel_min),
            covtree_bootstrap_draws=int(args.bootstrap_draws),
            covtree_bootstrap_seed=int(args.bootstrap_seed),
            covtree_alpha=float(args.alpha),
            covtree_rank_rtol=float(args.rank_rtol),
            covtree_split_spec_out=(
                str(Path(args.split_spec_out).expanduser().resolve())
                if args.split_spec_out
                else ""
            ),
            covtree_diagnostic_out=str(
                Path(args.diagnostic_out).expanduser().resolve()
            ),
            pcg_tol=float(original.pcg_tol),
            max_pcg_iters=int(original.max_pcg_iters),
        )
        diagnostic_summary = sparse._run_covtree_diagnostic(
            args=replay_args,
            fitter=fitter,
            ops=ops,
            grm_index=grm_index,
            component_spec_path=original.component_spec,
            bed_prefix=bed_prefix,
            marker_score_path=marker_path,
            residual_raw=residual,
            covar=covar,
            phenotype_scale=float(completed["phenotype_scale"]),
            theta=theta,
        )
        print(
            json.dumps(
                {
                    "fit_dir": str(fit_dir),
                    "h2": completed.get("h2_chive_guarded"),
                    "marker_score": diagnostic_summary["marker_score"],
                    "covtree": diagnostic_summary,
                },
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        fitter.close()
        jax.clear_caches()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
