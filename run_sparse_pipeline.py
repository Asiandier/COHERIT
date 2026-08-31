#!/usr/bin/env python3
"""Production sparse COHERIT workflow.

Two modes are intentionally supported:

``fixed``
    Fit a user-specified fixed GRM partition (or one whole-genome GRM), with
    validation R2 selecting lambda inside every alpha/theta outer iteration.

``adaptive``
    Fit K=1 as above, freeze its sparse mean, add LD-rank covariance boundaries
    while the global LD-CUSUM score is significant, jointly refit alpha/theta
    with validation-selected lambda at the endpoint, and then perform the same
    final combined-sample refit as fixed mode.

Neither mode accepts or reads a prediction phenotype.  Once model selection is
complete, the selected lambda/lambda-max ratio is frozen and the model is
warm-refit on training + validation samples before effects and predictions are
emitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

from .adaptive_ld import (
    add_ld_boundary,
    atomic_json,
    read_json,
    write_root_component_spec,
)
from .component_spec import load_component_specs
from .ld_score import (
    compute_ld_scores_with_plink2,
    load_aligned_ld_scores,
    load_variant_records,
    write_ld_rank_artifact,
)


def _now() -> float:
    return time.time()


def _resolved_file(value: str, label: str) -> Path:
    path = Path(value).expanduser().resolve(strict=True)
    if not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")
    return path


def _validate_prefix(value: str, genotype_format: str) -> Path:
    prefix = Path(value).expanduser().resolve(strict=False)
    suffixes = (
        (".bed", ".bim", ".fam")
        if genotype_format == "bed"
        else (".pgen", ".pvar", ".psam")
    )
    for suffix in suffixes:
        if not Path(str(prefix) + suffix).is_file():
            raise FileNotFoundError(str(prefix) + suffix)
    return prefix


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _input_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": _sha256(path),
    }


def _large_file_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _genotype_signature(prefix: Path, genotype_format: str) -> dict[str, Any]:
    suffixes = (
        (".bed", ".bim", ".fam")
        if genotype_format == "bed"
        else (".pgen", ".pvar", ".psam")
    )
    return {
        "format": genotype_format,
        "prefix": str(prefix),
        "files": {
            suffix.lstrip("."): _large_file_signature(Path(str(prefix) + suffix))
            for suffix in suffixes
        },
    }


def _atomic_text(path: Path, text: str) -> None:
    if path.is_file() and path.read_text(encoding="utf-8") == text:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _read_source_sample_ids(prefix: Path, genotype_format: str) -> list[str]:
    path = Path(str(prefix) + (".fam" if genotype_format == "bed" else ".psam"))
    result: list[str] = []
    header: list[str] | None = None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            fields = stripped.lstrip("#").split()
            if genotype_format == "pgen" and stripped.startswith("#"):
                if "IID" in fields:
                    header = fields
                continue
            index = (
                header.index("IID")
                if genotype_format == "pgen" and header is not None
                else 1 if len(fields) >= 2 else 0
            )
            if len(fields) <= index:
                raise ValueError(f"Malformed sample row in {path}: {stripped!r}")
            result.append(fields[index])
    if not result or len(result) != len(set(result)):
        raise ValueError(f"Source sample IDs are empty or duplicated: {path}")
    return result


def _read_keep(path: Path) -> list[str]:
    result: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.split()
            if not fields:
                continue
            iid = fields[1] if len(fields) >= 2 else fields[0]
            if not iid:
                raise ValueError(f"Empty IID in {path} row {line_number}.")
            result.append(iid)
    if not result or len(result) != len(set(result)):
        raise ValueError(f"Keep IDs are empty or duplicated: {path}")
    return result


def _read_phenotype_map(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.split()
            if len(fields) < 3:
                continue
            try:
                value = float(fields[2])
            except ValueError:
                continue
            if not np.isfinite(value) or value == -9.0:
                continue
            iid = fields[1]
            if iid in values:
                raise ValueError(f"Duplicate phenotype IID {iid!r} in {path}.")
            values[iid] = value
    if not values:
        raise ValueError(f"No finite phenotype values found in {path}.")
    return values


def _prepare_combined_fit_inputs(args: argparse.Namespace) -> tuple[Path, Path, bool]:
    if args.fit_pheno_txt:
        return args.fit_pheno_txt, args.fit_keep_path, False

    train_ids = _read_keep(args.keep_path)
    validation_ids = _read_keep(args.validation_keep_path)
    overlap = set(train_ids).intersection(validation_ids)
    if overlap:
        raise ValueError(
            "Training and validation keep files must be disjoint; first overlap: "
            f"{next(iter(overlap))!r}."
        )
    selected = set(train_ids).union(validation_ids)
    source_order = _read_source_sample_ids(args.genotype_prefix, args.genotype_format)
    combined_ids = [iid for iid in source_order if iid in selected]
    if len(combined_ids) != len(selected):
        missing = selected.difference(combined_ids)
        raise ValueError(f"Combined keep ID is absent from genotype: {next(iter(missing))!r}.")

    train_values = _read_phenotype_map(args.pheno_txt)
    validation_values = _read_phenotype_map(args.validation_pheno_txt)
    combined_values = dict(train_values)
    for iid, value in validation_values.items():
        if iid in combined_values and not math.isclose(
            combined_values[iid], value, rel_tol=0.0, abs_tol=0.0
        ):
            raise ValueError(f"Conflicting phenotype values for IID {iid!r}.")
        combined_values[iid] = value
    absent = [iid for iid in combined_ids if iid not in combined_values]
    if absent:
        raise ValueError(
            f"Combined fit phenotype is missing selected IID {absent[0]!r}."
        )

    combined_dir = args.work_dir / "combined_fit_inputs"
    phenotype_path = combined_dir / "train_plus_validation.pheno"
    keep_path = combined_dir / "train_plus_validation.keep"
    _atomic_text(
        keep_path,
        "".join(f"{iid}\t{iid}\n" for iid in combined_ids),
    )
    _atomic_text(
        phenotype_path,
        "".join(
            f"{iid}\t{iid}\t{combined_values[iid]:.17g}\n" for iid in combined_ids
        ),
    )
    return phenotype_path, keep_path, True


def _run_command(command: Sequence[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    package_parent = str(Path(__file__).resolve().parent.parent)
    python_path = [package_parent]
    if environment.get("PYTHONPATH"):
        python_path.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_path)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(json.dumps({"started_at": _now(), "argv": list(command)}) + "\n")
        log.flush()
        completed = subprocess.run(
            list(command),
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
            check=False,
        )
        log.write(
            json.dumps(
                {"finished_at": _now(), "returncode": int(completed.returncode)}
            )
            + "\n"
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Sparse COHERIT stage exited {completed.returncode}; see {log_path}."
        )


def _source_options(prefix: Path, genotype_format: str, *, prediction: bool = False) -> list[str]:
    if prediction:
        flag = "--prediction-bed-prefix" if genotype_format == "bed" else "--prediction-pgen-prefix"
    else:
        flag = "--bed-prefix" if genotype_format == "bed" else "--pgen-prefix"
    return [flag, str(prefix)]


def _low_level_command(
    *,
    args: argparse.Namespace,
    phenotype: Path,
    keep: Path,
    component_spec: Path | None,
    out_prefix: Path,
    validation_selection: bool,
    warm_state_in: Path | None = None,
    warm_state_out: Path | None = None,
    theta_init: Sequence[float] | None = None,
    fixed_lam_ratio: float | None = None,
    final_outputs: bool = False,
) -> list[str]:
    if validation_selection == (fixed_lam_ratio is not None):
        raise ValueError("Choose validation selection or a frozen lambda ratio.")
    command = [
        str(args.python_bin),
        str(args.low_level_pipeline),
        *_source_options(args.genotype_prefix, args.genotype_format),
        "--pheno-txt",
        str(phenotype),
        "--keep-path",
        str(keep),
        "--out-prefix",
        str(out_prefix),
        "--device",
        args.device,
        "--gpu-budget-gib",
        format(float(args.gpu_budget_gib), ".8g"),
        "--cpu-threads",
        str(int(args.cpu_threads)),
        "--call-width",
        str(int(args.call_width)),
        "--ring-depth",
        str(int(args.ring_depth)),
        "--n-rand-vec",
        str(int(args.n_rand_vec)),
        "--slq-samples",
        str(int(args.slq_samples)),
        "--slq-m",
        str(int(args.slq_m)),
        "--minq-iter",
        str(int(args.minq_iter)),
        "--pcg-tol",
        format(float(args.pcg_tol), ".8g"),
        "--pcg-ridge",
        format(float(args.pcg_ridge), ".8g"),
        "--max-pcg-iters",
        str(int(args.max_pcg_iters)),
        "--reml-max-linesearch-trials",
        str(int(args.reml_max_linesearch_trials)),
        "--outer-max",
        str(int(args.outer_max)),
        "--h2-abs-tol",
        format(float(args.h2_abs_tol), ".8g"),
        "--effect-rel-tol",
        format(float(args.effect_rel_tol), ".8g"),
        "--screen-topk",
        str(int(args.screen_topk)),
        "--candidate-k",
        str(int(args.candidate_k)),
        "--kkt-add-topk",
        str(int(args.kkt_add_topk)),
        "--kkt-max-rounds",
        str(int(args.kkt_max_rounds)),
        "--lasso-lam-min-ratio",
        format(float(args.lam_min_ratio), ".8g"),
        "--lasso-n-lambda",
        str(int(args.n_lambda)),
        "--lasso-cd-max-iter",
        str(int(args.lasso_cd_max_iter)),
        "--validation-early-stopping-lag",
        str(int(args.validation_early_stopping_lag)),
    ]
    if args.covar_txt:
        command.extend(["--covar-txt", str(args.covar_txt)])
    if component_spec is not None:
        command.extend(["--component-spec", str(component_spec)])
    if validation_selection:
        command.extend(
            [
                *_source_options(
                    args.genotype_prefix, args.genotype_format, prediction=True
                ),
                "--prediction-keep-path",
                str(args.validation_keep_path),
                "--sparsity-validation-pheno-txt",
                str(args.validation_pheno_txt),
                "--sparsity-validation-out",
                str(out_prefix) + ".validation.json",
            ]
        )
        if args.covar_txt:
            command.extend(["--prediction-covar-txt", str(args.covar_txt)])
    else:
        command.extend(
            [
                "--lasso-fixed-lam-ratio",
                format(float(fixed_lam_ratio), ".17g"),
            ]
        )
    if warm_state_in is not None:
        command.extend(["--lasso-warm-state-in", str(warm_state_in)])
    if warm_state_out is not None:
        command.extend(["--lasso-warm-state-out", str(warm_state_out)])
    if theta_init is not None:
        command.extend(
            [
                "--variance-components-init",
                json.dumps(
                    np.asarray(theta_init, dtype=np.float64).tolist(),
                    separators=(",", ":"),
                ),
            ]
        )
    if final_outputs:
        if args.compute_effects:
            command.append("--compute-effects")
        if args.prediction_prefix is not None:
            command.extend(
                _source_options(
                    args.prediction_prefix,
                    args.prediction_format,
                    prediction=True,
                )
            )
            if args.prediction_covar_txt:
                command.extend(
                    ["--prediction-covar-txt", str(args.prediction_covar_txt)]
                )
            if args.prediction_keep_path:
                command.extend(
                    ["--prediction-keep-path", str(args.prediction_keep_path)]
                )
    if args.verbose:
        command.append("--verbose")
    return command


def _adaptive_stage_command(
    args: argparse.Namespace,
    command_name: str,
    *,
    component_spec: Path,
    state_path: Path,
    extra: Sequence[str],
) -> list[str]:
    return [
        str(args.python_bin),
        "-m",
        "GPU_REML.adaptive_ld",
        command_name,
        *_source_options(args.genotype_prefix, args.genotype_format),
        "--pheno-txt",
        str(args.pheno_txt),
        "--keep-path",
        str(args.keep_path),
        "--component-spec",
        str(component_spec),
        "--state-path",
        str(state_path),
        "--device",
        args.device,
        "--gpu-budget-gib",
        format(float(args.gpu_budget_gib), ".8g"),
        "--cpu-threads",
        str(int(args.cpu_threads)),
        "--call-width",
        str(int(args.call_width)),
        "--ring-depth",
        str(int(args.ring_depth)),
        "--n-rand-vec",
        str(int(args.n_rand_vec)),
        "--slq-samples",
        str(int(args.slq_samples)),
        "--slq-m",
        str(int(args.slq_m)),
        "--minq-iter",
        str(int(args.minq_iter)),
        "--pcg-tol",
        format(float(args.pcg_tol), ".8g"),
        "--pcg-ridge",
        format(float(args.pcg_ridge), ".8g"),
        "--max-pcg-iters",
        str(int(args.max_pcg_iters)),
        "--reml-max-linesearch-trials",
        str(int(args.reml_max_linesearch_trials)),
        *extra,
    ] + (["--covar-txt", str(args.covar_txt)] if args.covar_txt else []) + (
        ["--verbose"] if args.verbose else []
    )


def _validate_summary(
    prefix: Path,
    *,
    expected_k: int,
    expected_method: str,
    require_prediction: bool,
) -> dict[str, Any]:
    path = Path(str(prefix) + ".summary.json")
    summary = read_json(path)
    if int(summary.get("sparse_output_schema_version", -1)) != 7:
        raise ValueError(f"Unsupported sparse summary schema: {path}")
    if int(summary.get("n_grms", -1)) != int(expected_k):
        raise ValueError(f"Sparse summary K does not match expected K={expected_k}: {path}")
    if not bool(summary.get("lasso_branch_valid", False)):
        raise ValueError(f"Sparse summary does not contain a valid COHERIT branch: {path}")
    if summary.get("lambda_selection_method") != expected_method:
        raise ValueError(f"Unexpected lambda-selection method in {path}.")
    if expected_method == "validation_r2":
        if not bool(summary.get("validation_selection_inside_outer_loop", False)):
            raise ValueError("Validation lambda was not selected inside the outer loop.")
        if summary.get("sparse_prediction", {}).get("status") != "emitted":
            raise ValueError("Validation selection did not emit its validation predictions.")
    elif summary.get("lasso_path_role") != "frozen_ratio_target_only":
        raise ValueError("Final refit did not use the frozen-ratio target path.")
    if require_prediction and summary.get("sparse_prediction", {}).get("status") != "emitted":
        raise ValueError("Requested prediction output was not emitted.")
    return summary


def _stage_snapshot(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "k": int(summary["n_grms"]),
        "h2": float(summary["h2"]),
        "theta": [float(value) for value in summary["var_components_lasso_ml"]],
        "q_chive": float(summary["q_chive"]),
        "support_size": int(summary["support_size"]),
        "lambda_ratio": float(summary["lasso_selected_lam_ratio"]),
        "validation_r2": (
            None
            if summary.get("lasso_selected_validation_r2") is None
            else float(summary["lasso_selected_validation_r2"])
        ),
        "outer_iterations": int(summary["outer_iterations"]),
        "outer_stop_reason": str(summary["outer_stop_reason"]),
    }


def _component_count(path: Path | None) -> int:
    return len(load_component_specs(str(path))) if path is not None else 1


def _run_validation_fit(
    args: argparse.Namespace,
    *,
    directory: Path,
    component_spec: Path | None,
    theta_init: Sequence[float] | None = None,
    warm_state_in: Path | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    prefix = directory / "coherit"
    state = directory / "sparse_state.npz"
    summary_path = Path(str(prefix) + ".summary.json")
    expected_k = _component_count(component_spec)
    if not summary_path.is_file() or not state.is_file():
        _run_command(
            _low_level_command(
                args=args,
                phenotype=args.pheno_txt,
                keep=args.keep_path,
                component_spec=component_spec,
                out_prefix=prefix,
                validation_selection=True,
                warm_state_in=warm_state_in,
                warm_state_out=state,
                theta_init=theta_init,
            ),
            directory / "runner.log",
        )
    summary = _validate_summary(
        prefix,
        expected_k=expected_k,
        expected_method="validation_r2",
        require_prediction=True,
    )
    if not state.is_file():
        raise RuntimeError(f"Validation fit did not emit its sparse state: {state}")
    return prefix, state, summary


def _prepare_ld_rank(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    records = load_variant_records(str(args.genotype_prefix), args.genotype_format)
    rank_path = args.work_dir / "adaptive" / "ld_rank.npz"
    if args.ld_score:
        scores = load_aligned_ld_scores(args.ld_score, records)
        ld_metadata: dict[str, Any] = {
            "source": "user_provided",
            "score_path": str(args.ld_score),
            "n_variants": len(records),
            "cache_reused": True,
        }
    else:
        score_path = args.work_dir / "adaptive" / "computed_ld_score.tsv"
        scores, ld_metadata = compute_ld_scores_with_plink2(
            prefix=str(args.genotype_prefix),
            genotype_format=args.genotype_format,
            keep_path=str(args.keep_path),
            output_path=score_path,
            window_kb=int(args.ld_window_kb),
            threads=int(args.cpu_threads or (os.cpu_count() or 1)),
            plink2=str(args.plink2),
        )
    rank_metadata = write_ld_rank_artifact(
        rank_path, scores, bins=int(args.ld_rank_bins)
    )
    metadata = {"ld_score": ld_metadata, "rank": rank_metadata}
    atomic_json(args.work_dir / "adaptive" / "ld_score.json", metadata)
    return rank_path, metadata


def _run_adaptive_selection(
    args: argparse.Namespace,
) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    records = load_variant_records(str(args.genotype_prefix), args.genotype_format)
    root_spec = args.work_dir / "adaptive" / "k0001.component_spec.npz"
    rank_path_placeholder = args.work_dir / "adaptive" / "ld_rank.npz"
    if not root_spec.is_file():
        write_root_component_spec(
            root_spec,
            marker_count=len(records),
            rank_path=rank_path_placeholder,
        )
    _k1_prefix, k1_state, k1_summary = _run_validation_fit(
        args,
        directory=args.work_dir / "adaptive" / "k1_validation",
        component_spec=root_spec,
    )
    rank_path, ld_metadata = _prepare_ld_rank(args)

    current_k = 1
    current_spec = root_spec
    current_summary_path = Path(str(_k1_prefix) + ".summary.json")
    current_summary: dict[str, Any] = k1_summary
    path: list[dict[str, Any]] = [
        {
            "k": 1,
            "h2": float(k1_summary["h2"]),
            "q_chive": float(k1_summary["q_chive"]),
            "theta": [
                float(value) for value in k1_summary["var_components_lasso_ml"]
            ],
            "component_spec": str(root_spec.resolve()),
            "fit": str(current_summary_path.resolve()),
            "stage": "k1_validation_lambda",
        }
    ]
    stop_reason = "running"
    adaptive_dir = args.work_dir / "adaptive" / "fixed_alpha_path"
    while True:
        score_path = adaptive_dir / f"k{current_k:04d}.score.json"
        if not score_path.is_file():
            _run_command(
                _adaptive_stage_command(
                    args,
                    "score",
                    component_spec=current_spec,
                    state_path=k1_state,
                    extra=[
                        "--summary-path",
                        str(current_summary_path),
                        "--rank-path",
                        str(rank_path),
                        "--bootstrap-draws",
                        str(int(args.bootstrap_draws)),
                        "--bootstrap-seed",
                        str(int(args.bootstrap_seed) + current_k),
                        "--split-alpha",
                        format(float(args.split_alpha), ".8g"),
                        "--out",
                        str(score_path),
                    ],
                ),
                adaptive_dir / f"k{current_k:04d}.score.log",
            )
        score = read_json(score_path)
        p_value = float(score["diagnostics"]["global_sup_score_p_value"])
        selected = score.get("selected_candidate")
        path[-1].update(
            {
                "score_p": p_value,
                "score_accepted": bool(score["accepted"]),
                "proposed_boundary_position": (
                    int(selected["boundary_position"]) if selected else None
                ),
                "score_statistic": (
                    float(selected["score_statistic"]) if selected else None
                ),
                "score": str(score_path.resolve()),
            }
        )
        atomic_json(
            adaptive_dir / "path.json",
            {
                "schema_version": 1,
                "status": "running",
                "path": path,
                "current_k": current_k,
            },
        )
        if not bool(score["accepted"]):
            stop_reason = str(score.get("stop_reason", "global_score_rejected"))
            break
        if selected is None:
            raise RuntimeError("Accepted adaptive score has no selected boundary.")
        if current_k >= int(args.adaptive_max_k):
            raise RuntimeError(
                "Adaptive split signal remained significant at --adaptive-max-k; "
                "increase the safety cap instead of silently truncating the path."
            )

        next_k = current_k + 1
        next_spec = adaptive_dir / f"k{next_k:04d}.component_spec.npz"
        build_path = adaptive_dir / f"k{next_k:04d}.build.json"
        if not next_spec.is_file() or not build_path.is_file():
            build = add_ld_boundary(
                parent_spec=current_spec,
                parent_theta=current_summary["var_components_lasso_ml"],
                rank_path=rank_path,
                boundary_position=int(selected["boundary_position"]),
                output_spec=next_spec,
            )
            atomic_json(build_path, build)
        build = read_json(build_path)
        next_fit_path = adaptive_dir / f"k{next_k:04d}.frozen_fit.json"
        if not next_fit_path.is_file():
            _run_command(
                _adaptive_stage_command(
                    args,
                    "frozen-fit",
                    component_spec=next_spec,
                    state_path=k1_state,
                    extra=[
                        "--parent-summary",
                        str(current_summary_path),
                        "--theta-init-json",
                        json.dumps(
                            build["variance_components_init"], separators=(",", ":")
                        ),
                        "--out",
                        str(next_fit_path),
                    ],
                ),
                adaptive_dir / f"k{next_k:04d}.frozen_fit.log",
            )
        next_summary = read_json(next_fit_path)
        theta = np.asarray(next_summary["var_components_lasso_ml"], dtype=np.float64)
        if theta.shape != (next_k + 1,) or not np.all(np.isfinite(theta)):
            raise RuntimeError(f"Invalid frozen covariance fit: {next_fit_path}")
        path.append(
            {
                "k": next_k,
                "h2": float(next_summary["h2"]),
                "q_chive": float(next_summary["q_chive"]),
                "theta": theta.tolist(),
                "component_spec": str(next_spec.resolve()),
                "fit": str(next_fit_path.resolve()),
                "stage": "fixed_k1_sparse_mean_covariance_refit",
                "added_boundary_position": int(selected["boundary_position"]),
            }
        )
        current_k = next_k
        current_spec = next_spec
        current_summary_path = next_fit_path
        current_summary = next_summary

    path_payload = {
        "schema_version": 1,
        "status": "complete",
        "algorithm": "fixed_k1_sparse_mean_global_ld_cusum",
        "stop_reason": stop_reason,
        "selected_k": current_k,
        "selected_h2_fixed_alpha": float(current_summary["h2"]),
        "path": path,
    }
    atomic_json(adaptive_dir / "path.json", path_payload)

    if current_k == 1:
        endpoint_state = k1_state
        endpoint_summary = k1_summary
        endpoint_spec = root_spec
    else:
        _endpoint_prefix, endpoint_state, endpoint_summary = _run_validation_fit(
            args,
            directory=args.work_dir / "adaptive" / "endpoint_validation_refit",
            component_spec=current_spec,
            theta_init=current_summary["var_components_lasso_ml"],
            warm_state_in=k1_state,
        )
        endpoint_spec = current_spec
    adaptive_result = {
        "ld_score": ld_metadata,
        "k1_validation": _stage_snapshot(k1_summary),
        "fixed_alpha_path": path_payload,
        "endpoint_validation_refit": _stage_snapshot(endpoint_summary),
    }
    return endpoint_spec, endpoint_state, endpoint_summary, adaptive_result


def _run_fixed_selection(
    args: argparse.Namespace,
) -> tuple[Path | None, Path, dict[str, Any], dict[str, Any]]:
    _prefix, state, summary = _run_validation_fit(
        args,
        directory=args.work_dir / "fixed" / "validation_fit",
        component_spec=args.component_spec,
    )
    return args.component_spec, state, summary, {
        "validation_fit": _stage_snapshot(summary)
    }


def _run_final_refit(
    args: argparse.Namespace,
    *,
    component_spec: Path | None,
    selected_state: Path,
    selected_summary: Mapping[str, Any],
) -> dict[str, Any]:
    expected_k = _component_count(component_spec)
    final_summary_path = Path(str(args.out_prefix) + ".summary.json")
    if not final_summary_path.is_file():
        command = _low_level_command(
            args=args,
            phenotype=args.fit_pheno_txt,
            keep=args.fit_keep_path,
            component_spec=component_spec,
            out_prefix=args.out_prefix,
            validation_selection=False,
            warm_state_in=selected_state,
            theta_init=selected_summary["var_components_lasso_ml"],
            fixed_lam_ratio=float(selected_summary["lasso_selected_lam_ratio"]),
            final_outputs=True,
        )
        _run_command(command, args.work_dir / "final_refit.log")
    final = _validate_summary(
        args.out_prefix,
        expected_k=expected_k,
        expected_method="fixed_lam_ratio",
        require_prediction=args.prediction_prefix is not None,
    )
    if not math.isclose(
        float(final["lasso_selected_lam_ratio"]),
        float(selected_summary["lasso_selected_lam_ratio"]),
        rel_tol=1e-10,
        abs_tol=1e-12,
    ):
        raise RuntimeError("Final refit did not preserve the validation-selected lambda ratio.")
    if args.compute_effects and not Path(str(args.out_prefix) + ".sparse_effects.tsv").is_file():
        raise RuntimeError("Final refit did not emit requested sparse effect sizes.")
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("fixed", "adaptive"), default="fixed")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--bed-prefix", default="")
    source.add_argument("--pgen-prefix", default="")
    parser.add_argument(
        "--component-spec",
        default="",
        help="Optional fixed-K component partition; valid only in fixed mode.",
    )
    parser.add_argument(
        "--ld-score",
        default="",
        help=(
            "Optional adaptive-mode TSV with ID and ld_score columns. If omitted, "
            "the pipeline computes the pilot-compatible score on training samples."
        ),
    )
    parser.add_argument("--ld-window-kb", type=int, default=1000)
    parser.add_argument("--ld-rank-bins", type=int, default=1024)
    parser.add_argument("--plink2", default="plink2")
    parser.add_argument("--pheno-txt", required=True)
    parser.add_argument("--validation-pheno-txt", required=True)
    parser.add_argument("--covar-txt", default="")
    parser.add_argument("--keep-path", required=True)
    parser.add_argument("--validation-keep-path", required=True)
    parser.add_argument(
        "--fit-pheno-txt",
        default="",
        help=(
            "Optional train+validation phenotype for the final refit. If omitted, "
            "it is constructed automatically from the two selection phenotypes."
        ),
    )
    parser.add_argument(
        "--fit-keep-path",
        default="",
        help="Optional train+validation keep file; paired with --fit-pheno-txt.",
    )
    prediction = parser.add_mutually_exclusive_group()
    prediction.add_argument("--prediction-bed-prefix", default="")
    prediction.add_argument("--prediction-pgen-prefix", default="")
    parser.add_argument("--prediction-covar-txt", default="")
    parser.add_argument("--prediction-keep-path", default="")
    parser.add_argument("--compute-effects", action="store_true")
    parser.add_argument("--out-prefix", required=True)

    parser.add_argument("--lam-min-ratio", type=float, default=1e-3)
    parser.add_argument("--n-lambda", type=int, default=80)
    parser.add_argument("--lasso-cd-max-iter", type=int, default=10000)
    parser.add_argument("--validation-early-stopping-lag", type=int, default=5)
    parser.add_argument("--outer-max", type=int, default=20)
    parser.add_argument("--h2-abs-tol", type=float, default=0.01)
    parser.add_argument("--effect-rel-tol", type=float, default=0.05)
    parser.add_argument("--screen-topk", type=int, default=2000)
    parser.add_argument("--candidate-k", type=int, default=256)
    parser.add_argument("--kkt-add-topk", type=int, default=256)
    parser.add_argument("--kkt-max-rounds", type=int, default=20)

    parser.add_argument("--bootstrap-draws", type=int, default=199)
    parser.add_argument("--bootstrap-seed", type=int, default=20260831)
    parser.add_argument("--split-alpha", type=float, default=0.05)
    parser.add_argument("--adaptive-max-k", type=int, default=128)

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
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument(
        "--low-level-pipeline",
        default=str(Path(__file__).resolve().with_name("run_sparse_reml_pipeline.py")),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if bool(args.fit_pheno_txt) != bool(args.fit_keep_path):
        parser.error("--fit-pheno-txt and --fit-keep-path must be supplied together.")
    if args.mode == "adaptive" and args.component_spec:
        parser.error("Adaptive mode starts from K=1 and does not accept --component-spec.")
    if args.mode == "fixed" and args.ld_score:
        parser.error("--ld-score is used only by adaptive mode.")
    if args.prediction_covar_txt or args.prediction_keep_path:
        if not (args.prediction_bed_prefix or args.prediction_pgen_prefix):
            parser.error("Prediction covariate/keep inputs require prediction genotype.")
    positive_integer_names = (
        "ld_window_kb",
        "ld_rank_bins",
        "n_lambda",
        "lasso_cd_max_iter",
        "validation_early_stopping_lag",
        "outer_max",
        "screen_topk",
        "candidate_k",
        "kkt_add_topk",
        "kkt_max_rounds",
        "bootstrap_draws",
        "adaptive_max_k",
        "n_rand_vec",
        "slq_samples",
        "slq_m",
        "minq_iter",
        "max_pcg_iters",
        "reml_max_linesearch_trials",
    )
    for name in positive_integer_names:
        if int(getattr(args, name)) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if args.bootstrap_draws < 19:
        parser.error("--bootstrap-draws must be at least 19.")
    if args.ld_rank_bins < 2:
        parser.error("--ld-rank-bins must be at least 2.")
    if args.screen_topk < args.candidate_k:
        parser.error("--screen-topk must be at least --candidate-k.")
    if not 0.0 < args.lam_min_ratio <= 1.0:
        parser.error("--lam-min-ratio must lie in (0, 1].")
    if not 0.0 < args.split_alpha < 1.0:
        parser.error("--split-alpha must lie in (0, 1).")
    if args.h2_abs_tol <= 0.0 or args.effect_rel_tol <= 0.0:
        parser.error("Convergence tolerances must be positive.")
    if (
        args.gpu_budget_gib < 0.0
        or args.cpu_threads < 0
        or args.call_width < 0
        or args.ring_depth < 0
    ):
        parser.error("GPU budget, threads, call width, and ring depth must be nonnegative.")
    if args.pcg_tol <= 0.0 or args.pcg_ridge < 0.0:
        parser.error("PCG tolerance must be positive and ridge nonnegative.")
    return args


def _resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    args.genotype_format = "bed" if args.bed_prefix else "pgen"
    args.genotype_prefix = _validate_prefix(
        args.bed_prefix or args.pgen_prefix, args.genotype_format
    )
    for name in (
        "pheno_txt",
        "validation_pheno_txt",
        "keep_path",
        "validation_keep_path",
        "python_bin",
        "low_level_pipeline",
    ):
        setattr(args, name, _resolved_file(str(getattr(args, name)), name))
    for name in (
        "covar_txt",
        "component_spec",
        "ld_score",
        "fit_pheno_txt",
        "fit_keep_path",
        "prediction_covar_txt",
        "prediction_keep_path",
    ):
        value = str(getattr(args, name))
        setattr(args, name, _resolved_file(value, name) if value else None)
    args.prediction_prefix = None
    args.prediction_format = None
    if args.prediction_bed_prefix or args.prediction_pgen_prefix:
        args.prediction_format = "bed" if args.prediction_bed_prefix else "pgen"
        args.prediction_prefix = _validate_prefix(
            args.prediction_bed_prefix or args.prediction_pgen_prefix,
            args.prediction_format,
        )
    args.out_prefix = Path(args.out_prefix).expanduser().resolve(strict=False)
    args.work_dir = Path(str(args.out_prefix) + ".work")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.fit_pheno_txt, args.fit_keep_path, args.fit_inputs_constructed = (
        _prepare_combined_fit_inputs(args)
    )
    return args


def _configuration(args: argparse.Namespace) -> dict[str, Any]:
    input_paths = {
        "pheno": args.pheno_txt,
        "validation_pheno": args.validation_pheno_txt,
        "train_keep": args.keep_path,
        "validation_keep": args.validation_keep_path,
        "fit_pheno": args.fit_pheno_txt,
        "fit_keep": args.fit_keep_path,
    }
    if args.covar_txt:
        input_paths["covar"] = args.covar_txt
    if args.component_spec:
        input_paths["component_spec"] = args.component_spec
    if args.ld_score:
        input_paths["ld_score"] = args.ld_score
    return {
        "schema_version": 1,
        "mode": args.mode,
        "genotype": _genotype_signature(
            args.genotype_prefix, args.genotype_format
        ),
        "inputs": {
            name: _input_signature(path) for name, path in input_paths.items()
        },
        "fit_inputs_constructed": bool(args.fit_inputs_constructed),
        "prediction": (
            None
            if args.prediction_prefix is None
            else {
                "genotype": _genotype_signature(
                    args.prediction_prefix, args.prediction_format
                ),
                "keep": (
                    _input_signature(args.prediction_keep_path)
                    if args.prediction_keep_path
                    else None
                ),
                "covariates": (
                    _input_signature(args.prediction_covar_txt)
                    if args.prediction_covar_txt
                    else None
                ),
            }
        ),
        "compute_effects": bool(args.compute_effects),
        "lambda_path": {
            "minimum_ratio": float(args.lam_min_ratio),
            "points": int(args.n_lambda),
            "cd_max_iter": int(args.lasso_cd_max_iter),
            "early_stopping_lag": int(args.validation_early_stopping_lag),
        },
        "outer_convergence": {
            "max_iterations": int(args.outer_max),
            "h2_absolute_tolerance": float(args.h2_abs_tol),
            "fitted_mean_relative_tolerance": float(args.effect_rel_tol),
        },
        "working_set": {
            "screen_topk": int(args.screen_topk),
            "candidate_k": int(args.candidate_k),
            "kkt_add_topk": int(args.kkt_add_topk),
            "kkt_max_rounds": int(args.kkt_max_rounds),
        },
        "numerics": {
            "n_rand_vec": int(args.n_rand_vec),
            "slq_samples": int(args.slq_samples),
            "slq_m": int(args.slq_m),
            "minq_iter": int(args.minq_iter),
            "pcg_tol": float(args.pcg_tol),
            "pcg_ridge": float(args.pcg_ridge),
            "max_pcg_iters": int(args.max_pcg_iters),
            "reml_max_linesearch_trials": int(
                args.reml_max_linesearch_trials
            ),
            "call_width": int(args.call_width),
            "ring_depth": int(args.ring_depth),
            "device": str(args.device),
            "gpu_budget_gib": float(args.gpu_budget_gib),
            "cpu_threads": int(args.cpu_threads),
        },
        "adaptive": (
            None
            if args.mode != "adaptive"
            else {
                "ld_window_kb": int(args.ld_window_kb),
                "ld_rank_bins": int(args.ld_rank_bins),
                "bootstrap_draws": int(args.bootstrap_draws),
                "bootstrap_seed": int(args.bootstrap_seed),
                "split_alpha": float(args.split_alpha),
                "max_k_safety_cap": int(args.adaptive_max_k),
                "plink2": str(args.plink2),
            }
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _resolve_args(parse_args(argv))
    config = _configuration(args)
    config_path = args.work_dir / "pipeline_config.json"
    if config_path.is_file():
        previous = read_json(config_path)
        if previous != config:
            raise RuntimeError(
                f"Existing work directory belongs to a different run: {args.work_dir}"
            )
    else:
        atomic_json(config_path, config)

    status_path = args.work_dir / "status.json"
    atomic_json(
        status_path,
        {"state": "running", "stage": "model_selection", "started_at": _now()},
    )
    try:
        if args.mode == "adaptive":
            component_spec, state, selected_summary, selection_result = (
                _run_adaptive_selection(args)
            )
        else:
            component_spec, state, selected_summary, selection_result = (
                _run_fixed_selection(args)
            )
        atomic_json(
            status_path,
            {
                "state": "running",
                "stage": "combined_sample_final_refit",
                "selected_k": int(selected_summary["n_grms"]),
                "selected_lambda_ratio": float(
                    selected_summary["lasso_selected_lam_ratio"]
                ),
            },
        )
        final = _run_final_refit(
            args,
            component_spec=component_spec,
            selected_state=state,
            selected_summary=selected_summary,
        )
        result = {
            "schema_version": 1,
            "status": "complete",
            "mode": args.mode,
            "algorithm": (
                "fixed_k_validation_lambda_then_combined_refit"
                if args.mode == "fixed"
                else "k1_validation_then_fixed_alpha_ld_cusum_then_endpoint_validation_and_combined_refits"
            ),
            "selection": selection_result,
            "final_refit": {
                **_stage_snapshot(final),
                "fit_samples": int(final["n_samples"]),
                "lambda_ratio_frozen": True,
                "component_spec": (
                    str(component_spec.resolve()) if component_spec else None
                ),
                "summary": str(Path(str(args.out_prefix) + ".summary.json")),
                "selected_snps": str(
                    Path(str(args.out_prefix) + ".selected_snps.tsv")
                ),
                "effects": (
                    str(Path(str(args.out_prefix) + ".sparse_effects.tsv"))
                    if args.compute_effects
                    else None
                ),
                "prediction": (
                    str(Path(str(args.out_prefix) + ".sparse_prediction.tsv"))
                    if args.prediction_prefix is not None
                    else None
                ),
            },
            "prediction_phenotype_used": False,
            "config": str(config_path.resolve()),
            "finished_at": _now(),
        }
        result_path = Path(str(args.out_prefix) + ".pipeline.json")
        atomic_json(result_path, result)
        atomic_json(
            status_path,
            {
                "state": "complete",
                "stage": "complete",
                "result": str(result_path.resolve()),
                "finished_at": _now(),
            },
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as error:
        atomic_json(
            status_path,
            {
                "state": "failed",
                "stage": "failed",
                "error": f"{type(error).__name__}: {error}",
                "failed_at": _now(),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
