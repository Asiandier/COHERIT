#!/usr/bin/env python3
"""Heritability-first CovTree loop around fixed-K validation-lambda COHERIT.

Covariance-score evidence selects each proposed GRM split.  After a fitted
layer, a three-layer h2 plateau can stop further splitting and choose the
smallest K in that practically equivalent window for final refitting.
Validation prediction remains inside each fixed-K fit to select the Lasso
lambda, but it never selects K.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent
H2_STABILITY_WINDOW = 3


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def _read_prediction(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"iid", "lasso_phenotype_prediction"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Prediction table has an incompatible schema: {path}")
        iids: list[str] = []
        prediction: list[float] = []
        for row in reader:
            iids.append(str(row["iid"]))
            prediction.append(float(row["lasso_phenotype_prediction"]))
    values = np.asarray(prediction, dtype=np.float64)
    if len(iids) != len(set(iids)) or not np.all(np.isfinite(values)):
        raise ValueError("Prediction IDs or values are invalid.")
    return iids, values


def _read_phenotype(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    with path.open(encoding="utf-8") as handle:
        for row_index, line in enumerate(handle, start=1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) < 3:
                raise ValueError(f"Malformed phenotype row {row_index}: {path}")
            iid = fields[1]
            value = float(fields[2])
            if iid in values or not math.isfinite(value):
                raise ValueError(f"Invalid phenotype row {row_index}: {path}")
            values[iid] = value
    return values


def prediction_metrics(
    prediction_path: Path,
    phenotype_path: Path,
    *,
    phenotype_standardization: dict,
) -> dict[str, float | int]:
    iids, prediction = _read_prediction(prediction_path)
    phenotype = _read_phenotype(phenotype_path)
    if set(iids) != set(phenotype) or len(iids) != len(phenotype):
        raise ValueError("Prediction and validation phenotype IID sets differ.")
    outcome = np.asarray([phenotype[iid] for iid in iids], dtype=np.float64)
    mean = float(phenotype_standardization["mean"])
    standard_deviation = float(
        phenotype_standardization["standard_deviation"]
    )
    if (
        not math.isfinite(mean)
        or not math.isfinite(standard_deviation)
        or standard_deviation <= 0.0
    ):
        raise ValueError("Phenotype input-standardization metadata is invalid.")
    outcome = (outcome - mean) / standard_deviation
    centered_prediction = prediction - prediction.mean()
    centered_outcome = outcome - outcome.mean()
    prediction_ss = float(centered_prediction @ centered_prediction)
    outcome_ss = float(centered_outcome @ centered_outcome)
    if prediction_ss <= 0.0 or outcome_ss <= 0.0:
        raise ValueError("Prediction metrics require nonconstant finite values.")
    covariance = float(centered_prediction @ centered_outcome)
    correlation = covariance / math.sqrt(prediction_ss * outcome_ss)
    return {
        "n": int(outcome.size),
        "correlation_squared": float(correlation * correlation),
        "predictive_r2": float(1.0 - np.sum((outcome - prediction) ** 2) / outcome_ss),
        "mse": float(np.mean(np.square(outcome - prediction))),
        "calibration_slope": float(covariance / prediction_ss),
    }


def heritability_accuracy(
    estimate: float,
    true_h2: float | None,
) -> dict[str, float]:
    """Return evaluation-only h2 errors; truth never enters model selection."""
    if true_h2 is None:
        return {}
    estimate_f = float(estimate)
    truth_f = float(true_h2)
    signed_bias = estimate_f - truth_f
    return {
        "true_h2": truth_f,
        "signed_h2_bias": signed_bias,
        "absolute_h2_error": abs(signed_bias),
    }


def h2_stability_diagnostic(
    layers: Sequence[dict[str, object]],
    *,
    tolerance: float,
) -> dict[str, object]:
    """Detect a practical h2 plateau without using simulation truth.

    A single small parent-to-child change is unsafe: an early split can expose
    heterogeneity that only changes total h2 after a later split.  We therefore
    require the latest three converged fixed-K estimates to lie inside one
    absolute-h2 band.  Using the range also prevents two small changes in the
    same direction from being mistaken for a plateau.
    """
    tol = float(tolerance)
    if not math.isfinite(tol) or not 0.0 < tol < 1.0:
        raise ValueError("h2 stability tolerance must lie in (0, 1).")

    estimates: list[tuple[int, float]] = []
    for layer in layers:
        k = int(layer["K"])
        h2 = float(layer["h2"])
        if k < 1 or not math.isfinite(h2) or not 0.0 <= h2 <= 1.0:
            raise ValueError("CovTree layers contain an invalid K or h2 estimate.")
        if estimates and k <= estimates[-1][0]:
            raise ValueError("CovTree K values must increase strictly across layers.")
        estimates.append((k, h2))

    window = estimates[-H2_STABILITY_WINDOW:]
    enough_history = len(window) == H2_STABILITY_WINDOW
    h2_values = [value for _, value in window]
    h2_range = (
        float(max(h2_values) - min(h2_values))
        if enough_history
        else None
    )
    reached = bool(
        enough_history
        and h2_range is not None
        and (
            h2_range <= tol
            or math.isclose(h2_range, tol, rel_tol=1e-12, abs_tol=1e-15)
        )
    )
    return {
        "criterion": "latest_three_h2_range_at_most_tolerance",
        "window_size": H2_STABILITY_WINDOW,
        "tolerance": tol,
        "n_available_layers": len(estimates),
        "enough_history": enough_history,
        "K": [k for k, _ in window],
        "h2": h2_values,
        "range": h2_range,
        "reached": reached,
    }


def final_refit_layer(
    layers: Sequence[dict[str, object]],
    *,
    stop_reason: str,
) -> tuple[dict[str, object], str]:
    """Choose the simplest practically equivalent layer at an h2 plateau."""
    if not layers:
        raise ValueError("CovTree final selection requires at least one layer.")
    if stop_reason != "h2_stability_plateau":
        return layers[-1], "terminal_fitted_layer"

    diagnostic = layers[-1].get("h2_stability")
    if not isinstance(diagnostic, dict) or not bool(diagnostic.get("reached")):
        raise ValueError("h2 plateau stopping lacks a reached stability diagnostic.")
    window_k = diagnostic.get("K")
    if not isinstance(window_k, list) or len(window_k) != H2_STABILITY_WINDOW:
        raise ValueError("h2 plateau diagnostic has a malformed K window.")
    target_k = int(window_k[0])
    matches = [layer for layer in layers if int(layer["K"]) == target_k]
    if len(matches) != 1:
        raise ValueError("h2 plateau start does not identify exactly one fitted layer.")
    return matches[0], "smallest_K_in_terminal_h2_plateau"


def covtree_data_stop_reason(
    diagnostic: dict[str, object],
    h2_stability: dict[str, object],
) -> str | None:
    """Apply target-estimator stability before covariance fit diagnostics."""
    if bool(h2_stability.get("reached", False)):
        return "h2_stability_plateau"
    if not bool(diagnostic.get("accepted", False)):
        return str(diagnostic.get("stopping_reason") or "score_not_accepted")
    return None


_VALUE_FLAGS_TO_REPLACE = {
    "--component-spec",
    "--out-prefix",
    "--sparsity-validation-out",
    "--variance-components-init",
    "--sparse-state-in",
    "--sparse-state-out",
    "--marker-score-out",
    "--marker-score-probes",
    "--marker-score-seed",
    "--marker-score-min-validation-r2",
    "--covtree-diagnostic-out",
    "--covtree-split-spec-out",
    "--covtree-ld-score",
    "--covtree-bootstrap-draws",
    "--covtree-bootstrap-seed",
    "--covtree-alpha",
    "--covtree-rank-rtol",
    "--covtree-min-child-markers",
}

_FINAL_REFIT_VALUE_FLAGS_TO_REPLACE = {
    "--pheno-txt",
    "--keep-path",
    "--prediction-bed-prefix",
    "--prediction-covar-txt",
    "--prediction-keep-path",
    "--sparsity-validation-pheno-txt",
    "--lasso-fixed-lam-ratio",
}


def _without_value_flags(
    arguments: Sequence[str], flags: set[str]
) -> list[str]:
    cleaned: list[str] = []
    index = 0
    while index < len(arguments):
        value = str(arguments[index])
        if value in flags:
            if index + 1 >= len(arguments):
                raise ValueError(f"Pipeline flag lacks a value: {value}")
            index += 2
            continue
        cleaned.append(value)
        index += 1
    return cleaned


def _base_pipeline_arguments(arguments: Sequence[str]) -> list[str]:
    return _without_value_flags(arguments, _VALUE_FLAGS_TO_REPLACE)


def _final_refit_pipeline_arguments(arguments: Sequence[str]) -> list[str]:
    """Retain runtime/model options but remove all selection-sample inputs."""
    return _without_value_flags(
        _base_pipeline_arguments(arguments),
        _FINAL_REFIT_VALUE_FLAGS_TO_REPLACE,
    )


def _flag_value(arguments: Sequence[str], flag: str) -> str:
    try:
        index = list(arguments).index(flag)
    except ValueError as error:
        raise ValueError(f"Original pipeline arguments lack {flag}.") from error
    if index + 1 >= len(arguments):
        raise ValueError(f"Original pipeline flag lacks a value: {flag}")
    return str(arguments[index + 1])


def _run(command: Sequence[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"started_at": _now(), "argv": list(command)}) + "\n")
        handle.flush()
        completed = subprocess.run(
            list(command), stdout=handle, stderr=subprocess.STDOUT, check=False
        )
        handle.write(
            json.dumps({"finished_at": _now(), "returncode": completed.returncode})
            + "\n"
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Fixed-K sparse pipeline exited {completed.returncode}; see {log_path}."
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--initial-fit-dir", required=True)
    parser.add_argument("--initial-diagnostic", required=True)
    parser.add_argument(
        "--pipeline-trajectory",
        help=(
            "Optional h2_trajectory.json supplying the original fixed-K "
            "pipeline arguments. This permits continuation from an existing "
            "CovTree layer whose directory does not contain that audit file."
        ),
    )
    parser.add_argument(
        "--pipeline-run-config",
        help=(
            "Optional fixed-K validation-lambda run_config.json used to "
            "reconstruct the original selection command when no trajectory "
            "was recorded."
        ),
    )
    parser.add_argument(
        "--fit-pheno-txt",
        help=(
            "Combined training+validation phenotype for the automatic final "
            "refit. Inferred from --pipeline-run-config when available."
        ),
    )
    parser.add_argument(
        "--fit-keep",
        help=(
            "Combined training+validation keep file for the automatic final "
            "refit. Inferred from --pipeline-run-config when available."
        ),
    )
    parser.add_argument("--ld-score", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--true-h2",
        type=float,
        help="Simulation truth used only to report bias, never to select K.",
    )
    parser.add_argument("--max-k", type=int, default=1024)
    parser.add_argument(
        "--max-splits",
        type=int,
        help=(
            "Optional independent split-layer safety cap. By default there "
            "is no layer-count cap: covariance-score stopping or --max-k "
            "terminates the tree."
        ),
    )
    parser.add_argument("--bootstrap-draws", type=int, default=199)
    parser.add_argument("--bootstrap-seed", type=int, default=20260827)
    parser.add_argument("--score-alpha", type=float, default=0.05)
    parser.add_argument(
        "--h2-stability-tol",
        type=float,
        default=0.01,
        help=(
            "Stop before another split when the latest three converged "
            "fixed-K h2 estimates have max-minus-min no larger than this "
            "absolute tolerance."
        ),
    )
    parser.add_argument("--rank-rtol", type=float, default=1e-7)
    parser.add_argument("--min-child-markers", type=int, default=16)
    parser.add_argument(
        "--sparse-pipeline",
        default=str(REPO_ROOT / "run_sparse_reml_pipeline.py"),
    )
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def _selection_arguments_from_run_config(path: Path) -> list[str]:
    config = _read_json(path)
    inputs = config.get("inputs")
    runtime = config.get("runtime")
    algorithm = config.get("algorithm")
    if not all(isinstance(value, dict) for value in (inputs, runtime, algorithm)):
        raise ValueError("run_config.json lacks inputs/runtime/algorithm objects.")
    required = {
        "bed_prefix",
        "component_spec_snapshot",
        "train_pheno_txt",
        "train_keep",
        "validation_pheno_txt",
        "validation_keep",
    }
    missing = sorted(key for key in required if not inputs.get(key))
    if missing:
        raise ValueError(
            "run_config.json lacks required selection inputs: "
            + ", ".join(missing)
        )

    bed_prefix = str(inputs["bed_prefix"])
    arguments = [
        "--bed-prefix",
        bed_prefix,
        "--prediction-bed-prefix",
        bed_prefix,
        "--component-spec",
        str(inputs["component_spec_snapshot"]),
        "--pheno-txt",
        str(inputs["train_pheno_txt"]),
        "--keep-path",
        str(inputs["train_keep"]),
        "--prediction-keep-path",
        str(inputs["validation_keep"]),
        "--sparsity-validation-pheno-txt",
        str(inputs["validation_pheno_txt"]),
    ]
    if inputs.get("covar_txt"):
        covariates = str(inputs["covar_txt"])
        arguments.extend(
            [
                "--covar-txt",
                covariates,
                "--prediction-covar-txt",
                covariates,
            ]
        )

    runtime_flags = {
        "device": "--device",
        "gpu_budget_gib": "--gpu-budget-gib",
        "cpu_threads": "--cpu-threads",
        "screen_topk": "--screen-topk",
        "candidate_k": "--candidate-k",
        "kkt_add_topk": "--kkt-add-topk",
        "kkt_max_rounds": "--kkt-max-rounds",
    }
    for key, flag in runtime_flags.items():
        value = runtime.get(key)
        if value is not None:
            arguments.extend([flag, str(value)])
    algorithm_flags = {
        "lam_min_ratio": "--lasso-lam-min-ratio",
        "n_lambda": "--lasso-n-lambda",
        "lasso_cd_max_iter": "--lasso-cd-max-iter",
    }
    for key, flag in algorithm_flags.items():
        value = algorithm.get(key)
        if value is not None:
            arguments.extend([flag, str(value)])
    return arguments


def _final_refit_inputs_from_run_config(path: Path) -> tuple[str, str]:
    config = _read_json(path)
    inputs = config.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("run_config.json lacks an inputs object.")
    required = ("fit_pheno_txt", "fit_keep")
    missing = [key for key in required if not inputs.get(key)]
    if missing:
        raise ValueError(
            "run_config.json lacks automatic final-refit inputs: "
            + ", ".join(missing)
        )
    return str(inputs["fit_pheno_txt"]), str(inputs["fit_keep"])


def _final_refit_command(
    *,
    python_bin: Path,
    sparse_pipeline: Path,
    original_arguments: Sequence[str],
    component_spec: Path,
    fit_phenotype: Path,
    fit_keep: Path,
    prefix: Path,
    fixed_lam_ratio: float,
    theta_init: Sequence[float],
    verbose: bool,
) -> list[str]:
    ratio = float(fixed_lam_ratio)
    theta = np.asarray(theta_init, dtype=np.float64).reshape(-1)
    if not math.isfinite(ratio) or not 0.0 < ratio <= 1.0:
        raise ValueError("Final-refit lambda ratio must lie in (0, 1].")
    if theta.size < 2 or not np.all(np.isfinite(theta)):
        raise ValueError("Final-refit warm-start theta is malformed.")
    command = [
        str(python_bin),
        str(sparse_pipeline),
        *_final_refit_pipeline_arguments(original_arguments),
        "--component-spec",
        str(component_spec),
        "--pheno-txt",
        str(fit_phenotype),
        "--keep-path",
        str(fit_keep),
        "--out-prefix",
        str(prefix),
        "--lasso-fixed-lam-ratio",
        format(ratio, ".17g"),
        "--variance-components-init",
        json.dumps(theta.tolist(), separators=(",", ":")),
    ]
    if verbose and "--verbose" not in command:
        command.append("--verbose")
    return command


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if int(args.max_k) < 1:
        raise ValueError("max_k must be positive.")
    if args.max_splits is not None and int(args.max_splits) < 0:
        raise ValueError("max_splits must be nonnegative.")
    if int(args.bootstrap_draws) < 19:
        raise ValueError("bootstrap_draws must be at least 19.")
    if int(args.min_child_markers) < 1:
        raise ValueError("min_child_markers must be positive.")
    if not math.isfinite(float(args.score_alpha)) or not 0.0 < float(
        args.score_alpha
    ) < 1.0:
        raise ValueError("score_alpha must lie in (0, 1).")
    if not math.isfinite(float(args.h2_stability_tol)) or not 0.0 < float(
        args.h2_stability_tol
    ) < 1.0:
        raise ValueError("h2_stability_tol must lie in (0, 1).")
    if args.true_h2 is not None and (
        not math.isfinite(float(args.true_h2))
        or not 0.0 <= float(args.true_h2) <= 1.0
    ):
        raise ValueError("true_h2 must lie in [0, 1].")
    if not math.isfinite(float(args.rank_rtol)) or float(args.rank_rtol) <= 0.0:
        raise ValueError("rank_rtol must be finite and positive.")
    initial_dir = Path(args.initial_fit_dir).expanduser().resolve(strict=True)
    initial_summary_path = initial_dir / "coherit.summary.json"
    initial_prediction_path = initial_dir / "coherit.sparse_prediction.tsv"
    if args.pipeline_trajectory and args.pipeline_run_config:
        raise ValueError(
            "Supply at most one of --pipeline-trajectory and "
            "--pipeline-run-config."
        )
    for path in (initial_summary_path, initial_prediction_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    initial_diagnostic_path = Path(args.initial_diagnostic).expanduser().resolve(strict=True)
    ld_score_path = Path(args.ld_score).expanduser().resolve(strict=True)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "covtree_result.json"
    if result_path.exists():
        raise FileExistsError(f"Refusing to overwrite completed result: {result_path}")

    run_config_path = (
        Path(args.pipeline_run_config).expanduser().resolve(strict=True)
        if args.pipeline_run_config
        else None
    )
    fit_phenotype_value = args.fit_pheno_txt
    fit_keep_value = args.fit_keep
    if run_config_path is not None:
        original_arguments = _selection_arguments_from_run_config(
            run_config_path
        )
        inferred_fit_phenotype, inferred_fit_keep = (
            _final_refit_inputs_from_run_config(run_config_path)
        )
        fit_phenotype_value = fit_phenotype_value or inferred_fit_phenotype
        fit_keep_value = fit_keep_value or inferred_fit_keep
    else:
        trajectory_path = (
            Path(args.pipeline_trajectory).expanduser().resolve(strict=True)
            if args.pipeline_trajectory
            else initial_dir / "h2_trajectory.json"
        )
        if not trajectory_path.is_file():
            raise FileNotFoundError(trajectory_path)
        trajectory = _read_json(trajectory_path)
        original_arguments = trajectory.get("pipeline_args")
        if not isinstance(original_arguments, list):
            raise ValueError("Initial h2_trajectory.json lacks pipeline_args.")
    if not fit_phenotype_value or not fit_keep_value:
        raise ValueError(
            "Automatic final refit requires --fit-pheno-txt and --fit-keep, "
            "or a --pipeline-run-config containing both inputs."
        )
    fit_phenotype = Path(fit_phenotype_value).expanduser().resolve(strict=True)
    fit_keep = Path(fit_keep_value).expanduser().resolve(strict=True)
    python_bin = Path(args.python_bin).expanduser().resolve(strict=True)
    sparse_pipeline = Path(args.sparse_pipeline).expanduser().resolve(strict=True)
    validation_phenotype = Path(
        _flag_value(original_arguments, "--sparsity-validation-pheno-txt")
    ).expanduser().resolve(strict=True)
    base_arguments = _base_pipeline_arguments(original_arguments)
    initial_summary = _read_json(initial_summary_path)
    initial_diagnostic = _read_json(initial_diagnostic_path)
    if int(initial_diagnostic.get("current_k", -1)) != int(
        initial_summary["n_grms"]
    ):
        raise ValueError("Initial CovTree diagnostic and fixed-K fit disagree on K.")
    initial_metrics = prediction_metrics(
        initial_prediction_path,
        validation_phenotype,
        phenotype_standardization=initial_summary[
            "input_phenotype_standardization"
        ],
    )

    initial_h2 = float(initial_summary["h2_chive_guarded"])
    layers: list[dict[str, object]] = [
        {
            "step": 0,
            "K": int(initial_summary["n_grms"]),
            "accepted": True,
            "fit_dir": str(initial_dir),
            "summary": str(initial_summary_path),
            "diagnostic": str(initial_diagnostic_path),
            "prediction": str(initial_prediction_path),
            "validation": initial_metrics,
            "acceptance_reason": "initial_fitted_model",
            "h2": initial_h2,
            **heritability_accuracy(initial_h2, args.true_h2),
            "theta": initial_summary["var_components_lasso_ml"],
            "support_size": int(initial_summary["support_size"]),
            "elapsed_sec": float(initial_summary["elapsed_sec"]),
        }
    ]
    layers[0]["h2_stability"] = h2_stability_diagnostic(
        layers,
        tolerance=float(args.h2_stability_tol),
    )
    current_diagnostic = initial_diagnostic
    initial_state_metadata = initial_summary.get("sparse_state_out")
    current_sparse_state = (
        Path(str(initial_state_metadata["path"])).expanduser().resolve(strict=True)
        if isinstance(initial_state_metadata, dict)
        and initial_state_metadata.get("status") == "emitted"
        else None
    )
    stop_reason = None

    step = 0
    while True:
        data_stop_reason = covtree_data_stop_reason(
            current_diagnostic,
            layers[-1]["h2_stability"],
        )
        if data_stop_reason is not None:
            stop_reason = data_stop_reason
            break
        next_k = int(current_diagnostic["next_k"])
        if next_k > int(args.max_k):
            stop_reason = "max_k_safety_cap"
            break
        if args.max_splits is not None and step >= int(args.max_splits):
            stop_reason = "max_splits_safety_cap"
            break
        step += 1
        component_spec_value = current_diagnostic.get("selected_split_spec")
        warm_start = current_diagnostic.get("covariance_preserving_warm_start")
        if not component_spec_value or not isinstance(warm_start, list):
            raise ValueError("Accepted diagnostic lacks split spec or warm start.")
        component_spec = Path(str(component_spec_value)).expanduser().resolve(strict=True)
        layer_dir = out_dir / f"step_{step:02d}_K{next_k}"
        if layer_dir.exists() and any(layer_dir.iterdir()):
            raise FileExistsError(f"Refusing to reuse partial layer: {layer_dir}")
        layer_dir.mkdir(parents=True, exist_ok=True)
        prefix = layer_dir / "coherit"
        selection_out = layer_dir / "validation_path.json"
        marker_out = layer_dir / "marker_score.npz"
        diagnostic_out = layer_dir / "covtree_diagnostic.json"
        split_out = layer_dir / "accepted_split.npz"
        sparse_state_out = layer_dir / "sparse_state.npz"
        command = [
            str(python_bin),
            str(sparse_pipeline),
            *base_arguments,
            "--component-spec",
            str(component_spec),
            "--out-prefix",
            str(prefix),
            "--sparsity-validation-out",
            str(selection_out),
            "--variance-components-init",
            json.dumps(warm_start, separators=(",", ":")),
            "--sparse-state-out",
            str(sparse_state_out),
            "--marker-score-out",
            str(marker_out),
            "--covtree-diagnostic-out",
            str(diagnostic_out),
            "--covtree-split-spec-out",
            str(split_out),
            "--covtree-ld-score",
            str(ld_score_path),
            "--covtree-bootstrap-draws",
            str(int(args.bootstrap_draws)),
            "--covtree-bootstrap-seed",
            str(int(args.bootstrap_seed) + step),
            "--covtree-alpha",
            str(float(args.score_alpha)),
            "--covtree-rank-rtol",
            str(float(args.rank_rtol)),
            "--covtree-min-child-markers",
            str(int(args.min_child_markers)),
        ]
        if current_sparse_state is not None:
            command.extend(["--sparse-state-in", str(current_sparse_state)])
        if args.verbose and "--verbose" not in command:
            command.append("--verbose")
        print(f"[covtree] fitting step={step} K={next_k}", flush=True)
        _run(command, layer_dir / "runner.log")

        summary_path = Path(str(prefix) + ".summary.json")
        prediction_path = Path(str(prefix) + ".sparse_prediction.tsv")
        for path in (summary_path, prediction_path, diagnostic_out):
            if not path.is_file():
                raise RuntimeError(f"CovTree layer lacks expected output: {path}")
        summary = _read_json(summary_path)
        diagnostic = _read_json(diagnostic_out)
        if int(summary["n_grms"]) != next_k:
            raise RuntimeError("Fixed-K layer returned an unexpected component count.")
        metrics = prediction_metrics(
            prediction_path,
            validation_phenotype,
            phenotype_standardization=summary[
                "input_phenotype_standardization"
            ],
        )
        layer_h2 = float(summary["h2_chive_guarded"])
        record = {
            "step": step,
            "K": next_k,
            # Reaching this fit proves that the preceding fixed-K diagnostic
            # selected its split by adjusted covariance-score evidence.  Once
            # fitted, prediction cannot roll the layer back.
            "accepted": True,
            "acceptance_reason": "parent_covariance_score_selected_split",
            "fit_dir": str(layer_dir),
            "summary": str(summary_path),
            "diagnostic": str(diagnostic_out),
            "prediction": str(prediction_path),
            "validation": metrics,
            "prediction_role": "lambda_selection_and_audit_only",
            "h2": layer_h2,
            **heritability_accuracy(layer_h2, args.true_h2),
            "theta": summary["var_components_lasso_ml"],
            "support_size": int(summary["support_size"]),
            "elapsed_sec": float(summary["elapsed_sec"]),
            "split_that_created_layer": current_diagnostic.get("best_candidate_name"),
            "next_split_accepted_by_score": bool(diagnostic.get("accepted", False)),
            "next_split_adjusted_p": diagnostic.get("max_score_adjusted_p"),
        }
        layers.append(record)
        record["h2_stability"] = h2_stability_diagnostic(
            layers,
            tolerance=float(args.h2_stability_tol),
        )
        _atomic_json(layer_dir / "layer_result.json", record)
        print(
            "[covtree-h2] K=%s h2=%.8f absolute_error=%s "
            "next_split_score_accepted=%s"
            % (
                next_k,
                float(record["h2"]),
                (
                    "%.8f" % float(record["absolute_h2_error"])
                    if "absolute_h2_error" in record
                    else "not_available"
                ),
                bool(diagnostic.get("accepted", False)),
            ),
            flush=True,
        )
        current_diagnostic = diagnostic
        current_sparse_state = sparse_state_out

    accepted_layers = [layer for layer in layers if bool(layer["accepted"])]
    terminal_layer = accepted_layers[-1]
    selected, selection_reason = final_refit_layer(
        accepted_layers,
        stop_reason=str(stop_reason),
    )
    for layer in accepted_layers:
        layer["selected_for_final_refit"] = layer is selected
    selected_summary_path = Path(str(selected["summary"])).expanduser().resolve(
        strict=True
    )
    selected_summary = _read_json(selected_summary_path)
    selected_k = int(selected["K"])
    selected_lam_ratio = float(selected_summary["lasso_selected_lam_ratio"])
    selected_theta = np.asarray(
        selected_summary["var_components_lasso_ml"], dtype=np.float64
    )
    if selected_theta.shape != (selected_k + 1,):
        raise ValueError("Selected layer has a malformed covariance warm start.")
    selected_component_spec = Path(
        str(selected_summary["component_spec"])
    ).expanduser().resolve(strict=True)

    final_dir = out_dir / f"final_refit_K{selected_k}"
    final_dir.mkdir(parents=True, exist_ok=True)
    final_prefix = final_dir / "coherit"
    final_summary_path = Path(str(final_prefix) + ".summary.json")
    if not final_summary_path.is_file():
        partial_outputs = list(final_dir.glob("coherit.*"))
        if partial_outputs:
            raise RuntimeError(
                f"Incomplete automatic final-refit outputs exist; inspect {final_dir}."
            )
        final_command = _final_refit_command(
            python_bin=python_bin,
            sparse_pipeline=sparse_pipeline,
            original_arguments=original_arguments,
            component_spec=selected_component_spec,
            fit_phenotype=fit_phenotype,
            fit_keep=fit_keep,
            prefix=final_prefix,
            fixed_lam_ratio=selected_lam_ratio,
            theta_init=selected_theta,
            verbose=bool(args.verbose),
        )
        print(
            "[covtree-final] fitting training+validation with frozen "
            f"K={selected_k}, lambda ratio={selected_lam_ratio:.8g}, "
            "and selection-theta warm start",
            flush=True,
        )
        _run(final_command, final_dir / "runner.log")

    final_summary = _read_json(final_summary_path)
    if int(final_summary.get("sparse_output_schema_version", -1)) != 7:
        raise RuntimeError("Automatic final refit used an incompatible output schema.")
    if int(final_summary["n_grms"]) != selected_k:
        raise RuntimeError("Automatic final refit returned an unexpected K.")
    for field in (
        "lasso_branch_valid",
        "alpha_theta_pair_usable",
        "lasso_ml_outer_converged",
        "all_requested_estimators_valid",
    ):
        if not bool(final_summary.get(field, False)):
            raise RuntimeError(f"Automatic final refit failed validity check: {field}.")
    if str(final_summary.get("lambda_selection_method")) != "fixed_lam_ratio":
        raise RuntimeError("Automatic final refit did not freeze lambda ratio.")
    if str(final_summary.get("lasso_path_role")) != "frozen_ratio_target_only":
        raise RuntimeError("Automatic final refit did not solve only the frozen ratio.")
    if int(final_summary.get("lasso_path_points_solved", -1)) not in {1, 2}:
        raise RuntimeError("Automatic final refit solved an unexpected Lasso path.")
    if bool(final_summary.get("validation_selection_inside_outer_loop", True)):
        raise RuntimeError("Automatic final refit unexpectedly reselected lambda.")
    returned_ratio = float(final_summary["lasso_selected_lam_ratio"])
    if not math.isclose(
        returned_ratio, selected_lam_ratio, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise RuntimeError("Automatic final refit changed the selected lambda ratio.")
    initial_theta = np.asarray(
        final_summary.get("variance_components_initial"), dtype=np.float64
    )
    if initial_theta.shape != selected_theta.shape or not np.allclose(
        initial_theta, selected_theta, rtol=0.0, atol=1e-12
    ):
        raise RuntimeError("Automatic final refit did not use selection theta warm start.")
    if str(final_summary.get("variance_components_init_source")) != "command_line_json":
        raise RuntimeError("Automatic final refit did not record an explicit warm start.")
    if int(final_summary["n_samples"]) <= int(selected_summary["n_samples"]):
        raise RuntimeError(
            "Automatic final refit did not increase from training to "
            "training+validation samples."
        )
    final_h2 = float(final_summary["h2_chive_guarded"])
    final_accuracy = heritability_accuracy(final_h2, args.true_h2)
    final_top_level_accuracy = {
        f"final_{key}": value
        for key, value in final_accuracy.items()
        if key != "true_h2"
    }
    result = {
        "schema_version": 4,
        "algorithm": "coherit_covtree_heritability_first_v4",
        "case_id": args.case_id,
        "status": "complete",
        "created_at": _now(),
        "stop_reason": stop_reason,
        "selection_reason": selection_reason,
        "last_evaluated_step": int(terminal_layer["step"]),
        "last_evaluated_K": int(terminal_layer["K"]),
        "last_evaluated_h2": float(terminal_layer["h2"]),
        "selected_step": int(selected["step"]),
        "selected_K": int(selected["K"]),
        "selected_h2": float(selected["h2"]),
        **heritability_accuracy(float(selected["h2"]), args.true_h2),
        "selected_summary": selected["summary"],
        "h2_stability": terminal_layer["h2_stability"],
        "selected_h2_stability": selected["h2_stability"],
        "layers": layers,
        "final_h2": final_h2,
        **final_top_level_accuracy,
        "final_refit": {
            "status": "complete",
            "training_samples": int(final_summary["n_samples"]),
            "training_data": "training_plus_validation",
            "partition_frozen_before_refit": True,
            "component_spec": str(selected_component_spec),
            "lambda_ratio_frozen_before_refit": True,
            "fixed_lam_ratio": selected_lam_ratio,
            "lambda_selection_method": "fixed_lam_ratio",
            "theta_warm_started_from_selection": True,
            "theta_source_summary": str(selected_summary_path),
            "h2": final_h2,
            **final_accuracy,
            "support_size": int(final_summary["support_size"]),
            "summary": str(final_summary_path),
        },
        "configuration": {
            "max_k": int(args.max_k),
            "max_splits": (
                int(args.max_splits) if args.max_splits is not None else None
            ),
            "score_alpha": float(args.score_alpha),
            "bootstrap_draws": int(args.bootstrap_draws),
            "h2_stability_window": H2_STABILITY_WINDOW,
            "h2_stability_tolerance": float(args.h2_stability_tol),
            "h2_stability_model_choice": (
                "smallest_K_in_terminal_stability_window"
            ),
            "lambda_selection": "validation_r2_inside_each_fixed_K_inner_fit",
            "split_selection": "covariance_contrast_parametric_max_bootstrap",
            "layer_acceptance": (
                "layer_wise_max_bootstrap_p_and_no_three_layer_h2_plateau"
            ),
            "prediction_role": "lambda_selection_and_audit_only_not_K_selection",
            "final_refit": (
                "automatic_train_plus_validation_with_frozen_partition_"
                "frozen_lambda_ratio_and_selection_theta_warm_start"
            ),
            "true_h2_role": "evaluation_only_not_model_selection",
        },
    }
    _atomic_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
