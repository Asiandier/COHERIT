#!/usr/bin/env python3
"""Outer COHERIT-CovTree loop around the fixed-K validation-lambda pipeline."""
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
        required = {"iid", "lasso_phenotype_prediction_raw"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Prediction table has an incompatible schema: {path}")
        iids: list[str] = []
        prediction: list[float] = []
        for row in reader:
            iids.append(str(row["iid"]))
            prediction.append(float(row["lasso_phenotype_prediction_raw"]))
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
) -> dict[str, float | int]:
    iids, prediction = _read_prediction(prediction_path)
    phenotype = _read_phenotype(phenotype_path)
    if set(iids) != set(phenotype) or len(iids) != len(phenotype):
        raise ValueError("Prediction and validation phenotype IID sets differ.")
    outcome = np.asarray([phenotype[iid] for iid in iids], dtype=np.float64)
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


def paired_prediction_guardrail(
    *,
    previous_prediction_path: Path,
    current_prediction_path: Path,
    phenotype_path: Path,
    bootstrap_draws: int,
    seed: int,
    alpha: float,
) -> dict[str, object]:
    """Reject only when paired validation squared error clearly increases."""
    previous_iids, previous = _read_prediction(previous_prediction_path)
    current_iids, current = _read_prediction(current_prediction_path)
    if previous_iids != current_iids:
        raise ValueError("Paired prediction guardrail requires identical IID order.")
    phenotype = _read_phenotype(phenotype_path)
    if set(previous_iids) != set(phenotype):
        raise ValueError("Guardrail phenotype and prediction IID sets differ.")
    outcome = np.asarray([phenotype[iid] for iid in previous_iids], dtype=np.float64)
    paired_difference = np.square(outcome - current) - np.square(outcome - previous)
    draws = int(bootstrap_draws)
    if draws < 100:
        raise ValueError("prediction guardrail bootstrap_draws must be >= 100.")
    rng = np.random.default_rng(int(seed))
    bootstrap_mean = np.empty(draws, dtype=np.float64)
    n_samples = paired_difference.size
    for start in range(0, draws, 256):
        stop = min(start + 256, draws)
        sample = rng.integers(0, n_samples, size=(stop - start, n_samples))
        bootstrap_mean[start:stop] = np.mean(paired_difference[sample], axis=1)
    lower, upper = np.quantile(
        bootstrap_mean, [0.5 * float(alpha), 1.0 - 0.5 * float(alpha)]
    )
    observed = float(np.mean(paired_difference))
    return {
        "metric": "paired_validation_squared_error_current_minus_previous",
        "observed_mean_difference": observed,
        "confidence_level": float(1.0 - alpha),
        "confidence_interval": [float(lower), float(upper)],
        "bootstrap_draws": draws,
        "seed": int(seed),
        "material_degradation_supported": bool(lower > 0.0),
    }


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
    "--covtree-max-univariate-depth",
    "--covtree-parent-theta-abs-min",
    "--covtree-parent-theta-rel-min",
}


def _base_pipeline_arguments(arguments: Sequence[str]) -> list[str]:
    cleaned: list[str] = []
    index = 0
    while index < len(arguments):
        value = str(arguments[index])
        if value in _VALUE_FLAGS_TO_REPLACE:
            if index + 1 >= len(arguments):
                raise ValueError(f"Pipeline flag lacks a value: {value}")
            index += 2
            continue
        cleaned.append(value)
        index += 1
    return cleaned


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
    parser.add_argument("--ld-score", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-k", type=int, default=1024)
    parser.add_argument("--max-splits", type=int, default=10)
    parser.add_argument("--bootstrap-draws", type=int, default=199)
    parser.add_argument("--bootstrap-seed", type=int, default=20260827)
    parser.add_argument("--score-alpha", type=float, default=0.05)
    parser.add_argument("--rank-rtol", type=float, default=1e-7)
    parser.add_argument("--min-child-markers", type=int, default=16)
    parser.add_argument("--max-univariate-depth", type=int, default=2)
    parser.add_argument("--parent-theta-abs-min", type=float, default=1e-6)
    parser.add_argument("--parent-theta-rel-min", type=float, default=1e-4)
    parser.add_argument("--guardrail-bootstrap-draws", type=int, default=2000)
    parser.add_argument("--guardrail-alpha", type=float, default=0.05)
    parser.add_argument(
        "--sparse-pipeline",
        default=str(REPO_ROOT / "run_sparse_reml_pipeline.py"),
    )
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if int(args.max_k) < 1:
        raise ValueError("max_k must be positive.")
    if int(args.max_splits) < 0:
        raise ValueError("max_splits must be nonnegative.")
    if int(args.bootstrap_draws) < 19:
        raise ValueError("bootstrap_draws must be at least 19.")
    if int(args.guardrail_bootstrap_draws) < 100:
        raise ValueError("guardrail_bootstrap_draws must be at least 100.")
    if int(args.min_child_markers) < 1:
        raise ValueError("min_child_markers must be positive.")
    if not 1 <= int(args.max_univariate_depth) <= 10:
        raise ValueError("max_univariate_depth must lie in 1..10.")
    for name in ("score_alpha", "guardrail_alpha"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or not 0.0 < value < 1.0:
            raise ValueError(f"{name} must lie in (0, 1).")
    if not math.isfinite(float(args.rank_rtol)) or float(args.rank_rtol) <= 0.0:
        raise ValueError("rank_rtol must be finite and positive.")
    for name in ("parent_theta_abs_min", "parent_theta_rel_min"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative.")
    initial_dir = Path(args.initial_fit_dir).expanduser().resolve(strict=True)
    initial_summary_path = initial_dir / "coherit.summary.json"
    initial_prediction_path = initial_dir / "coherit.sparse_prediction.tsv"
    trajectory_path = initial_dir / "h2_trajectory.json"
    for path in (initial_summary_path, initial_prediction_path, trajectory_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    initial_diagnostic_path = Path(args.initial_diagnostic).expanduser().resolve(strict=True)
    ld_score_path = Path(args.ld_score).expanduser().resolve(strict=True)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "covtree_result.json"
    if result_path.exists():
        raise FileExistsError(f"Refusing to overwrite completed result: {result_path}")

    trajectory = _read_json(trajectory_path)
    original_arguments = trajectory.get("pipeline_args")
    if not isinstance(original_arguments, list):
        raise ValueError("Initial h2_trajectory.json lacks pipeline_args.")
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
    initial_metrics = prediction_metrics(initial_prediction_path, validation_phenotype)

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
            "h2": float(initial_summary["h2_chive_guarded"]),
            "theta": initial_summary["var_components_lasso_ml"],
            "support_size": int(initial_summary["support_size"]),
            "elapsed_sec": float(initial_summary["elapsed_sec"]),
        }
    ]
    current_diagnostic = initial_diagnostic
    current_prediction_path = initial_prediction_path
    initial_state_metadata = initial_summary.get("sparse_state_out")
    current_sparse_state = (
        Path(str(initial_state_metadata["path"])).expanduser().resolve(strict=True)
        if isinstance(initial_state_metadata, dict)
        and initial_state_metadata.get("status") == "emitted"
        else None
    )
    stop_reason = None

    for step in range(1, int(args.max_splits) + 1):
        if not bool(current_diagnostic.get("accepted", False)):
            stop_reason = current_diagnostic.get("stopping_reason") or "score_not_accepted"
            break
        next_k = int(current_diagnostic["next_k"])
        if next_k > int(args.max_k):
            stop_reason = "max_k_safety_cap"
            break
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
            str(Path(args.python_bin).expanduser().resolve(strict=True)),
            str(Path(args.sparse_pipeline).expanduser().resolve(strict=True)),
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
            "--covtree-max-univariate-depth",
            str(int(args.max_univariate_depth)),
            "--covtree-parent-theta-abs-min",
            str(float(args.parent_theta_abs_min)),
            "--covtree-parent-theta-rel-min",
            str(float(args.parent_theta_rel_min)),
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
        metrics = prediction_metrics(prediction_path, validation_phenotype)
        guardrail = paired_prediction_guardrail(
            previous_prediction_path=current_prediction_path,
            current_prediction_path=prediction_path,
            phenotype_path=validation_phenotype,
            bootstrap_draws=int(args.guardrail_bootstrap_draws),
            seed=int(args.bootstrap_seed) + 10000 + step,
            alpha=float(args.guardrail_alpha),
        )
        layer_accepted = not bool(guardrail["material_degradation_supported"])
        record = {
            "step": step,
            "K": next_k,
            "accepted": layer_accepted,
            "fit_dir": str(layer_dir),
            "summary": str(summary_path),
            "diagnostic": str(diagnostic_out),
            "prediction": str(prediction_path),
            "validation": metrics,
            "prediction_guardrail": guardrail,
            "h2": float(summary["h2_chive_guarded"]),
            "theta": summary["var_components_lasso_ml"],
            "support_size": int(summary["support_size"]),
            "elapsed_sec": float(summary["elapsed_sec"]),
            "split_that_created_layer": current_diagnostic.get("best_candidate_name"),
            "next_split_accepted_by_score": bool(diagnostic.get("accepted", False)),
            "next_split_adjusted_p": diagnostic.get("max_score_adjusted_p"),
        }
        layers.append(record)
        _atomic_json(layer_dir / "layer_result.json", record)
        print(
            "[covtree] K=%s h2=%.8f validation_R2=%.8f guardrail_degraded=%s"
            % (
                next_k,
                float(record["h2"]),
                float(metrics["correlation_squared"]),
                not layer_accepted,
            ),
            flush=True,
        )
        if not layer_accepted:
            stop_reason = "paired_prediction_guardrail_rejected_split"
            break
        current_diagnostic = diagnostic
        current_prediction_path = prediction_path
        current_sparse_state = sparse_state_out
    else:
        stop_reason = "max_splits"

    accepted_layers = [layer for layer in layers if bool(layer["accepted"])]
    selected = accepted_layers[-1]
    result = {
        "schema_version": 1,
        "algorithm": "coherit_covtree_validation_lambda_v1",
        "case_id": args.case_id,
        "status": "complete",
        "created_at": _now(),
        "stop_reason": stop_reason,
        "selected_step": int(selected["step"]),
        "selected_K": int(selected["K"]),
        "selected_h2": float(selected["h2"]),
        "selected_summary": selected["summary"],
        "layers": layers,
        "configuration": {
            "max_k": int(args.max_k),
            "max_splits": int(args.max_splits),
            "score_alpha": float(args.score_alpha),
            "bootstrap_draws": int(args.bootstrap_draws),
            "max_univariate_depth": int(args.max_univariate_depth),
            "lambda_selection": "validation_r2_inside_each_fixed_K_inner_fit",
            "split_selection": "nuisance_adjusted_reml_max_score_bootstrap",
            "prediction_role": "paired_noninferiority_guardrail_only",
        },
    }
    _atomic_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
