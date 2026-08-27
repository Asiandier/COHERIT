#!/usr/bin/env python3
"""Fit a fixed-K COHERIT partition with validation-selected sparsity."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent
PARENT = REPO_ROOT.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))
PKG_NAME = REPO_ROOT.name
_partition_mod = importlib.import_module(f"{PKG_NAME}.adaptive_partition")
_selection_mod = importlib.import_module(f"{PKG_NAME}.sparsity_selection")
_component_mod = importlib.import_module(f"{PKG_NAME}.component_spec")
_lasso_mod = importlib.import_module(f"{PKG_NAME}.lasso_cd")

AdaptiveComponent = _partition_mod.AdaptiveComponent
single_component = _partition_mod.single_component
validate_partition = _partition_mod.validate_partition
write_component_spec = _partition_mod.write_component_spec
evaluate_prediction_path = _selection_mod.evaluate_prediction_path
read_phenotype_aligned = _selection_mod.read_phenotype_aligned
load_component_specs = _component_mod.load_component_specs


SPARSE_PATH_MODE = "fixed_k_validation_lambda"


INPUT_PATH_NAMES = (
    "bed_prefix",
    "train_pheno_txt",
    "fit_pheno_txt",
    "validation_pheno_txt",
    "test_pheno_txt",
    "covar_txt",
    "train_keep",
    "validation_keep",
    "fit_keep",
    "test_keep",
)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def _atomic_tsv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _resolved_file(value: str, label: str) -> Path:
    path = Path(value).expanduser().resolve(strict=True)
    if not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")
    return path


def _validate_bed_prefix(value: str) -> Path:
    prefix = Path(value).expanduser().resolve(strict=False)
    for suffix in (".bed", ".bim", ".fam"):
        path = Path(str(prefix) + suffix)
        if not path.is_file():
            raise FileNotFoundError(path)
    return prefix


def _count_bim_variants(prefix: Path) -> int:
    with Path(str(prefix) + ".bim").open(encoding="utf-8") as handle:
        count = sum(1 for line in handle if line.strip())
    if count < 1:
        raise ValueError("BIM file contains no variants.")
    return int(count)


def _load_fixed_components(
    source: Path | None,
    *,
    n_variants: int,
) -> list[AdaptiveComponent]:
    """Load and validate a frozen user partition, defaulting to K=1."""
    if source is None:
        return single_component(n_variants)
    specs = load_component_specs(str(source))
    components = [
        AdaptiveComponent(
            name=str(spec.name),
            variant_indices=np.asarray(spec.variant_indices, dtype=np.int64),
            annotation=dict(spec.annotation or {}),
        )
        for spec in specs
    ]
    validate_partition(components, n_variants=int(n_variants))
    return components


def _same_component_membership(
    path: Path,
    components: Sequence[AdaptiveComponent],
) -> bool:
    observed = load_component_specs(str(path))
    return len(observed) == len(components) and all(
        np.array_equal(
            np.asarray(left.variant_indices, dtype=np.int64),
            np.asarray(right.variant_indices, dtype=np.int64),
        )
        for left, right in zip(observed, components)
    )


def _run_command(command: Sequence[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(json.dumps({"started_at": _now(), "argv": list(command)}) + "\n")
        log.flush()
        completed = subprocess.run(
            list(command),
            stdout=log,
            stderr=subprocess.STDOUT,
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
            f"Sparse COHERIT subprocess exited {completed.returncode}; see {log_path}."
        )


def _sparse_command(
    *,
    args: argparse.Namespace,
    component_spec: Path,
    phenotype: Path,
    keep: Path,
    prediction_keep: Path,
    prefix: Path,
    selection_pheno: Path | None,
    selection_output: Path | None,
    fixed_lam_ratio: float | None,
    theta_init: np.ndarray | None,
) -> list[str]:
    if bool(selection_pheno) != bool(selection_output):
        raise ValueError("Selection phenotype and output must be supplied together.")
    if selection_pheno is not None and fixed_lam_ratio is not None:
        raise ValueError("Selection scan and fixed-ratio refit are distinct stages.")
    if selection_pheno is None and fixed_lam_ratio is None:
        raise ValueError("Fixed-K sparse command requires an explicit lambda stage.")
    command = [
        str(args.python_bin),
        str(args.sparse_pipeline),
        "--bed-prefix",
        str(args.bed_prefix),
        "--component-spec",
        str(component_spec),
        "--pheno-txt",
        str(phenotype),
        "--covar-txt",
        str(args.covar_txt),
        "--keep-path",
        str(keep),
        "--prediction-bed-prefix",
        str(args.bed_prefix),
        "--prediction-covar-txt",
        str(args.covar_txt),
        "--prediction-keep-path",
        str(prediction_keep),
        "--out-prefix",
        str(prefix),
        "--device",
        str(args.device),
        "--gpu-budget-gib",
        str(args.gpu_budget_gib),
        "--cpu-threads",
        str(args.cpu_threads),
        "--screen-topk",
        str(args.screen_topk),
        "--candidate-k",
        str(args.candidate_k),
        "--kkt-add-topk",
        str(args.kkt_add_topk),
        "--kkt-max-rounds",
        str(args.kkt_max_rounds),
        "--lasso-lam-min-ratio",
        str(args.lam_min_ratio),
        "--lasso-n-lambda",
        str(args.n_lambda),
        "--lasso-cd-max-iter",
        str(args.lasso_cd_max_iter),
    ]
    if selection_pheno is not None:
        command.extend(
            [
                "--sparsity-validation-pheno-txt",
                str(selection_pheno),
                "--sparsity-validation-out",
                str(selection_output),
                "--validation-early-stopping-lag",
                str(args.validation_early_stopping_lag),
            ]
        )
    elif fixed_lam_ratio is not None:
        command.extend(
            [
                "--lasso-fixed-lam-ratio",
                format(float(fixed_lam_ratio), ".17g"),
            ]
        )
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
    if args.verbose:
        command.append("--verbose")
    return command


def _read_prediction(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        column = "lasso_phenotype_prediction"
        if not reader.fieldnames or not {"sample_index", "iid", column}.issubset(
            reader.fieldnames
        ):
            raise ValueError(f"Sparse prediction schema is incompatible: {path}")
        ids: list[str] = []
        values: list[float] = []
        for expected_index, row in enumerate(reader):
            if int(row["sample_index"]) != expected_index:
                raise ValueError("Prediction sample_index is not contiguous.")
            ids.append(str(row["iid"]))
            values.append(float(row[column]))
    prediction = np.asarray(values, dtype=np.float64)
    if len(ids) != len(set(ids)) or not np.all(np.isfinite(prediction)):
        raise ValueError("Prediction IDs or values are invalid.")
    return ids, prediction


def prediction_metrics(
    prediction_path: Path,
    phenotype_path: Path,
    *,
    phenotype_standardization: dict[str, Any],
) -> dict[str, float | int | None]:
    ids, prediction = _read_prediction(prediction_path)
    outcome = read_phenotype_aligned(str(phenotype_path), ids)
    mean = float(phenotype_standardization["mean"])
    standard_deviation = float(
        phenotype_standardization["standard_deviation"]
    )
    if (
        not np.isfinite(mean)
        or not np.isfinite(standard_deviation)
        or standard_deviation <= 0.0
    ):
        raise ValueError("Phenotype input-standardization metadata is invalid.")
    outcome = (outcome - mean) / standard_deviation
    metrics = evaluate_prediction_path(prediction[:, None], outcome)[0]
    metrics.pop("path_index")
    return {"n": int(outcome.size), **metrics}


def _validate_sparse_summary(
    prefix: Path,
    *,
    expected_method: str,
    expected_k: int = 1,
) -> dict[str, Any]:
    summary_path = Path(str(prefix) + ".summary.json")
    summary = _read_json(summary_path)
    if int(summary.get("sparse_output_schema_version", -1)) != 7:
        raise ValueError("Sparse summary must use output schema 7.")
    standardization = summary.get("input_phenotype_standardization")
    if not isinstance(standardization, dict):
        raise ValueError("Sparse summary lacks input-standardization metadata.")
    if int(expected_k) < 1:
        raise ValueError("expected_k must be positive.")
    if int(summary.get("n_grms", -1)) != int(expected_k):
        raise ValueError(
            "Sparse summary GRM count differs from the expected partition: "
            f"{summary.get('n_grms')} != {int(expected_k)}."
        )
    if not bool(summary.get("lasso_branch_valid", False)):
        raise ValueError("Sparse layer did not return a valid Lasso branch.")
    if summary.get("sparse_prediction", {}).get("status") != "emitted":
        raise ValueError("Sparse layer did not emit predictions.")
    if summary.get("lambda_selection_method") != expected_method:
        raise ValueError("Sparse layer used an unexpected lambda-selection method.")
    if expected_method == "validation_r2":
        role = summary.get("lasso_path_role")
        solved = int(summary.get("lasso_path_points_solved", -1))
        requested = int(summary.get("lasso_path_points_requested", -1))
        if role == "complete_validation_grid_weighted_basil":
            if solved != requested or not bool(
                summary.get("lasso_path_complete", False)
            ):
                raise ValueError(
                    "Fixed-K complete validation path has inconsistent counts."
                )
        elif role == (
            "early_stopped_kkt_certified_validation_prefix_weighted_basil"
        ):
            stopping = summary.get("lasso_validation_early_stopping")
            if not (
                isinstance(stopping, dict)
                and bool(stopping.get("stopped", False))
                and 1 <= solved < requested
                and int(stopping.get("n_evaluated", -1)) == solved
            ):
                raise ValueError(
                    "Fixed-K early-stopped validation prefix lacks a valid "
                    "stopping certificate."
                )
        else:
            raise ValueError(
                "Fixed-K selection did not return a certified validation path."
            )
    elif expected_method == "fixed_lam_ratio":
        if summary.get("lasso_path_role") != "frozen_ratio_target_only":
            raise ValueError("Fixed-K refit did not use the optimized target path.")
        if int(summary.get("lasso_path_points_solved", -1)) not in {1, 2}:
            raise ValueError("Frozen-ratio refit must solve only one or two points.")
    else:
        raise ValueError(f"Unsupported lambda-selection method: {expected_method}")
    return summary


def _existing_comparison(
    path: Path | None,
    *,
    case_id: str,
    fixed_k: int,
    prediction_r2: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {
            "case_id": case_id,
            "method": f"coherit_fixed_k{int(fixed_k)}_validation_lambda",
            "heldout_prediction_r2": float(prediction_r2),
            "iterative_validation_minus_method_r2": 0.0,
        }
    ]
    if path is None:
        return rows
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row.get("case_id") != case_id:
                continue
            value = float(row["heldout_prediction_r2"])
            rows.append(
                {
                    "case_id": case_id,
                    "method": row["method"],
                    "heldout_prediction_r2": value,
                    "iterative_validation_minus_method_r2": float(
                        prediction_r2 - value
                    ),
                }
            )
    return rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--bed-prefix", required=True)
    parser.add_argument(
        "--component-spec",
        default="",
        help=(
            "Frozen fixed-K component partition (.json or .npz). Omit for "
            "a single whole-genome GRM (K=1)."
        ),
    )
    parser.add_argument("--train-pheno-txt", required=True)
    parser.add_argument("--fit-pheno-txt", required=True)
    parser.add_argument("--validation-pheno-txt", required=True)
    parser.add_argument("--test-pheno-txt", required=True)
    parser.add_argument("--covar-txt", required=True)
    parser.add_argument("--train-keep", required=True)
    parser.add_argument("--validation-keep", required=True)
    parser.add_argument("--fit-keep", required=True)
    parser.add_argument("--test-keep", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--lam-min-ratio", type=float, default=1e-3)
    parser.add_argument("--n-lambda", type=int, default=80)
    parser.add_argument("--lasso-cd-max-iter", type=int, default=10000)
    parser.add_argument(
        "--validation-early-stopping-lag", type=int, default=5
    )
    parser.add_argument("--screen-topk", type=int, default=2000)
    parser.add_argument("--candidate-k", type=int, default=256)
    parser.add_argument("--kkt-add-topk", type=int, default=256)
    parser.add_argument("--kkt-max-rounds", type=int, default=20)
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--gpu-budget-gib", type=float, default=0.0)
    parser.add_argument("--cpu-threads", type=int, default=0)
    parser.add_argument(
        "--sparse-pipeline",
        default=str(REPO_ROOT / "run_sparse_reml_pipeline.py"),
    )
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--existing-prediction-results", default="")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if not math.isfinite(args.lam_min_ratio) or not 0.0 < args.lam_min_ratio <= 1.0:
        parser.error("--lam-min-ratio must lie in (0, 1].")
    for name in (
        "n_lambda",
        "lasso_cd_max_iter",
        "validation_early_stopping_lag",
        "screen_topk",
        "candidate_k",
        "kkt_add_topk",
        "kkt_max_rounds",
    ):
        if int(getattr(args, name)) < 1:
            parser.error(f"--{name.replace('_', '-')} must be >= 1.")
    if int(args.screen_topk) < int(args.candidate_k):
        parser.error("--screen-topk must be >= --candidate-k.")
    if float(args.gpu_budget_gib) < 0.0 or int(args.cpu_threads) < 0:
        parser.error("GPU budget and CPU threads must be nonnegative.")
    return args


def _resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    args.bed_prefix = _validate_bed_prefix(args.bed_prefix)
    args.component_spec = (
        _resolved_file(str(args.component_spec), "component_spec")
        if args.component_spec
        else None
    )
    for name in (
        "train_pheno_txt",
        "fit_pheno_txt",
        "validation_pheno_txt",
        "test_pheno_txt",
        "covar_txt",
        "train_keep",
        "validation_keep",
        "fit_keep",
        "test_keep",
        "sparse_pipeline",
        "python_bin",
    ):
        setattr(args, name, _resolved_file(str(getattr(args, name)), name))
    args.out_dir = Path(args.out_dir).expanduser().resolve(strict=False)
    args.existing_prediction_results = (
        _resolved_file(
            args.existing_prediction_results,
            "existing_prediction_results",
        )
        if args.existing_prediction_results
        else None
    )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _resolve_args(parse_args(argv))
    result_path = args.out_dir / "sparsity_result.json"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    n_variants = _count_bim_variants(args.bed_prefix)
    components = _load_fixed_components(
        args.component_spec,
        n_variants=n_variants,
    )
    fixed_k = len(components)
    component_spec = args.out_dir / f"component_spec_fixed_K{fixed_k}.npz"
    if not component_spec.exists():
        write_component_spec(
            component_spec,
            components,
            provenance={
                "algorithm": SPARSE_PATH_MODE,
                "case_id": args.case_id,
                "fixed_K": int(fixed_k),
                "n_variants": n_variants,
                "source_component_spec": (
                    str(args.component_spec)
                    if args.component_spec is not None
                    else None
                ),
            },
        )
    elif not _same_component_membership(component_spec, components):
        raise ValueError(
            f"Existing fixed component snapshot differs: {component_spec}"
        )

    config = {
        "schema_version": 2,
        "case_id": args.case_id,
        "algorithm": {
            "name": "fixed_k_validation_lambda_coherit",
            "sparse_path_mode": SPARSE_PATH_MODE,
            "fixed_K": int(fixed_k),
            "selection_metric": "squared_pearson_correlation_total_phenotype_prediction",
            "lam_min_ratio": float(args.lam_min_ratio),
            "n_lambda": int(args.n_lambda),
            "lasso_cd_max_iter": int(args.lasso_cd_max_iter),
            "lambda_path": "full_marker_kkt_certified_prefix",
            "validation_early_stopping": {
                "enabled": True,
                "stopping_lag": int(
                    args.validation_early_stopping_lag
                ),
                "rule": "earlier_max_exceeds_latest_window_max",
            },
            "global_kkt_solver": "weighted_basil",
            "selection_timing": "before_every_variance_component_update",
            "outer_update": "validation_selected_alpha_then_variance_components",
            "selection_samples": "training_to_validation",
            "final_refit": (
                "freeze_partition_and_selected_lambda_ratio_then_refit_"
                "train_plus_validation"
            ),
            "final_evaluation": "heldout_test_once_after_selection",
        },
        "inputs": {
            **{name: str(getattr(args, name)) for name in INPUT_PATH_NAMES},
            "component_spec_source": (
                str(args.component_spec) if args.component_spec is not None else None
            ),
            "component_spec_snapshot": str(component_spec),
        },
        "runtime": {
            "python_bin": str(args.python_bin),
            "sparse_pipeline": str(args.sparse_pipeline),
            "device": args.device,
            "gpu_budget_gib": float(args.gpu_budget_gib),
            "cpu_threads": int(args.cpu_threads),
            "screen_topk": int(args.screen_topk),
            "candidate_k": int(args.candidate_k),
            "kkt_add_topk": int(args.kkt_add_topk),
            "kkt_max_rounds": int(args.kkt_max_rounds),
        },
        "source_sha256": {
            "driver": _sha256(Path(__file__).resolve()),
            "sparse_pipeline": _sha256(args.sparse_pipeline),
            "sparsity_selection": _sha256(Path(_selection_mod.__file__).resolve()),
            "lasso_cd": _sha256(Path(_lasso_mod.__file__).resolve()),
            "component_spec_source": (
                _sha256(args.component_spec)
                if args.component_spec is not None
                else None
            ),
        },
        "created_at": _now(),
    }
    config_path = args.out_dir / "run_config.json"
    if config_path.exists():
        existing = _read_json(config_path)
        existing.pop("created_at", None)
        comparable = dict(config)
        comparable.pop("created_at", None)
        if existing != comparable:
            raise ValueError(
                f"Existing configuration differs; use another --out-dir: {config_path}"
            )
    else:
        _atomic_json(config_path, config)

    if result_path.is_file():
        cached = _read_json(result_path)
        if cached.get("status") != "complete" or cached.get("case_id") != args.case_id:
            raise ValueError(f"Existing result is incompatible: {result_path}")
        print(f"[cached] {result_path}")
        return 0

    selection_dir = args.out_dir / "selection_fit"
    selection_dir.mkdir(parents=True, exist_ok=True)
    selection_prefix = selection_dir / "coherit"
    selection_output = selection_dir / "validation_path.json"
    selection_summary_path = Path(str(selection_prefix) + ".summary.json")
    if not selection_summary_path.exists():
        partial = list(selection_dir.glob("coherit.*")) + list(
            selection_dir.glob("validation_path*")
        )
        if partial:
            raise RuntimeError(
                f"Incomplete selection outputs exist; inspect {selection_dir}."
            )
        command = _sparse_command(
            args=args,
            component_spec=component_spec,
            phenotype=args.train_pheno_txt,
            keep=args.train_keep,
            prediction_keep=args.validation_keep,
            prefix=selection_prefix,
            selection_pheno=args.validation_pheno_txt,
            selection_output=selection_output,
            fixed_lam_ratio=None,
            theta_init=None,
        )
        print(
            f"[fixed-K validation-lambda] fitting K={fixed_k} with "
            "validation-selected lambda inside every outer iteration",
            flush=True,
        )
        _run_command(command, selection_dir / "runner.log")

    selection_summary = _validate_sparse_summary(
        selection_prefix,
        expected_method="validation_r2",
        expected_k=fixed_k,
    )
    if not bool(selection_summary.get("validation_selection_inside_outer_loop")):
        raise ValueError(
            "Selection fit did not perform validation selection inside the outer loop."
        )
    selection_payload = _read_json(selection_output)
    if selection_payload.get("test_phenotype_used") is not False:
        raise ValueError("Selection output does not prove test-phenotype isolation.")
    if selection_payload.get("selection_role") != (
        "inside_every_alpha_theta_outer_iteration"
    ):
        raise ValueError("Selection output has the wrong validation timing.")
    selected = selection_payload.get("final_selected")
    if not isinstance(selected, dict):
        raise ValueError("Selection output lacks the selected path point.")
    selected_ratio = float(selected["lam_ratio"])
    selected_validation_r2 = float(selected["correlation_squared"])
    if not float(args.lam_min_ratio) <= selected_ratio <= 1.0:
        raise ValueError("Selected lambda ratio lies outside the requested path.")
    selection_prediction_path = Path(
        str(selection_prefix) + ".sparse_prediction.tsv"
    )
    selection_prediction_metrics = prediction_metrics(
        selection_prediction_path,
        args.validation_pheno_txt,
        phenotype_standardization=selection_summary[
            "input_phenotype_standardization"
        ],
    )
    if not math.isclose(
        float(selection_prediction_metrics["correlation_squared"]),
        selected_validation_r2,
        rel_tol=0.0,
        abs_tol=5e-5,
    ):
        raise ValueError(
            "Final validation prediction does not reproduce the selected "
            "outer-loop path R2."
        )

    final_dir = args.out_dir / "final_refit"
    final_dir.mkdir(parents=True, exist_ok=True)
    final_prefix = final_dir / "coherit"
    final_summary_path = Path(str(final_prefix) + ".summary.json")
    if not final_summary_path.exists():
        partial = list(final_dir.glob("coherit.*"))
        if partial:
            raise RuntimeError(f"Incomplete final outputs exist; inspect {final_dir}.")
        command = _sparse_command(
            args=args,
            component_spec=component_spec,
            phenotype=args.fit_pheno_txt,
            keep=args.fit_keep,
            prediction_keep=args.test_keep,
            prefix=final_prefix,
            selection_pheno=None,
            selection_output=None,
            fixed_lam_ratio=selected_ratio,
            theta_init=np.asarray(
                selection_summary["var_components_lasso_ml"],
                dtype=np.float64,
            ),
        )
        print(
            f"[fixed-K validation-lambda] final refit with frozen K={fixed_k} "
            f"and lambda/lambda_max={selected_ratio:.8g}",
            flush=True,
        )
        _run_command(command, final_dir / "runner.log")

    final_summary = _validate_sparse_summary(
        final_prefix,
        expected_method="fixed_lam_ratio",
        expected_k=fixed_k,
    )
    final_selected_ratio = float(final_summary["lasso_selected_lam_ratio"])
    if not math.isclose(final_selected_ratio, selected_ratio, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("Final refit did not use the frozen validation lambda ratio.")
    test_prediction_path = Path(str(final_prefix) + ".sparse_prediction.tsv")
    test_metrics = prediction_metrics(
        test_prediction_path,
        args.test_pheno_txt,
        phenotype_standardization=final_summary[
            "input_phenotype_standardization"
        ],
    )
    comparison = _existing_comparison(
        args.existing_prediction_results,
        case_id=args.case_id,
        fixed_k=fixed_k,
        prediction_r2=float(test_metrics["correlation_squared"]),
    )
    comparison_path = args.out_dir / "prediction_comparison.tsv"
    _atomic_tsv(
        comparison_path,
        comparison,
        (
            "case_id",
            "method",
            "heldout_prediction_r2",
            "iterative_validation_minus_method_r2",
        ),
    )

    result = {
        "schema_version": 3,
        "status": "complete",
        "case_id": args.case_id,
        "completed_at": _now(),
        "algorithm": config["algorithm"],
        "sparse_path_mode": SPARSE_PATH_MODE,
        "fixed_K": int(fixed_k),
        "selection": {
            "training_samples": int(selection_summary["n_samples"]),
            "validation_samples": int(selection_payload["n_validation_samples"]),
            "selected_lam_ratio": selected_ratio,
            "selected_lam": float(selected["lam"]),
            "selected_support_size": int(selected["support_size"]),
            "selected_validation_r2": selected_validation_r2,
            "final_validation_prediction_metrics": (
                selection_prediction_metrics
            ),
            "selection_role": selection_payload["selection_role"],
            "n_path_selections": int(selection_payload["n_path_selections"]),
            "outer_converged": bool(selection_payload["outer_converged"]),
            "outer_stop_reason": selection_payload["outer_stop_reason"],
            "path_json": str(selection_output),
            "path_tsv": str(selection_output)[:-5] + ".path.tsv",
            "test_phenotype_used": False,
        },
        "final_refit": {
            "training_samples": int(final_summary["n_samples"]),
            "partition_frozen_before_refit": True,
            "lambda_ratio_frozen_before_refit": True,
            "selected_lam_ratio": final_selected_ratio,
            "lasso_path_role": final_summary.get("lasso_path_role"),
            "lasso_path_points_solved": int(
                final_summary.get("lasso_path_points_solved", 0)
            ),
            "h2_chive": float(final_summary["h2_chive_guarded"]),
            "support_size": int(final_summary["support_size"]),
            "test_metrics": test_metrics,
            "test_phenotype_used_only_after_model_selection": True,
            "summary": str(final_summary_path),
            "prediction": str(test_prediction_path),
        },
        "comparison": comparison,
        "comparison_path": str(comparison_path),
        "component_spec": str(component_spec),
        "run_config": str(config_path),
    }
    _atomic_json(result_path, result)
    print(
        f"[fixed-K validation-lambda] complete K={fixed_k} "
        "validation_R2=%.8f "
        "test_R2=%.8f support=%s"
        % (
            selected_validation_r2,
            float(test_metrics["correlation_squared"]),
            int(final_summary["support_size"]),
        ),
        flush=True,
    )
    print(f"[fixed-K validation-lambda] result -> {result_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
