#!/usr/bin/env python3
"""Prediction-first Adaptive COHERIT with iterative four-way GRM splits."""
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

AdaptiveComponent = _partition_mod.AdaptiveComponent
covariance_preserving_warm_start = (
    _partition_mod.covariance_preserving_warm_start
)
four_way_split = _partition_mod.four_way_split
single_component = _partition_mod.single_component
validate_partition = _partition_mod.validate_partition
write_component_spec = _partition_mod.write_component_spec


SIGNAL_HIGH_FRACTION = 0.15
MAX_ADAPTIVE_DEPTH = 5
VALIDATION_DECLINE_TOLERANCE = 1e-12
INPUT_PATH_NAMES = (
    "bed_prefix",
    "ld_score",
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
        while True:
            block = handle.read(8 << 20)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def validation_declined(
    previous_r2: float,
    current_r2: float,
    *,
    tolerance: float = VALIDATION_DECLINE_TOLERANCE,
) -> bool:
    """Return true only for a genuine one-step validation R2 decrease."""
    previous = float(previous_r2)
    current = float(current_r2)
    tol = float(tolerance)
    if not math.isfinite(previous) or not math.isfinite(current):
        raise ValueError("Validation R2 values must be finite.")
    if not math.isfinite(tol) or tol < 0.0:
        raise ValueError("Validation decline tolerance must be finite and nonnegative.")
    return current < previous - tol


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


def load_aligned_ld_score(
    ld_score_path: Path,
    bim_path: Path,
) -> np.ndarray:
    """Load LD scores while proving exact BIM-ID order alignment."""
    scores: list[float] = []
    with ld_score_path.open(encoding="utf-8", newline="") as ld_handle:
        reader = csv.DictReader(ld_handle, delimiter="\t")
        if not reader.fieldnames or not {"ID", "ld_score"}.issubset(
            reader.fieldnames
        ):
            raise ValueError("LD-score table must contain ID and ld_score columns.")
        with bim_path.open(encoding="utf-8") as bim_handle:
            row_count = 0
            for row_count, (ld_row, bim_line) in enumerate(
                zip(reader, bim_handle, strict=True),
                start=1,
            ):
                fields = bim_line.split()
                if len(fields) < 2:
                    raise ValueError(f"Malformed BIM row {row_count}: {bim_path}")
                if str(ld_row["ID"]) != fields[1]:
                    raise ValueError(
                        "LD-score/BIM marker order mismatch at row "
                        f"{row_count}: {ld_row['ID']!r} != {fields[1]!r}."
                    )
                score = float(ld_row["ld_score"])
                if not math.isfinite(score):
                    raise ValueError(f"Non-finite LD score at row {row_count}.")
                scores.append(score)
    result = np.asarray(scores, dtype=np.float64)
    if result.size == 0:
        raise ValueError("LD-score table is empty.")
    return result


def _read_phenotype(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) < 3:
                raise ValueError(f"Malformed phenotype row {line_number}: {path}")
            iid = fields[1]
            value = float(fields[2])
            if iid in values or not math.isfinite(value):
                raise ValueError(f"Invalid phenotype IID/value at row {line_number}.")
            values[iid] = value
    if not values:
        raise ValueError(f"Phenotype file is empty: {path}")
    return values


def _read_prediction(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected = {
            "sample_index",
            "iid",
            "lasso_phenotype_prediction_raw",
        }
        if not reader.fieldnames or not expected.issubset(reader.fieldnames):
            raise ValueError(f"Sparse prediction schema is incompatible: {path}")
        iids: list[str] = []
        values: list[float] = []
        for expected_index, row in enumerate(reader):
            if int(row["sample_index"]) != expected_index:
                raise ValueError(f"Prediction sample_index is not contiguous: {path}")
            iids.append(row["iid"])
            values.append(float(row["lasso_phenotype_prediction_raw"]))
    prediction = np.asarray(values, dtype=np.float64)
    if len(iids) != len(set(iids)) or not np.all(np.isfinite(prediction)):
        raise ValueError(f"Prediction IIDs/values are invalid: {path}")
    return iids, prediction


def prediction_metrics(
    prediction_path: Path,
    phenotype_path: Path,
) -> dict[str, float | int]:
    iids, prediction = _read_prediction(prediction_path)
    phenotype = _read_phenotype(phenotype_path)
    if set(iids) != set(phenotype) or len(iids) != len(phenotype):
        raise ValueError(
            f"Prediction and phenotype IID sets differ: {prediction_path}, "
            f"{phenotype_path}"
        )
    outcome = np.asarray([phenotype[iid] for iid in iids], dtype=np.float64)
    centered_prediction = prediction - float(prediction.mean())
    centered_outcome = outcome - float(outcome.mean())
    prediction_ss = float(centered_prediction @ centered_prediction)
    outcome_ss = float(centered_outcome @ centered_outcome)
    if prediction_ss <= 0.0 or outcome_ss <= 0.0:
        raise ValueError("Prediction or phenotype has zero centered variance.")
    covariance = float(centered_prediction @ centered_outcome)
    correlation = covariance / math.sqrt(prediction_ss * outcome_ss)
    correlation = min(1.0, max(-1.0, correlation))
    return {
        "n": int(outcome.size),
        "correlation": float(correlation),
        "correlation_squared": float(correlation * correlation),
        "mse": float(np.mean((outcome - prediction) ** 2)),
        "calibration_slope": float(covariance / prediction_ss),
        "predictive_r2": float(
            1.0 - np.sum((outcome - prediction) ** 2) / outcome_ss
        ),
    }


def _load_signal_score(path: Path, n_variants: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        required = {"source_variant_index", "signal_score"}
        if not required.issubset(payload.files):
            raise ValueError(f"Adaptive marker-score file is incomplete: {path}")
        indices = np.asarray(payload["source_variant_index"], dtype=np.int64)
        score = np.asarray(payload["signal_score"], dtype=np.float64)
    if not np.array_equal(indices, np.arange(int(n_variants), dtype=np.int64)):
        raise ValueError("Adaptive marker scores are not in exact source order.")
    if score.shape != (int(n_variants),) or not np.all(np.isfinite(score)):
        raise ValueError("Adaptive marker score is malformed.")
    return score


def _validate_sparse_layer(
    *,
    prefix: Path,
    expected_k: int,
    prediction_phenotype: Path,
    marker_score_path: Path | None,
    n_variants: int,
) -> tuple[dict[str, Any], dict[str, float | int], np.ndarray | None]:
    summary_path = Path(str(prefix) + ".summary.json")
    prediction_path = Path(str(prefix) + ".sparse_prediction.tsv")
    summary = _read_json(summary_path)
    if int(summary.get("n_grms", -1)) != int(expected_k):
        raise ValueError(f"Sparse layer used K={summary.get('n_grms')}, expected {expected_k}.")
    if not bool(summary.get("lasso_branch_valid", False)):
        raise ValueError("Sparse layer did not emit a valid COHERIT branch.")
    if summary.get("sparse_prediction", {}).get("status") != "emitted":
        raise ValueError("Sparse layer did not emit validation/test prediction.")
    theta = np.asarray(summary.get("var_components_lasso_ml"), dtype=np.float64)
    if theta.shape != (int(expected_k) + 1,) or not np.all(np.isfinite(theta)):
        raise ValueError("Sparse layer returned invalid variance components.")
    metrics = prediction_metrics(prediction_path, prediction_phenotype)
    score = None
    if marker_score_path is not None:
        score = _load_signal_score(marker_score_path, n_variants)
        marker_metadata = summary.get("adaptive_marker_score")
        if not isinstance(marker_metadata, dict) or marker_metadata.get("status") != "emitted":
            raise ValueError("Sparse summary lacks adaptive marker-score metadata.")
    return summary, metrics, score


def _component_spec_matches(
    path: Path,
    components: Sequence[AdaptiveComponent],
) -> bool:
    """Check exact source-marker membership before reusing a fitted layer."""
    with np.load(path, allow_pickle=False) as payload:
        expected_keys = {f"arr_{index}" for index in range(len(components))}
        observed_keys = {key for key in payload.files if key.startswith("arr_")}
        if observed_keys != expected_keys:
            return False
        return all(
            np.array_equal(
                np.asarray(payload[f"arr_{index}"], dtype=np.int64),
                np.asarray(component.variant_indices, dtype=np.int64),
            )
            for index, component in enumerate(components)
        )


def _load_reusable_layer_records(
    reuse_run_dir: Path | None,
    *,
    args: argparse.Namespace,
) -> dict[int, dict[str, Any]]:
    """Load compatible completed layers from an earlier shallower search."""
    if reuse_run_dir is None:
        return {}
    result = _read_json(reuse_run_dir / "adaptive_result.json")
    config = _read_json(reuse_run_dir / "run_config.json")
    if result.get("status") != "complete" or result.get("case_id") != args.case_id:
        raise ValueError(f"Reusable adaptive result is incompatible: {reuse_run_dir}")
    if config.get("case_id") != args.case_id:
        raise ValueError(f"Reusable run configuration has the wrong case: {reuse_run_dir}")

    prior_inputs = config.get("inputs")
    if not isinstance(prior_inputs, dict):
        raise ValueError("Reusable run configuration lacks input provenance.")
    for name in INPUT_PATH_NAMES:
        prior_value = prior_inputs.get(name)
        if prior_value is None:
            raise ValueError(f"Reusable run configuration lacks input {name}.")
        prior_path = Path(str(prior_value)).expanduser().resolve(strict=False)
        current_path = Path(str(getattr(args, name))).expanduser().resolve(strict=False)
        if prior_path != current_path:
            raise ValueError(f"Reusable run input differs for {name}.")

    prior_algorithm = config.get("algorithm")
    if not isinstance(prior_algorithm, dict):
        raise ValueError("Reusable run configuration lacks algorithm provenance.")
    expected_algorithm_values = {
        "name": "prediction_first_adaptive_coherit_iterative_four_split",
        "signal_high_fraction": SIGNAL_HIGH_FRACTION,
        "ld_split": "within_signal_bin_median_stable_rank",
        "validation_metric": "squared_pearson_correlation",
        "marker_information_estimator": "sample_space_rademacher_hutchinson",
        "marker_score_probes": int(args.marker_score_probes),
        "marker_score_seed": int(args.marker_score_seed),
    }
    for key, expected in expected_algorithm_values.items():
        if prior_algorithm.get(key) != expected:
            raise ValueError(f"Reusable run algorithm differs for {key}.")
    prior_sparse_hash = config.get("source_sha256", {}).get("sparse_pipeline")
    if prior_sparse_hash != _sha256(args.sparse_pipeline):
        raise ValueError("Reusable run used a different sparse pipeline source.")

    records: dict[int, dict[str, Any]] = {}
    for raw_record in result.get("valid_layers", []):
        if not isinstance(raw_record, dict):
            raise ValueError("Reusable run contains a malformed layer record.")
        depth = int(raw_record.get("depth", -1))
        if depth < 0 or int(raw_record.get("K", -1)) != 4**depth:
            raise ValueError("Reusable run contains an invalid depth/K pair.")
        if depth in records:
            raise ValueError("Reusable run contains duplicate depths.")
        records[depth] = raw_record
    return records


def _reusable_layer_artifacts(
    record: Mapping[str, Any] | None,
    *,
    components: Sequence[AdaptiveComponent],
    need_marker_score: bool,
) -> tuple[Path, Path, Path | None] | None:
    """Resolve a reusable component spec, sparse prefix, and optional score."""
    if record is None:
        return None
    marker_metadata = record.get("marker_score")
    if need_marker_score and not isinstance(marker_metadata, dict):
        return None
    component_spec = Path(str(record["component_spec"])).expanduser().resolve(
        strict=True
    )
    if not _component_spec_matches(component_spec, components):
        raise ValueError(
            f"Reusable component membership differs from the current split: {component_spec}"
        )
    marker_score_path = None
    if need_marker_score:
        marker_score_path = Path(str(marker_metadata["path"])).expanduser().resolve(
            strict=True
        )
    prefix = component_spec.parent / "coherit"
    return component_spec, prefix, marker_score_path


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
    marker_score_path: Path | None,
    marker_score_seed: int,
    theta_init: np.ndarray | None,
) -> list[str]:
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
    ]
    if marker_score_path is not None:
        command.extend(
            [
                "--marker-score-out",
                str(marker_score_path),
                "--marker-score-probes",
                str(args.marker_score_probes),
                "--marker-score-seed",
                str(marker_score_seed),
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


def _existing_comparison(
    path: Path | None,
    *,
    case_id: str,
    adaptive_r2: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {
            "case_id": case_id,
            "method": "coherit_adaptive",
            "heldout_prediction_r2": float(adaptive_r2),
            "adaptive_minus_method_r2": 0.0,
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
                    "adaptive_minus_method_r2": float(adaptive_r2 - value),
                }
            )
    return rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--bed-prefix", required=True)
    parser.add_argument("--ld-score", required=True)
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
    parser.add_argument("--max-depth", type=int, default=MAX_ADAPTIVE_DEPTH)
    parser.add_argument("--marker-score-probes", type=int, default=32)
    parser.add_argument("--marker-score-seed", type=int, default=20260825)
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--gpu-budget-gib", type=float, default=0.0)
    parser.add_argument("--cpu-threads", type=int, default=0)
    parser.add_argument(
        "--sparse-pipeline",
        default=str(REPO_ROOT / "run_sparse_reml_pipeline.py"),
    )
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--existing-prediction-results", default="")
    parser.add_argument(
        "--reuse-run-dir",
        default="",
        help=(
            "Completed shallower Adaptive COHERIT run whose compatible fitted "
            "layers may be reused. A layer is rerun when the deeper search "
            "requires a marker score that the earlier run did not emit."
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if not 0 <= int(args.max_depth) <= MAX_ADAPTIVE_DEPTH:
        parser.error(f"--max-depth must be in 0..{MAX_ADAPTIVE_DEPTH}.")
    if int(args.marker_score_probes) < 1:
        parser.error("--marker-score-probes must be >= 1.")
    if float(args.gpu_budget_gib) < 0.0:
        parser.error("--gpu-budget-gib must be nonnegative.")
    if int(args.cpu_threads) < 0:
        parser.error("--cpu-threads must be nonnegative.")
    return args


def _resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    args.bed_prefix = _validate_bed_prefix(args.bed_prefix)
    for name in (
        "ld_score",
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
    args.reuse_run_dir = (
        Path(args.reuse_run_dir).expanduser().resolve(strict=True)
        if args.reuse_run_dir
        else None
    )
    if args.reuse_run_dir is not None:
        if not args.reuse_run_dir.is_dir():
            raise NotADirectoryError(args.reuse_run_dir)
        if args.reuse_run_dir == args.out_dir:
            raise ValueError("--reuse-run-dir must differ from --out-dir.")
    args.existing_prediction_results = (
        _resolved_file(args.existing_prediction_results, "existing_prediction_results")
        if args.existing_prediction_results
        else None
    )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _resolve_args(parse_args(argv))
    result_path = args.out_dir / "adaptive_result.json"
    if result_path.is_file():
        cached = _read_json(result_path)
        if cached.get("status") != "complete" or cached.get("case_id") != args.case_id:
            raise ValueError(f"Existing adaptive result is incompatible: {result_path}")
        print(f"[cached] {result_path}")
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ld_score = load_aligned_ld_score(
        args.ld_score,
        Path(str(args.bed_prefix) + ".bim"),
    )
    n_variants = int(ld_score.size)
    components = single_component(n_variants)
    validate_partition(components, n_variants=n_variants)

    config = {
        "schema_version": 1,
        "case_id": args.case_id,
        "algorithm": {
            "name": "prediction_first_adaptive_coherit_iterative_four_split",
            "max_depth": int(args.max_depth),
            "k_path": [4**depth for depth in range(int(args.max_depth) + 1)],
            "signal_high_fraction": SIGNAL_HIGH_FRACTION,
            "ld_split": "within_signal_bin_median_stable_rank",
            "validation_metric": "squared_pearson_correlation",
            "early_stop_rule": "first_strict_validation_r2_decrease",
            "validation_decline_tolerance": VALIDATION_DECLINE_TOLERANCE,
            "marker_information_estimator": "sample_space_rademacher_hutchinson",
            "marker_score_probes": int(args.marker_score_probes),
            "marker_score_seed": int(args.marker_score_seed),
        },
        "inputs": {
            name: str(getattr(args, name))
            for name in INPUT_PATH_NAMES
        },
        "runtime": {
            "python_bin": str(args.python_bin),
            "sparse_pipeline": str(args.sparse_pipeline),
            "device": args.device,
            "gpu_budget_gib": float(args.gpu_budget_gib),
            "cpu_threads": int(args.cpu_threads),
            "reuse_run_dir": (
                str(args.reuse_run_dir) if args.reuse_run_dir is not None else None
            ),
        },
        "source_sha256": {
            "adaptive_driver": _sha256(Path(__file__).resolve()),
            "adaptive_partition": _sha256(Path(_partition_mod.__file__).resolve()),
            "sparse_pipeline": _sha256(args.sparse_pipeline),
        },
        "created_at": _now(),
    }
    config_path = args.out_dir / "run_config.json"
    if config_path.exists():
        existing_config = _read_json(config_path)
        comparable_existing = dict(existing_config)
        comparable_current = dict(config)
        comparable_existing.pop("created_at", None)
        comparable_current.pop("created_at", None)
        if comparable_existing != comparable_current:
            raise ValueError(
                f"Existing run configuration differs; use another --out-dir: {config_path}"
            )
    else:
        _atomic_json(config_path, config)

    reusable_layers = _load_reusable_layer_records(
        args.reuse_run_dir,
        args=args,
    )
    layer_records: list[dict[str, Any]] = []
    layer_components: list[list[AdaptiveComponent]] = []
    layer_failures: list[dict[str, Any]] = []
    search_stop: dict[str, Any] | None = None
    theta_init = None

    for depth in range(int(args.max_depth) + 1):
        expected_k = 4**depth
        if len(components) != expected_k:
            raise RuntimeError(
                f"Adaptive partition has K={len(components)} at depth={depth}."
        )
        layer_dir = args.out_dir / f"depth_{depth}_K{expected_k}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        need_marker_score = depth < int(args.max_depth)
        reused_artifacts = _reusable_layer_artifacts(
            reusable_layers.get(depth),
            components=components,
            need_marker_score=need_marker_score,
        )
        reused_from = None
        if reused_artifacts is not None:
            component_spec, prefix, marker_score_path = reused_artifacts
            reused_from = str(component_spec.parent)
            print(
                f"[adaptive] reusing depth={depth} K={expected_k} from {reused_from}",
                flush=True,
            )
        else:
            component_spec = layer_dir / "component_spec.npz"
            if not component_spec.exists():
                write_component_spec(
                    component_spec,
                    components,
                    provenance={
                        "algorithm": "adaptive_coherit_iterative_four_split",
                        "case_id": args.case_id,
                        "depth": int(depth),
                        "signal_high_fraction": SIGNAL_HIGH_FRACTION,
                    },
                )
            prefix = layer_dir / "coherit"
            marker_score_path = (
                layer_dir / "marker_score.npz" if need_marker_score else None
            )
        summary_path = Path(str(prefix) + ".summary.json")
        try:
            if reused_artifacts is None and not summary_path.is_file():
                partial_outputs = list(layer_dir.glob("coherit.*"))
                if partial_outputs:
                    raise RuntimeError(
                        "Incomplete sparse layer outputs already exist; use a new "
                        f"--out-dir or inspect {layer_dir}."
                    )
                command = _sparse_command(
                    args=args,
                    component_spec=component_spec,
                    phenotype=args.train_pheno_txt,
                    keep=args.train_keep,
                    prediction_keep=args.validation_keep,
                    prefix=prefix,
                    marker_score_path=marker_score_path,
                    marker_score_seed=int(args.marker_score_seed) + depth,
                    theta_init=theta_init,
                )
                print(f"[adaptive] fitting depth={depth} K={expected_k}", flush=True)
                _run_command(command, layer_dir / "runner.log")
            summary, metrics, signal_score = _validate_sparse_layer(
                prefix=prefix,
                expected_k=expected_k,
                prediction_phenotype=args.validation_pheno_txt,
                marker_score_path=marker_score_path,
                n_variants=n_variants,
            )
        except Exception as error:
            failure = {
                "depth": int(depth),
                "K": int(expected_k),
                "error": f"{type(error).__name__}: {error}",
                "recorded_at": _now(),
            }
            layer_failures.append(failure)
            search_stop = {
                "reason": "layer_failure",
                "failed_depth": int(depth),
                "failed_K": int(expected_k),
            }
            _atomic_json(args.out_dir / "layer_failures.json", {"failures": layer_failures})
            if not layer_records:
                raise
            print(
                f"[adaptive] stopping after failed depth={depth}: {error}",
                flush=True,
            )
            break

        theta_fit = np.asarray(
            summary["var_components_lasso_ml"], dtype=np.float64
        )
        record = {
            "depth": int(depth),
            "K": int(expected_k),
            "component_spec": str(component_spec),
            "validation": metrics,
            "h2_chive": float(summary["h2_chive_guarded"]),
            "support_size": int(summary["support_size"]),
            "var_components_lasso_ml": theta_fit.tolist(),
            "elapsed_sec": float(summary["elapsed_sec"]),
            "outer_stop_reason": summary.get("outer_stop_reason"),
            "marker_score": summary.get("adaptive_marker_score"),
            "executed_this_run": reused_artifacts is None,
            "reused_from": reused_from,
        }
        _atomic_json(layer_dir / "layer_result.json", record)
        layer_records.append(record)
        layer_components.append(list(components))
        print(
            "[adaptive] depth=%s K=%s validation_R2=%.8f support=%s"
            % (
                depth,
                expected_k,
                float(metrics["correlation_squared"]),
                int(summary["support_size"]),
            ),
            flush=True,
        )

        if len(layer_records) >= 2:
            previous_r2 = float(
                layer_records[-2]["validation"]["correlation_squared"]
            )
            current_r2 = float(metrics["correlation_squared"])
            if validation_declined(previous_r2, current_r2):
                search_stop = {
                    "reason": "validation_r2_decreased",
                    "previous_depth": int(layer_records[-2]["depth"]),
                    "previous_K": int(layer_records[-2]["K"]),
                    "previous_validation_r2": previous_r2,
                    "declined_depth": int(depth),
                    "declined_K": int(expected_k),
                    "declined_validation_r2": current_r2,
                }
                print(
                    "[adaptive] early stop: validation_R2 decreased "
                    f"{previous_r2:.8f} -> {current_r2:.8f}",
                    flush=True,
                )
                break

        if depth >= int(args.max_depth):
            search_stop = {
                "reason": "maximum_depth_reached",
                "depth": int(depth),
                "K": int(expected_k),
            }
            break

        assert signal_score is not None
        children = four_way_split(
            components,
            signal_score=signal_score,
            ld_score=ld_score,
            high_fraction=SIGNAL_HIGH_FRACTION,
            child_depth=depth + 1,
        )
        theta_init = covariance_preserving_warm_start(
            theta_fit,
            components,
            children,
        )
        components = children

    best_r2 = max(
        float(record["validation"]["correlation_squared"])
        for record in layer_records
    )
    tied = [
        index
        for index, record in enumerate(layer_records)
        if best_r2 - float(record["validation"]["correlation_squared"]) <= 1e-12
    ]
    selected_index = min(tied, key=lambda index: int(layer_records[index]["depth"]))
    selected = layer_records[selected_index]
    selected_components = layer_components[selected_index]
    selected_k = int(selected["K"])
    selected_depth = int(selected["depth"])

    final_dir = args.out_dir / f"final_refit_K{selected_k}"
    final_dir.mkdir(parents=True, exist_ok=True)
    final_component_spec = final_dir / "component_spec.npz"
    if not final_component_spec.exists():
        write_component_spec(
            final_component_spec,
            selected_components,
            provenance={
                "algorithm": "adaptive_coherit_frozen_training_partition",
                "case_id": args.case_id,
                "selected_depth": selected_depth,
                "selected_K": selected_k,
                "partition_source": selected["component_spec"],
            },
        )
    final_prefix = final_dir / "coherit"
    final_summary_path = Path(str(final_prefix) + ".summary.json")
    if not final_summary_path.is_file():
        partial_outputs = list(final_dir.glob("coherit.*"))
        if partial_outputs:
            raise RuntimeError(
                f"Incomplete final-refit outputs exist; inspect {final_dir}."
            )
        final_command = _sparse_command(
            args=args,
            component_spec=final_component_spec,
            phenotype=args.fit_pheno_txt,
            keep=args.fit_keep,
            prediction_keep=args.test_keep,
            prefix=final_prefix,
            marker_score_path=None,
            marker_score_seed=int(args.marker_score_seed),
            theta_init=np.asarray(
                selected["var_components_lasso_ml"], dtype=np.float64
            ),
        )
        print(
            f"[adaptive] final 10k refit with frozen K={selected_k} partition",
            flush=True,
        )
        _run_command(final_command, final_dir / "runner.log")

    final_summary, test_metrics, _ = _validate_sparse_layer(
        prefix=final_prefix,
        expected_k=selected_k,
        prediction_phenotype=args.test_pheno_txt,
        marker_score_path=None,
        n_variants=n_variants,
    )
    comparison = _existing_comparison(
        args.existing_prediction_results,
        case_id=args.case_id,
        adaptive_r2=float(test_metrics["correlation_squared"]),
    )
    comparison_path = args.out_dir / "prediction_comparison.tsv"
    _atomic_tsv(
        comparison_path,
        comparison,
        (
            "case_id",
            "method",
            "heldout_prediction_r2",
            "adaptive_minus_method_r2",
        ),
    )

    result = {
        "schema_version": 1,
        "status": "complete",
        "case_id": args.case_id,
        "completed_at": _now(),
        "algorithm": config["algorithm"],
        "valid_layers": layer_records,
        "layer_failures": layer_failures,
        "search_stop": search_stop,
        "selection": {
            "metric": "validation_squared_pearson_correlation",
            "selected_depth": selected_depth,
            "selected_K": selected_k,
            "selected_validation_r2": float(
                selected["validation"]["correlation_squared"]
            ),
            "tie_rule": "smaller_K_within_1e-12",
            "search_stop_reason": (
                search_stop.get("reason") if search_stop is not None else None
            ),
        },
        "final_refit": {
            "training_samples": int(final_summary["n_samples"]),
            "partition_frozen_before_10k_refit": True,
            "component_spec": str(final_component_spec),
            "h2_chive": float(final_summary["h2_chive_guarded"]),
            "support_size": int(final_summary["support_size"]),
            "test_metrics": test_metrics,
            "test_phenotype_used_only_after_model_selection": True,
            "summary": str(final_summary_path),
            "prediction": str(final_prefix) + ".sparse_prediction.tsv",
        },
        "comparison": comparison,
        "comparison_path": str(comparison_path),
        "run_config": str(config_path),
    }
    _atomic_json(result_path, result)
    print(
        "[adaptive] complete selected_K=%s validation_R2=%.8f test_R2=%.8f"
        % (
            selected_k,
            float(selected["validation"]["correlation_squared"]),
            float(test_metrics["correlation_squared"]),
        ),
        flush=True,
    )
    print(f"[adaptive] result -> {result_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
