"""Validation-set selection utilities for the COHERIT Lasso path."""

from __future__ import annotations

import csv
import json
import math
import os
from collections.abc import Mapping, Sequence

import numpy as np


def read_phenotype_aligned(
    path: str,
    sample_ids: Sequence[str],
) -> np.ndarray:
    """Read ``FID IID phenotype`` and return values in ``sample_ids`` order."""
    phenotype: dict[str, float] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) < 3:
                raise ValueError(
                    f"Malformed phenotype row {line_number}: {path}"
                )
            iid = str(fields[1])
            try:
                value = float(fields[2])
            except ValueError as error:
                raise ValueError(
                    f"Non-numeric phenotype at row {line_number}: {path}"
                ) from error
            if iid in phenotype:
                raise ValueError(
                    f"Duplicated phenotype IID {iid!r} at row {line_number}."
                )
            if not math.isfinite(value) or value == -9.0:
                raise ValueError(
                    f"Missing/non-finite phenotype at row {line_number}: {path}"
                )
            phenotype[iid] = value

    ids = [str(value) for value in sample_ids]
    if len(ids) != len(set(ids)):
        raise ValueError("Prediction sample IDs contain duplicates.")
    expected = set(ids)
    observed = set(phenotype)
    if expected != observed:
        missing = sorted(expected - observed)[:10]
        extra = sorted(observed - expected)[:10]
        raise ValueError(
            "Prediction and phenotype IID sets differ: "
            f"missing={missing}, extra={extra}."
        )
    if not ids:
        raise ValueError("Validation sample set is empty.")
    return np.asarray([phenotype[iid] for iid in ids], dtype=np.float64)


def evaluate_prediction_path(
    prediction: np.ndarray,
    outcome: np.ndarray,
) -> list[dict[str, float | int | None]]:
    """Compute prediction metrics for columns of an aligned path matrix."""
    pred = np.asarray(prediction, dtype=np.float64)
    y = np.asarray(outcome, dtype=np.float64).reshape(-1)
    if pred.ndim != 2 or pred.shape[0] != y.size or pred.shape[1] < 1:
        raise ValueError(
            "prediction must be a non-empty (n_samples, n_path) matrix."
        )
    if y.size < 2 or not np.all(np.isfinite(pred)) or not np.all(np.isfinite(y)):
        raise ValueError("Prediction path and outcome must be finite with n >= 2.")

    y_centered = y - float(np.mean(y))
    outcome_ss = float(y_centered @ y_centered)
    if outcome_ss <= 0.0:
        raise ValueError("Validation phenotype has zero centered variance.")

    pred_centered = pred - np.mean(pred, axis=0, keepdims=True)
    pred_ss = np.sum(pred_centered * pred_centered, axis=0)
    covariance = pred_centered.T @ y_centered
    squared_error = np.sum((pred - y[:, None]) ** 2, axis=0)

    rows: list[dict[str, float | int | None]] = []
    for index in range(pred.shape[1]):
        if float(pred_ss[index]) <= 0.0:
            correlation = None
            correlation_squared = None
            calibration_slope = None
        else:
            correlation_value = float(
                covariance[index]
                / math.sqrt(float(pred_ss[index]) * outcome_ss)
            )
            correlation_value = min(1.0, max(-1.0, correlation_value))
            correlation = correlation_value
            correlation_squared = correlation_value * correlation_value
            calibration_slope = float(covariance[index] / pred_ss[index])
        rows.append(
            {
                "path_index": int(index),
                "correlation": correlation,
                "correlation_squared": correlation_squared,
                "mse": float(squared_error[index] / y.size),
                "calibration_slope": calibration_slope,
                "predictive_r2": float(1.0 - squared_error[index] / outcome_ss),
            }
        )
    return rows


def select_validation_path_index(
    path_rows: Sequence[Mapping[str, object]],
    metrics: Sequence[Mapping[str, object]],
) -> int:
    """Maximize squared correlation; prefer sparser and larger-lambda ties."""
    if len(path_rows) != len(metrics) or not path_rows:
        raise ValueError("Lasso path and validation metrics must align.")

    eligible: list[int] = []
    for index, metric in enumerate(metrics):
        value = metric.get("correlation_squared")
        if value is not None and math.isfinite(float(value)):
            eligible.append(index)
    if not eligible:
        raise ValueError("No validation path point has a finite correlation.")

    return max(
        eligible,
        key=lambda index: (
            float(metrics[index]["correlation_squared"]),
            -int(path_rows[index]["k"]),
            float(path_rows[index]["lam_ratio"]),
        ),
    )


def merge_path_diagnostics(
    path_rows: Sequence[Mapping[str, object]],
    metrics: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Combine solver and validation diagnostics without coefficient arrays."""
    if len(path_rows) != len(metrics):
        raise ValueError("Lasso path and validation metrics must align.")
    return [dict(path_row) | dict(metric) for path_row, metric in zip(path_rows, metrics)]


def write_selection_outputs(path: str, payload: Mapping[str, object]) -> dict[str, str]:
    """Write the complete JSON audit trail and a compact path TSV."""
    if not str(path).lower().endswith(".json"):
        raise ValueError("Validation selection output must end in .json.")
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)

    path_rows = payload.get("path")
    if not isinstance(path_rows, list) or not path_rows:
        raise ValueError("Validation selection payload requires a non-empty path.")
    tsv_path = path[:-5] + ".path.tsv"
    fieldnames = [
        "path_index",
        "lam",
        "lam_ratio",
        "k",
        "correlation",
        "correlation_squared",
        "mse",
        "calibration_slope",
        "predictive_r2",
        "ebic",
        "rss",
        "cd_iter",
        "converged",
        "kkt_passed",
    ]
    with open(tsv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in path_rows:
            writer.writerow({name: row.get(name) for name in fieldnames})
    return {"json": path, "path_tsv": tsv_path}


__all__ = [
    "evaluate_prediction_path",
    "merge_path_diagnostics",
    "read_phenotype_aligned",
    "select_validation_path_index",
    "write_selection_outputs",
]
