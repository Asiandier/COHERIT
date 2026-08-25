from __future__ import annotations

import importlib
import json
import os
import sys

import numpy as np
import pytest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT = os.path.dirname(REPO_ROOT)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)
PKG = os.path.basename(REPO_ROOT)
SELECTION = importlib.import_module(f"{PKG}.sparsity_selection")


def test_read_phenotype_aligns_exactly_by_iid(tmp_path):
    path = tmp_path / "validation.pheno"
    path.write_text("f2 i2 2\nf1 i1 1\nf3 i3 3\n", encoding="utf-8")

    result = SELECTION.read_phenotype_aligned(path, ["i1", "i2", "i3"])

    np.testing.assert_array_equal(result, np.asarray([1.0, 2.0, 3.0]))


def test_read_phenotype_rejects_mismatched_sample_set(tmp_path):
    path = tmp_path / "validation.pheno"
    path.write_text("f1 i1 1\nf2 i2 2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="IID sets differ"):
        SELECTION.read_phenotype_aligned(path, ["i1", "i3"])


def test_validation_selection_uses_r2_then_sparsity_then_lambda():
    outcome = np.asarray([0.0, 1.0, 2.0, 3.0])
    prediction = np.column_stack(
        [
            outcome,
            2.0 * outcome,
            np.asarray([0.0, 1.0, 3.0, 2.0]),
        ]
    )
    metrics = SELECTION.evaluate_prediction_path(prediction, outcome)
    path = [
        {"k": 4, "lam_ratio": 0.8},
        {"k": 2, "lam_ratio": 0.5},
        {"k": 1, "lam_ratio": 0.2},
    ]

    assert SELECTION.select_validation_path_index(path, metrics) == 1
    assert metrics[0]["correlation_squared"] == pytest.approx(1.0)
    assert metrics[1]["calibration_slope"] == pytest.approx(0.5)


def test_write_selection_outputs_emits_json_and_tsv(tmp_path):
    path = tmp_path / "selection.json"
    payload = {
        "selected_index": 0,
        "path": [
            {
                "path_index": 0,
                "lam": 2.0,
                "lam_ratio": 1.0,
                "k": 0,
                "correlation": 0.2,
                "correlation_squared": 0.04,
                "mse": 1.0,
                "calibration_slope": 0.3,
                "predictive_r2": -0.1,
                "ebic": 5.0,
                "rss": 2.0,
                "cd_iter": 1,
                "converged": True,
                "kkt_passed": True,
            }
        ],
    }

    outputs = SELECTION.write_selection_outputs(str(path), payload)

    assert json.loads(path.read_text())["selected_index"] == 0
    assert os.path.exists(outputs["path_tsv"])
    assert "correlation_squared" in open(
        outputs["path_tsv"], encoding="utf-8"
    ).readline()
