from __future__ import annotations

import importlib
import os
import sys

import numpy as np
import pytest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT = os.path.dirname(REPO_ROOT)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

PKG = os.path.basename(REPO_ROOT)
DRIVER = importlib.import_module(f"{PKG}.run_adaptive_sparse_reml_pipeline")
SPARSE = importlib.import_module(f"{PKG}.run_sparse_reml_pipeline")


def _required_driver_args(tmp_path):
    values = {
        "case-id": "case",
        "bed-prefix": str(tmp_path / "geno"),
        "ld-score": str(tmp_path / "ld.tsv"),
        "train-pheno-txt": str(tmp_path / "train.pheno"),
        "fit-pheno-txt": str(tmp_path / "fit.pheno"),
        "validation-pheno-txt": str(tmp_path / "validation.pheno"),
        "test-pheno-txt": str(tmp_path / "test.pheno"),
        "covar-txt": str(tmp_path / "covar.txt"),
        "train-keep": str(tmp_path / "train.keep"),
        "validation-keep": str(tmp_path / "validation.keep"),
        "fit-keep": str(tmp_path / "fit.keep"),
        "test-keep": str(tmp_path / "test.keep"),
        "out-dir": str(tmp_path / "out"),
    }
    return [item for key, value in values.items() for item in (f"--{key}", value)]


def test_prediction_metrics_aligns_by_iid(tmp_path):
    phenotype = tmp_path / "validation.pheno"
    phenotype.write_text("f2 i2 2\nf1 i1 1\nf3 i3 3\n", encoding="utf-8")
    prediction = tmp_path / "prediction.tsv"
    prediction.write_text(
        "sample_index\tiid\tlasso_phenotype_prediction_raw\n"
        "0\ti1\t1\n1\ti2\t2\n2\ti3\t3\n",
        encoding="utf-8",
    )

    metrics = DRIVER.prediction_metrics(prediction, phenotype)

    assert metrics["n"] == 3
    assert metrics["correlation_squared"] == pytest.approx(1.0)
    assert metrics["mse"] == pytest.approx(0.0)
    assert metrics["calibration_slope"] == pytest.approx(1.0)


def test_ld_score_loader_requires_exact_bim_order(tmp_path):
    bim = tmp_path / "geno.bim"
    bim.write_text("1 rs1 0 1 A C\n1 rs2 0 2 G T\n", encoding="utf-8")
    ld = tmp_path / "ld.tsv"
    ld.write_text("ID\tld_score\nrs1\t1.5\nrs2\t2.5\n", encoding="utf-8")

    assert np.array_equal(
        DRIVER.load_aligned_ld_score(ld, bim), np.asarray([1.5, 2.5])
    )

    ld.write_text("ID\tld_score\nrs2\t2.5\nrs1\t1.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="order mismatch"):
        DRIVER.load_aligned_ld_score(ld, bim)


def test_variance_component_json_parser_validates_length_and_sign():
    parsed = SPARSE._parse_variance_components_init("[0.1,0.2,0.7]", n_grm=2)
    assert np.array_equal(parsed, np.asarray([0.1, 0.2, 0.7]))
    with pytest.raises(ValueError, match="expected 3"):
        SPARSE._parse_variance_components_init("[0.2,0.8]", n_grm=2)
    with pytest.raises(ValueError, match="nonnegative"):
        SPARSE._parse_variance_components_init("[-0.1,0.2,0.9]", n_grm=2)


def test_sparse_parser_accepts_adaptive_score_and_warm_start_args(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gpu-reml-sparse",
            "--marker-score-out",
            "scores.npz",
            "--marker-score-probes",
            "12",
            "--marker-score-seed",
            "42",
            "--variance-components-init",
            "[0.1,0.2,0.7]",
        ],
    )

    args = SPARSE.parse_args()

    assert args.marker_score_out == "scores.npz"
    assert args.marker_score_probes == 12
    assert args.marker_score_seed == 42
    assert args.variance_components_init == "[0.1,0.2,0.7]"


def test_adaptive_driver_defaults_to_k1024_and_rejects_deeper_search(tmp_path):
    required = _required_driver_args(tmp_path)

    assert DRIVER.parse_args(required).max_depth == 5
    assert DRIVER.parse_args([*required, "--max-depth", "5"]).max_depth == 5
    with pytest.raises(SystemExit):
        DRIVER.parse_args([*required, "--max-depth", "6"])


def test_validation_decline_is_strict_with_numerical_tolerance():
    assert DRIVER.validation_declined(0.20, 0.19)
    assert not DRIVER.validation_declined(0.20, 0.20)
    assert not DRIVER.validation_declined(0.20, 0.20 - 5e-13)
    assert DRIVER.validation_declined(0.20, 0.20 - 2e-12)
