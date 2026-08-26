from __future__ import annotations

import csv
import importlib
import os
import sys

import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT = os.path.dirname(REPO_ROOT)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

DRIVER = importlib.import_module(
    f"{os.path.basename(REPO_ROOT)}.run_covtree_sparse_reml_pipeline"
)


def _write_prediction(path, outcome, prediction):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["iid", "lasso_phenotype_prediction_raw"])
        for index, value in enumerate(prediction):
            writer.writerow([f"i{index}", value])


def _write_phenotype(path, outcome):
    with path.open("w", encoding="utf-8") as handle:
        for index, value in enumerate(outcome):
            handle.write(f"f i{index} {value}\n")


def test_base_pipeline_arguments_replace_only_covtree_layer_outputs():
    original = [
        "--bed-prefix",
        "geno",
        "--component-spec",
        "old.npz",
        "--out-prefix",
        "old/out",
        "--sparsity-validation-out",
        "old/path.json",
        "--lasso-n-lambda",
        "80",
    ]
    assert DRIVER._base_pipeline_arguments(original) == [
        "--bed-prefix",
        "geno",
        "--lasso-n-lambda",
        "80",
    ]


def test_paired_guardrail_rejects_clear_degradation(tmp_path):
    rng = np.random.default_rng(4)
    outcome = rng.normal(size=400)
    previous = outcome + rng.normal(scale=0.1, size=400)
    current = outcome + rng.normal(scale=1.0, size=400)
    phenotype_path = tmp_path / "validation.pheno"
    previous_path = tmp_path / "previous.tsv"
    current_path = tmp_path / "current.tsv"
    _write_phenotype(phenotype_path, outcome)
    _write_prediction(previous_path, outcome, previous)
    _write_prediction(current_path, outcome, current)

    result = DRIVER.paired_prediction_guardrail(
        previous_prediction_path=previous_path,
        current_prediction_path=current_path,
        phenotype_path=phenotype_path,
        bootstrap_draws=1000,
        seed=9,
        alpha=0.05,
    )

    assert result["observed_mean_difference"] > 0.0
    assert result["confidence_interval"][0] > 0.0
    assert result["material_degradation_supported"] is True


def test_paired_guardrail_keeps_sampling_noise(tmp_path):
    rng = np.random.default_rng(8)
    outcome = rng.normal(size=300)
    prediction = outcome + rng.normal(scale=0.5, size=300)
    phenotype_path = tmp_path / "validation.pheno"
    previous_path = tmp_path / "previous.tsv"
    current_path = tmp_path / "current.tsv"
    _write_phenotype(phenotype_path, outcome)
    _write_prediction(previous_path, outcome, prediction)
    _write_prediction(current_path, outcome, prediction.copy())

    result = DRIVER.paired_prediction_guardrail(
        previous_prediction_path=previous_path,
        current_prediction_path=current_path,
        phenotype_path=phenotype_path,
        bootstrap_draws=500,
        seed=10,
        alpha=0.05,
    )

    assert result["observed_mean_difference"] == 0.0
    assert result["material_degradation_supported"] is False
