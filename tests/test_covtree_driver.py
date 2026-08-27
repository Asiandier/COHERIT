from __future__ import annotations

import csv
import importlib
import json
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


def test_prediction_metrics_remain_available_for_audit(tmp_path):
    rng = np.random.default_rng(4)
    outcome = rng.normal(size=400)
    prediction = outcome + rng.normal(scale=0.1, size=400)
    phenotype_path = tmp_path / "validation.pheno"
    prediction_path = tmp_path / "prediction.tsv"
    _write_phenotype(phenotype_path, outcome)
    _write_prediction(prediction_path, outcome, prediction)

    result = DRIVER.prediction_metrics(
        prediction_path,
        phenotype_path,
    )

    assert result["n"] == 400
    assert result["correlation_squared"] > 0.9
    assert result["mse"] > 0.0


def test_heritability_accuracy_is_evaluation_only():
    result = DRIVER.heritability_accuracy(0.84, 0.7)

    assert result["true_h2"] == 0.7
    np.testing.assert_allclose(result["signed_h2_bias"], 0.14)
    np.testing.assert_allclose(result["absolute_h2_error"], 0.14)


def test_driver_exposes_no_prediction_guardrail_options():
    args = DRIVER.parse_args(
        [
            "--case-id",
            "case",
            "--initial-fit-dir",
            "fit",
            "--initial-diagnostic",
            "diagnostic.json",
            "--ld-score",
            "ld.tsv",
            "--out-dir",
            "out",
            "--true-h2",
            "0.7",
        ]
    )

    assert not hasattr(args, "guardrail_alpha")
    assert not hasattr(args, "guardrail_bootstrap_draws")
    assert args.true_h2 == 0.7
    assert args.max_k == 1024
    assert args.max_splits is None


def test_run_config_reconstructs_complete_selection_contract(tmp_path):
    config = tmp_path / "run_config.json"
    config.write_text(
        json.dumps(
            {
                "inputs": {
                    "bed_prefix": "/geno",
                    "component_spec_snapshot": "/component.npz",
                    "train_pheno_txt": "/train.pheno",
                    "train_keep": "/train.keep",
                    "validation_pheno_txt": "/validation.pheno",
                    "validation_keep": "/validation.keep",
                    "covar_txt": "/covar.txt",
                },
                "runtime": {
                    "device": "gpu",
                    "gpu_budget_gib": 80.0,
                    "cpu_threads": 56,
                    "screen_topk": 2000,
                    "candidate_k": 256,
                    "kkt_add_topk": 256,
                    "kkt_max_rounds": 20,
                },
                "algorithm": {
                    "lam_min_ratio": 0.001,
                    "n_lambda": 80,
                    "lasso_cd_max_iter": 10000,
                },
            }
        ),
        encoding="utf-8",
    )

    arguments = DRIVER._selection_arguments_from_run_config(config)

    assert DRIVER._flag_value(arguments, "--bed-prefix") == "/geno"
    assert DRIVER._flag_value(arguments, "--prediction-bed-prefix") == "/geno"
    assert DRIVER._flag_value(arguments, "--pheno-txt") == "/train.pheno"
    assert DRIVER._flag_value(arguments, "--keep-path") == "/train.keep"
    assert (
        DRIVER._flag_value(arguments, "--sparsity-validation-pheno-txt")
        == "/validation.pheno"
    )
    assert DRIVER._flag_value(arguments, "--lasso-n-lambda") == "80"
