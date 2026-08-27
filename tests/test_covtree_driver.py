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
        writer.writerow(["iid", "lasso_phenotype_prediction"])
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


def test_final_refit_arguments_remove_selection_sample_contract():
    original = [
        "--bed-prefix",
        "geno",
        "--prediction-bed-prefix",
        "geno",
        "--pheno-txt",
        "train.pheno",
        "--keep-path",
        "train.keep",
        "--prediction-keep-path",
        "validation.keep",
        "--sparsity-validation-pheno-txt",
        "validation.pheno",
        "--covar-txt",
        "covar.txt",
        "--prediction-covar-txt",
        "covar.txt",
        "--component-spec",
        "selection.npz",
        "--out-prefix",
        "selection/coherit",
        "--lasso-n-lambda",
        "80",
    ]

    result = DRIVER._final_refit_pipeline_arguments(original)

    assert result == [
        "--bed-prefix",
        "geno",
        "--covar-txt",
        "covar.txt",
        "--lasso-n-lambda",
        "80",
    ]


def test_final_refit_command_freezes_lambda_and_warm_starts_theta(tmp_path):
    command = DRIVER._final_refit_command(
        python_bin=tmp_path / "python",
        sparse_pipeline=tmp_path / "pipeline.py",
        original_arguments=[
            "--bed-prefix",
            "geno",
            "--pheno-txt",
            "train.pheno",
            "--keep-path",
            "train.keep",
            "--sparsity-validation-pheno-txt",
            "validation.pheno",
        ],
        component_spec=tmp_path / "selected.npz",
        fit_phenotype=tmp_path / "fit.pheno",
        fit_keep=tmp_path / "fit.keep",
        prefix=tmp_path / "final" / "coherit",
        fixed_lam_ratio=0.75,
        theta_init=[0.4, 0.6],
        verbose=True,
    )

    assert DRIVER._flag_value(command, "--pheno-txt") == str(
        tmp_path / "fit.pheno"
    )
    assert DRIVER._flag_value(command, "--keep-path") == str(
        tmp_path / "fit.keep"
    )
    assert float(DRIVER._flag_value(command, "--lasso-fixed-lam-ratio")) == 0.75
    assert json.loads(DRIVER._flag_value(command, "--variance-components-init")) == [
        0.4,
        0.6,
    ]
    assert "--sparsity-validation-pheno-txt" not in command
    assert "--prediction-keep-path" not in command
    assert command[-1] == "--verbose"


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
        phenotype_standardization={
            "mean": 0.0,
            "standard_deviation": 1.0,
        },
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


def test_run_config_supplies_automatic_final_refit_inputs(tmp_path):
    config = tmp_path / "run_config.json"
    config.write_text(
        json.dumps(
            {
                "inputs": {
                    "fit_pheno_txt": "/fit.pheno",
                    "fit_keep": "/fit.keep",
                }
            }
        ),
        encoding="utf-8",
    )

    assert DRIVER._final_refit_inputs_from_run_config(config) == (
        "/fit.pheno",
        "/fit.keep",
    )


def test_natural_stop_automatically_runs_warm_final_refit(tmp_path, monkeypatch):
    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    component_spec = tmp_path / "selected_partition.npz"
    component_spec.write_bytes(b"partition")
    validation_pheno = tmp_path / "validation.pheno"
    validation_prediction = initial_dir / "coherit.sparse_prediction.tsv"
    outcome = np.asarray([-1.0, 0.5, 1.5])
    _write_phenotype(validation_pheno, outcome)
    _write_prediction(validation_prediction, outcome, outcome + 0.1)

    initial_summary = {
        "n_grms": 1,
        "n_samples": 8,
        "h2_chive_guarded": 0.4,
        "input_phenotype_standardization": {
            "mean": 0.0,
            "standard_deviation": 1.0,
        },
        "var_components_lasso_ml": [0.4, 0.6],
        "lasso_selected_lam_ratio": 0.75,
        "component_spec": str(component_spec),
        "support_size": 2,
        "elapsed_sec": 1.0,
    }
    (initial_dir / "coherit.summary.json").write_text(
        json.dumps(initial_summary), encoding="utf-8"
    )
    diagnostic = tmp_path / "diagnostic.json"
    diagnostic.write_text(
        json.dumps(
            {
                "current_k": 1,
                "next_k": 1,
                "accepted": False,
                "stopping_reason": "max_score_not_significant",
            }
        ),
        encoding="utf-8",
    )
    fit_pheno = tmp_path / "fit.pheno"
    fit_keep = tmp_path / "fit.keep"
    fit_pheno.write_text("f i0 0\n", encoding="utf-8")
    fit_keep.write_text("f i0\n", encoding="utf-8")
    ld_score = tmp_path / "ld.tsv"
    ld_score.write_text("snp\tld\n", encoding="utf-8")
    run_config = tmp_path / "run_config.json"
    run_config.write_text(
        json.dumps(
            {
                "inputs": {
                    "bed_prefix": "/geno",
                    "component_spec_snapshot": "/selection.npz",
                    "train_pheno_txt": "/train.pheno",
                    "train_keep": "/train.keep",
                    "validation_pheno_txt": str(validation_pheno),
                    "validation_keep": "/validation.keep",
                    "fit_pheno_txt": str(fit_pheno),
                    "fit_keep": str(fit_keep),
                },
                "runtime": {},
                "algorithm": {},
            }
        ),
        encoding="utf-8",
    )
    captured: list[str] = []

    def fake_run(command, log_path):
        captured.extend(command)
        assert json.loads(
            DRIVER._flag_value(command, "--variance-components-init")
        ) == [0.4, 0.6]
        prefix = DRIVER._flag_value(command, "--out-prefix")
        final_summary = {
            "sparse_output_schema_version": 7,
            "n_grms": 1,
            "n_samples": 10,
            "lasso_branch_valid": True,
            "alpha_theta_pair_usable": True,
            "lasso_ml_outer_converged": True,
            "all_requested_estimators_valid": True,
            "lambda_selection_method": "fixed_lam_ratio",
            "lasso_path_role": "frozen_ratio_target_only",
            "lasso_path_points_solved": 1,
            "validation_selection_inside_outer_loop": False,
            "lasso_selected_lam_ratio": 0.75,
            "variance_components_initial": [0.4, 0.6],
            "variance_components_init_source": "command_line_json",
            "h2_chive_guarded": 0.42,
            "support_size": 3,
        }
        with open(prefix + ".summary.json", "w", encoding="utf-8") as handle:
            json.dump(final_summary, handle)

    monkeypatch.setattr(DRIVER, "_run", fake_run)
    out_dir = tmp_path / "result"
    assert (
        DRIVER.main(
            [
                "--case-id",
                "case",
                "--initial-fit-dir",
                str(initial_dir),
                "--initial-diagnostic",
                str(diagnostic),
                "--pipeline-run-config",
                str(run_config),
                "--ld-score",
                str(ld_score),
                "--out-dir",
                str(out_dir),
                "--true-h2",
                "0.7",
            ]
        )
        == 0
    )

    result = json.loads((out_dir / "covtree_result.json").read_text())
    assert result["schema_version"] == 3
    assert result["selected_h2"] == 0.4
    assert result["final_h2"] == 0.42
    np.testing.assert_allclose(result["final_absolute_h2_error"], 0.28)
    assert result["final_refit"]["theta_warm_started_from_selection"] is True
    assert result["final_refit"]["training_data"] == "training_plus_validation"
    assert DRIVER._flag_value(captured, "--pheno-txt") == str(fit_pheno)
    assert DRIVER._flag_value(captured, "--keep-path") == str(fit_keep)
    assert "--sparsity-validation-pheno-txt" not in captured
