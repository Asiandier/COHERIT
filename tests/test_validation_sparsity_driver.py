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
DRIVER = importlib.import_module(f"{PKG}.run_validation_sparsity_pipeline")


def _required_args(tmp_path):
    values = {
        "case-id": "case",
        "bed-prefix": str(tmp_path / "geno"),
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


def test_driver_defaults_to_expanded_untruncated_path(tmp_path):
    args = DRIVER.parse_args(_required_args(tmp_path))

    assert args.lam_min_ratio == pytest.approx(1e-3)
    assert args.n_lambda == 80
    assert args.lasso_cd_max_iter == 10000
    assert args.validation_early_stopping_lag == 5


def test_driver_accepts_only_certified_early_stopped_validation_prefix(tmp_path):
    prefix = tmp_path / "selection"
    summary_path = tmp_path / "selection.summary.json"
    summary = {
        "sparse_output_schema_version": 7,
        "input_phenotype_standardization": {
            "mean": 0.0,
            "standard_deviation": 1.0,
        },
        "n_grms": 1,
        "sparse_grm_mode": "single_whole_genome_grm",
        "lasso_branch_valid": True,
        "sparse_prediction": {"status": "emitted"},
        "lambda_selection_method": "validation_r2",
        "lasso_path_role": (
            "early_stopped_kkt_certified_validation_prefix_weighted_basil"
        ),
        "lasso_path_complete": False,
        "lasso_path_points_solved": 33,
        "lasso_path_points_requested": 80,
        "lasso_validation_early_stopping": {
            "stopped": True,
            "n_evaluated": 33,
            "stopping_lag": 5,
        },
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    validated = DRIVER._validate_sparse_summary(
        prefix, expected_method="validation_r2"
    )
    assert validated["lasso_path_points_solved"] == 33

    summary["lasso_validation_early_stopping"]["stopped"] = False
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="stopping certificate"):
        DRIVER._validate_sparse_summary(
            prefix, expected_method="validation_r2"
        )


def test_sparse_commands_isolate_validation_and_final_test_stages(tmp_path):
    args = DRIVER.parse_args(_required_args(tmp_path))
    args.python_bin = tmp_path / "python"
    args.sparse_pipeline = tmp_path / "pipeline.py"
    args.bed_prefix = tmp_path / "geno"
    args.covar_txt = tmp_path / "covar"
    warm_state = tmp_path / "lasso_warm_state.npz"

    selection = DRIVER._sparse_command(
        args=args,
        phenotype=tmp_path / "train.pheno",
        keep=tmp_path / "train.keep",
        prediction_keep=tmp_path / "validation.keep",
        prefix=tmp_path / "selection",
        selection_pheno=tmp_path / "validation.pheno",
        selection_output=tmp_path / "validation.json",
        fixed_lam_ratio=None,
        theta_init=None,
        warm_state_in=None,
        warm_state_out=warm_state,
    )
    final = DRIVER._sparse_command(
        args=args,
        phenotype=tmp_path / "fit.pheno",
        keep=tmp_path / "fit.keep",
        prediction_keep=tmp_path / "test.keep",
        prefix=tmp_path / "final",
        selection_pheno=None,
        selection_output=None,
        fixed_lam_ratio=0.025,
        theta_init=np.asarray([0.2, 0.8]),
        warm_state_in=warm_state,
        warm_state_out=None,
    )

    assert "--sparsity-validation-pheno-txt" in selection
    assert "--lasso-selection-mode" not in selection
    assert selection[selection.index("--lasso-cd-max-iter") + 1] == "10000"
    assert selection[
        selection.index("--validation-early-stopping-lag") + 1
    ] == "5"
    assert str(tmp_path / "validation.pheno") in selection
    assert str(tmp_path / "test.keep") not in selection
    assert selection[selection.index("--lasso-warm-state-out") + 1] == str(
        warm_state
    )
    assert "--component-spec" not in selection
    assert "--lasso-selection-mode" not in final
    assert float(
        final[final.index("--lasso-fixed-lam-ratio") + 1]
    ) == pytest.approx(0.025)
    assert "--sparsity-validation-pheno-txt" not in final
    assert str(tmp_path / "validation.pheno") not in final
    assert final[final.index("--lasso-warm-state-in") + 1] == str(
        warm_state
    )

    with pytest.raises(ValueError, match="explicit lambda stage"):
        DRIVER._sparse_command(
            args=args,
            phenotype=tmp_path / "fit.pheno",
            keep=tmp_path / "fit.keep",
            prediction_keep=tmp_path / "test.keep",
            prefix=tmp_path / "invalid",
            selection_pheno=None,
            selection_output=None,
            fixed_lam_ratio=None,
            theta_init=None,
            warm_state_in=None,
            warm_state_out=None,
        )


def test_prediction_metrics_aligns_by_iid(tmp_path):
    phenotype = tmp_path / "test.pheno"
    phenotype.write_text("f2 i2 2\nf1 i1 1\nf3 i3 3\n", encoding="utf-8")
    prediction = tmp_path / "prediction.tsv"
    prediction.write_text(
        "sample_index\tiid\tlasso_phenotype_prediction\n"
        "0\ti1\t-1\n1\ti2\t0\n2\ti3\t1\n",
        encoding="utf-8",
    )

    metrics = DRIVER.prediction_metrics(
        prediction,
        phenotype,
        phenotype_standardization={
            "mean": 2.0,
            "standard_deviation": 1.0,
        },
    )

    assert metrics["correlation_squared"] == pytest.approx(1.0)
    assert metrics["mse"] == pytest.approx(0.0)
