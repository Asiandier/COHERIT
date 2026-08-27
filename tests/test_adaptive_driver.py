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

    defaults = DRIVER.parse_args(required)
    assert defaults.max_depth == 5
    assert defaults.lam_min_ratio == pytest.approx(1e-3)
    assert defaults.n_lambda == 80
    assert defaults.lasso_cd_max_iter == 10000
    assert DRIVER.parse_args([*required, "--max-depth", "5"]).max_depth == 5
    with pytest.raises(SystemExit):
        DRIVER.parse_args([*required, "--max-depth", "6"])


def test_adaptive_driver_contract_uses_two_means_without_signal_fraction(tmp_path):
    args = DRIVER.parse_args(_required_driver_args(tmp_path))

    algorithm = DRIVER._algorithm_config(args)

    assert algorithm["sparse_path_mode"] == "adaptive_k_validation_lambda"
    assert algorithm["lambda_selection"] == (
        "validation_r2_inside_every_alpha_theta_outer_iteration"
    )
    assert algorithm["lambda_path"] == {
        "lam_min_ratio": 1e-3,
        "n_lambda": 80,
        "lasso_cd_max_iter": 10000,
        "complete_path": True,
        "early_stopping": False,
    }
    assert algorithm["signal_split"] == "exact_1d_two_means"
    assert "signal_high_fraction" not in algorithm
    assert algorithm["k_path"] == [1, 4, 16, 64, 256, 1024]


def test_adaptive_sparse_commands_isolate_validation_and_final_test_stages(tmp_path):
    args = DRIVER.parse_args(_required_driver_args(tmp_path))
    args.python_bin = tmp_path / "python"
    args.sparse_pipeline = tmp_path / "pipeline.py"
    args.bed_prefix = tmp_path / "geno"
    args.covar_txt = tmp_path / "covar"
    component = tmp_path / "component.npz"

    selection = DRIVER._sparse_command(
        args=args,
        component_spec=component,
        phenotype=tmp_path / "train.pheno",
        keep=tmp_path / "train.keep",
        prediction_keep=tmp_path / "validation.keep",
        prefix=tmp_path / "selection",
        selection_pheno=tmp_path / "validation.pheno",
        selection_output=tmp_path / "validation_path.json",
        fixed_lam_ratio=None,
        marker_score_path=tmp_path / "marker_score.npz",
        marker_score_seed=17,
        marker_score_min_validation_r2=0.2,
        theta_init=None,
    )
    final = DRIVER._sparse_command(
        args=args,
        component_spec=component,
        phenotype=tmp_path / "fit.pheno",
        keep=tmp_path / "fit.keep",
        prediction_keep=tmp_path / "test.keep",
        prefix=tmp_path / "final",
        selection_pheno=None,
        selection_output=None,
        fixed_lam_ratio=0.125,
        marker_score_path=None,
        marker_score_seed=17,
        marker_score_min_validation_r2=None,
        theta_init=np.asarray([0.2, 0.8]),
    )

    assert "--sparsity-validation-pheno-txt" in selection
    assert "--lasso-fixed-lam-ratio" not in selection
    assert str(tmp_path / "validation.pheno") in selection
    assert str(tmp_path / "test.keep") not in selection
    assert selection[selection.index("--lasso-n-lambda") + 1] == "80"
    assert selection[selection.index("--lasso-lam-min-ratio") + 1] == "0.001"
    assert "--marker-score-out" in selection
    assert float(
        selection[
            selection.index("--marker-score-min-validation-r2") + 1
        ]
    ) == pytest.approx(0.2)

    assert "--sparsity-validation-pheno-txt" not in final
    assert float(final[final.index("--lasso-fixed-lam-ratio") + 1]) == pytest.approx(
        0.125
    )
    assert str(tmp_path / "validation.pheno") not in final
    assert "--marker-score-out" not in final


def test_adaptive_layer_requires_iterative_validation_contract(tmp_path):
    prefix = tmp_path / "coherit"
    summary = {
        "sparse_output_schema_version": 7,
        "input_phenotype_standardization": {
            "mean": 0.0,
            "standard_deviation": 1.0,
        },
        "n_grms": 1,
        "lasso_branch_valid": True,
        "sparse_prediction": {"status": "emitted"},
        "lambda_selection_method": "validation_r2",
        "lasso_path_role": "complete_validation_grid_weighted_basil",
        "lasso_path_complete": True,
        "lasso_path_points_solved": 80,
        "lasso_path_points_requested": 80,
        "validation_selection_inside_outer_loop": True,
        "lasso_selected_lam_ratio": 0.25,
        "var_components_lasso_ml": [0.2, 0.8],
    }
    (tmp_path / "coherit.summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    (tmp_path / "coherit.sparse_prediction.tsv").write_text(
        "sample_index\tiid\tlasso_phenotype_prediction\n"
        "0\ti1\t1\n1\ti2\t2\n2\ti3\t3\n",
        encoding="utf-8",
    )
    phenotype = tmp_path / "validation.pheno"
    phenotype.write_text("f1 i1 1\nf2 i2 2\nf3 i3 3\n", encoding="utf-8")
    selection_output = tmp_path / "validation_path.json"
    selection_output.write_text(
        json.dumps(
            {
                "selection_role": "inside_every_alpha_theta_outer_iteration",
                "test_phenotype_used": False,
                "n_path_selections": 3,
                "final_selected": {
                    "lam": 2.0,
                    "lam_ratio": 0.25,
                    "correlation_squared": 1.0,
                },
            }
        ),
        encoding="utf-8",
    )

    loaded, metrics, score, audit = DRIVER._validate_sparse_layer(
        prefix=prefix,
        expected_k=1,
        prediction_phenotype=phenotype,
        marker_score_path=None,
        n_variants=4,
        expected_selection_method="validation_r2",
        selection_output=selection_output,
    )

    assert loaded["lambda_selection_method"] == "validation_r2"
    assert metrics["correlation_squared"] == pytest.approx(1.0)
    assert score is None
    assert audit["final_selected"]["lam_ratio"] == pytest.approx(0.25)


def test_validation_decline_is_strict_with_numerical_tolerance():
    assert DRIVER.validation_declined(0.20, 0.19)
    assert not DRIVER.validation_declined(0.20, 0.20)
    assert not DRIVER.validation_declined(0.20, 0.20 - 5e-13)
    assert DRIVER.validation_declined(0.20, 0.20 - 2e-12)
