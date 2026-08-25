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


def test_sparse_commands_isolate_validation_and_final_test_stages(tmp_path):
    args = DRIVER.parse_args(_required_args(tmp_path))
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
        selection_output=tmp_path / "validation.json",
        fixed_lam_ratio=None,
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
        fixed_lam_ratio=0.025,
        theta_init=np.asarray([0.2, 0.8]),
    )

    assert "--sparsity-validation-pheno-txt" in selection
    assert selection[selection.index("--lasso-cd-max-iter") + 1] == "10000"
    assert str(tmp_path / "validation.pheno") in selection
    assert str(tmp_path / "test.keep") not in selection
    assert "--lasso-selection-mode" in final
    assert final[final.index("--lasso-selection-mode") + 1] == "fixed_ratio"
    assert "--sparsity-validation-pheno-txt" not in final
    assert str(tmp_path / "validation.pheno") not in final


def test_prediction_metrics_aligns_by_iid(tmp_path):
    phenotype = tmp_path / "test.pheno"
    phenotype.write_text("f2 i2 2\nf1 i1 1\nf3 i3 3\n", encoding="utf-8")
    prediction = tmp_path / "prediction.tsv"
    prediction.write_text(
        "sample_index\tiid\tlasso_phenotype_prediction_raw\n"
        "0\ti1\t1\n1\ti2\t2\n2\ti3\t3\n",
        encoding="utf-8",
    )

    metrics = DRIVER.prediction_metrics(prediction, phenotype)

    assert metrics["correlation_squared"] == pytest.approx(1.0)
    assert metrics["mse"] == pytest.approx(0.0)
