from __future__ import annotations

import importlib
import json
import os
import sys

import pytest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT = os.path.dirname(REPO_ROOT)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

PKG = os.path.basename(REPO_ROOT)
REPLAY = importlib.import_module(f"{PKG}.run_covtree_replay")


def test_run_config_reconstructs_selection_model_inputs(tmp_path):
    path = tmp_path / "run_config.json"
    path.write_text(
        json.dumps(
            {
                "inputs": {
                    "bed_prefix": "/geno",
                    "component_spec_snapshot": "/component.npz",
                    "train_pheno_txt": "/train.pheno",
                    "train_keep": "/train.keep",
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
            }
        ),
        encoding="utf-8",
    )

    arguments = REPLAY._pipeline_arguments_from_run_config(path)
    parsed = REPLAY._parse_pipeline_args(arguments)

    assert parsed.bed_prefix == "/geno"
    assert parsed.component_spec == "/component.npz"
    assert parsed.pheno_txt == "/train.pheno"
    assert parsed.keep_path == "/train.keep"
    assert parsed.covar_txt == "/covar.txt"
    assert parsed.device == "gpu"
    assert parsed.gpu_budget_gib == pytest.approx(80.0)
    assert parsed.cpu_threads == 56


def test_missing_trajectory_requires_run_config(tmp_path):
    with pytest.raises(FileNotFoundError, match="supply --run-config"):
        REPLAY._original_pipeline_arguments(
            fit_dir=tmp_path,
            pipeline_trajectory="",
            run_config="",
        )
