import argparse
import importlib
import os
from pathlib import Path
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT = os.path.dirname(REPO_ROOT)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)
PKG = importlib.import_module(os.path.basename(REPO_ROOT))
RUN_SPARSE = importlib.import_module(f"{PKG.__name__}.run_sparse_pipeline")
_prepare_combined_fit_inputs = RUN_SPARSE._prepare_combined_fit_inputs


def score_args():
    return ["--mode", "adaptive", "--bed-prefix", "geno", "--pheno-txt", "train.pheno",
            "--validation-pheno-txt", "validation.pheno", "--keep-path", "train.keep",
            "--validation-keep-path", "validation.keep", "--out-prefix", "result"]


def test_trace_score_cli_defaults():
    args = RUN_SPARSE.parse_args(score_args())
    assert args.score_core_rank == 64
    assert args.score_reference_samples == 16383
    assert args.score_trace_probes == 512
    assert args.score_trace_max_probes == 4096
    assert args.score_trace_tol == 0.05
    assert args.score_trace_seed >= 0
    assert not any("bootstrap" in name for name in vars(args))


@pytest.mark.parametrize("option,value", [
    ("--score-trace-probes", "1"), ("--score-trace-max-probes", "64"),
    ("--score-trace-tol", "0"), ("--score-trace-tol", "nan"),
    ("--score-trace-seed", "-1"), ("--bootstrap-draws", "199"),
    ("--score-core-rank", "0"), ("--score-reference-samples", "1"),
    ("--split-alpha", "0.00001"),
])
def test_invalid_trace_configuration_is_rejected(option, value):
    with pytest.raises(SystemExit) as error:
        RUN_SPARSE.parse_args(score_args()+[option, value])
    assert error.value.code == 2


def test_resume_digest_detects_source_changes(monkeypatch, tmp_path: Path) -> None:
    script = tmp_path / "run_sparse_pipeline.py"
    script.write_text("# runner\n")
    algorithm = tmp_path / "adaptive_ld.py"
    algorithm.write_text("# first version\n")
    monkeypatch.setattr(RUN_SPARSE, "__file__", str(script))
    initial = RUN_SPARSE._algorithm_source_digest()
    assert RUN_SPARSE._algorithm_source_digest() == initial
    algorithm.write_text("# changed algorithm\n")
    assert RUN_SPARSE._algorithm_source_digest() != initial


def test_combined_fit_inputs_follow_source_sample_order(tmp_path: Path) -> None:
    prefix = tmp_path / "geno"
    (tmp_path / "geno.fam").write_text(
        "f b 0 0 0 -9\n"
        "f a 0 0 0 -9\n"
        "f d 0 0 0 -9\n"
        "f c 0 0 0 -9\n",
        encoding="utf-8",
    )
    train_keep = tmp_path / "train.keep"
    train_keep.write_text("f a\nf c\n", encoding="utf-8")
    validation_keep = tmp_path / "validation.keep"
    validation_keep.write_text("f b\nf d\n", encoding="utf-8")
    train_pheno = tmp_path / "train.pheno"
    train_pheno.write_text("f a 1\nf c 3\n", encoding="utf-8")
    validation_pheno = tmp_path / "validation.pheno"
    validation_pheno.write_text("f b 2\nf d 4\n", encoding="utf-8")
    args = argparse.Namespace(
        fit_pheno_txt=None,
        fit_keep_path=None,
        keep_path=train_keep,
        validation_keep_path=validation_keep,
        genotype_prefix=prefix,
        genotype_format="bed",
        pheno_txt=train_pheno,
        validation_pheno_txt=validation_pheno,
        work_dir=tmp_path / "work",
    )
    phenotype, keep, constructed = _prepare_combined_fit_inputs(args)
    assert constructed is True
    assert keep.read_text(encoding="utf-8").splitlines() == [
        "b\tb",
        "a\ta",
        "d\td",
        "c\tc",
    ]
    assert phenotype.read_text(encoding="utf-8").splitlines() == [
        "b\tb\t2",
        "a\ta\t1",
        "d\td\t4",
        "c\tc\t3",
    ]
