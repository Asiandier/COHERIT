import argparse
import importlib
import os
from pathlib import Path
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT = os.path.dirname(REPO_ROOT)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)
PKG = importlib.import_module(os.path.basename(REPO_ROOT))
RUN_SPARSE = importlib.import_module(f"{PKG.__name__}.run_sparse_pipeline")
_prepare_combined_fit_inputs = RUN_SPARSE._prepare_combined_fit_inputs


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
