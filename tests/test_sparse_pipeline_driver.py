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


@pytest.fixture
def cohort_inputs(tmp_path):
    prefix = tmp_path / "geno"
    prefix.with_suffix(".fam").write_text("".join(f"f {iid} 0 0 0 -9\n" for iid in "abcde"))
    contents = {
        "train.keep": "f a\nf c\n", "validation.keep": "f b\nf d\n",
        "fit.keep": "f d\nf c\nf b\nf a\n", "test.keep": "f e\n",
        "train.pheno": "f a 1\nf c 3\n", "validation.pheno": "f b 2\nf d 4\n",
        "fit.pheno": "f a 1\nf b 2\nf c 3\nf d 4\n",
    }
    for name, content in contents.items():
        (tmp_path/name).write_text(content)
    return argparse.Namespace(
        fit_pheno_txt=tmp_path/"fit.pheno", fit_keep_path=tmp_path/"fit.keep",
        keep_path=tmp_path/"train.keep", validation_keep_path=tmp_path/"validation.keep",
        genotype_prefix=prefix, genotype_format="bed", pheno_txt=tmp_path/"train.pheno",
        validation_pheno_txt=tmp_path/"validation.pheno", work_dir=tmp_path/"work",
        prediction_prefix=prefix, prediction_format="bed", prediction_keep_path=tmp_path/"test.keep",
    )


def test_explicit_final_inputs_are_validated_without_rewriting(cohort_inputs):
    args = cohort_inputs
    modified = args.fit_pheno_txt.stat().st_mtime_ns
    assert _prepare_combined_fit_inputs(args) == (args.fit_pheno_txt, args.fit_keep_path, False)
    assert args.fit_pheno_txt.stat().st_mtime_ns == modified
    assert not args.work_dir.exists()


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("bad", ["selection_overlap", "prediction_overlap", "prediction_missing",
                                 "own_phenotype_missing"])
def test_cohort_isolation_is_required_with_both_input_modes(cohort_inputs, explicit, bad):
    args = cohort_inputs
    if not explicit:
        args.fit_pheno_txt = args.fit_keep_path = None
    if bad == "selection_overlap":
        args.validation_keep_path.write_text("f a\nf d\n")
        match = "Training and validation.*disjoint"
    elif bad == "prediction_overlap":
        args.prediction_keep_path.write_text("f a\nf e\n")
        match = "Prediction and training.*disjoint"
    elif bad == "prediction_missing":
        args.prediction_keep_path.write_text("f unknown\n")
        match = "Prediction keep ID is absent"
    else:
        args.pheno_txt.write_text("f a 1\n")
        args.validation_pheno_txt.write_text("f b 2\nf c 3\nf d 4\n")
        match = "Training phenotype is missing"
    with pytest.raises(ValueError, match=match):
        _prepare_combined_fit_inputs(args)
    assert not args.work_dir.exists()


@pytest.mark.parametrize("bad", ["missing_id", "extra_id", "missing_value", "changed_value"])
def test_explicit_final_inputs_must_match_selection(cohort_inputs, bad):
    args = cohort_inputs
    if bad == "missing_id":
        args.fit_keep_path.write_text("f a\nf b\nf c\n")
    elif bad == "extra_id":
        args.fit_keep_path.write_text("f a\nf b\nf c\nf d\nf e\n")
    elif bad == "missing_value":
        args.fit_pheno_txt.write_text("f a 1\nf b 2\nf c 3\n")
    else:
        args.fit_pheno_txt.write_text("f a 1\nf b 2\nf c 3\nf d 40\n")
    with pytest.raises(ValueError, match="Final-fit"):
        _prepare_combined_fit_inputs(args)


def test_prediction_without_keep_still_checks_the_actual_sample_set(cohort_inputs):
    cohort_inputs.prediction_keep_path = None
    with pytest.raises(ValueError, match="Prediction and training.*disjoint"):
        _prepare_combined_fit_inputs(cohort_inputs)
