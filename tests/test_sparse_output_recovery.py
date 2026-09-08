"""Completion markers cover real files, not just successful status strings."""
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
PIPELINE = importlib.import_module(f"{ROOT.name}.run_sparse_pipeline")
IO = importlib.import_module(f"{ROOT.name}.io_utils")


def emit_stage(prefix, *, validation=False, state=None):
    prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = [Path(str(prefix) + suffix) for suffix in (
        ".history.json", ".selected_snps.tsv", ".sparse_prediction.tsv",
        ".sparse_prediction_metadata.json", ".sparse_effects.tsv",
    )]
    if state is not None:
        outputs.append(state)
    for path in outputs:
        path.write_text("complete artifact\n")
    summary = dict(
        sparse_output_schema_version=11, n_grms=1, lasso_branch_valid=True,
        lambda_selection_method="validation_predictive_r2" if validation else "fixed_lam_ratio",
        validation_selection_inside_outer_loop=validation,
        lasso_path_role="frozen_ratio_target_only",
        sparse_prediction={"status": "emitted"},
        var_components_lasso_ml=[0.3, 0.5], lasso_selected_lam_ratio=0.1,
        output_artifacts=IO.output_manifest(outputs),
    )
    IO.atomic_json(Path(str(prefix) + ".summary.json"), summary)
    return summary


@pytest.mark.parametrize("stage", ["validation", "final"])
@pytest.mark.parametrize("damage", [
    None, "missing_prediction", "truncated_prediction", "missing_effects",
    "missing_selected_snps", "invalid_summary", "missing_manifest", "missing_state",
])
def test_resume_reuses_complete_stage_and_rebuilds_incomplete_outputs(
    tmp_path, monkeypatch, stage, damage,
):
    validation = stage == "validation"
    prefix = tmp_path / "coherit"
    state = tmp_path / "sparse_state.npz"
    selected = emit_stage(prefix, validation=validation, state=state)
    summary_path = Path(str(prefix) + ".summary.json")
    if damage == "invalid_summary":
        summary_path.write_text('{"interrupted":')
    elif damage == "missing_manifest":
        broken = {key: value for key, value in selected.items() if key != "output_artifacts"}
        IO.atomic_json(summary_path, broken)
    elif damage == "truncated_prediction":
        Path(str(prefix) + ".sparse_prediction.tsv").write_text("partial\n")
    elif damage is not None:
        missing = {
            "missing_prediction": Path(str(prefix) + ".sparse_prediction.tsv"),
            "missing_effects": Path(str(prefix) + ".sparse_effects.tsv"),
            "missing_selected_snps": Path(str(prefix) + ".selected_snps.tsv"),
            "missing_state": state,
        }[damage]
        missing.unlink()
    args = SimpleNamespace(
        out_prefix=prefix, work_dir=tmp_path, prediction_prefix=tmp_path / "test",
        compute_effects=True, pheno_txt="train", keep_path="keep",
        fit_pheno_txt="combined", fit_keep_path="combined.keep",
    )
    calls = []
    monkeypatch.setattr(PIPELINE, "_low_level_command", lambda **kwargs: ["fixture-fit"])

    def run(command, log):
        assert not summary_path.exists()  # no stale completion marker during retry
        calls.append(command)
        emit_stage(prefix, validation=validation, state=state)

    monkeypatch.setattr(PIPELINE, "_run_command", run)
    if validation:
        _, actual_state, summary = PIPELINE._run_validation_fit(
            args, directory=tmp_path, component_spec=None,
        )
        assert actual_state == state
    else:
        summary = PIPELINE._run_final_refit(
            args, component_spec=None, selected_state=state, selected_summary=selected,
        )
    assert summary["sparse_prediction"]["status"] == "emitted"
    assert len(calls) == int(damage is not None)


def test_failed_retry_does_not_leave_a_complete_marker(tmp_path, monkeypatch):
    prefix = tmp_path / "coherit"
    selected = emit_stage(prefix)
    Path(str(prefix) + ".sparse_prediction.tsv").unlink()
    args = SimpleNamespace(
        out_prefix=prefix, work_dir=tmp_path, prediction_prefix="test",
        compute_effects=True, fit_pheno_txt="combined", fit_keep_path="combined.keep",
    )
    monkeypatch.setattr(PIPELINE, "_low_level_command", lambda **kwargs: [])

    def interrupted(*args):
        raise RuntimeError("interrupted fit")

    monkeypatch.setattr(PIPELINE, "_run_command", interrupted)
    with pytest.raises(RuntimeError, match="interrupted fit"):
        PIPELINE._run_final_refit(
            args, component_spec=None, selected_state=tmp_path / "state", selected_summary=selected,
        )
    assert not Path(str(prefix) + ".summary.json").exists()


def test_semantically_wrong_fit_is_not_silently_overwritten(tmp_path):
    prefix = tmp_path / "coherit"
    emit_stage(prefix)
    with pytest.raises(ValueError, match="K does not match"):
        PIPELINE._completed_summary(
            prefix, expected_k=2, expected_method="fixed_lam_ratio", require_prediction=True,
        )
    assert Path(str(prefix) + ".summary.json").is_file()


def test_atomic_json_failure_preserves_prior_file_and_cleans_temporary(tmp_path):
    path = tmp_path / "summary.json"
    IO.atomic_json(path, {"complete": True})
    previous = path.read_bytes()
    with pytest.raises(ValueError):
        IO.atomic_json(path, {"nonfinite": float("nan")})
    assert path.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [path]
