"""Tiny real-file CLI runs, kept entirely outside formal experiment outputs."""
from pathlib import Path
import json
import os
import subprocess
import sys

import numpy as np
import pytest
from bed_reader import to_bed

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "mode", ["fixed", "fixed_multi", "adaptive", "alignment", "missing_single", "missing_multi"]
)
def test_sparse_pipeline_automatic_fit_effects_prediction(tmp_path, mode):
    rng = np.random.default_rng(206)
    n, m = 480, 96
    x = rng.binomial(2, 0.3, size=(n, m)).astype(np.int8)
    ids = [f"sample{i}" for i in range(n)]
    variants = [f"rs{i}" for i in range(m)]
    prefix = tmp_path / "genotype"
    z = (x - x.mean(0)) / x.std(0)
    beta = rng.normal(size=m)
    beta[m // 2:] = 0
    genetic = z @ beta
    y = np.sqrt(0.65) * genetic / genetic.std() + np.sqrt(0.35) * rng.normal(size=n)
    if mode.startswith("missing"):
        # Different atoms by component AND sample set expose stale or unit atoms.
        x[:64, :m // 2] = -127
        x[:32, m // 2:] = -127
        x[320:328, :m // 2] = -127
        x[320:344, m // 2:] = -127
    to_bed(str(prefix) + ".bed", x, properties={"iid": ids, "fid": ids, "sid": variants})
    subsets = {"train": range(320), "validation": range(320, 400), "test": range(400, n)}
    for name, indices in subsets.items():
        (tmp_path / f"{name}.keep").write_text("".join(f"{ids[i]} {ids[i]}\n" for i in indices))
        if name != "test":
            (tmp_path / f"{name}.pheno").write_text(
                "".join(f"{ids[i]} {ids[i]} {y[i]:.12g}\n" for i in indices)
            )
    output = tmp_path / "result"
    # Keep a background component in the missing-data fits so the trace-weighted
    # heritability checks below remain sensitive to incorrect variance scaling.
    min_ratio = "0.5" if mode.startswith("missing") else "0.1"
    command = [
        sys.executable, "-m", f"{ROOT.name}.run_sparse_pipeline",
        "--mode", "adaptive" if mode == "adaptive" else "fixed",
        "--bed-prefix", str(prefix), "--pheno-txt", str(tmp_path / "train.pheno"),
        "--validation-pheno-txt", str(tmp_path / "validation.pheno"),
        "--keep-path", str(tmp_path / "train.keep"),
        "--validation-keep-path", str(tmp_path / "validation.keep"),
        "--prediction-bed-prefix", str(prefix),
        "--prediction-keep-path", str(tmp_path / "test.keep"),
        "--compute-effects", "--out-prefix", str(output),
        "--device", os.environ.get("COHERIT_TEST_DEVICE", "cpu"),
        "--gpu-budget-gib", "0.25", "--cpu-threads", "2", "--call-width", "32",
        "--n-rand-vec", "24", "--slq-samples", "8", "--slq-m", "12",
        "--minq-iter", "30", "--outer-max", "12", "--n-lambda", "12",
        "--lam-min-ratio", min_ratio, "--screen-topk", "96", "--candidate-k", "32",
        "--pcg-tol", "0.001", "--verbose",
    ]
    if mode in {"fixed_multi", "missing_multi"}:
        spec = tmp_path / "partition.npz"
        np.savez(spec, arr_0=np.arange(m // 2), arr_1=np.arange(m // 2, m))
        command += ["--component-spec", str(spec)]
    if mode == "adaptive":
        ld = tmp_path / "ld.tsv"
        ld.write_text("ID\tld_score\n" + "".join(f"{v}\t{i + 1}\n" for i, v in enumerate(variants)))
        command += ["--ld-score", str(ld), "--ld-rank-bins", "4", "--score-trace-probes", "64"]
    if mode == "alignment":
        worker = tmp_path / "alignment_worker.py"
        worker.write_text(
            "import sys\n"
            f"sys.path.insert(0, {str(ROOT.parent)!r})\n"
            f"from {ROOT.name} import run_sparse_reml_pipeline as pipeline\n"
            "original = pipeline._alignment_action\n"
            "rejected_once = False\n"
            "def reject_first(**kwargs):\n"
            "    global rejected_once\n"
            "    action = original(**kwargs)\n"
            "    if not rejected_once and action == 'converged':\n"
            "        rejected_once = True\n"
            "        kwargs['h2_stable'] = False\n"
            "        return original(**kwargs)\n"
            "    return action\n"
            "pipeline._alignment_action = reject_first\n"
            "pipeline.main()\n"
        )
        command += ["--low-level-pipeline", str(worker)]
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "XLA_PYTHON_CLIENT_ALLOCATOR": "platform"}
    # A complete adaptive run includes several separately compiled fits. Allow
    # for the SLQ reverse pass and shared-GPU contention; numerical assertions
    # below (and the isolated performance benchmark) are unchanged.
    result = subprocess.run(command, cwd=ROOT.parent, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=480)
    logs = "\n".join(p.read_text()[-3500:] for p in tmp_path.glob("**/runner.log"))
    assert result.returncode == 0, result.stdout[-5000:] + logs
    pipeline = json.loads(output.with_suffix(".pipeline.json").read_text())
    assert pipeline["status"] == "complete"
    final = json.loads(output.with_suffix(".summary.json").read_text())
    assert final["sparse_output_schema_version"] == 11
    assert final["primary_h2_method"] == "information_corrected_sparse_reml"
    assert final["n_samples"] == 400
    assert final["lasso_branch_valid"]
    assert final["sparse_prediction"]["status"] == "emitted"
    # Summary is published last and describes the complete output set.
    for name, size in final["output_artifacts"].items():
        assert Path(name).is_file()
        assert Path(name).stat().st_size == size > 0
    for suffix in (".history.json", ".selected_snps.tsv", ".sparse_effects.tsv", ".sparse_prediction.tsv"):
        assert str(output.with_suffix(suffix)) in final["output_artifacts"]
    assert output.with_suffix(".sparse_effects.tsv").is_file()
    assert len(output.with_suffix(".sparse_prediction.tsv").read_text().splitlines()) == 81
    if mode in {"fixed_multi", "missing_multi"}:
        assert final["n_grms"] == 2
    if mode == "adaptive":
        score_paths = sorted(tmp_path.glob("**/fixed_alpha_path/k*.score.json"))
        assert score_paths
        for score_path in score_paths:
            score = json.loads(score_path.read_text())
            assert score["schema_version"] == 3
            assert score["method"] == "joint_quadratic_reml_ld_cusum"
            diagnostics = score["diagnostics"]
            assert score["accepted"] == (diagnostics["global_p_value"] <= 0.05)
            if score["candidates"]:
                assert diagnostics["calibration_method"] == "joint_core_probe_quadratic"
                assert diagnostics["max_score_trace_standard_error"] <= 0.05
                assert diagnostics["max_information_relative_standard_error"] <= 0.10
                assert score["selected_candidate"] == score["candidates"][0]
                assert all("trace_" in row["stage"] or row["stage"] in {
                    "adaptive_ld_observed", "adaptive_ld_common_core", "covariate_projection"}
                    for row in score["pcg"])
                assert score["selected_candidate"]["score_statistic"] == max(
                    row["score_statistic"] for row in score["candidates"])
                assert diagnostics["global_p_value"] >= 1/(diagnostics["reference_samples"]+1)
        assert not list(tmp_path.glob("**/.score-*"))
        single_path = next(tmp_path.glob("**/k1_validation/coherit.summary.json"))
        single = json.loads(single_path.read_text())
        assert single["primary_h2_method"] == "information_corrected_sparse_reml"
        frozen_paths = list(tmp_path.glob("**/fixed_alpha_path/k*.frozen_fit.json"))
        assert frozen_paths, "The adaptive fixture must exercise an ordinary covariance refit."
        for frozen_path in frozen_paths:
            frozen = json.loads(frozen_path.read_text())
            assert frozen["schema_version"] == 1
            assert frozen["method"] == "fixed_sparse_mean_covariance_reml"
            assert frozen["q_chive"] == pytest.approx(single["q_chive"])
            assert frozen["parent_q_sparse_held_fixed"] == pytest.approx(single["q_chive"])
            assert "mean_information_rank" not in frozen
        endpoint_path = next(tmp_path.glob("**/endpoint_validation_refit/coherit.summary.json"))
        endpoint = json.loads(endpoint_path.read_text())
        assert endpoint["primary_h2_method"] == "information_corrected_sparse_reml"

    for path in tmp_path.glob("**/*.history.json"):
        records = json.loads(path.read_text())
        summary = json.loads(path.with_name(path.name.replace(".history.json", ".summary.json")).read_text())
        assert summary["grm_variance_scale"] == "trace_weighted"
        assert summary["primary_h2_method"] == "information_corrected_sparse_reml"
        atoms = np.asarray(summary["genetic_trace_atoms"])
        # All markers in this fixture are polymorphic. Missing positions become
        # zero after centering; each standardized column's norm² is its count.
        n_fit = summary["n_samples"]
        counts = np.sum(x[:n_fit] >= 0, axis=0)
        groups = np.split(counts, np.cumsum(summary["m_per_grm"])[:-1])
        np.testing.assert_allclose(atoms, [g.mean() / n_fit for g in groups], atol=1e-7)
        theta_final = np.asarray(summary["var_components_lasso_ml"])
        bg = float(theta_final[:-1] @ atoms)
        if mode.startswith("missing"):
            # A nonzero background makes the final h2 assertion detect raw sums.
            assert bg > 0.0
        q_final = summary["q_chive"]
        parts = summary["q_chive_components"]
        assert q_final == pytest.approx(parts["term1_g2_over_n"] + parts["term2_cross"]
                                       - parts["term3_mean_uncertainty_subtracted"])
        assert summary["h2"] == pytest.approx((q_final + bg) / (q_final + bg + theta_final[-1]))
        assert summary["h2_background_lasso_ml"] == pytest.approx(bg / (bg + theta_final[-1]))
        for row in records:
            if row.get("coherit_h2") is None:
                continue
            theta, q = row["theta"], row["q_sparse"]
            bg = float(np.asarray(theta[:-1]) @ atoms)
            expected = (q + bg) / (q + bg + theta[-1])
            assert row["coherit_h2"] == pytest.approx(expected, abs=1e-10)
            if "theta_after_variance_update" in row:
                after = np.asarray(row["theta_after_variance_update"])
                bg_after = float(after[:-1] @ atoms)
                assert row["coherit_h2_after_variance_update"] == pytest.approx(
                    (row["q_sparse_after_variance_update"] + bg_after) /
                    (row["q_sparse_after_variance_update"] + bg_after + after[-1]), abs=1e-10
                )
        if summary["lasso_ml_outer_converged"]:
            assert records[-1]["stage"] == "final_covariance_lasso"
            assert records[-1]["alignment_action"] == "converged"
            assert records[-1]["h2_stable"] and records[-1]["effect_stable"]
            assert summary["h2"] == pytest.approx(records[-1]["coherit_h2"], abs=1e-7)
        if mode == "alignment":
            assert any(r.get("alignment_action") == "continue" for r in records)
            assert summary["lasso_ml_outer_converged"]
        audit = path.with_name(path.name.replace(".history.json", ".validation.json"))
        if audit.exists():
            traces = json.loads(audit.read_text())["outer_selection_trace"]
            assert len(traces) == summary["outer_iterations"] + 1
            outer_rows = {r["outer"]: r for r in records if r.get("stage") == "outer_update"}
            for trace in traces:
                if trace["stage"] == "outer_update":
                    row = outer_rows[trace["outer"]]
                    assert trace["theta"] == row["theta"]
                    assert trace["coherit_h2"] == row["coherit_h2"]
                    assert trace["selected"]["predictive_r2"] == row["validation_predictive_r2"]

    # Same inputs/code resume without touching already-completed fit artifacts.
    modified = output.with_suffix(".summary.json").stat().st_mtime_ns
    resumed = subprocess.run(command, cwd=ROOT.parent, env=env, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
    assert resumed.returncode == 0, resumed.stdout[-5000:]
    assert output.with_suffix(".summary.json").stat().st_mtime_ns == modified

    if mode == "fixed":
        # Keep the algorithm unchanged and turn off just the two optimizations
        # in a test-only worker. No production switch or alternative estimator.
        worker = tmp_path / "uncached_worker.py"
        worker.write_text(
            "import sys\n"
            f"sys.path.insert(0, {str(ROOT.parent)!r})\n"
            f"from {ROOT.name} import run_sparse_reml_pipeline as pipeline\n"
            "original_fitter = pipeline.InfinitesimalREMLFitter\n"
            "def fitter(config):\n"
            "    config.cache_reml_setup = False\n"
            "    return original_fitter(config)\n"
            "pipeline.InfinitesimalREMLFitter = fitter\n"
            "evaluate = pipeline._evaluate_lasso_path_on_validation\n"
            "def without_reuse(**kwargs):\n"
            "    kwargs.pop('hinv_residual_path', None)\n"
            "    kwargs.pop('training_score_path', None)\n"
            "    return evaluate(**kwargs)\n"
            "pipeline._evaluate_lasso_path_on_validation = without_reuse\n"
            "pipeline.main()\n"
        )
        reference_prefix = tmp_path / "reference"
        reference_command = command.copy()
        reference_command[reference_command.index("--out-prefix") + 1] = str(reference_prefix)
        reference_command += ["--low-level-pipeline", str(worker)]
        reference_run = subprocess.run(
            reference_command, cwd=ROOT.parent, env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=240,
        )
        assert reference_run.returncode == 0, reference_run.stdout[-5000:]
        reference = json.loads(reference_prefix.with_suffix(".summary.json").read_text())
        assert final["lasso_selected_lam_ratio"] == reference["lasso_selected_lam_ratio"]
        assert final["h2"] == pytest.approx(reference["h2"], abs=5e-4)
        np.testing.assert_allclose(final["var_components_lasso_ml"],
                                   reference["var_components_lasso_ml"], atol=5e-4)
