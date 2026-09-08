"""Self-contained regressions for the sparse/REML resource contracts."""
import importlib
from pathlib import Path
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
PLAN = importlib.import_module(f"{ROOT.name}.suggest_params_v3")
SLQ = importlib.import_module(f"{ROOT.name}.slq")
STREAM = importlib.import_module(f"{ROOT.name}.geno_stream")


def test_strict_planning_ignores_removed_probes_but_smile_reserves_them():
    args = dict(n_samples=50_000, p_list=[10_000]*64, n_grm=64,
                gpu_budget_bytes=4*1024**3, n_covar=10, slq_samples=100, slq_m=50)
    strict = PLAN.suggest_call_width(**args, n_rand_vec=100)
    absent = PLAN.suggest_call_width(**args, n_rand_vec=0)
    assert strict == absent
    smile = PLAN.suggest_call_width(**args, n_rand_vec=100, optimizer="smile_scoring")
    assert smile.call_width < strict.call_width
    assert strict.feasible


def test_planner_accounts_for_slq_depth_and_the_one_probe_minimum():
    short = SLQ.workspace_layout(8000, 8, 16, 1024**3)
    deep = SLQ.workspace_layout(8000, 8, 64, 1024**3)
    assert deep["batch_width"] == short["batch_width"] == 8
    assert deep["peak_bytes"] > short["peak_bytes"]
    minimum = SLQ.workspace_layout(50_000, 8, 50, 1)
    assert minimum["batch_width"] == 1
    assert minimum["peak_bytes"] > 1
    args = dict(n_samples=8000, p_list=[1000, 1000], gpu_budget_bytes=1024**3,
                slq_samples=8, requested_call_width=256)
    short_plan = PLAN.suggest_call_width(**args, slq_m=16)
    deep_plan = PLAN.suggest_call_width(**args, slq_m=64)
    assert deep_plan.gpu_slq_peak_gib > short_plan.gpu_slq_peak_gib


def test_single_grm_affine_planning_does_not_reserve_a_reverse_tape():
    args = dict(n_samples=8000, p_list=[1000], gpu_budget_bytes=1024**3,
                slq_samples=8, slq_m=50, requested_call_width=256)
    affine = PLAN.suggest_call_width(**args)
    general = PLAN.suggest_call_width(**args, identity_residual=False)
    assert general.gpu_slq_peak_gib > affine.gpu_slq_peak_gib


class ArraySource:
    n, m, missing_val = 12, 4, -9

    def read_block_variant_major(self, start, count):
        values = np.arange(self.n*self.m, dtype=np.int8).reshape(self.n, self.m) % 3
        return np.asfortranarray(values[:, start:start+count].T)


@pytest.mark.parametrize("explicit", [None, 0, 100_000])
def test_streamer_respects_environment_and_numba_ceiling(monkeypatch, explicit):
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    stream = STREAM.GenoBlockStreamer(ArraySource(), call_width=4, build_threads=explicit)
    try:
        expected = min(explicit or 2, STREAM.numba_config.NUMBA_NUM_THREADS)
        assert stream._build_threads == expected
    finally:
        stream.close()


def test_numba_mask_clamps_and_restores_after_an_exception():
    previous = STREAM.get_num_threads()
    ceiling = STREAM.numba_config.NUMBA_NUM_THREADS
    with pytest.raises(RuntimeError, match="interrupted build"):
        with STREAM._numba_thread_mask(ceiling + 1):
            assert STREAM.get_num_threads() == ceiling
            raise RuntimeError("interrupted build")
    assert STREAM.get_num_threads() == previous


def test_adaptive_import_is_independent_of_sparse_cli_and_keeps_precision():
    script = (
        f"import {ROOT.name}.adaptive_ld; import sys, jax; "
        f"assert '{ROOT.name}.run_sparse_reml_pipeline' not in sys.modules; "
        "assert jax.config.jax_default_matmul_precision == 'highest'"
    )
    subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT.parent, check=True, timeout=60,
        env={**os.environ, "GPU_REML_MATMUL_PRECISION": "highest", "JAX_PLATFORMS": "cpu"},
    )
