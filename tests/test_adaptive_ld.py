import importlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT = os.path.dirname(REPO_ROOT)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)
PKG = importlib.import_module(os.path.basename(REPO_ROOT))
ADAPTIVE = importlib.import_module(f"{PKG.__name__}.adaptive_ld")
LD_SCORE = importlib.import_module(f"{PKG.__name__}.ld_score")
add_ld_boundary = ADAPTIVE.add_ld_boundary
existing_boundary_positions = ADAPTIVE.existing_boundary_positions
load_component_groups = ADAPTIVE.load_component_groups
load_ld_rank = ADAPTIVE.load_ld_rank
write_root_component_spec = ADAPTIVE.write_root_component_spec
write_ld_rank_artifact = LD_SCORE.write_ld_rank_artifact


@pytest.mark.parametrize("beta", [np.zeros(4), np.array([.3, 0., -.2, 0.])])
def test_frozen_state_load_uses_active_source_ids_and_current_design(tmp_path, beta):
    rng = np.random.default_rng(801)
    source_ids = np.array([9, 2, 6, 4])
    design = rng.normal(size=(13, 4))
    covar = np.column_stack([np.ones(13), rng.normal(size=13)])
    state = tmp_path / "state.npz"
    np.savez(state, marker_indices=source_ids, selected_beta_snp=beta)
    source_to_cache = {9: 2, 2: 0, 6: 3, 4: 1}
    cached_design = design[:, [1, 3, 0, 2]]
    index = SimpleNamespace(
        cache_variant_indices=lambda ids: np.array([source_to_cache[i] for i in ids], dtype=int),
        extract_standardized_columns=lambda ids: cached_design[:, ids],
    )
    markers, loaded_beta, mean, information = ADAPTIVE._load_sparse_mean(state, index, covar)
    np.testing.assert_array_equal(markers, source_ids[beta != 0])
    np.testing.assert_array_equal(loaded_beta, beta[beta != 0])
    np.testing.assert_allclose(mean, design @ beta, atol=1e-12)
    assert information.rank == np.count_nonzero(beta)
    if information.rank:
        q = information.basis
        np.testing.assert_allclose(q @ q.T @ mean, mean, atol=1e-12)
    else:
        assert information.basis.shape == (13, 0)


def test_frozen_fit_corrects_covariance_and_recomputes_paired_sparse_variance(
    monkeypatch, tmp_path: Path,
) -> None:
    import jax.numpy as jnp

    rng = np.random.default_rng(118)
    n = 11
    theta = np.asarray([0.3, 0.2, 0.5])
    kernels = []
    for count in (7, 9):
        z = rng.normal(size=(n, count))
        kernels.append(z @ z.T / count)
    covariance = theta[0]*kernels[0] + theta[1]*kernels[1] + theta[2]*np.eye(n)
    atoms = np.array([np.trace(k)/n for k in kernels])
    design = rng.normal(size=(n, 2))
    covar = np.column_stack([np.ones(n), rng.normal(size=n)])
    sparse_mean = design @ np.array([.2, -.1])
    mean_info = ADAPTIVE.SparseMeanInformation(design, covar)
    calls = {}

    def fit_infinitesimal(response, covar, *, var_components_init, mean_information):
        calls["response"] = np.asarray(response)
        calls["theta_init"] = np.asarray(var_components_init)
        assert mean_information is mean_info
        return SimpleNamespace(
            var_components=theta,
            genetic_trace_atoms=atoms,
            final_loglik=-0.7,
            history=[{"stop_reason": "ll_down", "accepted": False, "converged": True}],
        )

    context = SimpleNamespace(
        groups=[np.arange(2), np.arange(2, 6)],
        grm_index=SimpleNamespace(m_total=6),
        y=rng.normal(size=n),
        covar=covar,
        fitter=SimpleNamespace(
            fit_infinitesimal=fit_infinitesimal,
            _assemble_reml_operators=lambda: None,
            _make_hv=lambda *args: lambda rhs: jnp.asarray(covariance) @ rhs,
            _make_effect_precond=lambda *args: None,
        ),
        close=lambda: calls.update(closed=True),
    )
    parent_path = tmp_path / "parent.json"
    # The parent has a different partition: its atom must not be reused.
    parent_path.write_text(json.dumps({
        "q_chive": 999.0, "support_size": 2, "genetic_trace_atoms": [0.8],
    }))
    args = SimpleNamespace(
        parent_summary=parent_path,
        state_path=tmp_path / "state.npz",
        component_spec=tmp_path / "child.npz",
        theta_init_json=json.dumps(theta.tolist()),
        out=tmp_path / "frozen.json",
        pcg_tol=1e-6, max_pcg_iters=200,
    )
    monkeypatch.setattr(ADAPTIVE, "build_analysis_context", lambda args: context)
    monkeypatch.setattr(ADAPTIVE, "_load_sparse_mean", lambda *args: (
        np.array([0, 1]), None, sparse_mean, mean_info,
    ))
    ADAPTIVE.run_frozen_fit(args)
    summary = json.loads(args.out.read_text())
    assert calls["closed"]
    np.testing.assert_allclose(calls["response"], context.y - sparse_mean)
    np.testing.assert_allclose(calls["theta_init"], theta)
    np.testing.assert_array_equal(summary["var_components_lasso_ml"], theta)
    np.testing.assert_array_equal(summary["genetic_trace_atoms"], atoms)
    assert summary["grm_variance_scale"] == "trace_weighted"
    inverse = np.linalg.inv(covariance)
    gamma = np.linalg.solve(covar.T @ inverse @ covar, covar.T @ inverse @ (context.y-sparse_mean))
    p = inverse-inverse @ covar @ np.linalg.solve(covar.T @ inverse @ covar, covar.T @ inverse)
    sigma = design @ np.linalg.solve(design.T @ p @ design, design.T)
    q = (sparse_mean @ sparse_mean + 2*sparse_mean @ (context.y-sparse_mean-covar @ gamma)
         - np.trace(sigma))/n
    assert summary["q_chive"] == pytest.approx(q, abs=1e-6)
    np.testing.assert_allclose(summary["beta_cov"], gamma, atol=1e-6)
    assert summary["mean_uncertainty_trace_per_n"] == pytest.approx(np.trace(sigma)/n, abs=1e-6)
    background = atoms @ theta[:-1]
    assert summary["h2"] == pytest.approx((q + background)/(q + background + 0.5), abs=2e-6)
    assert summary["method"] == "fixed_sparse_mean_information_corrected_reml"
    assert "parent_q_sparse_held_fixed" not in summary
    assert summary["stop_reason"] == "ll_down"


def test_add_ld_boundary_preserves_genetic_variance_and_rank_intervals(
    tmp_path: Path,
) -> None:
    rank = tmp_path / "rank.npz"
    write_ld_rank_artifact(
        rank,
        np.asarray([4.0, 1.0, 6.0, 3.0, 2.0, 5.0]),
        bins=3,
    )
    root = tmp_path / "k1.npz"
    write_root_component_spec(root, marker_count=6, rank_path=rank)
    child = tmp_path / "k2.npz"
    result = add_ld_boundary(
        parent_spec=root,
        parent_theta=[0.6, 0.4],
        rank_path=rank,
        boundary_position=2,
        output_spec=child,
    )
    groups = load_component_groups(child, 6)
    order, _positions = load_ld_rank(rank)
    np.testing.assert_array_equal(groups[0], np.sort(order[:2]))
    np.testing.assert_array_equal(groups[1], np.sort(order[2:]))
    np.testing.assert_array_equal(existing_boundary_positions(groups, order), [2])
    theta = np.asarray(result["variance_components_init"])
    assert theta.shape == (3,)
    np.testing.assert_allclose(theta[:-1].sum(), 0.6)
    assert theta[-1] == 0.4


def test_reml_projector_profiles_covariates_without_explicit_gls_residual() -> None:
    rng = np.random.default_rng(90210)
    n = 17
    covar = np.column_stack(
        [np.ones(n), rng.normal(size=n), rng.normal(size=n)]
    )
    covariance_root = rng.normal(size=(n, n))
    covariance = covariance_root @ covariance_root.T / n + 0.5 * np.eye(n)
    precision = np.linalg.inv(covariance)

    class ExactProjector:
        _covar = covar
        _vinv_c = precision @ covar
        _gram_inverse = np.linalg.inv(covar.T @ precision @ covar)

        @staticmethod
        def _solve(rhs, *, stage):
            del stage
            return precision @ rhs

    response = rng.normal(size=n)
    nuisance_shift = covar @ rng.normal(size=covar.shape[1])
    direct = ADAPTIVE.REMLProjector.apply(
        ExactProjector(), response, stage="direct"
    )
    shifted = ADAPTIVE.REMLProjector.apply(
        ExactProjector(), response + nuisance_shift, stage="shifted"
    )

    np.testing.assert_allclose(direct, shifted, rtol=5e-7, atol=5e-7)
    assert not hasattr(ADAPTIVE.REMLProjector, "fit_covariates")


def test_boundary_contractions_match_dense_kernels_with_permuted_order_and_monomorphic_markers():
    rng = np.random.default_rng(819)
    n, m = 13, 8
    z = rng.normal(size=(n, m))
    z[:, [1, 4]] = 0
    order = np.array([2, 0, 1, 5, 6, 7, 4, 3])
    positions = np.array([2, 3, 5, 7])
    candidates = np.array([2, 5, 7])
    contractions = ADAPTIVE.BoundaryContractions(
        order, positions, candidates, [(0, 3), (3, 8)], np.array([2, 4]),
    )
    left, right = rng.normal(size=(2, n, 7))
    actual = contractions.apply(left, right, z.T @ left, z.T @ right)
    directions = [z[:, :3] @ z[:, :3].T / 2, z[:, 3:] @ z[:, 3:].T / 4, np.eye(n)]
    ranked = z[:, order]
    directions += [ranked[:, :t] @ ranked[:, :t].T / t
                   - ranked[:, t:] @ ranked[:, t:].T / (m - t) for t in candidates]
    expected = np.einsum("ir,aij,jr->ar", left, np.stack(directions), right)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


@pytest.mark.parametrize("candidates,intervals", [
    ([3], [(0, 8)]), ([2], [(0, 3), (3, 8)]),
])
def test_boundary_contractions_reject_off_grid_directions(candidates, intervals):
    with pytest.raises(ValueError, match="LD-rank grid"):
        ADAPTIVE.BoundaryContractions(np.arange(8), np.array([2, 4, 6]),
                                      np.array(candidates), intervals, np.ones(len(intervals)))


@pytest.mark.parametrize("call_width", [3, 8, 32])
def test_covariance_factor_probes_match_dense_formula_and_preserve_batch_prefix(tmp_path, call_width, monkeypatch):
    from bed_reader import to_bed

    rng = np.random.default_rng(72)
    n, m = 11, 18
    raw = rng.binomial(2, .3, size=(n, m)).astype(np.int8)
    raw[:, 1] = 0
    raw[:, 12:] = 0
    raw[0, 2] = -127
    theta = np.array([0.3, 0.2, 0.0, 0.5])
    prefix = tmp_path/"factor"
    to_bed(str(prefix)+".bed", raw)
    groups = [np.arange(7), np.arange(7, 12), np.arange(12, m)]
    monkeypatch.setattr(ADAPTIVE, "_FACTOR_MAX_REDUCTION_WIDTH", 4)
    fitter = ADAPTIVE.InfinitesimalREMLFitter(ADAPTIVE.FitConfig(
        bed_prefix=str(prefix), component_variant_indices=groups, call_width=call_width,
        precond_rank=0, cpu_threads=1, keep_host_stats=True, verbose=False,
        device=os.environ.get("COHERIT_TEST_DEVICE", "cpu"),
    ))
    try:
        index = ADAPTIVE.MultiGRMIndex(fitter.streamers, component_variant_indices=groups)
        z = index.extract_standardized_columns(np.arange(m)).astype(float)
        normalizers = index.streamer._component_eff_m_host
        seeds = np.random.SeedSequence(92).spawn(17)
        streams = [np.random.default_rng(seed) for seed in seeds]
        actual = ADAPTIVE._factor_rhs(index, theta, streams)
        streams = [np.random.default_rng(seed) for seed in seeds]
        marker_signs = np.column_stack([2*r.integers(0, 2, size=m, dtype=np.int32)-1 for r in streams])
        residual_signs = np.column_stack([2*r.integers(0, 2, size=n, dtype=np.int32)-1 for r in streams])
        weights = np.repeat(np.sqrt(np.divide(theta[:-1], normalizers,
                            out=np.zeros(3), where=normalizers>0)), [7, 5, 6])
        expected = z @ (weights[:, None]*marker_signs) + np.sqrt(theta[-1])*residual_signs
        np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=3e-6)
        split = np.column_stack([
            ADAPTIVE._factor_rhs(index, theta, [np.random.default_rng(seed) for seed in group])
            for group in (seeds[:8], seeds[8:])
        ])
        np.testing.assert_allclose(split, actual, atol=3e-6, rtol=3e-6)
    finally:
        fitter.close()


def test_full_bilinear_prefixes_match_dense_with_arbitrary_cache_order(tmp_path):
    rng = np.random.default_rng(81)
    n, m = 17, 12
    z = rng.normal(size=(n, m))
    z[:, [1, 5]] = 0
    order = rng.permutation(m)
    positions = np.array([2, 4, 8, 10])
    ranked = z[:, order]
    normalizers = np.array([np.count_nonzero(np.any(ranked[:, :6], axis=0)),
                            np.count_nonzero(np.any(ranked[:, 6:], axis=0))])
    contractions = ADAPTIVE.BoundaryContractions(
        order, np.arange(2, m, 2), positions, [(0, 6), (6, m)], normalizers,
    )
    directions = np.stack([
        ranked[:, :6] @ ranked[:, :6].T / normalizers[0],
        ranked[:, 6:] @ ranked[:, 6:].T / normalizers[1], np.eye(n),
    ] + [ranked[:, :t] @ ranked[:, :t].T/t - ranked[:, t:] @ ranked[:, t:].T/(m-t)
         for t in positions])
    left, right = rng.normal(size=(n, 3)), rng.normal(size=(n, 7))
    output = np.memmap(tmp_path/"grams.bin", shape=(7, 3, 7), dtype=float, mode="w+")
    actual = contractions.bilinear(left, right, z.T@left, z.T@right, out=output)
    assert actual is output
    np.testing.assert_allclose(actual, left.T @ directions @ right, atol=1e-12)


def test_score_workspace_cleans_large_arrays_even_after_failure(tmp_path):
    with pytest.raises(RuntimeError, match="intentional"):
        with ADAPTIVE.ScoreWorkspace(tmp_path) as workspace:
            array = workspace.allocate((1024, 1024))
            assert isinstance(array, np.memmap)
            array[0] = 1
            raise RuntimeError("intentional")
    assert list(tmp_path.iterdir()) == []


def test_score_batch_width_uses_runtime_budget():
    index = SimpleNamespace(m_total=632255)
    config = SimpleNamespace(gpu_budget_bytes=20*1024**3)
    projector = SimpleNamespace(_covar=np.empty((8000, 0)), fitter=SimpleNamespace(cfg=config))
    assert ADAPTIVE._score_rhs_width(projector, index) == 128
    config.gpu_budget_bytes = 1024**3
    assert ADAPTIVE._score_rhs_width(projector, index) < 128


def test_common_core_random_draws_do_not_change_with_compute_batching(tmp_path, monkeypatch):
    import jax.numpy as jnp

    n, m, rank = 24, 80, 9
    inputs = []

    def kv(value):
        inputs.append(np.asarray(value).copy())
        return jnp.arange(1, n+1, dtype=jnp.float32)[:, None] * value

    index = SimpleNamespace(m_total=m, streamer=SimpleNamespace(kv=kv),
                            xtv_all=lambda value, **kw:np.zeros((m, value.shape[1])))
    projector = SimpleNamespace(_covar=np.ones((n, 1)), apply_reference=lambda value, **kw:value)

    def bilinear(left, right, *args, out):
        out[:] = left.T @ right

    contractions = SimpleNamespace(intervals=[(0, m)], candidates=np.array([40]), bilinear=bilinear)
    results, draws = [], []
    for width in (4, 12):
        monkeypatch.setattr(ADAPTIVE, "_score_rhs_width", lambda *args:width)
        inputs.clear()
        with ADAPTIVE.ScoreWorkspace(tmp_path) as workspace:
            h, _, _ = ADAPTIVE._build_score_core(projector, index, contractions, rank,
                                                 np.random.default_rng(87), workspace)
            results.append(h.copy())
            draws.append(np.column_stack(inputs))
    np.testing.assert_array_equal(draws[0], draws[1])
    np.testing.assert_allclose(results[0], results[1], atol=1e-12)


def test_workspace_releases_views_and_preserves_growing_cache(tmp_path):
    with ADAPTIVE.ScoreWorkspace(tmp_path) as workspace:
        cache = workspace.allocate((0, 1024))
        cache = workspace.grow_rows(cache, 2)
        cache[:] = np.arange(2048).reshape(2, 1024)
        expected = cache.copy()
        cache = workspace.grow_rows(cache, 1024)
        assert isinstance(cache, np.memmap)
        np.testing.assert_array_equal(cache[:2], expected)
        path = Path(cache.filename)
        cache = workspace.grow_rows(cache, 2048)
        assert Path(cache.filename) == path
        np.testing.assert_array_equal(cache[:2], expected)
        assert workspace.live_bytes == cache.nbytes
        workspace.release(np.asarray(cache))
        assert not path.exists() and workspace.live_bytes == 0
        other = workspace.allocate((1024, 1024))
        assert Path(other.filename) != path  # Released filenames are not reused.
    assert list(tmp_path.iterdir()) == []


def test_score_workspace_checks_scratch_space_before_allocation(tmp_path, monkeypatch):
    monkeypatch.setattr(ADAPTIVE.shutil, "disk_usage", lambda _:SimpleNamespace(free=1024))
    with ADAPTIVE.ScoreWorkspace(tmp_path) as workspace:
        with pytest.raises(RuntimeError, match="Insufficient score scratch space"):
            workspace.allocate((1024, 1024))
        assert workspace.live_bytes == 0
    assert list(tmp_path.iterdir()) == []


def test_probe_growth_releases_pilot_and_previous_tier(tmp_path, monkeypatch):
    ScoreProcess = importlib.import_module(f"{PKG.__name__}.score_process").ScoreProcess

    n, m, rank, count = 8, 40, 2, 3
    candidates = np.array([10, 20, 30])
    projector = SimpleNamespace(_covar=np.empty((n, 0)), apply=lambda r, **kw:r,
                                mean_information=None)
    index = SimpleNamespace(m_total=m, xtv_all=lambda r, **kw:np.zeros((m, r.shape[1])))
    contractions = SimpleNamespace(intervals=[(0, m)], candidates=candidates,
                                   apply=lambda *a:np.zeros((count+2, 1)))
    args = SimpleNamespace(score_trace_seed=7, score_core_rank=rank, score_trace_probes=32,
                           score_trace_max_probes=64, score_trace_tol=.05)
    with ADAPTIVE.ScoreWorkspace(tmp_path) as workspace:
        def core(*a):
            marker = workspace.allocate((m, rank)); marker[:] = 0
            matrix = workspace.allocate((count+2, rank, rank)); matrix[:] = 0
            return np.zeros((n, rank)), marker, matrix

        def fill(projector, index, h, seed, samples, markers, start, stop, *, stage):
            if "evaluation" in stage:
                # Only core, marker_core and the two probe caches remain alive.
                assert len(workspace._arrays) == 4
            if start:
                np.testing.assert_array_equal(samples[:start, 0], np.arange(start))
            samples[start:stop] = np.arange(start, stop)[:, None]
            markers[start:stop] = 0

        def sketch(contractions, h, marker_h, core, samples, markers, workspace, **kwargs):
            probes = len(samples)
            cross = workspace.allocate((count+2, rank, probes)); cross[:] = 0
            bulk = workspace.allocate((count+2, probes, probes)); bulk[:] = 0
            return ADAPTIVE.TraceSketch(core, cross, bulk, np.zeros((count+2, probes)))

        def moments(quadratics, *, evaluation, coefficients, boundary_positions, reference_out):
            reference_out[:] = 0
            return ScoreProcess(candidates, np.zeros(count), np.zeros(count), np.ones(count),
                                coefficients, reference_out, {
                "max_score_trace_standard_error": .1 if evaluation.probes == 32 else .01,
                "max_information_relative_standard_error": 0.,
            })

        monkeypatch.setattr(ADAPTIVE, "_build_score_core", core)
        monkeypatch.setattr(ADAPTIVE, "_fill_probe_cache", fill)
        monkeypatch.setattr(ADAPTIVE, "_probe_sketch", sketch)
        monkeypatch.setattr(ADAPTIVE, "score_process_moments", moments)
        result = ADAPTIVE._estimate_score_process(args, projector, index, contractions,
                                                 np.zeros(n), workspace=workspace)
        assert len(workspace._arrays) == 1
        assert workspace.live_bytes == result.reference_matrices.nbytes
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("selected_size", [0, 3])
def test_full_core_score_with_real_bed_pcg_matches_dense_corrected_model(tmp_path, selected_size):
    from bed_reader import to_bed

    rng = np.random.default_rng(312)
    n, m = 32, 64
    raw = rng.binomial(2, 0.3, size=(n, m)).astype(np.int8)
    raw[:, ::17] = 0
    raw[rng.random((n, m)) < 0.02] = -127
    prefix = tmp_path/"geno"
    to_bed(str(prefix)+".bed", raw)
    order = rng.permutation(m)
    groups = [np.sort(order[:32]), np.sort(order[32:])]
    fitter = ADAPTIVE.InfinitesimalREMLFitter(ADAPTIVE.FitConfig(
        bed_prefix=str(prefix), component_variant_indices=groups,
        device=os.environ.get("COHERIT_TEST_DEVICE", "cpu"), call_width=32, cpu_threads=1,
        precond_rank=0, keep_host_stats=True, reml_pcg_tol=1e-6, pcg_ridge=0,
        max_pcg_iters=200, verbose=False,
    ))
    try:
        index = ADAPTIVE.MultiGRMIndex(fitter.streamers, component_variant_indices=groups)
        covar = np.column_stack([np.ones(n), rng.normal(size=n)])
        theta = np.array([0.25, 0.35, 0.5])
        z = index.extract_standardized_columns(np.arange(m)).astype(float)
        selected = z[:, 1:1+selected_size]
        mean_info = ADAPTIVE.SparseMeanInformation(selected, covar)
        projector = ADAPTIVE.REMLProjector(
            fitter, fitter._assemble_reml_operators(), theta, covar, 1e-6, 200,
            mean_information=mean_info,
        )
        normalizers = np.asarray(index.streamer._component_eff_m_host)
        candidates = np.array([8, 16, 24, 40, 48, 56])
        contractions = ADAPTIVE.BoundaryContractions(
            index.cache_variant_indices(order), np.arange(8, m, 8), candidates,
            [(0, 32), (32, m)], normalizers,
        )
        args = SimpleNamespace(score_trace_seed=74, score_core_rank=64, score_trace_probes=32,
                               score_trace_max_probes=128, score_trace_tol=0.05)
        response = rng.normal(size=n).astype(np.float32)
        ranked = z[:, index.cache_variant_indices(order)]
        directions = np.stack([
            z[:, :32]@z[:, :32].T/normalizers[0],
            z[:, 32:]@z[:, 32:].T/normalizers[1], np.eye(n),
        ] + [ranked[:, :t]@ranked[:, :t].T/t - ranked[:, t:]@ranked[:, t:].T/(m-t)
             for t in candidates])
        v = np.einsum("a,aij->ij", theta, directions[:3])
        vi = np.linalg.inv(v)
        p = vi - vi@covar@np.linalg.solve(covar.T@vi@covar, covar.T@vi)
        sigma = (selected @ np.linalg.solve(selected.T @ p @ selected, selected.T)
                 if selected_size else np.zeros((n, n)))
        ps = p-p @ sigma @ p
        pd = ps@directions
        fisher = 0.5*np.einsum("aij,bji->ab", pd, pd)
        coefficients = np.linalg.solve(fisher[:3, :3], fisher[:3, 3:]).T
        raw_scores = 0.5*(np.einsum("i,aij,j->a", p@response, directions, p@response)
                          - np.trace(pd, axis1=1, axis2=2))
        information = fisher[3:, 3:] - coefficients @ fisher[:3, 3:]
        with ADAPTIVE.ScoreWorkspace(tmp_path) as workspace:
            process = ADAPTIVE._estimate_score_process(
                args, projector, index, contractions, response, workspace=workspace,
            )
            np.testing.assert_allclose(process.raw_scores, raw_scores[3:], atol=2e-4, rtol=2e-4)
            np.testing.assert_allclose(process.scores, raw_scores[3:]-coefficients@raw_scores[:3],
                                       atol=2e-4, rtol=2e-4)
            np.testing.assert_allclose(process.information, np.diag(information), atol=2e-4, rtol=2e-4)
            matrix = process.reference_matrices.astype(float)
            actual = 0.5*np.einsum("aij,bji->ab", matrix, matrix)
            scale = np.sqrt(np.diag(information))
            np.testing.assert_allclose(actual, information/scale[:, None]/scale, atol=2e-4, rtol=2e-4)
            assert process.diagnostics["core_rank"] == n-covar.shape[1]-selected_size
            assert process.diagnostics["mean_information_rank"] == selected_size
            np.testing.assert_allclose(projector.apply_reference(response, stage="oracle"),
                                       ps @ response, atol=3e-5)
            if selected_size:
                assert np.linalg.norm((p-ps) @ response) > .01
                # A finite-difference derivative proves these are ell_C scores,
                # not scores from fully profiling the selected SNP directions.
                def objective(covariance):
                    vi = np.linalg.inv(covariance)
                    pc = vi-vi@covar@np.linalg.solve(covar.T@vi@covar, covar.T@vi)
                    return -.5*(np.linalg.slogdet(covariance)[1]
                                + np.linalg.slogdet(covar.T@vi@covar)[1]
                                + np.linalg.slogdet(selected.T@pc@selected)[1]
                                + response@pc@response)
                derivatives = [(objective(v+1e-5*d)-objective(v-1e-5*d))/2e-5
                               for d in directions[3:]]
                np.testing.assert_allclose(process.raw_scores, derivatives, atol=3e-4, rtol=3e-4)
        assert not list(tmp_path.glob(".score-*"))
    finally:
        fitter.close()



def test_zero_parent_variance_stops_before_trace_solves(monkeypatch, tmp_path):
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"var_components_lasso_ml": [0.0, 1.0]}))
    rank = tmp_path / "rank.npz"
    write_ld_rank_artifact(rank, np.arange(8), bins=4)
    closed = []
    context = SimpleNamespace(groups=[np.arange(8)], close=lambda: closed.append(True))
    monkeypatch.setattr(ADAPTIVE, "build_analysis_context", lambda args: context)
    args = SimpleNamespace(summary_path=summary, rank_path=rank, split_alpha=0.05,
                           score_trace_seed=3, out=tmp_path / "score.json")
    ADAPTIVE.run_score(args)
    result = json.loads(args.out.read_text())
    assert result["stop_reason"] == "no_positive_parent_variance"
    assert not result["accepted"]
    assert result["zero_parent_boundaries_excluded"] == [2, 4, 6]
    assert result["method"] == "joint_quadratic_information_corrected_ld_cusum"
    assert result["diagnostics"]["global_p_value"] == 1.0
    assert closed == [True]
