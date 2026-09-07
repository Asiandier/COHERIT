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


def test_frozen_fit_uses_current_partition_trace_and_keeps_sparse_variance(
    monkeypatch, tmp_path: Path,
) -> None:
    theta = np.asarray([0.3, 0.2, 0.5])
    atoms = np.asarray([0.5, 1.5])
    q = 0.12
    sparse_mean = np.asarray([0.1, -0.1, 0.0])
    calls = {}

    def fit_infinitesimal(response, covar, *, var_components_init):
        calls["response"] = np.asarray(response)
        calls["theta_init"] = np.asarray(var_components_init)
        return SimpleNamespace(
            var_components=theta,
            genetic_trace_atoms=atoms,
            final_loglik=-0.7,
            history=[{"stop_reason": "ll_down", "accepted": False, "converged": True}],
        )

    context = SimpleNamespace(
        groups=[np.arange(2), np.arange(2, 6)],
        grm_index=SimpleNamespace(m_total=6),
        y=np.asarray([0.2, -0.3, 0.1]),
        covar=np.ones((3, 1)),
        fitter=SimpleNamespace(fit_infinitesimal=fit_infinitesimal),
        close=lambda: calls.update(closed=True),
    )
    parent_path = tmp_path / "parent.json"
    # The parent has a different partition: its atom must not be reused.
    parent_path.write_text(json.dumps({
        "q_chive": q, "support_size": 1, "genetic_trace_atoms": [0.8],
    }))
    args = SimpleNamespace(
        parent_summary=parent_path,
        state_path=tmp_path / "state.npz",
        component_spec=tmp_path / "child.npz",
        theta_init_json=json.dumps(theta.tolist()),
        out=tmp_path / "frozen.json",
    )
    monkeypatch.setattr(ADAPTIVE, "build_analysis_context", lambda args: context)
    monkeypatch.setattr(ADAPTIVE, "_load_sparse_mean", lambda *args: (None, None, sparse_mean))
    ADAPTIVE.run_frozen_fit(args)
    summary = json.loads(args.out.read_text())
    assert calls["closed"]
    np.testing.assert_allclose(calls["response"], context.y - sparse_mean)
    np.testing.assert_allclose(calls["theta_init"], theta)
    np.testing.assert_array_equal(summary["var_components_lasso_ml"], theta)
    np.testing.assert_array_equal(summary["genetic_trace_atoms"], atoms)
    assert summary["grm_variance_scale"] == "trace_weighted"
    assert summary["q_chive"] == q
    assert summary["h2"] == pytest.approx((q + 0.45) / (q + 0.45 + 0.5))
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


def test_covariance_factor_probes_match_dense_formula_and_preserve_batch_prefix():
    rng = np.random.default_rng(72)
    n, m = 11, 12
    z = rng.normal(size=(n, m)).astype(np.float32)
    z[:, 1] = 0
    z[:, 8:] = 0
    theta = np.array([0.3, 0.2, 0.0, 0.5])
    normalizers = np.array([3, 4, 0])
    index = SimpleNamespace(
        streamer=SimpleNamespace(n=n, _component_eff_m_host=normalizers), m_total=m,
        offsets=np.array([0, 4, 8, 12]), extract_standardized_columns=lambda ix: z[:, ix],
    )
    seeds = np.random.SeedSequence(92).spawn(17)
    streams = [np.random.default_rng(seed) for seed in seeds]
    actual = ADAPTIVE._factor_rhs(index, theta, streams)
    streams = [np.random.default_rng(seed) for seed in seeds]
    marker_signs = np.column_stack([2*r.integers(0, 2, size=m, dtype=np.int32)-1 for r in streams])
    residual_signs = np.column_stack([2*r.integers(0, 2, size=n, dtype=np.int32)-1 for r in streams])
    weights = np.repeat(np.sqrt([0.3/3, 0.2/4, 0]), 4)
    expected = z @ (weights[:, None]*marker_signs) + np.sqrt(theta[-1])*residual_signs
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)
    split = np.column_stack([
        ADAPTIVE._factor_rhs(index, theta, [np.random.default_rng(seed) for seed in group])
        for group in (seeds[:8], seeds[8:])
    ])
    np.testing.assert_allclose(split, actual, atol=2e-6, rtol=2e-6)


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


def test_full_core_score_with_real_bed_pcg_matches_dense_reml(tmp_path):
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
        projector = ADAPTIVE.REMLProjector(
            fitter, fitter._assemble_reml_operators(), theta, covar, 1e-6, 200,
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
        z = index.extract_standardized_columns(np.arange(m)).astype(float)
        ranked = z[:, index.cache_variant_indices(order)]
        directions = np.stack([
            z[:, :32]@z[:, :32].T/normalizers[0],
            z[:, 32:]@z[:, 32:].T/normalizers[1], np.eye(n),
        ] + [ranked[:, :t]@ranked[:, :t].T/t - ranked[:, t:]@ranked[:, t:].T/(m-t)
             for t in candidates])
        v = np.einsum("a,aij->ij", theta, directions[:3])
        vi = np.linalg.inv(v)
        p = vi - vi@covar@np.linalg.solve(covar.T@vi@covar, covar.T@vi)
        pd = p@directions
        fisher = 0.5*np.einsum("aij,bji->ab", pd, pd)
        coefficients = np.linalg.solve(fisher[:3, :3], fisher[:3, 3:]).T
        raw_scores = 0.5*(np.einsum("i,aij,j->a", p@response, directions, p@response)
                          - np.trace(pd, axis1=1, axis2=2))
        information = fisher[3:, 3:] - coefficients @ fisher[:3, 3:]
        with ADAPTIVE.ScoreWorkspace(tmp_path) as workspace:
            process = ADAPTIVE._estimate_score_process(
                args, projector, index, contractions, response, workspace=workspace,
            )
            np.testing.assert_allclose(process.scores, raw_scores[3:]-coefficients@raw_scores[:3],
                                       atol=2e-4, rtol=2e-4)
            np.testing.assert_allclose(process.information, np.diag(information), atol=2e-4, rtol=2e-4)
            matrix = process.reference_matrices.astype(float)
            actual = 0.5*np.einsum("aij,bji->ab", matrix, matrix)
            scale = np.sqrt(np.diag(information))
            np.testing.assert_allclose(actual, information/scale[:, None]/scale, atol=2e-4, rtol=2e-4)
            assert process.diagnostics["core_rank"] == n-covar.shape[1]
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
    assert result["method"] == "joint_quadratic_reml_ld_cusum"
    assert result["diagnostics"]["global_p_value"] == 1.0
    assert closed == [True]
