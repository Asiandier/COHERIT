"""Publication-critical regressions for the sparse--dense heritability path."""

from __future__ import annotations

import importlib
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT = os.path.dirname(REPO_ROOT)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

PKG = os.path.basename(REPO_ROOT)
SPARSE = importlib.import_module(f"{PKG}.run_sparse_reml_pipeline")
REML = importlib.import_module(f"{PKG}.reml")
LASSO = importlib.import_module(f"{PKG}.lasso_cd")


class _PartitionIndexStreamer:
    def __init__(self, cache_to_source, component_sizes):
        self._cache_to_source_variant_indices = np.asarray(
            cache_to_source, dtype=np.int64
        )
        self._component_snp_offsets = np.concatenate(
            [
                np.asarray([0], dtype=np.int64),
                np.cumsum(component_sizes, dtype=np.int64),
            ]
        )
        self.n_components = len(component_sizes)
        self.has_component_partition = True
        self.m = int(self._cache_to_source_variant_indices.size)
        self.n = 3
        self._columns = np.arange(self.n * self.m, dtype=np.float32).reshape(
            self.n, self.m
        )

    def extract_standardized_columns(self, indices):
        return self._columns[:, np.asarray(indices, dtype=np.int64)]


def test_multi_grm_index_preserves_cache_and_source_coordinates():
    streamer = _PartitionIndexStreamer(
        cache_to_source=[0, 2, 5, 1, 3, 4],
        component_sizes=[3, 3],
    )
    index = SPARSE.MultiGRMIndex(
        [streamer],
        component_variant_indices=[np.asarray([5, 0, 2]), np.asarray([4, 1, 3])],
    )

    assert index.n_grm == 2
    np.testing.assert_array_equal(index.m_per_grm, [3, 3])
    np.testing.assert_array_equal(
        index.source_variant_indices(np.asarray([0, 3, 5])), [0, 1, 4]
    )
    np.testing.assert_array_equal(
        index.cache_variant_indices(np.asarray([4, 0, 1])), [5, 0, 3]
    )
    groups = index.global_to_local(np.asarray([5, 1, 3]))
    assert [(g, local.tolist(), pos.tolist()) for g, local, pos in groups] == [
        (0, [1], [1]),
        (1, [2, 0], [0, 2]),
    ]
    np.testing.assert_array_equal(
        index.extract_standardized_columns(np.asarray([5, 1, 3])),
        streamer._columns[:, [5, 1, 3]],
    )


def test_k1_component_spec_index_matches_unpartitioned_coordinates():
    partitioned = _PartitionIndexStreamer(
        cache_to_source=[0, 1, 2, 3], component_sizes=[4]
    )
    indexed = SPARSE.MultiGRMIndex(
        [partitioned],
        component_variant_indices=[np.arange(4, dtype=np.int64)],
    )
    unpartitioned = SimpleNamespace(
        m=4,
        n=3,
        has_component_partition=False,
        n_components=1,
        extract_standardized_columns=partitioned.extract_standardized_columns,
    )
    plain = SPARSE.MultiGRMIndex([unpartitioned])
    marker_indices = np.asarray([3, 0, 2], dtype=np.int64)

    np.testing.assert_array_equal(
        indexed.source_variant_indices(marker_indices),
        plain.source_variant_indices(marker_indices),
    )
    np.testing.assert_array_equal(
        indexed.extract_standardized_columns(marker_indices),
        plain.extract_standardized_columns(marker_indices),
    )


def test_multi_grm_theta_parser_preserves_kernel_coefficients():
    theta = SPARSE._parse_variance_components_init(
        "[0.2, 0.3, 0.4]", n_grm=2
    )
    assert SPARSE._sparse_dense_h2(
        0.1, SPARSE.genetic_variance(theta[:-1], [0.5, 0.8]), float(theta[-1])
    ) == pytest.approx(
        0.44 / 0.84
    )
    np.testing.assert_array_equal(theta, [0.2, 0.3, 0.4])
    with pytest.raises(ValueError, match="expected 3"):
        SPARSE._parse_variance_components_init("[0.2, 0.8]", n_grm=2)


@pytest.mark.parametrize("scales", [(1.0,), (0.5,), (0.4, 1.4), (0.0, 0.7)])
def test_trace_weighted_variance_matches_dense_covariance_without_changing_reml(scales):
    rng = np.random.default_rng(903)
    n = 24
    kernels = []
    for scale in scales:
        z = rng.normal(size=(n, 8))
        kernel = z @ z.T
        kernels.append(kernel * (scale / np.mean(np.diag(kernel))))
    operators = [
        lambda value, kernel=REML.jnp.asarray(k, dtype=REML.jnp.float32): kernel @ value
        for k in kernels
    ]
    kwargs = dict(
        y=REML.jnp.asarray(rng.normal(size=n), dtype=REML.jnp.float32),
        K_mvs=operators,
        diag_list=[REML.jnp.asarray(np.diag(k)) for k in kernels],
        covar=REML.jnp.ones((n, 1)),
        param_init=REML.jnp.asarray([0.4 / len(scales)] * len(scales) + [0.6]),
        n_rand_vec=32,
        maxiter=100,
        minq_iter=3,
        slq_samples=32,
        slq_m=n,
        pcg_tol=1e-7,
        precond_conf=None,
        response_is_standardized=True,
        return_diagnostics=True,
        verbose=False,
    )
    old_theta, _, old_diagnostics = REML.fit_reml(
        **kwargs, unit_variance_components=True
    )
    theta, _, diagnostics = REML.fit_reml(**kwargs)
    np.testing.assert_array_equal(theta, old_theta)
    for key in ("grad", "ai", "loglik"):
        np.testing.assert_array_equal(diagnostics[key], old_diagnostics[key])
    atoms = np.asarray(diagnostics["genetic_trace_atoms"])
    np.testing.assert_allclose(atoms, scales, atol=1e-7)
    genetic_covariance = sum(float(t) * k for t, k in zip(theta[:-1], kernels))
    dense_variance = np.trace(genetic_covariance) / n
    background = SPARSE.genetic_variance(np.asarray(theta[:-1]), atoms)
    assert background == pytest.approx(dense_variance, abs=1e-7)
    q = 0.13
    assert SPARSE._sparse_dense_h2(q, background, float(theta[-1])) == pytest.approx(
        (q + dense_variance) / (q + dense_variance + float(theta[-1]))
    )


def test_component_spec_partition_must_be_exhaustive_and_disjoint():
    valid = SPARSE._validate_component_partition(
        [np.asarray([3, 1]), np.asarray([0, 2])], n_markers=4
    )
    np.testing.assert_array_equal(valid[0], [1, 3])
    np.testing.assert_array_equal(valid[1], [0, 2])
    with pytest.raises(ValueError, match="exactly once"):
        SPARSE._validate_component_partition(
            [np.asarray([0, 1]), np.asarray([1, 2])], n_markers=4
        )


def test_lasso_warm_state_reuses_selected_alpha_for_final_refit(tmp_path):
    state_path = tmp_path / "lasso_warm_state.npz"
    candidate = np.asarray([0, 2, 3], dtype=np.int64)
    support = np.asarray([2, 3], dtype=np.int64)
    selected_beta = np.asarray([0.0, 0.2, -0.1], dtype=np.float64)

    emitted = SPARSE._write_lasso_warm_state(
        str(state_path),
        candidate=candidate,
        support=support,
        selected_beta_snp=selected_beta,
        selected_lam_ratio=0.25,
    )
    assert emitted["coordinate_system"] == "single_grm_marker_index"
    assert emitted["marker_count"] == 2

    loaded = SPARSE._load_lasso_warm_state(
        str(state_path),
        n_markers=4,
        target_lam_ratio=0.25,
    )

    np.testing.assert_array_equal(loaded["candidate"], [2, 3])
    np.testing.assert_array_equal(loaded["support"], [2, 3])
    np.testing.assert_array_equal(loaded["beta_snp_path"][0], np.zeros(2))
    np.testing.assert_allclose(loaded["beta_snp_path"][1], [0.2, -0.1])
    assert loaded["selected_lam_ratio"] == pytest.approx(0.25)


def test_lasso_warm_state_reuses_certified_path_across_partitions(tmp_path):
    parent_streamer = _PartitionIndexStreamer(
        cache_to_source=[0, 2, 5, 1, 3, 4], component_sizes=[3, 3]
    )
    parent = SPARSE.MultiGRMIndex(
        [parent_streamer],
        component_variant_indices=[
            np.asarray([0, 2, 5]),
            np.asarray([1, 3, 4]),
        ],
    )
    child_streamer = _PartitionIndexStreamer(
        cache_to_source=[1, 2, 4, 0, 3, 5], component_sizes=[3, 3]
    )
    child = SPARSE.MultiGRMIndex(
        [child_streamer],
        component_variant_indices=[
            np.asarray([1, 2, 4]),
            np.asarray([0, 3, 5]),
        ],
    )
    state_path = tmp_path / "validation_path_state.npz"
    candidate = np.asarray([0, 1, 3, 5], dtype=np.int64)
    support = np.asarray([1, 5], dtype=np.int64)
    selected_beta = np.asarray([0.0, 0.2, 0.0, -0.1])
    beta_path = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.1, 0.2, 0.0, -0.1],
            [0.3, 0.4, -0.2, -0.5],
        ]
    )
    ratios = np.asarray([1.0, 0.5, 0.25])

    emitted = SPARSE._write_lasso_warm_state(
        str(state_path),
        grm_index=parent,
        candidate=candidate,
        support=support,
        selected_beta_snp=selected_beta,
        selected_lam_ratio=0.5,
        beta_snp_path=beta_path,
        path_lam_ratios=ratios,
    )
    loaded = SPARSE._load_lasso_warm_state(
        str(state_path),
        grm_index=child,
        target_path_lam_ratios=ratios,
    )

    assert emitted["path_rows"] == 3
    assert loaded["reuse_mode"] == "certified_path_prefix_for_validation"
    # Parent candidate source rows are [0, 2, 1, 4]. In the child cache they
    # become [3, 1, 0, 2], sorted to [0, 1, 2, 3].
    np.testing.assert_array_equal(loaded["candidate"], [0, 1, 2, 3])
    np.testing.assert_array_equal(loaded["support"], [1, 2])
    np.testing.assert_allclose(
        loaded["beta_snp_path"], beta_path[:, [2, 1, 3, 0]]
    )


def test_legacy_selected_alpha_is_safe_validation_path_fallback(tmp_path):
    state_path = tmp_path / "legacy_state.npz"
    with open(state_path, "wb") as handle:
        np.savez(
            handle,
            lasso_warm_state_schema_version=np.asarray(2),
            marker_indices=np.asarray([1, 3]),
            selected_beta_snp=np.asarray([0.2, -0.1]),
            selected_lam_ratio=np.asarray(0.25),
        )
    ratios = np.asarray([1.0, 0.5, 0.25, 0.125])

    loaded = SPARSE._load_lasso_warm_state(
        str(state_path), n_markers=4, target_path_lam_ratios=ratios
    )

    assert loaded["reuse_mode"] == "selected_alpha_broadcast_for_validation"
    np.testing.assert_array_equal(loaded["candidate"], [1, 3])
    np.testing.assert_array_equal(loaded["beta_snp_path"][0], [0.0, 0.0])
    np.testing.assert_allclose(
        loaded["beta_snp_path"][1:],
        np.asarray([[0.2, -0.1]] * 3),
    )


def test_validation_selected_path_materializes_alpha_used_downstream():
    lasso_path = {
        "beta_snp": np.asarray([0.0, 0.0]),
        "beta_cov": np.asarray([1.0]),
        "active_idx": np.empty((0,), dtype=np.int64),
        "lam": 10.0,
        "selected_index": 0,
        "selection_method": None,
        "selected_lam_ratio": 1.0,
        "lam_max": 10.0,
        "beta_snp_path": np.asarray([[0.0, 0.0], [2.0, -1.0]]),
        "beta_cov_path": np.asarray([[1.0], [3.0]]),
        "path": [
            {
                "lam": 10.0,
                "lam_ratio": 1.0,
                "k": 0,
                "converged": True,
                "kkt_passed": True,
                "global_kkt_passed": True,
            },
            {
                "lam": 2.0,
                "lam_ratio": 0.2,
                "k": 2,
                "converged": True,
                "kkt_passed": True,
                "global_kkt_passed": True,
            },
        ],
    }
    metrics = [
        {"path_index": 0, "predictive_r2": 0.01},
        {"path_index": 1, "predictive_r2": 0.25},
    ]

    selected_index = SPARSE._select_converged_validation_path_index(
        lasso_path["path"], metrics
    )
    selected, record = SPARSE._materialize_validation_selected_lasso(
        lasso_path,
        metrics,
        selected_index=selected_index,
        path_prediction_pcg_res=1e-4,
        path_prediction_pcg_iters=7,
    )

    assert selected_index == 1
    assert selected["selection_method"] == "validation_predictive_r2"
    assert selected["selected_lam_ratio"] == 0.2
    assert np.array_equal(selected["beta_snp"], [2.0, -1.0])
    assert np.array_equal(selected["beta_cov"], [3.0])
    assert np.array_equal(selected["active_idx"], [0, 1])
    assert record["selected"]["predictive_r2"] == 0.25

    residual = SPARSE._lasso_residual(
        y=np.asarray([10.0, 20.0]),
        covar=np.ones((2, 1)),
        geno=np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        beta_cov=selected["beta_cov"],
        beta_snp=selected["beta_snp"],
    )
    # This is the exact residual passed to the variance-component update.
    assert np.array_equal(residual, [5.0, 18.0])


def test_validation_selection_excludes_unconverged_path_points():
    path = [
        {
            "k": 5,
            "lam_ratio": 0.1,
            "converged": False,
            "kkt_passed": False,
            "global_kkt_passed": False,
        },
        {
            "k": 1,
            "lam_ratio": 0.5,
            "converged": True,
            "kkt_passed": True,
            "global_kkt_passed": True,
        },
    ]
    metrics = [
        {"predictive_r2": 0.9},
        {"predictive_r2": 0.2},
    ]

    assert SPARSE._select_converged_validation_path_index(path, metrics) == 1


def test_validation_early_stopping_waits_for_an_earlier_peak():
    flat = [
        {"predictive_r2": 0.2}
        for _ in range(9)
    ]
    assert not SPARSE._validation_path_early_stopping_decision(
        flat, stopping_lag=5
    )["stopped"]

    peaked = [
        {"predictive_r2": value}
        for value in [0.10, 0.20, 0.31, 0.30, 0.29, 0.28, 0.27, 0.26]
    ]
    decision = SPARSE._validation_path_early_stopping_decision(
        peaked, stopping_lag=5
    )
    assert decision["stopped"] is True
    assert decision["best_path_index"] == 2
    assert decision["best_predictive_r2"] == pytest.approx(0.31)


def test_batched_hinv_residual_path_matches_columnwise_algebra():
    hinv_y = np.asarray([3.0, 5.0, 7.0])
    hinv_covar = np.asarray([[1.0], [2.0], [3.0]])
    hinv_geno = np.asarray(
        [[1.0, 0.5], [0.0, 2.0], [2.0, -1.0]]
    )
    beta_cov_path = np.asarray([[0.0], [0.5], [-1.0]])
    beta_snp_path = np.asarray(
        [[0.0, 0.0], [1.0, -0.5], [0.25, 2.0]]
    )

    batched = SPARSE._build_hinv_lasso_residual_path(
        hinv_y=hinv_y,
        hinv_covar=hinv_covar,
        hinv_geno=hinv_geno,
        beta_cov_path=beta_cov_path,
        beta_snp_path=beta_snp_path,
    )
    expected = np.column_stack(
        [
            hinv_y
            - hinv_covar @ beta_cov_path[index]
            - hinv_geno @ beta_snp_path[index]
            for index in range(beta_snp_path.shape[0])
        ]
    )
    np.testing.assert_allclose(batched, expected, rtol=1e-6, atol=1e-6)
    assert batched.dtype == np.float32


def test_hinv_column_pcg_batches_bound_rhs_and_match_exact_solution():
    rhs = np.arange(35, dtype=np.float32).reshape(7, 5) + 1.0

    solution, diagnostic = SPARSE._solve_hinv_columns_batched(
        hv=lambda value: 2.0 * value,
        precond=lambda value: 0.5 * value,
        rhs=rhs,
        warm_start=np.zeros_like(rhs),
        tol=1e-6,
        maxiter=10,
        batch_size=2,
        stage="unit test",
    )

    np.testing.assert_allclose(solution, rhs / 2.0, rtol=1e-6, atol=1e-6)
    assert diagnostic["n_batches"] == 3
    assert diagnostic["n_columns"] == 5
    assert diagnostic["batch_size"] == 2
    assert diagnostic["max_iterations"] <= 2


def test_hinv_warm_hit_checks_the_true_residual_only_once_per_batch():
    rhs = np.arange(35, dtype=np.float32).reshape(7, 5) + 1.0
    calls = []

    def hv(value):
        calls.append(value.shape[1])
        return 2.0 * value

    solution, diagnostic = SPARSE._solve_hinv_columns_batched(
        hv=hv, precond=None, rhs=rhs, warm_start=rhs / 2.0,
        tol=1e-6, maxiter=10, batch_size=2, stage="warm-hit test",
    )
    np.testing.assert_array_equal(solution, rhs / 2.0)
    assert calls == [2, 2, 1]
    assert diagnostic["total_batch_iterations"] == 0
    assert diagnostic["max_true_relative_residual"] == 0.0




def test_complete_path_global_kkt_unions_violators_before_validation():
    candidate = np.asarray([0, 2], dtype=np.int64)
    beta_path = np.asarray(
        [[0.0, 0.0], [0.5, 0.0], [1.0, -0.25]]
    )
    path_rows = [
        {
            "lam": 10.0,
            "lam_ratio": 1.0,
            "k": 0,
            "converged": True,
            "kkt_passed": True,
        },
        {
            "lam": 5.0,
            "lam_ratio": 0.5,
            "k": 1,
            "converged": True,
            "kkt_passed": True,
        },
        {
            "lam": 1.0,
            "lam_ratio": 0.1,
            "k": 2,
            "converged": True,
            "kkt_passed": True,
        },
    ]
    # Rows are markers, columns are lambda path points.  Marker 3 violates
    # only lambda=5 and marker 4 violates only lambda=1.
    score_path = np.asarray(
        [
            [10.0, 5.0, 1.0],
            [9.0, 4.0, 0.9],
            [0.0, 0.0, -1.0],
            [8.0, 6.0, 0.5],
            [7.0, 4.5, 1.5],
        ]
    )

    result = SPARSE._certify_complete_lasso_path_kkt_from_scores(
        score_path=score_path,
        candidate=candidate,
        beta_candidate_path=beta_path,
        path_rows=path_rows,
        abs_tol=0.0,
        rel_tol=0.0,
    )

    assert result["passed"] is False
    assert result["n_path_points"] == 3
    assert result["n_violating_path_points"] == 2
    np.testing.assert_array_equal(result["outside_violators"], [3, 4])
    assert result["path_rows"][0]["global_kkt_passed"] is True
    assert result["path_rows"][1]["global_kkt_passed"] is False
    assert result["path_rows"][2]["global_kkt_passed"] is False
    assert result["priority_score"][3] == pytest.approx(6.0 / 5.0)
    assert result["priority_score"][4] == pytest.approx(1.5)


def test_basil_score_batch_allows_strong_rule_to_fill_working_set():
    selected = SPARSE._top_scored_markers_outside(
        score_abs=np.asarray([3.0, 2.0, 1.0]),
        excluded=np.asarray([0], dtype=np.int64),
        count=0,
    )
    assert selected.size == 0


def test_weighted_basil_path_matches_full_design_lasso():
    rng = np.random.RandomState(2718)
    n_samples, n_markers = 48, 32
    geno = rng.standard_normal((n_samples, n_markers)).astype(np.float32)
    geno -= geno.mean(axis=0, keepdims=True)
    geno /= geno.std(axis=0, keepdims=True)
    effects = np.zeros(n_markers, dtype=np.float64)
    effects[[2, 11, 23]] = [0.7, -0.5, 0.35]
    outcome = geno @ effects + rng.standard_normal(n_samples) * 0.4

    class _DenseIndex:
        m_total = n_markers

        def extract_standardized_columns(self, indices):
            return geno[:, np.asarray(indices, dtype=np.int64)]

        def xtv_all(self, vector, normalize=False):
            assert normalize is False
            return SPARSE.jnp.asarray(
                geno.T @ np.asarray(vector), dtype=SPARSE.jnp.float32
            )

    cfg = LASSO.LassoPathConfig(
        n_lambda=18,
        lam_min_ratio=0.08,
        max_cd_iter=10000,
        cd_tol=1e-9,
        kkt_abs_tol=1e-6,
        kkt_rel_tol=1e-6,
    )
    args = SimpleNamespace(
        candidate_k=4,
        screen_topk=6,
        kkt_add_topk=4,
        kkt_max_candidate=0,
        kkt_max_rounds=30,
        basil_marker_batch_size=6,
        basil_lambda_block_size=10,
        basil_max_iterations=30,
        candidate_pcg_rhs_batch_size=4,
        pcg_tol=1e-7,
        max_pcg_iters=20,
        lasso_ridge=1e-6,
        kkt_tol=1e-6,
        kkt_rel_tol=1e-6,
    )
    performance = {
        "candidate_hinv_pcg_batches": 0,
        "candidate_hinv_pcg_columns_solved": 0,
        "candidate_hinv_columns_reused": 0,
        "candidate_hinv_pcg_total_batch_iterations": 0,
        "lasso_path_solves": 0,
        "lasso_path_solve_seconds": 0.0,
        "lasso_validation_selection_seconds": 0.0,
        "lasso_cd_iterations": 0,
        "lasso_path_warm_start_rows_used": 0,
        "lasso_path_global_kkt_passes": 0,
        "lasso_path_global_kkt_seconds": 0.0,
        "lasso_path_global_kkt_points_checked": 0,
    }
    hinv_y = outcome / 2.0
    hinv_geno = geno / 2.0
    score = np.abs(geno.T @ hinv_y)
    validation_offset = [0]

    def _monotone_validation(**kwargs):
        n_rows = int(np.asarray(kwargs["beta_snp_path"]).shape[0])
        start = validation_offset[0]
        validation_offset[0] += n_rows
        return {
            "metrics": [
                {
                    "path_index": local,
                    "predictive_r2": float(start + local),
                }
                for local in range(n_rows)
            ],
            "pcg_rel_res": 0.0,
            "pcg_iters": 0,
        }

    basil = SPARSE._fit_complete_weighted_lasso_path_basil(
        args=args,
        grm_index=_DenseIndex(),
        hv=lambda value: 2.0 * value,
        precond=lambda value: 0.5 * value,
        y=outcome,
        covar=None,
        hinv_y=hinv_y,
        hinv_covar=None,
        initial_score_abs=score,
        path_cfg=cfg,
        previous_candidate=np.empty((0,), dtype=np.int64),
        previous_beta_path=None,
        previous_hinv_z={},
        outer=1,
        sparse_path_performance=performance,
        validation_evaluator=_monotone_validation,
        validation_stopping_lag=5,
    )
    direct = LASSO.fit_weighted_lasso_with_covariates(
        y=outcome,
        covar=None,
        geno=geno,
        Hinv_y=hinv_y,
        Hinv_covar=None,
        Hinv_geno=hinv_geno,
        cfg=cfg,
        ridge=args.lasso_ridge,
    )

    expanded = np.zeros_like(direct["beta_snp_path"])
    expanded[:, basil["candidate"]] = basil["lasso_path"]["beta_snp_path"]
    np.testing.assert_allclose(
        expanded, direct["beta_snp_path"], rtol=2e-4, atol=2e-5
    )
    assert basil["global_kkt"]["passed"] is True
    assert all(
        row["global_kkt_passed"]
        for row in basil["lasso_path"]["path"]
    )
    assert basil["candidate"].size < n_markers


def test_validation_selection_rejects_candidate_only_kkt_point():
    path = [
        {
            "k": 4,
            "lam_ratio": 0.2,
            "converged": True,
            "kkt_passed": True,
            "global_kkt_passed": False,
        },
        {
            "k": 2,
            "lam_ratio": 0.5,
            "converged": True,
            "kkt_passed": True,
            "global_kkt_passed": True,
        },
    ]
    metrics = [
        {"predictive_r2": 0.9},
        {"predictive_r2": 0.3},
    ]

    assert SPARSE._select_converged_validation_path_index(path, metrics) == 1


def test_sparse_dense_h2_is_invariant_to_phenotype_rescaling():
    """Input standardization removes arbitrary phenotype-unit changes."""
    rng = np.random.RandomState(314)
    n, s = 120, 7
    z_active = rng.standard_normal((n, s))
    z_active = (z_active - z_active.mean(axis=0)) / z_active.std(axis=0)
    beta = rng.standard_normal(s) * 0.15
    y = z_active @ beta + rng.standard_normal(n) * 0.7

    y_standardized, _, y_scale = SPARSE._standardize_phenotype_at_input(y)
    beta_standardized = beta / y_scale
    q, _, _ = SPARSE._chive_q_hat_given_active(
        z_active, y_standardized, beta_standardized
    )
    h2 = SPARSE._sparse_dense_h2(q, 0.22, 0.63)

    multiplier = 9.0
    y_rescaled, _, y_rescaled_scale = SPARSE._standardize_phenotype_at_input(
        multiplier * y
    )
    q_rescaled, _, _ = SPARSE._chive_q_hat_given_active(
        z_active,
        y_rescaled,
        multiplier * beta / y_rescaled_scale,
    )
    h2_rescaled = SPARSE._sparse_dense_h2(q_rescaled, 0.22, 0.63)

    np.testing.assert_allclose(y_standardized, y_rescaled, rtol=2e-6, atol=2e-6)
    assert np.isclose(q, q_rescaled, rtol=2e-6, atol=2e-6)
    assert np.isclose(h2, h2_rescaled, rtol=2e-6, atol=2e-6)


def test_chive_is_squared_fitted_mean_plus_residual_correction():
    rng = np.random.RandomState(1617)
    z_active = rng.standard_normal((100, 5))
    beta = rng.standard_normal(5) * 0.2
    y = z_active @ beta + rng.standard_normal(100)

    q_chive, q_squared_mean, correction = SPARSE._chive_q_hat_given_active(
        z_active,
        y,
        beta,
    )

    expected_squared_mean = float(np.mean(np.square(z_active @ beta)))
    assert np.isclose(q_squared_mean, expected_squared_mean)
    assert np.isclose(q_chive, q_squared_mean + correction)


def test_coherit_guard_accepts_only_a_certified_finite_estimator():
    guards = SPARSE._coherit_estimator_guard(
        alpha_theta_pair_certified=True,
        lasso_quadratics_available=True,
        h2_chive=0.3,
    )
    assert guards["lasso_branch_valid"] is True
    assert guards["lasso_outputs_finite"] is True
    assert guards["lasso_branch_invalid_reasons"] == []

    uncertified = SPARSE._coherit_estimator_guard(
        alpha_theta_pair_certified=False,
        lasso_quadratics_available=True,
        h2_chive=0.3,
    )
    assert uncertified["lasso_branch_valid"] is False
    assert "penalized_alpha_theta_pair_not_certified" in uncertified[
        "lasso_branch_invalid_reasons"
    ]

    nonfinite = SPARSE._coherit_estimator_guard(
        alpha_theta_pair_certified=True,
        lasso_quadratics_available=True,
        h2_chive=np.nan,
    )
    assert nonfinite["lasso_branch_valid"] is False
    assert "nonfinite_lasso_estimator" in nonfinite[
        "lasso_branch_invalid_reasons"
    ]


def test_sparse_output_contract_is_coherit_only():
    contract = SPARSE._sparse_output_contract()

    assert contract["sparse_output_schema_version"] == 11
    assert contract["estimator_mode"] == "coherit"
    assert contract["computed_estimators"] == ["h2_chive"]
    assert contract["selected_snp_columns"][-1] == "beta_lasso"


def test_fitted_mean_convergence_allows_equivalent_support_swaps():
    marker = np.asarray([-2.0, -1.0, 0.5, 1.0, 1.5])
    genotype = np.column_stack([marker, marker])
    previous_beta = np.asarray([0.4, 0.0])
    swapped_beta = np.asarray([0.0, 0.4])
    previous_mean = genotype @ previous_beta
    swapped_mean = genotype @ swapped_beta
    phenotype = previous_mean + np.asarray([0.1, -0.2, 0.0, 0.2, -0.1])

    swapped_change = SPARSE._relative_fitted_mean_change(
        swapped_mean,
        previous_mean,
        phenotype,
    )
    assert swapped_change == 0.0

    changed_mean = 2.0 * previous_mean
    changed_ratio = SPARSE._relative_fitted_mean_change(
        changed_mean,
        previous_mean,
        phenotype,
    )
    assert changed_ratio > 1e-2


def test_sparse_pipeline_defaults_to_iterative_validation_contract(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["gpu-reml-sparse"])
    args = SPARSE.parse_args()
    assert args.lasso_fixed_lam_ratio is None
    assert args.minq_iter == 50
    assert not hasattr(args, "kkt_check")
    assert not hasattr(args, "lasso_selection_mode")

    monkeypatch.setattr(sys, "argv", ["gpu-reml-sparse", "--kkt-check"])
    with np.testing.assert_raises(SystemExit):
        SPARSE.parse_args()

    monkeypatch.setattr(
        sys, "argv", ["gpu-reml-sparse", "--no-kkt-check"]
    )
    with np.testing.assert_raises(SystemExit):
        SPARSE.parse_args()


def test_fixed_lambda_ratio_canonicalizes_lambda_max_endpoint():
    assert SPARSE._canonical_fixed_lam_ratio(1.0) == 1.0
    assert SPARSE._canonical_fixed_lam_ratio(0.9999999999999998) == 1.0
    assert SPARSE._canonical_fixed_lam_ratio(1.0000000000000002) == 1.0
    assert SPARSE._canonical_fixed_lam_ratio(0.999999) == pytest.approx(0.999999)


@pytest.mark.parametrize(
    "minimum, selected",
    [
        (0.1, 0.09999999999999999),
        (0.1, 0.1),
        (0.1, np.nextafter(0.1, np.inf)),
        (0.1, 0.25),
        (1e-14, np.nextafter(1e-14, 0.0)),
    ],
)
def test_fixed_lambda_ratio_cli_accepts_lower_endpoint_roundoff(
    monkeypatch, minimum, selected
):
    monkeypatch.setattr(
        sys, "argv",
        [
            "gpu-reml-sparse",
            "--lasso-lam-min-ratio", str(minimum),
            "--lasso-fixed-lam-ratio", str(selected),
        ],
    )
    args = SPARSE.parse_args()
    monkeypatch.setattr(SPARSE, "parse_args", lambda: args)

    # Reaching the warm-start requirement proves the ratio passed validation,
    # without loading genotype data or running a fit.
    with pytest.raises(SystemExit, match="requires both its validation-selected"):
        SPARSE.main()
    assert args.lasso_fixed_lam_ratio == selected


@pytest.mark.parametrize(
    "minimum, selected",
    [(0.1, 0.099999999), (1e-14, 0.9e-14)],
)
def test_fixed_lambda_ratio_cli_rejects_values_below_path(
    monkeypatch, minimum, selected
):
    monkeypatch.setattr(
        sys, "argv",
        [
            "gpu-reml-sparse",
            "--lasso-lam-min-ratio", str(minimum),
            "--lasso-fixed-lam-ratio", str(selected),
        ],
    )
    with pytest.raises(SystemExit, match="must be at least --lasso-lam-min-ratio"):
        SPARSE.main()


def test_sparse_pipeline_has_no_max_active_cap(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["gpu-reml-sparse"])
    args = SPARSE.parse_args()
    assert not hasattr(args, "max_active")

    monkeypatch.setattr(
        sys,
        "argv",
        ["gpu-reml-sparse", "--max-active", "128"],
    )
    with np.testing.assert_raises(SystemExit):
        SPARSE.parse_args()


def test_lasso_candidate_keeps_complete_previous_support_above_seed_target():
    candidate = SPARSE._build_lasso_candidate(
        previous_support=np.asarray([9, 2, 7, 2]),
        screened_indices=np.asarray([7, 1, 3, 4]),
        candidate_target=2,
    )
    np.testing.assert_array_equal(candidate, np.asarray([2, 7, 9]))

    filled = SPARSE._build_lasso_candidate(
        previous_support=np.asarray([9, 2, 7, 2]),
        screened_indices=np.asarray([7, 1, 3, 4]),
        candidate_target=5,
    )
    np.testing.assert_array_equal(filled, np.asarray([1, 2, 3, 7, 9]))


def test_lasso_candidate_reuses_certified_set_and_unions_fresh_screen():
    candidate = SPARSE._build_lasso_candidate(
        previous_support=np.asarray([4, 7]),
        previous_candidate=np.asarray([1, 4, 9]),
        screened_indices=np.asarray([2, 4, 6]),
        candidate_target=2,
    )
    np.testing.assert_array_equal(candidate, np.asarray([1, 2, 4, 7, 9]))


def test_lasso_path_warm_start_remaps_global_marker_basis():
    previous_candidate = np.asarray([2, 7, 9])
    previous_path = np.asarray(
        [[0.0, 0.0, 0.0], [0.2, -0.7, 0.9]], dtype=np.float64
    )
    mapped, n_common = SPARSE._remap_lasso_beta_path(
        previous_candidate=previous_candidate,
        previous_beta_path=previous_path,
        candidate=np.asarray([1, 2, 7, 10]),
    )

    assert n_common == 2
    np.testing.assert_array_equal(
        mapped,
        np.asarray(
            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.2, -0.7, 0.0]]
        ),
    )


def test_buffered_kkt_expansion_adds_strict_and_near_threshold_markers():
    expansion = SPARSE._buffered_kkt_expansion_indices(
        score_abs=np.arange(20, dtype=np.float64),
        candidate=np.asarray([18, 19]),
        violators=np.asarray([16, 17]),
        max_add=16,
    )

    np.testing.assert_array_equal(
        expansion["add_indices"], np.asarray([14, 15, 16, 17])
    )
    assert expansion["n_strict_added"] == 2
    assert expansion["n_buffered_added"] == 2


def test_buffered_kkt_expansion_prioritizes_strict_violators_at_budget():
    expansion = SPARSE._buffered_kkt_expansion_indices(
        score_abs=np.asarray([0.0, 0.0, 0.0, 3.0, 0.0, 5.0, 0.0, 7.0]),
        candidate=np.asarray([0, 1]),
        violators=np.asarray([3, 5, 7]),
        max_add=2,
    )

    np.testing.assert_array_equal(
        expansion["add_indices"], np.asarray([5, 7])
    )
    assert expansion["n_strict_added"] == 2
    assert expansion["n_buffered_added"] == 0


def test_kkt_expansion_budget_absorbs_only_small_overflow():
    assert SPARSE._kkt_expansion_budget(267, 256) == 267
    assert SPARSE._kkt_expansion_budget(281, 256) == 281
    assert SPARSE._kkt_expansion_budget(282, 256) == 256
    assert SPARSE._kkt_expansion_budget(3, 2) == 2


def test_lasso_path_warm_start_is_reused_for_any_basis_overlap():
    assert SPARSE._allow_mapped_lasso_path_warm_start(
        previous_size=2015,
        current_size=2031,
        common_size=2015,
    )
    assert SPARSE._allow_mapped_lasso_path_warm_start(
        previous_size=1759,
        current_size=2015,
        common_size=1759,
    )
    assert SPARSE._allow_mapped_lasso_path_warm_start(
        previous_size=2015,
        current_size=2031,
        common_size=2000,
    )
    assert SPARSE._allow_mapped_lasso_path_warm_start(
        previous_size=2031,
        current_size=2015,
        common_size=2015,
    )
    assert not SPARSE._allow_mapped_lasso_path_warm_start(
        previous_size=2031,
        current_size=2015,
        common_size=0,
    )


def test_candidate_signed_kkt_distinguishes_inside_and_outside_failures():
    signed_pass = SPARSE._candidate_lasso_kkt_from_scores(
        score=np.asarray([0.5, -0.5, 0.49]),
        candidate=np.asarray([0, 1]),
        beta_candidate=np.asarray([0.2, -0.1]),
        lam=0.5,
        abs_tol=1e-8,
        rel_tol=0.0,
    )
    assert signed_pass["candidate_certificate"]["passed"] is True
    assert signed_pass["full_certificate"]["passed"] is True
    assert signed_pass["outside_violators"].size == 0

    inside_failure = SPARSE._candidate_lasso_kkt_from_scores(
        score=np.asarray([0.45, 0.49, 0.49]),
        candidate=np.asarray([0, 1]),
        beta_candidate=np.asarray([0.2, 0.0]),
        lam=0.5,
        abs_tol=1e-8,
        rel_tol=0.0,
    )
    assert inside_failure["candidate_certificate"]["passed"] is False
    assert inside_failure["full_certificate"]["passed"] is False
    assert inside_failure["outside_violators"].size == 0

    outside_failure = SPARSE._candidate_lasso_kkt_from_scores(
        score=np.asarray([0.5, 0.49, -0.51]),
        candidate=np.asarray([0, 1]),
        beta_candidate=np.asarray([0.2, 0.0]),
        lam=0.5,
        abs_tol=1e-8,
        rel_tol=0.0,
    )
    assert outside_failure["candidate_certificate"]["passed"] is True
    assert outside_failure["full_certificate"]["passed"] is False
    np.testing.assert_array_equal(
        outside_failure["outside_violators"], np.asarray([2])
    )


def test_outer_convergence_uses_absolute_coherit_h2_change():
    converged, change = SPARSE._heritability_converged(
        0.701,
        None,
        abs_tol=1e-2,
    )
    assert converged is False
    assert np.isinf(change)

    converged, change = SPARSE._heritability_converged(
        0.710,
        0.700,
        abs_tol=1e-2,
    )
    assert converged is True
    assert np.isclose(change, 1e-2)

    converged, change = SPARSE._heritability_converged(
        0.710001,
        0.700,
        abs_tol=1e-2,
    )
    assert converged is False
    assert change > 1e-2


@pytest.mark.parametrize(
    "provisional,h2_stable,effect_stable,outer,expected",
    [
        (True, True, True, 3, "converged"),
        (True, True, True, 20, "converged"),
        (True, False, True, 3, "continue"),
        (True, True, False, 3, "continue"),
        (True, False, True, 20, "outer_max"),
        (False, True, True, 20, "outer_max"),
    ],
)
def test_alignment_must_verify_the_returned_state(
    provisional, h2_stable, effect_stable, outer, expected
):
    assert SPARSE._alignment_action(
        provisional_convergence=provisional, h2_stable=h2_stable,
        effect_stable=effect_stable, outer=outer, outer_max=20,
    ) == expected


def test_outer_coherit_h2_matches_final_chive_functional():
    rng = np.random.RandomState(941)
    n_samples, n_markers = 80, 5
    genotype = rng.standard_normal((n_samples, n_markers))
    genotype -= genotype.mean(axis=0)
    beta = rng.standard_normal(n_markers) * 0.1
    sparse_mean = genotype @ beta
    residual = rng.standard_normal(n_samples) * 0.7
    residual -= residual.mean()
    phenotype = sparse_mean + residual
    phenotype, _, phenotype_standard_deviation = (
        SPARSE._standardize_phenotype_at_input(
            phenotype
        )
    )
    sparse_mean = sparse_mean / phenotype_standard_deviation
    residual = residual / phenotype_standard_deviation
    beta = beta / phenotype_standard_deviation

    expected_q, _, _ = SPARSE._chive_q_hat_given_active(
        genotype,
        phenotype,
        beta,
    )
    expected_h2 = SPARSE._sparse_dense_h2(expected_q, 0.2, 0.3)
    observed_h2, observed_q = (
        SPARSE._outer_coherit_h2_from_fitted_sparse_mean(
            sparse_mean,
            residual,
            background_genetic_variance=0.2,
            residual_variance=0.3,
        )
    )

    assert np.isclose(observed_q, expected_q)
    assert np.isclose(observed_h2, expected_h2)


def test_candidate_kkt_rejects_nonfinite_outside_score_with_empty_support():
    with np.testing.assert_raises_regex(
        ValueError, "Full-p KKT inputs must be finite"
    ):
        SPARSE._candidate_lasso_kkt_from_scores(
            score=np.asarray([np.nan]),
            candidate=np.empty((0,), dtype=np.int64),
            beta_candidate=np.empty((0,), dtype=np.float64),
            lam=0.5,
            abs_tol=1e-8,
            rel_tol=0.0,
        )


def test_sparse_defaults_use_twenty_outer_rounds_and_pcg_scaled_kkt_floor(
    monkeypatch,
):
    monkeypatch.delenv("PCG_TOL", raising=False)

    monkeypatch.setattr(sys, "argv", ["gpu-reml-sparse"])
    default_args = SPARSE.parse_args()
    default_floor = max(1e-4, 2.0 * default_args.pcg_tol)
    assert default_args.outer_max == 20
    assert np.isclose(default_args.h2_abs_tol, 1e-2)
    assert np.isclose(default_args.effect_rel_tol, 5e-2)
    assert np.isclose(default_args.kkt_tol, default_floor)
    assert np.isclose(default_args.kkt_rel_tol, default_floor)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gpu-reml-sparse",
            "--pcg-tol",
            "2e-3",
            "--kkt-tol",
            "1e-8",
            "--kkt-rel-tol",
            "3e-3",
        ],
    )
    floored_args = SPARSE.parse_args()
    requested_floor = max(1e-4, 2.0 * floored_args.pcg_tol)
    assert np.isclose(floored_args.kkt_tol, requested_floor)
    assert np.isclose(floored_args.kkt_rel_tol, requested_floor)


def test_pcg_returns_the_true_relative_residual_of_its_solution():
    matrix = SPARSE.jnp.asarray(
        [[2.0, 0.0], [0.0, 4.0]], dtype=SPARSE.jnp.float32
    )
    rhs = SPARSE.jnp.asarray(
        [[2.0, 1.0], [4.0, -2.0]], dtype=SPARSE.jnp.float32
    )
    exact = SPARSE.jnp.asarray(
        [[1.0, 0.5], [1.0, -0.5]], dtype=SPARSE.jnp.float32
    )
    hv = lambda value: matrix @ value

    _, residual, _ = SPARSE.pcg_solve(hv, rhs, X0=exact, maxiter=0)
    assert float(residual) == 0.0

    perturbed = exact.at[0, 0].add(0.1)
    _, observed, _ = SPARSE.pcg_solve(hv, rhs, X0=perturbed, maxiter=0)
    rhs_np = np.asarray(rhs)
    perturbed_np = np.asarray(perturbed)
    expected = np.max(
        np.linalg.norm(rhs_np - np.asarray(matrix) @ perturbed_np, axis=0)
        / (np.linalg.norm(rhs_np, axis=0) + 1e-12)
    )
    assert np.isclose(observed, expected)


def test_sparse_dense_h2_rejects_nonfinite_or_nonpositive_denominator():
    assert np.isnan(SPARSE._sparse_dense_h2(np.nan, 0.2, 0.8))
    assert np.isnan(SPARSE._sparse_dense_h2(-1.0, 0.2, 0.8))
    assert np.isnan(SPARSE._sparse_dense_h2(np.inf, 0.2, 0.8))


def test_json_safe_value_replaces_nested_nonfinite_diagnostics():
    value = {
        "finite": np.float64(0.4),
        "nested": [np.nan, np.float32(np.inf), True],
    }
    assert SPARSE._json_safe_value(value) == {
        "finite": 0.4,
        "nested": [None, None, True],
    }


def test_reml_state_validation_accepts_only_coherent_converged_states():
    accepted = SimpleNamespace(
        var_components=np.asarray([0.3, 0.7]),
        history=[
            {
                "accepted": True,
                "converged": True,
                "stop_reason": "rel_dll",
            }
        ],
    )
    theta, reason = SPARSE._accepted_reml_theta(
        accepted,
        expected_components=2,
        stage="test",
    )
    assert np.array_equal(theta, np.asarray([0.3, 0.7]))
    assert reason == "rel_dll"

    rejected_step = SimpleNamespace(
        var_components=np.asarray([0.3, 0.7]),
        history=[
            {
                "accepted": False,
                "converged": True,
                "stop_reason": "ll_down",
            }
        ],
    )
    theta, reason = SPARSE._accepted_reml_theta(
        rejected_step,
        expected_components=2,
        stage="test",
    )
    assert np.array_equal(theta, np.asarray([0.3, 0.7]))
    assert reason == "ll_down"

    for rejected in (
        SimpleNamespace(
            var_components=np.asarray([0.3, 0.7]),
            history=[],
        ),
        SimpleNamespace(
            var_components=np.asarray([0.3, 0.7]),
            history=[
                {
                    "accepted": True,
                    "converged": False,
                    "stop_reason": "max_iter",
                }
            ],
        ),
        SimpleNamespace(
            var_components=np.asarray([np.nan, 0.7]),
            history=[
                {
                    "accepted": True,
                    "converged": True,
                    "stop_reason": "rel_dll",
                }
            ],
        ),
    ):
        with np.testing.assert_raises(RuntimeError):
            SPARSE._accepted_reml_theta(
                rejected,
                expected_components=2,
                stage="test",
            )


def test_full_score_kkt_certificate_checks_active_and_inactive_coordinates():
    passed = SPARSE._lasso_kkt_certificate_from_scores(
        score=np.asarray([0.5, -0.5, 0.49]),
        beta=np.asarray([0.2, -0.1, 0.0]),
        lam=0.5,
        abs_tol=1e-8,
        rel_tol=0.0,
    )
    assert passed["passed"] is True

    active_failure = SPARSE._lasso_kkt_certificate_from_scores(
        score=np.asarray([0.45, -0.5, 0.49]),
        beta=np.asarray([0.2, -0.1, 0.0]),
        lam=0.5,
        abs_tol=1e-8,
        rel_tol=0.0,
    )
    assert active_failure["passed"] is False
    assert active_failure["max_active_error"] > 0.0

    inactive_failure = SPARSE._lasso_kkt_certificate_from_scores(
        score=np.asarray([0.5, -0.5, 0.51]),
        beta=np.asarray([0.2, -0.1, 0.0]),
        lam=0.5,
        abs_tol=1e-8,
        rel_tol=0.0,
    )
    assert inactive_failure["passed"] is False
    assert inactive_failure["max_inactive_excess"] > 0.0


def test_gram_cd_uses_score_kkt_not_coefficient_delta_to_stop():
    beta, gram_beta, n_iter, converged = LASSO.solve_lasso_cd_gram(
        Q=np.eye(2),
        q=np.asarray([2.0, -1.0]),
        lam=0.5,
        max_iter=10,
        tol=1e-12,
        active_set_period=5,
        kkt_abs_tol=1e-12,
        kkt_rel_tol=0.0,
    )

    # The first sweep moves beta by 1.5 (> tol), yet its exact score already
    # satisfies KKT and is therefore the convex optimum.
    assert converged
    assert n_iter == 1
    np.testing.assert_allclose(beta, np.asarray([1.5, -0.5]))
    np.testing.assert_allclose(
        np.asarray([2.0, -1.0]) - gram_beta,
        0.5 * np.sign(beta),
    )


def test_gram_cd_final_active_sweep_receives_exact_kkt_certificate():
    gram = np.asarray([[1.0, 0.5], [0.5, 1.0]])
    linear = np.asarray([2.0, 2.0])
    beta, gram_beta, n_iter, converged = LASSO.solve_lasso_cd_gram(
        gram,
        linear,
        lam=1.0,
        max_iter=2,
        tol=1e-12,
        active_set_period=5,
        kkt_abs_tol=0.1,
        kkt_rel_tol=0.0,
    )

    # Iteration two is active-only.  The unconditional return-time score check
    # must recognize its valid KKT point even though no periodic full sweep is
    # left in the iteration budget.
    assert n_iter == 2
    assert converged
    score = linear - gram_beta
    np.testing.assert_array_less(
        np.abs(score - np.sign(beta)),
        np.full(2, 0.1 + 1e-12),
    )


def test_gram_cd_kkt_certificate_on_scaled_high_ld_gram():
    k = 128
    lam = 550.0
    scale = 8.0e4
    correlation = 0.2
    gram = scale * (
        (1.0 - correlation) * np.eye(k)
        + correlation * np.ones((k, k))
    )
    linear = np.random.default_rng(7).normal(0.0, 700.0, k)
    abs_tol = 1e-4
    rel_tol = 1e-4
    tolerance = max(abs_tol, rel_tol * lam)

    beta, gram_beta, _, converged = LASSO.solve_lasso_cd_gram(
        gram,
        linear,
        lam,
        max_iter=5000,
        tol=1e-6,
        active_set_period=5,
        kkt_abs_tol=abs_tol,
        kkt_rel_tol=rel_tol,
    )

    score = linear - gram_beta
    active = beta != 0.0
    active_error = (
        float(
            np.max(
                np.abs(score[active] - lam * np.sign(beta[active]))
            )
        )
        if np.any(active)
        else 0.0
    )
    inactive_excess = (
        float(max(np.max(np.abs(score[~active])) - lam, 0.0))
        if np.any(~active)
        else 0.0
    )
    assert converged
    assert active_error <= tolerance
    assert inactive_excess <= tolerance


def test_complete_path_failure_does_not_fall_back_to_valid_null(monkeypatch):
    def fake_lambda_sequence(_lam_max, _lam_min_ratio, _n_lambda):
        return np.asarray([1.0, 0.1])

    def fake_cd(_gram, _linear, lam, **_kwargs):
        if np.isclose(lam, 1.0):
            return np.asarray([0.0]), np.asarray([0.0]), 1, True
        # Artificially attractive partial iterate: zero RSS but failed KKT.
        return np.asarray([1.0]), np.asarray([1.0]), 2, False

    monkeypatch.setattr(LASSO, "make_lambda_sequence", fake_lambda_sequence)
    monkeypatch.setattr(LASSO, "solve_lasso_cd_gram", fake_cd)

    with np.testing.assert_raises_regex(
        RuntimeError,
        "Lasso path failed score-KKT convergence",
    ):
        LASSO.solve_lasso_path(
            Q=np.asarray([[1.0]]),
            q=np.asarray([1.0]),
            yHy=1.0,
            cfg=LASSO.LassoPathConfig(
                n_lambda=2,
                kkt_abs_tol=1e-8,
                kkt_rel_tol=0.0,
            ),
        )


def test_covariate_contrast_reml_passes_full_design_without_rescaling():
    marker = SimpleNamespace(
        var_components=REML.jnp.asarray(
            [0.25, 0.75], dtype=REML.jnp.float32
        ),
        rep_var_components=REML.jnp.asarray(
            [[0.2, 0.8], [0.3, 0.7]], dtype=REML.jnp.float32
        ),
        monte_carlo_se_var=REML.jnp.asarray(
            [0.01, 0.02], dtype=REML.jnp.float32
        ),
        final_grad=REML.jnp.asarray([2.0, 4.0], dtype=REML.jnp.float32),
        final_ai=REML.jnp.asarray(
            [[3.0, 0.5], [0.5, 5.0]], dtype=REML.jnp.float32
        ),
        diagnostics={
            "theta": REML.jnp.asarray([0.25, 0.75]),
            "grad": REML.jnp.asarray([2.0, 4.0]),
            "ai": REML.jnp.asarray([[3.0, 0.5], [0.5, 5.0]]),
        },
        history=[{"params": [0.25, 0.75], "step_norm": 0.1, "grad_norm": 2.0}],
    )

    class RecordingFitter:
        def fit_infinitesimal(self, y, covar, **kwargs):
            self.y = np.asarray(y)
            self.covar = np.asarray(covar)
            self.kwargs = kwargs
            return marker

    fitter = RecordingFitter()
    residual = np.linspace(-1.0, 1.0, 19, dtype=np.float32)
    nuisance = np.column_stack(
        [
            np.ones(residual.size, dtype=np.float32),
            np.linspace(-2.0, 2.0, residual.size, dtype=np.float32),
            np.sin(np.linspace(0.0, 2.0, residual.size, dtype=np.float32)),
        ]
    ).astype(np.float32)
    theta = np.asarray([0.31, 0.69], dtype=np.float32)
    result = SPARSE._fit_covariate_contrast_residual_reml(
        fitter,
        residual,
        theta,
        covar=nuisance,
        h2_init=0.31,
    )

    assert result is marker
    assert np.array_equal(fitter.y, residual)
    assert np.array_equal(fitter.covar, nuisance)
    assert "standardize_y" not in fitter.kwargs
    assert np.isclose(fitter.kwargs["h2_init"], 0.31)
    assert np.array_equal(
        np.asarray(fitter.kwargs["var_components_init"]),
        theta,
    )
    assert result.var_components is marker.var_components
    assert result.rep_var_components is marker.rep_var_components
    assert result.monte_carlo_se_var is marker.monte_carlo_se_var
    assert result.final_grad is marker.final_grad
    assert result.final_ai is marker.final_ai
    assert result.diagnostics is marker.diagnostics
    assert result.history is marker.history


def test_covariate_contrast_reml_intercept_fallback_matches_legacy_design():
    class RecordingFitter:
        def fit_infinitesimal(self, y, covar, **kwargs):
            self.y = np.asarray(y)
            self.covar = np.asarray(covar)
            self.kwargs = kwargs
            return SimpleNamespace(
                var_components=REML.jnp.asarray([0.4, 0.6]),
                rep_var_components=None,
                monte_carlo_se_var=None,
                final_grad=None,
                final_ai=None,
                diagnostics=None,
                history=[],
            )

    residual = np.linspace(-0.8, 1.2, 23, dtype=np.float32)
    theta = np.asarray([0.37, 0.63], dtype=np.float32)
    explicit = RecordingFitter()
    fallback = RecordingFitter()

    SPARSE._fit_covariate_contrast_residual_reml(
        explicit,
        residual,
        theta,
        covar=np.ones((residual.size, 1), dtype=np.float32),
        h2_init=0.37,
    )
    SPARSE._fit_covariate_contrast_residual_reml(
        fallback,
        residual,
        theta,
        covar=None,
        h2_init=0.37,
    )

    assert np.array_equal(explicit.y, fallback.y)
    assert np.array_equal(explicit.covar, fallback.covar)
    assert explicit.kwargs.keys() == fallback.kwargs.keys()
    assert explicit.kwargs["h2_init"] == fallback.kwargs["h2_init"]
    assert np.array_equal(
        np.asarray(explicit.kwargs["var_components_init"]),
        np.asarray(fallback.kwargs["var_components_init"]),
    )


def test_covariate_contrast_quadratic_is_invariant_to_nuisance_shift():
    rng = np.random.RandomState(90210)
    n = 41
    nuisance = np.column_stack(
        [np.ones(n), rng.standard_normal(n), rng.standard_normal(n)]
    )
    a = rng.standard_normal((n, n))
    covariance = a @ a.T / n + 0.4 * np.eye(n)
    precision = np.linalg.inv(covariance)
    gram = nuisance.T @ precision @ nuisance
    projector = (
        precision
        - precision
        @ nuisance
        @ np.linalg.solve(gram, nuisance.T @ precision)
    )
    response = rng.standard_normal(n)
    shifted = response + nuisance @ rng.standard_normal(nuisance.shape[1])

    assert np.allclose(projector @ nuisance, 0.0, atol=2e-12)
    assert np.isclose(
        response @ projector @ response,
        shifted @ projector @ shifted,
        rtol=1e-12,
        atol=1e-11,
    )


def test_covariate_contrast_reml_core_is_invariant_to_nuisance_shift():
    """Exercise the real REML projection and the helper's scale mapping."""
    rng = np.random.RandomState(427)
    n_samples, n_markers = 36, 9
    geno = rng.standard_normal((n_samples, n_markers))
    geno = (geno - geno.mean(axis=0)) / geno.std(axis=0)
    geno_jax = REML.jnp.asarray(geno, dtype=REML.jnp.float32)
    diagonal = REML.jnp.asarray(
        np.mean(np.square(geno), axis=1), dtype=REML.jnp.float32
    )
    nuisance = np.column_stack(
        [np.ones(n_samples), np.linspace(-1.0, 1.0, n_samples)]
    ).astype(np.float32)
    residual = rng.standard_normal(n_samples).astype(np.float32)
    shifted = (
        residual
        + nuisance @ np.asarray([2.3, -1.7], dtype=np.float32)
    ).astype(np.float32)
    theta = np.asarray([0.27, 0.73], dtype=np.float32)

    def genetic_mv(value):
        return geno_jax @ (geno_jax.T @ value) / float(n_markers)

    class DirectREMLFitter:
        def fit_infinitesimal(
            self, response, covar, *, h2_init, var_components_init
        ):
            params, history, diagnostics = REML.fit_reml(
                y=response,
                K_mvs=[genetic_mv],
                diag_list=[diagonal],
                covar=covar,
                n_rand_vec=32,
                maxiter=100,
                seed=123,
                h2_init=h2_init,
                param_init=var_components_init,
                minq_iter=0,
                slq_samples=32,
                slq_m=24,
                precond_conf=None,
                pcg_tol=1e-7,
                response_is_standardized=True,
                unit_variance_components=True,
                return_diagnostics=True,
                verbose=False,
            )
            return SimpleNamespace(
                var_components=params,
                rep_var_components=None,
                monte_carlo_se_var=None,
                final_grad=diagnostics["grad"],
                final_ai=diagnostics["ai"],
                diagnostics=diagnostics,
                history=history,
            )

    fit = SPARSE._fit_covariate_contrast_residual_reml(
        DirectREMLFitter(),
        residual,
        theta,
        covar=nuisance,
        h2_init=0.27,
    )
    shifted_fit = SPARSE._fit_covariate_contrast_residual_reml(
        DirectREMLFitter(),
        shifted,
        theta,
        covar=nuisance,
        h2_init=0.27,
    )

    np.testing.assert_allclose(
        np.asarray(fit.var_components),
        np.asarray(shifted_fit.var_components),
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        np.asarray(fit.final_grad),
        np.asarray(shifted_fit.final_grad),
        rtol=2e-5,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        np.asarray(fit.final_ai),
        np.asarray(shifted_fit.final_ai),
        rtol=2e-5,
        atol=2e-6,
    )


def test_empty_support_reduces_to_background_only_heritability():
    background, residual = 0.28, 0.62
    expected = background / (background + residual)
    h2 = SPARSE._sparse_dense_h2(0.0, background, residual)

    assert np.isclose(h2, expected)


def test_prediction_emitter_uses_the_already_fitted_pair(
    monkeypatch,
):
    calls = {"closed": False}
    prediction = SimpleNamespace(pcg_rel_res=1e-4, pcg_iters=3)

    class _Index:
        def extract_standardized_columns(self, indices):
            np.testing.assert_array_equal(indices, [1, 3])
            return np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    context = SimpleNamespace(
        fitter=object(),
        grm_index=_Index(),
        covar=np.ones((2, 1), dtype=np.float32),
        sample_ids=["i1", "i2"],
        close=lambda: calls.__setitem__("closed", True),
    )
    monkeypatch.setattr(
        SPARSE,
        "predict_sparse_branch",
        lambda **kwargs: calls.setdefault("prediction_kwargs", kwargs)
        and prediction,
    )
    monkeypatch.setattr(
        SPARSE,
        "write_sparse_prediction_status",
        lambda **kwargs: calls.setdefault("status", kwargs),
    )
    monkeypatch.setattr(
        SPARSE,
        "write_sparse_prediction_outputs",
        lambda **kwargs: calls.setdefault("outputs", kwargs)
        and {"prediction": "second.sparse_prediction.tsv"},
    )

    result = SPARSE._emit_lasso_prediction(
        out_prefix="second",
        keep_path="v_cov.keep",
        prediction_context=context,
        prediction_bed_list=["geno"],
        prediction_pgen_prefix="",
        input_phenotype_mean=0.0,
        input_phenotype_standard_deviation=1.0,
        training_fitter=object(),
        y_train=np.asarray([0.1, -0.1]),
        train_covar=np.ones((2, 1)),
        train_support=np.ones((2, 2)),
        support_indices=np.asarray([1, 3]),
        beta_cov=np.asarray([0.2]),
        beta_active=np.asarray([0.3, 0.4]),
        theta=np.asarray([0.5, 0.5]),
        pcg_tol=5e-3,
        max_pcg_iters=400,
    )

    assert calls["closed"] is True
    assert calls["prediction_kwargs"]["theta"].tolist() == [0.5, 0.5]
    assert calls["outputs"]["metadata"]["prediction_keep_path"] == "v_cov.keep"
    assert result["status"] == "emitted"
    assert result["n_samples"] == 2


def test_reml_backtracking_reuses_the_accepted_state_warm_anchor(
    monkeypatch,
):
    """Rejected PCG states must not make the line search path-dependent."""
    jnp = REML.jnp
    calls = {"eval": 0, "candidate_warm_means": []}

    def fake_eval_once(ctx, pvec, warm_all, **_kwargs):
        call_idx = calls["eval"]
        calls["eval"] += 1
        if call_idx > 0:
            calls["candidate_warm_means"].append(
                float(np.asarray(warm_all).mean())
            )
        ll = jnp.asarray(0.0 if call_idx == 0 else -1e-3)
        grad = jnp.asarray([5e-5, 0.0], dtype=jnp.float32)
        warm_token = jnp.full(
            (ctx.n, 1), 10.0 + call_idx, dtype=jnp.float32
        )
        return (
            ll,
            grad,
            REML.AverageInfoMatrix(jnp.eye(2, dtype=jnp.float32), ridge=0.0),
            0,
            warm_token,
            jnp.zeros((ctx.n, 2), dtype=jnp.float32),
            jnp.ones((1,), dtype=jnp.float32),
            jnp.ones((1,), dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
        )

    monkeypatch.setattr(REML, "_eval_once", fake_eval_once)
    monkeypatch.setattr(
        REML,
        "_compute_traces_from_pcg",
        lambda _warm, _ctx: (
            jnp.ones((1,), dtype=jnp.float32),
            jnp.ones((1,), dtype=jnp.float32),
        ),
    )

    theta, history = REML.fit_reml(
        y=jnp.asarray([0.5, -0.1, 1.2, 0.3], dtype=jnp.float32),
        K_mvs=[lambda value: value],
        diag_list=[jnp.ones((4,), dtype=jnp.float32)],
        covar=None,
        n_rand_vec=2,
        maxiter=8,
        minq_iter=1,
        slq_samples=2,
        slq_m=3,
        precond_conf=None,
        param_init=jnp.asarray([0.5, 0.5], dtype=jnp.float32),
        max_linesearch_trials=3,
        scoring_step_tol=1e-4,
        verbose=False,
    )

    assert np.allclose(np.asarray(theta), [0.5, 0.5])
    assert history[-1]["stop_reason"] == "ll_down"
    assert history[-1]["accepted"] is False
    assert history[-1]["converged"] is True
    assert history[-1]["line_search_trials"] == 3
    assert np.allclose(history[-1]["params"], [0.5, 0.5])
    assert history[-1]["loglik"] == history[-1]["loglik_prev"]
    assert calls["candidate_warm_means"] == [10.0, 10.0, 10.0]


def test_strict_reml_stops_on_relative_likelihood_increment_alone(
    monkeypatch,
):
    jnp = REML.jnp
    calls = {"eval": 0}

    def fake_eval_once(ctx, pvec, warm_all, **_kwargs):
        del pvec
        call_idx = calls["eval"]
        calls["eval"] += 1
        ll = jnp.asarray(1.0 if call_idx == 0 else 1.0005)
        return (
            ll,
            jnp.asarray([0.1, 0.1], dtype=jnp.float32),
            REML.AverageInfoMatrix(jnp.eye(2, dtype=jnp.float32), ridge=0.0),
            0,
            warm_all,
            jnp.zeros((ctx.n, 2), dtype=jnp.float32),
            jnp.ones((1,), dtype=jnp.float32),
            jnp.ones((1,), dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
        )

    monkeypatch.setattr(REML, "_eval_once", fake_eval_once)
    monkeypatch.setattr(
        REML,
        "_compute_traces_from_pcg",
        lambda _warm, _ctx: (
            jnp.ones((1,), dtype=jnp.float32),
            jnp.ones((1,), dtype=jnp.float32),
        ),
    )

    _theta, history = REML.fit_reml(
        y=jnp.asarray([0.5, -0.1, 1.2, 0.3], dtype=jnp.float32),
        K_mvs=[lambda value: value],
        diag_list=[jnp.ones((4,), dtype=jnp.float32)],
        covar=None,
        n_rand_vec=2,
        maxiter=8,
        minq_iter=2,
        slq_samples=2,
        slq_m=3,
        precond_conf=None,
        param_init=jnp.asarray([0.5, 0.5], dtype=jnp.float32),
        rel_dll_tol=1e-3,
        verbose=False,
    )

    assert len(history) == 1
    assert history[0]["accepted"] is True
    assert history[0]["rel_dll"] < 1e-3
    assert history[0]["converged"] is True
    assert history[0]["stop_reason"] == "rel_dll"
    assert "proj_grad_inf" not in history[0]
