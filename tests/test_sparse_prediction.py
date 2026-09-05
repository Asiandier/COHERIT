from __future__ import annotations

import importlib
import json
import os
import sys

import jax.numpy as jnp
import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT = os.path.dirname(REPO_ROOT)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

PKG = importlib.import_module(os.path.basename(REPO_ROOT))
REML_MODEL = importlib.import_module(f"{PKG.__name__}.reml_model")
SPARSE_PRED = importlib.import_module(f"{PKG.__name__}.sparse_prediction")
DATA_UTILS = importlib.import_module(f"{PKG.__name__}.data_utils")
RUN_REML = importlib.import_module(f"{PKG.__name__}.run_reml_pipeline")
RUN_SPARSE = importlib.import_module(
    f"{PKG.__name__}.run_sparse_reml_pipeline"
)

FitConfig = REML_MODEL.FitConfig
InfinitesimalREMLFitter = REML_MODEL.InfinitesimalREMLFitter
SparseBranchPrediction = SPARSE_PRED.SparseBranchPrediction
predict_sparse_branch = SPARSE_PRED.predict_sparse_branch
predict_sparse_path_partitioned = SPARSE_PRED.predict_sparse_path_partitioned
predict_sparse_path_single_grm = SPARSE_PRED.predict_sparse_path_single_grm
write_sparse_prediction_outputs = SPARSE_PRED.write_sparse_prediction_outputs
write_sparse_prediction_status = SPARSE_PRED.write_sparse_prediction_status
load_pheno_covar_aligned_with_transform = (
    DATA_UTILS.load_pheno_covar_aligned_with_transform
)
load_covar_aligned = DATA_UTILS.load_covar_aligned


def test_prediction_keep_filters_in_fam_order_without_refitting_transform(
    tmp_path,
):
    train_fam = tmp_path / "train.fam"
    train_pheno = tmp_path / "train.pheno"
    test_fam = tmp_path / "test.fam"
    train_fam.write_text("f i1 0 0 0 -9\nf i2 0 0 0 -9\n")
    train_pheno.write_text("f i1 1\nf i2 2\n")
    test_fam.write_text(
        "f j1 0 0 0 -9\nf j2 0 0 0 -9\nf j3 0 0 0 -9\n"
    )
    _, _, _, _, transform = (
        load_pheno_covar_aligned_with_transform(
            str(train_fam),
            str(train_pheno),
            None,
            add_intercept=True,
        )
    )
    x_test, kept, dropped = load_covar_aligned(
        str(test_fam),
        None,
        transform=transform,
        keep_ids=["j3", "j1"],
    )
    assert kept == ["j1", "j3"]
    assert dropped == ["j2"]
    np.testing.assert_array_equal(
        x_test, np.ones((2, 1), dtype=np.float32)
    )


def test_prediction_cli_keep_options(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["gpu-reml", "--prediction-keep-path", "test.keep"],
    )
    assert RUN_REML.parse_args().prediction_keep_path == "test.keep"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gpu-reml-sparse",
            "--prediction-pgen-prefix",
            "test",
            "--prediction-covar-txt",
            "test.covar",
            "--prediction-keep-path",
            "test.keep",
        ],
    )
    args = RUN_SPARSE.parse_args()
    assert args.prediction_pgen_prefix == "test"
    assert args.prediction_covar_txt == "test.covar"
    assert args.prediction_keep_path == "test.keep"

def test_remove_sparse_prediction_outputs_clears_reused_prefix(tmp_path):
    prefix = str(tmp_path / "sparse")
    table = tmp_path / "sparse.sparse_prediction.tsv"
    metadata = tmp_path / "sparse.sparse_prediction_metadata.json"
    unrelated = tmp_path / "sparse.summary.json"
    table.write_text("stale prediction table\n", encoding="utf-8")
    metadata.write_text("{}\n", encoding="utf-8")
    unrelated.write_text("{}\n", encoding="utf-8")

    SPARSE_PRED.remove_sparse_prediction_outputs(prefix)

    assert not table.exists()
    assert not metadata.exists()
    assert unrelated.exists()


class _ArraySource:
    def __init__(self, block: np.ndarray, missing_val: int = -9):
        self._block = np.asarray(block, dtype=np.int8)
        self.n, self.m = self._block.shape
        self.missing_val = int(missing_val)

    def read_block_variant_major(
        self, snp_start: int, snp_count: int
    ) -> np.ndarray:
        return np.asfortranarray(
            self._block[:, snp_start : snp_start + snp_count].T
        )

    def close(self):
        return None


def _standardize_from_training(
    train: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean = train.astype(np.float64).mean(axis=0)
    variance = train.astype(np.float64).var(axis=0)
    inv_sd = np.where(
        variance > 0.0,
        1.0 / np.sqrt(np.maximum(variance, 1e-6)),
        0.0,
    )
    return (
        (train.astype(np.float64) - mean) * inv_sd,
        (test.astype(np.float64) - mean) * inv_sd,
    )


def _partitioned_prediction_genotypes() -> tuple[np.ndarray, np.ndarray]:
    train = np.asarray(
        [
            [0, 0, 1, 2, 0, 1],
            [1, 0, 2, 1, 1, 0],
            [2, 1, 0, 0, 2, 1],
            [0, 2, 1, 1, 0, 2],
            [1, 1, 2, 0, 1, 2],
            [2, 2, 0, 2, 2, 0],
            [0, 1, 1, 0, 2, 2],
            [1, 2, 0, 1, 0, 1],
            [2, 0, 2, 2, 1, 0],
            [0, 2, 0, 1, 2, 1],
            [1, 0, 1, 2, 0, 2],
            [2, 1, 2, 0, 1, 1],
        ],
        dtype=np.int8,
    )
    test = np.asarray(
        [
            [2, 2, 0, 0, 2, 1],
            [0, 1, 2, 2, 0, 0],
            [1, 2, 1, 0, 1, 2],
            [2, 0, 2, 1, 2, 0],
            [0, 0, 0, 2, 1, 2],
        ],
        dtype=np.int8,
    )
    return train, test


def test_partitioned_sparse_source_mapping_and_blup_match_dense_algebra():
    x_train, x_test = _partitioned_prediction_genotypes()
    groups = [
        np.asarray([4, 0, 2], dtype=np.int64),
        np.asarray([5, 3, 1], dtype=np.int64),
    ]
    train = InfinitesimalREMLFitter(
        FitConfig(
            sources=[_ArraySource(x_train)],
            component_variant_indices=groups,
            call_width=2,
            keep_host_stats=True,
            precond_rank=0,
            verbose=False,
        )
    )
    test = InfinitesimalREMLFitter(
        FitConfig(
            sources=[_ArraySource(x_test)],
            component_variant_indices=groups,
            call_width=2,
            standardization_overrides=[
                (
                    train.streamers[0]._means_host,
                    train.streamers[0]._inv_sds_host,
                )
            ],
            keep_host_stats=True,
            precond_rank=0,
            verbose=False,
        )
    )
    try:
        train_streamer = train.streamers[0]
        test_streamer = test.streamers[0]
        source_order = np.asarray([0, 2, 4, 1, 3, 5], dtype=np.int64)
        np.testing.assert_array_equal(
            train_streamer._cache_to_source_variant_indices, source_order
        )
        np.testing.assert_array_equal(
            train_streamer.component_source_variant_indices(0),
            source_order[:3],
        )
        np.testing.assert_array_equal(
            train_streamer.component_source_variant_indices(1),
            source_order[3:],
        )

        z_train_source, z_test_source = _standardize_from_training(
            x_train, x_test
        )
        cache_indices = np.arange(x_train.shape[1], dtype=np.int64)
        z_train_cache = train_streamer.extract_standardized_columns(
            cache_indices
        )
        z_test_cache = test_streamer.extract_standardized_columns(cache_indices)
        np.testing.assert_allclose(
            z_train_cache,
            z_train_source[:, source_order],
            rtol=2e-5,
            atol=2e-5,
        )
        np.testing.assert_allclose(
            z_test_cache,
            z_test_source[:, source_order],
            rtol=2e-5,
            atol=2e-5,
        )

        support_cache = np.asarray([0, 4], dtype=np.int64)
        z_active_train = z_train_cache[:, support_cache]
        z_active_test = z_test_cache[:, support_cache]
        c_train = np.ones((x_train.shape[0], 1), dtype=np.float64)
        c_test = np.ones((x_test.shape[0], 1), dtype=np.float64)
        beta_cov = np.asarray([0.4], dtype=np.float64)
        beta_active = np.asarray([0.35, -0.2], dtype=np.float64)
        theta = np.asarray([0.24, 0.31, 0.45], dtype=np.float64)
        y = (
            c_train @ beta_cov
            + z_active_train @ beta_active
            + np.linspace(-0.3, 0.35, x_train.shape[0])
        )

        prediction = predict_sparse_branch(
            name="partitioned",
            fitter=train,
            test_fitter=test,
            y_train=y,
            train_covar=c_train,
            test_covar=c_test,
            train_active_geno=z_active_train,
            test_active_geno=z_active_test,
            beta_cov=beta_cov,
            beta_active=beta_active,
            theta=theta,
            pcg_tol=1e-6,
            max_pcg_iters=1000,
        )

        residual = y - c_train @ beta_cov - z_active_train @ beta_active
        covariance = theta[-1] * np.eye(x_train.shape[0])
        expected_components = []
        for component_index in range(2):
            start = int(train_streamer._component_snp_offsets[component_index])
            stop = int(
                train_streamer._component_snp_offsets[component_index + 1]
            )
            count = float(train_streamer._component_eff_m_host[component_index])
            z_group_train = z_train_cache[:, start:stop]
            covariance += (
                theta[component_index]
                * (z_group_train @ z_group_train.T)
                / count
            )
        dual = np.linalg.solve(covariance, residual)
        for component_index in range(2):
            start = int(train_streamer._component_snp_offsets[component_index])
            stop = int(
                train_streamer._component_snp_offsets[component_index + 1]
            )
            count = float(train_streamer._component_eff_m_host[component_index])
            expected_components.append(
                theta[component_index]
                * z_test_cache[:, start:stop]
                @ (z_train_cache[:, start:stop].T @ dual)
                / count
            )
        expected_background = np.sum(expected_components, axis=0)
        np.testing.assert_allclose(
            prediction.fixed_snp_score,
            z_test_source[:, [0, 3]] @ beta_active,
            rtol=2e-5,
            atol=2e-5,
        )
        np.testing.assert_allclose(
            prediction.background_blup,
            expected_background,
            rtol=7e-4,
            atol=7e-4,
        )
        for observed, expected in zip(
            prediction.background_components, expected_components
        ):
            np.testing.assert_allclose(
                observed, expected, rtol=7e-4, atol=7e-4
            )

        beta_cov_path = np.stack([beta_cov, beta_cov + 0.1])
        beta_active_path = np.stack([beta_active, 0.5 * beta_active])
        path = predict_sparse_path_partitioned(
            fitter=train,
            test_fitter=test,
            y_train=y,
            train_covar=c_train,
            test_covar=c_test,
            train_candidate_geno=z_active_train,
            test_candidate_geno=z_active_test,
            beta_cov_path=beta_cov_path,
            beta_candidate_path=beta_active_path,
            theta=theta,
            pcg_tol=1e-6,
            max_pcg_iters=1000,
        )
        for path_index in range(beta_active_path.shape[0]):
            point = predict_sparse_branch(
                name=f"partitioned_{path_index}",
                fitter=train,
                test_fitter=test,
                y_train=y,
                train_covar=c_train,
                test_covar=c_test,
                train_active_geno=z_active_train,
                test_active_geno=z_active_test,
                beta_cov=beta_cov_path[path_index],
                beta_active=beta_active_path[path_index],
                theta=theta,
                pcg_tol=1e-6,
                max_pcg_iters=1000,
            )
            np.testing.assert_allclose(
                path.background_blup[:, path_index],
                point.background_blup,
                rtol=7e-4,
                atol=7e-4,
            )
            np.testing.assert_allclose(
                path.phenotype_prediction[:, path_index],
                point.phenotype_prediction,
                rtol=7e-4,
                atol=7e-4,
            )
    finally:
        train.close()
        test.close()


def test_partitioned_path_one_component_matches_single_grm_api():
    x_train, x_test = _partitioned_prediction_genotypes()
    one_group = [np.arange(x_train.shape[1], dtype=np.int64)]
    train = InfinitesimalREMLFitter(
        FitConfig(
            sources=[_ArraySource(x_train)],
            component_variant_indices=one_group,
            call_width=3,
            keep_host_stats=True,
            precond_rank=0,
            verbose=False,
        )
    )
    test = InfinitesimalREMLFitter(
        FitConfig(
            sources=[_ArraySource(x_test)],
            component_variant_indices=one_group,
            call_width=3,
            standardization_overrides=[
                (
                    train.streamers[0]._means_host,
                    train.streamers[0]._inv_sds_host,
                )
            ],
            keep_host_stats=True,
            precond_rank=0,
            verbose=False,
        )
    )
    try:
        z_train = train.streamers[0].extract_standardized_columns(
            np.asarray([1, 4], dtype=np.int64)
        )
        z_test = test.streamers[0].extract_standardized_columns(
            np.asarray([1, 4], dtype=np.int64)
        )
        c_train = np.ones((x_train.shape[0], 1), dtype=np.float64)
        c_test = np.ones((x_test.shape[0], 1), dtype=np.float64)
        beta_cov_path = np.asarray([[0.2], [0.35]], dtype=np.float64)
        beta_path = np.asarray(
            [[0.0, 0.0], [0.3, -0.15]], dtype=np.float64
        )
        theta = np.asarray([0.3, 0.7], dtype=np.float64)
        y = 0.25 + z_train @ np.asarray([0.2, -0.1]) + np.linspace(
            -0.2, 0.25, x_train.shape[0]
        )
        kwargs = dict(
            fitter=train,
            test_fitter=test,
            y_train=y,
            train_covar=c_train,
            test_covar=c_test,
            train_candidate_geno=z_train,
            test_candidate_geno=z_test,
            beta_cov_path=beta_cov_path,
            beta_candidate_path=beta_path,
            theta=theta,
            pcg_tol=1e-6,
            max_pcg_iters=1000,
        )
        partitioned = predict_sparse_path_partitioned(**kwargs)
        single = predict_sparse_path_single_grm(**kwargs)
        for field in (
            "residual",
            "dual",
            "nuisance_fixed_score",
            "fixed_snp_score",
            "background_blup",
            "genetic_score",
            "phenotype_prediction",
        ):
            np.testing.assert_allclose(
                getattr(partitioned, field),
                getattr(single, field),
                rtol=2e-6,
                atol=2e-6,
            )
    finally:
        train.close()
        test.close()


def test_single_grm_path_prediction_matches_pointwise_branches():
    x_train = np.asarray(
        [
            [0, 0, 1, 2], [1, 0, 2, 1], [2, 1, 0, 0],
            [0, 2, 1, 1], [1, 1, 2, 0], [2, 2, 0, 2],
            [0, 1, 1, 0], [1, 2, 0, 1], [2, 0, 2, 2],
            [0, 2, 0, 1], [1, 0, 1, 2], [2, 1, 2, 0],
        ],
        dtype=np.int8,
    )
    x_test = np.asarray(
        [
            [2, 2, 0, 0], [0, 1, 2, 2], [1, 2, 1, 0],
            [2, 0, 2, 1], [0, 0, 0, 2],
        ],
        dtype=np.int8,
    )
    train = InfinitesimalREMLFitter(
        FitConfig(
            sources=[_ArraySource(x_train)],
            call_width=2,
            keep_host_stats=True,
            precond_rank=0,
            verbose=False,
        )
    )
    test = InfinitesimalREMLFitter(
        FitConfig(
            sources=[_ArraySource(x_test)],
            call_width=2,
            standardization_overrides=[
                (
                    train.streamers[0]._means_host,
                    train.streamers[0]._inv_sds_host,
                )
            ],
            keep_host_stats=True,
            precond_rank=0,
            verbose=False,
        )
    )
    try:
        marker_idx = np.asarray([0, 2], dtype=np.int64)
        train_index = RUN_SPARSE.SingleGRMIndex(train.streamers)
        test_index = RUN_SPARSE.SingleGRMIndex(test.streamers)
        z_train = train_index.extract_standardized_columns(marker_idx)
        z_test = test_index.extract_standardized_columns(marker_idx)
        c_train = np.ones((x_train.shape[0], 1), dtype=np.float64)
        c_test = np.ones((x_test.shape[0], 1), dtype=np.float64)
        beta_cov_path = np.asarray([[0.2], [0.35]], dtype=np.float64)
        beta_path = np.asarray(
            [[0.0, 0.0], [0.3, -0.15]], dtype=np.float64
        )
        theta = np.asarray([0.3, 0.7], dtype=np.float64)
        y = (
            0.25
            + z_train @ np.asarray([0.2, -0.1])
            + np.linspace(-0.2, 0.25, x_train.shape[0])
        )

        path = predict_sparse_path_single_grm(
            fitter=train,
            test_fitter=test,
            y_train=y,
            train_covar=c_train,
            test_covar=c_test,
            train_candidate_geno=z_train,
            test_candidate_geno=z_test,
            beta_cov_path=beta_cov_path,
            beta_candidate_path=beta_path,
            theta=theta,
            pcg_tol=1e-6,
            max_pcg_iters=1000,
        )
        for path_index in range(beta_path.shape[0]):
            point = predict_sparse_branch(
                name=f"path_{path_index}",
                fitter=train,
                test_fitter=test,
                y_train=y,
                train_covar=c_train,
                test_covar=c_test,
                train_active_geno=z_train,
                test_active_geno=z_test,
                beta_cov=beta_cov_path[path_index],
                beta_active=beta_path[path_index],
                theta=theta,
                pcg_tol=1e-6,
                max_pcg_iters=1000,
            )
            np.testing.assert_allclose(
                path.residual[:, path_index], point.residual, atol=5e-5
            )
            np.testing.assert_allclose(
                path.phenotype_prediction[:, path_index],
                point.phenotype_prediction,
                rtol=5e-4,
                atol=5e-4,
            )
    finally:
        train.close()
        test.close()


@pytest.mark.parametrize("partitioned", [False, True])
@pytest.mark.parametrize("perturb", [False, True])
def test_path_reuses_certified_dual_and_refines_only_inaccurate_columns(
    monkeypatch, partitioned, perturb
):
    x_train, x_test = _partitioned_prediction_genotypes()
    groups = [[0, 2, 4], [1, 3, 5]] if partitioned else None
    config = dict(component_variant_indices=groups, call_width=2,
                  keep_host_stats=True, precond_rank=0, verbose=False)
    train = InfinitesimalREMLFitter(FitConfig(sources=[_ArraySource(x_train)], **config))
    test = InfinitesimalREMLFitter(FitConfig(
        sources=[_ArraySource(x_test)], standardization_overrides=[
            (train.streamers[0]._means_host, train.streamers[0]._inv_sds_host)
        ], **config,
    ))
    try:
        marker_indices = np.array([0, 2])
        z_train = train.streamers[0].extract_standardized_columns(marker_indices)
        z_test = test.streamers[0].extract_standardized_columns(marker_indices)
        kwargs = dict(
            fitter=train, test_fitter=test,
            y_train=np.linspace(-1, 1, x_train.shape[0]),
            train_covar=np.ones((x_train.shape[0], 1)),
            test_covar=np.ones((x_test.shape[0], 1)),
            train_candidate_geno=z_train, test_candidate_geno=z_test,
            beta_cov_path=np.array([[0.1], [0.2]]),
            beta_candidate_path=np.array([[0.1, -0.1], [0.2, 0.1]]),
            theta=np.array([0.2, 0.3, 0.5] if partitioned else [0.4, 0.6]),
            pcg_tol=1e-5, max_pcg_iters=100,
        )
        predict = predict_sparse_path_partitioned if partitioned else predict_sparse_path_single_grm
        fresh = predict(**kwargs)
        dual = fresh.dual.copy()
        if perturb:
            dual[:, 1] += 0.5
        scores = np.asarray(train.streamers[0].xtv(jnp.asarray(dual), normalize=False))
        solve_widths, score_widths = [], []
        original_pcg = SPARSE_PRED.pcg_solve
        original_xtv = train.streamers[0].xtv

        def solve(hv, rhs, **options):
            solve_widths.append(rhs.shape[1])
            return original_pcg(hv, rhs, **options)

        def xtv(value, **options):
            score_widths.append(value.shape[1])
            return original_xtv(value, **options)

        monkeypatch.setattr(SPARSE_PRED, "pcg_solve", solve)
        monkeypatch.setattr(train.streamers[0], "xtv", xtv)
        cached = predict(**kwargs, hinv_residual_path=dual, training_score_path=scores)
        assert solve_widths == ([1] if perturb else [])
        assert score_widths == ([1] if perturb else [])
        assert cached.pcg_rel_res <= 1.05e-5
        np.testing.assert_allclose(cached.phenotype_prediction, fresh.phenotype_prediction,
                                   rtol=2e-5, atol=2e-5)
        with pytest.raises(ValueError, match="matching dual"):
            predict(**kwargs, training_score_path=scores)
    finally:
        train.close()
        test.close()


def _manual_background(z_train, z_test, residual, theta):
    m = float(z_train.shape[1])
    covariance = (
        float(theta[0]) * (z_train @ z_train.T) / m
        + float(theta[1]) * np.eye(z_train.shape[0])
    )
    dual = np.linalg.solve(covariance, residual)
    return (
        float(theta[0])
        * z_test
        @ (z_train.T @ dual)
        / m
    )


def test_sparse_predictor_uses_lasso_mean_and_theta():
    x_train = np.asarray(
        [
            [0, 0, 1, 2, 0, 1],
            [1, 0, 2, 1, 1, 0],
            [2, 1, 0, 0, 2, 1],
            [0, 2, 1, 1, 0, 2],
            [1, 1, 2, 0, 1, 2],
            [2, 2, 0, 2, 2, 0],
            [0, 1, 1, 0, 2, 2],
            [1, 2, 0, 1, 0, 1],
            [2, 0, 2, 2, 1, 0],
            [0, 2, 0, 1, 2, 1],
            [1, 0, 1, 2, 0, 2],
            [2, 1, 2, 0, 1, 1],
        ],
        dtype=np.int8,
    )
    x_test = np.asarray(
        [
            [2, 2, 0, 0, 2, 1],
            [0, 1, 2, 2, 0, 0],
            [1, 2, 1, 0, 1, 2],
            [2, 0, 2, 1, 2, 0],
            [0, 0, 0, 2, 1, 2],
        ],
        dtype=np.int8,
    )
    train = InfinitesimalREMLFitter(
        FitConfig(
            sources=[_ArraySource(x_train)],
            call_width=3,
            keep_host_stats=True,
            precond_rank=0,
            verbose=False,
        )
    )
    overrides = [
        (streamer._means_host, streamer._inv_sds_host)
        for streamer in train.streamers
    ]
    test = InfinitesimalREMLFitter(
        FitConfig(
            sources=[_ArraySource(x_test)],
            call_width=3,
            standardization_overrides=overrides,
            keep_host_stats=True,
            precond_rank=0,
            verbose=False,
        )
    )
    try:
        all_idx = np.arange(x_train.shape[1], dtype=np.int64)
        support = np.asarray([1, 4], dtype=np.int64)
        z_train = np.asarray(
            train.streamers[0].extract_standardized_columns(all_idx),
            dtype=np.float64,
        )
        z_test = np.asarray(
            test.streamers[0].extract_standardized_columns(all_idx),
            dtype=np.float64,
        )
        train_mean = x_train.astype(np.float64).mean(axis=0)
        train_var = x_train.astype(np.float64).var(axis=0)
        train_inv = np.where(
            train_var > 0.0,
            1.0 / np.sqrt(np.maximum(train_var, 1e-6)),
            0.0,
        )
        np.testing.assert_allclose(
            z_test,
            (x_test.astype(np.float64) - train_mean) * train_inv,
            rtol=2e-5,
            atol=2e-5,
        )

        c_train = np.c_[
            np.ones(x_train.shape[0]),
            np.linspace(-1.0, 1.0, x_train.shape[0]),
        ]
        c_test = np.c_[
            np.ones(x_test.shape[0]),
            np.linspace(0.2, 1.0, x_test.shape[0]),
        ]
        y = (
            0.7
            + 0.15 * c_train[:, 1]
            + 0.5 * z_train[:, 1]
            - 0.25 * z_train[:, 4]
            + np.linspace(-0.25, 0.35, x_train.shape[0])
        )
        beta_cov = np.asarray([0.4, -0.2])
        beta_active = np.asarray([0.35, -0.10])
        theta = np.asarray([0.28, 0.72])
        prediction = predict_sparse_branch(
            name="lasso",
            fitter=train,
            test_fitter=test,
            y_train=y,
            train_covar=c_train,
            test_covar=c_test,
            train_active_geno=z_train[:, support],
            test_active_geno=z_test[:, support],
            beta_cov=beta_cov,
            beta_active=beta_active,
            theta=theta,
            pcg_tol=1e-7,
            max_pcg_iters=1000,
        )
        residual = y - c_train @ beta_cov - z_train[:, support] @ beta_active
        expected_background = _manual_background(
            z_train, z_test, residual, theta
        )
        np.testing.assert_allclose(
            prediction.residual, residual, rtol=1e-7, atol=1e-7
        )
        np.testing.assert_allclose(
            prediction.nuisance_fixed_score,
            c_test @ beta_cov,
            rtol=2e-5,
            atol=2e-5,
        )
        np.testing.assert_allclose(
            prediction.fixed_snp_score,
            z_test[:, support] @ beta_active,
            rtol=2e-5,
            atol=2e-5,
        )
        np.testing.assert_allclose(
            prediction.background_blup,
            expected_background,
            rtol=3e-4,
            atol=3e-4,
        )
        np.testing.assert_allclose(
            prediction.phenotype_prediction,
            c_test @ beta_cov
            + z_test[:, support] @ beta_active
            + expected_background,
            rtol=3e-4,
            atol=3e-4,
        )
    finally:
        train.close()
        test.close()


def _toy_branch(name: str, offset: float) -> SparseBranchPrediction:
    values = np.asarray([offset, offset + 1.0])
    return SparseBranchPrediction(
        name=name,
        theta=np.asarray([0.3, 0.7]),
        residual=np.asarray([0.0]),
        nuisance_fixed_score=values + 1.0,
        fixed_snp_score=values + 2.0,
        background_blup=values + 3.0,
        background_components=(values + 3.0,),
        genetic_score=2.0 * values + 5.0,
        phenotype_prediction=3.0 * values + 6.0,
        pcg_rel_res=1e-8,
        pcg_iters=4,
    )


def test_sparse_prediction_writer_emits_only_lasso_branch(tmp_path):
    prefix = str(tmp_path / "pred" / "fit")
    paths = write_sparse_prediction_outputs(
        out_prefix=prefix,
        sample_ids=["i1", "i2"],
        lasso=_toy_branch("lasso", 0.0),
        metadata={
            "test_phenotype_used": False,
            "emitted_branches": ["lasso"],
        },
    )
    with open(paths["prediction"], encoding="utf-8") as handle:
        header = handle.readline().strip().split("\t")
    assert "lasso_genetic_score" in header
    with open(paths["metadata"], encoding="utf-8") as handle:
        metadata = json.load(handle)
    assert metadata["status"] == "emitted"
    assert metadata["test_phenotype_used"] is False
    assert metadata["emitted_branches"] == ["lasso"]

    with pytest.raises(ValueError, match="valid COHERIT Lasso"):
        write_sparse_prediction_outputs(
            out_prefix=prefix,
            sample_ids=["i1", "i2"],
            lasso=None,
            metadata={"test_phenotype_used": False},
        )

    write_sparse_prediction_status(
        out_prefix=prefix,
        status="not_emitted_no_valid_branch",
        metadata={"sparse_fit_rejection_reasons": ["guard_failed"]},
    )
    assert not os.path.exists(paths["prediction"])
    with open(paths["metadata"], encoding="utf-8") as handle:
        metadata = json.load(handle)
    assert metadata["status"] == "not_emitted_no_valid_branch"
