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
predict_sparse_path_partitioned = (
    SPARSE_PRED.predict_sparse_path_partitioned
)
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
    assert args.compare_four_estimators is False

    monkeypatch.setattr(
        sys,
        "argv",
        ["gpu-reml-sparse", "--compare-four-estimators"],
    )
    assert RUN_SPARSE.parse_args().compare_four_estimators is True


def test_sparse_prediction_branches_follow_estimator_validity():
    branch_names = RUN_SPARSE._sparse_prediction_branch_names
    assert branch_names(
        comparison_enabled=False,
        lasso_branch_valid=False,
        selected_support_refit_branch_valid=False,
    ) == []
    assert branch_names(
        comparison_enabled=False,
        lasso_branch_valid=True,
        selected_support_refit_branch_valid=False,
    ) == ["lasso"]
    assert branch_names(
        comparison_enabled=True,
        lasso_branch_valid=True,
        selected_support_refit_branch_valid=True,
    ) == ["lasso", "selected_span"]
    with pytest.raises(ValueError, match="requires a valid Lasso"):
        branch_names(
            comparison_enabled=True,
            lasso_branch_valid=False,
            selected_support_refit_branch_valid=True,
        )

    # A downstream refit cannot leak into the default COHERIT prediction.
    assert branch_names(
        comparison_enabled=False,
        lasso_branch_valid=True,
        selected_support_refit_branch_valid=True,
    ) == ["lasso"]


def test_remove_sparse_prediction_outputs_clears_reused_prefix(tmp_path):
    prefix = str(tmp_path / "sparse")
    table = tmp_path / "sparse.sparse_prediction.tsv"
    metadata = tmp_path / "sparse.sparse_prediction_metadata.json"
    unrelated = tmp_path / "sparse.summary.json"
    table.write_text("stale selected-span table\n", encoding="utf-8")
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


def test_partitioned_one_component_matches_unpartitioned_sparse_coordinates():
    """A one-component spec must be only a covariance representation change."""
    genotype = np.asarray(
        [
            [0, 0, 1, 2, 0, 1],
            [1, 0, 2, 1, 1, 0],
            [2, 1, 0, 0, 2, 1],
            [0, 2, 1, 1, 0, 2],
            [1, 1, 2, 0, 1, 2],
            [2, 2, 0, 2, 2, 0],
        ],
        dtype=np.int8,
    )
    source_indices = np.arange(genotype.shape[1] - 1, -1, -1, dtype=np.int64)
    ordinary = InfinitesimalREMLFitter(
        FitConfig(
            sources=[_ArraySource(genotype)],
            call_width=2,
            keep_host_stats=True,
            precond_rank=0,
            verbose=False,
        )
    )
    partitioned = InfinitesimalREMLFitter(
        FitConfig(
            sources=[_ArraySource(genotype)],
            component_variant_indices=[source_indices],
            call_width=2,
            keep_host_stats=True,
            precond_rank=0,
            verbose=False,
        )
    )
    try:
        ordinary_index = RUN_SPARSE.MultiGRMIndex(ordinary.streamers)
        partitioned_index = RUN_SPARSE.MultiGRMIndex(
            partitioned.streamers,
            component_variant_indices=[source_indices],
        )
        cache_indices = np.arange(genotype.shape[1], dtype=np.int64)
        np.testing.assert_array_equal(
            partitioned_index.source_variant_indices(cache_indices),
            cache_indices,
        )
        np.testing.assert_allclose(
            partitioned_index.extract_standardized_columns(cache_indices),
            ordinary_index.extract_standardized_columns(cache_indices),
            rtol=0.0,
            atol=0.0,
        )

        vector = jnp.asarray(
            np.linspace(-0.75, 0.9, genotype.shape[0]), dtype=jnp.float32
        )
        np.testing.assert_allclose(
            partitioned_index.xtv_all(vector),
            ordinary_index.xtv_all(vector),
            rtol=1e-6,
            atol=1e-6,
        )
        ordinary_k = ordinary._assemble_reml_operators().K_mvs[0](vector)
        partitioned_k = partitioned._assemble_reml_operators().K_mvs[0](vector)
        np.testing.assert_allclose(
            np.asarray(partitioned_k),
            np.asarray(ordinary_k),
            rtol=1e-6,
            atol=1e-6,
        )
    finally:
        ordinary.close()
        partitioned.close()


def test_partitioned_sparse_source_ids_and_bim_rows_follow_streamer_mapping(
    tmp_path,
):
    """Selected cache coordinates must resolve to the SNP actually decoded."""
    genotype = np.asarray(
        [
            [0, 0, 1, 2, 0, 1],
            [1, 0, 2, 1, 1, 0],
            [2, 1, 0, 0, 2, 1],
            [0, 2, 1, 1, 0, 2],
            [1, 1, 2, 0, 1, 2],
            [2, 2, 0, 2, 2, 0],
        ],
        dtype=np.int8,
    )
    # Deliberately unsorted within each component.  The streamer canonicalizes
    # these memberships, so output mapping must be read back from the streamer
    # instead of concatenating the raw component specification.
    groups = [
        np.asarray([4, 0, 2], dtype=np.int64),
        np.asarray([5, 3, 1], dtype=np.int64),
    ]
    fitter = InfinitesimalREMLFitter(
        FitConfig(
            sources=[_ArraySource(genotype)],
            component_variant_indices=groups,
            call_width=2,
            keep_host_stats=True,
            precond_rank=0,
            verbose=False,
        )
    )
    try:
        index = RUN_SPARSE.MultiGRMIndex(
            fitter.streamers,
            component_variant_indices=groups,
        )
        expected_source_order = np.asarray([0, 2, 4, 1, 3, 5], dtype=np.int64)
        cache_indices = np.arange(expected_source_order.size, dtype=np.int64)
        np.testing.assert_array_equal(
            index.source_variant_indices(cache_indices),
            expected_source_order,
        )

        standardized, _ = _standardize_from_training(genotype, genotype)
        np.testing.assert_allclose(
            index.extract_standardized_columns(cache_indices),
            standardized[:, expected_source_order],
            rtol=2e-5,
            atol=2e-5,
        )
        score_vector = jnp.asarray(
            np.linspace(-0.6, 0.8, genotype.shape[0]), dtype=jnp.float32
        )
        np.testing.assert_allclose(
            index.xtv_all(score_vector),
            standardized[:, expected_source_order].T
            @ np.asarray(score_vector, dtype=np.float64),
            rtol=2e-5,
            atol=2e-5,
        )

        prefix = tmp_path / "markers"
        prefix.with_suffix(".bim").write_text(
            "".join(
                f"1 rs{source_idx} 0 {1000 + source_idx} A G\n"
                for source_idx in range(genotype.shape[1])
            ),
            encoding="utf-8",
        )
        rows = index.lookup_bim_rows(
            [str(prefix)], np.asarray([0, 3, 4, 5], dtype=np.int64)
        )
        assert rows[0][1] == "rs0"
        assert rows[3][1] == "rs1"
        assert rows[4][1] == "rs3"
        assert rows[5][1] == "rs5"

        with pytest.raises(IndexError, match="Global SNP indices"):
            index.source_variant_indices(np.asarray([-1], dtype=np.int64))
        with pytest.raises(IndexError, match="Global SNP indices"):
            index.extract_standardized_columns(
                np.asarray([genotype.shape[1]], dtype=np.int64)
            )
    finally:
        fitter.close()


def test_partitioned_sparse_prediction_uses_matching_source_coordinates():
    """Fixed-score and multi-GRM BLUP prediction must share one SNP mapping."""
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
        train_index = RUN_SPARSE.MultiGRMIndex(
            train.streamers, component_variant_indices=groups
        )
        test_index = RUN_SPARSE.MultiGRMIndex(
            test.streamers, component_variant_indices=groups
        )
        # Cache positions 0 and 4 are source SNPs 0 and 3 under the canonical
        # component order [0,2,4 | 1,3,5].
        support_cache = np.asarray([0, 4], dtype=np.int64)
        np.testing.assert_array_equal(
            train_index.source_variant_indices(support_cache),
            np.asarray([0, 3], dtype=np.int64),
        )
        z_active_train = train_index.extract_standardized_columns(support_cache)
        z_active_test = test_index.extract_standardized_columns(support_cache)

        z_train_source, z_test_source = _standardize_from_training(
            x_train, x_test
        )
        np.testing.assert_allclose(
            z_active_train,
            z_train_source[:, [0, 3]],
            rtol=2e-5,
            atol=2e-5,
        )
        np.testing.assert_allclose(
            z_active_test,
            z_test_source[:, [0, 3]],
            rtol=2e-5,
            atol=2e-5,
        )

        c_train = np.ones((x_train.shape[0], 1), dtype=np.float64)
        c_test = np.ones((x_test.shape[0], 1), dtype=np.float64)
        beta_cov = np.asarray([0.4], dtype=np.float64)
        beta_active = np.asarray([0.35, -0.2], dtype=np.float64)
        phenotype_scale = 1.3
        theta = np.asarray([0.24, 0.31, 0.45], dtype=np.float64)
        y = (
            c_train @ beta_cov
            + z_active_train @ beta_active
            + np.linspace(-0.3, 0.35, x_train.shape[0])
        )
        prediction = predict_sparse_branch(
            name="lasso",
            fitter=train,
            test_fitter=test,
            y_train_raw=y,
            train_covar=c_train,
            test_covar=c_test,
            train_active_geno=z_active_train,
            test_active_geno=z_active_test,
            beta_cov_raw=beta_cov,
            beta_active_raw=beta_active,
            theta_standardized=theta,
            phenotype_scale=phenotype_scale,
            pcg_tol=1e-7,
            max_pcg_iters=1000,
        )

        residual = (
            y - c_train @ beta_cov - z_active_train @ beta_active
        ) / phenotype_scale
        canonical_groups = (
            np.asarray([0, 2, 4], dtype=np.int64),
            np.asarray([1, 3, 5], dtype=np.int64),
        )
        covariance = theta[-1] * np.eye(x_train.shape[0])
        for component_idx, source_group in enumerate(canonical_groups):
            z_group = z_train_source[:, source_group]
            covariance += (
                theta[component_idx]
                * (z_group @ z_group.T)
                / float(source_group.size)
            )
        dual = np.linalg.solve(covariance, residual)
        expected_components = []
        for component_idx, source_group in enumerate(canonical_groups):
            expected_components.append(
                phenotype_scale
                * theta[component_idx]
                * z_test_source[:, source_group]
                @ (z_train_source[:, source_group].T @ dual)
                / float(source_group.size)
            )
        expected_background = np.sum(expected_components, axis=0)

        np.testing.assert_allclose(
            prediction.fixed_snp_score_raw,
            z_test_source[:, [0, 3]] @ beta_active,
            rtol=2e-5,
            atol=2e-5,
        )
        np.testing.assert_allclose(
            prediction.background_blup_raw,
            expected_background,
            rtol=5e-4,
            atol=5e-4,
        )
        for observed, expected in zip(
            prediction.background_components_raw, expected_components
        ):
            np.testing.assert_allclose(
                observed,
                expected,
                rtol=5e-4,
                atol=5e-4,
            )

        beta_cov_path = np.stack([beta_cov, beta_cov + 0.1], axis=0)
        beta_active_path = np.stack(
            [beta_active, 0.5 * beta_active], axis=0
        )
        path_prediction = predict_sparse_path_partitioned(
            fitter=train,
            test_fitter=test,
            y_train_raw=y,
            train_covar=c_train,
            test_covar=c_test,
            train_candidate_geno=z_active_train,
            test_candidate_geno=z_active_test,
            beta_cov_path_raw=beta_cov_path,
            beta_candidate_path_raw=beta_active_path,
            theta_standardized=theta,
            phenotype_scale=phenotype_scale,
            pcg_tol=1e-7,
            max_pcg_iters=1000,
        )
        for path_index in range(beta_active_path.shape[0]):
            branch = predict_sparse_branch(
                name=f"path_{path_index}",
                fitter=train,
                test_fitter=test,
                y_train_raw=y,
                train_covar=c_train,
                test_covar=c_test,
                train_active_geno=z_active_train,
                test_active_geno=z_active_test,
                beta_cov_raw=beta_cov_path[path_index],
                beta_active_raw=beta_active_path[path_index],
                theta_standardized=theta,
                phenotype_scale=phenotype_scale,
                pcg_tol=1e-7,
                max_pcg_iters=1000,
            )
            np.testing.assert_allclose(
                path_prediction.residual_standardized[:, path_index],
                branch.residual_standardized,
                rtol=5e-5,
                atol=5e-5,
            )
            np.testing.assert_allclose(
                path_prediction.fixed_snp_score_raw[:, path_index],
                branch.fixed_snp_score_raw,
                rtol=5e-5,
                atol=5e-5,
            )
            np.testing.assert_allclose(
                path_prediction.background_blup_raw[:, path_index],
                branch.background_blup_raw,
                rtol=5e-4,
                atol=5e-4,
            )
            np.testing.assert_allclose(
                path_prediction.phenotype_prediction_raw[:, path_index],
                branch.phenotype_prediction_raw,
                rtol=5e-4,
                atol=5e-4,
            )
    finally:
        train.close()
        test.close()


def _manual_background(
    z_train, z_test, residual_standardized, theta, phenotype_scale
):
    m = float(z_train.shape[1])
    covariance = (
        float(theta[0]) * (z_train @ z_train.T) / m
        + float(theta[1]) * np.eye(z_train.shape[0])
    )
    dual = np.linalg.solve(covariance, residual_standardized)
    return (
        float(phenotype_scale)
        * float(theta[0])
        * z_test
        @ (z_train.T @ dual)
        / m
    )


def test_sparse_predictors_use_branch_matched_mean_theta_and_raw_scale():
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
        phenotype_scale = 1.7
        branch_specs = {
            "lasso": (
                np.asarray([0.4, -0.2]),
                np.asarray([0.35, -0.10]),
                np.asarray([0.28, 0.72]),
            ),
            "selected_span": (
                np.asarray([0.55, -0.05]),
                np.asarray([0.42, -0.04]),
                np.asarray([0.48, 0.52]),
            ),
        }
        predictions = {}
        for name, (beta_cov, beta_active, theta) in branch_specs.items():
            predictions[name] = predict_sparse_branch(
                name=name,
                fitter=train,
                test_fitter=test,
                y_train_raw=y,
                train_covar=c_train,
                test_covar=c_test,
                train_active_geno=z_train[:, support],
                test_active_geno=z_test[:, support],
                beta_cov_raw=beta_cov,
                beta_active_raw=beta_active,
                theta_standardized=theta,
                phenotype_scale=phenotype_scale,
                pcg_tol=1e-7,
                max_pcg_iters=1000,
            )
            residual = (
                y
                - c_train @ beta_cov
                - z_train[:, support] @ beta_active
            ) / phenotype_scale
            expected_background = _manual_background(
                z_train, z_test, residual, theta, phenotype_scale
            )
            np.testing.assert_allclose(
                predictions[name].residual_standardized,
                residual,
                rtol=1e-7,
                atol=1e-7,
            )
            np.testing.assert_allclose(
                predictions[name].nuisance_fixed_score_raw,
                c_test @ beta_cov,
                rtol=2e-5,
                atol=2e-5,
            )
            np.testing.assert_allclose(
                predictions[name].fixed_snp_score_raw,
                z_test[:, support] @ beta_active,
                rtol=2e-5,
                atol=2e-5,
            )
            np.testing.assert_allclose(
                predictions[name].background_blup_raw,
                expected_background,
                rtol=3e-4,
                atol=3e-4,
            )
            np.testing.assert_allclose(
                predictions[name].phenotype_prediction_raw,
                c_test @ beta_cov
                + z_test[:, support] @ beta_active
                + expected_background,
                rtol=3e-4,
                atol=3e-4,
            )

        lasso_beta_cov, lasso_beta_active, _ = branch_specs["lasso"]
        selected_theta = branch_specs["selected_span"][2]
        lasso_residual = (
            y
            - c_train @ lasso_beta_cov
            - z_train[:, support] @ lasso_beta_active
        ) / phenotype_scale
        incorrectly_mixed = _manual_background(
            z_train,
            z_test,
            lasso_residual,
            selected_theta,
            phenotype_scale,
        )
        assert not np.allclose(
            predictions["lasso"].background_blup_raw,
            incorrectly_mixed,
            rtol=1e-3,
            atol=1e-3,
        )
    finally:
        train.close()
        test.close()


def _toy_branch(name: str, offset: float) -> SparseBranchPrediction:
    values = np.asarray([offset, offset + 1.0])
    return SparseBranchPrediction(
        name=name,
        theta_standardized=np.asarray([0.3, 0.7]),
        residual_standardized=np.asarray([0.0]),
        nuisance_fixed_score_raw=values + 1.0,
        fixed_snp_score_raw=values + 2.0,
        background_blup_raw=values + 3.0,
        background_components_raw=(values + 3.0,),
        genetic_score_raw=2.0 * values + 5.0,
        phenotype_prediction_raw=3.0 * values + 6.0,
        pcg_rel_res=1e-8,
        pcg_iters=4,
    )


def test_sparse_prediction_writer_supports_independent_branches(tmp_path):
    prefix = str(tmp_path / "pred" / "fit")
    paths = write_sparse_prediction_outputs(
        out_prefix=prefix,
        sample_ids=["i1", "i2"],
        lasso=_toy_branch("lasso", 0.0),
        selected_span=_toy_branch("selected_span", 10.0),
        metadata={"test_phenotype_used": False},
    )
    with open(paths["prediction"], encoding="utf-8") as handle:
        header = handle.readline().strip().split("\t")
    assert "lasso_genetic_score_raw" in header
    assert "selected_span_genetic_score_raw" in header
    with open(paths["metadata"], encoding="utf-8") as handle:
        metadata = json.load(handle)
    assert metadata["status"] == "emitted"
    assert metadata["test_phenotype_used"] is False

    lasso_only = write_sparse_prediction_outputs(
        out_prefix=prefix,
        sample_ids=["i1", "i2"],
        lasso=_toy_branch("lasso", 0.0),
        selected_span=None,
        metadata={
            "test_phenotype_used": False,
            "emitted_branches": ["lasso"],
        },
    )
    with open(lasso_only["prediction"], encoding="utf-8") as handle:
        lasso_header = handle.readline().strip().split("\t")
    assert "lasso_genetic_score_raw" in lasso_header
    assert not any(column.startswith("selected_span_") for column in lasso_header)
    with open(lasso_only["metadata"], encoding="utf-8") as handle:
        lasso_metadata = json.load(handle)
    assert lasso_metadata["emitted_branches"] == ["lasso"]

    with pytest.raises(ValueError, match="At least one valid"):
        write_sparse_prediction_outputs(
            out_prefix=prefix,
            sample_ids=["i1", "i2"],
            lasso=None,
            selected_span=None,
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
