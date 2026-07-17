"""Publication-critical regressions for the sparse--dense heritability path."""

from __future__ import annotations

import importlib
import os
import sys

import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT = os.path.dirname(REPO_ROOT)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

PKG = os.path.basename(REPO_ROOT)
SPARSE = importlib.import_module(f"{PKG}.run_sparse_reml_pipeline")


def test_sparse_dense_h2_is_invariant_to_phenotype_rescaling():
    """CHIVE q and REML theta must be combined on the standardized-y scale."""
    rng = np.random.RandomState(314)
    n, s = 120, 7
    z_active = rng.standard_normal((n, s))
    z_active = (z_active - z_active.mean(axis=0)) / z_active.std(axis=0)
    beta = rng.standard_normal(s) * 0.15
    y = z_active @ beta + rng.standard_normal(n) * 0.7

    q_raw, _, _ = SPARSE._chive_q_hat_given_active(z_active, y, beta)
    _, y_scale = SPARSE._phenotype_standardization_stats(y)
    q_std = SPARSE._quadratic_variance_to_reml_scale(q_raw, y_scale)
    h2 = SPARSE._sparse_dense_h2(q_std, 0.22, 0.63)

    multiplier = 9.0
    q_scaled_raw, _, _ = SPARSE._chive_q_hat_given_active(
        z_active, multiplier * y, multiplier * beta
    )
    _, y_scaled_scale = SPARSE._phenotype_standardization_stats(multiplier * y)
    q_scaled_std = SPARSE._quadratic_variance_to_reml_scale(
        q_scaled_raw, y_scaled_scale
    )
    h2_scaled = SPARSE._sparse_dense_h2(q_scaled_std, 0.22, 0.63)

    assert np.isclose(q_std, q_scaled_std, rtol=2e-6, atol=2e-6)
    assert np.isclose(h2, h2_scaled, rtol=2e-6, atol=2e-6)

    # The historical raw-q/standardized-theta mixture fails this invariance.
    h2_old = SPARSE._sparse_dense_h2(q_raw, 0.22, 0.63)
    h2_scaled_old = SPARSE._sparse_dense_h2(q_scaled_raw, 0.22, 0.63)
    assert abs(h2_old - h2_scaled_old) > 0.05


def test_primary_h2_uses_penalized_chive_not_post_selection_gls():
    assert SPARSE._primary_sparse_dense_h2(0.31, 0.47) == 0.31


def test_coherent_sparse_fixed_point_keeps_hybrid_as_primary():
    primary, fallback, reason = SPARSE._select_primary_h2_with_fallback(
        0.31,
        0.22,
        alpha_theta_fixed_point_coherent=True,
    )
    assert primary == 0.31
    assert fallback is False
    assert reason is None


def test_incoherent_sparse_fixed_point_falls_back_to_standard_reml():
    primary, fallback, reason = SPARSE._select_primary_h2_with_fallback(
        0.31,
        0.22,
        alpha_theta_fixed_point_coherent=False,
    )
    assert primary == 0.22
    assert fallback is True
    assert reason == "sparse_outer_not_alpha_theta_fixed_point"


def test_sparse_pipeline_ebic_defaults_to_full_model_space(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["gpu-reml-sparse"])
    args = SPARSE.parse_args()
    assert args.ebic_p_mode == "full"

    monkeypatch.setattr(
        sys,
        "argv",
        ["gpu-reml-sparse", "--ebic-p-mode", "candidate"],
    )
    assert SPARSE.parse_args().ebic_p_mode == "candidate"


def test_empty_support_reduces_to_background_only_heritability():
    background, residual = 0.28, 0.62
    expected = background / (background + residual)
    penalized = SPARSE._sparse_dense_h2(0.0, background, residual)
    post_gls = SPARSE._sparse_dense_h2(0.0, background, residual)

    assert np.isclose(penalized, expected)
    assert np.isclose(post_gls, expected)
    assert np.isclose(SPARSE._primary_sparse_dense_h2(penalized, post_gls), expected)


def test_same_sample_ols_refit_makes_chive_cross_term_vanish():
    """Locks in why the post-selection GLS/OLS result is diagnostic only."""
    rng = np.random.RandomState(2718)
    z_active = rng.standard_normal((90, 6))
    y = rng.standard_normal(90)
    beta_ols = np.linalg.solve(z_active.T @ z_active, z_active.T @ y)

    q_hat, term1, term2 = SPARSE._chive_q_hat_given_active(
        z_active, y, beta_ols
    )

    assert np.isclose(term2, 0.0, atol=1e-12)
    assert np.isclose(q_hat, term1, atol=1e-12)
    assert q_hat > 0.0
