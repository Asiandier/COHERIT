"""Publication-critical regressions for the sparse--dense heritability path."""

from __future__ import annotations

import importlib
import os
import sys
from types import SimpleNamespace

import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT = os.path.dirname(REPO_ROOT)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

PKG = os.path.basename(REPO_ROOT)
SPARSE = importlib.import_module(f"{PKG}.run_sparse_reml_pipeline")
REML = importlib.import_module(f"{PKG}.reml")
LASSO = importlib.import_module(f"{PKG}.lasso_cd")


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


def test_raw_lasso_plugin_is_chive_first_term_without_calibration():
    rng = np.random.RandomState(1617)
    z_active = rng.standard_normal((100, 5))
    beta = rng.standard_normal(5) * 0.2
    y = z_active @ beta + rng.standard_normal(100)

    q_chive, q_lasso_plugin, correction = SPARSE._chive_q_hat_given_active(
        z_active,
        y,
        beta,
    )

    expected_plugin = float(np.mean(np.square(z_active @ beta)))
    assert np.isclose(q_lasso_plugin, expected_plugin)
    assert np.isclose(q_chive, q_lasso_plugin + correction)


def test_four_estimator_h2_uses_lasso_ml_and_reml_branches():
    values = SPARSE._four_estimator_h2_from_branches(
        q_lasso_plugin_standardized=0.07,
        q_lasso_calibrated_standardized=0.13,
        q_selected_span_plugin_standardized=0.19,
        q_selected_span_trace_standardized=0.11,
        lasso_ml_background_variance=0.17,
        lasso_ml_residual_variance=0.71,
        selected_span_reml_background_variance=0.43,
        selected_span_reml_residual_variance=0.29,
    )

    assert np.isclose(
        values["h2_lasso_plugin"],
        SPARSE._sparse_dense_h2(0.07, 0.17, 0.71),
    )
    assert np.isclose(
        values["h2_chive"],
        SPARSE._sparse_dense_h2(0.13, 0.17, 0.71),
    )
    assert np.isclose(
        values["h2_ss_gls_plugin"],
        SPARSE._sparse_dense_h2(0.19, 0.43, 0.29),
    )
    assert np.isclose(
        values["h2_ss_gls_df_corrected"],
        SPARSE._sparse_dense_h2(0.11, 0.43, 0.29),
    )
    assert not np.isclose(
        values["h2_chive"],
        SPARSE._sparse_dense_h2(0.13, 0.43, 0.29),
    )


def test_branch_guards_preserve_valid_lasso_when_refit_is_unavailable():
    guards = SPARSE._sparse_estimator_branch_guards(
        alpha_theta_pair_certified=True,
        lasso_quadratics_available=True,
        selected_span_refit_ok=True,
        lasso_estimator_values=np.asarray([0.2, 0.3]),
        selected_support_estimator_values=np.asarray([0.4, -0.1]),
    )
    assert guards["lasso_branch_valid"] is True
    assert guards["selected_support_refit_branch_valid"] is True
    assert guards["all_four_estimators_valid"] is True
    assert guards["combined_invalid_reasons"] == []

    guards = SPARSE._sparse_estimator_branch_guards(
        alpha_theta_pair_certified=True,
        lasso_quadratics_available=True,
        selected_span_refit_ok=False,
        lasso_estimator_values=np.asarray([0.2, 0.3]),
        selected_support_estimator_values=np.asarray([np.nan, np.nan]),
    )
    assert guards["lasso_branch_valid"] is True
    assert guards["selected_support_refit_branch_valid"] is False
    assert guards["all_four_estimators_valid"] is False
    assert "selected_support_reml_gls_unavailable" in guards[
        "selected_support_refit_branch_invalid_reasons"
    ]
    assert "nonfinite_selected_support_estimator" in guards[
        "selected_support_refit_branch_invalid_reasons"
    ]


def test_invalid_lasso_branch_invalidates_downstream_refit_branch():
    guards = SPARSE._sparse_estimator_branch_guards(
        alpha_theta_pair_certified=False,
        lasso_quadratics_available=True,
        selected_span_refit_ok=True,
        lasso_estimator_values=np.asarray([0.2, 0.3]),
        selected_support_estimator_values=np.asarray([0.4, 0.5]),
    )
    assert guards["lasso_branch_valid"] is False
    assert guards["selected_support_refit_branch_valid"] is False
    assert "penalized_alpha_theta_pair_not_certified" in guards[
        "lasso_branch_invalid_reasons"
    ]
    assert "lasso_support_branch_not_valid" in guards[
        "selected_support_refit_branch_invalid_reasons"
    ]


def test_terminal_sparse_pair_stops_only_after_returned_full_p_kkt():
    common = {
        "terminal_verification": True,
        "stable_candidate": True,
    }
    assert not SPARSE._terminal_sparse_pair_may_stop(
        **common,
        returned_covariance_kkt={"passed": False},
    )
    assert SPARSE._terminal_sparse_pair_may_stop(
        **common,
        returned_covariance_kkt={"passed": True},
    )
    assert not SPARSE._terminal_sparse_pair_may_stop(
        terminal_verification=False,
        stable_candidate=True,
        returned_covariance_kkt={"passed": True},
    )


def test_sparse_pipeline_ebic_defaults_to_full_model_space(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["gpu-reml-sparse"])
    args = SPARSE.parse_args()
    assert args.ebic_p_mode == "full"
    assert args.minq_iter == 50
    assert not hasattr(args, "kkt_check")

    monkeypatch.setattr(sys, "argv", ["gpu-reml-sparse", "--kkt-check"])
    with np.testing.assert_raises(SystemExit):
        SPARSE.parse_args()

    monkeypatch.setattr(
        sys,
        "argv",
        ["gpu-reml-sparse", "--ebic-p-mode", "candidate"],
    )
    assert SPARSE.parse_args().ebic_p_mode == "candidate"

    monkeypatch.setattr(
        sys, "argv", ["gpu-reml-sparse", "--no-kkt-check"]
    )
    with np.testing.assert_raises(SystemExit):
        SPARSE.parse_args()


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


def test_ebic_path_failure_does_not_fall_back_to_valid_null(monkeypatch):
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
        LASSO.solve_lasso_path_and_select_ebic(
            Q=np.asarray([[1.0]]),
            q=np.asarray([1.0]),
            yHy=1.0,
            n_samples=100,
            p_total=10,
            cfg=LASSO.LassoPathConfig(
                n_lambda=2,
                ebic_early_stop=False,
                kkt_abs_tol=1e-8,
                kkt_rel_tol=0.0,
            ),
        )


def test_lasso_variance_block_uses_an_intercept_contrast_design():
    design = SPARSE._intercept_contrast_fixed_effect(37)

    assert design.shape == (37, 1)
    assert design.dtype == np.float32
    assert np.array_equal(design, np.ones((37, 1), dtype=np.float32))


def test_residual_ml_helper_passes_intercept_on_standardized_fit_path():
    marker = SimpleNamespace(
        var_components=REML.jnp.asarray(
            [0.25, 0.75], dtype=REML.jnp.float32
        ),
        rep_var_components=None,
        monte_carlo_se_var=None,
        final_grad=None,
        final_ai=None,
        diagnostics=None,
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
    theta = np.asarray([0.31, 0.69], dtype=np.float32)
    result = SPARSE._fit_intercept_contrast_residual_ml(
        fitter,
        residual,
        theta,
        h2_init=0.31,
    )

    assert result is marker
    assert np.array_equal(fitter.y, residual)
    assert np.array_equal(
        fitter.covar, np.ones((residual.size, 1), dtype=np.float32)
    )
    assert "standardize_y" not in fitter.kwargs
    assert np.isclose(fitter.kwargs["h2_init"], 0.31)
    _, residual_scale = SPARSE._phenotype_standardization_stats(residual)
    variance_scale = residual_scale**2
    assert np.array_equal(
        np.asarray(fitter.kwargs["var_components_init"]),
        np.asarray(theta / variance_scale, dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.asarray(result.var_components),
        np.asarray([0.25, 0.75]) * variance_scale,
    )
    np.testing.assert_allclose(
        result.history[0]["params"],
        np.asarray([0.25, 0.75]) * variance_scale,
    )
    assert np.isclose(
        result.history[0]["variance_scale_to_standardized_phenotype"],
        variance_scale,
    )


def test_empty_support_reduces_to_background_only_heritability():
    background, residual = 0.28, 0.62
    expected = background / (background + residual)
    penalized = SPARSE._sparse_dense_h2(0.0, background, residual)
    post_gls = SPARSE._sparse_dense_h2(0.0, background, residual)

    assert np.isclose(penalized, expected)
    assert np.isclose(post_gls, expected)


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


def test_selected_span_gls_df_correction_matches_fixed_span_formula():
    rng = np.random.RandomState(1618)
    n, k = 70, 4
    z_active = rng.standard_normal((n, k))
    z_active -= z_active.mean(axis=0)
    w = rng.standard_normal((n, n))
    v = w @ w.T / n + 0.7 * np.eye(n)
    vinv = np.linalg.inv(v)
    phenotype_scale = 2.5
    y = phenotype_scale * (
        z_active @ rng.standard_normal(k) + rng.multivariate_normal(np.zeros(n), v)
    )

    out = SPARSE._selected_span_gls_quadratics(
        y=y,
        covar=None,
        z_active=z_active,
        Hinv_y=vinv @ y,
        Hinv_covar=None,
        Hinv_z_active=vinv @ z_active,
        phenotype_scale=phenotype_scale,
    )

    gram_inv = np.linalg.inv(z_active.T @ vinv @ z_active)
    sparse_gram = z_active.T @ z_active / n
    expected_df_std = np.trace(sparse_gram @ gram_inv)
    expected_plugin_std = float(out["q_plugin_raw"]) / phenotype_scale**2

    assert np.isclose(out["df_correction_standardized"], expected_df_std)
    assert np.isclose(out["q_plugin_standardized"], expected_plugin_std)
    assert np.isclose(
        out["q_df_corrected_standardized"],
        expected_plugin_std - expected_df_std,
    )


def test_selected_span_gls_uses_independent_basis_for_duplicate_markers():
    rng = np.random.RandomState(2719)
    n = 60
    z1 = rng.standard_normal(n)
    z1 -= z1.mean()
    z_active = np.column_stack([z1, z1])
    covar = np.ones((n, 1))
    v = np.eye(n)
    y = 0.4 * z1 + rng.standard_normal(n)

    out = SPARSE._selected_span_gls_quadratics(
        y=y,
        covar=covar,
        z_active=z_active,
        Hinv_y=y,
        Hinv_covar=covar,
        Hinv_z_active=z_active,
        phenotype_scale=float(np.std(y)),
    )

    assert len(out["active_basis_idx"]) == 1
    assert np.isfinite(out["q_df_corrected_standardized"])

    beta_full = SPARSE._expand_selected_basis_coefficients(
        support_size=z_active.shape[1],
        basis_positions=out["active_basis_idx"],
        basis_coefficients=out["beta_active_basis"],
    )
    fitted = z_active @ beta_full
    assert np.count_nonzero(beta_full) == 1
    assert np.isclose(
        np.mean(np.square(fitted)),
        out["q_plugin_raw"],
        rtol=1e-12,
        atol=1e-12,
    )


def test_selected_span_gls_recovers_covariates_with_empty_support():
    rng = np.random.RandomState(2720)
    n = 50
    covar = np.ones((n, 1), dtype=np.float64)
    y = 1.25 + rng.standard_normal(n)
    z_active = np.empty((n, 0), dtype=np.float64)

    out = SPARSE._selected_span_gls_quadratics(
        y=y,
        covar=covar,
        z_active=z_active,
        Hinv_y=y,
        Hinv_covar=covar,
        Hinv_z_active=z_active,
        phenotype_scale=float(np.std(y)),
    )

    assert out["beta_cov"].shape == (1,)
    assert out["beta_active_basis"].shape == (0,)
    assert out["active_basis_idx"].shape == (0,)
    assert out["q_plugin_raw"] == 0.0
    assert out["q_df_corrected_raw"] == 0.0


def test_selected_span_basis_expansion_rejects_invalid_positions():
    with np.testing.assert_raises(ValueError):
        SPARSE._expand_selected_basis_coefficients(
            support_size=3,
            basis_positions=np.asarray([1, 1]),
            basis_coefficients=np.asarray([0.2, 0.3]),
        )
    with np.testing.assert_raises(ValueError):
        SPARSE._expand_selected_basis_coefficients(
            support_size=3,
            basis_positions=np.asarray([3]),
            basis_coefficients=np.asarray([0.2]),
        )


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
