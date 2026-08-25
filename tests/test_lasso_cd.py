"""
Pure numerical tests for lasso_cd.py.

These tests avoid GPU, streamer, and bed-file dependencies. They cover:
- coordinate-descent correctness
- lambda=0 -> OLS / GLS behavior
- KKT conditions for LASSO solutions
- EBIC basic monotonic behavior in simple controlled settings
"""

import importlib
import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PARENT = os.path.dirname(_REPO_ROOT)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
_PKG = os.path.basename(_REPO_ROOT)
_LASSO = importlib.import_module(f"{_PKG}.lasso_cd")

LassoPathConfig = _LASSO.LassoPathConfig
compute_projected_hinv_vector = _LASSO.compute_projected_hinv_vector
ebic_from_rss = _LASSO.ebic_from_rss
fit_weighted_lasso_with_covariates = _LASSO.fit_weighted_lasso_with_covariates
make_lambda_sequence = _LASSO.make_lambda_sequence
solve_lasso_cd_gram = _LASSO.solve_lasso_cd_gram
solve_lasso_path_and_select_ebic = _LASSO.solve_lasso_path_and_select_ebic


def _random_spd(k: int, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    A = rng.randn(k, k)
    return A.T @ A + 0.5 * np.eye(k)


class TestSolveLassoCdGram:
    def test_lambda_zero_matches_ols_solution(self):
        Q = _random_spd(6, seed=1)
        q = np.array([1.2, -0.4, 0.7, 2.1, -1.5, 0.3], dtype=np.float64)
        beta, _, _, converged = solve_lasso_cd_gram(
            Q, q, 0.0, max_iter=5000, tol=1e-10
        )
        beta_ref = np.linalg.solve(Q, q)
        np.testing.assert_allclose(beta, beta_ref, rtol=1e-6, atol=1e-7)
        assert converged

    def test_large_lambda_gives_all_zero(self):
        Q = _random_spd(5, seed=2)
        q = np.array([0.5, -1.0, 2.0, -0.75, 1.25], dtype=np.float64)
        lam = float(np.max(np.abs(q)))
        beta, Qb, _, converged = solve_lasso_cd_gram(Q, q, lam, max_iter=200, tol=1e-10)
        np.testing.assert_allclose(beta, 0.0, atol=0.0)
        np.testing.assert_allclose(Qb, 0.0, atol=0.0)
        assert converged

    def test_kkt_conditions_hold(self):
        Q = _random_spd(7, seed=3)
        q = np.array([1.5, -2.0, 0.25, 0.5, -0.1, 0.75, -1.2], dtype=np.float64)
        lam = 0.6
        beta, Qb, _, converged = solve_lasso_cd_gram(Q, q, lam, max_iter=5000, tol=1e-10)
        assert converged
        grad = Qb - q
        for j in range(beta.size):
            if abs(beta[j]) > 1e-8:
                stationarity = grad[j] + lam * np.sign(beta[j])
                assert abs(stationarity) < 1e-5
            else:
                assert abs(grad[j]) <= lam + 1e-5

    def test_warm_start_does_not_change_solution(self):
        Q = _random_spd(6, seed=4)
        q = np.array([0.3, -0.7, 1.1, -1.4, 0.9, 0.2], dtype=np.float64)
        lam = 0.4
        beta1, _, _, _ = solve_lasso_cd_gram(Q, q, lam, max_iter=5000, tol=1e-10)
        beta0 = np.linspace(-0.5, 0.5, 6)
        beta2, _, _, _ = solve_lasso_cd_gram(
            Q, q, lam, beta0=beta0, max_iter=5000, tol=1e-10
        )
        np.testing.assert_allclose(beta1, beta2, rtol=1e-6, atol=1e-7)

    def test_active_set_period_does_not_change_solution(self):
        Q = _random_spd(8, seed=9)
        q = np.array([0.8, -0.5, 1.4, -1.1, 0.6, 0.2, -0.3, 0.9], dtype=np.float64)
        lam = 0.35
        beta1, Qb1, _, converged1 = solve_lasso_cd_gram(
            Q, q, lam, max_iter=5000, tol=1e-10, active_set_period=1
        )
        beta2, Qb2, _, converged2 = solve_lasso_cd_gram(
            Q, q, lam, max_iter=5000, tol=1e-10, active_set_period=7
        )
        assert converged1
        assert converged2
        np.testing.assert_allclose(beta1, beta2, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(Qb1, Qb2, rtol=1e-6, atol=1e-7)

    def test_converged_requires_score_kkt_on_scaled_correlated_gram(self):
        """A small coefficient sweep is not enough at genomic Gram scales."""
        k = 128
        lam = 550.0
        scale = 8.0e4
        correlation = 0.2
        Q = scale * (
            (1.0 - correlation) * np.eye(k)
            + correlation * np.ones((k, k))
        )
        q = np.random.default_rng(7).normal(0.0, 700.0, k)
        kkt_abs_tol = 1e-4
        kkt_rel_tol = 1e-4
        tolerance = max(kkt_abs_tol, kkt_rel_tol * lam)

        beta, Qb, _, converged = solve_lasso_cd_gram(
            Q,
            q,
            lam,
            max_iter=5000,
            tol=1e-6,
            active_set_period=5,
            kkt_abs_tol=kkt_abs_tol,
            kkt_rel_tol=kkt_rel_tol,
        )

        score = q - Qb
        active = beta != 0.0
        active_error = (
            np.max(np.abs(score[active] - lam * np.sign(beta[active])))
            if np.any(active)
            else 0.0
        )
        inactive_excess = (
            max(np.max(np.abs(score[~active])) - lam, 0.0)
            if np.any(~active)
            else 0.0
        )
        assert converged
        assert active_error <= tolerance
        assert inactive_excess <= tolerance


class TestLambdaPathAndEbic:
    def test_make_lambda_sequence_descending(self):
        seq = make_lambda_sequence(10.0, 0.1, 5)
        assert seq.shape == (5,)
        assert np.all(seq[:-1] >= seq[1:])
        np.testing.assert_allclose(seq[0], 10.0)
        np.testing.assert_allclose(seq[-1], 1.0)

    def test_ebic_increases_with_k_when_rss_fixed(self):
        e1 = ebic_from_rss(n=1000, p=10000, k=1, rss=500.0, gamma=0.5, eps=1e-12)
        e2 = ebic_from_rss(n=1000, p=10000, k=10, rss=500.0, gamma=0.5, eps=1e-12)
        assert e1 < e2

    def test_path_result_matches_min_ebic_in_path(self):
        Q = _random_spd(5, seed=5)
        q = np.array([1.0, -0.8, 0.6, 0.0, 0.2], dtype=np.float64)
        out = solve_lasso_path_and_select_ebic(
            Q=Q,
            q=q,
            yHy=10.0,
            n_samples=200,
            p_total=1000,
            cfg=LassoPathConfig(n_lambda=12, lam_min_ratio=0.2, max_cd_iter=5000, cd_tol=1e-10),
        )
        path = out["path"]
        best = min(path, key=lambda d: (d["ebic"], d["k"]))
        assert abs(best["lam"] - out["lam"]) < 1e-12
        assert abs(best["ebic"] - out["best_ebic"]) < 1e-12
        assert all("kkt_passed" in row for row in path)
        assert all(
            row["converged"] <= row["kkt_passed"]
            for row in path
        )
        assert out["selection_mode"] == "ebic"
        assert out["beta_path"].shape == (len(path), q.size)
        np.testing.assert_allclose(
            out["beta"], out["beta_path"][out["selected_index"]]
        )

    def test_fixed_ratio_selection_uses_requested_path_location(self):
        Q = _random_spd(5, seed=12)
        q = np.array([1.2, -0.9, 0.55, 0.25, -0.1], dtype=np.float64)
        out = solve_lasso_path_and_select_ebic(
            Q=Q,
            q=q,
            yHy=15.0,
            n_samples=250,
            p_total=5000,
            cfg=LassoPathConfig(
                n_lambda=9,
                lam_min_ratio=0.01,
                selection_mode="fixed_ratio",
                fixed_lam_ratio=0.1,
                ebic_early_stop=True,
                max_cd_iter=5000,
                cd_tol=1e-10,
            ),
        )

        assert len(out["path"]) == 9
        assert out["selection_mode"] == "fixed_ratio"
        assert out["selected_lam_ratio"] == pytest.approx(0.1)
        assert out["path"][out["selected_index"]]["lam_ratio"] == pytest.approx(
            0.1
        )
        np.testing.assert_allclose(
            out["beta"], out["beta_path"][out["selected_index"]]
        )

    def test_fixed_ratio_selection_validates_ratio(self):
        with pytest.raises(ValueError, match="fixed_lam_ratio"):
            solve_lasso_path_and_select_ebic(
                Q=np.eye(2),
                q=np.ones(2),
                yHy=5.0,
                n_samples=20,
                p_total=100,
                cfg=LassoPathConfig(
                    selection_mode="fixed_ratio",
                    fixed_lam_ratio=0.0,
                ),
            )

    def test_path_clamps_tiny_negative_rss(self, monkeypatch):
        def _fake_solve_lasso_cd_gram(Q, q, lam, **_kwargs):
            del Q, q, lam
            beta = np.array([0.0], dtype=np.float64)
            Qb = np.array([0.0], dtype=np.float64)
            return beta, Qb, 1, True

        monkeypatch.setattr(_LASSO, "solve_lasso_cd_gram", _fake_solve_lasso_cd_gram)
        out = solve_lasso_path_and_select_ebic(
            Q=np.array([[1.0]], dtype=np.float64),
            q=np.array([1.0], dtype=np.float64),
            yHy=-1e-12,
            n_samples=20,
            p_total=1,
            cfg=LassoPathConfig(n_lambda=1, lam_min_ratio=1.0, ebic_early_stop=False),
        )

        assert out["path"][0]["rss"] == 0.0
        assert np.isfinite(out["best_ebic"])


class TestProjectedHinvVector:
    def test_projection_is_orthogonal_to_covariates_under_identity_metric(self):
        rng = np.random.RandomState(6)
        n, p = 100, 3
        C = rng.randn(n, p)
        target = rng.randn(n)
        projected = compute_projected_hinv_vector(
            covar=C,
            Hinv_covar=C,
            Hinv_target=target,
            ridge=1e-8,
        )
        ortho = C.T @ projected
        np.testing.assert_allclose(ortho, 0.0, atol=1e-6)


class TestFitWeightedLassoWithCovariates:
    def test_huge_lambda_reduces_to_covariate_only_gls(self):
        rng = np.random.RandomState(7)
        n, p_c, p_z = 80, 2, 4
        C = rng.randn(n, p_c).astype(np.float32)
        Z = rng.randn(n, p_z).astype(np.float32)
        beta_c_true = np.array([1.5, -0.5], dtype=np.float64)
        y = (C @ beta_c_true + 0.01 * rng.randn(n)).astype(np.float32)

        out = fit_weighted_lasso_with_covariates(
            y=y,
            covar=C,
            geno=Z,
            Hinv_y=y,
            Hinv_covar=C,
            Hinv_geno=Z,
            p_total=1000,
            cfg=LassoPathConfig(
                n_lambda=1,
                lam_min_ratio=1.0,
                max_cd_iter=5000,
                cd_tol=1e-10,
                ebic_early_stop=False,
            ),
            ridge=1e-8,
        )

        beta_cov_ref = np.linalg.solve(C.T @ C + 1e-8 * np.eye(p_c), C.T @ y)
        np.testing.assert_allclose(out["beta_snp"], 0.0, atol=0.0)
        np.testing.assert_allclose(out["beta_cov"], beta_cov_ref, rtol=1e-5, atol=1e-6)
        assert out["active_idx"].size == 0
        assert out["beta_snp_path"].shape == (1, p_z)
        assert out["beta_cov_path"].shape == (1, p_c)
        np.testing.assert_allclose(out["beta_cov_path"][0], out["beta_cov"])

    def test_no_covariate_single_lambda_path_starts_at_zero_model(self):
        rng = np.random.RandomState(8)
        n, p_z = 120, 5
        Z = rng.randn(n, p_z).astype(np.float32)
        beta_true = np.array([1.0, 0.0, -0.5, 0.25, 0.0], dtype=np.float64)
        y = (Z @ beta_true + 0.01 * rng.randn(n)).astype(np.float32)

        out = fit_weighted_lasso_with_covariates(
            y=y,
            covar=None,
            geno=Z,
            Hinv_y=y,
            Hinv_covar=None,
            Hinv_geno=Z,
            p_total=p_z,
            cfg=LassoPathConfig(
                n_lambda=1,
                lam_min_ratio=1.0,
                max_cd_iter=5000,
                cd_tol=1e-10,
                ebic_early_stop=False,
            ),
            ridge=1e-8,
        )

        assert float(np.max(np.abs(out["beta_snp"]))) < 1e-12
        assert out["beta_cov"].size == 0

    def test_float64_hinv_inputs_match_direct_weighted_system(self):
        rng = np.random.RandomState(10)
        n, p_c, p_z = 64, 3, 5
        C = rng.randn(n, p_c)
        Z = rng.randn(n, p_z).astype(np.float32)
        W_diag = 0.5 + rng.rand(n)
        Hinv = np.diag(W_diag)
        beta_c_true = np.array([0.7, -1.1, 0.3], dtype=np.float64)
        beta_z_true = np.array([1.2, 0.0, -0.4, 0.0, 0.5], dtype=np.float64)
        y = C @ beta_c_true + Z.astype(np.float64) @ beta_z_true + 0.01 * rng.randn(n)
        Hy = Hinv @ y
        HC = Hinv @ C
        HZ = (Hinv @ Z.astype(np.float64)).astype(np.float32)

        out = fit_weighted_lasso_with_covariates(
            y=y.astype(np.float32),
            covar=C.astype(np.float32),
            geno=Z,
            Hinv_y=Hy,
            Hinv_covar=HC,
            Hinv_geno=HZ,
            p_total=500,
            cfg=LassoPathConfig(
                n_lambda=1,
                lam_min_ratio=1.0,
                max_cd_iter=5000,
                cd_tol=1e-10,
                ebic_early_stop=False,
            ),
            ridge=1e-8,
        )

        Z64 = Z.astype(np.float64)
        GCC = C.T @ HC + 1e-8 * np.eye(p_c)
        GCZ = C.T @ (Hinv @ Z64)
        GZZ = Z64.T @ (Hinv @ Z64)
        gCy = C.T @ Hy
        gZy = Z64.T @ Hy
        Ainv_gCy = np.linalg.solve(GCC, gCy)
        Ainv_GCZ = np.linalg.solve(GCC, GCZ)
        Q = 0.5 * ((GZZ - GCZ.T @ Ainv_GCZ) + (GZZ - GCZ.T @ Ainv_GCZ).T)
        q = gZy - GCZ.T @ Ainv_gCy
        lam = float(np.max(np.abs(q)))
        beta_ref = np.zeros(p_z, dtype=np.float64)
        beta_cov_ref = np.linalg.solve(GCC, gCy - GCZ @ beta_ref)

        np.testing.assert_allclose(out["lam"], lam, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(out["beta_snp"], beta_ref, atol=1e-12)
        np.testing.assert_allclose(out["beta_cov"], beta_cov_ref, rtol=1e-6, atol=1e-7)

    def test_float64_inputs_are_not_downcast_before_gram_build(self, monkeypatch):
        rng = np.random.RandomState(11)
        n, p_c, p_z = 12, 2, 3
        y = rng.randn(n).astype(np.float64)
        C = rng.randn(n, p_c).astype(np.float64)
        Z = rng.randn(n, p_z).astype(np.float64)
        Hy = rng.randn(n).astype(np.float64)
        HC = rng.randn(n, p_c).astype(np.float64)
        HZ = rng.randn(n, p_z).astype(np.float64)
        orig_asarray = _LASSO.np.asarray
        seen: dict[str, object] = {}

        def _record_asarray(a, dtype=None, *args, **kwargs):
            if a is y:
                seen["y_dtype"] = dtype
            elif a is Z:
                seen["geno_dtype"] = dtype
            return orig_asarray(a, dtype=dtype, *args, **kwargs)

        monkeypatch.setattr(_LASSO.np, "asarray", _record_asarray)

        fit_weighted_lasso_with_covariates(
            y=y,
            covar=C,
            geno=Z,
            Hinv_y=Hy,
            Hinv_covar=HC,
            Hinv_geno=HZ,
            p_total=100,
            cfg=LassoPathConfig(
                n_lambda=1,
                lam_min_ratio=1.0,
                max_cd_iter=100,
                cd_tol=1e-8,
                ebic_early_stop=False,
            ),
            ridge=1e-8,
        )

        assert seen["y_dtype"] is np.float64
        assert seen["geno_dtype"] is np.float64
