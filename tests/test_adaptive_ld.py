import importlib
import os
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT = os.path.dirname(REPO_ROOT)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)
PKG = importlib.import_module(os.path.basename(REPO_ROOT))
ADAPTIVE = importlib.import_module(f"{PKG.__name__}.adaptive_ld")
LD_SCORE = importlib.import_module(f"{PKG.__name__}.ld_score")
add_ld_boundary = ADAPTIVE.add_ld_boundary
efficient_score_statistics = ADAPTIVE.efficient_score_statistics
existing_boundary_positions = ADAPTIVE.existing_boundary_positions
load_component_groups = ADAPTIVE.load_component_groups
load_ld_rank = ADAPTIVE.load_ld_rank
write_root_component_spec = ADAPTIVE.write_root_component_spec
write_ld_rank_artifact = LD_SCORE.write_ld_rank_artifact


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


def test_efficient_score_statistics_returns_global_bootstrap_gate() -> None:
    rng = np.random.default_rng(7)
    # Two nuisance quadratics and two candidate quadratics; first column is
    # observed, remaining 39 columns are parametric-bootstrap draws.
    quadratics = rng.normal(size=(4, 40))
    quadratics[2, 0] += 8.0
    rows, diagnostics = efficient_score_statistics(
        quadratics,
        nuisance_count=2,
        boundary_positions=np.asarray([10, 20]),
    )
    assert {row["boundary_position"] for row in rows} == {10, 20}
    assert rows[0]["boundary_position"] == 10
    assert 0.0 < diagnostics["global_sup_score_p_value"] <= 1.0


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
