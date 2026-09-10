"""Dense REML identities and the actual joint quadratic calibration backend."""
import importlib
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
SCORE = importlib.import_module(f"{ROOT.name}.score_process")


def sketch(directions, h, b):
    core = h.T @ directions @ h
    cross = h.T @ directions @ b
    bulk = b.T @ directions @ b
    diagonal = np.diagonal(bulk, axis1=1, axis2=2).copy()
    indices = np.arange(b.shape[1])
    bulk[:, indices, indices] = 0
    return SCORE.TraceSketch(core, cross, bulk, diagonal)


def full_core(directions, p):
    values, basis = np.linalg.eigh(p)
    h = basis * np.sqrt(np.maximum(values, 0))
    return sketch(directions, h, np.zeros((len(p), 3)))


@pytest.mark.parametrize("contrast_scale", [1.0, 1e-12, 1e12])
def test_full_core_matches_dense_reml_score_and_joint_covariance(contrast_scale):
    rng = np.random.default_rng(415)
    n = 12
    root = rng.normal(size=(n, n))
    k = root @ root.T / n
    v = 0.4 * k + 0.7 * np.eye(n)
    vi = np.linalg.inv(v)
    c = np.ones((n, 1))
    p = vi - vi @ c @ np.linalg.solve(c.T @ vi @ c, c.T @ vi)
    directions = [k, np.eye(n)]
    for _ in range(3):
        block = rng.normal(size=(n, 4))
        directions.append(contrast_scale * (block @ block.T / 4 - k))
    directions = np.stack(directions)
    r = rng.normal(size=n)
    py = p @ r
    quadratics = np.einsum("i,aij,j->a", py, directions, py)
    pd = p @ directions
    fisher = 0.5 * np.einsum("aij,bji->ab", pd, pd)
    raw = 0.5 * (quadratics - np.trace(pd, axis1=1, axis2=2))
    coefficients = np.linalg.solve(fisher[:2, :2], fisher[:2, 2:]).T
    expected_covariance = fisher[2:, 2:] - coefficients @ fisher[:2, 2:]
    exact = full_core(directions, p)
    actual_coef, _ = SCORE.fit_nuisance_projection(exact, 2)
    process = SCORE.score_process_moments(
        quadratics, evaluation=exact, coefficients=actual_coef, boundary_positions=np.array([10, 20, 30]),
    )
    np.testing.assert_allclose(exact.information_rows(2), fisher[:2], rtol=1e-11, atol=1e-23)
    np.testing.assert_allclose(process.scores / contrast_scale,
                               (raw[2:] - coefficients @ raw[:2]) / contrast_scale, atol=1e-11)
    np.testing.assert_allclose(process.information / contrast_scale**2,
                               np.diag(expected_covariance) / contrast_scale**2, atol=1e-11)
    covariance = 0.5 * np.einsum("aij,bji->ab", process.reference_matrices, process.reference_matrices)
    scale = np.sqrt(np.diag(expected_covariance))
    np.testing.assert_allclose(covariance, expected_covariance / scale[:, None] / scale, atol=1e-12)
    assert process.diagnostics["max_score_trace_standard_error"] == 0


def test_independent_evaluation_uses_actual_direction_not_second_schur():
    # Pilot F=[[1,.8],[.8,1]], independent evaluation F=I. The actual
    # direction has variance 1+.8², neither pilot Schur .36 nor eval Schur 1.
    a = np.diag([np.sqrt(2), 0.0])
    d = np.diag([0.0, np.sqrt(2)])
    pilot = full_core(np.stack([a, 0.8*a + 0.6*d]), np.eye(2))
    evaluation = full_core(np.stack([a, d]), np.eye(2))
    coefficients, _ = SCORE.fit_nuisance_projection(pilot, 1)
    process = SCORE.score_process_moments(
        np.array([2.0, 6.0]), evaluation=evaluation, coefficients=coefficients,
        boundary_positions=np.array([10]),
    )
    assert coefficients[0, 0] == pytest.approx(0.8)
    assert process.information[0] == pytest.approx(1.64)
    assert process.scores[0] == pytest.approx(
        0.5 * ((6 - np.sqrt(2)) - 0.8 * (2 - np.sqrt(2))))


def test_finite_probe_gram_is_psd_and_matches_reference_covariance_and_jackknife():
    rng = np.random.default_rng(42)
    n, count = 9, 13
    h, b = rng.normal(size=(n, 3)), rng.normal(size=(n, count))
    directions = rng.normal(size=(5, n, n))
    directions = (directions + directions.swapaxes(1, 2)) / 2
    pilot = sketch(directions, h, rng.normal(size=b.shape))
    evaluation = sketch(directions, h, b)
    coefficients, _ = SCORE.fit_nuisance_projection(pilot, 2)
    process = SCORE.score_process_moments(
        np.zeros(5), evaluation=evaluation, coefficients=coefficients,
        boundary_positions=np.array([1, 2, 3]),
    )
    fisher = evaluation.information_rows(5)
    assert np.linalg.eigvalsh(fisher)[0] >= -1e-10
    transform = np.column_stack([-coefficients, np.eye(3)])
    expected = transform @ fisher @ transform.T
    scale = np.sqrt(np.diag(expected))
    matrices = process.reference_matrices
    np.testing.assert_allclose(0.5*np.einsum("aij,bji->ab", matrices, matrices),
                               expected / scale[:, None] / scale, atol=1e-12)
    adjusted = directions[2:] - np.einsum("ta,aij->tij", coefficients, directions[:2])
    deleted = np.stack([
        np.diag(sketch(adjusted, h, np.delete(b, i, axis=1)).information_rows(3))
        for i in range(count)
    ])
    se = np.sqrt((count - 1) / count * np.sum((deleted - deleted.mean(0))**2, axis=0))
    assert process.diagnostics["max_information_relative_standard_error"] == pytest.approx(
        np.max(se / np.diag(expected)), rel=1e-11)


def test_zero_and_nuisance_span_directions_are_untestable():
    p = np.diag([0.5, 1, 2, 3.0])
    a = np.diag([1, 2, 1, 2.0])
    directions = np.stack([a, np.eye(4), 2*a - np.eye(4), np.zeros((4, 4))])
    exact = full_core(directions, p)
    coefficients, _ = SCORE.fit_nuisance_projection(exact, 2)
    process = SCORE.score_process_moments(
        np.zeros(4), evaluation=exact, coefficients=coefficients, boundary_positions=np.array([2, 4]),
    )
    np.testing.assert_array_equal(process.information, 0)
    np.testing.assert_array_equal(process.reference_matrices, 0)
    rows, diagnostics = SCORE.calibrate_boundary_maximum(process)
    assert rows == []
    assert diagnostics["global_p_value"] == 1
    assert not diagnostics["accepted"]
    assert diagnostics["reference_samples"] == 0


def test_redundant_nuisance_directions_use_psd_pseudoinverse():
    a, d = np.diag([1.0, 2, 3]), np.diag([3.0, -1, 2])
    exact = full_core(np.stack([a, 2*a, np.zeros_like(a), d]), np.eye(3))
    coefficients, eigenvalues = SCORE.fit_nuisance_projection(exact, 3)
    adjusted = d - coefficients[0, 0]*a - coefficients[0, 1]*2*a
    assert np.trace(a @ adjusted) == pytest.approx(0, abs=1e-12)
    assert np.count_nonzero(eigenvalues > 1e-10) == 1


def make_process(scores, matrices=None):
    scores = np.asarray(scores, dtype=float)
    if matrices is None:
        matrices = np.full((len(scores), 1, 1), np.sqrt(2.0))
    return SCORE.ScoreProcess(np.arange(len(scores)) + 1, scores, scores, np.ones_like(scores),
                              np.zeros((len(scores), 1)), matrices, {})


def test_single_rank_one_reference_matches_chi_square_tail():
    process = make_process([3.0])
    rows, result = SCORE.calibrate_boundary_maximum(process, samples=65535, seed=910)
    expected = chi2.sf(1 + 3*np.sqrt(2), 1)
    assert result["global_p_value"] == pytest.approx(expected, abs=0.002)
    assert rows[0]["standardized_score"] == 3
    assert result["calibration_method"] == "joint_core_probe_quadratic"


def test_duplicate_and_sign_reversed_candidates_do_not_increase_penalty():
    _, single = SCORE.calibrate_boundary_maximum(make_process([3.0]), samples=4095, seed=71)
    process = make_process([3.0, -3.0, 3.0], np.array([[[np.sqrt(2)]], [[-np.sqrt(2)]], [[np.sqrt(2)]]]))
    rows, repeated = SCORE.calibrate_boundary_maximum(process, samples=4095, seed=71)
    assert single["global_p_value"] == repeated["global_p_value"]
    assert single["reference_critical_value"] == repeated["reference_critical_value"]
    assert rows[0]["boundary_position"] == 1


def test_selection_uses_maximum_absolute_score_and_rank_p_never_zero():
    rows, diagnostics = SCORE.calibrate_boundary_maximum(make_process([4, -100, 2]), samples=1023, seed=42)
    assert [row["boundary_position"] for row in rows] == [2, 1, 3]
    assert diagnostics["maximum_absolute_score"] == 100
    assert diagnostics["global_p_value"] == 1 / 1024
    assert diagnostics["accepted"]


def test_integration_matches_dense_joint_law_and_is_batch_invariant(monkeypatch):
    rng = np.random.default_rng(602)
    matrices = rng.normal(size=(5, 7, 7))
    matrices = (matrices + matrices.swapaxes(1, 2)) / 8
    coordinates = np.random.default_rng(211).standard_normal((1031, 7), dtype=np.float32)
    expected = np.max(np.abs(0.5 * (
        np.einsum("bi,tij,bj->bt", coordinates, matrices, coordinates)
        - np.trace(matrices, axis1=1, axis2=2))), axis=1)
    actual = SCORE.integrate_boundary_maximum(matrices, samples=1031, seed=211, batch_size=128)
    monkeypatch.setattr(SCORE, "direction_batch_size", lambda _: 2)
    batched = SCORE.integrate_boundary_maximum(matrices, samples=1031, seed=211, batch_size=73)
    np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=3e-6)
    np.testing.assert_allclose(batched, expected, atol=3e-6, rtol=3e-6)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_sketch_is_rejected(bad):
    with pytest.raises(ValueError, match="finite"):
        SCORE.TraceSketch(np.full((2, 2, 2), bad), np.zeros((2, 2, 3)),
                          np.zeros((2, 3, 3)), np.zeros((2, 3)))


def test_overflowed_information_is_an_error_not_an_empty_family():
    huge = full_core(np.full((2, 2, 2), 1e200), np.eye(2))
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(FloatingPointError, match="Non-finite"):
            SCORE.fit_nuisance_projection(huge, 1)
        with pytest.raises(FloatingPointError, match="Non-finite"):
            SCORE.score_process_moments(
                np.zeros(2), evaluation=huge, coefficients=np.ones((1, 1)),
                boundary_positions=np.array([1]),
            )


@pytest.mark.parametrize("alpha,samples", [(0, 1023), (1, 1023), (0.0001, 255)])
def test_invalid_or_unresolvable_alpha_is_rejected(alpha, samples):
    with pytest.raises(ValueError, match="alpha"):
        SCORE.calibrate_boundary_maximum(make_process([3]), alpha=alpha, samples=samples)
