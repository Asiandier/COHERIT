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
ADAPTIVE = importlib.import_module(f"{PKG}.adaptive_partition")
COVTREE = importlib.import_module(f"{PKG}.covtree")
COVARIANCE_SCORE = importlib.import_module(f"{PKG}.covariance_score")
GENO_STREAM = importlib.import_module(f"{PKG}.geno_stream")


class _ArraySource:
    def __init__(self, genotype: np.ndarray):
        self._genotype = np.asarray(genotype, dtype=np.int8)
        self.n, self.m = self._genotype.shape
        self.missing_val = -9

    def read_block_variant_major(self, snp_start: int, snp_count: int):
        return np.asfortranarray(
            self._genotype[:, snp_start : snp_start + snp_count].T
        )

    def close(self):
        return None


def test_covtree_candidates_are_genotype_only_deterministic_and_complete():
    components = ADAPTIVE.single_component(12)
    ld = np.asarray([1.0] * 6 + [20.0] * 6)
    heterozygosity = np.asarray(([0.1, 0.4] * 6), dtype=np.float64)

    first, rejected = COVTREE.generate_covtree_candidates(
        components,
        ld_score=ld,
        heterozygosity=heterozygosity,
        min_child_markers=2,
    )
    second, _ = COVTREE.generate_covtree_candidates(
        components,
        ld_score=ld,
        heterozygosity=heterozygosity,
        min_child_markers=2,
    )

    assert {
        (item["split_kind"], item["reason"].split(":", 1)[0])
        for item in rejected
    } == {
        ("ld4_tree", "recursive_feature_split_failed"),
        ("maf4_tree", "recursive_feature_split_failed"),
    }
    assert [candidate.split_kind for candidate in first] == [
        "ld2",
        "maf2",
        "ld_maf4",
    ]
    assert all(
        all(np.array_equal(a, b) for a, b in zip(left.children, right.children))
        for left, right in zip(first, second)
    )
    assert [child.size for child in first[2].children] == [3, 3, 3, 3]
    for candidate in first:
        assert np.array_equal(
            np.sort(np.concatenate(candidate.children)), np.arange(12)
        )


def test_constant_feature_does_not_create_arbitrary_or_suppress_valid_candidates():
    candidates, rejected = COVTREE.generate_covtree_candidates(
        ADAPTIVE.single_component(8),
        ld_score=np.ones(8),
        heterozygosity=np.linspace(0.1, 0.4, 8),
        min_child_markers=2,
    )

    assert [candidate.split_kind for candidate in candidates] == ["maf2"]
    reasons = {(item["split_kind"], item["reason"]) for item in rejected}
    assert ("ld2", "feature_constant_within_parent") in reasons
    assert ("ld_maf4", "joint_feature_unavailable") in reasons


def test_selective_split_and_warm_start_replace_only_one_parent():
    roots = ADAPTIVE.single_component(8)
    first_candidates, _ = COVTREE.generate_covtree_candidates(
        roots,
        ld_score=np.asarray([0.0] * 4 + [10.0] * 4),
        heterozygosity=np.linspace(0.1, 0.4, 8),
        min_child_markers=1,
    )
    first_split = COVTREE.replace_parent(roots, first_candidates[0])
    parent_theta = np.asarray([0.2, 0.4, 0.4])

    second_candidates, _ = COVTREE.generate_covtree_candidates(
        first_split,
        ld_score=np.arange(8, dtype=np.float64),
        heterozygosity=np.linspace(0.1, 0.4, 8),
        min_child_markers=1,
    )
    candidate = next(item for item in second_candidates if item.parent_index == 0)
    updated = COVTREE.replace_parent(first_split, candidate)
    warm = COVTREE.selective_covariance_warm_start(
        parent_theta, first_split, candidate
    )
    effective_warm = COVTREE.selective_covariance_warm_start(
        parent_theta,
        first_split,
        candidate,
        child_effective_markers=np.ones(len(candidate.children)),
    )

    assert len(updated) == len(first_split) - 1 + len(candidate.children)
    assert warm.shape == (len(updated) + 1,)
    assert np.isclose(np.sum(warm[:-1]), np.sum(parent_theta[:-1]))
    assert warm[-1] == parent_theta[-1]
    np.testing.assert_allclose(
        effective_warm[: len(candidate.children)],
        parent_theta[0] / len(candidate.children),
    )
    assert updated[-1].name == first_split[-1].name


def test_recursive_four_way_candidate_can_replace_parent():
    components = ADAPTIVE.single_component(16)
    candidates, _ = COVTREE.generate_covtree_candidates(
        components,
        ld_score=np.expm1(np.arange(16, dtype=np.float64) / 4.0),
        heterozygosity=np.asarray(([0.1, 0.4] * 8), dtype=np.float64),
        min_child_markers=1,
    )

    candidate = next(item for item in candidates if item.split_kind == "ld4_tree")
    updated = COVTREE.replace_parent(components, candidate)

    assert len(updated) == 4
    assert [component.variant_indices.size for component in updated] == [4] * 4
    assert [component.annotation["child_label"] for component in updated] == [
        "ll",
        "lh",
        "hl",
        "hh",
    ]


def test_trace_orthogonal_contrasts_have_full_expected_rank():
    trace = np.asarray([0.98, 1.01, 1.00, 0.99])
    contrasts = COVTREE.trace_orthogonal_contrasts(trace)

    assert contrasts.shape == (4, 3)
    assert np.linalg.matrix_rank(contrasts) == 3
    np.testing.assert_allclose(trace @ contrasts, 0.0, atol=1e-10)
    np.testing.assert_allclose(contrasts.T @ contrasts, np.eye(3), atol=1e-10)


def test_bootstrap_max_score_selects_strong_candidate():
    rng = np.random.default_rng(19)
    bootstrap_scores = rng.multivariate_normal(
        np.zeros(2),
        np.asarray(
            [
                [1.0, 0.1],
                [0.1, 1.0],
            ]
        ),
        size=2000,
    ).T
    trace = np.asarray([12.0, 14.0])
    bootstrap_quadratics = trace[:, None] + 2.0 * bootstrap_scores
    observed_scores = np.asarray([5.0, 0.2])
    observed_quadratics = trace + 2.0 * observed_scores

    result = COVTREE.bootstrap_max_score_statistics(
        observed_quadratics=observed_quadratics,
        bootstrap_quadratics=bootstrap_quadratics,
        candidate_slices=[(0, 1), (1, 2)],
    )

    assert result["best_candidate_index"] == 0
    assert result["max_score_adjusted_p"] < 0.01
    assert all(candidate["rank_sufficient"] for candidate in result["candidates"])


def test_partitioned_null_sampler_matches_dense_target_covariance():
    genotype = np.asarray(
        [
            [0, 0, 1, 2],
            [0, 1, 1, 2],
            [1, 1, 2, 1],
            [2, 2, 0, 0],
            [2, 1, 2, 0],
        ],
        dtype=np.int8,
    )
    streamer = GENO_STREAM.GenoBlockStreamer(
        source=_ArraySource(genotype),
        component_variant_indices=[np.asarray([0, 1]), np.asarray([2, 3])],
        call_width=2,
        device="cpu",
        keep_host_stats=True,
    )
    try:
        theta = np.asarray([0.2, 0.3, 0.5])
        draws = COVARIANCE_SCORE.sample_partitioned_null_residuals(
            streamer,
            theta=theta,
            n_draws=4000,
            seed=23,
        )
        standardized = streamer.extract_standardized_columns(np.arange(4))
        expected = (
            theta[0] * standardized[:, :2] @ standardized[:, :2].T / 2.0
            + theta[1] * standardized[:, 2:] @ standardized[:, 2:].T / 2.0
            + theta[2] * np.eye(genotype.shape[0])
        )
        observed = np.cov(draws, bias=True)

        assert np.max(np.abs(np.mean(draws, axis=1))) < 0.05
        np.testing.assert_allclose(observed, expected, rtol=0.08, atol=0.05)
    finally:
        streamer.close()
