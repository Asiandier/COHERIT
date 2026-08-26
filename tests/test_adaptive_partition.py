from __future__ import annotations

import importlib
import os
import sys

import numpy as np
import pytest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT = os.path.dirname(REPO_ROOT)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

PKG = os.path.basename(REPO_ROOT)
ADAPTIVE = importlib.import_module(f"{PKG}.adaptive_partition")
COMPONENT_SPEC = importlib.import_module(f"{PKG}.component_spec")


def test_four_way_split_uses_exact_two_means_and_bin_specific_ld_medians():
    parents = ADAPTIVE.single_component(100)
    signal = np.arange(100, dtype=np.float64)
    ld = np.arange(100, dtype=np.float64)

    children = ADAPTIVE.four_way_split(
        parents,
        signal_score=signal,
        ld_score=ld,
        child_depth=1,
    )

    assert len(children) == 4
    assert [child.variant_indices.size for child in children] == [25, 25, 25, 25]
    high_signal = np.concatenate(
        [children[0].variant_indices, children[1].variant_indices]
    )
    low_signal = np.concatenate(
        [children[2].variant_indices, children[3].variant_indices]
    )
    assert np.array_equal(np.sort(high_signal), np.arange(50, 100))
    assert np.array_equal(np.sort(low_signal), np.arange(50))
    assert children[0].annotation["signal_split_method"] == "exact_1d_two_means"
    assert children[0].annotation["signal_high_fraction_realized"] == pytest.approx(
        0.5
    )
    assert children[0].annotation["signal_cluster_boundary"] == pytest.approx(49.5)
    assert children[0].annotation["ld_median_within_signal_bin"] == 74.5
    assert children[2].annotation["ld_median_within_signal_bin"] == 24.5
    ADAPTIVE.validate_partition(children, n_variants=100)


def test_four_way_split_is_deterministic_under_score_ties():
    parents = ADAPTIVE.single_component(40)
    signal = np.ones(40, dtype=np.float64)
    ld = np.ones(40, dtype=np.float64)

    first = ADAPTIVE.four_way_split(
        parents,
        signal_score=signal,
        ld_score=ld,
        child_depth=1,
    )
    second = ADAPTIVE.four_way_split(
        parents,
        signal_score=signal,
        ld_score=ld,
        child_depth=1,
    )

    assert all(
        np.array_equal(left.variant_indices, right.variant_indices)
        for left, right in zip(first, second)
    )
    selected_high = np.sort(
        np.concatenate([first[0].variant_indices, first[1].variant_indices])
    )
    assert np.array_equal(selected_high, np.arange(20, 40))
    assert [child.variant_indices.size for child in first] == [10, 10, 10, 10]
    assert first[0].annotation["signal_cluster_low_mean"] == pytest.approx(1.0)
    assert first[0].annotation["signal_cluster_high_mean"] == pytest.approx(1.0)


def test_four_way_split_two_means_finds_exact_separated_clusters():
    parents = ADAPTIVE.single_component(8)
    signal = np.asarray([0.0, 0.1, 0.2, 0.3, 10.0, 11.0, 12.0, 13.0])
    ld = np.asarray([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])

    children = ADAPTIVE.four_way_split(
        parents,
        signal_score=signal,
        ld_score=ld,
        child_depth=1,
    )

    high_signal = np.sort(
        np.concatenate([children[0].variant_indices, children[1].variant_indices])
    )
    low_signal = np.sort(
        np.concatenate([children[2].variant_indices, children[3].variant_indices])
    )
    assert np.array_equal(high_signal, np.arange(4, 8))
    assert np.array_equal(low_signal, np.arange(4))
    assert children[0].annotation["signal_split_method"] == "exact_1d_two_means"
    assert children[0].annotation["signal_high_fraction_realized"] == pytest.approx(
        0.5
    )
    assert children[0].annotation["signal_cluster_low_mean"] == pytest.approx(0.15)
    assert children[0].annotation["signal_cluster_high_mean"] == pytest.approx(11.5)
    ADAPTIVE.validate_partition(children, n_variants=8)


def test_four_way_split_two_means_is_deterministic_under_within_cluster_ties():
    parents = ADAPTIVE.single_component(12)
    signal = np.asarray([0.0] * 6 + [5.0] * 6)
    ld = np.ones(12)

    first = ADAPTIVE.four_way_split(
        parents,
        signal_score=signal,
        ld_score=ld,
        child_depth=1,
    )
    second = ADAPTIVE.four_way_split(
        parents,
        signal_score=signal,
        ld_score=ld,
        child_depth=1,
    )

    assert all(
        np.array_equal(left.variant_indices, right.variant_indices)
        for left, right in zip(first, second)
    )
    high_signal = np.sort(
        np.concatenate([first[0].variant_indices, first[1].variant_indices])
    )
    assert np.array_equal(high_signal, np.arange(6, 12))


def test_four_way_split_two_means_balances_constant_parent_scores():
    children = ADAPTIVE.four_way_split(
        ADAPTIVE.single_component(8),
        signal_score=np.ones(8),
        ld_score=np.arange(8, dtype=np.float64),
        child_depth=1,
    )

    assert [child.variant_indices.size for child in children] == [2, 2, 2, 2]
    high_signal = np.sort(
        np.concatenate([children[0].variant_indices, children[1].variant_indices])
    )
    assert np.array_equal(high_signal, np.arange(4, 8))


def test_iterative_split_has_only_power_of_four_k():
    signal = np.linspace(0.0, 1.0, 1000)
    ld = np.cos(np.arange(1000, dtype=np.float64))
    components = ADAPTIVE.single_component(1000)
    observed = [len(components)]
    for depth in (1, 2):
        components = ADAPTIVE.four_way_split(
            components,
            signal_score=signal,
            ld_score=ld,
            child_depth=depth,
        )
        observed.append(len(components))
    assert observed == [1, 4, 16]


def test_full_marker_panel_can_reach_k1024_with_nonempty_leaves():
    n_variants = 632_255
    signal = np.linspace(0.0, 1.0, n_variants)
    ld = np.linspace(1.0, 0.0, n_variants)
    components = ADAPTIVE.single_component(n_variants)
    observed = [(1, n_variants, n_variants)]

    for depth in range(1, 6):
        components = ADAPTIVE.four_way_split(
            components,
            signal_score=signal,
            ld_score=ld,
            child_depth=depth,
        )
        sizes = [component.variant_indices.size for component in components]
        observed.append((len(components), min(sizes), max(sizes)))

    assert [row[0] for row in observed] == [1, 4, 16, 64, 256, 1024]
    for depth, (_, minimum, maximum) in enumerate(observed):
        assert minimum == n_variants // (4**depth)
        assert maximum == int(np.ceil(n_variants / (4**depth)))
    ADAPTIVE.validate_partition(components, n_variants=n_variants)


def test_covariance_preserving_warm_start_uses_marker_proportions():
    parents = ADAPTIVE.single_component(100)
    signal = np.arange(100, dtype=np.float64)
    ld = np.arange(100, dtype=np.float64)
    children = ADAPTIVE.four_way_split(
        parents,
        signal_score=signal,
        ld_score=ld,
        child_depth=1,
    )

    warm = ADAPTIVE.covariance_preserving_warm_start(
        np.asarray([0.6, 0.4]),
        parents,
        children,
    )

    expected = np.asarray([0.15, 0.15, 0.15, 0.15, 0.4])
    assert np.allclose(warm, expected)
    assert warm[:-1].sum() == pytest.approx(0.6)


def test_written_component_spec_is_loadable(tmp_path):
    components = ADAPTIVE.single_component(12)
    path = ADAPTIVE.write_component_spec(
        tmp_path / "components.npz",
        components,
        provenance={"test": True},
    )

    loaded = COMPONENT_SPEC.load_component_specs(str(path))

    assert [component.name for component in loaded] == ["root"]
    assert np.array_equal(loaded[0].variant_indices, np.arange(12))
    assert loaded[0].provenance == {"test": True}


def test_four_way_split_rejects_parent_too_small_for_four_nonempty_children():
    with pytest.raises(ValueError, match="four-way split"):
        ADAPTIVE.four_way_split(
            ADAPTIVE.single_component(3),
            signal_score=np.ones(3),
            ld_score=np.ones(3),
            child_depth=1,
        )
