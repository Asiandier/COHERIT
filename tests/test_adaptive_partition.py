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


def test_four_way_split_uses_top_15_percent_and_bin_specific_ld_medians():
    parents = ADAPTIVE.single_component(100)
    signal = np.arange(100, dtype=np.float64)
    ld = np.arange(100, dtype=np.float64)

    children = ADAPTIVE.four_way_split(
        parents,
        signal_score=signal,
        ld_score=ld,
        high_fraction=0.15,
        child_depth=1,
    )

    assert len(children) == 4
    assert [child.variant_indices.size for child in children] == [8, 7, 43, 42]
    high_signal = np.concatenate(
        [children[0].variant_indices, children[1].variant_indices]
    )
    low_signal = np.concatenate(
        [children[2].variant_indices, children[3].variant_indices]
    )
    assert np.array_equal(np.sort(high_signal), np.arange(85, 100))
    assert np.array_equal(np.sort(low_signal), np.arange(85))
    assert children[0].annotation["ld_median_within_signal_bin"] == 92.0
    assert children[2].annotation["ld_median_within_signal_bin"] == 42.0
    ADAPTIVE.validate_partition(children, n_variants=100)


def test_four_way_split_is_deterministic_under_score_ties():
    parents = ADAPTIVE.single_component(40)
    signal = np.ones(40, dtype=np.float64)
    ld = np.ones(40, dtype=np.float64)

    first = ADAPTIVE.four_way_split(
        parents,
        signal_score=signal,
        ld_score=ld,
        high_fraction=0.15,
        child_depth=1,
    )
    second = ADAPTIVE.four_way_split(
        parents,
        signal_score=signal,
        ld_score=ld,
        high_fraction=0.15,
        child_depth=1,
    )

    assert all(
        np.array_equal(left.variant_indices, right.variant_indices)
        for left, right in zip(first, second)
    )
    selected_high = np.sort(
        np.concatenate([first[0].variant_indices, first[1].variant_indices])
    )
    assert np.array_equal(selected_high, np.arange(6))


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
            high_fraction=0.15,
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
            high_fraction=0.15,
            child_depth=depth,
        )
        sizes = [component.variant_indices.size for component in components]
        observed.append((len(components), min(sizes), max(sizes)))

    assert observed == [
        (1, 632_255, 632_255),
        (4, 47_419, 268_708),
        (16, 3_556, 114_201),
        (64, 267, 48_535),
        (256, 20, 20_627),
        (1024, 1, 8_766),
    ]
    ADAPTIVE.validate_partition(components, n_variants=n_variants)


def test_covariance_preserving_warm_start_uses_marker_proportions():
    parents = ADAPTIVE.single_component(100)
    signal = np.arange(100, dtype=np.float64)
    ld = np.arange(100, dtype=np.float64)
    children = ADAPTIVE.four_way_split(
        parents,
        signal_score=signal,
        ld_score=ld,
        high_fraction=0.15,
        child_depth=1,
    )

    warm = ADAPTIVE.covariance_preserving_warm_start(
        np.asarray([0.6, 0.4]),
        parents,
        children,
    )

    expected = np.asarray([0.048, 0.042, 0.258, 0.252, 0.4])
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
            ADAPTIVE.single_component(5),
            signal_score=np.ones(5),
            ld_score=np.ones(5),
            high_fraction=0.15,
            child_depth=1,
        )
