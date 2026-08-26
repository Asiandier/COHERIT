"""Deterministic four-way partitions for prediction-first Adaptive COHERIT."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class AdaptiveComponent:
    """One marker leaf in the current adaptive partition."""

    name: str
    variant_indices: np.ndarray
    annotation: dict[str, object]


def validate_partition(
    components: Sequence[AdaptiveComponent],
    *,
    n_variants: int,
) -> None:
    """Require nonempty, disjoint leaves that cover source variant order."""
    if n_variants <= 0:
        raise ValueError("n_variants must be positive.")
    if not components:
        raise ValueError("Adaptive partition must contain at least one component.")

    membership = np.full(int(n_variants), -1, dtype=np.int32)
    names: set[str] = set()
    for component_index, component in enumerate(components):
        if component.name in names:
            raise ValueError(f"Duplicate adaptive component name: {component.name}")
        names.add(component.name)
        indices = np.asarray(component.variant_indices, dtype=np.int64).reshape(-1)
        if indices.size == 0:
            raise ValueError(f"Adaptive component {component.name} is empty.")
        if np.any((indices < 0) | (indices >= int(n_variants))):
            raise ValueError(
                f"Adaptive component {component.name} contains an out-of-range marker."
            )
        if np.unique(indices).size != indices.size:
            raise ValueError(
                f"Adaptive component {component.name} contains duplicate markers."
            )
        if np.any(membership[indices] >= 0):
            raise ValueError("Adaptive components overlap.")
        membership[indices] = int(component_index)
    if np.any(membership < 0):
        raise ValueError("Adaptive partition does not cover every source marker.")


def single_component(n_variants: int) -> list[AdaptiveComponent]:
    if n_variants <= 0:
        raise ValueError("n_variants must be positive.")
    return [
        AdaptiveComponent(
            name="root",
            variant_indices=np.arange(int(n_variants), dtype=np.int64),
            annotation={"depth": 0, "path": "root"},
        )
    ]


def _exact_signal_two_means_split(
    indices: np.ndarray,
    signal_score: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    """Return the exact deterministic one-dimensional two-means partition.

    For squared Euclidean distance, the globally optimal clusters are
    contiguous after sorting the scalar scores.  Evaluating every admissible
    split therefore avoids the initialization and local-minimum ambiguity of
    iterative Lloyd k-means.  At least two markers are retained in each
    signal cluster because both clusters are subsequently split by LD rank.
    """
    count = int(indices.size)
    if count < 4:
        raise ValueError(
            "A four-way split requires at least four markers for signal "
            f"two-means; parent size={count}."
        )
    values = np.asarray(signal_score[indices], dtype=np.float64)
    # Primary key: ascending score. Secondary key: source marker index.
    order = np.lexsort((indices, values))
    sorted_values = values[order]

    prefix_sum = np.concatenate(
        (np.asarray([0.0]), np.cumsum(sorted_values, dtype=np.float64))
    )
    prefix_square = np.concatenate(
        (
            np.asarray([0.0]),
            np.cumsum(sorted_values * sorted_values, dtype=np.float64),
        )
    )
    split_positions = np.arange(2, count - 1, dtype=np.int64)
    low_count = split_positions.astype(np.float64)
    high_count = float(count) - low_count
    low_sum = prefix_sum[split_positions]
    high_sum = prefix_sum[count] - low_sum
    low_sse = prefix_square[split_positions] - low_sum * low_sum / low_count
    high_sse = (
        prefix_square[count]
        - prefix_square[split_positions]
        - high_sum * high_sum / high_count
    )
    objective = np.maximum(low_sse, 0.0) + np.maximum(high_sse, 0.0)
    # Several cut positions can have exactly the same global optimum when
    # scores contain ties (including the all-equal case).  Prefer the most
    # balanced optimum, then the smaller cut position.  This keeps every
    # split deterministic and prevents arbitrary tiny clusters.
    balance = np.abs(split_positions.astype(np.float64) - 0.5 * float(count))
    best_index = int(
        np.lexsort((split_positions, balance, objective))[0]
    )
    split = int(split_positions[best_index])

    low_order = order[:split]
    high_order = order[split:]
    low = np.sort(indices[low_order])
    high = np.sort(indices[high_order])
    low_mean = float(np.mean(values[low_order]))
    high_mean = float(np.mean(values[high_order]))
    if high_mean < low_mean:
        raise RuntimeError("Signal two-means reversed the ordered cluster means.")
    cutoff = float(0.5 * (low_mean + high_mean))
    diagnostics: dict[str, float | int] = {
        "signal_cluster_low_size": int(low.size),
        "signal_cluster_high_size": int(high.size),
        "signal_cluster_low_mean": low_mean,
        "signal_cluster_high_mean": high_mean,
        "signal_cluster_boundary": cutoff,
        "signal_cluster_within_sse": float(objective[best_index]),
    }
    return high, low, diagnostics


def _stable_ld_split(
    indices: np.ndarray,
    ld_score: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    count = int(indices.size)
    if count < 2:
        raise ValueError("An LD median split requires at least two markers.")
    values = ld_score[indices]
    # This rank definition is exactly a median split and remains deterministic
    # when many markers have the same LD score.
    order = np.lexsort((indices, values))
    low_count = count // 2
    low = np.sort(indices[order[:low_count]])
    high = np.sort(indices[order[low_count:]])
    median = float(np.median(values))
    return high, low, median


def four_way_split(
    parents: Sequence[AdaptiveComponent],
    *,
    signal_score: np.ndarray,
    ld_score: np.ndarray,
    child_depth: int,
) -> list[AdaptiveComponent]:
    """Split every parent by exact signal two-means, then by LD rank."""
    signal = np.asarray(signal_score, dtype=np.float64).reshape(-1)
    ld = np.asarray(ld_score, dtype=np.float64).reshape(-1)
    if signal.shape != ld.shape or signal.size == 0:
        raise ValueError("signal_score and ld_score must have one matching entry per marker.")
    if not np.all(np.isfinite(signal)) or np.any(signal < 0.0):
        raise ValueError("signal_score must be finite and nonnegative.")
    if not np.all(np.isfinite(ld)):
        raise ValueError("ld_score must be finite.")
    validate_partition(parents, n_variants=int(signal.size))

    children: list[AdaptiveComponent] = []
    for parent_index, parent in enumerate(parents):
        indices = np.asarray(parent.variant_indices, dtype=np.int64).reshape(-1)
        (
            signal_high,
            signal_low,
            signal_diagnostics,
        ) = _exact_signal_two_means_split(indices, signal)
        realized_high_fraction = float(signal_high.size / indices.size)
        sh_ld_high, sh_ld_low, sh_ld_median = _stable_ld_split(
            signal_high, ld
        )
        sl_ld_high, sl_ld_low, sl_ld_median = _stable_ld_split(
            signal_low, ld
        )
        child_definitions = (
            ("signal_high__ld_high", sh_ld_high, "high", "high", sh_ld_median),
            ("signal_high__ld_low", sh_ld_low, "high", "low", sh_ld_median),
            ("signal_low__ld_high", sl_ld_high, "low", "high", sl_ld_median),
            ("signal_low__ld_low", sl_ld_low, "low", "low", sl_ld_median),
        )
        for suffix, child_indices, signal_bin, ld_bin, ld_median in child_definitions:
            child_name = f"{parent.name}__{suffix}"
            children.append(
                AdaptiveComponent(
                    name=child_name,
                    variant_indices=child_indices,
                    annotation={
                        "depth": int(child_depth),
                        "path": child_name,
                        "parent_index": int(parent_index),
                        "parent_name": parent.name,
                        "parent_size": int(indices.size),
                        "signal_bin": signal_bin,
                        "signal_split_method": "exact_1d_two_means",
                        "signal_high_fraction_realized": realized_high_fraction,
                        "ld_bin": ld_bin,
                        "ld_median_within_signal_bin": float(ld_median),
                        **signal_diagnostics,
                    },
                )
            )
    validate_partition(children, n_variants=int(signal.size))
    if len(children) != 4 * len(parents):
        raise RuntimeError("Every adaptive parent must produce exactly four children.")
    return children


def covariance_preserving_warm_start(
    parent_theta: np.ndarray,
    parents: Sequence[AdaptiveComponent],
    children: Sequence[AdaptiveComponent],
) -> np.ndarray:
    """Map parent GRM coefficients to four children and retain residual theta."""
    theta = np.asarray(parent_theta, dtype=np.float64).reshape(-1)
    if theta.shape != (len(parents) + 1,):
        raise ValueError(
            "parent_theta must contain one coefficient per parent plus residual."
        )
    if len(children) != 4 * len(parents):
        raise ValueError("Expected exactly four children per parent.")
    if not np.all(np.isfinite(theta)) or np.any(theta[:-1] < 0.0) or theta[-1] <= 0.0:
        raise ValueError("parent_theta contains invalid variance components.")

    child_theta: list[float] = []
    for parent_index, parent in enumerate(parents):
        parent_size = int(np.asarray(parent.variant_indices).size)
        parent_children = children[4 * parent_index : 4 * parent_index + 4]
        child_sizes = np.asarray(
            [np.asarray(child.variant_indices).size for child in parent_children],
            dtype=np.float64,
        )
        if int(child_sizes.sum()) != parent_size:
            raise ValueError("Child marker counts do not sum to the parent count.")
        child_theta.extend(
            (float(theta[parent_index]) * child_sizes / float(parent_size)).tolist()
        )
    return np.asarray([*child_theta, float(theta[-1])], dtype=np.float64)


def write_component_spec(
    path: str | os.PathLike[str],
    components: Sequence[AdaptiveComponent],
    *,
    provenance: dict[str, object],
) -> Path:
    """Atomically write an NPZ component spec accepted by GPU_REML."""
    target = Path(path).expanduser().resolve(strict=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        f"arr_{index}": np.asarray(component.variant_indices, dtype=np.int64)
        for index, component in enumerate(components)
    }
    arrays["component_names"] = np.asarray(
        [component.name for component in components]
    )
    arrays["component_annotations_json"] = np.asarray(
        [json.dumps(component.annotation, sort_keys=True) for component in components]
    )
    arrays["component_provenance_json"] = np.asarray(
        [json.dumps(provenance, sort_keys=True) for _ in components]
    )
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, target)
    return target


__all__ = [
    "AdaptiveComponent",
    "covariance_preserving_warm_start",
    "four_way_split",
    "single_component",
    "validate_partition",
    "write_component_spec",
]
