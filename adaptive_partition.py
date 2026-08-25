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


def _stable_signal_split(
    indices: np.ndarray,
    signal_score: np.ndarray,
    *,
    high_fraction: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    count = int(indices.size)
    high_count = int(np.ceil(float(high_fraction) * count))
    if high_count < 2 or count - high_count < 2:
        raise ValueError(
            "A four-way split requires at least two markers in both signal bins; "
            f"parent size={count}, high_fraction={high_fraction}."
        )
    values = signal_score[indices]
    # Primary key: descending score. Secondary key: source marker index.
    order = np.lexsort((indices, -values))
    high = np.sort(indices[order[:high_count]])
    low = np.sort(indices[order[high_count:]])
    cutoff = float(values[order[high_count - 1]])
    return high, low, cutoff


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
    high_fraction: float = 0.15,
    child_depth: int,
) -> list[AdaptiveComponent]:
    """Split every parent into signal 15/85 and within-bin LD halves."""
    signal = np.asarray(signal_score, dtype=np.float64).reshape(-1)
    ld = np.asarray(ld_score, dtype=np.float64).reshape(-1)
    if signal.shape != ld.shape or signal.size == 0:
        raise ValueError("signal_score and ld_score must have one matching entry per marker.")
    if not np.all(np.isfinite(signal)) or np.any(signal < 0.0):
        raise ValueError("signal_score must be finite and nonnegative.")
    if not np.all(np.isfinite(ld)):
        raise ValueError("ld_score must be finite.")
    if not 0.0 < float(high_fraction) < 1.0:
        raise ValueError("high_fraction must lie strictly between zero and one.")
    validate_partition(parents, n_variants=int(signal.size))

    children: list[AdaptiveComponent] = []
    for parent_index, parent in enumerate(parents):
        indices = np.asarray(parent.variant_indices, dtype=np.int64).reshape(-1)
        signal_high, signal_low, signal_cutoff = _stable_signal_split(
            indices,
            signal,
            high_fraction=float(high_fraction),
        )
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
                        "signal_high_fraction": float(high_fraction),
                        "signal_cutoff": signal_cutoff,
                        "ld_bin": ld_bin,
                        "ld_median_within_signal_bin": float(ld_median),
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
