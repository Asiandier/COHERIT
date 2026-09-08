"""Shared sparse model contracts and marker indexing, independent of CLI setup."""
from __future__ import annotations

from typing import Any

import numpy as np


def validate_component_partition(
    groups: list[np.ndarray], *, n_markers: int
) -> list[np.ndarray]:
    """Require a nonempty, disjoint, exhaustive source-marker partition."""
    if not groups:
        return []
    normalized: list[np.ndarray] = []
    for component_idx, group in enumerate(groups):
        values = np.asarray(group, dtype=np.int64).reshape(-1)
        if values.size == 0:
            raise ValueError(
                f"Component {component_idx} is empty; every GRM must contain SNPs."
            )
        unique = np.unique(values)
        if unique.size != values.size:
            raise ValueError(f"Component {component_idx} contains duplicate SNPs.")
        if np.any((unique < 0) | (unique >= int(n_markers))):
            raise ValueError(
                f"Component {component_idx} contains an index outside "
                f"[0, {int(n_markers)})."
            )
        normalized.append(unique)
    assigned = np.concatenate(normalized)
    if assigned.size != int(n_markers) or np.unique(assigned).size != int(
        n_markers
    ):
        raise ValueError(
            "--component-spec must assign every source SNP exactly once "
            "across mutually exclusive components."
        )
    return normalized


class MultiGRMIndex:
    """Map the sparse cache order to component-local and source coordinates.

    A component-partitioned source is physically cached as the concatenation
    of its components.  Sparse optimization works in that cache order, while
    BIM/PVAR reporting must use original source-variant rows.  This class is
    the single authority for translating between the two coordinate systems.
    """

    def __init__(self, streamers, component_variant_indices=None):
        self.streamers = tuple(streamers)
        if len(self.streamers) != 1:
            raise ValueError(
                "Sparse fitting accepts exactly one physical genotype source."
            )
        self.streamer = self.streamers[0]
        self._partitioned_single_streamer = component_variant_indices is not None
        self._source_variant_indices: np.ndarray | None = None

        if self._partitioned_single_streamer:
            streamer = self.streamer
            if not bool(getattr(streamer, "has_component_partition", False)):
                raise ValueError(
                    "component_variant_indices were supplied, but the genotype "
                    "streamer is not component-partitioned."
                )
            requested_groups = [
                np.asarray(group, dtype=np.int64).reshape(-1)
                for group in component_variant_indices
            ]
            self.n_grm = int(streamer.n_components)
            component_offsets = np.asarray(
                streamer._component_snp_offsets, dtype=np.int64
            ).reshape(-1)
            if component_offsets.shape != (self.n_grm + 1,):
                raise ValueError("Invalid component offsets in partitioned streamer.")
            self.m_per_grm = np.diff(component_offsets)
            cache_to_source = np.asarray(
                streamer._cache_to_source_variant_indices, dtype=np.int64
            ).reshape(-1)
            if cache_to_source.size != int(streamer.m):
                raise ValueError(
                    "Partitioned streamer's cache-to-source SNP map has the "
                    "wrong length."
                )
            if not np.array_equal(
                np.sort(cache_to_source),
                np.arange(int(streamer.m), dtype=np.int64),
            ):
                raise ValueError(
                    "Sparse component partitions must cover every source SNP "
                    "exactly once."
                )
            if len(requested_groups) != self.n_grm:
                raise ValueError(
                    "Component count mismatch between component spec and "
                    "genotype streamer."
                )
            for component_idx, requested in enumerate(requested_groups):
                start = int(component_offsets[component_idx])
                stop = int(component_offsets[component_idx + 1])
                actual = cache_to_source[start:stop]
                if not np.array_equal(actual, np.unique(requested)):
                    raise ValueError(
                        "Component SNP mapping mismatch between component spec "
                        f"and genotype streamer for component {component_idx}."
                    )
            self._source_variant_indices = cache_to_source.copy()
        else:
            self.n_grm = 1
            self.m_per_grm = np.asarray([int(self.streamer.m)], dtype=np.int64)

        self.offsets = np.zeros(self.n_grm + 1, dtype=np.int64)
        np.cumsum(self.m_per_grm, out=self.offsets[1:])
        self.m_total = int(self.offsets[-1])

    def _validated_global_indices(self, global_idx: np.ndarray) -> np.ndarray:
        idx = np.asarray(global_idx, dtype=np.int64)
        if idx.ndim != 1:
            raise ValueError("Global SNP indices must be one-dimensional.")
        if np.any((idx < 0) | (idx >= self.m_total)):
            raise IndexError(
                f"Global SNP indices must lie in [0, {self.m_total})."
            )
        return idx

    def global_to_local(
        self, global_idx: np.ndarray
    ) -> list[tuple[int, np.ndarray, np.ndarray]]:
        idx = self._validated_global_indices(global_idx)
        grm_ids = np.searchsorted(self.offsets[1:], idx, side="right")
        grm_ids = np.clip(grm_ids, 0, self.n_grm - 1)
        groups: list[tuple[int, np.ndarray, np.ndarray]] = []
        for grm_idx in range(self.n_grm):
            positions = np.flatnonzero(grm_ids == grm_idx)
            if positions.size == 0:
                continue
            local = idx[positions] - int(self.offsets[grm_idx])
            groups.append((grm_idx, local, positions))
        return groups

    def xtv_all(self, u_jax: Any, normalize: bool = False, *, dtype=np.float64) -> np.ndarray:
        return np.asarray(
            self.streamer.xtv(u_jax, normalize=normalize),
            dtype=dtype,
        )

    def extract_standardized_columns(
        self, global_idx: np.ndarray
    ) -> np.ndarray:
        idx = self._validated_global_indices(global_idx)
        return self.streamer.extract_standardized_columns(idx)

    def source_variant_indices(self, global_idx: np.ndarray) -> np.ndarray:
        idx = self._validated_global_indices(global_idx)
        if self._source_variant_indices is None:
            return idx.copy()
        return self._source_variant_indices[idx].copy()

    def cache_variant_indices(self, source_idx: np.ndarray) -> np.ndarray:
        """Translate original source rows to the current concatenated cache."""
        source = np.asarray(source_idx, dtype=np.int64).reshape(-1)
        if self._source_variant_indices is None:
            return self._validated_global_indices(source).copy()
        if np.any((source < 0) | (source >= self.m_total)):
            raise IndexError(
                f"Source SNP indices must lie in [0, {self.m_total})."
            )
        source_to_cache = np.empty(self.m_total, dtype=np.int64)
        source_to_cache[self._source_variant_indices] = np.arange(
            self.m_total, dtype=np.int64
        )
        return source_to_cache[source]

    def lookup_bim_rows(
        self, bed_prefixes: list[str], global_idx: np.ndarray
    ) -> dict[int, tuple[str, str, str, str, str, str]]:
        idx = self._validated_global_indices(global_idx)
        result: dict[int, tuple[str, str, str, str, str, str]] = {}
        source_idx = self.source_variant_indices(idx)
        source_rows = lookup_bim_rows(bed_prefixes[0] + ".bim", source_idx)
        for global_snp, source_snp in zip(idx.tolist(), source_idx.tolist()):
            if int(source_snp) in source_rows:
                result[int(global_snp)] = source_rows[int(source_snp)]
        return result


class SingleGRMIndex(MultiGRMIndex):
    """Compatibility wrapper for callers that explicitly require K=1."""

    def __init__(self, streamers):
        if len(streamers) != 1:
            raise ValueError(
                "The sparse pipeline supports exactly one whole-genome GRM."
            )
        streamer = streamers[0]
        if bool(getattr(streamer, "has_component_partition", False)) or int(
            getattr(streamer, "n_components", 1)
        ) != 1:
            raise ValueError(
                "The single-GRM index does not accept component partitions."
            )
        super().__init__(streamers)

    def lookup_bim_rows(
        self, bed_prefix: str, marker_idx: np.ndarray
    ) -> dict[int, tuple[str, str, str, str, str, str]]:
        return super().lookup_bim_rows([bed_prefix], marker_idx)


def lookup_bim_rows(bim_path: str, snp_indices: np.ndarray) -> dict[int, tuple[str, str, str, str, str, str]]:
    idx = np.asarray(snp_indices, dtype=np.int64)
    if idx.size == 0:
        return {}
    need = set(int(i) for i in idx.tolist())
    rows: dict[int, tuple[str, str, str, str, str, str]] = {}
    with open(bim_path, "r") as f:
        for i, line in enumerate(f):
            if i not in need:
                continue
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            rows[i] = (parts[0], parts[1], parts[2], parts[3], parts[4], parts[5])
            if len(rows) == len(need):
                break
    return rows


def accepted_reml_theta(
    fit_result,
    *,
    expected_components: int,
    stage: str,
) -> tuple[np.ndarray, str]:
    """Return a valid, converged REML state.

    ``fit_reml`` returns the last accepted parameter vector when all
    line-search candidates are downhill. ``ll_down`` is an accepted terminal
    state: its rejected trial is discarded, but earlier accepted updates in
    the same covariance block are retained.
    """
    theta = np.asarray(fit_result.var_components, dtype=np.float64).reshape(-1)
    history = list(fit_result.history)
    stop_reason = (
        str(history[-1].get("stop_reason", "")) if history else ""
    )
    valid_theta = (
        theta.shape == (int(expected_components),)
        and np.all(np.isfinite(theta))
        and np.all(theta[:-1] >= 0.0)
        and theta[-1] > 0.0
    )
    last_history = history[-1] if history else {}
    candidate_accepted = bool(last_history.get("accepted", False))
    converged = bool(last_history.get("converged", False))
    usable_history = bool(
        history
        and converged
        and (
            (
                stop_reason in {"rel_dll", "scoring_step"}
                and candidate_accepted
            )
            or (stop_reason == "ll_down" and not candidate_accepted)
        )
    )
    if not valid_theta or not usable_history:
        raise RuntimeError(
            f"{stage} did not return a usable REML state: "
            f"theta={theta.tolist()}, history_rows={len(history)}, "
            f"stop_reason={stop_reason!r}, "
            f"last_accepted="
            f"{bool(history[-1].get('accepted', False)) if history else False}, "
            f"converged={converged}, "
            f"dll_true={last_history.get('dll_true')}, "
            f"max_rel_dp={last_history.get('max_rel_dp')}."
        )
    return theta, stop_reason


def sparse_dense_h2(
    q_sparse: float,
    background_genetic_variance: float,
    residual_variance: float,
) -> float:
    """Combine sparse, dense-background, and residual variance on one scale."""
    genetic = float(q_sparse) + float(background_genetic_variance)
    residual = float(residual_variance)
    denominator = genetic + residual
    if (
        not np.isfinite(genetic)
        or not np.isfinite(residual)
        or not np.isfinite(denominator)
        or denominator <= 0.0
    ):
        return float("nan")
    return genetic / denominator
