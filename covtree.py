"""Genotype-only partitions and bootstrap score inference for COHERIT-CovTree."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.stats import chi2

from .adaptive_partition import (
    AdaptiveComponent,
    exact_two_means_split,
    validate_partition,
)


@dataclass(frozen=True)
class CovTreeCandidate:
    """One phenotype-independent split proposed for a current parent."""

    name: str
    parent_index: int
    parent_name: str
    split_kind: str
    children: tuple[np.ndarray, ...]
    diagnostics: dict[str, object]

    @property
    def degrees_of_freedom(self) -> int:
        return len(self.children) - 1


def _intersect_sorted(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.intersect1d(left, right, assume_unique=True)


def _recursive_feature_leaves(
    parent_indices: np.ndarray,
    feature: np.ndarray,
    *,
    depth: int,
    min_child_markers: int,
    feature_name: str,
) -> tuple[np.ndarray, ...]:
    leaves = [np.asarray(parent_indices, dtype=np.int64)]
    for _ in range(int(depth)):
        next_leaves: list[np.ndarray] = []
        for leaf in leaves:
            leaf_values = np.asarray(feature[leaf], dtype=np.float64)
            variation_tolerance = np.finfo(np.float64).eps * max(
                1.0, float(np.max(np.abs(leaf_values)))
            )
            if float(np.ptp(leaf_values)) <= variation_tolerance:
                raise ValueError(f"{feature_name} is constant within a tree leaf")
            high, low, _ = exact_two_means_split(
                leaf,
                feature,
                min_cluster_size=int(min_child_markers),
                feature_name=feature_name,
            )
            next_leaves.extend([low, high])
        leaves = next_leaves
    return tuple(leaves)


def generate_covtree_candidates(
    components: Sequence[AdaptiveComponent],
    *,
    ld_score: np.ndarray,
    heterozygosity: np.ndarray,
    min_child_markers: int = 2,
    max_univariate_depth: int = 1,
) -> tuple[list[CovTreeCandidate], list[dict[str, object]]]:
    """Generate recursive LD/MAF and LD-by-MAF candidates in each parent.

    The cuts use exact deterministic one-dimensional two-means on
    ``log1p(LD score)`` and ``log(2p(1-p))``.  Phenotype values never enter
    candidate construction.  Depth one yields LD-2 and MAF-2; larger depths
    recursively bisect every leaf and yield 4, 8, ... children.
    """
    ld = np.asarray(ld_score, dtype=np.float64).reshape(-1)
    het = np.asarray(heterozygosity, dtype=np.float64).reshape(-1)
    minimum = int(min_child_markers)
    maximum_depth = int(max_univariate_depth)
    if ld.shape != het.shape or ld.size == 0:
        raise ValueError("ld_score and heterozygosity must have matching lengths.")
    if minimum < 1:
        raise ValueError("min_child_markers must be >= 1.")
    if maximum_depth < 1 or maximum_depth > 10:
        raise ValueError("max_univariate_depth must lie in 1..10.")
    if not np.all(np.isfinite(ld)) or np.any(ld < 0.0):
        raise ValueError("ld_score must be finite and nonnegative.")
    if not np.all(np.isfinite(het)) or np.any(het <= 0.0):
        raise ValueError("heterozygosity must be finite and positive.")
    validate_partition(components, n_variants=int(ld.size))

    transformed_ld = np.log1p(ld)
    transformed_maf = np.log(het)
    candidates: list[CovTreeCandidate] = []
    rejected: list[dict[str, object]] = []
    for parent_index, parent in enumerate(components):
        parent_indices = np.asarray(parent.variant_indices, dtype=np.int64)
        if parent_indices.size < 2 * minimum:
            rejected.append(
                {
                    "parent_index": int(parent_index),
                    "parent_name": parent.name,
                    "split_kind": "all",
                    "reason": "parent_too_small",
                    "parent_size": int(parent_indices.size),
                }
            )
            continue
        common = {
            "parent_size": int(parent_indices.size),
            "feature_transforms": {
                "ld": "log1p_ld_score",
                "maf": "log_2p1mp",
            },
        }
        split_results: dict[
            str, tuple[np.ndarray, np.ndarray, dict[str, float | int]]
        ] = {}
        for feature_name, feature_values in (
            ("ld", transformed_ld),
            ("maf", transformed_maf),
        ):
            parent_values = feature_values[parent_indices]
            variation_tolerance = np.finfo(np.float64).eps * max(
                1.0, float(np.max(np.abs(parent_values)))
            )
            if float(np.ptp(parent_values)) <= variation_tolerance:
                rejected.append(
                    {
                        "parent_index": int(parent_index),
                        "parent_name": parent.name,
                        "split_kind": f"{feature_name}2",
                        "reason": "feature_constant_within_parent",
                    }
                )
                continue
            try:
                feature_high, feature_low, feature_diag = exact_two_means_split(
                    parent_indices,
                    feature_values,
                    min_cluster_size=minimum,
                    feature_name=feature_name,
                )
            except (ValueError, IndexError) as error:
                rejected.append(
                    {
                        "parent_index": int(parent_index),
                        "parent_name": parent.name,
                        "split_kind": f"{feature_name}2",
                        "reason": f"feature_split_failed:{error}",
                    }
                )
                continue
            split_results[feature_name] = (
                feature_high,
                feature_low,
                feature_diag,
            )
            candidates.append(
                CovTreeCandidate(
                    name=f"p{parent_index:04d}__{feature_name}2",
                    parent_index=int(parent_index),
                    parent_name=parent.name,
                    split_kind=f"{feature_name}2",
                    children=(feature_low, feature_high),
                    diagnostics={**common, **feature_diag},
                )
            )
            for depth in range(2, maximum_depth + 1):
                try:
                    feature_children = _recursive_feature_leaves(
                        parent_indices,
                        feature_values,
                        depth=depth,
                        min_child_markers=minimum,
                        feature_name=feature_name,
                    )
                except (ValueError, IndexError) as error:
                    rejected.append(
                        {
                            "parent_index": int(parent_index),
                            "parent_name": parent.name,
                            "split_kind": f"{feature_name}{2**depth}_tree",
                            "reason": f"recursive_feature_split_failed:{error}",
                        }
                    )
                    break
                candidates.append(
                    CovTreeCandidate(
                        name=(
                            f"p{parent_index:04d}__{feature_name}{2**depth}_tree"
                        ),
                        parent_index=int(parent_index),
                        parent_name=parent.name,
                        split_kind=f"{feature_name}{2**depth}_tree",
                        children=feature_children,
                        diagnostics={
                            **common,
                            "recursive_feature": feature_name,
                            "recursive_depth": int(depth),
                            "cut_method": "recursive_exact_1d_two_means",
                        },
                    )
                )

        if not {"ld", "maf"}.issubset(split_results):
            rejected.append(
                {
                    "parent_index": int(parent_index),
                    "parent_name": parent.name,
                    "split_kind": "ld_maf4",
                    "reason": "joint_feature_unavailable",
                    "missing_features": sorted(
                        {"ld", "maf"}.difference(split_results)
                    ),
                }
            )
        else:
            ld_high, ld_low, ld_diag = split_results["ld"]
            maf_high, maf_low, maf_diag = split_results["maf"]
            joint_children = (
                _intersect_sorted(ld_low, maf_low),
                _intersect_sorted(ld_low, maf_high),
                _intersect_sorted(ld_high, maf_low),
                _intersect_sorted(ld_high, maf_high),
            )
            joint_sizes = [int(child.size) for child in joint_children]
            if min(joint_sizes) < minimum:
                rejected.append(
                    {
                        "parent_index": int(parent_index),
                        "parent_name": parent.name,
                        "split_kind": "ld_maf4",
                        "reason": "joint_child_too_small",
                        "child_sizes": joint_sizes,
                    }
                )
            else:
                candidates.append(
                    CovTreeCandidate(
                        name=f"p{parent_index:04d}__ld_maf4",
                        parent_index=int(parent_index),
                        parent_name=parent.name,
                        split_kind="ld_maf4",
                        children=joint_children,
                        diagnostics={
                            **common,
                            **ld_diag,
                            **maf_diag,
                            "child_order": [
                                "ld_low_maf_low",
                                "ld_low_maf_high",
                                "ld_high_maf_low",
                                "ld_high_maf_high",
                            ],
                        },
                    )
                )
    return candidates, rejected


def trace_orthogonal_contrasts(trace_atoms: np.ndarray) -> np.ndarray:
    """Return deterministic orthonormal contrasts perpendicular to trace."""
    trace = np.asarray(trace_atoms, dtype=np.float64).reshape(-1)
    child_count = int(trace.size)
    if child_count < 2 or not np.all(np.isfinite(trace)) or np.all(trace == 0.0):
        raise ValueError("trace_atoms must contain at least two finite nonzero values.")

    helmert = np.zeros((child_count, child_count - 1), dtype=np.float64)
    for column in range(child_count - 1):
        scale = np.sqrt(float((column + 1) * (column + 2)))
        helmert[: column + 1, column] = 1.0 / scale
        helmert[column + 1, column] = -float(column + 1) / scale
    projector = np.eye(child_count) - np.outer(trace, trace) / float(trace @ trace)
    projected = projector @ helmert
    basis, _ = np.linalg.qr(projected, mode="reduced")
    basis = basis[:, : child_count - 1]
    if np.linalg.matrix_rank(basis, tol=1e-10) != child_count - 1:
        raise ValueError("Child trace direction does not admit full-rank contrasts.")
    for column in range(basis.shape[1]):
        pivot = int(np.argmax(np.abs(basis[:, column])))
        if basis[pivot, column] < 0.0:
            basis[:, column] *= -1.0
    if not np.allclose(trace @ basis, 0.0, rtol=0.0, atol=1e-10):
        raise RuntimeError("Constructed contrasts are not trace-orthogonal.")
    return basis


def replace_parent(
    components: Sequence[AdaptiveComponent],
    candidate: CovTreeCandidate,
) -> list[AdaptiveComponent]:
    """Replace only the selected parent, retaining all other leaves."""
    if candidate.parent_index < 0 or candidate.parent_index >= len(components):
        raise IndexError("candidate parent_index is out of range.")
    parent = components[candidate.parent_index]
    if parent.name != candidate.parent_name:
        raise ValueError("candidate parent name does not match the current partition.")
    parent_union = np.sort(np.concatenate(candidate.children))
    if not np.array_equal(parent_union, np.sort(parent.variant_indices)):
        raise ValueError("candidate children do not exactly partition the parent.")

    child_count = len(candidate.children)
    if child_count == 2:
        labels = ("low", "high")
    elif child_count > 0 and child_count & (child_count - 1) == 0:
        label_depth = child_count.bit_length() - 1
        labels = tuple(
            format(index, f"0{label_depth}b").translate(str.maketrans("01", "lh"))
            for index in range(child_count)
        )
    else:
        labels = tuple(f"child{index:04d}" for index in range(child_count))
    depth = int(parent.annotation.get("depth", 0)) + 1
    children = [
        AdaptiveComponent(
            name=f"{parent.name}__{candidate.split_kind}_{label}",
            variant_indices=np.asarray(indices, dtype=np.int64),
            annotation={
                "depth": depth,
                "parent_name": parent.name,
                "split_kind": candidate.split_kind,
                "child_label": label,
                "candidate_name": candidate.name,
            },
        )
        for label, indices in zip(labels, candidate.children, strict=True)
    ]
    updated = [
        *components[: candidate.parent_index],
        *children,
        *components[candidate.parent_index + 1 :],
    ]
    n_variants = int(sum(component.variant_indices.size for component in components))
    validate_partition(updated, n_variants=n_variants)
    return updated


def selective_covariance_warm_start(
    parent_theta: np.ndarray,
    components: Sequence[AdaptiveComponent],
    candidate: CovTreeCandidate,
    *,
    child_effective_markers: Sequence[float] | None = None,
) -> np.ndarray:
    """Preserve covariance while replacing one parent by its children."""
    theta = np.asarray(parent_theta, dtype=np.float64).reshape(-1)
    if theta.shape != (len(components) + 1,):
        raise ValueError("parent_theta must contain K genetic values and one residual.")
    if not np.all(np.isfinite(theta)) or np.any(theta[:-1] < 0.0) or theta[-1] <= 0.0:
        raise ValueError("parent_theta contains invalid variance components.")
    parent = components[candidate.parent_index]
    raw_sizes = np.asarray(
        [child.size for child in candidate.children], dtype=np.float64
    )
    if int(raw_sizes.sum()) != int(parent.variant_indices.size):
        raise ValueError("candidate child sizes do not sum to the parent size.")
    weights = (
        raw_sizes
        if child_effective_markers is None
        else np.asarray(child_effective_markers, dtype=np.float64).reshape(-1)
    )
    if (
        weights.shape != raw_sizes.shape
        or not np.all(np.isfinite(weights))
        or np.any(weights <= 0.0)
    ):
        raise ValueError("child_effective_markers must be finite and positive.")
    child_theta = theta[candidate.parent_index] * weights / float(weights.sum())
    return np.concatenate(
        [
            theta[: candidate.parent_index],
            child_theta,
            theta[candidate.parent_index + 1 : -1],
            theta[-1:],
        ]
    )


def bootstrap_conditional_statistics(
    *,
    observed_quadratics: np.ndarray,
    bootstrap_quadratics: np.ndarray,
    nuisance_count: int,
    candidate_slices: Sequence[tuple[int, int]],
    rank_rtol: float = 1e-7,
) -> dict[str, object]:
    """Compute efficient score tests and a layer-wise max-score bootstrap.

    Quadratics are ``(P e)' A (P e)`` for nuisance atoms followed by all
    candidate contrast atoms.  Their bootstrap means estimate ``tr(PA)``;
    their bootstrap covariance estimates the expected score information.
    """
    observed = np.asarray(observed_quadratics, dtype=np.float64).reshape(-1)
    boot = np.asarray(bootstrap_quadratics, dtype=np.float64)
    nuisance = int(nuisance_count)
    rtol = float(rank_rtol)
    if boot.ndim != 2 or boot.shape[0] != observed.size or boot.shape[1] < 3:
        raise ValueError("bootstrap_quadratics must have shape (n_atoms, B>=3).")
    if nuisance < 1 or nuisance >= observed.size:
        raise ValueError("nuisance_count must leave at least one candidate atom.")
    if not np.all(np.isfinite(observed)) or not np.all(np.isfinite(boot)):
        raise ValueError("quadratics must be finite.")
    if not np.isfinite(rtol) or rtol <= 0.0:
        raise ValueError("rank_rtol must be finite and positive.")

    trace = np.mean(boot, axis=1)
    observed_score = 0.5 * (observed - trace)
    bootstrap_score = 0.5 * (boot - trace[:, None])
    information = np.cov(bootstrap_score, bias=False)
    information = 0.5 * (information + information.T)
    eta = np.arange(nuisance, dtype=np.int64)
    info_eta = information[np.ix_(eta, eta)]
    nuisance_eigenvalues = np.linalg.eigvalsh(info_eta)
    nuisance_scale = max(
        float(np.max(np.abs(nuisance_eigenvalues))), np.finfo(float).tiny
    )
    nuisance_rank = int(
        np.count_nonzero(nuisance_eigenvalues > rtol * nuisance_scale)
    )
    pinv_eta = np.linalg.pinv(info_eta, rcond=rtol, hermitian=True)

    candidate_results: list[dict[str, object]] = []
    bootstrap_statistics: list[np.ndarray] = []
    for start, stop in candidate_slices:
        if start < nuisance or stop <= start or stop > observed.size:
            raise ValueError("candidate_slices contain an invalid atom range.")
        d = np.arange(int(start), int(stop), dtype=np.int64)
        info_dd = information[np.ix_(d, d)]
        info_deta = information[np.ix_(d, eta)]
        adjustment = info_deta @ pinv_eta
        conditional = info_dd - adjustment @ information[np.ix_(eta, d)]
        conditional = 0.5 * (conditional + conditional.T)
        eigenvalues = np.linalg.eigvalsh(conditional)
        scale = max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny)
        rank = int(np.count_nonzero(eigenvalues > rtol * scale))
        expected_rank = int(stop - start)
        efficient_observed = observed_score[d] - adjustment @ observed_score[eta]
        efficient_bootstrap = (
            bootstrap_score[d] - adjustment @ bootstrap_score[eta]
        )
        if rank > 0:
            inverse = np.linalg.pinv(conditional, rcond=rtol, hermitian=True)
            statistic = float(efficient_observed @ inverse @ efficient_observed)
            statistic_boot = np.einsum(
                "ib,ij,jb->b",
                efficient_bootstrap,
                inverse,
                efficient_bootstrap,
                optimize=True,
            )
            statistic_boot = np.maximum(statistic_boot, 0.0)
            naive_p = float(chi2.sf(statistic, rank))
            empirical_p = float(
                (1 + np.count_nonzero(statistic_boot >= statistic))
                / (boot.shape[1] + 1)
            )
        else:
            statistic = float("nan")
            statistic_boot = np.full(boot.shape[1], np.nan)
            naive_p = float("nan")
            empirical_p = float("nan")
        candidate_results.append(
            {
                "atom_start": int(start),
                "atom_stop": int(stop),
                "expected_rank": expected_rank,
                "rank": rank,
                "rank_sufficient": bool(rank == expected_rank),
                "conditional_information": conditional.tolist(),
                "conditional_information_eigenvalues": eigenvalues.tolist(),
                "efficient_score": efficient_observed.tolist(),
                "statistic": statistic,
                "naive_chi_square_p": naive_p,
                "candidate_bootstrap_p": empirical_p,
            }
        )
        bootstrap_statistics.append(statistic_boot)

    eligible = [
        index
        for index, result in enumerate(candidate_results)
        if result["rank_sufficient"] and np.isfinite(result["statistic"])
    ]
    if eligible:
        best_index = max(eligible, key=lambda index: candidate_results[index]["statistic"])
        max_bootstrap = np.max(
            np.stack([bootstrap_statistics[index] for index in eligible], axis=0),
            axis=0,
        )
        best_statistic = float(candidate_results[best_index]["statistic"])
        adjusted_p = float(
            (1 + np.count_nonzero(max_bootstrap >= best_statistic))
            / (boot.shape[1] + 1)
        )
        mc_se = float(np.sqrt(adjusted_p * (1.0 - adjusted_p) / (boot.shape[1] + 1)))
    else:
        best_index = None
        adjusted_p = None
        mc_se = None

    return {
        "trace_estimates": trace.tolist(),
        "observed_scores": observed_score.tolist(),
        "nuisance_information": info_eta.tolist(),
        "nuisance_information_eigenvalues": nuisance_eigenvalues.tolist(),
        "nuisance_information_rank": nuisance_rank,
        "nuisance_expected_rank": nuisance,
        "candidates": candidate_results,
        "best_candidate_index": best_index,
        "max_score_adjusted_p": adjusted_p,
        "max_score_adjusted_p_mc_se": mc_se,
        "bootstrap_draws": int(boot.shape[1]),
        "rank_rtol": rtol,
    }


__all__ = [
    "CovTreeCandidate",
    "bootstrap_conditional_statistics",
    "generate_covtree_candidates",
    "replace_parent",
    "selective_covariance_warm_start",
    "trace_orthogonal_contrasts",
]
