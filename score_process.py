"""Efficient covariance scores with one joint Gaussian quadratic reference.

The common core and the off-diagonal probe remainder retain non-Gaussian tails
and dependence between cuts. Traces, fitted covariance and finite integration
make the reported p-value a plug-in approximation, not a certified bound.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class TracePrecisionError(RuntimeError):
    """The probes cannot yet resolve the score's centering and information."""


def direction_batch_size(dimension: int) -> int:
    """Bound the live float64 projection blocks, independently of family size."""
    return max(1, min(32, (32 * 1024**2) // (8 * max(1, dimension)**2)))


@dataclass
class TraceSketch:
    """Common-core/probe contractions for all nuisance and candidate directions.

    core=H'DH, cross=H'DB, diagonal=diag(B'DB); bulk is B'DB with its
    diagonal removed. B has independent columns with covariance P-HH'.
    Large arrays may be file-backed; validation and products use bounded blocks.
    """

    core: np.ndarray
    cross: np.ndarray
    bulk: np.ndarray
    diagonal: np.ndarray

    def __post_init__(self) -> None:
        for name in ("core", "cross", "bulk", "diagonal"):
            value = np.asarray(getattr(self, name))
            if value.dtype != np.float64:
                value = value.astype(np.float64)
            setattr(self, name, value)
        if self.core.ndim != 3 or self.bulk.ndim != 3:
            raise ValueError("Core and bulk contractions must be three-dimensional.")
        d, rank, rank2 = self.core.shape
        count = self.bulk.shape[-1]
        if (rank != rank2 or count < 3 or self.bulk.shape != (d, count, count)
                or self.cross.shape != (d, rank, count)
                or self.diagonal.shape != (d, count)):
            raise ValueError("Trace block shapes must align, with at least three probes.")
        width = direction_batch_size(rank + count)
        for start in range(0, d, width):
            for block in (self.core, self.cross, self.bulk, self.diagonal):
                if not np.all(np.isfinite(block[start:start + width])):
                    raise ValueError("Trace contractions must be finite.")
        if np.any(np.diagonal(self.bulk, axis1=1, axis2=2) != 0):
            raise ValueError("The probe bulk must have zero diagonal.")

    @property
    def probes(self) -> int:
        return self.bulk.shape[-1]

    @property
    def rank(self) -> int:
        return self.core.shape[-1]

    def information_rows(self, count: int) -> np.ndarray:
        """PSD all-pairs Fisher Gram, computing only the requested leading rows."""
        directions = len(self.core)
        if not 1 <= count <= directions:
            raise ValueError("Nuisance count must match the trace directions.")
        rows = np.zeros((count, directions), dtype=np.float64)
        weights = (0.5, 1 / self.probes, 1 / (2 * self.probes * (self.probes - 1)))
        # Contiguous feature slices avoid copying a full file-backed sketch.
        width = max(1, min(4096, 32 * 1024**2 // (8 * max(1, directions))))
        for block, weight in zip((self.core, self.cross, self.bulk), weights, strict=True):
            flat = block.reshape(directions, -1)
            for start in range(0, flat.shape[1], width):
                feature = np.ascontiguousarray(flat[:, start:start + width])
                rows += weight * (feature[:count] @ feature.T)
        if not np.all(np.isfinite(rows)):
            raise FloatingPointError("Non-finite Fisher information.")
        return rows


def fit_nuisance_projection(pilot: TraceSketch, nuisance_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit only nuisance Fisher rows using independent pilot probes."""
    q = int(nuisance_count)
    rows = pilot.information_rows(q)
    information = 0.5 * (rows[:, :q] + rows[:, :q].T)
    diagonal = np.diag(information)
    scale = np.sqrt(np.where(diagonal > 0, diagonal, 1.0))
    normalized = information / scale[:, None] / scale[None, :]
    eigenvalues, basis = np.linalg.eigh(normalized)
    tolerance = 1e-10 * max(1.0, float(eigenvalues[-1]))
    if eigenvalues[0] < -tolerance:
        raise FloatingPointError("Nuisance Fisher Gram is indefinite beyond roundoff.")
    keep = eigenvalues > tolerance
    inverse = (basis[:, keep] / eigenvalues[keep]) @ basis[:, keep].T
    inverse /= scale[:, None] * scale[None, :]
    coefficients = rows[:, q:].T @ inverse
    if not np.all(np.isfinite(coefficients)):
        raise FloatingPointError("Non-finite nuisance projection coefficients.")
    return coefficients, eigenvalues


@dataclass
class ScoreProcess:
    positions: np.ndarray
    raw_scores: np.ndarray
    scores: np.ndarray
    information: np.ndarray
    coefficients: np.ndarray
    reference_matrices: np.ndarray
    diagnostics: dict


def score_process_moments(
    quadratics: np.ndarray,
    *,
    evaluation: TraceSketch,
    coefficients: np.ndarray,
    boundary_positions: np.ndarray,
    reference_out: np.ndarray | None = None,
) -> ScoreProcess:
    """Evaluate the actual pilot-fitted directions, without another Schur fit.

    The reference covariance is exactly the same all-pairs Fisher estimate.
    Information uncertainty uses a delete-one jackknife of the combined cross
    and bulk terms, retaining their dependence. Zero-information rows have a
    zero reference matrix and are excluded from the test.
    """
    quadratics = np.asarray(quadratics, dtype=np.float64).reshape(-1)
    positions = np.asarray(boundary_positions, dtype=np.int64).reshape(-1)
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if coefficients.ndim != 2 or coefficients.shape[0] != len(positions):
        raise ValueError("Projection coefficients and candidate positions must align.")
    count, rank = evaluation.probes, evaluation.rank
    q, candidates = coefficients.shape[1], len(positions)
    if (q < 1 or len(evaluation.core) != q + candidates
            or quadratics.size != q + candidates):
        raise ValueError("Score directions and trace blocks do not align.")
    if (not np.all(np.isfinite(quadratics)) or not np.all(np.isfinite(coefficients))
            or np.any(np.diff(positions) <= 0)):
        raise ValueError("Scores must be finite and boundary positions strictly ordered.")
    shape = (candidates, rank + count, rank + count)
    reference = np.empty(shape) if reference_out is None else reference_out
    if reference.shape != shape or reference.dtype not in (np.float32, np.float64):
        raise ValueError("Reference output must be a floating matrix for each candidate.")
    traces = np.trace(evaluation.core, axis1=1, axis2=2) + evaluation.diagonal.mean(1)
    raw_scores = 0.5 * (quadratics - traces)
    scores = raw_scores[q:] - coefficients @ raw_scores[:q]
    information, trace_se, info_se = np.empty((3, candidates))
    width = direction_batch_size(rank + count)
    for start in range(0, candidates, width):
        stop = min(candidates, start + width)
        coef = coefficients[start:stop]
        a, e, c, diagonal = [
            block[q + start:q + stop] - np.tensordot(coef, block[:q], axes=1)
            for block in (evaluation.core, evaluation.cross, evaluation.bulk, evaluation.diagonal)
        ]
        e_rows = np.sum(e * e, axis=1)
        c_rows = np.sum(c * c, axis=2)
        info = (0.5 * np.sum(a * a, axis=(1, 2)) + e_rows.sum(1) / count
                + 0.5 * c_rows.sum(1) / (count * (count - 1)))
        # Compare cancellation with the raw direction's scale, not an absolute
        # cutoff, so multiplying a contrast by a tiny/large constant is harmless.
        raw_info = (
            0.5 * np.sum(evaluation.core[q + start:q + stop]**2, axis=(1, 2))
            + np.sum(evaluation.cross[q + start:q + stop]**2, axis=(1, 2)) / count
            + np.sum(evaluation.bulk[q + start:q + stop]**2, axis=(1, 2))
            / (2 * count * (count - 1))
        )
        if not all(np.all(np.isfinite(x)) for x in (info, raw_info, scores[start:stop])):
            raise FloatingPointError("Non-finite score information after trace contraction.")
        keep = info > 64 * np.finfo(float).eps * (q + 1) * raw_info
        information[start:stop] = np.where(keep, info, 0.0)
        trace_se[start:stop] = 0.5 * diagonal.std(1, ddof=1) / np.sqrt(count)
        deleted = ((e_rows.sum(1)[:, None] - e_rows) / (count - 1)
                   + 0.5 * (c_rows.sum(1)[:, None] - 2 * c_rows)
                   / ((count - 1) * (count - 2)))
        info_se[start:stop] = np.sqrt(
            (count - 1) / count * np.sum((deleted - deleted.mean(1)[:, None])**2, axis=1)
        )
        scale = np.sqrt(np.where(keep, info, 1.0))[:, None, None]
        target = reference[start:stop]
        target[:, :rank, :rank] = a / scale
        target[:, :rank, rank:] = e / (np.sqrt(count) * scale)
        target[:, rank:, :rank] = e.swapaxes(1, 2) / (np.sqrt(count) * scale)
        target[:, rank:, rank:] = c / (np.sqrt(count * (count - 1)) * scale)
        target[~keep] = 0
        if not np.all(np.isfinite(target)):
            raise FloatingPointError("Non-finite normalized quadratic reference.")
    keep = information > 0
    if not all(np.all(np.isfinite(x)) for x in (raw_scores, trace_se, info_se)):
        raise FloatingPointError("Non-finite score or trace uncertainty.")
    diagnostics = {
        "core_rank": rank,
        "evaluation_probes": count,
        "reference_dimension": rank + count,
        "nuisance_score": raw_scores[:q].tolist(),
        "max_score_trace_standard_error": float(np.max(
            trace_se[keep] / np.sqrt(information[keep]), initial=0.0)),
        "max_information_relative_standard_error": float(np.max(
            info_se[keep] / information[keep], initial=0.0)),
        "untestable_candidate_count": int(np.count_nonzero(~keep)),
        "untestable_boundary_positions": positions[~keep].tolist(),
    }
    return ScoreProcess(positions, raw_scores[q:], scores, information, coefficients,
                        reference, diagnostics)


def integrate_boundary_maximum(
    matrices: np.ndarray, *, samples: int, seed: int, batch_size: int = 256,
) -> np.ndarray:
    """Integrate one common quadratic law in bounded candidate/sample batches.

    Each candidate batch replays the same Gaussian stream. Changing candidate
    batching cannot make cuts independent. Only small matrices enter JAX; no P
    application, genotype access or phenotype fit occurs inside this loop.
    """
    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
        raise ValueError("The joint reference requires square matrices.")
    if samples < 1 or seed < 0 or batch_size < 1:
        raise ValueError("Integration requires positive counts and a nonnegative seed.")
    # Lazy import keeps trace algebra usable without initializing a JAX device.
    import jax
    import jax.numpy as jnp

    @jax.jit
    def maximum(block, coordinates):
        values = 0.5 * (
            jnp.einsum("bi,tij,bj->bt", coordinates, block, coordinates,
                       precision=jax.lax.Precision.HIGHEST, optimize="optimal")
            - jnp.trace(block, axis1=1, axis2=2)
        )
        return jnp.max(jnp.abs(values), axis=1)

    result = np.zeros(samples, dtype=np.float64)
    width = direction_batch_size(matrices.shape[-1])
    for start in range(0, len(matrices), width):
        host = np.asarray(matrices[start:start + width], dtype=np.float32)
        if not np.all(np.isfinite(host)):
            raise ValueError("Reference matrices must be finite.")
        block = jnp.asarray(host)
        rng = np.random.default_rng(seed)
        for offset in range(0, samples, batch_size):
            stop = min(samples, offset + batch_size)
            coordinates = rng.standard_normal((stop - offset, matrices.shape[-1]), dtype=np.float32)
            values = np.asarray(maximum(block, jnp.asarray(coordinates)), dtype=np.float64)
            if not np.all(np.isfinite(values)):
                raise FloatingPointError("Non-finite joint reference maximum.")
            result[offset:stop] = np.maximum(result[offset:stop], values)
    return result


def calibrate_boundary_maximum(
    process: ScoreProcess, *, alpha: float = 0.05, samples: int = 16383, seed: int = 20260831,
) -> tuple[list[dict], dict]:
    """Rank the maximum |Z| against its joint reference; select by |Z|.

    The +1 rank p-value is finite-MC valid only for an exchangeable exact null.
    Fitted covariance and finite probes leave this implementation a plug-in
    test. Integration size is fixed before examining its tail count.
    """
    if not 0 < alpha < 1 or samples < 1 or 1 / (samples + 1) > alpha:
        raise ValueError("Reference sample count must resolve the requested alpha in (0,1).")
    count = len(process.positions)
    if any(np.shape(x) != (count,) for x in (
        process.scores, process.information, process.raw_scores,
    )) or process.reference_matrices.shape[0] != count:
        raise ValueError("Score process shapes do not align.")
    if (not all(np.all(np.isfinite(x)) for x in (
            process.scores, process.information, process.raw_scores))
            or np.any(process.information < 0)):
        raise ValueError("Scores require finite values and nonnegative information.")
    keep = process.information > 0
    with np.errstate(over="raise", invalid="raise"):
        z = process.scores[keep] / np.sqrt(process.information[keep])
        squared = z * z
    rows = [{
        "boundary_position": int(position), "raw_score": float(raw),
        "efficient_score": float(score), "efficient_information": float(info),
        "standardized_score": float(standardized), "score_statistic": float(statistic),
    } for position, raw, score, info, standardized, statistic in zip(
        process.positions[keep], process.raw_scores[keep], process.scores[keep],
        process.information[keep], z, squared, strict=True
    )]
    rows.sort(key=lambda row: (-row["score_statistic"], row["boundary_position"]))
    observed = float(np.max(np.abs(z), initial=0.0))
    p_value, exceedances, critical = 1.0, 0, None
    if rows:
        reference = integrate_boundary_maximum(process.reference_matrices, samples=samples, seed=seed)
        exceedances = int(np.count_nonzero(reference >= observed))
        p_value = (1 + exceedances) / (samples + 1)
        index = samples - int(np.floor(alpha * (samples + 1)))
        critical = float(np.partition(reference, index)[index])
    diagnostics = {
        **process.diagnostics,
        "testable_candidate_count": len(rows),
        "maximum_absolute_score": observed,
        "global_p_value": p_value,
        "reference_samples": samples if rows else 0,
        "reference_seed": seed,
        "reference_tail_count": exceedances if rows else None,
        "reference_critical_value": critical,
        "calibration_method": "joint_core_probe_quadratic" if rows else "empty_family",
        "accepted": bool(rows and p_value <= alpha),
        "control_scope": "single_given_candidate_family",
        "calibration_status": "fitted_null_finite_probe_plugin",
    }
    return rows, diagnostics
