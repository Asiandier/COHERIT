"""Branch-matched prediction helpers for sparse REML fits."""

from __future__ import annotations

import dataclasses
import json
import os
from typing import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .pcg import pcg_solve
from .reml_model import (
    EffectEstimates,
    _copy_training_standardization_to_test_streamer,
    _validate_dense_prediction_streamers,
)


@dataclasses.dataclass(frozen=True)
class SparseBranchPrediction:
    """Raw-outcome-scale predictions for one internally coherent branch."""

    name: str
    theta_standardized: np.ndarray
    residual_standardized: np.ndarray
    nuisance_fixed_score_raw: np.ndarray
    fixed_snp_score_raw: np.ndarray
    background_blup_raw: np.ndarray
    background_components_raw: tuple[np.ndarray, ...]
    genetic_score_raw: np.ndarray
    phenotype_prediction_raw: np.ndarray
    pcg_rel_res: float
    pcg_iters: int


@dataclasses.dataclass(frozen=True)
class SparsePathPrediction:
    """Vectorized predictions for a coefficient path at one fixed covariance."""

    residual_standardized: np.ndarray
    dual_raw_objective_scale: np.ndarray
    nuisance_fixed_score_raw: np.ndarray
    fixed_snp_score_raw: np.ndarray
    background_blup_raw: np.ndarray
    genetic_score_raw: np.ndarray
    phenotype_prediction_raw: np.ndarray
    pcg_rel_res: float
    pcg_iters: int


def _as_design(value, *, n_rows: int, name: str) -> np.ndarray:
    if value is None:
        return np.empty((n_rows, 0), dtype=np.float64)
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2 or int(arr.shape[0]) != int(n_rows):
        raise ValueError(f"{name} must be a 2D matrix with {n_rows} rows.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def predict_sparse_branch(
    *,
    name: str,
    fitter,
    test_fitter,
    y_train_raw: np.ndarray,
    train_covar: np.ndarray | None,
    test_covar: np.ndarray | None,
    train_active_geno: np.ndarray,
    test_active_geno: np.ndarray,
    beta_cov_raw: np.ndarray,
    beta_active_raw: np.ndarray,
    theta_standardized: np.ndarray,
    phenotype_scale: float,
    pcg_tol: float,
    max_pcg_iters: int,
) -> SparseBranchPrediction:
    """Predict with one branch's own mean, residual and covariance estimate."""
    y = np.asarray(y_train_raw, dtype=np.float64).reshape(-1)
    n_train = int(y.size)
    if not fitter.streamers or int(fitter.streamers[0].n) != n_train:
        raise ValueError("Training phenotype and genotype row counts do not match.")
    if not test_fitter.streamers:
        raise ValueError("Prediction requires an initialized test fitter.")
    n_test = int(test_fitter.streamers[0].n)
    c_train = _as_design(train_covar, n_rows=n_train, name="train_covar")
    c_test = _as_design(test_covar, n_rows=n_test, name="test_covar")
    z_train = _as_design(
        train_active_geno, n_rows=n_train, name="train_active_geno"
    )
    z_test = _as_design(
        test_active_geno, n_rows=n_test, name="test_active_geno"
    )
    if c_train.shape[1] != c_test.shape[1]:
        raise ValueError("Training and prediction covariate widths do not match.")
    if z_train.shape[1] != z_test.shape[1]:
        raise ValueError("Training and prediction active-genotype widths do not match.")

    beta_cov = np.asarray(beta_cov_raw, dtype=np.float64).reshape(-1)
    beta_active = np.asarray(beta_active_raw, dtype=np.float64).reshape(-1)
    if beta_cov.size != c_train.shape[1] or beta_active.size != z_train.shape[1]:
        raise ValueError("Branch coefficient and design widths do not match.")
    theta = np.asarray(theta_standardized, dtype=np.float64).reshape(-1)
    scale = float(phenotype_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("phenotype_scale must be positive and finite.")
    if (
        not np.all(np.isfinite(theta))
        or np.any(theta[:-1] < 0.0)
        or theta[-1] <= 0.0
    ):
        raise ValueError(
            "theta_standardized requires nonnegative genetic and positive "
            "residual components."
        )

    ops = fitter._assemble_reml_operators()
    if theta.size != len(ops.K_mvs) + 1:
        raise ValueError(
            "theta_standardized length mismatch: expected "
            f"{len(ops.K_mvs) + 1}, got {theta.size}."
        )
    residual_raw = y - c_train @ beta_cov - z_train @ beta_active
    residual_standardized = residual_raw / scale
    theta_dev = jnp.asarray(theta, dtype=jnp.float32)
    fitter._ensure_projected_core_precond_ready(
        ops, var_components_init=theta_dev
    )
    hv = fitter._make_hv(ops, theta_dev[:-1], theta_dev[-1])
    precond = fitter._make_effect_precond(ops, theta_dev[:-1], theta_dev[-1])
    rhs = jnp.asarray(residual_standardized[:, None], dtype=jnp.float32)
    x0 = precond(rhs) if precond is not None else jnp.zeros_like(rhs)
    sol, rel_res, iters = pcg_solve(
        hv,
        rhs,
        M=precond,
        tol=float(pcg_tol),
        maxiter=int(max_pcg_iters),
        X0=x0,
    )
    rel = float(np.asarray(jax.device_get(rel_res)))
    if not np.isfinite(rel) or rel > float(pcg_tol) * 1.05:
        raise RuntimeError(
            f"{name} background-BLUP PCG did not converge: "
            f"relative residual={rel:.3e}."
        )
    dual = sol[:, 0]
    snp_effects = fitter._estimate_snp_effects(dual, theta_dev[:-1])
    fixed_raw = np.concatenate([beta_cov, beta_active])
    zeros_train = jnp.zeros((n_train,), dtype=jnp.float32)
    effects = EffectEstimates(
        fixed_effects=jnp.asarray(fixed_raw / scale, dtype=jnp.float32),
        random_effect=zeros_train,
        random_effect_components=tuple(
            zeros_train for _ in range(len(ops.K_mvs))
        ),
        snp_effects=snp_effects,
        pcg_rel_res=rel,
        pcg_iters=int(iters),
        y_mean=0.0,
        y_scale=scale,
    )
    test_design = np.concatenate([c_test, z_test], axis=1)
    predictions = fitter.predict(
        effects,
        test_fitter=test_fitter,
        test_covar=(
            jnp.asarray(test_design, dtype=jnp.float32)
            if test_design.shape[1]
            else None
        ),
    )
    nuisance_raw = c_test @ beta_cov
    fixed_snp_raw = z_test @ beta_active
    background_raw = scale * np.asarray(
        jax.device_get(predictions.random_effect), dtype=np.float64
    )
    component_raw = tuple(
        scale * np.asarray(jax.device_get(component), dtype=np.float64)
        for component in predictions.random_effect_components
    )
    genetic_raw = fixed_snp_raw + background_raw
    phenotype_raw = nuisance_raw + genetic_raw
    returned_raw = np.asarray(
        jax.device_get(predictions.y_pred), dtype=np.float64
    )
    if not np.allclose(returned_raw, phenotype_raw, rtol=5e-5, atol=5e-5):
        raise RuntimeError(
            f"{name} prediction scale decomposition is inconsistent."
        )
    return SparseBranchPrediction(
        name=str(name),
        theta_standardized=theta.copy(),
        residual_standardized=residual_standardized,
        nuisance_fixed_score_raw=nuisance_raw,
        fixed_snp_score_raw=fixed_snp_raw,
        background_blup_raw=background_raw,
        background_components_raw=component_raw,
        genetic_score_raw=genetic_raw,
        phenotype_prediction_raw=phenotype_raw,
        pcg_rel_res=rel,
        pcg_iters=int(iters),
    )


def _as_coefficient_path(
    value,
    *,
    n_columns: int,
    name: str,
) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2 or int(arr.shape[1]) != int(n_columns):
        raise ValueError(
            f"{name} must have shape (path_length, {int(n_columns)})."
        )
    if int(arr.shape[0]) < 1 or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain a finite, non-empty path.")
    return arr


def _pack_effect_path_by_call(streamer, effect_path: np.ndarray) -> jnp.ndarray:
    effects = np.asarray(effect_path, dtype=np.float32)
    if effects.ndim != 2 or int(effects.shape[0]) != int(streamer.m):
        raise ValueError(
            "effect_path must contain one row per partitioned-stream marker."
        )
    n_path = int(effects.shape[1])
    packed = np.zeros(
        (
            n_path,
            int(streamer._n_calls),
            int(streamer._max_unpack_width),
        ),
        dtype=np.float32,
    )
    for call_index in range(int(streamer._n_calls)):
        start = int(streamer._call_snp_starts[call_index])
        width = int(streamer._call_true_widths[call_index])
        packed[:, call_index, :width] = effects[
            start : start + width, :
        ].T
    return jax.device_put(jnp.asarray(packed), streamer.dev)


def predict_sparse_path_partitioned(
    *,
    fitter,
    test_fitter,
    y_train_raw: np.ndarray,
    train_covar: np.ndarray | None,
    test_covar: np.ndarray | None,
    train_candidate_geno: np.ndarray,
    test_candidate_geno: np.ndarray,
    beta_cov_path_raw: np.ndarray,
    beta_candidate_path_raw: np.ndarray,
    theta_standardized: np.ndarray,
    phenotype_scale: float,
    pcg_tol: float,
    max_pcg_iters: int,
) -> SparsePathPrediction:
    """Predict an entire Lasso path in one PCG and one genotype pass.

    Path points share a fixed covariance estimate.  This is the inexpensive
    validation scan used to select ``lambda / lambda_max`` before a full refit.
    """
    train_streamer = getattr(fitter, "_partitioned_streamer", None)
    test_streamer = getattr(test_fitter, "_partitioned_streamer", None)
    if train_streamer is None or test_streamer is None:
        raise ValueError(
            "Sparse path prediction requires a single-source component partition."
        )
    _validate_dense_prediction_streamers(
        fitter.streamers, test_fitter.streamers
    )
    _copy_training_standardization_to_test_streamer(
        train_streamer, test_streamer
    )

    y = np.asarray(y_train_raw, dtype=np.float64).reshape(-1)
    n_train = int(y.size)
    n_test = int(test_streamer.n)
    c_train = _as_design(train_covar, n_rows=n_train, name="train_covar")
    c_test = _as_design(test_covar, n_rows=n_test, name="test_covar")
    z_train = _as_design(
        train_candidate_geno,
        n_rows=n_train,
        name="train_candidate_geno",
    )
    z_test = _as_design(
        test_candidate_geno,
        n_rows=n_test,
        name="test_candidate_geno",
    )
    if c_train.shape[1] != c_test.shape[1]:
        raise ValueError("Training and validation covariate widths do not match.")
    if z_train.shape[1] != z_test.shape[1]:
        raise ValueError("Training and validation candidate widths do not match.")

    beta_cov_path = _as_coefficient_path(
        beta_cov_path_raw,
        n_columns=c_train.shape[1],
        name="beta_cov_path_raw",
    )
    beta_candidate_path = _as_coefficient_path(
        beta_candidate_path_raw,
        n_columns=z_train.shape[1],
        name="beta_candidate_path_raw",
    )
    if beta_cov_path.shape[0] != beta_candidate_path.shape[0]:
        raise ValueError("Covariate and candidate coefficient paths differ in length.")
    n_path = int(beta_candidate_path.shape[0])

    theta = np.asarray(theta_standardized, dtype=np.float64).reshape(-1)
    scale = float(phenotype_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("phenotype_scale must be positive and finite.")
    if (
        theta.shape != (int(train_streamer.n_components) + 1,)
        or not np.all(np.isfinite(theta))
        or np.any(theta[:-1] < 0.0)
        or theta[-1] <= 0.0
    ):
        raise ValueError("theta_standardized is incompatible with the partition.")

    nuisance_train = c_train @ beta_cov_path.T
    fixed_train = z_train @ beta_candidate_path.T
    residual_raw = y[:, None] - nuisance_train - fixed_train
    residual_standardized = residual_raw / scale

    ops = fitter._assemble_reml_operators()
    theta_dev = jnp.asarray(theta, dtype=jnp.float32)
    fitter._ensure_projected_core_precond_ready(
        ops, var_components_init=theta_dev
    )
    hv = fitter._make_hv(ops, theta_dev[:-1], theta_dev[-1])
    precond = fitter._make_effect_precond(
        ops, theta_dev[:-1], theta_dev[-1]
    )
    rhs = jnp.asarray(residual_standardized, dtype=jnp.float32)
    x0 = precond(rhs) if precond is not None else jnp.zeros_like(rhs)
    dual_standardized, rel_res, iters = pcg_solve(
        hv,
        rhs,
        M=precond,
        tol=float(pcg_tol),
        maxiter=int(max_pcg_iters),
        X0=x0,
    )
    rel = float(np.asarray(jax.device_get(rel_res)))
    if not np.isfinite(rel) or rel > float(pcg_tol) * 1.05:
        raise RuntimeError(
            "Sparse path background-BLUP PCG did not converge: "
            f"relative residual={rel:.3e}."
        )

    xt_dual = np.asarray(
        jax.device_get(
            train_streamer.xtv(dual_standardized, normalize=False)
        ),
        dtype=np.float32,
    )
    expected_xt_shape = (int(train_streamer.m), n_path)
    if xt_dual.shape != expected_xt_shape:
        raise RuntimeError(
            "Partitioned X'V^-1 residual path has the wrong shape: "
            f"{xt_dual.shape} != {expected_xt_shape}."
        )
    random_effect_path = np.zeros_like(xt_dual, dtype=np.float32)
    for component_index in range(int(train_streamer.n_components)):
        start = int(train_streamer._component_snp_offsets[component_index])
        stop = int(train_streamer._component_snp_offsets[component_index + 1])
        effective_m = float(
            train_streamer._component_eff_m_host[component_index]
        )
        if effective_m > 0.0:
            random_effect_path[start:stop, :] = (
                float(theta[component_index])
                / effective_m
                * xt_dual[start:stop, :]
            )

    from .kv_impl import zxb_impl_same_stream_multi

    test_streamer._prepare_kv_pass()
    packed = _pack_effect_path_by_call(
        train_streamer, random_effect_path
    )
    _summed_paths, path_predictions = zxb_impl_same_stream_multi(
        packed,
        test_streamer._true_widths_dev,
        test_streamer._means_by_call,
        test_streamer._inv_by_call,
        n=n_test,
        n_calls=int(test_streamer._n_calls),
        pop_block=test_streamer._pop_cached,
        missing_val=int(test_streamer._missing_val),
    )
    background_raw = scale * np.column_stack(
        [
            np.asarray(jax.device_get(values), dtype=np.float64)
            for values in path_predictions
        ]
    )
    if background_raw.shape != (n_test, n_path):
        raise RuntimeError("Sparse path background prediction shape mismatch.")

    nuisance_raw = c_test @ beta_cov_path.T
    fixed_raw = z_test @ beta_candidate_path.T
    genetic_raw = fixed_raw + background_raw
    phenotype_raw = nuisance_raw + genetic_raw
    arrays = (
        residual_standardized,
        nuisance_raw,
        fixed_raw,
        background_raw,
        genetic_raw,
        phenotype_raw,
    )
    if any(not np.all(np.isfinite(values)) for values in arrays):
        raise RuntimeError("Sparse path prediction contains non-finite values.")

    return SparsePathPrediction(
        residual_standardized=residual_standardized,
        # The Lasso objective uses H_std^-1 applied to the raw-scale residual.
        dual_raw_objective_scale=(
            scale
            * np.asarray(
                jax.device_get(dual_standardized), dtype=np.float64
            )
        ),
        nuisance_fixed_score_raw=nuisance_raw,
        fixed_snp_score_raw=fixed_raw,
        background_blup_raw=background_raw,
        genetic_score_raw=genetic_raw,
        phenotype_prediction_raw=phenotype_raw,
        pcg_rel_res=rel,
        pcg_iters=int(iters),
    )


def _metadata_path(out_prefix: str) -> str:
    return out_prefix + ".sparse_prediction_metadata.json"


def _table_path(out_prefix: str) -> str:
    return out_prefix + ".sparse_prediction.tsv"


def remove_sparse_prediction_outputs(out_prefix: str) -> None:
    """Remove prediction artifacts when a run did not request prediction."""
    for path in (_table_path(out_prefix), _metadata_path(out_prefix)):
        if os.path.exists(path):
            os.remove(path)


def write_sparse_prediction_status(
    *, out_prefix: str, status: str, metadata: Mapping[str, object]
) -> str:
    """Write status metadata and ensure rejected fits expose no branch table."""
    table_path = _table_path(out_prefix)
    if status != "emitted" and os.path.exists(table_path):
        os.remove(table_path)
    path = _metadata_path(out_prefix)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    payload = dict(metadata)
    payload.update({"schema_version": 1, "status": str(status)})
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
    return path


def write_sparse_prediction_outputs(
    *,
    out_prefix: str,
    sample_ids: Sequence[str],
    lasso: SparseBranchPrediction | None,
    selected_span: SparseBranchPrediction | None,
    metadata: Mapping[str, object],
) -> dict[str, str]:
    """Write every available sparse branch to one aligned prediction table."""
    n = len(sample_ids)
    branches = (
        ("lasso", lasso),
        ("selected_span", selected_span),
    )
    available = [(prefix, branch) for prefix, branch in branches if branch is not None]
    if not available:
        raise ValueError("At least one valid sparse prediction branch is required.")
    suffixes = (
        "fixed_snp_score_raw",
        "background_blup_raw",
        "genetic_score_raw",
        "nuisance_fixed_score_raw",
        "phenotype_prediction_raw",
    )
    arrays: list[np.ndarray] = []
    columns = ["sample_index", "iid"]
    for prefix, branch in available:
        columns.extend(f"{prefix}_{suffix}" for suffix in suffixes)
        arrays.extend(
            (
                branch.fixed_snp_score_raw,
                branch.background_blup_raw,
                branch.genetic_score_raw,
                branch.nuisance_fixed_score_raw,
                branch.phenotype_prediction_raw,
            )
        )
    if any(np.asarray(arr).size != n for arr in arrays):
        raise ValueError(
            "Sparse prediction arrays and sample IDs have different lengths."
        )
    table_path = _table_path(out_prefix)
    parent = os.path.dirname(table_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(table_path, "w", encoding="utf-8") as handle:
        handle.write("\t".join(columns) + "\n")
        for idx, iid in enumerate(sample_ids):
            values = [str(idx), str(iid)] + [
                f"{float(np.asarray(arr)[idx]):.10g}" for arr in arrays
            ]
            handle.write("\t".join(values) + "\n")
    status_path = write_sparse_prediction_status(
        out_prefix=out_prefix,
        status="emitted",
        metadata={
            **dict(metadata),
            "n_samples": n,
            "columns": columns,
        },
    )
    return {"prediction": table_path, "metadata": status_path}
