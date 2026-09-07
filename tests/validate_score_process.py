"""Offline validation of the production joint score reference under known nulls.

Run with MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1:
  python tests/validate_score_process.py --out /tmp/score-validation.json

Phenotype generation is only for this validation. The production calibrator
integrates small quadratic matrices; it never fits simulated phenotypes.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
SCORE = importlib.import_module(f"{ROOT.name}.score_process")
ADAPTIVE = importlib.import_module(f"{ROOT.name}.adaptive_ld")


def proportion(count, total):
    rate = count / total
    z = 1.959963984540054
    denominator = 1 + z*z/total
    center = (rate + z*z/(2*total)) / denominator
    radius = z*np.sqrt(rate*(1-rate)/total + z*z/(4*total*total)) / denominator
    return {"count": int(count), "total": total, "rate": rate,
            "wilson_95": [max(0.0, center-radius), min(1.0, center+radius)]}


def experiment(n, m, rho, bins, *, seed, repetitions, trace_seeds, reference_samples):
    rng = np.random.default_rng(seed)
    latent = np.repeat(rng.normal(size=(n, 8)), m//8, axis=1)
    z = np.sqrt(rho)*latent + np.sqrt(1-rho)*rng.normal(size=(n, m))
    z = (z-z.mean(0))/z.std(0)
    k = z@z.T/m
    theta = np.array([0.5, 0.5])
    v = 0.5*k + 0.5*np.eye(n)
    vi = np.linalg.inv(v)
    covar = np.ones((n, 1))
    p = vi-vi@covar@np.linalg.solve(covar.T@vi@covar, covar.T@vi)
    positions = np.arange(m//bins, m, m//bins)
    contractions = ADAPTIVE.BoundaryContractions(
        np.arange(m), positions, positions, [(0, m)], np.array([m]),
    )
    directions = np.stack([k, np.eye(n)] + [
        z[:, :t]@z[:, :t].T/t - z[:, t:]@z[:, t:].T/(m-t) for t in positions
    ])
    pd = p@directions
    trace = np.trace(pd, axis1=1, axis2=2)
    fisher = 0.5*np.einsum("aij,bji->ab", pd, pd)
    coef = np.linalg.solve(fisher[:2, :2], fisher[:2, 2:]).T
    info = np.diag(fisher[2:, 2:]-coef@fisher[:2, 2:])
    values, basis = np.linalg.eigh(p)
    root = (basis*np.sqrt(np.maximum(values, 0)))@basis.T
    adjusted = directions[2:]-np.einsum("ta,aij->tij", coef, directions[:2])
    exact_matrix = root@adjusted@root/np.sqrt(info)[:, None, None]
    raw_zero = -0.5*trace
    exact_process = SCORE.ScoreProcess(
        positions, raw_zero[2:], raw_zero[2:]-coef@raw_zero[:2], info, coef, exact_matrix, {},
    )
    _, exact_calibration = SCORE.calibrate_boundary_maximum(
        exact_process, samples=reference_samples, seed=seed+7000,
    )
    models = [{
        "coefficients": coef, "information": info, "offset": exact_process.scores,
        "critical": exact_calibration["reference_critical_value"], "diagnostics": {},
    }]
    index = SimpleNamespace(
        streamer=SimpleNamespace(n=n, _component_eff_m_host=np.array([m]),
                                 kv=lambda x: k@np.asarray(x)),
        m_total=m, offsets=np.array([0, m]),
        xtv_all=lambda x, **_: z.T@np.asarray(x),
        extract_standardized_columns=lambda indices: z[:, indices].astype(np.float32),
    )
    projector = SimpleNamespace(_covar=covar, theta=theta, apply=lambda x, **_: p@np.asarray(x))
    with tempfile.TemporaryDirectory(prefix="validate_joint_score_") as directory:
        for i in range(trace_seeds):
            args = SimpleNamespace(
                score_trace_seed=seed+100+i, score_core_rank=64,
                score_trace_probes=512, score_trace_max_probes=4096, score_trace_tol=0.05,
            )
            with ADAPTIVE.ScoreWorkspace(directory) as workspace:
                process = ADAPTIVE._estimate_score_process(
                    args, projector, index, contractions, np.zeros(n), workspace=workspace,
                )
                _, calibration = SCORE.calibrate_boundary_maximum(
                    process, samples=reference_samples, seed=process.diagnostics["reference_seed"],
                )
                assert np.all(process.information > 0)
                delta = process.coefficients-coef
                actual_info = info + np.einsum("ta,ab,tb->t", delta, fisher[:2, :2], delta)
                exact_offset = -0.5*(trace[2:]-process.coefficients@trace[:2])
                diagnostic = {
                    **process.diagnostics,
                    "max_information_relative_error": float(np.max(
                        np.abs(process.information/actual_info-1))),
                    "max_standardized_centering_error": float(np.max(
                        np.abs(process.scores-exact_offset)/np.sqrt(process.information))),
                }
                models.append({
                    "coefficients": process.coefficients.copy(), "information": process.information.copy(),
                    "offset": process.scores.copy(), "critical": calibration["reference_critical_value"],
                    "diagnostics": diagnostic,
                })
    half = m//2
    alternative = (0.49*z[:, :half]@z[:, :half].T/half
                   + 0.01*z[:, half:]@z[:, half:].T/half + 0.5*np.eye(n))
    value, basis = np.linalg.eigh(p@alternative@p)
    alternative_root = (basis*np.sqrt(np.maximum(value, 0)))@basis.T
    rates = {}
    for label, sampling_root in (("null", root), ("alternative", alternative_root)):
        counts = np.zeros(len(models), dtype=int)
        for offset in range(0, repetitions, 1000):
            py = sampling_root@rng.normal(size=(n, min(1000, repetitions-offset)))
            marker = z.T@py
            quadratic = contractions.apply(py, py, marker, marker)
            for i, model in enumerate(models):
                score = (0.5*(quadratic[2:]-model["coefficients"]@quadratic[:2])
                         + model["offset"][:, None])
                maximum = np.max(np.abs(score)/np.sqrt(model["information"][:, None]), axis=0)
                counts[i] += np.count_nonzero(maximum > model["critical"])
        rates[label] = [proportion(int(count), repetitions) for count in counts]
    return {
        "n": n, "m": m, "rho": rho, "candidate_count": len(positions),
        "reference_samples": reference_samples, "repetitions_per_model": repetitions,
        "exact_reference": {label: values[0] for label, values in rates.items()},
        "trace_references": [
            {"trace_seed": seed+100+i, **models[i+1]["diagnostics"],
             **{label: values[i+1] for label, values in rates.items()}}
            for i in range(trace_seeds)
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--repetitions", type=int, default=30000)
    parser.add_argument("--trace-seeds", type=int, default=2)
    parser.add_argument("--reference-samples", type=int, default=16383)
    args = parser.parse_args()
    if args.repetitions < 1 or args.trace_seeds < 1 or args.reference_samples < 255:
        parser.error("Repetitions/seeds must be positive; reference samples must be >=255.")
    output = {"scope": "fixed correct mean/covariance, one given candidate family", "results": []}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for i, (n, m, rho, bins) in enumerate(((128, 256, 0., 128), (128, 256, .9, 128),
                                         (512, 512, 0., 32), (512, 512, .9, 32))):
        result = experiment(n, m, rho, bins, seed=20260910+i, repetitions=args.repetitions,
                            trace_seeds=args.trace_seeds, reference_samples=args.reference_samples)
        output["results"].append(result)
        args.out.write_text(json.dumps(output, indent=2)+"\n")
        print(json.dumps(result), flush=True)
