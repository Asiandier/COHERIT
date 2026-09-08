"""Repeated-response REML setup is reusable, not its fitted covariance."""
import importlib
from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
REML = importlib.import_module(f"{ROOT.name}.reml")
MODEL = importlib.import_module(f"{ROOT.name}.reml_model")


def test_sparse_reml_setup_reuses_operators_and_invalidates_transforms(monkeypatch):
    class Streamer:
        n, m, _n_calls = 8, 16, 1

        def __init__(self):
            self._means_by_call = np.zeros(16)

        def kv(self, value, normalize=True):
            return value

        def diag(self):
            return jnp.ones(self.n)

    captured = []

    def fit(**kwargs):
        captured.append((kwargs["K_mvs"], kwargs["probe_cache"]))
        return jnp.array([0.3, 0.7]), []

    monkeypatch.setattr(MODEL, "GenoBlockStreamer", lambda **kwargs: Streamer())
    monkeypatch.setattr(MODEL, "fit_reml", fit)
    fitter = MODEL.InfinitesimalREMLFitter(MODEL.FitConfig(
        sources=[object()], cache_reml_setup=True, precond_rank=0, verbose=False,
    ))
    try:
        fitter.fit_infinitesimal(jnp.ones(8))
        fitter.fit_infinitesimal(jnp.arange(8))
        assert captured[0][0] is captured[1][0]
        assert captured[0][1] is captured[1][1]
        fitter.streamers[0]._means_by_call = np.ones(16)
        fitter.fit_infinitesimal(jnp.arange(8))
        assert captured[2][0] is not captured[1][0]
        assert captured[2][1] is not captured[1][1]
    finally:
        fitter.close()
    assert fitter._reml_fit_setup is None
    assert fitter._reml_probe_cache.kvrand_stack is None


@pytest.mark.parametrize("components", [1, 2])
@pytest.mark.parametrize("optimizer", ["strict", "smile_scoring"])
def test_probe_cache_preserves_changed_response_and_theta(monkeypatch, components, optimizer):
    rng = np.random.default_rng(73)
    n = 12
    kernels = []
    for _ in range(components):
        x = rng.normal(size=(n, 16))
        kernels.append(jnp.asarray(x @ x.T / 16, dtype=jnp.float32))
    operators = tuple(lambda v, k=k: k @ v for k in kernels)
    calls = {"probes": 0, "lanczos": 0}

    def stacked(v):
        if v.shape[1] == 7:
            calls["probes"] += 1
        return jnp.stack([op(v) for op in operators])

    original = REML._build_affine_slq_cache

    def build(*args, **kwargs):
        calls["lanczos"] += int(components == 1)
        return original(*args, **kwargs)

    monkeypatch.setattr(REML, "_build_affine_slq_cache", build)
    cache = REML.REMLProbeCache()
    base = dict(
        K_mvs=operators, diag_list=[jnp.diag(k) for k in kernels],
        stacked_kv=stacked, covar=jnp.ones((n, 1)), n_rand_vec=7,
        maxiter=100, minq_iter=0, slq_samples=4, slq_m=8,
        response_is_standardized=True, pcg_tol=1e-5, verbose=False,
        return_diagnostics=True,
        optimizer=optimizer,
    )
    trace_builds = int(optimizer == "smile_scoring")
    REML.fit_reml(y=rng.normal(size=n), probe_cache=cache, **base)
    assert calls == {"probes": trace_builds, "lanczos": int(components == 1)}
    response = rng.normal(size=n)
    theta = np.array([0.2] * components + [0.6])
    cached = REML.fit_reml(y=response, param_init=theta, probe_cache=cache, **base)
    assert calls == {"probes": trace_builds, "lanczos": int(components == 1)}
    fresh = REML.fit_reml(y=response, param_init=theta, **base)
    np.testing.assert_array_equal(cached[0], fresh[0])
    for key in ["loglik", "grad", "ai"]:
        np.testing.assert_allclose(cached[2][key], fresh[2][key], rtol=1e-6, atol=1e-6)
    before = calls.copy()
    REML.fit_reml(y=response, seed=91, probe_cache=cache, **base)
    assert calls["probes"] == before["probes"] + trace_builds
    assert calls["lanczos"] == before["lanczos"] + int(components == 1)
    before = calls.copy()
    # Same dimensions but different operators must never share products.
    previous_products = np.asarray(cache.kvrand_stack).copy() if trace_builds else None
    changed = {**base, "K_mvs": tuple(lambda v, k=k: 2 * k @ v for k in kernels)}
    changed["stacked_kv"] = None
    REML.fit_reml(y=response, seed=91, probe_cache=cache, **changed)
    if trace_builds:
        np.testing.assert_allclose(cache.kvrand_stack, 2 * previous_products, rtol=1e-6, atol=1e-6)
    else:
        assert cache.kvrand_stack is None
    assert cache.signature[0] == changed["K_mvs"]
    assert calls["lanczos"] == before["lanczos"] + int(components == 1)
