"""Runtime environment bootstrap for CLI entrypoints.

This must run before importing JAX so allocator-related environment
variables take effect.
"""

from __future__ import annotations

import os


def resolve_cpu_threads(explicit: int | None = None) -> tuple[int, str]:
    """Resolve one CPU budget without importing numerical libraries."""
    if explicit is not None:
        return max(1, int(explicit)), "explicit"
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        raw = os.environ.get(name, "").strip()
        if not raw:
            continue
        try:
            value = int(raw)
        except ValueError:
            continue
        if value > 0:
            return value, name
    try:
        return max(1, len(os.sched_getaffinity(0))), "sched_getaffinity"
    except (AttributeError, OSError):
        return max(1, int(os.cpu_count() or 1)), "os.cpu_count"


def configure_runtime_env() -> None:
    # Use the non-preallocating JAX client by default so CLI runs track the
    # project GPU budget more closely. Do not force the platform allocator: it
    # serializes allocation/free work and is substantially slower for the
    # repeated streamed kernels used here. Users can still opt into it through
    # XLA_PYTHON_CLIENT_ALLOCATOR for allocation diagnostics.
    # Limit backend discovery to CUDA/CPU by default. On non-TPU machines JAX
    # otherwise tries to initialize the TPU backend and emits a noisy libtpu
    # warning during every CLI run.
    os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


__all__ = ["configure_runtime_env", "resolve_cpu_threads"]
