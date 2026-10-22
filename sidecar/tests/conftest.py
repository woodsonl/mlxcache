"""Shared helpers for the TQ test suites.

No module-level mlx import: this conftest loads for EVERY test run, including
the documented mlx-free dev environment (`uv sync --group dev` provably
uninstalls the mlx extra), and an import-skip at conftest scope is a usage
error that kills the whole suite. The mlx-dependent helpers import lazily;
mlx-gated test modules skip themselves as before.
"""

from __future__ import annotations


def tq_cos(a, b) -> float:
    """Mean cosine over the last axis. NaN PROPAGATES on purpose: these
    helpers back fidelity floors, and a NaN row must fail the floor — the
    quality probe has its own finite-mean variant for reporting."""
    import mlx.core as mx  # noqa: PLC0415

    a32, b32 = a.astype(mx.float32), b.astype(mx.float32)
    per = (a32 * b32).sum(-1) / ((a32 * a32).sum(-1).sqrt() * (b32 * b32).sum(-1).sqrt())
    return float(per.mean())


def tq_basis(head_dim: int = 128, bits: int = 4):
    """The shared (rotation, codebook) pair every TQ test uses."""
    from mlxcache_sidecar.tq.codebook import lloyd_max_codebook  # noqa: PLC0415
    from mlxcache_sidecar.tq.rotation import build_rotation  # noqa: PLC0415

    return build_rotation(head_dim), lloyd_max_codebook(head_dim, bits)
