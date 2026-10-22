"""Pinned-seed random orthogonal rotation for the TurboQuant key path.

The paper (arXiv:2504.19874 §3.1) randomizes each vector with a random
rotation built by QR decomposition of an iid Gaussian matrix. This module
builds ONE rotation per head_dim from a PINNED seed so that every blob this
process writes uses the same canonical basis (a per-boot random basis would
make two blobs of the same prefix mutually non-adoptable at the byte level
and would break any future chunk-level dedup between them).

The matrix is embedded in every encoded frame, so decode never depends on
regenerating it — a future MLX whose RNG or QR differs writes blobs that
carry their own rotation and stay self-describing. Only freshly-written
blobs would drift from older ones byte-wise, never semantically.
"""

from __future__ import annotations

import functools

# One basis for every process, pinned. Any change to this constant re-keys
# nothing (the rotation travels in the frame), but new blobs stop
# byte-matching older ones, so treat a change like a format bump.
PINNED_SEED = 20260521


@functools.lru_cache(maxsize=8)
def build_rotation(head_dim: int, seed: int = PINNED_SEED):
    """A deterministic orthogonal f32 matrix of shape [head_dim, head_dim].

    Rows of R are the rotated basis: y = R·x is computed on tensors as
    ``x @ R.T`` and inverted as ``y @ R``. QR runs on a CPU stream — MLX's
    GPU path does not implement qr (mlx 0.32) — and the result is usable
    from regular GPU ops afterwards.
    """
    import mlx.core as mx  # noqa: PLC0415 — mlx-dependent by design

    g = mx.random.normal((head_dim, head_dim), key=mx.random.key(seed))
    with mx.stream(mx.cpu):
        q, _ = mx.linalg.qr(g)
    q = q.astype(mx.float32)
    err = float(mx.abs(q.T @ q - mx.eye(head_dim, dtype=mx.float32)).max())
    if err > 1e-4:
        # f32 QR orthogonality is ~1e-6; 1e-4 means the decomposition
        # silently degraded. Refuse rather than write a lossy basis.
        raise ValueError(f"rotation for head_dim={head_dim} not orthogonal (err={err})")
    return q
