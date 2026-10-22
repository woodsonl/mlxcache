"""Compressed-key scoring tests — mlx-gated.

`scores_from_packed` must be numerically the SAME quantity as scoring
against the dequantized keys (that equality is its correctness proof), and
the QJL correction must move the projected-score error DOWN on average (the
paper's unbiasedness claim, checked empirically rather than trusted).
"""

from __future__ import annotations

import pytest

pytest.importorskip("mlx.core")

import mlx.core as mx  # noqa: E402
from mlxcache_sidecar.tq.codebook import lloyd_max_codebook  # noqa: E402
from mlxcache_sidecar.tq.codec import _unpack_indices, encode_keys  # noqa: E402
from mlxcache_sidecar.tq.rotation import build_rotation  # noqa: E402
from mlxcache_sidecar.tq.score import (  # noqa: E402
    qjl_correction,
    residual_signs_from_y,
    scores_from_packed,
)

D = 128


def _basis(bits=4):
    return build_rotation(D), lloyd_max_codebook(D, bits)


def test_scores_from_packed_equal_dequantized_scores():
    """The same quantity two ways: q @ dequantize(packed)ᵀ vs the packed
    gather-dot. Equal up to fp ordering (different op order)."""
    rotation, codebook = _basis()
    k = (mx.random.normal((1, 4, 40, D), key=mx.random.key(21)) / 17).astype(mx.float16)
    q = (mx.random.normal(shape=(D,), key=mx.random.key(22)) / 5).astype(mx.float16)
    packed, norms = encode_keys(k, codebook, rotation)

    packed_scores = scores_from_packed(q, packed, norms, D, codebook, rotation)

    levels, _b, _m = codebook
    idx = _unpack_indices(packed, 4, D)
    centroids = mx.array(levels, dtype=mx.float32)
    yhat = mx.take(centroids, idx.reshape(-1)).reshape(idx.shape)
    k_deq = (yhat / mx.sqrt(float(D))) @ rotation  # decode_keys' math
    k_deq = k_deq * norms.astype(mx.float32)[..., None]
    dense_scores = mx.einsum("d,bhsd->bhs", q.astype(mx.float32), k_deq)

    diff = mx.abs(packed_scores - dense_scores).max()
    assert float(diff) < 1e-2, f"packed scoring diverged from dense: {float(diff)}"


def test_qjl_correction_reduces_score_error_on_average():
    """Empirical unbiasedness: the projected scores are biased low (centroid
    shrinkage); adding the QJL residual estimate must reduce the mean score
    error against the TRUE (pre-quantization) inner products."""
    rotation, codebook = _basis(bits=3)
    k = (mx.random.normal((1, 2, 64, D), key=mx.random.key(23)) / 17).astype(mx.float16)
    q = (mx.random.normal(shape=(D,), key=mx.random.key(24)) / 5).astype(mx.float16)
    packed, norms = encode_keys(k, codebook, rotation)

    # True scores against the UNQUANTIZED keys.
    true_scores = mx.einsum("d,bhsd->bhs", q.astype(mx.float32), k.astype(mx.float32))

    projected = scores_from_packed(q, packed, norms, D, codebook, rotation)
    err_before = mx.abs(projected - true_scores).mean()

    # Residual signs from the encoder's own pre-quantization y.
    levels, boundaries, _m = codebook
    k32 = k.astype(mx.float32)
    norms32 = mx.sqrt(mx.sum(k32 * k32, axis=-1))
    u = k32 / mx.maximum(norms32, 1e-12)[..., None]
    y = (u @ rotation.T) * mx.sqrt(float(D))
    idx = mx.zeros_like(y).astype(mx.uint32)
    for t in boundaries:
        idx = idx + (y > t).astype(mx.uint32)
    signs = residual_signs_from_y(y, idx, codebook, D)
    corrected = projected + qjl_correction(q, signs, rotation, D, norms=norms)
    err_after = mx.abs(corrected - true_scores).mean()

    assert float(err_after) < float(err_before), (
        f"QJL correction did not reduce score error: {float(err_before):.4f} -> "
        f"{float(err_after):.4f}"
    )
