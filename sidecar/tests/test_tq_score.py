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
from conftest import tq_basis  # noqa: E402
from mlxcache_sidecar.tq.codec import _unpack_indices, encode_keys  # noqa: E402
from mlxcache_sidecar.tq.score import (  # noqa: E402
    qjl_correction,
    residual_signs_from_y,
    scores_from_packed,
)

D = 128


def test_scores_from_packed_equal_dequantized_scores():
    """The same quantity two ways: q @ dequantize(packed)ᵀ vs the packed
    gather-dot. Equal up to fp ordering (different op order)."""
    rotation, codebook = tq_basis(D)
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
    """The QJL paper's claim is UNBIASEDNESS, and that is what is asserted:
    the projected scores carry a systematic signed bias (centroid
    shrinkage); adding the QJL residual estimate drives the MEAN signed
    error toward zero. Per-token ABSOLUTE error is a different quantity —
    the estimator's own variance (≈ π/(2d)·‖Rq‖²‖r‖², ~0.5 at d=128) can
    exceed the bias it removes for a single query, which is exactly why
    the module records this trade instead of promising error reduction."""
    rotation, codebook = tq_basis(D, bits=3)
    k = (mx.random.normal((1, 2, 64, D), key=mx.random.key(23)) / 17).astype(mx.float16)
    q = (mx.random.normal(shape=(D,), key=mx.random.key(24)) / 5).astype(mx.float16)
    packed, norms = encode_keys(k, codebook, rotation)

    # True scores against the UNQUANTIZED keys.
    true_scores = mx.einsum("d,bhsd->bhs", q.astype(mx.float32), k.astype(mx.float32))

    projected = scores_from_packed(q, packed, norms, D, codebook, rotation)
    signed_bias_before = float((true_scores - projected).mean())

    # Residual signs and norms from the encoder's own pre-quantization y.
    levels, boundaries, _m = codebook
    k32 = k.astype(mx.float32)
    norms32 = mx.sqrt(mx.sum(k32 * k32, axis=-1))
    u = k32 / mx.maximum(norms32, 1e-12)[..., None]
    y = (u @ rotation.T) * mx.sqrt(float(D))
    idx = mx.zeros_like(y).astype(mx.uint32)
    for t in boundaries:
        idx = idx + (y > t).astype(mx.uint32)
    signs, r_norms = residual_signs_from_y(y, idx, codebook, D)
    corrected = projected + qjl_correction(q, signs, r_norms, rotation, D, norms=norms)
    signed_bias_after = float((true_scores - corrected).mean())

    assert abs(signed_bias_after) < abs(signed_bias_before) / 2, (
        f"QJL did not remove the signed bias: {signed_bias_before:.4f} -> {signed_bias_after:.4f}"
    )
