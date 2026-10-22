"""Scoring attention from COMPRESSED keys — the math and its measured cost.

The compressed keys are k_s = norm_s · Rᵀ ŷ_s with ŷ_s the per-dimension
centroid values (the Lloyd-Max codebook is a SCALAR quantizer: one level per
coordinate). So for a decode-step query q,

    <q, k_s> = norm_s · Σ_i (R q)_i · c[idx_{s,i}],

computable straight from the packed state: rotate the query once (d² flops),
then a gather-dot over per-token, per-dimension centroid values. This module
implements that exactly (`scores_from_packed`) plus the paper's QJL residual
correction (`qjl_correction`), which restores the unbiased inner product the
MSE quantizer biases (QJL, Def 1: 1 stored sign bit per dimension of the
rotated residual).

The honest compute accounting, because it decides the wiring question:

- The K-side bandwidth of the packed state is ~4x smaller than dense f16
  (3-4 bits per element vs 16), so a FUSED gather-dot kernel is
  bandwidth-bound where dense score matmuls are — the theoretical win.
- As plain MLX tensor ops there is NO win: `_unpack_indices` and the gather
  materialize a [S, d] intermediate (4 bytes/element — bigger than the f16
  it replaces), and MLX has no fused unpack-gather-reduce. The probe
  measures the tensor-op path anyway; the fusion boundary is real.
- The QJL correction is O(S·d) on [S, d] sign bits — same complexity as
  dense scoring; it buys UNBIASEDNESS, not per-token accuracy. Measured at
  d=128/k3: the projected scores' systematic bias is removed (mean signed
  error toward zero, pinned by test), but the estimator's own variance
  (≈ π/(2d)·‖Rq‖²‖‖r‖², ~0.5 here) EXCEEDS the bias it removes for
  single-query attention scoring — per-token absolute error gets WORSE.
  That is the honest answer to "should the decode path carry QJL bits":
  no, unless a future estimator amortizes the variance across queries.

So this module is the validated math plus the fidelity record; wiring it
into the decode hot loop means either a custom Metal kernel (mx.fast) or a
cache-aware attention op in mlx-lm itself. That decision stays open with
the numbers attached.
"""

from __future__ import annotations

import functools

SQRT_PI_OVER_2 = 1.2533141373155003  # √(π/2), the QJL dequantization scale


@functools.lru_cache(maxsize=8)
def _qjl_matrix(head_dim: int, seed: int = 20261101):
    """S ∈ R^{d×d} with iid N(0,1) entries, pinned seed (QJL, Def 1).

    Pinned like the rotation: every process agrees on one canonical sketch.
    Stored implicitly — d² floats would dwarf the d-bit payload the sketch
    protects; frames carrying QJL bits are self-describing relative to THIS
    matrix only, so a seed change is a format change."""
    import mlx.core as mx  # noqa: PLC0415

    return mx.random.normal((head_dim, head_dim), key=mx.random.key(seed)).astype(mx.float32)


def scores_from_packed(query, packed, norms, d: int, codebook, rotation):
    """<q, k_s> for every cached s, from the compressed state.

    `query` is a single per-head decode query [d]; `packed` u32 [B,H,S,W]
    and `norms` [B,H,S] are the resident tier's state. Returns [B,H,S].
    Numerically equal (up to fp ordering) to `q @ dequantize(packed)ᵀ` —
    the probe asserts that, which is the module's correctness proof."""
    import mlx.core as mx  # noqa: PLC0415

    from mlxcache_sidecar.tq.codec import _unpack_indices  # noqa: PLC0415

    levels, _boundaries, _mse = codebook
    bits = (len(levels) - 1).bit_length()
    rq = rotation @ query.astype(mx.float32)  # [d]
    idx = _unpack_indices(packed, bits, d)  # [B,H,S,d] u32 (materialized)
    centroids = mx.array(levels, dtype=mx.float32)
    yhat = mx.take(centroids, idx.reshape(-1)).reshape(idx.shape)
    # The /sqrt(d) is the u-units -> x-units conversion every decode path
    # applies (see decode_keys); missing it scales scores by sqrt(d).
    return (yhat * rq).sum(-1) * norms.astype(mx.float32) / mx.sqrt(float(d))


def qjl_correction(query, residual_sign, residual_norms, rotation, head_dim: int, norms=None):
    """The unbiasedness term: E[<Rq, r>] estimated from stored sign bits.

    `residual_sign` int8 [B,H,S,d] holds sign(S·r) of the rotated residual
    r = y − ĉ and `residual_norms` [B,H,S] its per-token norms, both
    computed at write time by `residual_signs_from_y` — sign bits are
    scale-invariant, so the estimator must carry ‖r‖ explicitly or it
    under-corrects by exactly that factor. Returns [B,H,S]:
    √(π/2)·‖r‖/d · <S R q, z_s> per token, scaled by key norms/√d when
    `norms` [B,H,S] is given (the same unit conversion
    `scores_from_packed` applies) — add the result to the projected scores
    for the unbiased inner-product estimate."""
    import mlx.core as mx  # noqa: PLC0415

    rq = rotation @ query.astype(mx.float32)  # [d]
    sketched = _qjl_matrix(head_dim) @ rq  # [d]
    correction = (
        (residual_sign.astype(mx.float32) @ sketched)
        * (SQRT_PI_OVER_2 / head_dim)
        * residual_norms.astype(mx.float32)
    )
    if norms is not None:
        correction = correction * norms.astype(mx.float32) / mx.sqrt(float(head_dim))
    return correction


def residual_signs_from_y(y, idx, codebook, head_dim: int):
    """sign(S·r̃) AND ‖r̃‖ for tokens still in the encoder's hand, where the
    coherent residual is r̃ = s·y − ŷ: the compensated-norm score computes
    (n/s)·<Rq, ŷ>/√d against the true n·<Rq, y>/√d, so the missing term is
    exactly (n/s)·<Rq, s·y − ŷ>/√d with s = ‖ŷ‖/√d per row. `y` is the
    pre-quantization rotated unit vector in u-units [.., d], `idx` the
    just-computed centroid indices [.., d]. Returns (int8 ±1, f32 norms).

    Storage: d bits per token per head (16 B at d=128 — the frame grows by
    roughly a third at k3) plus the per-token norm. Nothing persists these
    by default; the probe computes them to measure what the correction buys
    before anyone pays that cost."""
    import mlx.core as mx  # noqa: PLC0415

    levels, _boundaries, _mse = codebook
    centroids = mx.array(levels, dtype=mx.float32)
    yhat = mx.take(centroids, idx.reshape(-1)).reshape(idx.shape)
    s_row = mx.sqrt(mx.sum(yhat * yhat, axis=-1)) / mx.sqrt(float(head_dim))
    residual = s_row[..., None] * y.astype(mx.float32) - yhat
    r_norms = mx.sqrt(mx.sum(residual * residual, axis=-1))
    sketched = residual @ _qjl_matrix(head_dim).T  # S·r̃ per row
    return mx.where(sketched >= 0, 1, -1).astype(mx.int8), r_norms
