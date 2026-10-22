"""TurboQuantKVCache (live tier) tests — mlx-gated.

The invariant that holds: quantize-at-write is the same elementwise math the
persistence codec applies at capture, so for the same token stream the live
resident state EQUALS the persistence frame's tensors and `serialize_live`
emits the byte-identical frame `encode_cache` would (pinned below). What
does NOT hold — measured, not assumed — is token identity between live-tier
and persistence-tier GENERATION: quantizing during the forward changes the
computation recursively (every position attends through quantized history),
so the real gate pins each tier's determinism plus bounded cross-tier
divergence instead. Fidelity floors and append stability follow from the
codec's, but are proven here against the live class itself.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mlx.core")

import mlx.core as mx  # noqa: E402
from conftest import tq_basis  # noqa: E402
from conftest import tq_cos as _cos
from mlxcache_sidecar.tq import decode_payload, encode_cache  # noqa: E402
from mlxcache_sidecar.tq.live import TurboQuantKVCache, serialize_live  # noqa: E402

B, H, D = 1, 4, 128


def _kv(seed: int, tokens: int):
    k = (mx.random.normal((B, H, tokens, D), key=mx.random.key(seed)) / 17).astype(mx.float16)
    v = (mx.random.normal((B, H, tokens, D), key=mx.random.key(seed + 500)) / 3).astype(mx.float16)
    return k, v


def test_update_and_fetch_tracks_the_f16_cache():
    """Feeding the same rows through the live cache and an f16 KVCache: the
    returned history must track the f16 rows within the tier's fidelity
    floor, and PRIOR rows must not change when later tokens arrive."""
    from mlx_lm.models.cache import KVCache  # noqa: PLC0415

    live = TurboQuantKVCache(4, 4, 64, basis=tq_basis(D))
    plain = KVCache()
    k1, v1 = _kv(11, 30)
    k2, v2 = _kv(12, 20)
    lk, lv = live.update_and_fetch(k1, v1)
    pk, pv = plain.update_and_fetch(k1, v1)
    assert lk.dtype == mx.float16 and lk.shape == (B, H, 30, D)
    assert _cos(lk, pk[..., :30, :]) >= 0.985
    assert _cos(lv, pv[..., :30, :]) >= 0.985
    resident_before = live.keys[..., :30, :] * 1  # copy: arrays are semantically immutable

    lk2, lv2 = live.update_and_fetch(k2, v2)
    # The RESIDENT state is append-only: prior packed rows are byte-identical
    # after the append. The dequantized RETURN differs at most by last-ulp
    # matmul rounding — MLX batches the rotation matmul over S, and kernel
    # tiling can differ between an S=30 and an S=50 batch.
    assert bool((live.keys[..., :30, :] == resident_before).all())
    assert _cos(lk2[..., :30, :], lk) > 0.9999
    assert live.offset == 50 and live.size() == 50


def test_resident_state_is_compressed():
    """The point of the live tier: the RESIDENT state is ~3.6x smaller than
    the f16 equivalent at tq-k4v4-g64 (f16 = 2 tensors x D x 2B per head-
    token; compressed ~= 1,120/4,096). Measured at an exact capacity
    boundary (STEP=256): capacity is step-rounded like mlx-lm's KVCache, so
    a short context carries rounded-up padding — the amortized ratio is what
    long contexts actually see."""
    live = TurboQuantKVCache(4, 4, 64, basis=tq_basis(D))
    k, v = _kv(13, 256)
    live.update_and_fetch(k, v)
    f16_bytes = 2 * B * H * 256 * D * 2
    assert live.nbytes * 3 < f16_bytes, (
        f"resident state {live.nbytes} is not compressed vs f16 {f16_bytes}"
    )


def test_serialize_live_equals_encode_cache_bytes():
    """THE equality invariant: the live tier's serialized frame is
    byte-identical to the persistence codec encoding the same token stream —
    one tier label, one format, two runtimes."""
    from mlx_lm.models.cache import KVCache  # noqa: PLC0415

    k, v = _kv(14, 80)
    live = TurboQuantKVCache(4, 4, 64, basis=tq_basis(D))
    live.update_and_fetch(k, v)
    plain = KVCache()
    plain.update_and_fetch(k, v)
    assert serialize_live([live], 4, 4, 64) == encode_cache([plain], 4, 4, 64)


def test_adopt_state_resumes_and_extends():
    """Adopting a frame's compressed tensors directly (no dequantize-
    requantize): the resumed cache keeps the adopted rows byte-identical and
    appends new quantized rows past them."""
    basis = tq_basis(D)
    k, v = _kv(15, 40)
    src = TurboQuantKVCache(4, 4, 64, basis=basis)
    src.update_and_fetch(k, v)
    frame = serialize_live([src], 4, 4, 64)

    # adopt path: pull the raw tensors like the engine does

    from mlxcache_sidecar.tq.codec import decode_frame  # noqa: PLC0415

    arrays, meta = decode_frame(frame)
    adopted = TurboQuantKVCache(4, 4, 64, basis=basis)
    adopted.adopt_state(
        arrays["layers.0.k.idx"],
        arrays["layers.0.k.norm"],
        arrays["layers.0.v.q"],
        arrays["layers.0.v.s"],
        arrays["layers.0.v.b"],
        meta["offset"],
        meta["kv_dtype"],
    )
    assert adopted.offset == 40
    before = adopted.state[0]
    k2, v2 = _kv(16, 10)
    lk, lv = adopted.update_and_fetch(k2, v2)
    assert adopted.offset == 50
    after_packed = adopted.keys[..., :40, :]
    assert bool((after_packed == before).all()), "adopted rows changed on append"
    assert _cos(lk[..., :40, :], k) >= 0.985


def test_live_frame_decodes_through_the_standard_path():
    """A frame written by the live tier round-trips through the stock
    decode_payload (the persistence adopt path) — same format, verified
    from the other side."""
    k, v = _kv(17, 50)
    live = TurboQuantKVCache(4, 4, 64, basis=tq_basis(D))
    live.update_and_fetch(k, v)
    frame = serialize_live([live], 4, 4, 64)
    (out,) = decode_payload(frame)
    k0, v0 = out.state
    assert _cos(k0, k) >= 0.985 and _cos(v0, v) >= 0.985
    assert out.offset == 50


def test_trim_size_empty_mask():
    live = TurboQuantKVCache(4, 4, 64, basis=tq_basis(D))
    assert live.empty() and live.size() == 0
    k, v = _kv(18, 20)
    live.update_and_fetch(k, v)
    assert live.is_trimmable() and live.size() == 20
    assert live.trim(5) == 5 and live.size() == 15 and live.offset == 15
    assert live.trim(99) == 15 and live.empty() is False
    # make_mask delegates to mlx-lm's causal mask with our offset
    m = live.make_mask(4, return_array=True)
    assert m is not None


def test_adopted_live_cache_encodes_appends_with_the_frames_codebook():
    """Regression: frame_basis used to return EMPTY bucketing boundaries, so
    a cache adopted from a frame encoded every appended key coordinate as
    index 0 — silent corruption of delta prefill. An adopted cache must
    quantize appended rows identically to a from-scratch cache."""
    basis = tq_basis(D)
    k, v = _kv(41, 30)
    src = TurboQuantKVCache(4, 4, 64, basis=basis)
    src.update_and_fetch(k, v)
    frame = serialize_live([src], 4, 4, 64)

    from mlxcache_sidecar.tq.codec import decode_frame, frame_basis  # noqa: PLC0415

    arrays, meta = decode_frame(frame)
    adopted_basis = frame_basis(arrays, meta)
    adopted = TurboQuantKVCache(4, 4, 64, basis=adopted_basis)
    adopted.adopt_state(
        arrays["layers.0.k.idx"],
        arrays["layers.0.k.norm"],
        arrays["layers.0.v.q"],
        arrays["layers.0.v.s"],
        arrays["layers.0.v.b"],
        meta["offset"],
        meta["kv_dtype"],
    )
    k2, v2 = _kv(42, 10)
    adopted.update_and_fetch(k2, v2)

    scratch = TurboQuantKVCache(4, 4, 64, basis=tq_basis(D))
    scratch.update_and_fetch(k, v)
    scratch.update_and_fetch(k2, v2)
    assert bool((adopted.keys[..., 30:40, :] == scratch.keys[..., 30:40, :]).all()), (
        "adopted cache encoded appended rows differently than a fresh cache"
    )
    # The reconstructed boundaries match the solver's fixed point to f32
    # storage precision (the frame carries the levels as f32; the solver's
    # midpoints are float64 — identical bucketing, last-ulp differences).
    for got, want in zip(adopted_basis[1][1], tq_basis(D)[1][1], strict=True):
        assert abs(got - want) < 1e-5


def test_decode_rejects_rank3_packed_keys():
    """Regression: a rank-3 k.idx passed the [:3] shape checks and only
    exploded as IndexError on the first live append — a 500 loop, not a 422
    quarantine."""
    import io

    from mlx_lm.models.cache import KVCache  # noqa: PLC0415

    k0, v0 = _kv(43, 20)
    plain = KVCache()
    plain.update_and_fetch(k0, v0)
    payload = encode_cache([plain], 3, 4, 64)
    blob = io.BytesIO(payload)
    blob.name = "x.safetensors"
    arr, meta = mx.load(blob, return_metadata=True)
    arr["layers.0.k.idx"] = arr["layers.0.k.idx"][:, :, 0, :]  # rank 3
    buf = io.BytesIO()
    mx.save_safetensors(buf, arr, meta)
    with pytest.raises(ValueError, match="ranks are wrong"):
        decode_payload(buf.getvalue())
