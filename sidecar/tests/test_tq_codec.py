"""TurboQuant persistence codec tests (mlx-gated: the frame is mlx tensors).

Covers the three properties the tier's gates rest on:

1. FIDELITY — worst-case (iid Gaussian keys) cosine bounds per the paper's
   Theorem 1: E[cos] = 1 − D_mse/2 per coordinate aggregate. Random data is
   the paper's own worst case, so real activations can only do better.
2. APPEND STABILITY — encoding a grown cache leaves every prior token's
   packed bytes identical (the chunk-identity property REPORT.md P1 demands
   of any future chunked store; element-wise quantizers guarantee it, the
   test proves the implementation didn't break it).
3. FRAME VALIDITY — the payload is ordinary safetensors, so the existing
   wire-level structural check passes and decode rebuilds a live KVCache.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mlx.core")

import mlx.core as mx  # noqa: E402
from mlxcache_sidecar.tq import (  # noqa: E402
    DEFAULT_GROUP_SIZE,
    FORMAT,
    decode_payload,
    encode_cache,
    parse_tier,
)
from mlxcache_sidecar.tq.codec import _pack_indices, _unpack_indices  # noqa: E402

B, H, D = 1, 4, 128


def _cache(layers: int, tokens: int, seed: int):
    """A dense KVCache stack filled with structured-but-generic tensors.

    Rows are sliced from ONE generated tensor per (layer, tensor-kind) so a
    shorter cache is exactly the prefix of a longer one — mx.random.normal
    is not itself prefix-stable across shapes, and the append-stability
    test needs controlled inputs to isolate codec behavior."""
    from mlx_lm.models.cache import KVCache  # noqa: PLC0415

    out = []
    for layer in range(layers):
        k_all = (mx.random.normal((B, H, tokens, D), key=mx.random.key(seed + layer)) / 17).astype(
            mx.float16
        )
        v_all = (
            mx.random.normal((B, H, tokens, D), key=mx.random.key(seed + 100 + layer)) / 3
        ).astype(mx.float16)
        c = KVCache()
        c.state = (k_all, v_all)
        out.append(c)
    return out


def _cache_pair(short_tokens: int, long_tokens: int, layers: int, seed: int):
    """(short, long) caches whose shared prefix rows are bit-identical."""
    from mlx_lm.models.cache import KVCache  # noqa: PLC0415

    short, long_ = [], []
    for layer in range(layers):
        k_all = (
            mx.random.normal((B, H, long_tokens, D), key=mx.random.key(seed + layer)) / 17
        ).astype(mx.float16)
        v_all = (
            mx.random.normal((B, H, long_tokens, D), key=mx.random.key(seed + 100 + layer)) / 3
        ).astype(mx.float16)
        for target, up_to in ((short, short_tokens), (long_, long_tokens)):
            c = KVCache()
            c.state = (k_all[:, :, :up_to], v_all[:, :, :up_to])
            target.append(c)
    return short, long_


def _cos(a, b) -> float:
    a32, b32 = a.astype(mx.float32), b.astype(mx.float32)
    per = (a32 * b32).sum(-1) / ((a32 * a32).sum(-1).sqrt() * (b32 * b32).sum(-1).sqrt())
    return float(per.mean())


def _worst(a, b) -> float:
    a32, b32 = a.astype(mx.float32), b.astype(mx.float32)
    per = (a32 * b32).sum(-1) / ((a32 * a32).sum(-1).sqrt() * (b32 * b32).sum(-1).sqrt())
    return float(per.min())


# --- tier strings ---------------------------------------------------------


@pytest.mark.parametrize(
    "tier,expected",
    [
        ("tq-k3v4-g64", (3, 4, 64)),
        ("tq-k3v2-g64", (3, 2, 64)),
        ("tq-k4v4-g32", (4, 4, 32)),
    ],
)
def test_tier_parse(tier, expected):
    assert parse_tier(tier) == expected


@pytest.mark.parametrize(
    "bad", ["", "tq", "q8", "tq-k3", "tq-k9v4-g64", "tq-k3v9-g64", "tq-k3v4-g1", "k3v4", "tq-k3v4"]
)
def test_tier_rejects_unknown(bad):
    with pytest.raises(ValueError, match="tier"):
        parse_tier(bad)


def test_default_group_is_pinned():
    # The canonical labels the README documents parse to the documented
    # parameters; 64 is the group size every shipped label carries.
    assert DEFAULT_GROUP_SIZE == 64
    assert parse_tier("tq-k4v4-g64") == (4, 4, 64)
    assert parse_tier("tq-k3v2-g64") == (3, 2, 64)


# --- bit packing ----------------------------------------------------------


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_pack_unpack_roundtrip(bits):
    n = 32 // bits
    idx = mx.floor(
        mx.random.uniform(shape=(2, 5, 130), key=mx.random.key(3), low=0, high=1 << bits)
    ).astype(mx.uint32)
    idx = mx.minimum(idx, mx.array((1 << bits) - 1, dtype=mx.uint32))
    packed = _pack_indices(idx, bits)
    assert packed.shape[-1] == -(-130 // n)
    assert _unpack_indices(packed, bits, 130).dtype == mx.uint32
    assert bool((_unpack_indices(packed, bits, 130) == idx).all())


# --- fidelity -------------------------------------------------------------


@pytest.mark.parametrize(
    "kb,vb,key_min,val_min",
    [
        # Worst-case iid keys: E[cos] = 1 - D/2 by Theorem 1; the min over
        # 200 tokens gets a further slack band. Values: group-quant anchors
        # from the paper's own quality table (0.94 @2b, 0.997 @4b).
        (3, 2, 0.978, 0.90),
        (3, 4, 0.978, 0.985),
        (4, 4, 0.990, 0.985),
    ],
)
def test_roundtrip_fidelity(kb, vb, key_min, val_min):
    layers = _cache(2, 200, seed=11)
    out = decode_payload(encode_cache(layers, kb, vb, 64))
    assert len(out) == 2
    for orig, reco in zip(layers, out, strict=True):
        k0, v0 = orig.state
        k1, v1 = reco.state
        assert k1.dtype == mx.float16 and k1.shape == k0.shape
        assert _cos(k0, k1) >= key_min
        assert _cos(v0, v1) >= val_min


def test_decoded_cache_is_live():
    """The decoded stack must be a working cache: offset set, and
    update_and_fetch appends without touching the adopted history."""
    from mlx_lm.models.cache import KVCache  # noqa: PLC0415

    layers = _cache(1, 50, seed=5)
    payload = encode_cache(layers, 3, 4, 64)
    (c,) = decode_payload(payload)
    assert type(c) is KVCache and c.offset == 50
    new_k = (mx.random.normal((B, H, 3, D), key=mx.random.key(9)) / 17).astype(mx.float16)
    new_v = (mx.random.normal((B, H, 3, D), key=mx.random.key(10)) / 3).astype(mx.float16)
    k_all, v_all = c.update_and_fetch(new_k, new_v)
    assert k_all.shape[2] == 53
    assert bool((k_all[..., :50, :] == c.keys[..., :50, :]).all())


def test_compression_ratio():
    """Asymptotic ratios (rotation amortized at 2000 tokens): k3v4 packs to
    512 B/token against f16's 2048 (4.0x exactly), k3v2 to 384 B/token
    (5.3x). The gate keeps margin for the fixed rotation/header bytes."""
    layers = _cache(2, 2000, seed=13)
    f16_bytes = 2 * B * H * 2000 * D * 2 * 2  # K+V, f16
    assert f16_bytes / len(encode_cache(layers, 3, 4, 64)) > 3.5
    assert f16_bytes / len(encode_cache(layers, 3, 2, 64)) > 4.5


# --- append stability -----------------------------------------------------


def test_append_stability():
    """Prior tokens' bytes never change when the cache grows: the packed
    indices, norms, and group-quantized values for the first S rows are
    identical whether encoded alone or as a prefix of a longer cache."""
    import io

    short, long_ = _cache_pair(120, 200, layers=2, seed=21)
    k_s, _ = short[0].state
    k_l, _ = long_[0].state
    assert bool((k_s == k_l[..., :120, :]).all())  # controlled inputs

    def tensors_of(payload):
        blob = io.BytesIO(payload)
        blob.name = "x.safetensors"
        arr, _ = mx.load(blob, return_metadata=True)
        return arr

    a = tensors_of(encode_cache(short, 3, 4, 64))
    b = tensors_of(encode_cache(long_, 3, 4, 64))
    for name in a:
        if name.startswith(("rot.", "cb.")):
            continue  # codec constants, not per-token tensors
        # Every per-token tensor carries S on axis 2 (the packed key indices,
        # the norms, and the group-quantized values alike).
        assert bool((a[name][:, :, :120] == b[name][:, :, :120]).all()), f"{name} not append-stable"


# --- frame validity and error paths ---------------------------------------


def test_frame_is_valid_safetensors():
    from mlxcache_sidecar.server import _valid_safetensors  # noqa: PLC0415

    payload = encode_cache(_cache(2, 40, seed=31), 3, 4, 64)
    assert _valid_safetensors(payload)


def test_decode_rejects_non_tq_frame():
    import io

    from mlx_lm.models.cache import save_prompt_cache  # noqa: PLC0415

    buf = io.BytesIO()
    save_prompt_cache(buf, _cache(1, 20, seed=41))
    with pytest.raises(ValueError, match="TurboQuant"):
        decode_payload(buf.getvalue())


def test_decode_rejects_missing_rotation():
    import io

    payload = encode_cache(_cache(1, 20, seed=51), 3, 4, 64)
    blob = io.BytesIO(payload)
    blob.name = "x.safetensors"
    arr, meta = mx.load(blob, return_metadata=True)
    del arr["rot.128"]
    buf = io.BytesIO()
    mx.save_safetensors(buf, arr, meta)
    with pytest.raises(ValueError, match="rotation"):
        decode_payload(buf.getvalue())


def test_decode_uses_the_frame_codebook_not_a_recomputed_one():
    """The blob is self-describing: mutating the frame's embedded levels must
    change the decoded keys. If decode recomputed the codebook this test
    would fail silently on any future solver drift."""
    import io

    layers = _cache(1, 20, seed=52)
    k0, _ = layers[0].state
    payload = encode_cache(layers, 3, 4, 64)
    blob = io.BytesIO(payload)
    blob.name = "x.safetensors"
    arr, meta = mx.load(blob, return_metadata=True)
    arr["cb.128.3"] = arr["cb.128.3"] + 0.25  # shift every reconstruction level
    buf = io.BytesIO()
    mx.save_safetensors(buf, arr, meta)
    (out,) = decode_payload(buf.getvalue())
    k1, _ = out.state
    assert _cos(k0, k1) < 0.999, "decode ignored the frame's codebook"


def test_decode_rejects_a_non_contiguous_layer_set():
    """A planted frame that drops a layer's tensors must 422 (quarantine),
    not build a short stack that IndexErrors inside model forward."""
    import io

    payload = encode_cache(_cache(3, 20, seed=53), 3, 4, 64)
    blob = io.BytesIO(payload)
    blob.name = "x.safetensors"
    arr, meta = mx.load(blob, return_metadata=True)
    for name in [n for n in arr if n.startswith("layers.1.")]:
        del arr[name]
    buf = io.BytesIO()
    mx.save_safetensors(buf, arr, meta)
    with pytest.raises(ValueError, match="contiguous"):
        decode_payload(buf.getvalue())


def test_encode_rejects_non_dense_stack():
    class FakeHybrid:
        keys = None

    with pytest.raises(ValueError, match="dense"):
        encode_cache([FakeHybrid()], 3, 4, 64)


def test_encode_rejects_mixed_head_dims(monkeypatch):
    """Uniform head_dim is a documented tier constraint: refuse loudly at
    capture, not silently at decode."""
    layers = _cache(1, 20, seed=61)
    k, v = layers[0].state
    other = _cache(1, 20, seed=62)[0]
    k2 = mx.pad(other.keys, [(0, 0), (0, 0), (0, 0), (0, 16)])  # D=144
    v2 = mx.pad(other.values, [(0, 0), (0, 0), (0, 0), (0, 16)])
    from mlx_lm.models.cache import KVCache  # noqa: PLC0415

    second = KVCache()
    second.state = (k2, v2)
    with pytest.raises(ValueError, match="uniform"):
        encode_cache([layers[0], second], 3, 4, 64)


def test_encode_rejects_indivisible_group():
    layers = _cache(1, 20, seed=71)
    with pytest.raises(ValueError, match="group"):
        encode_cache(layers, 3, 4, 96)  # 128 % 96 != 0


def test_zero_norm_rows_are_safe():
    """An all-zero key row reconstructs as ~0 rather than NaN."""
    layers = _cache(1, 30, seed=81)
    c = layers[0]
    k, v = c.state
    k = mx.pad(k, [(0, 0), (0, 0), (0, 0), (0, 0)], constant_values=0)
    k = k * mx.array([1, 1, 0, 1]).astype(mx.float16)[None, :, None, None]  # head 2 zeroed
    c.state = (k, v)
    (out,) = decode_payload(encode_cache(layers, 3, 4, 64))
    k1, _ = out.state
    assert bool(mx.isfinite(k1).all())
    assert float(mx.abs(k1[:, 2]).max()) == pytest.approx(0.0, abs=1e-6)


def test_key_norms_survive_chained_decode_encode_cycles():
    """Regression: the stored norm used to be ||k|| verbatim, but the
    dequantized direction has norm s < 1, so every decode->encode cycle
    (delta prefill adopting an ancestor, settle of a resumed cache) shrank
    the oldest keys by another factor of s — ~1.5% per turn at k3, invisible
    to cosine (scale-invariant). The compensated norm must keep per-row
    magnitudes stable across five chained cycles."""
    import mlx.core as mx

    layers = _cache(1, 40, seed=91)
    k0, _ = layers[0].state
    orig_norms = mx.sqrt(mx.sum(k0.astype(mx.float32) ** 2, axis=-1))
    cache = layers
    for _cycle in range(5):
        cache = decode_payload(encode_cache(cache, 3, 4, 64))
    k5, _ = cache[0].state
    final_norms = mx.sqrt(mx.sum(k5.astype(mx.float32) ** 2, axis=-1))
    ratio = final_norms / orig_norms
    assert float(ratio.min()) > 0.999 and float(ratio.max()) < 1.001, (
        f"norms drifted after 5 cycles: min={float(ratio.min()):.5f} max={float(ratio.max()):.5f}"
    )


def test_decode_rejects_out_of_range_metadata():
    """key_bits=0 must die as ValueError (422/quarantine), not as an
    uncaught ZeroDivisionError that 500s forever while the poison stays
    selectable."""
    import io

    payload = encode_cache(_cache(1, 20, seed=92), 3, 4, 64)
    blob = io.BytesIO(payload)
    blob.name = "x.safetensors"
    arr, meta = mx.load(blob, return_metadata=True)
    meta = dict(meta)
    meta["key_bits"] = "0"
    arr = {"cb.128.0": arr.pop("cb.128.3"), **dict(arr)}
    buf = io.BytesIO()
    mx.save_safetensors(buf, arr, meta)
    with pytest.raises(ValueError, match="key_bits"):
        decode_payload(buf.getvalue())


def test_encode_rejects_mismatched_key_value_head_dims():
    """An architecture with kdim != vdim would encode (frame names derive
    from the key dim) and then fail to decode (d re-derived from the value
    scales): refuse at encode."""
    from mlx_lm.models.cache import KVCache  # noqa: PLC0415

    layers = _cache(1, 20, seed=93)
    other = _cache(1, 20, seed=94)[0]
    k2, v2 = other.state
    v2 = v2[..., :64]  # value head_dim 64 != key head_dim 128
    second = KVCache()
    second.state = (k2, v2)
    with pytest.raises(ValueError, match="head_dim"):
        encode_cache([layers[0], second], 3, 4, 64)


def test_decode_rejects_non_float_kv_dtype():
    """A frame naming a non-floating dtype would decode into garbage KV
    rather than erroring; the boundary rejects it before use."""
    import io

    payload = encode_cache(_cache(1, 20, seed=95), 3, 4, 64)
    blob = io.BytesIO(payload)
    blob.name = "x.safetensors"
    arr, meta = mx.load(blob, return_metadata=True)
    meta = dict(meta)
    meta["kv_dtype"] = "uint8"
    buf = io.BytesIO()
    mx.save_safetensors(buf, arr, meta)
    with pytest.raises(ValueError, match="floating dtype"):
        decode_payload(buf.getvalue())


def test_encode_rejects_mixed_key_value_dtypes():
    """A single kv_dtype field in the frame means a mixed f16-key/f32-value
    cache would silently downcast the values at decode; refuse it."""
    from mlx_lm.models.cache import KVCache  # noqa: PLC0415

    layers = _cache(1, 20, seed=96)
    other = _cache(1, 20, seed=97)[0]
    k2, _ = other.state
    v2 = other.state[1].astype(mx.float32)
    second = KVCache()
    second.state = (k2.astype(mx.float16), v2)
    with pytest.raises(ValueError, match="uniform"):
        encode_cache([layers[0], second], 3, 4, 64)


def test_decode_rejects_key_value_shape_disagreement():
    """A planted frame whose value tensors cover a different sequence length
    than its keys/offset would pass KVCache.from_state and explode inside
    update_and_fetch on the first resume — past the 422 boundary. (Batch and
    head mismatches already die during decode as mx broadcast ValueErrors,
    which the adapter maps to 422; the value-side sequence mismatch is the
    case that slips to from_state.) Reject at decode instead."""
    import io

    payload = encode_cache(_cache(1, 20, seed=98), 3, 4, 64)
    blob = io.BytesIO(payload)
    blob.name = "x.safetensors"
    arr, meta = mx.load(blob, return_metadata=True)
    for name in ("layers.0.v.q", "layers.0.v.s", "layers.0.v.b"):
        arr[name] = arr[name][:, :, :18, ...]  # values claim 18 positions, offset 20
    buf = io.BytesIO()
    mx.save_safetensors(buf, arr, meta)
    with pytest.raises(ValueError, match="cover the claimed offset|shapes disagree"):
        decode_payload(buf.getvalue())


def test_decode_rejects_an_empty_layer_set():
    """An empty layer set passed the contiguity check trivially (range(0))
    and handed mlx-lm a zero-layer cache, which zip() accepts by skipping
    every layer — invalid output, no error, no quarantine."""
    import io

    payload = encode_cache(_cache(2, 20, seed=99), 3, 4, 64)
    blob = io.BytesIO(payload)
    blob.name = "x.safetensors"
    arr, meta = mx.load(blob, return_metadata=True)
    arr = {k: v for k, v in arr.items() if not k.startswith("layers.")}
    buf = io.BytesIO()
    mx.save_safetensors(buf, arr, meta)
    with pytest.raises(ValueError, match="empty or not contiguous"):
        decode_payload(buf.getvalue())


def test_load_cache_rejects_a_short_layer_count():
    """The adapter's own trust boundary: a checkpoint whose cache entry
    count disagrees with the model's layer count 422s instead of letting
    mlx-lm's zip() silently skip layers. Uses the TQ branch via the real
    engine path (MlxLmEngine._load_cache_from_payload)."""
    pytest.importorskip("mlx.core")
    from mlxcache_sidecar.server import CheckpointRejectedError, MlxLmEngine

    engine = object.__new__(MlxLmEngine)
    engine.kv_tier = "tq-k4v4-g64"
    engine._cache_layer_count = 3
    layers = _cache(2, 20, seed=100)
    payload = encode_cache(layers, 3, 4, 64)
    with pytest.raises(CheckpointRejectedError, match="cache entries"):
        engine._load_cache_from_payload(payload)


def test_rotation_determinism_across_builds():
    """Same seed -> identical matrix across separate computations (the
    lru_cache hides this from every other test); different seed differs;
    both orthogonal."""
    from mlxcache_sidecar.tq.rotation import PINNED_SEED, build_rotation  # noqa: PLC0415

    build_rotation.cache_clear()
    a = build_rotation(64)
    build_rotation.cache_clear()
    b = build_rotation(64)
    assert bool((a == b).all()), "same seed rebuilt a different rotation"
    build_rotation.cache_clear()
    c = build_rotation(64, seed=PINNED_SEED + 1)
    assert not bool((a == c).all()), "different seeds built the same rotation"
    for m in (a, c):
        err = float(mx.abs(m.T @ m - mx.eye(64, dtype=mx.float32)).max())
        assert err < 1e-4


def test_format_constant():
    assert FORMAT == "tq1"
