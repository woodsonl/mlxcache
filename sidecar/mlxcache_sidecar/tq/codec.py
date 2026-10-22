"""The TurboQuant persistence codec: encode a dense mlx-lm prompt cache to a
compressed safetensors frame and decode it back to a live cache.

Scope (deliberate): this is the PERSISTENCE tier of arXiv:2504.19874, not the
live-compute tier. Encode runs at capture time (the cache is already resident
in f16), decode runs at adopt time and rebuilds a stock ``KVCache`` — the
engine never computes attention from compressed state. What the algorithm
contributes here is the byte count of the persisted blob:

- KEYS: per-(token,head) L2 norm in f16 + a random-orthogonal rotation
  (paper §3.1) + a b-bit Lloyd-Max codebook on the rotated unit-vector
  coordinates (see ``codebook``), bit-packed ⌊32/b⌋ indices per u32 word.
  Element-wise by construction: appended tokens never rewrite prior bytes,
  which is what a chunked store needs for block identity.
- VALUES: stock ``mx.quantize`` group quantization (per-group scale/bias),
  the same mechanism ``QuantizedKVCache`` uses — proven append-stable by the
  dedup forensics (REPORT.md E2/P1).
- QJL residual sign bits are NOT included: they exist to score attention
  directly from compressed keys (the paper's inner-product path). Decode
  always dequantizes, so the estimator machinery has nothing to add here.

The rotation matrix and codebook parameters travel INSIDE the frame, so a
decoded cache depends on nothing but the frame itself. The frame is ordinary
safetensors: the existing wire-level structural validation
(``_valid_safetensors``) and sha256 integrity stamping apply unchanged.

Lossy by design: a resumed generation approximates a scratch run (bounded by
the codebook/group-quant distortion; the paper reports quality-neutral KV at
3.5 bits/channel). The exact-resume contract does not hold on this tier —
that is why the tier is pinned in the checkpoint fingerprint and can never
cross-serve the f16/q8 tiers.
"""

from __future__ import annotations

import io

FORMAT = "tq1"

# bits accepted by the tier strings. Key bits are bounded by the codebook
# solver; value bits by mx.quantize's supported widths.
KEY_BITS_CHOICES = (2, 3, 4)
VALUE_BITS_CHOICES = (2, 3, 4, 6, 8)
DEFAULT_GROUP_SIZE = 64


def parse_tier(tier: str) -> tuple[int, int, int]:
    """Parse a tier label like ``tq-k3v4-g64`` into (key_bits, value_bits,
    group_size). Raises ValueError on anything unknown — a typo'd env var
    must refuse to boot, not silently fall back to f16. The label grammar is
    pinned by the golden tests' literals; there is no formatter because
    nothing in production composes labels, operators spell them."""
    parts = tier.split("-")
    if len(parts) != 3 or parts[0] != "tq":
        raise ValueError(f"unknown kv tier {tier!r} (expected e.g. 'tq-k3v4-g64')")
    kv = parts[1]
    if not kv.startswith("k") or "v" not in kv[1:]:
        raise ValueError(f"unknown kv tier {tier!r} (expected e.g. 'tq-k3v4-g64')")
    try:
        key_bits = int(kv[1 : kv.index("v")])
        value_bits = int(kv[kv.index("v") + 1 :])
        group_size = int(parts[2][1:]) if parts[2].startswith("g") else int(parts[2])
    except ValueError as exc:
        raise ValueError(f"unknown kv tier {tier!r}: {exc}") from exc
    if key_bits not in KEY_BITS_CHOICES:
        raise ValueError(f"key bits {key_bits} not in {KEY_BITS_CHOICES} (tier {tier!r})")
    if value_bits not in VALUE_BITS_CHOICES:
        raise ValueError(f"value bits {value_bits} not in {VALUE_BITS_CHOICES} (tier {tier!r})")
    if group_size < 2:
        raise ValueError(f"group size {group_size} must be >= 2 (tier {tier!r})")
    return key_bits, value_bits, group_size


def _pack_indices(idx, bits: int):
    """Pack [..., D] small uint32 indices into [..., W] u32 words, n=⌊32/b⌋
    per word, zero-padded past D. For b=3 the last 2 bits of every word are
    unused (10 indices × 3 bits = 30); b=2 and b=4 pack exactly. A
    straddle-free layout keeps pack/unpack as pure shift/mask tensor ops."""
    import mlx.core as mx  # noqa: PLC0415

    n = 32 // bits
    d = idx.shape[-1]
    words = -(-d // n)
    pad = words * n - d
    if pad:
        idx = mx.pad(idx, [(0, 0)] * (idx.ndim - 1) + [(0, pad)])
    idx = idx.reshape(*idx.shape[:-1], words, n)
    shifts = (bits * mx.arange(n)).astype(mx.uint32)
    return mx.sum(idx << shifts, axis=-1).astype(mx.uint32)


def _unpack_indices(words, bits: int, d: int):
    """Inverse of ``_pack_indices``; drops the zero padding past ``d``."""
    import mlx.core as mx  # noqa: PLC0415

    n = 32 // bits
    count = words.shape[-1]
    shifts = (bits * mx.arange(n)).astype(mx.uint32)
    mask = mx.array((1 << bits) - 1, dtype=mx.uint32)
    idx = (words[..., None] >> shifts) & mask
    return idx.reshape(*idx.shape[:-2], count * n)[..., :d]


def encode_keys(keys, codebook, rotation):
    """(packed_idx u32 [B,H,S,W], norms f32 [B,H,S]) for one layer's keys.

    `codebook` is (levels, boundaries) from ``codebook.lloyd_max_codebook``,
    in u-units (u = √d · coordinate of the rotated unit vector).
    """
    import mlx.core as mx  # noqa: PLC0415

    levels, boundaries, _mse = codebook
    bits = (len(levels) - 1).bit_length()
    k32 = keys.astype(mx.float32)
    norms = mx.sqrt(mx.sum(k32 * k32, axis=-1))
    # A zero key row (if one ever occurs) divides cleanly to u=0 and
    # reconstructs as ~0: clamping the divisor is exact for every other row.
    u = k32 / mx.maximum(norms, 1e-12)[..., None]
    y = u @ rotation.T
    y = y * mx.sqrt(float(keys.shape[-1]))  # x-units -> u-units
    b = mx.array(boundaries, dtype=mx.float32)
    idx = (y[..., None] > b).sum(-1).astype(mx.uint32)
    # Norms in f32, not f16: a key whose elements sit near the f16 ceiling
    # has ||k|| up to ~65504*sqrt(d), which overflows f16 storage and would
    # reconstruct as inf. The 2 extra bytes per (head, token) are ~1% of the
    # frame.
    return _pack_indices(idx, bits), norms.astype(mx.float32)


def decode_keys(packed, norms, d: int, codebook, rotation, dtype):
    """Inverse of ``encode_keys``: [B,H,S,d] keys in `dtype`."""
    import mlx.core as mx  # noqa: PLC0415

    levels, _boundaries, _mse = codebook
    bits = (len(levels) - 1).bit_length()
    idx = _unpack_indices(packed, bits, d)
    flat = mx.take(mx.array(levels, dtype=mx.float32), idx.reshape(-1))
    y = flat.reshape(idx.shape)
    u = (y / mx.sqrt(float(d))) @ rotation
    return (u * norms.astype(mx.float32)[..., None]).astype(dtype)


def encode_cache(cache, key_bits: int, value_bits: int, group_size: int) -> bytes:
    """Serialize a dense KVCache list to the compressed safetensors frame.

    Raises ValueError for anything but a uniform dense stack (the tier's
    documented scope — hybrid compositions are refused at engine boot, this
    guards a cache that slipped past that).
    """
    import mlx.core as mx  # noqa: PLC0415
    from mlx_lm.models.cache import KVCache  # noqa: PLC0415

    from mlxcache_sidecar.tq.codebook import lloyd_max_codebook  # noqa: PLC0415
    from mlxcache_sidecar.tq.rotation import build_rotation  # noqa: PLC0415

    if not cache:
        raise ValueError("empty cache list")
    # Pass 1 — validate the whole stack before encoding anything, so a
    # rejected cache costs no quantize work and the first error is the
    # structural one (not a tensor-shape complaint from deep inside).
    offsets = set()
    head_dims = set()
    dtypes = set()
    for i, c in enumerate(cache):
        if type(c) is not KVCache:
            raise ValueError(
                f"TQ tier requires a dense KVCache stack; layer {i} is {type(c).__name__}"
            )
        if c.keys is None:
            raise ValueError(f"layer {i} holds no state (empty cache)")
        k, _ = c.state
        offsets.add(k.shape[2])
        head_dims.add(k.shape[-1])
        # str(dtype) is "mlx.core.float16"; store the bare name so decode can
        # getattr(mx, name) — the same normalization _kv_dtype() applies.
        dtypes.add(str(k.dtype).replace("mlx.core.", ""))
    if len(offsets) != 1 or len(head_dims) != 1 or len(dtypes) != 1:
        raise ValueError(
            "TQ tier requires uniform offset/head_dim/dtype across layers "
            f"(offsets={offsets}, head_dims={head_dims}, dtypes={dtypes})"
        )
    offset = offsets.pop()
    d = head_dims.pop()
    dtype = dtypes.pop()
    if d % group_size:
        raise ValueError(f"head_dim {d} not divisible by value group size {group_size}")

    # Pass 2 — encode.
    tensors: dict[str, mx.array] = {}
    rotation = build_rotation(d)
    codebook = lloyd_max_codebook(d, key_bits)
    levels, _boundaries, _mse = codebook
    # The codebook LEVELS travel in the frame alongside the rotation: decode
    # must not recompute them (a future solver change would silently shift
    # what every existing blob decodes to — the digest still matches, the
    # KV silently changes). Boundaries are encode-only and stay out.
    tensors[f"cb.{d}.{key_bits}"] = mx.array(levels, dtype=mx.float32)
    for i, c in enumerate(cache):
        k, v = c.state
        tensors[f"layers.{i}.k.idx"], tensors[f"layers.{i}.k.norm"] = encode_keys(
            k, codebook, rotation
        )
        tensors[f"layers.{i}.v.q"], tensors[f"layers.{i}.v.s"], tensors[f"layers.{i}.v.b"] = (
            mx.quantize(v, group_size=group_size, bits=value_bits)
        )
    tensors[f"rot.{d}"] = rotation

    buf = io.BytesIO()
    mx.save_safetensors(
        buf,
        tensors,
        {
            "format": FORMAT,
            "key_bits": str(key_bits),
            "value_bits": str(value_bits),
            "group_size": str(group_size),
            "offset": str(offset),
            "kv_dtype": dtype,
        },
    )
    return buf.getvalue()


def decode_payload(payload: bytes) -> list:
    """Rebuild a dense KVCache list from the compressed frame.

    The result is a live, stock cache: offsets set from the tensor shapes,
    ready for delta prefill. Structural corruption raises the same exception
    family ``_load_cache_from_payload`` maps to CheckpointRejectedError."""
    import mlx.core as mx  # noqa: PLC0415
    from mlx_lm.models.cache import KVCache  # noqa: PLC0415

    blob = io.BytesIO(payload)
    # mx.load infers safetensors from the extension and reads `.name`:
    # a bare BytesIO raises AttributeError (the pinned landmine).
    blob.name = "checkpoint.safetensors"
    arrays, metadata = mx.load(blob, return_metadata=True)
    if not metadata or metadata.get("format") != FORMAT:
        raise ValueError("not a TurboQuant frame (missing format metadata)")
    key_bits = int(metadata["key_bits"])
    value_bits = int(metadata["value_bits"])
    group_size = int(metadata["group_size"])
    offset = int(metadata["offset"])
    dtype = getattr(mx, metadata["kv_dtype"])

    rotations = {}
    codebooks = {}
    # Five tensors name each layer (k.idx, k.norm, v.q, v.s, v.b): collect ids
    # as a set first, or each layer decodes five times.
    layer_ids = sorted({int(name.split(".")[1]) for name in arrays if name.startswith("layers.")})
    if layer_ids != list(range(len(layer_ids))):
        # A digest-consistent-but-planted frame could drop layer tensors; a
        # short or non-contiguous stack would IndexError inside model
        # forward — past the 422 boundary — and 500 forever while the
        # daemon never quarantines it. Reject here instead.
        raise ValueError(f"frame layer ids are not contiguous from 0: {layer_ids}")
    cache = []
    for i in layer_ids:
        k_idx = arrays[f"layers.{i}.k.idx"]
        norms = arrays[f"layers.{i}.k.norm"]
        v_q = arrays[f"layers.{i}.v.q"]
        v_s = arrays[f"layers.{i}.v.s"]
        v_b = arrays[f"layers.{i}.v.b"]
        d = v_s.shape[-1] * group_size
        if d not in rotations:
            rot_name = f"rot.{d}"
            if rot_name not in arrays:
                raise ValueError(f"frame carries no rotation for head_dim {d}")
            # The levels come from the FRAME, never recomputed: the blob is
            # self-describing even if this process's solver drifts.
            cb_name = f"cb.{d}.{key_bits}"
            if cb_name not in arrays:
                raise ValueError(f"frame carries no codebook for head_dim {d}")
            levels = arrays[cb_name]
            if levels.shape != (1 << key_bits,):
                raise ValueError(f"codebook has {levels.shape[0]} levels for {key_bits} bits")
            rotations[d] = arrays[rot_name]
            codebooks[d] = (levels.tolist(), (), 0.0)
        k = decode_keys(k_idx, norms, d, codebooks[d], rotations[d], dtype)
        v = mx.dequantize(v_q, v_s, v_b, group_size=group_size, bits=value_bits).astype(dtype)
        if k.shape[2] != offset:
            raise ValueError(f"layer {i} offset {k.shape[2]} != frame offset {offset}")
        cache.append(KVCache.from_state((k, v), ""))
    return cache
