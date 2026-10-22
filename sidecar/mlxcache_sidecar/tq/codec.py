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
# The only dtypes a KV cache can hold; a frame naming anything else (uint8,
# int64, ...) would decode "successfully" into garbage KV.
FLOAT_DTYPES = ("float16", "bfloat16", "float32")


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
    # One threshold per pass rather than a [B,H,S,D,k-1] broadcast: MLX does
    # not fuse elementwise into reductions on Metal, so the broadcast would
    # materialize ~768 MB of transient bool at 50K tokens, k=16. The running
    # sum computes the identical bucket index.
    idx = mx.zeros_like(y).astype(mx.uint32)
    for t in boundaries:
        idx = idx + (y > t).astype(mx.uint32)
    # Compensate the stored norm for the reconstructed direction's scale:
    # the dequantized unit vector has norm s = ||levels[idx]||/sqrt(d) < 1
    # (a centroid shrinks toward the mean), so storing ||k|| verbatim would
    # scale the decoded row to ||k||*s — and every decode->encode cycle
    # (delta prefill adopting an ancestor, settle of a resumed cache) would
    # multiply another factor of s, shrinking the oldest keys ~1.5% per turn
    # at k3. Dividing by s makes decode return ||k|| exactly, and chains
    # converge (re-quantizing an already-quantized direction re-selects the
    # same centroid).
    recon = mx.take(mx.array(levels, dtype=mx.float32), idx.reshape(-1)).reshape(idx.shape)
    s = mx.sqrt(mx.sum(recon * recon, axis=-1)) / mx.sqrt(float(keys.shape[-1]))
    norms = norms / mx.maximum(s, 1e-12)
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
        k, v = c.state
        offsets.add(k.shape[2])
        # Key AND value head dims: the frame names rot.{d}/cb.{d} by the KEY
        # dim while decode derives d from the VALUE scales — an architecture
        # with kdim != vdim would encode and then fail to decode, so refuse
        # it here (uniform across layers and equal to each other).
        head_dims.add(k.shape[-1])
        head_dims.add(v.shape[-1])
        # One dtype for keys AND values per layer: the frame records a single
        # kv_dtype, so a mixed f16-key/f32-value cache would silently downcast
        # the values at decode. Record the KEY dtype (the one decode round-
        # trips first) and require the value dtype to agree with it.
        for name, tensor in (("key", k), ("value", v)):
            dtype_name = str(tensor.dtype).replace("mlx.core.", "")
            if dtype_name not in FLOAT_DTYPES:
                raise ValueError(f"layer {i} {name}s cache in non-float dtype {dtype_name!r}")
            dtypes.add(dtype_name)
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


def decode_frame(payload: bytes) -> tuple[dict, dict]:
    """Validated tensors and metadata of a TQ frame, WITHOUT rebuilding a
    cache — the live tier adopts the compressed tensors directly. Applies
    every boundary check decode_payload applies, so callers can trust the
    frame the same way."""
    import mlx.core as mx  # noqa: PLC0415

    blob = io.BytesIO(payload)
    # mx.load infers safetensors from the extension and reads `.name`:
    # a bare BytesIO raises AttributeError (the pinned landmine).
    blob.name = "checkpoint.safetensors"
    arrays, metadata = mx.load(blob, return_metadata=True)
    if not metadata or metadata.get("format") != FORMAT:
        raise ValueError("not a TurboQuant frame (missing format metadata)")
    try:
        key_bits = int(metadata["key_bits"])
        value_bits = int(metadata["value_bits"])
        group_size = int(metadata["group_size"])
        offset = int(metadata["offset"])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"frame metadata invalid: {exc}") from exc
    # Range-check at the boundary: an out-of-range value (e.g. key_bits=0)
    # would otherwise surface as ZeroDivisionError deep in _unpack_indices —
    # outside the ValueError family the adapter maps to a 422 quarantine, so
    # the daemon would 500-loop on a poison frame instead of retiring it.
    if key_bits not in KEY_BITS_CHOICES:
        raise ValueError(f"frame key_bits {key_bits} not in {KEY_BITS_CHOICES}")
    if value_bits not in VALUE_BITS_CHOICES:
        raise ValueError(f"frame value_bits {value_bits} not in {VALUE_BITS_CHOICES}")
    if group_size < 2:
        raise ValueError(f"frame group_size {group_size} must be >= 2")
    if offset < 0:
        raise ValueError(f"frame offset {offset} must be >= 0")
    dtype_name = metadata.get("kv_dtype", "")
    if dtype_name not in FLOAT_DTYPES:
        raise ValueError(f"frame kv_dtype {dtype_name!r} is not a floating dtype")
    # Five tensors name each layer (k.idx, k.norm, v.q, v.s, v.b). An EMPTY
    # set passes a contiguity check trivially and would hand the model a
    # zero-layer cache (mlx-lm zips layers with the cache, silently skipping
    # every layer) — reject it as corruption, along with short or
    # non-contiguous stacks: they fail past the 422 boundary and 500-loop
    # while the daemon never quarantines them.
    layer_ids = sorted({int(name.split(".")[1]) for name in arrays if name.startswith("layers.")})
    if not layer_ids or layer_ids != list(range(len(layer_ids))):
        raise ValueError(f"frame layer ids are empty or not contiguous from 0: {layer_ids}")
    # Per-layer structural checks, run for EVERY consumer (the dense decode
    # path and the live adopt path alike): tensors must cover the claimed
    # offset, key and value tensors must agree on (batch, heads, sequence),
    # and the packed widths must match the declared bits. Without these a
    # planted frame either silently clamps to a short history or explodes
    # inside model forward — past the 422 boundary, unretired.
    for i in layer_ids:
        k_idx = arrays.get(f"layers.{i}.k.idx")
        norms = arrays.get(f"layers.{i}.k.norm")
        v_q = arrays.get(f"layers.{i}.v.q")
        v_s = arrays.get(f"layers.{i}.v.s")
        v_b = arrays.get(f"layers.{i}.v.b")
        if any(x is None for x in (k_idx, norms, v_q, v_s, v_b)):
            raise ValueError(f"layer {i} is missing tensors")
        # Rank first: a rank-3 k.idx passes the [:3] comparisons below (a
        # rank-4 value tensor shares its leading axes) and only explodes as
        # IndexError on the first live append — past the 422 boundary.
        if k_idx.ndim != 4 or v_q.ndim != 4 or norms.ndim != 3:
            raise ValueError(f"layer {i} tensor ranks are wrong")
        # Scale/bias pair must be rank-4, identical to each other, and share
        # the packed values' leading axes: the live path stores them raw and
        # only touches them inside the model forward, where a mismatch is a
        # 500 loop instead of a 422 quarantine.
        if v_s.ndim != 4 or v_b.ndim != 4 or v_s.shape != v_b.shape:
            raise ValueError(f"layer {i} value scale/bias tensors disagree")
        if v_s.shape[:3] != v_q.shape[:3]:
            raise ValueError(f"layer {i} value scale/bias leading axes disagree with v.q")
        if min(k_idx.shape[2], norms.shape[2], v_q.shape[2], v_s.shape[2], v_b.shape[2]) < offset:
            raise ValueError(f"layer {i} tensors do not cover the claimed offset {offset}")
        if k_idx.shape[:3] != v_q.shape[:3] or norms.shape[:3] != v_q.shape[:3]:
            raise ValueError(
                f"layer {i} key/value shapes disagree: {tuple(k_idx.shape)} vs {tuple(v_q.shape)}"
            )
        # Packed widths must match the declared bits: a width mismatch would
        # either unpack garbage indices or misalign the group-quantized rows.
        d = v_s.shape[-1] * group_size
        expected_k_w = -(-d // (32 // key_bits)) if key_bits else 0
        expected_v_w = -(-d * value_bits // 32)
        if k_idx.shape[-1] != expected_k_w or v_q.shape[-1] != expected_v_w:
            raise ValueError(
                f"layer {i} packed widths {k_idx.shape[-1]}/{v_q.shape[-1]} do not match "
                f"bits {key_bits}/{value_bits} for head_dim {d}"
            )
    meta = {
        "key_bits": key_bits,
        "value_bits": value_bits,
        "group_size": group_size,
        "offset": offset,
        "kv_dtype": getattr(mx, dtype_name),
        "layer_ids": layer_ids,
    }
    return arrays, meta


def frame_basis(arrays: dict, meta: dict):
    """The (rotation, codebook) pair a frame carries, in the shape the
    codec's encode/decode helpers expect. One source for both adopt paths.
    The codebook LENGTH is pinned to the declared bits, and the bucketing
    boundaries are RECONSTRUCTED from the frame's own levels (the solver's
    fixed point makes boundaries the midpoints of adjacent centroids —
    pinned by test_fixed_point_optimality), so an adopted live cache
    quantizes appended tokens with the frame's exact codebook instead of an
    empty-boundary book that would encode every coordinate as index 0."""
    d = arrays[f"layers.{meta['layer_ids'][0]}.v.s"].shape[-1] * meta["group_size"]
    rot_name, cb_name = f"rot.{d}", f"cb.{d}.{meta['key_bits']}"
    if rot_name not in arrays or cb_name not in arrays:
        raise ValueError(f"frame carries no rotation/codebook for head_dim {d}")
    rotation = arrays[rot_name]
    if rotation.ndim != 2 or rotation.shape != (d, d):
        raise ValueError(f"frame rotation is not [{d},{d}]: {tuple(rotation.shape)}")
    levels = arrays[cb_name]
    if levels.shape != (1 << meta["key_bits"],):
        raise ValueError(f"codebook has {levels.shape[0]} levels for {meta['key_bits']} bits")
    level_values = levels.tolist()
    boundaries = tuple(
        (a + b) / 2.0 for a, b in zip(level_values[:-1], level_values[1:], strict=True)
    )
    return rotation, (level_values, boundaries, 0.0)


def decode_payload(payload: bytes) -> list:
    """Rebuild a dense KVCache list from the compressed frame.

    The result is a live, stock cache: offsets set from the tensor shapes,
    ready for delta prefill. Structural corruption raises the same exception
    family ``_load_cache_from_payload`` maps to CheckpointRejectedError."""
    import mlx.core as mx  # noqa: PLC0415
    from mlx_lm.models.cache import KVCache  # noqa: PLC0415

    arrays, meta = decode_frame(payload)
    key_bits = meta["key_bits"]
    value_bits = meta["value_bits"]
    group_size = meta["group_size"]
    offset = meta["offset"]
    dtype = meta["kv_dtype"]

    rotations = {}
    codebooks = {}
    # decode_frame already validated the layer-id set; collect ids once
    # (each layer contributes five tensor names).
    layer_ids = sorted({int(name.split(".")[1]) for name in arrays if name.startswith("layers.")})
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
        # Key/value shapes must AGREE per layer (batch, heads, sequence): a
        # planted frame with mismatched heads passes KVCache.from_state and
        # then explodes inside update_and_fetch on the first resume — past
        # the 422 boundary, so the poison would 500-loop unretired.
        if k.shape[:3] != v.shape[:3]:
            raise ValueError(
                f"layer {i} key/value shapes disagree: {tuple(k.shape)} vs {tuple(v.shape)}"
            )
        cache.append(KVCache.from_state((k, v), ""))
    return cache
