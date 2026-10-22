"""TurboQuantKVCache: the LIVE tier — compressed KV resident in unified
memory, dequantized on read.

The persistence codec (``codec``) compresses at capture and decompresses at
adopt; the engine's live cache stays f16. This class inverts that: it IS the
cache mlx-lm's attention reads from, and it holds the packed Lloyd-Max key
indices + norms and the group-quantized values as its resident state. On
``update_and_fetch`` it quantizes the new tokens into that state and returns
the dequantized FULL history, so stock attention code runs unchanged — no
mlx-lm monkey-patching; the cache-class API is the intended extension point
(the wrapper's no-patches property is preserved).

Resident-memory math (Qwen2.5-7B, d=128, 4 KV heads): f16 KV is 4,096 B per
token per layer; the tq-k4v4-g64 state is ~1,120 B — the live tier trades
~3.6x smaller resident KV for one transient dequantized buffer per forward
plus the per-step dequant compute (the probe measures both).

Correctness invariant: quantize-at-write applies the same elementwise math
as the persistence codec's ``encode_keys``/``mx.quantize`` over the same
rows, so the resident packed state equals the persistence frame's tensors
for the same token stream, and `serialize_live` emits the byte-identical
frame `encode_cache` would (pinned by test at the tensor level).

What the live tier does NOT promise: token identity with a persistence-tier
generation. Quantizing during the forward changes the computation itself —
every position's keys are computed attending through quantize-then-
dequantize history, recursively — while the persistence tier's blob was
computed by an f16 prefill and quantized afterward. Same tier label, same
frame, different forward; the real gate pins each tier's determinism and a
bounded cross-tier divergence instead.

The class does not own its basis: every layer shares one (rotation,
codebook) per head_dim, supplied by the engine and shared by reference; the
serialized frame carries it exactly once, as in every TQ frame.
"""

from __future__ import annotations

import io

from mlxcache_sidecar.tq.codec import FORMAT, decode_keys, encode_keys

# Capacity grows in position steps, mirroring mlx-lm's KVCache, so appends
# stay amortized O(new tokens) instead of O(history) reallocations.
STEP = 256


class TurboQuantKVCache:
    """One attention layer's compressed KV state.

    Shapes follow mlx-lm's KVCache ([B, n_kv_heads, S, D]); `update_and_fetch`
    returns dequantized arrays shaped exactly like KVCache's slices.
    """

    def __init__(self, key_bits: int, value_bits: int, group_size: int, basis=None):
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.group_size = group_size
        self._basis = basis  # (rotation [d,d] f32, codebook tuple) or None
        self.keys = None  # packed idx u32 [B,H,cap,W]
        self.norms = None  # f32 [B,H,cap]
        self.values = None  # (q u32 [B,H,cap,Wv], scales, biases)
        self.offset = 0
        self._kv_dtype = None

    # -- basis ---------------------------------------------------------------

    def _resolve_basis(self, head_dim: int):
        if self._basis is None:
            from mlxcache_sidecar.tq.codebook import lloyd_max_codebook  # noqa: PLC0415
            from mlxcache_sidecar.tq.rotation import build_rotation  # noqa: PLC0415

            self._basis = (build_rotation(head_dim), lloyd_max_codebook(head_dim, self.key_bits))
        return self._basis

    def adopt_state(self, packed, norms, v_q, v_s, v_b, offset, dtype, basis):
        """Build from an already-decoded TQ frame (no dequantize-then-
        requantize round trip: the resident state IS the frame's tensors)."""
        self.keys = packed
        self.norms = norms
        self.values = (v_q, v_s, v_b)
        self.offset = offset
        self._kv_dtype = dtype
        self._basis = basis

    # -- mlx-lm cache protocol ----------------------------------------------

    def update_and_fetch(self, keys, values):
        import mlx.core as mx  # noqa: PLC0415

        b, h, s, d = keys.shape
        v_dim = values.shape[-1]
        prev = self.offset
        rotation, codebook = self._resolve_basis(d)

        if self.keys is None:
            cap = ((s + STEP - 1) // STEP) * STEP
            w_k = -(-d // (32 // self.key_bits))
            self.keys = mx.zeros((b, h, cap, w_k), dtype=mx.uint32)
            self.norms = mx.zeros((b, h, cap), dtype=mx.float32)
            self.values = (
                mx.zeros((b, h, cap, -(-v_dim * self.value_bits // 32)), dtype=mx.uint32),
                mx.zeros((b, h, cap, v_dim // self.group_size), dtype=values.dtype),
                mx.zeros((b, h, cap, v_dim // self.group_size), dtype=values.dtype),
            )
        elif prev + s > self.keys.shape[2]:
            cap = ((prev + s + STEP - 1) // STEP) * STEP

            def grow(x):
                pad = mx.zeros((*x.shape[:2], cap - x.shape[2], x.shape[3]), x.dtype)
                return mx.concatenate([x, pad], axis=2)

            self.keys = grow(self.keys)
            self.norms = mx.concatenate(
                [self.norms, mx.zeros((b, h, cap - self.norms.shape[2]), mx.float32)], axis=2
            )
            self.values = tuple(grow(x) for x in self.values)

        self.offset += s
        self._kv_dtype = keys.dtype

        # Quantize the NEW rows (elementwise: prior bytes untouched).
        packed, norms = encode_keys(keys, codebook, rotation)
        self.keys[..., prev : self.offset, :] = packed
        self.norms[..., prev : self.offset] = norms
        q, scales, biases = mx.quantize(values, group_size=self.group_size, bits=self.value_bits)
        self.values[0][..., prev : self.offset, :] = q
        self.values[1][..., prev : self.offset, :] = scales
        self.values[2][..., prev : self.offset, :] = biases

        return self._fetch()

    def _fetch(self):
        """Dequantize the resident state up to `offset` — the arrays stock
        attention consumes. One transient full-history buffer per call; MLX's
        allocator reuses it across steps (same shape every decode step)."""
        import mlx.core as mx  # noqa: PLC0415

        rotation, codebook = self._basis
        d = rotation.shape[0]
        o = self.offset
        k = decode_keys(
            self.keys[..., :o, :],
            self.norms[..., :o],
            d,
            codebook,
            rotation,
            self._kv_dtype,
        )
        q, scales, biases = self.values
        v = mx.dequantize(
            q[..., :o, :],
            scales[..., :o, :],
            biases[..., :o, :],
            group_size=self.group_size,
            bits=self.value_bits,
        ).astype(self._kv_dtype)
        return k, v

    @property
    def state(self):
        """The compressed per-layer pytree up to `offset` (what our
        serializer writes and `mx.eval` materializes; NOT mlx-lm's k/v
        pair — this class never round-trips through save_prompt_cache)."""
        if self.keys is None:
            return []
        o = self.offset
        return [
            self.keys[..., :o, :],
            self.norms[..., :o],
            self.values[0][..., :o, :],
            self.values[1][..., :o, :],
            self.values[2][..., :o, :],
        ]

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        total = self.keys.nbytes + self.norms.nbytes
        for x in self.values:
            total += x.nbytes
        return int(total)

    def is_trimmable(self):
        return True

    def trim(self, n: int) -> int:
        n = min(self.offset, n)
        self.offset -= n
        return n

    def size(self) -> int:
        return self.offset

    def empty(self) -> bool:
        return self.keys is None

    def make_mask(self, *args, **kwargs):
        from mlx_lm.models.cache import create_attention_mask  # noqa: PLC0415

        # Signature mirrors mlx-lm's KVCache: window_size is the optional
        # second positional; the factory demands it positionally.
        if len(args) == 1 and "window_size" not in kwargs:
            kwargs["window_size"] = None
        return create_attention_mask(*args, offset=self.offset, **kwargs)


def serialize_live(cache: list, key_bits: int, value_bits: int, group_size: int) -> bytes:
    """Write a live stack's resident state as the standard TQ frame.

    Byte-identical to `encode_cache` over the same token stream (same
    elementwise quantizers applied to the same rows at write time), so a
    blob written by a live-tier engine is indistinguishable from one written
    by the persistence tier — one tier label, one format, two runtimes."""
    import mlx.core as mx  # noqa: PLC0415

    if not cache:
        raise ValueError("empty cache list")
    offsets = {c.offset for c in cache}
    head_dims = {c._basis[0].shape[0] for c in cache}
    dtypes = {
        str(c._kv_dtype).replace("mlx.core.", "") if c._kv_dtype is not None else None
        for c in cache
    }
    if len(offsets) != 1 or len(head_dims) != 1 or None in dtypes or len(dtypes) != 1:
        raise ValueError(
            f"live stack is not uniform (offsets={offsets}, head_dims={head_dims}, dtypes={dtypes})"
        )
    offset = offsets.pop()
    d = head_dims.pop()
    dtype = dtypes.pop()
    rotation, codebook = cache[0]._basis
    levels = mx.array(codebook[0], dtype=mx.float32)

    # Insertion order mirrors encode_cache exactly (codebook, per-layer
    # tensors, rotation last): the frame is byte-identical, not merely
    # semantically equal — one tier label, one format, two runtimes.
    tensors: dict[str, mx.array] = {f"cb.{d}.{key_bits}": levels}
    for i, c in enumerate(cache):
        k_idx, norms, v_q, v_s, v_b = c.state
        tensors[f"layers.{i}.k.idx"] = k_idx
        tensors[f"layers.{i}.k.norm"] = norms
        tensors[f"layers.{i}.v.q"] = v_q
        tensors[f"layers.{i}.v.s"] = v_s
        tensors[f"layers.{i}.v.b"] = v_b
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
