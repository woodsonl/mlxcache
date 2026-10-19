"""MLX port of the `qwen4_exp_text` stack (Qwen3.8-Flash-Next).

The reference is `transformers/models/qwen4_exp/modeling_qwen4_exp.py`; every
block here follows it op-for-op, including the places where the reference does
something unusual (weight-centred RMSNorm, gate-after-norm, duplicated mRoPE
frequencies, hyper-connection residual streams, block-pooled QSA indexer).

Coupling to the weight loader is the `WeightSource` protocol. Names are the
FULL checkpoint names from `model.safetensors.index.json`; nothing is stripped
here. `weight_shapes()` enumerates every name the model can ask for, so the
loader and the shape-verification test share one source of truth.
"""

from __future__ import annotations

import math
from typing import Any, Protocol

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import ArraysCache, CacheList, KVCache

from .config import LINEAR, Qwen4TextConfig, build_layer_multipliers

LAYER_PREFIX = "model.language_model.layers"
EMBED_NAME = "model.language_model.embed_tokens.weight"
LM_HEAD_NAME = "lm_head.weight"
MIXER_PREFIX = "model.language_model.hyper_connection_mixer"

# QSA block scales are stored in 128x128 blocks for the fp8 PLE shards.
FP8_BLOCK = 128

# Query rows the indexer processes per vectorised step. Keeps the gathered
# block-key tensor bounded: queries x budget x ratio x head_dim.
INDEXER_QUERY_TILE = 32


class WeightSource(Protocol):
    """The only coupling between the architecture and the checkpoint reader."""

    def get(self, name: str) -> mx.array: ...

    def get_slice(self, name: str, rows: slice) -> mx.array: ...


# --------------------------------------------------------------------- helpers


def _act(name: str, x: mx.array) -> mx.array:
    if name == "silu":
        return x * mx.sigmoid(x)
    if name == "sigmoid":
        return mx.sigmoid(x)
    raise ValueError(f"unsupported activation: {name!r}")


def _linear(x: mx.array, weight: mx.array, bias: mx.array | None = None) -> mx.array:
    y = mx.matmul(x, weight.swapaxes(-1, -2))
    return y if bias is None else y + bias


def _rms_norm_groups(x: mx.array, weight: mx.array, eps: float, group_size: int | None) -> mx.array:
    """`output * (1 + weight)` after an optional within-group RMS.

    float32 throughout matches the reference, which upcasts before the mean and
    only casts back at the end; the `1 +` centring is the checkpoint convention.
    """
    x32 = x.astype(mx.float32)
    if group_size is not None:
        shape = x32.shape
        grouped = x32.reshape(*shape[:-1], -1, group_size)
        normed = grouped * mx.rsqrt(mx.mean(grouped * grouped, axis=-1, keepdims=True) + eps)
        x32 = normed.reshape(shape)
    else:
        x32 = x32 * mx.rsqrt(mx.mean(x32 * x32, axis=-1, keepdims=True) + eps)
    return (x32 * (1.0 + weight.astype(mx.float32))).astype(x.dtype)


class Qwen4RMSNorm(nn.Module):
    def __init__(self, weights: WeightSource, name: str, eps: float, group_size: int | None = None):
        super().__init__()
        self.weight = weights.get(name)
        self.eps = eps
        self.group_size = group_size

    def __call__(self, x: mx.array) -> mx.array:
        return _rms_norm_groups(x, self.weight, self.eps, self.group_size)


class Qwen4RMSNormGated(nn.Module):
    """Norm, scale, THEN gate (the reference normalises before gating)."""

    def __init__(self, weights: WeightSource, name: str, eps: float, activation: str):
        super().__init__()
        self.weight = weights.get(name)
        self.eps = eps
        self.activation = activation

    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        out_dtype = x.dtype
        x32 = x.astype(mx.float32)
        variance = mx.mean(x32 * x32, axis=-1, keepdims=True)
        normed = x32 * mx.rsqrt(variance + self.eps)
        # The reference multiplies by the weight in the input dtype, then by the
        # (float32) gate activation, and casts once at the end.
        scaled = (self.weight * normed.astype(out_dtype)).astype(mx.float32)
        gated = scaled * _act(self.activation, gate.astype(mx.float32))
        return gated.astype(out_dtype)


def _rotate_half(x: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    return mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def _apply_rotary(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Partial rotary over the leading `cos.shape[-1]` channels of the last axis.

    `cos`/`sin` broadcast over the head axis: pass them as [B,1,S,rot] for
    attention heads or [S,rot] for one key per position.
    """
    rotary_dim = cos.shape[-1]
    rope, keep = x[..., :rotary_dim], x[..., rotary_dim:]
    rotated = rope * cos + _rotate_half(rope) * sin
    if keep.shape[-1] == 0:
        return rotated
    return mx.concatenate([rotated, keep], axis=-1)


class Qwen4RotaryEmbedding(nn.Module):
    """3-axis mRoPE with the reference's frequency recomposition.

    For text-only input all three position grids are identical, so the
    recomposition reduces to duplicating the frequency vector; the general form
    is kept because the vision tower feeds real (T, H, W) grids through it.
    """

    def __init__(self, config: Qwen4TextConfig):
        super().__init__()
        dim = config.rotary_dim
        self.dim = dim
        self.mrope_section = list(config.mrope_section)
        inv_freq = 1.0 / (config.rope_theta ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
        self.inv_freq = inv_freq
        self.attention_scaling = config.attention_scaling
        # The frequency axis is half the rotary width: the reference rotates with
        # `cat((freqs, freqs))`, so its recomposition writes into `dim // 2`.
        half = dim // 2
        positions = mx.arange(half)
        masks = []
        running = mx.zeros((half,), dtype=mx.bool_)
        for axis, offset in ((1, 1), (2, 2)):
            length = self.mrope_section[axis] * 3
            mask = (mx.remainder(positions - offset, 3) == 0) & (positions < length)
            masks.append(mask & ~running)
            running = running | mask
        self.axis_masks = masks

    def frequency_grid(self, position_ids: mx.array, batch: int = 0) -> mx.array:
        """Normalise position ids to [3, B, S] (the reference's `[1:]` slice)."""
        if position_ids.ndim == 1:
            position_ids = position_ids[None, None, :]
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, :, :]
        if position_ids.shape[0] == 4:
            return position_ids[1:]
        if position_ids.shape[0] == 1:
            return mx.broadcast_to(position_ids, (3,) + position_ids.shape[1:])
        if position_ids.shape[0] == 3:
            return position_ids
        raise ValueError(f"mRoPE expects 1, 3 or 4 position axes, got {position_ids.shape[0]}.")

    def __call__(self, position_ids: mx.array) -> tuple[mx.array, mx.array]:
        grids = self.frequency_grid(position_ids).astype(mx.float32)
        freqs = grids[..., None] * self.inv_freq  # [3, B, S, dim/2]
        cos = mx.cos(freqs) * self.attention_scaling
        sin = mx.sin(freqs) * self.attention_scaling
        cos = self.recompose(cos)
        sin = self.recompose(sin)
        return mx.concatenate([cos, cos], axis=-1), mx.concatenate([sin, sin], axis=-1)

    def recompose(self, freq: mx.array) -> mx.array:
        thw = freq[0]
        for axis, mask in zip((1, 2), self.axis_masks, strict=True):
            thw = mx.where(mask, freq[axis], thw)
        return thw


def _causal_mask(q_len: int, kv_len: int) -> mx.array:
    """Boolean [q_len, kv_len] lower-right-aligned causal mask."""
    q_pos = mx.arange(kv_len - q_len, kv_len)
    return mx.arange(kv_len)[None, :] <= q_pos[:, None]


def _shift_right_ignore_eos(ids: mx.array, shift: int, eos: int) -> mx.array:
    """Shift token ids right by `shift`, treating EOS as a segment boundary."""
    if shift == 0:
        return ids
    batch, seq_len = ids.shape
    positions = mx.arange(seq_len)
    eos_positions = mx.where(ids == eos, positions[None, :], -1)
    inclusive = mx.cummax(eos_positions, axis=1)
    previous = mx.concatenate(
        [mx.full((batch, 1), -1, dtype=inclusive.dtype), inclusive[:, :-1]], axis=1
    )
    segment_start = previous + 1
    in_segment = positions[None, :] - segment_start
    source = positions - shift
    gather = mx.maximum(source, 0)[None, :]
    gather = mx.broadcast_to(gather, (batch, seq_len))
    shifted = mx.take_along_axis(ids, gather, axis=1)
    valid = (in_segment >= shift) & (source[None, :] >= 0)
    return mx.where(valid, shifted, mx.full(ids.shape, eos, dtype=ids.dtype))


def _unit_lower_inverse(lower: mx.array) -> mx.array:
    """(I + lower)^-1 for strictly-lower `lower`, by Newton doubling.

    Six doubling steps cover any strictly-lower 64x64 block (degree 63), and
    the whole batch of chunks/heads runs as one batched matmul per step. The
    reference solves the same system by forward substitution when the fast
    triangular solver is unavailable; both are exact for a unit-triangular
    system, this form just keeps the work on the GPU stream.
    """
    n = lower.shape[-1]
    eye = mx.eye(n, dtype=lower.dtype)
    inv = eye - lower
    for _ in range(max(1, math.ceil(math.log2(n)))):
        inv = inv @ (2.0 * eye - (eye + lower) @ inv)
    return inv


def _softplus(x: mx.array) -> mx.array:
    return mx.logaddexp(x, 0.0)


def _l2norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    return x * mx.rsqrt(mx.sum(x * x, axis=-1, keepdims=True) + eps)


def _split_last(x: mx.array, *sizes: int) -> list[mx.array]:
    """Split the last axis at cumulative offsets (an empty list when len(sizes) == 1)."""
    bounds: list[int] = []
    total = 0
    for size in sizes[:-1]:
        total += size
        bounds.append(total)
    return mx.split(x, bounds, axis=-1)


def _dequant_fp8_blocks(payload: mx.array, scale_inv: mx.array, block: int = FP8_BLOCK) -> mx.array:
    """fp8 e4m3 payload with one bf16 scale per `block` x `block` tile.

    The stored grid is `ceil(out/block) x ceil(in/block)`; a partial trailing
    tile still occupies a full stored scale, so the expanded grid is trimmed
    back to the payload shape.

    The loader owns this step for checkpoint reads (it has the byte ranges); this
    copy is for callers holding an already-decoded fp8 payload and its scale
    grid, which is what the shape tables describe.
    """
    values = mx.from_fp8(payload, dtype=mx.float32)
    rows, cols = payload.shape
    expanded = mx.repeat(mx.repeat(scale_inv.astype(mx.float32), block, axis=0), block, axis=1)
    return (values * expanded[:rows, :cols]).astype(mx.bfloat16)


# --------------------------------------------------------------- linear attention

GDN_CHUNK = 64


def _chunk_gated_delta_rule(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    decay: mx.array,
    beta: mx.array,
    state: mx.array | None,
    chunk: int = GDN_CHUNK,
) -> tuple[mx.array, mx.array]:
    """Chunked gated delta rule over [B, H, S, D] tensors, in float32.

    Mirrors the reference's UT-transform chunking: the intra-chunk correction is
    a unit-triangular solve (one inverted factor plus a matmul) and the state
    scan is a sequential loop over chunks. Right padding to whole chunks is
    zeros, matching the reference's `F.pad`, and the output is sliced back.
    """
    batch, heads, seq_len, key_dim = key.shape
    value_dim = value.shape[-1]
    pad = (-seq_len) % chunk
    if pad:
        query = mx.pad(query, [(0, 0), (0, 0), (0, pad), (0, 0)])
        key = mx.pad(key, [(0, 0), (0, 0), (0, pad), (0, 0)])
        value = mx.pad(value, [(0, 0), (0, 0), (0, pad), (0, 0)])
        decay = mx.pad(decay, [(0, 0), (0, 0), (0, pad)])
        beta = mx.pad(beta, [(0, 0), (0, 0), (0, pad)])
    total = seq_len + pad
    chunks = total // chunk
    shape = (batch, heads, chunks, chunk)
    query = query.reshape(*shape, key_dim)
    key = key.reshape(*shape, key_dim)
    value = value.reshape(*shape, value_dim)
    decay = decay.reshape(*shape)
    beta = beta.reshape(*shape)

    v_beta = value * beta[..., None]
    k_beta = key * beta[..., None]

    cum_decay = mx.cumsum(decay, axis=-1)
    positions = mx.arange(chunk)
    strictly_upper = positions[None, :] > positions[:, None]
    spread = cum_decay[..., :, None] - cum_decay[..., None, :]
    pairwise = mx.exp(mx.where(strictly_upper, -mx.inf, spread))

    ut_system = (k_beta @ key.swapaxes(-1, -2)) * pairwise
    intra = (query @ key.swapaxes(-1, -2)) * pairwise
    decayed_k_beta = k_beta * mx.exp(cum_decay)[..., None]

    inverse = _unit_lower_inverse(mx.tril(ut_system, -1))
    new_values = inverse @ v_beta
    k_cumdecay = inverse @ decayed_k_beta

    decayed_query = query * mx.exp(cum_decay)[..., None]
    decayed_key = key * mx.exp(cum_decay[..., -1:] - cum_decay)[..., None]
    chunk_decay = mx.exp(cum_decay[..., -1])

    if state is None:
        state = mx.zeros((batch, heads, key_dim, value_dim), dtype=new_values.dtype)
    out = mx.zeros_like(new_values)
    for index in range(chunks):
        v_new = new_values[:, :, index] - k_cumdecay[:, :, index] @ state
        inter = decayed_query[:, :, index] @ state
        out[:, :, index] = inter + intra[:, :, index] @ v_new
        state = state * chunk_decay[:, :, index][..., None, None] + (
            decayed_key[:, :, index].swapaxes(-1, -2) @ v_new
        )
    return out.reshape(batch, heads, total, value_dim)[:, :, :seq_len], state


def _recurrent_gated_delta_rule(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    decay: mx.array,
    beta: mx.array,
    state: mx.array | None,
) -> tuple[mx.array, mx.array]:
    """Single-token recurrent form; the reference uses it for cached decode."""
    batch, heads, seq_len, key_dim = key.shape
    value_dim = value.shape[-1]
    decayed = mx.exp(decay)[..., None, None]
    if state is None:
        state = mx.zeros((batch, heads, key_dim, value_dim), dtype=value.dtype)
    out = mx.zeros_like(value)
    for index in range(seq_len):
        q_t, k_t, v_t = query[:, :, index], key[:, :, index], value[:, :, index]
        stepped = state * decayed[:, :, index]
        kv_mem = (stepped * k_t[..., None]).sum(axis=-2)
        delta = (v_t - kv_mem) * beta[:, :, index][..., None]
        state = stepped + k_t[..., None] * delta[..., None, :]
        out[:, :, index] = (state * q_t[..., None]).sum(axis=-2)
    return out, state


class Qwen4GatedDeltaNet(nn.Module):
    """Gated delta-rule linear attention with the reference's causal short conv."""

    def __init__(self, config: Qwen4TextConfig, weights: WeightSource, prefix: str):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv_kernel = config.linear_conv_kernel_dim
        self.repeat = self.num_v_heads // self.num_k_heads
        self.activation = config.hidden_act
        self.eps = config.rms_norm_eps

        self.in_proj_qkv = weights.get(f"{prefix}.in_proj_qkv.weight")
        self.in_proj_z = weights.get(f"{prefix}.in_proj_z.weight")
        self.in_proj_b = weights.get(f"{prefix}.in_proj_b.weight")
        self.in_proj_a = weights.get(f"{prefix}.in_proj_a.weight")
        self.conv1d = weights.get(f"{prefix}.conv1d.weight")
        self.A_log = weights.get(f"{prefix}.A_log")
        self.dt_bias = weights.get(f"{prefix}.dt_bias")
        self.out_proj = weights.get(f"{prefix}.out_proj.weight")
        self.norm = Qwen4RMSNormGated(
            weights,
            f"{prefix}.norm.weight",
            self.eps,
            config.output_gate_type or config.hidden_act,
        )

    def __call__(self, x: mx.array, cache: ArraysCache) -> mx.array:
        batch, seq_len, _ = x.shape
        mixed = _linear(x, self.in_proj_qkv)
        z = _linear(x, self.in_proj_z).reshape(batch, seq_len, self.num_v_heads, self.head_v_dim)
        beta = mx.sigmoid(_linear(x, self.in_proj_b)).swapaxes(1, 2)
        a = _linear(x, self.in_proj_a).swapaxes(1, 2)

        keep = self.conv_kernel - 1
        conv_state = cache[0]
        if conv_state is None:
            conv_state = mx.zeros((batch, keep, self.conv_dim), dtype=mixed.dtype)
        else:
            conv_state = _align_batch(conv_state, batch)
        conv_input = mx.concatenate([conv_state, mixed], axis=1)
        cache[0] = conv_input[:, conv_input.shape[1] - keep :, :]
        weight = self.conv1d.reshape(self.conv_dim, self.conv_kernel, 1)
        convolved = mx.conv1d(conv_input, weight, groups=self.conv_dim)
        convolved = _act(self.activation, convolved)

        query, key, value = _split_last(convolved, self.key_dim, self.key_dim, self.value_dim)
        query = query.reshape(batch, seq_len, self.num_k_heads, self.head_k_dim).swapaxes(1, 2)
        key = key.reshape(batch, seq_len, self.num_k_heads, self.head_k_dim).swapaxes(1, 2)
        value = value.reshape(batch, seq_len, self.num_v_heads, self.head_v_dim).swapaxes(1, 2)

        decay = -mx.exp(self.A_log.astype(mx.float32))[:, None] * _softplus(
            a.astype(mx.float32) + self.dt_bias.astype(mx.float32)[:, None]
        )
        query = query.astype(mx.float32)
        key = key.astype(mx.float32)
        value = value.astype(mx.float32)
        if self.repeat > 1:
            query = mx.repeat(query, self.repeat, axis=1)
            key = mx.repeat(key, self.repeat, axis=1)
        query = _l2norm(query) * self.head_k_dim**-0.5
        key = _l2norm(key)

        state = cache[1]
        if state is not None:
            state = _align_batch(state, batch)
        if state is not None and seq_len == 1:
            out, state = _recurrent_gated_delta_rule(query, key, value, decay, beta, state)
        else:
            out, state = _chunk_gated_delta_rule(query, key, value, decay, beta, state)
        cache[1] = state

        out = out.swapaxes(1, 2).reshape(-1, self.head_v_dim)
        out = self.norm(out, z.reshape(-1, self.head_v_dim))
        return _linear(out.reshape(batch, seq_len, self.value_dim), self.out_proj)


# -------------------------------------------------------------------- QSA indexer


class Qwen4QSAIndexer(nn.Module):
    """Block-pooled top-k selection: the sparse mask full-attention layers use.

    The reference walks queries one at a time, groups each query's *visible*
    tokens into consecutive blocks of `compress_ratio`, pools the complete
    blocks, scores the pooled block keys, keeps the best `budget // ratio`
    blocks, and always keeps the incomplete tail. Consecutive visible tokens
    equal consecutive positions only for an unpadded causal mask, so padding is
    rejected rather than approximated: a visible row that is not a causal
    prefix raises. Queries are tiled so the gathered block-key tensor stays
    bounded.
    """

    def __init__(self, config: Qwen4TextConfig, weights: WeightSource, prefix: str):
        super().__init__()
        for field in (
            "indexer_n_heads",
            "indexer_kv_heads",
            "indexer_head_dim",
            "indexer_budget",
            "indexer_compress_ratio",
        ):
            if getattr(config, field) is None:
                raise ValueError(f"Qwen4-Exp QSA indexer requires config.{field}.")
        self.n_heads = int(config.indexer_n_heads)
        self.head_dim = int(config.indexer_head_dim)
        self.token_budget = int(config.indexer_budget)
        self.ratio = int(config.indexer_compress_ratio)
        self.block_topk = self.token_budget // self.ratio
        self.scale = self.head_dim**-0.5
        self.eps = config.rms_norm_eps
        self.index_qk_proj = weights.get(f"{prefix}.index_qk_proj.weight")
        self.q_norm = Qwen4RMSNorm(weights, f"{prefix}.q_layernorm.weight", self.eps)
        self.k_norm = Qwen4RMSNorm(weights, f"{prefix}.k_layernorm.weight", self.eps)

    def __call__(
        self,
        hidden: mx.array,
        rope: Qwen4RotaryEmbedding,
        position_ids: mx.array,
        visible: mx.array,
        cached_keys: mx.array | None,
    ) -> tuple[mx.array, mx.array]:
        batch, seq_len, _ = hidden.shape
        qk = _linear(hidden, self.index_qk_proj)
        q_raw, k_raw = _split_last(qk, self.n_heads * self.head_dim, self.head_dim)
        queries = self.q_norm(q_raw.reshape(batch, seq_len, self.n_heads, self.head_dim))
        queries = _apply_rotary(queries, *[axis[:, :, None, :] for axis in rope(position_ids)])
        raw_keys = k_raw.reshape(batch, seq_len, self.head_dim)
        if cached_keys is None or cached_keys.shape[1] == 0:
            keys = raw_keys
        else:
            keys = mx.concatenate([_align_batch(cached_keys, batch), raw_keys], axis=1)
        kv_len = keys.shape[1]

        if visible.ndim == 4:
            visible = visible[:, 0]
        if visible.shape[0] != batch:
            visible = mx.broadcast_to(visible, (batch, *visible.shape[1:]))
        counts = mx.sum(visible.astype(mx.int32), axis=-1)
        prefix = mx.arange(kv_len)[None, None, :] < counts[..., None]
        if not mx.all(visible == prefix):
            raise ValueError(
                "Qwen4-Exp QSA indexer requires an unpadded causal mask; the visible rows "
                "are not causal prefixes, so block pooling would not match the reference."
            )

        selected = mx.zeros((batch, seq_len, kv_len), dtype=mx.bool_)
        for b in range(batch):
            for start in range(0, seq_len, INDEXER_QUERY_TILE):
                stop = start + INDEXER_QUERY_TILE
                selected[b, start:stop] = self._select(
                    keys[b], rope, queries[b, start:stop], counts[b, start:stop]
                )
        return selected[:, None, :, :], keys

    def _select(
        self,
        keys: mx.array,
        rope: Qwen4RotaryEmbedding,
        queries: mx.array,
        counts: mx.array,
    ) -> mx.array:
        tile, kv_len = queries.shape[0], keys.shape[0]
        blocks = int(counts.max().item()) // self.ratio
        out = mx.zeros((tile, kv_len + 1), dtype=mx.bool_)
        tail = (counts // self.ratio * self.ratio)[:, None] + mx.arange(self.ratio)[None, :]
        tails = mx.where(tail < counts[:, None], tail, kv_len)
        out = mx.put_along_axis(out, tails, mx.ones(tails.shape, dtype=mx.bool_), axis=-1)
        if blocks == 0:
            return out[:, :kv_len]

        keep = min(self.block_topk, blocks)
        pooled = self._pool(keys, rope, blocks)
        scores = mx.matmul(queries.astype(mx.float32), pooled.astype(mx.float32).swapaxes(-1, -2))
        oriented = scores.swapaxes(-1, -2)  # [T, blocks, heads]
        scores = mx.maximum(oriented, 0.0).sum(axis=-1) * self.scale
        valid = mx.arange(blocks)[None, :] < (counts // self.ratio)[:, None]
        scores = mx.where(valid, scores, -mx.inf)
        picked = mx.argpartition(scores, kth=-keep, axis=-1)[..., -keep:]
        picked = mx.where(mx.take_along_axis(valid, picked, axis=-1), picked, blocks)
        tokens = mx.arange(blocks * self.ratio).reshape(blocks, self.ratio)[
            mx.sort(picked, axis=-1)
        ]
        index = mx.concatenate([tokens.reshape(tile, keep * self.ratio), tails], axis=-1)
        index = mx.where(index < kv_len, index, kv_len)
        out = mx.put_along_axis(out, index, mx.ones(index.shape, dtype=mx.bool_), axis=-1)
        return out[:, :kv_len]

    def _pool(self, keys: mx.array, rope: Qwen4RotaryEmbedding, blocks: int) -> mx.array:
        """Pool each complete block at its first position, as the reference does."""
        grouped = keys[: blocks * self.ratio].reshape(blocks, self.ratio, self.head_dim)
        pooled = self.k_norm(mx.mean(grouped.astype(keys.dtype), axis=1))
        starts = mx.arange(blocks) * self.ratio
        cos, sin = rope(starts[None, :])
        return _apply_rotary(pooled[None], cos, sin)[0]


# ------------------------------------------------------------------ attention


class Qwen4Attention(nn.Module):
    """Full attention with the reference's gated query projection and QSA mask."""

    def __init__(self, config: Qwen4TextConfig, weights: WeightSource, prefix: str):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        self.eps = config.rms_norm_eps
        bias = config.attention_bias
        self.q_proj = weights.get(f"{prefix}.q_proj.weight")
        self.k_proj = weights.get(f"{prefix}.k_proj.weight")
        self.v_proj = weights.get(f"{prefix}.v_proj.weight")
        self.o_proj = weights.get(f"{prefix}.o_proj.weight")
        self.q_bias = weights.get(f"{prefix}.q_proj.bias") if bias else None
        self.k_bias = weights.get(f"{prefix}.k_proj.bias") if bias else None
        self.v_bias = weights.get(f"{prefix}.v_proj.bias") if bias else None
        self.o_bias = weights.get(f"{prefix}.o_proj.bias") if bias else None
        self.q_norm = Qwen4RMSNorm(weights, f"{prefix}.q_norm.weight", self.eps)
        self.k_norm = Qwen4RMSNorm(weights, f"{prefix}.k_norm.weight", self.eps)
        self.indexer = Qwen4QSAIndexer(config, weights, f"{prefix}.indexer")

    def __call__(
        self,
        x: mx.array,
        rope: Qwen4RotaryEmbedding,
        position_ids: mx.array,
        mask: mx.array,
        cache: CacheList,
    ) -> mx.array:
        kv_cache, key_cache = cache[0], cache[1]
        offset = kv_cache.offset
        selected, index_keys = self.indexer(
            x, rope, position_ids, mask, key_cache[0] if offset else None
        )
        key_cache[0] = index_keys
        mask = mask & selected

        batch, seq_len, _ = x.shape
        q = _linear(x, self.q_proj, self.q_bias).reshape(
            batch, seq_len, self.n_heads, 2 * self.head_dim
        )
        queries, gate = mx.split(q, 2, axis=-1)
        keys = _linear(x, self.k_proj, self.k_bias).reshape(
            batch, seq_len, self.n_kv_heads, self.head_dim
        )
        values = _linear(x, self.v_proj, self.v_bias).reshape(
            batch, seq_len, self.n_kv_heads, self.head_dim
        )

        queries = self.q_norm(queries)
        keys = self.k_norm(keys)
        cos, sin = rope(position_ids)
        cos, sin = cos[:, :, None, :], sin[:, :, None, :]
        queries = _apply_rotary(queries, cos, sin).swapaxes(1, 2)
        keys = _apply_rotary(keys, cos, sin).swapaxes(1, 2)
        values = values.swapaxes(1, 2)
        keys, values = kv_cache.update_and_fetch(keys, values)

        out = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        out = out.swapaxes(1, 2).reshape(batch, seq_len, -1)
        out = out * mx.sigmoid(gate.reshape(batch, seq_len, -1))
        return _linear(out, self.o_proj, self.o_bias)


# ---------------------------------------------------------------- mixture of experts


class Qwen4SparseMoeBlock(nn.Module):
    """Router + per-token expert application + gated shared expert.

    Routed expert weights are fetched from the `WeightSource` at use time, one
    call per (token, top-k slot): the residency policy lives in the loader, and
    nothing here can know which experts the next token will route to. The
    per-token loop mirrors the reference's accumulation order closely enough to
    keep the default-order sum, and it is what makes a 512-expert layer
    expressible without resident expert weights.
    """

    def __init__(self, config: Qwen4TextConfig, weights: WeightSource, prefix: str):
        super().__init__()
        self.weights = weights
        self.prefix = prefix
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.activation = config.hidden_act
        self.gate = weights.get(f"{prefix}.gate.weight")
        self.shared_gate_proj = weights.get(f"{prefix}.shared_expert.gate_proj.weight")
        self.shared_up_proj = weights.get(f"{prefix}.shared_expert.up_proj.weight")
        self.shared_down_proj = weights.get(f"{prefix}.shared_expert.down_proj.weight")
        self.shared_expert_gate = weights.get(f"{prefix}.shared_expert_gate.weight")
        self.expert_loader: Any = None

    def _expert(self, index: int, name: str) -> mx.array:
        key = (self.prefix, index, name)
        if self.expert_loader is not None:
            return self.expert_loader(key, f"{self.prefix}.experts.{index}.{name}.weight")
        return self.weights.get(f"{self.prefix}.experts.{index}.{name}.weight")

    def __call__(self, x: mx.array) -> mx.array:
        batch, seq_len, hidden = x.shape
        flat = x.reshape(-1, hidden)
        logits = _linear(flat, self.gate)
        probs = mx.softmax(logits.astype(mx.float32), axis=-1, precise=True)
        indices = mx.argpartition(probs, kth=-self.top_k, axis=-1)[..., -self.top_k :]
        scores = mx.take_along_axis(probs, indices, axis=-1)
        if self.norm_topk_prob:
            scores = scores / mx.sum(scores, axis=-1, keepdims=True)
        scores = scores.astype(x.dtype)

        # Group (token, slot) assignments BY EXPERT before applying them. A routed
        # expert is read and dequantized on demand, so iterating token-major
        # re-reads every expert once per token that uses it: on a 256-token
        # prefill that is 256 x 10 reads of 3 matrices per layer, and the same
        # expert is dequantized ten times over. Expert-major makes it one read
        # and one batched matmul per expert per layer, which is also how the
        # reference accumulates (`index_add_` over the expert's token set).
        routed = mx.zeros_like(flat)
        tokens = flat.shape[0]
        # Flatten (token, slot) pairs, then visit each DISTINCT expert once.
        pair_token = mx.repeat(mx.arange(tokens), self.top_k)
        pair_expert = indices.reshape(-1)
        pair_score = scores.reshape(-1)
        distinct = mx.array(sorted({int(e) for e in pair_expert.tolist()}), dtype=mx.int32)

        for expert in distinct.tolist():
            member = pair_expert == expert
            picked = mx.argsort(mx.where(member, 0, 1))[: int(mx.sum(member).item())]
            rows = mx.take(pair_token, picked, axis=0)
            weights = mx.take(pair_score, picked, axis=0)
            current = mx.take(flat, rows, axis=0)
            gate = _act(self.activation, _linear(current, self._expert(expert, "gate_proj")))
            up = _linear(current, self._expert(expert, "up_proj"))
            down = _linear(gate * up, self._expert(expert, "down_proj"))
            routed = routed.at[rows].add(down * weights[:, None])

        shared = _act(self.activation, _linear(flat, self.shared_gate_proj))
        shared = _linear(shared * _linear(flat, self.shared_up_proj), self.shared_down_proj)
        shared = mx.sigmoid(_linear(flat, self.shared_expert_gate)) * shared
        return (routed + shared).reshape(batch, seq_len, hidden)


# ---------------------------------------------------------- hyper-connections


class Qwen4GatedResidual(nn.Module):
    """The reference's hyper-connection mixer: mix streams, inject block output.

    `hc_norm` normalises each of the `hc_count` residual streams independently
    (group_size == hidden_size), which is why it is a grouped RMSNorm over the
    concatenated width.
    """

    def __init__(
        self,
        config: Qwen4TextConfig,
        weights: WeightSource,
        prefix: str,
        use_combine: bool = True,
    ):
        super().__init__()
        self.hc_count = config.hc_count
        self.hidden_size = config.hidden_size
        width = config.hc_hidden_size
        self.hc_norm = Qwen4RMSNorm(
            weights, f"{prefix}.hc_norm.weight", config.rms_norm_eps, self.hidden_size
        )
        self.input_mix_down = weights.get(f"{prefix}.input_mix_weight_down.weight")
        self.input_mix_up = weights.get(f"{prefix}.input_mix_weight_up.weight")
        self.block_inject = (
            weights.get(f"{prefix}.block_inject_weight.weight") if use_combine else None
        )
        if self.input_mix_down.shape != (config.hc_lowrank, width):
            raise ValueError(
                f"{prefix}.input_mix_weight_down has shape {self.input_mix_down.shape}, expected "
                f"{(config.hc_lowrank, width)}."
            )

    def __call__(self, hyper: mx.array) -> tuple[mx.array, mx.array, mx.array | None]:
        batch, seq_len, width = hyper.shape
        if width != self.hc_count * self.hidden_size:
            raise ValueError(
                f"expected {self.hc_count * self.hidden_size} hyper-connection features, "
                f"got {width}."
            )
        normed = self.hc_norm(hyper)
        mix = _act("silu", _linear(normed, self.input_mix_down) / self.hc_count)
        mix = mx.sigmoid(_linear(mix, self.input_mix_up))
        mix = mix.reshape(batch, seq_len, self.hc_count, self.hidden_size)
        streams = normed.reshape(batch, seq_len, self.hc_count, self.hidden_size)
        mixed = mx.mean(mix * streams, axis=-2)
        if self.block_inject is None:
            return mixed, hyper, None
        inject = 2.0 * mx.sigmoid(_linear(normed, self.block_inject) / self.hc_count)
        return mixed, hyper, inject


def _inject(hyper: mx.array, out: mx.array, inject: mx.array, hc_count: int) -> mx.array:
    """Add a block output to every residual stream, weighted per stream."""
    batch, seq_len, hidden = out.shape
    injection = out[..., None, :] * inject[..., None]
    return hyper + injection.reshape(batch, seq_len, hc_count * hidden)


# ------------------------------------------------------- hashed n-gram embeddings


class Qwen4NGramEmbedding(nn.Module):
    """Hashed n-gram lookup through the checkpoint's sharded fp8 table.

    The table is 51 GiB, so it is never resident: rows are requested one
    contiguous run at a time from `WeightSource.get_slice`, grouped per stored
    shard, and gathered back with `searchsorted`. Dequantization (the per-tensor
    fp8 scale) is the loader's business, so slices arrive as float rows.

    Hashing follows the reference exactly: per n-gram order, a fixed odd
    multiplier times each shifted token id, XOR-combined over the order, taken
    modulo a per-head prime vocabulary and offset into that head's row range.
    Shift-right treats EOS as a segment boundary, so an n-gram never spans two
    documents.
    """

    def __init__(self, config: Qwen4TextConfig, weights: WeightSource, prefix: str, rank: int):
        super().__init__()
        self.weights = weights
        self.ngram_size = config.ngram_size
        self.context_len = config.ngram_size - 1
        self.heads_per_ngram = config.heads_per_ngram
        self.ngram_heads = config.ngram_heads
        self.head_dim = config.ngram_head_dim
        self.eos = config.eos_token
        self.multipliers = build_layer_multipliers(
            config.vocab_size, config.ngram_size, rank, config.seed
        )
        sizes = config.head_vocab_sizes(rank)
        offsets = config.head_offsets(rank)
        self.vocab_sizes = mx.array(sizes, dtype=mx.int32)
        self.offsets = mx.array(offsets, dtype=mx.int32)
        self.total_rows = config.ngram_vocab_total(rank)
        self.parts = config.split_ngram_parts
        self.rows_per_part = -(-self.total_rows // self.parts)
        self.shard_names = [
            f"{prefix}.ngram_embedding.shard_{index}.weight" for index in range(self.parts)
        ]

    def __call__(self, ids: mx.array, cache: ArraysCache) -> mx.array:
        batch, seq_len = ids.shape
        context = cache[3]
        if context is None:
            context = mx.full((batch, self.context_len), -1, dtype=ids.dtype)
        else:
            context = _align_batch(context, batch)
        # A -1 row means "before this sequence started", which the reference
        # fills with EOS so no n-gram is formed across the boundary.
        previous = mx.where(context < 0, self.eos, context)
        history = mx.concatenate([previous, ids], axis=1)
        cache[3] = history[:, history.shape[1] - self.context_len :]

        shifted = [
            _shift_right_ignore_eos(history, shift, self.eos) for shift in range(self.ngram_size)
        ]
        blocks = []
        for order in range(2, self.ngram_size + 1):
            start = (order - 2) * self.heads_per_ngram
            stop = start + self.heads_per_ngram
            mixed = shifted[0].astype(mx.int64) * self.multipliers[0]
            for position in range(1, order):
                mixed = mx.bitwise_xor(
                    mixed, shifted[position].astype(mx.int64) * self.multipliers[position]
                )
            sizes = self.vocab_sizes[start:stop].astype(mx.int64)
            offsets = self.offsets[start:stop].astype(mx.int64)
            blocks.append(mx.remainder(mixed[..., None], sizes) + offsets)
        rows = mx.concatenate(blocks, axis=-1)[:, -seq_len:, :]
        return self._lookup(rows)

    def _lookup(self, rows: mx.array) -> mx.array:
        """Gather `rows` [B, S, heads] from the sharded table, one run per shard."""
        batch, seq_len, heads = rows.shape
        flat = rows.reshape(-1)
        parts = mx.floor_divide(flat, self.rows_per_part).astype(mx.int32)
        local = mx.remainder(flat, self.rows_per_part).astype(mx.int32)
        out = mx.zeros((flat.shape[0], self.head_dim), dtype=mx.float32)
        for part in range(self.parts):
            member = (parts == part).astype(mx.int32)
            count = int(mx.sum(member).item())
            if count == 0:
                continue
            picked = mx.sort(mx.argsort(-member)[:count]).astype(mx.int32)
            gathered = _gather_rows(self.weights, self.shard_names[part], local[picked])
            out[picked] = gathered
        return out.reshape(batch, seq_len, heads * self.head_dim)


def _align_batch(state: mx.array, batch: int) -> mx.array:
    if state.shape[0] == batch:
        return state
    if state.shape[0] == 1:
        return mx.broadcast_to(state, (batch,) + state.shape[1:])
    raise ValueError(f"cache batch size {state.shape[0]} does not match input batch {batch}.")


def _flags_to_positions(flags: mx.array) -> mx.array:
    """Ascending indices where `flags` is true, without boolean indexing."""
    count = int(mx.sum(flags).item())
    if count == 0:
        return mx.array([], dtype=mx.int32)
    ranked = mx.argsort(-flags.astype(mx.int32))
    return mx.sort(ranked[:count]).astype(mx.int32)


def _unique_with_index(values: mx.array) -> tuple[mx.array, mx.array]:
    """Sorted unique values and, for each input element, its slot in them."""
    ordered = mx.sort(values)
    if ordered.shape[0] <= 1:
        return ordered, mx.zeros(values.shape, dtype=mx.int32)
    changes = mx.concatenate(
        [mx.ones((1,), dtype=mx.int32), (ordered[1:] != ordered[:-1]).astype(mx.int32)], axis=0
    )
    unique = mx.take(ordered, _flags_to_positions(changes))
    return unique, mx.searchsorted(unique, values).astype(mx.int32)


def _gather_rows(weights: WeightSource, name: str, rows: mx.array) -> mx.array:
    """Fetch arbitrary rows of a sharded table using contiguous slices only.

    `WeightSource.get_slice` addresses a contiguous row range, so scattered
    requests are coalesced into maximal runs of consecutive rows: a shard-local
    run of `k` rows costs one call instead of `k`. The row LRU in the loader is
    what makes repeats free; this function only decides how to batch.
    """
    unique, index = _unique_with_index(rows)
    if unique.shape[0] == 0:
        return mx.zeros((0, 1), dtype=mx.float32)
    starts = mx.concatenate(
        [mx.ones((1,), dtype=mx.int32), (unique[1:] != unique[:-1] + 1).astype(mx.int32)], axis=0
    )
    run_start = _flags_to_positions(starts)
    run_first = mx.take(unique, run_start)
    lengths = mx.concatenate([run_start[1:], mx.array([unique.shape[0]], dtype=mx.int32)], axis=0)
    lengths = lengths - run_start

    tables = [
        weights.get_slice(name, slice(int(run_first[r]), int(run_first[r] + lengths[r])))
        for r in range(run_start.shape[0])
    ]
    table = tables[0] if len(tables) == 1 else mx.concatenate(tables, axis=0)

    # Table row of each unique row: the run's offset plus its distance from the
    # run's first row. Runs are contiguous in `unique`, so a cumulative sum of
    # the preceding run lengths gives each run's offset.
    run_of = mx.cumsum(starts) - 1
    offsets = mx.concatenate([mx.zeros((1,), dtype=mx.int32), mx.cumsum(lengths)[:-1]], axis=0)
    position = offsets[run_of] + (unique - run_first[run_of])
    return mx.take(table, mx.take(position, index), axis=0)


# ------------------------------------------------------------------ PLE layer


class Qwen4PLELayer(nn.Module):
    """Per-layer embedding: hashed n-grams gated into every residual stream.

    The gate is the sign-preserving square root of the per-stream dot product
    between the normalised key and the normalised stream activation, which is
    the reference's odd but deliberate choice: it keeps the gate's magnitude
    tied to alignment while preserving the direction.
    """

    def __init__(self, config: Qwen4TextConfig, weights: WeightSource, prefix: str, rank: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.hc_count = config.hc_count
        width = config.hc_hidden_size
        self.embedding = Qwen4NGramEmbedding(config, weights, f"{prefix}.ple_embedding", rank)
        self.conv_kernel = config.ple_conv_kernel_size
        self.conv_dilation = config.ngram_size
        self.state_len = (self.conv_kernel - 1) * self.conv_dilation
        self.key_proj = weights.get(f"{prefix}.key_proj.weight")
        self.value_proj = weights.get(f"{prefix}.value_proj.weight")
        self.conv1d = weights.get(f"{prefix}.conv1d.weight")
        self.eps = config.rms_norm_eps
        self.norm_key = Qwen4RMSNorm(
            weights, f"{prefix}.norm_key.weight", self.eps, self.hidden_size
        )
        self.norm_query = Qwen4RMSNorm(
            weights, f"{prefix}.norm_query.weight", self.eps, self.hidden_size
        )
        self.norm_conv = Qwen4RMSNorm(
            weights, f"{prefix}.norm_conv.weight", self.eps, self.hidden_size
        )
        if self.key_proj.shape != (width, config.ple_embed_dim):
            raise ValueError(
                f"{prefix}.key_proj has shape {self.key_proj.shape}, expected "
                f"{(width, config.ple_embed_dim)}."
            )

    def __call__(self, hidden: mx.array, ids: mx.array, cache: ArraysCache) -> mx.array:
        batch, seq_len, _ = hidden.shape
        width = self.hc_count * self.hidden_size
        embeddings = self.embedding(ids, cache)
        keys = self.norm_key(_linear(embeddings, self.key_proj)).reshape(
            batch, seq_len, self.hc_count, self.hidden_size
        )
        queries = self.norm_query(hidden).reshape(batch, seq_len, self.hc_count, self.hidden_size)
        gate = mx.sum(keys * queries, axis=-1, keepdims=True) / math.sqrt(self.hidden_size)
        gate = mx.sqrt(mx.maximum(mx.abs(gate), 1e-6)) * mx.sign(gate)
        value = _linear(embeddings, self.value_proj)
        gated = (mx.sigmoid(gate) * value[..., None, :]).reshape(batch, seq_len, width)
        convolved = self._short_conv(self.norm_conv(gated), cache)
        return gated + convolved

    def _short_conv(self, x: mx.array, cache: ArraysCache) -> mx.array:
        """Dilated depthwise conv over a fixed left-context window."""
        batch, seq_len, width = x.shape
        state = cache[2]
        if state is None:
            state = mx.zeros((batch, self.state_len, width), dtype=x.dtype)
        else:
            state = _align_batch(state, batch)
        window = mx.concatenate([state, x], axis=1)
        if window.shape[1] < self.state_len + seq_len:
            window = mx.pad(
                window, [(0, 0), (self.state_len + seq_len - window.shape[1], 0), (0, 0)]
            )
        cache[2] = window[:, seq_len : seq_len + self.state_len]
        weight = self.conv1d.reshape(width, self.conv_kernel, 1)
        return _act("silu", mx.conv1d(window, weight, groups=width, dilation=self.conv_dilation))


# --------------------------------------------------------------- decoder layer


class Qwen4DecoderLayer(nn.Module):
    def __init__(self, config: Qwen4TextConfig, weights: WeightSource, layer_idx: int, prefix: str):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        self.is_linear = self.layer_type == LINEAR
        if self.is_linear:
            self.linear_attn = Qwen4GatedDeltaNet(config, weights, f"{prefix}.linear_attn")
        else:
            self.self_attn = Qwen4Attention(config, weights, f"{prefix}.self_attn")
        self.mlp = Qwen4SparseMoeBlock(config, weights, f"{prefix}.mlp")
        self.attn_hyper = Qwen4GatedResidual(config, weights, f"{prefix}.attn_hyper_connection")
        self.mlp_hyper = Qwen4GatedResidual(config, weights, f"{prefix}.mlp_hyper_connection")
        rank = config.ple_rank(layer_idx)
        self.ple = (
            Qwen4PLELayer(config, weights, f"{prefix}.ple", rank) if rank is not None else None
        )
        self.hc_count = config.hc_count

    def __call__(
        self,
        hidden: mx.array,
        rope: Qwen4RotaryEmbedding,
        position_ids: mx.array,
        mask: mx.array,
        cache: Any,
        ids: mx.array | None,
    ) -> mx.array:
        if self.ple is not None:
            hidden = hidden + self.ple(hidden, ids, cache)
        streams, hyper, inject = self.attn_hyper(hidden)
        if self.is_linear:
            out = self.linear_attn(streams, cache)
        else:
            out = self.self_attn(streams, rope, position_ids, mask, cache)
        hidden = _inject(hyper, out, inject, self.hc_count)

        streams, hyper, inject = self.mlp_hyper(hidden)
        hidden = _inject(hyper, self.mlp(streams), inject, self.hc_count)
        return hidden


# ------------------------------------------------------------------- the model


class Qwen4TextModel(nn.Module):
    """The `qwen4_exp_text` stack: embeddings, decoder layers, final mixer, head."""

    def __init__(self, config: Qwen4TextConfig, weights: WeightSource):
        super().__init__()
        self.config = config
        self.embed = weights.get(EMBED_NAME)
        self.lm_head = weights.get(LM_HEAD_NAME)
        self.rope = Qwen4RotaryEmbedding(config)
        self.layers = [
            Qwen4DecoderLayer(config, weights, index, f"{LAYER_PREFIX}.{index}")
            for index in range(config.num_hidden_layers)
        ]
        self.mixer = Qwen4GatedResidual(config, weights, MIXER_PREFIX, use_combine=False)
        self.ple_layers = bool(config.ple_layer_ids)
        self.eps = config.rms_norm_eps

    # --------------------------------------------------------------- caches

    def make_cache(self, batch: int = 1) -> list[Any]:
        """One cache entry per layer, every slot populated.

        Layout (documented in `docs/designs/qwen4-exp-engine.md`):

        - linear layer: `ArraysCache(size=2)`, `[0]` the qkv conv window,
          `[1]` the GDN recurrent state. With PLE on the layer the entry is
          `ArraysCache(size=4)` and `[2]`/`[3]` hold the PLE conv window and the
          PLE token context (the reference's `conv_states[1]`/`conv_states[2]`).
        - full layer: `CacheList(KVCache(), ArraysCache(size=1))`, the KV pair
          plus `[0]` of the ArraysCache holding the QSA indexer's token keys.

        Slots are initialised with zero arrays rather than left `None`:
        mlx-lm's `save_prompt_cache` cannot serialise an `ArraysCache` with an
        unset slot, so a lazily-filled slot would make the checkpoint
        unsaveable exactly once, on short prompts.
        """
        caches: list[Any] = []
        for layer in self.layers:
            if layer.is_linear:
                conv = mx.zeros(
                    (1, layer.linear_attn.conv_kernel - 1, layer.linear_attn.conv_dim),
                    dtype=mx.bfloat16,
                )
                recurrent = mx.zeros(
                    (
                        1,
                        layer.linear_attn.num_v_heads,
                        layer.linear_attn.head_k_dim,
                        layer.linear_attn.head_v_dim,
                    ),
                    dtype=mx.float32,
                )
                if layer.ple is None:
                    cache = ArraysCache(size=2)
                    cache[0], cache[1] = conv, recurrent
                else:
                    cache = ArraysCache(size=4)
                    cache[0], cache[1] = conv, recurrent
                    cache[2] = mx.zeros(
                        (1, layer.ple.state_len, self.config.hc_hidden_size), dtype=mx.bfloat16
                    )
                    cache[3] = mx.full(
                        (1, layer.ple.embedding.context_len),
                        -1,
                        dtype=mx.int32,
                    )
                caches.append(cache)
            else:
                index_keys = mx.zeros((1, 0, layer.self_attn.indexer.head_dim), dtype=mx.bfloat16)
                keys = ArraysCache(size=1)
                keys[0] = index_keys
                caches.append(CacheList(KVCache(), keys))
        return caches

    # ---------------------------------------------------------------- forward

    def __call__(
        self,
        input_ids: mx.array,
        cache: list[Any] | None = None,
        position_ids: mx.array | None = None,
    ) -> mx.array:
        """Next-token logits for `input_ids`, advancing `cache` in place."""
        ids = input_ids.astype(mx.int32)
        batch, seq_len = ids.shape
        if cache is None:
            cache = self.make_cache()
        if position_ids is None:
            offset = _cache_offset(cache)
            position_ids = mx.arange(offset, offset + seq_len)[None, None, :]
            position_ids = mx.broadcast_to(position_ids, (3, batch, seq_len))

        hidden = mx.take(self.embed, ids.reshape(-1), axis=0).reshape(batch, seq_len, -1)
        # Tile, not repeat: the hyper-connection mixer unflattens the width into
        # (hc_count, hidden_size), so stream `s` must occupy the `s`-th whole
        # block of `hidden` features. `mx.repeat` interleaves elements instead
        # (1,2,3 -> 1,1,1,2,2,2), which would scatter every stream across all
        # four and feed the mixer a permutation of the intended input.
        hidden = mx.tile(hidden, (1, 1, self.config.hc_count))

        mask = _causal_mask(seq_len, _cache_offset(cache) + seq_len)
        mask = mx.broadcast_to(mask[None, None], (batch, 1, seq_len, mask.shape[1]))
        for layer, layer_cache in zip(self.layers, cache, strict=True):
            hidden = layer(hidden, self.rope, position_ids, mask, layer_cache, ids)

        mixed = self.mixer(hidden)[0]
        return _linear(mixed, self.lm_head)

    # ------------------------------------------------------------- weight table

    def weight_shapes(self) -> dict[str, tuple[int, ...]]:
        return weight_shapes(self.config)


def _cache_offset(cache: list[Any]) -> int:
    """Tokens already represented by `cache`, read from the entry classes."""
    for entry in cache:
        if isinstance(entry, CacheList):
            return entry[0].offset
    for entry in cache:
        if isinstance(entry, ArraysCache) and entry[0] is not None:
            return 0
    return 0


def _attn_shapes(config: Qwen4TextConfig) -> dict[str, tuple[int, ...]]:
    n_h, n_kv, dim, hidden = (
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
        config.hidden_size,
    )
    indexer_prefix = "self_attn.indexer"
    shapes = {
        "self_attn.q_proj.weight": (n_h * dim * 2, hidden),
        "self_attn.k_proj.weight": (n_kv * dim, hidden),
        "self_attn.v_proj.weight": (n_kv * dim, hidden),
        "self_attn.o_proj.weight": (hidden, n_h * dim),
        "self_attn.q_norm.weight": (dim,),
        "self_attn.k_norm.weight": (dim,),
        f"{indexer_prefix}.index_qk_proj.weight": (
            (int(config.indexer_n_heads) + int(config.indexer_kv_heads))
            * int(config.indexer_head_dim),
            hidden,
        ),
        f"{indexer_prefix}.q_layernorm.weight": (int(config.indexer_head_dim),),
        f"{indexer_prefix}.k_layernorm.weight": (int(config.indexer_head_dim),),
    }
    if config.attention_bias:
        shapes["self_attn.q_proj.bias"] = (n_h * dim * 2,)
        shapes["self_attn.k_proj.bias"] = (n_kv * dim,)
        shapes["self_attn.v_proj.bias"] = (n_kv * dim,)
        shapes["self_attn.o_proj.bias"] = (hidden,)
    return shapes


def _linear_attn_shapes(config: Qwen4TextConfig) -> dict[str, tuple[int, ...]]:
    key_dim = config.linear_key_head_dim * config.linear_num_key_heads
    value_dim = config.linear_value_head_dim * config.linear_num_value_heads
    conv_dim = 2 * key_dim + value_dim
    hidden = config.hidden_size
    return {
        "linear_attn.in_proj_qkv.weight": (conv_dim, hidden),
        "linear_attn.in_proj_z.weight": (value_dim, hidden),
        "linear_attn.in_proj_b.weight": (config.linear_num_value_heads, hidden),
        "linear_attn.in_proj_a.weight": (config.linear_num_value_heads, hidden),
        "linear_attn.conv1d.weight": (conv_dim, 1, config.linear_conv_kernel_dim),
        "linear_attn.A_log": (config.linear_num_value_heads,),
        "linear_attn.dt_bias": (config.linear_num_value_heads,),
        "linear_attn.norm.weight": (config.linear_value_head_dim,),
        "linear_attn.out_proj.weight": (hidden, value_dim),
    }


def _mlp_shapes(config: Qwen4TextConfig) -> dict[str, tuple[int, ...]]:
    hidden = config.hidden_size
    inter = config.moe_intermediate_size
    shared = config.shared_expert_intermediate_size
    shapes = {
        "mlp.gate.weight": (config.num_experts, hidden),
        "mlp.shared_expert.gate_proj.weight": (shared, hidden),
        "mlp.shared_expert.up_proj.weight": (shared, hidden),
        "mlp.shared_expert.down_proj.weight": (hidden, shared),
        "mlp.shared_expert_gate.weight": (1, hidden),
    }
    for expert in range(config.num_experts):
        shapes[f"mlp.experts.{expert}.gate_proj.weight"] = (inter, hidden)
        shapes[f"mlp.experts.{expert}.up_proj.weight"] = (inter, hidden)
        shapes[f"mlp.experts.{expert}.down_proj.weight"] = (hidden, inter)
    return shapes


def _hyper_shapes(config: Qwen4TextConfig, combine: bool) -> dict[str, tuple[int, ...]]:
    """Hyper-connection weights, relative to the connection's own prefix.

    `combine=False` is the final stream mixer, which has no injection weights:
    nothing follows it to inject into.
    """
    width = config.hc_hidden_size
    shapes = {
        "hc_norm.weight": (width,),
        "input_mix_weight_down.weight": (config.hc_lowrank, width),
        "input_mix_weight_up.weight": (width, config.hc_lowrank),
    }
    if combine:
        shapes["block_inject_weight.weight"] = (config.hc_count, width)
    return shapes


def _ple_shapes(config: Qwen4TextConfig) -> dict[str, tuple[int, ...]]:
    """PLE projection/conv/norm weights, relative to the layer's `ple` prefix."""
    width = config.hc_hidden_size
    embed = int(config.ple_embed_dim)
    return {
        "key_proj.weight": (width, embed),
        "value_proj.weight": (config.hidden_size, embed),
        "conv1d.weight": (width, 1, config.ple_conv_kernel_size),
        "norm_key.weight": (width,),
        "norm_query.weight": (width,),
        "norm_conv.weight": (width,),
    }


def ple_weight_names(config: Qwen4TextConfig, prefix: str) -> list[str]:
    """Names of the n-gram table shards for one PLE layer, in order."""
    return [
        f"{prefix}.ple_embedding.ngram_embedding.shard_{index}.weight"
        for index in range(config.split_ngram_parts)
    ]


def weight_shapes(config: Qwen4TextConfig) -> dict[str, tuple[int, ...]]:
    """Every checkpoint name this model can request, with its stored shape.

    Stored shapes, not dequantized ones: an NVFP4 payload is two values per
    byte and an fp8 payload carries a scale tensor beside it, and the loader is
    what turns each of those into a dense matrix. Enumerating the stored layout
    is what lets the shape test check the model against the real index without
    reading a checkpoint.
    """
    hidden = config.hidden_size
    shapes: dict[str, tuple[int, ...]] = {
        EMBED_NAME: (config.vocab_size, hidden),
        LM_HEAD_NAME: (config.vocab_size, hidden),
    }
    shapes.update(
        {f"{MIXER_PREFIX}.{name}": shape for name, shape in _hyper_shapes(config, False).items()}
    )
    for index, layer_type in enumerate(config.layer_types):
        prefix = f"{LAYER_PREFIX}.{index}"
        attention = _linear_attn_shapes(config) if layer_type == LINEAR else _attn_shapes(config)
        for name, shape in attention.items():
            shapes[f"{prefix}.{name}"] = shape
        for name, shape in _mlp_shapes(config).items():
            shapes[f"{prefix}.{name}"] = shape
        for name, shape in _hyper_shapes(config, True).items():
            shapes[f"{prefix}.attn_hyper_connection.{name}"] = shape
            shapes[f"{prefix}.mlp_hyper_connection.{name}"] = shape
        rank = config.ple_rank(index)
        if rank is not None:
            for name, shape in _ple_shapes(config).items():
                shapes[f"{prefix}.ple.{name}"] = shape
            for name in ple_weight_names(config, f"{prefix}.ple"):
                shapes[name] = (config.ngram_rows_per_shard(rank), config.ngram_head_dim)
            shapes[f"{prefix}.ple.ple_embedding.layer_multipliers"] = (config.ngram_size,)
            shapes[f"{prefix}.ple.ple_embedding.ngram_heads_vocab_sizes"] = (config.ngram_heads,)
            shapes[f"{prefix}.ple.ple_embedding.ngram_heads_offsets"] = (config.ngram_heads,)
    return shapes


# NVFP4: two 4-bit values per stored byte, and a per-group scale every
# `group_size` inputs (ModelOpt layout).
NVFP4_GROUP = 16
NVFP4_PACKED_DIVISOR = 2


def stored_weight_shapes(config: Qwen4TextConfig) -> dict[str, tuple[int, ...]]:
    """The checkpoint's own tensor table: names and STORED shapes.

    `weight_shapes` answers "what does the model request", which is dense
    matrices because the loader dequantizes. This answers "what is on disk",
    which is what a header-only check against the real index can verify:
    NVFP4 experts are packed two values per byte with a group scale and two
    per-tensor scalars beside them, and each MoE layer carries one global scale
    per expert per projection.
    """
    stored: dict[str, tuple[int, ...]] = {}
    for name, shape in weight_shapes(config).items():
        if name in {EMBED_NAME, LM_HEAD_NAME}:
            stored[name] = shape
            continue
        if ".mlp.experts." in name and name.endswith(".weight"):
            rows, cols = shape
            stored[name] = (rows, cols // NVFP4_PACKED_DIVISOR)
            stored[name.replace(".weight", ".weight_scale")] = (rows, cols // NVFP4_GROUP)
            stored[name.replace(".weight", ".weight_scale_2")] = ()
            stored[name.replace(".weight", ".input_scale")] = ()
            continue
        if ".ple_embedding.ngram_embedding.shard_" in name:
            stored[name] = shape
            continue
        stored[name] = shape

    # Each sharded PLE table carries one per-tensor scale.
    for index in range(config.num_hidden_layers):
        prefix = f"{LAYER_PREFIX}.{index}"
        if config.ple_rank(index) is None:
            continue
        stored[f"{prefix}.ple.ple_embedding.ngram_embedding.weight_scale"] = (1,)

    # One global scale per expert, per routed projection, per MoE layer.
    for index in range(config.num_hidden_layers):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            stored[f"{LAYER_PREFIX}.{index}.mlp.switch_mlp.{projection}.global_scale"] = (
                config.num_experts,
            )
    return stored
