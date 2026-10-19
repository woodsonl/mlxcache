"""Qwen3.8-Flash-Next (``qwen4_exp``) vision tower, in MLX.

Forward semantics are ported from ``Qwen3_5MoeVisionModel`` (the base the
``Qwen4ExpVisionModel`` subclass inherits unchanged) plus the shared
``transformers.vision_utils`` precompute helpers it calls:

* patch embedding: ``Qwen3_5MoeVisionPatchEmbed`` -- a stride-``(t, p, p)``
  Conv3d with bias.
* position embedding: ``Qwen3_5MoeVisionModel.__init__`` sets
  ``interpolation_align_corners=True`` / ``interpolation_mode="bilinear"`` and
  resamples the learned ``(num_grid_per_side, num_grid_per_side)`` table with
  ``get_vision_interpolation_indices_and_weights``, i.e. a separable bilinear
  gather whose per-axis source coordinate is ``i * (side - 1) / (size - 1)``.
* rotary: ``Qwen3_5MoeVisionRotaryEmbedding`` -- *axial* rope over
  ``head_dim // 4`` frequencies, H and W sharing the same frequency table, then
  duplicated once to cover the full head dim.
* attention: ``Qwen3_5MoeVisionAttention`` with the eager/SDPA path's
  ``cu_seqlens`` chunking (``get_vision_cu_seqlens`` with
  ``merge_temporal=False``: one segment per frame, so nothing attends across
  images or across frames of a video).
* blocks: ``Qwen3_5MoeVisionBlock`` -- pre-norm residual, LayerNorm eps 1e-6.
* merger: ``Qwen3_5MoeVisionPatchMerger`` with ``use_postshuffle_norm=False``
  and its *exact* (erf) GELU, unlike the MLP's tanh-approximated GELU.

Numerics: activations travel in the dtype the weights arrived in (the
checkpoint is bf16), while every LayerNorm accumulates in float32 the way
torch's LayerNorm does for low-precision input. All computations are pure
functions of the inputs, so identical calls are bit-identical.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import mlx.core as mx
import mlx.nn as nn
import numpy as np

GELU_TANH = "gelu_pytorch_tanh"
GELU_ERF = "gelu"

# LayerNorm epsilon used by every norm in the vision tower (blocks and merger).
_LAYER_NORM_EPS = 1e-6


class WeightSource(Protocol):
    """Checkpoint reader keyed by full parameter name."""

    def get(self, name: str) -> mx.array: ...


def _square(value: int | Sequence[int]) -> int:
    """Accept the scalar or ``[p, p]`` spellings configs allow for patch sizes."""
    if isinstance(value, int):
        return value
    values = tuple(int(v) for v in value)
    if len(values) != 2 or values[0] != values[1]:
        raise ValueError(f"expected a square patch size, got {value!r}")
    return values[0]


@dataclass(frozen=True)
class Qwen4VisionConfig:
    depth: int = 27
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    num_position_embeddings: int = 2304
    out_hidden_size: int = 2560
    hidden_act: str = GELU_TANH
    # ``rope_parameters`` defaults to {"rope_type": "axial", "rope_theta": 10000.0}
    # for this config class; the checkpoint stores neither key.
    rope_theta: float = 10000.0

    @classmethod
    def from_vision_config(cls, d: dict) -> Qwen4VisionConfig:
        """Build from a checkpoint's ``vision_config`` mapping."""
        return cls(
            depth=int(d["depth"]),
            hidden_size=int(d["hidden_size"]),
            intermediate_size=int(d["intermediate_size"]),
            num_heads=int(d["num_heads"]),
            in_channels=int(d.get("in_channels", 3)),
            patch_size=_square(d["patch_size"]),
            spatial_merge_size=int(d.get("spatial_merge_size", 2)),
            temporal_patch_size=_square(d.get("temporal_patch_size", 2)),
            num_position_embeddings=int(d["num_position_embeddings"]),
            out_hidden_size=int(d["out_hidden_size"]),
            hidden_act=str(d.get("hidden_act", GELU_TANH)),
            rope_theta=float(d.get("rope_theta", 10000.0)),
        )

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def spatial_merge_unit(self) -> int:
        return self.spatial_merge_size**2

    @property
    def patch_dim(self) -> int:
        return self.in_channels * self.temporal_patch_size * self.patch_size**2

    @property
    def num_grid_per_side(self) -> int:
        """Side of the stored square position-embedding grid."""
        return int(self.num_position_embeddings**0.5)


def _gelu(x: mx.array, act: str) -> mx.array:
    if act == GELU_TANH:
        return nn.gelu_approx(x)
    if act == GELU_ERF:
        return nn.gelu(x)
    raise ValueError(f"unsupported vision hidden_act {act!r}")


def _layer_norm(x: mx.array, weight: mx.array, bias: mx.array, eps: float) -> mx.array:
    """LayerNorm with float32 accumulation, matching torch on bf16 input."""
    dtype = x.dtype
    xf = x.astype(mx.float32)
    mean = mx.mean(xf, axis=-1, keepdims=True)
    centred = xf - mean
    var = mx.mean(centred * centred, axis=-1, keepdims=True)
    normed = centred * mx.rsqrt(var + eps)
    return (normed * weight.astype(mx.float32) + bias.astype(mx.float32)).astype(dtype)


class _Linear(nn.Module):
    """``x @ weight.T + bias`` in a checkpoint's own ``(out, in)`` layout."""

    def __init__(self, weight: mx.array, bias: mx.array) -> None:
        super().__init__()
        self.weight = weight
        self.bias = bias

    def __call__(self, x: mx.array) -> mx.array:
        return x.astype(self.weight.dtype) @ self.weight.T + self.bias


class _Norm(nn.Module):
    def __init__(self, weight: mx.array, bias: mx.array) -> None:
        super().__init__()
        self.weight = weight
        self.bias = bias

    def __call__(self, x: mx.array) -> mx.array:
        return _layer_norm(x, self.weight, self.bias, _LAYER_NORM_EPS)


class _PatchEmbed(nn.Module):
    def __init__(self, config: Qwen4VisionConfig, weights: WeightSource) -> None:
        super().__init__()
        self.temporal_patch_size = config.temporal_patch_size
        self.patch_size = config.patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size
        # Checkpoints store Conv3d weights in torch's (out, in, kt, kh, kw);
        # MLX's conv3d wants (out, kt, kh, kw, in).
        stored = weights.get("model.visual.patch_embed.proj.weight")
        self.weight = mx.transpose(stored, (0, 2, 3, 4, 1))
        self.bias = weights.get("model.visual.patch_embed.proj.bias")
        self.stride = (self.temporal_patch_size, self.patch_size, self.patch_size)

    def __call__(self, pixel_values: mx.array) -> mx.array:
        n = pixel_values.shape[0]
        patches = pixel_values.astype(self.weight.dtype).reshape(
            n,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        # MLX conv3d takes channels last: (batch, kt, kh, kw, in).
        patches = mx.transpose(patches, (0, 2, 3, 4, 1))
        embedded = mx.conv3d(patches, self.weight, stride=self.stride) + self.bias
        return embedded.reshape(n, self.embed_dim)


class _PatchMerger(nn.Module):
    """2x2 patch merge: norm -> fc1 -> exact GELU -> fc2.

    The norm runs per unmerged token (``use_postshuffle_norm=False`` in the
    reference) and its activation is ``nn.GELU()`` -- the erf definition, *not*
    the ``gelu_pytorch_tanh`` the transformer MLP uses.
    """

    def __init__(self, config: Qwen4VisionConfig, weights: WeightSource) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size * config.spatial_merge_unit
        self.norm = _Norm(
            weights.get("model.visual.merger.norm.weight"),
            weights.get("model.visual.merger.norm.bias"),
        )
        self.linear_fc1 = _Linear(
            weights.get("model.visual.merger.linear_fc1.weight"),
            weights.get("model.visual.merger.linear_fc1.bias"),
        )
        self.linear_fc2 = _Linear(
            weights.get("model.visual.merger.linear_fc2.weight"),
            weights.get("model.visual.merger.linear_fc2.bias"),
        )

    def __call__(self, x: mx.array) -> mx.array:
        # Norm is applied per 1152-wide patch token, and only then are each
        # group of spatial_merge_unit consecutive tokens concatenated into one
        # 4608-wide merged token (the reference's `.view(-1, hidden_size)`
        # after a norm whose weight is hidden_size wide).
        normed = self.norm(x).reshape(-1, self.hidden_size)
        return self.linear_fc2(_gelu(self.linear_fc1(normed), GELU_ERF))


class _VisionMLP(nn.Module):
    def __init__(self, config: Qwen4VisionConfig, weights: WeightSource, prefix: str) -> None:
        super().__init__()
        self.act = config.hidden_act
        self.linear_fc1 = _Linear(
            weights.get(f"{prefix}.linear_fc1.weight"),
            weights.get(f"{prefix}.linear_fc1.bias"),
        )
        self.linear_fc2 = _Linear(
            weights.get(f"{prefix}.linear_fc2.weight"),
            weights.get(f"{prefix}.linear_fc2.bias"),
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear_fc2(_gelu(self.linear_fc1(x), self.act))


class _VisionAttention(nn.Module):
    def __init__(self, config: Qwen4VisionConfig, weights: WeightSource, prefix: str) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        self.qkv = _Linear(
            weights.get(f"{prefix}.attn.qkv.weight"),
            weights.get(f"{prefix}.attn.qkv.bias"),
        )
        self.proj = _Linear(
            weights.get(f"{prefix}.attn.proj.weight"),
            weights.get(f"{prefix}.attn.proj.bias"),
        )

    def __call__(
        self,
        x: mx.array,
        cos: mx.array,
        sin: mx.array,
        cu_seqlens: Sequence[int],
    ) -> mx.array:
        seq_len = x.shape[0]
        qkv = self.qkv(x).reshape(seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        q = _apply_rotary(q, cos, sin)
        k = _apply_rotary(k, cos, sin)
        # (heads, seq, head_dim) with a leading batch axis for SDPA.
        q = mx.expand_dims(q.transpose(1, 0, 2), 0)
        k = mx.expand_dims(k.transpose(1, 0, 2), 0)
        v = mx.expand_dims(v.transpose(1, 0, 2), 0)
        attn = _segment_attention(q, k, v, cu_seqlens, self.scale)
        # SDPA yields (1, heads, seq, head_dim); the output projection consumes
        # heads concatenated per token, so transpose heads back first.
        attn = attn.transpose(0, 2, 1, 3).reshape(seq_len, -1)
        return self.proj(attn)


def _apply_rotary(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    dtype = x.dtype
    xf = x.astype(mx.float32)
    cosf = mx.expand_dims(cos.astype(mx.float32), -2)
    sinf = mx.expand_dims(sin.astype(mx.float32), -2)
    return (xf * cosf + _rotate_half(xf) * sinf).astype(dtype)


def _rotate_half(x: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    return mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def _segment_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    cu_seqlens: Sequence[int],
    scale: float,
) -> mx.array:
    """Attend within each packed segment, never across them.

    The reference's non-flash path splits the packed sequence at ``cu_seqlens``
    and runs attention per chunk. Splitting is exactly the block-diagonal mask
    the packed layout implies, and because each chunk is computed with the same
    shapes whether or not siblings are batched alongside it, a multi-image call
    reproduces single-image calls bit for bit.
    """
    outs = []
    start = 0
    for length in cu_seqlens:
        end = start + length
        outs.append(
            mx.fast.scaled_dot_product_attention(
                q[:, :, start:end], k[:, :, start:end], v[:, :, start:end], scale=scale
            )
        )
        start = end
    return mx.concatenate(outs, axis=2)


class _VisionBlock(nn.Module):
    def __init__(self, config: Qwen4VisionConfig, weights: WeightSource, prefix: str) -> None:
        super().__init__()
        self.norm1 = _Norm(
            weights.get(f"{prefix}.norm1.weight"), weights.get(f"{prefix}.norm1.bias")
        )
        self.norm2 = _Norm(
            weights.get(f"{prefix}.norm2.weight"), weights.get(f"{prefix}.norm2.bias")
        )
        self.attn = _VisionAttention(config, weights, prefix)
        self.mlp = _VisionMLP(config, weights, f"{prefix}.mlp")

    def __call__(
        self, x: mx.array, cos: mx.array, sin: mx.array, cu_seqlens: Sequence[int]
    ) -> mx.array:
        x = x + self.attn(self.norm1(x), cos, sin, cu_seqlens)
        return x + self.mlp(self.norm2(x))


def _interpolation_taps(
    index: np.ndarray, size: np.ndarray, side: int
) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear taps into a ``side``-long table, ``align_corners=True``.

    Ported from ``vision_utils._interpolation_axis_taps_weights``: the source
    coordinate is ``index * (side - 1) / max(size - 1, 1)`` (the closed form of
    ``linspace(0, side - 1, size)[index]``), taps are clamped to the table, and
    weights are the linear hat kernel ``clamp(1 - |src - tap|, 0)``.
    """
    src = (
        index.astype(np.float32)
        * np.float32(side - 1)
        / np.maximum(size.astype(np.float32) - 1.0, 1.0)
    )
    floor = np.floor(src)
    offsets = np.arange(2, dtype=np.int64)
    raw_taps = floor.astype(np.int64)[:, None] + offsets[None, :]
    taps = np.clip(raw_taps, 0, side - 1)
    distance = np.abs(src[:, None] - floor[:, None] - offsets[None, :].astype(np.float32))
    weights = np.maximum(1.0 - distance, 0.0).astype(np.float32)
    return taps, weights


def _interpolation_indices_and_weights(
    grids: np.ndarray, side: int, merge: int
) -> tuple[np.ndarray, np.ndarray]:
    """Packed per-patch gather indices/weights for the learned position table.

    Ported from ``vision_utils.get_vision_interpolation_indices_and_weights``
    with ``mode="bilinear"``, ``align_corners=True``.
    """
    counts = grids[:, 0] * grids[:, 1] * grids[:, 2]
    total = int(counts.sum())
    heights = np.repeat(grids[:, 1], counts)
    widths = np.repeat(grids[:, 2], counts)
    starts = np.repeat(np.concatenate(([0], np.cumsum(counts)[:-1])), counts)
    within = (np.arange(total) - starts) % (heights * widths)
    # Decode into (row, col) in spatial-merge-block order, the inverse of the
    # merge reordering the tokens use.
    blocks_w = widths // merge
    in_col = within % merge
    in_row = (within // merge) % merge
    block_col = (within // (merge * merge)) % blocks_w
    block_row = within // (merge * merge * blocks_w)
    rows = block_row * merge + in_row
    cols = block_col * merge + in_col

    h_taps, h_weights = _interpolation_taps(rows, heights, side)
    w_taps, w_weights = _interpolation_taps(cols, widths, side)
    indices = (h_taps[:, :, None] * side + w_taps[:, None, :]).reshape(-1, 4)
    weights = (h_weights[:, :, None] * w_weights[:, None, :]).reshape(-1, 4)
    return indices, weights


def _vision_position_ids(grids: np.ndarray, merge: int) -> np.ndarray:
    """``(total_tokens, 2)`` (h, w) ids in merge-block order.

    Ported from ``vision_utils.get_vision_position_ids`` with
    ``include_temporal=False``.
    """
    out = []
    for grid_t, grid_h, grid_w in grids:
        hpos, wpos = np.meshgrid(np.arange(grid_h), np.arange(grid_w), indexing="ij")
        block_shape = (grid_h // merge, merge, grid_w // merge, merge)
        hpos = hpos.reshape(block_shape).transpose(0, 2, 1, 3).flatten()
        wpos = wpos.reshape(block_shape).transpose(0, 2, 1, 3).flatten()
        pairs = np.stack([hpos, wpos], axis=-1)
        out.append(np.tile(pairs, (int(grid_t), 1)))
    return np.concatenate(out, axis=0) if out else np.zeros((0, 2), dtype=np.int64)


def _frame_lengths(grids: np.ndarray) -> list[int]:
    """Per-frame attention segment lengths (``merge_temporal=False``)."""
    return [int(grid_h * grid_w) for grid_t, grid_h, grid_w in grids for _ in range(int(grid_t))]


class Qwen4VisionModel(nn.Module):
    """Vision tower returning merged features ``(num_merged_tokens, out_hidden_size)``."""

    def __init__(self, config: Qwen4VisionConfig, weights: WeightSource) -> None:
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.num_grid_per_side = config.num_grid_per_side
        self.patch_embed = _PatchEmbed(config, weights)
        self.pos_embed = weights.get("model.visual.pos_embed.weight")
        self.rotary_inv_freq = self._axial_inv_freq(config)
        self.blocks = [
            _VisionBlock(config, weights, f"model.visual.blocks.{layer}")
            for layer in range(config.depth)
        ]
        self.merger = _PatchMerger(config, weights)

    @staticmethod
    def _axial_inv_freq(config: Qwen4VisionConfig) -> mx.array:
        """``Qwen3_5MoeVisionRotaryEmbedding.compute_axial_rope_parameters``.

        ``head_dim // 4`` frequencies, stepped by two, shared by H and W.
        """
        spatial_dim = config.head_dim // 2
        exponents = mx.arange(0, spatial_dim, 2, dtype=mx.float32) / spatial_dim
        return 1.0 / (config.rope_theta**exponents)

    def _rotary(self, position_ids: np.ndarray) -> tuple[mx.array, mx.array]:
        """Cos/sin of the axial rope, recomposed to the full head dim.

        ``Qwen3_5MoeVisionRotaryEmbedding.forward`` +
        ``recomposition_frequencies``: ``cat([h, w], -1)`` then duplicated, so
        each head rotates (h, w, h, w) quarters. ``attention_scaling`` is 1.0.
        """
        freqs = mx.array(position_ids.astype(np.float32))[..., None] * self.rotary_inv_freq
        cos, sin = mx.cos(freqs), mx.sin(freqs)
        cos = mx.concatenate([cos[:, 0], cos[:, 1]], axis=-1)
        sin = mx.concatenate([sin[:, 0], sin[:, 1]], axis=-1)
        return mx.concatenate([cos, cos], axis=-1), mx.concatenate([sin, sin], axis=-1)

    def __call__(self, pixel_values: mx.array, grid_thw: mx.array) -> mx.array:
        config = self.config
        grids = np.array(grid_thw.tolist(), dtype=np.int64).reshape(-1, 3)
        if grids.size == 0:
            return mx.zeros((0, config.out_hidden_size), dtype=self.patch_embed.weight.dtype)
        total_patches = int((grids[:, 0] * grids[:, 1] * grids[:, 2]).sum())
        if pixel_values.shape[0] != total_patches:
            raise ValueError(
                f"pixel_values has {pixel_values.shape[0]} patches but grid_thw "
                f"implies {total_patches}"
            )
        if pixel_values.shape[1] != config.patch_dim:
            raise ValueError(
                f"pixel_values rows are {pixel_values.shape[1]}-wide, expected {config.patch_dim}"
            )

        hidden = self.patch_embed(pixel_values)

        interp_indices, interp_weights = _interpolation_indices_and_weights(
            grids, self.num_grid_per_side, self.spatial_merge_size
        )
        gathered = self.pos_embed[mx.array(interp_indices)]
        pos = (gathered * mx.array(interp_weights)[:, :, None]).sum(axis=1)
        hidden = hidden + pos.astype(hidden.dtype)

        position_ids = _vision_position_ids(grids, self.spatial_merge_size)
        cos, sin = self._rotary(position_ids)
        cu_seqlens = _frame_lengths(grids)

        for block in self.blocks:
            hidden = block(hidden, cos, sin, cu_seqlens)

        return self.merger(hidden)
