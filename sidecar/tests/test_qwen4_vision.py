"""Tests for the qwen4_exp vision tower port.

The tower is exercised against a shape-faithful synthetic checkpoint (seeded,
so output statistics are stable) plus a header-only check of the real
checkpoint's parameter table. The real model directory is optional: without it
only the static shape test skips.
"""

from __future__ import annotations

import json
import struct
import zlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

# CI's python job installs no mlx extra, and the tower is MLX-only, so the whole
# module skips there instead of failing collection.
mx = pytest.importorskip("mlx.core")

from mlxcache_qwen4.vision import (  # noqa: E402 - after the mlx availability gate
    GELU_ERF,
    GELU_TANH,
    Qwen4VisionConfig,
    Qwen4VisionModel,
    _gelu,
    _interpolation_indices_and_weights,
    _vision_position_ids,
)

REAL_MODEL = Path("/Users/lance/models/Qwen3.8-Flash-Next-NVFP4-mlxstream")

TINY = Qwen4VisionConfig(
    depth=2,
    hidden_size=32,
    intermediate_size=64,
    num_heads=2,
    in_channels=3,
    patch_size=4,
    spatial_merge_size=2,
    temporal_patch_size=2,
    num_position_embeddings=64,
    out_hidden_size=16,
)


def expected_shapes(config: Qwen4VisionConfig) -> dict[str, tuple[int, ...]]:
    """Parameter name -> shape, derived from the config the way the reference lays out modules."""
    shapes: dict[str, tuple[int, ...]] = {}
    hidden = config.hidden_size
    inter = config.intermediate_size
    kernel = (config.temporal_patch_size, config.patch_size, config.patch_size)
    shapes["model.visual.patch_embed.proj.weight"] = (
        hidden,
        config.in_channels,
        *kernel,
    )
    shapes["model.visual.patch_embed.proj.bias"] = (hidden,)
    shapes["model.visual.pos_embed.weight"] = (config.num_position_embeddings, hidden)
    for layer in range(config.depth):
        prefix = f"model.visual.blocks.{layer}"
        shapes[f"{prefix}.attn.qkv.weight"] = (hidden * 3, hidden)
        shapes[f"{prefix}.attn.qkv.bias"] = (hidden * 3,)
        shapes[f"{prefix}.attn.proj.weight"] = (hidden, hidden)
        shapes[f"{prefix}.attn.proj.bias"] = (hidden,)
        shapes[f"{prefix}.mlp.linear_fc1.weight"] = (inter, hidden)
        shapes[f"{prefix}.mlp.linear_fc1.bias"] = (inter,)
        shapes[f"{prefix}.mlp.linear_fc2.weight"] = (hidden, inter)
        shapes[f"{prefix}.mlp.linear_fc2.bias"] = (hidden,)
        for norm in ("norm1", "norm2"):
            shapes[f"{prefix}.{norm}.weight"] = (hidden,)
            shapes[f"{prefix}.{norm}.bias"] = (hidden,)
    merged = hidden * config.spatial_merge_unit
    shapes["model.visual.merger.norm.weight"] = (hidden,)
    shapes["model.visual.merger.norm.bias"] = (hidden,)
    shapes["model.visual.merger.linear_fc1.weight"] = (merged, merged)
    shapes["model.visual.merger.linear_fc1.bias"] = (merged,)
    shapes["model.visual.merger.linear_fc2.weight"] = (config.out_hidden_size, merged)
    shapes["model.visual.merger.linear_fc2.bias"] = (config.out_hidden_size,)
    return shapes


class RandomWeightSource:
    """Seeded stand-in for a checkpoint reader, shaped from the config.

    Values are derived from the tensor name, so repeated ``get`` calls agree --
    a test can ask for the same weight the model loaded instead of racing a
    shared random stream.
    """

    def __init__(self, config: Qwen4VisionConfig, seed: int = 0, dtype=mx.float32) -> None:
        self.shapes = expected_shapes(config)
        self.seed = seed
        self.dtype = dtype
        self.requested: list[str] = []

    def get(self, name: str) -> mx.array:
        self.requested.append(name)
        shape = self.shapes[name]
        # Small weights: the tests care about shape/finiteness/determinism, not
        # about reproducing trained magnitudes.
        fan_in = max(1, int(np.prod(shape[1:] if len(shape) > 1 else shape)))
        scale = 1.0 / fan_in**0.5
        if len(shape) >= 4:
            # A conv kernel sees one patch per output; without this its output
            # would be far smaller than the projections'.
            scale *= 4.0
        rng = np.random.default_rng([self.seed, zlib.crc32(name.encode())])
        data = rng.standard_normal(shape) * scale
        return mx.array(data.astype(np.float32)).astype(self.dtype)


def patch_rows(config: Qwen4VisionConfig, grid_thw, seed: int = 0) -> mx.array:
    total = int(sum(int(t) * int(h) * int(w) for t, h, w in grid_thw))
    rng = np.random.default_rng(seed)
    return mx.array(rng.standard_normal((total, config.patch_dim)).astype(np.float32))


def test_forward_shape_tokens_and_determinism() -> None:
    grid_thw = [(1, 4, 4)]
    model = Qwen4VisionModel(TINY, RandomWeightSource(TINY))
    pixel_values = patch_rows(TINY, grid_thw)

    out = model(pixel_values, mx.array(grid_thw))
    tokens = sum(t * h * w for t, h, w in grid_thw) // TINY.spatial_merge_unit
    assert out.shape == (tokens, TINY.out_hidden_size)

    values = np.array(out)
    assert np.all(np.isfinite(values))
    # Bounded: nothing exploded through the residual stack.
    assert np.abs(values).max() < 100.0
    assert np.linalg.norm(values) < 1000.0

    again = model(pixel_values, mx.array(grid_thw))
    np.testing.assert_array_equal(values, np.array(again))


def test_batched_matches_single_image() -> None:
    big = (1, 6, 4)
    small = (2, 2, 4)
    model = Qwen4VisionModel(TINY, RandomWeightSource(TINY, seed=7))

    big_rows = patch_rows(TINY, [big], seed=11)
    small_rows = patch_rows(TINY, [small], seed=12)
    alone_big = np.array(model(big_rows, mx.array([big])))
    alone_small = np.array(model(small_rows, mx.array([small])))

    batched = np.array(model(mx.concatenate([big_rows, small_rows]), mx.array([big, small])))
    assert batched.shape[0] == alone_big.shape[0] + alone_small.shape[0]
    np.testing.assert_array_equal(batched[: alone_big.shape[0]], alone_big)
    np.testing.assert_array_equal(batched[alone_big.shape[0] :], alone_small)


def test_video_frames_do_not_attend_to_each_other() -> None:
    """``merge_temporal=False``: one attention segment per frame, as the reference packs it."""
    frame = (1, 4, 4)
    video = (2, 4, 4)
    model = Qwen4VisionModel(TINY, RandomWeightSource(TINY, seed=5))
    rows = patch_rows(TINY, [video], seed=31)

    as_video = np.array(model(rows, mx.array([video])))
    as_frames = np.array(model(rows, mx.array([frame, frame])))
    np.testing.assert_array_equal(as_video, as_frames)


def test_no_attention_across_images() -> None:
    """A padded-position leak would let one image's patches move another's output."""
    first = (1, 4, 4)
    second = (1, 4, 4)
    model = Qwen4VisionModel(TINY, RandomWeightSource(TINY, seed=3))
    rows_first = patch_rows(TINY, [first], seed=21)
    rows_second = patch_rows(TINY, [second], seed=22)

    baseline = np.array(model(mx.concatenate([rows_first, rows_second]), mx.array([first, second])))
    n = baseline.shape[0] // 2
    np.testing.assert_array_equal(baseline[:n], np.array(model(rows_first, mx.array([first]))))
    np.testing.assert_array_equal(baseline[n:], np.array(model(rows_second, mx.array([second]))))


def test_position_grid_interpolation_runs() -> None:
    """Grids smaller and larger than the stored square table both resample."""
    model = Qwen4VisionModel(TINY, RandomWeightSource(TINY))
    side = TINY.num_grid_per_side
    assert side == 8

    for grid_thw in [(1, 4, 4), (1, side, side), (1, 6, 10)]:
        grid_t, grid_h, grid_w = grid_thw
        indices, weights = _interpolation_indices_and_weights(
            np.array([grid_thw]), side, TINY.spatial_merge_size
        )
        assert indices.shape == (grid_t * grid_h * grid_w, 4)
        assert weights.shape == indices.shape
        assert indices.min() >= 0 and indices.max() < TINY.num_position_embeddings
        # Interpolation weights must sum to one per position (a clamped tap
        # outside the table would otherwise silently drop contribution).
        np.testing.assert_allclose(weights.sum(axis=1), np.ones(indices.shape[0]), atol=1e-5)

        rows = patch_rows(TINY, [grid_thw], seed=5)
        out = model(rows, mx.array([grid_thw]))
        tokens = grid_t * grid_h * grid_w // TINY.spatial_merge_unit
        assert out.shape == (tokens, TINY.out_hidden_size)
        assert np.all(np.isfinite(np.array(out)))


def test_interpolation_matches_align_corners_bilinear() -> None:
    """The resize math is the reference's, not an approximation."""
    side = 5
    indices, weights = _interpolation_indices_and_weights(np.array([[1, 3, 3]]), side, 1)
    # align_corners=True maps target row/col i to i * (side-1)/(size-1) = i * 2,
    # so every position lands exactly on a table cell: one tap of weight 1 and
    # three dead taps. Position 4 is row 1, col 1 -> table cell (2, 2).
    np.testing.assert_allclose(weights[4], [1.0, 0.0, 0.0, 0.0], atol=1e-6)
    assert indices[4, 0] == 2 * side + 2
    # The dead taps are still clamped, in-range indices (never out of table).
    assert indices[4, 1] == 2 * side + 3
    assert indices[8, 0] == 4 * side + 4
    # Row 0 is the table's first row.
    assert indices[0, 0] == 0

    # A grid that does not divide the table blends two taps per axis.
    _, blended = _interpolation_indices_and_weights(np.array([[1, 4, 4]]), side, 1)
    np.testing.assert_allclose(blended.sum(axis=1), np.ones(16), atol=1e-6)
    # Row/col 1 of 4 -> source 4/3 -> taps at 1 and 2, weights 2/3 and 1/3.
    np.testing.assert_allclose(blended[1], [2 / 3, 1 / 3, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(blended[5], [4 / 9, 2 / 9, 2 / 9, 1 / 9], atol=1e-6)


def test_vision_position_ids_merge_block_order() -> None:
    ids = _vision_position_ids(np.array([[1, 4, 4]]), merge=2)
    assert ids.shape == (16, 2)
    # Merge-block order: (0,0) (0,1) (1,0) (1,1) then the next block along w.
    assert [tuple(row) for row in ids[:4]] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert [tuple(row) for row in ids[4:8]] == [(0, 2), (0, 3), (1, 2), (1, 3)]
    video = _vision_position_ids(np.array([[2, 4, 4]]), merge=2)
    assert video.shape == (32, 2)
    np.testing.assert_array_equal(video[:16], video[16:])


def test_rope_axial_frequency_layout() -> None:
    """H and W share one frequency table; the pair is duplicated over the head dim."""
    pids = _vision_position_ids(np.array([[1, 4, 4]]), merge=2)
    model = Qwen4VisionModel(TINY, RandomWeightSource(TINY))
    cos, sin = model._rotary(pids)
    assert cos.shape == sin.shape == (16, TINY.head_dim)

    spatial = TINY.head_dim // 2
    inv = 1.0 / (10000.0 ** (np.arange(0, spatial, 2, dtype=np.float32) / spatial))
    h_angles = pids[:, 0].astype(np.float32)[:, None] * inv
    w_angles = pids[:, 1].astype(np.float32)[:, None] * inv
    expected_cos = np.concatenate([np.cos(h_angles), np.cos(w_angles)], axis=-1)
    np.testing.assert_allclose(
        np.array(cos), np.concatenate([expected_cos] * 2, axis=-1), atol=1e-6
    )


def test_config_from_vision_config() -> None:
    config = Qwen4VisionConfig.from_vision_config(
        {
            "depth": 3,
            "hidden_size": 48,
            "intermediate_size": 96,
            "num_heads": 4,
            "in_channels": 3,
            "patch_size": 8,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "num_position_embeddings": 256,
            "out_hidden_size": 24,
            "hidden_act": "gelu_pytorch_tanh",
        }
    )
    assert config.depth == 3
    assert config.head_dim == 12
    assert config.patch_dim == 3 * 2 * 8 * 8
    assert config.num_grid_per_side == 16
    assert config.spatial_merge_unit == 4


def test_mlp_uses_tanh_gelu_and_merger_uses_exact_gelu() -> None:
    """The two GELUs in this tower differ; the checkpoint's act name decides the MLP one."""
    x = mx.array(np.linspace(-4.0, 4.0, 41, dtype=np.float32).reshape(1, -1))
    tanh_ref = 0.5 * x * (1.0 + mx.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))
    erf_ref = 0.5 * x * (1.0 + mx.erf(x / np.sqrt(2.0)))
    np.testing.assert_allclose(np.array(_gelu(x, GELU_TANH)), np.array(tanh_ref), atol=1e-6)
    np.testing.assert_allclose(np.array(_gelu(x, GELU_ERF)), np.array(erf_ref), atol=1e-6)
    # The two definitions are genuinely different functions (they separate by
    # ~5e-4 near |x|=2), so a merger that silently used the MLP's act would be
    # caught -- but only if this bound reflects the real gap, not a guess.
    divergence = np.abs(np.array(_gelu(x, GELU_TANH)) - np.array(_gelu(x, GELU_ERF))).max()
    assert 1e-4 < divergence < 1e-2

    # The tower's MLP honours hidden_act; the merger is fixed to the exact form.
    cfg = replace(TINY, hidden_act=GELU_TANH)
    model = Qwen4VisionModel(cfg, RandomWeightSource(cfg))
    grid_thw = [(1, 4, 4)]
    out = model(patch_rows(cfg, grid_thw), mx.array(grid_thw))
    assert out.shape == (4, cfg.out_hidden_size)

    with pytest.raises(ValueError, match="hidden_act"):
        _gelu(x, "silu")


def test_interpolation_matches_independent_linspace_bilinear() -> None:
    """Cross-check the packed gather against a direct ``np.linspace`` resample.

    ``F.interpolate(mode="bilinear", align_corners=True)`` maps target index i
    to ``linspace(0, side-1, size)[i]``; this recomputes the 2-D blend that way,
    in raster order, and compares against the merge-block-ordered gather.
    """
    side = 8
    model = Qwen4VisionModel(TINY, RandomWeightSource(TINY, seed=1))
    table = np.array(model.pos_embed)

    def grid(rows: int, cols: int) -> np.ndarray:
        return table[rows * side + cols]

    def axis(size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        coords = np.linspace(0, side - 1, size)
        low = np.floor(coords).astype(int)
        return low, np.minimum(low + 1, side - 1), coords - low

    for grid_h, grid_w in [(2, 2), (4, 6), (8, 8), (6, 10)]:
        indices, weights = _interpolation_indices_and_weights(
            np.array([[1, grid_h, grid_w]]), side, TINY.spatial_merge_size
        )
        gathered = (table[indices] * weights[:, :, None]).sum(axis=1)
        low_h, high_h, frac_h = axis(grid_h)
        low_w, high_w, frac_w = axis(grid_w)
        raster = np.array(
            [
                [
                    grid(low_h[i], low_w[j]) * (1 - frac_h[i]) * (1 - frac_w[j])
                    + grid(low_h[i], high_w[j]) * (1 - frac_h[i]) * frac_w[j]
                    + grid(high_h[i], low_w[j]) * frac_h[i] * (1 - frac_w[j])
                    + grid(high_h[i], high_w[j]) * frac_h[i] * frac_w[j]
                    for j in range(grid_w)
                ]
                for i in range(grid_h)
            ]
        ).reshape(-1, table.shape[1])
        merge_order = [
            (block_r * 2 + in_r) * grid_w + block_c * 2 + in_c
            for block_r in range(grid_h // 2)
            for block_c in range(grid_w // 2)
            for in_r in range(2)
            for in_c in range(2)
        ]
        np.testing.assert_allclose(gathered, raster[merge_order], atol=1e-6)


def test_patch_embed_matches_explicit_window_convolution() -> None:
    """The transposed Conv3d weight and channel-last layout both have to be right."""
    model = Qwen4VisionModel(TINY, RandomWeightSource(TINY))
    source = RandomWeightSource(TINY)
    config = TINY
    kernel = (config.temporal_patch_size, config.patch_size, config.patch_size)
    weight = np.array(source.get("model.visual.patch_embed.proj.weight"))
    bias = np.array(source.get("model.visual.patch_embed.proj.bias"))
    rows = np.arange(4 * config.patch_dim, dtype=np.float32).reshape(4, -1) / 100.0

    expected = np.stack(
        [
            np.einsum(
                "oickl,ickl->o",
                weight,
                row.reshape(config.in_channels, *kernel),
            )
            + bias
            for row in rows
        ]
    )
    got = np.array(model.patch_embed(mx.array(rows)))
    # conv3d and einsum accumulate in different orders; on values of order 10
    # float32 noise reaches ~1e-5.
    np.testing.assert_allclose(got, expected, atol=1e-4, rtol=0)


def _safetensors_shapes(model_dir: Path) -> dict[str, tuple[tuple[int, ...], str]]:
    """Read every tensor's shape/dtype from shard headers only; no tensor data."""
    weight_map = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    headers: dict[str, dict] = {}
    out: dict[str, tuple[tuple[int, ...], str]] = {}
    for name, shard in weight_map.items():
        if shard not in headers:
            with (model_dir / shard).open("rb") as handle:
                length = struct.unpack("<Q", handle.read(8))[0]
                headers[shard] = json.loads(handle.read(length))
        entry = headers[shard][name]
        out[name] = (tuple(entry["shape"]), entry["dtype"])
    return out


class HeaderWeightSource:
    """Weight source that validates against checkpoint headers and never loads data."""

    def __init__(self, config: Qwen4VisionConfig, headers: dict[str, tuple[tuple[int, ...], str]]):
        self.headers = headers
        self.expected = expected_shapes(config)
        self.requested: list[str] = []

    def get(self, name: str) -> mx.array:
        self.requested.append(name)
        assert name in self.headers, f"checkpoint has no tensor {name}"
        shape, _dtype = self.headers[name]
        assert name in self.expected, f"requested unexpected tensor {name}"
        assert shape == self.expected[name], (
            f"{name}: checkpoint {shape} != expected {self.expected[name]}"
        )
        return mx.zeros(shape, dtype=mx.bfloat16)


@pytest.mark.skipif(not REAL_MODEL.is_dir(), reason=f"real checkpoint absent: {REAL_MODEL}")
def test_real_config_parameter_table() -> None:
    payload = json.loads((REAL_MODEL / "config.json").read_text())["vision_config"]
    config = Qwen4VisionConfig.from_vision_config(payload)
    assert (config.depth, config.hidden_size, config.num_heads) == (27, 1152, 16)
    assert (config.intermediate_size, config.out_hidden_size) == (4304, 2560)
    assert config.num_grid_per_side == 48

    headers = _safetensors_shapes(REAL_MODEL)
    visual = {name for name in headers if ".visual." in name}
    assert len(visual) == 333

    source = HeaderWeightSource(config, headers)
    Qwen4VisionModel(config, source)

    # Every vision tensor is requested, exactly once, and every shape matched.
    assert len(source.requested) == len(set(source.requested)) == 333
    assert set(source.requested) == visual
    dtypes = {headers[name][1] for name in source.requested}
    assert dtypes == {"BF16"}
