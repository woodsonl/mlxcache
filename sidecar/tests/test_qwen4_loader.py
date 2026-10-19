"""B3: streaming checkpoint loader.

Two halves, by design:

* Hermetic tests build a tiny synthetic safetensors checkpoint in ``tmp_path``
  from tensors produced by ``mx.quantize(..., mode="nvfp4")`` / ``mx.to_fp8``,
  so the dequantisation paths are exercised against genuine quantised payloads
  with known ground truth.
* Real-checkpoint tests run only where the 132.6 GB Qwen3.8 checkpoint is
  present, and skip cleanly everywhere else (CI never has it).

mlx is an optional extra (``uv sync --group dev`` alone does not install it),
so every mlx import lives inside a test or fixture body and the module is
guarded by ``importorskip("mlx.core")``. Importing
``mlxcache_qwen4.loader`` itself imports mlx at module scope, so that import is
also inside the guards: collection must succeed with mlx absent.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("numpy")

import numpy as np  # noqa: E402

CHECKPOINT = Path("/Users/lance/models/Qwen3.8-Flash-Next-NVFP4-mlxstream")


def _require_real_checkpoint() -> Path:
    if not (CHECKPOINT / "model.safetensors.index.json").is_file():
        pytest.skip(f"real checkpoint absent: {CHECKPOINT}")
    return CHECKPOINT


# ---------------------------------------------------------------------------
# synthetic checkpoint helpers (hermetic)
# ---------------------------------------------------------------------------
def _write_safetensors(path: Path, tensors: dict[str, tuple[str, tuple, bytes]]) -> None:
    """Write a minimal safetensors file from raw little-endian payloads.

    Written by hand rather than through ``safetensors.numpy.save_file`` because
    this environment's numpy cannot represent BF16 or F8_E4M3, and the fixture
    must be able to store exactly the bytes the real checkpoint stores.
    """
    header: dict[str, object] = {"__metadata__": {"format": "pt"}}
    blobs: list[bytes] = []
    offset = 0
    for name, (dtype, shape, buf) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + len(buf)],
        }
        blobs.append(buf)
        offset += len(buf)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    with path.open("wb") as fh:
        fh.write(struct.pack("<Q", len(encoded)))
        fh.write(encoded)
        for blob in blobs:
            fh.write(blob)


def _u8_bytes(array) -> bytes:
    return np.asarray(array).tobytes()


def _bf16_bytes(value) -> bytes:
    """Little-endian bytes of a bf16 mlx array (exported as its uint16 view)."""
    return np.asarray(value.view(__import__("mlx.core", fromlist=["core"]).uint16)).tobytes()


def _f32_bytes(value: float) -> bytes:
    return np.asarray(np.array([value], dtype=np.float32)).tobytes()


def _build_checkpoint(root: Path) -> dict[str, object]:
    """Build a miniature checkpoint with the real checkpoint's naming scheme."""
    import mlx.core as mx

    mx.random.seed(7)
    root.mkdir(parents=True, exist_ok=True)
    out: dict[str, tuple[str, tuple, bytes]] = {}

    layer = 0
    prefix = f"model.language_model.layers.{layer}"
    # ground truth for the nvfp4 round-trip assertion
    truth: dict[str, object] = {}

    # Two experts per projection, gate [32,64], up [32,64], down [64,32].
    global_scales = {proj: [] for proj in ("gate_proj", "up_proj", "down_proj")}
    for expert in (0, 1):
        for proj, shape in (
            ("gate_proj", (32, 64)),
            ("up_proj", (32, 64)),
            ("down_proj", (64, 32)),
        ):
            weight = (mx.random.normal(shape) * 0.02).astype(mx.bfloat16)
            global_scale = mx.array([0.25 + 0.05 * expert], dtype=mx.float32)
            packed, scales = mx.quantize(
                weight, group_size=16, bits=4, mode="nvfp4", global_scale=global_scale
            )
            base = f"{prefix}.mlp.experts.{expert}.{proj}"
            out[f"{base}.weight"] = (
                "U8",
                packed.view(mx.uint8).shape,
                _u8_bytes(packed.view(mx.uint8)),
            )
            out[f"{base}.weight_scale"] = ("F8_E4M3", scales.shape, _u8_bytes(scales))
            out[f"{base}.weight_scale_2"] = ("F32", (), _f32_bytes(float(global_scale) / (6 * 448)))
            out[f"{base}.input_scale"] = ("F32", (), _f32_bytes(0.001))
            global_scales[proj].append(float(global_scale))
            truth[f"{base}.weight"] = weight
    for proj, values in global_scales.items():
        payload = np.array(values, dtype=np.float32).tobytes()
        out[f"{prefix}.mlp.switch_mlp.{proj}.global_scale"] = ("F32", (len(values),), payload)

    # BF16 resident tensors: a shared expert and an embedding-like table.
    shared = (mx.random.normal((32, 64)) * 0.02).astype(mx.bfloat16)
    out[f"{prefix}.mlp.shared_expert.gate_proj.weight"] = (
        "BF16",
        shared.shape,
        _bf16_bytes(shared),
    )
    embedding = (mx.random.normal((48, 16)) * 0.05).astype(mx.bfloat16)
    out["model.language_model.embed_tokens.weight"] = (
        "BF16",
        embedding.shape,
        _bf16_bytes(embedding),
    )
    truth["model.language_model.embed_tokens.weight"] = embedding
    truth[f"{prefix}.mlp.shared_expert.gate_proj.weight"] = shared

    # FP8 expert weights with [8, 8]-blocked scale_inv. The loader dispatches
    # dequantization on stored dtype, so this path must hold for any future
    # fp8 expert family, not just the one the current checkpoint ships.
    # The payload is stored normalised so each block's amax hits the e4m3
    # maximum (448) and scale_inv carries the magnitude, which is what the real
    # blocks do -- measured amax 448.0 for every block under that layout.
    for proj, shape in (("gate_proj", (32, 64)), ("down_proj", (64, 32))):
        weight = (mx.random.normal(shape) * 0.02).astype(mx.float32)
        block_rows, block_cols = 8, 8
        blocks = weight.reshape(
            shape[0] // block_rows, block_rows, shape[1] // block_cols, block_cols
        )
        scale_inv = (mx.abs(blocks).max(axis=(1, 3)).astype(mx.float32) / 448.0).astype(mx.bfloat16)
        wide = mx.repeat(
            mx.repeat(scale_inv.astype(mx.float32), block_rows, axis=0), block_cols, axis=1
        )
        fp8 = mx.to_fp8(weight / wide).view(mx.uint8)
        name = f"model.language_model.layers.3.mlp.experts.0.{proj}.weight"
        out[name] = ("F8_E4M3", fp8.shape, _u8_bytes(fp8))
        out[name.replace(".weight", ".weight_scale_inv")] = (
            "BF16",
            scale_inv.shape,
            _bf16_bytes(scale_inv),
        )
        truth[name] = weight
        truth["__block__" + name] = (block_rows, block_cols, scale_inv)

    # PLE: two tiny n-gram shards of F8 rows plus the shared BF16 row scale.
    # Stored rows are value/scale so that value = from_fp8(row) * weight_scale,
    # matching the real table where the scalar carries the magnitude.
    ple_scale = 1.992e-4
    ple_rows = 6
    ple_prefix = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
    for shard in (0, 1):
        rows = (mx.random.normal((ple_rows, 8)) * 0.05).astype(mx.float32)
        fp8 = mx.to_fp8(rows / ple_scale).view(mx.uint8)
        out[f"{ple_prefix}.shard_{shard}.weight"] = ("F8_E4M3", fp8.shape, _u8_bytes(fp8))
        truth[f"{ple_prefix}.shard_{shard}.weight"] = rows
    out[ple_prefix + ".weight_scale"] = (
        "BF16",
        (1,),
        _bf16_bytes(mx.array([ple_scale], dtype=mx.bfloat16)),
    )

    # A resident 2-D table larger than a deliberately tiny ceiling, used to
    # prove oversized whole-tensor reads are refused.
    big = (mx.random.normal((256, 64)) * 0.01).astype(mx.bfloat16)
    out["vision.big_table.weight"] = ("BF16", big.shape, _bf16_bytes(big))

    _write_safetensors(root / "model.safetensors", out)

    index = {"metadata": {}, "weight_map": dict.fromkeys(out, "model.safetensors")}
    (root / "model.safetensors.index.json").write_text(json.dumps(index))

    rows_per_shard = ple_rows
    manifest = {
        "layout": "fp8_e4m3_safetensors_ranges",
        "source_root": ".",
        "row_width": 8,
        "row_count": rows_per_shard * 2,
        "scale_bf16_bits": struct.unpack("<H", _bf16_bytes(mx.array([ple_scale], mx.bfloat16)))[0],
        "shards": [
            {
                "row_start": shard * rows_per_shard,
                "row_count": rows_per_shard,
                "file": "model.safetensors",
                "offset": _data_offset(
                    root / "model.safetensors", f"{ple_prefix}.shard_{shard}.weight"
                ),
            }
            for shard in (0, 1)
        ],
    }
    (root / "ple_fp8_manifest.json").write_text(json.dumps(manifest))
    return truth


def _data_offset(path: Path, name: str) -> int:
    with path.open("rb") as fh:
        (header_len,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(header_len))
    return 8 + header_len + header[name]["data_offsets"][0]


@pytest.fixture
def synthetic(tmp_path):
    truth = _build_checkpoint(tmp_path / "checkpoint")
    return tmp_path / "checkpoint", truth


def _loader(root, **kwargs):
    from mlxcache_qwen4.loader import CheckpointLoader

    return CheckpointLoader(root, **kwargs)


# ---------------------------------------------------------------------------
# hermetic: nvfp4 dequantisation
# ---------------------------------------------------------------------------
def test_nvfp4_roundtrip_within_tolerance(synthetic):
    """Dequantising the packed expert reproduces the original bf16 weights."""
    import mlx.core as mx

    root, truth = synthetic
    loader = _loader(root, expert_resident_n=2)
    name = "model.language_model.layers.0.mlp.experts.0.gate_proj.weight"
    got = loader.get(name).astype(mx.float32)
    want = truth[name].astype(mx.float32)
    assert got.shape == want.shape
    # 4-bit e2m1 with 16-element groups: relative RMS error around 10% is the
    # expected quantisation floor, not a defect.
    rel = float(mx.sqrt(mx.mean((got - want) ** 2)) / mx.sqrt(mx.mean(want**2)))
    assert rel < 0.2, f"nvfp4 round-trip relative RMS error {rel:.4f} too large"
    # And the magnitude is right, which is what a wrong global_scale breaks by
    # orders of magnitude (measured 20551x on the real checkpoint).
    ratio = float(mx.mean(mx.abs(got)) / mx.mean(mx.abs(want)))
    assert 0.6 < ratio < 1.6, f"mean|dequantised|/mean|truth| = {ratio:.4g}"
    loader.close()


def test_nvfp4_global_scale_is_not_weight_scale_2(synthetic):
    """Use the per-expert table entry, not weight_scale_2 (2688x smaller)."""
    import mlx.core as mx

    root, truth = synthetic
    loader = _loader(root)
    name = "model.language_model.layers.0.mlp.experts.0.gate_proj.weight"
    got = loader.get(name).astype(mx.float32)
    want = truth[name].astype(mx.float32)

    packed, _header = loader.get_raw(name)
    scale_name = name.replace(".weight", ".weight_scale")
    scales, _ = loader.get_raw(scale_name)
    ws2, _ = loader.get_raw(name.replace(".weight", ".weight_scale_2"))
    wrong = mx.dequantize(
        packed.view(mx.uint32).reshape(packed.shape[0], packed.shape[1] // 4),
        scales=scales,
        group_size=16,
        bits=4,
        mode="nvfp4",
        global_scale=ws2.reshape(()),
        dtype=mx.bfloat16,
    ).astype(mx.float32)
    good_ratio = float(mx.mean(mx.abs(got)) / mx.mean(mx.abs(want)))
    wrong_ratio = float(mx.mean(mx.abs(wrong)) / mx.mean(mx.abs(want)))
    assert 0.6 < good_ratio < 1.6
    assert wrong_ratio < 0.01, f"weight_scale_2 alone gave ratio {wrong_ratio:.3g}; not a detector"
    loader.close()


def test_nvfp4_global_scale_identity(synthetic):
    """6 * 448 * weight_scale_2 == the per-expert global scale, as measured."""

    root, _truth = synthetic
    loader = _loader(root)
    for proj in ("gate_proj", "up_proj", "down_proj"):
        for expert in (0, 1):
            ws2, _ = loader.get_raw(
                f"model.language_model.layers.0.mlp.experts.{expert}.{proj}.weight_scale_2"
            )
            table, _ = loader.get_raw(
                f"model.language_model.layers.0.mlp.switch_mlp.{proj}.global_scale"
            )
            implied = float(ws2.reshape(())) * 6 * 448
            assert implied == pytest.approx(float(table[expert]), rel=1e-6)
    loader.close()


# ---------------------------------------------------------------------------
# hermetic: slices, byte accounting, errors
# ---------------------------------------------------------------------------
def test_get_slice_matches_full_tensor_rows(synthetic):
    import mlx.core as mx

    root, _truth = synthetic
    loader = _loader(root)
    for name in (
        "model.language_model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.language_model.layers.0.mlp.experts.1.down_proj.weight",
        "model.language_model.layers.3.mlp.experts.0.gate_proj.weight",
    ):
        full = loader.get(name)
        sliced = loader.get_slice(name, slice(3, 9))
        assert sliced.shape[0] == 6
        assert bool(mx.all(sliced == full[3:9])), f"{name}: slice disagrees with full tensor"
        strided = loader.get_slice(name, slice(0, 12, 3))
        assert bool(mx.all(strided == full[0:12:3])), f"{name}: strided slice disagrees"
    loader.close()


def test_ple_slice_reads_only_requested_rows(synthetic):
    root, _truth = synthetic
    loader = _loader(root, ple_cache_mb=1)
    name = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight"
    rows = 4
    got = loader.get_slice(name, slice(1, 1 + rows))
    stats = loader.stats()
    assert got.shape == (rows, 8)
    assert stats.ple_rows_loaded == rows
    # 8 columns of fp8 per row: exactly rows*8 bytes, not the whole shard.
    assert stats.ple_bytes_loaded == rows * 8
    # A second identical read is a cache hit and transfers nothing new.
    loader.get_slice(name, slice(1, 1 + rows))
    assert loader.stats().ple_hits == 1
    assert loader.stats().ple_bytes_loaded == rows * 8
    loader.close()


def test_expert_read_transfers_far_less_than_shard(synthetic):
    root, _truth = synthetic
    loader = _loader(root, expert_resident_n=0)
    name = "model.language_model.layers.0.mlp.experts.0.gate_proj.weight"
    loader.get(name)
    stats = loader.stats()
    shard_bytes = (root / "model.safetensors").stat().st_size
    loaded = stats.expert_bytes_loaded
    assert loaded > 0
    assert loaded < shard_bytes / 4, (
        f"reading one expert transferred {loaded} of {shard_bytes} shard bytes"
    )
    assert stats.expert_misses == 1
    loader.close()


def test_lru_evicts_by_bytes(synthetic):
    import mlx.core as mx

    root, _truth = synthetic
    # Each fixture expert decodes to three bf16 matrices of 32*64*2 = 4096
    # bytes (gate, up, down), so one expert is 12288 bytes and N=1 provisions
    # exactly one expert: fetching a second expert's matrix must evict the
    # first expert's.
    loader = _loader(root, expert_resident_n=1)
    per_expert = 3 * 4096
    assert loader._experts.capacity == per_expert
    assert loader._experts.capacity % 3 == 0, "an expert is three matrices, not one"
    prefix = "model.language_model.layers.0.mlp.experts"
    matrices = [
        f"{prefix}.{expert}.{proj}.weight"
        for expert in (0, 1)
        for proj in ("gate_proj", "up_proj", "down_proj")
    ]
    # One expert's three matrices fill the budget exactly.
    for name in matrices[:3]:
        loader.get(name)
    assert loader._experts.nbytes == per_expert
    assert loader.stats().expert_misses == 3
    # Any matrix of the next expert evicts one of the first expert's, so the
    # first expert cannot be held whole alongside the second -- which is what
    # "N experts" has to mean.
    loader.get(matrices[3])
    assert loader.stats().expert_misses == 4
    assert loader._experts.nbytes <= loader._experts.capacity
    miss_before = loader.stats().expert_misses
    # The entry fetched last survives: a hit needs no read at all.
    assert isinstance(loader.get(matrices[3]), mx.array)
    assert loader.stats().expert_misses == miss_before
    assert loader.stats().expert_hits == 1
    # ...while the expert evicted to make room for it does not.
    assert isinstance(loader.get(matrices[0]), mx.array)
    assert loader.stats().expert_misses == miss_before + 1
    assert loader._experts.nbytes <= loader._experts.capacity
    loader.close()


def test_lru_disabled_never_caches(synthetic):
    root, _truth = synthetic
    loader = _loader(root, expert_resident_n=0)
    name = "model.language_model.layers.0.mlp.experts.0.gate_proj.weight"
    loader.get(name)
    loader.get(name)
    stats = loader.stats()
    assert stats.expert_misses == 2
    assert stats.expert_hits == 0
    loader.close()


def test_unknown_name_errors_clearly(synthetic):
    root, _truth = synthetic
    loader = _loader(root)
    with pytest.raises(KeyError) as excinfo:
        loader.get("model.language_model.layers.0.mlp.experts.0.bogus_proj.weight")
    message = str(excinfo.value)
    assert "not a tensor" in message
    assert "refusing to guess" in message
    loader.close()


def test_oversized_tensor_is_refused_not_loaded(synthetic):
    root, _truth = synthetic
    # Ceiling below the 256x64 bf16 table (32768 bytes decoded).
    loader = _loader(root, resident_max_bytes=16 * 1024)
    with pytest.raises(ValueError) as excinfo:
        loader.get("vision.big_table.weight")
    message = str(excinfo.value)
    assert "resident ceiling" in message
    assert "get_slice" in message
    loader.close()


def test_ple_table_cannot_be_fetched_whole(synthetic):
    root, _truth = synthetic
    loader = _loader(root)
    name = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight"
    with pytest.raises(ValueError) as excinfo:
        loader.get(name)
    assert "PLE n-gram table" in str(excinfo.value)
    assert "get_slice" in str(excinfo.value)
    loader.close()


def test_classify_names(synthetic):
    from mlxcache_qwen4.loader import classify

    assert classify("model.language_model.layers.0.mlp.experts.0.gate_proj.weight") == "expert"
    assert (
        classify("model.language_model.layers.3.mlp.experts.7.down_proj.weight_scale_inv")
        == "expert"
    )
    assert (
        classify("model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_5.weight")
        == "ple"
    )
    assert classify("model.language_model.embed_tokens.weight") == "resident"
    assert classify("lm_head.weight") == "resident"
    assert classify("model.language_model.layers.0.self_attn.q_proj.weight") == "resident"
    assert classify("model.language_model.layers.0.linear_attn.A_log") == "resident"
    assert classify("model.visual.blocks.0.attn.qkv.weight") == "resident"
    assert classify("model.language_model.layers.0.mlp.gate.weight") == "resident"
    # A PLE scalar (the BF16 row scale) is resident: the PLE path needs it.
    assert (
        classify("model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale")
        == "resident"
    )


def test_stats_start_at_zero(synthetic):
    root, _truth = synthetic
    loader = _loader(root)
    stats = loader.stats()
    assert stats.expert_bytes_loaded == 0
    assert stats.expert_hits == 0
    assert stats.expert_misses == 0
    assert stats.ple_rows_loaded == 0
    assert stats.ple_bytes_loaded == 0
    assert stats.ple_hits == 0
    assert stats.ple_misses == 0
    loader.close()


def test_env_knobs_read_once_at_construction(synthetic, monkeypatch):
    root, _truth = synthetic
    monkeypatch.setenv("MLXCACHE_Q4_EXPERT_RESIDENT_N", "3")
    monkeypatch.setenv("MLXCACHE_Q4_PLE_CACHE_MB", "2")
    loader = _loader(root)
    assert loader.expert_resident_n == 3
    assert loader.ple_cache_mb == 2.0
    # Changing the environment afterwards must not move the knob.
    monkeypatch.setenv("MLXCACHE_Q4_EXPERT_RESIDENT_N", "0")
    monkeypatch.setenv("MLXCACHE_Q4_PLE_CACHE_MB", "0")
    assert loader.expert_resident_n == 3
    assert loader.ple_cache_mb == 2.0
    loader.close()


def test_constructor_kwargs_override_env(synthetic, monkeypatch):
    root, _truth = synthetic
    monkeypatch.setenv("MLXCACHE_Q4_EXPERT_RESIDENT_N", "5")
    monkeypatch.setenv("MLXCACHE_Q4_PLE_CACHE_MB", "64")
    loader = _loader(root, expert_resident_n=1, ple_cache_mb=0.5)
    assert loader.expert_resident_n == 1
    assert loader.ple_cache_mb == 0.5
    loader.close()


def test_missing_checkpoint_dir_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        _loader(tmp_path / "absent")


def test_resident_get_is_cached(synthetic):
    root, _truth = synthetic
    loader = _loader(root)
    name = "model.language_model.embed_tokens.weight"
    first = loader.get(name)
    second = loader.get(name)
    assert first is second
    loader.close()


def test_fp8_block_scale_layout(synthetic):
    """Block scale_inv tiles the weight as stored, not transposed."""
    import mlx.core as mx

    root, truth = synthetic
    loader = _loader(root)
    name = "model.language_model.layers.3.mlp.experts.0.gate_proj.weight"
    got = loader.get(name).astype(mx.float32)
    want = truth[name]
    rel = float(mx.sqrt(mx.mean((got - want) ** 2)) / mx.sqrt(mx.mean(want**2)))
    assert rel < 0.1, f"fp8 block dequant relative RMS {rel:.4f}"
    loader.close()


def test_ple_dequant_uses_the_bf16_row_scale(synthetic):
    import mlx.core as mx

    root, truth = synthetic
    loader = _loader(root)
    name = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight"
    got = loader.get_slice(name, slice(0, 2)).astype(mx.float32)
    scale, _ = loader.get_raw(
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale"
    )
    raw, _ = loader.get_raw(f"{name}")
    expected = (mx.from_fp8(raw[0:2], dtype=mx.float32) * scale.astype(mx.float32)).astype(
        mx.bfloat16
    )
    assert bool(mx.all(got == expected.astype(mx.float32)))
    # And it should be the right magnitude against the pre-quantisation rows.
    want = truth[name].astype(mx.float32)
    ratio = float(mx.mean(mx.abs(got)) / mx.mean(mx.abs(want)))
    assert 0.5 < ratio < 2.0, f"PLE rows off by {ratio:.3g}x"
    loader.close()


# ---------------------------------------------------------------------------
# real checkpoint (skipped when absent)
# ---------------------------------------------------------------------------
def test_real_expert_matches_shared_expert_magnitude():
    import mlx.core as mx

    root = _require_real_checkpoint()
    loader = _loader(root)
    ratios = []
    for layer in (0, 23):
        shared = loader.get(
            f"model.language_model.layers.{layer}.mlp.shared_expert.gate_proj.weight"
        )
        shared_mean = float(mx.mean(mx.abs(shared.astype(mx.float32))))
        expert = loader.get(f"model.language_model.layers.{layer}.mlp.experts.0.gate_proj.weight")
        expert_mean = float(mx.mean(mx.abs(expert.astype(mx.float32))))
        ratios.append(expert_mean / shared_mean)
    loader.close()
    for ratio in ratios:
        # Measured 1.12..1.61 across 20 samples; a wrong global_scale is
        # 0.00055x or 20551x, so this band separates the recipes decisively.
        assert 0.5 < ratio < 3.0, f"expert/shared magnitude ratio {ratio:.4g} out of band"


def test_real_nvfp4_global_scale_identity():
    """6 * 448 * weight_scale_2 equals the per-expert global_scale (measured)."""

    root = _require_real_checkpoint()
    loader = _loader(root)
    for layer in (0, 47):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            table, _ = loader.get_raw(
                f"model.language_model.layers.{layer}.mlp.switch_mlp.{proj}.global_scale"
            )
            for expert in (0, 511):
                ws2, _ = loader.get_raw(
                    f"model.language_model.layers.{layer}.mlp.experts.{expert}.{proj}.weight_scale_2"
                )
                implied = float(ws2.reshape(())) * 6 * 448
                assert implied == pytest.approx(float(table[expert]), rel=1e-5)
    loader.close()


def test_real_ple_rows_read_only_requested_bytes():
    root = _require_real_checkpoint()
    loader = _loader(root, ple_cache_mb=8)
    name = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight"
    rows = loader.get_slice(name, slice(4, 7))
    stats = loader.stats()
    loader.close()
    assert rows.shape == (3, 160)
    assert stats.ple_rows_loaded == 3
    assert stats.ple_bytes_loaded == 3 * 160, f"read {stats.ple_bytes_loaded} bytes for 3 rows"


def test_real_expert_read_is_bounded():
    root = _require_real_checkpoint()
    loader = _loader(root, expert_resident_n=0)
    loader.get("model.language_model.layers.0.mlp.experts.0.down_proj.weight")
    stats = loader.stats()
    loader.close()
    # down_proj [2560, 320] U8 = 819200 weight bytes + [2560, 40] F8 scales
    # 102400 + 4 bytes of global scale = 921604.
    assert 800_000 < stats.expert_bytes_loaded < 1_000_000, stats.expert_bytes_loaded


def test_real_resident_embed_tokens_bytes():
    import mlx.core as mx

    root = _require_real_checkpoint()
    loader = _loader(root)
    embed = loader.get("model.language_model.embed_tokens.weight")
    loader.close()
    assert embed.shape == (248320, 2560)
    assert embed.dtype == mx.bfloat16
    assert embed.nbytes == 248320 * 2560 * 2
