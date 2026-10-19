"""Hermetic tests for the `qwen4_exp` (Qwen3.8-Flash-Next) architecture port.

Everything here runs on a tiny random checkpoint built in memory, so no model
download and no GPU residency are involved. The real-checkpoint assertion at the
bottom reads safetensors *headers* only, and skips when the model is absent.

mlx is an optional extra: the module-level `importorskip` keeps collection clean
on a machine without it, and every mlx-dependent import (including this
package's own modules, which import mlx at module scope) happens inside a test
or fixture body so the guard is the only thing that runs at import time.
"""

from __future__ import annotations

import io
import json
import os
import struct
import zlib

import pytest

pytest.importorskip("mlx.core")

REAL_CHECKPOINT = "/Users/lance/models/Qwen3.8-Flash-Next-NVFP4-mlxstream"


# ---------------------------------------------------------------------- fixture


def tiny_config_dict() -> dict:
    """A `text_config`-shaped dict small enough to run in milliseconds.

    Every axis the checkpoint exercises is present and non-degenerate: 4 layers
    hit one full-attention layer (QSA indexer) and three linear ones, layer 2
    (one-indexed) carries PLE so the n-gram/hash path runs, and the sharded
    table has two shards so the run-merging lookup is exercised.
    """
    return {
        "model_type": "qwen4_exp_text",
        "vocab_size": 128,
        "hidden_size": 64,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 4096,
        "attention_bias": False,
        "tie_word_embeddings": False,
        "layer_types": [
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
        "full_attention_interval": 4,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
            "mrope_section": [3, 3, 2],
        },
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "output_gate_type": "sigmoid",
        "moe_intermediate_size": 16,
        "shared_expert_intermediate_size": 16,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "norm_topk_prob": True,
        "hc_count": 2,
        "hc_lowrank": 8,
        "ple_layer_ids": [2],
        "ple_embed_dim": 32,
        "ple_conv_kernel_size": 4,
        "ngram_size": 3,
        "heads_per_ngram": 2,
        "ngram_vocab_size_base": 100,
        "make_ngram_vocab_size_divisible_by": 8,
        "seed": 1234,
        "split_ngram_parts": 2,
        "indexer_n_heads": 2,
        "indexer_kv_heads": 1,
        "indexer_head_dim": 16,
        "indexer_budget": 8,
        "indexer_compress_ratio": 4,
        "mtp_num_hidden_layers": 1,
        "mtp": {"num_hidden_layers": 1, "layer_types": ["full_attention"]},
        "eos_token_id": 127,
        "bos_token_id": 1,
    }


class RandomWeightSource:
    """Deterministic in-memory stand-in for the checkpoint reader.

    Shapes come from the model's own tables (`weight_shapes`, `mtp_weight_shapes`),
    so the fixture cannot drift from the architecture: a name the model asks for
    that the tables do not describe is a test failure, not a silent zero. Values
    are seeded per name so two sources built from the same config agree bit for
    bit, which is what the determinism assertion rests on.

    Three encodings are reproduced because the model depends on each:

    - MTP routed experts: `uint8` fp8 payload plus a bf16 `weight_scale_inv`
      grid, so the dequantization path runs.
    - PLE n-gram shards: rows served through `get_slice`, the way a 51 GiB table
      is actually read.
    - everything else: bf16 weights.
    """

    def __init__(self, config, shapes: dict, seed: int = 0):
        import mlx.core as mx

        self.mx = mx
        self.config = config
        self.shapes = dict(shapes)
        self.seed = seed
        self.calls: list[tuple[str, tuple[int, ...] | int]] = []
        self.ple_slices: list[tuple[str, int, int]] = []
        self.ple_shards = {
            name for name in self.shapes if ".ple_embedding.ngram_embedding.shard_" in name
        }

    # -- value generation ---------------------------------------------------

    def _seed_for(self, name: str) -> int:
        return (zlib.crc32(name.encode()) + self.seed) % (2**31 - 1)

    def _values(self, name: str, shape: tuple[int, ...]):
        mx = self.mx
        mx.random.seed(self._seed_for(name))
        if name.endswith("A_log"):
            return mx.random.uniform(0.01, 1.0, shape)
        if name.endswith("dt_bias"):
            return mx.ones(shape)
        if ".mlp.experts." in name and name.endswith(".weight") and not name.startswith("mtp."):
            # Main-model experts are resident-dense for the tiny fixture.
            return mx.random.normal(shape).astype(mx.bfloat16) * 0.05
        if name.startswith("mtp.") and ".mlp.experts." in name and name.endswith(".weight"):
            return mx.to_fp8(mx.random.uniform(0.5, 1.5, shape).astype(mx.float32))
        if ".mlp.experts." in name and name.endswith(".weight_scale_inv"):
            return mx.random.uniform(0.5, 1.5, shape).astype(mx.bfloat16)
        if name.endswith("layer_multipliers") or name.endswith("ngram_heads_vocab_sizes"):
            return mx.zeros(shape, dtype=mx.int64)
        if name.endswith("ngram_heads_offsets"):
            return mx.zeros(shape, dtype=mx.int64)
        if name.endswith(".norm.weight") or "_norm" in name or name.endswith("hc_norm.weight"):
            return mx.random.normal(shape).astype(mx.bfloat16) * 0.02
        if name.endswith("embed_tokens.weight") or name.endswith("lm_head.weight"):
            return mx.random.normal(shape).astype(mx.bfloat16) * 0.05
        return mx.random.normal(shape).astype(mx.bfloat16) * 0.05

    def _table(self, name: str):
        mx = self.mx
        shape = self.shapes[name]
        mx.random.seed(self._seed_for(name))
        return mx.random.normal(shape).astype(mx.float32) * 0.05

    # -- WeightSource -------------------------------------------------------

    def get(self, name: str):
        if name not in self.shapes:
            raise KeyError(f"unexpected weight request: {name}")
        shape = self.shapes[name]
        self.calls.append((name, shape))
        return self._values(name, shape)

    def get_slice(self, name: str, rows: slice):
        if name not in self.shapes:
            raise KeyError(f"unexpected weight slice request: {name}")
        start, stop, _ = rows.indices(self.shapes[name][0])
        self.ple_slices.append((name, start, stop))
        self.calls.append((name, stop - start))
        return self._table(name)[start:stop]


@pytest.fixture(scope="module")
def bundle():
    import mlx.core as mx
    from mlxcache_qwen4 import Qwen4MTPModel, Qwen4TextConfig, Qwen4TextModel
    from mlxcache_qwen4.model import mtp_weight_shapes, weight_shapes

    config = Qwen4TextConfig.from_checkpoint(tiny_config_dict())
    shapes = weight_shapes(config)
    mtp_shapes = mtp_weight_shapes(config)
    source = RandomWeightSource(config, {**shapes, **mtp_shapes})
    mx.random.seed(7)
    model = Qwen4TextModel(config, source)
    mtp = Qwen4MTPModel(config, source)
    mx.eval(model.parameters(), mtp.parameters())
    return {
        "config": config,
        "model": model,
        "mtp": mtp,
        "source": source,
        "shapes": shapes,
        "mtp_shapes": mtp_shapes,
    }


def _greedy(model, cache, ids, steps: int):
    """Greedy decode `steps` tokens, feeding one id at a time from `ids`."""
    import mlx.core as mx

    prompt = list(ids)
    produced = []
    logits = None
    for token in prompt:
        logits = model(mx.array([[token]]), cache=cache)
    for _ in range(steps):
        nxt = int(mx.argmax(logits[0, -1]).item())
        produced.append(nxt)
        logits = model(mx.array([[nxt]]), cache=cache)
    return produced


# ------------------------------------------------------------------------ tests


def test_forward_shape_and_determinism(bundle):
    """Logits have checkpoint shape, and a rebuild from the same weights matches."""
    import mlx.core as mx
    from mlxcache_qwen4 import Qwen4TextModel

    config, model, source = bundle["config"], bundle["model"], bundle["source"]
    ids = mx.array([[1, 5, 9, 17, 23], [3, 4, 5, 6, 7]])
    out = model(ids, cache=model.make_cache())
    mx.eval(out)
    assert out.shape == (2, 5, config.vocab_size)
    assert mx.all(mx.isfinite(out))

    rebuilt = Qwen4TextModel(config, source)
    mx.eval(rebuilt.parameters())
    again = rebuilt(ids, cache=rebuilt.make_cache())
    mx.eval(again)
    assert mx.array_equal(out, again)

    # Every name the model asked for is one the table describes, with that shape.
    requested = {name for name, _ in source.calls}
    assert requested <= set(bundle["shapes"]) | set(bundle["mtp_shapes"])
    for name, shape in source.calls:
        expected = bundle["shapes"].get(name, bundle["mtp_shapes"].get(name))
        if isinstance(shape, tuple):
            assert shape == expected, name
    assert any(".ple_embedding.ngram_embedding.shard_" in name for name, _ in source.calls)


def test_hyper_streams_are_tiled_not_interleaved(bundle):
    """The embedding must widen into `hc_count` WHOLE copies of the hidden vector.

    The hyper-connection mixer unflattens the width into (hc_count, hidden_size),
    so stream `s` has to occupy the `s`-th contiguous block. `mx.repeat(x, n,
    axis=-1)` interleaves elements instead (1,2,3 -> 1,1,1,2,2,2), which puts
    every stream's features in all four slots and feeds every layer a
    permutation of the intended input. That is a whole-model error that still
    produces finite logits, so shape and finiteness checks cannot see it; this
    asserts the layout directly.
    """
    import mlx.core as mx

    config, model = bundle["config"], bundle["model"]
    ids = mx.array([[1, 5, 9, 17]])
    embedded = mx.take(model.embed, ids.reshape(-1), axis=0).reshape(1, -1, config.hidden_size)
    expected = mx.tile(embedded, (1, 1, config.hc_count))

    # Record what the first layer's attention mixer is handed: it is the first
    # consumer of the widened width. Special methods resolve on the type, so the
    # class is patched for the duration rather than the instance.
    from mlxcache_qwen4.model import Qwen4GatedResidual

    observed = {}
    original_call = Qwen4GatedResidual.__call__

    def spy(self, hyper):
        observed.setdefault("hyper", hyper)
        return original_call(self, hyper)

    Qwen4GatedResidual.__call__ = spy
    try:
        model(ids, cache=model.make_cache())
    finally:
        Qwen4GatedResidual.__call__ = original_call

    hyper = observed["hyper"]
    assert hyper.shape == expected.shape
    if not mx.array_equal(hyper, expected):
        raise AssertionError(
            "embedding was widened by interleaving, not tiling: hyper-connection "
            "stream `s` is not the s-th contiguous block of the hidden width"
        )


def test_cache_layout_and_roundtrip(bundle):
    """The cache is mlx-lm's own classes and survives a save/load cycle intact."""
    import mlx.core as mx
    from mlx_lm.models.cache import (
        ArraysCache,
        CacheList,
        KVCache,
        load_prompt_cache,
        save_prompt_cache,
    )

    config, model = bundle["config"], bundle["model"]
    cache = model.make_cache()
    assert len(cache) == config.num_hidden_layers
    for index, entry in enumerate(cache):
        if config.layer_types[index] == "linear_attention":
            assert isinstance(entry, ArraysCache)
            assert all(slot is not None for slot in entry.cache), (
                "an unset slot cannot be serialised"
            )
        else:
            assert isinstance(entry, CacheList)
            assert isinstance(entry[0], KVCache) and isinstance(entry[1], ArraysCache)
            assert all(slot is not None for slot in entry[1].cache)

    tokens = [1, 5, 9, 17, 23, 42, 8]
    model(mx.array([tokens[:-1]]), cache=cache)

    buffer = io.BytesIO()
    buffer.name = "checkpoint.safetensors"
    save_prompt_cache(buffer, cache, {"engine": "qwen4-exp"})
    buffer.seek(0)
    loaded, metadata = load_prompt_cache(buffer, return_metadata=True)

    assert metadata["engine"] == "qwen4-exp"
    assert [type(entry).__name__ for entry in loaded] == [type(entry).__name__ for entry in cache]
    for before, after in zip(cache, loaded, strict=True):
        if isinstance(before, CacheList):
            assert before[0].offset == after[0].offset
            # `state` is the trimmed view a checkpoint actually stores; the live
            # KVCache keeps its padded buffer.
            for left, right in zip(before[0].state, after[0].state, strict=True):
                mx.eval(left, right)
                assert mx.array_equal(left, right)
            for left, right in zip(before[1].state, after[1].state, strict=True):
                assert mx.array_equal(left, right)
        else:
            for left, right in zip(before.cache, after.cache, strict=True):
                assert mx.array_equal(left, right)

    # The loaded cache is usable: continuing from it produces finite logits.
    out = model(mx.array([[8]]), cache=loaded)
    mx.eval(out)
    assert mx.all(mx.isfinite(out))


def test_resume_after_reload_generates_identical_tokens(bundle):
    """The core claim: a reloaded cache continues the same generation.

    Prefill `tokens[:-1]`, save, load, then feed the final token and decode. The
    reference run prefills every token and decodes from scratch. Greedy token
    streams must match exactly, which is only true if the serialised cache
    carries every piece of state the forward pass reads.
    """
    import mlx.core as mx

    model = bundle["model"]
    tokens = [1, 5, 9, 17, 23, 42, 8, 11]

    # Reference: one shot prefill, then greedy decode.
    scratch = model.make_cache()
    logits = model(mx.array([tokens]), cache=scratch)
    scratch_tokens = []
    for _ in range(6):
        nxt = int(mx.argmax(logits[0, -1]).item())
        scratch_tokens.append(nxt)
        logits = model(mx.array([[nxt]]), cache=scratch)

    # Resumed: prefill all but the last token, checkpoint, reload, feed the tail.
    cache = model.make_cache()
    model(mx.array([tokens[:-1]]), cache=cache)
    buffer = io.BytesIO()
    buffer.name = "checkpoint.safetensors"
    from mlx_lm.models.cache import load_prompt_cache, save_prompt_cache

    save_prompt_cache(buffer, cache)
    buffer.seek(0)
    resumed = load_prompt_cache(buffer)
    logits = model(mx.array([[tokens[-1]]]), cache=resumed)
    resumed_tokens = []
    for _ in range(6):
        nxt = int(mx.argmax(logits[0, -1]).item())
        resumed_tokens.append(nxt)
        logits = model(mx.array([[nxt]]), cache=resumed)

    assert resumed_tokens == scratch_tokens


def test_delta_prefill_matches_full_prefill(bundle):
    """Prefilling in two batches is numerically the same as prefilling once.

    The chunked GDN path and the recurrent single-token path are different
    arithmetic on the same state, and the short convs, the indexer key cache and
    the indexer's block selection all have to line up. A relative bound rather
    than equality: the two paths legitimately sum in a different order.
    """
    import mlx.core as mx

    model = bundle["model"]
    tokens = [1, 5, 9, 17, 23, 42, 8, 11, 3, 6]
    split = 6

    full = model(mx.array([tokens]), cache=model.make_cache())

    cache = model.make_cache()
    model(mx.array([tokens[:split]]), cache=cache)
    delta = model(mx.array([tokens[split:]]), cache=cache)
    mx.eval(full, delta)

    reference = full[0, -1].astype(mx.float32)
    candidate = delta[0, -1].astype(mx.float32)
    scale = mx.max(mx.abs(reference)).item()
    diff = mx.max(mx.abs(reference - candidate)).item()
    assert diff / scale < 5e-2, f"relative delta {diff / scale:.3e} too large"


def test_contradictory_layer_types_rejected():
    from mlxcache_qwen4 import Qwen4TextConfig

    payload = tiny_config_dict()
    payload["layer_types"][3] = "linear_attention"  # interval 4 requires full here
    with pytest.raises(ValueError, match="contradicts full_attention_interval"):
        Qwen4TextConfig.from_checkpoint(payload)

    wrong_length = tiny_config_dict()
    wrong_length["layer_types"] = ["linear_attention"] * 3
    with pytest.raises(ValueError, match="entries but num_hidden_layers"):
        Qwen4TextConfig.from_checkpoint(wrong_length)


def test_mtp_forward(bundle):
    """The MTP head runs end to end and dequantizes its fp8 experts.

    The hidden state enters at the hyper-connected width, not at `hidden_size`:
    the head consumes every residual stream (`pre_fc_norm_hidden` is
    `hc_hidden_size` wide in the checkpoint), while the token embedding enters as
    a single `hidden_size` stream.
    """
    import mlx.core as mx

    config, head, source = bundle["config"], bundle["mtp"], bundle["source"]
    ids = mx.array([[1, 5, 9, 17, 23]])
    hidden = mx.random.normal((1, 5, config.hc_hidden_size)).astype(mx.bfloat16)
    out = head(hidden, ids, cache=head.make_cache())
    mx.eval(out)
    assert out.shape == (1, 5, config.vocab_size)
    assert mx.all(mx.isfinite(out))

    scales = [name for name, _ in source.calls if name.endswith("weight_scale_inv")]
    assert scales, "the MTP path must read the fp8 block scales"


def _real_config():
    path = os.path.join(REAL_CHECKPOINT, "config.json")
    if not os.path.exists(path):
        pytest.skip(f"real checkpoint not present at {REAL_CHECKPOINT}")
    with open(path) as handle:
        return json.load(handle)


def _safetensors_header(path: str) -> dict:
    with open(path, "rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        return json.loads(handle.read(length))


def test_weight_table_matches_real_checkpoint():
    """Every tensor the model reads exists at the shape the checkpoint stores.

    Headers only: the index maps each tensor to a shard and each shard's header
    carries dtype and shape, so the whole table is checked without reading a
    single tensor body. Shapes are the stored layout (packed NVFP4, fp8 block
    scales), because a header knows nothing about dequantization.
    """
    from mlxcache_qwen4 import Qwen4TextConfig
    from mlxcache_qwen4.model import stored_weight_shapes

    config = Qwen4TextConfig.from_checkpoint(_real_config())
    index_path = os.path.join(REAL_CHECKPOINT, "model.safetensors.index.json")
    with open(index_path) as handle:
        weight_map = json.load(handle)["weight_map"]

    headers: dict[str, dict] = {}
    shapes = stored_weight_shapes(config)
    missing = []
    mismatched = []
    for name, expected in shapes.items():
        shard = weight_map.get(name)
        if shard is None:
            missing.append(name)
            continue
        if shard not in headers:
            headers[shard] = _safetensors_header(os.path.join(REAL_CHECKPOINT, shard))
        actual = tuple(headers[shard][name]["shape"])
        if actual != expected:
            mismatched.append((name, actual, expected))

    assert not missing, f"{len(missing)} names absent from the checkpoint: {missing[:5]}"
    assert not mismatched, f"shape mismatches: {mismatched[:5]}"
