"""Engine registration and the construction seam shared by both model engines.

The adapter for a second architecture is only safe if it cannot drift from the
one that carries the cache convention and the checkpoint trust boundary. These
tests pin that seam: registration is explicit, an unknown engine fails loudly
instead of silently serving toy tokens, and a model built by ANY means goes
through the same post-load adoption.
"""

import importlib
import os

import pytest
from mlxcache_sidecar import server


def test_unknown_engine_is_a_hard_error(monkeypatch):
    monkeypatch.setenv("MLXCACHE_ENGINE", "gpt-9")
    with pytest.raises(ValueError, match="unknown MLXCACHE_ENGINE"):
        server.make_engine("some-model")


def test_default_engine_is_still_synthetic(monkeypatch):
    monkeypatch.delenv("MLXCACHE_ENGINE", raising=False)
    engine = server.make_engine("some-model")
    assert isinstance(engine, server.SyntheticEngine)
    # The default engine owns the historical key domain, so every blob written
    # before engine selection existed keeps its identity.
    assert engine.engine_id == "mlx-lm"


def test_redacted_engine_reports_its_own_key_domain():
    from mlxcache_sidecar.engine_redacted import RedactedEngineEngine

    # Distinct from the generic engine: the two architectures cannot exchange
    # blobs even if model id, tokenizer, dtype and KV tier all agree.
    assert RedactedEngineEngine.engine_id == "redacted-engine"
    assert RedactedEngineEngine.engine_id != server.MlxLmEngine.engine_id


def test_redacted_engine_is_a_construction_subclass():
    from mlxcache_sidecar.engine_redacted import RedactedEngineEngine

    # The adapter surface is inherited, not reimplemented: a second copy of the
    # cache convention is how wrong KV gets served silently.
    assert issubclass(RedactedEngineEngine, server.MlxLmEngine)
    for name in (
        "prefill",
        "generate_with_text",
        "stream",
        "prepare_stream",
        "stream_prepared",
        "_load_cache_delta",
        "_load_cache_from_payload",
        "_quantize_cache",
        "_assert_quantization_supported",
        "_cache_layout_of",
    ):
        assert getattr(RedactedEngineEngine, name) is getattr(server.MlxLmEngine, name), name


def test_redacted_engine_build_is_the_only_overridden_seam():
    from mlxcache_sidecar.engine_redacted import RedactedEngineEngine

    # Members of the parent the adapter replaces. `engine_id` is the key domain.
    overridden = {
        name
        for name, value in vars(RedactedEngineEngine).items()
        if not name.startswith("__") and name in vars(server.MlxLmEngine)
    }
    assert overridden == {"engine_id"}, overridden
    # `_build` is new (the parent constructs with mlx_lm.load), and it is the
    # only method the adapter adds. Anything else means it started diverging.
    added = {
        name
        for name, value in vars(RedactedEngineEngine).items()
        if not name.startswith("__") and name not in vars(server.MlxLmEngine) and callable(value)
    }
    assert added == {"_build"}, added


class _StubTokenizer:
    eos_token_id = 0
    _eos_token_ids = {0}

    def get_vocab(self):
        return {"a": 0, "b": 1}


class _StubModel:
    """Stands in for any architecture: `_adopt_model` must not care which."""

    def parameters(self):
        import mlx.core as mx

        return {"w": mx.zeros((1,), dtype=mx.bfloat16)}

    def make_cache(self):
        from mlx_lm.models.cache import KVCache

        return [KVCache()]


class _Probe(server.MlxLmEngine):
    """Inherits every post-load step; supplies only construction."""

    def __init__(self, model, tokenizer):
        self.model_id = "stub"
        self.model = model
        self.tokenizer = tokenizer
        self._adopt_model()

    def _prompt_cache_probe(self):
        return self.model.make_cache()


def test_adopt_model_is_construction_agnostic():
    pytest.importorskip("mlx.core")
    engine = _Probe(_StubModel(), _StubTokenizer())
    # Identity derived from the artifact, not from the load path.
    assert engine.tokenizer_hash == server.MlxLmEngine._hash_vocab(engine)
    assert engine.kv_dtype == "bfloat16"
    assert engine.kv_bits == 0 and engine.kv_group_size == 0
    assert engine.cache_layout == server.MlxLmEngine._cache_layout_of(engine.model.make_cache())
    assert engine.prefill_count == 0
    assert engine._eos_ids == {0}


def test_adopt_model_derives_the_kv_tier_from_env(monkeypatch):
    pytest.importorskip("mlx.core")
    monkeypatch.setenv("MLXCACHE_KV_BITS", "8")
    monkeypatch.delenv("MLXCACHE_KV_GROUP_SIZE", raising=False)
    engine = _Probe(_StubModel(), _StubTokenizer())
    # The documented default group size for the proven q8 tier.
    assert (engine.kv_bits, engine.kv_group_size) == (8, 64)


def test_adopt_model_refuses_quantization_on_a_hybrid_cache(monkeypatch):
    pytest.importorskip("mlx.core")
    from mlx_lm.models.cache import ArraysCache, CacheList, KVCache

    class _Hybrid(_Probe):
        def _prompt_cache_probe(self):
            return [KVCache(), CacheList(KVCache(), ArraysCache(size=2))]

    monkeypatch.setenv("MLXCACHE_KV_BITS", "8")
    with pytest.raises(ValueError, match="not a dense KV stack"):
        _Hybrid(_StubModel(), _StubTokenizer())


def test_cache_layout_separates_compositions():
    # Built from stand-ins: the signature hashes the ORDERED CLASS NAMES, so it
    # needs no mlx import and can run on CI, where the mlx extra is absent.
    class KVCache:  # name matches the real cache entry on purpose
        pass

    class ArraysCache:
        pass

    dense = server.MlxLmEngine._cache_layout_of([KVCache(), KVCache()])
    hybrid = server.MlxLmEngine._cache_layout_of([KVCache(), ArraysCache()])
    assert dense != hybrid
    # Stable across calls: it is folded into every key and header.
    assert dense == server.MlxLmEngine._cache_layout_of([KVCache(), KVCache()])


def test_engine_selection_does_not_leak_between_engines(monkeypatch):
    monkeypatch.setenv("MLXCACHE_ENGINE", "synthetic")
    first = server.make_engine("m")
    monkeypatch.setenv("MLXCACHE_ENGINE", "unknown-engine")
    with pytest.raises(ValueError):
        server.make_engine("m")
    assert isinstance(first, server.SyntheticEngine)


def _real_stack_available() -> bool:
    """The checkpoint AND every module the adapter builds from.

    Checking package presence alone is not enough: the package can exist while
    a module the engine imports is missing or half-written, which turns a skip
    into a failure that says nothing about the engine.
    """
    if not os.path.isdir(
        os.environ.get(
            "MLXCACHE_Q4_CHECKPOINT", "REDACTED-HOME/models/REDACTED-MODEL-NVFP4-REDACTED"
        )
    ):
        return False
    try:
        importlib.import_module("mlxcache_redacted.loader")
        importlib.import_module("mlxcache_redacted.model")
    except ImportError:
        return False
    return True


@pytest.mark.skipif(
    not _real_stack_available(),
    reason="redacted_engine checkpoint or architecture/loader modules not present",
)
def test_redacted_engine_boots_on_the_real_checkpoint():
    """The adapter over the real stack: identity fields must be real, not
    placeholders, and the cache composition must be a hybrid one."""
    from mlxcache_sidecar.engine_redacted import RedactedEngineEngine

    engine = RedactedEngineEngine(
        os.environ.get(
            "MLXCACHE_Q4_CHECKPOINT", "REDACTED-HOME/models/REDACTED-MODEL-NVFP4-REDACTED"
        )
    )
    assert engine.engine_id == "redacted-engine"
    assert len(engine.cache_layout) == 16
    assert engine.tokenizer_hash and engine.tokenizer_hash != "synthetic"
    assert engine.kv_dtype != "unknown"
    # Hybrid: 12 full-attention layers cannot be the whole stack, so the
    # composition must NOT be a dense KV repeat. The dense baseline is built
    # from KVCache entries directly: `type(c)()` cannot construct ArraysCache,
    # which requires a slot count.
    from mlx_lm.models.cache import KVCache  # noqa: PLC0415

    probe = engine._prompt_cache_probe()
    dense = server.MlxLmEngine._cache_layout_of([KVCache() for _ in range(len(probe))])
    assert len(probe) == 48
    assert engine.cache_layout != dense
