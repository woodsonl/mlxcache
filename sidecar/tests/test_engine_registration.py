"""Engine registration and the construction seam every engine shares.

An adapter for a second architecture is only safe if it cannot drift from the
one that carries the cache convention and the checkpoint trust boundary. These
tests pin that seam: registration is explicit, an unknown engine fails loudly
instead of silently serving toy tokens, and a model built by ANY means goes
through the same post-load adoption.
"""

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


def test_adopt_model_parses_a_named_tier(monkeypatch):
    pytest.importorskip("mlx.core")
    monkeypatch.delenv("MLXCACHE_KV_BITS", raising=False)
    monkeypatch.setenv("MLXCACHE_KV_TIER", "tq-k3v4-g64")
    engine = _Probe(_StubModel(), _StubTokenizer())
    assert engine.kv_tier == "tq-k3v4-g64"
    assert engine._tq_params == (3, 4, 64)
    # The named tier owns the axis: the scalar pair stays f16.
    assert (engine.kv_bits, engine.kv_group_size) == (0, 0)


@pytest.mark.parametrize("tier", ["tq", "tq-k9v4-g64", "tq-k3v4", "nope-k3v4-g64", "tq-k3v4-g1"])
def test_adopt_model_refuses_a_bad_named_tier(monkeypatch, tier):
    pytest.importorskip("mlx.core")
    monkeypatch.delenv("MLXCACHE_KV_BITS", raising=False)
    monkeypatch.setenv("MLXCACHE_KV_TIER", tier)
    with pytest.raises(ValueError):
        _Probe(_StubModel(), _StubTokenizer())


def test_named_tier_and_bits_are_mutually_exclusive(monkeypatch):
    pytest.importorskip("mlx.core")
    monkeypatch.setenv("MLXCACHE_KV_BITS", "8")
    monkeypatch.setenv("MLXCACHE_KV_TIER", "tq-k3v4-g64")
    with pytest.raises(ValueError, match="mutually exclusive"):
        _Probe(_StubModel(), _StubTokenizer())


def test_adopt_model_refuses_a_named_tier_on_a_hybrid_cache(monkeypatch):
    pytest.importorskip("mlx.core")
    from mlx_lm.models.cache import ArraysCache, CacheList, KVCache

    class _Hybrid(_Probe):
        def _prompt_cache_probe(self):
            return [KVCache(), CacheList(KVCache(), ArraysCache(size=2))]

    monkeypatch.delenv("MLXCACHE_KV_BITS", raising=False)
    monkeypatch.setenv("MLXCACHE_KV_TIER", "tq-k3v4-g64")
    with pytest.raises(ValueError, match="not a dense KV stack"):
        _Hybrid(_StubModel(), _StubTokenizer())


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
