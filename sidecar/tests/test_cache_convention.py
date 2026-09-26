"""Hermetic tests for MlxLmEngine's cache convention (no MLX, no model).

Pins the invariant the R1-5 thesis guard proves on real hardware: the saved
checkpoint cache covers tokens[:-1], and the resume path feeds tokens[cached-1:]
so the model predicts the final token from KV of the preceding ones. If the two
halves disagree, every cache hit generates different output than a scratch run.
"""

from __future__ import annotations

import sys
import types

import pytest
from mlxcache_sidecar import blob


class FakeCache:
    """An mlx prompt cache stand-in that records how many tokens it holds."""

    def __init__(self, length: int = 0) -> None:
        self.len = length
        self.state = "s"

    def __iter__(self):
        return iter([self])


class FakeArray:
    def __init__(self, data) -> None:
        self._d = list(data)

    def __getitem__(self, _idx):
        return self

    @property
    def shape(self):
        return (1, len(self._d))


def _install_fake_mlx(monkeypatch):
    """Install just enough of mlx / mlx_lm.models.cache to drive MlxLmEngine.

    Returns (prefill_lengths, model): the model records the token count it was
    called with, so a test can see what prefill actually seeded the cache with.
    """
    prefilled: list[int] = []
    saved: dict[str, int] = {}

    class FakeModel:
        def __call__(self, arr, cache=None):
            n = int(arr.shape[-1])
            cache.len = n
            prefilled.append(n)
            return types.SimpleNamespace()

    core = types.ModuleType("mlx.core")
    core.array = FakeArray
    core.eval = lambda *a, **k: None
    core.clear_cache = lambda: None

    mx = types.ModuleType("mlx")
    mx.core = core

    cache_mod = types.ModuleType("mlx_lm.models.cache")
    cache_mod.make_prompt_cache = lambda model: FakeCache(0)
    cache_mod.save_prompt_cache = lambda path, cache: saved.__setitem__(path, cache.len)
    cache_mod.load_prompt_cache = lambda path: FakeCache(saved.get(path, 0))

    mlx_lm = types.ModuleType("mlx_lm")
    mlx_lm.models = types.ModuleType("mlx_lm.models")
    mlx_lm.models.cache = cache_mod

    monkeypatch.setitem(sys.modules, "mlx", mx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.models", mlx_lm.models)
    monkeypatch.setitem(sys.modules, "mlx_lm.models.cache", cache_mod)
    return prefilled, FakeModel()


def _engine(model=None):
    from mlxcache_sidecar.server import MlxLmEngine

    eng = MlxLmEngine.__new__(MlxLmEngine)
    eng.model_id = "fake"
    eng.model = model if model is not None else types.SimpleNamespace()
    eng.tokenizer_hash = "h"
    return eng


def _write_blob(path, tokens):
    meta = blob.CheckpointMeta(
        fingerprint=blob.Fingerprint(
            model_id="fake", tokenizer_hash="h", kv_dtype="f16", kv_layout_version=1
        ),
        token_count=len(tokens),
        tokens=tokens,
    )
    with open(path, "wb") as fh:
        fh.write(blob.encode(meta, b"payload"))
    return path


def test_prefill_seeds_the_cache_with_tokens_minus_one(monkeypatch):
    # The saved cache must cover tokens[:-1]. Caching all tokens makes the hit
    # path re-feed the last token and diverge from a scratch run.
    prefilled, model = _install_fake_mlx(monkeypatch)
    eng = _engine(model)
    eng.prefill([10, 20, 30, 40])
    assert prefilled == [3]


def test_single_token_prefill_seeds_the_one_token(monkeypatch):
    # tokens[:-1] is empty for a one-token prompt; seed the token instead of
    # crashing on an empty prefill.
    prefilled, model = _install_fake_mlx(monkeypatch)
    eng = _engine(model)
    eng.prefill([7])
    assert prefilled == [1]


def test_resume_feeds_from_one_before_the_cached_count(monkeypatch, tmp_path):
    _install_fake_mlx(monkeypatch)
    eng = _engine()
    tokens = [1, 2, 3, 4, 5]
    blob_path = _write_blob(str(tmp_path / "b.ckpt"), tokens[:4])
    _cache, prompt = eng._load_cache_delta(tokens, blob_path)
    # Cache holds 4, so feed tokens[3:] and the model predicts token 5.
    assert prompt == [4, 5]


def test_seed_plus_delta_covers_the_whole_prompt(monkeypatch, tmp_path):
    # The end-to-end invariant: prefill(tokens) seeds len(tokens)-1, resume
    # feeds tokens[cached-1:], and together they cover every token exactly once.
    prefilled, model = _install_fake_mlx(monkeypatch)
    eng = _engine(model)
    tokens = [1, 2, 3, 4, 5, 6]
    eng.prefill(tokens)
    seeded = prefilled[-1]
    blob_path = _write_blob(str(tmp_path / "b.ckpt"), tokens)
    _cache, prompt = eng._load_cache_delta(tokens, blob_path)
    assert seeded + len(prompt) == len(tokens)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
