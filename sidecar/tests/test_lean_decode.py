"""Hermetic T21 lean-decode tests (no MLX, no model).

Regression coverage for the two adversarial-review defects in
MlxLmEngine._lean_greedy_stream:

1. HISTORY LOSS (Codex #1): with cache=None (every scratch request under
   MLXCACHE_LEAN_DECODE=1), the loop's single-token calls ran with NO
   context — output garbage after the first token. The fix allocates a
   fresh prompt cache so the first full-prompt call prefills it.
2. FINAL TEXT DROP (Codex #13): the EOS/cap exit yielded the pre-finalize
   segment; any detokenizer-buffered text never reached the consumer.
   The fix finalizes before the final yield.

Stub model semantics: logits are one-hot at (tokens seen so far + 1), so a
correct implementation generates [4, 5, 6, ...] for prompt [1, 2, 3]; an
implementation with the history bug generates [4, 2, 2, ...] (each step
sees only its own token).
"""

from __future__ import annotations

import sys
import types

import pytest


class _CountingCache:
    """Stands in for an mlx prompt cache: tracks how many tokens it holds."""

    def __init__(self) -> None:
        self.offset = 0


class _Shared:
    """Side channel: the model publishes its next-token id here and the
    mx.argmax stub returns it (decouples the loop from real tensor ops)."""

    def __init__(self) -> None:
        self.hot = 0


class _CountingModel:
    def __init__(self, shared: _Shared) -> None:
        self.shared = shared
        self.calls: list[int] = []

    def __call__(self, y, cache=None):
        n = int(y.shape[-1])
        self.calls.append(n)
        if cache is not None:
            cache.offset += n
        history = cache.offset if cache is not None else n
        self.shared.hot = (history + 1) % 256
        return _Logits()


class _Logits:
    """Slice-transparent logits marker: [:, -1, :] returns itself."""

    def __getitem__(self, _idx):
        return self


class _HotToken:
    def __init__(self, v: int) -> None:
        self._v = v

    def item(self) -> int:
        return self._v

    # The loop re-indexes the argmax result (y[None]) before feeding it back;
    # a 1-token batch is what the model stub expects to see.
    def __getitem__(self, _idx):
        return _FakeArray([self._v])


class _FakeArray:
    """[None]-index transparent: keeps (1, n) shape semantics."""

    def __init__(self, data) -> None:
        self.data = list(data)

    def __getitem__(self, _idx):
        return self

    @property
    def shape(self):
        return (1, len(self.data))


class _FakeDetok:
    """Detokenizer stub: segments are str(token); finalize() is observable
    (appends '!' so tests can see the FINALIZED segment was yielded)."""

    def __init__(self) -> None:
        self.last_segment = ""
        self.finalized = 0

    def add_token(self, token: int) -> None:
        self.last_segment = str(token)

    def finalize(self) -> None:
        self.finalized += 1
        self.last_segment = f"{self.last_segment}!"


class _FakeTokenizer:
    def __init__(self) -> None:
        self.detokenizer = _FakeDetok()


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def _install_stub_mlx(monkeypatch, shared: _Shared) -> None:
    core = types.ModuleType("mlx.core")
    core.array = lambda data: _FakeArray(data)
    core.new_thread_local_stream = lambda dev: object()
    core.default_device = lambda: object()
    core.stream = lambda dev: _nullcontext()
    core.async_eval = lambda *a, **k: None
    core.argmax = lambda x, axis=-1: _HotToken(shared.hot)
    core.eval = lambda *a, **k: None

    mx = types.ModuleType("mlx")
    mx.core = core
    monkeypatch.setitem(sys.modules, "mlx", mx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)

    # The cache=None fix imports make_prompt_cache inside the loop.
    cache_mod = types.ModuleType("mlx_lm.models.cache")
    cache_mod.make_prompt_cache = lambda model: _CountingCache()
    mlx_lm = types.ModuleType("mlx_lm")
    mlx_lm.models = types.ModuleType("mlx_lm.models")
    mlx_lm.models.cache = cache_mod
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.models", mlx_lm.models)
    monkeypatch.setitem(sys.modules, "mlx_lm.models.cache", cache_mod)


def _engine(monkeypatch):
    shared = _Shared()
    _install_stub_mlx(monkeypatch, shared)
    from mlxcache_sidecar.server import MlxLmEngine

    eng = MlxLmEngine.__new__(MlxLmEngine)
    eng.model = _CountingModel(shared)
    eng.tokenizer = _FakeTokenizer()
    eng._eos_ids = set()
    eng._lean_decode = True
    return eng


def test_lean_scratch_keeps_prompt_history(monkeypatch):
    # Regression (Codex #1): cache=None must allocate a prompt cache; the
    # generated sequence walks 4,5,6,7,... — not [4,2,2,2] (no EOS → the
    # stream runs to the internal 256-token cap).
    eng = _engine(monkeypatch)
    out = list(eng._lean_greedy_stream([1, 2, 3], None))
    assert [t for t, _ in out[:4]] == [4, 5, 6, 7], (
        f"scratch lean decode lost prompt history: {[t for t, _ in out[:8]]}"
    )
    assert len(out) == 256, "no EOS: the stream runs to the internal cap"
    # The first call consumed the WHOLE prompt (prefill), then one token at
    # a time — proving the history accumulated.
    assert eng.model.calls[:5] == [3, 1, 1, 1, 1]


def test_lean_with_existing_cache_continues(monkeypatch):
    eng = _engine(monkeypatch)
    cache = _CountingCache()
    cache.offset = 10  # a 10-token adopted checkpoint
    out = list(eng._lean_greedy_stream([11, 12, 13], cache))
    # History after prefill = 13; first argmax = 14...
    assert [t for t, _ in out[:4]] == [14, 15, 16, 17]
    assert eng.model.calls[:5] == [3, 1, 1, 1, 1]


def test_lean_finalizes_before_final_yield(monkeypatch):
    # Regression (Codex #13): the final segment must be the finalized one,
    # and the EOS token itself is still yielded (stream_generate parity).
    eng = _engine(monkeypatch)
    eng._eos_ids = {6}
    out = list(eng._lean_greedy_stream([1, 2, 3], None))
    assert eng.tokenizer.detokenizer.finalized == 1
    assert [t for t, _ in out] == [4, 5, 6]
    assert out[-1][1].endswith("!")


def test_lean_cap_exit_finalizes_too(monkeypatch):
    # The internal 256 cap ends the stream — finalize must still have run
    # and the last yielded segment must be the finalized one.
    eng = _engine(monkeypatch)
    out = list(eng._lean_greedy_stream([1, 2, 3], None))
    assert len(out) == 256
    assert eng.tokenizer.detokenizer.finalized == 1
    assert out[-1][1].endswith("!")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
