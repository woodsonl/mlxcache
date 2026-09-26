"""Hermetic tests for MlxLmEngine's cache convention (no MLX, no model).

Pins the invariant the R1-5 thesis guard proves on real hardware: the saved
checkpoint cache covers prefix[:-1], and the resume path feeds the uncovered tail
(tokens[len(prefix)-1:]) so the model predicts the final token from KV of the
preceding ones. If the two halves disagree, every cache hit generates different
output than a scratch run.
"""

from __future__ import annotations

import json
import sys
import types

import pytest
from mlxcache_sidecar import blob, server


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


def _install_fake_mlx(monkeypatch, on_load=None):
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

    def _load(path):
        if on_load is not None:
            on_load()
        return FakeCache(saved.get(path, 0))

    cache_mod = types.ModuleType("mlx_lm.models.cache")
    cache_mod.make_prompt_cache = lambda model: FakeCache(0)
    cache_mod.save_prompt_cache = lambda path, cache: saved.__setitem__(path, cache.len)
    cache_mod.load_prompt_cache = _load

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


def _valid_safetensors_bytes() -> bytes:
    header = b'{"__metadata__":{}}'
    return len(header).to_bytes(8, "little") + header


def _write_blob(path, tokens):
    meta = blob.CheckpointMeta(
        fingerprint=blob.Fingerprint(
            model_id="fake", tokenizer_hash="h", kv_dtype="f16", kv_layout_version=1
        ),
        token_count=len(tokens),
        tokens=tokens,
    )
    with open(path, "wb") as fh:
        fh.write(blob.encode(meta, _valid_safetensors_bytes()))
    return path


def test_prefill_seeds_the_cache_with_tokens_minus_one(monkeypatch):
    # The saved cache must cover tokens[:-1]. Caching all tokens makes the hit
    # path re-feed the last token and diverge from a scratch run.
    prefilled, model = _install_fake_mlx(monkeypatch)
    eng = _engine(model)
    eng.prefill([10, 20, 30, 40])
    assert prefilled == [3]


def test_single_token_prefill_caches_nothing(monkeypatch):
    # A one-token prompt has an empty prefix; mlx-lm cannot save/load an empty
    # prompt cache, and there is nothing to cache. prefill must not call the model
    # and must return an empty payload so the daemon skips publishing.
    prefilled, model = _install_fake_mlx(monkeypatch)
    eng = _engine(model)
    payload = eng.prefill([7])
    assert prefilled == [], "a one-token seed must not call the model"
    assert payload == b"", "a one-token prefill must cache nothing"


def test_resume_feeds_from_one_before_the_cached_count(monkeypatch, tmp_path):
    _install_fake_mlx(monkeypatch)
    eng = _engine()
    tokens = [1, 2, 3, 4, 5]
    blob_path = _write_blob(str(tmp_path / "b.ckpt"), tokens[:4])
    _cache, prompt = eng._load_cache_delta(tokens, blob_path)
    # The cache covers prefix[:-1] = 3 tokens, so feed tokens[3:] and the model
    # predicts token 5 from the KV of tokens 1..3.
    assert prompt == [4, 5]


def test_seed_plus_delta_covers_the_whole_prompt(monkeypatch, tmp_path):
    # The end-to-end invariant: prefill(tokens) seeds len(tokens)-1, resume feeds
    # the uncovered tail, and together they cover every token exactly once.
    # Checked for a multi-token prompt AND a one-token prompt (regression: the
    # one-token case used to feed the token twice).
    prefilled, model = _install_fake_mlx(monkeypatch)
    eng = _engine(model)
    for tokens in ([1, 2, 3, 4, 5, 6], [7]):
        prefilled.clear()
        eng.prefill(tokens)
        seeded = prefilled[-1] if prefilled else 0
        blob_path = _write_blob(str(tmp_path / f"b{len(tokens)}.ckpt"), tokens)
        _cache, prompt = eng._load_cache_delta(tokens, blob_path)
        assert seeded + len(prompt) == len(tokens), (
            f"tokens={tokens} seeded={seeded} delta={prompt}"
        )


def test_empty_payload_blob_is_not_adopted(monkeypatch, tmp_path):
    # A checkpoint with no KV payload (a one-token prompt) must run from scratch,
    # not attempt to load an empty mlx-lm cache.
    _install_fake_mlx(monkeypatch)
    eng = _engine()
    tokens = [7]
    meta = blob.CheckpointMeta(
        fingerprint=blob.Fingerprint(
            model_id="fake", tokenizer_hash="h", kv_dtype="f16", kv_layout_version=1
        ),
        token_count=len(tokens),
        tokens=tokens,
    )
    path = str(tmp_path / "empty.ckpt")
    with open(path, "wb") as fh:
        fh.write(blob.encode(meta, b""))
    cache, prompt = eng._load_cache_delta(tokens, path)
    assert cache is None
    assert prompt == tokens


def test_truncated_multi_token_blob_raises_for_quarantine(monkeypatch, tmp_path):
    # Regression (structured Codex review): a MULTI-token checkpoint truncated to
    # a valid header and empty payload is corrupt, not uncacheable. The loader
    # must reject it (CheckpointRejectedError -> 422) so the daemon quarantines it.
    # Swallowing it as scratch would report a hit and recompute forever.
    _install_fake_mlx(monkeypatch)
    eng = _engine()
    tokens = [1, 2, 3]
    meta = blob.CheckpointMeta(
        fingerprint=blob.Fingerprint(
            model_id="fake", tokenizer_hash="h", kv_dtype="f16", kv_layout_version=1
        ),
        token_count=len(tokens),
        tokens=tokens,
    )
    path = str(tmp_path / "truncated.ckpt")
    with open(path, "wb") as fh:
        fh.write(blob.encode(meta, b""))  # header valid, KV missing
    with pytest.raises(server.CheckpointRejectedError):
        eng._load_cache_delta(tokens, path)


def test_corrupt_safetensors_load_is_rejected_not_500(monkeypatch, tmp_path):
    # Regression (Codex adversarial): a multi-token blob with a nonempty but
    # corrupt payload makes load_prompt_cache raise. That is a bad checkpoint, so
    # it must become CheckpointRejectedError (422 -> daemon retires it), not a
    # generic 500 that leaves the poison Published and retried forever.
    _install_fake_mlx(monkeypatch)

    def _boom(_path):
        raise ValueError("invalid safetensors header")

    sys.modules["mlx_lm.models.cache"].load_prompt_cache = _boom
    eng = _engine()
    tokens = [1, 2, 3]
    path = _write_blob(str(tmp_path / "corrupt.ckpt"), tokens)
    with pytest.raises(server.CheckpointRejectedError):
        eng._load_cache_delta(tokens, path)


def test_malformed_header_is_rejected_not_500(monkeypatch, tmp_path):
    # Regression (Codex adversarial): a length-prefixed but schema-less header
    # (e.g. {}) makes the JSON decode produce missing fields -> KeyError/TypeError.
    # That is a bad checkpoint; it must be CheckpointRejectedError (422), not a
    # 500 that leaves it selectable forever.
    _install_fake_mlx(monkeypatch)
    eng = _engine()
    tokens = [1, 2, 3]
    path = str(tmp_path / "bad-header.ckpt")
    body = b"{}"
    with open(path, "wb") as fh:
        fh.write(len(body).to_bytes(4, "little") + body + b"payload")
    with pytest.raises(server.CheckpointRejectedError):
        eng._load_cache_delta(tokens, path)


def test_transient_read_error_is_not_quarantined(monkeypatch, tmp_path):
    # Regression (Codex adversarial): a transient read failure (EIO/ENFILE) is
    # NOT a bad checkpoint. It must propagate as-is (-> 500), not be converted to
    # CheckpointRejectedError, so the daemon does not retire a healthy entry.
    _install_fake_mlx(monkeypatch)
    eng = _engine()
    tokens = [1, 2, 3]
    path = _write_blob(str(tmp_path / "healthy.ckpt"), tokens)

    real_open = open

    def _flaky_open(p, *a, **k):
        if p == path:
            raise OSError(5, "Input/output error")
        return real_open(p, *a, **k)

    monkeypatch.setattr("builtins.open", _flaky_open)
    with pytest.raises(OSError):
        eng._load_cache_delta(tokens, path)


def test_invalid_safetensors_rejected_before_native_load(monkeypatch, tmp_path):
    # Regression (Codex pass 4): MLX's native parser raises RuntimeError for a bad
    # safetensors header, the same type as a transient read failure. We validate
    # the framing ourselves, so a payload with a bogus header length is rejected
    # (422) even though RuntimeError from the loader below is treated as
    # transient. The loader is monkeypatched to raise RuntimeError to prove the
    # structural check runs first.
    _install_fake_mlx(monkeypatch)

    def _native_boom(_path):
        raise RuntimeError("safetensors: invalid header length")

    sys.modules["mlx_lm.models.cache"].load_prompt_cache = _native_boom
    eng = _engine()
    tokens = [1, 2, 3]
    meta = blob.CheckpointMeta(
        fingerprint=blob.Fingerprint(
            model_id="fake", tokenizer_hash="h", kv_dtype="f16", kv_layout_version=1
        ),
        token_count=len(tokens),
        tokens=tokens,
    )
    path = str(tmp_path / "bad-inner.ckpt")
    with open(path, "wb") as fh:
        fh.write(blob.encode(meta, b"\x00" * 8))  # header length 0 -> invalid
    with pytest.raises(server.CheckpointRejectedError):
        eng._load_cache_delta(tokens, path)


def test_unhashable_dtype_is_rejected_not_500(monkeypatch, tmp_path):
    # Regression (Codex pass 7): a safetensors header with "dtype":[] or {}
    # must be a 422, not a TypeError escaping to a 500. The validator checks the
    # dtype is a str before the dict membership test.
    _install_fake_mlx(monkeypatch)
    eng = _engine()
    tokens = [1, 2, 3]
    for bad_dtype in ([], {}):
        header = json.dumps(
            {"w": {"dtype": bad_dtype, "shape": [1], "data_offsets": [0, 2]}}
        ).encode()
        payload = len(header).to_bytes(8, "little") + header + b"\x00\x00"
        meta = blob.CheckpointMeta(
            fingerprint=blob.Fingerprint(
                model_id="fake", tokenizer_hash="h", kv_dtype="f16", kv_layout_version=1
            ),
            token_count=len(tokens),
            tokens=tokens,
        )
        path = str(tmp_path / "bad-dtype.ckpt")
        with open(path, "wb") as fh:
            fh.write(blob.encode(meta, payload))
        with pytest.raises(server.CheckpointRejectedError):
            eng._load_cache_delta(tokens, path)


def test_blob_for_a_different_prefix_is_rejected(monkeypatch, tmp_path):
    # A blob whose recorded prefix does not match the request must not be
    # adopted: resuming from the wrong KV generates silently wrong output. It is
    # bad (mislabeled/corrupt), so reject it for quarantine, not silent scratch.
    _install_fake_mlx(monkeypatch)
    eng = _engine()
    tokens = [1, 2, 3, 4]
    blob_path = _write_blob(str(tmp_path / "b.ckpt"), [9, 9, 9, 9])
    with pytest.raises(server.CheckpointRejectedError):
        eng._load_cache_delta(tokens, blob_path)


def test_blob_without_recorded_prefix_is_rejected(monkeypatch, tmp_path):
    # The daemon always writes meta.tokens, so a matched blob WITHOUT them is
    # corrupt: it cannot be verified to cover this request's prefix, and adopting
    # it would trust token_count alone (any prefix of that length passes).
    # Silently running scratch while the daemon reports a hit is an accounting
    # lie, so reject it for retirement.
    _install_fake_mlx(monkeypatch)
    eng = _engine()
    tokens = [1, 2, 3, 4, 5]
    for count in (0, -1, 4, 9):
        meta = blob.CheckpointMeta(
            fingerprint=blob.Fingerprint(
                model_id="fake", tokenizer_hash="h", kv_dtype="f16", kv_layout_version=1
            ),
            token_count=count,
            tokens=[],
        )
        path = str(tmp_path / f"old{count}.ckpt")
        with open(path, "wb") as fh:
            fh.write(blob.encode(meta, b"payload"))
        with pytest.raises(server.CheckpointRejectedError):
            eng._load_cache_delta(tokens, path)


def test_recorded_prefix_wins_over_token_count(monkeypatch, tmp_path):
    # When a blob records meta.tokens, its length defines the prefix; token_count
    # is ignored for the delta. A blob whose token_count disagrees must still
    # resume from tokens[len(prefix)-1:], not token_count.
    _install_fake_mlx(monkeypatch)
    eng = _engine()
    tokens = [1, 2, 3, 4, 5]
    meta = blob.CheckpointMeta(
        fingerprint=blob.Fingerprint(
            model_id="fake", tokenizer_hash="h", kv_dtype="f16", kv_layout_version=1
        ),
        token_count=4,  # stale/lying; the recorded prefix is authoritative
        tokens=[1, 2, 3],
    )
    path = str(tmp_path / "mismatch.ckpt")
    with open(path, "wb") as fh:
        fh.write(blob.encode(meta, _valid_safetensors_bytes()))
    _cache, prompt = eng._load_cache_delta(tokens, path)
    # prefix = [1,2,3] -> covered = 2 -> delta = tokens[2:]
    assert prompt == [3, 4, 5]


def test_legacy_one_token_checkpoint_is_not_adopted(monkeypatch, tmp_path):
    # Regression (Codex P1): a pre-fix cache dir holds a NONEMPTY one-token
    # checkpoint. It must be rejected on prefix length, not payload emptiness,
    # or the hit path adopts it and feeds the single token twice.
    _install_fake_mlx(monkeypatch)
    eng = _engine()
    tokens = [7]
    meta = blob.CheckpointMeta(
        fingerprint=blob.Fingerprint(
            model_id="fake", tokenizer_hash="h", kv_dtype="f16", kv_layout_version=1
        ),
        token_count=1,
        tokens=tokens,
    )
    path = str(tmp_path / "legacy.ckpt")
    with open(path, "wb") as fh:
        fh.write(blob.encode(meta, b"NONEMPTY_CACHE_BYTES"))
    cache, prompt = eng._load_cache_delta(tokens, path)
    assert cache is None, "a one-token checkpoint must never be adopted"
    assert prompt == tokens


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
