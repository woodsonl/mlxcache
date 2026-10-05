"""B1.2 wrapper conformance — PersistentPromptCache + the serve CLI.

The mlx-lm codec is injected (fake save/load) so these tests exercise
the WRAPPER's contracts: which cache wins, the covered-positions math,
persistence triggers, failure fallbacks, and the CLI seam. Real-engine
rows live with the gated suites (B2.1).
"""

import copy
import sys

import pytest

# The wrapper subclasses mlx-lm's cache — importable only where mlx-lm is
# (Apple Silicon). CI (ubuntu) skips this module; the gated real-engine
# suite covers the live behavior.
pytest.importorskip("mlx_lm")

from mlx_lm.models.cache import LRUPromptCache  # noqa: E402
from mlxcache_serve import ENGINE_ID, PersistentPromptCache, fingerprint_for_model_dir, main
from mlxcache_store import Fingerprint, Store

FP = Fingerprint("model-a", "tok-a", "mlx-lm", 1)
MODEL = ("model-a", None, None)  # mlx-lm's model_key shape

FAKE_STATE = {}  # payload -> cache marker (the injected codec's "disk")


def fake_save(cache):
    payload = f"payload-for-{cache.marker}@{cache[0].positions}".encode()
    FAKE_STATE[payload] = cache
    return payload


def fake_load(payload):
    return copy.deepcopy(FAKE_STATE[payload])


class _KV:
    """One KV layer: needs .nbytes for the LRU byte accounting."""

    def __init__(self, marker: str):
        self.marker = marker
        self.nbytes = 8
        self.positions = 4  # simulated KV offset (all of T in these tests)

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        n = min(self.positions, n)
        self.positions -= n
        return n


class FakeCache(list):
    """Deep-copyable stand-in for a prompt cache (a list of KV layers)."""

    def __init__(self, marker: str):
        super().__init__([_KV(marker)])

    @property
    def marker(self) -> str:
        return self[0].marker


def make(tmp_path, **kw):
    store = Store(tmp_path, byte_budget=0)
    return PersistentPromptCache(store, FP, save_fn=fake_save, load_fn=fake_load, **kw)


# --- persistence ------------------------------------------------------------


def test_insert_persists_finished_assistant_streams(tmp_path):
    c = make(tmp_path)
    c.insert_cache(MODEL, [1, 2, 3, 4], FakeCache("full-stream"))
    assert c.persisted == 1
    matched, ref = c.store.lookup(ENGINE_ID, FP, [1, 2, 3, 4, 5])
    assert (matched, ref) == (4, ref)
    assert c.store.fetch(ref) == b"payload-for-full-stream@3", (
        "§3.2: the saved state covers T[:-1] (trimmed exactly one position)"
    )


def test_segment_and_undersized_inserts_are_not_persisted(tmp_path):
    c = make(tmp_path)
    c.insert_cache(MODEL, [1, 2, 3], FakeCache("segment"), cache_type="system")
    c.insert_cache(MODEL, [1], FakeCache("one-token"), cache_type="assistant")
    assert c.persisted == 0
    assert list(tmp_path.glob("*.ckpt")) == []


def test_persist_failure_never_breaks_serving(tmp_path):
    def boom(_cache):
        raise RuntimeError("save exploded")

    c = make(tmp_path)
    c._save = boom
    c.insert_cache(MODEL, [1, 2, 3], FakeCache("x"))  # must not raise
    assert c.persist_errors == 1 and c.persisted == 0


# --- fetch: which cache wins -------------------------------------------------


def _seed_disk(store, tokens, marker):
    # Persist the marker at `tokens` in §3.2 convention via the fake codec.
    payload = f"payload-for-{marker}@{len(tokens) - 1}".encode()
    FAKE_STATE[payload] = FakeCache(marker)
    store.put(ENGINE_ID, FP, list(tokens), payload)


def test_exact_hit_edge_whole_context_like(tmp_path):
    """matched == len(tokens): the disk payload covers len-1 positions, so
    the resume remainder is EXACTLY the last token (never empty, never 0)."""
    c = make(tmp_path)
    _seed_disk(c.store, [1, 2, 3], "disk-full")
    cache, rest = c.fetch_nearest_cache(MODEL, [1, 2, 3])
    assert cache.marker == "disk-full"
    assert rest == [3], "covered=2 → feed tokens[2:] — a double-feed here is F1"
    assert c.disk_hits == 1


def test_matched_under_two_never_served(tmp_path):
    c = make(tmp_path)
    _seed_disk(c.store, [1, 2], "tiny")
    cache, rest = c.fetch_nearest_cache(MODEL, [1, 2, 3])
    # matched=2 covers 1 position; upstream memory covers 0 → disk wins.
    # covered=1 → feed [2, 3].
    assert cache.marker == "tiny" and rest == [2, 3]


# --- fingerprint -------------------------------------------------------------


def test_fingerprint_determinism_and_inputs(tmp_path, tmp_path_factory):
    d1 = tmp_path_factory.mktemp("model")
    tok = d1 / "tokenizer.json"
    tok.write_bytes(b'{"tok": true}')
    d2 = tmp_path_factory.mktemp("model2")
    (d2 / "tokenizer.json").write_bytes(b'{"tok": false}')

    fa = fingerprint_for_model_dir(d1)
    fb = fingerprint_for_model_dir(d1)
    assert fa == fb, "same dir → same fingerprint (stable across restarts)"
    assert fa.model_id.endswith(d1.name) or fa.model_id == str(d1)
    assert fa.kv_dtype == "mlx-lm" and fa.kv_layout_version == 1

    fc = fingerprint_for_model_dir(d2)
    assert fc.tokenizer_hash != fa.tokenizer_hash, "tokenizer bytes matter"

    fq = fingerprint_for_model_dir(d1, kv_bits=8, kv_group_size=64)
    assert fq.kv_bits == 8 and fq != fa, "quantization tier namespaces the key"

    fo = fingerprint_for_model_dir(d1, model_id_override="daemon-model")
    assert fo.model_id == "daemon-model"


# --- CLI seam ----------------------------------------------------------------


def test_cli_swaps_the_cache_factory_and_restores(monkeypatch, tmp_path):
    import mlx_lm.server as mlx_server

    captured = {}

    def fake_main():
        factory = mlx_server.LRUPromptCache
        cache = factory(3)
        captured["is_persistent"] = isinstance(cache, PersistentPromptCache)
        captured["max_size"] = cache.max_size
        captured["store_dir"] = str(cache.store.dir)
        # The CLI wires the REAL codec (no injection point); persistence
        # semantics are covered by the cache-level tests above.
        captured["fingerprint"] = cache.fingerprint(MODEL)

    monkeypatch.setattr(mlx_server, "main", fake_main)
    rc = main(["--store-dir", str(tmp_path / "s"), "--kv-bits", "8", "--", "--model", "m"])
    assert rc == 0
    assert captured["is_persistent"] and captured["max_size"] == 3
    assert captured["fingerprint"].kv_bits == 8, "--kv-bits flows into the fingerprint"
    assert (tmp_path / "s").is_dir()
    assert mlx_server.LRUPromptCache is LRUPromptCache, "symbol restored on exit"


def test_cli_no_separator_passthrough(monkeypatch, tmp_path):
    """--model must reach mlx-lm verbatim: allow_abbrev would fold it into
    --model-id-override and silently repartition the store."""
    import mlx_lm.server as mlx_server

    captured = {}

    def fake_main():
        captured["argv"] = list(sys.argv)

    monkeypatch.setattr(mlx_server, "main", fake_main)
    rc = main(["--store-dir", str(tmp_path / "s"), "--model", "my-model", "--port", "9000"])
    assert rc == 0
    assert captured["argv"] == ["mlx_lm.server", "--model", "my-model", "--port", "9000"], (
        "no separator: mlx-lm receives the model"
    )


def test_cli_separator_passthrough(monkeypatch, tmp_path):
    import mlx_lm.server as mlx_server

    captured = {}

    def fake_main():
        captured["argv"] = list(sys.argv)

    monkeypatch.setattr(mlx_server, "main", fake_main)
    rc = main(["--store-dir", str(tmp_path / "s"), "--", "--model", "m"])
    assert rc == 0
    assert captured["argv"] == ["mlx_lm.server", "--model", "m"], (
        "py3.14 parse_known_args keeps the leading --; we strip exactly one"
    )


def test_adapter_switch_isolates_fingerprint(tmp_path):
    """LoRA-B must never serve LoRA-A's KV: the default fingerprint folds
    adapter and draft model into the model id."""
    from mlxcache_serve import default_fingerprint_for

    fp_a = default_fingerprint_for(("model-a", "lora-a", None))
    fp_b = default_fingerprint_for(("model-a", "lora-b", None))
    assert fp_a != fp_b and fp_a.model_id != fp_b.model_id
    fp_draft = default_fingerprint_for(("model-a", None, "draft"))
    assert fp_draft.model_id != fp_a.model_id, "draft-model switch isolates too"
    assert default_fingerprint_for(("model-a", None, None)) is default_fingerprint_for(
        ("model-a", None, None)
    ), "memoized: same key → same fingerprint object"


def test_cli_requires_store_dir():
    with pytest.raises(SystemExit):
        main(["--model", "m"])


def test_help_lists_our_flags(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    for flag in ("--store-dir", "--kv-bits", "--trust-legacy", "--anchor-window-s"):
        assert flag in out
