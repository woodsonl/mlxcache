"""B1.2 QA live probes — the wrapper's public contracts, real codec-free.

Fake save/load (positional-blind) is intentional for CLI/fallback rows;
the positional convention is guarded by tests/test_serve_real.py (gated).
"""

import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "sidecar"))

from mlxcache_store import Fingerprint, Store  # noqa: E402
from mlxcache_serve import ENGINE_ID, PersistentPromptCache  # noqa: E402

FP = Fingerprint("qa-model", "qa-tok", "mlx-lm", 1)
MODEL = ("model-a", None, None)
RESULTS = []


class _KV:
    def __init__(self, marker):
        self.marker = marker
        self.nbytes = 8
        self.positions = 16

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.positions, n)
        self.positions -= n
        return n


class FakeCache(list):
    def __init__(self, marker):
        super().__init__([_KV(marker)])

    @property
    def marker(self):
        return self[0].marker


def probe(name, contract):
    def wrap(fn):
        try:
            fn()
            RESULTS.append((name, contract, "pass", ""))
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001
            RESULTS.append((name, contract, "fail", repr(exc)))
            print(f"FAIL {name}: {exc!r}")
        return fn

    return wrap


def expect(cond, msg):
    if not cond:
        raise AssertionError(msg)


@probe("W1 persistence round trip", "successful execution")
def _():
    with tempfile.TemporaryDirectory() as d:
        store = Store(d, byte_budget=0)
        payloads = {}
        pc = PersistentPromptCache(
            store, FP,
            save_fn=lambda c: payloads.setdefault(c.marker, c.marker.encode()),
            load_fn=lambda b: FakeCache(b.decode()),
        )
        pc.insert_cache(MODEL, [1, 2, 3, 4, 5], FakeCache("turn-a"))
        expect(pc.persisted == 1, f"persisted {pc.persisted}")
        fresh = PersistentPromptCache(
            Store(d, byte_budget=0), FP,
            save_fn=lambda c: b"", load_fn=lambda b: FakeCache(b.decode()),
        )
        cache, rest = fresh.fetch_nearest_cache(MODEL, [1, 2, 3, 4, 5, 6])
        expect(cache.marker == "turn-a" and rest == [5, 6], f"resume {(cache.marker, rest)}")


@probe("W2 invalid input", "invalid/missing input — CLI rejects missing store dir")
def _():
    r = subprocess.run(
        [sys.executable, "-m", "mlxcache_serve", "--model", "x"],
        capture_output=True, text=True, cwd=str(REPO / "sidecar"),
        env={"PATH": "/usr/bin:/bin", **dict(__import__("os").environ)},
    )
    expect(r.returncode != 0, "missing --store-dir must fail")
    expect("store-dir" in r.stderr, f"usage names the flag: {r.stderr[-200:]}")


@probe("W3 isolation", "authorization boundary — adapter switch isolates")
def _():
    from mlxcache_serve import default_fingerprint_for

    a = default_fingerprint_for(("m", "lora-a", None))
    b = default_fingerprint_for(("m", "lora-b", None))
    expect(a.model_id != b.model_id, "LoRA isolation")


@probe("W4 state transitions", "WHOLE_CONTEXT-shaped resume: rest == last token")
def _():
    with tempfile.TemporaryDirectory() as d:
        store = Store(d, byte_budget=0)
        store.put(ENGINE_ID, FP, [1, 2, 3], b"payload-x")
        pc = PersistentPromptCache(
            store, FP, save_fn=lambda c: b"", load_fn=lambda b: FakeCache("x"),
        )
        cache, rest = pc.fetch_nearest_cache(MODEL, [1, 2, 3])
        expect(rest == [3], f"rest {rest} != [3]")


@probe("W5 fallback", "partial-failure recovery — reuse failure falls back")
def _():
    with tempfile.TemporaryDirectory() as d:
        store = Store(d, byte_budget=0)
        store.put(ENGINE_ID, FP, [1, 2, 3, 4], b"payload-broken")
        pc = PersistentPromptCache(
            store, FP,
            save_fn=lambda c: b"",
            load_fn=lambda b: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        pc.insert_cache(MODEL, [1, 2], FakeCache("mem"))
        cache, rest = pc.fetch_nearest_cache(MODEL, [1, 2, 3, 4, 5])
        expect(cache.marker == "mem" and pc.reuse_errors == 1, "fallback to memory")


@probe("W6 CLI passthrough", "CLI process contract — --model reaches mlx-lm")
def _():
    from mlxcache_serve.serve import build_parser

    args, extra = build_parser().parse_known_args(
        ["--store-dir", "S", "--model", "my-model", "--port", "9000"]
    )
    expect("--model" in extra and "my-model" in extra, f"passthrough {extra}")
    # In-process proof that mlx-lm's main receives the model flag (no
    # abbreviation capture, no separator leak) — a subprocess would make
    # mlx-lm attempt an HF download for an unknown model id.
    import mlx_lm.server as mlx_server

    captured = {}

    def fake_main():
        captured["argv"] = list(sys.argv)

    original_main, original_cache = mlx_server.main, mlx_server.LRUPromptCache
    mlx_server.main = fake_main
    try:
        from mlxcache_serve import main as serve_main

        serve_main(["--store-dir", tempfile.mkdtemp(), "--model", "my-model", "--port", "9000"])
    finally:
        mlx_server.main, mlx_server.LRUPromptCache = original_main, original_cache
    expect(captured["argv"] == ["mlx_lm.server", "--model", "my-model", "--port", "9000"],
           f"mlx-lm received {captured.get('argv')}")


@probe("W7 restart durability", "state across process — store persists, wrapper reuses")
def _():
    d = tempfile.mkdtemp()
    store = Store(d, byte_budget=0)
    payloads = {"turn": b"turn"}
    pc = PersistentPromptCache(
        store, FP, save_fn=lambda c: payloads[c.marker],
        load_fn=lambda b: FakeCache(b.decode()),
    )
    pc.insert_cache(MODEL, [9, 8, 7], FakeCache("turn"))
    out = None
    fresh = PersistentPromptCache(
        Store(d, byte_budget=0), FP, save_fn=lambda c: b"",
        load_fn=lambda b: FakeCache(b.decode()),
    )
    cache, rest = fresh.fetch_nearest_cache(MODEL, [9, 8, 7, 6, 5])
    expect(cache.marker == "turn" and rest == [7, 6, 5], f"restart {(cache.marker, rest)}")


if __name__ == "__main__":
    fails = [r for r in RESULTS if r[2] == "fail"]
    print(f"\n{len(RESULTS) - len(fails)}/{len(RESULTS)} contracts pass")
    sys.exit(1 if fails else 0)
