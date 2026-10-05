"""B1.1 QA live probes — public-contract checks on a real filesystem.

Distinct from pytest: each probe is a documented contract with isolated
setup, recorded evidence, and pass/fail against the PROTOCOL's stated
behavior (docs/connector-protocol.md), not the implementation's.
"""

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sidecar"))

from mlxcache_store import (  # noqa: E402
    CorruptCache,
    Fingerprint,
    Granularity,
    Refused,
    Store,
    Unavailable,
    blob_key,
)

FP = Fingerprint("qa-model", "qa-tok", "f16", 1)
RESULTS = []


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


@probe("P1 round-trip", "successful execution")
def _():
    with tempfile.TemporaryDirectory() as d:
        s = Store(d, byte_budget=0)
        ref = s.put("mlx-lm", FP, [10, 20, 30], b"kv-payload")
        m, r = s.lookup("mlx-lm", FP, [10, 20, 30, 40])
        expect((m, r) == (3, ref), f"lookup {(m, r)} != (3, {ref})")
        expect(s.fetch(r) == b"kv-payload", "payload mismatch")
        expect(Path(d, ref).is_file(), "file not durable")


@probe("P2 invalid inputs", "invalid/missing input")
def _():
    with tempfile.TemporaryDirectory() as d:
        s = Store(d, byte_budget=0)
        try:
            s.put("mlx-lm", FP, [1], b"kv")
            raise AssertionError("sub-2 put accepted")
        except Refused:
            pass
        try:
            s.fetch("../escape")
            raise AssertionError("traversal ref accepted")
        except ValueError:
            pass
        expect(list(Path(d).glob("*.ckpt")) == [], "rejected put left state")


@probe("P3 cross-engine isolation", "authorization boundary")
def _():
    with tempfile.TemporaryDirectory() as d:
        s = Store(d, byte_budget=0)
        a = s.put("mlx-lm", FP, [1, 2, 3], b"mlx")
        b = s.put("llama-cpp", FP, [1, 2, 3], b"llama")
        expect(s.lookup("mlx-lm", FP, [1, 2, 3, 4]) == (3, a), "mlx identity")
        expect(s.lookup("llama-cpp", FP, [1, 2, 3, 4]) == (3, b), "llama identity")
        expect(s.fetch(a) == b"mlx" and s.fetch(b) == b"llama", "payloads crossed")


@probe("P4 granularity", "state transitions")
def _():
    with tempfile.TemporaryDirectory() as d:
        s = Store(d, byte_budget=0)
        w = s.put("llama-cpp", FP, [1, 2, 3, 4], b"whole", Granularity.WHOLE_CONTEXT)
        any_ref = s.put("llama-cpp", FP, [1, 2], b"any", Granularity.ANY_PREFIX)
        expect(s.lookup("llama-cpp", FP, [1, 2, 3, 4]) == (4, w), "whole: exact serves")
        expect(s.lookup("llama-cpp", FP, [1, 2, 3, 4, 5]) == (2, any_ref), "extension falls back")


@probe("P5 end-anchored", "matching semantics §3.3")
def _():
    with tempfile.TemporaryDirectory() as d:
        s = Store(d, byte_budget=0)
        ref = s.put("mlx-lm", FP, [1, 2, 3, 4], b"kv")
        m, r = s.lookup("mlx-lm", FP, [1, 2, 3, 9, 9])
        expect((m, r) == (4, ref), f"end-anchored {(m, r)}")


@probe("P6 supersede idempotency", "duplicates/idempotency")
def _():
    with tempfile.TemporaryDirectory() as d:
        s = Store(d, byte_budget=0)
        s.put("mlx-lm", FP, [5, 6], b"v1")
        r2 = s.put("mlx-lm", FP, [5, 6], b"v2")
        files = list(Path(d).glob("*.ckpt"))
        expect(len(files) == 1, f"{len(files)} durable effects for one key")
        expect(s.fetch(r2) == b"v2", "newest not served")


@probe("P7 cross-process concurrency", "concurrency/order")
def _():
    d = tempfile.mkdtemp()
    child = (
        "import sys\n"
        "sys.path.insert(0, sys.argv[3])\n"
        "from mlxcache_store import Store, Fingerprint\n"
        "fp = Fingerprint('qa-model', 'qa-tok', 'f16', 1)\n"
        "s = Store(sys.argv[1], byte_budget=0)\n"
        "tag = sys.argv[2]\n"
        "for i in range(8):\n"
        "    s.put('mlx-lm', fp, [1, 2], ('w' + tag).encode())\n"  # contended key
        "    s.put('mlx-lm', fp, [9, 9, int(tag)], b'x')\n"  # private key
    )
    child_path = Path(d).parent / "qa_child_writer.py"
    child_path.write_text(child)
    procs = [
        subprocess.Popen([sys.executable, str(child_path), d, str(t), os.getcwd() + "/sidecar"])
        for t in range(4)
    ]
    for p in procs:
        expect(p.wait() == 0, "writer crashed")
    fresh = Store(d, byte_budget=0)
    survivors = list(Path(d).glob("*.ckpt"))
    expect(len(survivors) == 5, f"{len(survivors)} survivors != 5 keys (1 contended + 4 private)")
    for f in survivors:
        fresh.fetch(f.name)  # every survivor digest-valid


@probe("P8 partial-failure recovery", "partial-failure recovery")
def _():
    with tempfile.TemporaryDirectory() as d:
        s = Store(d, byte_budget=0)
        good = s.put("mlx-lm", FP, [1, 2], b"kv")
        # crashed-writer tmp + poison header + digest-flipped blob
        Path(d, f"{good}.tmporphan").write_bytes(b"partial")
        Path(d, "poison.ckpt").write_bytes(b"\x00\x01\x02")
        raw = bytearray(Path(d, good).read_bytes())
        raw[-1] ^= 0xFF
        Path(d, "flip.ckpt").write_bytes(bytes(raw))
        os.utime(Path(d, f"{good}.tmporphan"), (0, 0))  # aged past sweep threshold
        s2 = Store(d, byte_budget=0)  # must not raise
        expect(s2.lookup("mlx-lm", FP, [1, 2, 3]) == (2, good), "good entry lost")
        expect(not Path(d, "poison.ckpt").exists(), "poison not reclaimed")
        expect(not Path(d, "flip.ckpt").exists(), "flip not reclaimed")
        expect(not Path(d, f"{good}.tmporphan").exists(), "aged tmp not swept")


@probe("P9 restart durability", "state transitions across process")
def _():
    d = tempfile.mkdtemp()
    s = Store(d, byte_budget=0)
    ref = s.put("mlx-lm", FP, [7, 8, 9], b"durable")
    out = subprocess.run(
        [sys.executable, "-c",
         f"import sys;sys.path.insert(0,{os.getcwd() + '/sidecar'!r});"
         "from mlxcache_store import Store,Fingerprint;"
         f"s=Store({d!r},byte_budget=0);"
         "print(s.lookup('mlx-lm',Fingerprint('qa-model','qa-tok','f16',1),[7,8,9,10]),"
         f"s.fetch({ref!r}))"],
        capture_output=True, text=True, check=True)
    expect(f"(3, {ref!r})" in out.stdout, f"restart lookup: {out.stdout!r}")
    expect("durable" in out.stdout, "restart payload")


@probe("P10 budget invariant", "resource bound §7")
def _():
    with tempfile.TemporaryDirectory() as d:
        s = Store(d, byte_budget=8192, anchor_window_s=-1.0)
        for i in range(16):
            s.put("mlx-lm", FP, [100 + i, 200 + i], bytes(1024))
        total = sum(p.stat().st_size for p in Path(d).glob("*.ckpt"))
        expect(total <= 8192 + 1200, f"directory {total}B over budget+1blob")


@probe("P11 error taxonomy", "declared rejections §3.4")
def _():
    with tempfile.TemporaryDirectory() as d:
        s = Store(d, byte_budget=0)
        ref = s.put("mlx-lm", FP, [1, 2], b"kv")
        Path(d, ref).unlink()
        try:
            s.fetch(ref)
            raise AssertionError("vanished ref served")
        except Unavailable:
            pass
        s2 = Store(d, byte_budget=0)
        r2 = s2.put("mlx-lm", FP, [1, 2], b"kv2")
        raw = bytearray(Path(d, r2).read_bytes())
        raw[-1] ^= 0xFF
        Path(d, r2).write_bytes(bytes(raw))
        try:
            s2.fetch(r2)
            raise AssertionError("tampered payload served")
        except CorruptCache as exc:
            expect(exc.kind == "digest", f"kind {exc.kind}")
        expect(s2.lookup("mlx-lm", FP, [1, 2, 3]) == (0, None), "no self-heal")


if __name__ == "__main__":
    fails = [r for r in RESULTS if r[2] == "fail"]
    print(f"\n{len(RESULTS) - len(fails)}/{len(RESULTS)} contracts pass")
    sys.exit(1 if fails else 0)
