#!/usr/bin/env python3
"""Wrapper benchmark: persistent KV reuse through mlxcache_serve.

Drives the REAL mlx_lm.server (wrapped by mlxcache_serve) over five legs
against a local mlx-lm model:

  cold-turn1    : empty store -> full-price prefill
  warm-turn1    : same process, same request again -> in-memory exact hit
  memory-turn2  : SAME process, conversation GROWN with turn 1's reply ->
                  resumes from the in-memory trie (the reference lineage)
  restart-turn2 : SIGTERM, fresh process, SAME store -> the disk tier must
                  reproduce the memory reference (the §3.2 guard)
  second-turn2  : a second fresh process, SAME store -> determinism check

Correctness gate: every disk leg's answer must be token-identical to the
memory-turn2 reference. A coverage off-by-one on the write path shifts only
the disk legs, so this comparison catches it; comparing against a fresh
batch prefill instead would flake on near-tied argmax at temp 0.

Reuse gate: token identity alone cannot prove the disk tier served the
request (a full prefill is token-identical at temp 0). Each disk leg also
reads /mlxcache/stats and must report disk_hits > 0, so a gate cannot pass
while the store is merely populated and never read.

Metrics per leg: wall_ms, text, verdict (token-identical | DIVERGED),
reused (coverage + observed disk_hits + verdict notes).

Output: JSON artifact (bench/wrapper-bench-<ts>.json) + a markdown table on
stdout. The model must be a local absolute snapshot path or a cached HF id,
resolved BEFORE launch so the bench and the server child agree on the store
fingerprint; the 7B-4bit default needs ~4GB and Apple Silicon.

This harness measures the WRAPPER path's disk tier. It is not the prefix-
caching benchmark: see scripts/bench_prefix.py (it lands with the
prefix-benchmark unit of this change stack) for declared-vs-served prefix
accounting and the cold/warm/pressure grid.

Usage:
  uv run python scripts/bench_wrapper.py [--model PATH_OR_HF_ID] [--port N]
      [--max-tokens N] [--out PATH]

Env: MLXCACHE_BENCH_MODEL overrides the model default.
"""

from __future__ import annotations

import argparse
import atexit
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _harness import (  # noqa: E402 — path insert above
    DEFAULT_MODEL,
    REPO,
    WrapperServer,
    free_port,
    resolve_model,
)

TURN1 = {
    "messages": [
        {"role": "user", "content": "Explain the theory of persistent KV cache reuse. " * 24}
    ],
    "temperature": 0.0,
    "max_tokens": 24,
}


class Server(WrapperServer):
    """One mlxcache_serve pair, warmup-checked through a real chat call."""

    def __init__(self, store_dir: str, model: str, port: int):
        super().__init__(
            store_dir,
            model,
            port,
            log_prefix="mlxcache-bench-srv-",
            warmup="chat",
        )


# Turn 2 GROWS the conversation: the request stream extends turn 1's stored
# prompt+response stream, which is the shape the protocol can serve from disk
# (an entry that extends the request is deliberately NOT a match — §3.3).
# A re-posted identical single-turn request can NEVER disk-hit; benching one
# would measure the wrong thing.
def turn2_body(resp_a: str, max_tokens: int) -> dict:
    return {
        "messages": TURN1["messages"]
        + [
            {"role": "assistant", "content": resp_a},
            {"role": "user", "content": "Summarize those properties in one sentence."},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }


def _leg_result(name, wall_ms, text, baseline, extra=None):
    row = {
        "leg": name,
        "wall_ms": round(wall_ms, 1),
        "text": text,
        "verdict": "token-identical" if baseline is None or text == baseline else "DIVERGED",
        "reused": extra or {},
    }
    if baseline is not None and text != baseline:
        # First divergence point for the report.
        for i, (a, b) in enumerate(zip(baseline, text, strict=False)):
            if a != b:
                row["reused"]["first_divergence_char"] = i
                break
    return row


def _mlxcache_tokenize(model: str, body: dict) -> list[int]:
    """The exact token stream the server computes for this chat body
    (chat template, no system prompt), so the store lookup matches."""
    from mlx_lm.utils import load_tokenizer

    tok = load_tokenizer(model)
    return tok.apply_chat_template(body["messages"], add_generation_prompt=True)


def _fp_for(model: str):
    from mlxcache_serve import default_fingerprint_for

    return default_fingerprint_for((model, None, None))


def _gate_ok(results, skip_pct, observed_hits, reuse_errors=None) -> tuple[bool, list[str]]:
    """The acceptance predicate. Returns (ok, reasons). A reusable blob
    is not reuse: the disk legs must have OBSERVED a disk hit, so a store
    that is populated but never read cannot pass. The hits are DELTAS across
    each measured leg (the counters are process-cumulative and the warmup
    probe can itself hit the disk tier), and a leg that logged reuse errors
    fails even when its hit count is positive."""
    reasons = []
    bad = [r["leg"] for r in results if r["verdict"] != "token-identical"]
    if bad:
        reasons.append(f"token divergence in {bad}")
    if skip_pct < 90:
        reasons.append(f"prefill skip {skip_pct}% < 90%")
    reused = all(v is not None and v > 0 for v in observed_hits.values())
    if not reused:
        reasons.append(f"disk tier not observed serving (disk_hits={observed_hits})")
    errored = {k: v for k, v in (reuse_errors or {}).items() if v}
    if errored:
        reasons.append(f"reuse errors on measured legs: {errored}")
    return (not reasons), reasons


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--port", type=int, default=free_port())
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    model = resolve_model(args.model)
    if args.port == 0:
        # A literal 0 would be passed to the child, which then binds a
        # kernel-assigned port while we probe 127.0.0.1:0 until the 180s
        # deadline; pick the port here instead.
        args.port = free_port()
    TURN1["max_tokens"] = args.max_tokens

    print(f"model: {model}")

    results: list[dict] = []
    tmp = tempfile.mkdtemp(prefix="mlxcache-bench-")
    # The store holds GB-scale checkpoints at real-model sizes; strand it in
    # $TMPDIR on every run and the machine eventually fills the volume the
    # harness itself needs. atexit covers normal return, SystemExit and
    # KeyboardInterrupt; a SIGTERM/SIGKILL mid-run still strands the dir —
    # the known, accepted limit of this cleanup.
    atexit.register(shutil.rmtree, tmp, True)
    store_dir = str(Path(tmp) / "store")

    # ---- one server serves turn 1 (cold + warm) AND the memory turn-2
    # reference, so the in-memory trie is genuinely warm for turn 2. A fresh
    # process here would take the same DISK path as the legs it guards and
    # the comparison would be disk-vs-disk — blind to a writer-side §3.2
    # off-by-one.
    srv = Server(store_dir, model, args.port)
    try:
        srv.wait_ready()
        wall_cold, text_cold = srv.chat(TURN1)
        results.append(_leg_result("cold-turn1", wall_cold, text_cold, None))
        baseline1 = text_cold
        wall_warm, text_warm = srv.chat(TURN1)
        results.append(_leg_result("warm-turn1", wall_warm, text_warm, baseline1))

        body2 = turn2_body(text_cold, args.max_tokens)
        wall_mem2, text_mem2 = srv.chat(body2)
        results.append(_leg_result("memory-turn2", wall_mem2, text_mem2, None))
        baseline2 = text_mem2
    finally:
        srv.stop()

    # Snapshot the store's answer for body2 BEFORE any restart leg runs, so
    # the lookup provably reads the turn-1 state the legs resume from (each
    # leg publishes its own longer stream; today the index cannot match a
    # longer entry, so ordering is belt-and-braces against future index
    # changes, not a live self-match bug). The gate that cannot be fooled is
    # the disk_hits DELTA below.
    from mlxcache_store import Store as _Store

    def _delta(pre: dict, post: dict, key: str) -> int:
        # `or 0` coercion would report "0 hits" when the counter is actually
        # MISSING from /mlxcache/stats — the opposite of what a gate wants
        # to hear. Fail loudly instead.
        if pre.get(key) is None or post.get(key) is None:
            raise RuntimeError(f"/mlxcache/stats did not report {key!r}: cannot gate on it")
        return post[key] - pre[key]

    st = _Store(store_dir, byte_budget=0)
    tok = _mlxcache_tokenize(model, body2)
    fp = _fp_for(model)
    matched, _ref = st.lookup("mlx-lm", fp, tok)
    n_blobs = len(list(Path(store_dir).glob("*.ckpt")))
    if n_blobs == 0:
        raise SystemExit(
            "no cache entries persisted: the writer stored nothing "
            "(can_trim_prompt_cache false, non-trimmable cache, or a "
            "failed publish) — the gate cannot measure reuse"
        )
    if matched == 0:
        raise SystemExit(
            f"fingerprint mismatch: store has {n_blobs} entries but the "
            "lookup missed — the gate would report 0% skip on a working "
            "system (model/path resolution differs between bench and server)"
        )
    covered = max(matched - 1, 0)
    skip_pct = round(100 * covered / max(len(tok) - 1, 1))

    # ---- restart turn 2 (fresh process, SAME store): the request stream
    # extends the persisted turn-1 stream, so the disk tier should cover the
    # whole first exchange minus one position (§3.2). Compared against the
    # MEMORY reference.
    with Server(store_dir, model, args.port) as srv2:
        srv2.wait_ready()
        # disk_hits is process-CUMULATIVE and the warmup itself can hit the
        # disk tier; gate on the DELTA across the measured request, or the
        # warmup alone could satisfy a "store was read" gate.
        pre = srv2.stats() or {}
        wall_re, text_re = srv2.chat(body2)
        post = srv2.stats() or {}
        restart_stats = {
            "disk_hits": _delta(pre, post, "disk_hits"),
            "reuse_errors": _delta(pre, post, "reuse_errors"),
        }
        results.append(_leg_result("restart-turn2", wall_re, text_re, baseline2, restart_stats))

    # ---- second process (another pair, SAME store): determinism ----------
    with Server(store_dir, model, args.port) as srv3:
        srv3.wait_ready()
        pre = srv3.stats() or {}
        wall_2nd, text_2nd = srv3.chat(body2)
        post = srv3.stats() or {}
        second_stats = {
            "disk_hits": _delta(pre, post, "disk_hits"),
            "reuse_errors": _delta(pre, post, "reuse_errors"),
        }
        results.append(_leg_result("second-turn2", wall_2nd, text_2nd, baseline2, second_stats))

    # ---- acceptance ------------------------------------------------------
    # Prefill-skip, measured as WORK not wall, from the PRE-LEG snapshot
    # above: a disk hit covers (matched-1) positions of the turn-1 prefix,
    # so the prefill actually computed is the remainder. The disk_hits
    # DELTAS prove the restart legs actually took a blob from disk rather
    # than re-prefilling (at temp 0 the two are token-identical, so the
    # token verdict alone cannot).
    observed_hits = {
        "restart-turn2": (restart_stats or {}).get("disk_hits"),
        "second-turn2": (second_stats or {}).get("disk_hits"),
    }
    reuse_errors = {
        "restart-turn2": (restart_stats or {}).get("reuse_errors"),
        "second-turn2": (second_stats or {}).get("reuse_errors"),
    }
    for r in results:
        if r["leg"] in ("restart-turn2", "second-turn2"):
            r["reused"]["prefill_skip_pct"] = skip_pct
            r["reused"]["request_tokens"] = len(tok)
            r["reused"]["disk_covered"] = covered
    # A blob existing is not reuse. At temperature 0 a full prefill and a
    # disk resume are token-identical, so the token verdict cannot prove the
    # disk tier served anything; the delta can. Require an observed hit and
    # zero reuse errors on the measured legs.
    ok, reasons = _gate_ok(results, skip_pct, observed_hits, reuse_errors)

    out_path = args.out or str(REPO / "bench" / f"wrapper-bench-{int(time.time())}.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps({"model": model, "results": results}, indent=2))

    print("\n| leg | wall_ms | verdict | notes |")
    print("|-----|---------|---------|-------|")
    for r in results:
        notes = json.dumps(r["reused"]) if r["reused"] else ""
        print(f"| {r['leg']} | {r['wall_ms']} | {r['verdict']} | {notes} |")
    print(f"\nartifact: {out_path}")
    if ok:
        print("GATE: PASS")
    else:
        print("GATE: FAIL —", "; ".join(reasons))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
