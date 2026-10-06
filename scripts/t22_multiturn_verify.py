#!/usr/bin/env python3
"""T22 real-engine multi-turn verification (GPU leg).

Drives a GROWING three-turn conversation (~20K-token turn-1 context built from
real repository files) against the daemon at MLXCACHE_BASE and records, per
turn: verdict, tokens_total, tokens_cached, prefill_from, lookup_ms, wall_ms,
and the full generated text.

Modes:
  warm  — capture replies to <out .replies.json> (used to build the turns).
  cold  — rebuild the SAME three requests byte-identically from a warm run's
          replies file, so a fresh-blob daemon's scratch output can be diffed
          against the cache-served output turn by turn (thesis guard T2 on the
          multi-turn path).

Usage:
  MLXCACHE_BASE=http://127.0.0.1:18520 uv run python scripts/t22_multiturn_verify.py warm \
      /tmp/t22warm.jsonl
  MLXCACHE_BASE=http://127.0.0.1:18522 uv run python scripts/t22_multiturn_verify.py cold \
      /tmp/t22cold.jsonl /tmp/t22warm.replies.json

MLXCACHE_DEMO_MODEL must name a model the daemon SERVES (it 404s unknown
models); the daemon quickstart serves "demo-model".

The prompt corpus is a PINNED snapshot (scripts/fixtures/t22), not the live
repo: the requests embed file contents, and live files drift when fixes land
between passes — which once faked a KV-parity failure (see FIXTURES below).
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

BASE = os.environ.get("MLXCACHE_BASE", "http://127.0.0.1:18520")
MODEL = os.environ.get("MLXCACHE_DEMO_MODEL", "mlx-community/Qwen2.5-7B-Instruct-4bit")
REPO = os.environ.get("MLXCACHE_DEMO_REPO", os.getcwd())


def _read(rel: str, max_bytes: int) -> str:
    with open(os.path.join(REPO, rel), encoding="utf-8", errors="replace") as fh:
        return fh.read(max_bytes)


# PROMPT SOURCES ARE PINNED SNAPSHOTS, NOT LIVE REPO FILES.
#
# The prompts embed file contents (the realistic ~20K-token shape). Reading
# the LIVE files makes the request stream drift whenever those files are
# edited between runs — which is exactly what poisoned the first parity
# attempt (2026-10-04): the "scratch oracle" pass ran after fixes landed in
# index.rs/orchestrator.rs/the design doc, so its turn-2 request was 424
# tokens shorter and asked a different question than the served pass. The
# comparison compared two different requests and looked like a KV-resume
# bug. Pin the bytes so every pass tokenizes the identical request; bump the
# fixtures deliberately (all passes together) when the corpus must change.
FIXTURES = os.environ.get("MLXCACHE_T22_FIXTURES", "scripts/fixtures/t22")


def _fixture(name: str) -> str:
    with open(os.path.join(REPO, FIXTURES, name), encoding="utf-8") as fh:
        return fh.read()


FILE_A = _fixture("design_doc.md")
FILE_B = _fixture("index.rs")
FILE_C = _fixture("orchestrator.rs")

SYSTEM = "You are a coding agent working in a Rust repository. Be concise."

Q1 = (
    "Here is the daemon design document you wrote earlier:\n\n"
    f"```markdown\n{FILE_A}\n```\n\n"
    "And the prefix index and orchestrator sources:\n\n"
    f"```rust\n{FILE_B}\n```\n\n"
    f"```rust\n{FILE_C}\n```\n\n"
    "In two sentences: what does the eviction reaper guarantee, and how does "
    "delta prefill choose an ancestor checkpoint?"
)
Q2 = (
    "Now name the one env var that disables the reaper entirely and the env "
    "var that caps published entries, with their defaults."
)
Q3 = (
    "Finally: in one sentence, why is mid-prefix divergence left unserved "
    "instead of doing byte-level cache surgery?"
)


def stats() -> dict:
    with urllib.request.urlopen(f"{BASE}/stats", timeout=30) as r:
        return json.loads(r.read())


def post(messages: list, max_tokens: int = 64) -> dict:
    body = json.dumps(
        {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "stream": False}
    ).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as resp:
        payload = json.loads(resp.read())
    wall_ms = round((time.perf_counter() - t0) * 1000, 1)
    mx = payload.get("mlxcache", {})
    gen = payload.get("generated_tokens") or []
    return {
        "wall_ms": wall_ms,
        "verdict": mx.get("verdict"),
        "tokens_total": mx.get("tokens_total"),
        "tokens_cached": mx.get("tokens_cached"),
        "prefill_from": mx.get("prefill_from"),
        "lookup_ms": mx.get("lookup_ms"),
        "total_ms": mx.get("total_ms"),
        "gen_n": len(gen),
        "generated": gen,
    }


def run(mode: str, out_path: str, replies_path: str | None) -> int:
    replies: list[str] = []
    if mode == "cold":
        assert replies_path, "cold mode needs the warm run's replies JSON path"
        with open(replies_path) as rf:
            replies = json.load(rf)
        assert len(replies) == 3, f"expected 3 warm replies, got {len(replies)}"

    # The handle lives for the whole function (emit() closes over it) and is
    # closed at the end — a context manager would fight the closure.
    fh = open(out_path, "a")  # noqa: SIM115

    def emit(rec: dict):
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        print(json.dumps(rec), flush=True)

    turns: list[list] = []
    convo: list = [{"role": "system", "content": SYSTEM}]
    single = int(os.environ.get("T22_SINGLE_TURN", "0"))
    # Build EVERY turn's full request (the scratch oracle must see the whole
    # conversation prefix), then post only the selected turn when
    # T22_SINGLE_TURN is set (0 = all).
    for _, q in enumerate((Q1, Q2, Q3)):
        convo = convo + [{"role": "user", "content": q}]
        # The daemon's non-stream response carries generated token ids and no
        # message content, so the assistant turn appended here is OUR choice:
        # it must be identical in BOTH modes or the request byte streams
        # (and therefore tokenization, KV, and published keys) diverge before
        # the first user turn does. Parity compares generated token arrays.
        convo = convo + [{"role": "assistant", "content": ""}]
        turns.append(list(convo))
    for i, turn in enumerate(turns):
        if single and i + 1 != single:
            continue
        r = post(turn)
        # The cold leg replays against these replies (it asserts there are
        # three), so warm mode must actually collect them.
        replies.append(r["generated"])
        emit(
            {k: v for k, v in r.items() if k != "generated"}
            | {"turn": i + 1, "n_messages": len(turn)}
        )
        emit({"turn": i + 1, "generated": r["generated"], "stats": stats()})
    fh.close()
    if mode == "warm":
        rp = replies_path or (out_path + ".replies.json")
        with open(rp, "w") as wf:
            json.dump(replies, wf)
        print(f"replies -> {rp}", flush=True)
    return 0


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "warm"
    out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/t22multiturn.jsonl"
    rp = sys.argv[3] if len(sys.argv) > 3 else None
    sys.exit(run(mode, out, rp))
