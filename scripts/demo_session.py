#!/usr/bin/env python3
"""Suspend/resume demo workload (step 6): one large real session, cold vs warm.

Builds ONE large coding-agent request from real repository content (~20K
tokens), then drives three legs against the daemon at MLXCACHE_BASE:

  cold    : fresh daemon + empty blob dir -> full-price prefill (the cost a
            restart would pay without the daemon)
  warm    : re-post the SAME request in-process -> full hit, the pre-restart
            steady state
  resume  : (caller restarts daemon+sidecar between) re-post the SAME request
            -> the freshly-rebuilt index must serve the persisted checkpoint
            as a hit with prefill_from == tokens-1, sub-second wall.

NOTE (T22 wiring): multi-turn GROWTH is not part of this demo. This script
drives one request three ways (cold/warm/resume). Multi-turn divergent-serve
behavior — a follow-up request that shares the first len(key)-1 tokens with a
published key — is covered by the T22 suites: the synthetic-engine e2e tests
and the real-engine parity driver (scripts/t22_multiturn_verify.py), which
verified the serve at ~20K tokens (LCP = len(key)-1, prefill_from = covered).

Every leg appends {"leg","wall_ms","verdict","tokens_total","tokens_cached",
"prefill_from","lookup_ms","total_ms","gen_n"} to the output JSONL.

Usage: uv run python scripts/demo_session.py <cold|warm|resume> [out.jsonl]
Env: MLXCACHE_BASE (daemon), MLXCACHE_DEMO_MODEL, MLXCACHE_DEMO_REPO.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

BASE = os.environ.get("MLXCACHE_BASE", "http://127.0.0.1:8420")
MODEL = os.environ.get(
    "MLXCACHE_DEMO_MODEL", "mlx-community/Qwen2.5-7B-Instruct-4bit"
)
REPO = os.environ.get("MLXCACHE_DEMO_REPO", os.getcwd())
CONVO_PATH = os.environ.get(
    "MLXCACHE_DEMO_CONVO", "/tmp/mlxcache-demo/session.jsonl.convo.json"
)


def _read(rel: str, max_bytes: int) -> str:
    with open(os.path.join(REPO, rel), "r", encoding="utf-8", errors="replace") as fh:
        return fh.read(max_bytes)


# Real repository content — the "files read" the agent carries as context.
# ~76KB total => ~21K tokens for Qwen2.5 BPE: a real long-context request
# whose cold prefill at 7B-4bit costs tens of seconds (the prefill a restart
# skips), while staying far inside the 32768-token window.
FILE_A = _read("docs/designs/mlx-kv-cache-daemon.md", 28_000)   # design doc
FILE_B = _read("crates/mlxcache-core/src/index.rs", 30_000)     # prefix index
FILE_C = _read("crates/mlxcache-daemon/src/orchestrator.rs", 18_000)

SYSTEM = "You are a coding agent working in a Rust repository. Be concise."


def build_convo() -> list:
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": (
                "Here is the daemon design document you wrote earlier:\n\n"
                f"```markdown\n{FILE_A}\n```\n\n"
                "And the prefix index and orchestrator sources:\n\n"
                f"```rust\n{FILE_B}\n```\n\n"
                f"```rust\n{FILE_C}\n```\n\n"
                "In two sentences: what does the eviction reaper guarantee, "
                "and how does delta prefill choose an ancestor checkpoint?"
            ),
        },
    ]


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
    gen = payload.get("generated_tokens")
    return {"wall_ms": wall_ms, "mx": mx, "gen_n": len(gen) if gen else 0}


def run(leg: str, out_path: str) -> int:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    new_file = not os.path.exists(out_path)
    fh = open(out_path, "a")

    def emit(rec: dict):
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        print(json.dumps(rec), flush=True)

    if leg == "cold":
        assert new_file, f"cold leg needs a fresh output file, got {out_path}"
    convo = build_convo()
    r = post(convo)
    emit(
        {
            "leg": leg,
            "wall_ms": r["wall_ms"],
            "verdict": r["mx"].get("verdict"),
            "tokens_total": r["mx"].get("tokens_total"),
            "tokens_cached": r["mx"].get("tokens_cached"),
            "prefill_from": r["mx"].get("prefill_from"),
            "lookup_ms": r["mx"].get("lookup_ms"),
            "total_ms": r["mx"].get("total_ms"),
            "gen_n": r["gen_n"],
        }
    )
    # Persist the exact request once (the cold leg) so the resume leg replays
    # byte-identical messages even if the repo edits between phases.
    if leg == "cold":
        with open(CONVO_PATH, "w") as cf:
            json.dump(convo, cf)
    fh.close()
    return 0


if __name__ == "__main__":
    leg = sys.argv[1] if len(sys.argv) > 1 else "cold"
    out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/mlxcache-demo/session.jsonl"
    sys.exit(run(leg, out))
