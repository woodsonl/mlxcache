#!/usr/bin/env python3
"""Chunked-store dedup e2e on the REAL engine, conversation-shaped.

A multi-turn conversation where each turn appends the previous assistant
reply verbatim (the real client shape): turn r's tokens are an EXACT prefix
of turn r+1's, so the daemon partial-hits, delta-prefills only the new
tokens, and the KV bytes of every prior position are preserved (REPORT P1).
That is the workload the chunk store exists for: each turn's blob shares
every chunk of the previous turn's blob, and the store's physical bytes grow
by only the new tokens' blocks.

Usage: uv run --extra mlx python scripts/dedup/chunk_e2e_real.py \
          [--model mlx-community/Qwen2.5-7B-Instruct-4bit] [--turns 5]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def wait_health(port: int, daemon: bool = False) -> None:
    path = "/healthz" if daemon else "/health"
    for _ in range(120):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=2)
            return
        except Exception:
            time.sleep(1)
    raise SystemExit(f"port {port} never became healthy")


def post(port: int, body: dict) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
    ap.add_argument("--turns", type=int, default=5)
    args = ap.parse_args()

    blobs = tempfile.mkdtemp(prefix="chunk-dedup-")
    sc_port, d_port = 18471, 8471
    sc = subprocess.Popen(
        [f"{REPO}/.venv/bin/python", "-m", "mlxcache_sidecar.server"],
        cwd=REPO,
        env={
            **os.environ,
            "MLXCACHE_ENGINE": "mlx-lm",
            "MLXCACHE_MODEL": args.model,
            "MLXCACHE_SIDECAR_PORT": str(sc_port),
        },
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    wait_health(sc_port)
    dm = subprocess.Popen(
        [f"{REPO}/target/release/mlxcache-daemon"],
        cwd=REPO,
        env={
            **os.environ,
            "MLXCACHE_MODELS": args.model,
            "MLXCACHE_SIDECAR_URL": f"http://127.0.0.1:{sc_port}",
            "MLXCACHE_BLOBS": blobs,
            "MLXCACHE_ADDR": f"127.0.0.1:{d_port}",
        },
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    wait_health(d_port, daemon=True)

    messages: list[dict] = []
    verdicts = []
    questions = [
        "Name the three measurement gates in the mlxcache design doc.",
        "What does the settle convention cover and why the minus-one?",
        "Explain single-flight prefill in one paragraph.",
        "What is the anchor rule in the eviction policy?",
        "Summarize the R1-5 gate in two sentences.",
        "Why is the radix tree keyed by adapter-reported token ids?",
    ]
    try:
        for turn in range(args.turns):
            messages.append({"role": "user", "content": questions[turn % len(questions)]})
            body = {"model": args.model, "messages": [m.copy() for m in messages], "max_tokens": 48}
            out = post(d_port, body)
            verdicts.append(
                (
                    out["mlxcache"]["verdict"],
                    out["mlxcache"]["tokens_cached"],
                    out["mlxcache"]["tokens_total"],
                )
            )
            # Append the ASSISTANT reply verbatim: turn r's tokens become an
            # exact prefix of turn r+1's request.
            messages.append(
                {"role": "assistant", "content": out["choices"][0]["message"]["content"]}
            )
            time.sleep(1)

        time.sleep(3)  # let the last settle publish land
        sys.path.insert(0, str(REPO / "sidecar"))
        from mlxcache_store.chunks import read_manifest

        d = Path(blobs)
        logical = 0
        ckpts = list(d.glob("*.ckpt"))
        for ckm in d.glob("*.ckm"):
            logical += read_manifest(ckm).payload_len
        seg_bytes = sum(p.stat().st_size for p in (d / "chunks").glob("*.seg"))
        print(
            json.dumps(
                {
                    "turns": verdicts,
                    "blobs": len(ckpts),
                    "logical_payload_bytes": logical,
                    "segment_bytes": seg_bytes,
                    "dedup_ratio": round(logical / max(seg_bytes, 1), 2),
                },
                indent=1,
            )
        )
        partials = sum(1 for v, _, _ in verdicts[1:] if v in ("partial", "hit"))
        print(f"partials+hits after turn 1: {partials}/{len(verdicts) - 1}")
        print(f"blobs_dir={blobs}")
        return 0 if partials > 0 else 1
    finally:
        dm.terminate()
        sc.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
