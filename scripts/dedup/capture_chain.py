#!/usr/bin/env python3
"""E0 — capture a growing multi-turn conversation through the real engine.

Spawns the mlx-lm sidecar + the daemon, drives N turns of a pinned
conversation, snapshots the blob store (hardlink farm) after every turn,
and records per-turn /stats, replies, and blob inventory into
<out>/manifest.json. The snapshots preserve every turn's blobs even if a
later republish supersedes them, at zero extra disk cost (blobs are
immutable after the atomic rename).

Modes:
  grow   — the capture itself. Turn k+1 resumes from turn k's blob, so the
           chain blobs are by-construction overlaps (probe P1's input).
  single — replay the FULL conversation from a grow manifest as ONE request
           against a fresh store: the whole prompt prefilled in one model
           call (probe P2/P3's input: cold, single-shape computation).

The prompt corpus is a pinned fixture slice + fixed questions (T22
precedent: live repo files drift between passes and once faked a parity
failure).

Usage:
  uv run python scripts/dedup/capture_chain.py --out <dir> --model <id> \
      --kv-bits 0 [--turns 8] [--max-tokens 340]
  uv run python scripts/dedup/capture_chain.py --out <dir> --model <id> \
      --kv-bits 0 --mode single --single-from <grow_dir>/manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DOC = REPO_ROOT / "scripts" / "fixtures" / "t22" / "design_doc.md"

SYSTEM = "You are a coding agent working in a Rust repository. Answer precisely and concisely."

# Pinned follow-ups (fixed order, fixed text — the conversation shape is part
# of the measurement).
QUESTIONS = [
    "In two sentences: what does the eviction reaper guarantee, and how does "
    "delta prefill choose an ancestor checkpoint?",
    "Name the two env vars that bound the store and disable the reaper, with their defaults.",
    "Why is mid-prefix divergence left unserved instead of doing byte-level cache surgery?",
    "Summarize the single-flight rule in one sentence.",
    "What does the boot rebuild verify, and what happens to a blob that fails it?",
    "Give the two rules the reaper never breaks.",
    "How does the daemon avoid re-prefilling a concurrent identical request?",
    "In one sentence, why must a checkpoint cover tokens[:-1] and not all tokens?",
]


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def wait_http(url: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, OSError, TimeoutError):
            pass
        time.sleep(1.0)
    raise SystemExit(f"timeout waiting for {url}")


def get_json(url: str, timeout_s: float = 60) -> dict:
    with urllib.request.urlopen(url, timeout=timeout_s) as r:
        return json.loads(r.read())


def post_chat(base: str, model: str, messages: list[dict], max_tokens: int) -> dict:
    body = json.dumps(
        {"model": model, "messages": messages, "max_tokens": max_tokens, "stream": False}
    ).encode()
    req = urllib.request.Request(
        f"{base}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as resp:
        payload = json.loads(resp.read())
    payload["_wall_s"] = round(time.perf_counter() - t0, 3)
    return payload


def snapshot_blobs(blobs_dir: Path, snap_dir: Path) -> list[dict]:
    """Hardlink every blob into snap_dir; return the inventory."""
    snap_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for p in sorted(blobs_dir.glob("*.ckpt")):
        dst = snap_dir / p.name
        if not dst.exists():
            os.link(p, dst)
        out.append({"name": p.name, "bytes": p.stat().st_size})
    return out


def wait_publish(blobs_dir: Path, before: int, timeout_s: float = 120.0) -> int:
    """Wait until the store has more blobs than `before`.

    Publish happens after the response is returned to the client, so a
    snapshot taken immediately can miss this turn's blob — an observed race
    that silently dropped a turn (7 blobs for 8 turns) and made the turn
    comparison meaningless. Poll instead of assuming ordering.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        n = len(list(blobs_dir.glob("*.ckpt")))
        if n > before:
            return n
        time.sleep(0.05)
    raise SystemExit(f"timeout: no new blob after {timeout_s}s (have {before})")


def spawn(env_extra: dict, argv: list[str], log: Path | None = None) -> subprocess.Popen:
    env = {**os.environ, **env_extra}
    out = open(log, "wb") if log else subprocess.DEVNULL  # noqa: SIM115 — closed in teardown
    return subprocess.Popen(argv, cwd=REPO_ROOT, env=env, stdout=out, stderr=subprocess.STDOUT)


def teardown(procs: list[subprocess.Popen]) -> None:
    for p in procs:
        if p.poll() is None:
            p.send_signal(signal.SIGTERM)
    for p in procs:
        try:
            p.wait(timeout=15)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait(timeout=10)


def user_msg(base_doc: str, idx: int) -> str:
    if idx == 0:
        return (
            "Here is the daemon design document you wrote earlier:\n\n"
            f"```markdown\n{base_doc}\n```\n\n{QUESTIONS[0]}"
        )
    return QUESTIONS[idx]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--kv-bits", type=int, default=0, choices=(0, 8))
    ap.add_argument("--mode", choices=("grow", "single"), default="grow")
    ap.add_argument("--single-from", type=Path, help="grow manifest to replay (single mode)")
    ap.add_argument("--turns", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=340)
    ap.add_argument("--base-doc-chars", type=int, default=19_000)
    ap.add_argument("--health-timeout-s", type=float, default=900.0)
    args = ap.parse_args()

    if args.mode == "single" and not args.single_from:
        raise SystemExit("--mode single requires --single-from <grow manifest>")
    if args.turns > len(QUESTIONS):
        raise SystemExit(f"--turns max is {len(QUESTIONS)}")

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    doc = FIXTURE_DOC.read_text(encoding="utf-8")[: args.base_doc_chars]
    doc_sha = hashlib.sha256(doc.encode()).hexdigest()

    replies = None
    if args.mode == "single":
        grow = json.loads(args.single_from.read_text())
        replies = [t["reply"] for t in grow["turns"]]
        if len(replies) != args.turns:
            raise SystemExit(
                f"manifest has {len(replies)} turns but --turns={args.turns}; match them"
            )

    sidecar_port, daemon_port = free_port(), free_port()
    sidecar = spawn(
        {
            "MLXCACHE_ENGINE": "mlx-lm",
            "MLXCACHE_MODEL": args.model,
            "MLXCACHE_SIDECAR_PORT": str(sidecar_port),
            "MLXCACHE_KV_BITS": str(args.kv_bits),
        },
        [sys.executable, "-m", "mlxcache_sidecar.server"],
        log=out / "sidecar.log",
    )
    wait_http(f"http://127.0.0.1:{sidecar_port}/health", args.health_timeout_s)

    daemon = spawn(
        {
            "MLXCACHE_ADDR": f"127.0.0.1:{daemon_port}",
            "MLXCACHE_MODELS": args.model,
            "MLXCACHE_SIDECAR_URL": f"http://127.0.0.1:{sidecar_port}",
            "MLXCACHE_BLOBS": str(out / "blobs"),
            "MLXCACHE_TRACE": str(out / "trace.jsonl"),
            "MLXCACHE_EVICT_MAX_BYTES": "0",
            "MLXCACHE_EVICT_INTERVAL_S": "0",
            "MLXCACHE_SHUTDOWN_GRACE_S": "2",
            "RUST_LOG": "mlxcache_daemon=debug,info",
        },
        [str(REPO_ROOT / "target" / "release" / "mlxcache-daemon")],
        log=out / "daemon.log",
    )
    base = f"http://127.0.0.1:{daemon_port}"
    try:
        wait_http(f"{base}/healthz", 60.0)
        manifest = {
            "mode": args.mode,
            "model": args.model,
            "kv_bits": args.kv_bits,
            "turns_requested": args.turns,
            "max_tokens": args.max_tokens,
            "base_doc_chars": args.base_doc_chars,
            "base_doc_sha256": doc_sha,
            "sidecar_port": sidecar_port,
            "daemon_port": daemon_port,
            "turns": [],
        }
        # grow: a real multi-turn conversation — every reply re-enters the
        # history, so turn k+1's prompt strictly extends turn k's.
        # single: ONE request carrying the entire pinned conversation (the
        # same message list the grow run's final turn saw).
        history: list[dict] = [{"role": "system", "content": SYSTEM}]
        if args.mode == "single":
            for i in range(args.turns):
                history.append({"role": "user", "content": user_msg(doc, i)})
                if i < args.turns - 1:  # the request ends at the final user
                    # turn, exactly like the grow run's final request
                    history.append({"role": "assistant", "content": replies[i]})
            requests = [(args.turns, list(history))]
        else:
            requests = None  # built incrementally below
        n_req = 1 if args.mode == "single" else args.turns
        for i in range(1, n_req + 1):
            if args.mode == "grow":
                history.append({"role": "user", "content": user_msg(doc, i - 1)})
                messages = list(history)
                turn_no = i
            else:
                messages = requests[0][1]
                turn_no = args.turns
            n_before = len(list((out / "blobs").glob("*.ckpt")))
            snap_rel = None
            if args.mode == "grow":
                snap_rel = f"snapshots/turn-{i:02d}"
                snapshot_blobs(out / "blobs", out / snap_rel)
            resp = post_chat(base, args.model, messages, args.max_tokens)
            reply = resp["choices"][0]["message"]["content"]
            verdict = (resp.get("mlxcache") or {}).get("verdict")
            if verdict != "hit":
                # a hit republishes nothing; anything else must add a blob
                wait_publish(out / "blobs", n_before)
            inv = None
            if args.mode == "grow":
                history.append({"role": "assistant", "content": reply})
                inv = snapshot_blobs(out / "blobs", out / snap_rel)
            stats = get_json(f"{base}/stats")
            manifest["turns"].append(
                {
                    "turn": turn_no,
                    "snapshot": snap_rel,
                    "blobs": inv if args.mode == "grow" else None,
                    "stats": stats,
                    "reply": reply,
                    "wall_s": resp["_wall_s"],
                    "mlxcache": resp.get("mlxcache"),
                }
            )
            print(
                f"turn {turn_no}: verdict={resp.get('mlxcache', {}).get('verdict')} "
                f"cached={resp.get('mlxcache', {}).get('tokens_cached')}/"
                f"{resp.get('mlxcache', {}).get('tokens_total')} wall={resp['_wall_s']}s",
                flush=True,
            )
        if args.mode == "single":
            inv = snapshot_blobs(out / "blobs", out / "snapshots/single")
            manifest["turns"][-1]["blobs"] = inv
            manifest["turns"][-1]["snapshot"] = "snapshots/single"
        manifest["final_stats"] = get_json(f"{base}/stats")
        (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
        trace = out / "trace.jsonl"
        n_trace = sum(1 for _ in trace.open()) if trace.exists() else 0
        print(f"manifest -> {out / 'manifest.json'}; trace lines: {n_trace}")
    finally:
        teardown([daemon, sidecar])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
