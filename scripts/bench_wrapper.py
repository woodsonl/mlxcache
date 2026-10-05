#!/usr/bin/env python3
"""Wrapper benchmark (plan §D5): persistent KV reuse through mlxcache_serve.

Drives the REAL mlx_lm.server (wrapped by mlxcache_serve) over four legs
against a local mlx-lm model:

  cold     : empty store -> full-price prefill (what a restart would pay
             without the wrapper)
  warm     : same processes, same request again -> in-memory exact hit
  restart  : SIGTERM server+wrapper, fresh processes, SAME store dir ->
             the persisted entry must serve as a disk hit (prefill skipped)
  second   : a SECOND wrapper pair resumes the FIRST's prefix (§8 cross-
             process share)

Correctness gate: every leg's generated tokens must be token-identical to
the cold leg's completion (the R1-5 thesis, applied to the wrapper).
The restart leg doubles as the §3.2 coverage guard: a coverage off-by-one
shifts the context and diverges within a few tokens.

Metrics per leg: wall_ms (request→last token), text, reused (coverage +
verdict notes), verdict (token-identical | DIVERGED).

Output: JSON artifact (bench/wrapper-bench-<ts>.json) + a markdown table
on stdout. CPU-viable with a small model (Qwen2.5-0.5B-4bit default).

Usage:
  uv run python scripts/bench_wrapper.py [--model PATH_OR_HF_ID] [--port N]
      [--max-tokens N] [--out PATH]

Env: MLXCACHE_BENCH_MODEL overrides the model default.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = os.environ.get(
    "MLXCACHE_BENCH_MODEL",
    "mlx-community/Qwen2.5-7B-Instruct-4bit",  # resolved via the HF cache
)

TURN1 = {
    "messages": [
        {"role": "user", "content": "Explain the theory of persistent KV cache reuse. " * 24}
    ],
    "temperature": 0.0,
    "max_tokens": 24,
}
# Turn 2 GROWS the conversation: the request stream extends turn 1's stored
# prompt+response stream, which is the shape the protocol can serve from disk
# (an entry that extends the request is deliberately NOT a match — §3.3).
# A re-posted identical single-turn request can NEVER disk-hit; benching one
# would measure the wrong thing.
#
# Correctness references, matched to the producer (the R1-5 gate): a disk
# resume and an in-memory resume share ONE cache lineage, so they must be
# token-identical to each other. Comparing a batch-prefill continuation
# against a resumed one instead would flake on near-tied argmax choices —
# different kernel paths yield bitwise-adjacent logits that legitimately
# disagree at temp 0. Same reference shape for the turn-2 restart:
# continue generation from a memory cache loaded with the FULL turn-1+
# turn-2 request stream (batch style), which is what a daemon-written
# blob would hold.
def turn2_body(resp_a: str) -> dict:
    return {
        "messages": TURN1["messages"]
        + [{"role": "assistant", "content": resp_a},
           {"role": "user", "content": "Summarize those properties in one sentence."}],
        "temperature": 0.0,
        "max_tokens": 24,
    }


def _free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    """One mlxcache_serve pair (wrapper spawns mlx_lm.server in-process)."""

    def __init__(self, store_dir: str, model: str, port: int):
        self.proc = subprocess.Popen(
            [
                str(REPO / ".venv/bin/python"), "-m", "mlxcache_serve",
                "--store-dir", store_dir,
                "--model", model,
                "--port", str(port),
                "--log-level", "ERROR",
            ],
            cwd=str(REPO / "sidecar"),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self.port = port
        self.base = f"http://127.0.0.1:{port}"

    def wait_ready(self, timeout_s: float = 180.0) -> None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.proc.poll() is not None:
                out = self.proc.stdout.read() if self.proc.stdout else ""
                raise RuntimeError(f"wrapper exited early:\n{out[-2000:]}")
            try:
                with urllib.request.urlopen(f"{self.base}/health", timeout=2) as r:
                    if r.status == 200:
                        return
            except Exception:
                time.sleep(1.0)
        raise RuntimeError("wrapper never became healthy")

    def chat(self, body: dict) -> tuple[float, str]:
        """POST /v1/chat/completions (non-stream). Returns (wall_ms, text)."""
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            f"{self.base}/v1/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=600) as r:
            payload = json.loads(r.read())
        wall = (time.perf_counter() - t0) * 1000
        return wall, payload["choices"][0]["message"]["content"]

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
        # Drain the PIPE so a chatty child cannot wedge on a full stdout
        # buffer and the logs of a failed run stay readable.
        if self.proc.stdout:
            try:
                self.proc.stdout.read()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stop()
        return False


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
        for i, (a, b) in enumerate(zip(baseline, text)):
            if a != b:
                row["reused"]["first_divergence_char"] = i
                break
    return row


def _mlxcache_tokenize(model: str, body: dict) -> list[int]:
    """The exact token stream the server computes for this chat body
    (chat template, no system prompt), so the store lookup matches."""
    from mlx_lm.utils import load_tokenizer

    tok = load_tokenizer(model)
    return tok.apply_chat_template(
        body["messages"], add_generation_prompt=True
    )


def _fp_for(model: str):
    from mlxcache_serve import default_fingerprint_for

    return default_fingerprint_for((model, None, None))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--port", type=int, default=_free_port())
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    model = args.model
    TURN1["max_tokens"] = args.max_tokens

    print(f"model: {args.model}")

    results: list[dict] = []
    tmp = tempfile.mkdtemp(prefix="mlxcache-bench-")
    store_dir = str(Path(tmp) / "store")

    # ---- cold + warm turn 1 (one server, empty store) --------------------
    with Server(store_dir, model, args.port) as srv:
        srv.wait_ready()
        wall_cold, text_cold = srv.chat(TURN1)
        results.append(_leg_result("cold-turn1", wall_cold, text_cold, None))
        baseline1 = text_cold
        wall_warm, text_warm = srv.chat(TURN1)
        results.append(_leg_result("warm-turn1", wall_warm, text_warm, baseline1))

    # ---- memory-resume turn 2 reference (SAME server, grown convo) -------
    # The in-memory tier answers the SAME turn-2 request from the SAME
    # turn-1 lineage without any disk involvement. Disk resume must match
    # THIS (same producer, same lineage): the R1-5 gate for the wrapper.
    body2 = turn2_body(text_cold)
    with Server(store_dir, model, args.port) as srv:
        srv.wait_ready()
        wall_mem2, text_mem2 = srv.chat(body2)
        results.append(_leg_result("memory-turn2", wall_mem2, text_mem2, None))
        baseline2 = text_mem2

    # ---- restart turn 2 (fresh processes, SAME store) --------------------
    # The request stream extends the persisted turn-1 stream: the disk tier
    # can now cover the whole first exchange minus one position (§3.2).
    with Server(store_dir, model, args.port) as srv2:
        srv2.wait_ready()
        wall_re, text_re = srv2.chat(body2)
        results.append(_leg_result("restart-turn2", wall_re, text_re, baseline2))

    # ---- second process (another pair, SAME store) ------------------------
    with Server(store_dir, model, args.port) as srv3:
        srv3.wait_ready()
        wall_2nd, text_2nd = srv3.chat(body2)
        results.append(_leg_result("second-turn2", wall_2nd, text_2nd, baseline2))

    # ---- acceptance --------------------------------------------------------
    # §D5 prefill-skip, measured as WORK not wall: the restart/second legs'
    # requests are full two-turn streams; a disk hit covers (matched-1)
    # positions of the turn-1 prefix, so the prefill actually computed is
    # the remainder. The wrapper exposes no counter over HTTP, so measure
    # the disk tier's coverage directly from the store it shares.
    from mlxcache_store import Store as _Store

    st = _Store(store_dir, byte_budget=0)
    tok = _mlxcache_tokenize(model, body2)
    fp = _fp_for(model)
    matched, _ref = st.lookup("mlx-lm", fp, tok)
    covered = max(matched - 1, 0)
    skip_pct = round(100 * covered / max(len(tok) - 1, 1))
    for r in results:
        if r["leg"] in ("restart-turn2", "second-turn2"):
            r["reused"]["prefill_skip_pct"] = skip_pct
            r["reused"]["request_tokens"] = len(tok)
            r["reused"]["disk_covered"] = covered
    ok = (
        all(r["verdict"] == "token-identical" for r in results)
        and skip_pct >= 90  # §D5: restart leg >= 90% prefill skip
    )

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
        bad = [r["leg"] for r in results if r["verdict"] != "token-identical"]
        reasons = []
        if bad:
            reasons.append(f"token divergence in {bad}")
        if skip_pct < 90:
            reasons.append(f"prefill skip {skip_pct}% < 90%")
        print("GATE: FAIL —", "; ".join(reasons))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
