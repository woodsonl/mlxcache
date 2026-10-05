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

Usage:
  uv run python scripts/bench_wrapper.py [--model PATH_OR_HF_ID] [--port N]
      [--max-tokens N] [--out PATH]

Env: MLXCACHE_BENCH_MODEL overrides the model default.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import subprocess
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


def _resolve_model(raw: str) -> str:
    """Absolute local model dir, resolved identically for the server child
    and the gate's store lookup.

    The server child runs with cwd=sidecar, and mlx_lm keeps the raw --model
    string in its cache key; the store fingerprint is built from
    os.path.realpath of that string, so a relative path or bare HF id
    realpaths differently in each process and the lookup never matches —
    the gate would then read 0% skip on a working system. Pin one absolute
    path before launch so both sides agree.
    """
    m = os.path.abspath(os.path.expanduser(raw))
    if (Path(m) / "config.json").is_file():
        return m
    # Resolve exactly as mlx_lm would: snapshot_download honours refs/<rev>
    # (default main), so a cache holding several revisions selects the same
    # snapshot the server child will load. A bare glob of the snapshot dirs
    # would sort by commit hash — unrelated to the revision fetched.
    try:
        from huggingface_hub import snapshot_download

        snap = snapshot_download(raw, local_files_only=True)
        if (Path(snap) / "config.json").is_file():
            return snap
    except Exception:  # noqa: BLE001 — falls through to the clean SystemExit
        pass
    raise SystemExit(
        f"model {raw!r} is neither a local dir nor a cached HF snapshot; "
        "pass an absolute snapshot path or set MLXCACHE_BENCH_MODEL"
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


def _free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    """One mlxcache_serve pair (wrapper spawns mlx_lm.server in-process)."""

    def __init__(self, store_dir: str, model: str, port: int):
        # Log to a FILE, never a PIPE: a child that writes more than the
        # 64KB pipe buffer before stop() (an HF download's tqdm, a startup
        # traceback) would block forever on a full pipe and wedge mid-startup.
        fd, log_path = tempfile.mkstemp(prefix="mlxcache-bench-srv-", suffix=".log")
        self.log_path = Path(log_path)
        self.log_fh = os.fdopen(fd, "w")
        try:
            self.proc = subprocess.Popen(
                [
                    str(REPO / ".venv/bin/python"),
                    "-m",
                    "mlxcache_serve",
                    "--store-dir",
                    store_dir,
                    "--model",
                    model,
                    "--port",
                    str(port),
                    "--log-level",
                    "ERROR",
                ],
                cwd=str(REPO / "sidecar"),
                stdout=self.log_fh,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except Exception:
            # A failed spawn would otherwise leak the fd and the temp file.
            with contextlib.suppress(Exception):
                self.log_fh.close()
            with contextlib.suppress(Exception):
                self.log_path.unlink()
            raise
        self.port = port
        self.base = f"http://127.0.0.1:{port}"

    def _log_tail(self) -> str:
        # Best effort: the child holds its own fd, so only what it has
        # flushed to the OS is visible here.
        with contextlib.suppress(Exception):
            self.log_fh.flush()
            return self.log_path.read_text()[-2000:]
        return "(log unreadable)"

    def stats(self) -> dict:
        """GET /mlxcache/stats: the live cache's reuse counters. This is the
        only signal that the disk tier SERVED a request — response equality
        at temperature 0 cannot distinguish a disk resume from a full
        prefill."""
        with urllib.request.urlopen(f"{self.base}/mlxcache/stats", timeout=10) as r:
            return json.loads(r.read())

    def wait_ready(self, timeout_s: float = 180.0) -> None:
        deadline = time.time() + timeout_s
        last_err: str | None = None
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"wrapper exited early:\n{self._log_tail()}")
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                with urllib.request.urlopen(f"{self.base}/health", timeout=2) as r:
                    if r.status != 200:
                        time.sleep(1.0)
                        continue
            except Exception:
                time.sleep(1.0)
                continue
            # The port binds before the lazy model load, so /health alone
            # proves nothing. A 1-token warmup forces the load; the timeout
            # is bounded by the readiness deadline so a hung request cannot
            # outlive it. A warmup that fails (connection reset mid-load, or
            # a 5xx) means NOT ready yet: retry until the deadline rather
            # than crashing or reading a broken server as ready.
            try:
                self.chat(
                    {
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 1,
                        "temperature": 0.0,
                    },
                    timeout=max(1.0, min(remaining, 30.0)),
                )
                return
            except Exception as exc:
                last_err = repr(exc)
                time.sleep(1.0)
        raise RuntimeError(
            f"wrapper never became healthy (last warmup error: {last_err}):\n{self._log_tail()}"
        )

    def chat(self, body: dict, timeout: float = 600) -> tuple[float, str]:
        """POST /v1/chat/completions (non-stream). Returns (wall_ms, text).
        Raises on non-2xx so callers cannot mistake an error for a result."""
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            f"{self.base}/v1/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=timeout) as r:
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
                # A reaped child is not guaranteed after SIGKILL; a raise
                # here would skip the close/unlink below and re-leak the log.
                with contextlib.suppress(subprocess.TimeoutExpired):
                    self.proc.wait(timeout=10)
        with contextlib.suppress(Exception):
            self.log_fh.close()
        with contextlib.suppress(Exception):
            self.log_path.unlink()

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


def _gate_ok(results, skip_pct, observed_hits) -> tuple[bool, list[str]]:
    """The acceptance predicate. Returns (ok, reasons). A reusable blob
    is not reuse: the disk legs must have OBSERVED a disk hit, so a store
    that is populated but never read cannot pass."""
    reasons = []
    bad = [r["leg"] for r in results if r["verdict"] != "token-identical"]
    if bad:
        reasons.append(f"token divergence in {bad}")
    if skip_pct < 90:
        reasons.append(f"prefill skip {skip_pct}% < 90%")
    reused = all(v is not None and v > 0 for v in observed_hits.values())
    if not reused:
        reasons.append(f"disk tier not observed serving (disk_hits={observed_hits})")
    return (not reasons), reasons


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--port", type=int, default=_free_port())
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    model = _resolve_model(args.model)
    TURN1["max_tokens"] = args.max_tokens

    print(f"model: {model}")

    results: list[dict] = []
    tmp = tempfile.mkdtemp(prefix="mlxcache-bench-")
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

    # ---- restart turn 2 (fresh process, SAME store): the request stream
    # extends the persisted turn-1 stream, so the disk tier should cover the
    # whole first exchange minus one position (§3.2). Compared against the
    # MEMORY reference.
    with Server(store_dir, model, args.port) as srv2:
        srv2.wait_ready()
        wall_re, text_re = srv2.chat(body2)
        restart_stats = srv2.stats()
        results.append(_leg_result("restart-turn2", wall_re, text_re, baseline2, restart_stats))

    # ---- second process (another pair, SAME store): determinism ----------
    with Server(store_dir, model, args.port) as srv3:
        srv3.wait_ready()
        wall_2nd, text_2nd = srv3.chat(body2)
        second_stats = srv3.stats()
        results.append(_leg_result("second-turn2", wall_2nd, text_2nd, baseline2, second_stats))

    # ---- acceptance ------------------------------------------------------
    # Prefill-skip, measured as WORK not wall: the restart/second legs'
    # requests are full two-turn streams; a disk hit covers (matched-1)
    # positions of the turn-1 prefix, so the prefill actually computed is
    # the remainder. Coverage (how MUCH a blob holds) is read from the
    # shared store; the disk_hits counter from /mlxcache/stats proves the
    # server actually TOOK that blob rather than the memory twin.
    from mlxcache_store import Store as _Store

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
    observed_hits = {
        "restart-turn2": (restart_stats or {}).get("disk_hits"),
        "second-turn2": (second_stats or {}).get("disk_hits"),
    }
    for r in results:
        if r["leg"] in ("restart-turn2", "second-turn2"):
            r["reused"]["prefill_skip_pct"] = skip_pct
            r["reused"]["request_tokens"] = len(tok)
            r["reused"]["disk_covered"] = covered
    # A blob existing is not reuse. At temperature 0 a full prefill and a
    # disk resume are token-identical, so the token verdict cannot prove the
    # disk tier served anything; the counter can. Require an observed hit.
    ok, reasons = _gate_ok(results, skip_pct, observed_hits)

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
