#!/usr/bin/env python3
"""Wrapper QA battery (plan §D6): chat/completions + completions + streaming,
with a restart mid-suite, against a real mlxcache_serve pair.

Probes (each records pass/fail + evidence):
  Q1  chat completions non-stream: valid body -> 200, message content, finish_reason
  Q2  chat completions streaming: SSE chunks parse; final content == non-stream (temp 0)
  Q3  completions non-stream: prompt -> 200, text content
  Q4  restart mid-suite: SIGTERM after Q1-Q3; a fresh process over the SAME
      store serves a GROWN conversation (extends the persisted stream)
  Q5  disk-resume identity: the SAME grown conversation on a SECOND fresh
      process returns the same answer (store-level determinism)
  Q6  error shape: malformed body -> HTTP 4xx, JSON error object
  Q7  store hygiene: every *.ckpt in the store dir is digest-valid (fetch)

Usage:
  uv run python scripts/qa_wrapper.py [--model PATH] [--port N]

Env: MLXCACHE_BENCH_MODEL overrides the model default. Exit 0 = all pass.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = os.environ.get(
    "MLXCACHE_BENCH_MODEL",
    "mlx-community/Qwen2.5-7B-Instruct-4bit",  # resolved via the HF cache
)
RESULTS: list[tuple[str, str, str]] = []


def _resolve_model(raw: str) -> str:
    """Absolute local model dir, resolved identically for the server child
    and the parent (see bench_wrapper._resolve_model: the child runs with
    cwd=sidecar, so a relative path fingerprints differently per process)."""
    m = os.path.abspath(os.path.expanduser(raw))
    if (Path(m) / "config.json").is_file():
        return m
    # Resolve exactly as mlx_lm would: snapshot_download honours refs/<rev>
    # (default main), so a cache holding several revisions selects the same
    # snapshot the server child will load.
    with contextlib.suppress(ImportError, Exception):
        from huggingface_hub import snapshot_download

        snap = snapshot_download(raw, local_files_only=True)
        if (Path(snap) / "config.json").is_file():
            return snap
    base = (
        Path.home() / ".cache/huggingface/hub" / f"models--{raw.replace('/', '--')}" / "snapshots"
    )
    snaps = sorted(base.glob("*")) if base.is_dir() else []
    if snaps and (snaps[-1] / "config.json").is_file():
        return str(snaps[-1])
    raise SystemExit(
        f"model {raw!r} is neither a local dir nor a cached HF snapshot; "
        "pass an absolute snapshot path or set MLXCACHE_BENCH_MODEL"
    )


def probe(name):
    def wrap(fn):
        try:
            fn()
            RESULTS.append((name, "pass", ""))
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001
            RESULTS.append((name, "fail", repr(exc)))
            print(f"FAIL {name}: {exc!r}")
        return fn

    return wrap


def expect(cond, msg):
    if not cond:
        raise AssertionError(msg)


class Server:
    def __init__(self, store_dir: str, model: str, port: int):
        # File, never PIPE: more than 64KB of child output before stop() (an
        # HF download's progress) would wedge the child on a full pipe.
        fd, log_path = tempfile.mkstemp(prefix="mlxcache-qa-srv-", suffix=".log")
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
        with contextlib.suppress(Exception):
            self.log_fh.flush()
            return self.log_path.read_text()[-2000:]
        return "(log unreadable)"

    def wait_ready(self, timeout_s: float = 180.0):
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
            # outlive it. post() returns errors as (code, body) rather than
            # raising, so the status is checked explicitly, and any failure
            # (5xx, or a reset mid-load) means NOT ready yet: retry.
            try:
                code, body = self.post(
                    "/v1/chat/completions",
                    {
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 1,
                        "temperature": 0.0,
                    },
                    timeout=max(1.0, min(remaining, 30.0)),
                )
            except Exception as exc:
                last_err = repr(exc)
                time.sleep(1.0)
                continue
            if code != 200:
                last_err = f"warmup HTTP {code}: {str(body)[:120]}"
                time.sleep(1.0)
                continue
            return
        raise RuntimeError(
            f"wrapper never became healthy (last warmup error: {last_err}):\n{self._log_tail()}"
        )

    def post(self, path: str, body: dict, timeout: float = 600) -> tuple[int, dict | bytes]:
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            f"{self.base}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read()
                return r.status, json.loads(raw)
        except urllib.error.HTTPError as e:
            raw = e.read()
            try:
                return e.code, json.loads(raw)
            except Exception:
                return e.code, raw

    def stream(self, path: str, body: dict, timeout: float = 600) -> list[dict]:
        """Return parsed SSE `data:` payloads (skipping [DONE]).

        urllib's line iteration over the response is buffered, so a `data:`
        record split across TCP reads is reassembled before we see it.
        """
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            f"{self.base}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        events = []
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw_line in r:
                line = raw_line.decode().strip()
                if line.startswith("data: "):
                    payload = line[6:]
                    if payload == "[DONE]":
                        break
                    events.append(json.loads(payload))
        return events

    def stats(self) -> dict:
        """GET /mlxcache/stats: the live cache's reuse counters. Response
        equality at temperature 0 cannot distinguish a disk resume from a
        full prefill; the counter can."""
        with urllib.request.urlopen(f"{self.base}/mlxcache/stats", timeout=10) as r:
            return json.loads(r.read())

    def stop(self):
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--port", type=int, default=0)
    args = ap.parse_args()
    model = _resolve_model(args.model)

    import socket

    if args.port == 0:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            args.port = s.getsockname()[1]

    tmp = tempfile.mkdtemp(prefix="mlxcache-qa-")
    store_dir = str(Path(tmp) / "store")

    chat_body = {
        "messages": [{"role": "user", "content": "Name three properties of a good cache. " * 8}],
        "temperature": 0.0,
        "max_tokens": 20,
    }
    stream_body = {**chat_body, "stream": True}
    comp_body = {"prompt": "The capital of France is", "temperature": 0.0, "max_tokens": 8}

    state: dict = {}
    srv = Server(store_dir, model, args.port)
    try:
        _run_suite(srv, state, chat_body, stream_body, comp_body, store_dir, model, args.port)
    finally:
        srv.stop()
    fails = [r for r in RESULTS if r[1] == "fail"]
    print(f"\n{len(RESULTS) - len(fails)}/{len(RESULTS)} pass")
    return 1 if fails else 0


def _grown_body(chat_body: dict, chat1: str) -> dict:
    """Grow the conversation with turn 1's reply so the request extends the
    persisted turn-1 stream — the shape the disk tier actually serves (§3.3:
    an entry that extends the request is not a match, so a re-posted
    single-turn prompt can never disk-hit)."""
    return {
        "messages": chat_body["messages"]
        + [
            {"role": "assistant", "content": chat1},
            {"role": "user", "content": "Now summarize those properties."},
        ],
        "temperature": 0.0,
        "max_tokens": 20,
    }


def _run_suite(srv, state, chat_body, stream_body, comp_body, store_dir, model, port):
    srv.wait_ready()

    @probe("Q1 chat non-stream")
    def _():
        code, body = srv.post("/v1/chat/completions", chat_body)
        expect(code == 200, f"status {code}: {str(body)[:200]}")
        expect(body["choices"][0]["message"]["content"], "empty content")
        expect(body["choices"][0].get("finish_reason"), "no finish_reason")
        state["chat1"] = body["choices"][0]["message"]["content"]

    @probe("Q2 chat streaming")
    def _():
        chat1 = state.get("chat1")
        expect(chat1 is not None, "Q1 failed; dependent probe skipped")
        events = srv.stream("/v1/chat/completions", stream_body)
        expect(len(events) >= 2, f"{len(events)} events")
        expect(events[0].get("object") == "chat.completion.chunk", "not chunk objects")
        text = "".join(
            c["choices"][0]["delta"].get("content", "") for c in events if c.get("choices")
        )
        expect(
            text == chat1,
            f"stream text diverged:\n A={text[:80]!r}\n B={chat1[:80]!r}",
        )

    @probe("Q3 completions non-stream")
    def _():
        code, body = srv.post("/v1/completions", comp_body)
        expect(code == 200, f"status {code}: {str(body)[:200]}")
        expect(body["choices"][0]["text"], "empty completion")

    @probe("Q6 error shape")
    def _():
        code, body = srv.post("/v1/chat/completions", {"messages": "not-a-list"})
        expect(400 <= code < 500, f"status {code}")
        expect(isinstance(body, dict), "non-JSON error body")

    srv.stop()  # SIGTERM both processes

    @probe("Q4 restart mid-suite (grown conversation)")
    def _():
        chat1 = state.get("chat1")
        expect(chat1 is not None, "Q1 failed; dependent probe skipped")
        with Server(store_dir, model, port) as srv2:
            srv2.wait_ready()
            code, body = srv2.post("/v1/chat/completions", _grown_body(chat_body, chat1))
            expect(code == 200, f"status {code}")
            state["grown"] = body["choices"][0]["message"]["content"]
            # The fresh process must have REUSED the persisted turn-1 stream,
            # not merely re-prefilled it: the counter proves it took the disk
            # path (at temp 0 the two are token-identical).
            hits = srv2.stats().get("disk_hits") or 0
            expect(hits > 0, f"restart served without a disk hit (disk_hits={hits})")

    @probe("Q5 disk-resume identity (second fresh process)")
    def _():
        grown_state = state.get("grown")
        chat1 = state.get("chat1")
        expect(grown_state is not None, "Q4 failed; dependent probe skipped")
        with Server(store_dir, model, port) as srv3:
            srv3.wait_ready()
            code, body = srv3.post("/v1/chat/completions", _grown_body(chat_body, chat1))
            expect(code == 200, f"status {code}")
            expect(
                body["choices"][0]["message"]["content"] == grown_state,
                f"second disk-resumed answer diverged:\n"
                f" A={grown_state[:80]!r}\n B={body['choices'][0]['message']['content'][:80]!r}",
            )
            hits = srv3.stats().get("disk_hits") or 0
            expect(hits > 0, f"second process served without a disk hit (disk_hits={hits})")

    @probe("Q7 store hygiene")
    def _():
        sys.path.insert(0, str(REPO / "sidecar"))
        from mlxcache_store import Store

        s = Store(store_dir, byte_budget=0)
        n = 0
        for f in Path(store_dir).glob("*.ckpt"):
            s.fetch(f.name)  # digest-verified
            n += 1
        expect(n >= 1, "no checkpoints persisted")


if __name__ == "__main__":
    raise SystemExit(main())
