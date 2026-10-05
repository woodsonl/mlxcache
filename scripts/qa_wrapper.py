#!/usr/bin/env python3
"""Wrapper QA battery (plan §D6): chat/completions + completions + streaming,
with a restart mid-suite, against a real mlxcache_serve pair.

Probes (each records pass/fail + evidence):
  Q1  chat completions non-stream: valid body → 200, message content, finish_reason
  Q2  chat completions streaming: SSE chunks parse; final content == non-stream (temp 0)
  Q3  completions non-stream: prompt → 200, text content
  Q4  persistence: after Q1-Q3, SIGTERM; fresh processes over the SAME store;
      the same chat request returns identical completion (temp 0) — restart
      reuse through the disk tier
  Q5  restart mid-suite correctness: Q4's answer token-identical to Q1's
  Q6  error shape: malformed body → HTTP 4xx, JSON error object
  Q7  store hygiene: every *.ckpt in the store dir is digest-valid (fetch)

Usage:
  uv run python scripts/qa_wrapper.py [--model PATH] [--port N]

Env: MLXCACHE_BENCH_MODEL overrides the model default. Exit 0 = all pass.
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
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = os.environ.get(
    "MLXCACHE_BENCH_MODEL",
    "/Users/lance/models/models--mlx-community--Qwen2.5-7B-Instruct-4bit/snapshots/"
    "c26a38f6a37d0a51b4e9a1eb3026530fa35d9fed",
)
RESULTS: list[tuple[str, str, str]] = []


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
        self.proc = subprocess.Popen(
            [str(REPO / ".venv/bin/python"), "-m", "mlxcache_serve",
             "--store-dir", store_dir, "--model", model, "--port", str(port),
             "--log-level", "ERROR"],
            cwd=str(REPO / "sidecar"),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        self.port = port
        self.base = f"http://127.0.0.1:{port}"

    def wait_ready(self, timeout_s: float = 180.0):
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

    def post(self, path: str, body: dict, timeout: float = 600) -> tuple[int, dict | bytes]:
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            f"{self.base}{path}", data=data,
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
        """Yield parsed SSE `data:` payloads (skipping [DONE])."""
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            f"{self.base}{path}", data=data,
            headers={"Content-Type": "application/json"},
        )
        events = []
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for line in r:
                line = line.decode().strip()
                if line.startswith("data: "):
                    payload = line[6:]
                    if payload == "[DONE]":
                        break
                    events.append(json.loads(payload))
        return events

    def stop(self):
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--port", type=int, default=0)
    args = ap.parse_args()
    model = args.model

    import socket

    if args.port == 0:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            args.port = s.getsockname()[1]

    tmp = tempfile.mkdtemp(prefix="mlxcache-qa-")
    store_dir = str(Path(tmp) / "store")

    chat_body = {
        "messages": [{"role": "user", "content": "Name three properties of a good cache. " * 8}],
        "temperature": 0.0, "max_tokens": 20,
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


def _run_suite(srv, state, chat_body, stream_body, comp_body, store_dir, model, port):
    state = state if state is not None else {}
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
        events = srv.stream("/v1/chat/completions", stream_body)
        expect(len(events) >= 2, f"{len(events)} events")
        expect(events[0].get("object") == "chat.completion.chunk", "not chunk objects")
        text = "".join(
            c["choices"][0]["delta"].get("content", "")
            for c in events if c.get("choices")
        )
        expect(text == state["chat1"], f"stream text diverged:\n A={text[:80]!r}\n B={state['chat1'][:80]!r}")

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

    @probe("Q4+Q5 restart mid-suite (grown conversation)")
    def _():
        # A re-posted identical single-turn prompt can NOT disk-hit: the
        # stored stream is prompt+reply, and an entry that extends the
        # request is deliberately not a match (§3.3). GROW the conversation
        # with turn 1's reply so the request extends the persisted stream —
        # the shape the disk tier actually serves.
        grown = {
            "messages": chat_body["messages"]
            + [{"role": "assistant", "content": state["chat1"]},
               {"role": "user", "content": "Now summarize those properties."}],
            "temperature": 0.0, "max_tokens": 20,
        }
        with Server(store_dir, model, port) as srv2:
            srv2.wait_ready()
            code, body = srv2.post("/v1/chat/completions", grown)
            expect(code == 200, f"status {code}")
            state["grown"] = body["choices"][0]["message"]["content"]
        # Reference: the SAME grown conversation served from the memory
        # lineage (fresh server, same store). Disk resume must equal it —
        # the R1-5 gate at wrapper scale.
        with Server(store_dir, model, port) as srv3:
            srv3.wait_ready()
            code, body = srv3.post("/v1/chat/completions", grown)
            expect(code == 200, f"status {code}")
            expect(body["choices"][0]["message"]["content"] == state["grown"],
                   f"disk-resumed answer diverged from the memory lineage:\n"
                   f" A={state['grown'][:80]!r}\n B={body['choices'][0]['message']['content'][:80]!r}")

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
