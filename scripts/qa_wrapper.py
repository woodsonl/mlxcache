#!/usr/bin/env python3
"""Wrapper QA battery: chat/completions + completions + streaming,
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
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _harness  # noqa: E402 — path insert above
from _harness import DEFAULT_MODEL, REPO, WrapperServer, free_port  # noqa: E402

_resolve_model = _harness.resolve_model
RESULTS: list[tuple[str, str, str]] = []


class Server(WrapperServer):
    """One mlxcache_serve pair whose warmup inspects the HTTP status."""

    def __init__(self, store_dir: str, model: str, port: int):
        super().__init__(
            store_dir,
            model,
            port,
            log_prefix="mlxcache-qa-srv-",
            warmup="post",
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--port", type=int, default=0)
    args = ap.parse_args()
    model = _resolve_model(args.model)

    if args.port == 0:
        args.port = free_port()

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
