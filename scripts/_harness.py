#!/usr/bin/env python3
"""Shared process harness for the benchmark/QA scripts.

Two scripts on this branch (``bench_wrapper``, ``qa_wrapper``) spawn a
localhost stack, wait for it to actually serve, and kill it cleanly; the
prefix-benchmark unit of this change stack adds a third consumer
(``bench_prefix``). The spawn, readiness and teardown rules are
safety-critical — a harness that trusts a server it did not start benchmarks
the wrong stack — so they live in one place instead of drifting copies.

Module path: scripts/ is not a package, so importers do
``sys.path.insert(0, str(Path(__file__).resolve().parent))`` then
``import _harness``.
"""

from __future__ import annotations

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

# Readiness and teardown budgets, named once so a slow stack tunes a constant
# instead of copying the whole loop into a subclass.
READY_TIMEOUT_S = 180.0
WARMUP_CAP_S = 30.0
RETRY_SLEEP_S = 1.0
HEALTH_CAP_S = 2.0
TERM_WAIT_S = 30.0
KILL_WAIT_S = 10.0


def free_port() -> int:
    """An unused localhost port.

    Bind-then-release is a TOCTOU race in principle: another process can claim
    the port between the close and the child's bind. It is kept because the
    child (mlx_lm.server / the daemon) cannot report a kernel-assigned port
    back to us, and every caller readiness-probes the exact base URL before
    use — a hijacked port fails the probe instead of being benchmarked. The
    e2e suites, which CAN read a printed port, use that instead.
    """
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def resolve_model(raw: str) -> str:
    """Absolute local model dir, resolved identically for the server child
    and the parent.

    The server child runs with cwd=sidecar, and mlx_lm keeps the raw --model
    string in its cache key; the store fingerprint is built from
    os.path.realpath of that string, so a relative path or bare HF id
    realpaths differently in each process and the lookup never matches —
    the harness would then read 0% reuse on a working system. Pin one
    absolute path before launch so both sides agree.

    Resolution is LOCAL-FILES-ONLY: a missing model fails loudly instead of
    downloading gigabytes behind the operator's back.
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
        "pass an absolute snapshot path or set MLXCACHE_BENCH_MODEL "
        "(this harness never downloads a model)"
    )


def stats_delta(pre: dict, post: dict, key: str) -> int:
    """Delta of a /mlxcache/stats counter across a measured request.

    Both wrappers gate reuse on this. `or 0` coercion would read a MISSING
    counter as "zero hits" — the opposite of what a gate wants to hear when
    the stats endpoint stops reporting a key — so a missing counter fails
    loudly instead.
    """
    if pre.get(key) is None or post.get(key) is None:
        raise RuntimeError(f"/mlxcache/stats did not report {key!r}: cannot gate on it")
    return post[key] - pre[key]


def sse_lines(resp, timeout: float):
    """Bounded-read SSE decoder: yield each `data:` payload string, stopping
    at EOF or [DONE]. urllib's line iteration over the response is buffered,
    so a record split across TCP reads is reassembled before we see it. The
    socket timeout bounds ONE read, not the stream — so the socket's timeout
    is reset to the REMAINING total budget before every read and a server
    dripping a frame every few seconds can neither hold the connection past
    the deadline nor block one read longer than the budget allows. Single
    decode policy for every consumer: errors="replace" (payloads are JSON;
    a torn multi-byte char should not kill a stream whose budget was fine)."""
    deadline = time.monotonic() + timeout
    sock = getattr(getattr(resp.fp, "raw", None), "_sock", None)
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(f"SSE stream exceeded its {timeout}s total budget")
        if sock is not None:
            # Bound the NEXT blocking read by what is left of the total
            # budget, not by the original per-read timeout.
            sock.settimeout(max(0.05, remaining))
        try:
            raw_line = resp.readline()
        except TimeoutError:
            # The remaining-budget read timed out: same verdict as the
            # deadline check, raised as the documented error.
            raise RuntimeError(f"SSE stream exceeded its {timeout}s total budget") from None
        if not raw_line:
            return  # EOF: the server closed the stream
        if time.monotonic() > deadline:
            raise RuntimeError(f"SSE stream exceeded its {timeout}s total budget")
        line = raw_line.decode(errors="replace").strip()
        if not line.startswith("data: "):
            continue
        payload = line[len("data: ") :]
        if payload == "[DONE]":
            return
        yield payload


class Server:
    """One localhost server child: spawn, readiness-probe, kill, reap, unlink.

    ``argv`` is the full command; ``cwd`` and ``env`` are passed through. The
    readiness probe is injected by subclasses via ``_warmup`` so each stack
    checks the endpoint that actually proves it serves.
    """

    def __init__(
        self,
        argv: list[str],
        cwd: Path | str | None = None,
        env: dict | None = None,
        port: int | None = None,
        base: str | None = None,
        log_prefix: str = "mlxcache-harness-",
    ):
        # Log to a FILE, never a PIPE: a child that writes more than the 64KB
        # pipe buffer before stop() (an HF download's tqdm, a startup
        # traceback) would block forever on a full pipe and wedge mid-startup.
        fd, log_path = tempfile.mkstemp(prefix=log_prefix, suffix=".log")
        self.log_path = Path(log_path)
        self.log_fh = os.fdopen(fd, "w")
        try:
            self.proc = subprocess.Popen(
                argv,
                cwd=str(cwd) if cwd is not None else None,
                env=env,
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
        # ``port`` is intentionally not stored: nothing reads it back (the
        # base URL carries everything the clients need), and a write-only
        # attribute is drift bait.
        self.base = base

    def _log_tail(self) -> str:
        # Best effort: the child holds its own fd, so only what it has
        # flushed to the OS is visible here. Read the LAST 2 KiB by seeking,
        # not the whole file: a chatty child over a 180s window can make a
        # full read big enough that the read itself fails and the suppress
        # swallows the diagnostic exactly when it is needed.
        try:
            with self.log_path.open("rb") as fh:
                fh.seek(0, os.SEEK_END)
                size = fh.tell()
                fh.seek(max(0, size - 2000))
                return fh.read().decode(errors="replace")
        except Exception:  # noqa: BLE001 — diagnostics must never raise
            return "(log unreadable)"

    def health_ok(self, timeout: float) -> None:
        """GET /health and require 200. A bound port is not a serving server."""
        with urllib.request.urlopen(f"{self.base}/health", timeout=timeout) as r:
            if r.status != 200:
                raise RuntimeError(f"health {r.status}")

    def _warmup(self, timeout: float) -> None:
        """Prove the server actually serves (not merely that the port bound).

        Default: GET /health and require 200. Stacks whose port binds before
        the model loads must override with a real inference probe.
        """
        self.health_ok(timeout)

    def wait_ready(self, timeout_s: float = READY_TIMEOUT_S) -> None:
        deadline = time.time() + timeout_s
        last_err: str | None = None
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited early:\n{self._log_tail()}")
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                self._warmup(max(1.0, min(remaining, WARMUP_CAP_S)))
            except Exception as exc:  # noqa: BLE001 — not ready yet, retry
                last_err = repr(exc)
                time.sleep(RETRY_SLEEP_S)
                continue
            # Post-probe liveness, OUTSIDE the retry's except (a raise inside
            # the try is just swallowed into the retry loop): the probe can
            # be answered by whatever NOW owns an explicit --port — a stale,
            # protocol-compatible listener — while our own child's bind
            # failure surfaces as an exit during the multi-second warmup.
            # Certifying ready without this re-check would benchmark a
            # server this harness never started.
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited during warmup:\n{self._log_tail()}")
            return
        raise RuntimeError(
            f"server never became healthy (last warmup error: {last_err}):\n{self._log_tail()}"
        )

    def get_json(self, path: str, timeout: float = 10.0) -> dict:
        with urllib.request.urlopen(f"{self.base}{path}", timeout=timeout) as r:
            return json.loads(r.read())

    def post(self, path: str, body: dict, timeout: float = 600.0) -> tuple[int, dict | bytes]:
        """POST JSON; errors come back as (code, body) rather than raising, so
        callers can distinguish a 5xx from a transport failure."""
        req = urllib.request.Request(
            f"{self.base}{path}",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            raw = e.read()
            try:
                return e.code, json.loads(raw)
            except Exception:  # noqa: BLE001 — non-JSON error body
                return e.code, raw

    def stream(self, path: str, body: dict, timeout: float = 600) -> list[dict]:
        """Return parsed SSE `data:` payloads (skipping [DONE]). The
        bounded-read and decode policies live in `sse_lines`, shared with
        every other SSE consumer."""
        req = urllib.request.Request(
            f"{self.base}{path}",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return [json.loads(p) for p in sse_lines(r, timeout)]

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=TERM_WAIT_S)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                # A reaped child is not guaranteed after SIGKILL; a raise
                # here would skip the close/unlink below and re-leak the log.
                with contextlib.suppress(subprocess.TimeoutExpired):
                    self.proc.wait(timeout=KILL_WAIT_S)
                if self.proc.poll() is None:
                    print(
                        f"warning: server pid {self.proc.pid} survived SIGKILL; "
                        "its log was unlinked but the process leaked",
                        file=sys.stderr,
                    )
        with contextlib.suppress(Exception):
            self.log_fh.close()
        with contextlib.suppress(Exception):
            self.log_path.unlink()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stop()
        return False


class WrapperServer(Server):
    """One mlxcache_serve pair (the wrapper spawns mlx_lm.server in-process).

    The port binds before the lazy model load, so /health alone proves
    nothing: a 1-token warmup forces the load. A warmup that fails
    (connection reset mid-load, or a 5xx) means NOT ready yet — retry until
    the deadline rather than crashing or reading a broken server as ready.
    """

    def __init__(
        self,
        store_dir: str,
        model: str,
        port: int,
        log_prefix: str = "mlxcache-srv-",
        warmup: str = "chat",
    ):
        # "chat" raises on failure (bench form); "post" returns (code, body)
        # so a 5xx is inspected rather than raised (QA form). Both must be
        # treated as NOT-ready and retried; the choice only decides which
        # client method the caller's tests patch. Anything else is a typo
        # that would silently select the chat probe and leave the subclass's
        # real warmup shape untested — and it must fail BEFORE the spawn in
        # super().__init__, or the child process and its log leak (no `with`
        # block is entered for a constructor that raises).
        if warmup not in ("chat", "post"):
            raise ValueError(f"warmup mode {warmup!r} must be 'chat' or 'post'")
        super().__init__(
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
            cwd=REPO / "sidecar",
            port=port,
            base=f"http://127.0.0.1:{port}",
            log_prefix=log_prefix,
        )
        # Validated above, before the spawn.
        self.warmup = warmup

    def _warmup(self, timeout: float) -> None:
        self.health_ok(min(timeout, HEALTH_CAP_S))
        # UNIQUE per spawned SERVER (the CHILD's pid, not this harness
        # process's: one run spawns several stacks from one parent). A
        # constant probe prompt would be published by the first server (the
        # store publishes any assistant stream of >= 2 tokens) and disk-hit
        # by every later server during ITS warmup, inflating disk_hits
        # before any measured request runs.
        body = {
            "messages": [{"role": "user", "content": f"harness warmup {self.proc.pid}"}],
            "max_tokens": 1,
            "temperature": 0.0,
        }
        if self.warmup == "post":
            code, payload = self.post("/v1/chat/completions", body, timeout=timeout)
            if code != 200:
                raise RuntimeError(f"warmup HTTP {code}: {str(payload)[:120]}")
            return
        self.chat(body, timeout=max(1.0, timeout))

    def chat(self, body: dict, timeout: float = 600) -> tuple[float, str]:
        """POST /v1/chat/completions (non-stream). Returns (wall_ms, text).
        Raises on non-2xx so callers cannot mistake an error for a result."""
        req = urllib.request.Request(
            f"{self.base}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = json.loads(r.read())
        wall = (time.perf_counter() - t0) * 1000
        return wall, payload["choices"][0]["message"]["content"]

    def stats(self) -> dict:
        """GET /mlxcache/stats: the live cache's reuse counters. This is the
        only signal that the disk tier SERVED a request — response equality
        at temperature 0 cannot distinguish a disk resume from a full
        prefill."""
        return self.get_json("/mlxcache/stats")
