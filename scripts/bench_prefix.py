#!/usr/bin/env python3
"""Prefix-caching benchmark on the REAL engine: declared vs served reuse.

This is the cache-benefit benchmark. It runs a real mlx-lm model behind the
daemon and reports, per workload, what reuse the traffic actually made
available and what the daemon actually served.

Why the distinction matters
---------------------------
`prefill_from` (a.k.a. `tokens_cached`) is the only honest reuse signal: a full
prefill and a KV resume are token-identical at temperature 0, so response
equality proves nothing. But `prefill_from > 0` alone does not prove the cache
did its job either — it must be measured against the reuse the workload
*declared*. Three numbers per request:

  declared  longest common token prefix with any earlier request in the leg
            (the reuse opportunity the traffic contains)
  served    the daemon's `prefill_from` for this request
  held      the trace's `matched_tokens - 1` (KV the blob actually hands over)

`served > declared` is a fabrication, `held != served` is an accounting drift,
and either one aborts the run instead of printing a number.

The serve rule this benchmark must respect
------------------------------------------
The index serves an entry only when the request agrees through the entry key's
LAST token (`crates/mlxcache-core/src/index.rs`). A request that diverges
mid-prefix is never served, because trimming quantized MLX KV at an arbitrary
position is not safe. Consequences the workloads encode:

  * A "long shared system prompt + short changing question" workload does NOT
    hit on its own: each request publishes a full-length key, and the next one
    diverges far from its end. W1 therefore runs BOTH shapes — primed (the
    prefix is itself published once, which the child rule can then serve) and
    unprimed (the natural production shape). The gap between them is the
    finding, not an accident.
  * Growing conversations and exact repeats are the shapes that reuse.

Workloads
---------
  W1  shared_prefix          fixed long prefix, N different short questions,
                             primed with one request whose stream IS the prefix
  W1np shared_prefix_unprimed  same traffic with no priming request
  W2  near_miss              shared prefix that diverges in its first few
                             tokens (a legitimate TRIE near-miss)
  W3  decode_heavy           distinct short prompts, 256-token answers
  W4  pressure_sweep         fixed prefix + k unrelated requests between
                             repeats, two arms: prefix-alone vs prefix-extended
                             (structurally anchored), against a tight entry cap
  W5  trace_replay           the captured trace of W1/W1np/W2/W3, scored with
                             scripts/replay_trace.py so trace and /stats agree

Legs
----
Every workload runs twice: `cold` (empty store, fresh process) and `warm`
(fresh process, SAME store) — the cross-restart tier, which is the whole point
of a persistent cache. The `direct` arm runs the SAME requests against plain
`mlx_lm.server`, whose `LRUPromptCache` dies with the process: its `warm` leg
is a cold restart by construction. It is the cross-process baseline.

Usage
-----
  # real engine, default grid (long: ~25-40 min)
  uv run python scripts/bench_prefix.py --model /abs/path/to/snapshot

  # smoke: one workload, few requests
  uv run python scripts/bench_prefix.py --workloads W1 --questions 2 --k 0

The model must already be on disk. This harness never downloads one.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _harness  # noqa: E402 — path insert above
import replay_trace  # noqa: E402 — path insert above
from _harness import REPO, free_port, resolve_model  # noqa: E402

P95_MIN_N = 100
# W5 is not a selectable workload: it is the per-leg trace/stats cross-check
# run automatically inside the daemon arm.
KNOWN_WORKLOADS = {"W1", "W1np", "W2", "W3", "W4"}
KNOWN_ARMS = {"daemon", "direct", "pressure"}
# Named budgets: a slow stack tunes a constant instead of copying a loop
# (the _harness rule). Model loads dominate readiness; requests stream well
# inside these, and a deadline exists so a drip can never pin a leg open.
MODEL_LOAD_TIMEOUT_S = 600.0
REQUEST_TIMEOUT_S = 900.0
TRACE_WAIT_S = 20.0
EVICT_WAIT_S = 25.0

# Filler that tokenizes to roughly one token per four characters. Repeated to
# build a long, boring, real-tokenizer prefix. Nothing here is special-cased by
# the engine; it is ordinary prose because the benchmark must not exploit the
# synthetic tokenizer's sentence hashing. The repeat count is CALIBRATED at
# runtime against the real tokenizer (measure_filler_tokens), so --prefix-tokens
# means tokens, not repeats-of-guess; W2 splices on _FILLER_OPENING, keeping the
# near-miss shape tied to this constant instead of a magic literal.
_FILLER = (
    "The retention policy for a shared cache depends on the working set of the "
    "caller and on how much of the prefix survives between requests. "
)
_FILLER_OPENING = "The retention"


def serialize_messages(messages: list[dict]) -> str:
    """The prompt string the daemon hashes, byte for byte.

    The daemon tokenizes `serde_json::to_string(&req.messages)` — compact
    separators, no ASCII escaping of non-ASCII text, `"` and `\\` escaped.
    Python's `json.dumps` with compact separators matches serde for this ASCII
    subset. The parity is not assumed: every request's real token count is
    compared against the trace's `n_tokens`, and a mismatch aborts the run.
    """
    return json.dumps(messages, separators=(",", ":"), ensure_ascii=False)


def lcp_len(a: list[int], b: list[int]) -> int:
    """Longest common prefix length of two token streams."""
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = max(0, min(len(s) - 1, round(p / 100 * (len(s) - 1))))
    return round(s[k], 2)


def tpot_ms(total_ms: float, ttft_ms: float, tokens: int) -> float:
    """Per-token decode cost over the tokens actually generated."""
    return (total_ms - ttft_ms) / max(tokens - 1, 1)


def filler_text(approx_tokens: int, tokens_per_repeat: int) -> str:
    """Deterministic prose filler of approximately `approx_tokens` tokens.

    `tokens_per_repeat` is MEASURED for this model by the caller (one encode of
    _FILLER); a guessed divisor here is exactly how the first run shipped a
    3,243-token prefix under a --prefix-tokens 2000 knob. The real count is
    measured through the real tokenizer anyway and reported; calibration makes
    the knob mean what it says.
    """
    if tokens_per_repeat < 1:
        raise ValueError(f"tokens_per_repeat must be >= 1, got {tokens_per_repeat}")
    repeats = max(1, -(-approx_tokens // tokens_per_repeat))
    return _FILLER * repeats


def measure_filler_tokens(model: str) -> int:
    """One tokenizer encode of _FILLER, for runtime prefix calibration."""
    from mlx_lm.utils import load_tokenizer  # noqa: PLC0415 — heavy import, once

    tok = load_tokenizer(model)
    return len(tok.encode(_FILLER))


def check_disk_headroom(store_parent: str, min_free_bytes: int) -> list[str]:
    """Refuse to start a run that cannot fit its checkpoints.

    A full disk does not fail loudly anywhere: the daemon treats a publish
    failure as "nothing published" (OV6 — a follower whose request was fine must
    not 502), so every request still returns a correct answer with a plausible
    latency while reuse silently collapses to zero. The result is a benchmark
    that reports 0% cache benefit and blames the cache. On 7B-4bit a ~1 K-token
    checkpoint is ~57 MB (measured: 56.7 MB at 972 tokens), so the grid needs
    gigabytes of headroom.
    """
    free = shutil.disk_usage(store_parent).free
    if free >= min_free_bytes:
        return []
    return [
        f"only {free / 1e9:.2f} GB free under {store_parent}, need "
        f"{min_free_bytes / 1e9:.2f} GB for checkpoints (a failed publish reads "
        "as a miss, so the run would understate reuse with no error)"
    ]


# ---------------------------------------------------------------------------
# Workload construction
# ---------------------------------------------------------------------------


def build_w1(prefix: str, questions: int, name: str, primed: bool = True) -> dict:
    """Fixed prefix + short changing question, optionally primed with the
    prefix alone.

    The priming request's stream ends exactly one token past the shared region
    (`"}]` closes the JSON array), which is the boundary the index's child rule
    can serve. Without it, every question diverges mid-key and is a miss — that
    is W1np, not a bug in this construction.
    """
    reqs = []
    if primed:
        reqs.append({"label": f"{name}:prime", "messages": [{"role": "system", "content": prefix}]})
    for i in range(questions):
        reqs.append(
            {
                "label": f"{name}:q{i}",
                "messages": [
                    {"role": "system", "content": prefix},
                    {"role": "user", "content": f"Question {i}: answer in one short paragraph."},
                ],
            }
        )
    return {"name": name, "requests": reqs, "max_tokens": 32}


def build_w1_unprimed(prefix: str, questions: int) -> dict:
    return build_w1(prefix, questions, "W1np_shared_prefix_unprimed", primed=False)


def build_w2(prefix: str, questions: int) -> dict:
    """Near miss: documents share a long opening, then diverge EARLY relative
    to each document's own length.

    A tail-only difference would be a legitimate hit under the end-anchored
    rule, so the divergence must come well before the end for this workload to
    mean "similar text, different document" rather than "same prefix". The
    shared region is long (most of the filler) and the splice hangs off
    _FILLER_OPENING, not a magic literal.
    """
    shared = prefix[: max(1, len(prefix) // 2)]
    other = (
        shared + "A different cache eviction strategy applies when the working set "
        "outgrows the resident set. "
    )
    reqs = []
    for i in range(questions):
        body = prefix if i % 2 == 0 else other
        reqs.append(
            {
                "label": f"W2:q{i}",
                "messages": [
                    {"role": "system", "content": body},
                    {"role": "user", "content": f"Question {i}: answer in one short paragraph."},
                ],
            }
        )
    return {"name": "W2_near_miss", "requests": reqs, "max_tokens": 32}


def build_w3(prompts: int) -> dict:
    """Little shared prefix, long answers: the decode-bound workload.

    Cache reuse cannot help here by construction; the row exists so a decode
    regression cannot hide behind a cache win, and so the "long answers dominate"
    case is measured rather than assumed.
    """
    reqs = [
        {
            "label": f"W3:q{i}",
            "messages": [
                {"role": "user", "content": f"Write a numbered list about topic {i} in detail."}
            ],
        }
        for i in range(prompts)
    ]
    return {"name": "W3_decode_heavy", "requests": reqs, "max_tokens": 256}


def build_w4(prefix: str, k: int, arm: str) -> dict:
    """Pressure sweep: a repeat of the prefix after k unrelated requests.

    `p_alone`    publishes the prefix and nothing else, so the prefix is the
                 longest entry and the reaper (score = token_count) takes it
                 first.
    `p_extended` publishes the prefix AND a request that extends it, which makes
                 the prefix a structural anchor: the reaper takes the extension
                 and leaves the prefix published.

    The sweep answers "does the benefit hold under cache pressure", and the two
    arms separate "biggest blob loses" from "anchored blob survives".

    The repeat is a SEPARATE phase (`probe`), because the reaper has to land
    between the noise and the measurement — sending them back to back would
    measure the pre-eviction cache and call it a pressure result.
    """
    noise = [{"label": f"W4/{arm}/k{k}:prime", "messages": [{"role": "system", "content": prefix}]}]
    if arm == "p_extended":
        noise.append(
            {
                "label": f"W4/{arm}/k{k}:extend",
                "messages": [
                    {"role": "system", "content": prefix},
                    {"role": "user", "content": "Summarize the policy above in one line."},
                ],
            }
        )
    for i in range(k):
        noise.append(
            {
                "label": f"W4/{arm}/k{k}:noise{i}",
                "messages": [{"role": "user", "content": f"Unrelated request {i}: say hello."}],
            }
        )
    probe = {
        "label": f"W4/{arm}/k{k}:repeat",
        "messages": [
            {"role": "system", "content": prefix},
            {"role": "user", "content": "Repeat question: answer in one short paragraph."},
        ],
    }
    return {
        "name": f"W4_pressure_{arm}_k{k}",
        "requests": noise,
        "probe": probe,
        "max_tokens": 32,
    }


# ---------------------------------------------------------------------------
# HTTP clients
# ---------------------------------------------------------------------------


def _generated_text(frame: dict) -> str | None:
    """The text this stream frame generated, in any of the three shapes the
    benchmark's servers emit.

    The daemon wraps the sidecar's NDJSON as {"token": ...}; OpenAI-style
    servers emit choices[0].delta.content (chat) or choices[0].text
    (completions). A frame carrying none of them — the daemon's leading
    verdict chunk, a role-only delta — is not a token, and counting it would
    fabricate both TTFT and token counts. The first benchmarked run counted
    only the daemon shape, so every direct-arm row reported 0 tokens and its
    "TPOT" was really end-to-end minus TTFT.
    """
    if "token" in frame:
        return str(frame["token"]) or None
    choice = (frame.get("choices") or [{}])[0] or {}
    piece = (choice.get("delta") or {}).get("content") or choice.get("text")
    return piece or None


def _post_stream(base: str, path: str, body: dict, timeout: float) -> tuple[dict, dict | None]:
    """POST a streaming completion; return (timing, mlxcache meta).

    TTFT is stamped on the first frame that carries generated text. The
    daemon's leading chunk carries the cache verdict as an extra and
    OpenAI-style servers open with a role-only chunk, so stamping on the
    first frame of any kind would fold the cache decision into TTFT and hide
    exactly the difference being measured. The whole consumption runs under a
    MONOTONIC TOTAL deadline: the socket timeout bounds one read only, and a
    server dripping a frame every few seconds would otherwise pin a leg open
    forever.
    """
    payload = dict(body, stream=True)
    req = urllib.request.Request(
        f"{base}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft: float | None = None
    tokens = 0
    meta: dict | None = None
    usage: dict | None = None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        # Shared bounded-read/decode policy (see _harness.sse_lines): one
        # loop for every SSE consumer, not a drifted copy per script.
        for frag in _harness.sse_lines(resp, timeout):
            try:
                frame = json.loads(frag)
            except json.JSONDecodeError:
                continue
            if "mlxcache" in frame:
                # The daemon emits TWO meta extras on a stream: the leading
                # chunk carries the routing verdict, and a trailing frame
                # carries timing (ttft_ms). Last-write-wins would erase the
                # verdict and report every request as "n/a".
                meta = {**(meta or {}), **frame["mlxcache"]}
                continue
            if "usage" in frame:
                # OpenAI-style servers can end with a usage frame when asked
                # (stream_options.include_usage); the server-side completion
                # count is the cross-check for the client's frame count.
                usage = frame["usage"]
                continue
            if "error" in frame:
                raise RuntimeError(f"stream error frame: {frame['error']}")
            if _generated_text(frame):
                if ttft is None:
                    ttft = (time.perf_counter() - t0) * 1000
                tokens += 1
    total = (time.perf_counter() - t0) * 1000
    if ttft is None:
        raise RuntimeError("stream closed before any token frame")
    server_count = (usage or {}).get("completion_tokens")
    if server_count is not None and server_count < tokens:
        # Every text-bearing frame carries at least one token, so the
        # server's token count can exceed the client's frame count (a chunk
        # may pack several tokens) but never fall below it — fewer tokens
        # than frames means frames were counted that carry no token.
        raise RuntimeError(
            f"server counted {server_count} completion tokens, client counted "
            f"{tokens} text frames — token accounting has drifted"
        )
    if server_count is not None:
        # The server's count is the true token count; the frame count only
        # lower-bounds it (multi-token chunks), so TPOT divides by this.
        tokens = server_count
    return {"ttft_ms": ttft, "total_ms": total, "tokens": tokens}, meta


def _post_json(base: str, path: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        f"{base}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        # Surface the daemon's own error object: a bare "HTTP 400" hides which
        # field it rejected, and that is exactly what a misconfigured stack
        # reports.
        detail = e.read().decode(errors="replace")
        raise RuntimeError(f"POST {path} -> {e.code}: {detail[:400]}") from e


def _stats(base: str) -> dict:
    with urllib.request.urlopen(f"{base}/stats", timeout=30) as r:
        return json.loads(r.read())


# ---------------------------------------------------------------------------
# Stacks
# ---------------------------------------------------------------------------


class _DaemonServer(_harness.Server):
    """The daemon side of a DaemonStack, on the shared readiness loop.

    The daemon exposes no /health; a 2xx /stats read proves the router is
    wired. Inheriting Server.wait_ready keeps the exit-fast poll, the
    bounded retry, the log tail on failure and the POST-PROBE liveness
    re-check (a stale listener on a recycled port cannot be certified).
    """

    def _warmup(self, timeout: float) -> None:
        _stats(self.base)


class DaemonStack:
    """sidecar (real mlx-lm engine) + daemon, over a chosen store dir."""

    def __init__(self, model: str, store_dir: str, env_extra: dict | None = None):
        self.model = model
        self.store_dir = store_dir
        self.sidecar_port = free_port()
        self.port = free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        self.sidecar_base = f"http://127.0.0.1:{self.sidecar_port}"
        Path(store_dir).mkdir(parents=True, exist_ok=True)

        env = dict(os.environ)
        env.pop("MLXCACHE_ENGINE", None)
        env.pop("MLXCACHE_TOKENIZE_GROW", None)  # synthetic-engine-only knob
        # The model and port reach the child via ENV, never interpolated into
        # executable source: the safety of `python -c` must not hang on a
        # repr() round-trip of an operator-controlled string.
        sidecar_env = dict(
            env,
            MLXCACHE_ENGINE="mlx-lm",
            PYTHONUNBUFFERED="1",
            MLXCACHE_BENCH_SIDECAR_DIR=str(REPO / "sidecar"),
            MLXCACHE_BENCH_ENGINE_MODEL=model,
            MLXCACHE_BENCH_SIDECAR_PORT=str(self.sidecar_port),
        )
        self.sidecar = _harness.Server(
            [
                str(REPO / ".venv/bin/python"),
                "-c",
                "import os, sys; sys.path.insert(0, os.environ['MLXCACHE_BENCH_SIDECAR_DIR']); "
                "from mlxcache_sidecar import server; "
                "server.Handler.engine = server.make_engine("
                "os.environ['MLXCACHE_BENCH_ENGINE_MODEL']); "
                "server.BurstServer(('127.0.0.1', "
                "int(os.environ['MLXCACHE_BENCH_SIDECAR_PORT'])), "
                "server.Handler).serve_forever()",
            ],
            cwd=REPO,
            env=sidecar_env,
            port=self.sidecar_port,
            base=self.sidecar_base,
            log_prefix="mlxcache-bench-sidecar-",
        )
        daemon_env = dict(
            env,
            MLXCACHE_MODELS=model,
            MLXCACHE_SIDECAR_URL=self.sidecar_base,
            MLXCACHE_BLOBS=store_dir,
            MLXCACHE_ADDR=f"127.0.0.1:{self.port}",
            RUST_LOG=os.environ.get("RUST_LOG", "warn"),
        )
        daemon_env.update(env_extra or {})
        try:
            self.daemon = self._spawn_daemon(daemon_env)
        except BaseException:
            # The sidecar is already running; a daemon that failed to spawn
            # must not strand it holding a port.
            self.sidecar.stop()
            raise

    def _spawn_daemon(self, env: dict) -> _DaemonServer:
        binary = REPO / "target/release/mlxcache-daemon"
        argv = (
            [str(binary)]
            if binary.is_file()
            else [
                "cargo",
                "run",
                "--release",
                "-p",
                "mlxcache-daemon",
                "--",
            ]
        )
        return _DaemonServer(
            argv,
            cwd=REPO,
            env=env,
            base=self.base,
            log_prefix="mlxcache-bench-daemon-",
        )

    def tokenize(self, prompt: str) -> list[int]:
        """Tokenize through the SAME engine the daemon uses, so a declared
        prefix and a served prefix are the same unit of measurement."""
        out = _post_json(self.sidecar_base, "/tokenize", {"prompt": prompt}, 120)
        return out["tokens"]

    def completion_request(self, messages: list[dict], max_tokens: int | None) -> tuple[str, dict]:
        """The daemon's OpenAI chat surface: it applies its own serde-compact
        serialization before tokenizing, so messages pass through unchanged."""
        body = {"model": self.model, "messages": messages}
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        return "/v1/chat/completions", body

    def wait_ready(self, timeout_s: float = MODEL_LOAD_TIMEOUT_S) -> None:
        """Wait for BOTH processes to serve, not just for the daemon's router.

        The daemon's /stats answers as soon as axum is up, while the sidecar
        binds its listener only AFTER the model finishes loading (it constructs
        the engine before `BurstServer`). Polling the daemon alone therefore
        races a multi-second model load and surfaces as a 503
        `adapter_unavailable` on the first real request.
        """
        self.daemon.wait_ready(timeout_s)
        self.sidecar.wait_ready(timeout_s)

    def stats(self) -> dict:
        return _stats(self.base)

    def stop(self) -> None:
        self.daemon.stop()
        self.sidecar.stop()


class DirectStack(_harness.Server):
    """Plain mlx_lm.server: in-memory LRUPromptCache, nothing survives a restart.

    This is the cross-process baseline. It is not crippled: its in-process
    repeat leg measures its own prompt cache, which is what a caller gets
    without mlxcache. Readiness rides the shared Server loop (exit-fast
    poll, bounded retry, post-probe liveness) with a REAL COMPLETION as the
    probe: mlx_lm.server binds its port before the weights are loaded, so a
    port check would race the model.
    """

    def __init__(self, model: str):
        self.model = model
        self._tok = None
        port = free_port()
        argv = [
            str(REPO / ".venv/bin/python"),
            "-m",
            "mlx_lm",
            "server",
            "--model",
            model,
            "--port",
            str(port),
            "--log-level",
            "ERROR",
        ]
        super().__init__(
            argv,
            cwd=REPO,
            base=f"http://127.0.0.1:{port}",
            log_prefix="mlxcache-bench-direct-",
        )

    def _warmup(self, timeout: float) -> None:
        path, body = self.completion_request([{"role": "user", "content": "hi"}], 1)
        _post_json(self.base, path, body, timeout)

    def tokenize(self, messages: list[dict]) -> list[int]:
        """Encode the same serialized string the daemon tokenizes, so a
        declared LCP in this arm and in the daemon arm live in one space."""
        if self._tok is None:
            from mlx_lm.utils import load_tokenizer

            self._tok = load_tokenizer(self.model)
        return self._tok.encode(serialize_messages(messages))

    def completion_request(self, messages: list[dict], max_tokens: int | None) -> tuple[str, dict]:
        """Send the DAEMON'S serialization as a raw completion prompt.

        The daemon tokenizes serde_json::to_string(messages) and the sidecar
        feeds those tokens straight to the model — no chat template anywhere
        in its path. A baseline that lets mlx_lm apply the Qwen template
        answers a DIFFERENT prompt: token counts, generation lengths (the
        JSON prompts ran to the 256-token cap where the templated ones
        stopped at ~55 by EOS) and therefore TTFT are not comparable. The
        raw /v1/completions endpoint with the identical string is the honest
        baseline: the same bytes the daemon's engine consumes. The usage
        frame is requested so the server's own completion count cross-checks
        the client's frame count.
        """
        body = {
            "model": self.model,
            "prompt": serialize_messages(messages),
            "stream_options": {"include_usage": True},
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        return "/v1/completions", body


# ---------------------------------------------------------------------------
# Running one leg
# ---------------------------------------------------------------------------


def run_leg(
    stack,
    workload: dict,
    tokenize_messages,
    requests=None,
    corpus: list[list[int]] | None = None,
) -> list[dict]:
    """Send every request in order, recording timing, declared and served reuse.

    `declared` is the reuse this request could have taken from ANY earlier
    request in the benchmark, which means the corpus must outlive one leg: the
    warm leg's store still holds the cold leg's keys, so a request that
    "declares" zero against its own leg alone would legitimately be served
    hundreds of tokens — and then be flagged as fabricated reuse. The caller
    threads one corpus through every leg for exactly that reason.

    Requests are serial: concurrency is held at 1 across every workload.
    """
    reqs = workload["requests"] if requests is None else requests
    seen = corpus if corpus is not None else []
    samples: list[dict] = []
    for req in reqs:
        tokens = tokenize_messages(req["messages"])
        declared = max((lcp_len(prev, tokens) for prev in seen), default=0)
        seen.append(tokens)
        path, body = stack.completion_request(req["messages"], workload["max_tokens"])
        timing, meta = _post_stream(stack.base, path, body, timeout=REQUEST_TIMEOUT_S)
        sample = {
            "label": req["label"],
            "prompt_tokens": len(tokens),
            "_tokens": tokens,
            "declared_lcp": declared,
            "ttft_ms": round(timing["ttft_ms"], 1),
            "total_ms": round(timing["total_ms"], 1),
            "tokens_out": timing["tokens"],
            "tpot_ms": round(tpot_ms(timing["total_ms"], timing["ttft_ms"], timing["tokens"]), 2),
            "verdict": (meta or {}).get("verdict"),
            "served": (meta or {}).get("prefill_from"),
        }
        samples.append(sample)
    return samples


def _await_trace_records(path: str, want: int, timeout_s: float = 20.0) -> list[dict]:
    """Wait until the trace holds at least `want` records.

    The writer is asynchronous (bounded channel drained by a thread), so the
    newest record can be one flush behind the response that produced it. Reading
    once would either raise on a short capture or — worse — read the baseline
    one record early and shift EVERY sample onto its predecessor's record, where
    the numbers are plausible and the report is silently wrong.
    """
    deadline = time.time() + timeout_s
    records = _read_trace(path)
    while len(records) < want and time.time() < deadline:
        time.sleep(0.05)
        records = _read_trace(path)
    return records


def join_trace(
    samples: list[dict], trace_path: str, base_n: int, timeout_s: float = TRACE_WAIT_S
) -> None:
    """Attach trace-side accounting and enforce the premises.

    The trace writer is asynchronous (bounded channel drained by a thread), so
    the last record can still be in flight when the final response lands. Poll
    for the expected count with a bounded wait instead of reading once — a
    single read would flake as "trace had N-1 records" and look like a daemon
    bug.

    Three premises, each of which has silently broken a benchmark before:
      1. serialization parity — the trace's n_tokens must equal the count the
         sidecar gives for OUR serialization of the same messages. A mismatch
         means the daemon tokenizes a different string and every declared
         prefix in this run is fiction. (Enforced on token COUNT: the trace
         carries no token ids, so count parity is the strongest check the
         recorded data supports — two different equal-length serializations
         would slip past it, which is the documented limit, not a promise of
         byte identity.)
      2. served == held — `prefill_from` must equal `matched_tokens - 1`.
      3. served <= declared — the daemon cannot serve reuse the traffic never
         offered.

    The record count must match EXACTLY: a longer capture would silently
    shift every later sample onto its predecessor's record (positional join),
    which is precisely the plausible-numbers failure this function exists to
    prevent.
    """
    records = _await_trace_records(trace_path, base_n + len(samples), timeout_s)[base_n:]
    if len(records) != len(samples):
        raise RuntimeError(
            f"{trace_path}: trace has {len(records)} records for {len(samples)} "
            f"requests after offset {base_n} (capture lag, a dropped write, or "
            "an extra record that would shift the positional join); the "
            "accounting cannot be joined"
        )
    for sample, rec in zip(samples, records, strict=True):
        sample["trace_n_tokens"] = rec["n_tokens"]
        sample["trace_matched"] = rec["matched_tokens"]
        sample["trace_verdict"] = rec["verdict"]
        held = rec["matched_tokens"] - 1 if rec["verdict"] != "miss" else 0
        sample["held"] = max(held, 0)


def _read_trace(path: str) -> list[dict]:
    out = []
    with open(path, "rb") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def check_premises(samples: list[dict], strict_serialization: bool = True) -> list[str]:
    """Return premise violations. A violation is a bug in the harness or the
    daemon's accounting, never a benchmark result — the caller aborts.

    A sample that was joined to a trace (has trace_n_tokens) must carry a
    served verdict: losing the meta extras would otherwise read as a
    premise-green 0% reuse — the exact confident-zero failure the disk
    preflight exists to refuse. served=None is legal only where no
    instrumentation exists (the direct arm's samples are never joined).
    """
    problems = []
    for s in samples:
        if "trace_n_tokens" in s and s.get("served") is None:
            problems.append(
                f"{s['label']}: trace-joined sample lost its served meta "
                "(lost meta reads as a confident 0% reuse)"
            )
        if (
            strict_serialization
            and "trace_n_tokens" in s
            and s["trace_n_tokens"] != s["prompt_tokens"]
        ):
            problems.append(
                f"{s['label']}: trace tokenized {s['trace_n_tokens']} tokens, "
                f"harness declared {s['prompt_tokens']} (serialization drift)"
            )
        if s.get("served") is not None and s.get("held") is None:
            problems.append(f"{s['label']}: served {s['served']} with no held value (unjoined)")
        if s.get("served") is not None and s.get("held") is not None and s["served"] != s["held"]:
            problems.append(f"{s['label']}: served {s['served']} != held {s['held']}")
        if s.get("served") is not None and s["served"] > s["declared_lcp"]:
            problems.append(
                f"{s['label']}: served {s['served']} > declared {s['declared_lcp']} "
                "(reuse invented)"
            )
    return problems


def summarize(name: str, leg: str, samples: list[dict]) -> dict:
    for s in samples:
        s.pop("_tokens", None)  # token ids are per-leg scratch, not artifact
    ttfts = [s["ttft_ms"] for s in samples]
    tpots = [s["tpot_ms"] for s in samples]
    req_tokens = sum(s["prompt_tokens"] for s in samples)
    served = sum(s["served"] or 0 for s in samples)
    declared = sum(s["declared_lcp"] for s in samples)
    instrumented = [s for s in samples if s.get("served") is not None]
    verdicts: dict[str, int] = {}
    for s in samples:
        v = s["verdict"] or "n/a"
        verdicts[v] = verdicts.get(v, 0) + 1
    # Reuse rates are None when no request carried served data: an
    # uninstrumented leg (the direct baseline) is NOT a measured zero, and
    # printing 0.0 for it invites a false daemon-vs-baseline reuse compare.
    any_served = bool(instrumented)
    return {
        "workload": name,
        "leg": leg,
        "n": len(samples),
        "n_instrumented": len(instrumented),
        "ttft_p50_ms": pct(ttfts, 50),
        "ttft_p95_ms": pct(ttfts, 95) if len(ttfts) >= P95_MIN_N else None,
        "ttft_mean_ms": round(statistics.fmean(ttfts), 1) if ttfts else None,
        "tpot_p50_ms": pct(tpots, 50),
        "prompt_tokens": req_tokens,
        "declared_prefix_tokens": declared,
        "served_prefix_tokens": served,
        "reuse_pct_of_requested": (
            round(100 * served / req_tokens, 2) if any_served and req_tokens else None
        ),
        "served_pct_of_declared": (
            round(100 * served / declared, 2) if any_served and declared else None
        ),
        "verdicts": verdicts,
    }


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------


def run_daemon_arm(model: str, args, prefix: str) -> tuple[list[dict], dict[str, str]]:
    """Every workload runs cold (empty store, fresh process) then warm (fresh
    process, SAME store).

    Each workload gets its OWN store. Sharing one would let W1's primed prefix
    satisfy W1np's requests, so W1np would report reuse it never offered — the
    exact confound the unprimed arm exists to expose.
    """
    rows: list[dict] = []
    hello = {"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}

    workloads = []
    if "W1" in args.workloads:
        workloads.append(build_w1(prefix, args.questions, "W1"))
    if "W1np" in args.workloads:
        workloads.append(build_w1_unprimed(prefix, args.questions))
    if "W2" in args.workloads:
        workloads.append(build_w2(prefix, args.questions))
    if "W3" in args.workloads:
        workloads.append(build_w3(args.questions))

    trace_paths: dict[str, str] = {}
    for wl in workloads:
        store = tempfile.mkdtemp(prefix=f"mlxcache-{wl['name']}-")
        # One corpus per workload, threaded through BOTH legs: the warm leg's
        # store still holds the cold leg's keys, so declared reuse must count
        # them or a legitimate cross-restart hit reads as fabricated reuse.
        corpus: list[list[int]] = []
        try:
            for leg in ("cold", "warm"):
                # A FRESH STORE is not cold under a live daemon: a missing blob
                # makes the sidecar 422, the daemon quarantines and retries, and
                # correct_retire rewrites the stats. Cold means a new store AND
                # a new process.
                #
                # One trace file PER (workload, leg), created with mkstemp:
                # TraceWriter truncates on start ("a trace is a session's
                # story"), so reusing a path would wipe earlier records — and a
                # PREDICTABLE path under a shared TMPDIR is a symlink a local
                # attacker can pre-plant for a truncation.
                key = f"{wl['name']}/{leg}"
                trace_fd, trace_path = tempfile.mkstemp(
                    prefix=f"trace-{wl['name']}-{leg}-", suffix=".jsonl"
                )
                os.close(trace_fd)
                trace_paths[key] = trace_path
                stack = DaemonStack(model, store, env_extra={"MLXCACHE_TRACE": trace_path})
                try:
                    stack.wait_ready()
                    # /stats baselines taken around the leg feed the W5
                    # cross-check below; the hello warmup's traffic is excluded
                    # by snapshotting after it settles.
                    _post_json(
                        stack.base,
                        "/v1/chat/completions",
                        hello,
                        REQUEST_TIMEOUT_S,
                    )
                    # The warmth request must be ON DISK before the baseline is
                    # taken, or every sample joins its predecessor's record. It
                    # is the first request this daemon serves, so exactly one
                    # record is expected.
                    base_n = len(_await_trace_records(trace_path, 1))
                    if base_n != 1:
                        raise SystemExit(f"{trace_path}: expected 1 warmup record, found {base_n}")
                    pre_stats = stack.stats()
                    samples = run_leg(stack, wl, _tokenize_for_daemon(stack), corpus=corpus)
                    join_trace(samples, trace_path, base_n)
                    problems = check_premises(samples)
                    post_stats = stack.stats()
                    stats_delta = _harness.stats_delta(pre_stats, post_stats, "requests")
                    row = summarize(wl["name"], f"daemon/{leg}", samples)
                    row["samples"] = samples
                    # W5: the trace slice for THIS leg must score identically to
                    # the per-response accounting above, and both must agree
                    # with the /stats request counter delta. A gap means one of
                    # the three views of the same requests is lying — abort,
                    # never print the leg.
                    scored = score_trace(trace_path, samples, offset=base_n)
                    row["trace_replay"] = scored
                    if problems:
                        raise SystemExit(
                            "PREMISE VIOLATION (harness/daemon bug):\n  " + "\n  ".join(problems)
                        )
                    if (
                        scored is None
                        or scored.get("error")
                        or not (
                            scored.get("tokens_total_matches_samples")
                            and scored.get("served_matches_samples")
                        )
                    ):
                        raise SystemExit(
                            f"TRACE DISAGREEMENT for {wl['name']}/{leg}: {scored} — "
                            "the trace and the response bodies are different "
                            "views of the same requests; the leg cannot be trusted"
                        )
                    if stats_delta != len(samples):
                        raise SystemExit(
                            f"/STATS DISAGREEMENT for {wl['name']}/{leg}: counter "
                            f"delta {stats_delta} != {len(samples)} requests — "
                            "the stats view and the responses disagree"
                        )
                    rows.append(row)
                    rows.append({"workload": "_stats", "leg": f"daemon/{leg}", "stats": post_stats})
                finally:
                    stack.stop()
        finally:
            shutil.rmtree(store, ignore_errors=True)
    return rows, trace_paths


def _tokenize_for_daemon(stack: DaemonStack):
    def tok(messages: list[dict]) -> list[int]:
        return stack.tokenize(serialize_messages(messages))

    return tok


def run_direct_arm(model: str, args, prefix: str) -> list[dict]:
    """Plain mlx_lm.server over the same traffic, twice, plus an in-process repeat.

    `restart` is the baseline's honest shape: its prompt cache is process-local,
    so a restart is always a cold prefill. `inproc` shows the cache it does have,
    so the comparison is not a straw man.

    The traffic is the daemon's own serialization, sent through
    /v1/completions (DirectStack.completion_request): the daemon never applies
    a chat template, so a templated baseline would answer a different prompt.
    Only W1 and W3 have baseline analogues; the arm runs the intersection of
    --workloads with those, so `--workloads W1np` does not silently benchmark
    traffic the daemon arm never ran.
    """
    rows: list[dict] = []
    workloads = []
    if "W1" in args.workloads:
        workloads.append(build_w1(prefix, args.questions, "W1"))
    if "W3" in args.workloads:
        workloads.append(build_w3(args.questions))
    if not workloads:
        return rows
    for wl in workloads:
        for leg in ("restart1", "restart2"):
            stack = DirectStack(model)
            try:
                stack.wait_ready()
                samples = run_leg(stack, wl, stack.tokenize)
                # The baseline gets the premise treatment too, minus the
                # serialization check (it has no trace to join): a served
                # value above declared, or a lost meta, is as fictional here
                # as on the daemon arm.
                problems = check_premises(samples, strict_serialization=False)
                if problems:
                    raise SystemExit(
                        "PREMISE VIOLATION (harness bug, direct arm):\n  " + "\n  ".join(problems)
                    )
                rows.append(summarize(wl["name"], f"direct/{leg}", samples))
                if leg == "restart1":
                    inproc = run_leg(stack, wl, stack.tokenize)
                    rows.append(summarize(wl["name"], "direct/inproc", inproc))
            finally:
                stack.stop()
    return rows


def run_pressure_sweep(model: str, args, prefix: str) -> list[dict]:
    """W4: does the prefix survive k unrelated requests under a tight cap?"""
    rows: list[dict] = []
    cap = 3
    for arm in ("p_alone", "p_extended"):
        for k in args.k:
            wl = build_w4(prefix, k, arm)
            store = tempfile.mkdtemp(prefix=f"mlxcache-w4-{arm}-{k}-")
            trace_fd, trace_path = tempfile.mkstemp(prefix=f"trace-w4-{arm}-{k}-", suffix=".jsonl")
            os.close(trace_fd)
            stack = DaemonStack(
                model,
                store,
                env_extra={
                    "MLXCACHE_EVICT_INTERVAL_S": "1",
                    "MLXCACHE_EVICT_MAX_ENTRIES": str(cap),
                    "MLXCACHE_EVICT_ANCHOR_WINDOW_S": "0",
                    "MLXCACHE_TRACE": trace_path,
                },
            )
            try:
                stack.wait_ready()
                before_stats = stack.stats()
                corpus: list[list[int]] = []
                noise = run_leg(stack, wl, _tokenize_for_daemon(stack), corpus=corpus)
                # Premise checks read held/serialization fields that only
                # join_trace attaches; checking unjoined samples is what once
                # crashed this arm with KeyError: 'held'.
                join_trace(noise, trace_path, 0)
                # Drop the scratch ids BEFORE the probe appends its own, so the
                # corpus carries the token streams and nothing else.
                corpus = [s.pop("_tokens") for s in noise]
                # Wait for one sweep to land BETWEEN the noise and the probe.
                # The reaper skips its first tick, so the first real pass lands
                # ~2s after boot at a 1s interval. Probing without this wait
                # would measure the pre-eviction cache and label it pressure.
                evicted = 0
                noise_stats = stack.stats()
                if "evictions" not in noise_stats or "checkpoints_published" not in noise_stats:
                    raise SystemExit(
                        "/stats did not report evictions/checkpoints_published: "
                        "a missing counter would reconcile as zero and route "
                        "the arm into the wrong branch"
                    )
                deadline = time.time() + EVICT_WAIT_S
                while time.time() < deadline:
                    now = _harness.stats_delta(before_stats, stack.stats(), "evictions")
                    if now > 0:
                        evicted = now
                        break
                    time.sleep(0.5)
                # An expired wait is not a pressure result: if the store holds
                # more than the cap, the reaper never bound and the row would
                # measure an un-evicted cache while reading green. The noise
                # samples still get the FULL premise treatment first — the
                # degraded-run path is exactly where an accounting lie must
                # abort, not print. Per-arm disk headroom re-check: the sweep
                # writes ~k checkpoints of ~190 MB at real sizes and the
                # one-shot preflight cannot see a disk filling mid-sweep.
                headroom = check_disk_headroom(tempfile.gettempdir(), args.min_free_gb * 1e9)
                if headroom:
                    raise SystemExit("PREFLIGHT (mid-sweep): " + "; ".join(headroom))
                problems = check_premises(noise)
                if problems:
                    raise SystemExit(
                        "PREMISE VIOLATION (harness/daemon bug, pressure):\n  "
                        + "\n  ".join(problems)
                    )
                # Refresh AFTER the premise checks and immediately before the
                # decision: the reaper removes entries one at a time and
                # records the count only when its pass returns, so an earlier
                # read can observe published > cap while evicted still holds
                # a stale zero from the wait loop.
                after_stats = stack.stats()
                evicted = _harness.stats_delta(before_stats, after_stats, "evictions")
                published_after = after_stats["checkpoints_published"]
                if evicted == 0 and published_after > cap:
                    row = summarize(
                        wl["name"], f"daemon/pressure_cap{cap}_evict_wait_timeout", noise
                    )
                    row["samples"] = noise
                    row["evictions_observed"] = 0
                    row["published_before_probe"] = published_after
                    row["repeat_verdict"] = None
                    row["repeat_served"] = None
                    row["repeat_declared"] = None
                    row["repeat_ttft_ms"] = None
                    rows.append(row)
                    continue
                probe = run_leg(
                    stack,
                    wl,
                    _tokenize_for_daemon(stack),
                    requests=[wl["probe"]],
                    corpus=corpus,
                )
                join_trace(probe, trace_path, len(noise))
                problems = check_premises(noise + probe)
                if problems:
                    raise SystemExit(
                        "PREMISE VIOLATION (harness/daemon bug):\n  " + "\n  ".join(problems)
                    )
                row = summarize(wl["name"], f"daemon/pressure_cap{cap}", noise)
                row["samples"] = noise
                row["probe_sample"] = probe[0]
                row["evictions_observed"] = evicted
                row["published_before_probe"] = published_after
                row["repeat_verdict"] = probe[0]["verdict"]
                row["repeat_served"] = probe[0]["served"]
                row["repeat_declared"] = probe[0]["declared_lcp"]
                row["repeat_ttft_ms"] = probe[0]["ttft_ms"]
                rows.append(row)
            finally:
                stack.stop()
                shutil.rmtree(store, ignore_errors=True)
                Path(trace_path).unlink(missing_ok=True)
    return rows


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def score_trace(trace_path: str, samples: list[dict], offset: int = 0) -> dict | None:
    """W5: score this leg's captured records with the trace tooling's scorer.

    The trace and the per-response body are two views of one run, so their
    token totals and reuse totals must agree. A gap means one of them is
    lying, and the leg's numbers in this report cannot be trusted — the
    caller treats any disagreement (or a missing/error result) as a premise
    violation and aborts.
    """
    if not samples or not Path(trace_path).is_file():
        return None
    records = replay_trace.read_trace(trace_path)[offset : offset + len(samples)]
    if len(records) != len(samples):
        return {"error": f"trace had {len(records)} records for {len(samples)} requests"}
    scored = replay_trace.score_records(records, "leg trace")
    scored["samples_tokens_total"] = sum(s["prompt_tokens"] for s in samples)
    scored["samples_served"] = sum(s["served"] or 0 for s in samples)
    scored["tokens_total_matches_samples"] = (
        scored["tokens_total"] == scored["samples_tokens_total"]
    )
    scored["served_matches_samples"] = scored["tokens_reused"] == scored["samples_served"]
    return scored


def print_table(rows: list[dict]) -> None:
    keep = [r for r in rows if r.get("workload") != "_stats"]
    print(
        "\n| workload | leg | n | TTFT p50 | TPOT p50 | served/req % "
        "| served/declared % | verdicts |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for r in keep:
        sd = r["served_pct_of_declared"]
        print(
            f"| {r['workload']} | {r['leg']} | {r['n']} | {r['ttft_p50_ms']} "
            f"| {r['tpot_p50_ms']} | {r['reuse_pct_of_requested']} "
            f"| {'n/a' if sd is None else sd} | {r['verdicts']} |"
        )


def _chip_identity() -> str:
    """Best-effort chip name for artifact provenance; empty when unknown."""
    try:
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001 — provenance must never fail a run
        return ""


def _git_out(*args: str) -> str | None:
    """Best-effort, status-aware git query for artifact provenance: bounded
    (a hung git must not stall the run). None means the command could NOT be
    completed — unavailable git, a non-repository, a timeout, a nonzero exit;
    an empty STRING is a successful empty result. Callers must not infer
    "clean" from an empty output without checking for None first."""
    try:
        out = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=REPO,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:  # noqa: BLE001 — provenance must never fail a run
        return None


def _daemon_spawn_mode() -> str:
    """How the daemon child was spawned: prebuilt binary path or cargo run."""
    binary = REPO / "target/release/mlxcache-daemon"
    return str(binary) if binary.is_file() else "cargo run --release"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default=_harness.DEFAULT_MODEL)
    ap.add_argument("--workloads", default="W1,W1np,W2,W3")
    ap.add_argument("--questions", type=int, default=6, help="varying requests per workload")
    ap.add_argument("--prefix-tokens", type=int, default=2000, help="approx shared-prefix size")
    ap.add_argument("--k", default="0,1,2,4,8,16", help="W4 noise counts (comma separated)")
    ap.add_argument("--arms", default="daemon,direct", help="daemon,direct,pressure")
    ap.add_argument("--json", default=None, help="write the full artifact here")
    ap.add_argument(
        "--min-free-gb",
        type=float,
        default=3.0,
        help="refuse to start below this much free disk in the temp dir "
        "(checkpoints are ~57 MB per 1 K tokens at 7B-4bit; a failed publish "
        "is reported as a miss, so a full disk silently zeroes the result)",
    )
    args = ap.parse_args()

    args.workloads = [w.strip() for w in args.workloads.split(",") if w.strip()]
    args.arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    args.k = [int(x) for x in str(args.k).split(",") if str(x).strip()]

    unknown_w = [w for w in args.workloads if w not in KNOWN_WORKLOADS]
    unknown_a = [a for a in args.arms if a not in KNOWN_ARMS]
    if unknown_w or unknown_a:
        raise SystemExit(
            f"unknown workload(s) {unknown_w} / arm(s) {unknown_a}; "
            f"workloads: {sorted(KNOWN_WORKLOADS)}, arms: {sorted(KNOWN_ARMS)}"
        )
    if "W4" in args.workloads and "pressure" not in args.arms:
        raise SystemExit("W4 is pressure-only: select it with --arms pressure")
    if "daemon" in args.arms and not set(args.workloads) & {"W1", "W1np", "W2", "W3"}:
        raise SystemExit("the daemon arm has nothing to run: no W1/W1np/W2/W3 selected")
    if "direct" in args.arms and not set(args.workloads) & {"W1", "W3"}:
        raise SystemExit("the direct arm has nothing to run: no W1/W3 selected")
    if set(args.workloads) & {"W1np", "W2"} and "direct" in args.arms:
        print("note: W1np/W2 have no direct-baseline analogue; direct arm runs W1/W3 only")

    model = resolve_model(args.model)
    tokens_per_repeat = measure_filler_tokens(model)
    prefix = filler_text(args.prefix_tokens, tokens_per_repeat)
    print(f"model:  {model}")
    print(
        f"prefix: ~{args.prefix_tokens} tokens requested "
        f"(calibrated: {tokens_per_repeat} tokens/filler-repeat)"
    )
    print(f"grid:   workloads={args.workloads} arms={args.arms} n={args.questions} k={args.k}\n")

    # Refuse a run whose checkpoints cannot fit: a failed publish is silently
    # reported as a miss, so a full disk reads as "the cache does nothing".
    disk = check_disk_headroom(tempfile.gettempdir(), args.min_free_gb * 1e9)
    if disk:
        raise SystemExit("PREFLIGHT: " + "; ".join(disk))

    rows: list[dict] = []
    traces: dict[str, str] = {}
    if "daemon" in args.arms:
        daemon_rows, traces = run_daemon_arm(model, args, prefix)
        rows += daemon_rows
        print_table(daemon_rows)
    if "direct" in args.arms:
        direct_rows = run_direct_arm(model, args, prefix)
        rows += direct_rows
        print_table(direct_rows)
    if "pressure" in args.arms:
        pressure_rows = run_pressure_sweep(model, args, prefix)
        rows += pressure_rows
        print_table(pressure_rows)

    artifact = {
        "model": model,
        "provenance": {
            "argv": sys.argv,
            "source_revision": _git_out("rev-parse", "HEAD") or "unknown",
            # None = the status query itself could not run (non-repo, no git,
            # timeout): unknown, NOT inferred-clean from an empty output.
            "source_dirty": (
                None if (status := _git_out("status", "--porcelain")) is None else bool(status)
            ),
            "machine": platform.machine(),
            "system": platform.platform(),
            "chip": _chip_identity(),
            "daemon": _daemon_spawn_mode(),
            "correctness": "premise guards enforced (serialization parity, "
            "served==held, served<=declared, W5 trace+stats agreement; any "
            "violation aborts before this artifact is written)",
        },
        "prefix_tokens_requested": args.prefix_tokens,
        "questions": args.questions,
        "k": args.k,
        "traces": traces,
        "rows": rows,
    }
    out = args.json or str(REPO / "bench" / f"prefix-bench-{int(time.time())}.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(artifact, indent=2))
    print(f"\nartifact: {out}")
    for leg, path in traces.items():
        print(f"trace[{leg}]: {path}")
        print(f"  score: scripts/replay_trace.py --trace {path} --summary-only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
