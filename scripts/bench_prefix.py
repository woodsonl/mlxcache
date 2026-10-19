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

  # arithmetic/wiring only, no model (CI): synthetic engine
  uv run python scripts/bench_prefix.py --engine synthetic --self-check

The model must already be on disk. This harness never downloads one.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import statistics
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _harness  # noqa: E402 — path insert above
from _harness import REPO, free_port, resolve_model  # noqa: E402

P95_MIN_N = 100

# Filler that tokenizes to roughly one token per four characters. Repeated to
# build a long, boring, real-tokenizer prefix. Nothing here is special-cased by
# the engine; it is ordinary prose because the benchmark must not exploit the
# synthetic tokenizer's sentence hashing.
_FILLER = (
    "The retention policy for a shared cache depends on the working set of the "
    "caller and on how much of the prefix survives between requests. "
)


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


def filler_text(approx_tokens: int) -> str:
    """Deterministic prose filler of roughly `approx_tokens` tokens.

    Approximate on purpose: the real count is measured through the real
    tokenizer and reported. Trimming to an exact count would require the
    tokenizer at workload-build time in both arms, which changes the text the
    direct arm sees and makes the two arms measure different prompts.
    """
    return _FILLER * max(1, approx_tokens // 16)


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


def build_w1(prefix: str, questions: int, name: str) -> dict:
    """Fixed prefix + short changing question, primed with the prefix itself.

    The priming request's stream ends exactly one token past the shared region
    (`"}]` closes the JSON array), which is the boundary the index's child rule
    can serve. Without it, every question diverges mid-key and is a miss — that
    is W1np, not a bug in this construction.
    """
    reqs = [{"label": f"{name}:prime", "messages": [{"role": "system", "content": prefix}]}]
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
    reqs = []
    for i in range(questions):
        reqs.append(
            {
                "label": f"W1np:q{i}",
                "messages": [
                    {"role": "system", "content": prefix},
                    {"role": "user", "content": f"Question {i}: answer in one short paragraph."},
                ],
            }
        )
    return {"name": "W1np_shared_prefix_unprimed", "requests": reqs, "max_tokens": 32}


def build_w2(prefix: str, questions: int) -> dict:
    """Near miss: the shared region diverges in its FIRST tokens.

    A tail-only difference would be a legitimate hit under the end-anchored
    rule, so the divergence has to be early for this workload to mean
    "similar text, different document" rather than "same prefix".
    """
    other = "Instead of " + prefix[len("The retention") :]
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
    exactly the difference being measured.
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
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data: "):
                continue
            frag = line[len("data: ") :]
            if frag == "[DONE]":
                break
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
            if "error" in frame:
                raise RuntimeError(f"stream error frame: {frame['error']}")
            if _generated_text(frame):
                if ttft is None:
                    ttft = (time.perf_counter() - t0) * 1000
                tokens += 1
    total = (time.perf_counter() - t0) * 1000
    if ttft is None:
        raise RuntimeError("stream closed before any token frame")
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


def _wait_stats(base: str, timeout_s: float) -> None:
    """The daemon has no serving readiness gate: /stats answering 200 means the
    router is wired. The sidecar's model load is proven separately by a real
    request, because a bound port is not a loaded model."""
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        try:
            _stats(base)
            return
        except urllib.error.HTTPError:
            return
        except Exception as exc:  # noqa: BLE001 — not up yet
            last = repr(exc)
            time.sleep(0.3)
    raise RuntimeError(f"daemon never answered /stats ({last})")


# ---------------------------------------------------------------------------
# Stacks
# ---------------------------------------------------------------------------


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
        sidecar_env = dict(env, MLXCACHE_ENGINE="mlx-lm", PYTHONUNBUFFERED="1")
        self.sidecar = _harness.Server(
            [
                str(REPO / ".venv/bin/python"),
                "-c",
                "import sys; sys.path.insert(0, 'sidecar'); "
                "from mlxcache_sidecar import server; "
                f"server.Handler.engine = server.make_engine({model!r}); "
                f"server.BurstServer(('127.0.0.1', {self.sidecar_port}), "
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
        self.daemon = self._spawn_daemon(daemon_env)

    def _spawn_daemon(self, env: dict) -> _harness.Server:
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
        return _harness.Server(
            argv,
            cwd=REPO,
            env=env,
            port=self.port,
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

    def wait_ready(self, timeout_s: float = 600.0) -> None:
        """Wait for BOTH processes to serve, not just for the daemon's router.

        The daemon's /stats answers as soon as axum is up, while the sidecar
        binds its listener only AFTER the model finishes loading (it constructs
        the engine before `BurstServer`). Polling the daemon alone therefore
        races a multi-second model load and surfaces as a 503
        `adapter_unavailable` on the first real request.
        """
        _wait_stats(self.base, timeout_s)
        self.sidecar.wait_ready(timeout_s)

    def stats(self) -> dict:
        return _stats(self.base)

    def stop(self) -> None:
        self.daemon.stop()
        self.sidecar.stop()


class DirectStack:
    """Plain mlx_lm.server: in-memory LRUPromptCache, nothing survives a restart.

    This is the cross-process baseline. It is not crippled: its in-process
    repeat leg measures its own prompt cache, which is what a caller gets
    without mlxcache.
    """

    def __init__(self, model: str, prompt_cache_size: int | None = None):
        self.model = model
        self._tok = None
        self.port = free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        argv = [
            str(REPO / ".venv/bin/python"),
            "-m",
            "mlx_lm",
            "server",
            "--model",
            model,
            "--port",
            str(self.port),
            "--log-level",
            "ERROR",
        ]
        if prompt_cache_size is not None:
            argv += ["--prompt-cache-size", str(prompt_cache_size)]
        self.proc = _harness.Server(
            argv, cwd=REPO, port=self.port, base=self.base, log_prefix="mlxcache-bench-direct-"
        )

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
        baseline: the same bytes the daemon's engine consumes.
        """
        body = {"model": self.model, "prompt": serialize_messages(messages)}
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        return "/v1/completions", body

    def wait_ready(self, timeout_s: float = 600.0) -> None:
        """A real completion, not /health: mlx_lm.server binds its port before
        the weights are loaded, so a port check would race the model."""
        deadline = time.time() + timeout_s
        last = None
        while time.time() < deadline:
            try:
                path, body = self.completion_request([{"role": "user", "content": "hi"}], 1)
                _post_json(self.base, path, body, 120)
                return
            except Exception as exc:  # noqa: BLE001 — still loading
                last = repr(exc)
                if self.proc.proc.poll() is not None:
                    raise RuntimeError(f"direct server exited:\n{self.proc._log_tail()}") from exc
                time.sleep(1.0)
        raise RuntimeError(f"direct server never served ({last})")

    def stop(self) -> None:
        self.proc.stop()


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
        timing, meta = _post_stream(stack.base, path, body, timeout=900)
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
            "daemon_tokens_total": (meta or {}).get("tokens_total"),
            "daemon_ttft_ms": (meta or {}).get("ttft_ms"),
            "lookup_ms": (meta or {}).get("lookup_ms"),
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


def join_trace(samples: list[dict], trace_path: str, base_n: int) -> None:
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
         prefix in this run is fiction.
      2. served == held — `prefill_from` must equal `matched_tokens - 1`.
      3. served <= declared — the daemon cannot serve reuse the traffic never
         offered.
    """
    records = _await_trace_records(trace_path, base_n + len(samples))[base_n:]
    if len(records) < len(samples):
        raise RuntimeError(
            f"{trace_path}: trace has {len(records)} records for {len(samples)} "
            f"requests after offset {base_n} (capture lag or a dropped write); "
            "the accounting cannot be joined"
        )
    for sample, rec in zip(samples, records, strict=False):
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
    daemon's accounting, never a benchmark result — the caller aborts."""
    problems = []
    for s in samples:
        if (
            strict_serialization
            and "trace_n_tokens" in s
            and s["trace_n_tokens"] != s["prompt_tokens"]
        ):
            problems.append(
                f"{s['label']}: trace tokenized {s['trace_n_tokens']} tokens, "
                f"harness declared {s['prompt_tokens']} (serialization drift)"
            )
        if s["served"] is not None and s["held"] is not None and s["served"] != s["held"]:
            problems.append(f"{s['label']}: served {s['served']} != held {s['held']}")
        if s["served"] is not None and s["served"] > s["declared_lcp"]:
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
    verdicts: dict[str, int] = {}
    for s in samples:
        v = s["verdict"] or "n/a"
        verdicts[v] = verdicts.get(v, 0) + 1
    return {
        "workload": name,
        "leg": leg,
        "n": len(samples),
        "ttft_p50_ms": pct(ttfts, 50),
        "ttft_p95_ms": pct(ttfts, 95) if len(ttfts) >= P95_MIN_N else None,
        "ttft_mean_ms": round(statistics.fmean(ttfts), 1) if ttfts else None,
        "tpot_p50_ms": pct(tpots, 50),
        "tokens_per_s": None,
        "prompt_tokens": req_tokens,
        "declared_prefix_tokens": declared,
        "served_prefix_tokens": served,
        "reuse_pct_of_requested": round(100 * served / req_tokens, 2) if req_tokens else 0.0,
        "served_pct_of_declared": round(100 * served / declared, 2) if declared else None,
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
                # One trace file PER (workload, leg): TraceWriter truncates on
                # start ("a trace is a session's story"), so reusing a path
                # would wipe earlier records.
                key = f"{wl['name']}/{leg}"
                trace_path = str(
                    Path(store).parent / f"trace-{wl['name']}-{leg}-{int(time.time())}.jsonl"
                )
                trace_paths[key] = trace_path
                stack = DaemonStack(model, store, env_extra={"MLXCACHE_TRACE": trace_path})
                try:
                    stack.wait_ready()
                    _post_json(stack.base, "/v1/chat/completions", hello, 900)
                    # The warmth request must be ON DISK before the baseline is
                    # taken, or every sample joins its predecessor's record. It
                    # is the first request this daemon serves, so exactly one
                    # record is expected.
                    base_n = len(_await_trace_records(trace_path, 1))
                    if base_n != 1:
                        raise SystemExit(f"{trace_path}: expected 1 warmup record, found {base_n}")
                    samples = run_leg(stack, wl, _tokenize_for_daemon(stack), corpus=corpus)
                    join_trace(samples, trace_path, base_n)
                    problems = check_premises(samples)
                    if problems:
                        raise SystemExit(
                            "PREMISE VIOLATION (harness/daemon bug):\n  " + "\n  ".join(problems)
                        )
                    row = summarize(wl["name"], f"daemon/{leg}", samples)
                    row["samples"] = samples
                    # W5: the trace slice for THIS leg must score identically to
                    # the per-response accounting above. A gap means the trace
                    # and the response body disagree about the same request.
                    row["trace_replay"] = score_trace(trace_path, samples, offset=base_n)
                    rows.append(row)
                    rows.append(
                        {"workload": "_stats", "leg": f"daemon/{leg}", "stats": stack.stats()}
                    )
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
    """
    rows: list[dict] = []
    workloads = [build_w1(prefix, args.questions, "W1")]
    if "W3" in args.workloads:
        workloads.append(build_w3(args.questions))
    for wl in workloads:
        for leg in ("restart1", "restart2"):
            stack = DirectStack(model)
            try:
                stack.wait_ready()
                samples = run_leg(stack, wl, stack.tokenize)
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
            trace_fd, trace_path = tempfile.mkstemp(prefix=f"trace-w4-{arm}-{k}-")
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
                before = stack.stats().get("evictions", 0)
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
                deadline = time.time() + 25
                while time.time() < deadline:
                    now = stack.stats().get("evictions", 0)
                    if now > before:
                        evicted = now - before
                        break
                    time.sleep(0.5)
                after_stats = stack.stats()
                published_after = after_stats.get("checkpoints_published")
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
    lying, and the leg's numbers in this report cannot be trusted.
    """
    if not samples or not Path(trace_path).is_file():
        return None
    spec = importlib.util.spec_from_file_location(
        "replay_trace", Path(__file__).resolve().parent / "replay_trace.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    records = mod.read_trace(trace_path)[offset : offset + len(samples)]
    if len(records) != len(samples):
        return {"error": f"trace had {len(records)} records for {len(samples)} requests"}
    scored = mod.score_records(records, "leg trace")
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
    ap.add_argument("--engine", default="real", choices=["real", "synthetic"])
    ap.add_argument("--json", default=None, help="write the full artifact here")
    ap.add_argument(
        "--self-check", action="store_true", help="exit after the premise arithmetic tests"
    )
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

    if args.self_check:
        return _self_check()

    if args.engine == "synthetic":
        raise SystemExit(
            "the synthetic engine hashes whole prompts into 8 tokens and cannot "
            "express prefix reuse; use scripts/bench_workloads.py for plumbing, "
            "or --self-check for the arithmetic"
        )

    model = resolve_model(args.model)
    prefix = filler_text(args.prefix_tokens)
    print(f"model:  {model}")
    print(f"prefix: ~{args.prefix_tokens} tokens requested")
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


def _self_check() -> int:
    """Arithmetic and premise guards, no model. Runs in CI."""
    failures = []
    if lcp_len([1, 2, 3], [1, 2, 9]) != 2:
        failures.append("lcp_len basic")
    if lcp_len([], [1]) != 0:
        failures.append("lcp_len empty")
    if tpot_ms(1100.0, 100.0, 11) != 100.0:
        failures.append("tpot_ms divisor")
    # A 1-token completion must not divide by zero or go negative.
    if tpot_ms(1100.0, 100.0, 1) != 1000.0:
        failures.append("tpot_ms single token")
    good = [
        {
            "label": "a",
            "prompt_tokens": 10,
            "served": 9,
            "held": 9,
            "declared_lcp": 9,
            "trace_n_tokens": 10,
        }
    ]
    if check_premises(good):
        failures.append("premises reject a valid sample")
    bad_served = [dict(good[0], served=11)]
    if not check_premises(bad_served):
        failures.append("premises accept served > declared")
    bad_held = [dict(good[0], held=8)]
    if not check_premises(bad_held):
        failures.append("premises accept served != held")
    bad_ser = [dict(good[0], trace_n_tokens=11)]
    if not check_premises(bad_ser):
        failures.append("premises accept serialization drift")
    ser = serialize_messages([{"role": "user", "content": 'a "b"'}])
    if ser != '[{"role":"user","content":"a \\"b\\""}]':
        failures.append(f"serde serialization mismatch: {ser}")
    if failures:
        print("SELF-CHECK FAILED: " + "; ".join(failures))
        return 1
    print("self-check: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
