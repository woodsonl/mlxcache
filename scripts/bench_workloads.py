#!/usr/bin/env python3
"""Plumbing regression harness: daemon+sidecar request path over real HTTP.

This is a WIRING check, not a cache-benefit benchmark. It runs the synthetic
engine, whose tokenizer hashes the whole prompt to 8 tokens
(sidecar/mlxcache_sidecar/server.py `_hash_tokens`) and whose "prefill" emits
a deterministic byte payload instead of running a model. A cold request
therefore costs an 8-token prefill, so TTFT differences here measure proxy
plumbing and blob writes — never prefill savings from a long shared prefix.

What it does cover, and should keep covering:

- cold path (first request for a prompt)
- warm exact hit (second request for the same prompt)
- growing conversation (delta prefill — the OV3 lever)
- concurrent identical load (single-flight coalescing under contention)
- concurrent distinct load (hot-path parallelism)

Every workload reports p50 (and p95 when n>=100) for TTFT (first token of the
assistant reply) and TPOT (per decoded token, from the REAL completion token
count), so a regression in ANY layer — tokenize, route, prefill, publish,
stream — shows up where the client feels it.

For real-engine cache-benefit numbers (declared vs served prefix, cold/warm,
pressure sweep) use scripts/bench_prefix.py (it lands with the
prefix-benchmark unit of this change stack).

Usage:
    uv run python scripts/bench_workloads.py [--daemon URL] [--requests N]
                                             [--concurrency C] [--json OUT]

The daemon and sidecar are started automatically if --daemon is not given
(uv run cargo run --release -- serve ...), exercised, and torn down.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _harness import KILL_WAIT_S, TERM_WAIT_S, free_port  # noqa: E402 — path insert above


def stream_chat(url: str, body: dict, timeout: float = 120.0) -> tuple[float, float, int]:
    """POST a streaming chat completion; return (ttft_ms, total_ms, tokens).

    TTFT = request sent → first token frame received. The daemon can emit an
    `mlxcache` stats frame or an error frame before the first token, so the
    stamp waits for a frame that is none of those (and not [DONE]) — stamping
    on the first `data:` line of any kind would overstate TTFT by the cache
    decision time. The per-frame loop reads the raw SSE byte stream like a
    real client, under a MONOTONIC TOTAL deadline: the socket timeout bounds
    one read only, so a server dripping a frame every few seconds would
    otherwise hold this consumer open forever.
    """
    body = dict(body, stream=True)
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    deadline = time.monotonic() + timeout
    ttft: float | None = None
    tokens = 0
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            if time.monotonic() > deadline:
                raise RuntimeError(f"SSE stream exceeded its {timeout}s total budget")
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[len("data: ") :]
            if payload == "[DONE]":
                break
            try:
                frame = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if "mlxcache" in frame or "error" in frame:
                continue  # stats/error frame, not a token
            # First non-stats, non-error, non-[DONE] frame: the first token.
            if ttft is None:
                ttft = (time.perf_counter() - t0) * 1000
            # Token frames forward the sidecar's native NDJSON shape:
            # {"token": id, "text": piece} (see http.rs push_frame).
            if "token" in frame:
                tokens += 1
    total = (time.perf_counter() - t0) * 1000
    if ttft is None:
        raise RuntimeError("stream closed before any data frame")
    return ttft, total, tokens


def pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = max(0, min(len(s) - 1, round(p / 100 * (len(s) - 1))))
    return s[k]


# A p95 over 30 samples is the second-largest sample wearing a percentile
# label. Report tails only where the sample can actually support them.
P95_MIN_N = 100


def tpot_ms(total_ms: float, ttft_ms: float, tokens: int) -> float:
    """Per-token decode cost, dividing by the tokens ACTUALLY generated.

    A constant divisor here is a lie the moment a workload changes
    max_tokens or the model stops early on EOS: the divisor and the
    measurement drift apart and nothing fails.
    """
    return (total_ms - ttft_ms) / max(tokens - 1, 1)


def report(name: str, ttfts: list[float], totals: list[float], token_counts: list[int]) -> dict:
    if any(n < 1 for n in token_counts):
        raise RuntimeError(
            f"{name}: a request reported {min(token_counts)} generated tokens; "
            "the token counting has drifted and every TPOT below would be fiction"
        )
    # TPOT needs at least one decoded token AFTER the first: a tokens==1
    # sample has an empty post-TTFT window and dividing would fabricate a
    # per-token number out of nothing.
    tpots = [
        tpot_ms(t, ttft, n)
        for ttft, t, n in zip(ttfts, totals, token_counts, strict=True)
        if n >= 2
    ]
    row = {
        "workload": name,
        "n": len(ttfts),
        "ttft_p50_ms": round(pct(ttfts, 50), 2),
        "ttft_p95_ms": round(pct(ttfts, 95), 2) if len(ttfts) >= P95_MIN_N else None,
        "ttft_mean_ms": round(statistics.fmean(ttfts), 2),
        "tpot_p50_ms": round(pct(tpots, 50), 2) if tpots else None,
        "tpot_p95_ms": round(pct(tpots, 95), 2) if len(tpots) >= P95_MIN_N else None,
        "tokens_mean": round(statistics.fmean(token_counts), 1),
    }
    ttft_p95 = f"  p95 {row['ttft_p95_ms']:>8.2f}ms" if row["ttft_p95_ms"] is not None else ""
    tpot_p50 = f"{row['tpot_p50_ms']:>6.2f}ms" if row["tpot_p50_ms"] is not None else "   n/a"
    tpot_p95 = f"  p95 {row['tpot_p95_ms']:>7.2f}ms/token" if row["tpot_p95_ms"] is not None else ""
    print(
        f"  {name:<34} TTFT p50 {row['ttft_p50_ms']:>8.2f}ms{ttft_p95}"
        f"   TPOT p50 {tpot_p50}{tpot_p95}"
        f"  (n={len(ttfts)}, tokens~{row['tokens_mean']})"
    )
    return row


def wait_healthy(url: str, daemon: subprocess.Popen, deadline_s: float = 30.0) -> None:
    # The daemon exposes no /health; a 404 from /stats proves the listener is
    # accepting and routing (any response but connection-refused works).
    # Liveness is checked every iteration too: a daemon that exits at startup
    # (bad env, bind failure after the free_port TOCTOU) must fail FAST with
    # its exit status, not burn the whole deadline on "not healthy".
    deadline = time.time() + deadline_s
    while time.time() < deadline:
        rc = daemon.poll()
        if rc is not None:
            raise RuntimeError(f"daemon exited during startup (code {rc}) at {url}")
        try:
            with urllib.request.urlopen(f"{url}/stats", timeout=1) as r:
                if r.status == 200:
                    return
        except urllib.error.HTTPError:
            return  # routed response: daemon is up
        except Exception:  # noqa: BLE001 — not up yet
            time.sleep(0.2)
    raise RuntimeError(f"daemon did not become healthy at {url}")


def _free_port() -> int:
    # Kept as a thin alias so the call sites read locally; the logic lives in
    # _harness so the port-picking rules cannot drift between harnesses.
    return free_port()


def start_stack(blob_dir: str) -> tuple[subprocess.Popen, subprocess.Popen, int]:
    """Start sidecar (uv, synthetic engine) + daemon (release build).

    The daemon binary is BUILT to completion before ANY spawn: `cargo run
    --release` would fold a cold build (minutes on a fresh checkout) into
    the daemon's 30s readiness window, and the harness would fail as "not
    healthy" while the compiler was still running. Building first also
    keeps partial-failure cleanup simple: once a child exists, every later
    failure path SIGTERMs it before re-raising, so a failed stack never
    strands a process holding a port."""
    sidecar_port = _free_port()
    env = dict(os.environ, MLXCACHE_TOKENIZE_GROW="growing conversation seed")
    subprocess.run(
        ["uv", "run", "cargo", "build", "--release", "-p", "mlxcache-daemon"],
        check=True,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    sidecar = subprocess.Popen(
        [
            "uv",
            "run",
            "python",
            "-c",
            "import sys; sys.path.insert(0, 'sidecar'); "
            "from mlxcache_sidecar import server; "
            "server.Handler.engine = server.make_engine('bench-model'); "
            f"server.BurstServer(('127.0.0.1', {sidecar_port}), "
            "server.Handler).serve_forever()",
        ],
        env=env,
    )
    daemon_port = _free_port()
    try:
        daemon = subprocess.Popen(
            [
                str(Path(__file__).resolve().parents[1] / "target" / "release" / "mlxcache-daemon"),
            ],
            env=dict(
                env,
                MLXCACHE_MODELS="bench-model",
                MLXCACHE_SIDECAR_URL=f"http://127.0.0.1:{sidecar_port}",
                MLXCACHE_BLOBS=blob_dir,
                MLXCACHE_ADDR=f"127.0.0.1:{daemon_port}",
            ),
        )
    except BaseException:
        sidecar.send_signal(signal.SIGTERM)
        try:
            sidecar.wait(timeout=10)
        except subprocess.TimeoutExpired:
            sidecar.kill()
        raise
    return sidecar, daemon, daemon_port


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--daemon", help="existing daemon base URL (skip auto-start)")
    ap.add_argument(
        "--requests",
        type=int,
        default=P95_MIN_N,
        help="per-workload sample count (default keeps p95 emittable)",
    )
    ap.add_argument("--concurrency", type=int, default=8, help="concurrent workers")
    ap.add_argument("--grow-rounds", type=int, default=8, help="conversation growth rounds")
    ap.add_argument(
        "--blobs",
        default=None,
        help="blob dir to REUSE (default: a fresh temp dir per run — a warm disk "
        "from a previous run turns the growing workload's partials into hits)",
    )
    ap.add_argument("--json", help="write results to this file")
    args = ap.parse_args()
    if args.requests < 1 or args.concurrency < 1 or args.grow_rounds < 1:
        raise SystemExit(
            "--requests, --concurrency and --grow-rounds must all be >= 1 "
            f"(got {args.requests}/{args.concurrency}/{args.grow_rounds})"
        )

    procs: list[subprocess.Popen] = []
    url = args.daemon
    blob_dir = args.blobs or tempfile.mkdtemp(prefix="mlxcache-bench-blobs-")
    daemon_proc: subprocess.Popen | None = None
    try:
        if url is None:
            os.makedirs(blob_dir, exist_ok=True)
            sidecar, daemon, port = start_stack(blob_dir)
            daemon_proc = daemon
            procs += [sidecar, daemon]
            url = f"http://127.0.0.1:{port}"
        try:
            wait_healthy(url, daemon_proc)
            results = run_workloads(url, args)
        finally:
            for p in procs:
                p.send_signal(signal.SIGTERM)
            for p in procs:
                try:
                    p.wait(timeout=TERM_WAIT_S)
                except subprocess.TimeoutExpired:
                    p.kill()
                    # A killed child still needs reaping, and a reaped child
                    # is not guaranteed: mirroring the shared Server.stop()
                    # contract instead of leaving a weaker local copy.
                    with contextlib.suppress(subprocess.TimeoutExpired):
                        p.wait(timeout=KILL_WAIT_S)
    finally:
        # The blob dir is GB-scale at real-model sizes; stranding it in $TMPDIR
        # on every run is how a machine fills the volume the harness itself
        # needs for temp logs. (``--blobs ""`` counts as unset: argparse
        # cannot express "empty string means reuse" sanely.)
        if not args.blobs:
            import shutil

            shutil.rmtree(blob_dir, ignore_errors=True)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nwrote {args.json}")
    return 0


def run_workloads(url: str, args) -> list[dict]:
    rows: list[dict] = []
    print(f"daemon {url}  requests={args.requests} concurrency={args.concurrency}\n")

    # 1. Cold path: unique prompts, each a first request. (An 8-token
    #    "prefill" in the synthetic engine — this is overhead, not prefill.)
    ttfts, totals, ntoks = [], [], []
    for i in range(args.requests):
        ttft, total, ntok = stream_chat(
            url,
            {
                "model": "bench-model",
                "messages": [{"role": "user", "content": f"cold prompt number {i}"}],
            },
        )
        ttfts.append(ttft)
        totals.append(total)
        ntoks.append(ntok)
        assert ntok > 0
    rows.append(report("cold_prefill", ttfts, totals, ntoks))

    # 2. Warm exact hit: the same prompt twice; the 2nd+ go through the index.
    prompt = {
        "model": "bench-model",
        "messages": [{"role": "user", "content": "warm exact hit prompt"}],
    }
    stream_chat(url, prompt)  # prime (cold, discarded)
    ttfts, totals, ntoks = [], [], []
    for _ in range(args.requests):
        ttft, total, ntok = stream_chat(url, prompt)
        ttfts.append(ttft)
        totals.append(total)
        ntoks.append(ntok)
    rows.append(report("warm_exact_hit", ttfts, totals, ntoks))

    # 3. Growing conversation: each round appends a new " | "-separated
    #    segment; the synthetic tokenizer hashes each segment independently,
    #    so every request after the first is a partial whose prefill covers
    #    only the delta (the same prefix-growth property real tokenizers
    #    have on appended text).
    content = "growing conversation seed"
    ttfts, totals, ntoks = [], [], []
    for r in range(args.grow_rounds):
        content += f" | round {r}"
        ttft, total, ntok = stream_chat(
            url,
            {"model": "bench-model", "messages": [{"role": "user", "content": content}]},
        )
        ttfts.append(ttft)
        totals.append(total)
        ntoks.append(ntok)
    rows.append(report("growing_conversation", ttfts, totals, ntoks))

    # Guard the workload's own premise: rounds 1+ must classify as partials
    # (round 0 is the cold miss that seeds the chain). If a tokenizer change
    # ever breaks the prefix property, this bench must fail loudly instead of
    # quietly benchmarking full prefills.
    with urllib.request.urlopen(f"{url}/stats", timeout=5) as r:
        grow_stats = json.loads(r.read())
    expected_partials = max(args.grow_rounds - 1, 0)
    assert grow_stats["partials"] >= expected_partials, (
        f"growing_conversation produced {grow_stats['partials']} partials, "
        f"expected >= {expected_partials} — the token prefix chain is broken"
    )

    # 4. Concurrent identical load: C workers fire the SAME uncached prompt at
    #    once; single-flight must coalesce them into one prefill.
    def worker(shared: dict, idx: int) -> None:
        try:
            ttft, total, ntok = stream_chat(
                url,
                {
                    "model": "bench-model",
                    "messages": [
                        {"role": "user", "content": f"concurrent burst {shared['burst']}"}
                    ],
                },
            )
        except Exception as exc:  # noqa: BLE001 — collected, re-raised after join
            shared["errors"].append(f"worker {idx}: {exc!r}")
            return
        shared["ttfts"].append(ttft)
        shared["totals"].append(total)
        shared["ntoks"].append(ntok)

    ttfts, totals, ntoks = [], [], []
    # Ceiling division: floor() would quantize the sample count DOWN, and at
    # requests=100/concurrency=8 the tail gate would go silent on exactly the
    # contention workloads it exists for.
    bursts = -(-args.requests // args.concurrency)
    for burst in range(bursts):
        shared = {"burst": burst, "ttfts": [], "totals": [], "ntoks": [], "errors": []}

        threads = [
            threading.Thread(target=worker, args=(shared, i)) for i in range(args.concurrency)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # A worker that dies in a thread would otherwise just shorten the
        # aggregate — a server 500ing under contention would read as a
        # slightly smaller n and a green run.
        if shared["errors"]:
            raise RuntimeError(
                f"concurrent_identical lost {len(shared['errors'])} worker(s): "
                f"{shared['errors'][0]}"
            )
        ttfts += shared["ttfts"]
        totals += shared["totals"]
        ntoks += shared["ntoks"]
    rows.append(report(f"concurrent_identical_c{args.concurrency}", ttfts, totals, ntoks))

    # 5. Concurrent distinct load: C workers on DIFFERENT prompts (no
    #    coalescing possible; pure parallelism of the hot path).
    ttfts, totals, ntoks = [], [], []
    for burst in range(bursts):
        shared = {"ttfts": [], "totals": [], "ntoks": [], "errors": []}

        def worker_distinct(i: int, burst: int = burst, shared: dict = shared) -> None:
            try:
                ttft, total, ntok = stream_chat(
                    url,
                    {
                        "model": "bench-model",
                        "messages": [{"role": "user", "content": f"distinct burst {burst}-{i}"}],
                    },
                )
            except Exception as exc:  # noqa: BLE001 — collected, re-raised after join
                shared["errors"].append(f"worker {i}: {exc!r}")
                return
            shared["ttfts"].append(ttft)
            shared["totals"].append(total)
            shared["ntoks"].append(ntok)

        threads = [
            threading.Thread(target=worker_distinct, args=(i,)) for i in range(args.concurrency)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if shared["errors"]:
            raise RuntimeError(
                f"concurrent_distinct lost {len(shared['errors'])} worker(s): {shared['errors'][0]}"
            )
        ttfts += shared["ttfts"]
        totals += shared["totals"]
        ntoks += shared["ntoks"]
    rows.append(report(f"concurrent_distinct_c{args.concurrency}", ttfts, totals, ntoks))

    with urllib.request.urlopen(f"{url}/stats", timeout=5) as r:
        stats = json.loads(r.read())
    print(
        f"\n  server stats: requests={stats['requests']} hits={stats['hits']} "
        f"partials={stats['partials']} misses={stats['misses']} "
        f"published={stats['checkpoints_published']}"
    )
    print(f"  hit_rate={(stats['hit_rate'] * 100):.1f}% (tokens_cached/tokens_total)")
    return rows


if __name__ == "__main__":
    sys.exit(main())
