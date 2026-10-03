#!/usr/bin/env python3
"""B1 workload driver: end-to-end daemon+sidecar benchmark over real HTTP.

Complements the criterion micro-benches (which pin the daemon's in-process
costs) with the numbers that actually matter to a client:

- cold prefill (first request for a prompt)
- warm exact hit (second request for the same prompt)
- growing conversation (delta prefill — the OV3 lever)
- concurrent identical load (single-flight coalescing under contention)

Every workload reports p50/p99 for TTFT (first byte of the assistant reply)
and TPOT (per decoded token), so a regression in ANY layer — tokenize, route,
prefill, publish, stream — shows up where the client feels it.

Usage:
    uv run python scripts/bench_workloads.py [--daemon URL] [--requests N]
                                             [--concurrency C] [--json OUT]

The daemon and sidecar are started automatically if --daemon is not given
(uv run cargo run --release -- serve ...), exercised, and torn down.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

BLOB_DIR_DEFAULT = "/tmp/mlxcache-bench-blobs"


def http_json(url: str, payload: dict, timeout: float = 60.0) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def stream_chat(url: str, body: dict, timeout: float = 120.0) -> tuple[float, float, int]:
    """POST a streaming chat completion; return (ttft_ms, total_ms, tokens).

    TTFT = request sent → first `data:` frame received. The per-frame loop
    reads the raw SSE byte stream like a real client.
    """
    body = dict(body, stream=True)
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    ttft: float | None = None
    tokens = 0
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data: "):
                continue
            if ttft is None:
                ttft = (time.perf_counter() - t0) * 1000
            payload = line[len("data: ") :]
            if payload == "[DONE]":
                break
            try:
                frame = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if "mlxcache" in frame or "error" in frame:
                continue  # stats/error frame, not a token
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


def report(name: str, ttfts: list[float], totals: list[float], tokens_per_req: int) -> dict:
    tpots = [
        (t - ttft) / max(tokens_per_req - 1, 1) for ttft, t in zip(ttfts, totals)
    ]
    row = {
        "workload": name,
        "n": len(ttfts),
        "ttft_p50_ms": round(pct(ttfts, 50), 2),
        "ttft_p99_ms": round(pct(ttfts, 99), 2),
        "ttft_mean_ms": round(statistics.fmean(ttfts), 2),
        "tpot_p50_ms": round(pct(tpots, 50), 2),
        "tpot_p99_ms": round(pct(tpots, 99), 2),
    }
    print(
        f"  {name:<34} TTFT p50 {row['ttft_p50_ms']:>8.2f}ms  p99 {row['ttft_p99_ms']:>8.2f}ms"
        f"   TPOT p50 {row['tpot_p50_ms']:>6.2f}ms  p99 {row['tpot_p99_ms']:>6.2f}ms  (n={len(ttfts)})"
    )
    return row


def wait_healthy(url: str, deadline_s: float = 30.0) -> None:
    # The daemon exposes no /health; a 404 from /stats proves the listener is
    # accepting and routing (any response but connection-refused works).
    deadline = time.time() + deadline_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/stats", timeout=1) as r:
                if r.status == 200:
                    return
        except urllib.error.HTTPError:
            return  # routed response: daemon is up
        except Exception:  # noqa: BLE001 — not up yet
            time.sleep(0.2)
    raise RuntimeError(f"daemon did not become healthy at {url}")


def start_stack(blob_dir: str) -> tuple[subprocess.Popen, subprocess.Popen, int]:
    """Start sidecar (uv, synthetic engine) + daemon (release build)."""
    sidecar_port = 8421
    env = dict(os.environ, MLXCACHE_TOKENIZE_GROW="growing conversation seed")
    sidecar = subprocess.Popen(
        [
            "uv", "run", "python", "-c",
            "import sys; sys.path.insert(0, 'sidecar'); "
            "from mlxcache_sidecar import server; "
            "server.Handler.engine = server.make_engine('bench-model'); "
            f"server.ThreadingHTTPServer(('127.0.0.1', {sidecar_port}), "
            "server.Handler).serve_forever()",
        ],
        env=env,
    )
    daemon_port = 8420
    daemon = subprocess.Popen(
        [
            "uv", "run", "cargo", "run", "--release", "-p", "mlxcache-daemon", "--",
        ],
        env=dict(
            env,
            MLXCACHE_MODELS="bench-model",
            MLXCACHE_SIDECAR_URL=f"http://127.0.0.1:{sidecar_port}",
            MLXCACHE_BLOBS=blob_dir,
            MLXCACHE_ADDR=f"127.0.0.1:{daemon_port}",
        ),
    )
    return sidecar, daemon, daemon_port


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--daemon", help="existing daemon base URL (skip auto-start)")
    ap.add_argument("--requests", type=int, default=30, help="per-workload sample count")
    ap.add_argument("--concurrency", type=int, default=8, help="concurrent workers")
    ap.add_argument("--grow-rounds", type=int, default=8, help="conversation growth rounds")
    ap.add_argument("--json", help="write results to this file")
    args = ap.parse_args()

    procs: list[subprocess.Popen] = []
    url = args.daemon
    blob_dir = BLOB_DIR_DEFAULT
    if url is None:
        os.makedirs(blob_dir, exist_ok=True)
        sidecar, daemon, port = start_stack(blob_dir)
        procs += [sidecar, daemon]
        url = f"http://127.0.0.1:{port}"
    try:
        wait_healthy(url)
        results = run_workloads(url, args)
    finally:
        for p in procs:
            p.send_signal(signal.SIGTERM)
        for p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nwrote {args.json}")
    return 0


def run_workloads(url: str, args) -> list[dict]:
    rows: list[dict] = []
    print(f"daemon {url}  requests={args.requests} concurrency={args.concurrency}\n")

    # 1. Cold prefill: unique prompts, each a first request.
    ttfts, totals = [], []
    for i in range(args.requests):
        ttft, total, ntok = stream_chat(
            url,
            {"model": "bench-model", "messages": [
                {"role": "user", "content": f"cold prompt number {i}"}]},
        )
        ttfts.append(ttft)
        totals.append(total)
        assert ntok > 0
    rows.append(report("cold_prefill", ttfts, totals, 64))

    # 2. Warm exact hit: the same prompt twice; the 2nd+ go through the index.
    prompt = {"model": "bench-model", "messages": [
        {"role": "user", "content": "warm exact hit prompt"}]}
    stream_chat(url, prompt)  # prime (cold, discarded)
    ttfts, totals = [], []
    for _ in range(args.requests):
        ttft, total, ntok = stream_chat(url, prompt)
        ttfts.append(ttft)
        totals.append(total)
    rows.append(report("warm_exact_hit", ttfts, totals, 64))

    # 3. Growing conversation: each round appends a new " | "-separated
    #    segment; the synthetic tokenizer hashes each segment independently,
    #    so every request after the first is a partial whose prefill covers
    #    only the delta (the same prefix-growth property real tokenizers
    #    have on appended text).
    content = "growing conversation seed"
    ttfts, totals = [], []
    for r in range(args.grow_rounds):
        content += f" | round {r}"
        ttft, total, ntok = stream_chat(
            url,
            {"model": "bench-model", "messages": [{"role": "user", "content": content}]},
        )
        ttfts.append(ttft)
        totals.append(total)
    rows.append(report("growing_conversation", ttfts, totals, 64))

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
        ttft, total, ntok = stream_chat(
            url,
            {"model": "bench-model", "messages": [
                {"role": "user", "content": f"concurrent burst {shared['burst']}"}]},
        )
        shared["ttfts"].append(ttft)
        shared["totals"].append(total)

    ttfts, totals = [], []
    for burst in range(max(1, args.requests // args.concurrency)):
        shared = {"burst": burst, "ttfts": [], "totals": []}
        import threading

        threads = [
            threading.Thread(target=worker, args=(shared, i))
            for i in range(args.concurrency)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        ttfts += shared["ttfts"]
        totals += shared["totals"]
    rows.append(report(f"concurrent_identical_c{args.concurrency}", ttfts, totals, 64))

    # 5. Concurrent distinct load: C workers on DIFFERENT prompts (no
    #    coalescing possible; pure parallelism of the hot path).
    ttfts, totals = [], []
    for burst in range(max(1, args.requests // args.concurrency)):
        shared = {"ttfts": [], "totals": []}
        import threading

        def worker_distinct(i: int, shared=shared) -> None:
            ttft, total, ntok = stream_chat(
                url,
                {"model": "bench-model", "messages": [
                    {"role": "user", "content": f"distinct burst {burst}-{i}"}]},
            )
            shared["ttfts"].append(ttft)
            shared["totals"].append(total)

        threads = [threading.Thread(target=worker_distinct, args=(i,)) for i in range(args.concurrency)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        ttfts += shared["ttfts"]
        totals += shared["totals"]
    rows.append(report(f"concurrent_distinct_c{args.concurrency}", ttfts, totals, 64))

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
