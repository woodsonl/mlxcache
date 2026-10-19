#!/usr/bin/env python3
"""Prefill and decode cost as context grows, on the real NVFP4 checkpoint.

The README quotes a decode figure at 8K that is 7.8x the 2K figure while the
8K prefill is only 1.07x the 2K prefill. A single decode step cannot cost 23%
of the whole 8K prefill, so that number was a machine-state artifact and was
labelled as such. This probe exists to replace it with a measured point.

What it measures, per context length:

  1. prefill  — build the cache for `size` tokens through the adapter's own
     `_prefill_cache`, so the number includes the path the daemon uses.
  2. ttft     — prefill plus the step that consumes the last prompt token.
  3. decode   — `--decode` greedy steps, reported as a median per token so one
     stalled step cannot move the number.
  4. blob     — serialized cache size, which is the bytes/token growth claim.

The self-check is the argument the README already makes, made mechanical: a
decode step's attention work is bounded by the prefill's attention work over the
same positions, so the decode ratio between the largest and smallest context
cannot exceed the prefill ratio over the same pair. The contaminated 8K run
fails this immediately (7.9x decode against 1.07x prefill) while a real
attention cost would raise both. A failure refuses to report any timing.

Swap is sampled at the start and end of the run. Any swap in use means the
decode timings are not measuring the model, because the weights are streamed
from the checkpoint and a paging filesystem turns that into disk thrash.

Run: uv run --extra mlx python sidecar/probes/qwen4_context_sweep.py \
        --sizes 256,1024,2048,8192 --out sidecar/probes/context-sweep.json
"""

from __future__ import annotations

import argparse
import gc
import io
import itertools
import json
import os
import statistics
import subprocess
import time

CHECKPOINT = os.environ.get(
    "MLXCACHE_Q4_CHECKPOINT", "/Users/lance/models/Qwen3.8-Flash-Next-NVFP4-mlxstream"
)

SEED = (
    "The daemon keeps a prompt cache on disk so that a restart can resume a long "
    "conversation without paying the prefill again. A cache is only worth keeping "
    "if reloading it is cheaper than recomputing it, and only trustworthy if the "
    "reloaded state continues the same stream the live state would have produced. "
)


def machine_state() -> dict[str, object]:
    """Swap and load, the two signals that make a streaming decode untrustworthy."""
    state: dict[str, object] = {}
    try:
        out = subprocess.run(
            ["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, check=True
        ).stdout
        # "total = 0.00M  used = 0.00M  free = 0.00M  (encrypted)"
        used = out.split("used =")[1].split("M")[0].strip()
        state["swap_used_mb"] = float(used)
    except (subprocess.CalledProcessError, IndexError, ValueError):
        state["swap_used_mb"] = None
    try:
        load = os.getloadavg()[0]
        state["load_1m"] = round(load, 2)
    except OSError:
        state["load_1m"] = None
    return state


def build_tokens(tokenizer, size: int) -> list[int]:
    """Exactly `size` tokens of varied prose, so no position is a degenerate repeat."""
    text = SEED
    while len(tokenizer.encode(text)) < size:
        text += SEED
    return tokenizer.encode(text)[:size]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=CHECKPOINT)
    ap.add_argument("--sizes", default="256,1024,2048,8192")
    ap.add_argument("--decode", type=int, default=4, help="greedy steps per size")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from probe_lock import checkpoint_lock

    # The lock spans load through report: taking it around only the loop would
    # let a second probe start loading while this one is still measuring.
    with checkpoint_lock("qwen4_context_sweep"):
        return _sweep(args)


def _sweep(args: argparse.Namespace) -> int:
    import mlx.core as mx
    from mlx_lm.models.cache import save_prompt_cache
    from mlxcache_sidecar.engine_qwen4 import Qwen4ExpEngine

    sizes = [int(s) for s in args.sizes.split(",")]
    before = machine_state()
    print(f"machine before  {before}", flush=True)

    engine = Qwen4ExpEngine(args.checkpoint)
    tok = engine.tokenizer

    rows: list[dict[str, float]] = []
    print(f"\n{'tokens':>7} {'prefill_s':>10} {'ttft_s':>8} {'decode_s/tok':>13} {'blob_MB':>8}")
    for size in sizes:
        ids = build_tokens(tok, size)

        t0 = time.perf_counter()
        cache, _logits = engine._prefill_cache(ids[:-1])
        prefill = time.perf_counter() - t0
        # TTFT adds the one step that consumes the final prompt token, kept
        # separate so `prefill_s` stays comparable to the README's figure.
        t0 = time.perf_counter()
        prev = int(mx.argmax(engine.model(mx.array([[ids[-1]]]), cache=cache)[0, -1]).item())
        ttft = prefill + (time.perf_counter() - t0)

        step_times: list[float] = []
        for _ in range(args.decode):
            t0 = time.perf_counter()
            logits = engine.model(mx.array([[prev]]), cache=cache)
            prev = int(mx.argmax(logits[0, -1]).item())
            step_times.append(time.perf_counter() - t0)
        decode = statistics.median(step_times)

        blob = io.BytesIO()
        # `load_prompt_cache` dispatches on the name, so keep the extension the
        # loader understands even when the payload only needs its length.
        blob.name = "sweep.safetensors"
        save_prompt_cache(blob, cache)
        payload = blob.getvalue()

        rows.append(
            {
                "tokens": size,
                "prefill_s": round(prefill, 2),
                "ttft_s": round(ttft, 2),
                "decode_s": round(decode, 3),
                "blob_bytes": len(payload),
            }
        )
        print(
            f"{size:>7} {prefill:>10.2f} {ttft:>8.2f} {decode:>13.2f} {len(payload) / 1e6:>8.1f}",
            flush=True,
        )

        del cache, logits, blob, payload
        gc.collect()

    after = machine_state()
    print(f"\nmachine after   {after}", flush=True)

    # --- self-check ------------------------------------------------------
    small, large = rows[0], rows[-1]
    prefill_ratio = large["prefill_s"] / small["prefill_s"]
    decode_ratio = large["decode_s"] / small["decode_s"]
    print(
        f"\n{large['tokens']} vs {small['tokens']}: prefill {prefill_ratio:.2f}x, "
        f"decode {decode_ratio:.2f}x"
    )
    failures = []
    if decode_ratio > prefill_ratio:
        failures.append(
            f"decode grew {decode_ratio:.2f}x but prefill only {prefill_ratio:.2f}x. "
            "Attention over the same positions cannot be the cause; the decode "
            "numbers were taken under contention."
        )
    for a, b in itertools.pairwise(rows):
        if b["prefill_s"] < a["prefill_s"]:
            failures.append(
                f"prefill at {b['tokens']} ({b['prefill_s']:.2f}s) is cheaper than at "
                f"{a['tokens']} ({a['prefill_s']:.2f}s); more tokens is more work."
            )
    swap_after = after["swap_used_mb"]
    swap_before = before["swap_used_mb"]
    # Growth is the causal signal. Stale swap that nothing is paging through
    # does not slow a decode; swap that grows DURING the run means the weights
    # were evicted while the clock was running. The collided run that motivated
    # this check went 0 -> 7 GB.
    if (
        isinstance(swap_after, float)
        and isinstance(swap_before, float)
        and swap_after > swap_before + 256
    ):
        failures.append(
            f"swap grew {swap_after - swap_before:.0f} MB during the run "
            f"({swap_before:.0f} -> {swap_after:.0f} MB): the weights were "
            "being paged while the clocks ran."
        )

    if failures:
        for f in failures:
            print(f"SELF-CHECK FAILED: {f}", flush=True)
        print("No timing from this run is usable.", flush=True)
        return 2
    print("self-check -> PASS", flush=True)

    report = {"machine_before": before, "machine_after": after, "rows": rows}
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"json -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
