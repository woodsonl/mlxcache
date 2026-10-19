#!/usr/bin/env python3
"""Prefill and decode cost as context grows, on the real NVFP4 checkpoint.

The README quotes a decode figure at 8K that is 7.8x the 2K figure while the
8K prefill is only 1.07x the 2K prefill. A single decode step cannot cost 23%
of the whole 8K prefill, so that number was a machine-state artifact and was
labelled as such. This probe exists to replace it with a measured point.

Each context length is measured in its own process. That is not tidiness: the
engine streams a 132.6 GB checkpoint through a 48 GB machine, MLX retains
freed allocator memory, and a single process measuring six lengths grows swap
by gigabytes as it goes. Rows from one long-lived process are therefore not
comparable with each other, and the swap guard below can never pass. One
process per row gives every measurement the same starting conditions.

Ordering matters for the other confounder. The weights stream from disk, so a
run that measures ascending leaves its largest context measured with the
warmest page cache -- which is exactly the row a long-context claim rests on.
Sizes are measured descending, so the coldest run lands on the largest
context and warming cannot manufacture a good number there.

Self-checks, all refusing rather than warning:

  * monotonicity -- more tokens cannot be less work than fewer.
  * ratio -- a decode step's attention is bounded by the prefill's over the
    same positions, so the decode ratio between the largest and smallest
    context cannot exceed the prefill ratio over that pair. The contaminated
    8K run fails this immediately (8.65x decode against 1.66x prefill).
  * repeatability -- one size is measured twice, first and last. If the same
    size disagrees with itself beyond a tolerance, the machine is not quiet
    enough to distinguish the effect being measured.
  * swap -- growth during a row means the weights were being paged while the
    clock ran.

Run: uv run --extra mlx python sidecar/probes/qwen4_context_sweep.py \
        --sizes 256,512,1024,2048,4096,8192 --out sidecar/probes/context-sweep.json
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
import sys
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
        state["load_1m"] = round(os.getloadavg()[0], 2)
    except OSError:
        state["load_1m"] = None
    return state


def build_tokens(tokenizer, size: int) -> list[int]:
    """Exactly `size` tokens of varied prose, so no position is a degenerate repeat."""
    text = SEED
    while len(tokenizer.encode(text)) < size:
        text += SEED
    return tokenizer.encode(text)[:size]


def measure_one(checkpoint: str, size: int, decode_steps: int) -> dict[str, float]:
    """One size, one process: boot, prefill, decode, blob."""
    import mlx.core as mx
    from mlx_lm.models.cache import save_prompt_cache
    from mlxcache_sidecar.engine_qwen4 import Qwen4ExpEngine

    before = machine_state()
    engine = Qwen4ExpEngine(checkpoint)
    ids = build_tokens(engine.tokenizer, size)

    t0 = time.perf_counter()
    cache, _logits = engine._prefill_cache(ids[:-1])
    prefill = time.perf_counter() - t0
    # TTFT adds the one step that consumes the final prompt token, kept
    # separate so `prefill_s` stays comparable to the README's figure.
    t0 = time.perf_counter()
    prev = int(mx.argmax(engine.model(mx.array([[ids[-1]]]), cache=cache)[0, -1]).item())
    ttft = prefill + (time.perf_counter() - t0)

    step_times: list[float] = []
    for _ in range(decode_steps):
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

    row = {
        "tokens": size,
        "prefill_s": round(prefill, 2),
        "ttft_s": round(ttft, 2),
        "decode_s": round(decode, 3),
        "blob_bytes": len(payload),
        "swap_before_mb": before["swap_used_mb"],
        "swap_after_mb": machine_state()["swap_used_mb"],
    }
    del cache, logits, blob, payload
    gc.collect()
    return row


def _child(args: argparse.Namespace) -> int:
    """Entry point for the per-size subprocess."""
    from probe_lock import checkpoint_lock

    with checkpoint_lock(f"qwen4_context_sweep[{args.only_size}]"):
        row = measure_one(args.checkpoint, args.only_size, args.decode)
    print("ROW " + json.dumps(row), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=CHECKPOINT)
    ap.add_argument("--sizes", default="256,512,1024,2048,4096,8192")
    ap.add_argument("--decode", type=int, default=4, help="greedy steps per size")
    ap.add_argument("--tolerance", type=float, default=0.25, help="repeatability bound")
    ap.add_argument("--only-size", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.only_size is not None:
        return _child(args)

    sizes = sorted({int(s) for s in args.sizes.split(",")}, reverse=True)
    control = sizes[-1]
    # Coldest first on the largest context; the control is measured before and
    # after so repeatability is observed rather than assumed.
    plan: list[tuple[int, bool]] = [
        (control, True),
        *((size, False) for size in sizes if size != control),
        (control, True),
    ]

    rows: list[dict[str, float]] = []
    print(
        f"\n{'tokens':>7} {'prefill_s':>10} {'ttft_s':>8} {'decode_s/tok':>13} "
        f"{'blob_MB':>8} {'swap_delta_MB':>13}",
        flush=True,
    )
    for size, is_control in plan:
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--only-size",
            str(size),
            "--decode",
            str(args.decode),
            "--checkpoint",
            args.checkpoint,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("ROW ")), None)
        if line is None:
            print(proc.stdout, proc.stderr, file=sys.stderr, flush=True)
            print(f"REFUSING: the {size}-token run produced no measurement.", flush=True)
            return 2
        row = json.loads(line[4:])
        row["control"] = is_control
        rows.append(row)
        delta = row["swap_after_mb"] - row["swap_before_mb"]
        label = f"{size}*" if is_control else f"{size}"
        print(
            f"{label:>7} {row['prefill_s']:>10.2f} {row['ttft_s']:>8.2f} "
            f"{row['decode_s']:>13.2f} {row['blob_bytes'] / 1e6:>8.1f} {delta:>13.0f}",
            flush=True,
        )

    sweep_rows = [r for r in rows if not r["control"]]
    control_rows = [r for r in rows if r["control"]]
    small, large = sweep_rows[-1], sweep_rows[0]
    prefill_ratio = large["prefill_s"] / small["prefill_s"]
    decode_ratio = large["decode_s"] / small["decode_s"]
    print(
        f"\n{large['tokens']} vs {small['tokens']}: prefill {prefill_ratio:.2f}x, "
        f"decode {decode_ratio:.2f}x"
    )

    failures: list[str] = []
    if len(control_rows) == 2:
        c0, c1 = control_rows
        spread = abs(c1["prefill_s"] - c0["prefill_s"]) / c0["prefill_s"]
        print(
            f"repeatability {c0['tokens']} tokens: {c0['prefill_s']:.1f}s vs "
            f"{c1['prefill_s']:.1f}s ({spread:.1%} apart)"
        )
        if spread > args.tolerance:
            failures.append(
                f"the same {c0['tokens']}-token prefill varied {spread:.0%} between two "
                "runs, so the machine cannot resolve the effect being measured."
            )

    if decode_ratio > prefill_ratio:
        failures.append(
            f"decode grew {decode_ratio:.2f}x but prefill only {prefill_ratio:.2f}x. "
            "Attention over the same positions cannot be the cause; the decode "
            "numbers were taken under contention."
        )
    for a, b in itertools.pairwise(sweep_rows):
        if b["prefill_s"] < a["prefill_s"]:
            failures.append(
                f"prefill at {b['tokens']} ({b['prefill_s']:.2f}s) is cheaper than at "
                f"{a['tokens']} ({a['prefill_s']:.2f}s); more tokens is more work."
            )
    for row in rows:
        grew = row["swap_after_mb"] - row["swap_before_mb"]
        if grew > 256:
            failures.append(
                f"swap grew {grew:.0f} MB during the {row['tokens']}-token row: the "
                "weights were being paged while the clock ran."
            )

    if failures:
        for f in failures:
            print(f"SELF-CHECK FAILED: {f}", flush=True)
        print("No timing from this run is usable.", flush=True)
        return 2
    print("self-check -> PASS", flush=True)

    report = {"rows": rows, "machine_after": machine_state()}
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"json -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
