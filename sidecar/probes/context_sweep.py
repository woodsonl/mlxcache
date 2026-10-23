#!/usr/bin/env python3
"""Prefill, decode, and checkpoint cost as context grows, on a real mlx-lm model.

The README's long-context claims need numbers measured on the supported path
(the mlx-lm adapter's own calls: make_prompt_cache, model(inp, cache=...),
save_prompt_cache). This sweep produces them and refuses to produce anything
else when the machine is not quiet enough to trust.

Method, each rule earned by a contaminated number that came before it:

  * One process per context length. MLX retains freed allocator memory, so
    rows from one long-lived process measure the allocator's history, not the
    model. A fresh process gives every row the same starting conditions.
  * Warmup first: a resident model's FIRST load streams its weights through
    a cold page cache (measured 45% slower on the 7B), and that one-time
    cost must not land on a measured row. The warmup leg absorbs it and is
    excluded from every check.
  * Control first and last: the smallest size is measured twice, bracketing
    the sweep, so repeatability is observed rather than assumed.
  * Every size descends after the leading control, so the coldest SWEEP run
    lands on the largest context. The control's weight load warms the page
    cache for the first sweep row, biasing the largest context's prefill
    down (in its favor); the before/after control pair bounds that bias.
  * Timings refuse rather than warn: a contaminated number can be explained
    away afterward, and that is how a bad figure ends up in a README.

Self-checks, all refusing (exit 2, no JSON):

  * monotonicity — fewer tokens cannot cost more prefill than more tokens.
  * ratio — a decode step's attention is bounded by the prefill's over the
    same positions, so the decode ratio between the largest and smallest
    context cannot exceed the prefill ratio over that pair (the check that
    caught an 8K decode figure 7.8x its 2K value while prefill grew 1.07x).
  * repeatability — the control pair must agree within a tolerance, else the
    machine cannot resolve the effect being measured.
  * swap — growth during a row means weights were paged while the clock ran.
  * row sanity — a missing or unreadable row (child died mid-write, swap
    read failed) is a refusal, never a skipped row.

One sweep at a time: the PARENT process holds an flock on
~/.mlxcache/probe-sweep.lock for the entire plan (override with
MLXCACHE_PROBE_LOCK_PATH, bypass with MLXCACHE_PROBE_LOCK=off — bypassing
admits contaminated timings on purpose). flock is released by the OS if the
probe is killed; the in-file holder text is display-only.

Run: uv run --extra mlx python sidecar/probes/context_sweep.py \
        --model mlx-community/Qwen2.5-7B-Instruct-4bit \
        --sizes 256,512,1024,2048,4096,8192,16384,32768 \
        --out sidecar/probes/context-sweep.json
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import gc
import io
import itertools
import json
import os
import resource
import statistics
import subprocess
import sys
import tempfile
import time

DEFAULT_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"

SEED = (
    "The daemon keeps a prompt cache on disk so that a restart can resume a long "
    "conversation without paying the prefill again. A cache is only worth keeping "
    "if reloading it is cheaper than recomputing it, and only trustworthy if the "
    "reloaded state continues the same stream the live state would have produced. "
)


def machine_state() -> dict[str, object]:
    """Swap is the guarded signal (paging live weights out is the proven
    contamination mode). Load is recorded for context only — a load check
    would flag the measured child's own CPU."""
    state: dict[str, object] = {}
    try:
        out = subprocess.run(
            ["sysctl", "-n", "vm.swapusage"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout
        used = out.split("used =")[1].split("M")[0].strip()
        state["swap_used_mb"] = float(used)
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        IndexError,
        ValueError,
        OSError,
    ):
        # A missing or wedged sysctl must degrade to the None that
        # validate_row turns into a clean refusal, never a traceback out
        # of a child that holds no context for the error.
        state["swap_used_mb"] = None
    try:
        state["load_1m"] = round(os.getloadavg()[0], 2)
    except OSError:
        state["load_1m"] = None
    return state


def build_tokens(tokenizer, size: int) -> list[int]:
    """Exactly `size` tokens of varied prose, so no position is a degenerate
    repeat. The corpus doubles per iteration (one append-one-SEED per loop
    re-encodes the whole text quadratically and dominates startup at 32K),
    then one final encode truncates to size."""
    text = SEED
    while len(tokenizer.encode(text)) < size:
        text += text
    return tokenizer.encode(text)[:size]


def measure_one(model_id: str, size: int, decode_steps: int) -> dict[str, float]:
    """One size, one process: load, prefill, decode, checkpoint bytes.

    Uses the same calls the sidecar's prefill path makes, so the numbers are
    the supported path's numbers, not a lookalike's."""
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache, save_prompt_cache

    before = machine_state()
    model, tokenizer = load(model_id)
    ids = build_tokens(tokenizer, size)

    # Prefill: the sidecar's own sequence — fresh cache, forward pass over
    # tokens[:-1], eval so the KV is resident, not lazily pending.
    t0 = time.perf_counter()
    cache = make_prompt_cache(model)
    logits = model(mx.array(ids[:-1])[None], cache=cache)
    mx.eval([c.state for c in cache], logits)
    prefill = time.perf_counter() - t0

    # The checkpoint is captured HERE, exactly as the adapter's publish
    # path captures it: the prompt-only cache, before any generated token
    # mutates it. Saving after decode would fold the generated positions
    # into blob_bytes and overstate the checkpoint size. The serialized
    # bytes are released immediately: holding ~1 GB of dead payload through
    # the decode measurement would perturb the timing and the swap guard
    # with the probe's own retention.
    blob = io.BytesIO()
    # load_prompt_cache dispatches on the name; keep the extension it reads.
    blob.name = "sweep.safetensors"
    save_prompt_cache(blob, cache)
    checkpoint_bytes = len(blob.getvalue())
    del blob

    # TTFT adds the one step that consumes the final prompt token, kept
    # separate so `prefill_s` stays the comparable figure.
    t0 = time.perf_counter()
    prev = int(mx.argmax(model(mx.array([[ids[-1]]]), cache=cache)[0, -1]).item())
    ttft = prefill + (time.perf_counter() - t0)

    step_times: list[float] = []
    logits = None  # bound here: --decode 0 skips the loop, and the del below
    # must not NameError away a completed measurement.
    for _ in range(decode_steps):
        t0 = time.perf_counter()
        logits = model(mx.array([[prev]]), cache=cache)
        prev = int(mx.argmax(logits[0, -1]).item())
        step_times.append(time.perf_counter() - t0)
    decode = statistics.median(step_times) if step_times else 0.0

    row = {
        "tokens": size,
        "prefill_s": round(prefill, 2),
        "ttft_s": round(ttft, 2),
        "decode_s": round(decode, 3),
        "blob_bytes": checkpoint_bytes,
        "swap_before_mb": before["swap_used_mb"],
        "swap_after_mb": machine_state()["swap_used_mb"],
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576, 1),
    }
    del cache, logits
    gc.collect()
    return row


@contextlib.contextmanager
def sweep_lock(what: str):
    """One sweep at a time, or explain why this one must not run."""
    if os.environ.get("MLXCACHE_PROBE_LOCK", "").lower() in {"off", "0", "false", "no"}:
        print("lock bypassed (MLXCACHE_PROBE_LOCK=off); timings may be contaminated")
        yield
        return
    lock_path = os.environ.get(
        "MLXCACHE_PROBE_LOCK_PATH",
        os.path.join(os.path.expanduser("~"), ".mlxcache", "probe-sweep.lock"),
    )
    lock_dir = os.path.dirname(lock_path)
    if lock_dir:  # dirname('') is '' for a bare filename; makedirs('') raises
        os.makedirs(lock_dir, exist_ok=True)
    with open(lock_path, "a+") as fh:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fh.seek(0)
            holder = fh.read().strip() or "another sweep"
            print(
                f"REFUSING to run {what}: the machine is held by {holder}.\n"
                "Two timing sweeps on one machine contaminate each other's rows.\n"
                "Wait for it to finish, or set MLXCACHE_PROBE_LOCK=off to overlap "
                "on purpose.",
                file=sys.stderr,
                flush=True,
            )
            raise SystemExit(4) from None
        fh.seek(0)
        fh.truncate()
        fh.write(f"{what} (pid {os.getpid()})")
        fh.flush()
        yield


def build_plan(sizes: list[int]) -> list[tuple[int, str]]:
    """Warmup, then control (smallest) first and last, every size descending
    between.

    The warmup leg absorbs the one-time cost a resident model pays on its
    FIRST load — streaming the weights through a cold page cache — which
    otherwise lands on the leading control and reads as a 45% repeatability
    failure (measured on Qwen2.5-7B-Instruct-4bit: cold first row 0.69 s vs
    0.38 s for every later load of the same size). It is excluded from every
    check and from the sweep rows; the report keeps it as the cold-load
    figure.

    Every size is a measured sweep row — the smallest included — so the
    ratio's small end is a requested size, never the second-smallest."""
    ordered = sorted(set(sizes), reverse=True)
    if len(ordered) < 2:
        raise ValueError("the sweep needs at least two distinct sizes")
    control = ordered[-1]
    return [
        (control, "warmup"),
        (control, "control"),
        *((size, "sweep") for size in ordered),
        (control, "control"),
    ]


def validate_row(row: dict, size: int) -> str | None:
    """A row the sweep cannot vouch for, as a refusal reason."""
    if row.get("tokens") != size:
        return f"the {size}-token run reported {row.get('tokens')} tokens"
    if row.get("swap_before_mb") is None or row.get("swap_after_mb") is None:
        return f"the {size}-token run could not read swap state; the swap guard cannot vouch for it"
    prefill = row.get("prefill_s")
    if not isinstance(prefill, (int, float)) or prefill <= 0:
        # Below the clock's resolution (a sub-5 ms prefill rounds to 0.0);
        # the ratios would divide by zero downstream.
        return (
            f"the {size}-token run's prefill is unmeasurable ({prefill!r}); "
            "sizes this small are below the clock"
        )
    return None


def monotonicity_failures(sweep_rows: list[dict]) -> list[str]:
    """Rows run largest to smallest, so `b` is the SMALLER context: fewer
    tokens must be cheaper. A smaller context measuring slower is the
    machine lying."""
    failures = []
    for a, b in itertools.pairwise(sweep_rows):
        if b["prefill_s"] > a["prefill_s"]:
            failures.append(
                f"prefill at {b['tokens']} ({b['prefill_s']:.2f}s) is slower than at "
                f"{a['tokens']} ({a['prefill_s']:.2f}s); fewer tokens should cost less."
            )
    return failures


def ratio_summary(sweep_rows: list[dict]) -> tuple[float, float | None]:
    """(prefill_ratio, decode_ratio|None) between the largest and smallest
    sweep row. decode_ratio is None when decode is disabled or below the
    clock's resolution: 0/0 must not masquerade as a ratio."""
    small, large = sweep_rows[-1], sweep_rows[0]
    prefill_ratio = large["prefill_s"] / small["prefill_s"]
    decode_ratio = None
    if large["decode_s"] > 0 and small["decode_s"] > 0:
        decode_ratio = large["decode_s"] / small["decode_s"]
    return prefill_ratio, decode_ratio


def ratio_failures(sweep_rows: list[dict]) -> list[str]:
    """A decode step's attention is bounded by the prefill's over the same
    positions: decode growing proportionally MORE than prefill between the
    same two sizes means the decode numbers were taken under contention.
    This is the check that caught a contaminated 8K figure at 7.8x its 2K
    value while prefill had grown only 1.07x."""
    prefill_ratio, decode_ratio = ratio_summary(sweep_rows)
    if decode_ratio is not None and decode_ratio > prefill_ratio:
        return [
            f"decode grew {decode_ratio:.2f}x but prefill only {prefill_ratio:.2f}x. "
            "Attention over the same positions cannot be the cause; the decode "
            "numbers were taken under contention."
        ]
    return []


def repeatability_failure(control_rows: list[dict], tolerance: float) -> str | None:
    if len(control_rows) != 2:
        return None
    c0, c1 = control_rows
    spread = abs(c1["prefill_s"] - c0["prefill_s"]) / c0["prefill_s"]
    if spread > tolerance:
        return (
            f"the same {c0['tokens']}-token prefill varied {spread:.0%} between two "
            "runs, so the machine cannot resolve the effect being measured."
        )
    return None


def swap_failures(rows: list[dict], base_limit_mb: float = 256.0) -> list[str]:
    """Growth beyond half the row's own high-water memory is the refusal
    signal; growth within it is the arriving working set displacing idle
    pages and is not.

    Four calibration runs earned this boundary: the 32K row's prefill held
    a 5.4% span across swap deltas of +2.8 GB, -1.1 GB, +12.3 GB, and
    +1.1 GB — the arriving row brings ~8 GB (weights + KV + allocator) and
    displaces up to about that much, harmlessly. What the guard exists for
    is the streamed-engine failure mode: swap growth while live weights
    thrash, which lands FAR beyond the row's own footprint (a 3.5x
    step-time blowup in the era that motivated this check). The half
    factor is the conservative side of the boundary: prefer refusing."""
    failures = []
    for row in rows:
        grew = row["swap_after_mb"] - row["swap_before_mb"]
        limit = max(base_limit_mb, (row.get("peak_rss_mb") or 0.0) / 2)
        if grew > limit:
            failures.append(
                f"swap grew {grew:.0f} MB during the {row['tokens']}-token row, beyond "
                f"half the row's own {row.get('peak_rss_mb', 0):.0f} MB footprint: the "
                "machine was thrashing while the clock ran."
            )
    return failures


def _child(args: argparse.Namespace) -> int:
    """Entry point for the per-size subprocess. The PARENT holds the sweep
    lock for the whole plan; a child never needs it (and taking it would
    refuse against the parent's own hold)."""
    row = measure_one(args.model, args.only_size, args.decode)
    print("ROW " + json.dumps(row), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--sizes", default="256,512,1024,2048,4096,8192,16384,32768")
    ap.add_argument("--decode", type=int, default=4, help="greedy steps per size")
    ap.add_argument("--tolerance", type=float, default=0.25, help="repeatability bound")
    ap.add_argument("--only-size", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.only_size is not None:
        return _child(args)

    try:
        plan = build_plan([int(s) for s in args.sizes.split(",")])
    except ValueError as exc:
        print(f"REFUSING: {exc}.", flush=True)
        return 2

    rows: list[dict[str, float]] = []
    # The lock is held by the PARENT for the whole sweep: per-child locking
    # would leave gaps between rows where a concurrent sweep's child slips
    # in, and this sweep's next child would then be refused by a lock it
    # itself held a moment ago — losing every row already measured.
    with sweep_lock(f"context_sweep[{args.sizes}]"):
        # A refused run must not leave a PREVIOUS run's report sitting at
        # --out, where it reads as this run's result.
        if args.out and os.path.exists(args.out):
            os.unlink(args.out)
        print(
            f"\n{'tokens':>7} {'prefill_s':>10} {'ttft_s':>8} {'decode_s/tok':>13} "
            f"{'blob_MB':>8} {'swap_delta_MB':>13}",
            flush=True,
        )
        for size, phase in plan:
            cmd = [
                sys.executable,
                os.path.abspath(__file__),
                "--only-size",
                str(size),
                "--decode",
                str(args.decode),
                "--model",
                args.model,
            ]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
            except subprocess.TimeoutExpired:
                # A hung child (model fetch stalled, Metal wedged) must not
                # wedge the sweep with it; 20 min exceeds the slowest planned
                # row by an order of magnitude.
                print(f"REFUSING: the {size}-token run did not finish (child hung).", flush=True)
                return 2
            line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("ROW ")), None)
            if line is None:
                print(proc.stdout, proc.stderr, file=sys.stderr, flush=True)
                print(f"REFUSING: the {size}-token run produced no measurement.", flush=True)
                return 2
            # A child killed mid-write (OOM inside the ROW print) or emitting
            # garbage after "ROW " is a refusal, not a traceback: one bad child
            # must not lose the measurements already collected.
            try:
                row = json.loads(line[4:])
            except json.JSONDecodeError:
                print(proc.stdout, proc.stderr, file=sys.stderr, flush=True)
                print(f"REFUSING: the {size}-token run produced an unreadable row.", flush=True)
                return 2
            bad = validate_row(row, size)
            if bad is not None:
                print(proc.stdout, proc.stderr, file=sys.stderr, flush=True)
                print(f"REFUSING: {bad}.", flush=True)
                return 2
            row["phase"] = phase
            rows.append(row)
            delta = row["swap_after_mb"] - row["swap_before_mb"]
            label = {"warmup": "+", "control": "*", "sweep": ""}[phase]
            print(
                f"{(str(size) + label):>7} {row['prefill_s']:>10.2f} {row['ttft_s']:>8.2f} "
                f"{row['decode_s']:>13.2f} {row['blob_bytes'] / 1e6:>8.1f} {delta:>13.0f}",
                flush=True,
            )

        sweep_rows = [r for r in rows if r["phase"] == "sweep"]
        control_rows = [r for r in rows if r["phase"] == "control"]
        prefill_ratio, decode_ratio = ratio_summary(sweep_rows)
        head = (
            f"{sweep_rows[0]['tokens']} vs {sweep_rows[-1]['tokens']}: prefill {prefill_ratio:.2f}x"
        )
        if decode_ratio is not None:
            head += f", decode {decode_ratio:.2f}x"
        else:
            head += ", decode n/a (disabled or below the clock)"
        print(f"\n{head}")

        failures = monotonicity_failures(sweep_rows)
        failures.extend(ratio_failures(sweep_rows))
        repeat = repeatability_failure(control_rows, args.tolerance)
        if repeat is not None:
            failures.append(repeat)
        # The warmup row is excluded from every check, the swap guard included:
        # the cold first load legitimately displaces pages, and its timing is
        # never quoted.
        failures.extend(swap_failures([r for r in rows if r["phase"] != "warmup"]))

        if failures:
            for f in failures:
                print(f"SELF-CHECK FAILED: {f}", flush=True)
            print("No timing from this run is usable.", flush=True)
            return 2
        print("self-check -> PASS", flush=True)

        report = {"model": args.model, "rows": rows, "machine_after": machine_state()}
        if args.out:
            # Written INSIDE the lock, and atomically: a concurrent sweep
            # sharing this path can neither observe a half-written report nor
            # have its own report clobbered by this one.
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            fd, tmp = tempfile.mkstemp(
                dir=os.path.dirname(args.out) or ".",
                prefix=os.path.basename(args.out) + ".",
                suffix=".part",
            )
            with os.fdopen(fd, "w") as fh:
                json.dump(report, fh, indent=2)
            os.replace(tmp, args.out)
            print(f"json -> {args.out}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
