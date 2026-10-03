#!/usr/bin/env python3
"""T11: mlx capability probe — what the runtime supports, measured, not assumed.

Answers the optimization plan's gating questions on THIS machine + version:

1. Metal availability + build (T14 FFI compute depends on it).
2. Prefill throughput at f16 (the baseline every tier must beat).
3. mx.compile on the model's forward pass: win or not (T14 must beat the
   Python-driven loop, and compile is the cheapest approximation of what a
   native FFI could win).
4. QuantizedKVCache availability + save/load round trip (T12's prerequisite).
5. Tokenizer hot path cost (T13's baseline: what native tokenize must beat).

Run: uv run --extra mlx python sidecar/probes/mlx_capability_probe.py \
        [--model Qwen/Qwen2-0.5B-Instruct] [--tokens 512]
Output: JSON on stdout (machine-readable) + human notes on stderr.
"""

from __future__ import annotations

import argparse
import json
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2-0.5B-Instruct")
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--out", help="also write JSON here")
    args = ap.parse_args()

    import mlx.core as mx

    report: dict = {"model": args.model, "tokens": args.tokens}
    report["metal"] = {
        "available": bool(mx.metal.is_available()),
        "device": str(mx.default_device()),
    }
    if not report["metal"]["available"]:
        json.dump(report, sys.stdout, indent=2)
        print("\nno Metal — everything below is moot", file=sys.stderr)
        return 1

    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    t0 = time.perf_counter()
    model, tokenizer = load(args.model)
    report["load_s"] = round(time.perf_counter() - t0, 2)

    # Deterministic token stream of the requested length.
    text = "The quick brown fox jumps over the lazy dog. "
    tokens = tokenizer.encode(text * (args.tokens // len(tokenizer.encode(text)) + 1))[
        : args.tokens
    ]
    report["n_tokens"] = len(tokens)

    def prefill_once() -> float:
        cache = make_prompt_cache(model)
        t = time.perf_counter()
        logits = model(mx.array(tokens)[None], cache=cache)
        mx.eval([c.state for c in cache], logits)
        return time.perf_counter() - t

    # Warm Metal pipelines once, then measure.
    prefill_once()
    t = min(prefill_once() for _ in range(3))
    report["prefill_f16"] = {
        "wall_s": round(t, 4),
        "tok_per_s": round(len(tokens) / t, 1),
    }

    # mx.compile: JIT-compile the model call. If this wins, a native FFI that
    # skips the Python loop must win by MORE than this (it also skips the
    # graph build each call).
    try:
        compiled_call = mx.compile(model)
        cache = make_prompt_cache(model)
        t0 = time.perf_counter()
        logits = compiled_call(mx.array(tokens)[None], cache=cache)
        mx.eval([c.state for c in cache], logits)
        wall = time.perf_counter() - t0
        # One warm + one measured (first call pays compile).
        cache = make_prompt_cache(model)
        t0 = time.perf_counter()
        logits = compiled_call(mx.array(tokens)[None], cache=cache)
        mx.eval([c.state for c in cache], logits)
        t = time.perf_counter() - t0
        report["prefill_f16_compiled"] = {
            "first_call_s": round(wall, 4),
            "steady_s": round(t, 4),
            "tok_per_s": round(len(tokens) / t, 1),
            "speedup_vs_eager": round(t and (min(prefill_once() for _ in range(3)) / t), 2),
        }
    except Exception as exc:  # noqa: BLE001 — capability probe must not die
        report["prefill_f16_compiled"] = {"error": f"{type(exc).__name__}: {exc}"}

    # QuantizedKVCache round trip (T12 prerequisite): can we even build one
    # for this model, save it, load it back?
    try:
        from mlx_lm.models.cache import (
            load_prompt_cache,
            save_prompt_cache,
        )

        cache = make_prompt_cache(model)
        logits = model(mx.array(tokens[:256])[None], cache=cache)
        mx.eval([c.state for c in cache], logits)
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as fh:
            path = fh.name
        save_prompt_cache(path, cache)
        size = os.path.getsize(path)
        restored = load_prompt_cache(path)
        os.unlink(path)
        report["kv_roundtrip_f16"] = {
            "bytes": size,
            "bytes_per_token": round(size / 256, 1),
            "load_ok": restored is not None,
        }
    except Exception as exc:  # noqa: BLE001
        report["kv_roundtrip_f16"] = {"error": f"{type(exc).__name__}: {exc}"}

    # Tokenizer hot path (T13 baseline): the daemon tokenizes EVERY request.
    prompt = json.dumps([{"role": "user", "content": text * 20}])
    tokenizer.encode(prompt)  # warm
    n = 200
    t0 = time.perf_counter()
    for _ in range(n):
        tokenizer.encode(prompt)
    per_call = (time.perf_counter() - t0) / n * 1e6
    report["tokenize"] = {"per_call_us": round(per_call, 1), "prompt_chars": len(prompt)}

    json.dump(report, sys.stdout, indent=2)
    print()
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
