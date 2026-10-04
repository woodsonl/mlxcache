#!/usr/bin/env python3
"""T12: quantized-KV persistence probe — can the disk tier shrink 4-8x?

The R1-5 budget math made raw f16 KV too big for long contexts (13 GB at
32B-50K tokens). mlx-lm >= 0.28 supports QuantizedKVCache (4/8-bit, grouped).
If a quantized cache (a) round-trips through save/load_prompt_cache, (b) keeps
generation token-identical for greedy decoding (or within a measured, bounded
KL/acceptance delta for sampled decoding), and (c) actually shrinks the blob,
the persistence tier can serve 4-8x longer prefixes within the same budget.

Three experiments, each printing JSON:
1. parity_greedy: f16 KV vs quantized KV, greedy decode — must be IDENTICAL
   token streams to ship as a byte-compatible tier (or we pin a known-good
   (bits, group) with a bounded-delta contract instead). SCOPE CAVEAT (QA
   D4): parity is validated at ONE prompt length per run (the --tokens
   argument, defaulting small). Quantization error is scale-dependent in
   general; the shipped q8 g64 tier's parity contract is additionally pinned
   by sidecar/tests at 0.5B and by the 20K-token 7B replay evidence in the
   design doc, but this probe alone does not prove length-invariance.
2. sizes: bytes/token f16 vs 8-bit vs 4-bit at a realistic prefix length.
3. resume_budget: load + delta-prefill wall time at 50K tokens vs the 2s TTFT
   budget (R1-5).

Run: uv run --extra mlx python sidecar/probes/quantized_kv_probe.py \
        [--model Qwen/Qwen2-0.5B-Instruct]

Caveat: the parity check prefills the full prompt on both paths and feeds the
final token again — it validates argmax parity only, NOT the shipped
save/reload delta-prefill convention end-to-end; the authoritative byte-parity
proofs are sidecar/tests/test_roundtrip_real.py and the daemon e2e suite.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time


def gen_with_cache(model, tokenizer, prompt_tokens, cache, max_tokens=32):
    """Greedy decode with an explicit cache; returns token list."""
    import mlx.core as mx

    logits = model(mx.array(prompt_tokens)[None], cache=cache)
    mx.eval(logits)
    out = []
    for _ in range(max_tokens):
        token = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        out.append(token)
        logits = model(mx.array([token])[None], cache=cache)
        mx.eval(logits)
    return out


def prefill_into_cache(model, cache, tokens):
    import mlx.core as mx

    logits = model(mx.array(tokens)[None], cache=cache)
    mx.eval([c.state for c in cache], logits)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2-0.5B-Instruct")
    ap.add_argument("--prefix-tokens", type=int, default=256)
    ap.add_argument("--gen-tokens", type=int, default=32)
    ap.add_argument("--long-tokens", type=int, default=20000)
    ap.add_argument("--out", help="also write JSON here")
    args = ap.parse_args()

    from mlx_lm import load
    from mlx_lm.models.cache import (
        load_prompt_cache,
        make_prompt_cache,
        save_prompt_cache,
    )

    model, tokenizer = load(args.model)
    base = "The quick brown fox jumps over the lazy dog. "
    unit = tokenizer.encode(base)

    def make_tokens(n: int) -> list[int]:
        reps = n // len(unit) + 1
        return (tokenizer.encode(base * reps))[:n]

    report: dict = {"model": args.model}

    # Shared scratch prefix for all three configs.
    tokens = make_tokens(args.prefix_tokens)
    continuation = make_tokens(args.long_tokens)

    # ---- 1. greedy parity: f16 vs quantized (8-bit, 4-bit) ----
    scratch = make_prompt_cache(model)
    prefill_into_cache(model, scratch, tokens)
    f16_out = gen_with_cache(model, tokenizer, tokens[-1:], scratch, args.gen_tokens)

    # Quantized AFTER prefill (the persistence shape: quantize once, store,
    # reload) — via to_quantized on each layer, then save/load.
    parity: dict = {}
    for bits, group in [(8, 64), (4, 64), (4, 32)]:
        cache = make_prompt_cache(model)
        prefill_into_cache(model, cache, tokens)
        try:
            quantized = [c.to_quantized(group_size=group, bits=bits) for c in cache]
        except Exception as exc:  # noqa: BLE001
            parity[f"{bits}bit_g{group}"] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        # Generation must run on the QUANTIZED cache: swap into a fresh list.
        out = gen_with_cache(model, tokenizer, tokens[-1:], quantized, args.gen_tokens)
        match = out == f16_out
        n_diff = sum(a != b for a, b in zip(out, f16_out, strict=True))
        parity[f"{bits}bit_g{group}"] = {
            "identical": match,
            "n_diff": n_diff,
            "of": len(f16_out),
            "first_diff": next(
                (i for i, (a, b) in enumerate(zip(out, f16_out, strict=True)) if a != b),
                None,
            ),
        }
    report["parity_greedy"] = {"f16": f16_out, "quantized": parity}

    # ---- 2. sizes: bytes/token per config at the long-prefix length ----
    sizes: dict = {}
    for label, bits, group in [("f16", None, None), ("8bit_g64", 8, 64), ("4bit_g64", 4, 64)]:
        cache = make_prompt_cache(model)
        prefill_into_cache(model, cache, continuation)
        if bits is not None:
            cache = [c.to_quantized(group_size=group, bits=bits) for c in cache]
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as fh:
            path = fh.name
        save_prompt_cache(path, cache)
        size = os.path.getsize(path)
        os.unlink(path)
        sizes[label] = {
            "bytes": size,
            "bytes_per_token": round(size / len(continuation), 1),
            "gb_at_50k": round(size / len(continuation) * 50_000 / 1e9, 2),
        }
        import mlx.core as _mx

        _mx.clear_cache()
    report["sizes"] = sizes

    # ---- 3. resume budget at the long length: load + delta prefill ----
    cache = make_prompt_cache(model)
    prefill_into_cache(model, cache, continuation)
    cache = [c.to_quantized(group_size=64, bits=8) for c in cache]
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as fh:
        path = fh.name
    save_prompt_cache(path, cache)
    t0 = time.perf_counter()
    restored = load_prompt_cache(path)
    load_ms = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    # Delta = the tokens the reloaded cache does not cover: the final one.
    gen_with_cache(model, tokenizer, continuation[-1:], restored, 4)
    delta_ms = (time.perf_counter() - t0) * 1000
    os.unlink(path)
    report["resume_budget_8bit"] = {
        "prefix_tokens": len(continuation),
        "load_ms": round(load_ms, 1),
        "delta_prefill_ms": round(delta_ms, 1),
        "budget_2s_ok": (load_ms + delta_ms) < 2000,
    }

    json.dump(report, sys.stdout, indent=2)
    print()
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
