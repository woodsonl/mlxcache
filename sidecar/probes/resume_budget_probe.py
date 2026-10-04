#!/usr/bin/env python3
"""Step-2 probe: 7B resume budget at 30-50K tokens — wall-to-first-token.

Question (R1-5 at 7B scale): when the daemon loads a persisted checkpoint for
Qwen2.5-7B-Instruct-4bit at 30K-50K tokens, how long until the FIRST generated
token can be produced? The budget is 2s TTFT.

What the probe measures, per prefix length (30K, 50K) and per KV tier (f16,
8-bit g64 — the T12 parity-safe config):
  1. publish: prefill f16 -> quantize (q8 only) -> save to disk = blob size.
  2. resume (fresh process-shaped): drop all refs, then load_prompt_cache +
     feed ONLY the uncovered tail (the last token, the daemon's delta
     convention) -> mx.eval -> time-to-first-logits = WALL TO FIRST TOKEN,
     page-in from disk included (the earlier 0.6 ms number was mmap-lazy and
     did NOT include page-in; this probe closes that gap).
  3. resume-after-sleep: repeat after 2s idle to let the mmap cool, so the
     cold-start number is honest, not a warm-page artifact.
  4. warm re-generate: a second generate on the already-paged cache = the
     steady-state continuation speed (context).

Also measures process-shape parity: f16 vs q8 first token identity on the
same prompt (the q8 tier only ships if the first generated token matches —
full 32-token greedy parity was proven in T12 at 0.5B; the 7B check here is
the same greedy stream from the reloaded cache vs the f16 in-memory run).

Run: uv run --extra mlx python sidecar/probes/resume_budget_probe.py \
        --model mlx-community/Qwen2.5-7B-Instruct-4bit --out /tmp/resume7b.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import tempfile
import time


def make_tokens(tokenizer, n: int) -> list[int]:
    base = "The quick brown fox jumps over the lazy dog. "
    unit = tokenizer.encode(base)
    reps = n // len(unit) + 1
    return (tokenizer.encode(base * reps))[:n]


def prefill_into_cache(model, cache, tokens):
    import mlx.core as mx

    logits = model(mx.array(tokens)[None], cache=cache)
    mx.eval([c.state for c in cache], logits)
    return logits


def first_token_from_cache(model, cache, last_token: int) -> tuple[int, float]:
    """Feed the single uncovered token, eval, argmax -> (token, wall_ms).

    This is exactly the daemon's resume shape: load blob, feed tokens[-1:],
    take the first prediction. Wall time covers the whole call — for a fresh
    mmap that INCLUDES page-in of the blob bytes.
    """
    import mlx.core as mx

    t0 = time.perf_counter()
    logits = model(mx.array([last_token])[None], cache=cache)
    tok = int(mx.argmax(logits[:, -1, :], axis=-1).item())
    mx.eval(logits)
    return tok, (time.perf_counter() - t0) * 1000


def greedy_stream(model, cache, last_token: int, n: int) -> list[int]:
    import mlx.core as mx

    logits = model(mx.array([last_token])[None], cache=cache)
    mx.eval(logits)
    out = []
    for _ in range(n):
        tok = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        out.append(tok)
        logits = model(mx.array([tok])[None], cache=cache)
        mx.eval(logits)
    return out


def save_cache(cache, path: str) -> float:
    from mlx_lm.models.cache import save_prompt_cache

    t0 = time.perf_counter()
    save_prompt_cache(path, cache)
    return (time.perf_counter() - t0) * 1000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
    ap.add_argument("--lengths", type=int, nargs="+", default=[30000, 50000])
    ap.add_argument("--gen-tokens", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from mlx_lm import load
    from mlx_lm.models.cache import load_prompt_cache, make_prompt_cache

    t0 = time.perf_counter()
    model, tokenizer = load(args.model)
    load_model_s = time.perf_counter() - t0

    report: dict = {
        "model": args.model,
        "model_load_s": round(load_model_s, 2),
        "budget_ms": 2000,
        "lengths": {},
    }

    for n_tokens in args.lengths:
        tokens = make_tokens(tokenizer, n_tokens)
        entry: dict = {"prefix_tokens": len(tokens)}

        # ---- f16 tier: publish (prefill+save) then resume ----
        cache = make_prompt_cache(model)
        t0 = time.perf_counter()
        prefill_into_cache(model, cache, tokens[:-1])
        prefill_ms = (time.perf_counter() - t0) * 1000
        entry["prefill_f16_ms"] = round(prefill_ms, 0)
        f16_first = None
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as fh:
            f16_path = fh.name
        entry["save_f16_ms"] = round(save_cache(cache, f16_path), 1)
        entry["f16_blob_gb"] = round(os.path.getsize(f16_path) / 1e9, 3)
        entry["f16_bytes_per_token"] = round(os.path.getsize(f16_path) / len(tokens), 0)
        # Reference stream from the in-memory f16 cache (parity target).
        f16_first = greedy_stream(model, cache, tokens[-1], args.gen_tokens)
        del cache
        gc.collect()
        import mlx.core as mx

        mx.clear_cache()

        # Resume, cold-ish: fresh load_prompt_cache + first token, twice, with
        # a 2s gap so the second load is not page-cache-flattered. The FIRST
        # number is the honest cold-start (disk was just written, some pages
        # may be cached — that flatters in the daemon's favor on a warm
        # restart, which is the common case; the second load after idle is
        # the pessimistic bound).
        for attempt in (1, 2):
            if attempt == 2:
                time.sleep(2.0)
            t0 = time.perf_counter()
            r_cache = load_prompt_cache(f16_path)
            load_ms = (time.perf_counter() - t0) * 1000
            tok, ttft_ms = first_token_from_cache(model, r_cache, tokens[-1])
            entry[f"f16_resume{attempt}_load_ms"] = round(load_ms, 1)
            entry[f"f16_resume{attempt}_ttft_ms"] = round(ttft_ms, 1)
            entry[f"f16_resume{attempt}_budget_ok"] = (load_ms + ttft_ms) < 2000
            # First-token identity vs the f16 reference stream.
            entry[f"f16_resume{attempt}_first_token_ok"] = tok == f16_first[0]
            del r_cache
            gc.collect()
        # q8 parity at this scale: reload f16 blob, quantize, full stream.
        r_cache = load_prompt_cache(f16_path)
        q_ref = [c.to_quantized(group_size=64, bits=8) for c in r_cache]
        q_first = greedy_stream(model, q_ref, tokens[-1], args.gen_tokens)
        entry["q8_from_f16_stream_matches_f16"] = q_first == f16_first
        del r_cache, q_ref
        gc.collect()
        mx.clear_cache()
        os.unlink(f16_path)

        # ---- q8 tier: publish (prefill+quantize+save) then resume ----
        cache = make_prompt_cache(model)
        t0 = time.perf_counter()
        prefill_into_cache(model, cache, tokens[:-1])
        prefill_ms = (time.perf_counter() - t0) * 1000
        q_cache = [c.to_quantized(group_size=64, bits=8) for c in cache]
        quantize_ms = (time.perf_counter() - t0) * 1000 - prefill_ms
        entry["prefill_q8_ms"] = round(prefill_ms, 0)
        entry["quantize_ms"] = round(quantize_ms, 0)
        del cache
        gc.collect()
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as fh:
            q8_path = fh.name
        entry["save_q8_ms"] = round(save_cache(q_cache, q8_path), 1)
        entry["q8_blob_gb"] = round(os.path.getsize(q8_path) / 1e9, 3)
        entry["q8_bytes_per_token"] = round(os.path.getsize(q8_path) / len(tokens), 0)
        q8_first = greedy_stream(model, q_cache, tokens[-1], args.gen_tokens)
        entry["q8_persisted_stream_matches_f16"] = q8_first == f16_first
        del q_cache
        gc.collect()
        mx.clear_cache()

        for attempt in (1, 2):
            if attempt == 2:
                time.sleep(2.0)
            t0 = time.perf_counter()
            r_cache = load_prompt_cache(q8_path)
            load_ms = (time.perf_counter() - t0) * 1000
            tok, ttft_ms = first_token_from_cache(model, r_cache, tokens[-1])
            entry[f"q8_resume{attempt}_load_ms"] = round(load_ms, 1)
            entry[f"q8_resume{attempt}_ttft_ms"] = round(ttft_ms, 1)
            entry[f"q8_resume{attempt}_budget_ok"] = (load_ms + ttft_ms) < 2000
            entry[f"q8_resume{attempt}_first_token_ok"] = tok == f16_first[0]
            del r_cache
            gc.collect()
        os.unlink(q8_path)

        report["lengths"][n_tokens] = entry
        # Incremental output: print after each length so a crash still leaves
        # the earlier measurement.
        with open(args.out or "/tmp/resume_budget_7b.json", "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"--- {n_tokens} tokens measured", file=sys.stderr)
        print(json.dumps({n_tokens: entry}, indent=2), file=sys.stderr)

    with open(args.out or "/tmp/resume_budget_7b.json", "w") as fh:
        json.dump(report, fh, indent=2)
    json.dump(report, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
