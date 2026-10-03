#!/usr/bin/env python3
"""T21 A/B: lean greedy decode vs stock stream_generate on the real engine.

Run: MLXCACHE_LEAN_DECODE=1 uv run --extra mlx python sidecar/probes/lean_decode_ab.py
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sidecar"))


def main() -> int:
    from mlxcache_sidecar.server import MlxLmEngine

    engine = MlxLmEngine(os.environ.get("MLXCACHE_BENCH_MODEL", "Qwen/Qwen2-0.5B-Instruct"))
    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlx_lm.models.cache import make_prompt_cache

    prompt = engine.tokenize(
        json.dumps([{"role": "user", "content": "Count from one to ten, then summarize."}])
    )
    report = {"n_prompt_tokens": len(prompt)}

    # Stock path tokens (ground truth). NOTE: stream_generate's max_tokens
    # bounds FRAMES, not decoded tokens (its final frame carries the EOS
    # token), and it stops early at EOS. The lean loop's own cap is 256
    # tokens. Parity = the shared prefix must be identical.
    cache = make_prompt_cache(engine.model)
    stock = [
        r.token
        for r in stream_generate(
            engine.model,
            engine.tokenizer,
            prompt=mx.array(prompt),
            prompt_cache=cache,
            max_tokens=64,
        )
    ]

    # Lean path tokens.
    cache = make_prompt_cache(engine.model)
    lean = [t for t, _text in engine._lean_greedy_stream(prompt, cache)]

    report["parity"] = lean[: len(stock)] == stock
    report["stock_len"] = len(stock)
    report["lean_len"] = len(lean)
    if not report["parity"]:
        report["first_diff"] = next(
            (i for i, (a, b) in enumerate(zip(stock, lean[: len(stock)], strict=False)) if a != b),
            None,
        )

    # Text parity over the same token window: rebuild lean text capped at
    # stock_len tokens (the lean stream may be longer).
    cache = make_prompt_cache(engine.model)
    capped = []
    for tok, text in engine._lean_greedy_stream(prompt, cache):
        capped.append((tok, text))
        if len(capped) >= len(stock):
            break
    lean_text = "".join(t for _tok, t in capped)
    cache = make_prompt_cache(engine.model)
    stock_text = "".join(
        r.text
        for r in stream_generate(
            engine.model,
            engine.tokenizer,
            prompt=mx.array(prompt),
            prompt_cache=cache,
            max_tokens=64,
        )
    )
    report["text_parity"] = lean_text == stock_text
    if lean_text != stock_text:
        report["lean_text_head"] = lean_text[:120]
        report["stock_text_head"] = stock_text[:120]

    # Throughput: median of 3 runs each (decode-dominated: long output).
    def bench(fn):
        times = []
        for _ in range(3):
            cache = make_prompt_cache(engine.model)
            t0 = time.perf_counter()
            n = sum(1 for _ in fn(prompt, cache))
            times.append((time.perf_counter() - t0) * 1000)
        times.sort()
        return times[1], n

    lean_ms, lean_n = bench(engine._lean_greedy_stream)
    report["lean"] = {
        "median_ms": round(lean_ms, 1),
        "tokens": lean_n,
        "tok_per_s": round(lean_n / (lean_ms / 1000), 1),
    }

    def stock_stream(prompt, cache):
        for r in stream_generate(
            engine.model,
            engine.tokenizer,
            prompt=mx.array(prompt),
            prompt_cache=cache,
            max_tokens=256,
        ):
            yield r.token, r.text

    stock_ms, stock_n = bench(stock_stream)
    report["stock"] = {
        "median_ms": round(stock_ms, 1),
        "tokens": stock_n,
        "tok_per_s": round(stock_n / (stock_ms / 1000), 1),
    }
    report["speedup"] = round(stock_ms / lean_ms, 3)

    print(json.dumps(report, indent=2))
    return 0 if report["parity"] and report["text_parity"] else 1


if __name__ == "__main__":
    sys.exit(main())
