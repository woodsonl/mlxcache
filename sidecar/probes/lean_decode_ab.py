#!/usr/bin/env python3
"""T21 A/B: lean greedy decode vs stock stream_generate on the real engine.

Self-contained since the D3 cleanup: the lean loop was REMOVED from the
sidecar (measured 1.01x, default-off — see the design doc T21 note), so its
implementation lives HERE as the preserved measurement and the resurrection
template if a native forward pass (T14) ever needs the structure again.

Run: MLXCACHE_BENCH_MODEL=Qwen/Qwen2-0.5B-Instruct uv run --extra mlx python sidecar/probes/lean_decode_ab.py
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sidecar"))


def _lean_greedy_stream(engine, prompt, cache):
    """T21 lean decode: argmax-only generation with no per-token fluff.

    Yields (token_id, text_piece) exactly like mlx-lm's stream_generate.
    Keeps the async_eval pipeline (queue step i+1 before materializing step i)
    but skips: the full-vocab logsumexp (we only need argmax of logits), the
    GenerationResponse dataclass + peak-memory + TPS accounting per token, two
    levels of generator wrapping, and the wired_limit context. GREEDY only:
    any sampling (temperature/top_p) must use the mlx-lm path, where the
    softmax weights matter.
    """
    import mlx.core as mx  # noqa: PLC0415

    if cache is None:
        from mlx_lm.models.cache import make_prompt_cache  # noqa: PLC0415

        cache = make_prompt_cache(engine.model)

    # A detokenizer is STATEFUL per stream: one per call, never cached on the
    # engine (ThreadingHTTPServer interleaves streams).
    detok = engine.tokenizer.detokenizer
    eos = engine._eos_ids

    y = mx.array(prompt)[None]
    logits = engine.model(y, cache=cache)
    y = mx.argmax(logits[:, -1, :], axis=-1)

    n = 0
    while True:
        # Queue the next step before materializing this token: the GPU works
        # on step i+1 while Python detokenizes step i.
        next_logits = engine.model(y[None], cache=cache)
        next_y = mx.argmax(next_logits[:, -1, :], axis=-1)
        mx.async_eval(next_y)

        token = int(y.item())
        detok.add_token(token)
        # Finalize BEFORE the final yield (EOS, or the cap consumed by this
        # step): the last yielded segment must be the finalized one, or any
        # text still buffered in the detokenizer is silently dropped.
        last = token in eos or n + 1 >= 256
        if last:
            detok.finalize()
        yield token, detok.last_segment
        if last:
            break

        y = next_y
        n += 1


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
    lean = [t for t, _text in _lean_greedy_stream(engine, prompt, cache)]

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
    for tok, text in _lean_greedy_stream(engine, prompt, cache):
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

    lean_ms, lean_n = bench(lambda p, c: _lean_greedy_stream(engine, p, c))
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
