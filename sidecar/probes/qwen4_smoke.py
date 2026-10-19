#!/usr/bin/env python3
"""End-to-end smoke on the real NVFP4 checkpoint.

Builds the engine through the same adapter path the daemon uses, runs a
prefill, decodes a few tokens, checkpoints the cache, reloads it, and
continues. Prints timings and the token streams so the resume claim can be
checked against a literal token comparison rather than a summary.

This is a probe, not a test: it needs the 132.6 GB checkpoint and a 48 GB
machine, so it never runs in CI.
"""

from __future__ import annotations

import io
import os
import sys
import time

CHECKPOINT = os.environ.get(
    "MLXCACHE_Q4_CHECKPOINT", "/Users/lance/models/Qwen3.8-Flash-Next-NVFP4-mlxstream"
)

PROMPT = os.environ.get("Q4_SMOKE_PROMPT", "The capital of France is")
TOKENS = int(os.environ.get("Q4_SMOKE_TOKENS", "4"))


def argmax_token(logits) -> int:
    import mlx.core as mx

    return int(mx.argmax(logits[0, -1]).item())


def main() -> int:
    import mlx.core as mx
    from mlx_lm.models.cache import load_prompt_cache, save_prompt_cache
    from mlxcache_sidecar.engine_qwen4 import Qwen4ExpEngine

    t0 = time.perf_counter()
    engine = Qwen4ExpEngine(CHECKPOINT)
    print(f"boot            {time.perf_counter() - t0:7.2f}s", flush=True)
    print(f"engine_id       {engine.engine_id}")
    print(f"kv_dtype        {engine.kv_dtype}  bits={engine.kv_bits} group={engine.kv_group_size}")
    print(f"cache_layout    {engine.cache_layout}")
    print(f"tokenizer_hash  {engine.tokenizer_hash[:16]}")

    tokenizer = engine.tokenizer
    ids = tokenizer.encode(PROMPT)
    print(f"prompt tokens   {len(ids)}  {ids}")
    print(f"resident before {engine.loader.stats()}", flush=True)

    # Prefill through the adapter's own path and decode from the returned cache.
    # The cache covers tokens[:-1] by the pinned convention, so the last prompt
    # token is fed to produce the first continuation token.
    t0 = time.perf_counter()
    cache, logits = engine._prefill_cache(ids[:-1])
    first = argmax_token(engine.model(mx.array([[ids[-1]]]), cache=cache))
    ttft = time.perf_counter() - t0
    print(f"ttft            {ttft:7.2f}s  -> {first} ({tokenizer.decode([first])!r})", flush=True)

    t0 = time.perf_counter()
    stream = [first]
    for _ in range(TOKENS - 1):
        logits = engine.model(mx.array([[stream[-1]]]), cache=cache)
        stream.append(argmax_token(logits))
    decode = time.perf_counter() - t0
    steps = max(TOKENS - 1, 1)
    print(f"decode {TOKENS} tok  {decode:7.2f}s  ({decode / steps:.2f}s/token)")
    print(f"stream          {stream}")
    print(f"text            {tokenizer.decode(stream)!r}")
    print(f"resident after  {engine.loader.stats()}", flush=True)

    # The resume claim: serialize the cache and continue from the reloaded copy.
    # The name must carry the `.safetensors` extension: `load_prompt_cache`
    # dispatches on the file name, not on the payload, so a BufferIO named
    # otherwise fails to reload even though the bytes are valid.
    t0 = time.perf_counter()
    buffer = io.BytesIO()
    buffer.name = "smoke.safetensors"
    save_prompt_cache(buffer, cache)
    save = time.perf_counter() - t0
    payload = buffer.getvalue()
    print(
        f"serialize       {save:7.2f}s  {len(payload)} bytes "
        f"({len(payload) / (len(ids) + len(stream)):.0f} bytes/token)"
    )

    t0 = time.perf_counter()
    buffer.seek(0)
    restored = load_prompt_cache(buffer)
    print(f"deserialize     {time.perf_counter() - t0:7.2f}s")

    # Scratch path: keep decoding the live cache one more step.
    logits = engine.model(mx.array([[stream[-1]]]), cache=cache)
    scratch_next = argmax_token(logits)
    # Resumed path: same step from the reloaded cache.
    resumed_next = argmax_token(engine.model(mx.array([[stream[-1]]]), cache=restored))
    print(f"next token      scratch={scratch_next}  resumed={resumed_next}")
    if resumed_next != scratch_next:
        print("RESUME MISMATCH: reloaded cache does not continue the same stream")
        return 1
    print("RESUME OK: reloaded cache reproduces the next token")

    # The adapter's full generate path, which is what the daemon calls.
    t0 = time.perf_counter()
    tokens, text = engine.generate_with_text(ids, TOKENS)
    print(f"generate        {time.perf_counter() - t0:7.2f}s  {tokens} {text!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
