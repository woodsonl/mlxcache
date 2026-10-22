#!/usr/bin/env python3
"""TQ live-tier probe: what the compressed-resident cache costs and saves.

One run answers the live tier's two gates on a real model:

1. TPOT — decode tokens/second with the live cache (dequant per step) vs the
   f16 cache, at a matched context. The trade being measured: ~3.6x smaller
   resident KV bought with per-step dequant compute.
2. RESIDENT — the live stack's nbytes vs an f16 stack's at the same context
   (the unified-memory saving, capacity rounding included).

Run: uv run --extra mlx python sidecar/probes/tq_live_probe.py \
       [--model mlx-community/Qwen2.5-7B-Instruct-4bit] [--tokens 2000] [--gen 32]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DOC = REPO_ROOT / "scripts" / "fixtures" / "t22" / "design_doc.md"


def _load(model_id: str):
    from mlx_lm import load

    return load(model_id)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
    parser.add_argument("--tokens", type=int, default=2000)
    parser.add_argument("--gen", type=int, default=32)
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlx_lm.models.cache import make_prompt_cache
    from mlxcache_sidecar.tq.codebook import lloyd_max_codebook
    from mlxcache_sidecar.tq.live import TurboQuantKVCache
    from mlxcache_sidecar.tq.rotation import build_rotation

    model, tokenizer = _load(args.model)
    corpus = FIXTURE_DOC.read_text()
    tokens = tokenizer.encode(corpus)[: args.tokens]
    n_layers = len(make_prompt_cache(model))

    def build_live():
        basis = (build_rotation(128), lloyd_max_codebook(128, 4))
        return [TurboQuantKVCache(4, 4, 64, basis=basis) for _ in range(n_layers)]

    def decode_stats(cache, label):
        # Feed prompt[:-1] through the cache, then time `gen` decode steps.
        model(mx.array(tokens[:-1])[None], cache=cache)
        mx.eval([c.state for c in cache])
        t0 = time.perf_counter()
        out = []
        for resp in stream_generate(
            model,
            tokenizer,
            prompt=mx.array([tokens[-1]]),
            prompt_cache=cache,
            max_tokens=args.gen,
        ):
            out.append(resp.token)
        dt = time.perf_counter() - t0
        nbytes = sum(c.nbytes for c in cache)
        print(
            f"{label}: {len(out)} tokens in {dt:.2f}s -> {len(out) / dt:.1f} tok/s, "
            f"resident {nbytes / 1e6:.1f} MB",
            file=sys.stderr,
        )
        return {"tok_s": round(len(out) / dt, 2), "resident_mb": round(nbytes / 1e6, 1), "out": out}

    f16 = make_prompt_cache(model)
    f16_stats = decode_stats(f16, "f16  ")
    mx.clear_cache()
    live = build_live()
    live_stats = decode_stats(live, "live ")
    mx.clear_cache()

    report = {
        "model": args.model,
        "prefix_tokens": len(tokens),
        "gen_tokens": args.gen,
        "f16": {k: f16_stats[k] for k in ("tok_s", "resident_mb")},
        "live": {k: live_stats[k] for k in ("tok_s", "resident_mb")},
        "resident_ratio": round(f16_stats["resident_mb"] / max(live_stats["resident_mb"], 1e-9), 2),
        "tpot_ratio": round(live_stats["tok_s"] / max(f16_stats["tok_s"], 1e-9), 3),
        "tokens_identical": f16_stats["out"] == live_stats["out"],
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
