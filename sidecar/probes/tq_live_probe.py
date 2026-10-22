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


def _score_fidelity(model, tokenizer, tokens, head_dim: int, key_bits: int):
    """Score-from-packed fidelity on real KV: the packed gather-dot vs exact
    dequantized scoring, and the QJL correction's effect on the bias against
    TRUE (pre-quantization) inner products. This is what `tq.score` exists
    to record."""
    import mlx.core as mx  # noqa: PLC0415
    from mlx_lm.models.cache import make_prompt_cache
    from mlxcache_sidecar.tq.codebook import lloyd_max_codebook
    from mlxcache_sidecar.tq.codec import encode_keys
    from mlxcache_sidecar.tq.rotation import build_rotation
    from mlxcache_sidecar.tq.score import (
        qjl_correction,
        residual_signs_from_y,
        scores_from_packed,
    )

    cache = make_prompt_cache(model)
    model(mx.array(tokens)[None], cache=cache)
    mx.eval([c.state for c in cache])
    rotation = build_rotation(head_dim)
    codebook = lloyd_max_codebook(head_dim, key_bits)

    errs_packed, errs_corrected = [], []
    for c in cache:
        k, _ = c.state
        k_head = k[:, :1, :, :].astype(mx.float32)  # head 0: [B,1,S,d]
        packed, norms = encode_keys(k[:, :1, :, :], codebook, rotation)
        q = k_head[:, :, -1, :]  # the last position's keys as the query stand-in
        true_scores = mx.einsum("bhd,bhsd->bhs", q, k_head)
        projected = scores_from_packed(q[0, 0], packed, norms, head_dim, codebook, rotation)
        errs_packed.append(float(mx.abs(projected - true_scores).mean()))
        levels, boundaries, _m = codebook
        norms32 = mx.sqrt(mx.sum(k_head * k_head, axis=-1))
        u = k_head / mx.maximum(norms32, 1e-12)[..., None]
        y = (u @ rotation.T) * mx.sqrt(float(head_dim))
        idx = mx.zeros_like(y).astype(mx.uint32)
        for t in boundaries:
            idx = idx + (y > t).astype(mx.uint32)
        signs, r_norms = residual_signs_from_y(y, idx, codebook, head_dim)
        corrected = projected + qjl_correction(
            q[0, 0], signs, r_norms, rotation, head_dim, norms=norms
        )
        errs_corrected.append(float(mx.abs(corrected - true_scores).mean()))
    n = len(errs_packed)
    return {
        "score_err_packed_mean": round(sum(errs_packed) / n, 4),
        "score_err_qjl_corrected_mean": round(sum(errs_corrected) / n, 4),
        "key_bits": key_bits,
    }


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

    def build_live(head_dim: int):
        basis = (build_rotation(head_dim), lloyd_max_codebook(head_dim, 4))
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
    # Derive the head_dim from the model's real cache (the --model override
    # may not be d=128); the basis is built after, not hardcoded.
    head_dim = f16[0].keys.shape[-1]
    live = build_live(head_dim)
    live_stats = decode_stats(live, "live ")
    mx.clear_cache()
    score_stats = _score_fidelity(model, tokenizer, tokens, head_dim, 3)

    report = {
        "model": args.model,
        "prefix_tokens": len(tokens),
        "gen_tokens": args.gen,
        "f16": {k: f16_stats[k] for k in ("tok_s", "resident_mb")},
        "live": {k: live_stats[k] for k in ("tok_s", "resident_mb")},
        "resident_ratio": round(f16_stats["resident_mb"] / max(live_stats["resident_mb"], 1e-9), 2),
        "tpot_ratio": round(live_stats["tok_s"] / max(f16_stats["tok_s"], 1e-9), 3),
        "tokens_identical": f16_stats["out"] == live_stats["out"],
        "packed_scoring": score_stats,
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
