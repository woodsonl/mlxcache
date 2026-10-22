#!/usr/bin/env python3
"""TQ tier quality probe: the pre-registered gates, measured on a real model.

One run answers the six questions the TQ persistence tier's ship decision
rests on (thresholds recorded in docs/designs/tq-tier.md before this probe
was first run against the 7B):

1. G1 fidelity — per-layer cosine of quantize→dequantize against the f16 KV
   the engine actually holds, for the TQ tiers (k3/k4 keys, v2/v4 values)
   AND the plain group-quant tiers (q8, q4) on the same tensors: the
   bake-off. The paper's Theorem 1 puts worst-case rotated-key cos at
   1 − D/2 (0.985 at k3, 0.9955 at k4); real keys should beat it.
2. G5 bytes — payload bytes/token for every tier vs f16.
3. G2 adopt budget — decode wall (TQ) vs mlx-lm load wall (f16) at the
   same prefix.
4. G4 append stability — encode(A) vs encode(A+B): prior tokens' bytes
   must be identical (the chunk-store identity property, on REAL KV).
5. G3 generation — greedy decode from a TQ-resumed cache vs scratch:
   agreement fraction and first-divergence position (lossy by design; the
   number is the contract, not a token-identity claim).
6. q4 bake-off — plain bits=4 rides the existing plumbing with zero port
   work; if it matches TQ k3v4 fidelity at similar bytes, the port must
   justify itself on the k3v2 arm alone.

Run: uv run --extra mlx python sidecar/probes/tq_quality_probe.py \
       [--model mlx-community/Qwen2.5-7B-Instruct-4bit] [--tokens 1500]
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DOC = REPO_ROOT / "scripts" / "fixtures" / "t22" / "design_doc.md"


def _load(model_id: str):
    from mlx_lm import load

    return load(model_id)


def _prefill(model, tokens: list[int]):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    logits = model(mx.array(tokens)[None], cache=cache)
    mx.eval([c.state for c in cache], logits)
    return cache


def _cos(a, b) -> float:
    import mlx.core as mx

    a32, b32 = a.astype(mx.float32), b.astype(mx.float32)
    per = (a32 * b32).sum(-1) / ((a32 * a32).sum(-1).sqrt() * (b32 * b32).sum(-1).sqrt())
    return float(per.mean())


def _per_layer_cos(orig, reco) -> list[float]:
    out = []
    for o, r in zip(orig, reco, strict=True):
        k0, v0 = o.state
        k1, v1 = r.state
        out.append(min(_cos(k0, k1), _cos(v0, v1)))
    return out


def _f16_bytes(cache) -> int:
    return sum(c.nbytes for c in cache)


def _to_quantized(cache, bits: int, group: int = 64):
    return [c.to_quantized(group_size=group, bits=bits) for c in cache]


def _q_state_bytes(quant_cache) -> int:
    return sum(c.nbytes for c in quant_cache)


def _greedy(model, tokenizer, prompt_tokens, cache, max_tokens: int) -> list[int]:
    import mlx.core as mx
    from mlx_lm import stream_generate

    out = []
    for resp in stream_generate(
        model, tokenizer, prompt=mx.array(prompt_tokens), prompt_cache=cache, max_tokens=max_tokens
    ):
        out.append(resp.token)
        if len(out) >= max_tokens:
            break
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
    parser.add_argument("--tokens", type=int, default=1500)
    parser.add_argument("--gen-tokens", type=int, default=48)
    args = parser.parse_args()

    from mlxcache_sidecar.tq import decode_payload, encode_cache

    model, tokenizer = _load(args.model)
    corpus = FIXTURE_DOC.read_text()
    tokens = tokenizer.encode(corpus)[: args.tokens]
    print(f"model={args.model} prefix_tokens={len(tokens)}", file=sys.stderr)

    report: dict = {"model": args.model, "prefix_tokens": len(tokens), "tiers": {}}

    # --- f16 reference state (a copy for every arm; caches mutate on decode)
    f16_cache = _prefill(model, tokens)
    len(f16_cache)

    # --- G1 + G5: fidelity and bytes per tier -----------------------------
    arms = {
        "q8": ("bits", 8),
        "q4": ("bits", 4),
        "q3": ("bits", 3),
        "q2": ("bits", 2),
        "tq-k3v4-g64": ("tq", (3, 4, 64)),
        "tq-k3v2-g64": ("tq", (3, 2, 64)),
        "tq-k4v4-g64": ("tq", (4, 4, 64)),
    }
    f16_total = _f16_bytes(f16_cache)
    for name, (kind, param) in arms.items():
        base = _prefill(model, tokens)  # fresh capture: caches are stateful
        if kind == "bits":
            quant = _to_quantized(base, param)
            # dequantized view for fidelity: rebuild via a round trip through
            # the state setter on a fresh KVCache stack
            reco = []
            from mlx_lm.models.cache import KVCache

            for q in quant:
                k = q.state[0]
                v = q.state[1]
                import mlx.core as mx

                kd = mx.dequantize(*k, group_size=q.group_size, bits=q.bits)
                vd = mx.dequantize(*v, group_size=q.group_size, bits=q.bits)
                c = KVCache()
                c.state = (kd, vd)
                reco.append(c)
            payload_bytes = _q_state_bytes(quant)
            # the wire payload is the safetensors frame, not the raw tensors
            from mlx_lm.models.cache import save_prompt_cache

            buf = io.BytesIO()
            save_prompt_cache(buf, quant)
            payload_bytes = len(buf.getvalue())
        else:
            kb, vb, g = param
            payload = encode_cache(base, kb, vb, g)
            reco = decode_payload(payload)
            payload_bytes = len(payload)
        cos = _per_layer_cos(f16_cache, reco)
        report["tiers"][name] = {
            "payload_bytes": payload_bytes,
            "bytes_per_token": round(payload_bytes / len(tokens), 1),
            "x_vs_f16": round(f16_total / payload_bytes, 2),
            "min_layer_cos": round(min(cos), 5),
            "mean_layer_cos": round(sum(cos) / len(cos), 5),
        }
        print(f"{name:>14}: {report['tiers'][name]}", file=sys.stderr)

    # --- G2: adopt budget (TQ decode vs f16 load at the same prefix) ------
    import mlx.core as mx
    from mlx_lm.models.cache import load_prompt_cache

    buf = io.BytesIO()
    from mlx_lm.models.cache import save_prompt_cache

    save_prompt_cache(buf, f16_cache)
    f16_payload = buf.getvalue()
    t0 = time.perf_counter()
    load_prompt_cache(_named(io.BytesIO(f16_payload)))
    f16_load_ms = (time.perf_counter() - t0) * 1000
    tq_payload = encode_cache(_prefill(model, tokens), 3, 4, 64)
    t0 = time.perf_counter()
    decode_payload(tq_payload)
    tq_decode_ms = (time.perf_counter() - t0) * 1000
    report["adopt_budget_ms"] = {
        "f16_load": round(f16_load_ms, 1),
        "tq_decode_k3v4": round(tq_decode_ms, 1),
        "tq_delta": round(tq_decode_ms - f16_load_ms, 1),
        "prefix_tokens": len(tokens),
    }

    # --- G4: append stability on real KV -----------------------------------
    short_tokens, long_tokens = tokens[:1000], tokens[:1500]
    a = encode_cache(_prefill(model, short_tokens), 3, 4, 64)
    b = encode_cache(_prefill(model, long_tokens), 3, 4, 64)
    ta, _ = mx.load(_named(io.BytesIO(a)), return_metadata=True)
    tb, _ = mx.load(_named(io.BytesIO(b)), return_metadata=True)
    stable = all(
        bool((ta[n][:, :, :1000] == tb[n][:, :, :1000]).all())
        for n in ta
        if not n.startswith("rot.")
    )
    report["append_stable_real_kv"] = stable

    # --- G3: greedy agreement, scratch vs TQ-resumed -----------------------
    prompts = [
        tokens + tokenizer.encode("\n\nSummarize the document in one paragraph."),
        tokens + tokenizer.encode("\n\nWhat is the KV cache convention this document pins?"),
        tokens + tokenizer.encode("\n\nList the three gates named in the document."),
    ]
    agreements, first_div = [], []
    for p in prompts:
        scratch_cache = _prefill(model, p[:-1])
        scratch_out = _greedy(model, tokenizer, p[-1:], scratch_cache, args.gen_tokens)
        tq_payload = encode_cache(_prefill(model, p[:-1]), 3, 4, 64)
        resumed = decode_payload(tq_payload)
        resumed_out = _greedy(model, tokenizer, p[-1:], resumed, args.gen_tokens)
        same = [x == y for x, y in zip(scratch_out, resumed_out, strict=False)]
        agreements.append(sum(same) / max(len(same), 1))
        k = next((i for i, s in enumerate(same) if not s), None)
        first_div.append(k)
    report["generation_greedy_k3v4"] = {
        "agreement_frac": [round(a, 3) for a in agreements],
        "first_divergence_pos": first_div,
        "gen_tokens": args.gen_tokens,
    }

    print(json.dumps(report, indent=2))
    return 0


def _named(buf: io.BytesIO) -> io.BytesIO:
    buf.name = "checkpoint.safetensors"
    return buf


if __name__ == "__main__":
    raise SystemExit(main())
