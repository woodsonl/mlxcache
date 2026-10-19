#!/usr/bin/env python3
"""Decompose WHERE KV byte identity breaks on the delta-prefill path.

The smoke capture showed chain blobs diverging at an exact per-KV-head
boundary (one head byte-stable, the next rewritten from byte 0). This probe
drives mlx-lm directly — no daemon — and isolates each step:

  A  prefill(tokens[:P])            -> save
  B  SAME cache + feed delta        -> save   (pure in-memory append)
  C  fresh load(A) + feed delta     -> save   (the daemon's delta path)
  D  fresh one-shot prefill(all)    -> save   (cross-shape cold compute)

Comparisons, per tensor AND per KV-head block inside each tensor:
  B vs A  — does appending preserve the old bytes at all?
  C vs B  — is the load->append->save round trip byte-exact vs append?
  C vs A  — the chain question (P1) in isolation.
  D vs B  — does a differently-shaped prefill produce the same bytes (P3)?

Usage:
  uv run python scripts/dedup/roundtrip_probe.py [--model ID] \
      [--prefix-tokens 1500] [--delta-tokens 300]
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import mlx.core as mx

REPO_ROOT = Path(__file__).resolve().parents[2]
DOC = REPO_ROOT / "scripts" / "fixtures" / "t22" / "design_doc.md"
SYSTEM = "You are a coding agent working in a Rust repository. Answer precisely and concisely."


def save_cache(cache) -> bytes:
    from mlx_lm.models.cache import save_prompt_cache

    buf = io.BytesIO()
    save_prompt_cache(buf, cache)
    return buf.getvalue()


def load_cache(payload: bytes):
    from mlx_lm.models.cache import load_prompt_cache

    buf = io.BytesIO(payload)
    buf.name = "checkpoint.safetensors"  # mx infers safetensors from .name
    return load_prompt_cache(buf)


def feed(model, cache, tokens: list[int]):
    inp = mx.array(tokens)[None]
    logits = model(inp, cache=cache)
    mx.eval([c.state for c in cache], logits)
    return logits


def head_blocks(shape: list[int], nbytes: int) -> int:
    """Number of KV-head blocks along axis 1 (the smoke divergence sat at
    exactly one such boundary)."""
    return shape[1] if len(shape) == 4 else 1


def diagnose_heads(tname, a, b, d, e, ta, tb, td, te) -> None:
    """Where did A's second KV-head block GO? Zero check, tail check, and
    whether a pure re-save (E) agrees with the appended view (B)."""
    na = ta[tname]["end"] - ta[tname]["begin"]
    hsz = na // 2
    seg = {
        "A": a[ta[tname]["begin"] : ta[tname]["end"]],
        "B": b[tb[tname]["begin"] : tb[tname]["begin"] + na],
        "D": d[td[tname]["begin"] : td[tname]["begin"] + na],
        "E": e[te[tname]["begin"] : te[tname]["begin"] + na],
    }
    print(f"-- head diagnosis {tname}")
    for k, s in seg.items():
        h0, h1 = s[:hsz], s[hsz:]
        print(
            f"   {k}: head0 nonzero={sum(1 for x in h0[:4096] if x)}/4096 "
            f"head1 nonzero={sum(1 for x in h1[:4096] if x)}/4096"
        )
    b_full = b[tb[tname]["begin"] : tb[tname]["end"]]
    hsz_b = len(b_full) // 2
    a_h1 = seg["A"][hsz:]
    print(f"   A.head1 == B.head1 tail  : {a_h1 == b_full[-hsz:]}")
    print(f"   A.head1 == B.head1 [na:] : {a_h1 == b_full[hsz_b : hsz_b + hsz]}")
    print(f"   E.head1[:hsz] == B.head1[:hsz]: {seg['E'][hsz:] == b_full[hsz_b : hsz_b + hsz]}")


def compare(
    name: str,
    a: bytes,
    b: bytes,
    tensors_a: dict,
    tensors_b: dict,
    record: list[dict],
) -> None:
    print(f"-- {name}")
    exact = total = 0
    detail: list[dict] = []
    for tname in sorted(tensors_a)[:4]:
        ta, tb = tensors_a[tname], tensors_b[tname]
        na = ta["end"] - ta["begin"]
        seg_a = a[ta["begin"] : ta["end"]]
        seg_b = b[tb["begin"] : tb["begin"] + na]
        h = head_blocks(ta["shape"], na)
        hsz = na // h
        heads = []
        for i in range(h):
            s = i * hsz
            d = (
                next((j for j in range(s, s + hsz) if seg_a[j] != seg_b[j]), None)
                if seg_a[s : s + hsz] != seg_b[s : s + hsz]
                else None
            )
            total += 1
            exact += d is None
            heads.append("==" if d is None else f"diff@{d - s}")
        detail.append({"tensor": tname, "heads": heads})
        print(f"   {tname}: heads [{' '.join(heads)}]")
    if len(tensors_a) > 4:
        print(f"   ... {len(tensors_a) - 4} more tensors")
    record.append(
        {
            "comparison": name,
            "heads_exact": exact,
            "heads_total": total,
            "verdict": "exact" if exact == total else "partial",
            "detail": detail,
        }
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2-0.5B-Instruct")
    ap.add_argument("--prefix-tokens", type=int, default=1500)
    ap.add_argument("--delta-tokens", type=int, default=300)
    ap.add_argument("--doc-chars", type=int, default=19_000)
    ap.add_argument("--json", type=Path, help="write the per-head comparison table here")
    args = ap.parse_args()

    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    model, tokenizer = load(args.model)
    record: list[dict] = []
    doc = DOC.read_text(encoding="utf-8")[: args.doc_chars]
    prompt = f"{SYSTEM}\n\nHere is the document:\n\n{doc}"
    tokens = tokenizer.encode(prompt)[: args.prefix_tokens + args.delta_tokens]
    p, d = args.prefix_tokens, args.delta_tokens
    print(f"model={args.model} prefix={p} delta={d} (token stream: {len(tokens)} available)")

    # A: fresh prefill of the prefix
    cache = make_prompt_cache(model)
    feed(model, cache, tokens[:p])
    payload_a = save_cache(cache)

    # B: SAME cache object, append the delta
    feed(model, cache, tokens[p : p + d])
    payload_b = save_cache(cache)

    # C: fresh process-state: load A, append the same delta (daemon path)
    cache_c = load_cache(payload_a)
    feed(model, cache_c, tokens[p : p + d])
    payload_c = save_cache(cache_c)

    # D: one-shot cold prefill of the full stream (different batch shape)
    cache_d = make_prompt_cache(model)
    feed(model, cache_d, tokens[: p + d])
    payload_d = save_cache(cache_d)

    # E: load A and save it right back — pure round trip, no compute
    payload_e = save_cache(load_cache(payload_a))

    from blob_forensics import parse_safetensors

    ta, _ = parse_safetensors(payload_a)
    tb, _ = parse_safetensors(payload_b)
    tc, _ = parse_safetensors(payload_c)
    td, _ = parse_safetensors(payload_d)
    te, _ = parse_safetensors(payload_e)

    compare("B vs A (pure in-memory append)", payload_a, payload_b, ta, tb, record)
    compare("C vs B (load round trip vs append)", payload_b, payload_c, tb, tc, record)
    compare("C vs A (daemon delta path, P1)", payload_a, payload_c, ta, tc, record)
    compare("D vs B (one-shot vs appended, P3)", payload_b, payload_d, tb, td, record)
    compare("E vs A (pure re-save, no compute)", payload_a, payload_e, ta, te, record)
    compare("E vs B head-prefix (re-save vs appended)", payload_b, payload_e, tb, te, record)
    diagnose_heads("0.0", payload_a, payload_b, payload_d, payload_e, ta, tb, td, te)

    sizes = {
        k: len(v)
        for k, v in (("A", payload_a), ("B", payload_b), ("C", payload_c), ("D", payload_d))
    }
    print(f"payload sizes: {sizes}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "model": args.model,
                    "prefix_tokens": p,
                    "delta_tokens": d,
                    "comparisons": record,
                    "payload_bytes": sizes,
                },
                indent=1,
            )
        )
        print(f"json -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
