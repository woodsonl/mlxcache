#!/usr/bin/env python3
"""Byte inventory of the NVFP4 checkpoint: what must be resident to run.

Streams every shard header (no tensor data) and aggregates stored bytes by
layer / module family / dtype, so the memory plan is arithmetic, not guesswork.
"""

from __future__ import annotations

import collections
import json
import os
import struct
import sys

DTYPE_BYTES = {
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "U8": 1,
    "I8": 1,
    "I32": 4,
    "I64": 8,
    "U32": 4,
    "U16": 2,
    "I16": 2,
    "BOOL": 1,
}

ROOT = (
    sys.argv[1] if len(sys.argv) > 1 else "/Users/lance/models/Qwen3.8-Flash-Next-NVFP4-mlxstream"
)


def iter_header(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n))


def nbytes(dtype, shape):
    per = DTYPE_BYTES[dtype]
    total = 1
    for d in shape:
        total *= int(d)
    # FP4 packs two values per byte.
    if dtype in ("U8",) and per == 1:
        return total
    return total * per


def family(key: str) -> str:
    k = key.replace("model.language_model.", "")
    parts = k.split(".")
    if parts[0] == "layers":
        layer = int(parts[1])
        rest = ".".join(parts[2:])
        if rest.startswith("mlp.experts."):
            return f"layer{layer}:experts:{parts[4] if len(parts) > 4 else '?'}"
        if rest.startswith("mlp.switch_mlp"):
            return f"layer{layer}:experts:switch_mlp"
        if rest.startswith("mlp."):
            return f"layer{layer}:shared_expert"
        if rest.startswith("ple."):
            return f"layer{layer}:ple"
        if rest.startswith("linear_attn.") or rest.startswith("self_attn."):
            head = rest.split(".")[0]
            return f"layer{layer}:{head}"
        if rest.startswith("mtp") or ".mtp" in rest:
            return "mtp"
        return f"layer{layer}:other"
    if parts[0] == "mtp":
        return "mtp"
    return parts[0]


def module_family(key: str) -> str:
    k = key
    for pat in (
        "linear_attn",
        "self_attn",
        "ple_embedding",
        "switch_mlp",
        "experts",
        "shared_expert",
        "mtp",
        "embed_tokens",
        "lm_head",
        "norm",
        "indexer",
    ):
        if pat in k:
            return pat
    return "other"


def main() -> int:
    with open(os.path.join(ROOT, "model.safetensors.index.json"), encoding="utf-8") as fh:
        index = json.load(fh)
    files = sorted(set(index["weight_map"].values()))
    by_family = collections.Counter()
    by_mod = collections.Counter()
    by_dtype = collections.Counter()
    counts = collections.Counter()
    layers_seen = set()
    total = 0
    for name in files:
        hdr = iter_header(os.path.join(ROOT, name))
        for key, info in hdr.items():
            if key == "__metadata__":
                continue
            b = nbytes(info["dtype"], info["shape"])
            total += b
            by_family[family(key)] += b
            by_mod[module_family(key)] += b
            by_dtype[info["dtype"]] += b
            counts[module_family(key)] += 1
            parts = key.split(".")
            if len(parts) > 3 and parts[2] == "layers":
                layers_seen.add(int(parts[3]))
        print(f"scanned {name}: {len(hdr) - 1} tensors", file=sys.stderr)

    # global scales file
    gs = os.path.join(ROOT, "nvfp4_global_scales.safetensors")
    if os.path.exists(gs):
        hdr = iter_header(gs)
        for key, info in hdr.items():
            if key == "__metadata__":
                continue
            b = nbytes(info["dtype"], info["shape"])
            total += b
            by_mod["global_scales"] += b
            by_dtype[info["dtype"]] += b
            counts["global_scales"] += 1

    print(f"ROOT {ROOT}")
    print(f"layers seen: {len(layers_seen)} (min {min(layers_seen)}, max {max(layers_seen)})")
    print(f"TOTAL stored: {total / 1e9:.1f} GB ({total} bytes)")
    print("\n== by dtype ==")
    for d, b in by_dtype.most_common():
        print(f"  {d:10s} {b / 1e9:8.2f} GB")
    print("\n== by module family ==")
    for m, b in by_mod.most_common():
        print(f"  {m:20s} {b / 1e9:8.2f} GB  ({counts[m]} tensors)")
    print("\n== by layer family (top 20) ==")
    for f, b in by_family.most_common(20):
        print(f"  {f:32s} {b / 1e9:8.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
