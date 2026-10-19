#!/usr/bin/env python3
"""E1/E2 — blob layout facts and the byte-identity probes.

E1 decodes the safetensors layout of real blobs: per-tensor table, bytes per
covered position, header/payload split. The blob framing is the daemon's
(u32 LE header_len + JSON header + payload) and the payload is a safetensors
file (u64 LE header_len + JSON + data; offsets relative to the data start).

E2 measures byte identity where every chunk-dedup claim hinges on it:
  P1 chain-by-construction — turn k+1's blob vs turn k's blob, per tensor.
     The delta prefill loaded turn k's cache and appended, so per tensor the
     expectation is EXACT leading equality over A's whole range; the first
     divergence should sit at A's end. Anything earlier is drift.
  P2 cold-twice — two independent single-shot runs (fresh processes, fresh
     stores) of the same full conversation: same-shape recompute identity.
  P3 cross-shape — the chain-built final blob vs the single-shot blob of the
     same total prompt: identical bytes computed under different prefill
     batch shapes. Reported as first divergence plus 4 KiB block containment
     (identical content anywhere dedups, so containment is the number that
     matters even when the first byte diverges).

Usage:
  uv run python scripts/dedup/blob_forensics.py <grow_run_dir> \
      [--single DIR] [--single2 DIR] [--json PATH] [--block-kib 4]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path


def load_blob(path: Path) -> tuple[dict, bytes]:
    raw = path.read_bytes()
    (hlen,) = struct.unpack("<I", raw[:4])
    meta = json.loads(raw[4 : 4 + hlen])
    return meta, raw[4 + hlen :]


def parse_safetensors(payload: bytes) -> tuple[dict, int]:
    (shlen,) = struct.unpack("<Q", payload[:8])
    header = json.loads(payload[8 : 8 + shlen])
    data_start = 8 + shlen
    tensors = {
        name: {
            "dtype": spec["dtype"],
            "shape": spec["shape"],
            "begin": data_start + spec["data_offsets"][0],
            "end": data_start + spec["data_offsets"][1],
        }
        for name, spec in header.items()
        if name != "__metadata__"
    }
    return tensors, data_start


def layout(meta: dict, tensors: dict, payload_len: int) -> dict:
    positions = meta["token_count"] - 1  # a blob covers tokens[:-1] (§3.2)
    rows = []
    for name in sorted(tensors):
        t = tensors[name]
        nbytes = t["end"] - t["begin"]
        unit, rem = divmod(nbytes, positions)
        rows.append(
            {
                "tensor": name,
                "dtype": t["dtype"],
                "shape": t["shape"],
                "bytes": nbytes,
                "bytes_per_position": unit,
                "unit_remainder": rem,
            }
        )
    return {
        "token_count": meta["token_count"],
        "positions": positions,
        "payload_bytes": payload_len,
        "framing_bytes": 4 + len(json.dumps(meta)),
        "kv_bits": meta["fingerprint"].get("kv_bits", 0),
        "tensors": rows,
    }


def first_diff(a: bytes, b: bytes) -> int | None:
    """First differing offset within min(len) — None when one is a prefix of
    the other (or equal). Block-scanned, then bisected."""
    n = min(len(a), len(b))
    if a[:n] == b[:n]:
        return None
    blk = 1 << 20
    lo = 0
    for start in range(0, n, blk):
        if a[start : start + blk] != b[start : start + blk]:
            lo = start
            break
    else:
        return n
    # bisect within [lo, lo+blk]
    bad, good = lo + blk, lo
    while bad - good > 1:
        mid = (good + bad) // 2
        if a[good:mid] == b[good:mid]:
            good = mid
        else:
            bad = mid
    return good


def block_digests(data: bytes, block: int) -> list[str]:
    return [hashlib.sha256(data[i : i + block]).hexdigest() for i in range(0, len(data), block)]


def block_containment(a: bytes, b: bytes, block: int) -> tuple[float, float]:
    """(fraction of A's blocks whose digest appears in B, the reverse)."""
    da, db = block_digests(a, block), set(block_digests(b, block))
    sa = set(da)
    return (
        sum(1 for d in da if d in db) / len(da) if da else 1.0,
        sum(1 for d in db if d in sa) / len(db) if db else 1.0,
    )


def turn_blobs(run_dir: Path) -> list[tuple[int, Path]]:
    """The one blob each grow turn added, in turn order."""
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if manifest["mode"] != "grow":
        raise SystemExit(f"{run_dir}: not a grow run")
    out = []
    seen: set[str] = set()
    for t in manifest["turns"]:
        names = {b["name"] for b in t["blobs"]}
        new = sorted(names - seen)
        if len(new) != 1:
            raise SystemExit(f"turn {t['turn']}: expected exactly 1 new blob, got {new}")
        seen |= names
        out.append((t["turn"], run_dir / t["snapshot"] / new[0]))
    return out


def single_blob(run_dir: Path) -> Path:
    blobs = sorted((run_dir / "snapshots" / "single").glob("*.ckpt"))
    if len(blobs) != 1:
        raise SystemExit(f"{run_dir}: expected exactly 1 single-run blob, got {blobs}")
    return blobs[0]


def compare_pair(label: str, a_path: Path, b_path: Path, block: int) -> dict:
    """Per-tensor identity of B against A (A is the shorter/earlier blob).

    TWO views, because the safetensors layout is per-KV-head contiguous
    ([head0 all positions][head1 all positions] ...): when the sequence
    grows, every head block grows and the FOLLOWING heads shift — a flat
    leading-bytes comparison misreads that shift as rewrite. The per-head
    view compares each head's own leading bytes at its own offset; the flat
    view is kept because it is what a whole-file chunker would actually
    see."""
    meta_a, payload_a = load_blob(a_path)
    meta_b, payload_b = load_blob(b_path)
    ta, _ = parse_safetensors(payload_a)
    tb, _ = parse_safetensors(payload_b)
    if sorted(ta) != sorted(tb):
        return {"label": label, "error": "tensor sets differ", "a": list(ta), "b": list(tb)}
    rows = []
    for name in sorted(ta):
        a_bytes = payload_a[ta[name]["begin"] : ta[name]["end"]]
        n_heads = ta[name]["shape"][1] if len(ta[name]["shape"]) == 4 else 1
        hsz_a = len(a_bytes) // n_heads
        hsz_b = (tb[name]["end"] - tb[name]["begin"]) // n_heads
        unit = hsz_a // (meta_a["token_count"] - 1)
        b_base = tb[name]["begin"]
        flat = first_diff(a_bytes, payload_b[b_base : b_base + len(a_bytes)])
        heads = []
        for h in range(n_heads):
            a_head = a_bytes[h * hsz_a : (h + 1) * hsz_a]
            b_head = payload_b[b_base + h * hsz_b : b_base + h * hsz_b + hsz_a]
            d = first_diff(a_head, b_head)
            heads.append(
                {
                    "head": h,
                    "leading_equal": d is None,
                    "first_diff_byte": d,
                    "first_diff_position": None if d is None else d // max(unit, 1),
                }
            )
        rows.append(
            {
                "tensor": name,
                "a_bytes": len(a_bytes),
                "n_heads": n_heads,
                "heads_equal": sum(1 for x in heads if x["leading_equal"]),
                "heads": heads,
                "flat_leading_equal": flat is None,
                "flat_first_diff_byte": flat,
                # positions covered by A within ONE head block
                "a_end_position": hsz_a // max(unit, 1),
            }
        )
    whole_a2b, whole_b2a = block_containment(payload_a, payload_b, block)
    return {
        "label": label,
        "a": {"path": str(a_path), "tokens": meta_a["token_count"]},
        "b": {"path": str(b_path), "tokens": meta_b["token_count"]},
        "tensors": rows,
        "block_containment_a_in_b": round(whole_a2b, 6),
        "block_containment_b_in_a": round(whole_b2a, 6),
        "block_bytes": block,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("grow_dir", type=Path)
    ap.add_argument("--single", type=Path, help="single-shot run dir (P3)")
    ap.add_argument("--single2", type=Path, help="second single-shot run dir (P2)")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--block-kib", type=int, default=4)
    args = ap.parse_args()
    block = args.block_kib * 1024

    report: dict = {"block_bytes": block}

    # E1: layout of every turn blob.
    turns = turn_blobs(args.grow_dir)
    lay = []
    for turn, path in turns:
        meta, payload = load_blob(path)
        tensors, _ = parse_safetensors(payload)
        lay.append({"turn": turn, **layout(meta, tensors, len(payload))})
    report["layout"] = lay

    last = lay[-1]
    print(f"== E1 layout (turn {last['turn']}, kv_bits={last['kv_bits']})")
    print(
        f"tokens={last['token_count']} positions={last['positions']} "
        f"payload={last['payload_bytes']:,}B tensors={len(last['tensors'])}"
    )
    units = {t["bytes_per_position"] for t in last["tensors"]}
    rems = {t["unit_remainder"] for t in last["tensors"]}
    print(f"bytes/position per tensor: min={min(units)} max={max(units)} remainders={rems}")
    for t in last["tensors"][:3]:
        print(
            f"  {t['tensor']}: {t['dtype']} shape={t['shape']} "
            f"{t['bytes']:,}B ({t['bytes_per_position']} B/pos)"
        )
    if len(last["tensors"]) > 3:
        print(f"  ... {len(last['tensors']) - 3} more")

    # P1: consecutive chain pairs.
    p1 = []
    print("\n== P1 chain-by-construction (per HEAD: does each KV head keep its bytes?)")
    for (ta_i, pa), (tb_i, pb) in zip(turns, turns[1:], strict=False):
        r = compare_pair(f"P1 t{ta_i}->t{tb_i}", pa, pb, block)
        p1.append(r)
        heads_ok = sum(x["heads_equal"] for x in r["tensors"])
        heads_all = sum(x["n_heads"] for x in r["tensors"])
        print(
            f"t{ta_i}->t{tb_i}: heads exact {heads_ok}/{heads_all} "
            f"flat containment {r['block_containment_a_in_b']:.4f} "
            f"(the shift a whole-file chunker would pay)"
        )
    report["p1_chain"] = p1

    # P3 / P2 need the single runs.
    if args.single:
        grow_final = turns[-1][1]
        p3 = compare_pair("P3 grow-final vs single", grow_final, single_blob(args.single), block)
        report["p3_cross_shape"] = p3
        print("\n== P3 cross-shape (grow final vs single-shot)")
        heads_ok = sum(x["heads_equal"] for x in p3["tensors"])
        heads_all = sum(x["n_heads"] for x in p3["tensors"])
        print("\n== P3 cross-shape (grow final vs single-shot)")
        print(
            f"heads exact-lead {heads_ok}/{heads_all} "
            f"flat containment grow-in-single={p3['block_containment_a_in_b']:.4f} "
            f"single-in-grow={p3['block_containment_b_in_a']:.4f}"
        )
        for x in p3["tensors"][:2]:
            for h in x["heads"]:
                print(
                    f"  {x['tensor']} h{h['head']}: first_diff_pos="
                    f"{h['first_diff_position']} A-positions={x['a_end_position']}"
                )
    if args.single and args.single2:
        p2 = compare_pair(
            "P2 single vs single2", single_blob(args.single), single_blob(args.single2), block
        )
        report["p2_cold_twice"] = p2
        heads_ok = sum(x["heads_equal"] for x in p2["tensors"])
        heads_all = sum(x["n_heads"] for x in p2["tensors"])
        print("\n== P2 cold-twice (two independent single-shot runs)")
        print(
            f"heads exact-lead {heads_ok}/{heads_all} "
            f"flat containment={p2['block_containment_a_in_b']:.4f}"
        )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=1))
        print(f"\njson -> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
