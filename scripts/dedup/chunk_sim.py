#!/usr/bin/env python3
"""E3 — chunking simulation over the captured chain blobs.

Two chunkings, measured side by side on the SAME captured bytes:

  CDC (gear)  — content-defined chunks via a vectorized gear hash. The gear
    recurrence's low m bits depend only on the last m bytes, so the cut
    condition (h & ((1<<m)-1)) == 0 over every eligible position is a
    sliding-window sum — vectorizable, and verified bit-for-bit against a
    naive sequential implementation on a sample (default on; --no-verify to
    skip). Envelope:
    min = avg/4, max = 4*avg (FastCDC-style).
  token-block — per-tensor slices of B covered positions, keyed by content
    digest. Exact by construction because we KNOW the token boundaries (the
    thing the llama.cpp FUSE author did not); it needs an adapter-side
    relayout to become a physical codec, which CDC does not.

Scopes: 'file' chunks the whole payload as one stream; 'tensor' chunks each
safetensors tensor independently (cuts never straddle layer seams).

Reported per config: per-turn NEW bytes vs the cumulative store (the honest
store view — chunks from earlier turns stay stored), chunk size percentiles,
and CDC boundary stability across turns.

Usage (defaults reproduce the committed tables):
  uv run python scripts/dedup/chunk_sim.py <grow_run_dir> \
      [--avg-kib 16,64,256] [--token-blocks 64,256] \
      [--scopes head,tensor,file] [--json PATH] [--no-verify]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from blob_forensics import kv_heads, load_blob, parse_safetensors, turn_blobs

_MASK64 = (1 << 64) - 1


def gear_table() -> np.ndarray:
    """256 fixed 64-bit gears (splitmix64 stream, fixed state): deterministic
    across runs and processes."""
    state = 0x9E3779B97F4A7C15
    out = []
    for _ in range(256):
        state = (state + 0x9E3779B97F4A7C15) & _MASK64
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
        out.append((z ^ (z >> 31)) & _MASK64)
    return np.array(out, dtype=np.uint64)


_TABLE = gear_table()

# Byte-block size for the windowed gear gather: bounds peak temporaries to
# ~3x this value regardless of payload size. Tests shrink it to force the
# multi-block path.
_GATHER_BLOCK = 1 << 24


def mask_bits_for(avg: int) -> int:
    """log2(avg): a cut probability of 1/avg per byte. Single definition —
    the verifier must exercise the same mask the chunker uses."""
    return max(avg.bit_length() - 1, 8)


def cdc_envelope(avg: int) -> tuple[int, int]:
    """FastCDC-style min/max chunk sizes. Single definition — boundary
    stability must be measured over the exact envelope the store simulation
    chunks with."""
    return avg // 4, avg * 4


def gear_cut_candidates(arr: np.ndarray, table: np.ndarray, mask_bits: int) -> np.ndarray:
    """Candidate chunk-end offsets where the gear hash's low bits are zero.

    The window at row r covers bytes [r, r+mask_bits); its cut would end the
    chunk at r+mask_bits. Eligible positions always have full history within
    the chunk because min_size >> mask_bits.

    Processed in bounded blocks (with a mask_bits-1 carry between them) so a
    426 MB payload does not materialize ~24x itself in uint64 temporaries;
    the unblocked math is identical. Output is identical to computing all
    windows at once."""
    m = mask_bits
    if arr.size < m:
        return np.empty(0, dtype=np.int64)
    g = table & np.uint64((1 << m) - 1)
    block = _GATHER_BLOCK
    out: list[np.ndarray] = []
    start = 0
    while start + m <= arr.size:
        stop = min(start + block, arr.size)
        # ensure every window starting in this block is complete within it
        win_end = min(stop, arr.size)
        seg = arr[start:win_end]
        if seg.size < m:
            break
        w = np.lib.stride_tricks.sliding_window_view(seg, m)
        acc = np.zeros(w.shape[0], dtype=np.uint64)
        for j in range(m):
            acc += g[w[:, j]] << np.uint64(m - 1 - j)
        acc &= np.uint64((1 << m) - 1)  # the cut condition is on the LOW m bits
        rows = np.nonzero(acc == 0)[0]
        out.append(rows.astype(np.int64) + start + m)
        if win_end >= arr.size:
            break
        start = win_end - (m - 1)
    if not out:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(out)


def apply_envelope(candidates: np.ndarray, n: int, min_size: int, max_size: int) -> list[int]:
    """Chunk ends: natural cuts spaced [min, max]; forced cuts at max."""
    cands = candidates.tolist()
    out: list[int] = []
    last = 0
    idx = 0
    while last < n:
        lo = last + min_size
        while idx < len(cands) and cands[idx] < lo:
            idx += 1
        if idx < len(cands) and cands[idx] <= last + max_size:
            end = cands[idx]
            idx += 1
        else:
            end = min(last + max_size, n)
        if end >= n:
            break
        out.append(end)
        last = end
    return out


def naive_gear_cdc(data: bytes, mask_bits: int, min_size: int, max_size: int) -> list[int]:
    """Sequential reference (the real algorithm); the vectorized path must
    produce identical cuts. Kept obviously-correct on purpose."""
    mask = (1 << mask_bits) - 1
    table = _TABLE.tolist()
    cuts: list[int] = []
    last = 0
    h = 0
    n = len(data)
    for i, b in enumerate(data):
        h = ((h << 1) + table[b]) & _MASK64
        end = i + 1
        if (
            end - last >= min_size
            and (h & mask) == 0
            and end < n
            or end - last >= max_size
            and end < n
        ):
            cuts.append(end)
            last = end
            h = 0
    return cuts


def verify_vectorized(sample: bytes, mask_bits: int, min_size: int, max_size: int) -> None:
    arr = np.frombuffer(sample, dtype=np.uint8)
    got = apply_envelope(
        gear_cut_candidates(arr, _TABLE, mask_bits), len(sample), min_size, max_size
    )
    want = naive_gear_cdc(sample, mask_bits, min_size, max_size)
    if got != want:
        n = min(len(got), len(want))
        at = next((i for i in range(n) if got[i] != want[i]), n)
        raise SystemExit(
            f"CDC verifier MISMATCH (mask_bits={mask_bits}): "
            f"{len(got)} vs {len(want)} cuts, first at index {at}"
        )


def chunk_stream(
    data: bytes, avg: int, min_size: int, max_size: int
) -> tuple[list[tuple[int, int, str]], list[int]]:
    """(chunks, cuts): per chunk (end_offset, size, sha256) relative to
    `data`, and the raw cut list (for boundary-stability)."""
    arr = np.frombuffer(data, dtype=np.uint8)
    cuts = apply_envelope(
        gear_cut_candidates(arr, _TABLE, mask_bits_for(avg)), len(data), min_size, max_size
    )
    out = []
    start = 0
    for end in [*cuts, len(data)]:
        chunk = data[start:end]
        out.append((end, end - start, hashlib.sha256(chunk).hexdigest()))
        start = end
    return out, cuts  # type: ignore[return-value]


def blob_chunk_sets(
    payload: bytes, tensors: dict, scope: str, avg: int
) -> tuple[list[tuple[int, int, str]], dict[str, list[int]]]:
    """Chunks over the payload. Scopes:
    file   — one stream over the whole payload (what a FUSE-style whole-file
             chunker sees; every head boundary shift re-cuts downstream).
    tensor — per safetensors tensor (one internal shift boundary each).
    head   — per KV-head block (axis-1 slices): the layout-stable unit the
             forensics probe identified; cuts never cross a head boundary
             and appends never shift later heads' bytes.
    Returns (chunks with payload-absolute offsets, cut map for stability)."""
    min_size, max_size = cdc_envelope(avg)
    chunks: list[tuple[int, int, str]] = []
    cuts_by: dict[str, list[int]] = {}
    if scope == "file":
        cs, cuts = chunk_stream(payload, avg, min_size, max_size)
        chunks.extend(cs)
        cuts_by["<file>"] = cuts
    elif scope == "tensor":
        for name in sorted(tensors):
            t = tensors[name]
            seg = payload[t["begin"] : t["end"]]
            cs, cuts = chunk_stream(seg, avg, min_size, max_size)
            chunks.extend((t["begin"] + e, s, d) for e, s, d in cs)
            cuts_by[name] = cuts
    else:
        for name in sorted(tensors):
            t = tensors[name]
            n_heads = kv_heads(t)
            hsz = (t["end"] - t["begin"]) // n_heads
            for h in range(n_heads):
                seg = payload[t["begin"] + h * hsz : t["begin"] + (h + 1) * hsz]
                cs, cuts = chunk_stream(seg, avg, min_size, max_size)
                chunks.extend((t["begin"] + h * hsz + e, s, d) for e, s, d in cs)
                cuts_by[f"{name}#{h}"] = [h * hsz + c for c in cuts]
    return chunks, cuts_by


def token_block_chunks(payload: bytes, tensors: dict, positions: int, block: int):
    """Blocks of `block` covered positions. A block's bytes are NOT
    contiguous in the layout (each KV head holds its own slice for those
    positions), so the block's identity is the digest over the
    concatenation of its per-head slices — exact by construction, and the
    layout a future codec would re-materialize at save time."""
    chunks: list[tuple[int, int, str]] = []
    for name in sorted(tensors):
        t = tensors[name]
        nbytes = t["end"] - t["begin"]
        n_heads = kv_heads(t)
        hsz = nbytes // n_heads
        unit, rem = divmod(hsz, positions)
        if rem:
            raise SystemExit(f"tensor {name}: {hsz} not divisible by {positions} positions")
        for b in range(0, positions, block):
            n = min(block, positions - b)
            parts = []
            for h in range(n_heads):
                base = t["begin"] + h * hsz
                parts.append(payload[base + b * unit : base + (b + n) * unit])
            digest = hashlib.sha256(b"".join(parts)).hexdigest()
            chunks.append((t["begin"] + n_heads * (b + n) * unit, n * nbytes // positions, digest))
    return chunks


def pct(values: list[int], p: float) -> int:
    if not values:
        return 0
    s = sorted(values)
    return s[min(len(s) - 1, int(len(s) * p))]


def simulate(turns: list[Path], cfg_name: str, chunker) -> dict:
    """Per-turn new bytes against the cumulative store digest set."""
    store: set[str] = set()
    rows = []
    for turn, path in turns:
        meta, payload = load_blob(path)
        chunks = chunker(payload, meta)
        new_bytes = sum(size for _, size, digest in chunks if digest not in store)
        sizes = [size for _, size, _ in chunks]
        rows.append(
            {
                "turn": turn,
                "tokens": meta["token_count"],
                "payload_bytes": len(payload),
                "chunks": len(chunks),
                "new_bytes": new_bytes,
                "new_frac": round(new_bytes / len(payload), 4),
                "chunk_p50": pct(sizes, 0.5),
                "chunk_p95": pct(sizes, 0.95),
            }
        )
        store.update(d for _, _, d in chunks)
    if len(rows) < 2:
        raise SystemExit(
            "simulate: need >= 2 turns for a new-bytes mean "
            f"(got {len(rows)} for config {cfg_name})"
        )
    mean_new = sum(r["new_bytes"] for r in rows[1:]) / sum(r["payload_bytes"] for r in rows[1:])
    return {
        "config": cfg_name,
        "turns": rows,
        "mean_new_frac_turns2plus": round(mean_new, 4),
        "store_unique_bytes": sum(r["new_bytes"] for r in rows),
    }


def boundary_stability(turns: list[Path], avg: int) -> float:
    """Fraction of turn k's CDC cuts (per KV-HEAD block, within the shared
    byte range) that reappear identically in turn k+1."""
    min_size, max_size = cdc_envelope(avg)
    total, shared_total = 0, 0
    prev: dict[str, tuple[int, list[int]]] | None = None
    for _, path in turns:
        meta, payload = load_blob(path)
        tensors, _ = parse_safetensors(payload)
        cur: dict[str, tuple[int, list[int]]] = {}
        for name in sorted(tensors):
            t = tensors[name]
            n_heads = kv_heads(t)
            hsz = (t["end"] - t["begin"]) // n_heads
            for h in range(n_heads):
                seg = payload[t["begin"] + h * hsz : t["begin"] + (h + 1) * hsz]
                _, cuts = chunk_stream(seg, avg, min_size, max_size)
                cur[f"{name}#{h}"] = (len(seg), cuts)
        if prev is not None:
            for key, (plen, pcuts) in prev.items():
                clen, ccuts = cur[key]
                shared = min(plen, clen)
                cs = {c for c in ccuts if c < shared}
                total += len(pcuts)
                shared_total += sum(1 for c in pcuts if c in cs)
        prev = cur
    return round(shared_total / total, 4) if total else 1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("grow_dir", type=Path)
    ap.add_argument("--avg-kib", default="16,64,256")
    ap.add_argument("--token-blocks", default="64,256")
    ap.add_argument("--scopes", default="head,tensor,file")
    ap.add_argument("--json", type=Path)
    # The vectorized gear hash shipped a real boundary bug once; verification
    # is the default so a default invocation cannot produce a table from a
    # chunker that disagrees with the naive reference.
    ap.add_argument("--no-verify", action="store_true", help="skip the naive-reference cross-check")
    args = ap.parse_args()

    turns = turn_blobs(args.grow_dir)
    if not args.no_verify:
        for kib in (int(x) for x in args.avg_kib.split(",")):
            avg = kib * 1024
            # the sample must hold enough natural cuts to make the equality
            # meaningful: at avg=256 KiB a fixed 2 MiB sample has a handful,
            # and at 1024 KiB nearly none, so scale with the target size
            sample = load_blob(turns[0][1])[1][: max(2 << 20, 32 * avg)]
            verify_vectorized(sample, mask_bits_for(avg), *cdc_envelope(avg))
        print(
            "CDC verifier: vectorized == naive "
            f"({', '.join(x.strip() for x in args.avg_kib.split(','))} KiB configs)"
        )

    results = []
    for scope in args.scopes.split(","):
        for kib in (int(x) for x in args.avg_kib.split(",")):
            avg = kib * 1024

            def chunker(payload: bytes, meta: dict, _avg=avg, _scope=scope):
                tensors, _ = parse_safetensors(payload)
                return blob_chunk_sets(payload, tensors, _scope, _avg)[0]

            r = simulate(turns, f"cdc/{scope}/{kib}KiB", chunker)
            r["boundary_stability"] = boundary_stability(turns, avg) if scope == "head" else None
            results.append(r)
            tail = ", ".join(f"{x['new_frac']:.3f}" for x in r["turns"][1:])
            print(
                f"{r['config']:<22} new/turn t2+: {tail}  mean={r['mean_new_frac_turns2plus']:.4f}"
            )
    for tb in (int(x) for x in args.token_blocks.split(",")):

        def tb_chunker(payload: bytes, meta: dict, _tb=tb):
            tensors, _ = parse_safetensors(payload)
            return token_block_chunks(payload, tensors, meta["token_count"] - 1, _tb)

        r = simulate(turns, f"token-block/{tb}", tb_chunker)
        results.append(r)
        tail = ", ".join(f"{x['new_frac']:.3f}" for x in r["turns"][1:])
        print(f"{r['config']:<22} new/turn t2+: {tail}  mean={r['mean_new_frac_turns2plus']:.4f}")

    for r in results[:1]:
        print(f"\n{r['config']} detail:")
        for x in r["turns"]:
            print(
                f"  t{x['turn']}: tokens={x['tokens']} payload={x['payload_bytes']:,} "
                f"chunks={x['chunks']} new={x['new_bytes']:,} ({x['new_frac']:.4f}) "
                f"p50={x['chunk_p50']} p95={x['chunk_p95']}"
            )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=1))
        print(f"json -> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
