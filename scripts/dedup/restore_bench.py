#!/usr/bin/env python3
"""E4 — restore-path prototype cost: whole-file read vs chunk-manifest read.

Materializes the chosen chunking of ONE real blob as actual chunk files on
disk (content-addressed), then times, page-cache warm for both sides:

  plain   — read the blob file + sha256 the payload (today's fetch path:
            whole-file read; the daemon verifies the payload digest).
  chunked — manifest-driven parallel read (N workers), per-chunk sha256,
            assembly into a preallocated buffer at each chunk's absolute
            payload offset. Non-tensor spans (the safetensors JSON header;
            any alignment gaps) are carried INLINE in the manifest — a few
            KB — exactly as a real codec would.

Assembled output is verified byte-equal to the payload before timing. The
restore-relevant number is the delta between the medians.

Usage:
  uv run python scripts/dedup/restore_bench.py <blob.ckpt> \
      [--avg-kib 16] [--workers 8] [--reps 10] [--pack-mib 64] \
      --scratch DIR
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path

from blob_forensics import load_blob, parse_safetensors
from chunk_sim import blob_chunk_sets


def materialize(
    payload: bytes,
    tensors: dict,
    avg: int,
    scratch: Path,
    pack_mib: int = 0,
):
    """Write chunks once; return the manifest: chunk items with absolute
    payload offsets, plus the inline (non-chunked) spans. pack_mib > 0 packs
    chunks sequentially into segment files (restic-style; the layout a real
    codec needs — one file per 19 KB chunk is syscall-bound on restore).
    Scope is fixed at head: tensor/file chunking failed G1, so their restore
    cost is not a number anyone needs."""
    chunks, _ = blob_chunk_sets(payload, tensors, "head", avg)
    chunk_dir = scratch / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    covered = bytearray(len(payload))
    items = []
    seg_size = pack_mib * (1 << 20) if pack_mib else 0
    seg_idx = -1
    seg_off = 0
    seg_fh = None
    with ExitStack() as stack:
        for end, size, digest in sorted(chunks):
            start = end - size
            data = payload[start:end]
            if seg_size:
                if seg_fh is None or seg_off + size > seg_size:
                    if seg_fh is not None:
                        seg_fh.close()
                    seg_idx += 1
                    seg_off = 0
                    seg_fh = stack.enter_context(open(chunk_dir / f"seg-{seg_idx:04d}", "wb"))
                seg_fh.write(data)
                loc = {
                    "seg": seg_idx,
                    "seg_off": seg_off,
                    "size": size,
                    "offset": start,
                    "digest": digest,
                }
                seg_off += size
            else:
                (chunk_dir / digest).write_bytes(data)
                loc = {"digest": digest, "size": size, "offset": start}
            items.append(loc)
            covered[start:end] = b"\x01" * size
    if seg_fh is not None:
        seg_fh.close()
    inline = []
    i = 0
    n = len(covered)
    while i < n:
        if covered[i]:
            i += 1
            continue
        j = i
        while j < n and not covered[j]:
            j += 1
        inline.append({"offset": i, "size": j - i, "data": payload[i:j]})
        i = j
    total = sum(it["size"] for it in items) + sum(s["size"] for s in inline)
    if total != len(payload):
        raise SystemExit(f"coverage {total} != payload {len(payload)}")
    return {
        "items": items,
        "inline": inline,
        "packed": bool(seg_size),
        "chunk_bytes": sum(it["size"] for it in items),
        "inline_bytes": sum(s["size"] for s in inline),
    }


def bench_plain(blob_path: Path, payload_len: int) -> float:
    t0 = time.perf_counter()
    raw = blob_path.read_bytes()
    payload = raw[len(raw) - payload_len :]
    hashlib.sha256(payload).hexdigest()
    return time.perf_counter() - t0


def bench_chunked(manifest: dict, scratch: Path, workers: int, out: bytearray) -> float:
    t0 = time.perf_counter()
    for span in manifest["inline"]:
        out[span["offset"] : span["offset"] + span["size"]] = span["data"]
    chunk_dir = scratch / "chunks"
    if manifest["packed"]:
        fhs = {
            i: os.open(str(chunk_dir / f"seg-{i:04d}"), os.O_RDONLY)
            for i in range(max(it["seg"] for it in manifest["items"]) + 1)
        }

        def read_one(item: dict) -> None:
            data = os.pread(fhs[item["seg"]], item["size"], item["seg_off"])
            hashlib.sha256(data).hexdigest()
            out[item["offset"] : item["offset"] + item["size"]] = data

        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(read_one, manifest["items"]))
        for fd in fhs.values():
            os.close(fd)
    else:

        def read_one(item: dict) -> None:
            data = (chunk_dir / item["digest"]).read_bytes()
            hashlib.sha256(data).hexdigest()
            out[item["offset"] : item["offset"] + item["size"]] = data

        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(read_one, manifest["items"]))
    return time.perf_counter() - t0


def stats(times: list[float]) -> dict:
    s = sorted(times)
    return {
        "min_s": round(s[0], 4),
        "median_s": round(s[len(s) // 2], 4),
        "p90_s": round(s[min(len(s) - 1, int(len(s) * 0.9))], 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("blob", type=Path)
    ap.add_argument("--avg-kib", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--pack-mib", type=int, default=0, help="pack chunks into N-MiB segments")
    ap.add_argument("--scratch", type=Path, required=True)
    args = ap.parse_args()

    blob = args.blob
    if blob.is_dir():  # a snapshot dir: bench its LARGEST blob (the final turn's)
        blob = max(blob.glob("*.ckpt"), key=lambda p: p.stat().st_size)
    meta, payload = load_blob(blob)
    tensors, _ = parse_safetensors(payload)
    avg = args.avg_kib * 1024

    if args.scratch.exists():
        shutil.rmtree(args.scratch)
    manifest = materialize(payload, tensors, avg, args.scratch, pack_mib=args.pack_mib)

    assembled = bytearray(len(payload))
    bench_chunked(manifest, args.scratch, args.workers, assembled)  # warm + verify
    if bytes(assembled) != payload:
        raise SystemExit("FATAL: assembled chunks != payload")

    plain_times = [bench_plain(blob, len(payload)) for _ in range(args.reps)]
    chunk_times = [
        bench_chunked(manifest, args.scratch, args.workers, bytearray(len(payload)))
        for _ in range(args.reps)
    ]

    ps, cs = stats(plain_times), stats(chunk_times)
    delta = round((cs["median_s"] - ps["median_s"]) * 1000, 1)
    gib = len(payload) / (1 << 30)
    print(f"blob: {args.blob.name}  payload={len(payload):,}B  tokens={meta['token_count']}")
    print(
        f"chunks: {len(manifest['items'])}  scope=head  avg-target={args.avg_kib}KiB  "
        f"workers={args.workers}  inline={manifest['inline_bytes']:,}B"
    )
    print(f"plain   {ps}")
    print(f"chunked {cs}  ({gib / cs['median_s']:.2f} GiB/s)")
    print(f"median delta: {delta:+.1f} ms  ({delta / 1000 / gib:+.3f} s/GiB)")
    print(f"chunk-store bytes on disk (this blob): {manifest['chunk_bytes']:,}B")

    shutil.rmtree(args.scratch)
    return 0


if __name__ == "__main__":
    sys.exit(main())
