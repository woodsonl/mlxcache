"""Chunked-blob (format_version 3) reader — the Python mirror of the Rust
chunk store (crates/mlxcache-daemon/src/chunkstore.rs).

The daemon publishes v3 blobs as a HEADER-ONLY `.ckpt` plus a binary `.ckm`
manifest plus packed `chunks/seg-*.seg` segments; the payload is reassembled
byte-exactly from per-chunk placements and verified against the header's
sha256 (the same D1 contract as whole blobs — corruption anywhere in the
manifest or segments yields wrong bytes and a digest mismatch, never a wrong
verdict).

Binary format (all little-endian; mirrors the Rust writer byte-for-byte):

- Segment: 8-byte header (magic ``MCSG`` + u16 version + 2 reserved zeros),
  then records ``[u32 body_len][32B digest][body]``. A truncated tail record
  (crash mid-append) ends the scan of that segment cleanly.
- Manifest (.ckm): magic ``MCKM`` + u16 version, u64 payload_len, u64
  chunk_count, then 56-byte records: 32B digest, u32 per_head_len, u32
  n_heads, u64 first_off, u64 head_stride. A chunk's body is the per-head
  concatenation (head order); reassembly scatters per_head_len bytes at
  ``first_off + h*head_stride`` for each head.

This module only READS. Python publishers (the L1 store's ``put``) keep
writing whole-blob v2 — one writer of the chunk format per store directory
avoids cross-language allocator drift; the daemon is that writer. Both
layouts coexist in one directory and both bindings read both.
"""

from __future__ import annotations

import hashlib
import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SEGMENT_MAGIC = b"MCSG"
MANIFEST_MAGIC = b"MCKM"
FORMAT_VERSION = 1
READ_WORKERS = 8
"""Segment-read parallelism, mirroring READ_WORKERS in the Rust reader: a
blob's chunks span every segment written over its lifetime, file reads
release the GIL, and only the scatter stays on the calling thread."""
_HEADER_BYTES = 8
_RECORD_HEADER = struct.Struct("<I32s")  # body_len, digest
_MANIFEST_HEAD = struct.Struct("<4sHQQ")  # magic, version, payload_len, count
_CHUNK_RECORD = struct.Struct("<32sIIQQ")  # digest, per_head_len, n_heads, first_off, head_stride


class ChunkFormatError(ValueError):
    """A deterministically invalid chunked layout (bad magic/version,
    truncated manifest, missing chunk). Callers map this to the same
    corruption taxonomy as a whole-blob digest mismatch."""


class _SegmentVanishedError(Exception):
    """Internal: a segment disappeared between scan and open (a daemon
    compaction sweep re-homed its records). Signals a rescan; never
    escapes this module."""

    def __init__(self, name: str):
        super().__init__(name)
        self.name = name


def read_manifest(blob_path: str | Path) -> Manifest:
    """The parsed .ckm manifest for a v3 blob."""
    ckm = Path(blob_path).with_suffix(".ckm")
    try:
        data = ckm.read_bytes()
    except OSError as exc:
        raise ChunkFormatError(f"chunk manifest unreadable: {exc}") from exc
    return Manifest.decode(data)


class ChunkRef:
    __slots__ = ("digest", "per_head_len", "n_heads", "first_off", "head_stride")

    def __init__(
        self, digest: bytes, per_head_len: int, n_heads: int, first_off: int, head_stride: int
    ):
        self.digest = digest
        self.per_head_len = per_head_len
        self.n_heads = n_heads
        self.first_off = first_off
        self.head_stride = head_stride

    @property
    def body_len(self) -> int:
        return self.n_heads * self.per_head_len


class Manifest:
    __slots__ = ("payload_len", "chunks")

    def __init__(self, payload_len: int, chunks: list[ChunkRef]):
        self.payload_len = payload_len
        self.chunks = chunks

    @classmethod
    def decode(cls, data: bytes) -> Manifest:
        if len(data) < 22:
            raise ChunkFormatError("chunk manifest corrupt: too short")
        magic, version, payload_len, count = _MANIFEST_HEAD.unpack_from(data, 0)
        if magic != MANIFEST_MAGIC:
            raise ChunkFormatError("chunk manifest corrupt: bad magic")
        if version != FORMAT_VERSION:
            raise ChunkFormatError(f"chunk manifest corrupt: version {version}")
        need = 22 + count * _CHUNK_RECORD.size
        if len(data) < need:
            raise ChunkFormatError("chunk manifest corrupt: truncated records")
        chunks = []
        at = 22
        max_end = 0
        for _ in range(count):
            digest, per_head_len, n_heads, first_off, head_stride = _CHUNK_RECORD.unpack_from(
                data, at
            )
            at += _CHUNK_RECORD.size
            if n_heads == 0 or per_head_len == 0:
                raise ChunkFormatError("chunk manifest corrupt: empty chunk geometry")
            end = first_off + (n_heads - 1) * head_stride + per_head_len
            if end > payload_len:
                raise ChunkFormatError("chunk manifest corrupt: placement exceeds payload")
            max_end = max(max_end, end)
            chunks.append(ChunkRef(digest, per_head_len, n_heads, first_off, head_stride))
        # The chunker guarantees chunks cover the payload: a bit-flipped
        # payload_len would otherwise decode fine and make reassembly
        # allocate attacker-sized memory (MemoryError escapes the
        # ChunkFormatError taxonomy).
        if max_end != payload_len and not (payload_len == 0 and not chunks):
            raise ChunkFormatError("chunk manifest corrupt: payload_len does not match coverage")
        return cls(payload_len, chunks)


_SEGMENT_CACHE: dict[Path, dict[str, tuple[int, dict[bytes, tuple[int, int]]]]] = {}
"""dir -> {segment name -> (scanned size, {digest -> (offset, body_len)})}.

Per-segment, process-lifetime cache with INCREMENTAL tail scans: a segment
that only grew since the last scan contributes only its new records (the
daemon appends on every publish, so a whole-store signature would re-read
every segment after every publication); a shrunk or replaced segment
rescans from zero; a vanished segment is skipped (the reassembler reports
genuinely missing chunks). Reads walk each segment ONCE with pread-style
seeks — no whole-segment loads.
"""


def _read_segment_regions(
    blob_dir: Path, name: str, regions: list[tuple[bytes, int, int]]
) -> dict[bytes, bytes]:
    """One segment's chunk bodies, read while the file is open."""
    bodies: dict[bytes, bytes] = {}
    with open(blob_dir / "chunks" / name, "rb") as fh:
        for digest, offset, body_len in regions:
            fh.seek(offset)
            body = fh.read(body_len)
            if len(body) != body_len:
                raise ChunkFormatError("segment truncated under a live chunk")
            bodies[digest] = body
    return bodies


class ChunkReader:
    """Reads chunked payloads from one blob directory.

    The segment index is shared per directory and cached process-wide,
    keyed by the segment files' (name, size) signature: a re-scan happens
    only when the set changes (a segment rolls or GC deletes one). Without
    the cache every v3 read would re-scan every segment byte — GBs on a
    serving hot path.
    """

    def __init__(self, blob_dir: str | Path):
        self.dir = Path(blob_dir)

    def _scan_segments(self) -> dict[bytes, tuple[str, int, int]]:
        chunks_dir = self.dir / "chunks"
        if not chunks_dir.is_dir():
            return {}
        cache = _SEGMENT_CACHE.setdefault(self.dir, {})
        # Purge cached segments that no longer exist (GC deleted them);
        # otherwise their digests keep shadowing live placements forever.
        live_names = {seg.name for seg in chunks_dir.glob("seg-*.seg")}
        for name in list(cache):
            if name not in live_names:
                del cache[name]
        # Pass 1: refresh each segment's digest list (incremental for
        # growth, full rescan for shrink/replace, skip for vanish).
        for seg in sorted(chunks_dir.glob("seg-*.seg")):
            try:
                size = seg.stat().st_size
                with open(seg, "rb") as fh:
                    header = fh.read(_HEADER_BYTES)
                    if len(header) < _HEADER_BYTES or header[0:4] != SEGMENT_MAGIC:
                        # Crash artifact (short file) or foreign file: no
                        # records; the manifest digest check stays the
                        # correctness guard.
                        cache.pop(seg.name, None)
                        continue
                    version = struct.unpack_from("<H", header, 4)[0]
                    if version != FORMAT_VERSION:
                        raise ChunkFormatError(f"{seg.name}: segment version {version}")
                    cached = cache.get(seg.name)
                    if cached is not None and cached[0] == size:
                        continue  # unchanged: nothing to do
                    start_offset = (
                        _HEADER_BYTES if cached is None or cached[0] > size else cached[0]
                    )
                    digests = [] if cached is None or cached[0] > size else list(cached[1])
                    fh.seek(start_offset)
                    at = start_offset
                    while True:
                        rec = fh.read(_RECORD_HEADER.size)
                        if len(rec) < _RECORD_HEADER.size:
                            break  # truncated tail: dead bytes, stop cleanly
                        body_len, digest = _RECORD_HEADER.unpack(rec)
                        body_start = at + _RECORD_HEADER.size
                        if body_start + body_len > size:
                            break
                        digests.append(digest)
                        at = body_start + body_len
                        fh.seek(body_start + body_len)
                    cache[seg.name] = (size, digests)
            except FileNotFoundError:
                # Deleted by GC between glob and open: skip. A genuinely
                # missing chunk surfaces in reassemble.
                cache.pop(seg.name, None)
                continue
        # Pass 2: derive digest -> placement from the cached digests. The
        # placement offsets are implied by the record layout: walk each
        # segment's digests reconstructing offsets. HIGHEST seq first, so
        # the first-wins rule (setdefault) keeps the NEWEST copy of a
        # duplicate digest — the same resolution order as the Rust boot
        # scan. Duplicates only exist after a compaction crash (new segment
        # durable, old not yet deleted); both readers must agree on the
        # new segment so the old one is fully reclaimable.
        placement: dict[bytes, tuple[str, int, int]] = {}
        for name, (_size, digests) in sorted(cache.items(), reverse=True):
            at = _HEADER_BYTES
            for digest in digests:
                body_start = at + _RECORD_HEADER.size
                body_len = None
                # body length is not cached; re-read just the 4-byte header
                # of each record to recover offsets exactly.
                try:
                    with open(chunks_dir / name, "rb") as fh:
                        fh.seek(at)
                        raw = fh.read(4)
                except FileNotFoundError as exc:
                    # Deleted between Pass 1 and this open (a sweep
                    # re-homed the records): the caller must rescan, so
                    # the fresh pass sees the replacement segment —
                    # raising a raw FileNotFoundError here would bypass
                    # the retry and reject a healthy checkpoint.
                    raise _SegmentVanishedError(name) from exc
                if len(raw) < 4:
                    break
                body_len = struct.unpack("<I", raw)[0]
                placement.setdefault(digest, (name, body_start, body_len))
                at = body_start + body_len
        return placement

    @property
    def placement(self) -> dict[bytes, tuple[str, int, int]]:
        return self._scan_segments()

    def reassemble(self, manifest: Manifest) -> bytes:
        """The blob's payload bytes, scattered from the segments."""
        # A daemon compaction sweep can delete a segment between this
        # scan and the opens below (it moves live records to a new
        # segment first). Rescan and retry — bounded, because a sweep
        # moves each digest at most once; a segment still missing after
        # that is real corruption.
        last_vanished: _SegmentVanishedError | None = None
        for _ in range(3):
            try:
                return self._reassemble_from(manifest, self.placement)
            except _SegmentVanishedError as exc:
                last_vanished = exc
        raise ChunkFormatError(
            f"segment {last_vanished.name} vanished under a live placement"
        ) from last_vanished

    def _reassemble_from(
        self, manifest: Manifest, placement: dict[bytes, tuple[str, int, int]]
    ) -> bytes:
        out = bytearray(manifest.payload_len)
        # Group the reads per segment — one open per segment, no long-lived
        # handles (the reader can sit on a serving thread).
        by_segment: dict[str, list[tuple[bytes, int, int]]] = {}
        for c in manifest.chunks:
            where = placement.get(c.digest)
            if where is None:
                raise ChunkFormatError(f"chunk {c.digest[:8].hex()} missing from the store")
            name, offset, body_len = where
            by_segment.setdefault(name, []).append((c.digest, offset, body_len))
        bodies: dict[bytes, bytes] = {}
        # Segments read in parallel (file reads release the GIL; the Rust
        # reader does the same with 8 pread workers): a blob's chunks span
        # every segment written over its lifetime, and a serial walk made
        # restore latency scale with the segment count.
        with ThreadPoolExecutor(max_workers=min(READ_WORKERS, max(1, len(by_segment)))) as pool:
            futures = {
                pool.submit(_read_segment_regions, self.dir, name, regions): name
                for name, regions in by_segment.items()
            }
            for fut, name in futures.items():
                try:
                    bodies.update(fut.result())
                except FileNotFoundError as exc:
                    # The segment vanished between scan and open (a sweep
                    # deleted it after re-homing its records): rescan.
                    raise _SegmentVanishedError(name) from exc
        for c in manifest.chunks:
            body = bodies[c.digest]
            if len(body) != c.body_len:
                # Manifest geometry disagrees with the stored record (bit
                # rot): wrong bytes for the digest to catch, never an
                # uncontrolled ValueError from the slice assign below.
                raise ChunkFormatError(
                    f"chunk {c.digest[:8].hex()} geometry disagrees with the stored record"
                )
            for h in range(c.n_heads):
                dst = c.first_off + h * c.head_stride
                out[dst : dst + c.per_head_len] = body[
                    h * c.per_head_len : (h + 1) * c.per_head_len
                ]
        return bytes(out)

    def read_payload(self, blob_path: str | Path, expected_sha256: str | None) -> bytes:
        """The verified payload for a v3 blob: manifest + reassembly + the
        D1 digest check (never skipped — a v3 header without a digest is
        corrupt, same rule as v2)."""
        manifest = read_manifest(blob_path)
        payload = self.reassemble(manifest)
        if expected_sha256 is None:
            raise ChunkFormatError("format version 3 requires payload_sha256")
        actual = hashlib.sha256(payload).hexdigest()
        if actual != expected_sha256.lower():
            raise ChunkFormatError(
                f"payload digest mismatch (expected {expected_sha256}, got {actual})"
            )
        return payload


def read_payload_v3(blob_dir: str | Path, blob_path: str | Path, expected_sha256: str) -> bytes:
    """The verified payload of one v3 blob, from its store directory.

    The single v3 read path: both the sidecar's checkpoint boundary and the
    L1 store's open/fetch call this, so reassembly, the digest contract,
    and the ChunkFormatError taxonomy have exactly one implementation per
    language.
    """
    return ChunkReader(blob_dir).read_payload(blob_path, expected_sha256)
