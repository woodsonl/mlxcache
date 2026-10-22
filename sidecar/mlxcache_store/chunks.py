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
from pathlib import Path

SEGMENT_MAGIC = b"MCSG"
MANIFEST_MAGIC = b"MCKM"
FORMAT_VERSION = 1
_HEADER_BYTES = 8
_RECORD_HEADER = struct.Struct("<I32s")  # body_len, digest
_MANIFEST_HEAD = struct.Struct("<4sHQQ")  # magic, version, payload_len, count
_CHUNK_RECORD = struct.Struct("<32sIIQQ")  # digest, per_head_len, n_heads, first_off, head_stride


class ChunkFormatError(ValueError):
    """A deterministically invalid chunked layout (bad magic/version,
    truncated manifest, missing chunk). Callers map this to the same
    corruption taxonomy as a whole-blob digest mismatch."""


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
            chunks.append(ChunkRef(digest, per_head_len, n_heads, first_off, head_stride))
        return cls(payload_len, chunks)


class ChunkReader:
    """Reads chunked payloads from one blob directory.

    The segment index is built lazily on first use and cached: a boot-time
    scan of every segment is the same cost class the daemon pays, and the
    reader is per-process.
    """

    def __init__(self, blob_dir: str | Path):
        self.dir = Path(blob_dir)
        self._placement: dict[bytes, tuple[str, int, int]] | None = (
            None  # digest -> (seg name, offset, len)
        )

    def _scan_segments(self) -> dict[bytes, tuple[str, int, int]]:
        chunks_dir = self.dir / "chunks"
        placement: dict[bytes, tuple[str, int, int]] = {}
        if not chunks_dir.is_dir():
            return placement
        for seg in sorted(chunks_dir.glob("seg-*.seg")):
            data = seg.read_bytes()
            if len(data) < _HEADER_BYTES or data[0:4] != SEGMENT_MAGIC:
                raise ChunkFormatError(f"{seg.name}: bad segment magic")
            version = struct.unpack_from("<H", data, 4)[0]
            if version != FORMAT_VERSION:
                raise ChunkFormatError(f"{seg.name}: segment version {version}")
            at = _HEADER_BYTES
            while at + _RECORD_HEADER.size <= len(data):
                body_len, digest = _RECORD_HEADER.unpack_from(data, at)
                body_start = at + _RECORD_HEADER.size
                body_end = body_start + body_len
                if body_end > len(data):
                    break  # truncated tail: dead bytes, stop cleanly
                placement.setdefault(digest, (seg.name, body_start, body_len))
                at = body_end
        return placement

    @property
    def placement(self) -> dict[bytes, tuple[str, int, int]]:
        if self._placement is None:
            self._placement = self._scan_segments()
        return self._placement

    def reassemble(self, manifest: Manifest) -> bytes:
        """The blob's payload bytes, scattered from the segments."""
        out = bytearray(manifest.payload_len)
        placement = self.placement
        # Read every needed segment region while the file is open — one
        # open per segment, no long-lived handles (the reader can sit on a
        # serving thread).
        by_segment: dict[str, list[tuple[bytes, int, int]]] = {}
        for c in manifest.chunks:
            where = placement.get(c.digest)
            if where is None:
                raise ChunkFormatError(f"chunk {c.digest[:8].hex()} missing from the store")
            name, offset, body_len = where
            by_segment.setdefault(name, []).append((c.digest, offset, body_len))
        bodies: dict[bytes, bytes] = {}
        for name, regions in by_segment.items():
            with open(self.dir / "chunks" / name, "rb") as fh:
                for digest, offset, body_len in regions:
                    fh.seek(offset)
                    body = fh.read(body_len)
                    if len(body) != body_len:
                        raise ChunkFormatError("segment truncated under a live chunk")
                    bodies[digest] = body
        for c in manifest.chunks:
            body = bodies[c.digest]
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
