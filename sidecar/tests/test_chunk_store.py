"""Chunked-blob store (format_version 3) tests — the Python half.

Two halves:
- FORMAT tests build a store with the RUST daemon (spawned via the same
  harness the e2e suites use is too heavy here; instead they exercise the
  Python READER against a fixture directory committed under
  scripts/fixtures/chunk-store/, which the Rust writer produced once — the
  cross-language pin: if either side drifts, these fail).
- UNIT tests cover the reader's own edges (truncated tails, missing chunks,
  tampered segments, mislabeled manifests).

Python publishes stay whole-blob v2 by design (one chunk-format writer per
directory — the daemon); both layouts coexist and both bindings read both.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import struct
from pathlib import Path

import pytest
from mlxcache_sidecar import blob
from mlxcache_store.chunks import (
    _SEGMENT_CACHE,
    FORMAT_VERSION,
    MANIFEST_MAGIC,
    SEGMENT_MAGIC,
    ChunkFormatError,
    ChunkReader,
    Manifest,
)

FIXTURE = Path(__file__).resolve().parents[2] / "scripts" / "fixtures" / "chunk-store"


def _wire_header(meta: dict) -> bytes:
    header = json.dumps(meta).encode()
    return struct.pack("<I", len(header)) + header


@pytest.fixture()
def fixture_store(tmp_path):
    """Function-scoped: the tamper test MUTATES its copy, and a
    module-scoped fixture made results depend on test order."""
    if not FIXTURE.is_dir():
        pytest.skip("chunk-store fixture not built (cargo chunkstore::fixture_dump)")
    dst = tmp_path / "chunk-store"
    shutil.copytree(FIXTURE, dst, dirs_exist_ok=True)
    return dst


def test_fixture_round_trips_through_the_python_reader(fixture_store):
    """The Rust-written store reads back byte-exactly in Python."""
    ckpt = next(fixture_store.glob("*.ckpt"))
    raw = ckpt.read_bytes()
    (hl,) = struct.unpack("<I", raw[:4])
    meta = json.loads(raw[4 : 4 + hl])
    assert meta["format_version"] == 3
    payload = ChunkReader(fixture_store).read_payload(ckpt, meta["payload_sha256"])
    # The fixture's expected payload sits beside the store.
    expected = (fixture_store / "expected-payload.bin").read_bytes()
    assert payload == expected


def test_tampered_segment_fails_the_digest(fixture_store):
    ckpt = next(fixture_store.glob("*.ckpt"))
    raw = ckpt.read_bytes()
    (hl,) = struct.unpack("<I", raw[:4])
    meta = json.loads(raw[4 : 4 + hl])
    seg = next((fixture_store / "chunks").glob("*.seg"))
    data = bytearray(seg.read_bytes())
    # Tamper INSIDE the first record's body (offset 8-byte header + 36-byte
    # record header): aa.ckpt's first chunk, so aa's read must fail. The
    # segment's last bytes belong to bb's tail chunk.
    data[8 + 36] ^= 0xFF
    seg.write_bytes(bytes(data))
    with pytest.raises(ChunkFormatError, match="digest mismatch"):
        ChunkReader(fixture_store).read_payload(ckpt, meta["payload_sha256"])


def test_missing_manifest_is_a_format_error(tmp_path):
    ckpt = tmp_path / "deadbeef-1.ckpt"
    meta = {
        "fingerprint": {
            "model_id": "m",
            "tokenizer_hash": "h",
            "kv_dtype": "f16",
            "kv_layout_version": 1,
        },
        "token_count": 4,
        "tokens": [1, 2, 3, 4],
        "format_version": 3,
        "payload_sha256": "0" * 64,
    }
    ckpt.write_bytes(_wire_header(meta))
    with pytest.raises(ChunkFormatError, match="manifest|No such file"):
        ChunkReader(tmp_path).read_payload(ckpt, meta["payload_sha256"])


def test_manifest_decode_rejects_truncated_and_foreign():
    with pytest.raises(ChunkFormatError):
        Manifest.decode(b"MCKM\x01\x00" + b"\x00" * 4)  # too short
    with pytest.raises(ChunkFormatError, match="magic"):
        Manifest.decode(b"XXXX" + b"\x00" * 40)
    # Version skew: refuse like every other versioned format here.
    bad_version = b"MCKM" + struct.pack("<H", 2) + struct.pack("<QQ", 8, 1)
    with pytest.raises(ChunkFormatError, match="version 2"):
        Manifest.decode(bad_version + b"\x00" * 56)


def test_blob_decoder_accepts_v3_and_requires_its_digest():
    meta = {
        "fingerprint": {
            "model_id": "m",
            "tokenizer_hash": "h",
            "kv_dtype": "f16",
            "kv_layout_version": 1,
        },
        "token_count": 4,
        "tokens": [1, 2, 3, 4],
        "format_version": 3,
        "payload_sha256": hashlib.sha256(b"x").hexdigest(),
    }
    wire = _wire_header(meta) + b""
    m, payload = blob.decode(wire)
    assert m.format_version == 3 and payload == b""
    meta_v3_nodigest = {k: v for k, v in meta.items() if k != "payload_sha256"}
    wire2 = _wire_header(meta_v3_nodigest)
    with pytest.raises(blob.HeaderRejected, match="requires payload_sha256"):
        blob.decode(wire2)


def test_reader_tolerates_missing_chunks_dir(tmp_path):
    """An empty/absent chunk store is legitimate (no chunked blobs yet)."""
    reader = ChunkReader(tmp_path)
    assert reader.placement == {}


def test_duplicate_digests_resolve_to_the_newest_segment(tmp_path):
    """A compaction crash can strand one digest in TWO segments (the
    rewrite durable, the old file not yet deleted). The reader must
    resolve to the NEWEST copy — the same order as the Rust boot scan —
    so the old segment counts as fully dead and reclaims. A
    lowest-seq-wins reader would pin the stale file forever."""
    body = bytes(range(64))
    digest = hashlib.sha256(body).digest()
    record = struct.pack("<I32s", len(body), digest) + body
    header = SEGMENT_MAGIC + struct.pack("<H", FORMAT_VERSION) + b"\x00\x00"
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    (chunks_dir / "seg-0000000000.seg").write_bytes(header + record)  # stale copy
    (chunks_dir / "seg-0000000001.seg").write_bytes(header + record)  # the rewrite
    ckm = struct.pack("<4sHQQ", MANIFEST_MAGIC, FORMAT_VERSION, len(body), 1) + struct.pack(
        "<32sIIQQ", digest, len(body), 1, 0, 0
    )
    (tmp_path / "aa.ckm").write_bytes(ckm)
    meta = {
        "fingerprint": {
            "model_id": "m",
            "tokenizer_hash": "h",
            "kv_dtype": "f16",
            "kv_layout_version": 1,
        },
        "token_count": 4,
        "tokens": [1, 2, 3, 4],
        "format_version": 3,
        "payload_sha256": hashlib.sha256(body).hexdigest(),
    }
    ckpt = tmp_path / "aa.ckpt"
    ckpt.write_bytes(_wire_header(meta))
    reader = ChunkReader(tmp_path)
    name, offset, body_len = reader.placement[digest]
    assert name == "seg-0000000001.seg", "the newest segment wins a shared digest"
    assert (offset, body_len) == (8 + 36, len(body))
    assert reader.read_payload(ckpt, meta["payload_sha256"]) == body


def test_reassemble_rescans_when_a_segment_vanishes_mid_read(tmp_path, monkeypatch):
    """A compaction sweep can delete a segment after this reader scanned
    placements but before it opened the file. The read must rescan and
    land on the surviving copy, not surface FileNotFoundError."""
    body = bytes(range(64))
    digest = hashlib.sha256(body).digest()
    record = struct.pack("<I32s", len(body), digest) + body
    header = SEGMENT_MAGIC + struct.pack("<H", FORMAT_VERSION) + b"\x00\x00"
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    (chunks_dir / "seg-0000000000.seg").write_bytes(header + record)
    ckm = struct.pack("<4sHQQ", MANIFEST_MAGIC, FORMAT_VERSION, len(body), 1) + struct.pack(
        "<32sIIQQ", digest, len(body), 1, 0, 0
    )
    (tmp_path / "aa.ckm").write_bytes(ckm)
    sha = hashlib.sha256(body).hexdigest()
    (tmp_path / "aa.ckpt").write_bytes(_wire_header({"format_version": 3, "payload_sha256": sha}))

    calls = {"n": 0}
    real_scan = ChunkReader._scan_segments

    def stale_then_fresh(self):
        calls["n"] += 1
        if calls["n"] == 1:
            # The pre-sweep snapshot: placements point at a segment the
            # sweep is about to delete (no such file on disk).
            return {digest: ("seg-0000000009.seg", 8 + 36, len(body))}
        return real_scan(self)

    monkeypatch.setattr(ChunkReader, "_scan_segments", stale_then_fresh)
    assert ChunkReader(tmp_path).read_payload(tmp_path / "aa.ckpt", sha) == body
    assert calls["n"] == 2, "exactly one rescan, then success"


def test_reassemble_rescans_when_a_segment_vanishes_mid_scan(tmp_path):
    """Pass 2 of the scan re-opens segment files to rebuild offsets; a
    sweep can delete one between Pass 1's glob and that open. The
    FileNotFoundError must become a rescan (which resolves the digest to
    the surviving copy), never a rejected checkpoint."""
    body = bytes(range(64))
    digest = hashlib.sha256(body).digest()
    record = struct.pack("<I32s", len(body), digest) + body
    header = SEGMENT_MAGIC + struct.pack("<H", FORMAT_VERSION) + b"\x00\x00"
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    (chunks_dir / "seg-0000000000.seg").write_bytes(header + record)
    newest = chunks_dir / "seg-0000000001.seg"
    newest.write_bytes(header + record)  # a compaction rewrite
    ckm = struct.pack("<4sHQQ", MANIFEST_MAGIC, FORMAT_VERSION, len(body), 1) + struct.pack(
        "<32sIIQQ", digest, len(body), 1, 0, 0
    )
    (tmp_path / "aa.ckm").write_bytes(ckm)
    sha = hashlib.sha256(body).hexdigest()
    ckpt = tmp_path / "aa.ckpt"
    ckpt.write_bytes(_wire_header({"format_version": 3, "payload_sha256": sha}))

    reader = ChunkReader(tmp_path)
    assert reader.placement[digest][0] == newest.name
    # Replay the race: an in-flight scan's Pass 1 already cached the
    # newest segment when the sweep deleted it — Pass 2 now opens a file
    # that is gone. The cache entry stands in for that in-flight scan.
    size = 8 + 36 + len(body)
    _SEGMENT_CACHE[tmp_path][newest.name] = (size, [digest])
    newest.unlink()
    # The read must survive: the vanish signal triggers one rescan, and
    # the fresh scan resolves the digest to the surviving old copy.
    assert reader.read_payload(ckpt, sha) == body
    assert reader.placement[digest][0] == "seg-0000000000.seg"
