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
