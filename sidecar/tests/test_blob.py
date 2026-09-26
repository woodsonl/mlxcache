"""blob.py codec tests: byte-compatibility with the Rust daemon's persistence.rs."""

from __future__ import annotations

import json
import struct

from mlxcache_sidecar import blob


def _meta(tokens: list[int]) -> blob.CheckpointMeta:
    return blob.CheckpointMeta(
        fingerprint=blob.Fingerprint(
            model_id="m", tokenizer_hash="h", kv_dtype="f16", kv_layout_version=1
        ),
        token_count=len(tokens),
        tokens=tokens,
    )


def test_encode_decode_roundtrip():
    meta = _meta([1, 2, 3])
    encoded = blob.encode(meta, b"payload")
    decoded, payload = blob.decode(encoded)
    assert decoded.token_count == 3
    assert decoded.tokens == [1, 2, 3]
    assert payload == b"payload"


def test_header_layout_matches_rust():
    # u32 LE header length + JSON header + payload, exactly as persistence.rs.
    encoded = blob.encode(_meta([7]), b"KV")
    (header_len,) = struct.unpack("<I", encoded[:4])
    header = json.loads(encoded[4 : 4 + header_len])
    assert header["token_count"] == 1
    assert header["tokens"] == [7]
    assert header["format_version"] == 1
    assert header["fingerprint"]["model_id"] == "m"
    assert encoded[4 + header_len :] == b"KV"


def test_decode_rejects_truncated_header_length():
    try:
        blob.decode(b"xy")
    except ValueError as e:
        assert "truncated" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_decode_rejects_header_longer_than_blob():
    # header_len says 100 but only 4 bytes follow.
    bad = struct.pack("<I", 100) + b"{}"
    try:
        blob.decode(bad)
    except ValueError as e:
        assert "exceeds" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_decode_rejects_version_mismatch():
    meta = _meta([1])
    meta.format_version = 99
    try:
        blob.decode(blob.encode(meta, b"x"))
    except ValueError as e:
        assert "version" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_decode_tolerates_missing_tokens_field():
    # Older blobs (pre-tokens) must still decode, with an empty prefix.
    header = json.dumps(
        {
            "fingerprint": {
                "model_id": "m",
                "tokenizer_hash": "h",
                "kv_dtype": "f16",
                "kv_layout_version": 1,
            },
            "token_count": 5,
            "format_version": 1,
        }
    ).encode()
    encoded = struct.pack("<I", len(header)) + header + b"kv"
    meta, payload = blob.decode(encoded)
    assert meta.tokens == []
    assert payload == b"kv"
