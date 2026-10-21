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


def test_decode_accepts_t12_daemon_blobs_with_kv_tier_fields():
    # Regression (2026-10-03): the daemon's serde now serializes kv_bits and
    # kv_group_size into the fingerprint. The first delta-prefill run against
    # that daemon 422-quarantined every healthy ancestor because
    # Fingerprint(**raw) raised TypeError on the unexpected kwargs. The reader
    # must accept the SUPERSET (a newer daemon's header) and adopt the tier.
    rust_style = {
        "fingerprint": {
            "model_id": "mlx-community/Qwen2.5-7B-Instruct-4bit",
            "tokenizer_hash": "1a2b3c4d5e6f7081",
            "kv_dtype": "float16",
            "kv_layout_version": 1,
            "kv_bits": 8,
            "kv_group_size": 64,
        },
        "token_count": 8,
        "tokens": [10, 11, 12, 13, 14, 15, 16, 17],
        "format_version": 1,
    }
    header = json.dumps(rust_style).encode()
    blob_bytes = struct.pack("<I", len(header)) + header + b"KV"
    meta, _payload = blob.decode(blob_bytes)
    assert meta.fingerprint.kv_bits == 8
    assert meta.fingerprint.kv_group_size == 64
    assert meta.tokens == [10, 11, 12, 13, 14, 15, 16, 17]


def test_decode_drops_unknown_fingerprint_fields():
    # Forward compatibility: a future daemon adds a field this sidecar has
    # never heard of. The blob must still decode (dropping the unknown key),
    # not 422-quarantine the checkpoint.
    rust_style = {
        "fingerprint": {
            "model_id": "m",
            "tokenizer_hash": "h",
            "kv_dtype": "f16",
            "kv_layout_version": 1,
            "kv_bits": 0,
            "kv_group_size": 0,
            "some_future_field": "x",
        },
        "token_count": 3,
        "tokens": [1, 2, 3],
        "format_version": 1,
    }
    header = json.dumps(rust_style).encode()
    meta, _ = blob.decode(struct.pack("<I", len(header)) + header + b"KV")
    assert meta.fingerprint.model_id == "m"
    assert meta.fingerprint.kv_bits == 0


def test_t12_blobs_still_decode_after_default_fingerprint_write():
    # The sidecar's own encode path (asdict) round-trips the tier fields so a
    # q8 blob written by this sidecar reads back as q8.
    meta = blob.CheckpointMeta(
        fingerprint=blob.Fingerprint(
            model_id="m",
            tokenizer_hash="h",
            kv_dtype="f16",
            kv_layout_version=1,
            kv_bits=8,
            kv_group_size=64,
        ),
        token_count=3,
        tokens=[1, 2, 3],
    )
    meta2, _ = blob.decode(blob.encode(meta, b"payload"))
    assert meta2.fingerprint.kv_bits == 8
    assert meta2.fingerprint.kv_group_size == 64


def test_decode_accepts_daemon_blobs_with_cache_layout():
    # Regression guard for the B1 field, by the same discipline as the T12
    # tier fields above: a daemon that records cache composition serializes
    # one more fingerprint field. The reader must adopt it (not strip it, not
    # TypeError), and a header WITHOUT it must read as "" (unrecorded).
    rust_style = {
        "fingerprint": {
            "model_id": "mlx-community/Qwen2.5-7B-Instruct-4bit",
            "tokenizer_hash": "1a2b3c4d5e6f7081",
            "kv_dtype": "float16",
            "kv_layout_version": 1,
            "kv_bits": 0,
            "kv_group_size": 0,
            "cache_layout": "9f8e7d6c5b4a3928",
        },
        "token_count": 4,
        "tokens": [1, 2, 3, 4],
        "format_version": 1,
    }
    header = json.dumps(rust_style).encode()
    meta, _ = blob.decode(struct.pack("<I", len(header)) + header + b"KV")
    assert meta.fingerprint.cache_layout == "9f8e7d6c5b4a3928"

    legacy = dict(rust_style)
    legacy["fingerprint"] = {
        k: v for k, v in rust_style["fingerprint"].items() if k != "cache_layout"
    }
    header = json.dumps(legacy).encode()
    meta, _ = blob.decode(struct.pack("<I", len(header)) + header + b"KV")
    assert meta.fingerprint.cache_layout == "", "absent layout means unrecorded"


def test_encode_omits_an_unrecorded_cache_layout():
    # An empty layout is "composition unknown", which is exactly what a pre-B1
    # header says by saying nothing. Emitting `"cache_layout":""` would make
    # every publish differ from the pinned cross-language wire golden while
    # carrying no information.
    encoded = blob.encode(_meta([1, 2]), b"kv")
    (header_len,) = struct.unpack("<I", encoded[:4])
    header = json.loads(encoded[4 : 4 + header_len])
    assert "cache_layout" not in header["fingerprint"]

    recorded = _meta([1, 2])
    recorded.fingerprint.cache_layout = "abc123"
    encoded = blob.encode(recorded, b"kv")
    (header_len,) = struct.unpack("<I", encoded[:4])
    header = json.loads(encoded[4 : 4 + header_len])
    assert header["fingerprint"]["cache_layout"] == "abc123"


def test_decode_rejects_non_string_cache_layout():
    # Untrusted header field: a non-str layout would raise on the fold's
    # .encode() outside every guarded region — the nested poison-pill path.
    for bad in (5, None, ["x"], {"a": 1}):
        rust_style = {
            "fingerprint": {
                "model_id": "m",
                "tokenizer_hash": "h",
                "kv_dtype": "f16",
                "kv_layout_version": 1,
                "cache_layout": bad,
            },
            "token_count": 2,
            "tokens": [1, 2],
            "format_version": 1,
        }
        header = json.dumps(rust_style).encode()
        try:
            blob.decode(struct.pack("<I", len(header)) + header + b"KV")
        except blob.HeaderRejected as exc:
            assert exc.kind == "header", bad
        else:
            raise AssertionError(f"non-string cache_layout {bad!r} must be rejected")


def test_decode_accepts_and_omits_content_digest():
    # B2 forward compatibility, by the same discipline as cache_layout: a
    # daemon that records non-text content serializes one more field, and the
    # reader must adopt it. An empty digest is "no non-text content" — exactly
    # what a pre-B2 header says by saying nothing — so it must not be emitted.
    rust_style = {
        "fingerprint": {
            "model_id": "mlx-community/Qwen2.5-7B-Instruct-4bit",
            "tokenizer_hash": "1a2b3c4d5e6f7081",
            "kv_dtype": "float16",
            "kv_layout_version": 1,
            "content_digest": "b1946ac92492d2347c6235b4d2611184",
        },
        "token_count": 4,
        "tokens": [1, 2, 3, 4],
        "format_version": 1,
    }
    header = json.dumps(rust_style).encode()
    meta, _ = blob.decode(struct.pack("<I", len(header)) + header + b"KV")
    assert meta.fingerprint.content_digest == "b1946ac92492d2347c6235b4d2611184"

    encoded = blob.encode(_meta([1, 2]), b"kv")
    (header_len,) = struct.unpack("<I", encoded[:4])
    header = json.loads(encoded[4 : 4 + header_len])
    assert "content_digest" not in header["fingerprint"]

    recorded = _meta([1, 2])
    recorded.fingerprint.content_digest = "abc123"
    encoded = blob.encode(recorded, b"kv")
    (header_len,) = struct.unpack("<I", encoded[:4])
    header = json.loads(encoded[4 : 4 + header_len])
    assert header["fingerprint"]["content_digest"] == "abc123"


def test_decode_rejects_non_string_content_digest():
    # Same poison-pill path as cache_layout: a non-str digest would raise on
    # the fold's .encode() outside every guarded region.
    for bad in (5, None, ["x"], {"a": 1}):
        rust_style = {
            "fingerprint": {
                "model_id": "m",
                "tokenizer_hash": "h",
                "kv_dtype": "f16",
                "kv_layout_version": 1,
                "content_digest": bad,
            },
            "token_count": 2,
            "tokens": [1, 2],
            "format_version": 1,
        }
        header = json.dumps(rust_style).encode()
        try:
            blob.decode(struct.pack("<I", len(header)) + header + b"KV")
        except blob.HeaderRejected as exc:
            assert exc.kind == "header", bad
        else:
            raise AssertionError(f"non-string content_digest {bad!r} must be rejected")


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
