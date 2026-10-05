"""Connector-protocol FORMAT conformance (B0.3) — the Python half.

The golden blob SHA is asserted by BOTH suites (the Rust twin:
crates/mlxcache-daemon/tests/conformance.rs). Both writers producing
byte-identical files IS the cross-language round-trip (protocol §9).
Store-semantics rows (granularity matching, budget, concurrency) land
with the L1 client in B1.1.
"""

import hashlib

import pytest
from mlxcache_sidecar import blob, server

GOLDEN_BLOB_SHA = "5da07a82f4336e3f8d1bed1bba3decfa582ce4dc0e1d92f2e6f8dd4b465c7204"


def _golden_meta() -> blob.CheckpointMeta:
    return blob.CheckpointMeta(
        fingerprint=blob.Fingerprint(
            model_id="conformance-model",
            tokenizer_hash="conf-tok",
            kv_dtype="f16",
            kv_layout_version=1,
        ),
        token_count=4,
        tokens=[1, 2, 3, 4],
        # The Rust writer's publish path stamps these; we craft the
        # identical final bytes explicitly.
        format_version=blob.FORMAT_VERSION_D1,
        payload_sha256=hashlib.sha256(b"conformance-payload").hexdigest(),
    )


def test_golden_blob_bytes_match_the_rust_writer():
    wire = blob.encode(_golden_meta(), b"conformance-payload")
    assert len(wire) == 306, "wire length drifted"
    assert hashlib.sha256(wire).hexdigest() == GOLDEN_BLOB_SHA, (
        "the Python codec no longer matches the cross-language golden"
    )


def test_unknown_header_field_is_stripped_not_fatal():
    # §2: unknown fields are ignorable (the T12 rule). A newer daemon wrote
    # a field this reader does not model.
    wire = bytearray(blob.encode(_golden_meta(), b"kv"))
    header_len = int.from_bytes(wire[0:4], "little")
    # splice an unknown field right after the opening brace
    field = b'"future_field":42,'
    wire = wire[0:5] + field + wire[5:]
    new_len = header_len + len(field)
    wire[0:4] = new_len.to_bytes(4, "little")
    meta, payload = blob.decode(bytes(wire))
    assert payload == b"kv"
    assert meta.token_count == 4
    assert "future" not in blob.asdict(meta), "unknown fields never surface"


def test_foreign_engine_opaque_payload_round_trip(tmp_path):
    # §1 opacity + §2 [B0.2] fields: a llama-cpp-shaped entry with
    # NON-safetensors bytes passes the codec and the serve-time boundary
    # (framing checks are engine-side; the digest is the integrity rule).
    meta = blob.CheckpointMeta(
        fingerprint=blob.Fingerprint(
            model_id="gguf:model",
            tokenizer_hash="cpp-tok",
            kv_dtype="opaque",
            kv_layout_version=1,
        ),
        token_count=2,
        tokens=[7, 9],
        format_version=blob.FORMAT_VERSION_D1,
        payload_sha256=hashlib.sha256(b"\x00\x01raw-state").hexdigest(),
        engine_id="llama-cpp",
        granularity=1,  # WHOLE_CONTEXT
    )
    path = tmp_path / "llama.ckpt"
    path.write_bytes(blob.encode(meta, b"\x00\x01raw-state"))
    m, payload, usable = server.read_wire_checkpoint(str(path), [7, 9], check_safetensors=False)
    assert usable is not False
    assert payload == b"\x00\x01raw-state", "opaque bytes pass uninterpreted"
    assert m.engine_id == "llama-cpp"
    assert m.granularity == 1


def test_version_matrix():
    fp = blob.Fingerprint(model_id="m", tokenizer_hash="h", kv_dtype="f16", kv_layout_version=1)
    # v1 legacy: readable, no digest, unverified
    v1 = blob.encode(blob.CheckpointMeta(fingerprint=fp, token_count=2, tokens=[1, 2]), b"kv")
    m, payload = blob.decode(v1)
    assert payload == b"kv"
    assert m.payload_sha256 is None
    # foreign version: refused, named
    v3 = blob.encode(
        blob.CheckpointMeta(fingerprint=fp, token_count=2, tokens=[1, 2], format_version=3),
        b"kv",
    )
    with pytest.raises(ValueError, match="format version 3"):
        blob.decode(v3)
