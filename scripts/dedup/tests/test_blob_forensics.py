"""Hermetic invariants for the blob forensics primitives.

Every committed table routes through load_blob's framing, parse_safetensors'
offset math, and first_diff's bisection; a bug here corrupts all
forensics/chunk/evict evidence while CI stays green (the analysis tests
hand-build tensor dicts). These pin the primitives to hand-framed inputs."""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blob_forensics import first_diff, kv_heads, layout, load_blob, parse_safetensors  # noqa: E402


def st_payload(tensors: dict[str, bytes]) -> bytes:
    off, header = 0, {}
    for name, b in tensors.items():
        header[name] = {"dtype": "U8", "shape": [len(b)], "data_offsets": [off, off + len(b)]}
        off += len(b)
    hj = json.dumps(header).encode()
    return struct.pack("<Q", len(hj)) + hj + b"".join(tensors.values())


def test_parse_safetensors_offsets_are_data_relative():
    payload = st_payload({"a": b"1234", "b": b"567890"})
    tensors, data_start = parse_safetensors(payload)
    assert data_start == len(payload) - 10
    assert payload[tensors["a"]["begin"] : tensors["a"]["end"]] == b"1234"
    assert payload[tensors["b"]["begin"] : tensors["b"]["end"]] == b"567890"


def test_load_blob_framing_roundtrip(tmp_path):
    meta = {"token_count": 5, "fingerprint": {"kv_bits": 0}}
    h = json.dumps(meta).encode()
    p = tmp_path / "x.ckpt"
    p.write_bytes(struct.pack("<I", len(h)) + h + b"PAYLOAD")
    assert load_blob(p) == (meta, b"PAYLOAD")


def test_first_diff_edges():
    a = bytes(5000)
    assert first_diff(a, a) is None
    assert first_diff(a, a + b"x") is None  # strict prefix
    assert first_diff(a, b"\x01" + a[1:]) == 0
    assert first_diff(a, a[:-1] + b"\x01") == 4999


def test_kv_heads_shape_rule():
    assert kv_heads({"shape": [1, 4, 7434, 128]}) == 4
    assert kv_heads({"shape": [1, 4, 7434, 32]}) == 4
    assert kv_heads({"shape": [7434]}) == 1  # non-KV tensor: one stream


def test_layout_framing_from_file_not_reserialization(tmp_path):
    """framing_bytes must be the file's actual header size, not a
    re-serialized guess (json.dumps spacing need not match the writer)."""
    meta = {"token_count": 3, "fingerprint": {"kv_bits": 0}}
    h = json.dumps(meta, separators=(",", ":")).encode()  # compact, unlike json.dumps
    payload = st_payload({"k": b"xyz"})
    p = tmp_path / "x.ckpt"
    framing = 4 + len(h)
    p.write_bytes(struct.pack("<I", len(h)) + h + payload)
    m, pl = load_blob(p)
    tensors, _ = parse_safetensors(pl)
    lay = layout(m, tensors, len(pl), framing)
    assert lay["framing_bytes"] == framing
    assert lay["framing_bytes"] != 4 + len(json.dumps(m)) or framing == 4 + len(json.dumps(m))
