"""Checkpoint blob codec — byte-compatible with the Rust daemon's persistence.rs.

Format: u32 LE header length + JSON header + payload.
Header mirrors mlxcache_core::contract::CheckpointMeta exactly (serde field names).
"""

from __future__ import annotations

import json
import struct
from dataclasses import asdict, dataclass, field

FORMAT_VERSION = 1


@dataclass
class Fingerprint:
    model_id: str
    tokenizer_hash: str
    kv_dtype: str
    kv_layout_version: int


@dataclass
class CheckpointMeta:
    fingerprint: Fingerprint
    token_count: int
    tokens: list[int] = field(default_factory=list)
    format_version: int = FORMAT_VERSION


def encode(meta: CheckpointMeta, payload: bytes) -> bytes:
    header = json.dumps(asdict(meta), separators=(",", ":")).encode()
    return struct.pack("<I", len(header)) + header + payload


def decode(blob: bytes) -> tuple[CheckpointMeta, bytes]:
    if len(blob) < 4:
        raise ValueError("truncated header length")
    (header_len,) = struct.unpack("<I", blob[:4])
    if 4 + header_len > len(blob):
        raise ValueError("header length exceeds blob")
    raw = json.loads(blob[4 : 4 + header_len])
    meta = CheckpointMeta(
        fingerprint=Fingerprint(**raw["fingerprint"]),
        token_count=raw["token_count"],
        tokens=raw.get("tokens", []),
        format_version=raw["format_version"],
    )
    if meta.format_version != FORMAT_VERSION:
        raise ValueError(f"format version {meta.format_version}")
    return meta, blob[4 + header_len :]
