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
    # KV quantization tier (T12 adoption, mirrors the Rust ModelFingerprint's
    # serde defaults): 0/0 = f16. Lines 1-2 below strip fields a NEWER daemon
    # wrote that this sidecar does not know (forward compat), then the
    # explicit kwargs adopt the two fields written by the current daemon —
    # WITHOUT them, Fingerprint(**raw['fingerprint']) raises TypeError on
    # every blob the T12-era daemon publishes (observed as a 422 quarantine
    # of every healthy ancestor: delta-prefill e2e regressed to 'miss').
    # Being strict about UNKNOWN fields is wrong here: the daemon serde serializes
    # the fingerprint struct it has; this reader must accept a superset and
    # preserve what it recognizes.
    kv_bits: int = 0
    kv_group_size: int = 0


def _known_fields(data: dict) -> dict:
    """Drop keys this sidecar does not model (forward compatibility)."""
    import dataclasses

    known = {f.name for f in dataclasses.fields(Fingerprint)}
    return {k: v for k, v in data.items() if k in known}

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
    # The header is untrusted: a fingerprint of the wrong JSON type (null, a
    # list, a number) would raise AttributeError from .items() inside
    # _known_fields — a 500-forever poison instead of a clean rejection.
    # Validate the shape HERE so every malformed header is a ValueError.
    if not isinstance(raw["fingerprint"], dict):
        raise ValueError("fingerprint must be an object")
    meta = CheckpointMeta(
        fingerprint=Fingerprint(**_known_fields(raw["fingerprint"])),
        token_count=raw["token_count"],
        tokens=raw.get("tokens", []),
        format_version=raw["format_version"],
    )
    if meta.format_version != FORMAT_VERSION:
        raise ValueError(f"format version {meta.format_version}")
    return meta, blob[4 + header_len :]
