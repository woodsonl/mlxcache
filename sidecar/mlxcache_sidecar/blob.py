"""Checkpoint blob codec — byte-compatible with the Rust daemon's persistence.rs.

Format: u32 LE header length + JSON header + payload.
Header mirrors mlxcache_core::contract::CheckpointMeta exactly (serde field names).
"""

from __future__ import annotations

import json
import struct
from dataclasses import asdict, dataclass, field

FORMAT_VERSION = 1
# D1 integrity: version 2 adds payload_sha256 (written by the daemon at
# publish, verified at load). Version 1 blobs predate it and stay readable.
FORMAT_VERSION_D1 = 2


@dataclass
class Fingerprint:
    model_id: str
    tokenizer_hash: str
    kv_dtype: str
    kv_layout_version: int
    # KV quantization tier (T12 adoption, mirrors the Rust ModelFingerprint's
    # serde defaults): 0/0 = f16. _known_fields() (this module) strips fields a
    # NEWER daemon wrote that this sidecar does not know, at decode time
    # (forward compat); the two defaulted dataclass fields adopt the fields the
    # current daemon writes — WITHOUT them, Fingerprint(**_known_fields(...))
    # raises TypeError on every blob the T12-era daemon publishes (observed as
    # a 422 quarantine of every healthy ancestor: delta-prefill e2e regressed
    # to 'miss').
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
    # D1 integrity (format_version 2, written by the daemon at publish):
    # sha256 hex of the payload bytes. None on legacy v1 blobs — unverified.
    payload_sha256: str | None = None
    # Connector protocol §2/§4 [B0.2]: engine identity for namespaced keys
    # (None = the default engine "mlx-lm", which contributes zero fold
    # bytes) and resume-granularity declaration (None = 0 = ANY_PREFIX).
    engine_id: str | None = None
    granularity: int | None = None


def encode(meta: CheckpointMeta, payload: bytes) -> bytes:
    # None-valued fields are OMITTED, matching the Rust writer's
    # skip_serializing_if (byte-exactness across bindings, protocol §2:
    # absent means default). asdict alone would emit `"engine_id": null`.
    raw = {k: v for k, v in asdict(meta).items() if v is not None}
    header = json.dumps(raw, separators=(",", ":")).encode()
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
        payload_sha256=raw.get("payload_sha256"),
        engine_id=raw.get("engine_id"),
        granularity=raw.get("granularity"),
    )
    # Version policy mirrors the daemon's load(): 1 = legacy (no digest
    # field), 2 = current (D1 digest contract). Anything else is a layout
    # this reader must not interpret.
    if meta.format_version not in (1, FORMAT_VERSION_D1):
        raise ValueError(f"format version {meta.format_version}")
    return meta, blob[4 + header_len :]
