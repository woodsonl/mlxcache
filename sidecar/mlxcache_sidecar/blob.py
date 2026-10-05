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


MAX_HEADER_BYTES = 1 << 20  # headers are tiny JSON; larger = malformed (§6)


class HeaderRejected(ValueError):  # noqa: N818 — protocol §3.4 taxonomy kind carrier
    """A deterministically invalid header, with its taxonomy kind (§3.4):
    "version" (version policy) or "header" (shape/schema). Message text is
    stable for callers that match on it."""

    def __init__(self, kind: str, message: str):
        self.kind = kind
        super().__init__(message)


def decode_header(header: bytes) -> CheckpointMeta:
    """Parse + validate a header slice. EVERY malformed shape raises
    HeaderRejected (a ValueError): the header is untrusted on-disk bytes
    and a KeyError/TypeError/RecursionError escape would poison-brick
    every reader (§6). Validated: container types, tokens as a u32 list
    (bools are not ints), granularity in {0,1}, string engine_id /
    payload_sha256, int token_count (advisory) and format_version."""
    try:
        raw = json.loads(header)
    except RecursionError as exc:
        raise HeaderRejected("header", "header too deeply nested") from exc
    if not isinstance(raw, dict):
        raise HeaderRejected("header", "header must be a JSON object")
    if not isinstance(raw.get("fingerprint"), dict):
        raise HeaderRejected("header", "fingerprint must be an object")
    try:
        meta = CheckpointMeta(
            fingerprint=Fingerprint(**_known_fields(raw["fingerprint"])),
            token_count=raw["token_count"],
            tokens=raw.get("tokens", []),
            format_version=raw["format_version"],
            payload_sha256=raw.get("payload_sha256"),
            engine_id=raw.get("engine_id"),
            granularity=raw.get("granularity"),
        )
    except (KeyError, TypeError) as exc:
        raise HeaderRejected("header", f"missing or invalid header field: {exc}") from exc
    fp = meta.fingerprint
    if (
        not isinstance(fp.model_id, str)
        or not isinstance(fp.tokenizer_hash, str)
        or not isinstance(fp.kv_dtype, str)
        or type(fp.kv_layout_version) is not int
        or type(fp.kv_bits) is not int
        or type(fp.kv_group_size) is not int
    ):
        # The dataclass does not enforce types; a non-str field would
        # explode later in the fold (.encode()) outside every guarded
        # region — the nested poison-pill path.
        raise HeaderRejected("header", "fingerprint field types are invalid")
    if not isinstance(meta.tokens, list) or not all(
        type(t) is int and 0 <= t <= 0xFFFFFFFF for t in meta.tokens
    ):
        # tokens is the AUTHORitative prefix (token_count is advisory —
        # the sidecar's recorded-prefix-wins rule); it must be a u32 list.
        raise HeaderRejected("header", "tokens must be a list of u32")
    if type(meta.token_count) is not int:
        raise HeaderRejected("header", "token_count must be an integer")
    if type(meta.format_version) is not int:
        raise HeaderRejected("header", "format_version must be an integer")
    if meta.granularity is not None and (
        type(meta.granularity) is not int or meta.granularity not in (0, 1)
    ):
        raise HeaderRejected("header", "granularity must be 0 or 1")
    if meta.engine_id is not None and not isinstance(meta.engine_id, str):
        raise HeaderRejected("header", "engine_id must be a string")
    if meta.payload_sha256 is not None and not isinstance(meta.payload_sha256, str):
        raise HeaderRejected("header", "payload_sha256 must be a string")
    # Version policy mirrors the daemon's load(): 1 = legacy (no digest
    # field), 2 = current (D1 digest contract; the digest is REQUIRED —
    # serving a v2 header without one would be unverified). Anything else
    # is a layout this reader must not interpret.
    if meta.format_version not in (1, FORMAT_VERSION_D1):
        raise HeaderRejected("version", f"format version {meta.format_version}")
    if meta.format_version == FORMAT_VERSION_D1 and meta.payload_sha256 is None:
        raise HeaderRejected("version", "format version 2 requires payload_sha256")
    return meta


def decode(blob: bytes) -> tuple[CheckpointMeta, bytes]:
    if len(blob) < 4:
        raise ValueError("truncated header length")
    (header_len,) = struct.unpack("<I", blob[:4])
    if header_len > MAX_HEADER_BYTES:
        raise HeaderRejected("header", f"header length {header_len} exceeds cap")
    if 4 + header_len > len(blob):
        raise ValueError("header length exceeds blob")
    return decode_header(blob[4 : 4 + header_len]), blob[4 + header_len :]
