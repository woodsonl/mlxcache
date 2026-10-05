"""Fingerprint + the normative key fold (connector protocol §4).

The fold is byte-identical to the Rust implementation
(crates/mlxcache-daemon/src/http.rs :: blob_key / prefix_hash / fnv1a):
length-prefixed fields → u32-LE chunks → domain separators → dual-lane
FNV-1a. Cross-language keys are pinned by the golden constants in
sidecar/tests/test_store.py and crates/…/tests (B0.2).
"""

from __future__ import annotations

import struct

# One fingerprint definition across the codebase: the codec's own
# dataclass (identical fields/defaults; nothing hashes it — unhashable).
from mlxcache_sidecar.blob import Fingerprint  # noqa: F401 — re-export

MASK64 = (1 << 64) - 1
FNV1A_BASIS = 0xCBF29CE484222325
FNV1A_PRIME = 0x00000100000001B3
FNV1_SEED = 0x9E3779B97F4A7C15
FNV1_PRIME = 0x880355F21E6D1965
DEFAULT_ENGINE = "mlx-lm"
_DOMAIN_SEP = 0xFFFFFFFF


def _fnv1a(seed: int, prime: int, words: list[int]) -> int:
    h = seed
    for w in words:
        h ^= w & 0xFFFFFFFF
        h = (h * prime) & MASK64
    return h


def _prefix_hash(words: list[int]) -> int:
    lo = _fnv1a(FNV1A_BASIS, FNV1A_PRIME, words)
    hi = _fnv1a(FNV1_SEED, FNV1_PRIME, words)
    return (hi << 64) | lo


def _chunks_of_concat(parts: list[str]) -> list[int]:
    """Concatenate `len:field` encodings, then chunk the WHOLE buffer into
    zero-padded u32-LE words — the Rust fold chunks once across all field
    boundaries (per-field chunking would pad differently)."""
    b = "".join(f"{len(p.encode())}:{p}" for p in parts).encode()  # len in BYTES (Rust String::len)
    out = []
    for i in range(0, len(b), 4):
        chunk = b[i : i + 4]
        out.append(struct.unpack("<I", chunk.ljust(4, b"\x00"))[0])
    return out


def blob_key(fp: Fingerprint, engine_id: str | None, tokens: list[int]) -> int:
    """The 128-bit store key (protocol §4, normative by construction).

    Non-default engines fold their length-prefixed id closed by a domain
    separator BEFORE all fingerprint bytes; the default engine (and None)
    contributes zero bytes, so legacy keys are byte-identical.
    """
    words: list[int] = []
    engine = engine_id or DEFAULT_ENGINE
    if engine != DEFAULT_ENGINE:
        words.extend(_chunks_of_concat([engine]))
        words.append(_DOMAIN_SEP)
    words.extend(
        _chunks_of_concat(
            [
                fp.model_id,
                fp.tokenizer_hash,
                fp.kv_dtype,
                str(fp.kv_layout_version),
                str(fp.kv_bits),
                str(fp.kv_group_size),
            ]
        )
    )
    words.append(_DOMAIN_SEP)
    words.extend(t & 0xFFFFFFFF for t in tokens)
    return _prefix_hash(words)
