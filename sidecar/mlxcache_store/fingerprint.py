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


def cache_layout_compatible(entry_layout: str, request_layout: str) -> bool:
    """Cache-composition compatibility (the B1 identity rule).

    `cache_layout` is a signature of the ordered tuple of cache-entry classes
    an engine builds for a model — the SHAPE of the persisted payload. `""`
    means "composition unknown", the value every pre-B1 blob carries.

    An entry is compatible only with a request reporting the SAME layout:

    - a request with a KNOWN (non-empty) layout is never served an
      unknown-layout entry — nothing proves that payload has the shape this
      engine will deserialize, and a dense KV stack loaded as a hybrid
      model's recurrent state is silent wrong output, not an error;
    - an unknown-layout request may still be served an unknown-layout entry,
      so a store written before layouts were recorded keeps working until the
      engine rewrites its blobs (legacy-to-legacy only: the moment either
      side knows its composition the comparison is exact).
    """
    return entry_layout == request_layout


def content_digest_compatible(entry_digest: str, request_digest: str) -> bool:
    """Non-text content identity (the B2 rule): exact equality.

    Not the legacy-tolerant rule that `cache_layout` uses. `""` means "this
    request carries no non-text content" — a claim, not a missing measurement,
    since the daemon computes it from the request before lookup. So a
    text-only entry and a text-only request agree, and a text-only entry can
    never serve a request carrying pixels (or any other modality).
    """
    return entry_digest == request_digest


def identity_compatible(entry: Fingerprint, request: Fingerprint) -> bool:
    """The single identity relation every matching site uses. Field-by-field
    on purpose: the composition rule is a named decision, not an accident of
    dataclass equality, so adding a fingerprint field cannot silently drop
    it."""
    return (
        entry.model_id == request.model_id
        and entry.tokenizer_hash == request.tokenizer_hash
        and entry.kv_dtype == request.kv_dtype
        and entry.kv_layout_version == request.kv_layout_version
        and entry.kv_bits == request.kv_bits
        and entry.kv_group_size == request.kv_group_size
        and cache_layout_compatible(entry.cache_layout, request.cache_layout)
        and content_digest_compatible(entry.content_digest, request.content_digest)
        and entry.kv_tier == request.kv_tier
    )


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

    The same zero-byte rule applies to `fingerprint.cache_layout`: a NON-EMPTY
    layout folds as one more length-prefixed field AFTER kv_group_size and
    before the closing domain separator, so two cache compositions of the same
    model can never share a key; an EMPTY layout (composition unrecorded, as
    on every pre-B1 blob) contributes zero bytes and leaves existing keys
    byte-identical.
    """
    words: list[int] = []
    engine = engine_id or DEFAULT_ENGINE
    if engine != DEFAULT_ENGINE:
        words.extend(_chunks_of_concat([engine]))
        words.append(_DOMAIN_SEP)
    fields = [
        fp.model_id,
        fp.tokenizer_hash,
        fp.kv_dtype,
        str(fp.kv_layout_version),
        str(fp.kv_bits),
        str(fp.kv_group_size),
    ]
    if fp.cache_layout:
        fields.append(fp.cache_layout)
    # Content identity, folded last (the same position as the Rust fold) and
    # only when recorded: an empty digest adds zero bytes, keeping text-only
    # keys byte-identical to the pre-digest fold.
    if fp.content_digest:
        fields.append(fp.content_digest)
    # Named KV tier, folded only when present and LAST among fingerprint
    # fields (the same position as the Rust fold): an empty tier adds zero
    # bytes, keeping pre-tier keys byte-identical; a TQ tier namespaces its
    # own blobs.
    if fp.kv_tier:
        fields.append(fp.kv_tier)
    words.extend(_chunks_of_concat(fields))
    words.append(_DOMAIN_SEP)
    words.extend(t & 0xFFFFFFFF for t in tokens)
    return _prefix_hash(words)
