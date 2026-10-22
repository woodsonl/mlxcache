"""Sidecar HTTP server: the mlx-lm compatibility adapter.

Endpoints (daemon -> sidecar protocol):
- POST /tokenize  {"prompt": str} -> {"tokens": [u32], "tokenizer_hash": str,
                   "kv_dtype": str, "kv_bits": int, "kv_group_size": int,
                   "engine_id": str, "cache_layout": str, "kv_tier": str}
- POST /prefill   {"tokens": [u32], "ancestor_blob_path": str|null}
                   -> KV checkpoint payload (binary; EMPTY body = nothing
                   cacheable). 400 on bad input; 422 when a declared ancestor
                   is missing or unloadable (the daemon quarantines it and
                   retries from scratch).
- POST /generate  {"tokens": [u32], "max_tokens": int, "blob_path": str|null,
                   "stream": bool, "settle_path": str|null} -> NDJSON token
                   stream (or a JSON body when stream is false). `settle_path`
                   asks the engine to persist the POST-GENERATION cache — the
                   prompt plus the tokens it just produced — as a wire
                   checkpoint at that path, and to report the outcome in the
                   response (`settle`, on the JSON body and on the terminal
                   done frame). The cache only exists inside this request, so
                   it must be captured here; a separate endpoint could not
                   rebuild it from tokens.

Engine selection: MLXCACHE_ENGINE=synthetic (default, hermetic) or mlx-lm
(requires a local model). The daemon is the only client; this
server binds 127.0.0.1 by default.

Run: uv run python -m mlxcache_sidecar.server
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import PurePath


class CheckpointRejectedError(Exception):
    """A checkpoint the daemon asked us to resume from is bad: corrupt payload,
    undecodable header, or a recorded prefix that disagrees with the request.
    Mapped to HTTP 422 so the daemon retires (quarantines) the entry. Distinct
    from a decode/generation failure, which must NOT retire a healthy
    checkpoint."""


# Phase timers (B2). Aggregates, not per-request traces: /stats exposes totals
# + counts so the benchmark harness can decompose TTFT. The GIL makes dict
# increments safe under ThreadingHTTPServer's thread-per-request model (same
# reasoning as prefill_count); the lock keeps multi-field updates coherent for
# the reader.
_phase_lock = threading.Lock()
_phase_totals: dict[str, float] = {}
_phase_counts: dict[str, int] = {}


def _phase_add(name: str, ms: float) -> None:
    """Record one operation of phase `name` taking `ms` wall milliseconds.
    Aggregates `{name}_total` (ms) and `{name}_count`."""
    with _phase_lock:
        key = f"{name}_total"
        _phase_totals[key] = _phase_totals.get(key, 0.0) + ms
        _phase_counts[name] = _phase_counts.get(name, 0) + 1


def _phase_stats() -> dict[str, float]:
    out: dict[str, float] = {}
    with _phase_lock:
        for name, count in _phase_counts.items():
            total = _phase_totals.get(f"{name}_total", 0.0)
            out[f"{name}_count"] = count
            out[f"{name}_ms_total"] = round(total, 3)
            out[f"{name}_ms_avg"] = round(total / (count or 1), 3)
    return out


_SAFETENSORS_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
}


def _reject_json_constant(name: str) -> None:
    """Reject NaN/Infinity: the JSON spec forbids them and MLX's parser does too,
    but Python's json accepts them by default."""
    raise ValueError(f"non-standard JSON constant: {name}")


def _has_lone_surrogate(obj: object) -> bool:
    """True if any string anywhere in the parsed JSON contains a lone surrogate
    (U+D800-U+DFFF), which is not valid UTF-8 and which MLX rejects."""
    if isinstance(obj, str):
        return any("\ud800" <= c <= "\udfff" for c in obj)
    if isinstance(obj, dict):
        return any(_has_lone_surrogate(k) or _has_lone_surrogate(v) for k, v in obj.items())
    if isinstance(obj, list):
        return any(_has_lone_surrogate(v) for v in obj)
    return False


def _valid_safetensors(payload: bytes) -> bool:
    """Structural check of a safetensors blob: an 8-byte little-endian header
    length, then that many bytes of JSON object, then each tensor's dtype, shape,
    and `data_offsets` consistent with the remaining data. Deterministic
    corruption is caught here so it can be classified as a rejection regardless
    of which exception the native loader would raise."""
    if len(payload) < 8:
        return False
    n = int.from_bytes(payload[:8], "little")
    if n == 0 or 8 + n > len(payload):
        return False
    try:
        # Match MLX's strict parser, not Python's lenient one. Python accepts
        # UTF-16 headers, lone surrogates, and NaN/Infinity; MLX rejects all
        # three with a RuntimeError we would misread as transient. Decode as
        # strict UTF-8 and forbid the non-standard numeric constants.
        text = payload[8 : 8 + n].decode("utf-8")
        header = json.loads(text, parse_constant=_reject_json_constant)
        # A lone surrogate (a \uD800-\uDFFF escape) parses fine in Python but is
        # not valid UTF-8; MLX rejects it. The strict decode above catches raw
        # bytes; this catches escaped surrogates in keys and values. It is inside
        # the guard because the traversal recurses: a deeply nested header raises
        # RecursionError here, and an escaping RecursionError is a RuntimeError
        # the daemon would misread as transient (500) instead of retiring it.
        if _has_lone_surrogate(header):
            return False
    except (ValueError, TypeError, RecursionError):
        return False
    if not isinstance(header, dict):
        return False
    data_len = len(payload) - 8 - n
    ranges: list[tuple[int, int]] = []
    for name, spec in header.items():
        if name == "__metadata__":
            # Metadata is a string->string map; anything else, MLX rejects with a
            # RuntimeError we would misread as transient.
            if not isinstance(spec, dict) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in spec.items()
            ):
                return False
            continue
        if not isinstance(spec, dict):
            return False
        dtype = spec.get("dtype")
        shape = spec.get("shape")
        offs = spec.get("data_offsets")
        if not isinstance(dtype, str) or dtype not in _SAFETENSORS_DTYPE_BYTES:
            return False
        if not isinstance(shape, list) or not all(
            isinstance(d, int) and not isinstance(d, bool) and d >= 0 for d in shape
        ):
            return False
        if (
            not isinstance(offs, list)
            or len(offs) != 2
            or not all(isinstance(o, int) and not isinstance(o, bool) for o in offs)
        ):
            return False
        start, end = offs
        elements = 1
        for d in shape:
            elements *= d
        expected = elements * _SAFETENSORS_DTYPE_BYTES[dtype]
        if start < 0 or end < start or end > data_len or end - start != expected:
            return False
        ranges.append((start, end))
    # Safetensors forbids overlapping tensor ranges; the native loader rejects
    # them. Two tensors claiming the same bytes would otherwise pass here and
    # raise a RuntimeError we misread as transient.
    ranges.sort()
    return all(ranges[i][0] >= ranges[i - 1][1] for i in range(1, len(ranges)))


def _payload_digest_ok(meta: object, payload: bytes) -> bool:
    """True when the payload matches the daemon-stamped digest (D1).

    Verified on EVERY load, no per-path caching: the bytes are already in
    hand here (one hash pass over memory the process already paid to read),
    and a once-per-process cache would make the check vacuous for exactly the
    hottest blobs — a blob verified once could rot in place and keep serving
    corrupt KV with a `hit` verdict until the next daemon restart (red-team
    2026-10-04). Legacy v1 blobs carry no digest and pass unverified; the
    daemon's boot sweep applies the same policy to the whole store. A v2+
    header WITHOUT a digest is corrupt (the field is part of the v2
    contract, and the daemon refuses the same blob at boot) — reject rather
    than serve unverified.
    """
    if getattr(meta, "format_version", 1) >= 2 and getattr(meta, "payload_sha256", None) is None:
        return False
    expected = getattr(meta, "payload_sha256", None)
    if expected is None:
        return True
    return hashlib.sha256(payload).hexdigest() == expected.lower()


# ---------------------------------------------------------------------------
# Post-generation settling
#
# The KV for a request exists only inside the request that built it: /prefill
# constructs it and returns, and /generate constructs or loads it and returns.
# Nothing outside those calls can reconstruct it from token ids, which is why a
# settled checkpoint has to be produced and handed out from INSIDE the generate
# request instead of from a separate endpoint.
#
# A settle keeps the repository's coverage convention: a persisted cache covers
# `meta.tokens[:-1]`, so the reader's `covered = len(meta.tokens) - 1` tells the
# next turn exactly which positions are already resident. The cache an engine
# holds at the end of generation may sit one position deeper than that (the
# sampler feeds the last generated token to produce the token AFTER it, which
# is then discarded), so coverage is MEASURED and trimmed to the convention
# rather than assumed — an off-by-one here makes the next turn feed a token the
# cache already holds, which degrades output without failing any structural
# check.
# ---------------------------------------------------------------------------

# Generous but finite: settling is best-effort work on the request's critical
# path, so it must never be unbounded. The token cap bounds positions (and the
# payload that follows from them); the byte cap bounds a payload whose shape
# does not follow the token count (wide KV heads, many layers).
SETTLE_MAX_TOKENS_DEFAULT = 32768
SETTLE_MAX_BYTES_DEFAULT = 2 * 1024 * 1024 * 1024


def _settle_limits() -> tuple[int | None, int | None]:
    """(max_tokens, max_bytes) for one settled checkpoint. A malformed value
    falls back to the default rather than disabling settling silently.

    An explicitly-set 0 means REFUSE EVERYTHING, not "no cap": `0` is the
    natural "turn this off" for the two knobs that bound on-disk writes, so
    treating it as falsy-disabled would invert the operator's intent. An
    unset or unparsable variable keeps the default — a malformed value must
    never leave the caps silently off."""

    def read(name: str, default: int) -> int:
        raw = os.environ.get(name)
        if raw is None or raw == "":
            return default
        try:
            return int(raw)
        except ValueError:
            return default

    return (
        read("MLXCACHE_SETTLE_MAX_TOKENS", SETTLE_MAX_TOKENS_DEFAULT),
        read("MLXCACHE_SETTLE_MAX_BYTES", SETTLE_MAX_BYTES_DEFAULT),
    )


def _settle_fingerprint(engine: object) -> object:
    """The header fingerprint a settled checkpoint carries.

    Identity is owned by the DAEMON (it is the side that keys and publishes),
    so this header is not the authority for routing: it is what makes the file
    a self-describing wire checkpoint the daemon can decode, cross-check
    against the identity it already resolved for the request, and re-stamp
    with its own metadata. Writing the engine's own identity keeps the two
    comparable instead of forcing the daemon to trust an opaque blob.
    """
    from .blob import Fingerprint  # noqa: PLC0415

    return Fingerprint(
        model_id=str(getattr(engine, "model_id", "")),
        tokenizer_hash=str(getattr(engine, "tokenizer_hash", "")),
        kv_dtype=str(getattr(engine, "kv_dtype", "unknown")),
        kv_layout_version=1,
        kv_bits=int(getattr(engine, "kv_bits", 0) or 0),
        kv_group_size=int(getattr(engine, "kv_group_size", 0) or 0),
        cache_layout=str(getattr(engine, "cache_layout", "") or ""),
        kv_tier=str(getattr(engine, "kv_tier", "") or ""),
    )


def settle_checkpoint(
    engine: object,
    settle_path: str,
    tokens: list[int],
    payload: bytes,
    *,
    covered: int | None = None,
) -> dict:
    """Write one post-generation checkpoint (wire format) and report the outcome.

    `tokens` is the FULL sequence the checkpoint's cache is described by
    (prompt + generated); the cache itself covers `tokens[:-1]` by the
    repository's convention. `covered` is the measured number of positions the
    cache actually holds; the caller has already brought it to
    `len(tokens) - 1` (or declines to settle), so this function only records
    it.

    Never raises. A settle is best-effort work: the model's answer is already
    produced, and a failed settle (ENOSPC, a bad path, a refused write) must
    not turn a successful completion into an error. The returned report says
    what happened so the daemon can log the reason and so "wrote nothing" is
    never mistaken for "wrote something".
    """
    max_tokens, max_bytes = _settle_limits()
    if len(tokens) < 2:
        # A prefix shorter than two tokens caches nothing (mirrors prefill and
        # read_wire_checkpoint): publishing it would index a checkpoint that
        # can never be resumed from.
        return {"written": False, "tokens": len(tokens), "reason": "too_short"}
    if str(getattr(engine, "tokenizer_hash", "")) == "missing":
        # No cryptographic vocabulary identity: the engine's fingerprint is
        # the non-persistence sentinel, and a settled blob under it would
        # key reusable KV on an unverifiable identity. Fail closed.
        return {"written": False, "tokens": len(tokens), "reason": "identity_unavailable"}
    if max_tokens is not None and len(tokens) > max_tokens:
        return {"written": False, "tokens": len(tokens), "reason": "token_cap"}
    if not payload:
        return {"written": False, "tokens": len(tokens), "reason": "empty_payload"}
    if max_bytes is not None and len(payload) > max_bytes:
        return {"written": False, "tokens": len(tokens), "reason": "size_cap"}

    from .blob import CheckpointMeta, encode  # noqa: PLC0415

    meta = CheckpointMeta(
        fingerprint=_settle_fingerprint(engine),
        token_count=len(tokens),
        tokens=list(tokens),
        # No payload digest here: the daemon stamps format_version 2 and the
        # digest at its single publish choke point (publish_atomic), exactly as
        # it does for a prefill payload. Duplicating that here would create a
        # second, independently-maintained integrity contract.
        engine_id=_engine_id_meta(engine),
    )
    wire = encode(meta, payload)
    digest = hashlib.sha256(payload).hexdigest()
    try:
        _atomic_write(settle_path, wire)
    except OSError as exc:
        return {"written": False, "tokens": len(tokens), "reason": f"write_failed: {exc}"}
    return {
        "written": True,
        "tokens": len(tokens),
        "covered": len(tokens) - 1 if covered is None else covered,
        "sha256": digest,
        "bytes": len(wire),
    }


def _atomic_write(path: str, data: bytes) -> None:
    """Write `data` to `path` so a concurrent reader (the daemon reading the
    staging file after this response) never sees a torn write, without
    requiring the caller to coordinate a lock. mkstemp gives the uniqueness:
    a FIXED ".part" name would let two overlapping requests (a daemon retry
    racing an abandoned attempt) corrupt each other's temp.

    A failed write leaves nothing behind: a stranded temp in the blob dir
    would be invisible to every reader and accumulate per request. mkstemp
    can itself fail (a read-only blob dir), in which case there is no temp;
    cleanup errors during that failure must not mask the original OSError."""
    tmp_path: str | None = None
    try:
        fd, tmp_path = tempfile.mkstemp(
            dir=os.path.dirname(path) or ".",
            prefix=os.path.basename(path) + ".",
            suffix=".part",
        )
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except OSError:
        if tmp_path is not None:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
        raise


def _engine_id_meta(engine: object) -> str | None:
    """`engine_id` for the header, using the same zero-byte default rule as the
    daemon: the default engine records nothing so its header stays
    byte-identical to a pre-namespacing one."""
    engine_id = str(getattr(engine, "engine_id", "mlx-lm") or "mlx-lm")
    return None if engine_id == "mlx-lm" else engine_id


def _is_staging_shaped(path: str) -> bool:
    """True when `path` is NAMED like the daemon's own settle staging path:
    basename `.settle-<pid>-<seq>-<nanos>.staging`, no traversal components.
    Anything else is not a path the daemon would ever send."""
    import re

    if re.fullmatch(r"\.settle-\d+-\d+-\d+\.staging", os.path.basename(path)) is None:
        return False
    return not any(part in ("..", ".") for part in PurePath(path).parts)


def _staging_dir_matches(path: str) -> bool:
    """True when `path`'s parent directory is the pinned staging directory.

    The pin is the daemon's blob dir in real deployments. A mismatch means
    misconfiguration (the daemon restarted against a new blob dir while the
    old one still exists), not an attack: the caller never gets to write
    anywhere either way, because the mismatch DEGRADES the settle rather
    than the request -- generation has value, a best-effort settle does not.
    """
    parent = os.path.realpath(os.path.dirname(path) or ".")
    with _SETTLE_DIR_LOCK:
        pinned = _SETTLE_DIR_PIN[0]
        # Re-pin only when the pinned directory is GONE: a live directory can
        # never be displaced by another caller's path.
        if pinned is not None and pinned != parent and not os.path.isdir(pinned):
            pinned = None
            _SETTLE_DIR_PIN[0] = None
        if pinned is None:
            _SETTLE_DIR_PIN[0] = parent
            pinned = parent
        return pinned == parent


_SETTLE_DIR_LOCK = threading.Lock()
_SETTLE_DIR_PIN: list[str | None] = [None]


def _cache_positions(cache) -> int | None:
    """How many positions a prompt cache actually holds, or None when the
    composition cannot report it.

    Measured, never assumed: the sampler's lookahead means a finished
    generation can leave the cache exactly one position deeper than the
    `tokens[:-1]` convention, and the amount differs with where the loop
    stopped (token budget, EOS).
    """
    if cache is None:
        return None
    try:
        entries = list(cache)
    except TypeError:
        return None
    if not entries:
        return None
    sizes: list[int] = []
    for entry in entries:
        size = getattr(entry, "size", None)
        if callable(size):
            sizes.append(int(size()))
        elif isinstance(getattr(entry, "offset", None), int):
            sizes.append(int(entry.offset))
        else:
            return None
    return max(sizes)


def _trim_cache_to(cache, target: int) -> bool:
    """Trim `cache` in place down to `target` positions. False when the
    composition cannot be trimmed (recurrent/SSM hybrid state has no
    per-position trim), or when trimming does not reach the target."""
    from mlx_lm.models.cache import can_trim_prompt_cache, trim_prompt_cache  # noqa: PLC0415

    current = _cache_positions(cache)
    if current is None or current <= target:
        return current == target
    if not can_trim_prompt_cache(cache):
        return False
    trim_prompt_cache(cache, current - target)
    return _cache_positions(cache) == target


def read_wire_checkpoint(
    blob_path: str, tokens: list[int] | None = None, check_safetensors: bool = True
) -> tuple[object | None, bytes, bool]:
    """Read and validate a daemon wire-format checkpoint off disk.

    Shared trust boundary for BOTH engines (prefill-with-ancestor and generate
    resume). Returns ``(meta, payload, usable)``:

    - A corrupt/invalid checkpoint (bad header, prefix disagreement with
      `tokens`, empty payload, invalid safetensors) raises
      CheckpointRejectedError → HTTP 422 → the daemon quarantines the entry.
    - A legitimately uncacheable checkpoint (prefix shorter than 2 tokens) sets
      ``usable=False`` with meta/payload intact — the caller runs from scratch
      WITHOUT retiring the entry.
    - A transient OS read failure (EIO, ENFILE) propagates as OSError → HTTP
      500 → the daemon retries without retiring a healthy entry.

    When ``tokens`` is given the recorded prefix must equal ``tokens[:n]``; a
    mismatch is a mislabeled/corrupt entry and is REJECTED, never silently
    ignored (resuming from wrong KV would break identical-to-scratch parity).

    ``check_safetensors`` validates the safetensors framing of the payload —
    the real MLX engine requires it (a malformed inner header would otherwise
    raise the same RuntimeError type as a transient OS failure); the synthetic
    engine's deterministic KV bytes are not safetensors and pass False.
    """
    from .blob import decode  # noqa: PLC0415

    # Read first. A MISSING file is a raced/retired checkpoint (the daemon
    # only passes paths its index currently points at): reject it (422 →
    # quarantine + scratch retry), not a 500 — a 500 makes the daemon treat
    # it as transient, the client eats a 502, and the dead entry stays
    # selectable. EVERY OTHER OSError (EIO, ENFILE, ...) is transient: let it
    # propagate as a 500 — a failed stat/open must never retire a healthy
    # entry (see os.stat pre-checks in the handlers).
    try:
        with open(blob_path, "rb") as fh:
            raw = fh.read()
    except FileNotFoundError as exc:
        raise CheckpointRejectedError(f"blob missing: {blob_path}") from exc
    try:
        meta, payload = decode(raw)
        # The JSON header is untrusted: a field of the wrong type (e.g.
        # "tokens":123) must be rejected here, inside the boundary. Letting
        # len() raise a bare TypeError outside it would 500 forever and leave
        # the poison selectable.
        if not isinstance(meta.tokens, list) or not all(
            isinstance(t, int) and not isinstance(t, bool) for t in meta.tokens
        ):
            raise ValueError("tokens must be a list of integers")
    except (ValueError, KeyError, TypeError, AttributeError, RecursionError) as exc:
        # A header that does not decode or lacks required fields is a bad
        # checkpoint: retire it rather than 500 forever while it stays
        # selectable. AttributeError (fingerprint: null/[]/5 → .items() on a
        # non-dict) and RecursionError (deeply nested JSON) were both
        # reproducible escapes — outside this handler they are a 500 with the
        # poison left selectable (mirror _valid_safetensors, which already
        # covers them).
        raise CheckpointRejectedError(f"checkpoint header invalid: {exc}") from exc
    # The checkpoint prefix must be self-describing: only meta.tokens tells
    # us which prefix the KV actually covers. The daemon always writes
    # meta.tokens, so a blob the index matched WITHOUT them cannot be
    # verified against this request. Adopting it would trust it covers
    # tokens[:token_count] by construction; silently running scratch while
    # the daemon reports a hit would be an accounting lie. Reject it.
    if not meta.tokens:
        raise CheckpointRejectedError("checkpoint has no recorded token prefix")
    prefix_len = len(meta.tokens)
    if prefix_len < 2:
        # A prefix shorter than 2 caches nothing. This also rejects legacy
        # one-token checkpoints whose nonempty KV already holds that token:
        # adopting one and feeding the whole prompt would double-feed it.
        return meta, payload, False
    if tokens is not None:
        if prefix_len > len(tokens):
            return meta, payload, False
        # The adapter is a trust boundary: verify the blob really covers the
        # KV it claims for THIS request. T22 (prefix-stable multi-turn): the
        # persisted cache covers meta.tokens[:-1], and the KV at position j
        # depends only on tokens ..=j — so a request that diverges from the
        # recorded prefix at its LAST token still gets byte-identical KV for
        # every covered position (a multi-turn wire prompt that grew: the
        # previous turn's checkpoint key ends in a closing bracket the next
        # turn replaced). Divergence anywhere else — including the final
        # COVERED position — means the file does not match the index entry
        # that pointed at it (mislabeled/corrupt): retire it, do not
        # silently resume from or ignore wrong KV.
        if meta.tokens[:-1] != tokens[: prefix_len - 1]:
            raise CheckpointRejectedError("checkpoint prefix does not match the request")
    # A multi-token checkpoint with no KV payload is CORRUPT (a truncated
    # write), not uncacheable: reject so the daemon quarantines the entry.
    if not payload:
        raise CheckpointRejectedError("checkpoint payload is empty")
    # Validate the safetensors framing ourselves, BEFORE handing it to MLX.
    # MLX's native parser raises RuntimeError for a bad header length, the
    # same type it uses for a transient OS read failure, so we cannot
    # classify from the exception alone. A malformed inner header is
    # deterministic corruption: reject it here so it is retired, while
    # RuntimeError from the loader below stays transient (500).
    if check_safetensors and not _valid_safetensors(payload):
        raise CheckpointRejectedError("checkpoint payload is not valid safetensors")
    # D1 integrity: verify the payload digest the daemon stamped at publish.
    # This catches framing-valid corruption (bit rot, a torn rewrite with an
    # intact header) that every structural check above misses. Verified once
    # per digest per process — the bytes are already in hand here, but the
    # hash is not free on GB-scale KV, and a blob that verified once does not
    # rot differently on the next load from the same immutable file.
    if not _payload_digest_ok(meta, payload):
        raise CheckpointRejectedError(f"payload digest mismatch (expected {meta.payload_sha256})")
    return meta, payload, True


def _parse_kv_tier_env() -> tuple[str, tuple[int, int, int] | None]:
    """MLXCACHE_KV_TIER → (tier label, parsed (key_bits, value_bits,
    group_size)); ("", None) when unset.

    Refuses at boot on anything ``parse_tier`` rejects and on a simultaneous
    MLXCACHE_KV_BITS: the two knob families select different payload formats
    for the SAME fingerprint axis, so an accidental double set would
    fingerprint one tier and encode the other's bytes."""
    from mlxcache_sidecar.tq import parse_tier  # noqa: PLC0415

    tier = (os.environ.get("MLXCACHE_KV_TIER", "") or "").strip()
    if not tier:
        return "", None
    if os.environ.get("MLXCACHE_KV_BITS", "0") not in ("0", "", None):
        raise ValueError(
            "MLXCACHE_KV_TIER and MLXCACHE_KV_BITS are mutually exclusive: "
            "they select different payload formats for the same fingerprint "
            "axis. Set exactly one."
        )
    return tier, parse_tier(tier)


class SyntheticEngine:
    """Hermetic engine: deterministic tokens + KV bytes, no MLX dependency."""

    name = "synthetic"
    # Engine identity and cache composition are part of the reported identity
    # (connector protocol §2/§4), so a future engine must set both:
    #
    # - `engine_id` namespaces the store key domain. It is "mlx-lm" — NOT
    #   "synthetic" — deliberately: this engine stands in for mlx-lm in
    #   hermetic tests, and a checkpoint minted by a test run must be
    #   indistinguishable (byte-for-byte fingerprints, hence keys) from one
    #   minted by the real engine. A distinct id would re-key every hermetic
    #   e2e blob and hide exactly the cross-binding drift those tests exist to
    #   catch.
    # - `cache_layout` signs the payload's SHAPE. The synthetic engine has no
    #   real cache, so it reports a fixed constant standing in for a dense KV
    #   stack: the composition nothing may be cross-served with, and stable
    #   across processes because it is a literal.
    engine_id = "mlx-lm"
    cache_layout = "synthetic-dense-kv"

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.prefill_count = 0
        self.tokenizer_hash = "synthetic"
        self.kv_dtype = "synthetic"
        # The synthetic engine reports the same quantization knobs so daemon
        # e2e tests can exercise the q8 fingerprint tier without a GPU: the
        # payload is fake bytes either way, but the fingerprint (and blob
        # names) must respond exactly as the real engine's would.
        self.kv_bits = max(0, int(os.environ.get("MLXCACHE_KV_BITS", "0") or 0))
        self.kv_group_size = max(0, int(os.environ.get("MLXCACHE_KV_GROUP_SIZE", "0") or 0))
        if self.kv_bits and not self.kv_group_size:
            # Parity with MlxLmEngine (review 2026-10-04): a group size is
            # mandatory with quantization and 64 is the proven one. The two
            # engines' parses drifted here — different fingerprints/blob names
            # for the same operator config, breaking the synthetic engine's
            # own must-respond-exactly contract.
            self.kv_group_size = 64
        self.kv_tier, self._tq_params = _parse_kv_tier_env()
        # Test knob: emulate a tokenizer that grows a base prompt's tokens —
        # tokenize(base) is a strict token-prefix of tokenize(base + rest), so
        # daemon tests can exercise partial hits and delta prefill (which the
        # fixed 8-token hash alone can never produce).
        self._grow_base = os.environ.get("MLXCACHE_TOKENIZE_GROW")
        # Test knob (T22): emulate the real multi-turn wire shape. When set,
        # one extra token is appended whose value hinges on the WHOLE
        # serialized prompt string: turn r's stream = grow tokens + hash(
        # turn-r-serialization). A grown conversation re-serializes with the
        # closing bracket replaced, so the appended token differs while the
        # grow tokens agree — the stream diverges from the previous turn's
        # checkpoint key exactly at its last token, the shape the end-
        # anchored serve rule turns into reuse.
        self._diverge_tail = os.environ.get("MLXCACHE_TOKENIZE_DIVERGE") == "1"
        # Test knob: fail /prefill with a transient 500 on exactly the Nth
        # call (counting from 1). Exercises the daemon's publish-failure
        # path with an ADOPTED ancestor: a 500 must NOT quarantine the
        # ancestor, and the next identical request must retry cleanly.
        self._prefill_fail_at = int(os.environ.get("MLXCACHE_PREFILL_FAIL_AT", "0") or 0)
        # Test knob: fail every non-stream /generate with a transient 500 from
        # the first call onward. Exercises the daemon's failed-request path:
        # 502 to the client, exactly one trace record, and the stats claim
        # corrected so capture and /stats keep agreeing.
        self._generate_fail = os.environ.get("MLXCACHE_GENERATE_FAIL") == "1"
        # Test knob: fail exactly the Nth non-stream /generate (1-based), then
        # succeed. Exercises the scratch-retry-success path on a blob-backed
        # request: the abandoned attempt leaves its staging file, the retry
        # writes its own, and the settled checkpoint must publish from the
        # RETRY's path.
        self._generate_fail_at = int(os.environ.get("MLXCACHE_GENERATE_FAIL_AT", "0") or 0)
        self._generate_count = 0
        # Phase timers (B2): last prefill's coverage, for the delta-prefill test.
        self.last_prefill_tokens = 0
        self.last_prefill_delta_tokens = 0

    def tokenize(self, prompt: str) -> list[int]:
        # Deterministic token stream derived from the prompt hash. Not a real
        # tokenizer — the daemon contract only needs stable u32 tokens.
        # Test knob: emulate a real tokenizer returning zero tokens.
        if os.environ.get("MLXCACHE_TOKENIZE_EMPTY") == "1":
            return []
        # Test knob: emulate a one-token prompt (nothing cacheable).
        if os.environ.get("MLXCACHE_TOKENIZE_ONE") == "1":
            return [12345]
        tokens = self._tokenize_inner(prompt)
        if self._diverge_tail:
            # T22 knob: hinge ONE token on the WHOLE serialized prompt. A
            # grown conversation re-serializes (closing bracket replaced),
            # so the hinge token differs between turns while every earlier
            # token agrees — turn r+1's stream diverges from turn r's
            # checkpoint key exactly at its last token, the shape the
            # end-anchored serve rule turns into reuse.
            tokens = tokens + [self._hash1(prompt) & 0x7FFFFFFF]
        return tokens

    def _tokenize_inner(self, prompt: str) -> list[int]:
        if self._grow_base is not None:
            # The daemon tokenizes the JSON-serialized message list, so the
            # user content appears VERBATIM between JSON quotes. Emulate a
            # real tokenizer over the CONTENT: the first occurrence of the
            # grow base starts the token stream, and the content runs to the
            # closing JSON quote. The remainder is split on " | " separators
            # and each segment hashed independently, so appending a new
            # segment EXTENDS the token stream: tokenize(round r) is a strict
            # token-prefix of tokenize(round r+1) — the property that makes a
            # growing conversation partial-match at every round (delta
            # prefill's OV3 showcase), the way real BPE keeps deep prefix
            # matches on appended text.
            idx = prompt.find(self._grow_base)
            if idx >= 0:
                end = prompt.find('"', idx + len(self._grow_base))
                content = prompt[idx:end] if end >= 0 else prompt[idx:]
                base = self._hash_tokens(self._grow_base)
                if len(content) == len(self._grow_base):
                    return base
                segments = content[len(self._grow_base) :].removeprefix(" | ").split(" | ")
                return base + [t for seg in segments for t in self._hash_tokens(seg)]
        return self._hash_tokens(prompt)

    @staticmethod
    def _hash1(text: str) -> int:
        return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little")

    @staticmethod
    def _hash_tokens(text: str) -> list[int]:
        digest = hashlib.sha256(text.encode()).digest()
        return [int.from_bytes(digest[i : i + 4], "little") % 2**31 for i in range(0, 32, 4)]

    def prefill(self, tokens: list[int], ancestor_blob_path: str | None = None) -> bytes:
        # KV payload: 1024 bytes/token, deterministic from token ids.
        # ponytail: test-only counter; ThreadingHTTPServer is thread-per-request
        # but the GIL makes this increment effectively safe. Add a lock if the
        # coalescing e2e ever sees a lost count.
        self.prefill_count += 1
        # Test knob (see __init__): a transient RuntimeError on exactly the
        # Nth call. The handler maps it to a 500, which the daemon must
        # classify as a NON-rejection (no quarantine) — the ancestor blob
        # was never invalid, the publish just failed.
        if self._prefill_fail_at and self.prefill_count == self._prefill_fail_at:
            raise RuntimeError(
                f"synthetic induced prefill failure (test knob, call #{self.prefill_count})"
            )
        # Optional delay (test knob): widens the single-flight window so
        # concurrent identical requests are provably coalesced. It runs BEFORE the
        # empty-prefix return below so a one-token prompt also holds the window
        # open, letting the concurrency test exercise the follower path.
        delay = float(os.environ.get("MLXCACHE_PREFILL_DELAY", "0"))
        if delay:
            time.sleep(delay)
        # A prompt shorter than 2 tokens has an empty cache prefix (mirrors
        # MlxLmEngine): cache nothing so the daemon skips publishing an empty
        # payload.
        if len(tokens) < 2:
            return b""
        covered = 0
        if ancestor_blob_path is not None:
            meta, _payload, usable = read_wire_checkpoint(
                ancestor_blob_path, tokens, check_safetensors=False
            )
            if usable and meta is not None:
                # read_wire_checkpoint verified meta.tokens == tokens[:n]
                # against THIS request (the daemon only ever passes the matched
                # ancestor of the same token sequence).
                covered = len(meta.tokens) - 1
        # Phase accounting (B2): what a real engine would recompute. The
        # synthetic KV is a pure function of (token, position), so "prefilling
        # the delta" reproduces the identical full payload — byte-parity with a
        # scratch prefill by construction, while last_prefill_delta_tokens
        # records how many model steps the delta path actually performed
        # (scratch = len-1 steps; delta = len-covered-1 steps), so e2e tests
        # can assert the delta path ran and how much it saved.
        self.last_prefill_tokens = len(tokens)
        self.last_prefill_delta_tokens = max(len(tokens) - 1 - covered, 0)
        # Raw payload only; the daemon writes the checkpoint header.
        return self._kv_bytes(tokens)

    @staticmethod
    def _kv_bytes(tokens: list[int]) -> bytes:
        """The synthetic KV payload for a token sequence: a pure function of
        (token id, position), which is what makes this engine's scratch
        prefill, delta prefill, and settled checkpoint byte-identical."""
        return b"".join(
            ((t * 31 + i) & 0xFFFFFFFF).to_bytes(4, "little") * 256 for i, t in enumerate(tokens)
        )

    def generate(self, tokens: list[int], max_tokens: int) -> list[int]:
        # Deterministic continuation: token ids derived from context length.
        if self._generate_fail:
            raise RuntimeError("synthetic induced generate failure (test knob)")
        self._generate_count += 1
        if self._generate_fail_at and self._generate_count == self._generate_fail_at:
            raise RuntimeError("synthetic induced generate failure (test knob)")
        base = len(tokens)
        return [(base + i) % 2**31 for i in range(max_tokens)]

    def settle(
        self, prompt_tokens: list[int], generated: list[int], cache, settle_path: str
    ) -> dict:
        """Persist the synthetic KV for prompt+generated.

        The synthetic KV is a pure function of (token id, position), so the
        settled payload is byte-identical to what a scratch prefill of the same
        sequence returns — this engine CAN honor a settle honestly, and doing so
        is what lets the hermetic e2e prove the daemon's settle→resume routing.
        It is not a fabricated checkpoint: the bytes are exactly the KV this
        engine would compute for those tokens.

        The documented no-op rule still binds this engine where it cannot
        produce a real settle: with no generated tokens, fewer than two total
        tokens, or a payload over the caps, it writes NOTHING and says so
        (settle_checkpoint enforces those, and the reason is reported).
        """
        del cache  # this engine persists no state object, only the token-ordered KV
        total = list(prompt_tokens) + list(generated)
        if not generated:
            return {"written": False, "tokens": len(total), "reason": "no_generated_tokens"}
        lie = os.environ.get("MLXCACHE_SETTLE_LIE", "")
        # The tokens/identity/engine lies must change the FILE, not the
        # report: the daemon cross-checks the staged header against the
        # request, so these write a checkpoint whose header disagrees with
        # what the engine reports, the exact shape the publish guards reject.
        # "identity:<field>" targets one same_shape conjunct individually;
        # bare "identity" defaults to tokenizer_hash.
        lie_field = None
        if lie.startswith("identity"):
            lie_field = lie.split(":", 1)[1] if ":" in lie else "tokenizer_hash"
        lie_tokens = lie == "tokens"
        lie_engine = lie == "engine"
        written_tokens = list(total)
        if lie_tokens and len(written_tokens) >= 2:
            # Record a sequence one token LONGER than the one the engine
            # reports: header tokens != request+generated.
            written_tokens.append(written_tokens[-1] + 1)
        report = settle_checkpoint(
            self,
            settle_path,
            written_tokens,
            self._kv_bytes(written_tokens),
        )
        if (lie_field or lie_engine) and report.get("written"):
            # Rewrite the header so exactly ONE guard can catch it: the
            # payload bytes are untouched (every digest check passes), and
            # the forged value keeps the field's own type so the daemon's
            # reader still parses the header — the lie is caught by the
            # same_shape conjunct or the engine-domain check it targets,
            # never by a serde rejection that would make the test pass
            # vacuously.
            import struct as _struct  # noqa: PLC0415

            from .blob import CheckpointMeta, Fingerprint, encode  # noqa: PLC0415

            # Only the fields the daemon's same_shape check actually compares
            # (http.rs publish_settled_blob): a knob for any other field is a
            # trap — model_id is never compared (publication proceeds, the
            # test fails confusingly), and content_digest is OMITTED from the
            # header when empty, so forging it KeyErrors out of the settle
            # path as a 500 instead of a discard.
            same_shape_fields = frozenset(
                {
                    "tokenizer_hash",
                    "kv_dtype",
                    "kv_bits",
                    "kv_group_size",
                    "kv_layout_version",
                    "cache_layout",
                    "kv_tier",
                }
            )
            if lie_field and lie_field not in same_shape_fields:
                raise ValueError(f"identity lie targets unknown field {lie_field!r}")
            try:
                with open(settle_path, "rb") as fh:
                    payload = fh.read()
                (hlen,) = _struct.unpack_from("<I", payload)
                meta = json.loads(payload[4 : 4 + hlen])
                if lie_field:
                    fp = dict(meta["fingerprint"])
                    current = fp[lie_field]
                    # Wrong VALUE, right TYPE: a str field gets a foreign
                    # hash, an int field a foreign count.
                    fp[lie_field] = "0" * 64 if isinstance(current, str) else current + 1
                    meta["fingerprint"] = fp
                else:
                    # lie_engine: engine_id is a CheckpointMeta field, not a
                    # fingerprint field — a foreign engine DOMAIN, so the
                    # engine-domain guard fires while same_shape still
                    # agrees.
                    meta["engine_id"] = "llama-cpp"
                fields = {k: v for k, v in meta.items() if k != "fingerprint"}
                wire = encode(
                    CheckpointMeta(fingerprint=Fingerprint(**meta["fingerprint"]), **fields),
                    payload[4 + hlen :],
                )
                _atomic_write(settle_path, wire)
            except OSError as exc:
                # A rewrite that cannot complete is an unsuccessful settle —
                # the same write_failed contract settle_checkpoint honors —
                # never an exception through the generation path.
                return {
                    "written": False,
                    "tokens": len(total),
                    "reason": f"write_failed: {exc}",
                }
        # Test knob (same family as MLXCACHE_GENERATE_FAIL): let a e2e spawn
        # an engine that lies about its digest, or omits it, so the daemon's
        # publish-time guards are exercised over the real HTTP path.
        if lie and report.get("written"):
            if lie == "omit":
                report.pop("sha256", None)
            elif lie == "bad":
                report["sha256"] = "0" * 64
            elif lie == "pathy":
                # Simulate the leak shape the client scrub guards against: an
                # OSError reason embedding an absolute staging path.
                report["reason"] = (
                    f"write_failed: [Errno 13] Permission denied: '{settle_path}.Nx8fJa.part'"
                )
        return report

    def generate_collect(
        self,
        tokens: list[int],
        max_tokens: int,
        blob_path: str | None = None,
        settle_path: str | None = None,
    ):
        """(token_ids, text, cache, settle_report) — the collecting entry the
        handler uses when a settle was requested."""
        out, text = self.generate_with_text(tokens, max_tokens, blob_path)
        settle = None
        if settle_path is not None:
            settle = self.settle(tokens, out, None, settle_path)
        return out, text, None, settle

    def generate_with_text(
        self,
        tokens: list[int],
        max_tokens: int,
        blob_path: str | None = None,
    ):
        """(token_ids, text) — mirrors the stream path's pieces exactly, so
        streaming and non-streaming clients see the same detokenization."""
        if blob_path:
            # The synthetic engine has no KV to resume, but a resumed blob
            # must still VALIDATE like the real adapter would (wire decodes,
            # digest matches) — otherwise corrupt checkpoints serve "fine"
            # here and 422 in production (QA probe 2026-10-04).
            read_wire_checkpoint(blob_path, None, check_safetensors=False)
        out = self.generate(tokens, max_tokens)
        return out, "".join(f"tok{i} " for i in range(len(out)))

    def stream(self, tokens: list[int], blob_path: str | None):
        # Synthetic engine streams its deterministic tokens as text pieces.
        for i, t in enumerate(self.generate(tokens, 64)):
            yield t, f"tok{i} "

    def prepare_stream(self, tokens: list[int], blob_path: str | None):
        """Load any checkpoint and return (prompt, cache) BEFORE streaming.

        Split from generation so the handler can surface a rejected checkpoint as
        a clean 422 (which the daemon quarantines) rather than a mid-stream
        failure. The synthetic engine has no cache, but a resumed blob must
        still VALIDATE (wire + digest) exactly like the real adapter — the
        generation is arithmetic, the trust boundary is not optional.
        """
        if blob_path:
            read_wire_checkpoint(blob_path, None, check_safetensors=False)
        return tokens, None

    def stream_prepared(self, prompt: list[int], cache):
        """Yield (token_id, text) from an already-prepared prompt/cache."""
        # Test knob (daemon idle-timeout e2e): stall before the first token so
        # the daemon's per-chunk idle budget can be exercised deterministically.
        delay = float(os.environ.get("MLXCACHE_FIRST_TOKEN_DELAY", "0"))
        if delay:
            time.sleep(delay)
        for i, t in enumerate(self.generate(prompt, 64)):
            yield t, f"tok{i} "


class MlxLmEngine:
    """Real mlx-lm engine. Lazy import; requires a downloaded model.

    KV checkpoints are real mlx-lm prompt caches (safetensors). The blob is the
    sidecar/daemon wire format: u32 header length + JSON meta + safetensors bytes.
    """

    name = "mlx-lm"
    # The default engine's key domain: zero fold bytes (protocol §4).
    engine_id = "mlx-lm"

    def __init__(self, model_id: str) -> None:
        from mlx_lm import load  # noqa: PLC0415 — lazy, only when selected

        self.model_id = model_id
        self.model, self.tokenizer = load(model_id)
        self._adopt_model()

    def _adopt_model(self) -> None:
        """Everything that follows loading a model, independent of HOW it was
        loaded. A second engine that builds its model some other way (a
        streaming loader over an architecture mlx-lm cannot construct) calls
        this same method, so the identity fields and boot guards have exactly
        one implementation and cannot drift apart from the load path."""
        self._materialize_weights()
        # Hash the vocabulary, not the model name: R1-2 pins the tokenizer
        # artifact, and two different tokenizers can share a name_or_path-free
        # identity. The vocab is deterministic and artifact-derived.
        self.tokenizer_hash = self._hash_vocab()
        self.kv_dtype = self._kv_dtype()
        # T12 adoption knobs: KV quantization applied before persisting.
        # bits=0 (default) = f16, byte-identical to the historical behavior.
        # Only 8-bit group-64 is parity-proven (T12); anything else is the
        # operator's explicit choice and is fingerprinted as its own tier.
        self.kv_bits = max(0, int(os.environ.get("MLXCACHE_KV_BITS", "0") or 0))
        self.kv_group_size = max(0, int(os.environ.get("MLXCACHE_KV_GROUP_SIZE", "0") or 0))
        if self.kv_bits and not self.kv_group_size:
            # A group size is mandatory with quantization; 64 is the proven one.
            self.kv_group_size = 64
        # Named persistence tier (TQ codecs): validated for the same reasons
        # and mutually exclusive with the bits tier — see _parse_kv_tier_env.
        self.kv_tier, self._tq_params = _parse_kv_tier_env()
        # The LIVE variant of a named tier: hold the compressed state as the
        # engine's own cache (unified-memory savings) instead of compressing
        # only at capture. Same tier label, same frame format, same
        # fingerprint — a runtime memory trade, not a new format.
        self.tq_live = self.kv_tier != "" and os.environ.get("MLXCACHE_TQ_LIVE") == "1"
        # One cache construction serves BOTH boot checks below. It allocates
        # no arrays (state fills on the first forward), so probing every model
        # — quantized or not — is free. Composition must be signed for EVERY
        # model: an empty (unknown) layout would let this engine serve blobs
        # whose payload shape it cannot verify.
        probe = self._prompt_cache_probe()
        self.cache_layout = self._cache_layout_of(probe)
        # The per-model layer count every adopted checkpoint must match: a
        # short cache list makes mlx-lm's zip(model.layers, cache) skip
        # layers silently (invalid output, no error) rather than 422.
        self._cache_layer_count = len(probe)
        self._assert_quantization_supported(probe)
        if self.tq_live:
            # The live tier's shared basis: one (rotation, codebook) pair per
            # model, referenced by every layer's cache. The geometry guard
            # above already materialized the uniform head_dim via its
            # one-token forward.
            from mlxcache_sidecar.tq.codebook import lloyd_max_codebook  # noqa: PLC0415
            from mlxcache_sidecar.tq.rotation import build_rotation  # noqa: PLC0415

            head_dim = probe[0].keys.shape[-1]
            self._tq_basis = (
                build_rotation(head_dim),
                lloyd_max_codebook(head_dim, self._tq_params[0]),
            )
        self.prefill_count = 0
        # Metal allocator hygiene: MLX pools freed GPU buffers by size-class
        # indefinitely (set_cache_limit defaults high). A day of mixed-leg
        # 20K prefills (each a new size class) then grows the pool for the
        # life of the process. Bound the cache; freed buffers beyond it are
        # released to the OS instead of pooled. (Read the env as MB; 0 keeps
        # the library default for tests that never want reclaim churn.)
        metal_cache_mb = os.environ.get("MLXCACHE_METAL_CACHE_MB", "1024")
        try:
            import mlx.core as mx  # noqa: PLC0415

            limit = int(metal_cache_mb) * 1024 * 1024
            try:
                mx.set_cache_limit(limit)
            except AttributeError:  # older MLX: knob lived under mx.metal
                mx.metal.set_cache_limit(limit)
        except (ImportError, AttributeError, ValueError):
            pass  # CPU-only build or older MLX without the knob: nothing to bound
        # EOS for THIS tokenizer artifact (identity-pinned alongside it).
        self._eos_ids = getattr(self.tokenizer, "_eos_token_ids", None) or {
            self.tokenizer.eos_token_id
        }

    def _materialize_weights(self) -> None:
        """Evaluate every model parameter on the boot thread.

        MLX defers array construction, and a lazy graph built on one thread
        cannot be evaluated on another: doing so raises
        `RuntimeError: There is no Stream(gpu, 0) in current thread`. The engine
        is built on the thread that calls `serve()`, while
        `ThreadingHTTPServer` dispatches each request on a worker thread, so a
        model whose weights are still unevaluated when the first request arrives
        fails on the very first forward pass — a 500 on `/prefill`, before any
        token is produced.

        `parameters()` is a lazy view, so this is one `mx.eval` over the whole
        weight tree, on the thread that owns it. It costs the load's own I/O,
        which the loader is already streaming at boot; it does not hold a second
        copy. Engines whose model is not an MLX module (the synthetic engine, a
        third-party adapter) are skipped.
        """
        parameters = getattr(self.model, "parameters", None)
        if parameters is None:
            return
        try:
            import mlx.core as mx  # noqa: PLC0415
        except ImportError:
            return
        tree = parameters()
        if tree:
            mx.eval(tree)

    def _prompt_cache_probe(self):
        """One freshly constructed prompt cache for this model, shared by the
        boot checks. Construction allocates no arrays (states fill on the
        first forward), so this is cheap and side-effect free."""
        from mlx_lm.models.cache import make_prompt_cache  # noqa: PLC0415

        return make_prompt_cache(self.model)

    @staticmethod
    def _cache_layout_of(cache) -> str:
        """The cache-composition signature: the ORDERED tuple of cache-entry
        class names, hashed. A dense model yields one class repeated per
        layer; a hybrid model yields CacheList/ArraysCache entries carrying
        linear-attention or SSM state. Two architectures can agree on model
        id, tokenizer, dtype and quantization tier while their payloads have
        different shapes, so composition must be part of the identity.

        The signature is a digest, not the raw tuple: it is folded into every
        key and stamped in every header, and a short stable hex string keeps
        both cheap. Truncated to 16 hex chars (64 bits) — collisions between
        two real cache compositions are not a realistic concern and, unlike
        the key fold, this value only ever needs to separate architectures
        that differ structurally."""
        names = ",".join(type(c).__name__ for c in cache)
        return hashlib.sha256(names.encode()).hexdigest()[:16]

    def _assert_quantization_supported(self, cache) -> None:
        """Boot-time guard for the quantized tiers (q8 via kv_bits, TQ via
        kv_tier): parity is proven for dense attention stacks only, and the
        tier's format constraints must hold for THIS model before the first
        request — a violation discovered at encode time is a 500 loop on
        every cache-miss request, not a boot error.

        Hybrid caches (CacheList/ArraysCache entries carrying linear-
        attention / SSM / GDN state) have no ``to_quantized`` and no TQ
        encoder path; uniform head_dim divisible by the group size is
        required by both the TQ frame and mx.quantize. The probe cache
        allocates no arrays until a forward runs, so one single-token
        forward materializes the per-layer shapes the checks need."""
        if not self.kv_bits and not getattr(self, "kv_tier", ""):
            return
        from mlx_lm.models.cache import KVCache  # noqa: PLC0415

        if not all(type(c) is KVCache for c in cache):
            raise ValueError(
                "A KV quantization tier is set but this model's prompt cache "
                "is not a dense KV stack (a hybrid architecture: linear "
                "attention, SSM, or GDN layers). Quantized tiers are "
                "parity-proven for dense models only; hybrid checkpoints "
                "persist their state unquantized. Restart with "
                "MLXCACHE_KV_BITS=0 and MLXCACHE_KV_TIER unset."
            )
        self._assert_tier_geometry(cache)

    def _assert_tier_geometry(self, cache) -> None:
        """The tier's SHAPE constraints for this model, via one single-token
        forward (the probe cache allocates no arrays until a forward runs,
        so this is what materializes the per-layer head dims). A violation
        discovered at encode time would be a 500 loop on every cache-miss
        request; discovered here it is a boot error with the fix in the
        message.

        Covered by test_engine_registration's boot-refusal tests (real mx,
        shape-real stubs); the cache-convention suite fakes the whole mlx
        layer and bypasses this step."""
        import mlx.core as mx  # noqa: PLC0415

        logits = self.model(mx.array([[0]]), cache=cache)
        mx.eval([c.state for c in cache], logits)
        # Key AND value head dims: the TQ frame derives its tensor names from
        # the key dim and decode re-derives from the value scales, so
        # kdim != vdim must refuse at boot; mx.quantize needs divisibility
        # on both.
        head_dims = {c.keys.shape[-1] for c in cache} | {c.values.shape[-1] for c in cache}
        if len(head_dims) != 1:
            raise ValueError(
                f"A KV quantization tier requires a uniform head_dim across "
                f"layers; this model mixes {sorted(head_dims)}. Restart with "
                "MLXCACHE_KV_BITS=0 and MLXCACHE_KV_TIER unset."
            )
        d = head_dims.pop()
        group = self.kv_group_size if self.kv_bits else self._tq_params[2]
        if d % group:
            raise ValueError(
                f"A KV quantization tier is set but this model's head_dim "
                f"{d} is not divisible by the group size {group}. Restart "
                "with MLXCACHE_KV_BITS=0 and MLXCACHE_KV_TIER unset."
            )

    def _kv_dtype(self) -> str:
        """The KV/compute dtype this model will cache in (bf16 vs f16 changes the
        persisted bytes and makes a checkpoint from the other dtype unsafe to
        serve). Derived from a real model parameter, not a constant."""
        try:

            def flatten(node):
                if hasattr(node, "dtype"):
                    yield node
                elif isinstance(node, dict):
                    for v in node.values():
                        yield from flatten(v)
                elif isinstance(node, (list, tuple)):
                    for v in node:
                        yield from flatten(v)

            for p in flatten(self.model.parameters()):
                return str(p.dtype).replace("mlx.core.", "")
        except Exception:  # noqa: BLE001 — best-effort identity, never fatal
            pass
        return "unknown"

    def _hash_vocab(self) -> str:
        """Cryptographic digest of the tokenizer's vocabulary, or the
        literal "missing" sentinel when the tokenizer exposes no
        vocabulary. The previous fallback hashed name_or_path / the model
        id — an identity two different tokenizers can share and a path the
        cache must not pin: a non-cryptographic identity now fails CLOSED
        (the sentinel disables persistence) instead of silently keying
        reusable KV on a name."""
        vocab = getattr(self.tokenizer, "get_vocab", None)
        if not callable(vocab):
            return "missing"
        items = sorted(vocab().items())
        payload = repr(items).encode()
        return hashlib.sha256(payload).hexdigest()[:16]

    def tokenize(self, prompt: str) -> list[int]:
        return self.tokenizer.encode(prompt)

    def _prefill_cache(self, tokens: list[int]):
        """Run tokens through the model filling a fresh prompt cache. Returns
        (cache, last_logits) with the KV state resident and evaluated."""
        import mlx.core as mx  # noqa: PLC0415

        cache = self._new_cache()
        inp = mx.array(tokens)[None]
        logits = self.model(inp, cache=cache)
        mx.eval([c.state for c in cache], logits)
        return cache, logits

    def _load_cache_from_payload(self, payload: bytes):
        """Load a prompt cache from an ALREADY-VALIDATED payload.

        Delta prefill needs the same load the generate path does, but without a
        second disk read (the caller just validated the wire checkpoint). A
        pure-Python parse/schema failure is corruption → CheckpointRejectedError
        (422/quarantine); OSError/RuntimeError stay transient (500).

        The TQ tier decodes its own frame here: the engine's tier and the
        blob's tier always agree (the daemon only routes same-tier blobs —
        the fingerprint pins it), so the branch is sound. The LIVE variant
        adopts the compressed tensors directly (no dequantize-requantize
        round trip: the resident state IS the frame). Both branches verify
        the decoded cache has the layer count THIS model expects: a short
        list makes mlx-lm zip() skip layers silently instead of erroring."""
        cache = None
        if getattr(self, "kv_tier", ""):
            try:
                if getattr(self, "tq_live", False):
                    from mlxcache_sidecar.tq.codec import (  # noqa: PLC0415
                        decode_frame,
                        frame_basis,
                    )
                    from mlxcache_sidecar.tq.live import TurboQuantKVCache  # noqa: PLC0415

                    arrays, meta = decode_frame(payload)
                    # The frame's declared format must equal the engine's
                    # configured tier: _fetch dequantizes with the ENGINE's
                    # bits/group and serialize_live re-labels the codebook
                    # tensor with the engine's key bits, so a mismatching
                    # frame would decode to silent garbage here (the dense
                    # path is frame-consistent instead).
                    kb, vb, g = self._tq_params
                    if (meta["key_bits"], meta["value_bits"], meta["group_size"]) != (kb, vb, g):
                        raise ValueError(
                            f"frame format {meta['key_bits']}/{meta['value_bits']}/"
                            f"{meta['group_size']} != engine tier {kb}/{vb}/{g}"
                        )
                    basis = frame_basis(arrays, meta)
                    cache = []
                    for i in meta["layer_ids"]:
                        c = TurboQuantKVCache(kb, vb, g, basis=basis)
                        c.adopt_state(
                            arrays[f"layers.{i}.k.idx"],
                            arrays[f"layers.{i}.k.norm"],
                            arrays[f"layers.{i}.v.q"],
                            arrays[f"layers.{i}.v.s"],
                            arrays[f"layers.{i}.v.b"],
                            meta["offset"],
                            meta["kv_dtype"],
                        )
                        cache.append(c)
                else:
                    from mlxcache_sidecar.tq import decode_payload  # noqa: PLC0415

                    cache = decode_payload(payload)
            except (ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:
                raise CheckpointRejectedError(
                    f"TQ checkpoint failed to decode: {type(exc).__name__}: {exc}"
                ) from exc
        else:
            from mlx_lm.models.cache import load_prompt_cache  # noqa: PLC0415

            # Parse from memory: the daemon just sent this payload over the wire
            # and validated it — a temp-file write + read doubled disk I/O per
            # PARTIAL hit at GB scale (performance review 2026-10-04).
            # load_prompt_cache forwards to mx.load WITHOUT a format argument, so
            # mlx infers safetensors from the file EXTENSION and reads `.name` —
            # a bare BytesIO raises AttributeError. Naming the buffer satisfies
            # the inference; the data itself still never touches disk.
            blob = io.BytesIO(payload)
            blob.name = "checkpoint.safetensors"
            try:
                cache = load_prompt_cache(blob)
            except (ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:
                raise CheckpointRejectedError(
                    f"checkpoint failed to load: {type(exc).__name__}: {exc}"
                ) from exc
        expected = getattr(self, "_cache_layer_count", None)
        if expected is not None and len(cache) != expected:
            raise CheckpointRejectedError(
                f"checkpoint holds {len(cache)} cache entries; this model needs {expected}"
            )
        return cache

    def _quantize_cache(self, cache):
        """Apply the configured bits-tier KV quantization (T12). No-op at
        bits=0 and on the TQ tier (its compression happens in
        ``_serialize_cache`` — the live cache is never quantized).

        A cache already holding QuantizedKVCache layers (a delta prefill that
        adopted a quantized ancestor) must NOT be re-quantized: to_quantized
        runs mx.quantize over the packed uint32 words as if they were floats,
        corrupting the KV (the fingerprint pins kv_bits/kv_group_size, so an
        adoptable ancestor is always this same tier — skipping is exact).
        """
        if not self.kv_bits or getattr(self, "kv_tier", ""):
            return cache
        from mlx_lm.models.cache import QuantizedKVCache  # noqa: PLC0415

        return [
            c
            if isinstance(c, QuantizedKVCache)
            else c.to_quantized(group_size=self.kv_group_size, bits=self.kv_bits)
            for c in cache
        ]

    def _serialize_cache(self, cache) -> bytes:
        """Persist the cache in the configured tier's payload format.

        f16/q8: mlx-lm's own safetensors frame (via save_prompt_cache). TQ:
        the compressed frame — the whole point of the tier, ~4x smaller
        bytes at the cost of a bounded-distortion resume."""
        import io  # noqa: PLC0415

        if getattr(self, "kv_tier", ""):
            from mlxcache_sidecar.tq import encode_cache  # noqa: PLC0415

            key_bits, value_bits, group_size = self._tq_params
            if getattr(self, "tq_live", False):
                # The live cache IS the compressed state: serialize it
                # directly (byte-identical to encode_cache over the same
                # token stream — the equality test pins it).
                from mlxcache_sidecar.tq.live import serialize_live  # noqa: PLC0415

                return serialize_live(cache, key_bits, value_bits, group_size)
            return encode_cache(cache, key_bits, value_bits, group_size)
        from mlx_lm.models.cache import save_prompt_cache  # noqa: PLC0415

        buf = io.BytesIO()
        save_prompt_cache(buf, cache)
        return buf.getvalue()

    def prefill(self, tokens: list[int], ancestor_blob_path: str | None = None) -> bytes:
        import mlx.core as mx  # noqa: PLC0415

        self.prefill_count = getattr(self, "prefill_count", 0) + 1
        # Cache convention (pinned; verified by the R1-5 thesis guard): the saved
        # cache covers tokens[:-1], NOT all tokens. On resume the adapter feeds the
        # uncovered tail (tokens[len(prefix)-1:]) so the model predicts the final
        # token from KV of the preceding ones — identical to a scratch run.
        #
        # A one-token prompt has an empty prefix: mlx-lm cannot save/load an empty
        # prompt cache, and there is nothing to cache, so return no payload. The
        # daemon skips publishing an empty payload and the request runs from
        # scratch.
        if len(tokens) < 2:
            return b""
        cache = None
        covered = 0
        if ancestor_blob_path is not None:
            # Delta prefill (OV3 — the single biggest TTFT lever on a growing
            # conversation): adopt the ancestor's KV instead of recomputing it.
            # read_wire_checkpoint enforces the trust boundary (prefix must
            # agree with THIS request; bad → 422 → the daemon quarantines and
            # retries from scratch).
            meta, payload, usable = read_wire_checkpoint(ancestor_blob_path, tokens)
            if usable:
                t0 = time.perf_counter()
                cache = self._load_cache_from_payload(payload)
                _phase_add("cache_load", (time.perf_counter() - t0) * 1000)
                # The persisted cache covers meta.tokens[:-1] = tokens[:covered].
                covered = len(meta.tokens) - 1
        if cache is None:
            cache, _ = self._prefill_cache(tokens[:-1])
            self.last_prefill_tokens = len(tokens)
            self.last_prefill_delta_tokens = len(tokens) - 1
        else:
            # Feed ONLY the uncovered tokens, stopping BEFORE the final token so
            # the saved cache covers tokens[:-1] — byte-identical convention to a
            # scratch prefill. Work drops from len(tokens)-1 model steps to
            # len(tokens)-1-covered.
            delta = tokens[covered:-1]
            self.last_prefill_tokens = len(tokens)
            self.last_prefill_delta_tokens = len(delta)
            if delta:
                t0 = time.perf_counter()
                inp = mx.array(delta)[None]
                logits = self.model(inp, cache=cache)
                mx.eval([c.state for c in cache], logits)
                _phase_add("delta_prefill", (time.perf_counter() - t0) * 1000)
        # Serialize to an IN-MEMORY buffer: the payload must be in RAM for the
        # HTTP response regardless, so the old temp-file roundtrip (a full
        # GB-scale disk write + read back at 20K tokens, before the daemon
        # could even publish) was pure TTFT tax with zero durability benefit
        # (performance review 2026-10-04). mx.save_safetensors — and thus
        # save_prompt_cache — accepts file objects.
        # Quantize/persist BEFORE save (T12 tier): the persisted blob holds
        # the quantized state, so bytes/token drops ~47% at 8-bit g64 (and
        # ~4x further on the TQ tier's compressed frame). The fingerprint the
        # daemon recorded already pins (kv_bits, kv_group_size, kv_tier) from
        # /tokenize, so tiers never cross-serve.
        cache = self._quantize_cache(cache)
        t0 = time.perf_counter()
        payload = self._serialize_cache(cache)
        _phase_add("cache_save", (time.perf_counter() - t0) * 1000)
        mx.clear_cache()

        # Return the raw safetensors payload. The daemon owns the checkpoint
        # header (it writes meta via publish_atomic) and the wire format is
        # header + this payload; encoding one here would double-wrap it.
        return payload

    def generate(self, tokens: list[int], max_tokens: int) -> list[int]:
        """Generate from scratch over `tokens` (no persisted cache).

        A FRESH cache is constructed here — passing None down used to crash
        the first trim probe in the streaming loop, and a scratch run needs a
        real cache object anyway (the settle path captures it). The blob-
        resuming variants are generate_from_blob / generate_collect.
        """
        return self._generate_with_cache(tokens, max_tokens, cache=self._new_cache())

    def _new_cache(self):
        """A fresh prompt cache owned by the caller, so generation mutates it in
        place and it survives the call.

        Passing None to mlx-lm makes IT construct the cache internally, which
        leaves nothing to settle: the post-generation KV would be unreachable
        from the request that produced it, and settling would be impossible
        without replaying generation.

        The live TQ tier swaps the class: TurboQuantKVCache holds the
        compressed state as THE cache (unified-memory savings), quantizing at
        write and dequantizing on read — stock attention, no patches.
        """
        if getattr(self, "tq_live", False):
            from mlxcache_sidecar.tq.live import TurboQuantKVCache  # noqa: PLC0415

            kb, vb, g = self._tq_params
            return [
                TurboQuantKVCache(kb, vb, g, basis=self._tq_basis)
                for _ in range(self._cache_layer_count)
            ]
        from mlx_lm.models.cache import make_prompt_cache  # noqa: PLC0415

        return make_prompt_cache(self.model)

    def settle(
        self, prompt_tokens: list[int], generated: list[int], cache, settle_path: str
    ) -> dict:
        """Persist the POST-GENERATION cache (prompt + generated) as a wire
        checkpoint, so the next turn resumes with only its own new tokens.

        Coverage is measured and reconciled, never assumed. mlx-lm's sampler
        feeds the most recent token to predict the NEXT one, and the final
        prediction is discarded, so the cache at the end of generation can hold
        one position MORE than the `tokens[:-1]` convention. Two directions,
        both handled:

        - Deeper than the convention: trim the surplus position(s). If the
          composition cannot be trimmed (recurrent/SSM hybrid state has no
          per-position trim), do NOT settle — a covered count one too deep
          makes the next turn re-feed a token the cache already holds, which
          degrades output silently instead of failing.
        - Shallower than the convention (immediate EOS, a prompt-only run):
          REFUSE. The earlier behavior shortened the recorded prefix and wrote
          the file, but the daemon requires the settled sequence to equal
          request+generated exactly (publish_settled_blob discards anything
          else), so a shortened settle was dead work that still reported
          `written: true` — a report nothing downstream ever acted on.
        """
        total = list(prompt_tokens) + list(generated)
        if not generated:
            return {"written": False, "tokens": len(total), "reason": "no_generated_tokens"}
        if cache is None:
            return {"written": False, "tokens": len(total), "reason": "no_cache"}
        if len(total) < 2:
            return {"written": False, "tokens": len(total), "reason": "too_short"}
        # Same tier as prefill persists: the header's fingerprint pins
        # kv_bits/kv_group_size/kv_tier, so a settled blob must hold the same
        # shape of state as a prefilled one for the two to be interchangeable.
        cache = self._quantize_cache(cache)
        positions = _cache_positions(cache)
        if positions is None:
            return {"written": False, "tokens": len(total), "reason": "unmeasurable_cache"}
        target = len(total) - 1
        if positions > target:
            if not _trim_cache_to(cache, target):
                return {"written": False, "tokens": len(total), "reason": "untrimmable_cache"}
            # `covered` must describe the state as persisted, not as first
            # measured: the trim above is what makes them differ.
            positions = _cache_positions(cache)
            if positions is None or positions != target:
                return {"written": False, "tokens": len(total), "reason": "untrimmable_cache"}
        elif positions < target:
            return {"written": False, "tokens": len(total), "reason": "shallow_cache"}
        payload = self._serialize_cache(cache)
        return settle_checkpoint(self, settle_path, total, payload, covered=positions)

    def generate_with_text(
        self,
        tokens: list[int],
        max_tokens: int,
        blob_path: str | None = None,
    ):
        """(token_ids, detokenized_text) for non-streaming callers.

        Collects from the same streaming decode path so token-for-token (and
        character-for-character) it matches what a streaming client sees; the
        plain `generate` below discards text for callers that only want ids.
        """
        out, text, _cache, _settle = self.generate_collect(tokens, max_tokens, blob_path, None)
        return out, text

    def generate_collect(
        self,
        tokens: list[int],
        max_tokens: int,
        blob_path: str | None = None,
        settle_path: str | None = None,
    ):
        """(token_ids, text, cache, settle_report) for the non-streaming leg.

        The cache is returned rather than discarded so a settle can persist the
        post-generation KV: it exists only inside this call, so the request
        that built it is the only place it can be captured.
        """
        if max_tokens <= 0:
            return [], "", None, None
        cache = None
        prompt = tokens
        if blob_path:
            cache, prompt = self._load_cache_delta(tokens, blob_path)
            if cache is None:
                prompt = tokens  # unexpectable cache: scratch
        if cache is None:
            cache = self._new_cache()
        out: list[int] = []
        pieces: list[str] = []
        for token, text in self._stream_with_cache(prompt, cache):
            out.append(token)
            pieces.append(text)
            if len(out) >= max_tokens:
                break
        settle = None
        if settle_path is not None:
            # Never raises, mirroring _settle_stream: generation already
            # succeeded, and a failed settle (an import, an untrimmable cache
            # composition, a quantize step) must not turn the completed
            # request into a 500. The report carries the reason instead.
            try:
                settle = self.settle(tokens, out, cache, settle_path)
            except Exception as exc:  # noqa: BLE001 — a settle is best-effort
                traceback.print_exc()
                settle = {
                    "written": False,
                    "tokens": len(tokens) + len(out),
                    "reason": f"settle_failed: {exc}",
                }
        return out, "".join(pieces), cache, settle

    def stream(self, tokens: list[int], blob_path: str | None):
        """Yield (token_id, text) for this request, resuming from `blob_path`
        when present. This is the streaming entry the handler drives."""
        prompt, cache = self.prepare_stream(tokens, blob_path)
        yield from self.stream_prepared(prompt, cache)

    def prepare_stream(self, tokens: list[int], blob_path: str | None):
        """Load the checkpoint (may raise for a corrupt blob) and return
        (prompt, cache). Done before headers are sent so a load failure is a
        clean 422, not a truncated stream. First-token decode happens later, in
        stream_prepared, so a decode error does not look like blob corruption.

        A cache is ALWAYS returned (constructed here when there is no blob to
        adopt): generation mutates it in place, and the settled checkpoint is
        captured from exactly that object after the last token. Letting mlx-lm
        build a throwaway internal cache would make the post-generation KV
        unreachable from the request that produced it.
        """
        cache = None
        prompt = tokens
        if blob_path:
            cache, prompt = self._load_cache_delta(tokens, blob_path)
            if cache is None:
                prompt = tokens  # unexpectable cache: scratch
        if cache is None:
            cache = self._new_cache()
        return prompt, cache

    def stream_prepared(self, prompt, cache):
        """Yield (token_id, text) after the checkpoint is already loaded."""
        # Test knob (daemon idle-timeout e2e): stall before the first token so
        # the daemon's per-chunk idle budget can be exercised deterministically.
        delay = float(os.environ.get("MLXCACHE_FIRST_TOKEN_DELAY", "0"))
        if delay:
            time.sleep(delay)
        yield from self._stream_with_cache(prompt, cache)

    def _stream_with_cache(self, prompt, cache):
        yield from self.stream_with_cache(prompt, cache)

    def stream_with_cache(self, tokens, cache):
        """Yield (token_id, text_piece) per decode step.

        Two loops, chosen by what the cache can do, because they differ by
        exactly the position `settle` has to reconcile.

        `mlx_lm.stream_generate` runs one forward per token PLUS one: the token
        it yielded last is fed back before the budget check, so after N tokens
        the cache holds N+1 positions. `settle` trims that surplus back to the
        `tokens[:-1]` convention, which is fine for a KV cache —
        `KVCache.trim` drops positions.

        A hybrid cache cannot. This architecture's GDN layers accumulate
        recurrent state with no position axis, so `is_trimmable()` is False and
        the surplus is unfixable after the fact: `settle` refuses every
        checkpoint, and without a checkpoint every turn re-prefills from
        scratch. For those caches the loop below forwards only when the consumer
        asks for another token, so the last forward is the one that produced the
        last yielded token and the cache lands on the convention exactly. That
        also saves one forward per request.

        Token choice is identical either way: `stream_generate`'s default
        sampler is argmax over the log-probabilities, which is argmax over the
        logits.
        """
        from mlx_lm.models.cache import can_trim_prompt_cache  # noqa: PLC0415

        if can_trim_prompt_cache(cache):
            import mlx.core as mx  # noqa: PLC0415
            from mlx_lm import stream_generate  # noqa: PLC0415

            for resp in stream_generate(
                self.model, self.tokenizer, prompt=mx.array(tokens), prompt_cache=cache
            ):
                yield resp.token, resp.text
            return
        yield from self._stream_untrimmable(tokens, cache)

    def _stream_untrimmable(self, tokens, cache):
        """Decode loop that ends on the cache convention, for caches without trim.

        The prompt is fed up to but excluding its last token; that tail is the
        first input to the loop, so its output predicts the first generated token
        with the whole prompt already resident.
        """
        import mlx.core as mx  # noqa: PLC0415
        from mlx_lm.tokenizer_utils import TokenizerWrapper  # noqa: PLC0415

        if not tokens:
            raise ValueError("stream_with_cache needs at least one prompt token")
        tokenizer = self.tokenizer
        if not isinstance(tokenizer, TokenizerWrapper):
            tokenizer = TokenizerWrapper(tokenizer)
        eos = set(getattr(tokenizer, "eos_token_ids", ()) or ())
        detokenizer = tokenizer.detokenizer

        if len(tokens) > 1:
            self.model(mx.array([tokens[:-1]]), cache=cache)
        pending = tokens[-1]
        while True:
            logits = self.model(mx.array([[pending]]), cache=cache)
            token = int(mx.argmax(logits[0, -1]).item())
            if token in eos:
                # Not yielded: `stream_generate` ends before emitting EOS, so
                # yielding it would change the token sequence callers see.
                #
                # The cache is one position DEEPER than the `tokens[:-1]`
                # convention here, not on it: the forward that discovered the
                # EOS is the one that consumed the last yielded token, so the
                # cache now covers the whole `prompt + generated` sequence. A
                # trimmable cache drops that surplus; this one cannot, so
                # `settle` refuses the checkpoint rather than record a coverage
                # count the cache does not hold. The answer is unaffected; the
                # cost is that an EOS-terminated reply on a hybrid engine
                # publishes no checkpoint. A token-budget stop, which is the
                # path the settle tests drive, lands exactly on the convention.
                detokenizer.finalize()
                return
            detokenizer.add_token(token)
            yield token, detokenizer.last_segment
            pending = token

    def _generate_with_cache(self, tokens: list[int], max_tokens: int, cache) -> list[int]:
        """Collect a bounded generation from the streaming path, so the
        non-streaming /generate mode reuses the identical decode logic."""
        if max_tokens <= 0:
            return []
        out: list[int] = []
        for token, _text in self._stream_with_cache(tokens, cache):
            out.append(token)
            if len(out) >= max_tokens:
                break
        return out

    def _load_cache_delta(self, tokens: list[int], blob_path: str):
        """Returns (cache, delta_prompt), or (None, tokens) when the checkpoint
        is legitimately uncacheable for this request (a prefix shorter than 2
        tokens, or a legacy blob with no recorded tokens). Raises
        CheckpointRejectedError when the checkpoint is BAD — corrupt payload, or a
        recorded prefix that disagrees with the request — so the daemon retires
        the entry instead of resuming from wrong KV.

        The persisted cache covers the checkpoint prefix MINUS its final token
        (see prefill), so the delta is exactly the tokens the cache does not
        cover: tokens[covered:]. Uniform for every prompt length, including a
        one-token prompt whose cache covers 0 tokens (delta = the whole prompt)."""
        meta, payload, usable = read_wire_checkpoint(blob_path, tokens)
        if not usable:
            return None, tokens
        covered = len(meta.tokens) - 1
        # From-memory load via the shared tier-aware loader — same rationale
        # and same named-buffer requirement as _load_cache_from_payload
        # (performance review 2026-10-04), and one place that knows the
        # configured tier's payload format.
        # A pure-Python parse/schema failure is corruption: reject so the
        # daemon retires it. MemoryError, OSError, and RuntimeError are
        # transient (a real OOM, a disk fault; MLX's native reader raises
        # RuntimeError on an OS read failure): let them propagate as a 500 so
        # a healthy checkpoint is not retired.
        cache = self._load_cache_from_payload(payload)
        return cache, tokens[covered:]

    def generate_from_blob(self, tokens: list[int], blob_path: str, max_tokens: int) -> list[int]:
        """Resume generation from a persisted wire-format checkpoint (a cache
        hit): strip the daemon header, load the prompt cache, generate.

        The checkpoint holds KV for a token prefix; the delta (uncached tail)
        is what the model still needs to see. We pass the tail so the cache
        supplies the cached prefix and generation continues identically to a
        scratch run over the full prompt (thesis guard, T2).

        A legitimately uncacheable blob (a prefix shorter than 2 tokens)
        falls back to a FRESH cache over the whole prompt — the same contract
        `prepare_stream` applies on the streaming path. Passing cache=None on
        used to crash the first trim probe; an uncacheable hit is a scratch
        run, not an error."""
        cache, prompt = self._load_cache_delta(tokens, blob_path)
        if cache is None:
            cache = self._new_cache()
            prompt = tokens
        return self._generate_with_cache(prompt, max_tokens, cache)


def make_engine(model_id: str) -> SyntheticEngine | MlxLmEngine:
    """Select the engine from `MLXCACHE_ENGINE`.

    An unknown value is a hard error, never a silent fallback: falling back to
    the synthetic engine would answer real requests with its deterministic
    toy tokens while every health check still passed.
    """
    engine = os.environ.get("MLXCACHE_ENGINE", "synthetic")
    if engine == "mlx-lm":
        return MlxLmEngine(model_id)
    if engine != "synthetic":
        raise ValueError(f"unknown MLXCACHE_ENGINE={engine!r}; expected 'synthetic' or 'mlx-lm'")
    return SyntheticEngine(model_id)


class Handler(BaseHTTPRequestHandler):
    engine: object = None  # set by serve()

    # HTTP/1.1 keep-alive (Content-Length is always set on every response,
    # including errors — see _json/_binary). The daemon's reqwest pool REUSES
    # connections; with the HTTP/1.0 default each response closed the socket
    # while the pool raced the next request onto it, surfacing as spurious
    # ConnectionReset/500s under concurrent load (the e2e coalescing tests
    # tripped this flakily in release builds).
    protocol_version = "HTTP/1.1"

    # A pooled keep-alive socket that sits idle between bursts must not die
    # mid-teardown when the next request lands: None disables the socket
    # timeout (BaseHTTPRequestHandler's default), relying on the daemon's own
    # budgets (JSON total timeout, stream open/idle) for liveness.
    timeout = None

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write("sidecar: %s\n" % (fmt % args))

    def _json(self, code: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    @staticmethod
    def _require(req: dict, key: str):
        if key not in req:
            raise Handler._BadRequestError(f"missing required field '{key}'")
        return req[key]

    @staticmethod
    def _require_tokens(req: dict) -> list[int]:
        """The daemon always sends a JSON list of ints; anything else is a
        client error (400), not a 500 raised from inside the engine. bool is
        excluded: isinstance(True, int) in Python."""
        tokens = Handler._require(req, "tokens")
        if not isinstance(tokens, list) or not all(
            isinstance(t, int) and not isinstance(t, bool) for t in tokens
        ):
            raise Handler._BadRequestError("tokens must be a list of integers")
        return tokens

    def _binary(self, code: int, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # Hard cap on request bodies (the daemon is the only client; this bounds a
    # malformed/hostile Content-Length from pinning a worker thread).
    MAX_BODY = 64 * 1024 * 1024

    class _BadRequestError(Exception):
        pass

    def _read_json(self) -> dict:
        raw_len = self.headers.get("Content-Length")
        if raw_len is None:
            return {}
        try:
            length = int(raw_len)
        except (TypeError, ValueError) as exc:
            raise Handler._BadRequestError("invalid Content-Length") from exc
        if length < 0 or length > self.MAX_BODY:
            raise Handler._BadRequestError("Content-Length out of range")
        if length == 0:
            return {}
        body = self.rfile.read(length)
        try:
            return json.loads(body)
        except (ValueError, UnicodeDecodeError) as exc:
            raise Handler._BadRequestError("malformed JSON body") from exc

    def _settle_stream(
        self,
        prompt_tokens: list,
        generated: list,
        cache,
        settle_path: str | None,
    ) -> dict | None:
        """Settle the post-generation cache on the streaming leg, BEFORE the
        done frame is emitted so the terminal frame can report the outcome.

        Returns None when the client did not ask to settle (no field at all,
        distinct from "asked and nothing was written"). Never raises: a settle
        failure must not corrupt a stream whose tokens are already delivered —
        the engine reports the reason and the frame carries it.
        """
        if settle_path is None:
            return None
        settle = getattr(self.engine, "settle", None)
        if settle is None:
            # An engine that does not implement settling writes NOTHING. It
            # must not be assumed to have written: the daemon publishes what
            # the report describes, never what the request asked for.
            return {"written": False, "tokens": len(prompt_tokens) + len(generated)}
        try:
            return settle(prompt_tokens, generated, cache, settle_path)
        except Exception as exc:  # noqa: BLE001 — a settle is best-effort
            traceback.print_exc()
            return {
                "written": False,
                "tokens": len(prompt_tokens) + len(generated),
                "reason": f"settle_failed: {exc}",
            }

    def _stream_ndjson(
        self,
        tokens: list,
        blob_path: str | None,
        max_tokens: int,
        settle_path: str | None = None,
        settle_refusal: bool = False,
    ) -> None:
        """Stream generation as newline-delimited JSON: one line per token,
        then a final done line. Framed by connection close (HTTP/1.0-style)
        so reqwest reads until EOF — no hand-rolled chunked encoding."""
        # Load/validate the checkpoint BEFORE sending 200. A bad blob raises
        # CheckpointRejectedError here and do_POST maps it to a clean 422, which the
        # daemon quarantines. Only the load is primed: first-token decode runs
        # after headers, so a decode error (OOM) truncates the stream without
        # being mistaken for blob corruption. Once bytes are sent we cannot
        # switch to a 422.
        prompt, cache = self.engine.prepare_stream(tokens, blob_path)

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        # Test knob: flood the stream with N bytes and NO newline, then hold
        # the connection open. Exercises the daemon's partial-line cap
        # (MAX_PARTIAL_LINE): the idle budget only governs byte ARRIVALS, so
        # a fast newline-free flood must trip the explicit "unbounded line"
        # upstream error instead of growing the bridge buffer.
        flood = int(os.environ.get("MLXCACHE_STREAM_FLOOD", "0") or 0)
        if flood:
            self.wfile.write(b"x" * flood)
            self.wfile.flush()
            time.sleep(30)
            return

        n = 0
        generated: list[int] = []
        try:
            gen = self.engine.stream_prepared(prompt, cache)
            while n < max_tokens:
                try:
                    token, text = next(gen)
                except StopIteration:
                    break
                self.wfile.write((json.dumps({"token": token, "text": text}) + "\n").encode())
                self.wfile.flush()
                generated.append(token)
                n += 1
            # Settle BEFORE the done frame: the terminal frame must carry the
            # outcome, and a client that stops reading at `done` must not miss
            # it. The cache object here is the one generation just mutated.
            done = {"done": True, "tokens": n}
            settle = self._settle_stream(tokens, generated, cache, settle_path)
            if settle_refusal:
                # The staging pin refused this request's path: report the
                # refusal rather than an absent settle, so the daemon logs a
                # reason instead of wondering whether one was requested.
                settle = {
                    "written": False,
                    "tokens": len(tokens) + len(generated),
                    "reason": "staging_path_refused",
                }
            if settle is not None:
                done["settle"] = settle
            self.wfile.write((json.dumps(done) + "\n").encode())
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            # Client went away mid-stream (R1-4 semantics): stop cleanly.
            self.close_connection = True
        except Exception:  # noqa: BLE001 — after headers, never emit a 2nd response
            # A mid-stream engine failure (OOM, decode error) must not fall
            # through to do_POST's 500 writer: that would append a JSON error
            # body to a live 200 stream, which the daemon reads as a token frame.
            # Truncate the stream by closing, WITHOUT a `done` line, so the daemon
            # sees EOF-without-completion and surfaces an upstream error.
            traceback.print_exc()
            self.close_connection = True

    def do_POST(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler API
        try:
            if self.path == "/tokenize":
                req = self._read_json()
                # Client error, not a 500: a non-string prompt would raise
                # inside the tokenizer (AttributeError/TypeError) after the
                # handler's trust boundary — reject it before any engine call.
                prompt = req.get("prompt")
                if not isinstance(prompt, str):
                    self._json(400, {"error": "prompt must be a string"})
                    return
                t0 = time.perf_counter()
                tokens = self.engine.tokenize(prompt)
                _phase_add("tokenize", (time.perf_counter() - t0) * 1000)
                self._json(
                    200,
                    {
                        "tokens": tokens,
                        "tokenizer_hash": getattr(self.engine, "tokenizer_hash", "synthetic"),
                        "kv_dtype": getattr(self.engine, "kv_dtype", "unknown"),
                        "kv_bits": getattr(self.engine, "kv_bits", 0),
                        "kv_group_size": getattr(self.engine, "kv_group_size", 0),
                        # Connector protocol §2/§4 identity: the engine that
                        # owns this key domain and the signature of the cache
                        # composition it persists. Both are fingerprinted by
                        # the daemon; an engine that omits them defaults to
                        # the mlx-lm domain and an unknown composition.
                        "engine_id": getattr(self.engine, "engine_id", "mlx-lm"),
                        "cache_layout": getattr(self.engine, "cache_layout", ""),
                        # Named KV persistence tier ("" = none): fingerprinted
                        # like kv_bits so a TQ blob never cross-serves f16/q8.
                        "kv_tier": getattr(self.engine, "kv_tier", ""),
                    },
                )
            elif self.path == "/prefill":
                req = self._read_json()
                # Delta prefill (OV3): the daemon passes the matched ancestor
                # checkpoint's absolute path when this request extends one. The
                # engine verifies it as a trust boundary (a mismatching or
                # corrupt ancestor → 422 → quarantine + scratch retry) and
                # prefills only the uncovered delta.
                ancestor = req.get("ancestor_blob_path")
                if ancestor is not None:
                    if not isinstance(ancestor, str) or not ancestor:
                        self._json(400, {"error": "ancestor_blob_path must be a non-empty string"})
                        return
                    try:
                        os.stat(ancestor)
                    except FileNotFoundError:
                        # The index pointed at a blob that is now gone: the same
                        # treatment as a missing generate blob (422 → quarantine
                        # → the daemon retries from scratch).
                        self._json(422, {"error": f"ancestor blob missing: {ancestor}"})
                        return
                t0 = time.perf_counter()
                blob = self.engine.prefill(self._require_tokens(req), ancestor)
                _phase_add("prefill", (time.perf_counter() - t0) * 1000)
                self._binary(200, blob)
            elif self.path == "/generate":
                req = self._read_json()
                tokens = self._require_tokens(req)
                # A MISSING blob is a bad checkpoint: 422 so the daemon
                # quarantines it and retries from scratch. Use os.stat, not
                # os.path.exists, so a transient stat failure (EIO, ENFILE)
                # raises and becomes a 500 instead of being misread as "gone"
                # and retiring a healthy entry. Real engines also fail to load a
                # missing blob; this pre-check only makes the missing case a
                # clean 422 before any bytes are written.
                blob_path = req.get("blob_path")
                if blob_path:
                    try:
                        os.stat(blob_path)
                    except FileNotFoundError:
                        self._json(422, {"error": f"blob missing: {blob_path}"})
                        return
                # Validate max_tokens before any bytes: a non-int here would
                # blow up mid-stream after headers are sent (truncated 200), and
                # a negative/huge value silently changes streaming. The daemon
                # sends a bounded int; reject anything else as a client error.
                max_tokens = req.get("max_tokens", 64)
                if (
                    not isinstance(max_tokens, int)
                    or isinstance(max_tokens, bool)
                    or max_tokens < 0
                ):
                    self._json(400, {"error": "max_tokens must be a non-negative integer"})
                    return
                # An optional filesystem path the engine writes the settled
                # checkpoint to. Validated before any generation work starts: a
                # bad type here must be a clean 400, and an empty-string path
                # would silently mean "settle somewhere unusable".
                #
                # The path is ALSO confined to the daemon's staging naming
                # convention (`.settle-<pid>-<seq>-<nanos>.staging`, generated
                # by the daemon inside its own blob dir). This server is an
                # unauthenticated loopback HTTP server: without the confinement
                # any local process could hand it an arbitrary path and make
                # the engine create or clobber a file there. The daemon is the
                # only intended caller and always sends a convention-named
                # path, so anything else is refused rather than written.
                settle_path = req.get("settle_path")
                settle_refused = False
                if settle_path is not None and (
                    not isinstance(settle_path, str)
                    or not settle_path
                    or not _is_staging_shaped(settle_path)
                ):
                    # Not a path the daemon would ever send: refuse the request.
                    self._json(400, {"error": "settle_path must be a daemon staging path"})
                    return
                if settle_path is not None and not _staging_dir_matches(settle_path):
                    # A staging-shaped path in a directory other than the
                    # pinned one. Nothing is written either way, so degrade
                    # the best-effort settle instead of failing the whole
                    # (already-completable) generation -- a daemon restarted
                    # against a new blob dir must not 502 every request
                    # until the sidecar restarts.
                    settle_refused = True
                    settle_path = None

                if req.get("stream"):
                    self._stream_ndjson(
                        tokens,
                        req.get("blob_path"),
                        max_tokens,
                        settle_path,
                        settle_refusal=settle_refused,
                    )
                else:
                    blob_path = req.get("blob_path")
                    settle: dict | None = None
                    collect = getattr(self.engine, "generate_collect", None)
                    if collect is not None:
                        # The only path that can settle: it returns the cache
                        # generation mutated. Engines without it fall through to
                        # the contract-level entries below, which have nowhere to
                        # surface post-generation KV.
                        out, text, _cache, settle = collect(
                            tokens, max_tokens, blob_path, settle_path
                        )
                    else:
                        gen = getattr(self.engine, "generate_with_text", None)
                        if gen is not None:
                            out, text = gen(tokens, max_tokens, blob_path)
                        elif hasattr(self.engine, "stream"):
                            # Contract-conforming engine without generate_with_text:
                            # collect from its stream so non-stream clients still
                            # get detokenized text (api-contract review: falling
                            # through to empty choices[].content is a silent
                            # compatibility trap for third-party adapters).
                            out, pieces = [], []
                            for i, (tok, piece) in enumerate(self.engine.stream(tokens, blob_path)):
                                out.append(tok)
                                pieces.append(piece)
                                if i + 1 >= max_tokens:
                                    break
                            text = "".join(pieces)
                        else:
                            out = self.engine.generate(tokens, max_tokens)
                            text = ""
                        if settle_path is not None:
                            # An engine with no collecting entry cannot hand out
                            # the cache it just used: report that nothing was
                            # written rather than pretending it settled.
                            settle = {
                                "written": False,
                                "tokens": len(tokens) + len(out),
                                "reason": "engine_cannot_settle",
                            }
                    if settle_refused:
                        settle = {
                            "written": False,
                            "tokens": len(tokens) + len(out),
                            "reason": "staging_path_refused",
                        }
                    body: dict = {"tokens": out, "text": text}
                    if settle is not None:
                        body["settle"] = settle
                    self._json(200, body)

            else:
                self._json(404, {"error": f"unknown path {self.path}"})
        except Handler._BadRequestError as exc:
            # Client error: 400 with a stable message (no internal detail).
            self._json(400, {"error": str(exc)})
        except CheckpointRejectedError as exc:
            # 422: the checkpoint is bad, not the request or the engine. The
            # daemon quarantines the entry on this status and retries scratch.
            self._json(422, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001 — map all engine errors to 500 JSON
            traceback.print_exc()
            self._json(500, {"error": str(exc)})

    def do_GET(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler API
        if self.path == "/health":
            self._json(200, {"status": "ok"})
        elif self.path == "/stats":
            # Test observability: how many prefills the engine actually ran.
            # Proves single-flight coalesced concurrent identical requests.
            # Phase timers (B2) decompose where prefill time goes (cache load,
            # delta prefill, save) and what the delta path actually saved.
            engine = self.engine
            self._json(
                200,
                {
                    "prefill_count": getattr(engine, "prefill_count", None),
                    "generate_count": getattr(engine, "_generate_count", -1),
                    "last_prefill_tokens": getattr(engine, "last_prefill_tokens", 0),
                    "last_prefill_delta_tokens": getattr(engine, "last_prefill_delta_tokens", 0),
                    "phases": _phase_stats(),
                },
            )
        else:
            self._json(404, {"error": f"unknown path {self.path}"})


class BurstServer(ThreadingHTTPServer):
    """Sidecar HTTP server with a production-shaped listen backlog.

    Stock ThreadingHTTPServer has request_queue_size = 5: a connect burst
    larger than 5 (single-flight releases a whole coalesced burst at once,
    plus health probes) resets surplus connections on a loaded machine,
    which the daemon classifies as adapter_unreachable → 503 — the exact
    failure two e2e single-flight tests hit under parallel test load
    (2026-10-04). Backlog 128 + daemon threads: accept bursts drain fast
    because every request gets a thread immediately.
    """

    request_queue_size = 128
    daemon_threads = True


def serve(addr: str = "127.0.0.1", port: int = 8421) -> None:
    model_id = os.environ.get("MLXCACHE_MODEL", "synthetic-model")
    Handler.engine = make_engine(model_id)
    server = BurstServer((addr, port), Handler)
    print(f"sidecar: {Handler.engine.name} engine on {addr}:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    serve(
        addr=os.environ.get("MLXCACHE_SIDECAR_ADDR", "127.0.0.1"),
        port=int(os.environ.get("MLXCACHE_SIDECAR_PORT", "8421")),
    )
