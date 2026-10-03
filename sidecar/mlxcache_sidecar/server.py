"""Sidecar HTTP server: the mlx-lm compatibility adapter.

Endpoints (daemon -> sidecar protocol):
- POST /tokenize  {"prompt": str} -> {"tokens": [u32], "tokenizer_hash": str}
- POST /prefill   {"prompt": str, "tokens": [u32]} -> KV checkpoint blob (binary)
- POST /generate  {"tokens": [u32], "max_tokens": int, "blob_path": str|null,
                   "stream": bool} -> NDJSON token stream

Engine selection: MLXCACHE_ENGINE=synthetic (default, hermetic) or mlx-lm
(lazy import; requires a local model). The daemon is the only client; this
server binds 127.0.0.1 by default.

Run: uv run python -m mlxcache_sidecar.server
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


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

    # Read first: an OSError here is TRANSIENT, not a bad checkpoint (see
    # docstring). A MISSING file is also surfaced as rejection: the daemon
    # only passes paths its index currently points at, so a miss here means
    # the entry is already bad (retired/raced); 422 cleans it up.
    with open(blob_path, "rb") as fh:
        raw = fh.read()
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
    except (ValueError, KeyError, TypeError) as exc:
        # A header that does not decode or lacks required fields is a bad
        # checkpoint: retire it rather than 500 forever while it stays
        # selectable.
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
        # The adapter is a trust boundary: verify the blob really covers this
        # request's prefix. A disagreement means the file does not match the
        # index entry that pointed at it (mislabeled/corrupt): retire it, do
        # not silently resume from or ignore wrong KV.
        if meta.tokens != tokens[:prefix_len]:
            raise CheckpointRejectedError("checkpoint prefix does not match the request")
    # A multi-token checkpoint with no KV payload is CORRUPT (a truncated
    # write), not uncacheable: reject so the daemon quarantines the entry.
    if not payload:
        raise CheckpointRejectedError("checkpoint payload is empty")
    if check_safetensors:
        # Validate the safetensors framing ourselves, BEFORE handing it to MLX.
        # MLX's native parser raises RuntimeError for a bad header length, the
        # same type it uses for a transient OS read failure, so we cannot
        # classify from the exception alone. A malformed inner header is
        # deterministic corruption: reject it here so it is retired, while
        # RuntimeError from the loader below stays transient (500).
        if not _valid_safetensors(payload):
            raise CheckpointRejectedError("checkpoint payload is not valid safetensors")
    return meta, payload, True


class SyntheticEngine:
    """Hermetic engine: deterministic tokens + KV bytes, no MLX dependency."""

    name = "synthetic"

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.prefill_count = 0
        self.tokenizer_hash = "synthetic"
        self.kv_dtype = "synthetic"
        # Test knob: emulate a tokenizer that grows a base prompt's tokens —
        # tokenize(base) is a strict token-prefix of tokenize(base + rest), so
        # daemon tests can exercise partial hits and delta prefill (which the
        # fixed 8-token hash alone can never produce).
        self._grow_base = os.environ.get("MLXCACHE_TOKENIZE_GROW")
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
                segments = (
                    content[len(self._grow_base) :]
                    .removeprefix(" | ")
                    .split(" | ")
                )
                return base + [
                    t for seg in segments for t in self._hash_tokens(seg)
                ]
        return self._hash_tokens(prompt)

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
        payload = b"".join(
            ((t * 31 + i) & 0xFFFFFFFF).to_bytes(4, "little") * 256 for i, t in enumerate(tokens)
        )
        # Raw payload only; the daemon writes the checkpoint header.
        return payload

    def generate(self, tokens: list[int], max_tokens: int) -> list[int]:
        # Deterministic continuation: token ids derived from context length.
        base = len(tokens)
        return [(base + i) % 2**31 for i in range(max_tokens)]

    def stream(self, tokens: list[int], blob_path: str | None):
        # Synthetic engine streams its deterministic tokens as text pieces.
        for i, t in enumerate(self.generate(tokens, 64)):
            yield t, f"tok{i} "

    def prepare_stream(self, tokens: list[int], blob_path: str | None):
        """Load any checkpoint and return (prompt, cache) BEFORE streaming.

        Split from generation so the handler can surface a rejected checkpoint as
        a clean 422 (which the daemon quarantines) rather than a mid-stream
        failure. The synthetic engine has no cache; returns the prompt as-is.
        """
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

    def __init__(self, model_id: str) -> None:
        from mlx_lm import load  # noqa: PLC0415 — lazy, only when selected

        self.model_id = model_id
        self.model, self.tokenizer = load(model_id)
        # Hash the vocabulary, not the model name: R1-2 pins the tokenizer
        # artifact, and two different tokenizers can share a name_or_path-free
        # identity. The vocab is deterministic and artifact-derived.
        self.tokenizer_hash = self._hash_vocab()
        self.kv_dtype = self._kv_dtype()
        self.prefill_count = 0

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
        vocab = getattr(self.tokenizer, "get_vocab", None)
        if callable(vocab):
            items = sorted(vocab().items())
            payload = repr(items).encode()
        else:
            # Fall back to whatever identity the tokenizer exposes.
            payload = repr(getattr(self.tokenizer, "name_or_path", self.model_id)).encode()
        return hashlib.sha256(payload).hexdigest()[:16]

    def tokenize(self, prompt: str) -> list[int]:
        return self.tokenizer.encode(prompt)

    def _prefill_cache(self, tokens: list[int]):
        """Run tokens through the model filling a fresh prompt cache. Returns
        (cache, last_logits) with the KV state resident and evaluated."""
        import mlx.core as mx  # noqa: PLC0415
        from mlx_lm.models.cache import make_prompt_cache  # noqa: PLC0415

        cache = make_prompt_cache(self.model)
        inp = mx.array(tokens)[None]
        logits = self.model(inp, cache=cache)
        mx.eval([c.state for c in cache], logits)
        return cache, logits

    def _load_cache_from_payload(self, payload: bytes):
        """Load a prompt cache from an ALREADY-VALIDATED safetensors payload.

        Delta prefill needs the same load the generate path does, but without a
        second disk read (the caller just validated the wire checkpoint). A
        pure-Python parse/schema failure is corruption → CheckpointRejectedError
        (422/quarantine); OSError/RuntimeError stay transient (500)."""
        from mlx_lm.models.cache import load_prompt_cache  # noqa: PLC0415

        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as fh:
            tmp = fh.name
            fh.write(payload)
        try:
            try:
                return load_prompt_cache(tmp)
            except (ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:
                raise CheckpointRejectedError(
                    f"checkpoint failed to load: {type(exc).__name__}: {exc}"
                ) from exc
        finally:
            os.unlink(tmp)

    def prefill(self, tokens: list[int], ancestor_blob_path: str | None = None) -> bytes:
        import mlx.core as mx  # noqa: PLC0415
        from mlx_lm.models.cache import save_prompt_cache  # noqa: PLC0415

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
        # safetensors needs a real file; write to a temp, then read the bytes.
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as fh:
            tmp = fh.name
        try:
            t0 = time.perf_counter()
            save_prompt_cache(tmp, cache)
            _phase_add("cache_save", (time.perf_counter() - t0) * 1000)
            with open(tmp, "rb") as fh:
                payload = fh.read()
        finally:
            os.unlink(tmp)
        mx.clear_cache()

        # Return the raw safetensors payload. The daemon owns the checkpoint
        # header (it writes meta via publish_atomic) and the wire format is
        # header + this payload; encoding one here would double-wrap it.
        return payload

    def generate(self, tokens: list[int], max_tokens: int) -> list[int]:
        """Generate continuing from an optional persisted cache.

        When the daemon supplies a blob the KV cache is loaded from it and the
        uncovered tail is prefilled; otherwise generation runs from scratch. The
        daemon supplies the blob via the /generate request body; see the handler.
        """
        return self._generate_with_cache(tokens, max_tokens, cache=None)

    def stream(self, tokens: list[int], blob_path: str | None):
        """Yield (token_id, text) for this request, resuming from `blob_path`
        when present. This is the streaming entry the handler drives."""
        prompt, cache = self.prepare_stream(tokens, blob_path)
        yield from self.stream_prepared(prompt, cache)

    def prepare_stream(self, tokens: list[int], blob_path: str | None):
        """Load the checkpoint (may raise for a corrupt blob) and return
        (prompt, cache). Done before headers are sent so a load failure is a
        clean 422, not a truncated stream. First-token decode happens later, in
        stream_prepared, so a decode error does not look like blob corruption."""
        cache = None
        prompt = tokens
        if blob_path:
            cache, prompt = self._load_cache_delta(tokens, blob_path)
            if cache is None:
                prompt = tokens  # unexpectable cache: scratch
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
        """Yield (token_id, text_piece) per decode step from mlx-lm."""
        import mlx.core as mx  # noqa: PLC0415
        from mlx_lm import stream_generate  # noqa: PLC0415

        for resp in stream_generate(
            self.model, self.tokenizer, prompt=mx.array(tokens), prompt_cache=cache
        ):
            yield resp.token, resp.text

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
        from mlx_lm.models.cache import load_prompt_cache  # noqa: PLC0415

        meta, payload, usable = read_wire_checkpoint(blob_path, tokens)
        if not usable:
            return None, tokens
        covered = len(meta.tokens) - 1
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as fh:
            tmp = fh.name
            fh.write(payload)
        try:
            # A pure-Python parse/schema failure is corruption: reject so the
            # daemon retires it. MemoryError, OSError, and RuntimeError are
            # transient (a real OOM, a disk fault; MLX's native reader raises
            # RuntimeError on an OS read failure): let them propagate as a 500 so
            # a healthy checkpoint is not retired.
            try:
                return load_prompt_cache(tmp), tokens[covered:]
            except (ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:
                raise CheckpointRejectedError(
                    f"checkpoint failed to load: {type(exc).__name__}: {exc}"
                ) from exc
        finally:
            os.unlink(tmp)

    def generate_from_blob(self, tokens: list[int], blob_path: str, max_tokens: int) -> list[int]:
        """Resume generation from a persisted wire-format checkpoint (a cache
        hit): strip the daemon header, load the prompt cache, generate.

        The checkpoint holds KV for a token prefix; the delta (uncached tail)
        is what the model still needs to see. We pass the tail so the cache
        supplies the cached prefix and generation continues identically to a
        scratch run over the full prompt (thesis guard, T2)."""
        cache, prompt = self._load_cache_delta(tokens, blob_path)
        if cache is None:
            return self._generate_with_cache(tokens, max_tokens, cache=None)
        return self._generate_with_cache(prompt, max_tokens, cache)


def make_engine(model_id: str) -> SyntheticEngine | MlxLmEngine:
    if os.environ.get("MLXCACHE_ENGINE", "synthetic") == "mlx-lm":
        return MlxLmEngine(model_id)
    return SyntheticEngine(model_id)


class Handler(BaseHTTPRequestHandler):
    engine: object = None  # set by serve()

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

    def _stream_ndjson(self, tokens: list, blob_path: str | None, max_tokens: int) -> None:
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

        n = 0
        try:
            gen = self.engine.stream_prepared(prompt, cache)
            while n < max_tokens:
                try:
                    token, text = next(gen)
                except StopIteration:
                    break
                self.wfile.write((json.dumps({"token": token, "text": text}) + "\n").encode())
                self.wfile.flush()
                n += 1
            self.wfile.write((json.dumps({"done": True, "tokens": n}) + "\n").encode())
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
                t0 = time.perf_counter()
                tokens = self.engine.tokenize(req.get("prompt", ""))
                _phase_add("tokenize", (time.perf_counter() - t0) * 1000)
                self._json(
                    200,
                    {
                        "tokens": tokens,
                        "tokenizer_hash": getattr(self.engine, "tokenizer_hash", "synthetic"),
                        "kv_dtype": getattr(self.engine, "kv_dtype", "unknown"),
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
                blob = self.engine.prefill(self._require(req, "tokens"), ancestor)
                _phase_add("prefill", (time.perf_counter() - t0) * 1000)
                self._binary(200, blob)
            elif self.path == "/generate":
                req = self._read_json()
                tokens = self._require(req, "tokens")
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
                if req.get("stream"):
                    self._stream_ndjson(tokens, req.get("blob_path"), max_tokens)
                else:
                    blob_path = req.get("blob_path")
                    if blob_path and hasattr(self.engine, "generate_from_blob"):
                        out = self.engine.generate_from_blob(tokens, blob_path, max_tokens)
                    else:
                        out = self.engine.generate(tokens, max_tokens)
                    self._json(200, {"tokens": out})
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
                    "last_prefill_tokens": getattr(engine, "last_prefill_tokens", 0),
                    "last_prefill_delta_tokens": getattr(
                        engine, "last_prefill_delta_tokens", 0
                    ),
                    "phases": _phase_stats(),
                },
            )
        else:
            self._json(404, {"error": f"unknown path {self.path}"})


def serve(addr: str = "127.0.0.1", port: int = 8421) -> None:
    model_id = os.environ.get("MLXCACHE_MODEL", "synthetic-model")
    Handler.engine = make_engine(model_id)
    server = ThreadingHTTPServer((addr, port), Handler)
    print(f"sidecar: {Handler.engine.name} engine on {addr}:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    serve(
        addr=os.environ.get("MLXCACHE_SIDECAR_ADDR", "127.0.0.1"),
        port=int(os.environ.get("MLXCACHE_SIDECAR_PORT", "8421")),
    )
