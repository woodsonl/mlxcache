"""PersistentPromptCache — the L1 integration seam (connector protocol §8).

Subclass mlx-lm's in-memory ``LRUPromptCache`` so every served model gains
disk-backed prefix reuse with zero daemon involvement:

- ``fetch_nearest_cache``: memory first (an exact in-memory hit returns
  immediately); when memory leaves a remainder, the store is consulted,
  and a disk entry that covers MORE positions wins. Disk reuse is an
  optimization only — any store failure falls back to the memory result,
  never to an error.
- ``insert_cache``: memory exactly as upstream, plus a durable publish of
  the token stream and its KV state. The persisted payload follows the
  protocol's §3.2 writer convention EXACTLY as the daemon does: the state
  handed to insert covers ALL of T (generate_step feeds each token before
  yielding it), so the payload is trimmed ONE position before saving —
  a blob at T covers T[:-1], the daemon-convention. Non-trimmable caches
  and other cache_types are not persisted (a wrong-convention blob would
  double-feed a token on resume: silent wrong KV).

Blob convention (pinned by the daemon + sidecar suites): a payload stored
at token stream T covers exactly T[:-1] positions; a lookup returning
``matched`` therefore resumes by feeding ``tokens[matched-1:]``.
"""

from __future__ import annotations

import copy
import hashlib
import io
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mlx_lm.models.cache import LRUPromptCache
from mlxcache_store import Fingerprint, Store

ENGINE_ID = "mlx-lm"  # the default engine: zero fold bytes (protocol §4)

log = logging.getLogger("mlxcache_serve")

# Injectable (test) seams; defaults bind the real mlx-lm codec lazily.
SaveFn = Callable[[Any], bytes]
LoadFn = Callable[[bytes], Any]
FingerprintSource = Callable[[Any], Fingerprint]


def _default_save(cache: Any) -> bytes:
    from mlx_lm.models.cache import save_prompt_cache  # noqa: PLC0415

    buf = io.BytesIO()
    save_prompt_cache(buf, cache)
    return buf.getvalue()


def _default_load(payload: bytes) -> Any:
    from mlx_lm.models.cache import load_prompt_cache  # noqa: PLC0415

    # mlx infers the safetensors format from the file EXTENSION and reads
    # ``.name`` — a bare BytesIO raises AttributeError (the pinned landmine).
    blob = io.BytesIO(payload)
    blob.name = "checkpoint.safetensors"
    return load_prompt_cache(blob)


def fingerprint_for_model_dir(
    model_path: str | os.PathLike,
    kv_bits: int = 0,
    kv_group_size: int = 0,
    model_id_override: str | None = None,
) -> Fingerprint:
    """Deterministic fingerprint for an mlx-lm model directory.

    ``model_id`` is the REALPATH of the model directory (stable across
    restarts and ~ vs absolute invocations). The tokenizer hash reads the
    artifact bytes without following a symlink planted at tokenizer.json.
    """
    p = Path(model_path)
    tokenizer = p / "tokenizer.json"
    tokenizer_hash = "missing"
    try:
        fd = os.open(tokenizer, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError:
        tokenizer_hash = "missing"
    else:
        with os.fdopen(fd, "rb") as fh:
            tokenizer_hash = hashlib.sha256(fh.read()).hexdigest()
    return Fingerprint(
        model_id=model_id_override or os.path.realpath(p),
        tokenizer_hash=tokenizer_hash,
        kv_dtype="mlx-lm",
        kv_layout_version=1,
        kv_bits=kv_bits,
        kv_group_size=kv_group_size,
    )


def _memoized_dir_fingerprint(
    model_path: str, kv_bits: int, kv_group_size: int, model_id_override: str | None
) -> Fingerprint:
    """Cache per (path, tier): tokenizer.json re-reads cost milliseconds per
    REQUEST on the serving thread otherwise (multi-MB sha256 each call)."""
    key = (model_path, kv_bits, kv_group_size, model_id_override)
    fp = _FP_CACHE.get(key)
    if fp is None:
        fp = fingerprint_for_model_dir(
            model_path,
            kv_bits=kv_bits,
            kv_group_size=kv_group_size,
            model_id_override=model_id_override,
        )
        _FP_CACHE[key] = fp
    return fp


_FP_CACHE: dict[tuple, Fingerprint] = {}


class PersistentPromptCache(LRUPromptCache):
    """``LRUPromptCache`` + a persistent ``mlxcache_store`` behind it.

    ``fingerprint`` is either a ``Fingerprint`` (single-model server) or a
    callable taking the server's ``model_key`` — the ``(model, adapter,
    draft_model)`` tuple — and returning one. The DEFAULT fingerprint
    (``default_fingerprint_for``) folds the adapter and draft model into
    the model id: LoRA-B must never serve LoRA-A's KV.
    """

    def __init__(
        self,
        store: Store,
        fingerprint: Fingerprint | FingerprintSource,
        max_size: int = 10,
        max_bytes: int = 1 << 63,
        save_fn: SaveFn | None = None,
        load_fn: LoadFn | None = None,
    ):
        super().__init__(max_size=max_size, max_bytes=max_bytes)
        self.store = store
        self.fingerprint = fingerprint
        self._save: SaveFn = save_fn or _default_save
        self._load: LoadFn = load_fn or _default_load
        # Observability (tests + logging); not a contract.
        self.disk_hits = 0
        self.persisted = 0
        self.reuse_errors = 0
        self.persist_errors = 0

    def _fp(self, model: Any) -> Fingerprint:
        fp = self.fingerprint
        return fp(model) if callable(fp) else fp

    def fetch_nearest_cache(self, model: Any, tokens: list[int]):
        # super() deepcopies the memory entry before we know whether disk
        # wins; avoiding that copy means reimplementing upstream's trie
        # walk. Accepted cost: the copy is GB-scale ONLY on entries big
        # enough that the disk alternative saves a full prefill.
        cache, rest = super().fetch_nearest_cache(model, tokens)
        if not rest:
            return cache, rest  # exact in-memory hit: nothing on disk can beat it
        try:
            matched, ref = self.store.lookup(ENGINE_ID, self._fp(model), list(tokens))
            if ref is None or matched < 2:
                return cache, rest
            disk_covered = matched - 1  # payload at T covers T[:-1]
            memory_covered = len(tokens) - len(rest)
            if disk_covered <= memory_covered:
                return cache, rest  # memory already covers at least as much
            disk_cache = self._load(self.store.fetch(ref))
            self.disk_hits += 1
            return disk_cache, tokens[disk_covered:]
        except Exception:  # noqa: BLE001 — reuse is best-effort, serving is not
            self.reuse_errors += 1
            log.warning("disk cache reuse failed; serving from memory", exc_info=True)
            return cache, rest

    def insert_cache(
        self,
        model: Any,
        tokens: list[int],
        prompt_cache: list[Any],
        *,
        cache_type: str = "assistant",
    ):
        super().insert_cache(model, tokens, prompt_cache, cache_type=cache_type)
        if cache_type != "assistant" or len(tokens) < 2:
            # Segment saves / mid-stream re-saves do not follow the T[:-1]
            # coverage convention (see module docstring); sub-2 streams
            # cover nothing (§3.2).
            return
        try:
            from mlx_lm.models.cache import (  # noqa: PLC0415
                can_trim_prompt_cache,
                trim_prompt_cache,
            )

            if not can_trim_prompt_cache(prompt_cache):
                # Non-trimmable caches cannot be trimmed to the convention;
                # persisting one would violate §3.2 (wrong KV on resume).
                return
            trimmed = copy.deepcopy(prompt_cache)
            if trim_prompt_cache(trimmed, 1) != 1:
                return  # could not trim exactly one position: do not persist
            payload = self._save(trimmed)
            if payload:
                self.store.put(ENGINE_ID, self._fp(model), list(tokens), payload)
                self.persisted += 1
        except Exception:  # noqa: BLE001 — persistence failure never breaks serving
            self.persist_errors += 1
            log.warning("disk cache publish failed", exc_info=True)


def default_fingerprint_for(model_key, kv_bits: int = 0, kv_group_size: int = 0):
    """Default fingerprint source: model dir realpath + tokenizer bytes,
    with the ADAPTER and DRAFT MODEL folded into the model id (upstream's
    memory trie keys the full triple — a LoRA or draft-model switch must
    never share KV with the base model). Memoized per (path, tier)."""
    model, adapter, draft = (list(model_key) + [None, None, None])[:3]
    adapter = os.path.realpath(adapter) if adapter else ""
    draft = os.path.realpath(draft) if draft else ""
    model_id = "|".join((os.path.realpath(model), adapter, draft))
    return _memoized_dir_fingerprint(os.path.realpath(model), kv_bits, kv_group_size, model_id)
