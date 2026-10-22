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
import functools
import hashlib
import io
import logging
import os
import stat
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


def _artifact_bytes(path: Path) -> bytes | None:
    """Read a model artifact's bytes, refusing a symlink that escapes the model.

    HF cache layouts symlink every snapshot file into a sibling ``blobs/``
    directory (``snapshots/<rev>/tokenizer.json -> ../../blobs/<sha>``), so a
    blanket ``O_NOFOLLOW`` hashed nothing for EVERY cached model — every
    HF-cached model then fingerprinted as ``"missing"``, erasing tokenizer
    identity for all of them (a tokenizer swap under the same model dir went
    undetected, and all symlinked models aliased one sentinel). The two
    engine paths hash different artifacts by design (the wrapper hashes
    these file bytes; the daemon hashes the vocabulary), so this guard
    restores identity pinning WITHIN each path, not agreement between them.

    The guard's intent survives: a link is followed only when its target
    resolves inside the model directory itself or inside a ``blobs/``
    sibling of it — the model's own storage, which is the declared trust
    boundary: an attacker with write access to that tree could already swap
    blob contents or plant a regular file. Anything resolving elsewhere — a
    link pointing at another model's artifact or at an arbitrary file — is
    still refused. The final open uses ``O_NOFOLLOW`` so a swap of the FINAL
    component between the check and the open fails closed; the opened
    file's ``(st_dev, st_ino)`` is then reconciled against the checked
    path's so an intermediate-directory swap mid-window is caught too
    (usual inode-reuse caveat applies). A target that is a directory (a
    corrupt pull can leave the link at ``blobs/`` itself) reads as missing,
    not as an error. Swap DETECTION is restart-scoped: the fingerprint is
    memoized for the process lifetime, so a tokenizer swapped under a live
    server is caught on its next restart, not mid-run.
    """
    target = path
    if path.is_symlink():
        resolved = Path(os.path.realpath(path))
        model_dir = Path(os.path.realpath(path.parent))
        # HF cache: <hub>/models--org--name/{snapshots/<rev>,blobs/<sha>} with
        # model_dir being the SNAPSHOT dir, so blobs/ sits two levels up —
        # trusted ONLY in that shape; a plain local model may keep a blobs/
        # sibling one level up. Anything resolving outside those — another
        # model's artifact, an arbitrary file — is a forgery and stays
        # refused.
        allowed = [model_dir, model_dir.parent / "blobs"]
        if model_dir.parent.name == "snapshots":
            allowed.append(model_dir.parent.parent / "blobs")
        if not any(resolved.is_relative_to(root) for root in allowed):
            return None
        target = resolved
    try:
        checked = os.stat(target)
        if stat.S_ISDIR(checked.st_mode):
            return None  # a directory is not an artifact (corrupt-pull shape)
        fd = os.open(target, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError:
        return None
    try:
        # The context manager owns the fd from here (a manual close would
        # double-close on a read error and leak on the mismatch return).
        # O_NOFOLLOW only guards the final component, so an intermediate
        # directory swapped for a symlink mid-window would otherwise be
        # followed silently; the inode reconciliation catches it.
        with os.fdopen(fd, "rb") as fh:
            opened = os.fstat(fh.fileno())
            if (opened.st_dev, opened.st_ino) != (checked.st_dev, checked.st_ino):
                return None
            return fh.read()
    except OSError:
        return None


def fingerprint_for_model_dir(
    model_path: str | os.PathLike,
    kv_bits: int = 0,
    kv_group_size: int = 0,
    model_id_override: str | None = None,
    kv_tier: str = "",
) -> Fingerprint:
    """Deterministic fingerprint for an mlx-lm model directory.

    ``model_id`` is the REALPATH of the model directory (stable across
    restarts and ~ vs absolute invocations). The tokenizer hash reads the
    artifact bytes without following a symlink that escapes the model tree.
    ``kv_tier`` names a persistence tier the scalar bits pair cannot express
    (the TQ codecs); empty = none.
    """
    p = Path(model_path)
    payload = _artifact_bytes(p / "tokenizer.json")
    tokenizer_hash = hashlib.sha256(payload).hexdigest() if payload is not None else "missing"
    return Fingerprint(
        model_id=model_id_override or os.path.realpath(p),
        tokenizer_hash=tokenizer_hash,
        kv_dtype="mlx-lm",
        kv_layout_version=1,
        kv_bits=kv_bits,
        kv_group_size=kv_group_size,
        kv_tier=kv_tier,
    )


def _memoized_dir_fingerprint(
    model_path: str,
    kv_bits: int,
    kv_group_size: int,
    model_id_override: str | None,
    kv_tier: str,
) -> Fingerprint:
    """Cache per (path, tier): tokenizer.json re-reads cost milliseconds per
    REQUEST on the serving thread otherwise (multi-MB sha256 each call)."""
    key = (model_path, kv_bits, kv_group_size, model_id_override, kv_tier)
    fp = _FP_CACHE.get(key)
    if fp is None:
        fp = fingerprint_for_model_dir(
            model_path,
            kv_bits=kv_bits,
            kv_group_size=kv_group_size,
            model_id_override=model_id_override,
            kv_tier=kv_tier,
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

    def _persistent_fp(self, model: Any) -> Fingerprint | None:
        """The fingerprint to key the STORE by, or None when the identity is
        unresolved. `default_fingerprint_for` pins unresolved identities as
        the "missing"/"unresolved" sentinel; this is the enforcement point —
        a sentinel must never reach the store in either direction (a lookup
        would miss anyway, but a PUT would persist an entry no future
        resolved identity can ever adopt, under an id two different
        unresolved spellings share)."""
        fp = self._fp(model)
        if fp.tokenizer_hash == "missing" or fp.model_id == "unresolved":
            return None
        return fp

    def fetch_nearest_cache(self, model: Any, tokens: list[int]):
        # super() deepcopies the memory entry before we know whether disk
        # wins; avoiding that copy means reimplementing upstream's trie
        # walk. Accepted cost: the copy is GB-scale ONLY on entries big
        # enough that the disk alternative saves a full prefill.
        cache, rest = super().fetch_nearest_cache(model, tokens)
        if not rest:
            return cache, rest  # exact in-memory hit: nothing on disk can beat it
        try:
            fp = self._persistent_fp(model)
            if fp is None:
                return cache, rest  # unresolved identity: store is unreachable by design
            matched, ref = self.store.lookup(ENGINE_ID, fp, list(tokens))
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
        # Resolve the store identity FIRST: an unresolved identity pays for
        # no persistence work at all — the deepcopy/trim/save below would
        # process a multi-GB cache only for the put to be refused at the end.
        try:
            fp = self._persistent_fp(model)
            if fp is None:
                return  # unresolved identity: never persist under a sentinel
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
                self.store.put(ENGINE_ID, fp, list(tokens), payload)
                self.persisted += 1
        except Exception:  # noqa: BLE001 — persistence failure never breaks serving
            self.persist_errors += 1
            log.warning("disk cache publish failed", exc_info=True)


@functools.lru_cache(maxsize=64)
def _resolve_snapshot(raw: str, marker: str | None = "config.json") -> str:
    """Resolve a bare HF id (org/name) or relative path to a LOCAL snapshot
    directory, without ever downloading; returns "" when nothing resolves.
    An unresolved identifier must not be fingerprinted as itself (two
    different ids would then collide on whatever string was passed), and
    must not invent a path the tokenizer hasher would refuse anyway.
    `marker` names the file whose presence identifies a directory:
    config.json for models and drafts, adapter_config.json for LoRA
    adapters; None accepts any existing directory (no marker required).
    Memoized per (raw, marker) — failures included, so a resolution miss
    costs one probe per identifier for the process lifetime."""
    expanded = os.path.abspath(os.path.expanduser(raw))
    if marker is None:
        return expanded if Path(expanded).is_dir() else ""
    if (Path(expanded) / marker).is_file():
        return expanded
    try:
        from huggingface_hub import snapshot_download

        snap = snapshot_download(raw, local_files_only=True)
        if snap and (Path(snap) / marker).is_file():
            return snap
    except Exception:  # noqa: BLE001 — falls through to the sentinel
        pass
    return ""


def default_fingerprint_for(model_key, kv_bits: int = 0, kv_group_size: int = 0, kv_tier: str = ""):
    """Default fingerprint source: model dir realpath + tokenizer bytes,
    with the ADAPTER and DRAFT MODEL folded into the model id (upstream's
    memory trie keys the full triple — a LoRA or draft-model switch must
    never share KV with the base model). Memoized per (path, tier).

    Non-directory identifiers (a bare `org/name`, a relative path) are
    resolved to local snapshot directories FIRST: fingerprinting the raw
    string would key the cache on the spelling of a name. Adapters resolve
    against adapter_config.json (mlx-lm requires it to load a LoRA anyway);
    models and drafts against config.json. Any identifier that was SUPPLIED
    but fails to resolve DISABLES persistent caching via the `"missing"`
    tokenizer sentinel — a vanished adapter must not silently collapse to
    the empty component and serve the base model's KV as if no LoRA was
    requested."""
    model_raw, adapter_raw, draft_raw = (list(model_key) + [None, None, None])[:3]
    model = _resolve_snapshot(model_raw) if model_raw else ""
    adapter = _resolve_snapshot(adapter_raw, marker="adapter_config.json") if adapter_raw else ""
    draft = _resolve_snapshot(draft_raw) if draft_raw else ""
    supplied_but_unresolved = bool(
        (model_raw and not model) or (adapter_raw and not adapter) or (draft_raw and not draft)
    )
    if supplied_but_unresolved:
        # Unresolvable identity: return a fingerprint whose tokenizer hash is
        # the "missing" sentinel — the wrapper treats it as non-persistent.
        return Fingerprint(
            model_id="unresolved",
            tokenizer_hash="missing",
            kv_dtype="f16",
            kv_layout_version=1,
            kv_bits=kv_bits,
            kv_group_size=kv_group_size,
            cache_layout="",
            kv_tier=kv_tier,
        )
    adapter = os.path.realpath(adapter) if adapter else ""
    draft = os.path.realpath(draft) if draft else ""
    model_id = "|".join((os.path.realpath(model), adapter, draft))
    return _memoized_dir_fingerprint(
        os.path.realpath(model), kv_bits, kv_group_size, model_id, kv_tier
    )
