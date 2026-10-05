"""The embedded Store (connector protocol §3/§6/§7).

lookup / fetch / put / invalidate over one directory of L0 blobs.
Atomic puts (O_EXCL temp + rename), digest verification on every fetch,
byte-budget eviction over the DIRECTORY contents (not just what this
process indexed), and corrupt-on-open reclamation.

Threat model: the directory's contents are untrusted bytes (anything may
have written them). Integrity is enforced by digests and by name binding
(a v2 blob's filename must be the fold of its own header — a mislabeled
header cannot impersonate another entry); authenticity is NOT — anyone
with directory write access can plant self-consistent entries. Sharing a
store directory assumes same-user cooperating writers.

Multiple processes may share the directory: filenames are immutable per
publication and every read verifies its digest — the worst case is a
missed reuse, never wrong KV. `invalidate(prefix=[])` erases EVERYTHING
(all engines) — subtree semantics applied at the root.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import struct
import time
import uuid
from pathlib import Path

from mlxcache_sidecar import blob

from .errors import CorruptCache, Refused, Unavailable
from .fingerprint import DEFAULT_ENGINE, Fingerprint, blob_key
from .index import ANY_PREFIX, Entry, Granularity, PrefixIndex

CKPT_SUFFIX = ".ckpt"
DEFAULT_BUDGET = 32 * 1024 * 1024 * 1024  # 32 GiB (protocol §7 default)
DEFAULT_ANCHOR_WINDOW_S = 900.0  # protocol §7
_CHUNK = 1 << 20  # streaming digest chunk
_TMP_MAX_AGE_S = 24 * 3600  # crashed-writer tmp sweep threshold


class Store:
    def __init__(
        self,
        store_dir: str | os.PathLike,
        byte_budget: int = DEFAULT_BUDGET,
        anchor_window_s: float = DEFAULT_ANCHOR_WINDOW_S,
        trust_legacy: bool = False,
    ):
        """`trust_legacy=True` indexes on-disk v1 (digest-less) blobs for
        lookup; the default skips them — over an untrusted directory a
        planted v1 blob is unverifiable, so it is readable by explicit
        fetch but never served by lookup."""
        self.dir = Path(store_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.byte_budget = byte_budget
        self.index = PrefixIndex(anchor_window_s=anchor_window_s)
        self.trust_legacy = trust_legacy
        self._recovered_corrupt = 0
        self._skipped_unreadable = 0
        self._bytes = 0
        self._open()

    # -- open -------------------------------------------------------------

    def _open(self) -> None:
        """Whole-store open (protocol §6): sweep crashed-writer temps,
        index every entry with a streaming digest verify, reclaim the
        deterministically corrupt, dedupe same-key generations (newest
        wins), and skip — never fail on — files this process cannot read.
        A single unreadable or poison file must not brick the store."""
        self._sweep_tmps()
        indexed: list[tuple[str, Entry]] = []
        for path in sorted(self.dir.glob(f"*{CKPT_SUFFIX}"), reverse=True):
            # reverse = newest generation first: last writer wins (§3.3)
            if not path.is_file():
                continue  # a directory named *.ckpt is foreign junk, not ours
            try:
                meta, size = self._read_and_verify(path)
            except ValueError:
                # Deterministic corruption (foreign version, v2-without-
                # digest, malformed/mislabeled header, digest mismatch):
                # reclaim at open (§6) — the daemon-side quarantine.
                with contextlib.suppress(OSError):
                    os.unlink(path)
                self._recovered_corrupt += 1
                continue
            except OSError:
                # Unreadable now (permissions, transient): skip without
                # indexing. Missing reuse is the safe failure; bricking
                # every future open on one file is not.
                self._skipped_unreadable += 1
                continue
            if meta.payload_sha256 is None and not self.trust_legacy:
                continue  # unverifiable v1: kept on disk, not indexed
            entry = Entry(
                name=path.name,
                tokens=tuple(meta.tokens),
                fingerprint=meta.fingerprint,
                engine_id=meta.engine_id or DEFAULT_ENGINE,
                granularity=meta.granularity or ANY_PREFIX,
                last_used=time.monotonic(),  # seeded just-used (§7)
            )
            self.index.insert(entry)
            self._bytes += size
            indexed.append((meta, path.name, entry))
        # Same-key generations (a crash between publish and supersede, or
        # another process's file): keep the best, reclaim the rest (§3.3).
        # Rank is (format_version, name): a digest-verified v2 ALWAYS beats
        # an unverifiable v1 for the same identity — a v1 blob predates
        # every v2, and a lexically-newer v1 name must not delete the
        # genuine v2 file (wrong-KV inversion under trust_legacy).
        seen: dict[str, tuple[int, str]] = {}
        for meta, name, entry in indexed:  # newest-first order
            ident = f"{blob_key(entry.fingerprint, entry.engine_id, list(entry.tokens)):032x}"
            rank = (meta.format_version, name)
            if ident not in seen or rank > seen[ident]:
                seen[ident] = rank
        for _, name, entry in indexed:
            ident = f"{blob_key(entry.fingerprint, entry.engine_id, list(entry.tokens)):032x}"
            if seen[ident][1] != name:
                with contextlib.suppress(OSError):
                    self._bytes -= (self.dir / name).stat().st_size
                self.index.remove_exact(entry.tokens, name)
                with contextlib.suppress(OSError):
                    os.unlink(self.dir / name)

    def _read_and_verify(self, path: Path) -> tuple[blob.CheckpointMeta, int]:
        """Stream a blob: cap + parse the header, verify the payload
        digest without holding it in memory, and bind a v2 blob's filename
        to the fold of its own header (a mislabeled header cannot
        impersonate another entry's key). Raises ValueError on every
        deterministic defect, OSError on read failures."""
        with open(path, "rb") as fh:
            prefix = fh.read(4)
            if len(prefix) < 4:
                raise ValueError("truncated header length")
            (header_len,) = struct.unpack("<I", prefix)
            if header_len > blob.MAX_HEADER_BYTES:
                raise ValueError(f"header length {header_len} exceeds cap")
            header = fh.read(header_len)
            if len(header) < header_len:
                raise ValueError("header length exceeds blob")
            meta = blob.decode_header(header)
            digest = hashlib.sha256()
            while chunk := fh.read(_CHUNK):
                digest.update(chunk)
        if meta.payload_sha256 is not None:
            if digest.hexdigest() != meta.payload_sha256:
                raise ValueError("payload digest mismatch")
            try:
                expected = blob_key(meta.fingerprint, meta.engine_id, list(meta.tokens))
            except (AttributeError, TypeError) as exc:
                raise ValueError(f"unfingerprintable header: {exc}") from exc
            if not path.name.startswith(f"{expected:032x}-"):
                raise ValueError("filename does not bind to header (mislabeled)")
        size = path.stat().st_size
        return meta, size

    def _sweep_tmps(self) -> None:
        cutoff = time.time() - _TMP_MAX_AGE_S
        for path in self.dir.glob(f"*{CKPT_SUFFIX}.tmp*"):
            with contextlib.suppress(OSError):
                if path.is_file() and path.stat().st_mtime < cutoff:
                    os.unlink(path)

    # -- operations (protocol §3) ------------------------------------------

    def lookup(
        self, engine_id: str, fingerprint: Fingerprint, tokens: list[int]
    ) -> tuple[int, str | None]:
        matched, entry = self.index.lookup(engine_id, fingerprint, tokens)
        if entry is None:
            return 0, None
        return matched, entry.name

    def fetch(self, ref: str) -> bytes:
        if (
            ref != os.path.basename(ref)
            or ref in ("", ".", "..")
            or "/" in ref
            or "\\" in ref
            or "\x00" in ref
        ):
            raise ValueError(f"ref is not a bare store filename: {ref!r}")
        path = self.dir / ref
        try:
            raw = path.read_bytes()
        except FileNotFoundError as exc:
            raise Unavailable(f"ref vanished (evicted?): {ref}") from exc
        except OSError as exc:
            raise Unavailable(str(exc)) from exc
        try:
            meta, payload = blob.decode(raw)
        except ValueError as exc:
            raise CorruptCache(getattr(exc, "kind", "header"), f"{ref}: {exc}") from exc
        if meta.payload_sha256 is None:
            return payload  # v1 legacy: readable, never verified (§5)
        if hashlib.sha256(payload).hexdigest() != meta.payload_sha256:
            # Self-heal (daemon quarantine mirror): retire the entry so the
            # next lookup misses instead of re-serving the failure.
            entry = self.index.entry_by_name(ref)
            if entry is not None and self.index.remove_exact(entry.tokens, ref):
                with contextlib.suppress(OSError):
                    self._bytes -= (self.dir / ref).stat().st_size
                    os.unlink(self.dir / ref)
            raise CorruptCache("digest", ref)
        return payload

    def put(
        self,
        engine_id: str,
        fingerprint: Fingerprint,
        tokens: list[int],
        payload: bytes,
        granularity: Granularity = Granularity.ANY_PREFIX,
    ) -> str:
        if len(tokens) < 2:
            # §3.2 coverage: a prefix of length n covers exactly the first
            # n-1 positions — a sub-2-token prefix caches nothing.
            raise Refused("prefix-under-two", "a sub-2-token prefix covers zero positions (§3.2)")
        if int(granularity) not in (0, 1):
            raise ValueError(f"granularity must be 0 or 1, got {int(granularity)}")
        # Mirror decode_header's schema at the write boundary: a header we
        # would reclaim at the next open must never earn a CacheRef (and
        # non-u32 tokens would alias in the fold — one key, two entries).
        if not all(type(t) is int and 0 <= t <= 0xFFFFFFFF for t in tokens):
            raise Refused("invalid-tokens", "tokens must be a list of u32")
        if not isinstance(engine_id, str):
            raise ValueError(f"engine_id must be a str, got {type(engine_id).__name__}")
        meta = blob.CheckpointMeta(
            fingerprint=fingerprint,
            token_count=len(tokens),
            tokens=list(tokens),
            format_version=blob.FORMAT_VERSION_D1,
            payload_sha256=hashlib.sha256(payload).hexdigest(),
            engine_id=engine_id,
            granularity=int(granularity),
        )
        key = blob_key(fingerprint, engine_id, tokens)
        # §3.4 generation recipe (ms<<16 | pid) is "practically unique" —
        # the local exists-check closes the same-millisecond collision.
        generation = (int(time.time() * 1000) << 16) | (os.getpid() & 0xFFFF)
        name = f"{key:032x}-{generation:016x}{CKPT_SUFFIX}"
        bump = 0
        while (self.dir / name).exists():
            bump += 1
            name = f"{key:032x}-{generation + bump:016x}{CKPT_SUFFIX}"
        final = self.dir / name
        # Collision-proof tmp suffix: id(self) is recycled after GC, so
        # pid+id+thread-ident can repeat across instances and O_EXCL would
        # spuriously refuse the write. O_EXCL also refuses a pre-planted
        # symlink at the tmp path.
        tmp = self.dir / f"{name}.tmp{uuid.uuid4().hex}"
        data = blob.encode(meta, payload)
        try:
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            os.replace(tmp, final)  # atomic publication
        except OSError as exc:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise Unavailable(str(exc)) from exc
        self._bytes += len(data)
        # Supersede every EARLIER generation of the SAME key — including
        # files OTHER processes published after our open (§3.3 last writer
        # wins BY GENERATION): only lexically-older names are unlinked, so
        # a concurrent newer winner is never crossed out (two racing puts
        # whose supersede globs interleave must not zero the key).
        for stale in self.dir.glob(f"{key:032x}-*{CKPT_SUFFIX}"):
            if stale.name >= name or not stale.is_file():
                continue
            for e in self.index.entries_at(tuple(tokens)):
                if e.name == stale.name:
                    self.index.remove_exact(tuple(tokens), stale.name)
            with contextlib.suppress(OSError):
                self._bytes -= stale.stat().st_size
                os.unlink(stale)
        self.index.insert(
            Entry(
                name=name,
                tokens=tuple(tokens),
                fingerprint=fingerprint,
                engine_id=engine_id,
                granularity=int(granularity),
                last_used=time.monotonic(),
            )
        )
        self._enforce_budget()
        return name

    def invalidate(self, ref: str | None = None, prefix: list[int] | None = None) -> int:
        """Remove by CacheRef and/or by token subtree (protocol §3.4).
        Removal unlinks the file after the generation re-verify. NOTE:
        `prefix=[]` erases the ENTIRE store (subtree semantics at the
        root, all engines); this is subtree erasure plus equal-length
        end-anchored siblings — not full match-set erasure."""
        n = 0
        if ref is not None:
            entry = self.index.entry_by_name(ref)
            if entry is not None and self.index.remove_exact(entry.tokens, ref):
                n += 1
                with contextlib.suppress(OSError):
                    self._bytes -= (self.dir / ref).stat().st_size
                    os.unlink(self.dir / ref)
        if prefix is not None:
            for name in self.index.remove_subtree(tuple(prefix)):
                n += 1
                with contextlib.suppress(OSError):
                    self._bytes -= (self.dir / name).stat().st_size
                    os.unlink(self.dir / name)
        return n

    # -- budget (protocol §7) ------------------------------------------------

    def _dir_bytes(self) -> int:
        total = 0
        for path in self.dir.glob(f"*{CKPT_SUFFIX}"):
            if not path.is_file():
                continue
            try:
                total += path.stat().st_size
            except OSError:
                continue  # raced unlink: not present, not counted
        return total

    def _enforce_budget(self) -> None:
        if self.byte_budget <= 0:
            return
        if self._bytes <= self.byte_budget:
            return  # running total under budget: no per-put directory walk
        # Tracked total says over: rescan authoritatively (other processes
        # may have changed the directory), then evict.
        total = self._dir_bytes()
        self._bytes = total
        if total <= self.byte_budget:
            return
        now = time.monotonic()
        cands = self.index.candidates_for_eviction(now)
        # §7: non-anchors ordered by token_count DESCENDING; recency and
        # extension confer anchor STATUS only, never ordering.
        evictable = [e for e, anchored in cands if not anchored]
        evictable.sort(key=lambda e: len(e.tokens), reverse=True)
        for entry in evictable:
            if total <= self.byte_budget:
                break
            if not self.index.remove_exact(entry.tokens, entry.name):
                continue  # raced: another process already removed it
            try:
                size = (self.dir / entry.name).stat().st_size
                os.unlink(self.dir / entry.name)
                total -= size
            except OSError:
                continue  # raced: another process reclaimed it first
        self._bytes = total
        # Still over budget (held by anchors / unindexed foreign files):
        # hold and continue — the protocol's over-budget-by-anchors rule.
