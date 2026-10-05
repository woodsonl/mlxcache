"""Granularity-aware prefix index (connector protocol §3.3).

A token trie whose nodes hold MULTIPLE entries (keyed by CacheRef/name) —
the same token path can carry entries from different engines/fingerprints
("two models can share token ids"), and every entry stays reachable for
lookup, eviction, and invalidate-by-ref.

Semantics (ported from the daemon's radix index + protocol §3):

- longest-prefix match over entries that are engine- and fingerprint-
  identical, honoring per-entry granularity (ANY_PREFIX: request extends
  the entry's tokens — guaranteed by path construction, never re-checked;
  WHOLE_CONTEXT: exact full-stream equality only, with fallback to a
  shorter ANY_PREFIX entry below);
- the end-anchored rule: an entry agreeing for k tokens, diverging at
  token k, and terminating still matches — its payload covers exactly
  the k shared positions. The scan runs at EVERY depth (including where
  the on-path walk stops) over TERMINAL edges only (edges to children
  holding entries — an entry at depth d has exactly d tokens by
  construction), and a LONGER end-anchored match beats a shorter on-path
  one (on-path wins ties), mirroring the Rust index's "deepest published
  child beats any walk best";
- `matched_len` is always len(entry.tokens), never capped (§3.2).

Anchors (§7): an entry is structurally anchored while a LIVE published
entry extends it. Liveness is computed bottom-up during the eviction
walk itself (the walk visits every node anyway) — not cached as a sticky
insert-time flag, which would permanently anchor the ancestors of every
ever-inserted chain and degrade a budgeted store to write-through.

Thread safety: an RLock around mutations and lookups.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import IntEnum

from .fingerprint import Fingerprint

ANY_PREFIX = 0
WHOLE_CONTEXT = 1


class Granularity(IntEnum):
    """Resume granularity declared at put (protocol §2/§3.3)."""

    ANY_PREFIX = ANY_PREFIX
    WHOLE_CONTEXT = WHOLE_CONTEXT


@dataclass
class Entry:
    name: str  # store-relative filename (the CacheRef)
    tokens: tuple[int, ...]
    fingerprint: Fingerprint
    engine_id: str
    granularity: int
    last_used: float  # monotonic seconds, per process (protocol §7)


class _Node:
    __slots__ = ("children", "entries", "terminal_edges")

    def __init__(self):
        self.children: dict[int, _Node] = {}
        self.entries: dict[str, Entry] = {}
        # Edges to children that hold entries — the ONLY siblings the
        # end-anchored scan ever needs to consider (kept exact on insert
        # and removal; a stale edge would only widen the EA scan).
        self.terminal_edges: set[int] = set()


class PrefixIndex:
    def __init__(self, anchor_window_s: float = 900.0):
        self._root = _Node()
        self._lock = threading.RLock()
        self._anchor_window = anchor_window_s
        self._by_name: dict[str, Entry] = {}

    def insert(self, entry: Entry) -> None:
        with self._lock:
            node = self._root
            for t in entry.tokens:
                nxt = node.children.get(t)
                if nxt is None:
                    nxt = _Node()
                    node.children[t] = nxt
                node = nxt
            node.entries[entry.name] = entry
            self._by_name[entry.name] = entry
            if entry.tokens:
                # The last hop leads to a node with an entry: terminal edge.
                parent = self._root
                for t in entry.tokens[:-1]:
                    parent = parent.children[t]
                parent.terminal_edges.add(entry.tokens[-1])

    def remove_exact(self, tokens: tuple[int, ...], name: str) -> bool:
        """Remove the entry at exactly `tokens` with CacheRef `name` (the
        generation re-verify, protocol §3.4)."""
        with self._lock:
            node = self._root
            parent: _Node | None = None
            for t in tokens:
                parent, node = node, node.children.get(t)
                if node is None:
                    return False
            removed = node.entries.pop(name, None) is not None
            if removed:
                self._by_name.pop(name, None)
                if not node.entries and parent is not None and tokens:
                    parent.terminal_edges.discard(tokens[-1])
            return removed

    def remove_subtree(self, prefix: tuple[int, ...]) -> list[str]:
        """Remove the exact entry at `prefix` (if any) AND every entry
        whose tokens extend it — including END-ANCHORED siblings: entries
        of length len(prefix) that agree for len-1 tokens and diverge at
        the final token. The sibling cleanup runs even when the main walk
        misses (invalidate of a diverged prefix must still purge what
        lookup would serve from it). Returns removed names.

        Scope: this is subtree erasure at `prefix` (+ equal-length
        divergent siblings), NOT match-set erasure — an invalidate(prefix)
        does not remove shorter end-anchored entries that a longer request
        through `prefix` could still match."""
        with self._lock:
            removed: list[str] = []
            # End-anchored siblings first (independent of walk success):
            # at the parent of `prefix`, terminal edges OTHER than
            # prefix[-1] whose child holds entries are exactly the
            # equal-length divergent siblings.
            if len(prefix) >= 1:
                parent = self._root
                for t in prefix[:-1]:
                    parent = parent.children.get(t)
                    if parent is None:
                        break
                else:
                    for edge in list(parent.terminal_edges):
                        if edge == prefix[-1]:
                            continue
                        sib = parent.children.get(edge)
                        if sib is None:
                            continue
                        for name in list(sib.entries):
                            removed.append(sib.entries.pop(name).name)
                            self._by_name.pop(name, None)
                        if not sib.entries:
                            parent.terminal_edges.discard(edge)
            # Exact entry + every extension below `prefix`.
            node = self._root
            for t in prefix:
                node = node.children.get(t)
                if node is None:
                    return removed
            removed.extend(node.entries.keys())
            for name in node.entries:
                self._by_name.pop(name, None)
            node.entries.clear()
            stack = [node]
            while stack:
                cur = stack.pop()
                for child in cur.children.values():
                    removed.extend(child.entries.keys())
                    for name in child.entries:
                        self._by_name.pop(name, None)
                    child.entries.clear()
                    stack.append(child)
            return removed

    def lookup(
        self,
        engine_id: str,
        fingerprint: Fingerprint,
        tokens: list[int],
    ) -> tuple[int, Entry | None]:
        """(matched_len, entry) — the longest conforming match, or (0, None).
        The winner's last_used is stamped under the same lock (protocol §7)."""
        with self._lock:
            best: Entry | None = None  # on-path (request extends entry)
            node = self._root
            for depth in range(len(tokens)):
                child = node.children.get(tokens[depth])
                if child is None:
                    break
                node = child
                for e in node.entries.values():
                    if self._identity_ok(e, engine_id, fingerprint, tokens) and (
                        best is None or len(e.tokens) > len(best.tokens)
                    ):
                        best = e
            if best is not None and len(best.tokens) == len(tokens):
                best.last_used = time.monotonic()
                return len(best.tokens), best  # cannot be beaten

            # End-anchored scan: terminal sibling edges at EVERY depth
            # (including the walk's stopping depth). A child on a terminal
            # edge holds entries of exactly depth k+1 tokens — the
            # terminate-right-after-divergence shape.
            node = self._root
            best_ea: Entry | None = None
            k = 0
            while k < len(tokens):
                for edge in node.terminal_edges:
                    if edge == tokens[k]:
                        continue  # the on-path edge; siblings only
                    for e in node.children[edge].entries.values():
                        if self._identity_ok(e, engine_id, fingerprint, tokens) and (
                            best_ea is None or len(e.tokens) > len(best_ea.tokens)
                        ):
                            best_ea = e
                nxt = node.children.get(tokens[k])
                if nxt is None:
                    break
                node = nxt
                k += 1
            # A longer end-anchored match beats a shorter on-path one;
            # on-path wins ties (the walk is the canonical path).
            winner = (
                best_ea
                if (
                    best_ea is not None and (best is None or len(best_ea.tokens) > len(best.tokens))
                )
                else best
            )
            if winner is None:
                return 0, None
            winner.last_used = time.monotonic()
            return len(winner.tokens), winner

    def _identity_ok(
        self,
        e: Entry,
        engine_id: str,
        fingerprint: Fingerprint,
        tokens: list[int],
    ) -> bool:
        if e.engine_id != engine_id or e.fingerprint != fingerprint:
            return False  # cross-engine/fingerprint: structurally no match
        if e.granularity == Granularity.WHOLE_CONTEXT:
            return len(e.tokens) == len(tokens) and e.tokens == tuple(tokens)
        return True  # ANY_PREFIX: the walk already guarantees the request
        # extends the entry's tokens (on-path construction / EA depth k+1
        # with k < len(tokens)); no extension check is needed here.

    def entries_at(self, tokens: tuple[int, ...]) -> list[Entry]:
        """Every entry published at EXACTLY `tokens` (all engines) — the
        supersede scan's targeted accessor, O(depth)."""
        with self._lock:
            node = self._root
            for t in tokens:
                node = node.children.get(t)
                if node is None:
                    return []
            return list(node.entries.values())

    def entry_by_name(self, name: str) -> Entry | None:
        return self._by_name.get(name)

    def all_entries(self) -> list[Entry]:
        with self._lock:
            return list(self._by_name.values())

    def candidates_for_eviction(self, now: float):
        """(entry, is_anchor) for every indexed entry — the store applies
        its own ordering policy (protocol §7). Structural anchoring is
        computed LIVE: an entry is anchored iff its node has a live
        published DESCENDANT (strictly below — same-node multi-engine
        entries do not extend each other). Iterative post-order so token
        chains deeper than the recursion limit cannot crash the walk."""
        with self._lock:
            out = []
            # node -> True iff this subtree (incl. the node) holds entries.
            subtree_has: dict[_Node, bool] = {}
            stack: list[tuple[_Node, bool]] = [(self._root, False)]
            while stack:
                node, entered = stack.pop()
                if not entered:
                    stack.append((node, True))
                    for child in node.children.values():
                        stack.append((child, False))
                    continue
                live_below = any(subtree_has[c] for c in node.children.values())
                for e in node.entries.values():
                    out.append((e, live_below or (now - e.last_used) <= self._anchor_window))
                subtree_has[node] = live_below or bool(node.entries)
            return out
