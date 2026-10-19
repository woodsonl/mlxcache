#!/usr/bin/env python3
"""E5 — eviction simulation: whole-blob store vs refcounted chunk store.

The representation changes the policy problem. Today the reaper evicts
whole files ("coldest non-anchor, biggest first" = token_count DESC); under
a chunk store evicting a blob manifest frees only the chunks whose refcount
hits zero — for a chain, its unique TAIL, not its size. The sim mirrors the
real semantics (store.py _enforce_budget + index.py structural anchors:
an entry is anchored while a live published entry extends it) over the
captured chain and reports, per scenario and byte budget:

  S1 chain growth   — bytes trajectory: full-blob per turn vs unique chunks.
  S2 two conversations, tight budget, interleaved — hit rate + evictions.
     Conversation B shares turn 1 exactly (same system+doc+Q1 — the prefix
     index already dedups that logically in BOTH stores) and then diverges
     with per-turn deltas sized from A's measured deltas (synthetic-derived,
     labeled as such in the report).
  S3 evict-then-re-request — cascade-evict the chain, replay it: republish
     cost is full bytes (whole) vs the chunks whose refcount hit zero
     (chunk; measured ~1.3-6% of whole-blob republish at the x1 budget).

Budgets are expressed as a fraction of ONE whole final blob, so a budget can
land below the chunk store's steady-state size. That distinction is the whole
point of the gate: a budget larger than the chunk store's total bytes cannot
evict anything, so it tests nothing. Every S2 row records the absolute budget
and whether the chunk store actually reached it (``binding``).

Invariants (also asserted by test_evict_sim.py on a synthetic chain):
budget held after every sweep unless only anchors remain; chunks with
refcount > 0 never freed; structurally anchored entries never evicted;
chunked hit rate >= whole-blob hit rate at a binding budget.

Usage (defaults reproduce the committed tables):
  uv run python scripts/dedup/evict_sim.py <grow_run_dir> \
      [--avg-kib 16] [--budgets 0.5,0.9,1,1.25,2] [--json PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from blob_forensics import load_blob, parse_safetensors, turn_blobs
from chunk_sim import blob_chunk_sets


@dataclass
class Entry:
    convo: str
    turn: int
    tokens: int  # token_count (positions + 1)
    size: int  # full blob bytes
    chunks: dict[str, int]  # digest -> bytes
    parent: Entry | None = None
    live: bool = True


def anchored_ancestors(entries: list[Entry]) -> set[int]:
    """Structural anchoring (index.py): an entry is anchored while a LIVE
    published entry extends it. Mark every ancestor of a live entry."""
    marked: set[int] = set()
    for e in entries:
        if not e.live:
            continue
        p = e.parent
        while p is not None and id(p) not in marked:
            marked.add(id(p))
            p = p.parent
    return marked


class WholeStore:
    """Today's store: bytes = sum of live blob file sizes."""

    def __init__(self) -> None:
        self.entries: list[Entry] = []

    def bytes(self) -> int:
        return sum(e.size for e in self.entries if e.live)

    def publish(self, e: Entry) -> None:
        e.live = True
        self.entries.append(e)

    def enforce(self, budget: int) -> dict:
        anchored = anchored_ancestors(self.entries)
        cands = [e for e in self.entries if e.live and id(e) not in anchored]
        cands.sort(key=lambda e: e.tokens, reverse=True)  # store.py: biggest first
        total = self.bytes()
        evicted = 0
        for e in cands:
            if total <= budget:
                break
            e.live = False
            total -= e.size
            evicted += 1
        return {
            "bytes": total,
            "evicted": evicted,
            "over_budget_held_by_anchors": total > budget,
        }


class ChunkStore:
    """Chunk-refcount store: bytes = unique stored chunk bytes."""

    def __init__(self) -> None:
        self.entries: list[Entry] = []
        self.ref: dict[str, int] = {}
        self.chunks_on_disk: dict[str, int] = {}

    def bytes(self) -> int:
        return sum(self.chunks_on_disk.values())

    def publish(self, e: Entry) -> None:
        e.live = True
        self.entries.append(e)
        for d, sz in e.chunks.items():
            self.ref[d] = self.ref.get(d, 0) + 1
            self.chunks_on_disk[d] = sz

    def _unique_bytes(self, e: Entry) -> int:
        return sum(sz for d, sz in e.chunks.items() if self.ref.get(d, 0) == 1)

    def enforce(self, budget: int) -> dict:
        # chunk-store policy: evict the biggest unique-tail first (freeing a
        # shared chunk costs the store nothing, so unique bytes are the only
        # bytes an eviction can actually reclaim)
        anchored = anchored_ancestors(self.entries)
        cands = [e for e in self.entries if e.live and id(e) not in anchored]
        cands.sort(key=self._unique_bytes, reverse=True)
        total = self.bytes()
        evicted = 0
        for e in cands:
            if total <= budget:
                break
            e.live = False
            evicted += 1
            for d in e.chunks:
                self.ref[d] -= 1
                if self.ref[d] == 0:
                    # invariant: a chunk leaves disk only at refcount 0
                    del self.chunks_on_disk[d]
                assert self.ref[d] >= 0
            total = self.bytes()
        return {
            "bytes": total,
            "evicted": evicted,
            "over_budget_held_by_anchors": total > budget,
        }


@dataclass
class SimResult:
    name: str
    store_kind: str
    hit_rate: float
    reused_positions: int
    requested_positions: int
    final_bytes: int
    evictions: int
    over_budget: bool
    trajectory: list[int] = field(default_factory=list)
    # absolute budget for the row; None for the unbudgeted S1 scenario
    budget_bytes: int | None = None
    # did this store ever reach its budget during the replay? A budget the
    # store never approaches cannot evict, so its hit rate is not evidence
    # about policy under pressure.
    binding: bool | None = None
    # bytes the re-request after eviction had to re-store (S3 only)
    restore_bytes: int | None = None


def replay(
    store: WholeStore | ChunkStore,
    requests: list[tuple[str, int]],
    budget: int,
    entries_by: dict[tuple[str, int], Entry],
    track_restore: bool = False,
) -> SimResult:
    """requests: (convo, turn) in arrival order. Lookup is TOKEN-PREFIX
    semantics like the daemon's index (never label-scoped): the request's own
    entry, if already live, is an exact hit; otherwise the nearest live
    ancestor on the parent chain (the blob whose tokens are the longest
    prefix of the request) serves a partial hit of tokens-1 positions.
    Misses publish the entry, then the budget is enforced (store.py mirrors
    a put -> _enforce_budget). Entry liveness is reset per replay: entries
    are shared objects across scenarios and stores.

    With ``track_restore``, the result also reports bytes published after the
    first eviction: the republish cost once the store is under pressure, which
    is what a chunk store is supposed to cut (its live tail survives, a whole
    blob's does not)."""
    kind = "chunk" if isinstance(store, ChunkStore) else "whole"
    for e in entries_by.values():
        e.live = False
    store.entries.clear()
    if isinstance(store, ChunkStore):
        store.ref.clear()
        store.chunks_on_disk.clear()
    reused = requested = 0
    evictions = 0
    over = False
    traj: list[int] = []
    peak = 0  # largest pre-sweep live bytes: did the store ever reach budget?
    restore = 0
    for convo, turn in requests:
        own = entries_by[(convo, turn)]
        requested += own.tokens - 1
        if own.live:
            reused += own.tokens - 1  # exact hit: blob already published
        else:
            p = own.parent
            best: Entry | None = None
            while p is not None:
                if p.live:
                    best = p
                    break
                p = p.parent
            reused += (best.tokens - 1) if best else 0
            before = store.bytes()
            store.publish(own)
            peak = max(peak, store.bytes())
            if track_restore and evictions:
                restore += store.bytes() - before
            r = store.enforce(budget)
            evictions += r["evicted"]
            over = over or r["over_budget_held_by_anchors"]
        traj.append(store.bytes())
    bounded = budget < (1 << 62)
    return SimResult(
        name="",
        store_kind=kind,
        hit_rate=round(reused / requested, 4) if requested else 1.0,
        reused_positions=reused,
        requested_positions=requested,
        final_bytes=store.bytes(),
        evictions=evictions,
        over_budget=over,
        trajectory=traj,
        budget_bytes=budget if bounded else None,
        binding=(peak > budget) if bounded else None,
        restore_bytes=restore if track_restore else None,
    )


def build_entries(turns: list[tuple[int, Path]], avg: int, scope: str = "head") -> dict:
    out: dict[tuple[str, int], Entry] = {}
    parent = None
    for turn, path in turns:
        meta, payload = load_blob(path)
        tensors, _ = parse_safetensors(payload)
        chunks, _ = blob_chunk_sets(payload, tensors, scope, avg)
        e = Entry(
            convo="A",
            turn=turn,
            tokens=meta["token_count"],
            size=path.stat().st_size,
            chunks={d: s for _, s, d in chunks},
            parent=parent,
        )
        out[("A", turn)] = e
        parent = e
    return out


def synthetic_convo_b(entries: dict[tuple[str, int], Entry]) -> dict[tuple[str, int], Entry]:
    """Conversation B: shares turn 1 EXACTLY (same system+doc+Q1 — in the
    real index that is one blob serving both conversations; B's turn 1 is an
    exact hit, not a publish), then diverges with per-turn deltas sized from
    A's measured growth (synthetic-derived, labeled as such)."""
    out: dict[tuple[str, int], Entry] = {}
    a1 = entries[("A", 1)]
    out[("B", 1)] = a1  # alias: the SAME blob serves both conversations
    parent = a1
    for k in sorted({t for c, t in entries if c == "A"}):
        if k == 1:
            continue
        ak = entries[("A", k)]
        delta_bytes = max(ak.size - entries[("A", k - 1)].size, 0)
        own = {f"b{k}-{i}": sz for i, sz in enumerate([delta_bytes] if delta_bytes else [])}
        e = Entry(
            "B",
            k,
            ak.tokens,
            ak.size,
            {**a1.chunks, **own},
            parent=parent,
        )
        out[("B", k)] = e
        parent = e
    return out


def run_scenarios(
    entries_a: dict[tuple[str, int], Entry],
    budgets: list[float],
    s3_mult: float = 1.0,
) -> list[SimResult]:
    all_entries = dict(entries_a)
    all_entries.update(synthetic_convo_b(entries_a))
    a_turns = sorted({t for c, t in entries_a if c == "A"})
    chain = [("A", t) for t in a_turns]
    results: list[SimResult] = []

    # S1: chain growth, no budget.
    for store in (WholeStore(), ChunkStore()):
        r = replay(store, chain, budget=1 << 62, entries_by=all_entries)
        r.name = "S1 chain growth (no budget)"
        results.append(r)

    # S2: interleaved conversations under scaled budgets.
    interleaved = [(c, t) for t in a_turns for c in ("A", "B")]
    final_size = entries_a[("A", max(a_turns))].size
    for mult in budgets:
        budget = int(mult * final_size)
        rw = replay(WholeStore(), interleaved, budget, all_entries)
        rw.name = f"S2 interleaved x{mult:g} budget"
        rc = replay(ChunkStore(), interleaved, budget, all_entries)
        rc.name = rw.name
        results.extend([rw, rc])

    # S3: evict the chain under a binding budget, then re-request it. Both
    # passes run in one replay so the second one starts from the evicted store.
    # The budget must leave the store able to serve anything at all: at x0.5
    # both representations thrash to zero bytes and republish cost is
    # meaningless.
    tight = int(s3_mult * final_size)
    for store in (WholeStore(), ChunkStore()):
        r = replay(store, chain + chain, tight, all_entries, track_restore=True)
        r.name = f"S3 evict-then-re-request x{s3_mult:g} budget"
        results.append(r)
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("grow_dir", type=Path)
    # defaults reproduce the committed tables/evict_sim.json rows
    ap.add_argument("--avg-kib", type=int, default=16)
    ap.add_argument(
        "--budgets",
        default="0.5,0.9,1,1.25,2",
        help="budgets as a fraction of one whole final blob; values below 1 "
        "are the only ones that can bind on a deduped store, and 0.5 is the "
        "degenerate floor where both stores thrash to zero",
    )
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    turns = turn_blobs(args.grow_dir)
    entries = build_entries(turns, args.avg_kib * 1024)
    results = run_scenarios(entries, [float(x) for x in args.budgets.split(",")])

    print(
        f"{'scenario':<34} {'store':<6} {'hit':>7} {'final MiB':>10} "
        f"{'budget MiB':>10} {'bind':>5} {'evict':>6} over"
    )
    for r in results:
        mib = r.final_bytes / (1 << 20)
        bud = "-" if r.budget_bytes is None else f"{r.budget_bytes / (1 << 20):.1f}"
        bind = "-" if r.binding is None else ("yes" if r.binding else "no")
        print(
            f"{r.name:<34} {r.store_kind:<6} {r.hit_rate:>7.4f} {mib:>10.1f} "
            f"{bud:>10} {bind:>5} {r.evictions:>6} {'yes' if r.over_budget else '-'}"
        )
    s1w, s1c = results[0], results[1]
    print(
        f"\nS1 growth: whole {s1w.final_bytes / (1 << 20):.1f} MiB vs chunk "
        f"{s1c.final_bytes / (1 << 20):.1f} MiB "
        f"({s1w.final_bytes / max(s1c.final_bytes, 1):.1f}x)"
    )
    for r in results[2:]:
        if r.store_kind == "chunk":
            w = next(x for x in results if x.name == r.name and x.store_kind == "whole")
            assert r.hit_rate >= w.hit_rate - 1e-9, (
                f"{r.name}: chunked hit rate {r.hit_rate} < whole {w.hit_rate}"
            )
    for r in results[2:]:
        if r.store_kind == "chunk" and r.name.startswith("S2") and not r.binding:
            print(
                f"WARNING {r.name}: budget {r.budget_bytes} never reached; "
                "chunk hit rate is not evidence under pressure"
            )
    for r in results:
        if r.restore_bytes is not None:
            print(
                f"{r.name:<34} {r.store_kind:<6} republish after eviction: "
                f"{r.restore_bytes / (1 << 20):.1f} MiB"
            )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps([r.__dict__ for r in results], indent=1))
        print(f"json -> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
