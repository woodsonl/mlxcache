#!/usr/bin/env python3
"""E5 — eviction simulation: whole-blob store vs refcounted chunk store.

The representation changes the policy problem. Today the reaper evicts whole
files; under a chunk store, evicting a blob manifest frees only the chunks
whose refcount hits zero — for a chain, its unique TAIL, not its size.

What the sim mirrors, and what it proposes:

  MIRRORED (whole-blob arm) — the sidecar's store.py ``_enforce_budget``:
    candidates are non-anchors ordered by token_count DESCENDING (§7), the
    budget is enforced after each publish, and a store that is still over
    budget with only anchored entries HOLDS (over-budget-by-anchors), it does
    not break anchors. Anchor status mirrors index.py: an entry is anchored
    while a live published entry extends it (structural) OR it was served
    within the anchor window (recency; ``MLXCACHE_EVICT_ANCHOR_WINDOW_S``,
    default 900 s, same shape in the daemon's orchestrator.rs ``evict_cold``).

  PROPOSED (chunk arm) — there is no real chunk store yet; its ordering
    (biggest unique-tail first) is the policy this gate exists to test, not
    an implementation claim.

The anchor window is modeled in request ticks, not seconds, so both regimes
that matter are runnable without wall-clock sleeps:

  --anchor-window-ticks 0  "aged": every request arrives after the previous
                            entries' windows expired. Steady-state reaper.
  fast arm (window >= N)   every request of the replay lands inside one
                            anchor window: recently touched entries cannot
                            be evicted and a binding budget is held by
                            anchors in BOTH stores — the daemon's documented
                            hold-and-continue rule, not thrash.

Scenarios (budgets are fractions of ONE whole final blob, so a budget can
land below the chunk store's steady-state size — that distinction is the
point of the gate; a budget larger than the store cannot evict anything and
tests nothing):

  S1 chain growth, no budget — bytes trajectory: full-blob per turn vs
     unique chunks.
  S2 two conversations, interleaved, scaled budgets — hit rate + evictions,
     run in BOTH regimes. Conversation B shares turn 1 exactly (same
     system+doc+Q1 — the prefix index dedups that logically in BOTH stores)
     and then diverges with per-turn deltas sized from A's measured deltas
     (synthetic-derived, labeled as such); B's chunks are cumulative, as a
     real chain's are.
  S3 evict-then-re-request — run the chain under a binding budget, then
     re-request it (aged, so the anchors from pass one have expired):
     republish cost is full bytes (whole) vs the chunks whose refcount hit
     zero (chunk; measured 0.8-5.5% of whole-blob republish at the x1
     budget across codecs and tiers).

Every row records the absolute budget, whether the store ever reached it
(``binding``), and the anchor-window regime it ran under. Invariants are
asserted by tests/test_evict_sim.py on synthetic chains.

Usage (defaults reproduce the committed tables):
  uv run python scripts/dedup/evict_sim.py <grow_run_dir> \
      [--avg-kib 16] [--codec cdc|token-block] [--budgets 0.5,0.9,1,1.25,2] \
      [--json PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from blob_forensics import load_blob, parse_safetensors, turn_blobs
from chunk_sim import TOKEN_BLOCK, blob_chunk_sets, token_block_chunks

# "no budget": larger than any reachable store size; rows under it record
# budget_bytes=None / binding=None so an unbudgeted row can never be misread
# as policy evidence.
UNBUDGETED = 1 << 62


@dataclass
class Entry:
    convo: str
    turn: int
    tokens: int  # token_count (positions + 1)
    size: int  # payload bytes (both stores count payload, not file framing)
    chunks: dict[str, int]  # digest -> bytes
    parent: Entry | None = None
    live: bool = True
    last_used: int = 0  # tick of the last serve/publish (index.py §7 stamps the winner)


def anchored_ids(entries: list[Entry], tick: int, window: int) -> set[int]:
    """Anchor status as index.py computes it: structural (a live published
    entry extends this one) OR served within the anchor window."""
    marked: set[int] = set()
    for e in entries:
        if not e.live:
            continue
        p = e.parent
        while p is not None and id(p) not in marked:
            marked.add(id(p))
            p = p.parent
    for e in entries:
        if e.live and tick - e.last_used <= window:
            marked.add(id(e))
    return marked


class WholeStore:
    """Today's store: bytes = sum of live blob payload sizes."""

    def __init__(self) -> None:
        self.entries: list[Entry] = []
        self._seen: set[int] = set()

    def bytes(self) -> int:
        return sum(e.size for e in self.entries if e.live)

    def publish(self, e: Entry, tick: int) -> None:
        e.live = True
        e.last_used = tick
        # an evicted entry can be re-published within the same replay
        # (S3's re-request): it must occupy the ledger once, or bytes() and
        # the sweep double-count it
        if id(e) not in self._seen:
            self._seen.add(id(e))
            self.entries.append(e)

    def enforce(self, budget: int, tick: int, window: int) -> dict:
        anchored = anchored_ids(self.entries, tick, window)
        cands = [e for e in self.entries if e.live and id(e) not in anchored]
        cands.sort(key=lambda e: e.tokens, reverse=True)  # store.py §7: token_count DESC
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
        self._seen: set[int] = set()

    def bytes(self) -> int:
        return sum(self.chunks_on_disk.values())

    def publish(self, e: Entry, tick: int) -> None:
        e.live = True
        e.last_used = tick
        # ledger once per entry (a re-published evicted entry must not be
        # swept twice), but refcounts on EVERY publish: each live holder
        # counts, and a re-published entry's chunks must return to disk
        if id(e) not in self._seen:
            self._seen.add(id(e))
            self.entries.append(e)
        for d, sz in e.chunks.items():
            self.ref[d] = self.ref.get(d, 0) + 1
            self.chunks_on_disk[d] = sz

    def _unique_bytes(self, e: Entry) -> int:
        return sum(sz for d, sz in e.chunks.items() if self.ref.get(d, 0) == 1)

    def enforce(self, budget: int, tick: int, window: int) -> dict:
        # proposed chunk-store policy: evict the biggest unique tail first
        # (freeing a shared chunk reclaims nothing, so unique bytes are the
        # only bytes an eviction can reclaim)
        anchored = anchored_ids(self.entries, tick, window)
        cands = [e for e in self.entries if e.live and id(e) not in anchored]
        cands.sort(key=self._unique_bytes, reverse=True)
        total = self.bytes()
        evicted = 0
        for e in cands:
            if total <= budget:
                break
            e.live = False
            evicted += 1
            freed = 0
            for d in e.chunks:
                self.ref[d] -= 1
                assert self.ref[d] >= 0
                if self.ref[d] == 0:
                    # invariant: a chunk leaves disk only at refcount 0
                    freed += self.chunks_on_disk.pop(d)
            total -= freed
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
    # anchor-window regime in request ticks (0 = aged; >= len(requests) = fast)
    window_ticks: int | None = None
    # bytes the re-request after eviction had to re-store (S3 only)
    restore_bytes: int | None = None


def replay(
    store: WholeStore | ChunkStore,
    agenda: list[tuple[str, int]],
    budget: int,
    entries_by: dict[tuple[str, int], Entry],
    window: int = 0,
    track_restore: bool = False,
) -> SimResult:
    """requests: (convo, turn) in arrival order, plus the special agenda
    entry ("reap", n) which advances the clock by n background-reaper sweeps
    with no request (the daemon's interval sweeper running after a
    conversation goes idle: anchors expire, leaves become evictable, and
    repeated sweeps cascade leaf -> parent). Tick = agenda index.
    Lookup is TOKEN-PREFIX semantics like the daemon's index (never
    label-scoped): the request's own entry, if already live, is an exact hit;
    otherwise the nearest live ancestor on the parent chain serves a partial
    hit of tokens-1 positions, and the SERVED entry's last_used is stamped
    (index.py §7 stamps the winner). Misses publish the entry, then the
    budget is enforced (put -> _enforce_budget). Liveness and store state are
    reset per replay: entries are shared objects across scenarios and stores.

    With ``track_restore``, the result also reports bytes published after the
    first eviction: the republish cost once the store is under pressure."""
    kind = "chunk" if isinstance(store, ChunkStore) else "whole"
    for e in entries_by.values():
        e.live = False
        e.last_used = 0
    store.entries.clear()
    store._seen.clear()
    if isinstance(store, ChunkStore):
        store.ref.clear()
        store.chunks_on_disk.clear()
    reused = requested = 0
    evictions = 0
    over = False
    traj: list[int] = []
    peak = 0  # largest pre-sweep live bytes: did the store ever reach budget?
    restore = 0
    for tick, req in enumerate(agenda):
        if req[0] == "reap":
            for _ in range(req[1]):
                r = store.enforce(budget, tick=tick, window=window)
                evictions += r["evicted"]
                over = over or r["over_budget_held_by_anchors"]
            traj.append(store.bytes())
            continue
        convo, turn = req
        own = entries_by[(convo, turn)]
        requested += own.tokens - 1
        if own.live:
            reused += own.tokens - 1  # exact hit: blob already published
            own.last_used = tick
        else:
            p = own.parent
            best: Entry | None = None
            while p is not None:
                if p.live:
                    best = p
                    break
                p = p.parent
            reused += (best.tokens - 1) if best else 0
            if best is not None:
                best.last_used = tick  # the ancestor served the request
            before = store.bytes()
            store.publish(own, tick)
            peak = max(peak, store.bytes())
            if track_restore and evictions:
                restore += store.bytes() - before
            r = store.enforce(budget, tick=tick, window=window)
            evictions += r["evicted"]
            over = over or r["over_budget_held_by_anchors"]
        traj.append(store.bytes())
    bounded = budget < UNBUDGETED
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
        window_ticks=window if bounded else None,
        restore_bytes=restore if track_restore else None,
    )


def build_entries(turns: list[tuple[int, Path]], avg: int, codec: str = "cdc") -> dict:
    out: dict[tuple[str, int], Entry] = {}
    parent = None
    for turn, path in turns:
        meta, payload = load_blob(path)
        tensors, _ = parse_safetensors(payload)
        if codec == "token-block":
            chunks = token_block_chunks(payload, tensors, meta["token_count"] - 1, TOKEN_BLOCK)
        else:
            chunks = blob_chunk_sets(payload, tensors, "head", avg)
        e = Entry(
            convo="A",
            turn=turn,
            tokens=meta["token_count"],
            size=len(payload),
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
    A's measured growth (synthetic-derived, labeled as such). B's chunk set
    is CUMULATIVE — B-k contains B-(k-1)'s chunks plus its own delta, the
    way a real growing chain's blobs nest."""
    out: dict[tuple[str, int], Entry] = {}
    a1 = entries[("A", 1)]
    out[("B", 1)] = a1  # alias: the SAME blob serves both conversations
    parent = a1
    cumulative = dict(a1.chunks)
    for k in sorted({t for c, t in entries if c == "A"}):
        if k == 1:
            continue
        ak = entries[("A", k)]
        prev = entries[("A", k - 1)]
        delta_bytes = max(ak.size - prev.size, 0)
        if delta_bytes:
            cumulative[f"b{k}"] = delta_bytes
        e = Entry(
            "B",
            k,
            ak.tokens,
            ak.size,
            dict(cumulative),
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

    def run_pair(name: str, reqs: list[tuple[str, int]], budget: int, window: int):
        rw = replay(WholeStore(), reqs, budget, all_entries, window=window)
        rw.name = name
        rc = replay(ChunkStore(), reqs, budget, all_entries, window=window)
        rc.name = name
        results.extend([rw, rc])

    # S1: chain growth, no budget.
    for store in (WholeStore(), ChunkStore()):
        r = replay(store, chain, budget=UNBUDGETED, entries_by=all_entries)
        r.name = "S1 chain growth (no budget)"
        results.append(r)

    # S2: interleaved conversations under scaled budgets, in BOTH anchor
    # regimes. The aged ladder is the steady-state reaper; the fast arm is
    # the same x1 budget replayed inside one anchor window, which is what a
    # tight burst actually does to both stores (held by anchors, not evicted).
    interleaved = [(c, t) for t in a_turns for c in ("A", "B")]
    final_size = entries_a[("A", max(a_turns))].size
    for mult in budgets:
        budget = int(mult * final_size)
        run_pair(f"S2 interleaved x{mult:g} aged", interleaved, budget, window=0)
    run_pair(
        "S2 interleaved x1 fast-window",
        interleaved,
        int(1 * final_size),
        window=len(interleaved),
    )

    # S3: grow the chain under a binding budget — a growing chain HOLDS over
    # budget by anchors (every member is structurally anchored, the leaf is
    # freshly served), which is the daemon's documented hold-and-continue
    # rule. Then model the background reaper: idle ticks where the anchor
    # window has expired and the sweeper runs repeatedly until a fixed point,
    # cascading leaf -> parent as successors die. THEN re-request the chain:
    # republish cost is full bytes (whole) vs the chunks whose refcount hit
    # zero (chunk). Skipping the reaper phase makes S3 measure nothing —
    # nothing ever evicts mid-chain.
    tight = int(s3_mult * final_size)
    for store in (WholeStore(), ChunkStore()):
        r = replay(
            store,
            chain + [("reap", len(chain))] + chain,
            tight,
            all_entries,
            window=0,
            track_restore=True,
        )
        r.name = f"S3 evict-then-re-request x{s3_mult:g} aged"
        results.append(r)
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("grow_dir", type=Path)
    # defaults reproduce the committed tables/evict_sim.json rows
    ap.add_argument("--avg-kib", type=int, default=16)
    ap.add_argument(
        "--codec",
        choices=("cdc", "token-block"),
        default="cdc",
        help="chunking for the simulated chunk store: cdc/head at --avg-kib, "
        "or token-block/64 (the recommended v1 codec)",
    )
    ap.add_argument(
        "--budgets",
        default="0.5,0.9,1,1.25,2",
        help="budgets as a fraction of one whole final blob; values below 1 "
        "are the only ones that can bind on a deduped store",
    )
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    turns = turn_blobs(args.grow_dir)
    entries = build_entries(turns, args.avg_kib * 1024, args.codec)
    results = run_scenarios(entries, [float(x) for x in args.budgets.split(",")])

    print(
        f"{'scenario':<36} {'store':<6} {'hit':>7} {'final MiB':>10} "
        f"{'budget MiB':>10} {'bind':>5} {'evict':>6} {'over':>5} {'win':>4}"
    )
    for r in results:
        mib = r.final_bytes / (1 << 20)
        bud = "-" if r.budget_bytes is None else f"{r.budget_bytes / (1 << 20):.1f}"
        bind = "-" if r.binding is None else ("yes" if r.binding else "no")
        win = "-" if r.window_ticks is None else str(r.window_ticks)
        print(
            f"{r.name:<36} {r.store_kind:<6} {r.hit_rate:>7.4f} {mib:>10.1f} "
            f"{bud:>10} {bind:>5} {r.evictions:>6} "
            f"{'yes' if r.over_budget else '-':>5} {win:>4}"
        )
    # select by name/kind, not position: an inserted scenario must never
    # silently become the headline ratio
    s1w = next(r for r in results if r.name.startswith("S1") and r.store_kind == "whole")
    s1c = next(r for r in results if r.name.startswith("S1") and r.store_kind == "chunk")
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
        if r.store_kind == "chunk" and r.name.startswith("S2") and r.binding is False:
            print(
                f"WARNING {r.name}: budget {r.budget_bytes} never reached; "
                "chunk hit rate is not evidence under pressure"
            )
    for r in results:
        if r.restore_bytes is not None:
            print(
                f"{r.name:<36} {r.store_kind:<6} republish after eviction: "
                f"{r.restore_bytes / (1 << 20):.1f} MiB"
            )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps([r.__dict__ for r in results], indent=1))
        print(f"json -> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
