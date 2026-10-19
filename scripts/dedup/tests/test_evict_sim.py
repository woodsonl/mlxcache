"""Hermetic invariants for the eviction simulation (E5).

Synthetic chain — no capture, no model. What is asserted:
  - a chunk leaves the store only at refcount 0 (shared chunks survive the
    eviction of one of their holders);
  - structurally anchored entries are never evicted; a chain that fits
    nowhere reports over-budget-by-anchors instead of breaking the contract;
  - the budget holds after EVERY sweep that had candidates, not just at the
    end of the replay;
  - recency anchoring matches index.py's semantics: inside one anchor window
    nothing recently served is evicted (budget held by anchors), and once the
    window expires entries become evictable;
  - at a binding budget the chunk store's hit rate is never worse than the
    whole-blob store's, and a budget the store never reaches reports
    binding=False so a slack row cannot be read as policy evidence;
  - a growing chain costs unique bytes, not full-blob bytes per turn;
  - re-request after eviction re-stores fewer bytes in the chunk store.

Run: uv run pytest scripts/dedup/tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evict_sim import (  # noqa: E402
    UNBUDGETED,
    ChunkStore,
    Entry,
    WholeStore,
    anchored_ids,
    replay,
)


def synth_chain(convo: str = "A", turns: int = 8) -> dict[tuple[str, int], Entry]:
    out: dict[tuple[str, int], Entry] = {}
    parent: Entry | None = None
    shared = {f"{convo}-p{i}": 200_000 for i in range(5)}
    for k in range(1, turns + 1):
        e = Entry(
            convo=convo,
            turn=k,
            tokens=1_000 + 100 * k,
            size=1_000_000 + 100_000 * k,
            chunks={**shared, **{f"{convo}-u{k}-{i}": 60_000 for i in range(3)}},
            parent=parent,
        )
        out[(convo, k)] = e
        parent = e
    return out


def divergent_convo() -> dict[tuple[str, int], Entry]:
    a = synth_chain("A")
    b: dict[tuple[str, int], Entry] = {}
    a1 = a[("A", 1)]
    b[("B", 1)] = a1  # the SAME turn-1 blob serves both conversations
    parent: Entry | None = a1
    cumulative = dict(a1.chunks)
    for k in range(2, 9):
        cumulative[f"B-u{k}"] = 60_000
        e = Entry(
            convo="B",
            turn=k,
            tokens=1_000 + 100 * k,
            size=1_000_000 + 100_000 * k,
            chunks=dict(cumulative),
            parent=parent,
        )
        b[("B", k)] = e
        parent = e
    both = dict(a)
    both.update(b)
    return both


def test_shared_chunk_survives_single_eviction():
    """Drive the eviction through the production enforce() path (not a
    hand-rolled copy of its bookkeeping): turn 1 becomes evictable only when
    its successor is gone, and evicting the successor must free exactly the
    successor's unique bytes while every shared chunk stays on disk."""
    s = ChunkStore()
    entries = synth_chain()
    s.publish(entries[("A", 1)], tick=0)
    s.publish(entries[("A", 2)], tick=1)
    before = s.bytes()
    # both entries are structurally anchored while both are live; make turn 2
    # evictable by the policy's own rules (aged out, leaf, over budget)
    s.entries[-1].last_used = -10  # aged: tick 1 - (-10) > window 0
    r = s.enforce(0, tick=1, window=0)
    assert r["evicted"] == 1
    # turn 2's unique chunks freed; the shared prefix (held by turn 1) stays
    unique2 = 3 * 60_000
    assert before - s.bytes() == unique2
    assert len(s.chunks_on_disk) == len(entries[("A", 1)].chunks)
    assert all(d in s.chunks_on_disk for d in entries[("A", 1)].chunks)


def test_anchored_chain_never_broken():
    """A strictly growing chain under a binding budget HOLDS over budget by
    anchors — the daemon's documented hold-and-continue rule — rather than
    evicting the just-published blob the way an anchor-blind policy would."""
    entries = synth_chain()
    s = WholeStore()
    r = replay(s, [("A", t) for t in range(1, 5)], 2_500_000, entries, window=0)
    total = sum(entries[("A", t)].size for t in range(1, 5))
    assert total > 2_500_000  # the budget genuinely binds
    assert r.evictions == 0  # nothing evictable: chain is anchored end to end
    assert r.over_budget is True
    assert all(entries[("A", t)].live for t in range(1, 5))


def test_budget_holds_after_every_sweep_with_candidates():
    """The per-sweep bound, not just the final state: whenever the store is
    over budget, no evictable (non-anchored, live) candidate may remain."""
    for store in (WholeStore(), ChunkStore()):
        entries = divergent_convo()
        for e in entries.values():
            e.live = False
            e.last_used = 0
        store.entries.clear()
        if isinstance(store, ChunkStore):
            store.ref.clear()
            store.chunks_on_disk.clear()
        budget = 3_000_000
        for tick, (convo, turn) in enumerate((c, t) for t in range(1, 9) for c in ("A", "B")):
            own = entries[(convo, turn)]
            if own.live:
                own.last_used = tick
                continue
            store.publish(own, tick)
            r = store.enforce(budget, tick=tick, window=0)
            anchored = anchored_ids(store.entries, tick, 0)
            evictable = any(e.live and id(e) not in anchored for e in store.entries)
            assert not (r["bytes"] > budget and evictable), (convo, turn, r["bytes"])


def test_chunked_hit_rate_not_worse_only_where_binding():
    """The chunk store's hit rate is never worse at ANY tested budget; a
    budget a given store never reached is asserted as its own contract
    (that store evicted nothing — slack, not policy evidence)."""
    entries = divergent_convo()
    reqs = [(c, t) for t in range(1, 9) for c in ("A", "B")]
    for budget in (1_500_000, 2_000_000, 2_500_000, 3_000_000, 4_000_000, 8_000_000):
        w = replay(WholeStore(), reqs, budget, entries, window=0)
        c = replay(ChunkStore(), reqs, budget, entries, window=0)
        if not w.binding:
            assert w.evictions == 0, (budget, w.evictions)
        if not c.binding:
            assert c.evictions == 0, (budget, c.evictions)
        assert c.hit_rate >= w.hit_rate - 1e-9, (budget, c.hit_rate, w.hit_rate)


def test_chain_growth_unique_not_full():
    entries = synth_chain()
    w = replay(WholeStore(), [("A", t) for t in range(1, 9)], UNBUDGETED, entries)
    c = replay(ChunkStore(), [("A", t) for t in range(1, 9)], UNBUDGETED, entries)
    # a growing chain is partial hits after turn 1 (daemon semantics), and
    # both representations serve identical lookups with no budget pressure
    assert w.hit_rate == c.hit_rate > 0.8
    assert w.evictions == c.evictions == 0
    # 8 turns: whole stores every full blob; chunk stores prefix + deltas
    assert c.final_bytes < w.final_bytes / 4
    # unbudgeted rows must record budget_bytes/binding/window as None so an
    # S1 row can never be misread as policy evidence
    for r in (w, c):
        assert r.budget_bytes is None and r.binding is None and r.window_ticks is None


def test_binding_flag_distinguishes_pressure_from_slack():
    """A budget the store never reaches must report binding=False.

    This is the gate's own failure mode: a budget expressed as a multiple of
    one whole blob can exceed a deduped store's entire contents, in which case
    zero evictions means "never under pressure", not "policy holds"."""
    entries = divergent_convo()
    reqs = [(c, t) for t in range(1, 9) for c in ("A", "B")]
    tight = 2_500_000
    slack = 1 << 40

    pressured = replay(ChunkStore(), reqs, tight, entries, window=0)
    assert pressured.binding is True
    assert pressured.budget_bytes == tight

    relaxed = replay(ChunkStore(), reqs, slack, entries, window=0)
    assert relaxed.binding is False
    assert relaxed.evictions == 0


def test_recency_window_holds_recent_entries():
    """index.py semantics: an entry served within the anchor window is
    anchored. Inside one window, a binding budget is HELD by anchors (nothing
    evicts, over_budget reports true); aged (window=0), the same budget
    evicts."""
    entries = divergent_convo()
    reqs = [(c, t) for t in range(1, 9) for c in ("A", "B")]
    budget = 2_500_000

    fast = replay(ChunkStore(), reqs, budget, entries, window=len(reqs))
    assert fast.evictions == 0
    assert fast.over_budget is True  # held by anchors, not broken

    fast_w = replay(WholeStore(), reqs, budget, entries, window=len(reqs))
    assert fast_w.evictions == 0
    assert fast_w.over_budget is True

    aged = replay(ChunkStore(), reqs, budget, entries, window=0)
    assert aged.evictions > 0


def test_cumulative_conversation_b_nests():
    """B's chain must be cumulative: B-k contains B-(k-1)'s unique chunks,
    so evicting B-k frees its own delta, not the shared prefix."""
    from evict_sim import synthetic_convo_b

    entries = synth_chain()
    b = synthetic_convo_b(entries)
    for k in range(3, 9):
        assert set(b[("B", k - 1)].chunks) < set(b[("B", k)].chunks)


def test_republish_after_eviction_counts_once():
    """S3's re-request republishes evicted entries within one replay. A
    double-published entry used to appear twice in the ledger (bytes and
    refcounts double-counted, inflating whole-store republish ~2x and
    pinning chunk refcounts above zero)."""
    entries = synth_chain()
    s = WholeStore()
    s.publish(entries[("A", 1)], tick=0)
    s.entries[0].live = False  # evict
    before = s.bytes()
    s.publish(entries[("A", 1)], tick=1)  # re-request
    assert s.bytes() - before == entries[("A", 1)].size
    assert len(s.entries) == 1

    c = ChunkStore()
    c.publish(entries[("A", 2)], tick=0)
    c.entries[0].live = False
    for d in entries[("A", 2)].chunks:
        c.ref[d] -= 1
        if c.ref[d] == 0:
            del c.chunks_on_disk[d]
    cbefore = c.bytes()
    c.publish(entries[("A", 2)], tick=1)
    # chunk store re-stores exactly the entry's chunk bytes, once
    assert c.bytes() - cbefore == sum(entries[("A", 2)].chunks.values())
    assert len(c.entries) == 1
    assert all(v == 1 for v in c.ref.values())


def test_restore_bytes_only_count_after_first_eviction():
    """Grow under a binding budget, let the background reaper cascade (the
    idle sweeps age out anchors), then re-request: republish is full bytes
    for the whole store and only the missing tail for the chunk store."""
    entries = synth_chain()
    chain = [("A", t) for t in range(1, 9)]
    agenda = chain + [("reap", 8)] + chain
    r = replay(WholeStore(), agenda, 2_000_000, entries, track_restore=True)
    c = replay(ChunkStore(), agenda, 2_000_000, entries, track_restore=True)
    assert r.evictions > 0 and c.evictions > 0  # the reaper really ran
    assert r.restore_bytes and c.restore_bytes
    assert c.restore_bytes < r.restore_bytes


def test_enforce_evicts_largest_first_per_policy():
    """VALUE: the two stores' eviction ORDER is the policy the sim exists to
    measure — WholeStore mirrors store.py §7 (token_count DESC); ChunkStore
    proposes biggest-unique-tail first (only unique bytes reclaim space). A
    silent sort-key change (ascending, or plain entry size) would attribute a
    different policy's numbers to the design evidence while every existing
    invariant still holds, so the evicted SET is pinned directly."""
    # WholeStore: token_count DESC — the biggest-token entry dies first
    w = WholeStore()
    ws = []
    for k, tok in enumerate((1000, 2000, 3000)):
        e = Entry(convo="A", turn=k + 1, tokens=tok, size=100, chunks={}, parent=None)
        ws.append(e)
        w.publish(e, tick=k)
    r = w.enforce(250, tick=99, window=0)
    assert r["evicted"] == 1
    assert not ws[2].live and ws[0].live and ws[1].live

    # ChunkStore: biggest unique tail first — e3 owns the most unique bytes
    # (40) despite the smallest total size, so it is evicted before the fat
    # shared holders; the doubly-referenced chunk must survive
    c = ChunkStore()
    e1 = Entry("B", 1, tokens=10, size=220, chunks={"s": 200, "u1": 20}, parent=None)
    e2 = Entry("B", 2, tokens=20, size=210, chunks={"s": 200, "u2": 10}, parent=None)
    e3 = Entry("B", 3, tokens=30, size=40, chunks={"u3": 40}, parent=None)
    for k, e in enumerate((e1, e2, e3)):
        c.publish(e, tick=k)
    r = c.enforce(250, tick=99, window=0)  # disk 270 B, budget 250: one eviction
    assert r["evicted"] == 1
    assert not e3.live and e1.live and e2.live
    assert c.ref["s"] == 2 and c.chunks_on_disk["s"] == 200
