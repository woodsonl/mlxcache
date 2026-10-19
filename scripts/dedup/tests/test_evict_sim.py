"""Hermetic invariants for the eviction simulation (E5).

Synthetic chain — no capture, no model. What is asserted:
  - a chunk leaves the store only at refcount 0 (shared chunks survive the
    eviction of one of their holders);
  - structurally anchored entries are never evicted; a chain that fits
    nowhere reports over-budget-by-anchors instead of breaking the contract;
  - the budget holds after every sweep that had candidates;
  - at a binding budget, the chunk store's hit rate is never worse than the
    whole-blob store's, and a growing chain costs unique bytes, not
    full-blob bytes per turn;
  - a budget above the store's reachable size is reported as non-binding, so
    a row that never evicted cannot be read as policy evidence.

Run: uv run pytest scripts/dedup/tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evict_sim import ChunkStore, Entry, WholeStore, replay  # noqa: E402


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
    for k in range(2, 9):
        e = Entry(
            convo="B",
            turn=k,
            tokens=1_000 + 100 * k,
            size=1_000_000 + 100_000 * k,
            chunks={**a1.chunks, **{f"B-u{k}-{i}": 60_000 for i in range(3)}},
            parent=parent,
        )
        b[("B", k)] = e
        parent = e
    both = dict(a)
    both.update(b)
    return both


def test_shared_chunk_surves_single_eviction():
    s = ChunkStore()
    entries = synth_chain()
    s.publish(entries[("A", 1)])
    s.publish(entries[("A", 2)])
    before = s.bytes()
    s.entries[0].live = False  # simulate direct eviction of turn 1
    for d in entries[("A", 1)].chunks:
        s.ref[d] -= 1
        if s.ref[d] == 0:
            del s.chunks_on_disk[d]
    # turn 2 still holds every shared chunk; only turn-1-unique chunks freed
    unique1 = 3 * 60_000
    assert before - s.bytes() == unique1
    assert len(s.chunks_on_disk) == len(entries[("A", 2)].chunks)


def test_anchored_chain_never_broken():
    entries = synth_chain()
    s = WholeStore()
    # budget fits A1+A2 (2.3M) but not A3: the sweep may evict only the
    # unanchored newest entry; A1/A2 stay anchored by each successor.
    r = replay(s, [("A", t) for t in range(1, 5)], 2_500_000, entries)
    assert r.evictions == 2  # A3 and A4, each evicted by its own sweep
    assert r.over_budget is False
    assert entries[("A", 1)].live and entries[("A", 2)].live
    assert not entries[("A", 3)].live and not entries[("A", 4)].live
    # A4 was served from A2's blob (partial hit), never from scratch
    assert r.reused_positions >= 2 * (entries[("A", 2)].tokens - 1)


def test_budget_held_when_candidates_exist():
    entries = divergent_convo()
    reqs = [(c, t) for t in range(1, 9) for c in ("A", "B")]
    budget = 4_000_000
    r = replay(WholeStore(), reqs, budget, entries)
    assert r.final_bytes <= budget or r.over_budget
    r2 = replay(ChunkStore(), reqs, budget, entries)
    assert r2.final_bytes <= budget or r2.over_budget


def test_chunked_hit_rate_not_worse_same_budget():
    entries = divergent_convo()
    reqs = [(c, t) for t in range(1, 9) for c in ("A", "B")]
    for budget in (3_000_000, 4_000_000, 8_000_000):
        w = replay(WholeStore(), reqs, budget, entries)
        c = replay(ChunkStore(), reqs, budget, entries)
        assert c.hit_rate >= w.hit_rate - 1e-9, (budget, c.hit_rate, w.hit_rate)


def test_chain_growth_unique_not_full():
    entries = synth_chain()
    w = replay(WholeStore(), [("A", t) for t in range(1, 9)], 1 << 62, entries)
    c = replay(ChunkStore(), [("A", t) for t in range(1, 9)], 1 << 62, entries)
    # a growing chain is partial hits after turn 1 (daemon semantics), and
    # both representations serve identical lookups with no budget pressure
    assert w.hit_rate == c.hit_rate > 0.8
    assert w.evictions == c.evictions == 0
    # 8 turns: whole stores every full blob; chunk stores prefix + deltas
    assert c.final_bytes < w.final_bytes / 4


def test_binding_flag_distinguishes_pressure_from_slack():
    """A budget the store never reaches must report binding=False.

    This is the gate's own failure mode: a budget expressed as a multiple of
    one whole blob can exceed a deduped store's entire contents, in which case
    zero evictions means "never under pressure", not "policy holds"."""
    entries = divergent_convo()
    reqs = [(c, t) for t in range(1, 9) for c in ("A", "B")]
    tight = 2_500_000
    slack = 1 << 40

    pressured = replay(ChunkStore(), reqs, tight, entries)
    assert pressured.binding is True
    assert pressured.evictions > 0
    assert pressured.budget_bytes == tight

    relaxed = replay(ChunkStore(), reqs, slack, entries)
    assert relaxed.binding is False
    assert relaxed.evictions == 0


def test_binding_budget_chunked_hit_rate_not_worse():
    """The hit-rate invariant only means something where the budget binds."""
    entries = divergent_convo()
    reqs = [(c, t) for t in range(1, 9) for c in ("A", "B")]
    for budget in (1_500_000, 2_000_000, 2_500_000, 3_000_000):
        w = replay(WholeStore(), reqs, budget, entries)
        c = replay(ChunkStore(), reqs, budget, entries)
        assert c.binding and w.binding, (budget, c.binding, w.binding)
        assert c.hit_rate >= w.hit_rate - 1e-9, (budget, c.hit_rate, w.hit_rate)


def test_restore_bytes_only_count_after_first_eviction():
    entries = synth_chain()
    chain = [("A", t) for t in range(1, 9)]
    r = replay(WholeStore(), chain + chain, 2_000_000, entries, track_restore=True)
    c = replay(ChunkStore(), chain + chain, 2_000_000, entries, track_restore=True)
    # both re-request the whole chain after eviction; the chunk store rebuilds
    # shared prefix chunks that a whole-blob store must re-store in full
    assert r.restore_bytes and c.restore_bytes
    assert c.restore_bytes <= r.restore_bytes
