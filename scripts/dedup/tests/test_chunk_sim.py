"""Hermetic invariants for the token-block chunker (G1's measurement path).

Synthetic per-head layout only — no blob, no capture. The property under test
is the one G1 depends on: a block's identity covers exactly its positions, so
a prefix of the positions yields a subset of the full chunk set. A block that
leaks bytes from a neighbouring position would still look "stable" here, so
the test also pins the digest to the documented construction.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chunk_sim  # noqa: E402
from chunk_sim import token_block_chunks  # noqa: E402

HEADS = 4
UNIT = 8  # bytes per position per head


def make_layout(positions: int) -> tuple[bytes, dict]:
    """One tensor, per-head contiguous: [head0 positions][head1 positions]..."""
    per_head = UNIT * positions
    payload = bytes((h * 37 + p) % 256 for h in range(HEADS) for p in range(per_head))
    tensors = {
        "kv": {
            "begin": 0,
            "end": HEADS * per_head,
            "shape": (1, HEADS, positions, UNIT),
        }
    }
    return payload, tensors


def test_block_cuts_cover_every_position_exactly_once():
    positions, block = 16, 4
    payload, tensors = make_layout(positions)
    chunks = token_block_chunks(payload, tensors, positions, block)
    assert len(chunks) == positions // block
    # cut offsets are monotonically increasing and end at the tensor end
    cuts = [c[0] for c in chunks]
    assert cuts == sorted(cuts)
    assert cuts[-1] == tensors["kv"]["end"]
    assert sum(c[1] for c in chunks) == tensors["kv"]["end"]


def test_partial_tail_block_shorter_than_block():
    positions, block = 10, 4
    payload, tensors = make_layout(positions)
    chunks = token_block_chunks(payload, tensors, positions, block)
    assert len(chunks) == 3
    assert chunks[-1][1] == 2 * HEADS * UNIT  # 2 leftover positions


def test_prefix_positions_yield_subset_of_full_chunks():
    """The G1 property: truncating the chain must not re-key earlier blocks.

    Per-head layout means a naive flat cut would shift every later head block;
    this chunker must be immune to that."""
    full_positions, prefix = 16, 8
    block = 4
    payload, tensors = make_layout(full_positions)
    full = token_block_chunks(payload, tensors, full_positions, block)

    truncated = bytes((h * 37 + p) % 256 for h in range(HEADS) for p in range(UNIT * prefix))
    pref_tensors = {
        "kv": {
            "begin": 0,
            "end": HEADS * UNIT * prefix,
            "shape": (1, HEADS, prefix, UNIT),
        }
    }
    part = token_block_chunks(truncated, pref_tensors, prefix, block)

    full_digests = {d for _, _, d in full}
    part_digests = [d for _, _, d in part]
    assert set(part_digests) <= full_digests
    assert part_digests == [d for _, _, d in full][: len(part_digests)]


def test_digest_is_over_positions_not_flat_bytes():
    """Pin the construction: block=1 chunk digest == sha256 of one position's
    slice from every head, in head order."""
    positions, block = 4, 1
    payload, tensors = make_layout(positions)
    chunks = token_block_chunks(payload, tensors, positions, block)

    expected = []
    per_head = UNIT * positions
    for p in range(positions):
        parts = [
            payload[h * per_head + p * UNIT : h * per_head + (p + 1) * UNIT] for h in range(HEADS)
        ]
        expected.append(hashlib.sha256(b"".join(parts)).hexdigest())
    assert [d for _, _, d in chunks] == expected


def test_indivisible_head_size_fails_loudly():
    payload, tensors = make_layout(8)
    # 4 extra bytes across 4 heads leaves each head at 33 bytes: not a whole
    # number of positions, which must fail loudly rather than mis-slice.
    bad = {"kv": {"begin": 0, "end": tensors["kv"]["end"] + 4, "shape": (1, HEADS, 8, UNIT)}}
    with pytest.raises(SystemExit, match="not divisible"):
        token_block_chunks(payload, bad, 8, 4)


def test_simulate_counts_new_bytes_against_cumulative_store(tmp_path, monkeypatch):
    """VALUE: simulate() produces G1's new-bytes-per-turn number. Catches the
    accounting regressions that would silently move the committed gate line:
    updating the store before counting (new_bytes collapses to 0 and G1 looks
    perfect), counting against only the previous turn (re-inflates), or
    letting turn 1 into the mean's denominator. The chunker is injected, so
    this pins accounting semantics, not any chunker's behavior."""
    payloads = {
        "t1.ckpt": ({"token_count": 5}, b"AAAA"),
        "t2.ckpt": ({"token_count": 9}, b"AAAABBBB"),
    }
    monkeypatch.setattr(chunk_sim, "load_blob", lambda p: payloads[p.name])
    turns = [(1, tmp_path / "t1.ckpt"), (2, tmp_path / "t2.ckpt")]
    table = {
        b"AAAA": [(4, 4, "d1"), (8, 4, "d2")],
        b"AAAABBBB": [(4, 4, "d1"), (8, 4, "d2"), (12, 4, "d3")],
    }
    r = chunk_sim.simulate(turns, "cfg", lambda payload, meta: table[payload])
    assert r["turns"][0]["new_bytes"] == 8
    assert r["turns"][1]["new_bytes"] == 4  # only d3 is new vs the CUMULATIVE store
    assert r["turns"][1]["new_frac"] == 0.5
    assert r["mean_new_frac_turns2plus"] == 0.5  # mean excludes turn 1 entirely
    assert r["store_unique_bytes"] == 12
