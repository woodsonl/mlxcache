"""Hermetic invariants for the CDC chunker (G1's measurement path).

Synthetic bytes only — no blob, no capture. The property under test is the
one the committed tables depend on: the vectorized gear hash produces
exactly the naive sequential reference's cuts, under the same mask and
envelope the measurement path uses, including through the multi-block
gather path (real payloads are hundreds of MB; the block carry must not
change a single cut)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chunk_sim  # noqa: E402
from chunk_sim import (  # noqa: E402
    _TABLE,
    apply_envelope,
    cdc_envelope,
    chunk_stream,
    gear_cut_candidates,
    mask_bits_for,
    naive_gear_cdc,
)


def prng_bytes(n: int, salt: int) -> bytes:
    out = bytearray(n)
    s = salt
    for i in range(n):
        s = (s * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        out[i] = (s >> 33) & 0xFF
    return bytes(out)


@pytest.mark.parametrize("mask_bits", [8, 12, 14])
def test_vectorized_matches_naive(mask_bits: int):
    data = prng_bytes(300_000, salt=7)
    arr = np.frombuffer(data, dtype=np.uint8)
    avg = 1 << mask_bits
    cands = gear_cut_candidates(arr, _TABLE, mask_bits)
    assert apply_envelope(cands, len(data), *cdc_envelope(avg)) == naive_gear_cdc(
        data, mask_bits, *cdc_envelope(avg)
    )


def test_blocked_gather_identical_to_single_block(monkeypatch):
    """Force many tiny blocks so the carry path runs constantly; the
    candidates must be byte-identical to one big block."""
    data = prng_bytes(200_000, salt=11)
    arr = np.frombuffer(data, dtype=np.uint8)
    monkeypatch.setattr(chunk_sim, "_GATHER_BLOCK", 64)
    blocked = gear_cut_candidates(arr, _TABLE, 14)
    monkeypatch.setattr(chunk_sim, "_GATHER_BLOCK", 1 << 24)
    single = gear_cut_candidates(arr, _TABLE, 14)
    assert np.array_equal(blocked, single)


@pytest.mark.parametrize("avg", [1 << 8, 1 << 12, 1 << 14])
def test_envelope_bounds_respected(avg: int):
    data = prng_bytes(300_000, salt=9)
    cuts = chunk_stream(data, avg, *cdc_envelope(avg))[1]
    sizes = [b - a for a, b in zip([0, *cuts[:-1]], cuts, strict=True)]
    assert sizes and min(sizes) >= cdc_envelope(avg)[0]
    assert max(sizes) <= cdc_envelope(avg)[1]


def test_mask_bits_clamped_for_small_targets():
    assert mask_bits_for(64) == 8  # clamp: log2(64)=6 would under-mask
    assert mask_bits_for(16_384) == 14


def test_input_shorter_than_mask_yields_no_candidates():
    assert gear_cut_candidates(np.frombuffer(b"abc", dtype=np.uint8), _TABLE, 8).size == 0


def test_simulate_single_turn_fails_loudly(tmp_path, monkeypatch):
    """A one-blob grow run cannot produce a turns-2-plus mean; it must fail
    with a message, not a ZeroDivisionError deep in the math."""
    blob = tmp_path / "only.ckpt"
    monkeypatch.setattr(chunk_sim, "load_blob", lambda p: ({"token_count": 4}, b"data"))
    with pytest.raises(SystemExit, match="need >= 2 turns"):
        chunk_sim.simulate([(1, blob)], "cfg", lambda payload, meta: [(4, 4, "d0")])
