"""Hermetic tests for roundtrip_probe's layout helpers and the codec-block
single-source contract.

Value: protects=head_blocks' shape rule and compare()'s per-head verdict
arithmetic (the layout table's math), and TOKEN_BLOCK's agreement with the
chunker CLI default (the eviction arm must measure the codec the decision
adopts); fails_when=head_blocks mishandles non-4D shapes, compare() miscounts
exact heads at full coverage, or the two 64s drift apart; why_new=the probe
module was unimportable without mlx (CI cannot import it) and nothing tied
the block-size definitions; seam=none (mlx import deferred to the functions
that need it); no_claim=real-engine probe behavior (mlx paths, opt-in).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chunk_sim  # noqa: F401 — asserted via DEFAULT_TOKEN_BLOCKS below
from chunk_sim import TOKEN_BLOCK  # noqa: E402
from roundtrip_probe import compare, head_blocks  # noqa: E402


def test_head_blocks_shape_rule():
    assert head_blocks([1, 4, 7434, 128]) == 4
    assert head_blocks([7434]) == 1


def _tensors(payload_len: int, n_heads: int, positions: int) -> tuple[bytes, dict]:
    per_head = payload_len // n_heads
    payload = bytes((h * 37 + p) % 256 for h in range(n_heads) for p in range(per_head))
    tensors = {"kv": {"begin": 0, "end": payload_len, "shape": [1, n_heads, positions, 1]}}
    return payload, tensors


def test_compare_counts_exact_heads_across_every_tensor():
    """Identical payloads: every head of every tensor must be exact."""
    payload, tensors = _tensors(4 * 64, n_heads=4, positions=64)
    record: list[dict] = []
    compare("same vs same", payload, payload, tensors, tensors, record)
    assert record == [
        {
            "comparison": "same vs same",
            "heads_exact": 4,
            "heads_total": 4,
            "verdict": "exact",
            "detail": [{"tensor": "kv", "heads": ["==", "==", "==", "=="]}],
        }
    ]


def test_compare_reports_first_diff_per_head():
    """Head 0 identical, head 1 differing at its byte 8: verdict partial with
    the offset recorded relative to the head start."""
    payload, tensors = _tensors(4 * 64, n_heads=4, positions=64)
    mutated = bytearray(payload)
    hsz = len(payload) // 4
    mutated[hsz + 8] ^= 0xFF
    record: list[dict] = []
    compare("mutated", payload, bytes(mutated), tensors, tensors, record)
    assert record[0]["verdict"] == "partial"
    assert record[0]["heads_exact"] == 3
    assert record[0]["detail"][0]["heads"][0] == "=="
    assert record[0]["detail"][0]["heads"][1] == "diff@8"


def test_token_block_constant_agrees_with_cli_default():
    """The eviction arm's codec block must be the chunker's default first arm;
    if the default list is edited, this fails instead of silently comparing
    two different codecs in the regenerated tables."""
    from chunk_sim import DEFAULT_TOKEN_BLOCKS

    first = int(DEFAULT_TOKEN_BLOCKS.split(",")[0])
    assert first == TOKEN_BLOCK
