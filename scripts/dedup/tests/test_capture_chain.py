"""Hermetic invariants for the capture driver's pure helpers.

The capture spawns the real engine pair, so the orchestration itself is an
opt-in real-engine run (like the T22 probes). The one behavior CI CAN pin is
the publish-race guard: a turn whose blob never lands must abort the capture
loudly, and an already-landed blob must not wait. The code comment records a
real incident (7 blobs for 8 turns silently dropping a turn's comparison).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from capture_chain import wait_publish  # noqa: E402


def test_wait_publish_waits_and_times_out_loudly(tmp_path):
    """VALUE: wait_publish is the capture's race guard. Catches both real
    regressions: `>=` for `>` returns immediately with the pre-turn count
    (the snapshot then misses the turn's blob, and the loss surfaces only
    hours later at analysis time), and dropping the timeout raise lets a
    no-publish turn pass silently. Timed behavior, not internals."""
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    (blobs / "a.ckpt").write_bytes(b"x")
    assert wait_publish(blobs, before=0) == 1  # new blob already present
    (blobs / "b.ckpt").write_bytes(b"x")
    assert wait_publish(blobs, before=1) == 2
    with pytest.raises(SystemExit, match="no new blob"):
        wait_publish(blobs, before=2, timeout_s=0.15)
