"""Hermetic invariants for the restore-path prototype (E4).

Two behaviors are pinned without any engine: the scratch guard (a
destructive rmtree must never touch a directory this script did not create)
and the manifest reconstruction property (both the unpacked and packed
materializations must reassemble the exact payload — the property the G2
restore claim rests on, and the layout the adopted token-block codec would
ship).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from restore_bench import (  # noqa: E402
    _MARKER,
    REPO_ROOT,
    _reset_scratch,
    bench_chunked,
    materialize,
)


def test_scratch_reset_refuses_foreign_dirs(tmp_path):
    """VALUE: _reset_scratch rmtree's its argument — the guard is the only
    thing between a typo'd --scratch and deleting an unrelated tree. Catches:
    dropping the marker check, dropping the forbidden-root check, or
    resetting before validating. Asserts the safety contract (refusal +
    nothing deleted), not how the check is written."""
    fresh = tmp_path / "fresh"
    _reset_scratch(fresh)  # creates dir + marker
    assert (fresh / _MARKER).exists()
    _reset_scratch(fresh)  # marker present: reset is safe and idempotent

    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "keep.txt").write_text("x")
    with pytest.raises(SystemExit, match="marker"):
        _reset_scratch(foreign)
    assert (foreign / "keep.txt").exists()  # refused BEFORE any deletion

    for forbidden in (Path("/"), Path.home(), Path.cwd(), REPO_ROOT):
        with pytest.raises(SystemExit, match="refusing"):
            _reset_scratch(forbidden)


def test_materialize_reconstruction_unpacked_and_packed(tmp_path):
    """VALUE: materialize + bench_chunked reassemble the payload byte-exactly
    in BOTH layouts — the property behind G2's "+171 ms / −68 ms" restore
    numbers. Catches: payload-offset bookkeeping errors (start = end - size),
    segment-rotation offset errors (seg_off after close/rotate), the inline
    non-tensor spans being dropped or misplaced, and the packed pread path
    reading the wrong segment offsets. Reconstruction is the observable
    contract; chunk boundaries themselves are free to move."""
    heads, positions, unit = 2, 150_000, 4  # 600 KiB per head, ~1.2 MiB payload
    per_head = positions * unit
    rng = np.random.default_rng(1234)
    payload = b"H" * 64 + rng.integers(0, 256, heads * per_head, dtype=np.uint8).tobytes()
    tensors = {
        "k": {"begin": 64, "end": 64 + heads * per_head, "shape": (1, heads, positions, unit)}
    }
    avg = 16 * 1024

    unpacked = materialize(payload, tensors, avg, tmp_path / "u")
    assert unpacked["packed"] is False
    assert unpacked["inline_bytes"] == 64  # the safetensors-header span rides inline
    out = bytearray(len(payload))
    bench_chunked(unpacked, tmp_path / "u", workers=4, out=out)
    assert bytes(out) == payload

    packed = materialize(payload, tensors, avg, tmp_path / "p", pack_mib=1)
    assert packed["packed"] is True
    assert packed["inline_bytes"] == 64
    segs = sorted((tmp_path / "p" / "chunks").glob("seg-*"))
    # 1.2 MiB of chunks cannot fit one 1 MiB segment: rotation must have run
    assert len(segs) >= 2
    assert all(s.stat().st_size <= 1 << 20 for s in segs)
    out2 = bytearray(len(payload))
    bench_chunked(packed, tmp_path / "p", workers=4, out=out2)  # os.pread path
    assert bytes(out2) == payload
