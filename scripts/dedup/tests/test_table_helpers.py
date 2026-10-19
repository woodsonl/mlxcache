"""Hermetic tests for the remaining table-producing helpers.

Value: protects=the boundary_stability column, the per-scope chunk offsets,
pct()/stats() conventions and first_diff's multi-block path in the committed
tables; fails_when=any of those helpers changes behavior that the tables'
numbers depend on; why_new=these helpers produce committed evidence columns
and had no direct tests; seam=none; no_claim=real-capture stability values
(these pin the helpers' contracts, not the measured numbers).
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chunk_sim  # noqa: E402
from blob_forensics import first_diff  # noqa: E402
from chunk_sim import blob_chunk_sets, boundary_stability, pct  # noqa: E402


def prng_bytes(n: int, salt: int) -> bytes:
    out = bytearray(n)
    s = salt
    for i in range(n):
        s = (s * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        out[i] = (s >> 33) & 0xFF
    return bytes(out)


def st_payload(tensors: dict[str, bytes]) -> bytes:
    import json
    import struct

    off, header = 0, {}
    for name, b in tensors.items():
        header[name] = {
            "dtype": "U8",
            "shape": [1, 1, len(b), 1],
            "data_offsets": [off, off + len(b)],
        }
        off += len(b)
    hj = json.dumps(header).encode()
    return struct.pack("<Q", len(hj)) + hj + b"".join(tensors.values())


def frame_blob(meta: dict, payload: bytes, path: Path) -> Path:
    import json
    import struct

    h = json.dumps(meta).encode()
    path.write_bytes(struct.pack("<I", len(h)) + h + payload)
    return path


def test_boundary_stability_prefix_growth_is_one(tmp_path, monkeypatch):
    """A byte-prefix extension must keep every prior CDC cut: stability 1.0.
    A mid-range mutation must drop below 1.0."""
    avg = 1 << 10
    head1 = prng_bytes(60_000, salt=3)
    head2 = head1 + prng_bytes(20_000, salt=4)  # pure extension

    def fake_load(path):
        return ({"token_count": 5}, path.read_bytes())

    def fake_parse(payload):
        return {"k": {"begin": 0, "end": len(payload), "shape": [1, 1, len(payload), 1]}}, 8

    p1 = frame_blob({"token_count": 5}, head1, tmp_path / "t1.ckpt")
    p2 = frame_blob({"token_count": 5}, head2, tmp_path / "t2.ckpt")
    monkeypatch.setattr(chunk_sim, "load_blob", fake_load)
    monkeypatch.setattr(chunk_sim, "parse_safetensors", fake_parse)
    assert boundary_stability([(1, p1), (2, p2)], avg) == 1.0

    mutated = bytearray(head2)
    # a single flipped byte rarely moves a CDC cut (the gear window resyncs
    # within mask_bits); an 8 KiB span crosses ~8 expected cuts at avg=1 KiB
    mutated[1000 : 1000 + 8192] = prng_bytes(8192, salt=9)
    p2m = frame_blob({"token_count": 5}, bytes(mutated), tmp_path / "t2m.ckpt")
    assert boundary_stability([(1, p1), (2, p2m)], avg) < 1.0


@pytest.mark.parametrize("scope", ["file", "tensor", "head"])
def test_blob_chunk_sets_scopes_digest_at_absolute_offsets(scope):
    """Whatever the scope, a chunk's digest must equal sha256 of the payload
    slice at its own absolute offset — the offset translation is what the
    per-scope table rows depend on."""
    t1 = prng_bytes(40_000, salt=5)
    t2 = prng_bytes(40_000, salt=6)
    payload = st_payload({"a": t1, "b": t2})
    import json
    import struct

    (shlen,) = struct.unpack("<Q", payload[:8])
    header = json.loads(payload[8 : 8 + shlen])
    tensors = {
        n: {
            "begin": 8 + shlen + s["data_offsets"][0],
            "end": 8 + shlen + s["data_offsets"][1],
            "shape": s["shape"],
        }
        for n, s in header.items()
    }
    avg = 1 << 10
    chunks = blob_chunk_sets(payload, tensors, scope, avg)
    assert chunks
    for end, size, digest in chunks:
        assert 0 < size <= 4 * avg
        assert hashlib.sha256(payload[end - size : end]).hexdigest() == digest


def test_pct_convention_pinned():
    assert pct([], 0.5) == 0
    assert pct(list(range(100)), 0.5) == 50
    # the clamp: p95 of 10 elements is the 10th (index 9), not the 9th
    assert pct(list(range(10)), 0.95) == 9


def test_first_diff_divergence_beyond_first_block():
    """Real payloads are 100+ MiB; the block scan's later blocks must work."""
    n = (2 << 20) + 100  # just past two 1 MiB blocks
    a = bytearray(n)
    a[1_500_000] = 0x01  # second block
    b = bytes(n)
    assert first_diff(bytes(a), b) == 1_500_000


def test_stats_median_convention_pinned():
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from restore_bench import stats

    # with 10 reps the recorded 'median' is the 6th smallest — the committed
    # restore deltas are median-minus-median on exactly this definition
    assert stats(list(range(1, 11))) == {"min_s": 1, "median_s": 6, "p90_s": 10}
