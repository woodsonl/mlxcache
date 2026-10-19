"""Hermetic invariants for the blob forensics primitives.

Every committed table routes through load_blob's framing, parse_safetensors'
offset math, and first_diff's bisection; a bug here corrupts all
forensics/chunk/evict evidence while CI stays green (the analysis tests
hand-build tensor dicts). These pin the primitives to hand-framed inputs."""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blob_forensics import (  # noqa: E402
    block_containment,
    compare_pair,
    first_diff,
    kv_heads,
    layout,
    load_blob,
    parse_safetensors,
    single_blob,
    turn_blobs,
)


def st_payload(tensors: dict[str, bytes]) -> bytes:
    off, header = 0, {}
    for name, b in tensors.items():
        header[name] = {"dtype": "U8", "shape": [len(b)], "data_offsets": [off, off + len(b)]}
        off += len(b)
    hj = json.dumps(header).encode()
    return struct.pack("<Q", len(hj)) + hj + b"".join(tensors.values())


def test_parse_safetensors_offsets_are_data_relative():
    payload = st_payload({"a": b"1234", "b": b"567890"})
    tensors, data_start = parse_safetensors(payload)
    assert data_start == len(payload) - 10
    assert payload[tensors["a"]["begin"] : tensors["a"]["end"]] == b"1234"
    assert payload[tensors["b"]["begin"] : tensors["b"]["end"]] == b"567890"


def test_load_blob_framing_roundtrip(tmp_path):
    meta = {"token_count": 5, "fingerprint": {"kv_bits": 0}}
    h = json.dumps(meta).encode()
    p = tmp_path / "x.ckpt"
    p.write_bytes(struct.pack("<I", len(h)) + h + b"PAYLOAD")
    assert load_blob(p) == (meta, b"PAYLOAD")


def test_first_diff_edges():
    a = bytes(5000)
    assert first_diff(a, a) is None
    assert first_diff(a, a + b"x") is None  # strict prefix
    assert first_diff(a, b"\x01" + a[1:]) == 0
    assert first_diff(a, a[:-1] + b"\x01") == 4999


def test_kv_heads_shape_rule():
    assert kv_heads({"shape": [1, 4, 7434, 128]}) == 4
    assert kv_heads({"shape": [1, 4, 7434, 32]}) == 4
    assert kv_heads({"shape": [7434]}) == 1  # non-KV tensor: one stream


def test_layout_framing_from_file_not_reserialization(tmp_path):
    """framing_bytes must be the file's actual header size, not a
    re-serialized guess (json.dumps spacing need not match the writer)."""
    meta = {"token_count": 3, "fingerprint": {"kv_bits": 0}}
    h = json.dumps(meta, separators=(",", ":")).encode()  # compact, unlike json.dumps
    payload = st_payload({"k": b"xyz"})
    p = tmp_path / "x.ckpt"
    framing = 4 + len(h)
    p.write_bytes(struct.pack("<I", len(h)) + h + payload)
    m, pl = load_blob(p)
    tensors, _ = parse_safetensors(pl)
    lay = layout(m, tensors, len(pl), framing)
    assert lay["framing_bytes"] == framing
    # the compact header's length must differ from a spaced re-serialization
    # (that is the point: never guess the writer's serialization)
    assert framing != 4 + len(json.dumps(m))


def st_payload_shapes(specs: dict[str, tuple[bytes, list[int]]]) -> bytes:
    off, header = 0, {}
    for name, (b, shape) in specs.items():
        header[name] = {"dtype": "U8", "shape": shape, "data_offsets": [off, off + len(b)]}
        off += len(b)
    hj = json.dumps(header).encode()
    return struct.pack("<Q", len(hj)) + hj + b"".join(b for b, _ in specs.values())


def write_ckpt(path: Path, specs: dict[str, tuple[bytes, list[int]]], token_count: int) -> None:
    meta = {"token_count": token_count, "fingerprint": {"kv_bits": 0}}
    h = json.dumps(meta).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<I", len(h)) + h + st_payload_shapes(specs))


def test_turn_blobs_validates_grow_manifest(tmp_path):
    """VALUE: turn_blobs is the entry point of every analysis script (forensics,
    chunk, evict); a malformed capture must fail loudly at load time, and the
    per-turn blob order must be the capture order. Catches: accepting a
    non-grow manifest, silent mis-ordering, and the turn-with-no-new-blob
    shape that a dropped publish produces. Asserts observable returns/errors,
    not internals, so a behavior-preserving refactor stays green."""
    run = tmp_path / "grow"
    names = ["a1.ckpt", "a2.ckpt", "a3.ckpt"]
    for i in range(3):
        snap = run / f"snap{i + 1}"
        snap.mkdir(parents=True)
        for n in names[: i + 1]:
            (snap / n).write_bytes(b"blob")
    manifest = {
        "mode": "grow",
        "turns": [
            {
                "turn": i + 1,
                "snapshot": f"snap{i + 1}",
                "blobs": [{"name": n} for n in names[: i + 1]],
            }
            for i in range(3)
        ],
    }
    (run / "manifest.json").write_text(json.dumps(manifest))
    got = turn_blobs(run)
    assert [t for t, _ in got] == [1, 2, 3]
    assert [p.name for _, p in got] == names

    (run / "manifest.json").write_text(json.dumps({**manifest, "mode": "single"}))
    with pytest.raises(SystemExit, match="not a grow run"):
        turn_blobs(run)

    bad_turns = manifest["turns"][:1] + [
        {"turn": 2, "snapshot": "snap2", "blobs": [{"name": "a2.ckpt"}, {"name": "b2.ckpt"}]}
    ]
    (run / "manifest.json").write_text(json.dumps({**manifest, "turns": bad_turns}))
    with pytest.raises(SystemExit, match="expected exactly 1 new blob"):
        turn_blobs(run)


def test_single_blob_requires_exactly_one(tmp_path):
    """VALUE: single_blob feeds P2/P3; two blobs in snapshots/single (a
    republish that landed mid-snapshot) must abort the probe instead of
    silently comparing the wrong file."""
    single = tmp_path / "run" / "snapshots" / "single"
    single.mkdir(parents=True)
    (single / "x.ckpt").write_bytes(b"1")
    assert single_blob(tmp_path / "run") == single / "x.ckpt"
    (single / "y.ckpt").write_bytes(b"2")
    with pytest.raises(SystemExit, match="exactly 1 single-run blob"):
        single_blob(tmp_path / "run")


def test_compare_pair_head_view_vs_flat_shift(tmp_path):
    """VALUE: compare_pair's per-head slicing (b_base + h*hsz_b) carries the
    P1 claim ("every head byte-exact across turns; the flat divergence is the
    shift"). An offset regression here flips the design decision's headline
    evidence while committed tables stay plausible. Built from synthetic
    framed blobs: B grows each of A's 2 KV heads by 8 bytes — per-head view
    must be exact while the flat view diverges at the head-1 shift."""
    head0 = bytes((7 * i) % 256 for i in range(32))
    head1 = bytes((50 + i) % 256 for i in range(32))
    a = tmp_path / "a.ckpt"
    b = tmp_path / "b.ckpt"
    write_ckpt(
        a,
        {"fl": (b"C" * 16, [16]), "kv": (head0 + head1, [1, 2, 8, 4])},
        token_count=9,
    )
    write_ckpt(
        b,
        {
            "fl": (b"C" * 16 + b"D" * 4, [20]),
            "kv": (head0 + b"\xaa" * 8 + head1 + b"\xbb" * 8, [1, 2, 10, 4]),
        },
        token_count=11,
    )
    r = compare_pair("t", a, b, block=16)
    by_name = {row["tensor"]: row for row in r["tensors"]}
    kv = by_name["kv"]
    assert kv["n_heads"] == 2 and kv["heads_equal"] == 2
    assert all(h["first_diff_byte"] is None for h in kv["heads"])
    # the shift a whole-file chunker would pay: A's head-1 block no longer
    # sits at A's flat offset in B
    assert kv["flat_leading_equal"] is False
    assert kv["flat_first_diff_byte"] == 32
    # a non-KV (1-stream) tensor has no shift: both views exact
    fl = by_name["fl"]
    assert fl["n_heads"] == 1 and fl["heads_equal"] == 1
    assert fl["flat_leading_equal"] is True and fl["flat_first_diff_byte"] is None


def test_block_containment_direction_and_edges():
    """VALUE: block_containment produces the committed containment columns;
    swapping the a/b direction or mis-counting blocks would silently invert
    the reported dedup fractions on the next table regeneration."""
    blk = b"ab" * 8
    ext = blk + b"cd" * 8
    assert block_containment(blk, ext, 16) == (1.0, 0.5)
    assert block_containment(ext, blk, 16) == (0.5, 1.0)
    assert block_containment(blk, blk, 16) == (1.0, 1.0)
    assert block_containment(b"xy" * 8, blk, 16) == (0.0, 0.0)


def test_compare_pair_tensor_set_mismatch_is_a_labeled_error(tmp_path):
    """A malformed capture must produce the labeled error row, not a
    KeyError stack trace."""
    from blob_forensics import compare_pair

    def blob(name_sizes: dict[str, bytes]) -> Path:
        meta = {"token_count": 4, "fingerprint": {"kv_bits": 0}}
        h = json.dumps(meta).encode()
        payload = st_payload(name_sizes)
        p = tmp_path / f"{len(name_sizes)}-{sum(map(len, name_sizes.values()))}.ckpt"
        p.write_bytes(struct.pack("<I", len(h)) + h + payload)
        return p

    a = blob({"k": b"12345678"})
    b = blob({"k": b"12345678", "v": b"87654321"})
    r = compare_pair("mismatch", a, b, 4096)
    assert r["error"] == "tensor sets differ"
    assert r["a"] == ["k"] and sorted(r["b"]) == ["k", "v"]
