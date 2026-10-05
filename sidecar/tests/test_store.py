"""L1 Store conformance (B1.1) — the store-semantics tier.

Everything the format tier (test_conformance.py) could not test without
an index: granularity matching, cross-engine isolation, budget eviction,
corrupt-on-open reclamation, concurrent writers, restart persistence —
plus the cross-language KEY goldens (the Rust twins live in
crates/mlxcache-daemon/src/http.rs tests; both sides pin the same
constants for the same fixtures).
"""

import struct
import threading
import time
from multiprocessing import Process

import pytest
from mlxcache_sidecar import blob
from mlxcache_store import (
    CorruptCache,
    Fingerprint,
    Granularity,
    Refused,
    Store,
    Unavailable,
    blob_key,
)

# --- fixtures -------------------------------------------------------------
# The B0.2 key-golden fixture (see blob_key_engine_namespacing_and_legacy_
# stability in http.rs — same fields, same constants, both languages).
FP_GOLDEN = Fingerprint("golden-model", "golden-tok", "f16", 1)
GOLDEN_KEY_8 = "2786690f2948dec17c5498a6267fc101"
GOLDEN_KEY_0 = "ca1873d2c14a2fe1f6be3cc0354ea7e9"
GOLDEN_KEY_LLAMA = "0f8b99131596e10dfed791048839d6e5"

FP = Fingerprint("model-a", "tok-a", "f16", 1)
FP_B = Fingerprint("model-b", "tok-a", "f16", 1)  # different model


def test_cross_language_key_goldens():
    tokens = list(range(1, 9))
    assert f"{blob_key(FP_GOLDEN, None, tokens):032x}" == GOLDEN_KEY_8
    assert f"{blob_key(FP_GOLDEN, 'mlx-lm', tokens):032x}" == GOLDEN_KEY_8
    assert f"{blob_key(FP_GOLDEN, None, []):032x}" == GOLDEN_KEY_0
    assert f"{blob_key(FP_GOLDEN, 'llama-cpp', tokens):032x}" == GOLDEN_KEY_LLAMA


# --- store semantics --------------------------------------------------------


def test_put_lookup_fetch_roundtrip(tmp_path):
    s = Store(tmp_path, byte_budget=0)
    ref = s.put("mlx-lm", FP, [1, 2, 3, 4], b"kv")
    m, r = s.lookup("mlx-lm", FP, [1, 2, 3, 4, 5])
    assert (m, r) == (4, ref), "matched_len is len(entry.tokens), never capped"
    assert s.fetch(r) == b"kv"


def test_lookup_isolation(tmp_path):
    s = Store(tmp_path, byte_budget=0)
    s.put("mlx-lm", FP, [1, 2, 3], b"kv")
    # Cross-engine: structurally no match (§4).
    assert s.lookup("llama-cpp", FP, [1, 2, 3, 4]) == (0, None)
    # Fingerprint mismatch: no match.
    assert s.lookup("mlx-lm", FP_B, [1, 2, 3, 4]) == (0, None)


def test_whole_context_granularity_and_fallback(tmp_path):
    s = Store(tmp_path, byte_budget=0)
    s.put("llama-cpp", FP, [1, 2, 3, 4], b"whole", Granularity.WHOLE_CONTEXT)
    # Exact full stream matches.
    m, r = s.lookup("llama-cpp", FP, [1, 2, 3, 4])
    assert m == 4 and r is not None
    assert r is not None and s.fetch(r) == b"whole"
    # An EXTENDING request does NOT match a WHOLE_CONTEXT entry (no partial).
    assert s.lookup("llama-cpp", FP, [1, 2, 3, 4, 5]) == (0, None)
    # Granularity fallback: an ANY_PREFIX entry below is found instead (§3.3).
    s.put("llama-cpp", FP, [1, 2], b"any", Granularity.ANY_PREFIX)
    m, r = s.lookup("llama-cpp", FP, [1, 2, 3, 4, 5])
    assert m == 2 and s.fetch(r) == b"any"


def test_end_anchored_divergence_match(tmp_path):
    s = Store(tmp_path, byte_budget=0)
    s.put("mlx-lm", FP, [1, 2, 3, 4], b"kv")
    # Request agrees for 3 tokens then diverges at the 4th (the entry's
    # last token) — still a match covering 3 positions (§3.3).
    m, r = s.lookup("mlx-lm", FP, [1, 2, 3, 9, 9])
    assert m == 4, "end-anchored: matched_len stays len(entry.tokens)"
    assert s.fetch(r) == b"kv"


def test_fetch_verifies_digest_every_read(tmp_path):
    s = Store(tmp_path, byte_budget=0)
    ref = s.put("mlx-lm", FP, [1, 2, 3], b"kv")
    path = tmp_path / ref
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 0xFF
    path.write_bytes(bytes(raw))
    with pytest.raises(CorruptCache):
        s.fetch(ref)


def test_legacy_v1_fetch_unverified(tmp_path):
    meta = blob.CheckpointMeta(
        fingerprint=blob.Fingerprint(
            model_id=FP.model_id,
            tokenizer_hash=FP.tokenizer_hash,
            kv_dtype=FP.kv_dtype,
            kv_layout_version=1,
        ),
        token_count=3,
        tokens=[1, 2, 3],
    )  # v1, no digest
    (tmp_path / "legacy.ckpt").write_bytes(blob.encode(meta, b"old-kv"))
    # Default posture: an unverifiable v1 blob is never SERVED by lookup
    # (a planted digest-less blob cannot poison lookups); it stays on
    # disk and remains fetchable by explicit ref for migration.
    s = Store(tmp_path, byte_budget=0)
    assert s.lookup("mlx-lm", FP, [1, 2, 3, 4]) == (0, None)
    assert (tmp_path / "legacy.ckpt").exists()
    assert s.fetch("legacy.ckpt") == b"old-kv"
    # Opt-in (known-good migration directory): indexed and served.
    s2 = Store(tmp_path, byte_budget=0, trust_legacy=True)
    m, r = s2.lookup("mlx-lm", FP, [1, 2, 3, 4])
    assert (m, r) == (3, "legacy.ckpt")
    assert s2.fetch(r) == b"old-kv"


def test_sub_two_token_put_refused(tmp_path):
    s = Store(tmp_path, byte_budget=0)
    from mlxcache_store import Refused

    with pytest.raises(Refused):
        s.put("mlx-lm", FP, [1], b"kv")


def test_supersede_last_writer_wins(tmp_path):
    s = Store(tmp_path, byte_budget=0)
    r1 = s.put("mlx-lm", FP, [1, 2, 3], b"old")
    r2 = s.put("mlx-lm", FP, [1, 2, 3], b"new")
    assert r1 != r2
    assert not (tmp_path / r1).exists(), "superseded generation unlinked"
    m, r = s.lookup("mlx-lm", FP, [1, 2, 3, 4])
    assert r == r2 and s.fetch(r) == b"new"


def test_invalidate_ref_and_subtree(tmp_path):
    s = Store(tmp_path, byte_budget=0)
    base = s.put("mlx-lm", FP, [1, 2], b"base")
    child = s.put("mlx-lm", FP, [1, 2, 3], b"child")
    grand = s.put("mlx-lm", FP, [1, 2, 3, 4], b"grand")
    other = s.put("mlx-lm", FP, [7, 7], b"other")
    n = s.invalidate(prefix=[1, 2])
    assert n >= 2, "exact + descendants"
    assert not (tmp_path / base).exists() and not (tmp_path / child).exists()
    assert not (tmp_path / grand).exists()
    assert (tmp_path / other).exists()


def test_budget_eviction_directory_basis(tmp_path):
    # anchor_window_s=-1.0: no entry can satisfy (now - last_used) <= -1,
    # so anchoring is off deterministically — no reliance on clock ticks.
    s = Store(tmp_path, byte_budget=2048, anchor_window_s=-1.0)
    for i in range(8):
        s.put("mlx-lm", FP, [10 + i, 20 + i, 30 + i], bytes(700))  # ~750B each
    total = sum(p.stat().st_size for p in tmp_path.glob("*.ckpt"))
    assert total <= 2048 + 800, f"directory basis: {total} bytes over budget"
    # The most recent puts survive; everything is still fetchable-or-gone.
    for p in tmp_path.glob("*.ckpt"):
        assert s.fetch(p.name) is not None


def test_corrupt_on_open_reclaims(tmp_path):
    s = Store(tmp_path, byte_budget=0)
    good = s.put("mlx-lm", FP, [1, 2, 3], b"kv")
    # Foreign version => deterministically corrupt.
    fp_meta = blob.Fingerprint(
        model_id=FP.model_id,
        tokenizer_hash=FP.tokenizer_hash,
        kv_dtype=FP.kv_dtype,
        kv_layout_version=1,
    )
    v3 = blob.encode(
        blob.CheckpointMeta(fingerprint=fp_meta, token_count=2, tokens=[9, 9], format_version=3),
        b"kv",
    )
    (tmp_path / "v3.ckpt").write_bytes(v3)
    # Tampered digest.
    tampered = bytearray((tmp_path / good).read_bytes())
    tampered[-1] ^= 0xFF
    (tmp_path / "tampered.ckpt").write_bytes(bytes(tampered))
    s2 = Store(tmp_path, byte_budget=0)
    assert not (tmp_path / "v3.ckpt").exists()
    assert not (tmp_path / "tampered.ckpt").exists()
    m, r = s2.lookup("mlx-lm", FP, [1, 2, 3, 4])
    assert (m, r) == (3, good), "healthy sibling still served"


def test_restart_persistence(tmp_path):
    s = Store(tmp_path, byte_budget=0)
    ref = s.put("mlx-lm", FP, [1, 2, 3, 4], b"kv")
    s2 = Store(tmp_path, byte_budget=0)
    m, r = s2.lookup("mlx-lm", FP, [1, 2, 3, 4])
    assert (m, r) == (4, ref)
    assert s2.fetch(r) == b"kv"


def test_concurrent_writers_share_the_store(tmp_path):
    a = Store(tmp_path, byte_budget=0)
    b = Store(tmp_path, byte_budget=0)
    errs = []

    def writer(store, tag):
        try:
            for i in range(10):
                store.put("mlx-lm", FP, [100 + i, 200 + i], f"{tag}-{i}".encode())
        except Exception as exc:  # noqa: BLE001
            errs.append(exc)

    ta = threading.Thread(target=writer, args=(a, "a"))
    tb = threading.Thread(target=writer, args=(b, "b"))
    ta.start()
    tb.start()
    ta.join()
    tb.join()
    assert not errs, errs
    # Every file present is digest-valid (fetch proves it).
    for p in tmp_path.glob("*.ckpt"):
        a.fetch(p.name)
    # And the union of both writers' generations is visible to a fresh open.
    fresh = Store(tmp_path, byte_budget=0)
    names = {p.name for p in tmp_path.glob("*.ckpt")}
    assert len(names) >= 10
    for n in names:
        fresh.fetch(n)


# --- review-finding regressions (B1.1 gauntlet) ----------------------------


def _write_raw(tmp_path, name: str, data: bytes) -> None:
    (tmp_path / name).write_bytes(data)


def test_malformed_header_shapes_reclaimed_not_bricked(tmp_path):
    """Every deterministically malformed header is reclaimed at open; a
    healthy sibling survives; open never raises (review CRITICAL 1/2)."""
    import struct

    good = Store(tmp_path, byte_budget=0).put("mlx-lm", FP, [1, 2, 3], b"kv")
    _write_raw(tmp_path, "garbage.ckpt", b"\x00\x01\x02")  # < 4 bytes
    _write_raw(
        tmp_path, "lenpast.ckpt", struct.pack("<I", 999) + b"{}" + b"x"
    )  # header length exceeds blob
    _write_raw(
        tmp_path, "jsonlist.ckpt", struct.pack("<I", 7) + b"[1,2,3]" + b"x"
    )  # non-object JSON header
    _write_raw(
        tmp_path,
        "fpint.ckpt",
        struct.pack("<I", 24) + b'{"fingerprint":5,"token_count":1}',
    )  # fingerprint not an object
    _write_raw(
        tmp_path,
        "notokens.ckpt",
        struct.pack("<I", 40) + b'{"fingerprint":{"model_id":"m"},"format_version":1}',
    )  # missing token_count
    # v2 header without a digest (CRITICAL 2): rejected at decode, reclaimed.
    meta = blob.CheckpointMeta(
        fingerprint=blob.Fingerprint(
            model_id=FP.model_id,
            tokenizer_hash=FP.tokenizer_hash,
            kv_dtype=FP.kv_dtype,
            kv_layout_version=1,
        ),
        token_count=2,
        tokens=[5, 6],
        format_version=blob.FORMAT_VERSION_D1,  # digest omitted → invalid
    )
    _write_raw(tmp_path, "v2nodigest.ckpt", blob.encode(meta, b"kv"))

    s2 = Store(tmp_path, byte_budget=0)  # must not raise
    for junk in (
        "garbage.ckpt",
        "lenpast.ckpt",
        "jsonlist.ckpt",
        "fpint.ckpt",
        "notokens.ckpt",
        "v2nodigest.ckpt",
    ):
        assert not (tmp_path / junk).exists(), f"{junk} not reclaimed"
    assert (tmp_path / good).exists()
    m, r = s2.lookup("mlx-lm", FP, [1, 2, 3, 4])
    assert (m, r) == (3, good)


def test_cross_engine_same_tokens_coexist(tmp_path):
    """One token path carries entries from TWO engines (multi-entry nodes);
    invalidate-by-ref reaches exactly one (review INFO 4)."""
    s = Store(tmp_path, byte_budget=0)
    a = s.put("mlx-lm", FP, [1, 2, 3], b"mlx-kv")
    b = s.put("llama-cpp", FP, [1, 2, 3], b"llama-kv")
    assert s.lookup("mlx-lm", FP, [1, 2, 3, 4]) == (3, a)
    assert s.lookup("llama-cpp", FP, [1, 2, 3, 4]) == (3, b)
    assert s.fetch(a) == b"mlx-kv" and s.fetch(b) == b"llama-kv"
    assert s.invalidate(ref=b) == 1
    assert s.lookup("llama-cpp", FP, [1, 2, 3, 4]) == (0, None)
    assert s.lookup("mlx-lm", FP, [1, 2, 3, 4]) == (3, a), "sibling untouched"


def test_longer_end_anchored_beats_shorter_on_path(tmp_path):
    """The EA scan always runs; a LONGER diverging entry beats a shorter
    extending one (review CRITICAL 3)."""
    s = Store(tmp_path, byte_budget=0)
    short = s.put("mlx-lm", FP, [1, 2], b"short")
    ea = s.put("mlx-lm", FP, [1, 2, 3, 4], b"ea")
    m, r = s.lookup("mlx-lm", FP, [1, 2, 3, 9, 9])
    assert (m, r) == (4, ea), "len-4 diverging beats len-2 extending"
    assert s.fetch(short)  # still present and fetchable


def test_invalidate_subtree_removes_end_anchored_siblings(tmp_path):
    """Invalidate(prefix) also removes equal-length entries that agree for
    len-1 tokens then diverge (review INFO 5)."""
    s = Store(tmp_path, byte_budget=0)
    on_path = s.put("mlx-lm", FP, [1, 2, 3], b"on")
    diverged = s.put("mlx-lm", FP, [1, 2, 9], b"off")  # EA sibling, len 3
    longer = s.put("mlx-lm", FP, [1, 2, 9, 9], b"deep")  # len 4: subtree of [1,2,9]
    n = s.invalidate(prefix=[1, 2, 3])
    assert n == 2, "exact + its end-anchored sibling"
    assert not (tmp_path / on_path).exists()
    assert not (tmp_path / diverged).exists()
    assert (tmp_path / longer).exists(), "len-4 extension of the sibling stays"


def test_directory_named_ckpt_does_not_brick_open(tmp_path):
    (tmp_path / "not-a-blob.ckpt").mkdir()
    s = Store(tmp_path, byte_budget=0)  # must not raise
    ref = s.put("mlx-lm", FP, [1, 2], b"kv")
    assert s.lookup("mlx-lm", FP, [1, 2, 3]) == (2, ref)
    s._enforce_budget()  # dir listing skips the directory too


# --- gauntlet-fix regressions (B1.1 review wave) ---------------------------


def test_noniterable_and_non_u32_tokens_rejected(tmp_path):
    """tokens: null / int / bool / negative / >u32 are deterministic
    header corruption — ValueError at decode, reclaimed at open, never a
    TypeError crash (multi-source CRITICAL)."""
    import json
    import struct

    from mlxcache_sidecar import blob

    fp = {"model_id": "m", "tokenizer_hash": "t", "kv_dtype": "f16", "kv_layout_version": 1}
    cases = {
        "nulltok": None,
        "inttok": 5,
        "booltok": [1, True],
        "negtok": [1, -5],
        "bigtok": [1, 1 << 40],
    }
    for name, tokens in cases.items():
        h = json.dumps(
            {"fingerprint": fp, "token_count": 2, "format_version": 1, "tokens": tokens}
        ).encode()
        (tmp_path / f"{name}.ckpt").write_bytes(struct.pack("<I", len(h)) + h + b"x")
        with pytest.raises(ValueError):
            blob.decode((tmp_path / f"{name}.ckpt").read_bytes())
    Store(tmp_path, byte_budget=0)  # never raises
    assert list(tmp_path.glob("*.ckpt")) == []


def test_fetch_rejects_traversal_refs(tmp_path):
    """fetch(ref) is store-directory-bounded: separators/absolute paths
    are a caller bug, not an arbitrary-file-read primitive."""
    s = Store(tmp_path, byte_budget=0)
    s.put("mlx-lm", FP, [1, 2], b"kv")
    for bad in ("../outside.ckpt", "/etc/passwd", "sub/dir/x.ckpt", "", ".."):
        with pytest.raises(ValueError, match="bare store filename"):
            s.fetch(bad)


def test_fetch_vanished_ref_raises_unavailable(tmp_path):
    s = Store(tmp_path, byte_budget=0)
    ref = s.put("mlx-lm", FP, [1, 2], b"kv")
    (tmp_path / ref).unlink()  # evicted by another process between lookup/fetch
    with pytest.raises(Unavailable):
        s.fetch(ref)


def test_fetch_header_and_version_kinds(tmp_path):
    s = Store(tmp_path, byte_budget=0)
    ref = s.put("mlx-lm", FP, [1, 2], b"kv")
    # Corrupt the HEADER region (JSON) → kind "header".
    raw = bytearray((tmp_path / ref).read_bytes())
    raw[6] ^= 0x20
    (tmp_path / "hdr.ckpt").write_bytes(
        struct.pack("<I", 40)
        + b'{"fingerprint":5,"token_count":1,"format_version":2,'
        + b'"payload_sha256":"'
        + b"0" * 64
        + b'","tokens":[1,2]}'
    )
    with pytest.raises(CorruptCache) as ei:
        s.fetch("hdr.ckpt")
    assert ei.value.kind == "header"
    # A v3 header fetched by explicit ref → kind "version" (the
    # fingerprint must be VALID so the flow reaches the version gate).
    v3h = (
        b'{"fingerprint":{"model_id":"m","tokenizer_hash":"t",'
        b'"kv_dtype":"f16","kv_layout_version":1},"token_count":2,'
        b'"format_version":3,"tokens":[1,2]}'
    )
    (tmp_path / "v3.ckpt").write_bytes(struct.pack("<I", len(v3h)) + v3h + b"x")
    with pytest.raises(CorruptCache) as ei:
        s.fetch("v3.ckpt")
    assert ei.value.kind == "version"


def test_invalidate_ref_unlinks_and_stays_gone(tmp_path):
    """invalidate(ref) removes the FILE too — no immortal corpse pinning
    the budget and resurrecting after restart (adversarial F2)."""
    s = Store(tmp_path, byte_budget=0)
    ref = s.put("mlx-lm", FP, [1, 2, 3], b"kv")
    assert s.invalidate(ref=ref) == 1
    assert not (tmp_path / ref).exists()
    s2 = Store(tmp_path, byte_budget=0)  # restart: not resurrected
    assert s2.lookup("mlx-lm", FP, [1, 2, 3, 4]) == (0, None)
    assert s2.invalidate(ref="nonexistent") == 0


def test_invalidate_diverged_prefix_purges_end_anchored_sibling(tmp_path):
    """invalidate(prefix=[1,2,9]) when only [1,2,3] exists: the walk at
    [1,2,9] misses, but the equal-length divergent sibling [1,2,3] is
    exactly what lookup([1,2,9,7]) would serve — it must be purged
    (adversarial F3)."""
    s = Store(tmp_path, byte_budget=0)
    ref = s.put("mlx-lm", FP, [1, 2, 3], b"kv")
    assert s.lookup("mlx-lm", FP, [1, 2, 9, 7]) == (3, ref)  # EA serve
    assert s.invalidate(prefix=[1, 2, 9]) == 1
    assert s.lookup("mlx-lm", FP, [1, 2, 9, 7]) == (0, None)
    assert not (tmp_path / ref).exists()


def test_stale_generation_loses_after_reopen(tmp_path):
    """Two same-key files (crash between publish and supersede): the NEWER
    generation must win after reopen (§3.3), not the first-indexed."""
    s = Store(tmp_path, byte_budget=0)
    old_ref = s.put("mlx-lm", FP, [1, 2, 3], b"old")
    # Simulate a stale twin: copy the file under a LARGER generation name.
    key = old_ref.split("-")[0]
    stale = f"{key}-ffffffffffffffff.ckpt"
    (tmp_path / stale).write_bytes((tmp_path / old_ref).read_bytes())
    s2 = Store(tmp_path, byte_budget=0)
    m, r = s2.lookup("mlx-lm", FP, [1, 2, 3, 4])
    assert (m, r) == (3, stale), "newest generation survives the dedupe"
    assert not (tmp_path / old_ref).exists(), "older twin reclaimed"


def test_mislabeled_filename_reclaimed(tmp_path):
    """A v2 blob whose filename does not bind to the fold of its own
    header cannot impersonate another entry's key (name-binding check)."""
    s = Store(tmp_path, byte_budget=0)
    ref = s.put("mlx-lm", FP, [1, 2, 3], b"kv")
    raw = (tmp_path / ref).read_bytes()
    mislabeled = "deadbeef" * 4 + "-0000000000000001.ckpt"
    (tmp_path / mislabeled).write_bytes(raw)
    s2 = Store(tmp_path, byte_budget=0)
    assert not (tmp_path / mislabeled).exists()
    assert s2.lookup("mlx-lm", FP, [1, 2, 3, 4]) == (3, ref)


def test_unreadable_file_skipped_not_fatal(tmp_path):
    """One permission-denied file must not brick every future open; the
    healthy sibling stays served (adversarial F7)."""
    s = Store(tmp_path, byte_budget=0)
    ref = s.put("mlx-lm", FP, [1, 2, 3], b"kv")
    locked = tmp_path / "locked.ckpt"
    locked.write_bytes((tmp_path / ref).read_bytes())
    locked.chmod(0)
    try:
        s2 = Store(tmp_path, byte_budget=0)  # must not raise
        assert s2.lookup("mlx-lm", FP, [1, 2, 3, 4]) == (3, ref)
        assert locked.exists(), "unreadable ≠ corrupt: left in place"
    finally:
        locked.chmod(0o644)


def test_unknown_granularity_rejected(tmp_path):
    """granularity=2 on disk is deterministic header corruption (never
    silently behave-as-ANY_PREFIX); put() rejects it as a caller bug."""
    import json

    fp_json = {
        "model_id": FP.model_id,
        "tokenizer_hash": FP.tokenizer_hash,
        "kv_dtype": FP.kv_dtype,
        "kv_layout_version": 1,
    }
    h = json.dumps(
        {
            "fingerprint": fp_json,
            "token_count": 2,
            "format_version": 2,
            "payload_sha256": "0" * 64,
            "granularity": 2,
            "tokens": [1, 2],
        }
    ).encode()
    (tmp_path / "gran2.ckpt").write_bytes(struct.pack("<I", len(h)) + h + b"x")
    s = Store(tmp_path, byte_budget=0)
    assert not (tmp_path / "gran2.ckpt").exists()
    with pytest.raises(ValueError, match="granularity"):
        s.put("mlx-lm", FP, [1, 2], b"kv", granularity=2)  # type: ignore[arg-type]


def test_tmp_files_swept_by_age(tmp_path):
    """Crashed-writer tmp files are swept once older than the threshold;
    fresh ones (possibly a live writer's) are left alone."""
    import os

    s = Store(tmp_path, byte_budget=0)
    ref = s.put("mlx-lm", FP, [1, 2], b"kv")
    stale = tmp_path / f"{ref}.tmpdeadbeef"
    fresh = tmp_path / f"{ref}.tmpfeedface"
    stale.write_bytes(b"x")
    fresh.write_bytes(b"x")
    old = time.time() - 48 * 3600
    os.utime(stale, (old, old))
    Store(tmp_path, byte_budget=0)
    assert not stale.exists()
    assert fresh.exists()


def _mp_writer(store_dir, tag):
    # Module-level: macOS spawn pickles the target.
    st = Store(store_dir, byte_budget=0)
    for i in range(5):
        st.put("mlx-lm", FP, [1, 2], f"{tag}-{i}".encode())  # same key
        st.put("mlx-lm", FP, [1, 3, tag], f"uniq-{tag}-{i}".encode())


def test_multiprocess_writers_share_directory(tmp_path):
    """The §8 process boundary: two real PROCESSES (not threads) putting
    overlapping + identical keys; every survivor is digest-valid, and the
    same-key union keeps exactly the newer generation per key."""
    ps = [Process(target=_mp_writer, args=(str(tmp_path), t)) for t in (0, 1)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    fresh = Store(tmp_path, byte_budget=0)
    survivors = list(tmp_path.glob("*.ckpt"))
    # 3 unique keys survive: the contended [1,2] (newest generation wins)
    # plus each writer's private [1,3,tag] key (self-superseded to one).
    assert len(survivors) == 3
    for f in survivors:
        fresh.fetch(f.name)  # every survivor digest-valid


# --- red-team-fix regressions (B1.1 re-review wave) -------------------------


def test_fingerprint_field_types_rejected(tmp_path):
    """A v2 blob with model_id: 5 (dataclass accepts it) must die at
    decode — not explode later in the fold's .encode() and brick opens."""
    import json

    h = json.dumps(
        {
            "fingerprint": {
                "model_id": 5,
                "tokenizer_hash": "t",
                "kv_dtype": "f16",
                "kv_layout_version": 1,
            },
            "token_count": 2,
            "format_version": 2,
            "payload_sha256": "0" * 64,
            "tokens": [1, 2],
        }
    ).encode()
    (tmp_path / "fpint.ckpt").write_bytes(struct.pack("<I", len(h)) + h + b"x")
    with pytest.raises(ValueError, match="fingerprint field types"):
        blob.decode((tmp_path / "fpint.ckpt").read_bytes())
    Store(tmp_path, byte_budget=0)  # reclaims, never raises
    assert not (tmp_path / "fpint.ckpt").exists()


def test_v2_beats_planted_v1_in_dedupe(tmp_path):
    """Under trust_legacy, a planted v1 blob with a lexically-max
    generation name must NOT delete the genuine v2 twin; the verified v2
    wins and the unverifiable v1 is reclaimed."""
    s = Store(tmp_path, byte_budget=0)
    v2_ref = s.put("mlx-lm", FP, [1, 2, 3], b"genuine")
    key = v2_ref.split("-")[0]
    meta = blob.CheckpointMeta(
        fingerprint=blob.Fingerprint(
            model_id=FP.model_id,
            tokenizer_hash=FP.tokenizer_hash,
            kv_dtype=FP.kv_dtype,
            kv_layout_version=1,
        ),
        token_count=3,
        tokens=[1, 2, 3],
    )
    planted = f"{key}-ffffffffffffffff.ckpt"
    (tmp_path / planted).write_bytes(blob.encode(meta, b"stale-planted"))
    s2 = Store(tmp_path, byte_budget=0, trust_legacy=True)
    m, r = s2.lookup("mlx-lm", FP, [1, 2, 3, 4])
    assert (m, r) == (3, v2_ref), "verified v2 survives; planted v1 loses"
    assert not (tmp_path / planted).exists()
    assert s2.fetch(r) == b"genuine"


def test_anchors_are_live_not_sticky(tmp_path):
    """After the extending entry is removed, the chain head must become
    evictable again — anchoring reflects LIVE descendants (§7), not the
    process's insert history."""
    s = Store(tmp_path, byte_budget=0, anchor_window_s=-1.0)
    head = s.put("mlx-lm", FP, [1, 2], b"head")
    ext = s.put("mlx-lm", FP, [1, 2, 3], b"ext")
    assert s.invalidate(ref=ext) == 1
    cands = {e.name: anchored for e, anchored in s.index.candidates_for_eviction(time.monotonic())}
    assert cands[head] is False, "no live descendant → not anchored"


def test_fetch_digest_failure_self_heals(tmp_path):
    """A file tampered AFTER open: fetch raises CorruptCache(digest) AND
    retires the entry — the next lookup misses instead of re-serving."""
    s = Store(tmp_path, byte_budget=0)
    ref = s.put("mlx-lm", FP, [1, 2, 3], b"kv")
    raw = bytearray((tmp_path / ref).read_bytes())
    raw[-1] ^= 0xFF
    (tmp_path / ref).write_bytes(bytes(raw))
    with pytest.raises(CorruptCache):
        s.fetch(ref)
    assert s.lookup("mlx-lm", FP, [1, 2, 3, 4]) == (0, None)
    assert not (tmp_path / ref).exists()


def test_put_input_validation_mirrors_decode(tmp_path):
    """put refuses what the next open would reclaim: non-u32 tokens
    (fold-aliasing hazard) and non-str engine ids never earn a CacheRef."""
    s = Store(tmp_path, byte_budget=0)
    for bad_tokens in ([1, -5], [1, 1 << 40], [1, True], [1, 2.5]):
        with pytest.raises(Refused):
            s.put("mlx-lm", FP, bad_tokens, b"kv")
    with pytest.raises(ValueError, match="engine_id"):
        s.put(5, FP, [1, 2], b"kv")  # type: ignore[arg-type]
