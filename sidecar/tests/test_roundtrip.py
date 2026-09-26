"""Round-trip benchmark tests (T2). Synthetic by default; MLXCACHE_BENCH_REAL=1 for real mlx-lm."""

import pytest
from mlxcache_sidecar.roundtrip import run_roundtrip


def test_roundtrip_identity() -> None:
    result = run_roundtrip(50_000)
    # Thesis guard: the codec round-trips the exact KV bytes.
    assert result.logits_identical
    # KV accounting is exact: the synthetic payload is 1024 bytes/token. Checks
    # the payload, not the header, so it fails if KV sizing regresses.
    assert result.payload_bytes_per_token() == pytest.approx(1024.0)
    # Header adds a positive amount, so total exceeds payload.
    assert result.bytes_total > result.bytes_payload


def test_roundtrip_scaling() -> None:
    small = run_roundtrip(1_000)
    big = run_roundtrip(10_000)
    assert big.bytes_total > small.bytes_total
    # Payload scales linearly: bytes/token is constant across N.
    assert big.payload_bytes_per_token() == pytest.approx(small.payload_bytes_per_token())
