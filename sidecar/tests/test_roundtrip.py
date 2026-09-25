"""Round-trip benchmark tests (T2). Synthetic by default; MLXCACHE_BENCH_REAL=1 for real mlx-lm."""

from mlxcache_sidecar.roundtrip import run_roundtrip


def test_roundtrip_identity() -> None:
    result = run_roundtrip(50_000)
    assert result.logits_identical
    assert result.bytes_per_token() == 1024.0 + 4 / 50_000 or result.bytes_per_token() > 1024


def test_roundtrip_scaling() -> None:
    small = run_roundtrip(1_000)
    big = run_roundtrip(10_000)
    assert big.bytes_total > small.bytes_total
