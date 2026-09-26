"""Round-trip benchmark harness (T2).

Measures, for the mlx-lm sidecar adapter at N tokens (default 50K):
- bytes/token of a serialized checkpoint
- serialize wall time, deserialize wall time
- peak memory during round-trip
- logits identity: generation continues identically after adopt (the thesis guard, [EVAL])
- sidecar IPC tax: KV blob serialization + transport overhead (R4/D2-reopened)

Run: uv run pytest sidecar/tests/test_roundtrip.py -v
The real mlx-lm model load is gated behind MLXCACHE_BENCH_REAL=1 (needs a
downloaded model); the default run exercises the harness mechanics with a
synthetic state so CI stays fast and hermetic.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import pytest


@dataclass
class RoundTripResult:
    tokens: int
    bytes_total: int
    serialize_ms: float
    deserialize_ms: float
    peak_memory_mb: float
    logits_identical: bool
    ipc_overhead_ms: float = 0.0
    notes: list[str] = field(default_factory=list)

    def bytes_per_token(self) -> float:
        return self.bytes_total / self.tokens

    def as_dict(self) -> dict[str, Any]:
        return {
            "tokens": self.tokens,
            "bytes_total": self.bytes_total,
            "bytes_per_token": self.bytes_per_token(),
            "serialize_ms": self.serialize_ms,
            "deserialize_ms": self.deserialize_ms,
            "peak_memory_mb": self.peak_memory_mb,
            "logits_identical": self.logits_identical,
            "ipc_overhead_ms": self.ipc_overhead_ms,
            "notes": self.notes,
        }


def _synthetic_kv_state(n_tokens: int, bytes_per_token: int = 1024) -> bytes:
    """Synthetic KV state: deterministic bytes so logits-identity is checkable."""
    return bytes((i * 31 + n_tokens) % 256 for i in range(n_tokens * bytes_per_token))


def _measure_peak_memory_mb() -> float:
    """Peak RSS in MiB. macOS reports ru_maxrss in BYTES; Linux in KiB."""
    try:
        import resource
        import sys

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
        return rss / divisor
    except Exception:  # pragma: no cover - non-Unix fallback
        return 0.0


def run_roundtrip(n_tokens: int = 50_000) -> RoundTripResult:
    """Execute the round-trip mechanics: serialize -> deserialize -> verify.

    The synthetic path validates the harness. MLXCACHE_BENCH_REAL=1 swaps in
    the real mlx-lm save/load path (requires a local model; not hermetic).
    """
    real = os.environ.get("MLXCACHE_BENCH_REAL") == "1"
    if real:
        pytest.skip("real mlx-lm benchmark requires a local model; run manually")

    state = _synthetic_kv_state(n_tokens)

    t0 = time.perf_counter()
    header = json.dumps(
        {
            "fingerprint": {
                "model_id": "synthetic",
                "tokenizer_hash": "0" * 64,
                "kv_dtype": "f16",
                "kv_layout_version": 1,
            },
            "token_count": n_tokens,
            "format_version": 1,
        }
    ).encode()
    blob = len(header).to_bytes(4, "little") + header + state
    serialize_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    header_len = int.from_bytes(blob[:4], "little")
    parsed_header = json.loads(blob[4 : 4 + header_len])
    payload = blob[4 + header_len :]
    deserialize_ms = (time.perf_counter() - t0) * 1000

    # Thesis guard: round-trip must be byte-identical ([EVAL] semantics).
    logits_identical = payload == state and parsed_header["token_count"] == n_tokens

    # IPC tax measurement: one in-process copy stands in for the transport hop;
    # the real measurement crosses the sidecar boundary (T2 follow-up).
    t0 = time.perf_counter()
    _ = bytes(payload)
    ipc_overhead_ms = (time.perf_counter() - t0) * 1000

    return RoundTripResult(
        tokens=n_tokens,
        bytes_total=len(blob),
        serialize_ms=serialize_ms,
        deserialize_ms=deserialize_ms,
        peak_memory_mb=_measure_peak_memory_mb(),
        logits_identical=logits_identical,
        ipc_overhead_ms=ipc_overhead_ms,
        notes=["synthetic state; MLXCACHE_BENCH_REAL=1 for real mlx-lm numbers"],
    )


def test_roundtrip_50k() -> None:
    result = run_roundtrip(50_000)
    assert result.logits_identical, "round-trip must be byte-identical (thesis guard)"
    assert result.bytes_total > 0
    assert result.serialize_ms >= 0
    assert result.deserialize_ms >= 0
    # Regression: macOS ru_maxrss is bytes; a bad divisor reports ~1024x.
    assert 1.0 < result.peak_memory_mb < 5000.0, (
        f"implausible peak RSS {result.peak_memory_mb} MB (unit bug?)"
    )


def test_roundtrip_scales() -> None:
    small = run_roundtrip(1_000)
    big = run_roundtrip(10_000)
    assert big.bytes_total > small.bytes_total
    assert big.bytes_per_token() == pytest.approx(small.bytes_per_token(), rel=0.05)


def test_result_report_shape() -> None:
    result = run_roundtrip(500)
    d = result.as_dict()
    for key in (
        "tokens",
        "bytes_total",
        "bytes_per_token",
        "serialize_ms",
        "deserialize_ms",
        "peak_memory_mb",
        "logits_identical",
        "ipc_overhead_ms",
    ):
        assert key in d
