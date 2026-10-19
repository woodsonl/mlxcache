"""Pure-arithmetic guards for scripts/bench_workloads.py.

report()/tpot_ms() are the plumbing harness's whole measurement contract;
until now nothing exercised them without a live daemon, so a regression in
the p95 gating or the token divisor would only surface as quietly wrong
numbers in a green run.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load(name: str):
    path = Path(__file__).resolve().parents[2] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so a later sys.path import resolves to this copy.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


bw = _load("bench_workloads")


def test_tpot_divides_by_real_token_count():
    assert bw.tpot_ms(730.0, 100.0, 64) == pytest.approx(10.0)


def test_tpot_degenerate_token_counts_do_not_divide_by_zero():
    assert bw.tpot_ms(50.0, 10.0, 1) == 40.0
    assert bw.tpot_ms(50.0, 10.0, 0) == 40.0


def test_report_withholds_p95_below_the_sample_floor():
    row = bw.report("cold", [100.0] * 30, [730.0] * 30, [64] * 30)
    assert row["ttft_p95_ms"] is None
    assert row["tpot_p95_ms"] is None
    assert row["tpot_p50_ms"] == pytest.approx(10.0)


def test_report_emits_p95_at_the_floor():
    row = bw.report(
        "warm",
        [float(i) for i in range(100)],
        [float(i) + 9.0 for i in range(100)],
        [10] * 100,
    )
    assert row["ttft_p95_ms"] is not None
    assert row["tpot_p95_ms"] is not None


def test_report_single_token_samples_do_not_fabricate_tpot():
    """A tokens==1 sample has an empty post-TTFT window; it must be excluded
    from the TPOT aggregate rather than divided into a plausible number."""
    row = bw.report("warmup", [100.0, 110.0], [150.0, 160.0], [1, 1])
    assert row["tpot_p50_ms"] is None


def test_report_fails_loudly_on_zero_token_drift():
    with pytest.raises(RuntimeError, match="token counting has drifted"):
        bw.report("bad", [100.0], [150.0], [0])


class _FakePopenProc:
    pid = 99

    def poll(self):
        return None

    def send_signal(self, _sig):
        pass

    def wait(self, timeout=None):
        return 0

    def kill(self):
        pass


def test_start_stack_spawns_prebuilt_daemon_not_cargo_run(monkeypatch, tmp_path):
    """Regression: start_stack spawned `cargo run --release`, folding a cold
    release build (minutes on a fresh checkout) into the daemon's 30s
    readiness window; the harness then failed as "not healthy" while the
    compiler was still running. The build must complete BEFORE spawn.
    Found by /qa on 2026-10-19
    Report: .gstack/qa-reports/qa-report-bench-workloads-2026-10-19.md

    Value: protects=start_stack spawns the prebuilt release daemon only after
    `cargo build --release` completed; fails_when=start_stack uses `cargo run`
    (build inside the readiness window); why_new=no other test covers
    start_stack's spawn shape; seam=none; no_claim=none beyond protects."""
    import subprocess as _subprocess

    builds, spawns = [], []

    def fake_run(cmd, *a, **k):
        builds.append(cmd)
        return _subprocess.CompletedProcess(cmd, 0)

    def fake_popen(argv, *a, **k):
        spawns.append(argv)
        return _FakePopenProc()

    monkeypatch.setattr(bw.subprocess, "run", fake_run)
    monkeypatch.setattr(bw.subprocess, "Popen", fake_popen)

    bw.start_stack(str(tmp_path))

    assert builds, "daemon must be built before spawn"
    assert builds[0][1:4] == ["run", "cargo", "build"]
    daemon_argv = spawns[-1]
    assert "cargo" not in daemon_argv
    assert str(daemon_argv[0]).endswith("target/release/mlxcache-daemon")


def test_report_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        bw.report("bad", [1.0], [1.0, 2.0], [1])
