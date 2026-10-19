"""Pure-arithmetic guards for scripts/bench_workloads.py.

report()/tpot_ms() are the plumbing harness's whole measurement contract;
until now nothing exercised them without a live daemon, so a regression in
the p95 gating or the token divisor would only surface as quietly wrong
numbers in a green run.
"""

from __future__ import annotations

import importlib.util
import json
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

    events: list[tuple[str, object]] = []

    def fake_run(cmd, *a, **k):
        events.append(("build", cmd))
        return _subprocess.CompletedProcess(cmd, 0)

    def fake_popen(argv, *a, **k):
        event = ("spawn", argv)
        events.append(event)
        return _FakePopenProc()

    monkeypatch.setattr(bw.subprocess, "run", fake_run)
    monkeypatch.setattr(bw.subprocess, "Popen", fake_popen)

    bw.start_stack(str(tmp_path))

    # The build must not merely precede `cargo run`: it must precede EVERY
    # spawn, or a cold build still burns inside a child's lifetime.
    kinds = [kind for kind, _ in events]
    assert kinds.index("build") < kinds.index("spawn")
    daemon_argv = events[-1][1]
    assert "cargo" not in daemon_argv
    assert str(daemon_argv[0]).endswith("target/release/mlxcache-daemon")


def test_main_cleans_own_blob_dir_but_not_a_given_one(monkeypatch, tmp_path):
    """Value: protects=main() removes the temp blob dir it created but never
    a caller-provided --blobs dir; fails_when=the cleanup is dropped (GB-scale
    dirs strand in $TMPDIR) or inverted (a user's --blobs dir is deleted);
    why_new=the only main() test exits at validation; seam=none; no_claim=
    run_workloads is stubbed, so workload behavior is not exercised here."""
    monkeypatch.setattr(bw, "wait_healthy", lambda url, daemon=None, deadline_s=30.0: None)
    monkeypatch.setattr(bw, "run_workloads", lambda url, args: [])

    mine = tmp_path / "mine"
    mine.mkdir()
    given = tmp_path / "given"
    given.mkdir()
    monkeypatch.setattr(bw.tempfile, "mkdtemp", lambda prefix="": str(mine))

    monkeypatch.setattr(sys, "argv", ["bench_workloads.py", "--daemon", "http://127.0.0.1:1"])
    bw.main()
    assert not mine.exists(), "self-made blob dir must be cleaned up"

    monkeypatch.setattr(
        sys, "argv", ["bench_workloads.py", "--daemon", "http://127.0.0.1:1", "--blobs", str(given)]
    )
    bw.main()
    assert given.exists(), "caller-provided --blobs dir must be preserved"


def test_report_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        bw.report("bad", [1.0], [1.0, 2.0], [1])


_STATS = {
    "requests": 9,
    "hits": 1,
    "partials": 3,
    "misses": 5,
    "checkpoints_published": 2,
    "hit_rate": 0.5,
}


class _StatsResp:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return json.dumps(_STATS).encode()


class _Args:
    def __init__(self, requests, concurrency, grow_rounds):
        self.requests = requests
        self.concurrency = concurrency
        self.grow_rounds = grow_rounds


def _patch_run_environment(monkeypatch, stream_fake):
    monkeypatch.setattr(bw, "stream_chat", stream_fake)
    monkeypatch.setattr(bw.urllib.request, "urlopen", lambda *a, **k: _StatsResp())


def test_concurrent_identical_worker_errors_fail_loudly(monkeypatch):
    """A worker that dies mid-burst must fail the run, not shrink the sample.

    Value: protects=concurrent_identical collects worker exceptions and
    raises after the join; fails_when=worker exceptions are swallowed by the
    thread (run completes green with a smaller n); why_new=no other test
    drives run_workloads — the error-capture branch was live-run-only;
    seam=none (module globals patched: stream_chat/urlopen, real
    run_workloads/threads); no_claim=none beyond protects.
    """

    def flaky(url, body):
        if "concurrent burst" in body["messages"][0]["content"]:
            raise RuntimeError("boom 500")
        return 1.0, 2.0, 4

    _patch_run_environment(monkeypatch, flaky)
    with pytest.raises(RuntimeError, match=r"concurrent_identical lost 2 worker\(s\)"):
        bw.run_workloads("http://127.0.0.1:1", _Args(requests=2, concurrency=2, grow_rounds=2))


def test_concurrent_distinct_worker_errors_fail_loudly(monkeypatch):
    """Same contract for the distinct-prompt workload (a separate copy of
    the capture pattern; a regression can silence either one alone).

    Value: protects=concurrent_distinct collects worker exceptions and
    raises after the join; fails_when=worker_distinct exceptions are
    swallowed by the thread; why_new=no other test reaches workload 5, and
    its capture code is a distinct copy from workload 4's; seam=none (same
    patched environment, real run_workloads); no_claim=none beyond protects.
    """

    def flaky(url, body):
        if "distinct burst" in body["messages"][0]["content"]:
            raise RuntimeError("boom 500")
        return 1.0, 2.0, 4

    _patch_run_environment(monkeypatch, flaky)
    with pytest.raises(RuntimeError, match=r"concurrent_distinct lost 2 worker\(s\)"):
        bw.run_workloads("http://127.0.0.1:1", _Args(requests=2, concurrency=2, grow_rounds=2))


def test_concurrent_burst_count_uses_ceiling(monkeypatch):
    """Bursts must be ceil(requests/concurrency), not floor: floor division
    quantizes the sample count down and silences the p95 tail gate on
    exactly the contention workloads it exists for.

    Value: protects=each concurrent workload runs ceil(requests/concurrency)
    bursts so total samples >= requests; fails_when=bursts uses floor
    division (5 requests / 2 workers -> 4 samples, not 6);
    why_new=no other test pins the burst arithmetic — it is inline in
    run_workloads and was live-run-only; seam=none (patched stream/stats,
    real run_workloads); no_claim=none beyond protects.
    """

    def ok(url, body):
        return 1.0, 2.0, 4

    _patch_run_environment(monkeypatch, ok)
    rows = bw.run_workloads("http://127.0.0.1:1", _Args(requests=5, concurrency=2, grow_rounds=2))
    by_name = {r["workload"]: r for r in rows}
    assert by_name["concurrent_identical_c2"]["n"] == 6
    assert by_name["concurrent_distinct_c2"]["n"] == 6


def test_main_rejects_sub_one_counts(monkeypatch):
    """--requests/--concurrency/--grow-rounds < 1 must fail with a clean
    SystemExit before anything spawns, instead of limping into empty
    workloads and a StatisticsError deep in report().

    Value: protects=main validates all three counts >= 1 up front;
    fails_when=validation is dropped or inverted (main proceeds toward
    starting the stack); why_new=no other test invokes main — the guard was
    live-run-only; seam=none (argv patched, start_stack double as a
    tripwire); no_claim=none beyond protects.
    """
    monkeypatch.setattr(sys, "argv", ["bench_workloads.py", "--requests", "0"])

    def _no_spawn(*a, **k):
        raise AssertionError("invalid counts must fail before any stack spawn")

    monkeypatch.setattr(bw, "start_stack", _no_spawn)
    with pytest.raises(SystemExit, match="must all be >= 1"):
        bw.main()
