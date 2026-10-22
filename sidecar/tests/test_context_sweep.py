"""The context sweep's self-checks and lock, tested without a model.

The sweep's product is trustworthy timings, and the checks are what make
them trustworthy: monotonicity in the descending row order (an earlier
generation of this check was inverted and flagged every healthy run), the
decode/prefill ratio bound, repeatability, swap growth, row sanity, and the
overlap lock that keeps two sweeps off one machine. All pure seams — the
probe imports cleanly without mlx.
"""

import os
import subprocess
import sys
import textwrap

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "probes"))
import context_sweep  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_lock_env():
    saved = {
        key: os.environ.get(key) for key in ("MLXCACHE_PROBE_LOCK", "MLXCACHE_PROBE_LOCK_PATH")
    }
    yield
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def test_plan_measures_every_size_with_control_brackets():
    plan = context_sweep.build_plan([8192, 256, 1024])
    assert plan[0] == (256, "warmup")
    assert plan[1] == (256, "control")
    assert plan[-1] == (256, "control")
    sweep = [size for size, phase in plan if phase == "sweep"]
    assert sweep == [8192, 1024, 256]
    # The ratio's small end is the requested smallest size, not the
    # second-smallest — and with exactly two sizes, not the same row twice.
    assert (sweep[0], sweep[-1]) == (8192, 256)


def test_plan_refuses_a_single_size():
    with pytest.raises(ValueError, match="at least two distinct sizes"):
        context_sweep.build_plan([256, 256])


def test_monotonicity_passes_a_healthy_descending_sweep():
    rows = [
        {"tokens": 8192, "prefill_s": 60.0},
        {"tokens": 1024, "prefill_s": 8.0},
        {"tokens": 256, "prefill_s": 2.0},
    ]
    assert context_sweep.monotonicity_failures(rows) == []


def test_monotonicity_flags_a_smaller_context_measuring_slower():
    # The regression pin: the original check compared in the wrong direction
    # and rejected every HEALTHY descending run. Smaller-but-slower is the
    # failure; smaller-and-cheaper is the expectation.
    rows = [
        {"tokens": 8192, "prefill_s": 60.0},
        {"tokens": 256, "prefill_s": 90.0},
    ]
    failures = context_sweep.monotonicity_failures(rows)
    assert len(failures) == 1
    assert "fewer tokens should cost less" in failures[0]


def test_ratio_summary_bounds_the_pair():
    rows = [
        {"tokens": 8192, "prefill_s": 60.0, "decode_s": 0.30},
        {"tokens": 256, "prefill_s": 2.0, "decode_s": 0.10},
    ]
    prefill_ratio, decode_ratio = context_sweep.ratio_summary(rows)
    assert prefill_ratio == pytest.approx(30.0)
    assert decode_ratio == pytest.approx(3.0)


def test_ratio_summary_reports_no_decode_ratio_when_decode_is_disabled():
    rows = [
        {"tokens": 8192, "prefill_s": 60.0, "decode_s": 0.0},
        {"tokens": 256, "prefill_s": 2.0, "decode_s": 0.0},
    ]
    _, decode_ratio = context_sweep.ratio_summary(rows)
    assert decode_ratio is None  # 0/0 must not masquerade as a ratio


def test_ratio_bound_fails_when_decode_outgrows_prefill():
    # The pin for the check that caught a contaminated 8K figure (decode
    # 7.8x while prefill grew 1.07x): decode growing proportionally more
    # than prefill between the same two sizes is contention, not scaling.
    contaminated = [
        {"tokens": 8192, "prefill_s": 2.14, "decode_s": 0.60},
        {"tokens": 1024, "prefill_s": 2.0, "decode_s": 0.08},
    ]
    failures = context_sweep.ratio_failures(contaminated)
    assert len(failures) == 1
    assert "under contention" in failures[0]


def test_ratio_bound_passes_when_decode_stays_under_prefill():
    healthy = [
        {"tokens": 32768, "prefill_s": 66.23, "decode_s": 0.021},
        {"tokens": 256, "prefill_s": 0.39, "decode_s": 0.015},
    ]
    assert context_sweep.ratio_failures(healthy) == []


def test_ratio_bound_skips_when_decode_unmeasured():
    rows = [
        {"tokens": 8192, "prefill_s": 60.0, "decode_s": 0.0},
        {"tokens": 256, "prefill_s": 2.0, "decode_s": 0.0},
    ]
    assert context_sweep.ratio_failures(rows) == []


def test_repeatability_within_tolerance_passes():
    rows = [{"tokens": 256, "prefill_s": 2.0}, {"tokens": 256, "prefill_s": 2.4}]
    assert context_sweep.repeatability_failure(rows, tolerance=0.25) is None


def test_repeatability_beyond_tolerance_fails():
    rows = [{"tokens": 256, "prefill_s": 2.0}, {"tokens": 256, "prefill_s": 3.0}]
    failure = context_sweep.repeatability_failure(rows, tolerance=0.25)
    assert failure is not None
    assert "256" in failure
    assert "cannot resolve" in failure


def test_swap_growth_over_the_limit_fails():
    rows = [
        {"tokens": 8192, "swap_before_mb": 0.0, "swap_after_mb": 512.0},
        {"tokens": 256, "swap_before_mb": 512.0, "swap_after_mb": 512.0},
    ]
    failures = context_sweep.swap_failures(rows)
    assert len(failures) == 1
    assert "8192" in failures[0]


def test_validate_row_rejects_wrong_size_and_missing_swap():
    assert context_sweep.validate_row({"tokens": 512}, 512) is not None
    swapless = {"tokens": 512, "swap_before_mb": None, "swap_after_mb": 1.0}
    assert "swap" in context_sweep.validate_row(swapless, 512)
    good = {"tokens": 512, "prefill_s": 0.68, "swap_before_mb": 0.0, "swap_after_mb": 0.0}
    assert context_sweep.validate_row(good, 512) is None


def test_validate_row_rejects_a_prefill_below_the_clock():
    # A sub-5 ms prefill rounds to 0.0 and would divide-by-zero the ratios
    # downstream; the refusal names it instead of crashing.
    row = {"tokens": 8, "prefill_s": 0.0, "swap_before_mb": 0.0, "swap_after_mb": 0.0}
    reason = context_sweep.validate_row(row, 8)
    assert reason is not None
    assert "below the clock" in reason


def _holder_script(lock_path: str) -> str:
    return textwrap.dedent(
        f"""
        import fcntl, os, time
        d = os.path.dirname({lock_path!r})
        if d:
            os.makedirs(d, exist_ok=True)
        fh = open({lock_path!r}, "a+")
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fh.write("sweep[old] (pid 1)")
        fh.flush()
        print("HELD", flush=True)
        time.sleep(60)
        """
    )


def _spawn_holder(lock: str) -> subprocess.Popen:
    holder = subprocess.Popen(
        [sys.executable, "-c", _holder_script(lock)],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert holder.stdout.readline().strip() == "HELD"
    return holder


def test_a_second_sweep_refuses_and_names_the_holder(tmp_path, monkeypatch):
    lock = str(tmp_path / "probe-sweep.lock")
    monkeypatch.setenv("MLXCACHE_PROBE_LOCK_PATH", lock)
    holder = _spawn_holder(lock)
    try:
        with pytest.raises(SystemExit) as exc, context_sweep.sweep_lock("sweep[8192]"):
            pass
        assert exc.value.code == 4
    finally:
        holder.kill()
        holder.wait()


def test_a_killed_holder_releases_the_lock(tmp_path, monkeypatch):
    # flock releases on process death; the stale in-file holder text is
    # display-only, never a gate.
    lock = str(tmp_path / "probe-sweep.lock")
    monkeypatch.setenv("MLXCACHE_PROBE_LOCK_PATH", lock)
    holder = _spawn_holder(lock)
    holder.kill()
    holder.wait()
    with context_sweep.sweep_lock("sweep[256]"):
        pass  # acquired: no refusal, no SystemExit


def test_the_env_override_bypasses_the_lock_with_a_warning(tmp_path, monkeypatch, capsys):
    lock = str(tmp_path / "probe-sweep.lock")
    monkeypatch.setenv("MLXCACHE_PROBE_LOCK_PATH", lock)
    monkeypatch.setenv("MLXCACHE_PROBE_LOCK", "off")
    holder = _spawn_holder(lock)
    try:
        with context_sweep.sweep_lock("sweep[256]"):
            pass
        out = capsys.readouterr().out
        assert "lock bypassed" in out
        assert "contaminated" in out
    finally:
        holder.kill()
        holder.wait()
