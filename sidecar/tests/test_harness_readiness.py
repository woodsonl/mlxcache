"""Readiness-gate unit tests for the bench/QA harness scripts.

These exercise the Server readiness loop directly with fakes: no model, no
GPU, no subprocess. They pin the behavior the harness must have so a server
that is not actually serving is never mistaken for ready.

The spawn/ready/stop core lives in scripts/_harness.py; bench_wrapper and
qa_wrapper subclass it and only choose which client call proves readiness.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
from pathlib import Path

import pytest


def _load(name: str):
    path = Path(__file__).resolve().parents[2] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    # Register BEFORE exec: the wrapper scripts import _harness through
    # sys.path, and without this the file-located copy and the imported copy
    # are two different module objects — a monkeypatch on `harness.X` would
    # then "succeed" without touching the classes bench/qa actually use.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


harness = _load("_harness")
bench = _load("bench_wrapper")
qa = _load("qa_wrapper")


class _FakeProc:
    pid = 4242

    def poll(self):
        return None

    def send_signal(self, _sig):
        pass

    def wait(self, timeout=None):
        return 0

    def kill(self):
        pass


class _DeadProc(_FakeProc):
    """A child whose port TOCTOU already claimed: poll() reports the exit."""

    def poll(self):
        return 17


class _ExitDuringWarmupProc(_FakeProc):
    """Alive at loop top (probe is answered by a stale listener), dead by the
    time the probe returns — poll() flips once _warmup has run."""

    def __init__(self):
        self.probed = False

    def poll(self):
        return 17 if self.probed else None


class _StubbornProc:
    """Ignores SIGTERM until killed; models the stop() escalation path."""

    pid = 4252

    def __init__(self):
        self.killed = False

    def poll(self):
        return None if not self.killed else 0

    def send_signal(self, _sig):
        pass

    def wait(self, timeout=None):
        if not self.killed:
            raise subprocess.TimeoutExpired("cmd", timeout)
        return 0

    def kill(self):
        self.killed = True


class _FakeResp:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _server(cls):
    srv = cls.__new__(cls)
    fd, log_path = tempfile.mkstemp(suffix=".log")
    srv.log_path = Path(log_path)
    srv.log_fh = os.fdopen(fd, "w")
    srv.proc = _FakeProc()
    srv.base = "http://127.0.0.1:1"
    srv.warmup = "post" if cls is qa.Server else "chat"
    return srv


def test_qa_warmup_non_200_is_not_ready(monkeypatch):
    """A warmup that returns 5xx must not be read as ready (post() returns
    errors as (code, body) instead of raising, so the status is checked)."""
    monkeypatch.setattr(harness.urllib.request, "urlopen", lambda *a, **k: _FakeResp())
    srv = _server(qa.Server)
    srv.post = lambda path, body, timeout=600: (500, {"error": "load failed"})
    try:
        t0 = time.time()
        with pytest.raises(RuntimeError, match="warmup HTTP 500"):
            srv.wait_ready(timeout_s=2.0)
        assert time.time() - t0 < 8
    finally:
        srv.log_fh.close()
        srv.log_path.unlink(missing_ok=True)


def test_bench_warmup_failure_is_bounded_and_reported(monkeypatch):
    """A warmup that keeps failing must retry with a deadline-bounded timeout
    and raise naming the error, not block on the client default (600s)."""
    monkeypatch.setattr(harness.urllib.request, "urlopen", lambda *a, **k: _FakeResp())
    srv = _server(bench.Server)
    seen = {}

    def fail(body, timeout=600):
        seen["timeout"] = timeout
        raise urllib.error.URLError("refused mid-load")

    srv.chat = fail
    try:
        t0 = time.time()
        with pytest.raises(RuntimeError, match="refused mid-load"):
            srv.wait_ready(timeout_s=2.0)
        assert seen["timeout"] <= 30
        assert time.time() - t0 < 8
    finally:
        srv.log_fh.close()
        srv.log_path.unlink(missing_ok=True)


def test_bench_warmup_retries_then_succeeds(monkeypatch):
    """A transient warmup failure (server still loading) must retry to success."""
    monkeypatch.setattr(harness.urllib.request, "urlopen", lambda *a, **k: _FakeResp())
    srv = _server(bench.Server)
    calls = {"n": 0}

    def flaky(body, timeout=600):
        calls["n"] += 1
        if calls["n"] < 2:
            raise urllib.error.URLError("not yet")
        return 10.0, "ok"

    srv.chat = flaky
    try:
        srv.wait_ready(timeout_s=10.0)
        assert calls["n"] == 2
    finally:
        srv.log_fh.close()
        srv.log_path.unlink(missing_ok=True)


def test_stop_unlinks_temp_log():
    # The shared implementation directly: after the dedup, neither wrapper
    # subclass overrides stop(), so per-subclass rows would replay the same
    # inherited contract twice.
    srv = _server(harness.Server)
    log_path = srv.log_path
    srv.stop()
    assert not log_path.exists()


def test_resolve_model_abs_and_hfid_agree():
    """An absolute local model dir resolves to itself; a bare HF id resolves
    via the hub (honouring refs/main), so both sides fingerprint identically.
    Skips when no snapshot is cached on this machine."""
    hfid = "mlx-community/Qwen2.5-7B-Instruct-4bit"
    huggingface_hub = pytest.importorskip("huggingface_hub")
    try:
        absolute = huggingface_hub.snapshot_download(hfid, local_files_only=True)
    except Exception:
        pytest.skip("no cached Qwen2.5-7B snapshot")
    absolute = os.path.realpath(absolute)
    for mod in (bench, qa):
        assert os.path.realpath(mod.resolve_model(absolute)) == absolute
        assert os.path.realpath(mod.resolve_model(hfid)) == absolute


def test_resolve_model_rejects_uncached_id():
    for mod in (bench, qa):
        with pytest.raises(SystemExit):
            mod.resolve_model("definitely/not-a-cached-model-xyz")


def test_wait_ready_fails_fast_when_child_exits():
    """The child-exited guard is the fast-fail defense against the port
    TOCTOU free_port's own docstring names: a dead child whose port was
    reclaimed must never be probed — the probe could be answered by whatever
    owns the port now, and the harness would benchmark the wrong stack."""
    srv = _server(qa.Server)
    srv.proc = _DeadProc()
    warmups = []
    srv._warmup = lambda timeout: warmups.append(timeout)
    try:
        with pytest.raises(RuntimeError, match="exited early"):
            srv.wait_ready(timeout_s=5.0)
        assert warmups == []  # never probed a port its child no longer owns
    finally:
        srv.log_fh.close()
        srv.log_path.unlink(missing_ok=True)


def test_wait_ready_rejects_child_that_exits_during_warmup():
    """The sneakier TOCTOU half: the child is alive when the loop starts and
    the probe SUCCEEDS (a stale listener on the explicit port answered it),
    but the child died mid-probe. Certifying ready would benchmark the stale
    listener, so the post-probe liveness check must raise — from OUTSIDE the
    retry's except, or the raise is swallowed and retried."""
    srv = _server(qa.Server)
    srv.proc = _ExitDuringWarmupProc()

    def successful_probe(_timeout):
        # Simulate a probe answered by a stale listener: it "succeeds", and
        # the child dies while it ran.
        srv.proc.probed = True

    srv._warmup = successful_probe
    try:
        with pytest.raises(RuntimeError, match="exited during warmup"):
            srv.wait_ready(timeout_s=5.0)
    finally:
        srv.log_fh.close()
        srv.log_path.unlink(missing_ok=True)


def test_stop_escalates_to_kill_and_still_unlinks():
    """A child that ignores SIGTERM must be SIGKILLed, and the escalation
    must not skip the close/unlink (an unsuppressed raise there would
    re-leak the log fd and temp file on every stuck teardown)."""
    srv = _server(harness.Server)
    srv.proc = proc = _StubbornProc()
    log_path = srv.log_path
    srv.stop()
    assert proc.killed
    assert not log_path.exists()


def test_spawn_failure_cleans_up_temp_log(monkeypatch, tmp_path):
    """If Popen fails, the log fd and temp file the constructor already
    created must be closed and unlinked, not leaked per failed spawn."""

    def fake_mkstemp(*_a, **_k):
        # mkstemp's real contract is a RAW os-level fd with no owning Python
        # file object; double ownership makes the cleanup close twice.
        fd = os.open(tmp_path / "spawn.log", os.O_CREAT | os.O_WRONLY, 0o600)
        return fd, str(tmp_path / "spawn.log")

    monkeypatch.setattr(harness.tempfile, "mkstemp", fake_mkstemp)

    def boom(*_a, **_k):
        raise OSError("exec failed")

    monkeypatch.setattr(harness.subprocess, "Popen", boom)
    with pytest.raises(OSError, match="exec failed"):
        harness.Server(["/nonexistent"], port=1, base="http://127.0.0.1:1")
    assert not (tmp_path / "spawn.log").exists()


def test_subclasses_choose_their_warmup_client(monkeypatch):
    """Production wiring, not the injection point: qa's Server must probe
    via post() (a 5xx is inspected, not raised) and bench's via chat()
    (raises AND returns the wall time its benchmark reports). A flipped
    default would leave one script's readiness shape untested while its
    tests keep passing against a hand-injected value."""
    seen = {}

    class _FakePopen:
        pid = 4243

        def __init__(self, argv, **_kw):
            seen["argv"] = argv

        def poll(self):
            return 0

        def send_signal(self, _sig):
            pass

        def wait(self, timeout=None):
            return 0

        def kill(self):
            pass

    monkeypatch.setattr(harness.subprocess, "Popen", _FakePopen)
    for mod, mode in ((qa, "post"), (bench, "chat")):
        srv = mod.Server("store", "model", 4321)
        try:
            assert srv.warmup == mode
            assert srv.base == "http://127.0.0.1:4321"
            assert seen["argv"][seen["argv"].index("--port") + 1] == "4321"
        finally:
            srv.stop()


def _leg(verdict):
    return {"leg": f"l-{verdict}", "wall_ms": 1.0, "text": "x", "verdict": verdict, "reused": {}}


def test_gate_requires_observed_disk_hits():
    """A populated store is not reuse: without an observed disk hit the gate
    must FAIL even when every leg is token-identical and coverage is high."""
    results = [
        _leg("token-identical"),
        _leg("token-identical"),
        _leg("token-identical"),
        _leg("token-identical"),
        _leg("token-identical"),
    ]
    ok, reasons = bench._gate_ok(results, 93, {"restart-turn2": 0, "second-turn2": 0})
    assert not ok and any("not observed serving" in r for r in reasons)

    ok, _ = bench._gate_ok(results, 93, {"restart-turn2": 1, "second-turn2": 2})
    assert ok


def test_gate_fails_on_divergence_or_low_skip():
    diverged = [_leg("token-identical"), _leg("DIVERGED")]
    ok, reasons = bench._gate_ok(diverged, 93, {"restart-turn2": 1})
    assert not ok and any("divergence" in r for r in reasons)

    ok, reasons = bench._gate_ok([_leg("token-identical")], 50, {"restart-turn2": 1})
    assert not ok and any("50% < 90%" in r for r in reasons)


def test_gate_fails_on_reuse_errors():
    """A measured leg that logged reuse errors must fail the gate even when
    its disk_hits delta is positive.

    Value: protects=_gate_ok treats any reuse_errors delta > 0 on a measured
    leg as a failure; fails_when=_gate_ok ignores the reuse_errors argument
    (legs with reuse errors pass while hitting the disk); why_new=no other
    test passes reuse_errors — the existing gate tests cover divergence,
    low skip and missing disk hits, so deleting this branch stays green;
    seam=none (pure function, real _gate_ok); no_claim=none beyond protects.
    """
    results = [_leg("token-identical") for _ in range(5)]
    ok, reasons = bench._gate_ok(
        results,
        93,
        {"restart-turn2": 1, "second-turn2": 1},
        {"restart-turn2": 2, "second-turn2": 0},
    )
    assert not ok
    assert any("reuse errors on measured legs" in r for r in reasons)

    # Control: the same clean legs with no reuse errors still pass.
    ok, _ = bench._gate_ok(results, 93, {"restart-turn2": 1, "second-turn2": 1})
    assert ok


def test_warmup_mode_typo_fails_before_spawn(monkeypatch):
    """Value: protects=WrapperServer rejects an unknown warmup mode with a
    ValueError BEFORE spawning the child; fails_when=validation runs after
    super().__init__ (a typo'd mode leaks the spawned process and its log) or
    is absent (silently selects the chat probe); why_new=the wiring test pins
    the two valid modes, not the rejection; seam=none; no_claim=none beyond
    protects."""
    spawned = []

    def no_spawn(*_a, **_k):
        spawned.append(1)
        raise AssertionError("constructor must not spawn on an invalid mode")

    monkeypatch.setattr(harness.subprocess, "Popen", no_spawn)
    with pytest.raises(ValueError, match="warmup mode"):
        harness.WrapperServer("store", "model", 1234, warmup="Chat")
    assert spawned == []


def test_delta_fails_loudly_on_a_missing_counter():
    """Value: protects=stats_delta (shared via _harness and re-exported by
    both wrappers) raises when /mlxcache/stats omits a counter instead of
    coercing it to zero; fails_when=the `or 0` coercion returns (a missing
    disk_hits reads as 'no hits' and can pass a gate on a half-broken stats
    endpoint); why_new=no test covers stats_delta; seam=none; no_claim=none
    beyond protects."""
    with pytest.raises(RuntimeError, match="did not report 'disk_hits'"):
        qa._delta({}, {"reuse_errors": 0}, "disk_hits")
    with pytest.raises(RuntimeError, match="did not report 'disk_hits'"):
        qa._delta({"disk_hits": 1}, {}, "disk_hits")
    assert qa._delta({"disk_hits": 2}, {"disk_hits": 5}, "disk_hits") == 3
    assert qa._delta is harness.stats_delta
    assert bench.stats_delta is harness.stats_delta


def test_warmup_prompt_is_unique_per_child(monkeypatch):
    """Value: protects=each spawned server's warmup probe prompt embeds its
    OWN child pid (a constant prompt is published by the first server and
    disk-hit by every later one during warmup, corrupting cumulative
    disk_hits counters before any measured request); fails_when=the prompt
    reverts to a constant string; why_new=the wiring test pins mode/base/
    argv but not the prompt body; seam=none; no_claim=none beyond protects."""
    bodies = []

    class _P:
        def __init__(self, pid):
            self.pid = pid

        def poll(self):
            return 0

        def send_signal(self, _sig):
            pass

        def wait(self, timeout=None):
            return 0

        def kill(self):
            pass

    for pid in (111, 222):

        def fake_popen(*a, _pid=pid, **k):
            return _P(_pid)

        monkeypatch.setattr(harness.subprocess, "Popen", fake_popen)
        srv = harness.WrapperServer("store", "model", 1, warmup="post")
        monkeypatch.setattr(
            srv, "post", lambda path, body, timeout=600: bodies.append(body) or (200, {})
        )
        monkeypatch.setattr(srv, "health_ok", lambda timeout: None)
        srv._warmup(5.0)
        srv.stop()
    assert "111" in bodies[0]["messages"][0]["content"]
    assert "222" in bodies[1]["messages"][0]["content"]
    assert bodies[0]["messages"][0]["content"] != bodies[1]["messages"][0]["content"]


def test_stop_warns_when_child_survives_sigkill(capsys):
    """Value: protects=stop() prints a warning when the child survives
    SIGKILL instead of returning silently; fails_when=the survived-branch
    warning is dropped (a leaked process goes unreported while its log is
    unlinked); why_new=the escalation test's child dies on kill, so the
    survivor branch is never reached; seam=none; no_claim=none beyond
    protects."""

    class _UnkillableProc(_StubbornProc):
        def kill(self):
            self.killed = True  # but poll() below keeps reporting "alive"

        def poll(self):
            return None

    srv = _server(harness.Server)
    srv.proc = _UnkillableProc()
    log_path = srv.log_path
    srv.stop()
    err = capsys.readouterr().err
    assert "survived SIGKILL" in err
    assert not log_path.exists()  # the leak warning must not skip cleanup


def test_health_ok_requires_200(monkeypatch):
    """Value: protects=health_ok raises on a non-200 status instead of
    treating any response as healthy; fails_when=the status check is removed
    (a 5xx listener reads as ready); why_new=only the 200 path was ever
    faked; seam=none; no_claim=none beyond protects."""

    class _Fake500:
        status = 500

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    srv = _server(harness.Server)
    monkeypatch.setattr(harness.urllib.request, "urlopen", lambda *a, **k: _Fake500())
    try:
        with pytest.raises(RuntimeError, match="health 500"):
            srv.health_ok(1.0)
    finally:
        srv.log_fh.close()
        srv.log_path.unlink(missing_ok=True)


def test_stream_enforces_its_total_deadline():
    """Value: protects=Server.stream raises when the SSE stream exceeds its
    total budget even though no single read times out (a server dripping one
    frame per few seconds would hang the harness forever); fails_when=the
    monotonic deadline check is removed (only the per-read socket timeout
    remains); why_new=no test reaches the deadline branch; seam=none (local
    HTTP server, real stream path); no_claim=none beyond protects."""
    import http.server
    import threading

    class _DripHandler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            # Frames arrive FASTER than the socket timeout but the stream
            # never ends: only the monotonic total-deadline check can stop
            # consumption (a per-read timeout would never fire).
            for _ in range(100):
                self.wfile.write(b'data: {"token": "a"}\n\n')
                self.wfile.flush()
                time.sleep(0.05)

        def log_message(self, *args):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), _DripHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    client = harness.Server.__new__(harness.Server)
    client.base = f"http://127.0.0.1:{srv.server_port}"
    try:
        with pytest.raises(RuntimeError, match="total budget"):
            client.stream("/v1/chat/completions", {}, timeout=0.3)
    finally:
        srv.shutdown()
