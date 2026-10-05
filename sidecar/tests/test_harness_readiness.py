"""Readiness-gate unit tests for the bench/QA harness scripts.

These exercise the Server readiness loop directly with fakes: no model, no
GPU, no subprocess. They pin the behavior the harness must have so a server
that is not actually serving is never mistaken for ready.
"""

from __future__ import annotations

import importlib.util
import os
import tempfile
import time
import urllib.error
from pathlib import Path

import pytest


def _load(name: str):
    path = Path(__file__).resolve().parents[2] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bench = _load("bench_wrapper")
qa = _load("qa_wrapper")


class _FakeProc:
    def poll(self):
        return None

    def send_signal(self, _sig):
        pass

    def wait(self, timeout=None):
        return 0

    def kill(self):
        pass


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
    return srv


def test_qa_warmup_non_200_is_not_ready(monkeypatch):
    """A warmup that returns 5xx must not be read as ready (post() returns
    errors as (code, body) instead of raising, so the status is checked)."""
    monkeypatch.setattr(qa.urllib.request, "urlopen", lambda *a, **k: _FakeResp())
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
    monkeypatch.setattr(bench.urllib.request, "urlopen", lambda *a, **k: _FakeResp())
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
    monkeypatch.setattr(bench.urllib.request, "urlopen", lambda *a, **k: _FakeResp())
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


@pytest.mark.parametrize("mod", [qa, bench])
def test_stop_unlinks_temp_log(mod):
    srv = _server(mod.Server)
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
        assert os.path.realpath(mod._resolve_model(absolute)) == absolute
        assert os.path.realpath(mod._resolve_model(hfid)) == absolute


def test_resolve_model_rejects_uncached_id():
    for mod in (bench, qa):
        with pytest.raises(SystemExit):
            mod._resolve_model("definitely/not-a-cached-model-xyz")


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
