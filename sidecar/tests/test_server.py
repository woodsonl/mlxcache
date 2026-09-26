"""Sidecar server tests: contract behavior over real HTTP (hermetic engine)."""

from __future__ import annotations

import threading

import httpx
import pytest
from mlxcache_sidecar import server


@pytest.fixture()
def sidecar_url(monkeypatch):
    monkeypatch.setenv("MLXCACHE_MODEL", "test-model")
    monkeypatch.setattr(server, "make_engine", lambda m: server.SyntheticEngine(m))
    server.Handler.engine = server.make_engine("test-model")
    httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_tokenize_deterministic(sidecar_url):
    r1 = httpx.post(f"{sidecar_url}/tokenize", json={"prompt": "hello world"}).json()
    r2 = httpx.post(f"{sidecar_url}/tokenize", json={"prompt": "hello world"}).json()
    assert r1["tokens"] == r2["tokens"]
    assert all(0 <= t < 2**31 for t in r1["tokens"])


def test_prefill_returns_raw_payload(sidecar_url):
    # The daemon owns the header; /prefill returns the raw payload bytes.
    tokens = [1, 2, 3, 4]
    payload = httpx.post(f"{sidecar_url}/prefill", json={"tokens": tokens}).content
    assert len(payload) == 4 * 1024


def test_generate_continuation(sidecar_url):
    r = httpx.post(
        f"{sidecar_url}/generate",
        json={"tokens": [1, 2, 3], "prefill_from": 0, "max_tokens": 5},
    ).json()
    assert len(r["tokens"]) == 5


def test_unknown_path_404(sidecar_url):
    r = httpx.post(f"{sidecar_url}/nope", json={})
    assert r.status_code == 404


def test_health(sidecar_url):
    r = httpx.get(f"{sidecar_url}/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_prefill_large_tokens_no_overflow(sidecar_url):
    # Regression: token ids up to 2**31 overflow the 4-byte payload pack.
    tokens = [2**31 - 1, 2**31 - 2, 0]
    payload = httpx.post(f"{sidecar_url}/prefill", json={"tokens": tokens}).content
    assert len(payload) == 3 * 1024
