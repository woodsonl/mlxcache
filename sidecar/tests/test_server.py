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


def test_tokenize_reports_kv_dtype(sidecar_url):
    # The daemon pins kv_dtype in the fingerprint. It must come from the engine,
    # not a daemon-side constant: a bf16 model's checkpoint is unsafe to serve
    # to an f16 request with the same model id and tokenizer.
    r = httpx.post(f"{sidecar_url}/tokenize", json={"prompt": "x"}).json()
    assert r["kv_dtype"] == "synthetic"
    server.Handler.engine.kv_dtype = "bfloat16"
    try:
        r2 = httpx.post(f"{sidecar_url}/tokenize", json={"prompt": "x"}).json()
        assert r2["kv_dtype"] == "bfloat16"
    finally:
        server.Handler.engine.kv_dtype = "synthetic"


def test_tokenize_returns_engine_hash_not_a_constant(sidecar_url):
    # R1-2: the daemon pins THIS hash. A constant would let checkpoints from
    # different tokenizers collide on fingerprint and be served wrongly.
    r = httpx.post(f"{sidecar_url}/tokenize", json={"prompt": "x"}).json()
    assert r["tokenizer_hash"] == "synthetic"  # the synthetic engine's own value
    # It must come from the engine, not a hardcoded string in the handler.
    server.Handler.engine.tokenizer_hash = "custom-abc"
    try:
        r2 = httpx.post(f"{sidecar_url}/tokenize", json={"prompt": "x"}).json()
        assert r2["tokenizer_hash"] == "custom-abc"
    finally:
        server.Handler.engine.tokenizer_hash = "synthetic"


def test_tokenize_can_return_zero_tokens(sidecar_url, monkeypatch):
    # The daemon rejects empty tokenizations; the sidecar must be able to
    # produce them for that path to be testable (real HF tokenizers do).
    monkeypatch.setenv("MLXCACHE_TOKENIZE_EMPTY", "1")
    r = httpx.post(f"{sidecar_url}/tokenize", json={"prompt": ""}).json()
    assert r["tokens"] == []


def test_prefill_returns_raw_payload(sidecar_url):
    # The daemon owns the header; /prefill returns the raw payload bytes.
    tokens = [1, 2, 3, 4]
    payload = httpx.post(f"{sidecar_url}/prefill", json={"tokens": tokens}).content
    assert len(payload) == 4 * 1024


def test_generate_continuation(sidecar_url):
    r = httpx.post(
        f"{sidecar_url}/generate",
        json={"tokens": [1, 2, 3], "max_tokens": 5},
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


def _raw_post(url: str, path: str, content_length: str, body: bytes = b"") -> tuple[int, str]:
    """Send a hand-built request so we can control a malformed Content-Length."""
    import socket
    from urllib.parse import urlparse

    u = urlparse(url)
    s = socket.create_connection((u.hostname, u.port), timeout=3)
    head = (
        f"POST {path} HTTP/1.1\r\nHost: x\r\nConnection: close\r\n"
        f"Content-Length: {content_length}\r\n\r\n"
    )
    s.sendall(head.encode() + body)
    data = b""
    try:
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            data += chunk
    except TimeoutError:
        pass
    s.close()
    status_line = data.split(b"\r\n", 1)[0].decode(errors="replace")
    code = int(status_line.split()[1])
    return code, data.split(b"\r\n\r\n", 1)[-1].decode(errors="replace")


def test_non_numeric_content_length_is_400(sidecar_url):
    # A malformed Content-Length must be a clean 400, not a 500 leaking the
    # internal ValueError text.
    code, body = _raw_post(sidecar_url, "/tokenize", "abc")
    assert code == 400
    assert "invalid Content-Length" in body
    assert "ValueError" not in body


def test_negative_content_length_is_400(sidecar_url):
    # read(-1) would block until EOF, pinning a worker thread (DoS).
    code, _ = _raw_post(sidecar_url, "/tokenize", "-1")
    assert code == 400


def test_oversized_content_length_is_400(sidecar_url):
    # A declared huge length with a tiny body must not block the worker.
    code, body = _raw_post(sidecar_url, "/tokenize", "999999999", b"{}")
    assert code == 400
    assert "out of range" in body


def test_missing_required_field_is_400(sidecar_url):
    code, body = _raw_post(sidecar_url, "/prefill", "2", b"{}")
    assert code == 400
    assert "missing required field" in body


def test_malformed_json_is_400(sidecar_url):
    code, body = _raw_post(sidecar_url, "/prefill", "3", b"{x}")
    assert code == 400
    assert "malformed JSON" in body


def test_stream_corrupt_blob_is_422_not_truncated_200(sidecar_url, tmp_path, monkeypatch):
    # Regression (Codex adversarial P1): the real engine loads the checkpoint on
    # the first step of its stream generator. If that raises AFTER 200 is sent,
    # the daemon sees a failed stream but cannot quarantine the checkpoint, and
    # the poison stays selectable forever. The load must run BEFORE headers, so a
    # rejected checkpoint yields a clean 422 (which the daemon quarantines on).
    class LoadThenStream:
        name = "load-then-stream"
        tokenizer_hash = "synthetic"
        kv_dtype = "synthetic"
        prefill_count = 0

        def prepare_stream(self, tokens, blob_path):  # noqa: ANN001, ANN201
            if blob_path:
                raise server.CheckpointRejectedError("corrupt safetensors payload")
            return tokens, None

        def stream_prepared(self, prompt, cache):  # noqa: ANN001, ANN201
            yield 1, "a"

    bad = tmp_path / "corrupt.ckpt"
    bad.write_bytes(b"not a real blob")
    monkeypatch.setattr(server.Handler, "engine", LoadThenStream())
    try:
        r = httpx.post(
            f"{sidecar_url}/generate",
            json={"tokens": [1, 2], "max_tokens": 1, "blob_path": str(bad), "stream": True},
        )
        assert r.status_code == 422, f"expected clean 422, got {r.status_code}"
        assert r.headers.get("content-type", "").startswith("application/json")
    finally:
        server.Handler.engine = server.make_engine("test-model")


def test_stream_generic_load_error_is_500_not_quarantine(sidecar_url, tmp_path, monkeypatch):
    # A non-rejection failure while opening the stream (e.g. OOM) is a 500, not a
    # 422: the daemon must retry from scratch WITHOUT retiring the checkpoint.
    # Only CheckpointRejectedError maps to 422.
    class OomThenStream:
        name = "oom-then-stream"
        tokenizer_hash = "synthetic"
        kv_dtype = "synthetic"
        prefill_count = 0

        def prepare_stream(self, tokens, blob_path):  # noqa: ANN001, ANN201
            if blob_path:
                raise MemoryError("out of memory loading cache")
            return tokens, None

        def stream_prepared(self, prompt, cache):  # noqa: ANN001, ANN201
            yield 1, "a"

    blob = tmp_path / "healthy.ckpt"
    blob.write_bytes(b"whatever")
    monkeypatch.setattr(server.Handler, "engine", OomThenStream())
    try:
        r = httpx.post(
            f"{sidecar_url}/generate",
            json={"tokens": [1, 2], "max_tokens": 1, "blob_path": str(blob), "stream": True},
        )
        assert r.status_code == 500, f"expected 500 (no quarantine), got {r.status_code}"
    finally:
        server.Handler.engine = server.make_engine("test-model")


def test_stream_midstream_error_does_not_inject_error_frame(sidecar_url, monkeypatch):
    # Regression (Codex adversarial F2): once headers are sent, a generator
    # failure must truncate the stream WITHOUT a done marker (so the daemon can
    # surface an upstream error), not fall through to do_POST's 500 writer which
    # would append a JSON error body to a live 200 and be read as a token frame.
    class FailsMidStream:
        name = "fails-mid-stream"
        tokenizer_hash = "synthetic"
        kv_dtype = "synthetic"
        prefill_count = 0

        def prepare_stream(self, tokens, blob_path):  # noqa: ANN001, ANN201
            return tokens, None

        def stream_prepared(self, prompt, cache):  # noqa: ANN001, ANN201
            yield 1, "a"  # load succeeded; the first token is emitted
            raise RuntimeError("decode blew up mid-stream")

    monkeypatch.setattr(server.Handler, "engine", FailsMidStream())
    try:
        r = httpx.post(
            f"{sidecar_url}/generate",
            json={"tokens": [1, 2], "max_tokens": 4, "stream": True},
        )
        assert r.status_code == 200, "headers were already committed"
        assert '"error"' not in r.text, "a mid-stream error must not inject an error frame"
        assert '"done"' not in r.text, "truncation must not send a completion marker"
    finally:
        server.Handler.engine = server.make_engine("test-model")


def test_stream_zero_max_tokens_yields_no_tokens(sidecar_url):
    # Regression (Codex adversarial F3): priming the generator must not force a
    # token when max_tokens <= 0.
    r = httpx.post(
        f"{sidecar_url}/generate",
        json={"tokens": [1, 2], "max_tokens": 0, "stream": True},
    )
    assert r.status_code == 200
    token_lines = [ln for ln in r.text.splitlines() if '"token"' in ln]
    assert token_lines == [], f"max_tokens=0 emitted tokens: {r.text}"
    assert '"done": true' in r.text or '"done":true' in r.text


def test_stream_load_error_not_treated_as_decode_error(sidecar_url, monkeypatch):
    # Regression (Codex adversarial F3): a first-token decode failure must NOT be
    # reported as blob corruption (500); the checkpoint loaded fine. It surfaces
    # after headers as a truncated stream, so the daemon does not quarantine a
    # healthy checkpoint.
    class DecodeFailsAfterLoad:
        name = "decode-fails"
        tokenizer_hash = "synthetic"
        kv_dtype = "synthetic"
        prefill_count = 0

        def prepare_stream(self, tokens, blob_path):  # noqa: ANN001, ANN201
            return tokens, None  # load ok

        def stream_prepared(self, prompt, cache):  # noqa: ANN001, ANN201
            raise MemoryError("first decode OOM")
            yield 1, "unreachable"  # pragma: no cover

    monkeypatch.setattr(server.Handler, "engine", DecodeFailsAfterLoad())
    try:
        r = httpx.post(
            f"{sidecar_url}/generate",
            json={"tokens": [1, 2], "max_tokens": 1, "stream": True},
        )
        assert r.status_code == 200, "a decode error after load must not be a 500"
        assert '"done"' not in r.text
    finally:
        server.Handler.engine = server.make_engine("test-model")


def test_prefill_fail_at_knob_fails_exactly_the_nth_call(monkeypatch):
    # MLXCACHE_PREFILL_FAIL_AT (test knob): a transient RuntimeError on
    # exactly the Nth /prefill call — not from-N-onward. The daemon e2e
    # drives this through HTTP (transient 500 → 502, ancestor intact, next
    # request recovers); this pins the exact-N semantics at the engine
    # level, where no HTTP layer can blur it.
    monkeypatch.setenv("MLXCACHE_PREFILL_FAIL_AT", "2")
    engine = server.SyntheticEngine("test-model")
    engine.prefill([1, 2, 3, 4], None)  # call #1: normal
    with pytest.raises(RuntimeError, match="synthetic induced prefill failure"):
        engine.prefill([1, 2, 3, 4, 5, 6], None)  # call #2: induced failure
    engine.prefill([1, 2], None)  # call #3: failures must not persist


def test_generate_nonstream_returns_detokenized_text(sidecar_url):
    # The choices[] compatibility layer: the sidecar detokenizes non-stream
    # generations (`generate_with_text`), and the text must be exactly the
    # streaming path's pieces joined — both legs agree char-for-char.
    r = httpx.post(
        f"{sidecar_url}/generate",
        json={"tokens": [1, 2, 3], "max_tokens": 3},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["tokens"] == [3, 4, 5], body
    assert body["text"] == "tok0 tok1 tok2 ", body


def test_engine_without_generate_with_text_collects_text_from_stream(sidecar_url, monkeypatch):
    # api-contract review: a contract-conforming engine that implements
    # stream() but not generate_with_text() must still produce detokenized
    # non-stream text — an empty choices[].content is a silent trap.
    class StreamOnly:
        tokenizer_hash = "s"

        def stream(self, tokens, blob_path):
            for i, t in enumerate(tokens):
                yield t + 100, f"piece{i} "

    monkeypatch.setattr(server.Handler, "engine", StreamOnly())
    try:
        r = httpx.post(f"{sidecar_url}/generate", json={"tokens": [1, 2, 3], "max_tokens": 2})
        assert r.status_code == 200
        body = r.json()
        assert body["tokens"] == [101, 102], body
        assert body["text"] == "piece0 piece1 ", body
    finally:
        server.Handler.engine = server.make_engine("test-model")
