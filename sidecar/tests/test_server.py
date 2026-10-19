"""Sidecar server tests: contract behavior over real HTTP (hermetic engine)."""

from __future__ import annotations

import threading
import types

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


def test_tokenize_reports_engine_id_and_cache_layout(sidecar_url):
    # Connector protocol §2/§4: the daemon folds these into the fingerprint and
    # the key, so they must come from the ENGINE (not a handler constant) and
    # the synthetic engine must report the DEFAULT engine id — its fingerprints
    # have to stay byte-identical to the real mlx-lm engine's, which is the
    # whole point of the hermetic stand-in.
    r = httpx.post(f"{sidecar_url}/tokenize", json={"prompt": "x"}).json()
    assert r["engine_id"] == "mlx-lm"
    assert r["cache_layout"], "composition must be signed, never empty"

    server.Handler.engine.engine_id = "llama-cpp"
    server.Handler.engine.cache_layout = "opaque-state"
    try:
        r2 = httpx.post(f"{sidecar_url}/tokenize", json={"prompt": "x"}).json()
        assert r2["engine_id"] == "llama-cpp"
        assert r2["cache_layout"] == "opaque-state"
    finally:
        server.Handler.engine.engine_id = "mlx-lm"
        server.Handler.engine.cache_layout = "synthetic-dense-kv"


def test_synthetic_engine_reports_the_mlx_lm_engine_domain():
    # Load-bearing: "synthetic" would be a distinct key domain and would
    # re-key every hermetic e2e blob, hiding cross-binding drift.
    eng = server.SyntheticEngine("test-model")
    assert eng.engine_id == "mlx-lm"
    assert eng.engine_id == server.MlxLmEngine.engine_id
    assert eng.cache_layout and eng.cache_layout == server.SyntheticEngine.cache_layout


def test_tokenize_protocol_violation_ok_to_be_400(sidecar_url):
    # An engine that does not report the new identity fields (a future engine
    # that forgot them) falls back to the default engine id and an UNKNOWN
    # composition rather than raising on the hot path. The fallback is
    # deliberately safe: unknown composition only ever serves legacy-to-legacy
    # (mlxcache_core::policy), so a misconfigured engine loses reuse instead
    # of serving a payload of the wrong shape.
    class Bare:
        tokenizer_hash = "h"

        def tokenize(self, prompt):
            return [1, 2]

    server.Handler.engine = Bare()
    try:
        r = httpx.post(f"{sidecar_url}/tokenize", json={"prompt": "x"}).json()
        assert r["engine_id"] == "mlx-lm"
        assert r["cache_layout"] == ""
    finally:
        server.Handler.engine = server.make_engine("test-model")


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


def test_adoption_materializes_weights_for_request_threads(sidecar_url, monkeypatch):
    """Adopting a model must leave it usable from a request thread.

    MLX records the stream that built a lazy array, and a graph built on one
    thread cannot be evaluated on another: the worker raises `There is no
    Stream(gpu, 0) in current thread`. The engine is built on the thread that
    calls `serve()` while ThreadingHTTPServer dispatches requests on workers, so
    a model whose weights are still unevaluated fails on the first forward pass
    — a 500 on /prefill, before any token exists.

    The hermetic engine never evaluates an array, so this cannot be seen without
    a real MLX model; the fake below is the smallest one that reproduces it.
    """
    mx = pytest.importorskip("mlx.core")

    class LazyModel:
        def __init__(self):
            # Deliberately not evaluated: a loader that materialises on demand
            # leaves exactly this shape behind.
            self.w = mx.array([[1.0, 2.0], [3.0, 4.0]]) * 1.0

        def parameters(self):
            return {"w": self.w}

        def __call__(self, x):
            return x @ self.w

    def run_on_request_thread(model):
        result = {}

        def work():
            try:
                result["value"] = model(mx.array([[1.0, 1.0]])).tolist()
            except RuntimeError as exc:
                result["error"] = str(exc)

        thread = threading.Thread(target=work)
        thread.start()
        thread.join()
        return result

    # The hazard is real: unevaluated weights cannot be used off the boot thread.
    unmaterialized = run_on_request_thread(LazyModel())
    assert "Stream(gpu, 0)" in unmaterialized.get("error", ""), unmaterialized

    # Adoption is what makes them usable, and it must be part of adopting —
    # not something a caller has to remember.
    engine = object.__new__(server.MlxLmEngine)
    engine.model = LazyModel()
    engine.tokenizer = types.SimpleNamespace(_eos_token_ids={1})
    engine.tokenizer_hash = ""  # set by the real method; stubbed below
    monkeypatch.setattr(engine, "_hash_vocab", lambda: "h", raising=False)
    monkeypatch.setattr(engine, "_kv_dtype", lambda: "bfloat16", raising=False)
    monkeypatch.setattr(engine, "_prompt_cache_probe", lambda: [], raising=False)
    monkeypatch.setattr(engine, "_cache_layout_of", staticmethod(lambda cache: "layout"))
    monkeypatch.setattr(engine, "_assert_quantization_supported", lambda cache: None, raising=False)

    engine._adopt_model()
    assert run_on_request_thread(engine.model) == {"value": [[4.0, 6.0]]}


def _untrimmable_loop_engine(script, eos_token):
    """An engine whose model reports one cache position per forward, so the
    loop's end position is measurable.

    The tokenizer is a real `TokenizerWrapper` subclass (the loop's isinstance
    check and its `eos_token_ids`/`detokenizer` use are the contract), with the
    HF work stubbed out.
    """
    pytest.importorskip("mlx_lm")
    import mlx.core as mx
    from mlx_lm.tokenizer_utils import TokenizerWrapper

    vocab = max(max(script), eos_token) + 8

    class Model:
        positions = 0

        def __call__(self, ids, cache=None):
            Model.positions += ids.shape[1]
            logits = mx.zeros((1, 1, vocab))
            if ids.shape[1] == 1:
                # Decode step: emit the next scripted token. The prompt-prefill
                # forward's logits are discarded by the loop, so it consumes no
                # script entry.
                logits[0, 0, script.pop(0)] = 10.0
            return logits

    class Detokenizer:
        last_segment = ""

        def add_token(self, _token):
            pass

        def finalize(self):
            pass

    class Tokenizer(TokenizerWrapper):
        def __init__(self):  # bypass the HF tokenizer requirement
            self._eos_token_ids = {eos_token}
            self._detokenizer = Detokenizer()

    engine = object.__new__(server.MlxLmEngine)
    engine.model = Model()
    engine.tokenizer = Tokenizer()
    return engine


def test_untrimmable_loop_ends_on_the_convention_at_a_token_budget():
    """A token-budget stop leaves the cache on `tokens[:-1]`.

    This is the path the settle tests drive: the last yielded token is never
    fed back, so the cache covers exactly the tokens the checkpoint describes
    minus one.
    """
    prompt = [1, 2, 3, 4]
    engine = _untrimmable_loop_engine([10, 11, 12, 13], eos_token=99)
    out = []
    for token, _text in engine._stream_untrimmable(prompt, None):
        out.append(token)
        if len(out) == 3:
            break
    assert out == [10, 11, 12]
    total = len(prompt) + len(out)
    assert engine.model.positions == total - 1, "budget stop must land on the convention"


def test_untrimmable_loop_eos_stop_is_one_position_deep():
    """An EOS stop leaves the cache one position DEEPER than `tokens[:-1]`.

    Measured, not assumed: the forward that discovers EOS is the one that
    consumed the last yielded token, so the cache covers the whole
    `prompt + generated` sequence. A trimmable cache drops that surplus; a
    hybrid (recurrent-state) cache cannot, so `settle` must refuse rather than
    record a coverage count the cache does not hold. Pinned here because the
    inline comment claimed the opposite and nothing else exercised the EOS
    branch.
    """
    prompt = [1, 2, 3, 4]
    engine = _untrimmable_loop_engine([10, 11, 99], eos_token=99)
    out = [token for token, _text in engine._stream_untrimmable(prompt, None)]
    assert out == [10, 11], "EOS itself is never yielded"
    total = len(prompt) + len(out)
    assert engine.model.positions == total, "the EOS forward leaves the surplus position"


# --- settle_path confinement -------------------------------------------------


def _reset_staging_pin(monkeypatch, directory=None):
    """One pinned staging directory per process; tests settle into fresh
    tmp dirs, so reset the pin and point it where this test wants it."""
    pin = [str(directory)] if directory is not None else [None]
    monkeypatch.setattr(server, "_SETTLE_DIR_PIN", pin)


def test_generate_rejects_settle_path_outside_the_staging_convention(
    sidecar_url, monkeypatch, tmp_path
):
    # The sidecar is unauthenticated loopback HTTP: without the confinement
    # any local process could make the engine write a file anywhere the
    # server user can write. Only the daemon's own staging naming convention
    # is accepted.
    # Pin the daemon's directory first so every target is judged against it.
    _reset_staging_pin(monkeypatch, directory=tmp_path)
    targets = [
        str(tmp_path / "ordinary.ckpt"),
        str(tmp_path / ".settle-1-1-1.staging/../../escape"),
        "/etc/.settle-1-1-1.staging",
    ]
    for bad in targets:
        res = httpx.post(
            f"{sidecar_url}/generate",
            json={"tokens": [1, 2, 3], "max_tokens": 1, "settle_path": bad},
        )
        assert res.status_code == 400, bad
        assert "staging" in res.json()["error"]


def test_generate_rejects_a_staging_shaped_path_in_a_foreign_directory(
    sidecar_url, monkeypatch, tmp_path
):
    # The parent-directory pin: after the first accepted settle, a
    # convention-named path in a DIFFERENT directory must also be refused.
    _reset_staging_pin(monkeypatch, directory=tmp_path)
    ok = str(tmp_path / ".settle-1-1-1.staging")
    res = httpx.post(
        f"{sidecar_url}/generate",
        json={"tokens": [1, 2, 3], "max_tokens": 1, "settle_path": ok},
    )
    assert res.status_code == 200, res.text
    assert (tmp_path / ".settle-1-1-1.staging").exists()

    _reset_staging_pin(monkeypatch, directory=tmp_path)
    other = tmp_path / "elsewhere"
    other.mkdir()
    res = httpx.post(
        f"{sidecar_url}/generate",
        json={
            "tokens": [1, 2, 3],
            "max_tokens": 1,
            "settle_path": str(other / ".settle-2-2-2.staging"),
        },
    )
    assert res.status_code == 400, "a pinned daemon must not be redirected to another directory"
    assert not (other / ".settle-2-2-2.staging").exists()
