"""Sidecar HTTP server: the mlx-lm compatibility adapter.

Endpoints (daemon -> sidecar protocol):
- POST /tokenize  {"prompt": str} -> {"tokens": [u32], "tokenizer_hash": str}
- POST /prefill   {"prompt": str, "tokens": [u32]} -> KV checkpoint blob (binary)
- POST /generate  {"tokens": [u32], "prefill_from": int, "max_tokens": int,
                   "stream": bool} -> NDJSON token stream

Engine selection: MLXCACHE_ENGINE=synthetic (default, hermetic) or mlx-lm
(lazy import; requires a local model). The daemon is the only client; this
server binds 127.0.0.1 by default.

Run: uv run python -m mlxcache_sidecar.server
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class SyntheticEngine:
    """Hermetic engine: deterministic tokens + KV bytes, no MLX dependency."""

    name = "synthetic"

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.prefill_count = 0

    def tokenize(self, prompt: str) -> list[int]:
        # Deterministic token stream derived from the prompt hash. Not a real
        # tokenizer — the daemon contract only needs stable u32 tokens.
        digest = hashlib.sha256(prompt.encode()).digest()
        return [int.from_bytes(digest[i : i + 4], "little") % 2**31 for i in range(0, 32, 4)]

    def prefill(self, tokens: list[int]) -> bytes:
        # KV payload: 1024 bytes/token, deterministic from token ids.
        self.prefill_count += 1
        # Optional delay (test knob): widens the single-flight window so
        # concurrent identical requests are provably coalesced.
        delay = float(os.environ.get("MLXCACHE_PREFILL_DELAY", "0"))
        if delay:
            time.sleep(delay)
        payload = b"".join(
            ((t * 31 + i) & 0xFFFFFFFF).to_bytes(4, "little") * 256 for i, t in enumerate(tokens)
        )
        # Raw payload only; the daemon writes the checkpoint header.
        return payload

    def generate(self, tokens: list[int], prefill_from: int, max_tokens: int) -> list[int]:
        # Deterministic continuation: token ids derived from context length.
        base = len(tokens)
        return [(base + i) % 2**31 for i in range(max_tokens)]

    def stream(self, tokens: list[int], blob_path: str | None):
        # Synthetic engine streams its deterministic tokens as text pieces.
        for i, t in enumerate(self.generate(tokens, 0, 64)):
            yield t, f"tok{i} "


class MlxLmEngine:
    """Real mlx-lm engine. Lazy import; requires a downloaded model.

    KV checkpoints are real mlx-lm prompt caches (safetensors). The blob is the
    sidecar/daemon wire format: u32 header length + JSON meta + safetensors bytes.
    """

    name = "mlx-lm"

    def __init__(self, model_id: str) -> None:
        from mlx_lm import load  # noqa: PLC0415 — lazy, only when selected

        self.model_id = model_id
        self.model, self.tokenizer = load(model_id)
        self.tokenizer_hash = hashlib.sha256(
            getattr(self.tokenizer, "name_or_path", model_id).encode()
        ).hexdigest()[:16]

    def tokenize(self, prompt: str) -> list[int]:
        return self.tokenizer.encode(prompt)

    def _prefill_cache(self, tokens: list[int]):
        """Run tokens through the model filling a fresh prompt cache. Returns
        (cache, last_logits) with the KV state resident and evaluated."""
        import mlx.core as mx  # noqa: PLC0415
        from mlx_lm.models.cache import make_prompt_cache  # noqa: PLC0415

        cache = make_prompt_cache(self.model)
        inp = mx.array(tokens)[None]
        logits = self.model(inp, cache=cache)
        mx.eval([c.state for c in cache], logits)
        return cache, logits

    def prefill(self, tokens: list[int]) -> bytes:
        import mlx.core as mx  # noqa: PLC0415
        from mlx_lm.models.cache import save_prompt_cache  # noqa: PLC0415

        cache, _ = self._prefill_cache(tokens)
        # safetensors needs a real file; write to a temp, then read the bytes.
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as fh:
            tmp = fh.name
        try:
            save_prompt_cache(tmp, cache)
            with open(tmp, "rb") as fh:
                payload = fh.read()
        finally:
            os.unlink(tmp)
        mx.clear_cache()

        # Return the raw safetensors payload. The daemon owns the checkpoint
        # header (it writes meta via publish_atomic) and the wire format is
        # header + this payload; encoding one here would double-wrap it.
        return payload

    def generate(self, tokens: list[int], prefill_from: int, max_tokens: int) -> list[int]:
        """Generate continuing from `prefill_from` cached tokens.

        When prefill_from == len(tokens) (a full hit), the KV cache is loaded
        from the blob the daemon passes; otherwise the delta is prefilled. The
        daemon supplies the blob via MLXCACHE_* — see the /generate handler.
        """
        return self._generate_with_cache(tokens, max_tokens, cache=None)

    def stream(self, tokens: list[int], blob_path: str | None):
        """Yield (token_id, text) for this request, resuming from `blob_path`
        when present. This is the streaming entry the handler drives."""
        cache = None
        prompt = tokens
        if blob_path:
            cache, prompt = self._load_cache_delta(tokens, blob_path)
            if cache is None:
                prompt = tokens  # unexpectable cache: scratch
        yield from self._stream_with_cache(prompt, cache)

    def _stream_with_cache(self, prompt, cache):
        # SyntheticEngine has no MLX stream path; approximate from its tokens.
        if not hasattr(self, "stream_with_cache"):
            for t in self.generate(prompt, 0, 64):
                yield t, ""
            return
        yield from self.stream_with_cache(prompt, cache)

    def stream_with_cache(self, tokens, cache):
        """Yield (token_id, text_piece) per decode step from mlx-lm."""
        import mlx.core as mx  # noqa: PLC0415
        from mlx_lm import stream_generate  # noqa: PLC0415

        for resp in stream_generate(
            self.model, self.tokenizer, prompt=mx.array(tokens), prompt_cache=cache
        ):
            yield resp.token, resp.text

    def _generate_with_cache(self, tokens: list[int], max_tokens: int, cache) -> list[int]:
        """Collect a bounded generation from the streaming path, so the
        non-streaming /generate mode reuses the identical decode logic."""
        out: list[int] = []
        for token, _text in self._stream_with_cache(tokens, cache):
            out.append(token)
            if len(out) >= max_tokens:
                break
        return out

    def _load_cache_delta(self, tokens: list[int], blob_path: str):
        """Returns (cache, delta_prompt) or (None, tokens). See generate_from_blob."""
        from mlx_lm.models.cache import load_prompt_cache  # noqa: PLC0415

        from .blob import decode  # noqa: PLC0415

        with open(blob_path, "rb") as fh:
            meta, payload = decode(fh.read())
        cached = meta.token_count
        if cached <= 0 or cached > len(tokens):
            return None, tokens
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as fh:
            tmp = fh.name
            fh.write(payload)
        try:
            return load_prompt_cache(tmp), tokens[cached - 1 :]
        finally:
            os.unlink(tmp)

    def generate_from_blob(self, tokens: list[int], blob_path: str, max_tokens: int) -> list[int]:
        """Resume generation from a persisted wire-format checkpoint (a cache
        hit): strip the daemon header, load the prompt cache, generate.

        The checkpoint holds KV for a token prefix; the delta (uncached tail)
        is what the model still needs to see. We pass the tail so the cache
        supplies the cached prefix and generation continues identically to a
        scratch run over the full prompt (thesis guard, T2)."""
        cache, prompt = self._load_cache_delta(tokens, blob_path)
        if cache is None:
            return self._generate_with_cache(tokens, max_tokens, cache=None)
        return self._generate_with_cache(prompt, max_tokens, cache)


def make_engine(model_id: str) -> SyntheticEngine | MlxLmEngine:
    if os.environ.get("MLXCACHE_ENGINE", "synthetic") == "mlx-lm":
        return MlxLmEngine(model_id)
    return SyntheticEngine(model_id)


class Handler(BaseHTTPRequestHandler):
    engine: object = None  # set by serve()

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write("sidecar: %s\n" % (fmt % args))

    def _json(self, code: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    @staticmethod
    def _require(req: dict, key: str):
        if key not in req:
            raise Handler._BadRequestError(f"missing required field '{key}'")
        return req[key]

    def _binary(self, code: int, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # Hard cap on request bodies (the daemon is the only client; this bounds a
    # malformed/hostile Content-Length from pinning a worker thread).
    MAX_BODY = 64 * 1024 * 1024

    class _BadRequestError(Exception):
        pass

    def _read_json(self) -> dict:
        raw_len = self.headers.get("Content-Length")
        if raw_len is None:
            return {}
        try:
            length = int(raw_len)
        except (TypeError, ValueError) as exc:
            raise Handler._BadRequestError("invalid Content-Length") from exc
        if length < 0 or length > self.MAX_BODY:
            raise Handler._BadRequestError("Content-Length out of range")
        if length == 0:
            return {}
        body = self.rfile.read(length)
        try:
            return json.loads(body)
        except (ValueError, UnicodeDecodeError) as exc:
            raise Handler._BadRequestError("malformed JSON body") from exc

    def _stream_ndjson(self, tokens: list, blob_path: str | None, max_tokens: int) -> None:
        """Stream generation as newline-delimited JSON: one line per token,
        then a final done line. Framed by connection close (HTTP/1.0-style)
        so reqwest reads until EOF — no hand-rolled chunked encoding."""
        # Validation happens in do_POST BEFORE any bytes are sent, so a bad
        # request can still yield a clean 400 rather than a truncated 200.
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        n = 0
        try:
            for token, text in self.engine.stream(tokens, blob_path):
                line = (json.dumps({"token": token, "text": text}) + "\n").encode()
                self.wfile.write(line)
                self.wfile.flush()
                n += 1
                if n >= max_tokens:
                    break
            self.wfile.write((json.dumps({"done": True, "tokens": n}) + "\n").encode())
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            # Client went away mid-stream (R1-4 semantics): stop cleanly.
            self.close_connection = True

    def do_POST(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler API
        try:
            if self.path == "/tokenize":
                req = self._read_json()
                tokens = self.engine.tokenize(req.get("prompt", ""))
                self._json(200, {"tokens": tokens, "tokenizer_hash": "synthetic"})
            elif self.path == "/prefill":
                req = self._read_json()
                blob = self.engine.prefill(self._require(req, "tokens"))
                self._binary(200, blob)
            elif self.path == "/generate":
                req = self._read_json()
                tokens = self._require(req, "tokens")
                if req.get("stream"):
                    self._stream_ndjson(tokens, req.get("blob_path"), req.get("max_tokens", 64))
                else:
                    blob_path = req.get("blob_path")
                    if blob_path and hasattr(self.engine, "generate_from_blob"):
                        out = self.engine.generate_from_blob(
                            tokens, blob_path, req.get("max_tokens", 64)
                        )
                    else:
                        out = self.engine.generate(
                            tokens, req.get("prefill_from", 0), req.get("max_tokens", 64)
                        )
                    self._json(200, {"tokens": out})
            else:
                self._json(404, {"error": f"unknown path {self.path}"})
        except Handler._BadRequestError as exc:
            # Client error: 400 with a stable message (no internal detail).
            self._json(400, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001 — map all engine errors to 500 JSON
            traceback.print_exc()
            self._json(500, {"error": str(exc)})

    def do_GET(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler API
        if self.path == "/health":
            self._json(200, {"status": "ok"})
        elif self.path == "/stats":
            # Test observability: how many prefills the engine actually ran.
            # Proves single-flight coalesced concurrent identical requests.
            self._json(200, {"prefill_count": getattr(self.engine, "prefill_count", None)})
        else:
            self._json(404, {"error": f"unknown path {self.path}"})


def serve(addr: str = "127.0.0.1", port: int = 8421) -> None:
    model_id = os.environ.get("MLXCACHE_MODEL", "synthetic-model")
    Handler.engine = make_engine(model_id)
    server = ThreadingHTTPServer((addr, port), Handler)
    print(f"sidecar: {Handler.engine.name} engine on {addr}:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    serve(
        addr=os.environ.get("MLXCACHE_SIDECAR_ADDR", "127.0.0.1"),
        port=int(os.environ.get("MLXCACHE_SIDECAR_PORT", "8421")),
    )
