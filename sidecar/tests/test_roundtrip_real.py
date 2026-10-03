"""Real mlx-lm round-trip: the R1-5 gate + thesis guard (T2, real path).

Verifies the core claim of the whole product: generation continued from a
saved-then-loaded prompt cache produces IDENTICAL tokens to generation that
re-prefilled from scratch. If this fails, the mlx-lm adapter is too lossy and
the disk tier is not viable (design doc R1-5 / success-criteria gating).

Cache semantics (verified against mlx-lm 0.31.3): a prompt cache holds the KV
for tokens[:N]. To continue generation you prefill the cache with tokens[:-1]
and pass tokens[-1:] as the delta prompt; the model then predicts the same
token stream as a scratch run over the full prompt.

Run: MLXCACHE_BENCH_REAL=1 uv run pytest sidecar/tests/test_roundtrip_real.py -v
Requires a downloaded model (Qwen2-0.5B-Instruct, ~500MB) and Apple Silicon.
"""

from __future__ import annotations

import contextlib
import os
import time

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("MLXCACHE_BENCH_REAL") != "1",
    reason="real mlx-lm benchmark; set MLXCACHE_BENCH_REAL=1 (needs a local model)",
)

MODEL = os.environ.get("MLXCACHE_BENCH_MODEL", "Qwen/Qwen2-0.5B-Instruct")


@pytest.fixture(scope="module")
def engine():
    from mlxcache_sidecar.server import MlxLmEngine

    return MlxLmEngine(MODEL)


class TempSafetensors:
    """Context manager yielding a temp .safetensors path, cleaned on exit."""

    def __enter__(self):
        import tempfile

        fh = tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False)
        fh.close()
        self.path = fh.name
        return self.path

    def __exit__(self, *exc):
        with contextlib.suppress(OSError):
            os.unlink(self.path)


def test_roundtrip_logits_identical(engine):
    """The thesis guard: resumed generation == scratch generation."""
    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlx_lm.models.cache import load_prompt_cache, make_prompt_cache, save_prompt_cache

    # 64 generated tokens (not 16): a short oracle can pass on coincidence —
    # identical prefixes diverge later; 64 tokens of equality is parity.
    tokens = engine.tokenize("The quick brown fox jumps over the lazy dog. " * 4)

    scratch = [
        r.token
        for r in stream_generate(
            engine.model, engine.tokenizer, prompt=mx.array(tokens), max_tokens=64
        )
    ]

    # Cache path: prefill tokens[:-1], save, load, feed the final token as delta.
    cache = make_prompt_cache(engine.model)
    engine.model(mx.array(tokens[:-1])[None], cache=cache)
    mx.eval([c.state for c in cache])
    with TempSafetensors() as path:
        save_prompt_cache(path, cache)
        resumed_cache = load_prompt_cache(path)
        resumed = [
            r.token
            for r in stream_generate(
                engine.model,
                engine.tokenizer,
                prompt=mx.array(tokens[-1:]),
                max_tokens=64,
                prompt_cache=resumed_cache,
            )
        ]

    assert resumed == scratch, (
        "resumed generation diverged from scratch:\n"
        f"  resumed={resumed[:8]}\n  scratch={scratch[:8]}"
    )


def test_adapter_prefill_resume_matches_scratch(engine):
    """Drives the REAL adapter path (MlxLmEngine.prefill -> generate_from_blob),
    not raw mlx-lm. This is what the daemon actually calls on a miss then a hit:
    the earlier convention bug (seeding the cache with all N tokens while the
    resume path fed tokens[cached-1:]) made every real hit double-feed the last
    token and diverge from scratch. This test would have caught it."""
    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlxcache_sidecar.blob import CheckpointMeta, Fingerprint, encode

    tokens = engine.tokenize("The quick brown fox jumps over the lazy dog. " * 4)

    # 64 generated tokens: see the oracle-width note in the raw test above.
    scratch = [
        r.token
        for r in stream_generate(
            engine.model, engine.tokenizer, prompt=mx.array(tokens), max_tokens=64
        )
    ]

    # Miss path: the adapter produces the raw KV payload the daemon persists.
    payload = engine.prefill(tokens)
    meta = CheckpointMeta(
        fingerprint=Fingerprint(
            model_id=engine.model_id,
            tokenizer_hash=engine.tokenizer_hash,
            kv_dtype=engine.kv_dtype,
            kv_layout_version=1,
        ),
        token_count=len(tokens),
        tokens=tokens,
    )
    blob = encode(meta, payload)

    with TempSafetensors() as path:
        with open(path, "wb") as fh:
            fh.write(blob)
        # Hit path: resume from the persisted blob exactly as the daemon does.
        resumed = engine.generate_from_blob(tokens, path, max_tokens=64)

    assert resumed == scratch, (
        "adapter resume diverged from scratch (cache convention bug):\n"
        f"  resumed={resumed[:8]}\n  scratch={scratch[:8]}"
    )


def test_adapter_delta_prefill_matches_scratch(engine):
    """OV3 parity on the REAL engine: generation resumed from a blob whose KV
    was built by DELTA prefill (ancestor KV + only the uncovered tokens) must
    be token-identical to scratch. This is the shipped fast path for a growing
    conversation — if adopting ancestor KV diverges, every partial hit after
    this change is silently wrong, so it is pinned here at oracle width."""
    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlxcache_sidecar.blob import CheckpointMeta, Fingerprint, encode

    base = engine.tokenize("The quick brown fox jumps over the lazy dog.")
    grown = engine.tokenize(
        "The quick brown fox jumps over the lazy dog. " * 4
    )
    assert len(grown) > len(base) and grown[: len(base)] == base, (
        "test premise: the grown prompt must extend the base token-for-token"
    )

    scratch = [
        r.token
        for r in stream_generate(
            engine.model, engine.tokenizer, prompt=mx.array(grown), max_tokens=64
        )
    ]

    # Step 1: publish the ancestor (full prefill of the short prompt).
    ancestor_payload = engine.prefill(base)
    with TempSafetensors() as ancestor_path:
        meta = CheckpointMeta(
            fingerprint=Fingerprint(
                model_id=engine.model_id,
                tokenizer_hash=engine.tokenizer_hash,
                kv_dtype=engine.kv_dtype,
                kv_layout_version=1,
            ),
            token_count=len(base),
            tokens=base,
        )
        with open(ancestor_path, "wb") as fh:
            fh.write(encode(meta, ancestor_payload))

        # Step 2: delta prefill of the grown prompt from the ancestor.
        delta_payload = engine.prefill(grown, ancestor_blob_path=ancestor_path)
        assert engine.last_prefill_delta_tokens < len(grown) - 1, (
            "delta prefill recomputed the whole prompt "
            f"({engine.last_prefill_delta_tokens} of {len(grown) - 1} steps)"
        )
        meta = CheckpointMeta(
            fingerprint=Fingerprint(
                model_id=engine.model_id,
                tokenizer_hash=engine.tokenizer_hash,
                kv_dtype=engine.kv_dtype,
                kv_layout_version=1,
            ),
            token_count=len(grown),
            tokens=grown,
        )
        with TempSafetensors() as delta_path:
            with open(delta_path, "wb") as fh:
                fh.write(encode(meta, delta_payload))
            # Step 3: resume generation from the delta-prefilled blob.
            resumed = engine.generate_from_blob(grown, delta_path, max_tokens=64)

    assert resumed == scratch, (
        "delta-prefill resume diverged from scratch:\n"
        f"  resumed={resumed[:8]}\n  scratch={scratch[:8]}"
    )


def test_adapter_one_token_prompt_matches_scratch(engine):
    """Regression: a ONE-token prompt. The cache covers tokens[:-1] = nothing, so
    the resume path must feed the whole prompt once. The earlier special case
    seeded the one token AND fed it as the delta, feeding it twice and diverging
    from scratch on a one-token hit."""
    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlxcache_sidecar.blob import CheckpointMeta, Fingerprint, encode

    tokens = engine.tokenize("Hello")[:1]
    assert len(tokens) == 1

    scratch = [
        r.token
        for r in stream_generate(
            engine.model, engine.tokenizer, prompt=mx.array(tokens), max_tokens=8
        )
    ]

    payload = engine.prefill(tokens)
    meta = CheckpointMeta(
        fingerprint=Fingerprint(
            model_id=engine.model_id,
            tokenizer_hash=engine.tokenizer_hash,
            kv_dtype=engine.kv_dtype,
            kv_layout_version=1,
        ),
        token_count=len(tokens),
        tokens=tokens,
    )
    with TempSafetensors() as path:
        with open(path, "wb") as fh:
            fh.write(encode(meta, payload))
        resumed = engine.generate_from_blob(tokens, path, max_tokens=8)

    assert resumed == scratch, (
        "one-token prompt resume diverged from scratch:\n"
        f"  resumed={resumed[:8]}\n  scratch={scratch[:8]}"
    )


def test_roundtrip_measures_bytes_and_time(engine):
    """R1-5 numbers: bytes/token + serialize/deserialize wall time at 50K."""
    import mlx.core as mx
    from mlx_lm.models.cache import load_prompt_cache, make_prompt_cache, save_prompt_cache

    n = int(os.environ.get("MLXCACHE_BENCH_TOKENS", "50000"))
    tokens = engine.tokenize("The quick brown fox jumps over the lazy dog. " * (n // 10))[:n]

    cache = make_prompt_cache(engine.model)
    t0 = time.perf_counter()
    engine.model(mx.array(tokens)[None], cache=cache)
    mx.eval([c.state for c in cache])
    prefill_ms = (time.perf_counter() - t0) * 1000

    with TempSafetensors() as path:
        t0 = time.perf_counter()
        save_prompt_cache(path, cache)
        serialize_ms = (time.perf_counter() - t0) * 1000
        size = os.path.getsize(path)
        t0 = time.perf_counter()
        load_prompt_cache(path)
        deserialize_ms = (time.perf_counter() - t0) * 1000

    bpt = size / len(tokens)
    print(
        f"\n[REAL] tokens={len(tokens)} bytes={size} bytes/token={bpt:.1f} "
        f"prefill_ms={prefill_ms:.0f} serialize_ms={serialize_ms:.0f} "
        f"deserialize_ms={deserialize_ms:.0f}"
    )
    assert size > 0
    assert bpt > 0
    # The 2s TTFT resume budget (R1-5): deserialize + delta prefill must fit.
    assert deserialize_ms < 2000, (
        f"deserialize {deserialize_ms:.0f}ms exceeds the 2s resume budget — "
        "v1 ships memory-resident only per R1-5"
    )
