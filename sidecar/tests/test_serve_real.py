"""Real-engine thesis guard for the WRAPPER (B1.2).

The wrapper's own gate: a disk-resumed generation through
PersistentPromptCache must be token-identical to scratch. The fake-codec
unit tests cannot see KV positions; THIS test can (it catches the §3.2
coverage off-by-one that the first review wave found: a payload stored
covering all of T instead of T[:-1] shifts the resumed context and
diverges within a few tokens).

Run: MLXCACHE_BENCH_REAL=1 uv run pytest sidecar/tests/test_serve_real.py -v
Requires a downloaded model (Qwen2-0.5B-Instruct, ~500MB) and Apple Silicon.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("MLXCACHE_BENCH_REAL") != "1",
    reason="real mlx-lm benchmark; set MLXCACHE_BENCH_REAL=1 (needs a local model)",
)

MODEL = os.environ.get("MLXCACHE_BENCH_MODEL", "Qwen/Qwen2-0.5B-Instruct")

PROMPT_TEXT = "The theory of persistent KV reuse rests on one convention. " * 8
MAX_TOKENS = 48


@pytest.fixture(scope="module")
def engine():
    from mlxcache_sidecar.server import MlxLmEngine

    return MlxLmEngine(MODEL)


def _sample(logits):
    import mlx.core as mx

    return int(mx.argmax(logits[:, -1, :], axis=-1).item())


def _generate_with_cache(engine, model, prompt_ids, cache, n):
    """Feed prompt_ids into `cache`, then generate n tokens step-by-step
    (generate_step feeds each token before yielding it — the exact
    producer shape the server's finish sites see). Returns (tokens,
    steps): the generated tokens and a done-flag."""
    import mlx.core as mx
    from mlx_lm.generate import generate_step

    logits = model(mx.array(prompt_ids)[None], cache=cache)
    mx.eval(logits)
    out = []
    token = _sample(logits)
    for _ in range(n):
        out.append(token)
        steps = generate_step(mx.array([token]), model, cache=cache)
        token = None
        for _, lg in steps:
            token = _sample(lg[None] if lg.ndim == 1 else lg)
            break
        if token is None:
            break
    return out


def test_disk_resumed_generation_matches_scratch(engine, tmp_path):
    """insert_cache (finish site, post-generation state covering ALL of T)
    → fetch on a fresh cache instance (simulated restart) → the wrapper's
    feed math must reproduce scratch EXACTLY."""
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    from mlxcache_serve import PersistentPromptCache, default_fingerprint_for
    from mlxcache_store import Store

    model_key = (MODEL, None, None)
    prompt = engine.tokenize(PROMPT_TEXT)
    assert len(prompt) > 32

    # 1. Scratch: fresh cache over the full prompt, then step.
    scratch_cache = make_prompt_cache(engine.model)
    scratch = _generate_with_cache(engine, engine.model, prompt, scratch_cache, MAX_TOKENS)
    assert len(scratch) == MAX_TOKENS

    # 2. The finish-site state: the SAME generation's cache now covers
    #    prompt + all generated tokens (each was fed before being yielded).
    full_stream = prompt + scratch

    # 3. Persist it the way the server's finish site does.
    store = Store(tmp_path, byte_budget=0)
    fp = default_fingerprint_for(model_key)
    pc = PersistentPromptCache(store, fp)
    pc.insert_cache(model_key, full_stream, scratch_cache, cache_type="assistant")
    assert pc.persisted == 1, "finish-site insert must persist"

    # 4. A FRESH cache instance over the SAME store (simulated restart):
    #    fetch + the wrapper's feed math.
    fresh = PersistentPromptCache(Store(tmp_path, byte_budget=0), fp)
    disk_cache, rest = fresh.fetch_nearest_cache(model_key, full_stream)
    assert disk_cache is not None, "disk entry must be found"
    assert rest == full_stream[len(full_stream) - 1 :], (
        "matched=len(T) → covered=len(T)-1 → feed exactly the LAST token"
    )

    # 5. Continue generation from the resumed cache and compare to scratch.
    from mlx_lm.generate import generate_step

    logits = engine.model(mx.array([rest[0]])[None], cache=disk_cache)
    mx.eval(logits)
    resumed = []
    token = _sample(logits)
    for _ in range(MAX_TOKENS - 1):
        resumed.append(token)
        token = None
        for _, lg in generate_step(mx.array([resumed[-1]]), engine.model, cache=disk_cache):
            token = _sample(lg[None] if lg.ndim == 1 else lg)
            break
        if token is None:
            break
    assert resumed == scratch, (
        "disk-resumed generation diverged from scratch — the §3.2 coverage convention is broken"
    )
