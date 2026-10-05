"""Real-engine thesis guard for the WRAPPER (B1.2).

The wrapper's own gate: the resumed CONTINUATION (context T = the stored
stream minus its last token) must be token-identical to the memory
lineage's continuation from the same context T. The fake-codec unit
tests cannot see KV positions; THIS test can: a payload stored covering
all of T instead of T[:-1] shifts the disk side to T+[dup] and diverges
within the window — the assert below pins the §3.2 convention.

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


def _generate_with_cache(model, prompt_ids, cache, n):
    """Feed prompt_ids into `cache`, then generate n tokens step-by-step
    through generate_step (each yielded token was fed into the cache
    before being produced — the exact producer shape the server's finish
    sites see). Returns the generated token ids."""
    import mlx.core as mx
    from mlx_lm.generate import generate_step

    out = []
    for generated, _logits in generate_step(
        mx.array(prompt_ids), model, prompt_cache=cache, max_tokens=n
    ):
        out.append(int(generated.item()) if hasattr(generated, "item") else int(generated))
        if len(out) >= n:
            break
    return out


def test_disk_resumed_generation_matches_scratch(engine, tmp_path):
    """insert_cache (finish site, post-generation state covering ALL of T)
    → fetch on a fresh cache instance (simulated restart) → the wrapper's
    feed math must reproduce the memory lineage's continuation EXACTLY."""
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    from mlxcache_serve import PersistentPromptCache, default_fingerprint_for
    from mlxcache_store import Store

    model_key = (MODEL, None, None)
    prompt = engine.tokenize(PROMPT_TEXT)
    assert len(prompt) > 32

    # 1. Scratch: fresh cache over the full prompt, then step.
    scratch_cache = make_prompt_cache(engine.model)
    scratch = _generate_with_cache(engine.model, prompt, scratch_cache, MAX_TOKENS)
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
    #    The loaded cache covers len(full_stream)-1 positions; feeding rest
    #    (the final token) completes the stream — generate_step resumes
    #    from prompt_cache and yields the continuation tokens.
    from mlx_lm.generate import generate_step

    resumed = [
        int(g.item()) if hasattr(g, "item") else int(g)
        for g, _lg in generate_step(
            mx.array(rest),
            engine.model,
            prompt_cache=disk_cache,
            max_tokens=MAX_TOKENS,
        )
    ][:MAX_TOKENS]
    # The resumed stream continues AFTER the stored generation: its first
    # MAX_TOKENS-1 tokens are the continuation the scratch lineage would
    # produce next. Produce that reference from the scratch side: keep
    # generating beyond scratch with the SAME memory lineage.
    # Reference: the memory TWIN of the disk path — the same trim-by-one
    # the writer applies, then the same single-token feed. Both sides then
    # sample from context T (the stored stream minus its last token); an
    # untrimmed payload would push the disk side to T+[dup] and diverge,
    # so this comparison is what pins the §3.2 convention.
    import copy as _copy

    from mlx_lm.models.cache import trim_prompt_cache

    ref_cache = _copy.deepcopy(scratch_cache)
    assert trim_prompt_cache(ref_cache, 1) == 1
    scratch_continuation = _generate_with_cache(engine.model, rest, ref_cache, MAX_TOKENS - 1)
    assert len(resumed) >= MAX_TOKENS - 1, f"short resume: {len(resumed)}"
    assert resumed[: len(scratch_continuation)] == scratch_continuation, (
        "disk-resumed generation diverged from the memory lineage — the §3.2 "
        "coverage convention is broken"
    )
