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


# The measured kernel-path noise between a batch prefill and a delta feed
# over bitwise-equal KV: ±0.3 per side. A sampled-token flip is
# noise-explainable only under the two-sided band.
NEAR_TIE_BAND = 0.6


class _RecordDecode:
    """Transparent model proxy capturing per-step decode logits. Generated
    token k comes from the k-th [1,1] call: stream_generate batch-prefills
    prompt[:-1] in one call, then decodes one token per call."""

    def __init__(self, model):
        self.model = model
        self.decode_logits = []

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __call__(self, *args, **kwargs):
        out = self.model(*args, **kwargs)
        shape = args[0].shape
        if len(shape) == 2 and shape[1] == 1:
            self.decode_logits.append(out.reshape(-1))
        return out


def _witness_scratch(resumed, scratch, twin_logits, scratch_logits=None):
    """The scratch canary shared by the resume-parity tests. Sampled-token
    divergence from a fresh batch prefill is permitted ONLY at a near-tie
    step. The flip is licensed by the top-2 margin at the first divergence,
    taken as the MIN of the margins available on each lineage (twin always;
    scratch too when recorded) — corruption on either lineage cannot
    manufacture its own near-tie license by narrowing its own gap."""
    if resumed == scratch:
        return
    k = next(
        (i for i, (a, b) in enumerate(zip(resumed, scratch, strict=False)) if a != b),
        None,
    )
    if k is None:
        pytest.fail(
            "one run is a strict prefix of the other (terminal-token "
            f"divergence at an EOS/continuation near-tie); "
            f"len(resumed)={len(resumed)} vs len(scratch)={len(scratch)}"
        )

    def _gap(logits):
        assert k < len(logits), "decode logits missing at the divergence step"
        top2 = sorted(logits[k].tolist(), reverse=True)[:2]
        return top2[0] - top2[1]

    gap = _gap(twin_logits)
    if scratch_logits is not None:
        gap = min(gap, _gap(scratch_logits))
    assert gap < NEAR_TIE_BAND, (
        f"resume diverged from scratch at step {k} with margin {gap:.4g} "
        f"above the {NEAR_TIE_BAND} noise band — not a near-tie flip; "
        "investigate the resume path"
    )


def _state_max_diff(cache_a, cache_b):
    """Max absolute elementwise difference between two prompt caches' states.
    Bitwise-equal producer states differ by the kernel-path noise (~0.3);
    wrong KV, an off-by-one feed, or a corrupted round-trip produces
    differences orders of magnitude larger."""
    import mlx.core as mx

    worst = 0.0
    for ca, cb in zip(cache_a, cache_b, strict=True):
        for sa, sb in zip(ca.state, cb.state, strict=True):
            diff = mx.abs(sa - sb).max().item()
            worst = max(worst, float(diff))
    return worst


class _ResumeSpy:
    """Wraps engine._load_cache_delta to prove the resume actually adopted a
    loaded cache instead of silently falling back to a scratch prefill — the
    direct read_wire_checkpoint assert cannot see a fallback that happens
    inside the engine's own resume call."""

    def __init__(self, engine):
        self.engine = engine
        self.adopted = None

    def __enter__(self):
        engine = self.engine
        spy = self

        def _spied(tokens, blob_path):
            cache, delta = engine._load_cache_delta.__wrapped__(tokens, blob_path)
            spy.adopted = cache is not None
            return cache, delta

        _spied.__wrapped__ = engine._load_cache_delta
        engine._load_cache_delta = _spied
        return self

    def __exit__(self, *exc):
        del self.engine._load_cache_delta
        return False


def _blob_bytes(engine, tokens, payload):
    """The daemon wire format for a published checkpoint (header + payload)."""
    from mlxcache_sidecar.blob import CheckpointMeta, Fingerprint, encode

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
    return encode(meta, payload)


def _twin_generate(engine, delta, twin_cache, max_tokens=64):
    """Generation from a memory-resident producer state, with per-step decode
    logits captured. Returns (tokens, decode_logits)."""
    recorder = _RecordDecode(engine.model)
    engine.model = recorder
    try:
        tokens = engine._generate_with_cache(delta, max_tokens, twin_cache)
    finally:
        engine.model = recorder.model
    return tokens, recorder.decode_logits


def test_roundtrip_logits_identical(engine):
    """The raw thesis guard: a saved-then-loaded prompt cache is faithful to
    the producer state that wrote it.

    The equality target is the producer twin — the identical prefill state
    held in memory — not a fresh batch prefill of the full prompt. Batch
    prefill and the final-token feed run different kernel shapes, and their
    ~0.3 logit disagreement decides zero-margin argmax steps per-device
    (issue #27); the twin isolates the variable under test, the save/load
    round-trip. The scratch run remains a margin-guarded canary (see
    _witness_scratch)."""
    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlx_lm.models.cache import load_prompt_cache, make_prompt_cache, save_prompt_cache

    # 64 generated tokens (not 16): a short oracle can pass on coincidence —
    # identical prefixes diverge later; 64 tokens of equality is parity.
    tokens = engine.tokenize("The quick brown fox jumps over the lazy dog. " * 4)

    scratch_recorder = _RecordDecode(engine.model)
    engine.model = scratch_recorder
    try:
        scratch = [
            r.token
            for r in stream_generate(
                scratch_recorder,
                engine.tokenizer,
                prompt=mx.array(tokens),
                max_tokens=64,
            )
        ]
    finally:
        engine.model = scratch_recorder.model

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

    # The twin: the identical producer state, memory-resident.
    twin_cache = make_prompt_cache(engine.model)
    engine.model(mx.array(tokens[:-1])[None], cache=twin_cache)
    mx.eval([c.state for c in twin_cache])
    twin, logits = _twin_generate(engine, tokens[-1:], twin_cache)

    assert resumed == twin, (
        "round-trip is not faithful to the producer state:\n"
        f"  resumed={resumed[:8]}\n  twin={twin[:8]}"
    )
    _witness_scratch(resumed, scratch, logits, scratch_recorder.decode_logits)


def test_adapter_prefill_resume_matches_scratch(engine):
    """Drives the REAL adapter path (MlxLmEngine.prefill -> generate_from_blob),
    not raw mlx-lm. This is what the daemon actually calls on a miss then a hit:
    the earlier convention bug (seeding the cache with all N tokens while the
    resume path fed tokens[cached-1:]) made every real hit double-feed the last
    token and diverge. This test would have caught it.

    The equality target is the producer twin (the identical
    _prefill_cache(tokens[:-1]) state, memory-resident), not a fresh batch
    prefill: the two kernel shapes disagree at zero-margin argmax steps
    per-device (issue #27). Scratch remains a margin-guarded canary."""
    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlxcache_sidecar.server import read_wire_checkpoint

    # The twin premise (blob == producer state) holds only unquantized; a
    # quantized run of this module is a supported config that these twin
    # comparators do not cover.
    if engine.kv_bits != 0:
        pytest.skip("twin parity is defined for f16; real-model quantized parity is uncovered")

    tokens = engine.tokenize("The quick brown fox jumps over the lazy dog. " * 4)

    # 64 generated tokens: see the oracle-width note in the raw test above.
    scratch_recorder = _RecordDecode(engine.model)
    engine.model = scratch_recorder
    try:
        scratch = [
            r.token
            for r in stream_generate(
                scratch_recorder,
                engine.tokenizer,
                prompt=mx.array(tokens),
                max_tokens=64,
            )
        ]
    finally:
        engine.model = scratch_recorder.model

    # Miss path: the adapter produces the raw KV payload the daemon persists.
    payload = engine.prefill(tokens)
    blob = _blob_bytes(engine, tokens, payload)

    with TempSafetensors() as path:
        with open(path, "wb") as fh:
            fh.write(blob)
        # The gate is vacuous if the resume silently fell back to a scratch
        # prefill: prove the checkpoint is consumable for this request.
        _, _, usable = read_wire_checkpoint(path, tokens)
        assert usable, "checkpoint unexpectedly unusable; the resume would fall back to scratch"
        # Hit path: resume from the persisted blob exactly as the daemon does,
        # with the spy proving the loaded cache was actually adopted.
        with _ResumeSpy(engine) as resume_spy:
            resumed = engine.generate_from_blob(tokens, path, max_tokens=64)
        assert resume_spy.adopted, "resume fell back to a scratch prefill"

    # The twin: the identical producer prefill, memory-resident. The payload
    # at T covers T[:-1], so the resume convention feeds tokens[len-1:].
    twin_cache, _ = engine._prefill_cache(tokens[:-1])
    twin, logits = _twin_generate(engine, tokens[len(tokens) - 1 :], twin_cache)

    assert resumed == twin, (
        "adapter resume diverged from its producer twin "
        "(round-trip or feed-convention defect):\n"
        f"  resumed={resumed[:8]}\n  twin={twin[:8]}"
    )
    _witness_scratch(resumed, scratch, logits, scratch_recorder.decode_logits)


def test_adapter_delta_prefill_matches_scratch(engine):
    """OV3 parity on the REAL engine: generation resumed from a blob whose KV
    was built by DELTA prefill (ancestor KV + only the uncovered tokens) must
    match its producer twin — the delta-prefilled state held in memory,
    before the second round-trip. This is the shipped fast path for a growing
    conversation; if adopting ancestor KV corrupted state, every partial hit
    would be silently wrong. Scratch remains a margin-guarded canary."""
    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlxcache_sidecar.server import read_wire_checkpoint

    if engine.kv_bits != 0:
        pytest.skip("twin parity is defined for f16; real-model quantized parity is uncovered")

    base = engine.tokenize("The quick brown fox jumps over the lazy dog.")
    grown = engine.tokenize("The quick brown fox jumps over the lazy dog. " * 4)
    assert len(grown) > len(base) and grown[: len(base)] == base, (
        "test premise: the grown prompt must extend the base token-for-token"
    )

    scratch_recorder = _RecordDecode(engine.model)
    engine.model = scratch_recorder
    try:
        scratch = [
            r.token
            for r in stream_generate(
                scratch_recorder,
                engine.tokenizer,
                prompt=mx.array(grown),
                max_tokens=64,
            )
        ]
    finally:
        engine.model = scratch_recorder.model

    # Step 1: publish the ancestor (full prefill of the short prompt).
    ancestor_payload = engine.prefill(base)
    with TempSafetensors() as ancestor_path:
        with open(ancestor_path, "wb") as fh:
            fh.write(_blob_bytes(engine, base, ancestor_payload))

        # Step 2: delta prefill of the grown prompt from the ancestor.
        delta_payload = engine.prefill(grown, ancestor_blob_path=ancestor_path)
        assert engine.last_prefill_delta_tokens < len(grown) - 1, (
            "delta prefill recomputed the whole prompt "
            f"({engine.last_prefill_delta_tokens} of {len(grown) - 1} steps)"
        )
        with TempSafetensors() as delta_path:
            with open(delta_path, "wb") as fh:
                fh.write(_blob_bytes(engine, grown, delta_payload))
            # The gate is vacuous if the resume silently fell back to scratch.
            _, _, usable = read_wire_checkpoint(delta_path, grown)
            assert usable, "delta checkpoint unexpectedly unusable"
            # Step 3: resume generation from the delta-prefilled blob, with
            # the spy proving the loaded cache was actually adopted.
            with _ResumeSpy(engine) as resume_spy:
                resumed = engine.generate_from_blob(grown, delta_path, max_tokens=64)
            assert resume_spy.adopted, "resume fell back to a scratch prefill"

        # The twin: the delta-prefilled producer state, memory-resident. Adopt
        # the ancestor payload and feed the uncovered tail exactly as prefill()
        # does, minus the second round-trip. Runs INSIDE the ancestor's
        # temp-file context: the ancestor bytes back both the delta prefill
        # and the twin.
        ancestor_meta, ancestor_payload2, ancestor_usable = read_wire_checkpoint(
            ancestor_path, grown
        )
        assert ancestor_usable, "ancestor checkpoint unexpectedly unusable for the twin"
        twin_cache = engine._load_cache_from_payload(ancestor_payload2)
        covered = len(ancestor_meta.tokens) - 1
        engine.model(mx.array(grown[covered:-1])[None], cache=twin_cache)
        mx.eval([c.state for c in twin_cache])

    # The composition gate: twin and resumed share the ancestor round-trip,
    # so twin equality alone cannot see an ancestor-adopt defect. The
    # delta-adopted state must also sit within kernel-path noise of a fresh
    # full prefill of the same prompt: measured composition noise on this
    # lineage is 0.5 max-abs, so a composition defect (wrong KV, wrong feed
    # position) lands well above the 1.0 bound. Computed BEFORE the twin
    # generates — generation extends the cache and would break the shape
    # pairing.
    fresh_cache, _ = engine._prefill_cache(grown[:-1])
    state_diff = _state_max_diff(twin_cache, fresh_cache)
    assert state_diff < 1.0, (
        "delta-adopted KV diverged from a fresh full prefill by "
        f"{state_diff:.4g} — the OV3 composition is corrupt"
    )

    # Generation extends twin_cache past the compared prefix, so the twin
    # runs after the state gate.
    twin, logits = _twin_generate(engine, grown[len(grown) - 1 :], twin_cache)

    assert resumed == twin, (
        "delta-prefill resume diverged from its producer twin "
        "(round-trip or feed-convention defect):\n"
        f"  resumed={resumed[:8]}\n  twin={twin[:8]}"
    )
    _witness_scratch(resumed, scratch, logits, scratch_recorder.decode_logits)


def test_adapter_divergent_resume_matches_scratch(engine):
    """T22 parity on the REAL engine: a request that shares the first
    len(key)-1 tokens with the published key (divergence AT the key's last
    token) must resume from the ancestor's blob, and the blob's round-trip
    must be faithful to the producer state that wrote it.

    The equality target is the resume's PRODUCER-LINEAGE TWIN, not a fresh
    batch prefill of the full request. A batch prefill runs different kernel
    shapes than the resume's delta feed, and the two disagree by up to ~0.3
    in the logits over bitwise-equal KV; at a zero-margin argmax step that
    noise decides the token, so raw token equality vs a scratch prefill
    flips per-device. The twin is the identical _prefill_cache(base[:-1])
    computation that wrote the blob, memory-resident: it differs from the
    resumed side ONLY by the f16 save/load round-trip, so any twin
    divergence is a real round-trip or feed-convention defect and is
    device-stable.

    The scratch run is still witnessed, softly — it is a canary, not the
    defect detector: sampled tokens may differ from the resume ONLY at a
    near-tie step. The twin's top-2 margin at the first divergence
    quantifies the tie; the acceptance band derives from the measured
    kernel-noise ceiling (±0.3 per side, so flips are noise-explainable up
    to ~0.6). At or above that band the resume produced something scratch's
    distribution does not support, and the test fails.
    """
    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlxcache_sidecar.server import read_wire_checkpoint

    # The twin premise (blob == producer state) holds only at f16: with KV
    # quantization on, the blob holds a QuantizedKVCache while the twin stays
    # f16, and the comparison stops isolating the round-trip (the T12 q8
    # tier has its own parity tests).
    if engine.kv_bits != 0:
        pytest.skip("twin parity is defined for f16; real-model quantized parity is uncovered")

    base = engine.tokenize("The quick brown fox jumps over the lazy dog. " * 40)
    assert len(base) > 64
    # T22 shape by construction: identical to the key through len(key)-2, then
    # the key's LAST token is replaced by three fresh tokens. LCP = len-1.
    divergent = base[:-1] + [2, 11, 5678]
    assert divergent[: len(base) - 1] == base[:-1] and divergent[len(base) - 1] != base[-1]

    scratch_recorder = _RecordDecode(engine.model)
    engine.model = scratch_recorder
    try:
        scratch = [
            r.token
            for r in stream_generate(
                scratch_recorder,
                engine.tokenizer,
                prompt=mx.array(divergent),
                max_tokens=64,
            )
        ]
    finally:
        engine.model = scratch_recorder.model

    # Publish the ancestor (the key), then resume the divergent request from it.
    payload = engine.prefill(base)
    with TempSafetensors() as path:
        with open(path, "wb") as fh:
            fh.write(_blob_bytes(engine, base, payload))
        # The gate is meaningless if the resume silently fell back to a
        # scratch prefill (the daemon contract is about the RESUMED path):
        # prove the checkpoint is consumable for this request before using it.
        _, _, usable = read_wire_checkpoint(path, divergent)
        assert usable, "checkpoint unexpectedly unusable; the resume would fall back to scratch"
        with _ResumeSpy(engine) as resume_spy:
            resumed = engine.generate_from_blob(divergent, path, max_tokens=64)
        assert resume_spy.adopted, "resume fell back to a scratch prefill"

    # The twin: the same producer prefill that wrote the blob, memory-resident.
    # Covered positions match the resume convention exactly (payload at T
    # covers T[:-1]; the delta is tokens[covered:]).
    twin_cache, _ = engine._prefill_cache(base[:-1])
    twin, logits = _twin_generate(engine, divergent[len(base) - 1 :], twin_cache)

    assert resumed == twin, (
        "divergent (T22) resume diverged from its producer twin:\n"
        f"  resumed={resumed[:8]}\n  twin={twin[:8]}"
    )
    _witness_scratch(resumed, scratch, logits, scratch_recorder.decode_logits)


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
