import sys
import tempfile

import mlx.core as mx

from mlxcache_sidecar.server import MlxLmEngine, read_wire_checkpoint
from mlxcache_sidecar.blob import CheckpointMeta, Fingerprint, encode

engine = MlxLmEngine("Qwen/Qwen2-0.5B-Instruct")


def blob(toks, pay):
    meta = CheckpointMeta(
        fingerprint=Fingerprint(
            model_id=engine.model_id,
            tokenizer_hash=engine.tokenizer_hash,
            kv_dtype=engine.kv_dtype,
            kv_layout_version=1,
        ),
        token_count=len(toks),
        tokens=toks,
    )
    return encode(meta, pay)


def state_diff(a, b):
    worst = 0.0
    for ca, cb in zip(a, b, strict=True):
        for sa, sb in zip(ca.state, cb.state, strict=True):
            worst = max(worst, float(mx.abs(sa - sb).max().item()))
    return worst


# --- delta lineage: composition ceiling (validates the 1.0 state bound) ---
base = engine.tokenize("The quick brown fox jumps over the lazy dog.")
grown = engine.tokenize("The quick brown fox jumps over the lazy dog. " * 4)
ap = tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False)
ap.close()
open(ap.name, "wb").write(blob(base, engine.prefill(base)))
dp = tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False)
dp.close()
open(dp.name, "wb").write(blob(grown, engine.prefill(grown, ancestor_blob_path=ap.name)))
ameta, apayload, ausable = read_wire_checkpoint(ap.name, grown)
assert ausable
twin3 = engine._load_cache_from_payload(apayload)
cov = len(ameta.tokens) - 1
engine.model(mx.array(grown[cov:-1])[None], cache=twin3)
mx.eval([c.state for c in twin3])
fresh, _ = engine._prefill_cache(grown[:-1])
d = state_diff(twin3, fresh)
print(f"delta composition state ceiling: {d:.4g} (bound 1.0)")
assert d < 1.0, "composition noise exceeds the state bound on honest bytes"

# --- vacuous-resume mutation: fallback must be caught by the spy ---
orig = MlxLmEngine._load_cache_delta


def fallback(self, t, p):
    return None, list(t)


# spy behavior mirror (same contract as the test's _ResumeSpy: bind whatever
# the resume path currently is, record whether it adopted a cache)
def spy_check(engine, tokens, blob_path):
    adopted = None
    original = engine._load_cache_delta

    def _spied(t, p):
        nonlocal adopted
        c, d = original(t, p)
        adopted = c is not None
        return c, d

    engine._load_cache_delta = _spied
    try:
        engine.generate_from_blob(tokens, blob_path, max_tokens=8)
    finally:
        del engine._load_cache_delta
    return adopted


# honest: adopted
a1 = spy_check(engine, grown, dp.name)
print("spy honest: adopted =", a1)
# mutated: fallback fires
MlxLmEngine._load_cache_delta = fallback
a2 = spy_check(engine, grown, dp.name)
MlxLmEngine._load_cache_delta = orig
print("spy mutated (fallback): adopted =", a2, "(must be False)")
assert a1 is True and a2 is False
print("VACUOUS-RESUME MUTATION CAUGHT")
