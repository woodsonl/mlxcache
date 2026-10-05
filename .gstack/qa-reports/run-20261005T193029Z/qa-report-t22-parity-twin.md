# Functional QA Report: T22 parity-twin comparator (issue #27 fix)

| Field | Value |
|---|---|
| Date / branch / revision | 2026-10-05 / t22/parity-twin / b99a425 + review-fix 05df264 (+ wording nit) |
| Caller / authority / depth | batch gauntlet (implement → ponytail → review+adversarial → QA → ship → land); permitted writes: repo source + report dir |
| Surfaces / scope | `sidecar/tests/test_roundtrip_real.py::test_adapter_divergent_resume_matches_scratch`, the weekly realmodel gate's T22 guard |
| Runtime / native tools | Python 3.14 (.venv), mlx-lm 0.31.x, Qwen2-0.5B-Instruct (fresh snapshot), Apple Silicon; `MLXCACHE_BENCH_REAL=1 uv run pytest ...::test_adapter_divergent_resume_matches_scratch` |
| Fixture ownership / destinations | fresh HF snapshot (~500MB); no shared state; 0.5B loads announced to the peer session |
| Probe budget / stop reason | 3+2 green reps, 2 mutation-red runs, full hermetic gates; complete |

## Contract outcomes

| Contract and source | Exact probe / evidence | Expected → observed | Outcome |
|---|---|---|---|
| The twin comparator passes on correct bytes | `twin-verify.log` rep1-3, `twin-verify2.log` rep1-2 | 5/5 PASS | pass |
| The comparator catches feed-convention bugs (teeth) | covered+1 mutation in both verify logs | twin diverges → assert fires → "MUTATION CAUGHT" | pass |
| The old comparator's failure mode is real (the red) | issue #27: 2/2 CI runner runs byte-identical (index 3, 220 vs 576); 6/6 local passes on the old bytes | documented red | pass |
| Hermetic gates unaffected | cargo test ✓ (146), hermetic pytest 149/7 ✓, fmt ✓, clippy -D warnings ✓, ruff ✓ | all green | pass |
| Review findings fixed | 5 review + 2 adversarial + 1 delta finding | all addressed in 05df264 + wording commit; delta pass verified all 6 checkpoints | pass |

## The defect and the fix

- **Symptom**: the weekly realmodel job failed on `main`: T22 divergent-resume, byte-identically, on every runner run.
- **Root cause** (confirmed by logits census, issue #27): the old comparator asserted raw sampled-token equality between the resumed generation and a fresh batch prefill. Those paths run different kernel shapes over bitwise-equal KV; logits differ by ≤0.3 max-abs; at a zero-margin argmax step (resumed top-2 gap exactly 0 at step 2) the noise decides the token, deterministically per device.
- **Fix**: equality target = the resume's producer-lineage twin (the identical `_prefill_cache(base[:-1])` computation that wrote the blob, memory-resident). Twin divergence = real round-trip or feed-convention defect, device-stable. The scratch run remains as a margin-guarded canary (divergence permitted only under the 0.6 two-sided noise band derived from the measured ±0.3 per-side ceiling).
- **Guards added during review**: checkpoint-usable assertion (no vacuous scratch fallback), f16 pin (the twin premise is f16-only), EOS-safe witness wording.

## Coverage limits

- The three remaining resumed-vs-scratch comparators in the same file are flip-capable by the same mechanism — tracked as issue #28 for incremental conversion with their own mutation checks; deliberately out of this batch.
- The 0.6 band's tail risk (legit noise ≥0.6) is unobserved in 8 measured runs; the twin equality remains the primary detector, so a false canary failure cannot mask a real defect.
