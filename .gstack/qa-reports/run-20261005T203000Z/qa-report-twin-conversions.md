# Functional QA Report: twin-conversion of the remaining parity comparators (issue #28)

| Field | Value |
|---|---|
| Date / branch / revision | 2026-10-05 / parity/twin-conversions / eae5693 + review-fix d33d219 |
| Caller / authority / depth | batch gauntlet (implement → ponytail → review+adversarial → QA → ship → land); permitted writes: repo source + report dir |
| Surfaces / scope | `sidecar/tests/test_roundtrip_real.py`: test_roundtrip_logits_identical, test_adapter_prefill_resume_matches_scratch, test_adapter_delta_prefill_matches_scratch converted to the producer-twin methodology; T22 deduped onto the shared helpers |
| Runtime / native tools | Python 3.14 (.venv), mlx-lm 0.31.x, Qwen2-0.5B-Instruct, Apple Silicon; `MLXCACHE_BENCH_REAL=1 uv run pytest sidecar/tests/test_roundtrip_real.py` |
| Fixture ownership / destinations | fresh HF snapshot; 0.5B loads announced to the peer session |
| Probe budget / stop reason | suite ×4 green runs across fix waves, 3 mutation checks ×2 runs, state-ceiling measurement, spy mutation check; complete |

## Contract outcomes

| Contract and source | Exact probe / evidence | Expected → observed | Outcome |
|---|---|---|---|
| All four parity tests green on honest bytes | `verify-wave1.log` (6/6 ×2) and `verify-final.log` (6/6 ×2) on the final bytes; `verify-wave2.log` records the intermediate 1-failed state that caught the state-gate ordering bug (shape mismatch), superseded by d33d219 | 6/6 PASS on the final bytes, twice | pass |
| Each comparator catches feed-convention corruption | `twin-mutations.py` (feed-content corruption): raw / prefill / delta | resumed ≠ twin on mutated bytes, all three → MUTATION CAUGHT | pass |
| The comparator catches wrong-blob round-trips (raw layer) | `mutations.log` (script `twin-mutations.py`, raw section): wrong prompt's cache saved into the blob path | resumed ≠ twin → RAW MUTATION CAUGHT | pass |
| The resume actually adopts the loaded cache | `mutations3-spy-ceiling.log` (script committed as `mutations3-spy-ceiling.py`): honest adopted=True; class-level fallback mutation adopted=False | fallback detected → VACUOUS-RESUME MUTATION CAUGHT | pass |
| The OV3 composition is state-faithful | composition state ceiling measured 0.5 max-abs on honest bytes (bound 1.0) | gate green with 2× headroom | pass |
| Hermetic gates unaffected | cargo ✓ (9 suites), hermetic pytest 149/7 ✓, fmt ✓, clippy -D warnings ✓, ruff ✓ | all green | pass |

## What changed and why

The three remaining resumed-vs-scratch comparators asserted bit-exact
sampled-token equality across kernel provenances — the comparator shape
that fails per-device at zero-margin argmax steps (issue #27). Each now
asserts against its producer twin (the identical producer prefill,
memory-resident) and keeps scratch as a margin-guarded canary. Review added
the composition state gate for the delta lineage (twin and resumed share the
ancestor round-trip, so the adopted state is additionally compared against a
fresh full prefill), the resume spy (a silent scratch fallback previously
stayed green), the dual-margin witness (a flip is licensed by the min of
both lineages' margins), and kv_bits skips (a quantized run is a supported
config, not a failure).

## Findings disposition

Full review (1 low) + adversarial (2 high, 2 medium, 1 low) + delta
certification (0) — all six substantive findings fixed in d33d219 and
verified by the delta pass with empirical checks. The pr29 deploy report on
this branch is documented carry-over (no home existed post-squash); this
report is the batch's own evidence record.

## Coverage limits

- The measured noise ceilings (±0.3 logits, 0.5 state composition) are
  0.5B/this-device numbers; other devices may differ but the twin equality
  — the defect detector — is device-stable by construction.
- The double-feed mutation is documented as a phase-aligned no-op on a
  periodic prompt; feed-content corruption is the discriminator. The
  double-feed bug class itself is additionally self-healing on mlx-lm 0.31.x
  (over-feed trimmed against the cache prefix); the synthetic convention
  tests remain the convention pin.
