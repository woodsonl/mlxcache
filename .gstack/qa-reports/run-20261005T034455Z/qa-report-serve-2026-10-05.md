# QA Report — B1.2 mlxcache_serve wrapper (functional target)

- Target: `sidecar/mlxcache_serve/` public API + CLI seam
- Mode: Full, diff-aware (branch `b1.2/wrapper`, commit `bf36d1f`)
- Probe harness: `qa_probes_b12.py` — 7 documented contracts, fake codec
  (positional semantics guarded separately by the gated real-engine thesis
  test `tests/test_serve_real.py`).

## Contract outcomes

| # | Contract | Outcome |
|---|----------|---------|
| W1 | Successful execution (insert → persist → fresh-instance resume, correct feed math) | pass |
| W2 | Invalid/missing input (`--store-dir` required; usage names it) | pass |
| W3 | Authorization boundary (adapter switch isolates fingerprint — LoRA-B never serves LoRA-A KV) | pass |
| W4 | State transitions (matched=len(T) → rest is exactly the last token — the §3.2 double-feed guard) | pass |
| W5 | Partial-failure recovery (reuse failure → falls back to memory, counted, never raises) | pass |
| W6 | CLI process contract (`--model` reaches mlx-lm's argv verbatim: no abbreviation capture, no `--` leak; swap+restore) | pass |
| W7 | State across process (restart over the same store: payload + feed math intact) | pass |

Stability: full pass after two probe-harness fixes (probe-side only:
a stale `--model-id-override` reference after the flag's removal, and a
subprocess leg that made mlx-lm attempt an HF download — replaced with
in-process argv capture). **No product issues found.**

## Review disposition (this batch)

- 2 review waves: wave 1 (specialists + adversarial) — the CRITICAL §3.2
  write-convention inversion (post-generation state covers all of T) fixed
  by trim-one-before-save, gated by `test_serve_real.py`; CLI abbreviation
  + py3.14 `--` leak fixed with regression rows. Wave 2 dispositions:
  generation-thread stall = documented v1 posture (module docstring +
  serve.py); `--model-id-override` dropped (daemon interop not achievable —
  different fingerprint schemes); weights-overwrite staleness documented;
  kv-bits relabeled as tier partition (honest semantics).
- pytest 132 passed / 7 skipped (12 wrapper unit tests); cargo EXIT=0;
  ruff/clippy/fmt clean.

**QA found 0 product issues in 7 contracts.** Ship-ready.
