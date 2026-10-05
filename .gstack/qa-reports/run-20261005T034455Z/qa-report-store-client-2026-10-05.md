# QA Report — B1.1 L1 store client (functional target)

- Target: `sidecar/mlxcache_store/` public API (connector protocol §3/§6/§7/§8)
- Mode: Full, diff-aware (branch `b1.1/store-client`, commit `1d734b2`)
- Probe harness: `qa_probes_b11.py` (this directory) — 11 documented contracts,
  isolated tmpfs setup per probe, real subprocesses for restart/concurrency rows.
- Gates at probe time: pytest 120P/6S (36 store), cargo test EXIT=0, ruff clean,
  clippy clean, fmt clean.

## Contract outcomes

| # | Contract | Outcome |
|---|----------|---------|
| P1 | Successful execution (put/lookup/fetch round trip, durable file) | pass |
| P2 | Invalid/missing input (sub-2 Refused, traversal ValueError, zero partial state) | pass |
| P3 | Authorization boundary (cross-engine isolation, no payload crossing) | pass |
| P4 | State transitions (WHOLE_CONTEXT exact-only + ANY_PREFIX fallback) | pass |
| P5 | Matching semantics §3.3 (end-anchored divergence serve) | pass |
| P6 | Duplicates/idempotency (re-put supersede → exactly 1 durable file, newest serves) | pass |
| P7 | Concurrency/order (4 real processes, contended + private keys; 5 survivors, all digest-valid) | pass |
| P8 | Partial-failure recovery (aged tmp swept, poison reclaimed, digest-flip reclaimed, open never raises) | pass |
| P9 | State across process (restart in a separate interpreter: lookup + payload intact) | pass |
| P10 | Resource bound §7 (16 puts over 8 KiB budget → directory ≤ budget + 1 blob) | pass |
| P11 | Declared rejections §3.4 (vanished → Unavailable; tamper → CorruptCache(digest) + self-heal) | pass |

Stability: 3 consecutive full runs, 11/11 each. (One probe-harness bug fixed during
the run: P7's child script short-circuited its second put via `or` — probe defect,
not product; corrected and re-run clean.)

## Verdict

**QA found 0 product issues in 11 contracts.** All review-wave fixes hold under
live probes. Ship-ready.

Known leftovers (documented, non-blocking):
- P7/P9 probe dirs use mkdtemp and persist in $TMPDIR until OS cleanup (harness only).
- Dispositioned review items: foreign-version reclaim (matches daemon quarantine,
  spec-intent question logged), cross-process supersede fully closed at open-dedupe
  (not per-put), budget held by anchors/unindexed files = hold-and-continue per §7.
