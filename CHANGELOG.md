# Changelog

All notable changes to mlxcache. Versions follow [SemVer](https://semver.org/).

## 0.1.3 — 2026-10-05

### Fixed

- **The e2e sidecar spawn race is closed** (issue #31). The harness picked
  sidecar ports with portpicker and trusted a health answer while the child
  was alive; a foreign server answering health inside that window died with
  its own test and the daemon 503'd a sidecar that was never ours (observed
  twice: the 2026-10-04 two_models contamination and this week's 1-in-8
  503). The child now binds a kernel-assigned port and prints it, so the
  URL derives from our own child's bind — no foreign server can know it.
- **Every spawn failure is loud**: the four failure paths panic with cause
  (spawn error, missing/garbled PORT line, 30s deadline, health exhaustion,
  sidecar exit) instead of silently skipping all 27 e2e tests. The child is
  killed and reaped before every panic, and a stdout drain thread prevents
  a mid-test pipe wedge.

## 0.1.2 — 2026-10-05

### Fixed

- **The three remaining real-engine parity comparators are flip-capable no
  more** (issue #28). `test_roundtrip_logits_identical`,
  `test_adapter_prefill_resume_matches_scratch`, and
  `test_adapter_delta_prefill_matches_scratch` asserted raw sampled-token
  equality between a resumed generation and a fresh batch prefill — the
  comparator shape that fails per-device at zero-margin argmax steps
  (the mechanism behind the 0.1.1 T22 fix). Each now asserts against its
  producer twin (the identical producer prefill, memory-resident) and keeps
  scratch as a margin-guarded canary.

### Added

- **Composition state gate** for the delta-prefill test: the adopted state
  must sit within 1.0 max-abs of a fresh full prefill (measured composition
  noise 0.5), closing the ancestor-round-trip blind spot that twin equality
  alone cannot see.
- **Resume spy** on the three twin-converted adapter parity tests: a
  silent scratch fallback inside the engine's resume path now fails the
  test instead of passing vacuously. The one-token test keeps the raw
  comparator on purpose — its resume legitimately falls back to the
  empty-cache scratch path.
- **Dual-margin scratch witness**: a sampled-token flip is licensed by the
  minimum of both lineages' top-2 margins, so corruption on either lineage
  cannot manufacture its own near-tie license.
- Mutation evidence per test: feed-content corruption and wrong-blob
  round-trips are caught; the double-feed mutation is documented as a
  phase-aligned no-op on a periodic prompt.

## 0.1.1 — 2026-10-05

### Fixed

- **T22 divergent-resume gate no longer flips per-device** (issue #27). The
  weekly realmodel job failed byte-identically on every runner run while
  local machines passed: the old comparator asserted raw sampled-token
  equality between the blob-resumed generation and a fresh batch prefill,
  and at a zero-margin argmax step the kernel-path noise between those two
  shapes (≤0.3 max-abs over bitwise-equal KV) decided the token. The
  equality target is now the resume's producer twin, the identical prefill
  that wrote the blob held in memory, so any divergence is a real
  round-trip or feed-convention defect and is device-stable. The scratch
  run remains as a margin-guarded canary (flips permitted only under the
  0.6 two-sided noise band). Mutation-checked: a feed-convention
  off-by-one is caught; five green reps on correct bytes.

## 0.1.0 — 2026-10-05

First versioned release: the mlx-lm connector (Phases 0–2) — any mlx-lm server
gains persistent, bounded, digest-checked prefix-KV reuse with zero mlx-lm
patches.

### Added

- **Connector protocol v1.0** (`docs/connector-protocol.md`): blob format,
  fingerprint fold, §3.2 coverage convention (a blob at T covers T[:-1]),
  §3.3 extends-not-a-match, conformance suite.
- **Daemon** (`crates/`): persistent prefix KV cache with integrity
  (sha256-at-publish, boot sweep, quarantine), recency-anchored eviction,
  OpenAI-style streaming with a cache-verdict leading frame.
- **L1 store client** (`sidecar/mlxcache_store/`): the same directory format
  embeddable in any Python engine.
- **Wrapper server** (`sidecar/mlxcache_serve/`): mlx-lm's own server with a
  `PersistentPromptCache` swapped in — no daemon, no mlx-lm patches. Serves
  `GET /mlxcache/stats` (disk_hits/persisted/error counters) so reuse is
  observable, not assumed.
- **Harnesses** (`scripts/bench_wrapper.py`, `scripts/qa_wrapper.py`): a
  five-leg bench (§D5) whose gate requires token-identity against the
  in-process memory reference, ≥90% prefill skip, and an OBSERVED disk hit
  per disk leg; a seven-probe QA battery (§D6).
- **Real-engine gates**: R1-5 round-trip thesis (resumed == scratch),
  divergent-resume parity (T22), delta-prefill parity, one-token regression —
  all against real mlx-lm models, gated behind `MLXCACHE_BENCH_REAL=1`.
- **CI**: Rust + Python jobs on every PR; the real-model gate on a weekly
  schedule and on dispatch.
