# Changelog

All notable changes to mlxcache. Versions follow [SemVer](https://semver.org/).

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
