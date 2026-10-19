# Changelog

All notable changes to mlxcache. Versions follow [SemVer](https://semver.org/).

## 0.1.5 — 2026-10-19

### Fixed

- **Benchmark gates can no longer pass on warmup traffic**: the wrapper
  harnesses' readiness probe used a constant prompt, which the first
  process published and every later process disk-hit during warmup —
  inflating the cumulative `disk_hits` counters before any measured
  request ran. Probes are now unique per spawned server, and both wrapper
  gates (bench `_gate_ok`, qa Q4/Q5) require the `disk_hits` DELTA across
  each measured leg with zero `reuse_errors`, failing loudly when the
  stats endpoint omits a counter (`stats_delta`, shared in
  `scripts/_harness.py`).
- **`bench_workloads.py` measures what it claims**: TPOT divides by the
  tokens actually generated (a constant divisor was a lie the moment a
  workload changed `max_tokens` or the model stopped early); p95 is
  withheld below a 100-sample floor instead of wearing a percentile
  label; concurrent workers record and re-raise thread failures instead
  of silently shrinking the sample; bursts use ceiling division so
  `--requests 100 --concurrency 8` yields n=104, not 96; invalid numeric
  args are rejected at parse. The synthetic-engine harness is also
  relabeled as a plumbing check — its 8-token "prefill" can never express
  prefix-cache benefit.
- **A cold checkout no longer breaks the workloads harness**: the release
  daemon is built to completion BEFORE any spawn (a cold build used to
  run inside the 30s readiness window and fail as "not healthy" while the
  compiler was still running); the sidecar port is picked at bind, the
  sidecar is proven listening before the daemon spawns, both children run
  with the repo as cwd, and any post-spawn failure SIGTERMs what already
  runs instead of stranding it.
- **`wait_ready`/`wait_healthy` cannot certify a server they did not
  start**: both recheck child liveness after a successful probe (outside
  the retry's `except`), so a stale listener on an explicit port cannot
  be read as ready; a dead daemon fails fast with its exit code.

### Added

- **`scripts/_harness.py`**: the spawn/readiness/teardown rules shared by
  `bench_wrapper`, `qa_wrapper` (and the prefix-benchmark unit of this
  stack) live in one place — SIGTERM→SIGKILL escalation with log unlink,
  per-child-pid warmup prompts, a monotonic SSE total deadline, and
  local-only model resolution that never downloads.
- 30 premise-guard tests across `test_harness_readiness.py` and
  `test_bench_workloads.py` pinning every failure path above.

## 0.1.4 — 2026-10-05

### Fixed

- **The t22 multi-turn harness works end to end again**: the warm leg
  never collected the replies it exists to produce (empty
  `.replies.json`, the cold leg's parity assert could never pass) —
  pre-existing since before 0.1.0. Fixed and verified live: warm writes
  three nonempty replies, cold leg passes parity. The harness documents
  that MLXCACHE_DEMO_MODEL must name a model the daemon serves.
- Lint findings in the legacy verify/demo scripts (unused loop variable,
  two long-lived JSONL handles that close at function end, docstring
  line lengths).

### Added

- **CI lints the gate-critical scripts**: the python job's ruff check and
  format check now cover `scripts/` alongside `sidecar/`, so a lint
  regression in the bench/QA harnesses is visible in CI, not only locally.
- **Stranger-clone dry-run evidence**: a fresh clone at the batch HEAD
  builds, and the README quickstart passes verbatim (miss then hit, from
  the clone's own servers, on pre-flight-verified ports) — the repo is
  self-contained.

## 0.1.3 — 2026-10-05

### Fixed

- **The e2e sidecar spawn race is closed** (issue #31). The harness picked
  sidecar ports with portpicker and trusted a health answer while the child
  was alive; a foreign server answering health inside that window died with
  its own test and the daemon 503'd a sidecar that was never ours (observed
  twice: the 2026-10-04 two_models contamination and this week's 1-in-8
  503). The child now binds a kernel-assigned port and prints it, so the
  URL derives from our own child's bind — no foreign server can know it.
- **Every spawn failure is loud**: every spawn-failure path panics with cause
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
