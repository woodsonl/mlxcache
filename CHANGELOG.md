# Changelog

All notable changes to mlxcache. Versions follow [SemVer](https://semver.org/).

## 0.1.7 — 2026-10-19

### Added

- **The chunk-granularity question is answered with primary evidence, and
  the answer is GO**: content-defined chunk dedup belongs in the blob
  store, with token-block/64 as the v1 codec (CDC/head/16 KiB as the
  fallback for adapters that cannot report position boundaries). The
  decision record, the pre-registered gates and every table that backs
  them live in `REPORT.md`, `docs/designs/mlx-kv-cache-daemon.md` and
  `scripts/dedup/`, measured on the real daemon and real mlx-lm sidecar,
  f16 and 8-bit KV tiers, 8-turn conversation over the pinned T22 corpus.
  Nothing in the daemon or sidecar runtime changes in this release; this
  ships the evidence and its tooling.
- **`scripts/dedup/` carries the full measurement pipeline**: capture
  harness (grow + single modes, per-turn manifest checkpointing), blob
  forensics (per-KV-head identity probes), CDC vs token-block chunker
  simulation (vectorized gear hash verified against the naive reference
  on every run), restore bench (packed and unpacked layouts), eviction
  simulation under the daemon's real anchor model (structural + recency),
  and the round-trip probe that isolated where KV byte identity actually
  breaks. Every committed table regenerates from a checked-in script.
- **CI runs the 52 dedup invariants** (`uv run pytest scripts/dedup/tests/`)
  after the python job's dev-group sync; numpy is now a pinned dev
  dependency (it previously entered the lock only via the mlx extra, so
  the step was a false green until fixed).

### Fixed

- **Eviction evidence now measures the real policy**: the simulation
  mirrors sidecar `store.py` §7 (non-anchors evicted token_count
  descending, over-budget holds by anchors) and `index.py`'s recency
  anchor window (`MLXCACHE_EVICT_ANCHOR_WINDOW_S`, default 900 s). An
  earlier revision omitted the recency term, read artificially aggressive
  eviction as policy evidence, and double-counted republished entries in
  the S3 scenario (an evicted entry re-entered the store ledger twice);
  both are fixed with regression tests, and all eviction tables were
  regenerated from a fresh capture. The adopted codec now has its own
  evidence arm: after the background reaper cascades and the chain is
  re-requested, the whole-blob store re-stores 2,494.8 MiB (f16) where
  the token-block chunk store re-stores 18.0 MiB.

### Removed

- Dead flexibility found during review: an unreachable eviction-ordering
  switch, a never-measured restore scope flag, an unused optional
  parameter, a dead cut-map return value, and an unreachable branch in
  `first_diff`.

## 0.1.6 — 2026-10-19

### Fixed

- **The sidecar fingerprints HF-cached models again**: every snapshot
  file in an HF cache is a symlink into a sibling `blobs/` directory, so
  the fingerprinter's blanket `O_NOFOLLOW` hashed nothing and every
  cached model read as `"missing"` — erasing tokenizer-identity pinning
  within each engine path. `_artifact_bytes` now follows a link only when
  its target resolves inside the model directory or its own `blobs/`
  storage (the two-level root trusted only in the real HF
  `snapshots/<rev>` shape); a link to another model's artifact or an
  arbitrary file is still refused. The opened file's `(st_dev, st_ino)`
  is reconciled against the checked path so an intermediate-directory
  swap mid-window fails closed, a directory artifact (a corrupt pull
  leaving the link at `blobs/` itself) reads as missing instead of
  raising, and the fd is owned end-to-end by its context manager (the
  mismatch path leaked it; a read error double-closed into an escaping
  EBADF). Swap detection is restart-scoped: the fingerprint is memoized
  per process. Seven tests pin the follow/refuse/missing corpus,
  including the corrupt-pull directory shape and the in-model link.

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
## Unreleased

### Added

- **`qwen4-exp` engine: Qwen3.8-Flash-Next served end to end**: a
  self-contained engine (`MLXCACHE_ENGINE=qwen4-exp`) for a local
  `Qwen3.8-Flash-Next` NVFP4 checkpoint (132.6 GB on disk) on one Apple
  Silicon machine. The architecture port, NVFP4 weight reader and PLE n-gram
  embedding live in this repo; nothing depends on another repository. The
  checkpoint's bundled MTP head and vision tower are not loaded: neither
  contributes cache state, and `weight_shapes()` enumerates 74,926 text
  tensors and zero from either family. Because the checkpoint cannot be resident on 48 GB,
  weights stream from disk under two residency knobs,
  `MLXCACHE_Q4_EXPERT_RESIDENT_N` (decoded experts per layer, default `0`) and
  `MLXCACHE_Q4_PLE_CACHE_MB` (decoded PLE row budget, default `512`); a budget
  that is too small costs decode speed, not correctness. Cache integration is
  the engine-generic path unchanged, so persistence, digests, eviction, delta
  resume and `engine_id` namespacing apply as-is. Residency accounting and
  troubleshooting are in `docs/ops.md`. Performance on the real checkpoint is
  recorded in the README's Measured section: boot 4.6 s, prefill 46.1 s at 256
  tokens to 76.6 s at 8K, decode 2.0-2.3 s/token to 2K. Decode at 8K was
  measured under 10 GB of swap and is flagged there as contaminated rather
  than quoted.
- **Hybrid-architecture models verified on the R1-5 gate**: mlx-lm's
  `save_prompt_cache`/`load_prompt_cache` are class-generic (cache classes are
  recorded in the blob and rebuilt via `from_state`), so the adapter's capture
  path needs no per-architecture code. Measured token-identical on
  `mlx-community/Falcon-H1-0.5B-Instruct-4bit`, whose every layer is a nested
  `CacheList(ArraysCache, KVCache)`. README documents the supported hybrid
  families and the boundary for models mlx-lm cannot load at all, which the
  self-contained `qwen4-exp` engine (above) serves instead.

### Fixed

- **The R1-5 state-diff helper handles nested cache state**: `_state_max_diff`
  assumed each entry's `.state` was a flat list of arrays; hybrid entries
  (`CacheList`, `ArraysCache`) nest lists (and `None` for unfilled slots), so
  the delta-prefill gate raised `TypeError` on every hybrid model. Now
  tree-recurses with `None` kept as an explicit leaf so shape mismatches stay
  visible. The bound also became per-leaf — `max(1.0, 25% of leaf scale)` —
  because hybrid state leaves differ in scale by orders of magnitude: on
  Falcon-H1-4bit the f32 SSM recurrent state (scale ~80) shows 9.7 max-abs of
  chunked-scan reassociation noise (11.7% relative) against the dense f16
  calibration of 1.0 absolute (measured 0.5). A composition defect replaces
  values wholesale (~100% relative) and still fails on every lineage; dense
  behavior is unchanged (floor governs, measured ratio 0.10).
- **The q8 KV tier refuses hybrid models at boot** instead of crashing on the
  first prefill: hybrid entries have no `to_quantized`, so
  `MLXCACHE_KV_BITS=8` against e.g. Falcon-H1 / Qwen3-Next / granite-hybrid
  died with `AttributeError` after prefill. A boot-time guard probes the cache
  composition (cheap: no arrays allocated) and exits with the fix
  (`MLXCACHE_KV_BITS=0`) in the message. Dense stacks are unaffected.

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
