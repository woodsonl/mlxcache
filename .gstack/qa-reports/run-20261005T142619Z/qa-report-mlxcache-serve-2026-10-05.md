# Functional QA Report: mlxcache_serve wrapper + harness scripts

| Field | Value |
|---|---|
| Date / branch / revision | 2026-10-05 / b2.3/retro-gauntlet / 7f7e4c3 + review-fix working tree (readiness + version-agreement tests, stop() guard, report corrections) |
| Caller / authority / depth | /gstack-qa (full gauntlet); permitted writes: repo source + report dir; bound: this branch |
| Surfaces / scope | HTTP API of `mlxcache_serve` (mlx_lm.server wrapper): /v1/chat/completions (stream + non-stream), /v1/completions, /health, error shape, disk+memory KV tiers; the two harness scripts under test (`scripts/qa_wrapper.py`, `scripts/bench_wrapper.py`) |
| Runtime / native tools | Python 3.14 (.venv), mlx-lm 0.31.x, real model `mlx-community/Qwen2.5-7B-Instruct-4bit` (local snapshot c26a38f6…), Apple Silicon. Commands: `.venv/bin/python scripts/qa_wrapper.py`, `.venv/bin/python scripts/bench_wrapper.py --max-tokens 16` |
| Fixture ownership / destinations | Isolated `tempfile.mkdtemp` store per run; no shared state; single GPU-resident model, announced to the peer session per plan §7b |
| Probe budget / guarded command time / stop reason | 7 QA probes + 5 bench legs, two independent runs (pre- and post-hardening). Warmup 1-token gate; request timeout 600s; probe budget not exhausted — complete |

## Contract outcomes

**Fix wave under review — `/gstack-review` + adversarial over `origin/main..HEAD` (commit e649f87) found 3 CRITICAL defects; all fixed in e82ea62 and re-probed here.**

| Contract and source | Exact probe / evidence | Expected → observed | Outcome |
|---|---|---|---|
| §D6 Q1 chat non-stream (protocol spec) | `scripts/qa_wrapper.py` Q1 | 200 + message content + finish_reason → PASS | pass |
| §D6 Q2 chat streaming == non-stream (temp 0) | Q2: SSE chunk parse, `object=chat.completion.chunk` | concatenated stream == non-stream content → PASS | pass |
| §D6 Q3 completions non-stream | Q3 | 200 + non-empty text → PASS | pass |
| §D6 Q6 error shape | Q6: `{"messages": "not-a-list"}` | 4xx + JSON error object → PASS | pass |
| §D6 Q4 restart mid-suite, grown conversation | Q4: SIGTERM, fresh process, same store, grown convo | 200 → PASS | pass |
| §D6 Q5 disk-resume identity (second fresh process) | Q5: third fresh process, same grown convo | answer == Q4 answer → PASS | pass |
| §D6 Q7 store hygiene | Q7: fetch every `*.ckpt` (digest-verified) | ≥1 checkpoint, all digest-valid → PASS | pass |
| §D5 bench gate: disk legs token-identical to MEMORY reference | `scripts/bench_wrapper.py` restart-turn2 / second-turn2 vs memory-turn2 | all legs `token-identical` → PASS | pass |
| §D5 bench gate: ≥90% prefill skip | store coverage of the turn-2 stream (281 computable positions) | 262/281 = 93% (disk_covered=262 of request_tokens=282) → PASS | pass |
| Changelog: warmup status is load-bearing (e82ea62) | `sidecar/tests/test_harness_readiness.py::test_qa_warmup_non_200_is_not_ready` | retries, raises naming the 500 (was: read as ready) → PASS | pass |
| Changelog: warmup bounded by deadline (e82ea62) | `test_bench_warmup_failure_is_bounded_and_reported` | bounded timeout ≤ 30s, retry, raise ≤ deadline (was: 600s block) → PASS | pass |
| Changelog: transient warmup retries (e82ea62) | `test_bench_warmup_retries_then_succeeds` | returns after 2 calls → PASS | pass |
| Changelog: temp log unlinked (e82ea62) | `test_stop_unlinks_temp_log[qa_wrapper|bench_wrapper]` | log file removed → PASS | pass |
| Fingerprint resolves per-process cwd (e649f87, the gate-breaker) | `test_resolve_model_abs_and_hfid_agree` / `test_resolve_model_rejects_uncached_id` | hfid and abs both → same absolute snapshot path → PASS | pass |
| Version sources agree (VERSION / Cargo / pyproject) | `sidecar/tests/test_version_agreement.py` (new in this batch) | all three == 0.1.0 → PASS | pass |

## Findings

### ISSUE-001: qa_wrapper readiness accepted a failing warmup (RESOLVED)

- Classification / severity: PRODUCT DEFECT / high. A broken server was declared healthy, then every probe failed with misleading downstream errors instead of the real cause.
- Intended contract and source: a readiness gate must mean "the model is loaded and serving", not "the port is bound" (`/health` binds before the lazy model load).
- Reproduction: mock `Server.post` returning `(500, {...})`; call `wait_ready(timeout_s=2)`.
- Observed (before): returned "ready" immediately. Observed (after): retries until deadline, raises `wrapper never became healthy (last warmup error: warmup HTTP 500: ...)`.
- Evidence: `scripts/qa_wrapper.py` `wait_ready`; `sidecar/tests/test_harness_readiness.py::test_qa_warmup_non_200_is_not_ready`.
- Diagnosis / next action: fixed in e82ea62 — the warmup status is checked and non-200 means not-ready.

### ISSUE-002: readiness warmup was unbounded (RESOLVED)

- Classification / severity: PRODUCT DEFECT / medium. A hung model-load thread blocked the readiness loop for 600s (the warmup's own timeout) while the readiness deadline was 180s.
- Intended contract and source: readiness must honour its own deadline.
- Reproduction: mock warmup that sleeps to its timeout; `wait_ready(timeout_s=2)`.
- Observed (before): could block up to 600s. Observed (after): warmup timeout `= max(1, min(remaining, 30))`; raises within ~2s.
- Evidence: both `wait_ready` implementations; mock transcript.
- Diagnosis / next action: fixed in e82ea62 — the warmup timeout is derived from the remaining deadline.

### ISSUE-003: targeted snapshot resolution could diverge from mlx_lm (RESOLVED)

- Classification / severity: PRODUCT DEFECT / low. `_resolve_model` picked `sorted(glob)[-1]` (lexicographic commit-hash order), unrelated to the `refs/main` revision mlx_lm fetches when several snapshots are cached.
- Intended contract and source: the bench and the server child must load the same revision.
- Reproduction: cache with >1 snapshot per repo (not present in this environment, so no live red).
- Observed (after): resolves via `huggingface_hub.snapshot_download(..., local_files_only=True)` (honours `refs/<rev>`), glob only as fallback.
- Evidence: `bench_wrapper._resolve_model`, `qa_wrapper._resolve_model`.
- Diagnosis / next action: fixed in e82ea62.

### ISSUE-004: per-server temp logs leaked (RESOLVED)

- Classification / severity: PRODUCT DEFECT / low. 3 temp files per run, plus fd+file on a failed `Popen`.
- Observed (after): `stop()` unlinks; spawn failure closes and unlinks. Mock: log removed.
- Evidence: `Server.stop` / `Server.__init__` in both scripts.
- Diagnosis / next action: fixed in e82ea62.

## Discoveries and permanent tests

| Hypothesis / discovery | Native test or proposed case | Red evidence before repair | Green + original + adjacent evidence | Parent disposition |
|---|---|---|---|---|
| The §D5 gate could not pass with a default (non-absolute) model: parent and child realpath the model differently → fingerprint mismatch → 0% skip | `_resolve_model` unit probe (hfid vs abs both → same path) | adversarial live repro `EQUAL: False` | both resolve to c26a38f6…; bench GATE PASS | authorized change (e649f87) |
| The turn-2 "memory reference" was disk-vs-disk (fresh server over same store) | bench memory-turn2 in the SAME process as turn 1 | reference took the disk path | memory-turn2 leg now in-process; disk legs match it | authorized change (e649f87) |
| Readiness warmup must check status and honour the deadline | `sidecar/tests/test_harness_readiness.py` (new in this batch) | qa 500 read as ready; unbounded warmup (600s) | all readiness tests PASS | authorized change (e82ea62) |
| KV-parity reference must share the producer lineage | prior learning `kv-parity-reference-shares-producer-lineage` | n/a (applied) | memory-twin reference now governs the gate | applied, confidence 10/10 |

## Coverage limits and cleanup

- **Browser surface:** none. This target is a headless HTTP server; no browser score applies. The gstack QA `sections/*.md` files referenced by the skill are absent in this install (only `references/` + `templates/` present) — degraded mode; functional path applied with safe defaults.
- **Not run:** no explicit malformed-SSE or partial-chunk delivery test beyond the buffered line iteration already covered; no concurrent-request test (bench is sequential by design).
- **Environment:** the 0.5B HF cache is locally broken (dangling symlinks), so all live runs used the 7B snapshot; the readiness fixes would surface that failure cleanly (ISSUE-001).
- **Cleanup:** all `mlxcache_serve` children reaped; temp logs unlinked; store dirs under `tempfile.mkdtemp`. No process left resident; GPU released to the peer.
- **Invalidation:** the pre-hardening GPU run (validated e649f87) is superseded by the post-hardening run (validated e82ea62); only the latter is current evidence.
