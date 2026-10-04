# QA Report — mlxcache daemon + sidecar (functional)

- **Date:** 2026-10-04 · **Branch:** `woodsonl/mlx` @ `eb5b032`
- **Mode:** Standard (fix critical+high+medium) · **Tier target:** functional HTTP stack, diff-aware (fix wave `a247793` + adjacent)
- **Isolation:** owned ports (daemon :18450, sidecar :18460), temp blob dir + trace under `$TMPDIR/opencode/qa-btLc`, synthetic engine (no GPU). Real-engine behavior evidenced separately by the gated pytest run this session (6 passed, incl. the new divergent-resume parity test).
- **Evidence dir:** `.gstack/qa-reports/evidence-2026-10-04/` (probe JSON responses, daemon/sidecar logs, trace.jsonl 0600)

## Contract map

| # | Contract | Probe | Expectation | Outcome |
|---|----------|-------|-------------|---------|
| 1 | Miss → publish → hit | identical request ×2 | R2 hit, same tokens, 1 publish | **PASS** (R1 miss 8 tok → R2 hit 7/8, identical output; stats consistent) |
| 2 | Invalid input rejection | malformed JSON / missing model / unknown model | declared 4xx, no state change | **PASS** (400 / 422 / 404; stats byte-identical before/after) |
| 3 | Missing Content-Type | POST without header | clean rejection, no crash | **PASS** (415, explanatory body) |
| 4 | T22 divergent serve (last-token hinge) | turn-2 request diverging at key's last token | serve ancestor KV: `prefill_from = len(key)-1`, hit | **PASS** (T1 miss 9 tok → T2 hit, prefill_from 8, identical output) |
| 5 | Mid-prefix divergence | turn-2 diverging before the key's last token | honest miss (no serve) | **PASS** (verdict miss, prefill_from 0) |
| 6 | Persistence across restart | kill daemon, restart on same blob dir | resume hit from disk | **PASS** (hit after restart, lookup_ms 1, identical output) |
| 7 | Checkpoint trust boundary | byte-flip inside payload; truncated payload | reject + quarantine for corrupt framing | **PASS for real-engine framing** (pytest: invalid safetensors → 422 → quarantine); **finding ISSUE-002 for synthetic payloads + no payload checksum (see below)** |
| 8 | Trace file permissions | batch C fix (0600) | `-rw-------` | **PASS** (0600 observed, 15 records written) |
| 9 | Single-flight coalescing | 2 concurrent identical misses + 1 s prefill delay | one prefill, one publish, follower serves hit | **PASS** (1 blob; follower lookup_ms 1012 ≈ one prefill window, not 2 s; byte-identical outputs) |
| 10 | Config surface | daemon boot env | fail-fast on misconfig that 404s all traffic | **FAIL → ISSUE-001 (medium)** |

## Findings

### ISSUE-001 (medium, verified): daemon boots with `MLXCACHE_MODELS` unset and then 404s every request
- **Repro:** start daemon without `MLXCACHE_MODELS`; log shows `WARN: MLXCACHE_MODELS is empty: every request will 404`; every completion request returns 404.
- **Impact:** a deployment typo produces a "healthy" daemon that serves nothing; a health-check on `/stats` (200) hides it.
- **Disposition:** deferred to a decision brief (behavior is arguably intentional for model-less dev runs; the fix is a boot-time error or a `/health` that reflects it). Not fixed in this run: it's a policy call, not a defect with an obvious contract.

### ISSUE-002 (medium, verified): blob payload has no integrity checksum; corruption inside the payload is not detectable at the boundary
- **Evidence:** flipped a byte inside a published blob's payload; the serve still returned the same tokens (synthetic engine never reads payload bytes — generation derives from token ids). For the **real engine**, framing-invalid corruption IS rejected+quarantined (pytest coverage), but a framing-valid bit-rotted float would load and serve confidently wrong tokens.
- **Root observation:** the wire format is `u32 len + JSON header + raw payload`; `token_count` + token-prefix equality validate *what* the KV claims but nothing validates the KV *bytes* themselves.
- **Impact:** silent wrong-token serves after disk bit-rot/partial-media corruption. No crash, no quarantine — worst failure class for a cache product (the "accounting lie" the design elsewhere refuses).
- **Disposition:** deferred to a decision brief — the honest fix (hash the payload, verify on serve) costs ~+0.5–1 s per serve at 20K-token blob sizes (2.4 GB read); that trades against D1 ("the only thing that matters is 100% optimization"). Options: (A) accept risk (bit-rot is rare on healthy media), (B) hash-at-publish + verify-at-serve, (C) hash-at-publish + verify only after any read error (cheap, catches nothing silent), (D) verify-at-serve sampled (e.g. every Nth).

## Fixed this run
None needed — all failing-surface candidates were either already covered by the fix wave's regression tests or are policy decisions above.

## Process cleanup
All owned sidecars/daemons stopped and verified (`pgrep` clean). Evidence retained under `.gstack/qa-reports/evidence-2026-10-04/`.

## Summary
> QA probed 10 contracts, 8 passed, 1 passed-with-finding (ISSUE-002 deferred to decision), 1 failed (ISSUE-001 deferred to decision). 0 fixes required in product code; 0 health-score regressions (functional target: contract outcomes, no score).
