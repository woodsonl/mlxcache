# mlxcache

A KV cache daemon for MLX on Apple Silicon. It sits between an OpenAI-compatible
client (coding agent, chat app, script) and a local MLX engine, and makes every
process reuse the expensive prefilled context it already paid for — system
prompts, repo context, agent histories — across requests, processes, and restarts.

If you run local inference on a Mac and re-prefill the same long context on every
turn, this is the layer that stops that.

## How it works

```
client ──HTTP──▶ mlxcache daemon ──▶ sidecar (mlx-lm) ──▶ model
                 │  radix-tree prefix index
                 │  single-flight prefill
                 └─ atomic KV checkpoints on disk (survive restart)
```

- The **daemon** (Rust) owns the cache: it hashes token prefixes, routes hits,
  collapses concurrent identical prefills into one, and persists KV checkpoints
  atomically (`write temp → sync → rename`). Zero Python on the hot path.
- The **sidecar** (Python, `mlx-lm`) is the compatibility adapter. It tokenizes,
  runs prefill, captures KV as safetensors, and streams generation. The daemon
  talks to it over local HTTP.
- A **hit** skips the prefill entirely and resumes generation from the stored
  cache. Verified token-for-token identical to a cold run (see below).
- The **L1 store client** (`sidecar/mlxcache_store/`) embeds the same cache
  without the daemon: any Python engine gets persistent, bounded, digest-checked
  prefix KV reuse over one directory ([`docs/connector-protocol.md`](docs/connector-protocol.md) §3). Same blob
  format, same key fold as the Rust daemon — a directory the daemon wrote is
  readable by the embedded client and vice versa.
- The **wrapper** (`sidecar/mlxcache_serve/`) turns that on for any mlx-lm
  server in one command — no daemon, no mlx-lm patches:
  `python -m mlxcache_serve --store-dir ~/.mlxcache/store --model <model>`.
  Finished conversations persist across restarts; restarts resume from disk
  instead of re-prefilling.

The contract is engine-agnostic ([`docs/contract-spec.md`](docs/contract-spec.md));
mlx-lm is the first adapter, not the only possible one.

## Requirements

- Apple Silicon Mac (M-series)
- Rust (for the daemon) — `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- Python 3.12+ and `uv` — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- For real inference: a local mlx-lm model, downloaded on first use

## Quickstart (synthetic engine — no model download)

Proves the cache works in about a minute.

```bash
# 1. Build the daemon
cargo build --release -p mlxcache-daemon

# 2. Start the sidecar (synthetic engine: deterministic, no model needed)
MLXCACHE_MODEL=demo-model uv run python -m mlxcache_sidecar.server &

# 3. Start the daemon, pointing at the sidecar
MLXCACHE_MODELS=demo-model \
MLXCACHE_SIDECAR_URL=http://127.0.0.1:8421 \
MLXCACHE_BLOBS=/tmp/mlxcache-blobs \
./target/release/mlxcache-daemon &

# 4. Ask the same question twice
curl -s -X POST http://127.0.0.1:8420/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"demo-model","messages":[{"role":"user","content":"hello"}]}'

curl -s -X POST http://127.0.0.1:8420/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"demo-model","messages":[{"role":"user","content":"hello"}]}'

# 5. Look at the cache
curl -s http://127.0.0.1:8420/stats
```

The first response carries `"verdict":"miss"`, the second `"verdict":"hit"` with
`"prefill_from":<n>` — `n` is the number of leading tokens the adapter resumed
from a stored checkpoint instead of pre-filling, so the daemon skipped
re-prefilling them. A checkpoint caches KV for `tokens[:-1]` (the adapter saves
the cache up to the last token), so for an `L`-token prompt a full hit reports
`L-1`. A request that has to run its own prefill (a miss, or the single-flight
leader even on a partial) reports `0`: it reused no prior KV even though
generation then resumes from the blob it just wrote. The response also carries
`"tokens_cached"`, the same count, and `"tokens_total"`. `/stats` reports the
running hit rate.

## API surface

The request is OpenAI-shaped, not OpenAI-complete. Exactly four fields are
honored: `model` (must be in `MLXCACHE_MODELS`), `messages`, `stream`, and
`max_tokens` (default `64`, capped at `8192` — larger values get a `400`).
Everything else (`temperature`, `top_p`, `stop`, `tools`, …) is ignored, not
rejected.

Responses decode with **stock OpenAI SDKs**: the non-stream body carries
`id`/`object`/`created`/`model`/`choices[0].message.content` (detokenized by
the sidecar — identical text to the streaming path) plus a `usage` block, and
every non-error SSE frame is a valid `chat.completion.chunk` with
`choices[0].delta.content` pieces, a role-delta first chunk, and a
`finish_reason` final chunk (`"length"` when generation hit your `max_tokens`
cap, `"stop"` otherwise) before `data: [DONE]`. Mid-stream engine failures
emit a bare `data: {"error":{...}}` frame (OpenAI's own error-stream shape)
before `[DONE]`. Point an SDK at `base_url=http://127.0.0.1:8420/v1` and it
works, streaming and not. One gap: `stream_options.include_usage` is not
honored — stream chunks carry no `usage` block (the non-stream response and
`/stats` have the numbers).

Cache telemetry rides along as extra fields the SDKs ignore: `mlxcache`
(verdict, tokens_cached, tokens_total, prefill_from, timings; on the first
stream chunk and the final body) and `generated_tokens` (raw token ids).

Every error is the same envelope: `{"error": {"message", "type"}}`, including
body-parse failures (`400`), a missing JSON content-type (`415`), and unknown
models (`404`, checked before any cache lookup). Status codes say what to
retry: `503 adapter_unavailable` means the sidecar is down or stalled (back
off and retry later), `502 adapter_error` means the sidecar answered but the
answer was bad — including its explicit `422`-rejections, which the daemon
already handled by quarantining the checkpoint and (on the next request)
serving from scratch. `GET /healthz` is a dependency-free liveness probe
(`{"status":"ok"}`); cache and adapter state live in `/stats`.

## Storage: integrity, quarantine, and eviction
The store is CHUNKED by default (v0.7.0): a published checkpoint is a
header-only file plus a content-addressed chunk manifest, with chunks
packed into shared 64 MiB segments. The token-block/64 codec makes a
shared token prefix yield byte-identical chunks, so a growing conversation
pays for each turn's NEW tokens instead of re-storing its whole history —
measured 2.2x on a 5-turn 7B conversation
([REPORT.md](REPORT.md), the GO record). Every reader
reassembles and verifies the same payload sha256, so corruption retires a
chunked blob exactly like a whole-blob flip; a chunk leaves disk only when
no live checkpoint references it. Setting `MLXCACHE_CHUNK_STORE=0` on the daemon publishes whole-blob v2
instead (both layouts coexist and stay readable forever).

Every published hit writes one KV checkpoint ("blob") — a safetensors payload
behind a JSON header — to its own `{hash}-{generation}.ckpt` file, written to
a temp name and atomically renamed. Blobs survive restarts; the daemon
rebuilds its index from disk at boot. Sizes are why the rest of this section
exists: KV runs 12–262 KB **per token** depending on model size (see
[Measured](#measured)), so a 20K-token agent context is a 1–5 GB file, and an
unbounded store is a disk-full incident waiting to happen.

**Integrity.** Each blob is stamped at publish with `format_version: 2` and a
sha256 digest of its payload. The daemon verifies every blob during the boot
rebuild; the sidecar re-verifies at first serve (once per file). A
framing-valid bit-rot flip — the corruption mode no structural check can
catch — surfaces as a `422`, the entry is quarantined, and the request re-runs
from scratch and republishes. A checkpoint the adapter cannot use is retired,
not retried forever: the adapter `422`s a blob that is gone, corrupt,
digest-mismatched, or whose recorded prefix disagrees with the request, and
retirement is keyed by publication generation, so a late failure unlinks only
that publication's file — never a fresh republish that replaced it. A generic
`500`/transport failure never retires a healthy checkpoint. Legacy
version-1 blobs (written before digests existed) stay readable, unverified.

**Identity.** A blob's header pins the model id, engine id, tokenizer hash,
KV dtype, KV layout version, quantization tier, a digest of any non-text
prompt content, AND cache composition — the ordered
tuple of cache-entry classes the engine builds for the model (a dense KV
stack vs. a hybrid model's `CacheList`/`ArraysCache` recurrent state). Two
architectures can agree on everything else while their payloads have
mutually unloadable shapes, so composition is folded into the filename key
and compared on every lookup: a checkpoint is never served to an engine
that would deserialize it as a different shape. A blob written before
compositions were recorded stays usable by an engine that also reports no
composition (legacy-to-legacy); the moment an engine knows its composition,
those older blobs are unreachable and get rewritten on the next publish.
Keys themselves are unchanged for the default engine and an unrecorded
composition, so upgrading does not re-key an existing store.

Engine domains are isolated: the engine id is part of the key, so a blob
published through one engine is unreachable from another even if every
other field agrees. Non-text prompt content (an image in a message) is
hashed into the key too: two requests with identical text but different
media never share a checkpoint, and an empty digest asserts "no media"
rather than "unknown media."

**Eviction.** A background reaper bounds the store. Every
`MLXCACHE_EVICT_INTERVAL_S` (default 60s) it measures the published blobs and,
if the store exceeds `MLXCACHE_EVICT_MAX_BYTES` (default **32 GiB** — on by
default), evicts until it fits: **coldest non-anchor blobs, biggest first**,
so each unlink frees the most bytes per sweep. Two things are never evicted:

- **Anchors** — a checkpoint that a longer published checkpoint extends (a
  chain base; evicting it would collapse the whole chain's future hit rate),
  or anything served within `MLXCACHE_EVICT_ANCHOR_WINDOW_S` (default 15 min).
  If anchors alone hold the store over budget, the reaper logs a warning and
  stops — it never breaks the anchor contract.
- **Fresh replacements** — removal is verified against the exact publication
  generation before the file is unlinked, so a sweep can never delete a newer
  republish of the same prefix.

Eviction is a throughput cost, never a correctness one: the next request that
would have hit an evicted checkpoint takes an honest miss and republishes it.
`/stats` carries an `evictions` counter. To run unbounded (the old behavior):
`MLXCACHE_EVICT_MAX_BYTES=0`.

## Real inference (mlx-lm)

Install the adapter, then run with the real engine:

```bash
uv sync --extra mlx                       # pins mlx-lm (see pyproject.toml)

MLXCACHE_ENGINE=mlx-lm \
MLXCACHE_MODEL=Qwen/Qwen2-0.5B-Instruct \
uv run python -m mlxcache_sidecar.server &

MLXCACHE_MODELS=Qwen/Qwen2-0.5B-Instruct \
MLXCACHE_SIDECAR_URL=http://127.0.0.1:8421 \
./target/release/mlxcache-daemon
```

The model id is whatever `mlx_lm.load()` accepts (an HF repo or a local path).
First run downloads the weights.

Hybrid architectures (linear attention / SSM / GDN mixed with attention —
Falcon-H1, Qwen3-Next, granite-hybrid, Nemotron-H, RWKV7, …) work through the
same path: mlx-lm's `save_prompt_cache`/`load_prompt_cache` are class-generic
(they record each entry's cache class and rebuild it via `from_state`), so the
adapter's capture path needs no per-architecture code. Verified
token-identical on `mlx-community/Falcon-H1-0.5B-Instruct-4bit` (every layer a
nested `CacheList(ArraysCache, KVCache)`) via the same R1-5 gate.

## Wrapper server (mlx-lm, no daemon)

Any mlx-lm server gains persistent prefix reuse in one command — no daemon, no
mlx-lm patches:

```bash
uv run python -m mlxcache_serve \
  --store-dir ~/.mlxcache/store \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit --port 8080
```

Unknown flags pass through to `mlx_lm.server`. Store flags mirror the
daemon's contract: `--store-budget-bytes` (default 32 GiB), `--anchor-window-s`
(default 900), `--trust-legacy` (index digest-less v1 blobs), and
`--kv-bits`/`--kv-group-size` (KV quantization tier — part of the fingerprint,
so tiers never share blobs). `--kv-tier` names a TurboQuant persistence
codec (`tq-k4v4-g64` and friends) that actually compresses what the wrapper
persists — unlike `--kv-bits`, which only labels the partition; the measured
bake-off lives in [docs/designs/tq-tier.md](docs/designs/tq-tier.md). One
extra endpoint rides along:

- `GET /mlxcache/stats` — the live cache's reuse counters:
  `{"disk_hits": .., "persisted": .., "reuse_errors": .., "persist_errors": ..}`.
  At temperature 0 a disk resume and a full prefill are token-identical, so
  response equality cannot prove the disk tier served a request; the counter
  can.

The wrapper is text-only by construction: it wraps mlx-lm's server, which
owns the request, so it never sees message JSON and every fingerprint it
mints carries `content_digest=""` — the no-media claim, true for every model
mlx-lm can load. Serving a multimodal model through it requires teaching that
fingerprint a content-digest producer first.

## Streaming

Set `"stream": true` and the daemon returns the OpenAI SSE shape from
[API surface](#api-surface): a leading `chat.completion.chunk` carrying the
assistant role delta with the `mlxcache` verdict as a top-level extra, one
valid `chat.completion.chunk` per generated token (`choices[0].delta.content`
pieces, a `finish_reason` chunk last), then `data: [DONE]`. Point a stock
OpenAI SDK at `base_url=http://127.0.0.1:8420/v1` with `stream=True` and the
frames parse. If the engine dies mid-stream the daemon emits a `data:
{"error":{"type":"upstream_error",...}}` frame before `[DONE]`, so a truncated
answer is not mistaken for a finished one.

```bash
curl -N -X POST http://127.0.0.1:8420/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"demo-model","messages":[{"role":"user","content":"hello"}],"stream":true}'
```

## Configuration

Daemon:

| Variable | Default | Meaning |
|---|---|---|
| `MLXCACHE_ADDR` | `127.0.0.1:8420` | daemon bind address |
| `MLXCACHE_MODELS` | (empty) | comma-separated served models; others 404 before any cache lookup; empty fails the boot unless `MLXCACHE_ALLOW_NO_MODELS=1` |
| `MLXCACHE_ALLOW_NO_MODELS` | (unset) | escape hatch to boot model-less (every request 404s) |
| `MLXCACHE_SIDECAR_URL` | (unset) | sidecar base URL; unset = no adapter, requests 503 |
| `MLXCACHE_BLOBS` | `/tmp/mlxcache-blobs` | where KV checkpoints are written |
| `MLXCACHE_SIDECAR_TIMEOUT_S` | `120` | per-request sidecar timeout; must exceed the slowest prefill |
| `MLXCACHE_STREAM_IDLE_TIMEOUT_S` | `60` | longest gap between bytes on a live generation stream before the sidecar is declared wedged |
| `MLXCACHE_NATIVE_TOKENIZER` | (unset) | path to the engine's `tokenizer.json`; when set, the daemon tokenizes natively (no sidecar round-trip) after PROVING parity against the engine over an adversarial probe set — any mismatch fails closed (503) rather than risk mis-routing checkpoints |
| `MLXCACHE_TRACE` | (unset) | capture one JSONL record per settled request (exact payload, verdict, covered KV); score and replay with `scripts/replay_trace.py` |
| `MLXCACHE_SHUTDOWN_GRACE_S` | `30` | SIGTERM drain window for in-flight streams |
| `MLXCACHE_EVICT_INTERVAL_S` | `60` | eviction sweep cadence; `0` disables the reaper |
| `MLXCACHE_EVICT_MAX_BYTES` | 32 GiB | store byte budget, on by default — see [Storage](#storage-integrity-quarantine-and-eviction); `0` = unbounded |
| `MLXCACHE_EVICT_MAX_ENTRIES` | `0` | entry cap on top of the byte budget; `0` = none |
| `MLXCACHE_EVICT_ANCHOR_WINDOW_S` | `900` | how recently a checkpoint must have served to stay an eviction-protected anchor |
| `MLXCACHE_CHUNK_STORE` | `1` | `0` publishes whole-blob v2 instead of chunked v3 (rollback knob). The chunked store dedups shared prefixes at the chunk level — the measured 2.2× on a 5-turn 7B conversation — with the same sha256 integrity contract over reassembled payloads; both layouts coexist in one directory and every reader handles both |

Sidecar:

| Variable | Default | Meaning |
|---|---|---|
| `MLXCACHE_ENGINE` | `synthetic` | `synthetic` (deterministic, no download) or `mlx-lm` |
| `MLXCACHE_MODEL` | `synthetic-model` | model the sidecar loads (HF repo or local path) |
| `MLXCACHE_SIDECAR_PORT` | `8421` | sidecar bind port |
| `MLXCACHE_KV_BITS` | `0` | KV quantization tier: `0` = f16, `8` = q8 — the tier is folded into the fingerprint, so tiers never share blobs. Dense attention models only: a hybrid model (linear attention / SSM / GDN layers — e.g. Falcon-H1, Qwen3-Next, granite-hybrid) is refused at boot with this tier set; its state persists unquantized on the f16 tier |
| `MLXCACHE_KV_GROUP_SIZE` | `0` | quantization group size; `0` with a nonzero `MLXCACHE_KV_BITS` resolves to `64` |
| `MLXCACHE_KV_TIER` | (unset) | named KV persistence tier (TurboQuant codecs, from arXiv:2504.19874): `tq-k4v4-g64` (recommended: 3.6× smaller than f16, per-layer cos ≥ 0.992), `tq-k3v4-g64` (4.0×), `tq-k3v2-g64` (5.3×, lossy). Compresses the persisted checkpoint; the live engine stays f16 unless `MLXCACHE_TQ_LIVE=1` swaps it too. LOSSY BY DESIGN: resume is deterministic but diverges from a scratch run (bounded-distortion contract, [docs/designs/tq-tier.md](docs/designs/tq-tier.md)); opt-in and fingerprint-isolated, so it never serves the f16/q8 tiers. Mutually exclusive with `MLXCACHE_KV_BITS`; dense models only |
| `MLXCACHE_TQ_LIVE` | (unset) | with `MLXCACHE_KV_TIER` set, `1` makes the engine's own cache hold the COMPRESSED state during decode (TurboQuantKVCache): 3.66× smaller resident KV at 0.843× decode throughput (16% fewer tok/s, ~1.19× per-token latency) on the 7B (measured in the design doc). Same tier label, same frame format — a runtime memory trade, not a new tier; outputs diverge from both f16 and persistence-tier runs (quantizing during the forward changes the computation) but are deterministic per run |
| `MLXCACHE_BENCH_REAL` | (unset) | set to `1` to run the real mlx-lm benchmark |
| `MLXCACHE_METAL_CACHE_MB` | `1024` | MLX allocator cache limit; freed buffers beyond it return to the OS instead of pooling |
| `MLXCACHE_SETTLE_MAX_TOKENS` | `32768` (unset) | cap on a settled checkpoint's token count; an explicit `0` refuses every settle rather than disabling the cap |
| `MLXCACHE_SETTLE_MAX_BYTES` | `2 GiB` (unset) | cap on a settled checkpoint's payload bytes; an explicit `0` refuses every settle rather than disabling the cap |

## Building and testing

```bash
uv sync --group dev                      # pytest, ruff
cargo fmt --all -- --check               # formatting gate
cargo test --workspace                   # Rust: core, daemon, chaos, conformance, e2e
cargo test -p mlxcache-daemon --test alloc_gate --release -- \
  --test-threads=1 --include-ignored     # alloc gate: one allocation per token frame
cargo clippy --workspace --all-targets -- -D warnings
uv run pytest sidecar/tests/             # Python: sidecar, blob codec, roundtrip
uv run pytest scripts/dedup/tests/       # Dedup: measurement invariants behind REPORT.md
uv run ruff check sidecar/ scripts/
uv run ruff format --check sidecar/ scripts/
```

Bench and QA harnesses live in `scripts/` on one shared core, `_harness.py`
(spawn, readiness that cannot certify a server it did not start, teardown):
`bench_workloads.py` is a plumbing check over the synthetic engine — its
8-token "prefill" can never express prefix reuse, so its cache numbers must
never be quoted as cache-benefit evidence; `bench_prefix.py` is the
real-engine prefix benchmark (premise-guarded, see
[docs/benchmarks.md](docs/benchmarks.md)); `bench_wrapper.py`/`qa_wrapper.py`
drive the wrapper path's own gates; `replay_trace.py` scores captured traces.
Timing probes live in `sidecar/probes/`: mlx capability, quantized-KV
persistence, decode A/B, context sweep, resume budget, and the two TQ
tier probes — each measures one question the cache's claims rest on.

The real mlx-lm round-trip gate (needs a downloaded model) is opt-in:

```bash
MLXCACHE_BENCH_REAL=1 uv run pytest sidecar/tests/test_roundtrip_real.py -v
```

It verifies the core claim: generation resumed from a saved-then-loaded KV cache
is **token-for-token identical** to a scratch run. If that ever fails, the
adapter has become lossy and checkpoints are not being trusted correctly.

## Measured

Qwen2-0.5B-Instruct, mlx-lm 0.31.3, Apple Silicon: 12,288 bytes/token,
serialize 20 ms, deserialize <1 ms. Qwen2.5-7B-Instruct-4bit: 57,344 bytes/token,
serialize 526 ms, deserialize <1 ms, prefill 10.2 s. Qwen3-32B-4bit: 262,150
bytes/token, serialize 203 ms, deserialize 1 ms, prefill 11.1 s. A hit eliminates
that prefill, and the 2 s TTFT resume budget holds with margin at every size.

Long-context cost curve on the reference model (Qwen2.5-7B-Instruct-4bit,
context sweep — one process per size, self-checked for monotonicity,
decode/prefill ratio, repeatability, and swap growth bounded by each row's
own footprint): prefill runs 0.36 s at 256 tokens to 27.0 s at 16K (1.4 to
1.65 ms/token as the attention history lengthens), while decode stays flat
across every size — 14 ms/token at 256, 18 ms at 16K (ratio 1.29x) — a
long context taxes the
prefill you pay once, not the decode you pay per token. Checkpoints scale
14.6 MB → 939.5 MB at 55.8-56.0 KiB/token, landing exactly on the
57,344-byte figure above: the probe captures the prompt-only cache, the
adapter's own checkpoint contract. The 16K row held a 25.0-29.1 s span
across seven runs on a machine that also hosted other workloads, and
scaling the 16K per-token cost to the restart test's 19,847 tokens
predicts 32.7 s against its measured 34.5 s — two independent methods
agreeing within 6%. The 32K
point is withheld: four of five attempts were refused by the sweep's own
guards — a 12 GB machine-churn event, and one confirmed thrash (79.9 s
prefill, a TTFT step 20x its normal cost, swap growth beyond the row's
own footprint) against the 64.7-68.3 s consensus of the runs whose rows
sat on the curve. That refusal is the methodology working; the era this
replaces once published a contaminated 8K figure instead. The probe is
`sidecar/probes/context_sweep.py`.

Restart resume (the claim that matters for a cache): a 19,847-token agent
context on Qwen2.5-7B-Instruct-4bit goes 34.5 s cold → 1.56 s warm → **1.59 s
after SIGTERM-killing both processes and restarting from disk** (verified in
both KV tiers, f16 and q8).

Full numbers and caveats in the [design doc](docs/designs/mlx-kv-cache-daemon.md) (R1-5).

Prefix-cache benchmark (Qwen2.5-7B-Instruct-4bit, runtime-calibrated
~1,975-token shared prefix, concurrency 1, premise-guarded with enforced
trace//stats cross-checks): after a restart, the daemon serves 99.95% of
requested prefix tokens and cuts TTFT 3.2 s → 0.15 s, with TPOT flat at
~13 ms — and the cross-restart cached path (~154 ms) now beats a raw local
`mlx_lm.server` re-prefilling the same bytes (~189 ms); only a live
process's in-memory LRU stays ahead (~121 ms) and it dies with the process.
The first request still pays ~3.2 s to publish a ~110 MB checkpoint. Under
a 3-entry cap the reaper evicts the longest (highest-value) prefix first
once pressure binds; anchor chains delay but do not prevent it. An earlier
"re-prefill is faster" claim here was an artifact of a miscalibrated
prefix (a guessed divisor benchmarked ~3,250 tokens under a 2,000-token
knob). Methodology, guards, and the full grid:
[docs/benchmarks.md](docs/benchmarks.md).

Named-tier bake-off (same 7B model, 1,500-token prefix): the TurboQuant
persistence tiers and the plain bits tiers measured head-to-head —
`tq-k4v4-g64` at 15.7 KB/token (3.64× vs f16, per-layer cos 0.992) strictly
dominates `q4` (16.1 KB/token, 3.55×, 0.990); the full frontier table and
the divergence disclosure are in [docs/designs/tq-tier.md](docs/designs/tq-tier.md).

## Operating it

See [`docs/ops.md`](docs/ops.md) — bypassing the daemon (SPOF), rollback,
launchd service, and the mlx-lm version-pinning policy.

## Status

Early. Working end to end (cache, persistence, single-flight, streaming, error
rescues, chaos tests). The R1-5 gate has run on Qwen2-0.5B,
Qwen2.5-7B-Instruct-4bit, and the named representative Qwen3-32B-4bit, all
token-identical to scratch. Hybrid architectures round-trip through the same
gate (measured on Falcon-H1-0.5B); the q8 KV tier is dense-only by boot-time
guard. Cache composition is part of every fingerprint (dense KV stack vs.
hybrid recurrent state), so a payload of one shape is never served as
another. The store is bounded by default (32 GiB eviction
budget with anchor protection) and every checkpoint carries a sha256 payload
digest verified at boot and first serve. The engine-agnostic contract has one
adapter today (mlx-lm), plus the hermetic synthetic engine the e2e suite
drives; engine domains are isolated in the key space, so a future adapter
cannot cross-serve another engine's checkpoints. No license assigned —
private build, all rights reserved.

The chunk-granularity question is answered: GO, with token-block/64 as the
v1 codec ([REPORT.md](REPORT.md)).
