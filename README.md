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

## When to use it

You re-pay the same prefill on every turn: a coding agent with a stable
system prompt and repo context, a chat app that reloads history, evals that
replay long contexts. mlxcache makes the second and every later request
resume from stored KV instead.

- vs **llama.cpp slot save/restore** — same idea, different engine: this is
  the MLX world, and the chunked store dedups at the token-block level, so a
  growing conversation writes only its new tokens (measured 2.2× on a 5-turn
  7B conversation).
- vs **mlx-lm's own server cache** — that is an in-process LRU; it dies with
  the process. Checkpoints here survive restarts, and the cross-restart
  cached path (~154 ms) beats a raw re-prefill of the same bytes (~189 ms)
  on the reference model ([Measured](#measured)).
- vs **vLLM-style prefix caching** — that targets multi-tenant GPU serving.
  This targets one Apple Silicon box and one user's workflows, with a
  32 GiB-bounded store and anchor-aware eviction instead of a fleet.

## Requirements

- Apple Silicon Mac (M-series)
- Rust (for the daemon) — `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- Python 3.12+ and `uv` — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- For real inference: a local mlx-lm model, downloaded on first use

## Installation

Source only for now — no published packages. The crate and package metadata
are staged so a PyPI/crates.io release later is a config change, not a
refactor.

```bash
git clone https://github.com/woodsonl/mlxcache.git
cd mlxcache
uv sync --group dev        # Python: sidecar, store client, test tooling
cargo build --release -p mlxcache-daemon   # daemon: target/release/mlxcache-daemon
```

That gives you the two processes: `mlxcache-daemon` (the cache, Rust) and
the sidecar (`uv run python -m mlxcache_sidecar.server`, the engine
adapter). The daemon needs the sidecar's URL; everything else is
[configuration](#configuration).

## Getting started (synthetic engine — no model download)

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

For a real model, continue to [Real inference (mlx-lm)](#real-inference-mlx-lm);
for what is happening on disk, [Storage](#storage-integrity-quarantine-and-eviction).

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
(`{"status":"ok"}`); cache and adapter state live in `/stats`, whose
`chunk_store` block reports the dedup gauges (ratio, dead bytes, and
per-blob unique/shared record bytes — what evicting each blob actually
frees). `POST /gc` reclaims dead chunk-store bytes on demand (see
[Storage](#storage-integrity-quarantine-and-eviction)).

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
no live checkpoint references it. Empty segments delete on the eviction
sweep; a PARTIALLY-dead segment (live chunks sandwiching dead ones) is
rewritten by segment compaction — the live records move to a fresh segment
and the old file deletes, so dead bytes leave disk instead of squatting
between live chunks. `POST /gc` runs the same maintenance on demand (it
seals the active append segment, compacts at the configured thresholds,
then deletes empty segments), so reclaiming dead bytes does not require a
restart; a failed sweep answers `500 store_error`, and a zero receipt
means nothing met the thresholds — `/stats`'s `dead_bytes` says whether
any remain. Reassembly differs by reader. The daemon builds one scatter
item per chunk head, splits them into byte-balanced groups, and runs the
groups on up to 8 scoped OS threads — a multi-chunk read parallelizes
even inside one segment; a single-chunk read runs on one worker. The
sidecar issues one pool task per segment (capped at 8 threads), staging
each chunk record before scattering its slices — a single-segment read
runs on one worker. Setting `MLXCACHE_CHUNK_STORE=0` on the daemon publishes whole-blob v2
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

### Compressed persistence tiers (TurboQuant)

By default a checkpoint is stored f16 — a byte-exact copy of what the engine
holds. A named TurboQuant tier compresses the persisted checkpoint instead:
3.6-5.3x smaller on disk, with quantization fidelity recorded at the bake-off
(below in [Measured](#measured)). This is opt-in, off by default, and never
silent: a tier label is part of the checkpoint's fingerprint, so compressed
and f16 blobs never serve each other.

To run the daemon's sidecar on the recommended tier (the `mlx` extra first —
the real engine is an optional dependency, per
[Real inference](#real-inference-mlx-lm)):

```bash
uv sync --group dev --extra mlx
MLXCACHE_KV_TIER=tq-k4v4-g64 MLXCACHE_ENGINE=mlx-lm \
MLXCACHE_MODEL=mlx-community/Qwen2.5-7B-Instruct-4bit \
uv run python -m mlxcache_sidecar.server &
```

The wrapper takes the same tier as a flag: `--kv-tier tq-k4v4-g64`. The
wrapper's disk-resume counter note (temperature-0 equality with a scratch
run) holds for the f16 path; a tier label makes wrapper resumes deterministic
but divergent like the sidecar's.

Tier labels are a grammar, not a closed list: `tq-k<key-bits>v<value-bits>-g<group-size>`
with key bits 2/3/4, value bits 2/3/4/6/8, group size ≥ 2 (the `g` prefix
is optional — `tq-k3v4-64` is accepted as `tq-k3v4-g64`; the raw label is
part of the fingerprint, so the two spellings are two cache identities:
pick one and stay on it). A typo refuses to
boot rather than falling back to f16, and the model's per-layer `head_dim`
must be divisible by the group size (the codec checks at encode time and
rejects loudly on a mismatch). Three labels are measured and
recommended — `tq-k4v4-g64` (3.6x smaller, per-layer cosine ≥ 0.992),
`tq-k3v4-g64` (4.0x), `tq-k3v2-g64` (5.3x, the lossiest) — and any other
valid label parses, subject to that geometry, but carries no measured
fidelity claim.

Two contracts bind every tier. LOSSY BY DESIGN: resuming from a compressed
checkpoint is deterministic but diverges from a scratch run (the
bounded-distortion contract; the full record is in git history at
`docs/designs/tq-tier.md`) — a coding agent resuming mid-conversation will
continue coherently, not token-identically. And tiers are mutually exclusive
with `MLXCACHE_KV_BITS` (the plain q8 tier) and dense-attention-only — the
sidecar refuses a hybrid architecture at boot; the wrapper validates the
tier/`--kv-bits` conflict at startup and defers the rest to its persistence
step's loud failures rather than refusing the boot.

Setting `MLXCACHE_TQ_LIVE=1` additionally swaps the ENGINE's live cache to
the compressed representation during decode: 3.66x smaller resident KV at
0.843x decode throughput on the 7B — a runtime memory trade with its own
deterministic divergence, separate from the persistence tier.

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
bake-off numbers are in [Measured](#measured). One
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
| `MLXCACHE_COMPACT` | `1` | `0` disables segment compaction (the sweep that rewrites partially-dead segments so dead bytes between live chunks leave disk; `/stats` reports the dead-byte total as `chunk_store.dead_bytes`) |
| `MLXCACHE_COMPACT_RATIO` | `0.5` | a segment compacts when at least this fraction of its record bytes is dead |
| `MLXCACHE_COMPACT_MIN_BYTES` | `16777216` | dead-byte floor per segment; below it the sweep leaves the segment alone (no churn on small fragments) |
| `MLXCACHE_COMPACT_MAX_SEGMENTS` | `2` | segments rewritten per sweep — the pacing that bounds each pass's I/O |

Sidecar:

| Variable | Default | Meaning |
|---|---|---|
| `MLXCACHE_ENGINE` | `synthetic` | `synthetic` (deterministic, no download) or `mlx-lm` |
| `MLXCACHE_MODEL` | `synthetic-model` | model the sidecar loads (HF repo or local path) |
| `MLXCACHE_SIDECAR_PORT` | `8421` | sidecar bind port |
| `MLXCACHE_KV_BITS` | `0` | KV quantization tier: `0` = f16, `8` = q8 — the tier is folded into the fingerprint, so tiers never share blobs. Dense attention models only: a hybrid model (linear attention / SSM / GDN layers — e.g. Falcon-H1, Qwen3-Next, granite-hybrid) is refused at boot with this tier set; its state persists unquantized on the f16 tier |
| `MLXCACHE_KV_GROUP_SIZE` | `0` | quantization group size; `0` with a nonzero `MLXCACHE_KV_BITS` resolves to `64` |
| `MLXCACHE_KV_TIER` | (unset) | named KV persistence tier (TurboQuant codecs, from arXiv:2504.19874; labels are a grammar, activation and contracts in [Compressed persistence tiers](#compressed-persistence-tiers-turboquant)): `tq-k4v4-g64` (recommended: 3.6× smaller than f16, per-layer cos ≥ 0.992), `tq-k3v4-g64` (4.0×), `tq-k3v2-g64` (5.3×, lossy). Compresses the persisted checkpoint; the live engine stays f16 unless `MLXCACHE_TQ_LIVE=1` swaps it too. LOSSY BY DESIGN: resume is deterministic but diverges from a scratch run (bounded-distortion contract); opt-in and fingerprint-isolated, so it never serves the f16/q8 tiers. Mutually exclusive with `MLXCACHE_KV_BITS`; dense models only |
| `MLXCACHE_TQ_LIVE` | (unset) | with `MLXCACHE_KV_TIER` set, `1` makes the engine's own cache hold the COMPRESSED state during decode (TurboQuantKVCache): 3.66× smaller resident KV at 0.843× decode throughput (16% fewer tok/s, ~1.19× per-token latency) on the 7B (measured; the full record is in git history at `docs/designs/tq-tier.md`). Same tier label, same frame format — a runtime memory trade, not a new tier; outputs diverge from both f16 and persistence-tier runs (quantizing during the forward changes the computation) but are deterministic per run |
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
own footprint): prefill runs 0.36 s at 256 tokens to 63.7 s at 32K (1.4 to
1.9 ms/token as the attention history lengthens), while decode stays flat
across every size — 13 ms/token at 256, 21 ms at 32K (ratio 1.62x) — a
long context taxes the prefill you pay once, not the decode you pay per
token. Checkpoints scale 14.6 MB → 1.88 GB at 55.8-56.0 KiB/token, landing
exactly on the 57,344-byte figure above: the probe captures the
prompt-only cache, the adapter's own checkpoint contract. The certified
32K run took nine attempts to land, and the trail is the point: every 32K
window on a shared machine displaces 1-3.4 GB as the row's ~8 GB working
set arrives (the KV lives in Metal allocator memory the process RSS
cannot see), one 12.3 GB thrash event measured 79.9 s and was refused and
excluded, and seven curve-consistent attempts spanning 63.7-68.3 s stood
as a published consensus until a machine with true headroom certified
63.7 s with zero displacement. The probe is
`sidecar/probes/context_sweep.py`.

Restart resume (the claim that matters for a cache): a 19,847-token agent
context on Qwen2.5-7B-Instruct-4bit goes 34.5 s cold → 1.56 s warm → **1.59 s
after SIGTERM-killing both processes and restarting from disk** (verified in
both KV tiers, f16 and q8).

Full numbers and caveats are in the R1-5 record, preserved in git history at `docs/designs/mlx-kv-cache-daemon.md`.

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
the divergence disclosure are preserved in git history at `docs/designs/tq-tier.md`.

## Operating it

See [`docs/ops.md`](docs/ops.md) — bypassing the daemon (SPOF), rollback,
launchd service, and the mlx-lm version-pinning policy.

## Related

- [kv-chunk-fs](https://github.com/the-code-foundry/kv-chunk-fs) and
  [local-llm-kv-cache](https://github.com/the-code-foundry/local-llm-kv-cache)
  — the llama.cpp-side siblings of this design: a deduplicating FUSE
  filesystem for KV snapshots and a slot-managing proxy. Their independent
  per-turn dedup measurements informed the chunked store's codec study
  ([REPORT.md](REPORT.md)) and v0.9.0's parallel restore and `POST /gc`.
- [MLX](https://github.com/ml-explore/mlx) and
  [mlx-lm](https://github.com/ml-explore/mlx-lm) — the engine and adapter
  underneath; the checkpoint round-trip is token-identical to a scratch run
  because of mlx-lm's save/load contract.

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
cannot cross-serve another engine's checkpoints. MIT licensed — see
[LICENSE](LICENSE).

The chunk-granularity question is answered: GO, with token-block/64 as the
v1 codec ([REPORT.md](REPORT.md)).

## Support me

<script type="text/javascript" src="https://cdnjs.buymeacoffee.com/1.0.0/button.prod.min.js" data-name="bmc-button" data-slug="lancewoodson" data-color="#FFDD00" data-emoji=""  data-font="Cookie" data-text="Buy me a coffee" data-outline-color="#000000" data-font-color="#000000" data-coffee-color="#ffffff" ></script>
mlxcache is free. If you like it, a coffee keeps the work on it going.
