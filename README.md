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
  atomically (`write temp → rename`). Zero Python on the hot path.
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

## Streaming

Set `"stream": true` and the daemon returns OpenAI-style SSE: a leading
`data: {"mlxcache":{...}}` frame with the cache verdict, one
`data: {"token":..,"text":..}` per generated token, then `data: [DONE]`. If the
engine dies mid-stream the daemon emits a `data:
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
| `MLXCACHE_NATIVE_TOKENIZER` | (unset) | path to the engine's `tokenizer.json`; when set, the daemon tokenizes natively (no sidecar round-trip) after PROVING parity against the engine over an adversarial probe set — any mismatch fails closed (503) rather than risk mis-routing checkpoints |
| `MLXCACHE_TRACE` | (unset) | capture one JSONL record per settled request (exact payload, verdict, covered KV); score and replay with `scripts/replay_trace.py` |
| `MLXCACHE_SHUTDOWN_GRACE_S` | `30` | SIGTERM drain window for in-flight streams |
| `MLXCACHE_EVICT_INTERVAL_S` | `60` | eviction sweep cadence; `0` disables the reaper |
| `MLXCACHE_EVICT_MAX_BYTES` | 32 GiB | store byte budget, on by default — see [Storage](#storage-integrity-quarantine-and-eviction); `0` = unbounded |
| `MLXCACHE_EVICT_MAX_ENTRIES` | `0` | entry cap on top of the byte budget; `0` = none |

Sidecar:

| Variable | Default | Meaning |
|---|---|---|
| `MLXCACHE_ENGINE` | `synthetic` | `synthetic` (deterministic, no download) or `mlx-lm` |
| `MLXCACHE_MODEL` | `synthetic-model` | model the sidecar loads (HF repo or local path) |
| `MLXCACHE_SIDECAR_PORT` | `8421` | sidecar bind port |
| `MLXCACHE_KV_BITS` | `0` | KV quantization tier: `0` = f16, `8` = q8 — the tier is folded into the fingerprint, so tiers never share blobs |
| `MLXCACHE_BENCH_REAL` | (unset) | set to `1` to run the real mlx-lm benchmark |

## Building and testing

```bash
uv sync --group dev                      # pytest, ruff (what CI installs)
cargo fmt --all -- --check               # formatting gate (CI fails on drift)
cargo test --workspace                   # Rust: core, daemon, chaos, e2e
cargo clippy --workspace --all-targets -- -D warnings
uv run pytest sidecar/tests/             # Python: sidecar, blob codec, roundtrip
uv run ruff check sidecar/
uv run ruff format --check sidecar/
```

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

Restart resume (the claim that matters for a cache): a 19,847-token agent
context on Qwen2.5-7B-Instruct-4bit goes 34.5 s cold → 1.56 s warm → **1.59 s
after SIGTERM-killing both processes and restarting from disk** (verified in
both KV tiers, f16 and q8).

Full numbers and caveats in the [design doc](docs/designs/mlx-kv-cache-daemon.md) (R1-5).

## Operating it

See [`docs/ops.md`](docs/ops.md) — bypassing the daemon (SPOF), rollback,
launchd service, and the mlx-lm version-pinning policy.

## Status

Early. Working end to end (cache, persistence, single-flight, streaming, error
rescues, chaos tests). The R1-5 gate has run on Qwen2-0.5B,
Qwen2.5-7B-Instruct-4bit, and the named representative Qwen3-32B-4bit, all
token-identical to scratch. The store is bounded by default (32 GiB eviction
budget with anchor protection) and every checkpoint carries a sha256 payload
digest verified at boot and first serve. The engine-agnostic contract has one
adapter (mlx-lm). No license assigned — private build, all rights reserved.
