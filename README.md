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

A checkpoint the adapter cannot use is retired, not retried forever. The adapter
answers `422` when the blob is gone, corrupt, or its recorded prefix disagrees
with the request; the daemon quarantines that entry and serves the request from
scratch. A generic decode or transport failure (`500`) does not retire a healthy
checkpoint. Each publication is written to its own file (`{hash}-{generation}
.ckpt`), and retirement is keyed by the publication generation the request used,
so a late failure unlinks only that publication's file — never a fresh republish
that replaced it.

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

| Variable | Default | Meaning |
|---|---|---|
| `MLXCACHE_ADDR` | `127.0.0.1:8420` | daemon bind address |
| `MLXCACHE_MODELS` | (empty) | comma-separated served models; others 404 before any cache lookup |
| `MLXCACHE_SIDECAR_URL` | (unset) | sidecar base URL; unset = no adapter, requests 503 |
| `MLXCACHE_BLOBS` | `/tmp/mlxcache-blobs` | where KV checkpoints are written |
| `MLXCACHE_SIDECAR_TIMEOUT_S` | `120` | per-request sidecar timeout; must exceed the slowest prefill |
| `MLXCACHE_ENGINE` | `synthetic` | sidecar engine: `synthetic` or `mlx-lm` |
| `MLXCACHE_MODEL` | `synthetic-model` | model the sidecar loads |
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
Full numbers and caveats in the [design doc](docs/designs/mlx-kv-cache-daemon.md) (R1-5).

## Operating it

See [`docs/ops.md`](docs/ops.md) — bypassing the daemon (SPOF), rollback,
launchd service, and the mlx-lm version-pinning policy.

## Status

Early. Working end to end (cache, persistence, single-flight, streaming, error
rescues, chaos tests). The R1-5 gate has run on Qwen2-0.5B,
Qwen2.5-7B-Instruct-4bit, and the named representative Qwen3-32B-4bit, all
token-identical to scratch. The engine-agnostic contract has one adapter
(mlx-lm). No license assigned — private build, all rights reserved.
