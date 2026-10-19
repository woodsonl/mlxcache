# mlxcache Operations

## SPOF bypass (single point of failure)

The daemon sits on every request path. If it crashes, clients lose inference
until it restarts. Bypass: point clients directly at the engine (sidecar or
owner engine endpoint) — the daemon is a pure middleman with no client state
to migrate. Recovery is a config change, zero downtime.

## Rollback

Rollback = the bypass above: repoint `base_url` from the daemon to the engine
directly. The daemon holds no authoritative state (checkpoints are an
optimization; losing them costs prefill time, not correctness).

## Service management (launchd)

For uvx/source installs, install the plist:

```
cp docs/com.mlxcache.daemon.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.mlxcache.daemon.plist
```

KeepAlive keeps the daemon running; a crash restarts it within seconds.
Streams in flight drop (R1-4); clients retry.

## Checkpoint durability and recovery

Checkpoints are an optimization; the daemon must serve correctly with an empty,
partial, or corrupt blob directory.

- **Immutable per-generation files.** Each publication writes a new file named
  `{hash}-{generation}.ckpt`. A generation is a monotonic counter reserved per
  publication, so a delayed retirement unlinks only its own file and never a
  republish that replaced it.
- **Startup rebuild.** `rebuild_from_disk` runs before the daemon binds its
  listener. It recovers in two passes: group blobs by token prefix, keep the
  highest generation per prefix, publish the winners, then delete the superseded
  files. Blobs that cache nothing or cannot be keyed — a prefix shorter than 2,
  a recorded count that disagrees with the prefix length, or an empty KV
  payload — are skipped and left on disk.
- **Generation floor.** The floor is computed from every filename it can parse
  before any blob is read, and seeded into the counter (highest `{generation}`
  wins, via `fetch_max`) before recovered entries are re-published, so a fresh
  publication can never reuse a generation that an existing file already
  claims. A filename that does not match the daemon's own
  `{hash:032x}-{generation:016x}.ckpt` shape is treated as foreign: it joins
  selection at generation 0 and is reclaimed if a parseable blob claims the same
  prefix. The daemon only ever writes parseable names, so this affects files not
  produced by this daemon (legacy or hand-placed).
- **Publish is refused while the floor is unknown.** If the startup scan fails
  (e.g. an unreadable blob directory), the daemon sets a recovered-failure
  guard and refuses all publications. Requests still serve — they run a full
  prefill and simply do not persist — so a failed scan degrades to
  no-cache, never to unindexed files or resurrected poison on the next start.

## Qwen3.8-Flash-Next residency (`qwen4-exp`)

`MLXCACHE_ENGINE=qwen4-exp` serves a local Qwen3.8-Flash-Next NVFP4
checkpoint. The checkpoint is 132.6 GB and the target machine has 48 GB of
unified memory, so the weights cannot be resident: the engine streams from
disk and keeps a bounded working set in memory. Nothing else in the cache
path changes. Persistence, digests, eviction, and delta resume are the same
engine-generic code the mlx-lm adapter uses.

### What stays resident, what streams

Measured checkpoint layout (132.6 GB total):

| family | bytes |
|---|---|
| experts (48 layers x 512 experts), NVFP4 packed `U8` + per-expert scales | 70.5 GB |
| PLE n-gram embedding table, fp8_e4m3, 128 shards, 320,001,536 rows x 160 | 51.2 GB |
| linear attention (gated-delta-net projections) | 4.2 GB |
| multi-token-prediction head | 2.7 GB |
| embeddings + `lm_head` | 2.5 GB |
| self attention | 1.3 GB |
| vision tower | ~0.9 GB |

(Families are measured and rounded independently, so they do not sum to the
132.6 GB total exactly.)

Resident: the non-expert weights (linear attention, self attention,
embeddings, `lm_head`, MTP head, vision tower), plus the decoded expert
window and the decoded PLE rows the two knobs below bound. Streamed per
token or per layer: expert weights (dequantized on use) and PLE rows (read
through the shard manifest, one hash lookup per n-gram head, 24 lookups per
PLE layer).

### Architecture in operator terms

48 layers with `full_attention_interval=4`: 12 full-attention layers and 36
gated-delta-net linear-attention layers. Each layer routes to 512 experts
top-10 plus a shared expert, carries hyper-connection residual streams, and
the full-attention layers run an indexer ("QSA") that budgets which key
blocks are attended. On top of the text stack sit a PLE n-gram embedding
layer, a multi-token-prediction head, and a vision tower.

Two consequences an operator sees:

- **The cache is not one shape.** Full-attention layers cache token KV plus
  QSA indexer key blocks; linear-attention layers cache convolution and
  recurrent state plus the PLE token context. That is why checkpoint
  identity carries a `cache_layout` signature: a payload of one composition
  is never served to an engine that would deserialize it as another.

### Knobs

| Variable | Default | Effect |
|---|---|---|
| `MLXCACHE_Q4_EXPERT_RESIDENT_N` | `0` | decoded experts per layer kept in an LRU. Raising it trades RAM for steady-state decode speed: a router that picks an expert already decoded skips the read and the dequantize. `0` keeps no expert between tokens, so every routed expert is read and dequantized again. |
| `MLXCACHE_Q4_PLE_CACHE_MB` | `512` | byte budget for decoded PLE rows. The table is a hash lookup, so the touch set per token is small; the LRU caches rows, never the table. Raising it buys hit rate on repeated n-gram heads across a conversation; lowering it forces re-reads from the shards. |

Both knobs are per process; the daemon's own memory and the KV checkpoint
payload are outside them.

Also in force for this engine:

- `MLXCACHE_KV_BITS` / `MLXCACHE_KV_GROUP_SIZE`: the KV quantization tier,
  folded into the fingerprint like every other engine. This model's cache is
  a mixed composition (full-attention KV plus linear-attention recurrent
  state), so the dense-only q8 tier does not apply; leave it at `0`.
- `MLXCACHE_METAL_CACHE_MB` (default `1024`): the MLX allocator pool limit.
  Streaming allocates a new size class per layer shape, so a bound here
  matters more than for a resident model: freed buffers beyond it return to
  the OS instead of being pooled. `0` keeps the library default.

### Failure mode when a budget is too small

Expect slowness, not a crash. Neither knob is a correctness bound: a too
small `MLXCACHE_Q4_EXPERT_RESIDENT_N` or `MLXCACHE_Q4_PLE_CACHE_MB` turns
into repeated disk reads and repeated dequantization, so decode slows down
while the process stays up and keeps producing correct tokens. That is an
expectation from the design (a cache miss on either structure re-reads the
same bytes), not a measurement. The observable signal is decode throughput
falling as the budgets shrink with everything else held constant. Watch
`/stats` for cache behavior and the process RSS for the resident set; a
working set that exceeds the machine shows up as memory pressure from
outside the engine, not as an engine error.

### Requirements and troubleshooting

- Point `MLXCACHE_MODEL` at the checkpoint directory (the one holding
  `config.json`, the sharded safetensors, and `ple_fp8_manifest.json`), not
  at an HF repo id. The engine does not download.
- Disk: 132.6 GB for the checkpoint. Because weights stream, the directory
  must stay readable for the life of the process; it is not a load-once
  artifact. Keep it on local storage: a network mount turns every expert
  read into a latency term.
- Hermetic tests for this engine skip when the checkpoint is absent, so a
  CI machine needs nothing downloaded.

Symptom to check, in order:

1. **Decode far slower than expected, memory comfortable.** Raise
   `MLXCACHE_Q4_EXPERT_RESIDENT_N` first (expert reads dominate the
   streamed bytes), then `MLXCACHE_Q4_PLE_CACHE_MB`.
2. **Memory pressure or swap.** Lower both, and lower
   `MLXCACHE_METAL_CACHE_MB`. The allocator pool and the two LRUs are the
   three things the engine holds beyond the non-expert weights.
3. **Boot fails on the checkpoint.** Confirm the path is the checkpoint
   directory itself and that shard symlinks resolve inside it.
4. **No cache hits at all.** Check the daemon serves the same model id the
   sidecar reports, as with any engine. Identity includes `engine_id`
   (`qwen4-exp`) and `cache_layout`, so blobs from another engine can never
   be served here.

## mlx-lm version pinning

The sidecar adapter wraps mlx-lm's save/load_prompt_cache, which is lossy
across versions. Policy: pin the exact mlx-lm version in pyproject.toml
(`mlx-lm==X.Y.Z`); the T2 round-trip benchmark must be re-run green on every
bump before shipping. A failing benchmark = do not bump.

## Environment

Daemon:

| Variable | Default | Meaning |
|---|---|---|
| MLXCACHE_ADDR | 127.0.0.1:8420 | daemon bind address |
| MLXCACHE_MODELS | (empty) | comma-separated models the daemon serves; requests for others 404 |
| MLXCACHE_SIDECAR_URL | (unset) | sidecar base URL; unset = no adapter, requests 503 |
| MLXCACHE_BLOBS | /tmp/mlxcache-blobs | checkpoint blob directory; set a durable path for a real install |
| MLXCACHE_SIDECAR_TIMEOUT_S | 120 | per-request sidecar timeout; must exceed the slowest prefill |
| RUST_LOG | info | tracing filter for the daemon's JSON logs |

Sidecar (compatibility adapter):

| Variable | Default | Meaning |
|---|---|---|
| MLXCACHE_ENGINE | synthetic | `mlx-lm` loads a real model; `qwen4-exp` serves a local NVFP4 Qwen3.8-Flash-Next checkpoint; `synthetic` is hermetic |
| MLXCACHE_MODEL | synthetic-model | model id to load (an mlx-lm repo or local path; for `qwen4-exp`, the checkpoint directory) |
| MLXCACHE_SIDECAR_ADDR | 127.0.0.1 | sidecar bind address |
| MLXCACHE_SIDECAR_PORT | 8421 | sidecar bind port |
| MLXCACHE_Q4_EXPERT_RESIDENT_N | 0 | `qwen4-exp`: decoded experts per layer kept in an LRU |
| MLXCACHE_Q4_PLE_CACHE_MB | 512 | `qwen4-exp`: byte budget for decoded PLE n-gram rows |
| MLXCACHE_METAL_CACHE_MB | 1024 | MLX allocator cache limit in MB; 0 keeps the library default |
| MLXCACHE_BENCH_REAL | (unset) | set to 1 for the real mlx-lm benchmark (needs a local model) |
