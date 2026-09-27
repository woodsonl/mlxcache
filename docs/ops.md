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
| MLXCACHE_ENGINE | synthetic | `mlx-lm` loads a real model; `synthetic` is hermetic |
| MLXCACHE_MODEL | synthetic-model | model id to load (an mlx-lm repo or local path) |
| MLXCACHE_SIDECAR_ADDR | 127.0.0.1 | sidecar bind address |
| MLXCACHE_SIDECAR_PORT | 8421 | sidecar bind port |
| MLXCACHE_BENCH_REAL | (unset) | set to 1 for the real mlx-lm benchmark (needs a local model) |
