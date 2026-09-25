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

## mlx-lm version pinning

The sidecar adapter wraps mlx-lm's save/load_prompt_cache, which is lossy
across versions. Policy: pin the exact mlx-lm version in pyproject.toml
(`mlx-lm==X.Y.Z`); the T2 round-trip benchmark must be re-run green on every
bump before shipping. A failing benchmark = do not bump.

## Environment

| Variable | Default | Meaning |
|---|---|---|
| MLXCACHE_ADDR | 127.0.0.1:8420 | daemon bind address |
| MLXCACHE_MODELS | (empty) | comma-separated models the daemon serves; requests for others 404 |
| MLXCACHE_SIDECAR_URL | (unset) | sidecar base URL; unset = no adapter, requests 503 |
| MLXCACHE_BENCH_REAL | (unset) | set to 1 for real mlx-lm benchmark (needs local model) |
