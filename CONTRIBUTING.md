# Contributing to mlxcache

Bug reports, measurements, and patches are welcome. The bar this repo holds
is that claims come with evidence: a benchmark number names its harness, a
bug report names its repro, a fix carries a regression that failed before
it.

## The gate battery

Every change lands with this battery green, run from the repo root:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo test -p mlxcache-daemon --test alloc_gate --release -- \
  --test-threads=1 --include-ignored
uv run pytest sidecar/tests/ scripts/dedup/tests/
uv run ruff check sidecar/ scripts/
uv run ruff format --check sidecar/ scripts/
```

CI runs on every PR with this battery plus the alloc gate and the
mlx-extra test pass; run the battery locally per batch anyway — CI is the
independent signal, and nothing merges with it red.
Real-engine gates (`MLXCACHE_BENCH_REAL=1` round-trip, prefix benchmarks)
also stay opt-in local: they need a downloaded model and a quiet machine.

## How to submit

1. Branch from `main`.
2. Work in batches: one coherent change per commit, the battery green at
   every commit, never a mixed batch. Review history shows this shape; keep
   it.
3. Open a PR against `main`. Every PR gets a CodeRabbit review; it must come
   back clean or have every item addressed in a follow-up commit that names
   the finding.
4. Updates to measured claims (README "Measured", REPORT.md,
   docs/benchmarks.md) must come from a checked-in harness run — a number
   with no script behind it will be asked for its script.

## Conventions worth knowing

- Checkpoints are an optimization, never a correctness dependency: the
  daemon must serve correctly with an empty, partial, or corrupt blob
  directory. Changes that make a healthy cache load-bearing will be pushed
  back on.
- The `tokens[:-1]` convention (a blob at token stream T covers T minus its
  last token) runs through everything; respect it in any new reader or
  writer.
- One chunk-format writer per store directory (the daemon); Python
  publishes whole-blob. Both layouts coexist forever; do not add a second
  chunk-format writer.
- Comments explain constraints and invariants, not history.

`AGENTS.md` carries the agent-facing version of these rules; if the two ever
disagree, this file wins for human contributors and a PR should reconcile
them.
