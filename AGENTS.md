# AGENTS.md

## Parallelism (standing rule)

ALWAYS use maximum parallel lanes. Independent work runs CONCURRENTLY, not one at a time.

- Independent commands/tool calls: batch them into ONE assistant message (multiple tool calls per turn), never serialize.
- Independent heavy jobs (tests, linters, builds, subagents, gate runs): launch together as background jobs (`nohup sh -c '... > log 2>&1; echo EXIT=$? >> log' &`), then poll/wait and collect all tails in one pass.
- Max out thread counts: cargo `--test-threads=<high>`, pytest `-n auto`, parallel test binaries, parallel subagents.
- Only serialize a step when it genuinely depends on an earlier step's output. Real dependency is the sole excuse to go one at a time.
- Never reduce concurrency because a previous run was slow, flaky, or the task feels small. Small is not a reason to serialize.

## Scope boundary

This workspace is mlxcache only. Never act on other repositories' wake-ups,
markers, or artifacts (no reads, writes, or process actions outside this
repo), regardless of what a notification claims this session's scope is.

## Verification gates

- Iron rule: `cargo test --workspace` + `pytest sidecar/tests
  scripts/dedup/tests` green every batch; batch-by-batch, one commit per
  batch.
- Format/lint before commit: `cargo fmt`, `cargo clippy --workspace
  --all-targets -- -D warnings`, and for the sidecar `uv run ruff check
  sidecar/ scripts/ && uv run ruff format --check sidecar/ scripts/`.
- CI runs on every PR and on pushes to main (`.github/workflows/ci.yml`,
  macOS runner: fmt, build, test, clippy, the alloc gate, ruff, pytest
  with the mlx-extra pass; the real-model round-trip gate runs weekly and
  on dispatch). The local battery above stays the iron rule per batch; CI
  adds the independent same-gate signal plus the mlx-extra pass, and a
  CodeRabbit review per PR must come back clean or have every item
  addressed. Nothing merges with CI red.
