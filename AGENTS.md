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

- Iron rule: `cargo test --workspace` + `pytest sidecar/tests` green every
  batch; batch-by-batch, one commit per batch.
- Format/lint before commit: `cargo fmt`, `cargo clippy --workspace
  --all-targets -- -D warnings`.
