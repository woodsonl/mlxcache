# Functional QA Report: B2.3 — CI audit, metadata verify, stranger-clone dry-run

| Field | Value |
|---|---|
| Date / branch / revision | 2026-10-05 / b23/batch / 196daac + t22-fix 9a4c0c6 + stranger-clone3 evidence |
| Caller / authority / depth | batch gauntlet (implement → ponytail → review+adversarial → QA → ship → land); permitted writes: CI workflow, scripts/, report dir |
| Surfaces / scope | `.github/workflows/ci.yml` (python job lint coverage), the six `scripts/` files, the README quickstart's self-containedness |
| Runtime / native tools | uv 0.16.9 ruff (locked), cargo, fresh git clone at 9a4c0c6 |
| Fixture ownership / destinations | stranger clone in a temp dir; /tmp/mlxcache-blobs2 for the quickstart run |
| Probe budget / stop reason | stranger-clone ×4 (1: harness deviation, dev-profile build; 2: README-verbatim build; 3: quickstart leg hit stale listeners, contaminated; 4: pre-flight free ports, PID ownership), t22 warm+cold end-to-end, full gates; complete |

## Contract outcomes

| Contract and source | Exact probe / evidence | Expected → observed | Outcome |
|---|---|---|---|
| CI lints the gate-critical scripts | `.github/workflows/ci.yml` python job: `ruff check sidecar/ scripts/` + `format --check sidecar/ scripts/` | exact commands pass locally (38 files) → PASS | pass |
| scripts/ actually clean | ruff check scripts/ after fixes | 0 findings (was: 11, of which 5 non-auto-fixable) | pass |
| The repo is self-contained for a stranger | `stranger-clone3.log` (fresh clone at 9a4c0c6 → `uv sync --group dev --locked` 1s → `cargo build --release -p mlxcache-daemon` 54s, warm registry) + `stranger-clone4.log` (README quickstart verbatim on that same clone, ports 8420/8421 pre-verified free, server PIDs logged) | request1 200 → daemon verdict **miss** → SECOND-REQUEST verdict **hit** (`ttft_ms: 0`, `kv_claimed: 7`), checkpoint in MLXCACHE_BLOBS | pass |
| The t22 multi-turn harness works end to end | warm run writes 3 nonempty replies; cold leg passes parity | warm `replies: 3, all nonempty: True`; cold-leg parity OK | pass (fixed this batch) |
| Metadata: version records agree | `sidecar/tests/test_version_agreement.py` | 3/3 PASS | pass |
| Metadata: license posture | Cargo.toml + pyproject both `UNLICENSED` | consistent with the private-repo posture pending the license decision | pass (blocked work tracked) |

## The stranger-clone finding that changed the script

The first dry-run attempt failed at step 3 (`target/release/mlxcache-daemon`
missing) — the dry-run script had deviated from the README (dev-profile
build, missing `MLXCACHE_BLOBS`). The README's own commands are
self-consistent; the second attempt followed them verbatim and passed.
Honest framing on the build: the 54s release build had a warm `~/.cargo`
registry cache; a first-ever cargo user adds crate downloads. The claim
this evidence supports is "the repo's files and quickstart commands are
self-consistent", not "cold toolchain works".

Second contamination, caught in review: clone3's quickstart leg shows both
fresh servers failing to bind (Address already in use, twice) before its
200/hit. Those responses came from stale listeners left running by earlier
attempts, not from the clone. clone4 re-ran the quickstart on the same
9a4c0c6 clone after asserting ports 8420/8421 free and logging the server
PIDs; its daemon logged miss (kv_claimed 0) then hit (ttft_ms 0,
kv_claimed 7) under those PIDs, and the checkpoint in MLXCACHE_BLOBS
carries clone4's mtime. clone3 is evidence for the clone/sync/build legs
only.

## Findings disposition

Review + adversarial found: the stranger-clone evidence missing from the
commit (fixed: log + this receipt committed), warm-build contamination
(fixed: honest framing above), the batch dry-run pinning the parent commit
(fixed: re-run at this branch's HEAD), the t22 replies bug (fixed: verified
end to end), clone3's stale-listener quickstart contamination (fixed:
clone4 re-run with pre-flight free ports and PID ownership, above), and
the PR#32 report riding the wrong commit (process note
for future merges: land receipts belong on or immediately after the merge;
recorded, not retroactively fixable).

## Coverage limits

- The e2e flake's readiness-poll follow-up (issue #31's second bullet) is
  out of this batch's scope; the spawn race itself closed in 0.1.3.
- The Dependency Graph dynamic workflow's capacity failures are GitHub
  infrastructure, not repo-fixable.
