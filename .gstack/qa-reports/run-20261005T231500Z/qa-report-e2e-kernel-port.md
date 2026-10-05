# Functional QA Report: kernel-assigned sidecar ports in the e2e harness (issue #31)

| Field | Value |
|---|---|
| Date / branch / revision | 2026-10-05 / e2e/kernel-assigned-port / 8cc4a2d + review-fixes 2f59960 + 5bb7f3e |
| Caller / authority / depth | batch gauntlet (implement → ponytail → review+adversarial → QA → ship → land); permitted writes: harness + report dir |
| Surfaces / scope | `crates/mlxcache-daemon/tests/e2e.rs::spawn_sidecar_with_env` — the spawn path all 27 e2e tests share |
| Runtime / native tools | Rust 1.x, tokio, reqwest; venv python direct-spawn (no uv); macOS |
| Fixture ownership / destinations | per-test tempdirs; sidecar children killed+reaped on every exit path (verified per path) |
| Probe budget / stop reason | e2e suite 27/27 ×5 across fix waves + workspace gates; complete |

## The defect

The harness picked sidecar ports with `portpicker` and trusted a health
answer while the child was alive. Two documented flakes (2026-10-04's
two_models cross-contamination and this week's 1-in-8 `two_models` 503,
issue #31) are that scheme's residual race: a foreign server answering
health inside the alive-check window dies with its own test, and the
daemon 503s a sidecar that was never ours.

## The fix

The child binds **port 0** (kernel-assigned) and prints `PORT=<n>` on
stdout; the harness reads its own child's bind. No foreign process can
know the port, so no foreign server can truthfully answer. The
portpicker retry loop is deleted. A drain thread keeps the child's
stdout open (no mid-test pipe wedge), and `SidecarHandle` kills and
reaps the child on drop.

Review hardening: every spawn-failure path now **panics with cause**
(spawn error, missing/garbled PORT line, 30s deadline, health
exhaustion, sidecar exit before or after a health answer) — the
previous silent-skip converted a broken environment into a green gate
with zero coverage across 27 tests. The alive-check TOCTOU window is
closed by re-checking after a successful health answer.

## Evidence

| Contract | Probe | Result |
|---|---|---|
| e2e green on the new harness | suite reps | 27/27 ×5 (once at 8cc4a2d, thrice at 2f59960, once at 5bb7f3e) |
| Teeth: every failure path is loud | code-path trace per panic + the venv assert | no silent-skip sink remains (grep: 0 `let Some((`, 0 `skipping`) |
| No child leak on any path | per-path audit + delta certification (which caught the health-exhaustion leak, fixed in 5bb7f3e) | kill+wait precedes every panic |
| Workspace gates | cargo 9 suites ✓, pytest 149/7 ✓, fmt ✓, clippy -D warnings ✓ | green |
| Red side | the two documented flakes (issue #31) | structural guarantee: a kernel-assigned port no other process knows cannot be answered by a foreign engine |

## Findings disposition

Full review (2 low: provenance dates, drain-comment mechanism) +
adversarial (1 high silent-skip, 1 medium TOCTOU) + delta certification
(1 medium: health-exhaustion leak) — all fixed; final delta on the
one-hunk fix clean. One adversarial observation recorded as design
documentation rather than a defect: the double-feed mutation is a
phase-aligned no-op on a periodic prompt (see the 0.1.2 QA report).

## Coverage limits

- The port-race fix is structural, not mutation-tested: a kernel-assigned
  port that no other process learns cannot be raced by construction.
- The 15s→30s PORT deadline was chosen for loaded CI runners; a cold start
  slower than 30s still panics (loud, by design).
