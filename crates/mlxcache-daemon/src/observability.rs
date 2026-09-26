//! Minimal observability (D3): per-request structured log line + counters.

use mlxcache_core::policy::{CacheVerdict, PolicyDecision};
use tracing::info;

/// Tokens whose KV the cache decision expects to reuse: matched_tokens - 1 for
/// hit/partial (the adapter caches tokens[:-1]), 0 for miss. This is the
/// decision's claim, logged BEFORE the checkpoint is opened; if adoption later
/// falls back to scratch (quarantine), the response reports prefill_from=0 and a
/// warn line records the mismatch. Same semantics as the response's
/// prefill_from when adoption succeeds.
fn kv_claimed(verdict: CacheVerdict, matched_tokens: usize) -> usize {
    match verdict {
        CacheVerdict::Miss => 0,
        CacheVerdict::Hit | CacheVerdict::Partial => matched_tokens.saturating_sub(1),
    }
}

/// Emit the per-request log line. Fields: model, prefix-hash, verdict, claimed
/// KV reuse/total, decision time. Kept allocation-light: all scalars, one
/// format call. This records the cache DECISION; the effective reuse is reported
/// in the response (prefill_from) and corrected here only on fallback.
pub fn log_request(model: &str, prefix_hash: u128, decision: &PolicyDecision, ttft_ms: u64) {
    let verdict_str = match decision.verdict {
        CacheVerdict::Hit => "hit",
        CacheVerdict::Miss => "miss",
        CacheVerdict::Partial => "partial",
    };
    info!(
        model = model,
        prefix_hash = prefix_hash,
        verdict = verdict_str,
        kv_claimed = kv_claimed(decision.verdict, decision.matched_tokens),
        tokens_cached = decision.matched_tokens,
        tokens_total = decision.request_tokens,
        ttft_ms = ttft_ms,
        "request"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlxcache_core::policy::PolicyDecision;

    #[test]
    fn log_line_does_not_panic() {
        let d = PolicyDecision {
            verdict: CacheVerdict::Hit,
            matched_tokens: 50,
            request_tokens: 60,
        };
        log_request("test-model", 0xdeadbeef, &d, 12);
    }

    #[test]
    fn kv_claimed_matches_decided_reuse() {
        // Hit/partial claim matched-1 (KV for tokens[:-1]); miss claims 0.
        assert_eq!(kv_claimed(CacheVerdict::Hit, 50), 49);
        assert_eq!(kv_claimed(CacheVerdict::Partial, 4), 3);
        assert_eq!(kv_claimed(CacheVerdict::Miss, 4), 0);
        // Degenerate match of 0 must not underflow.
        assert_eq!(kv_claimed(CacheVerdict::Partial, 0), 0);
    }
}
