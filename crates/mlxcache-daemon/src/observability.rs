//! Minimal observability (D3): per-request structured log line + counters.

use mlxcache_core::policy::{CacheVerdict, PolicyDecision};
use tracing::info;

/// Tokens whose KV came from the cache: matched_tokens - 1 for hit/partial
/// (the adapter caches tokens[:-1]), 0 for miss. Mirrors the response's
/// prefill_from so logs and the API agree.
fn tokens_cached(verdict: CacheVerdict, matched_tokens: usize) -> usize {
    match verdict {
        CacheVerdict::Miss => 0,
        CacheVerdict::Hit | CacheVerdict::Partial => matched_tokens.saturating_sub(1),
    }
}

/// Emit the per-request log line. Fields: model, prefix-hash, verdict,
/// tokens cached/total. Kept allocation-light: all scalars, one format call.
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
        tokens_cached = tokens_cached(decision.verdict, decision.matched_tokens),
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
    fn tokens_cached_matches_covered_kv() {
        // Hit/partial report matched-1 (KV for tokens[:-1]); miss reports 0.
        assert_eq!(tokens_cached(CacheVerdict::Hit, 50), 49);
        assert_eq!(tokens_cached(CacheVerdict::Partial, 4), 3);
        assert_eq!(tokens_cached(CacheVerdict::Miss, 4), 0);
        // Degenerate match of 0 must not underflow.
        assert_eq!(tokens_cached(CacheVerdict::Partial, 0), 0);
    }
}
