//! Minimal observability (D3): per-request structured log line + counters.

use mlxcache_core::policy::{CacheVerdict, PolicyDecision};
use tracing::info;

/// Emit the per-request log line. Fields: model, prefix-hash, verdict,
/// tokens cached/total. Kept allocation-light: all scalars, one format call.
pub fn log_request(model: &str, prefix_hash: u64, decision: &PolicyDecision, ttft_ms: u64) {
    let verdict_str = match decision.verdict {
        CacheVerdict::Hit => "hit",
        CacheVerdict::Miss => "miss",
        CacheVerdict::Partial => "partial",
    };
    info!(
        model = model,
        prefix_hash = prefix_hash,
        verdict = verdict_str,
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
}
