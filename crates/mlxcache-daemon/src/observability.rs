//! Minimal observability (D3): per-request structured log line + counters.

use mlxcache_core::policy::{covered_kv_tokens, CacheVerdict, PolicyDecision};
use tracing::info;

/// Emit the per-request log line. Fields: model, prefix-hash, verdict, claimed
/// KV reuse/total, decision time. Kept allocation-light: all scalars, one
/// format call. This records the cache DECISION; the effective reuse is reported
/// in the response (prefill_from) and corrected here only on fallback. The
/// claimed/covered count comes from `policy::covered_kv_tokens`, the same
/// definition the response's `prefill_from`/`tokens_cached` use, so the three
/// fields cannot drift apart by a token.
pub fn log_request(model: &str, prefix_hash: u128, decision: &PolicyDecision, ttft_ms: u64) {
    let verdict_str = match decision.verdict {
        CacheVerdict::Hit => "hit",
        CacheVerdict::Miss => "miss",
        CacheVerdict::Partial => "partial",
    };
    let covered = covered_kv_tokens(decision.verdict, decision.matched_tokens);
    info!(
        model = model,
        prefix_hash = prefix_hash,
        verdict = verdict_str,
        kv_claimed = covered,
        tokens_cached = covered,
        tokens_total = decision.request_tokens,
        ttft_ms = ttft_ms,
        "request"
    );
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use mlxcache_core::policy::PolicyDecision;

    /// A decision for the trace/log tests (kept public to crate siblings).
    pub(crate) fn decision(
        verdict: CacheVerdict,
        matched_tokens: usize,
        request_tokens: usize,
    ) -> PolicyDecision {
        PolicyDecision {
            verdict,
            matched_tokens,
            request_tokens,
        }
    }

    #[test]
    fn log_line_does_not_panic() {
        let d = decision(CacheVerdict::Hit, 50, 60);
        log_request("test-model", 0xdeadbeef, &d, 12);
    }
}
