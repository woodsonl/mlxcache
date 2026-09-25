//! Policy core: eviction, warming decisions, hit/miss classification.
//!
//! ds4-derived policy: protect extendable anchors (checkpoints a live session can
//! extend); evict cold, non-extendable blobs first. Hit-rate is the diagnostic;
//! total cost + p95 latency per completed task is the value (premise 5).

use crate::contract::ModelFingerprint;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheVerdict {
    /// Exact or longest-prefix hit; adapter can adopt the KV.
    Hit,
    /// No compatible checkpoint; full prefill required.
    Miss,
    /// Partial match: adopt longest checkpoint, prefill the delta.
    Partial,
}

/// Classification of a lookup against the index + fingerprint rules (R1-1).
pub fn classify(
    matched_tokens: Option<usize>,
    request_tokens: usize,
    matched_fingerprint: Option<&ModelFingerprint>,
    request_fingerprint: &ModelFingerprint,
) -> CacheVerdict {
    match matched_tokens {
        Some(0) => CacheVerdict::Miss,
        Some(n) => {
            if matched_fingerprint != Some(request_fingerprint) {
                return CacheVerdict::Miss;
            }
            if n == request_tokens {
                CacheVerdict::Hit
            } else {
                CacheVerdict::Partial
            }
        }
        None => CacheVerdict::Miss,
    }
}

/// Eviction candidate scoring. Higher score = evict first.
/// Anchors (recently extended) score low and survive; cold stale blobs score high.
#[derive(Debug, Clone)]
pub struct EvictionCandidate {
    pub token_count: u64,
    pub last_used: Instant,
    /// Whether any active session's prefix extends this checkpoint.
    pub is_anchor: bool,
}

pub fn eviction_score(c: &EvictionCandidate) -> u64 {
    if c.is_anchor {
        return 0; // never evict anchors (ds4 lesson)
    }
    c.token_count
}

/// A single policy decision record (feeds the per-request log line, D3).
#[derive(Debug, Clone)]
pub struct PolicyDecision {
    pub verdict: CacheVerdict,
    pub matched_tokens: usize,
    pub request_tokens: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fp(id: &str) -> ModelFingerprint {
        ModelFingerprint {
            model_id: id.into(),
            tokenizer_hash: "h".into(),
            kv_dtype: "f16".into(),
            kv_layout_version: 1,
        }
    }

    #[test]
    fn classify_hit_partial_miss() {
        let f = fp("m");
        assert_eq!(classify(Some(5), 5, Some(&f), &f), CacheVerdict::Hit);
        assert_eq!(classify(Some(3), 5, Some(&f), &f), CacheVerdict::Partial);
        assert_eq!(classify(None, 5, None, &f), CacheVerdict::Miss);
    }

    #[test]
    fn fingerprint_mismatch_is_miss() {
        let a = fp("m1");
        let b = fp("m2");
        assert_eq!(classify(Some(5), 5, Some(&a), &b), CacheVerdict::Miss);
    }

    #[test]
    fn anchors_never_evicted() {
        let anchor = EvictionCandidate {
            token_count: 100_000,
            last_used: Instant::now(),
            is_anchor: true,
        };
        let cold = EvictionCandidate {
            token_count: 10,
            last_used: Instant::now(),
            is_anchor: false,
        };
        assert_eq!(eviction_score(&anchor), 0);
        assert!(eviction_score(&cold) > 0);
    }
}
