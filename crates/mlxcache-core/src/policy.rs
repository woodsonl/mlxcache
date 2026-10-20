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

/// Cache-composition compatibility (the B1 identity rule).
///
/// `cache_layout` is a signature of the ordered tuple of cache-entry classes
/// the engine builds for a model — the SHAPE of the payload. `""` means
/// "composition unknown", the value every pre-B1 blob carries (and the value
/// any engine that does not report one would carry).
///
/// The rule: an entry is compatible only with a request reporting the SAME
/// layout string. Consequences are both deliberate:
///
/// - a request with a KNOWN (non-empty) layout is never served an
///   unknown-layout entry: nothing proves that entry's payload has the shape
///   the request's engine will deserialize, and a dense KV stack loaded as a
///   hybrid model's recurrent state is silent wrong output, not an error;
/// - an unknown-layout request may still be served an unknown-layout entry,
///   so a store written before layouts were recorded keeps working until the
///   current engine rewrites its blobs (legacy-to-legacy only — the moment
///   either side knows its composition, the comparison is exact).
pub fn cache_layout_compatible(entry_layout: &str, request_layout: &str) -> bool {
    entry_layout == request_layout
}

/// Content identity for the non-text part of a prompt: images, and any future
/// modality that tokenizes to ordinary ids.
///
/// Exact equality, deliberately, and NOT the legacy-tolerant rule that
/// `cache_layout` uses. An empty digest means "no non-text content", which is
/// a claim about the request, not a missing measurement: a text-only entry and
/// a text-only request agree, and a text-only entry can never serve a request
/// that carries a digest. There is no "unknown" state to be tolerant of — the
/// daemon computes this from the request itself, so a non-empty request digest
/// is always known before lookup.
pub fn content_digest_compatible(entry_digest: &str, request_digest: &str) -> bool {
    entry_digest == request_digest
}

/// The single identity relation every matching site uses (R1-1): the
/// fingerprint fields AND the cache-composition rule above. Field-by-field on
/// purpose — the composition rule is a named decision, not an accident of
/// `PartialEq` on the struct, so it cannot be dropped by adding a field.
pub fn identity_compatible(entry: &ModelFingerprint, request: &ModelFingerprint) -> bool {
    entry.model_id == request.model_id
        && entry.tokenizer_hash == request.tokenizer_hash
        && entry.kv_dtype == request.kv_dtype
        && entry.kv_layout_version == request.kv_layout_version
        && entry.kv_bits == request.kv_bits
        && entry.kv_group_size == request.kv_group_size
        && cache_layout_compatible(&entry.cache_layout, &request.cache_layout)
        && content_digest_compatible(&entry.content_digest, &request.content_digest)
}

/// Classification of a lookup against the index + fingerprint rules (R1-1).
///
/// A match of fewer than 2 tokens is treated as a miss: a checkpoint caches KV
/// for `tokens[:-1]`, so it holds nothing until it covers 2 tokens. The index
/// refuses to publish such prefixes, so this is defense in depth against any
/// path that still produces one — a hit/partial with zero covered KV would hand
/// the adapter a blob to resume from while reporting no reuse.
pub fn classify(
    matched_tokens: Option<usize>,
    request_tokens: usize,
    matched_fingerprint: Option<&ModelFingerprint>,
    request_fingerprint: &ModelFingerprint,
) -> CacheVerdict {
    match matched_tokens {
        Some(n) if n < 2 => CacheVerdict::Miss,
        Some(n) => {
            if !matched_fingerprint.is_some_and(|m| identity_compatible(m, request_fingerprint)) {
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

/// Tokens whose KV a decision actually covers: `matched_tokens - 1` for
/// hit/partial (the adapter caches `tokens[:-1]`), 0 for miss. This is the one
/// definition of the client-facing `prefill_from` / `tokens_cached` /
/// `kv_claimed` value, so those fields cannot drift apart by a token.
pub fn covered_kv_tokens(verdict: CacheVerdict, matched_tokens: usize) -> usize {
    match verdict {
        CacheVerdict::Miss => 0,
        CacheVerdict::Hit | CacheVerdict::Partial => matched_tokens.saturating_sub(1),
    }
}

/// The canonical "this request cached nothing" decision: a miss with zero
/// matched tokens. One definition so every force-scratch site (empty adapter
/// payload, publish refusal, publish failure, follower of a no-publish leader)
/// reports identical verdicts and cannot drift apart.
pub fn scratch_decision(request_tokens: usize) -> PolicyDecision {
    PolicyDecision {
        verdict: CacheVerdict::Miss,
        matched_tokens: 0,
        request_tokens,
    }
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
            ..Default::default()
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
    fn classify_match_under_two_is_miss() {
        // A match of <2 tokens covers no KV; never hit/partial. A 1-token
        // "exact" match (n == request_tokens == 1) must still be a miss, or the
        // request would be a hit with zero covered KV.
        let f = fp("m");
        assert_eq!(classify(Some(1), 1, Some(&f), &f), CacheVerdict::Miss);
        assert_eq!(classify(Some(1), 5, Some(&f), &f), CacheVerdict::Miss);
        assert_eq!(classify(Some(0), 5, Some(&f), &f), CacheVerdict::Miss);
    }

    #[test]
    fn fingerprint_mismatch_is_miss() {
        let a = fp("m1");
        let b = fp("m2");
        assert_eq!(classify(Some(5), 5, Some(&a), &b), CacheVerdict::Miss);
    }

    fn fp_with_layout(layout: &str) -> ModelFingerprint {
        ModelFingerprint {
            cache_layout: layout.into(),
            ..fp("m")
        }
    }

    fn fp_with_digest(digest: &str) -> ModelFingerprint {
        ModelFingerprint {
            content_digest: digest.into(),
            ..fp("m")
        }
    }

    #[test]
    fn same_tokens_different_content_is_a_miss() {
        // The collision this axis exists to prevent: a multimodal prompt
        // renders its media as ordinary token ids, so two different images
        // produce ONE token sequence and two different KV states. If the token
        // prefix is the whole identity, the second request is served the first
        // image's KV — no error, wrong answer.
        let image_a = fp_with_digest("a1b2");
        let image_b = fp_with_digest("c3d4");
        assert_eq!(
            classify(Some(5), 5, Some(&image_a), &image_b),
            CacheVerdict::Miss
        );

        // Same image, same tokens: still a hit.
        assert_eq!(
            classify(Some(5), 5, Some(&image_a), &fp_with_digest("a1b2")),
            CacheVerdict::Hit
        );
    }

    #[test]
    fn text_only_never_matches_or_is_matched_by_an_image() {
        // Empty means "no non-text content", not "unknown". A text-only entry
        // cannot prove it holds the KV for a request carrying pixels, and an
        // image entry must not be served to a text-only request either. Both
        // directions miss; only text-to-text agrees.
        let text = fp_with_digest("");
        let image = fp_with_digest("a1b2");
        assert_eq!(
            classify(Some(5), 5, Some(&text), &image),
            CacheVerdict::Miss
        );
        assert_eq!(
            classify(Some(5), 5, Some(&image), &text),
            CacheVerdict::Miss
        );
        assert_eq!(
            classify(Some(5), 5, Some(&text), &fp_with_digest("")),
            CacheVerdict::Hit
        );
    }

    #[test]
    fn known_layout_never_served_an_unknown_layout_blob() {
        // The legacy rule: a request that KNOWS its cache composition can
        // never prove an unrecorded (pre-B1) blob has that composition, and
        // a payload of the wrong shape is silent wrong output. Miss.
        let legacy = fp_with_layout("");
        let request = fp_with_layout("KV-KV");
        assert_eq!(
            classify(Some(5), 5, Some(&legacy), &request),
            CacheVerdict::Miss
        );
        assert_eq!(
            classify(Some(3), 5, Some(&legacy), &request),
            CacheVerdict::Miss
        );
    }

    #[test]
    fn unknown_layout_request_may_use_an_unknown_layout_blob() {
        // Legacy-to-legacy: an engine that reports no composition keeps
        // serving the store written before compositions were recorded, so an
        // upgrade does not invalidate an entire cache directory at once.
        let legacy = fp_with_layout("");
        let request = fp_with_layout("");
        assert_eq!(
            classify(Some(5), 5, Some(&legacy), &request),
            CacheVerdict::Hit
        );
        assert_eq!(
            classify(Some(3), 5, Some(&legacy), &request),
            CacheVerdict::Partial
        );
    }

    #[test]
    fn differing_layouts_never_cross_serve() {
        // The reason B1 exists: a dense-transformer blob and a hybrid-model
        // blob of the same model/tokenizer/tier are different payload shapes.
        let dense = fp_with_layout("KV-KV");
        let hybrid = fp_with_layout("CacheList-ArraysCache-KV");
        assert_eq!(
            classify(Some(5), 5, Some(&dense), &hybrid),
            CacheVerdict::Miss
        );
        assert_eq!(
            classify(Some(5), 5, Some(&hybrid), &dense),
            CacheVerdict::Miss
        );
        assert_eq!(
            classify(Some(5), 5, Some(&dense), &dense),
            CacheVerdict::Hit
        );
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

    #[test]
    fn covered_kv_is_matched_minus_one() {
        // The one definition of cached/covered KV: matched-1 for hit/partial
        // (adapter caches tokens[:-1]), 0 for miss, and never underflows.
        assert_eq!(covered_kv_tokens(CacheVerdict::Hit, 8), 7);
        assert_eq!(covered_kv_tokens(CacheVerdict::Partial, 8), 7);
        assert_eq!(covered_kv_tokens(CacheVerdict::Miss, 8), 0);
        assert_eq!(covered_kv_tokens(CacheVerdict::Partial, 0), 0);
        assert_eq!(covered_kv_tokens(CacheVerdict::Hit, 1), 0);
    }

    #[test]
    fn scratch_decision_is_a_zero_coverage_miss() {
        // The single force-scratch constructor: a miss that claims no matched
        // tokens and no covered KV, for whatever the request length was.
        let d = scratch_decision(17);
        assert_eq!(d.verdict, CacheVerdict::Miss);
        assert_eq!(d.matched_tokens, 0);
        assert_eq!(d.request_tokens, 17);
        assert_eq!(covered_kv_tokens(d.verdict, d.matched_tokens), 0);
    }
}
