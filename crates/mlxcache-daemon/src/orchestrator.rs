//! Policy orchestration: the request pipeline from HTTP to adapter.
//!
//! Flow (design doc data flow): validate → tokenize (adapter, R1-2) → index
//! lookup → classify (R1-1) → single-flight prefill (R1-3) → persist → stream.
//! Client disconnect mid-prefill still persists the checkpoint (Section 4).

use mlxcache_core::contract::{CheckpointMeta, ModelFingerprint};
use mlxcache_core::index::PrefixIndex;
use mlxcache_core::policy::{classify, CacheVerdict, PolicyDecision};
use mlxcache_core::singleflight::SingleFlight;
use std::sync::Arc;

/// What the daemon knows about the request after the index step.
pub struct RouteOutcome {
    pub decision: PolicyDecision,
    /// Covered KV: how many leading tokens the cache already holds, i.e. where
    /// prefill resumes. `matched_tokens - 1` for hit/partial, 0 for miss.
    pub prefill_from: usize,
    /// Published blob to resume from, with its publication generation and the
    /// token prefix it is keyed by (None on a full miss). The daemon hands the
    /// path to the adapter so the engine loads cached KV instead of
    /// re-prefilling; the generation and prefix let a late failure retire exactly
    /// this publication by descending its key, not a full-index scan. Kept as one
    /// option so the three cannot desynchronize.
    pub blob: Option<(String, u64, Vec<u32>)>,
}

pub struct Orchestrator {
    pub index: Arc<PrefixIndex>,
    pub singleflight: Arc<SingleFlight>,
}

impl Default for Orchestrator {
    fn default() -> Self {
        Self::new()
    }
}

impl Orchestrator {
    pub fn new() -> Self {
        Self {
            index: Arc::new(PrefixIndex::new()),
            singleflight: Arc::new(SingleFlight::new()),
        }
    }

    /// Route a tokenized request: lookup, classify, decide prefill origin.
    /// Tokenization and fingerprint come from the adapter (R1-2); the daemon
    /// never re-tokenizes.
    pub fn route(&self, tokens: &[u32], request_fingerprint: &ModelFingerprint) -> RouteOutcome {
        let lookup = self.index.lookup(tokens);
        let (matched_tokens, matched_fp, blob_path, blob_generation) = match &lookup {
            Some((entry, n)) => (
                Some(*n),
                Some(&entry.meta.fingerprint),
                Some(entry.blob_path.clone()),
                Some(entry.generation),
            ),
            None => (None, None, None, None),
        };
        let verdict = classify(
            matched_tokens,
            tokens.len(),
            matched_fp,
            request_fingerprint,
        );
        // A fingerprint mismatch classifies as Miss and must not reuse the blob.
        // The matched prefix is `tokens[..n]`, the exact key the entry lives at.
        let blob = match verdict {
            CacheVerdict::Miss => None,
            _ => blob_path
                .zip(blob_generation)
                .map(|(p, g)| (p, g, tokens[..matched_tokens.unwrap_or(0)].to_vec())),
        };
        // `prefill_from` is the client-facing count of tokens already covered by
        // cached KV, i.e. where prefill resumes. A checkpoint published for a
        // prefix of length N holds KV for N-1 tokens (the adapter caches
        // tokens[:-1]; see MlxLmEngine.prefill), so a hit or partial at a prefix
        // of length M reports M-1. Miss reports 0. This is the same value the
        // response's `tokens_cached` and the log's `kv_claimed` report, computed
        // once in `policy::covered_kv_tokens`.
        let prefill_from =
            mlxcache_core::policy::covered_kv_tokens(verdict, matched_tokens.unwrap_or(0));
        RouteOutcome {
            decision: PolicyDecision {
                verdict,
                matched_tokens: matched_tokens.unwrap_or(0),
                request_tokens: tokens.len(),
            },
            prefill_from,
            blob,
        }
    }

    /// Register a completed prefill: capture metadata and publish atomically.
    /// The blob rename must already be done (R1-3 ordering). Returns the
    /// publication generation so the caller can retire exactly this publication
    /// if the adapter later rejects the blob.
    pub fn publish_checkpoint(
        &self,
        tokens: &[u32],
        meta: CheckpointMeta,
        blob_path: String,
    ) -> u64 {
        self.index.publish(tokens, meta, blob_path)
    }

    /// Mark a checkpoint unusable (R1-1): a blob that failed to load at request
    /// time is quarantined so identical requests stop hitting it and fall back to
    /// scratch instead of erroring forever. Keyed by the published blob name,
    /// generation, and the token prefix the entry lives at, so it targets exactly
    /// the checkpoint that failed — never a healthy ancestor on a partial match,
    /// never a fresh republish that reused the same deterministic name, and in
    /// O(prefix) rather than a full-index scan. Returns true if an entry was
    /// marked.
    pub fn quarantine_checkpoint(&self, blob_path: &str, generation: u64, prefix: &[u32]) -> bool {
        self.index.quarantine_blob(blob_path, generation, prefix)
    }

    /// Count of quarantined checkpoints (observability/tests).
    pub fn quarantined_count(&self) -> usize {
        self.index.quarantined_count()
    }

    /// Count of published checkpoints (observability).
    pub fn published_count(&self) -> usize {
        self.index.published_count()
    }

    /// Rebuild the index from persisted checkpoints at startup (R1-4: persisted
    /// checkpoints survive a restart). Each blob's header carries its own token
    /// prefix, so the radix index can be reconstructed without a request. Blobs
    /// that fail to load are skipped (quarantine-by-omission): a corrupt blob
    /// must never prevent the daemon from starting or serving other checkpoints.
    pub fn rebuild_from_disk(
        &self,
        persistence: &crate::persistence::Persistence,
    ) -> RebuildReport {
        let mut report = RebuildReport::default();
        let blobs = match persistence.list_blobs() {
            Ok(b) => b,
            Err(e) => {
                report.errors.push(format!("list_blobs: {e}"));
                return report;
            }
        };
        for blob in blobs {
            match persistence.load(&blob) {
                Ok((meta, payload)) => {
                    // A prefix shorter than 2 tokens caches nothing (the adapter
                    // never produces one), and a prefix that disagrees with its
                    // own count cannot be keyed. Skip either rather than publish a
                    // mis-keyed entry the adapter would refuse to serve.
                    if meta.tokens.len() as u64 != meta.token_count || meta.tokens.len() < 2 {
                        report.skipped += 1;
                        continue;
                    }
                    // A multi-token checkpoint with an empty KV payload is
                    // corrupt (a truncated write the adapter rejected at runtime).
                    // The on-disk file can outlive its retirement (a repaired
                    // republish writes a different deterministic name), so
                    // re-indexing it at startup would resurrect the poison.
                    // Skip it, matching the adapter's own load-time check.
                    if payload.is_empty() {
                        report.skipped += 1;
                        continue;
                    }
                    // Key by the persisted token prefix, not the on-disk hash
                    // name, so lookups match real requests.
                    let name = blob
                        .file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_else(|| blob.to_string_lossy().into_owned());
                    let tokens = meta.tokens.clone();
                    self.publish_checkpoint(&tokens, meta, name);
                    report.rebuilt += 1;
                }
                Err(e) => {
                    report.skipped += 1;
                    report.errors.push(format!("{}: {e}", blob.display()));
                }
            }
        }
        report
    }
}

/// Outcome of a startup index rebuild (surfaced in logs / /stats).
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct RebuildReport {
    pub rebuilt: usize,
    pub skipped: usize,
    pub errors: Vec<String>,
}

pub mod test_support {
    use mlxcache_core::contract::ModelFingerprint;

    pub fn fp(id: &str) -> ModelFingerprint {
        ModelFingerprint {
            model_id: id.into(),
            tokenizer_hash: "h".into(),
            kv_dtype: "f16".into(),
            kv_layout_version: 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sidecar::SidecarConfig;

    fn fp(id: &str) -> ModelFingerprint {
        ModelFingerprint {
            model_id: id.into(),
            tokenizer_hash: "h".into(),
            kv_dtype: "f16".into(),
            kv_layout_version: 1,
        }
    }

    fn meta(id: &str, n: u64) -> CheckpointMeta {
        CheckpointMeta {
            fingerprint: fp(id),
            token_count: n,
            tokens: vec![1, 2, 3, 4, 5, 6],
            format_version: 1,
        }
    }

    #[test]
    fn miss_then_hit_roundtrip() {
        let orch = Orchestrator::new();
        let f = fp("m");
        let tokens = vec![1, 2, 3, 4];

        // Miss: nothing published
        let out = orch.route(&tokens, &f);
        assert_eq!(out.decision.verdict, CacheVerdict::Miss);
        assert_eq!(out.prefill_from, 0);

        // Publish after "prefill"
        orch.publish_checkpoint(&tokens, meta("m", 4), "blob-1".into());

        // Hit: full match. KV covers prefix[:-1] = 3 of 4 tokens.
        let out = orch.route(&tokens, &f);
        assert_eq!(out.decision.verdict, CacheVerdict::Hit);
        assert_eq!(out.prefill_from, 3);

        // Partial: extension of the published prefix. The 4-token checkpoint's
        // KV covers 3 tokens, so prefill resumes at 3.
        let ext = vec![1, 2, 3, 4, 5, 6];
        let out = orch.route(&ext, &f);
        assert_eq!(out.decision.verdict, CacheVerdict::Partial);
        assert_eq!(out.prefill_from, 3);
    }

    #[test]
    fn fingerprint_mismatch_never_hits() {
        let orch = Orchestrator::new();
        let tokens = vec![1, 2, 3];
        orch.publish_checkpoint(&tokens, meta("model-a", 3), "blob-a".into());
        let out = orch.route(&tokens, &fp("model-b"));
        assert_eq!(out.decision.verdict, CacheVerdict::Miss);
        assert_eq!(out.prefill_from, 0);
    }

    #[test]
    fn tokenizer_hash_difference_is_a_miss() {
        // R1-2: same model, different tokenizer artifact must not share a
        // checkpoint. A hardcoded tokenizer_hash in the daemon would break this.
        let orch = Orchestrator::new();
        let tokens = vec![1, 2, 3];
        let mut a = fp("m");
        a.tokenizer_hash = "tok-a".into();
        let mut b = fp("m");
        b.tokenizer_hash = "tok-b".into();

        let mut meta_a = meta("m", 3);
        meta_a.fingerprint = a.clone();
        orch.publish_checkpoint(&tokens, meta_a, "blob-a".into());
        assert_eq!(orch.route(&tokens, &a).decision.verdict, CacheVerdict::Hit);
        let out = orch.route(&tokens, &b);
        assert_eq!(out.decision.verdict, CacheVerdict::Miss);
        assert!(
            out.blob.is_none(),
            "must not reuse the other tokenizer's blob"
        );
    }

    #[test]
    fn sidecar_config_accessible() {
        // Wiring smoke: sidecar config constructs and is distinct from the
        // native path. The sidecar is never the hot-path default (R4).
        let cfg = SidecarConfig::new("http://127.0.0.1:8421".into(), "m".into());
        assert_eq!(cfg.model_id, "m");
    }
}
