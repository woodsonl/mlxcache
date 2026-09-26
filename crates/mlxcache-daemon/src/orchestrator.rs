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
    /// Token count to prefill as delta (0 for full miss = everything).
    pub prefill_from: usize,
    /// Published blob to resume from (None on a full miss). The daemon hands
    /// this to the adapter so the engine loads cached KV instead of re-prefilling.
    pub blob_path: Option<String>,
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
        let (matched_tokens, matched_fp, blob_path) = match &lookup {
            Some((entry, n)) => (
                Some(*n),
                Some(&entry.meta.fingerprint),
                Some(entry.blob_path.clone()),
            ),
            None => (None, None, None),
        };
        let verdict = classify(
            matched_tokens,
            tokens.len(),
            matched_fp,
            request_fingerprint,
        );
        // A fingerprint mismatch classifies as Miss and must not reuse the blob.
        let blob_path = match verdict {
            CacheVerdict::Miss => None,
            _ => blob_path,
        };
        let prefill_from = match verdict {
            CacheVerdict::Hit => tokens.len(), // nothing to prefill
            CacheVerdict::Partial => matched_tokens.unwrap_or(0),
            CacheVerdict::Miss => 0,
        };
        RouteOutcome {
            decision: PolicyDecision {
                verdict,
                matched_tokens: matched_tokens.unwrap_or(0),
                request_tokens: tokens.len(),
            },
            prefill_from,
            blob_path,
        }
    }

    /// Register a completed prefill: capture metadata and publish atomically.
    /// The blob rename must already be done (R1-3 ordering).
    pub fn publish_checkpoint(&self, tokens: &[u32], meta: CheckpointMeta, blob_path: String) {
        self.index.publish(tokens, meta, blob_path);
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
                Ok((meta, _payload)) => {
                    if meta.tokens.len() as u64 != meta.token_count || meta.tokens.is_empty() {
                        // A blob with no recoverable prefix cannot be indexed;
                        // skip it rather than publish a mis-keyed entry.
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

        // Hit: full match
        let out = orch.route(&tokens, &f);
        assert_eq!(out.decision.verdict, CacheVerdict::Hit);
        assert_eq!(out.prefill_from, 4);

        // Partial: extension of the published prefix
        let ext = vec![1, 2, 3, 4, 5, 6];
        let out = orch.route(&ext, &f);
        assert_eq!(out.decision.verdict, CacheVerdict::Partial);
        assert_eq!(out.prefill_from, 4);
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
    fn sidecar_config_accessible() {
        // Wiring smoke: sidecar config constructs and is distinct from the
        // native path. The sidecar is never the hot-path default (R4).
        let cfg = SidecarConfig::new("http://127.0.0.1:8421".into(), "m".into());
        assert_eq!(cfg.model_id, "m");
    }
}
