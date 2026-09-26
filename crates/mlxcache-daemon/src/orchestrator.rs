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
    /// Set when the startup scan FAILED, so the floor is unknown. Publishing is
    /// refused while set: an unseen higher-generation file could be on disk, and
    /// publishing below it would let the next restart discard this checkpoint.
    /// A daemon that never scanned (fresh, empty dir) may publish with floor 0.
    recovery_failed: std::sync::atomic::AtomicBool,
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
            recovery_failed: std::sync::atomic::AtomicBool::new(false),
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
    /// The blob rename must already be done (R1-3 ordering). `generation` is
    /// reserved by the caller before writing the blob so the on-disk name can be
    /// generation-specific. Returns false if the prefix was too short to publish.
    pub fn publish_checkpoint(
        &self,
        persistence: &crate::persistence::Persistence,
        tokens: &[u32],
        meta: CheckpointMeta,
        blob_path: String,
        generation: u64,
    ) -> bool {
        // Refuse to publish after a FAILED scan: the generation floor is unknown,
        // so an unseen higher-generation file could be on disk and the next
        // restart would discard a checkpoint published below it. Serving is
        // unaffected (the caller just does not cache).
        if self
            .recovery_failed
            .load(std::sync::atomic::Ordering::Acquire)
        {
            tracing::warn!("refusing to publish: startup recovery failed, floor unknown");
            return false;
        }
        self.index
            .publish(tokens, meta, blob_path, generation, |old| {
                if let Err(e) = persistence.remove(old) {
                    tracing::warn!(blob = %old, error = %e, "could not reclaim superseded blob file");
                }
            })
    }

    /// Reserve a publication generation for the next checkpoint write.
    pub fn reserve_generation(&self) -> u64 {
        self.index.reserve_generation()
    }

    /// Mark a checkpoint unusable (R1-1): a blob that failed to load at request
    /// time is quarantined so identical requests stop hitting it and fall back to
    /// scratch instead of erroring forever. Keyed by the published blob name,
    /// generation, and the token prefix the entry lives at, so it targets exactly
    /// the checkpoint that failed — never a healthy ancestor on a partial match,
    /// never a fresh republish that reused the same deterministic name, and in
    /// O(prefix) rather than a full-index scan. The blob file is deleted under the
    /// index write lock so a concurrent republish cannot lose its file. Returns
    /// true if an entry was marked.
    pub fn quarantine_checkpoint(
        &self,
        persistence: &crate::persistence::Persistence,
        blob_path: &str,
        generation: u64,
        prefix: &[u32],
    ) -> bool {
        self.index
            .quarantine_blob(blob_path, generation, prefix, || {
                if let Err(e) = persistence.remove(blob_path) {
                    tracing::warn!(blob = %blob_path, error = %e, "quarantined but could not delete blob file");
                }
            })
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
                // Mark the floor unknown: publishing stays disabled so a new
                // checkpoint cannot be assigned a generation below an on-disk one
                // this failed scan could not see.
                self.recovery_failed
                    .store(true, std::sync::atomic::Ordering::Release);
                return report;
            }
        };
        // The scan succeeded: the filename floor below is now authoritative.
        self.recovery_failed
            .store(false, std::sync::atomic::Ordering::Release);
        // Two passes. Directory order is arbitrary, and files now carry immutable
        // per-generation names, so processing in order and reclaiming as we go
        // would let an older corrupt generation delete the newer repair that
        // replaced it. Instead: load every valid candidate, keep only the highest
        // generation per token prefix (the latest publication), publish the
        // winners, then delete the superseded files. Recovery never reclaims a
        // file it is about to publish.
        struct Candidate {
            name: String,
            generation: u64,
            // Lowercased hex string prefix per file: {hash:032x}-{gen:016x}.ckpt.
            tokens: Vec<u32>,
            meta: mlxcache_core::contract::CheckpointMeta,
        }
        let mut best: std::collections::HashMap<Vec<u32>, Candidate> =
            std::collections::HashMap::new();
        let mut stale: Vec<String> = Vec::new();
        // Seed the generation floor from EVERY filename, before any I/O. A file
        // that fails to load now (transient read error) is still a publication
        // that could load later; if we only counted files that load, a restart
        // would hand a replacement a LOWER generation, and the recovered
        // higher-generation file would then delete it.
        let max_persisted = blobs
            .iter()
            .filter_map(|b| b.file_name())
            .filter_map(|n| parse_generation(&n.to_string_lossy()))
            .max()
            .unwrap_or(0);
        for blob in blobs {
            let name = blob
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| blob.to_string_lossy().into_owned());
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
                    // Skip it, matching the adapter's own load-time check.
                    if payload.is_empty() {
                        report.skipped += 1;
                        continue;
                    }
                    let generation = parse_generation(&name).unwrap_or(0);
                    // Group by the index's ACTUAL key (the token prefix only): it
                    // stores one entry per node, so two publications at the same
                    // prefix (e.g. a tokenizer-hash change) must be reconciled
                    // here too, or both survive selection and race for the node.
                    let key = meta.tokens.clone();
                    let cand = Candidate {
                        name,
                        generation,
                        tokens: meta.tokens.clone(),
                        meta,
                    };
                    match best.entry(key) {
                        std::collections::hash_map::Entry::Occupied(mut e) => {
                            if cand.generation > e.get().generation {
                                stale.push(e.get().name.clone());
                                e.insert(cand);
                            } else {
                                stale.push(cand.name);
                            }
                        }
                        std::collections::hash_map::Entry::Vacant(v) => {
                            v.insert(cand);
                        }
                    }
                }
                Err(e) => {
                    report.skipped += 1;
                    report.errors.push(format!("{}: {e}", blob.display()));
                }
            }
        }
        // Publish the winners, then remove every superseded file. A no-op reclaim
        // callback: the deletion is explicit and post-publish. Seed the counter
        // above every persisted generation first so the generations we reserve
        // here (and in later requests) sort above what is on disk.
        self.index.seed_generation(max_persisted);
        for (_, cand) in best {
            let generation = self.reserve_generation();
            self.index
                .publish(&cand.tokens, cand.meta, cand.name, generation, |_| {});
            report.rebuilt += 1;
        }
        for name in stale {
            if let Err(e) = persistence.remove(&name) {
                report
                    .errors
                    .push(format!("{name}: could not remove stale generation: {e}"));
            }
        }
        report
    }
}

/// Parse the generation embedded in an immutable blob name
/// (`{hash:032x}-{gen:016x}.ckpt`). None for a legacy/foreign name.
fn parse_generation(name: &str) -> Option<u64> {
    let stem = name.strip_suffix(".ckpt")?;
    let (hash, gen) = stem.rsplit_once('-')?;
    if hash.len() != 32 || gen.len() != 16 {
        return None;
    }
    u64::from_str_radix(gen, 16).ok()
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

    /// A Persistence over a leaked tempdir, for publish tests that do not
    /// exercise file reclamation. The deletion callback is a no-op on absent
    /// files, so the missing files are harmless.
    fn persist() -> crate::persistence::Persistence {
        crate::persistence::Persistence::new(tempfile::tempdir().unwrap().keep()).unwrap()
    }

    /// Publish in a test (publishing is allowed until a scan FAILS).
    fn publish(orch: &Orchestrator, tokens: &[u32], meta: CheckpointMeta, name: &str, gen: u64) {
        orch.publish_checkpoint(&persist(), tokens, meta, name.into(), gen);
    }

    #[test]
    fn publish_refused_after_a_failed_scan() {
        // If the startup scan fails, the generation floor is unknown, so
        // publishing must be refused (Codex pass 11). A nonexistent dir makes
        // list_blobs fail.
        let orch = Orchestrator::new();
        // Build a Persistence on a real dir, then replace the dir with a FILE so
        // read_dir fails and list_blobs errors.
        let parent = tempfile::tempdir().unwrap();
        let dir = parent.path().join("blobs");
        let missing = crate::persistence::Persistence::new(&dir).unwrap();
        std::fs::remove_dir(&dir).unwrap();
        std::fs::write(&dir, b"x").unwrap();
        let report = orch.rebuild_from_disk(&missing);
        assert!(
            !report.errors.is_empty(),
            "the scan must report the failure"
        );
        assert!(
            !orch.publish_checkpoint(&missing, &[1, 2, 3], meta("m", 3), "blob".into(), 1),
            "publishing must be refused while the floor is unknown"
        );
        assert_eq!(orch.published_count(), 0);
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
        publish(&orch, &tokens, meta("m", 4), "blob-1", 1);

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
        publish(&orch, &tokens, meta("model-a", 3), "blob-a", 1);
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
        publish(&orch, &tokens, meta_a, "blob-a", 1);
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
