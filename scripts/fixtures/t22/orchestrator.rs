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
        let lookup = self.index.lookup(tokens, Some(request_fingerprint));
        let (matched_tokens, matched_fp, blob_path, blob_generation, key) = match &lookup {
            Some((entry, n)) => (
                Some(*n),
                Some(&entry.meta.fingerprint),
                Some(entry.blob_path.clone()),
                Some(entry.generation),
                // T22: under end-anchored divergence the request's tokens do
                // NOT spell the entry's key. Quarantine/eviction key by the
                // entry's true key, never by the request prefix.
                Some(entry.key.clone()),
            ),
            None => (None, None, None, None, None),
        };
        let verdict = classify(
            matched_tokens,
            tokens.len(),
            matched_fp,
            request_fingerprint,
        );
        // A fingerprint mismatch classifies as Miss and must not reuse the blob.
        // T22: under end-anchored divergence the request's tokens do NOT spell
        // the entry's key, so the quarantine/eviction key MUST come from the
        // entry itself (`key` from the lookup), never from the request prefix.
        let blob = match verdict {
            CacheVerdict::Miss => None,
            _ => blob_path.zip(blob_generation).zip(key).map(|((p, g), k)| {
                (
                    p, g,
                    // Exact-depth matches: the key equals the request prefix,
                    // byte-identical to the historical behavior.
                    k,
                )
            }),
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
    ///
    /// Must not be called before the startup rebuild has seeded the floor:
    /// main.rs runs `rebuild_from_disk` before binding the listener, so no
    /// request can race the seed. The assert catches a future caller that
    /// reserves while a failed scan still has the floor unknown.
    pub fn reserve_generation(&self) -> u64 {
        debug_assert!(
            self.can_publish(),
            "reserve_generation before the generation floor is known"
        );
        self.index.reserve_generation()
    }

    /// Whether publishing is currently allowed. False after a failed startup
    /// scan (the generation floor is unknown). Callers must check this BEFORE
    /// writing a blob so a refusal does not leak an unindexed file.
    pub fn can_publish(&self) -> bool {
        !self
            .recovery_failed
            .load(std::sync::atomic::Ordering::Acquire)
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

    /// Evict cold checkpoints until the store holds at most `max_entries`
    /// published blobs (T-eviction: the ds4 anchor policy, now actually wired).
    ///
    /// Scoring (policy::eviction_score): anchors score 0 and are never evicted;
    /// an entry is an anchor when a longer published checkpoint extends it (a
    /// live chain's base) OR it was served within `anchor_window` (recently
    /// used). The remainder evict biggest-first (`score = token_count`: most
    /// bytes freed per unlink — the design doc's intent; "coldest-first" was a
    /// stale description of the same code).
    ///
    /// Safety: each removal re-verifies blob path + generation at the node, so
    /// an entry republished between the snapshot and the removal is untouched;
    /// the blob file is unlinked only after the index removal wins, and a
    /// failed unlink is logged (an orphan file is re-scanned and re-published
    /// by the next startup rebuild — never resurrected mid-run, since the index
    /// entry is already gone).
    ///
    /// Returns the number of entries evicted.
    pub fn evict_cold(
        &self,
        persistence: &crate::persistence::Persistence,
        max_entries: usize,
        anchor_window: std::time::Duration,
    ) -> usize {
        let candidates = self.index.eviction_candidates();
        if candidates.len() <= max_entries {
            return 0;
        }
        let now = std::time::Instant::now();
        let mut scored: Vec<(u64, mlxcache_core::index::IndexCandidate)> = candidates
            .into_iter()
            .map(|c| {
                let anchored = c.is_anchor || now.duration_since(c.last_used) <= anchor_window;
                let score = mlxcache_core::policy::eviction_score(
                    &mlxcache_core::policy::EvictionCandidate {
                        token_count: c.token_count as u64,
                        last_used: c.last_used,
                        is_anchor: anchored,
                    },
                );
                (score, c)
            })
            .collect();
        // Policy contract: "Higher score = evict first." Descending order puts
        // the biggest non-anchor blobs at the front (most bytes freed per
        // unlink); anchors score 0 and sink to the back, never reached.
        scored.sort_by_key(|(score, _)| std::cmp::Reverse(*score));
        let excess = scored.len().saturating_sub(max_entries);
        let mut evicted = 0usize;
        for (score, cand) in &scored {
            if evicted >= excess {
                break;
            }
            if *score == 0 {
                continue; // anchor: never evict
            }
            // The candidate carries the blob NAME (relative): removal is by
            // name under the blob dir, matching quarantine's convention.
            if self
                .index
                .remove_published(&cand.tokens, &cand.blob_path, cand.generation)
            {
                // Index removal won the verify race: now reclaim the file.
                if let Err(e) = persistence.remove(&cand.blob_path) {
                    tracing::warn!(
                        blob = %cand.blob_path,
                        error = %e,
                        "evicted from index but could not delete blob file (startup rescan will reconcile)"
                    );
                }
                evicted += 1;
            }
        }
        if evicted > 0 {
            tracing::info!(
                evicted,
                target = max_entries,
                "eviction: cold checkpoints reaped"
            );
        }
        evicted
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
        // Publish the winners, 