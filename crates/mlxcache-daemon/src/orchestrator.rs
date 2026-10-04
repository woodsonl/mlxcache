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
        // Borrow, don't clone: the full-prefix key (~80 KB at 20K tokens) and
        // the blob path are needed only when the verdict survives classify —
        // a fingerprint mismatch classifies as Miss and discards them, so
        // cloning up front paid a second full-prefix copy per request on the
        // hottest path (performance review 2026-10-04).
        let matched = lookup
            .as_ref()
            .map(|(entry, n)| (&entry.meta.fingerprint, *n, entry));
        let (matched_tokens, matched_fp) = match &matched {
            Some((fp, n, _)) => (Some(*n), Some(*fp)),
            None => (None, None),
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
            _ => matched.map(|(_, _, entry)| {
                (
                    entry.blob_path.clone(),
                    entry.generation,
                    // Exact-depth matches: the key equals the request prefix,
                    // byte-identical to the historical behavior.
                    entry.key.clone(),
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
        max_bytes: u64,
        anchor_window: std::time::Duration,
    ) -> usize {
        // Reap tombstones FIRST and unconditionally: the published-cap early
        // return below fires in the common under-cap steady state, and burial
        // behind it made the tombstone cap dead code exactly where unbounded
        // tombstone growth was possible (red-team 2026-10-04). usize::MAX
        // (the env-layer translation of "no entry cap") would make the
        // tombstone cap unbounded too — the same resurrection — so it maps
        // to 0, the aggressive full reap: tombstones are diagnostic records,
        // never serveable, and the default config must still bound them.
        let tombstone_cap = if max_entries == usize::MAX {
            0
        } else {
            max_entries
        };
        let reaped = self.reap_quarantined(tombstone_cap);
        let candidates = self.index.eviction_candidates();
        // File sizes for the byte budget (D-eviction): a missing file scores
        // 0 bytes — remove_published + the unlink reconcile it regardless.
        let size_of = |name: &str| -> u64 {
            std::fs::metadata(persistence.blob_dir.join(name))
                .map(|m| m.len())
                .unwrap_or(0)
        };
        let total_bytes: u64 = candidates.iter().map(|c| size_of(&c.blob_path)).sum();
        // Entry-cap semantics at the PASS level: `max_entries` is a hard cap,
        // 0 = allow zero published entries (evict everything evictable). The
        // "0 disables the knob" convention lives at the env layer (main.rs
        // translates 0 → usize::MAX = no entry cap), so pass-level 0 stays
        // the aggressive sweep the tests rely on.
        let under_entries = candidates.len() <= max_entries;
        let under_bytes = max_bytes == 0 || total_bytes <= max_bytes;
        if under_entries && under_bytes {
            return reaped;
        }
        let now = std::time::Instant::now();
        let mut scored: Vec<(u64, u64, mlxcache_core::index::IndexCandidate)> = candidates
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
                (score, size_of(&c.blob_path), c)
            })
            .collect();
        // Policy contract: "Higher score = evict first." Descending order puts
        // the biggest non-anchor blobs at the front (most bytes freed per
        // unlink); anchors score 0 and sink to the back, never reached.
        scored.sort_by_key(|(score, _, _)| std::cmp::Reverse(*score));
        let total_entries = scored.len();
        let mut evicted = 0usize;
        let mut remaining_bytes = total_bytes;
        for (score, size, cand) in &scored {
            let over_entries = total_entries - evicted > max_entries;
            let over_bytes = max_bytes > 0 && remaining_bytes > max_bytes;
            if !over_entries && !over_bytes {
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
                remaining_bytes = remaining_bytes.saturating_sub(*size);
            }
        }
        if max_bytes > 0 && remaining_bytes > max_bytes {
            // All that is left is anchors (or races): the budget is a
            // best-effort cap that never breaks the anchor contract.
            tracing::warn!(
                remaining_bytes,
                budget = max_bytes,
                "eviction byte budget unreachable: published anchors hold the store above it"
            );
        }
        if evicted > 0 {
            tracing::info!(
                evicted,
                target_entries = max_entries,
                budget_bytes = max_bytes,
                "eviction: cold checkpoints reaped"
            );
        }
        if total_entries - evicted > max_entries {
            // Silent-failure guard (red-team 2026-10-04): every excess
            // candidate was anchor/recency-protected (score 0) or lost its
            // removal race — the store stays above cap with no other signal.
            tracing::warn!(
                excess = total_entries - evicted,
                evicted,
                "eviction could not reach the cap: all excess candidates are anchor- or recency-protected; the store stays above cap until anchors age out"
            );
        }
        evicted + reaped
    }

    /// Cap quarantined tombstones at `max_entries` (QA ISSUE-002 follow-up):
    /// they are diagnostic records, not serveable assets, so the same cap that
    /// bounds published entries bounds them. Coldest stones go first; the
    /// exact-identity removal can never touch a published entry or a republish
    /// that reused a deterministic name. Blob files are already gone (quarantine
    /// unlinks them), so there is nothing on disk to reclaim.
    fn reap_quarantined(&self, max_entries: usize) -> usize {
        let candidates = self.index.quarantine_candidates(max_entries);
        let mut reaped = 0usize;
        for (tokens, blob_path, generation) in &candidates {
            if self
                .index
                .remove_quarantined(tokens, blob_path, *generation)
            {
                reaped += 1;
            }
        }
        if reaped > 0 {
            tracing::info!(
                reaped,
                cap = max_entries,
                "eviction: cold quarantine tombstones reaped"
            );
        }
        reaped
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
            ..Default::default()
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
            ..Default::default()
        }
    }

    fn meta(id: &str, n: u64) -> CheckpointMeta {
        CheckpointMeta {
            fingerprint: fp(id),
            token_count: n,
            tokens: vec![1, 2, 3, 4, 5, 6],
            format_version: 1,
            payload_sha256: None,
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
    fn tombstones_reap_even_when_published_are_under_cap() {
        // Red-team 2026-10-04: the tombstone reaper used to run only AFTER
        // evict_cold's published-cap early return, so a store under the
        // published cap — the normal steady state — never reaped, and the
        // ISSUE-002 tombstone cap was dead code exactly where unbounded
        // tombstone growth was possible.
        let orch = Orchestrator::new();
        let p = persist();
        publish(&orch, &[1, 2, 3, 4, 5, 6], meta("m", 6), "pub-a", 1);
        // Distinct keys: one trie node holds ONE entry, so same-key publishes
        // would replace each other's tombstones.
        for (tokens, name, gen) in [
            (&[20u32, 21, 22, 23][..], "q-a", 2),
            (&[30u32, 31, 32, 33][..], "q-b", 3),
            (&[40u32, 41, 42, 43][..], "q-c", 4),
        ] {
            publish(&orch, tokens, meta("m", 6), name, gen);
            assert!(orch.quarantine_checkpoint(&p, name, gen, tokens));
        }
        assert_eq!(orch.published_count(), 1);
        assert_eq!(orch.quarantined_count(), 3);
        // Published count (1) is far under the cap (2): the old early return
        // fired here and reaped nothing.
        let _ = orch.evict_cold(&p, 2, 0, std::time::Duration::from_secs(900));
        assert_eq!(
            orch.quarantined_count(),
            2,
            "tombstones must reap to the cap even with published entries under cap"
        );
        assert_eq!(orch.published_count(), 1, "reaping never touches published");
    }

    #[test]
    fn tombstones_reap_under_the_default_no_entry_cap() {
        // The default reaper config translates "no entry cap" (knob 0) to
        // usize::MAX — which must NOT flow into the tombstone cap (an
        // unbounded cap is the dead-reap resurrection). Stones must fully
        // reap on the default shape: byte budget only, entry cap MAX.
        let orch = Orchestrator::new();
        let p = persist();
        for (tokens, name, gen) in [
            (&[20u32, 21, 22, 23][..], "q-a", 2),
            (&[30u32, 31, 32, 33][..], "q-b", 3),
        ] {
            publish(&orch, tokens, meta("m", 6), name, gen);
            assert!(orch.quarantine_checkpoint(&p, name, gen, tokens));
        }
        assert_eq!(orch.quarantined_count(), 2);
        let evicted = orch.evict_cold(&p, usize::MAX, 0, std::time::Duration::ZERO);
        assert_eq!(evicted, 2, "entry-cap MAX must reap tombstones fully");
        assert_eq!(orch.quarantined_count(), 0);
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
            !orch.can_publish(),
            "publishing must be blocked after a failed scan"
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

    #[test]
    fn evict_cold_reaps_blobs_up_to_cap_and_unlinks_files() {
        // The reaper must evict down to the cap, unlink the evicted blob files
        // (a blob left on disk resurrects on the next restart rebuild), and
        // leave under-cap stores untouched. Window 0 disables the recency
        // anchor; all four prefixes are standalone, so structural anchors do
        // not apply either. (Recency anchoring is asserted separately below.)
        let orch = Orchestrator::new();
        let dir = tempfile::tempdir().unwrap();
        let persistence = crate::persistence::Persistence::new(dir.path().join("blobs")).unwrap();

        let mut names = Vec::new();
        for i in 0..4u64 {
            let name = format!("blob-{i}");
            // A real file per blob, so unlinking is observable.
            std::fs::write(persistence.blob_dir.join(&name), b"payload").unwrap();
            publish(&orch, &[i as u32, 2, 3], meta("m", 3), &name, i + 1);
            names.push(name);
        }
        assert_eq!(orch.published_count(), 4);

        // Under the cap: no-op.
        assert_eq!(
            orch.evict_cold(&persistence, 10, 0, std::time::Duration::ZERO),
            0
        );
        assert_eq!(orch.published_count(), 4);

        // Cap at 1: the reaper must remove 3. All scores tie (token_count 3),
        // so which one survives is walk-order dependent — but index removal
        // and file unlink must agree for every entry.
        let evicted = orch.evict_cold(&persistence, 1, 0, std::time::Duration::ZERO);
        assert_eq!(evicted, 3);
        assert_eq!(orch.published_count(), 1);
        for (i, name) in names.iter().enumerate() {
            let gone = !persistence.blob_dir.join(name).exists();
            let still_published =
                orch.route(&[i as u32, 2, 3], &fp("m")).decision.verdict == CacheVerdict::Hit;
            assert_eq!(
                gone, !still_published,
                "index removal and file unlink must agree for {name}"
            );
        }
    }

    #[test]
    fn evict_cold_enforces_the_byte_budget_biggest_first() {
        // The launch-config decision (design doc): the store is bounded by
        // BYTES, not entries — a long-context cache grows by GB-scale blobs
        // per publish. Entry cap off (usize::MAX); only the byte budget
        // drives the pass. Scores tie (3 tokens each), so the invariant
        // asserted is the budget itself, not which name went.
        let orch = Orchestrator::new();
        let persistence = persist();
        let names = ["small", "mid", "big"];
        for ((i, name), size) in names.iter().enumerate().zip([1024, 2048, 4096]) {
            std::fs::write(persistence.blob_dir.join(name), vec![0u8; size]).unwrap();
            let t = (i + 1) as u32;
            publish(&orch, &[t, t, t], meta("m", 3), name, (i + 1) as u64);
        }
        assert_eq!(orch.published_count(), 3);

        // 7 KiB published, 4 KiB budget: at least the biggest must go.
        let evicted = orch.evict_cold(&persistence, usize::MAX, 4096, std::time::Duration::ZERO);
        assert!(evicted >= 1, "over-budget store must evict");
        let remaining: u64 = names
            .iter()
            .map(|n| {
                std::fs::metadata(persistence.blob_dir.join(n))
                    .map(|m| m.len())
                    .unwrap_or(0)
            })
            .sum();
        assert!(
            remaining <= 4096,
            "remaining {remaining} bytes must fit the 4096 budget"
        );
        assert_eq!(
            orch.published_count(),
            3 - evicted,
            "index and disk must agree"
        );
    }

    #[test]
    fn evict_cold_byte_budget_never_breaks_anchors() {
        // A budget smaller than the anchor's own size is unreachable by
        // design: anchors are never evicted, the pass warns instead.
        let orch = Orchestrator::new();
        let persistence = persist();
        std::fs::write(persistence.blob_dir.join("anchor"), vec![0u8; 8192]).unwrap();
        publish(&orch, &[1, 2], meta("m", 2), "anchor", 1);
        let evicted = orch.evict_cold(
            &persistence,
            usize::MAX,
            1024,
            std::time::Duration::from_secs(900),
        );
        assert_eq!(evicted, 0, "an anchor must never be evicted for bytes");
        assert!(persistence.blob_dir.join("anchor").exists());
        assert_eq!(orch.published_count(), 1);
    }

    #[test]
    fn evict_cold_under_byte_budget_is_a_noop() {
        let orch = Orchestrator::new();
        let persistence = persist();
        std::fs::write(persistence.blob_dir.join("a"), vec![0u8; 100]).unwrap();
        publish(&orch, &[1, 2], meta("m", 2), "a", 1);
        let evicted = orch.evict_cold(&persistence, usize::MAX, 4096, std::time::Duration::ZERO);
        assert_eq!(evicted, 0);
        assert_eq!(orch.published_count(), 1);
    }

    #[test]
    fn evict_cold_never_reaps_chain_bases_and_highest_score_first() {
        // ds4 lesson: the base a live chain was built on must survive, even
        // when it is the "coldest" entry. Policy contract: higher score evicts
        // first (score = token_count), so with window 0 the 4-token chain
        // entry goes before the 3-token standalone, and the 2-token base
        // (anchor) is never reached.
        let orch = Orchestrator::new();
        let persistence = persist();

        // [1,2] anchors the chain [1,2,3,4]; [7,7,7] is standalone.
        publish(&orch, &[1, 2], meta("m", 2), "base", 1);
        publish(&orch, &[1, 2, 3, 4], meta("m", 4), "chain", 2);
        publish(&orch, &[7, 7, 7], meta("m", 3), "cold", 3);

        // Cap 2 (excess 1): the highest-score non-anchor is the chain entry.
        let evicted = orch.evict_cold(&persistence, 2, 0, std::time::Duration::ZERO);
        assert_eq!(evicted, 1);
        // The 4-token entry is gone: [1,2,3,4] now only PARTIALLY matches the
        // surviving base at depth 2.
        let out = orch.route(&[1, 2, 3, 4], &fp("m"));
        assert_eq!(
            out.decision.verdict,
            CacheVerdict::Partial,
            "the 4-token chain entry has the highest score and evicts first"
        );
        assert_eq!(out.decision.matched_tokens, 2);
        assert_eq!(
            orch.route(&[1, 2], &fp("m")).decision.verdict,
            CacheVerdict::Hit
        );
        assert_eq!(
            orch.route(&[7, 7, 7], &fp("m")).decision.verdict,
            CacheVerdict::Hit
        );

        // Cap 1 (excess 1): the standalone goes next; the base is untouched
        // (it is only ever reached after higher-scored non-anchors).
        let evicted = orch.evict_cold(&persistence, 1, 0, std::time::Duration::ZERO);
        assert_eq!(evicted, 1);
        assert_eq!(
            orch.route(&[7, 7, 7], &fp("m")).decision.verdict,
            CacheVerdict::Miss
        );

        // The base is now alone (its chain was reaped, so the structural
        // anchor has lapsed) — but it was just SERVED: within the recency
        // window it is an anchor and must survive even a cap of 0.
        orch.route(&[1, 2], &fp("m")); // touch via lookup
        let evicted = orch.evict_cold(&persistence, 0, 0, std::time::Duration::from_secs(900));
        assert_eq!(evicted, 0, "a recently served entry is never evicted");
        assert_eq!(
            orch.route(&[1, 2], &fp("m")).decision.verdict,
            CacheVerdict::Hit
        );

        // Outside the window (0) with no published extension, the same entry
        // is legitimately evictable: structural anchoring follows the live
        // chain, recency follows the window. A cold, unextended checkpoint is
        // dead weight no matter how it got there.
        let evicted = orch.evict_cold(&persistence, 0, 0, std::time::Duration::ZERO);
        assert_eq!(evicted, 1);
        assert_eq!(
            orch.route(&[1, 2], &fp("m")).decision.verdict,
            CacheVerdict::Miss
        );
    }

    #[test]
    fn evict_cold_recency_window_anchors_recently_used_entries() {
        // Within the anchor window, everything is anchored and nothing is
        // reaped (the live-working-set guarantee). Past the window (window 0
        // here), the same store evicts normally.
        let orch = Orchestrator::new();
        let persistence = persist();
        publish(&orch, &[7, 7, 7], meta("m", 3), "cold", 1); // standalone

        let evicted = orch.evict_cold(&persistence, 0, 0, std::time::Duration::from_secs(900));
        assert_eq!(evicted, 0, "a just-published entry is an anchor (recency)");
        assert_eq!(
            orch.route(&[7, 7, 7], &fp("m")).decision.verdict,
            CacheVerdict::Hit
        );
        // Window 0: not recently used, no descendants — evictable.
        let evicted = orch.evict_cold(&persistence, 0, 0, std::time::Duration::ZERO);
        assert_eq!(evicted, 1);
    }

    #[test]
    fn evict_cold_is_a_noop_when_store_is_empty() {
        let orch = Orchestrator::new();
        let persistence = persist();
        assert_eq!(
            orch.evict_cold(&persistence, 0, 0, std::time::Duration::ZERO),
            0
        );
    }
}
