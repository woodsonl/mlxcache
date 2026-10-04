//! Native prefix index: radix tree over token IDs.
//!
//! Longest-prefix match in O(prefix length). No SQLite on the hot path (D2-reopened).
//! Index rows are only visible after the checkpoint blob's atomic publish (R1-3).

use crate::contract::{CheckpointMeta, CheckpointState};
use std::collections::HashMap;
use std::sync::RwLock;

/// A node in the radix trie. Children are keyed by token-id "chunks" to bound
/// per-node fanout; for v1 a flat child map per token is correct and simple.
#[derive(Debug, Default)]
struct Node {
    children: HashMap<u32, Node>,
    /// Present only on nodes that terminate a published checkpoint.
    entry: Option<IndexEntry>,
}

#[derive(Debug)]
pub struct IndexEntry {
    pub meta: CheckpointMeta,
    /// Path of the published blob (post-rename, never a temp path).
    pub blob_path: String,
    /// Monotonic publication generation for this prefix. A blob name is
    /// deterministic (fingerprint+tokens), so a repaired checkpoint reuses the
    /// name; the generation lets a late failure retire only the exact
    /// publication it used, never a fresh republish that landed meanwhile.
    pub generation: u64,
    pub state: CheckpointState,
    /// Last time a lookup matched this entry — monotonic milliseconds from the
    /// process-start instant (see [`last_used_instant`]). Drives eviction
    /// anchoring (policy::eviction_score): a recently served checkpoint is the
    /// base of a live chain and must not be reaped. Atomic so `lookup` can
    /// touch it through the read lock.
    last_used_millis: std::sync::atomic::AtomicU64,
}

impl Clone for IndexEntry {
    fn clone(&self) -> Self {
        use std::sync::atomic::Ordering::Relaxed;
        Self {
            meta: self.meta.clone(),
            blob_path: self.blob_path.clone(),
            generation: self.generation,
            state: self.state,
            last_used_millis: std::sync::atomic::AtomicU64::new(
                self.last_used_millis.load(Relaxed),
            ),
        }
    }
}

/// Process-start instant backing [`IndexEntry::last_used_instant`]. Monotonic
/// clock (Instant), so eviction decisions are immune to wall-clock jumps.
static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();

fn now_millis() -> u64 {
    let start = START.get_or_init(std::time::Instant::now);
    start.elapsed().as_millis() as u64
}

fn last_used_instant(millis: u64) -> std::time::Instant {
    let start = START.get_or_init(std::time::Instant::now);
    *start + std::time::Duration::from_millis(millis)
}

#[derive(Debug, thiserror::Error)]
pub enum IndexError {
    #[error("index corrupt: {0}")]
    Corrupt(String),
}

/// One Published entry, shaped for the eviction policy (policy::eviction_score).
#[derive(Debug, Clone)]
pub struct IndexCandidate {
    /// The exact token prefix the entry lives at — the removal key.
    pub tokens: Vec<u32>,
    pub token_count: usize,
    pub last_used: std::time::Instant,
    /// True when a longer published checkpoint extends this prefix (a live
    /// chain's base). Feeds `EvictionCandidate::is_anchor`.
    pub is_anchor: bool,
    /// The published blob file (post-rename name) and its generation: eviction
    /// must remove the exact publication it snapshotted, never a republish.
    pub blob_path: String,
    pub generation: u64,
}

#[derive(Debug, Default)]
pub struct PrefixIndex {
    root: RwLock<Node>,
    /// Source of `IndexEntry::generation`. Bumped on every publish.
    next_generation: std::sync::atomic::AtomicU64,
}

impl PrefixIndex {
    pub fn new() -> Self {
        Self::default()
    }

    /// Recover a poisoned lock instead of panicking: the index is a cache, and
    /// a poisoned lock must not take down every later request. A recovered read
    /// may observe a torn write only if a panic interrupted one, which the
    /// short, allocation-only critical sections here make unlikely.
    fn read_lock(&self) -> std::sync::RwLockReadGuard<'_, Node> {
        self.root.read().unwrap_or_else(|e| e.into_inner())
    }

    fn write_lock(&self) -> std::sync::RwLockWriteGuard<'_, Node> {
        self.root.write().unwrap_or_else(|e| e.into_inner())
    }

    /// Longest published prefix match for `tokens`. Returns the entry and the
    /// number of tokens matched.
    pub fn lookup(&self, tokens: &[u32]) -> Option<(IndexEntry, usize)> {
        use std::sync::atomic::Ordering::Relaxed;
        let root = self.read_lock();
        let mut node: &Node = &root;
        let mut best: Option<(IndexEntry, usize)> = None;
        for (i, t) in tokens.iter().enumerate() {
            match node.children.get(t) {
                Some(child) => {
                    node = child;
                    if let Some(entry) = &node.entry {
                        if entry.state == CheckpointState::Published {
                            // Anchor touch: this entry just served a request, so
                            // it is the base of a live chain. Relaxed is fine —
                            // the value feeds a heuristic (eviction), and a torn
                            // read would still be *a* past time.
                            entry.last_used_millis.store(now_millis(), Relaxed);
                            best = Some((entry.clone(), i + 1));
                        }
                    }
                }
                None => break,
            }
        }
        best
    }

    /// Every Published entry, as an eviction candidate: its prefix tokens (the
    /// removal key), token count, last-use instant, and whether any *other*
    /// published entry extends it (an ancestor of a live chain — the ds4
    /// anchor rule: never reap the base a longer checkpoint was built on).
    pub fn eviction_candidates(&self) -> Vec<IndexCandidate> {
        use std::sync::atomic::Ordering::Relaxed;
        let root = self.read_lock();
        let mut out = Vec::new();
        // One DFS collects entries and marks extended prefixes: a published
        // node is "extended" if any published node below it shares its prefix.
        // Two passes over the same lock hold, so the snapshot is consistent.
        struct Frame<'a> {
            node: &'a Node,
            tokens: Vec<u32>,
        }
        let mut stack = vec![Frame {
            node: &root,
            tokens: Vec::new(),
        }];
        // (token_count, path_index into a flat list) pairs to patch later.
        let mut entries: Vec<(Vec<u32>, usize, u64, std::time::Instant, String, u64)> = Vec::new();
        while let Some(frame) = stack.pop() {
            if let Some(entry) = &frame.node.entry {
                if entry.state == CheckpointState::Published {
                    let millis = entry.last_used_millis.load(Relaxed);
                    entries.push((
                        frame.tokens.clone(),
                        entry.meta.token_count as usize,
                        millis,
                        last_used_instant(millis),
                        entry.blob_path.clone(),
                        entry.generation,
                    ));
                }
            }
            for (t, child) in &frame.node.children {
                let mut tokens = frame.tokens.clone();
                tokens.push(*t);
                stack.push(Frame {
                    node: child,
                    tokens,
                });
            }
        }
        // A candidate is an ANCHOR when its token prefix is a strict prefix of
        // another published candidate's prefix (a longer checkpoint extends it).
        for (tokens, count, _millis, instant, blob_path, generation) in &entries {
            let is_anchor = entries.iter().any(|(other, _, _, _, _, _)| {
                other.len() > tokens.len() && other.starts_with(tokens.as_slice())
            });
            out.push(IndexCandidate {
                tokens: tokens.clone(),
                token_count: *count,
                last_used: *instant,
                is_anchor,
                blob_path: blob_path.clone(),
                generation: *generation,
            });
        }
        out
    }

    /// Remove the Published entry at exactly `tokens` if it is still the same
    /// blob + generation. Returns true when removed (the blob file should then
    /// be unlinked by the caller). Never touches quarantined entries — they are
    /// diagnostics, not space.
    pub fn remove_published(&self, tokens: &[u32], blob_path: &str, generation: u64) -> bool {
        let mut root = self.write_lock();
        let mut node: &mut Node = &mut root;
        for t in tokens {
            match node.children.get_mut(t) {
                Some(child) => node = child,
                None => return false,
            }
        }
        match &mut node.entry {
            Some(entry)
                if entry.blob_path == blob_path
                    && entry.generation == generation
                    && entry.state == CheckpointState::Published =>
            {
                node.entry = None;
                true
            }
            _ => false,
        }
    }

    /// Reserve the next publication generation. Callers reserve BEFORE writing
    /// the blob so the file can carry a generation-specific (immutable) name: a
    /// republish never overwrites an earlier generation's file, so a delayed
    /// retirement cannot delete a healthy replacement.
    pub fn reserve_generation(&self) -> u64 {
        self.next_generation
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            + 1
    }

    /// Ensure the next reserved generation exceeds `max_persisted`. Called after
    /// a startup rebuild so a restart cannot hand a new publication a LOWER
    /// generation than a checkpoint already on disk — which the next rebuild
    /// would then discard as an older generation, losing the new work.
    pub fn seed_generation(&self, max_persisted: u64) {
        self.next_generation
            .fetch_max(max_persisted, std::sync::atomic::Ordering::Relaxed);
    }

    /// Publish an entry with a pre-reserved generation. Only callable for a prefix
    /// whose ancestors are consistent; the atomic blob rename must have completed
    /// BEFORE this call (R1-3).
    ///
    /// The daemon caches KV for `tokens[:-1]`, so a published entry must hold at
    /// least 2 tokens (a shorter prefix caches nothing and `lookup` would return
    /// `matched_tokens < 2`, classifying a request hit/partial with zero covered
    /// KV). Enforced in release, not only debug: the adapter (sidecar) is a trust
    /// boundary, and a non-conforming adapter could return a non-empty payload
    /// for a 1-token prompt. A short prefix is silently not published, matching
    /// the caller's "empty blob means nothing cached" convention; never panic, so
    /// a hostile adapter cannot crash the daemon. Returns true if published.
    pub fn publish(
        &self,
        tokens: &[u32],
        meta: CheckpointMeta,
        blob_path: String,
        generation: u64,
        on_replaced: impl FnOnce(&str),
    ) -> bool {
        if tokens.len() < 2 {
            return false;
        }
        let mut root = self.write_lock();
        let mut node: &mut Node = &mut root;
        for t in tokens {
            node = node.children.entry(*t).or_default();
        }
        // Reclaim the superseded generation's file: with immutable per-generation
        // names, replacing an entry (a fingerprint-miss republish at the same
        // prefix) would otherwise leak the old file forever. Runs under the write
        // lock so a concurrent retire of the old entry cannot race the unlink.
        if let Some(old) = &node.entry {
            if old.blob_path != blob_path {
                on_replaced(&old.blob_path);
            }
        }
        node.entry = Some(IndexEntry {
            meta,
            blob_path,
            generation,
            state: CheckpointState::Published,
            // A freshly published checkpoint counts as just-used: it must not
            // be the first thing the reaper picks the moment it goes cold,
            // since it is by definition the newest KV in the store.
            last_used_millis: std::sync::atomic::AtomicU64::new(now_millis()),
        });
        true
    }

    /// Quarantine the entry that points at `blob_path` (R1-1): keep it visible
    /// for diagnostics but never serve it. Returns true if an entry was marked.
    ///
    /// `prefix` is the token key the entry lives at (from `lookup`), so this
    /// descends directly in O(prefix) instead of scanning the whole tree. The
    /// name and generation are verified at that node, so it still cannot
    /// quarantine a healthy ancestor or a fresh republish that reused the same
    /// deterministic blob name.
    ///
    /// `on_retired` runs WHILE the write lock is held, after the entry is marked.
    /// The caller deletes the blob file there: holding the lock across the delete
    /// serializes it against `publish` (same lock), so a concurrent republish of
    /// the same deterministic name cannot land between the generation check and
    /// the unlink and lose its file. Keep the callback short (one unlink).
    pub fn quarantine_blob(
        &self,
        blob_path: &str,
        generation: u64,
        prefix: &[u32],
        on_retired: impl FnOnce(),
    ) -> bool {
        let mut root = self.write_lock();
        let mut node: &mut Node = &mut root;
        for t in prefix {
            match node.children.get_mut(t) {
                Some(child) => node = child,
                None => return false,
            }
        }
        match &mut node.entry {
            Some(entry)
                if entry.blob_path == blob_path
                    && entry.generation == generation
                    && entry.state == CheckpointState::Published =>
            {
                entry.state = CheckpointState::Quarantined;
                on_retired();
                true
            }
            _ => false,
        }
    }

    /// Count of published entries (for /stats).
    pub fn published_count(&self) -> usize {
        self.count_in_state(CheckpointState::Published)
    }

    /// Count of quarantined entries (for /stats and tests).
    pub fn quarantined_count(&self) -> usize {
        self.count_in_state(CheckpointState::Quarantined)
    }

    /// Count entries in `state`. Iterative so a very long prefix cannot overflow
    /// the thread stack.
    fn count_in_state(&self, state: CheckpointState) -> usize {
        let root = self.read_lock();
        let mut n = 0usize;
        let mut stack: Vec<&Node> = vec![&root];
        while let Some(node) = stack.pop() {
            n += usize::from(node.entry.as_ref().is_some_and(|e| e.state == state));
            for child in node.children.values() {
                stack.push(child);
            }
        }
        n
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::ModelFingerprint;

    fn fingerprint() -> ModelFingerprint {
        ModelFingerprint {
            model_id: "test-model".into(),
            tokenizer_hash: "abc123".into(),
            kv_dtype: "f16".into(),
            kv_layout_version: 1,
            ..Default::default()
        }
    }

    /// Reserve a generation and publish (the real call order), returning it.
    fn publish_entry(index: &PrefixIndex, tokens: &[u32], name: &str) -> u64 {
        let generation = index.reserve_generation();
        assert!(index.publish(tokens, meta(), name.into(), generation, |_| {}));
        generation
    }

    fn meta() -> CheckpointMeta {
        CheckpointMeta {
            fingerprint: fingerprint(),
            token_count: 3,
            tokens: vec![1, 2, 3],
            format_version: 1,
        }
    }

    #[test]
    fn pre_quantization_blobs_deserialize_with_default_f16_tier() {
        // Back-compat (T12 adoption): blobs written before kv_bits/kv_group_size
        // existed have neither field in their meta JSON. They must deserialize
        // to the 0/0 (f16) tier so restart rebuilds keep serving them, and the
        // 0/0 tier must be distinct from any quantized tier.
        let old_json = serde_json::json!({
            "model_id": "test-model",
            "tokenizer_hash": "abc123",
            "kv_dtype": "f16",
            "kv_layout_version": 1
        });
        let fp: ModelFingerprint = serde_json::from_value(old_json).unwrap();
        assert_eq!(fp.kv_bits, 0, "absent bits must default to f16");
        assert_eq!(fp.kv_group_size, 0);
        assert_eq!(fp, fingerprint(), "defaults match the f16 test fingerprint");

        // A quantized tier is a different fingerprint: never cross-serves.
        let mut q8 = fingerprint();
        q8.kv_bits = 8;
        q8.kv_group_size = 64;
        assert_ne!(fp, q8);

        // Round-trip: a q8 fingerprint serializes and restores exactly.
        let back: ModelFingerprint =
            serde_json::from_slice(&serde_json::to_vec(&q8).unwrap()).unwrap();
        assert_eq!(back, q8);
    }

    #[test]
    fn longest_prefix_match() {
        let index = PrefixIndex::new();
        let tokens = [1, 2, 3];
        assert!(index.lookup(&tokens).is_none());
        let _ = publish_entry(&index, &tokens, "blob-123");
        // Exact match
        let (entry, matched) = index.lookup(&tokens).unwrap();
        assert_eq!(matched, 3);
        assert_eq!(entry.blob_path, "blob-123");
        // Extension matches the same entry
        let (entry, matched) = index.lookup(&[1, 2, 3, 4, 5]).unwrap();
        assert_eq!(matched, 3);
        assert_eq!(entry.blob_path, "blob-123");
        // Non-matching prefix
        assert!(index.lookup(&[9, 9]).is_none());
    }

    #[test]
    fn quarantined_entries_not_served() {
        let index = PrefixIndex::new();
        let tokens = [1, 2, 3];
        let _ = publish_entry(&index, &tokens, "blob-123");
        let gen = index.lookup(&tokens).unwrap().0.generation;
        assert!(index.quarantine_blob("blob-123", gen, &tokens, || {}));
        assert!(index.lookup(&tokens).is_none());
        assert_eq!(index.published_count(), 0);
    }

    #[test]
    fn quarantine_on_partial_request_marks_the_matched_entry() {
        // Regression: a partial hit's entry ends at a SHORTER prefix than the
        // request. Quarantine must mark the matched entry (by its blob name), not
        // walk past it to the request terminal (where there is no entry), or a
        // poison blob stays Published and is served forever.
        let index = PrefixIndex::new();
        let _ = publish_entry(&index, &[1, 2], "short");
        // Request extends the published prefix; lookup matches at depth 2.
        let request = [1, 2, 3, 4];
        let (entry, matched) = index.lookup(&request).unwrap();
        assert_eq!(matched, 2);
        assert!(
            index.quarantine_blob(
                &entry.blob_path,
                entry.generation,
                &request[..matched],
                || {}
            ),
            "quarantine must mark the matched entry on a partial request"
        );
        assert!(
            index.lookup(&request).is_none(),
            "the poison entry must no longer be served"
        );
        assert_eq!(index.published_count(), 0);
    }

    #[test]
    fn quarantine_marks_deepest_of_nested_entries() {
        // With nested entries [1,2] and [1,2,3,4], a request [1,2,3,4,5]
        // matches the deepest, [1,2,3,4]; quarantine that one, not the ancestor.
        let index = PrefixIndex::new();
        let _ = publish_entry(&index, &[1, 2], "ancestor");
        let _ = publish_entry(&index, &[1, 2, 3, 4], "deep");
        let deep = index.lookup(&[1, 2, 3, 4, 5]).unwrap().0;
        assert_eq!(deep.blob_path, "deep");
        assert!(index.quarantine_blob(&deep.blob_path, deep.generation, &[1, 2, 3, 4], || {}));
        assert!(
            index.lookup(&[1, 2, 3, 4, 5]).is_some(),
            "the shallower ancestor is still published"
        );
        let (entry, matched) = index.lookup(&[1, 2, 3, 4, 5]).unwrap();
        assert_eq!(matched, 2);
        assert_eq!(entry.blob_path, "ancestor");
    }

    #[test]
    fn quarantine_blob_retires_only_the_exact_generation() {
        // A late failure must retire the publication it used, not a fresh
        // republish that reused the same deterministic blob name. Publish gen 1,
        // then republish the SAME name (gen 2); retiring gen 1 must be a no-op,
        // and retiring gen 2 must quarantine.
        let index = PrefixIndex::new();
        let _ = publish_entry(&index, &[1, 2, 3], "same-name");
        let gen1 = index.lookup(&[1, 2, 3]).unwrap().0.generation;
        let _ = publish_entry(&index, &[1, 2, 3], "same-name");
        let gen2 = index.lookup(&[1, 2, 3]).unwrap().0.generation;
        assert_ne!(gen1, gen2, "each publish bumps the generation");
        assert!(
            !index.quarantine_blob("same-name", gen1, &[1, 2, 3], || {}),
            "retiring a superseded generation must not touch the live entry"
        );
        assert!(
            index.lookup(&[1, 2, 3]).is_some(),
            "the fresh republish is still served"
        );
        assert!(
            index.quarantine_blob("same-name", gen2, &[1, 2, 3], || {}),
            "retiring the live generation quarantines it"
        );
        assert!(
            index.lookup(&[1, 2, 3]).is_none(),
            "the retired entry is no longer served"
        );
    }

    #[test]
    fn publish_returns_the_generation_that_retires_it() {
        // The leader records the generation `publish` returns so a later
        // rejection of its own blob retires it. If publish returned 0 (or the
        // caller kept None), quarantine would match nothing and leave poison
        // Published. Pin the round trip.
        let index = PrefixIndex::new();
        let generation = publish_entry(&index, &[1, 2, 3], "leader-blob");
        assert_ne!(generation, 0, "a real publish has a non-zero generation");
        assert!(
            index.quarantine_blob("leader-blob", generation, &[1, 2, 3], || {}),
            "the generation publish returned must retire the entry"
        );
        assert!(index.lookup(&[1, 2, 3]).is_none());
    }

    #[test]
    fn quarantine_blob_descends_the_given_prefix_only() {
        // Keyed descent: a wrong prefix must not find (or touch) the entry, even
        // with the right name+generation. This is why the caller passes the
        // matched prefix; it also makes retirement O(prefix), not a full scan.
        let index = PrefixIndex::new();
        let _ = publish_entry(&index, &[1, 2, 3], "b");
        let gen = index.lookup(&[1, 2, 3]).unwrap().0.generation;
        assert!(
            !index.quarantine_blob("b", gen, &[1, 2, 9], || {}),
            "a wrong prefix must not match the entry"
        );
        assert!(index.lookup(&[1, 2, 3]).is_some());
        assert!(index.quarantine_blob("b", gen, &[1, 2, 3], || {}));
        assert!(index.lookup(&[1, 2, 3]).is_none());
    }

    #[test]
    fn quarantine_callback_runs_only_on_match() {
        // The on_retired callback (blob deletion) must run only when the entry is
        // actually quarantined. A stale generation must not delete the live
        // entry's file.
        let index = PrefixIndex::new();
        let _ = publish_entry(&index, &[1, 2, 3], "b");
        let gen1 = index.lookup(&[1, 2, 3]).unwrap().0.generation;
        let _ = publish_entry(&index, &[1, 2, 3], "b"); // republish -> gen2
        let called = std::cell::Cell::new(false);
        assert!(!index.quarantine_blob("b", gen1, &[1, 2, 3], || called.set(true)));
        assert!(
            !called.get(),
            "stale generation must not run the delete callback"
        );
        assert!(index.quarantine_blob("b", gen1 + 1, &[1, 2, 3], || called.set(true)));
        assert!(called.get(), "the matching entry runs the callback");
    }

    #[test]
    fn publish_reclaims_the_replaced_generation() {
        // Immutable per-generation names mean a republish at the same prefix
        // (a fingerprint-miss overwrite) leaves the old file behind unless it is
        // reclaimed. The on_replaced callback must receive the superseded name.
        let index = PrefixIndex::new();
        let gen1 = index.reserve_generation();
        assert!(index.publish(&[1, 2, 3], meta(), "old".into(), gen1, |_| {}));
        let gen2 = index.reserve_generation();
        let reclaimed = std::cell::RefCell::new(None);
        assert!(
            index.publish(&[1, 2, 3], meta(), "new".into(), gen2, |old| {
                *reclaimed.borrow_mut() = Some(old.to_string());
            })
        );
        assert_eq!(reclaimed.borrow().as_deref(), Some("old"));
        // Re-publishing the SAME name (same file) must not try to delete it.
        let gen3 = index.reserve_generation();
        let mut called = false;
        assert!(index.publish(&[1, 2, 3], meta(), "new".into(), gen3, |_| {
            called = true;
        }));
        assert!(!called, "republishing the same path must not reclaim it");
    }

    #[test]
    fn publish_counts() {
        let index = PrefixIndex::new();
        let _ = publish_entry(&index, &[1, 2], "a");
        let _ = publish_entry(&index, &[1, 2, 3], "b");
        assert_eq!(index.published_count(), 2);
    }

    #[test]
    fn published_prefix_shorter_than_two_never_serves_kv() {
        // Invariant the daemon relies on: a published checkpoint caches KV for
        // tokens[:-1], so it must hold at least 2 tokens. A shorter prefix must
        // not be published, or `lookup` could return matched_tokens < 2 and the
        // request would be classified hit/partial with zero covered KV. The guard
        // is a real release-time check (the adapter is a trust boundary), so a
        // short prefix is silently dropped rather than indexed. Uses a prefix
        // whose second token would otherwise make lookup succeed at depth 1.
        let index = PrefixIndex::new();
        let gen = index.reserve_generation();
        assert!(!index.publish(&[42], meta(), "short".into(), gen, |_| {}));
        assert_eq!(
            index.published_count(),
            0,
            "a 1-token prefix must not index"
        );
        assert!(
            index.lookup(&[42]).is_none(),
            "a 1-token prefix must never match"
        );
    }

    #[test]
    fn matched_never_exceeds_request_length() {
        // A request that is a strict prefix of a longer published entry must
        // NOT match the longer entry: the walk ends when request tokens run out,
        // so no entry beyond the request tail is ever seen. This guards the
        // invariant that `classify` can never see matched_tokens > request_tokens.
        let index = PrefixIndex::new();
        let _ = publish_entry(&index, &[1, 2, 3, 4, 5], "long");
        assert!(
            index.lookup(&[1, 2]).is_none(),
            "short request must not match a longer entry"
        );
        // And the overlapping case: published [1,2], request [1,2,3] matches at 2.
        let index = PrefixIndex::new();
        let _ = publish_entry(&index, &[1, 2], "short");
        let (_, matched) = index.lookup(&[1, 2, 3]).unwrap();
        assert_eq!(matched, 2);
        assert!(matched <= 3);
    }

    #[test]
    fn poisoned_lock_is_recovered_not_panicked() {
        // A panic while holding the index lock poisons it. Later requests must
        // still be served, not abort the daemon.
        use std::sync::Arc;
        let index = Arc::new(PrefixIndex::new());
        let _ = publish_entry(&index, &[1, 2, 3], "b");

        let idx = index.clone();
        let _ = std::thread::spawn(move || {
            let _guard = idx.root.write().unwrap();
            panic!("poison the lock");
        })
        .join();

        assert!(index.root.is_poisoned(), "lock should be poisoned");
        // Must not panic, and the pre-existing entry is still readable.
        let (entry, matched) = index.lookup(&[1, 2, 3]).expect("recovered lookup");
        assert_eq!(matched, 3);
        assert_eq!(entry.blob_path, "b");
        // Writes still work too.
        let _ = publish_entry(&index, &[4, 5], "c");
        assert_eq!(index.published_count(), 2);
    }

    #[test]
    fn eviction_candidates_reports_anchors_and_sizes() {
        // [1,2] is the base of the live chain ending at [1,2,3,4]: it must be
        // reported as an anchor (is_anchor), while the standalone [9,9] is not.
        let index = PrefixIndex::new();
        let _ = publish_entry(&index, &[1, 2], "base");
        let _ = publish_entry(&index, &[1, 2, 3, 4], "chain");
        let _ = publish_entry(&index, &[9, 9], "loner");
        let mut cands = index.eviction_candidates();
        cands.sort_by(|a, b| a.tokens.cmp(&b.tokens));
        assert_eq!(cands.len(), 3);
        assert!(cands[0].is_anchor, "[1,2] extends to [1,2,3,4]");
        assert!(!cands[1].is_anchor, "[1,2,3,4] has no published extension");
        assert!(!cands[2].is_anchor, "[9,9] is standalone");
        assert_eq!(cands[0].token_count, 3);
        assert_eq!(cands[2].blob_path, "loner");
    }

    #[test]
    fn remove_published_removes_exact_entry_only() {
        let index = PrefixIndex::new();
        let _ = publish_entry(&index, &[1, 2], "base");
        let chain_gen = publish_entry(&index, &[1, 2, 3, 4], "chain");
        // Wrong generation / wrong blob / wrong prefix all refuse.
        assert!(!index.remove_published(&[1, 2, 3, 4], "chain", chain_gen + 1));
        assert!(!index.remove_published(&[1, 2, 3, 4], "other", chain_gen));
        assert!(!index.remove_published(&[1, 2], "chain", chain_gen));
        assert_eq!(index.published_count(), 2, "nothing removed yet");
        // Exact match removes, and only that entry.
        assert!(index.remove_published(&[1, 2, 3, 4], "chain", chain_gen));
        assert_eq!(index.published_count(), 1);
        // [1,2,3,4] now matches only the surviving base, at depth 2.
        let (entry, matched) = index.lookup(&[1, 2, 3, 4]).unwrap();
        assert_eq!(matched, 2, "the removed entry must no longer match");
        assert_eq!(entry.blob_path, "base");
        // Removing again is a no-op (idempotent-safe).
        assert!(!index.remove_published(&[1, 2, 3, 4], "chain", chain_gen));
    }

    #[test]
    fn quarantined_entries_are_not_eviction_candidates() {
        // Quarantine keeps a tombstone for diagnostics; eviction must not reap
        // (and unlink) it a second time.
        let index = PrefixIndex::new();
        let _ = publish_entry(&index, &[1, 2], "base");
        let gen = publish_entry(&index, &[7, 7, 7], "poison");
        assert!(index.quarantine_blob("poison", gen, &[7, 7, 7], || {}));
        let cands = index.eviction_candidates();
        assert_eq!(cands.len(), 1, "only the published entry is a candidate");
        assert_eq!(cands[0].tokens, vec![1, 2]);
    }

    #[test]
    fn lookup_touches_last_used_for_recency_anchoring() {
        // The eviction policy anchors recently-served checkpoints. The touch
        // happens through the read lock; verify it observes a later timestamp
        // than publish time. The millis clock truncates, so sleep past one
        // full tick (2ms) to guarantee strict progress on any platform.
        let index = PrefixIndex::new();
        let _ = publish_entry(&index, &[1, 2, 3], "blob");
        let before = index.eviction_candidates()[0].last_used;
        std::thread::sleep(std::time::Duration::from_millis(2));
        let _ = index.lookup(&[1, 2, 3, 9]);
        let after = index.eviction_candidates()[0].last_used;
        assert!(
            after > before,
            "lookup must advance the entry's last_used instant"
        );
    }
}
