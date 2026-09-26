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

#[derive(Debug, Clone)]
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
}

#[derive(Debug, thiserror::Error)]
pub enum IndexError {
    #[error("index corrupt: {0}")]
    Corrupt(String),
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
        let root = self.read_lock();
        let mut node: &Node = &root;
        let mut best: Option<(IndexEntry, usize)> = None;
        for (i, t) in tokens.iter().enumerate() {
            match node.children.get(t) {
                Some(child) => {
                    node = child;
                    if let Some(entry) = &node.entry {
                        if entry.state == CheckpointState::Published {
                            best = Some((entry.clone(), i + 1));
                        }
                    }
                }
                None => break,
            }
        }
        best
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
    ) -> bool {
        if tokens.len() < 2 {
            return false;
        }
        let mut root = self.write_lock();
        let mut node: &mut Node = &mut root;
        for t in tokens {
            node = node.children.entry(*t).or_default();
        }
        node.entry = Some(IndexEntry {
            meta,
            blob_path,
            generation,
            state: CheckpointState::Published,
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
        }
    }

    /// Reserve a generation and publish (the real call order), returning it.
    fn publish_entry(index: &PrefixIndex, tokens: &[u32], name: &str) -> u64 {
        let generation = index.reserve_generation();
        assert!(index.publish(tokens, meta(), name.into(), generation));
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
        assert!(!index.publish(&[42], meta(), "short".into(), gen));
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
}
