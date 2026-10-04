//! Native prefix index: radix tree over token IDs.
//!
//! Longest-prefix match in O(prefix length). No SQLite on the hot path (D2-reopened).
//! Index rows are only visible after the checkpoint blob's atomic publish (R1-3).

use crate::contract::{CheckpointMeta, CheckpointState, ModelFingerprint};
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
    /// The exact trie key this entry lives at (the path from the root).
    /// A lookup can serve this entry to a request that diverges from the
    /// key at its LAST token (T22: the blob holds KV for the key's
    /// prefix-minus-one, so the final key token's absence is by
    /// construction), which means the request's own tokens do NOT
    /// spell the key — callers that need the key (quarantine keying)
    /// must take it from here, never from the request prefix.
    pub key: Vec<u32>,
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
            key: self.key.clone(),
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
    ///
    /// T22 (prefix-stable multi-turn): an entry serves a request whose token
    /// stream diverges from the entry's key at the key's LAST token. The
    /// published blob holds KV for the key's `tokens[:-1]` (adapter
    /// convention), and the KV at position j depends only on tokens `..=j`,
    /// so agreement through `len(key) - 1` makes every covered position
    /// byte-identical for the request too — the same KV the full-containment
    /// case would have served. Trie-wise the walk breaks one edge short of
    /// the entry, so serving also considers the deepest-reached node's
    /// published direct children (key length = depth + 1, LCP = depth =
    /// len(key) - 1) at a break, and at walk end (request a strict prefix of
    /// the key). The returned `matched` is `min(len(key), tokens.len())`, so
    /// the policy's `covered = matched - 1` stays exact except the
    /// request-is-one-shorter boundary (matched capped to the request
    /// length under-reports coverage by one there — stats only; the adapter
    /// derives its true feed point from the blob's own meta). The entry
    /// always reports its TRUE `key`: a divergent request's tokens do not
    /// spell it, so quarantine/eviction must key by `entry.key`. Mid-prefix
    /// divergence is never served — KV past the LCP is not identical for the
    /// diverging request, and truncating persistent MLX cache states is
    /// unsafe on the quantized tier (explicitly out of scope).
    /// Longest-prefix lookup with T22 end-anchored divergence serving.
    ///
    /// `fingerprint` gates BOTH the result and the recency touch: only
    /// entries whose meta fingerprint matches are served, and only a
    /// fingerprint-matched touch counts as "recently served". A request with
    /// the wrong model/tokenizer/kv-tier must not refresh a cold blob's
    /// anchor window (an unauthenticated wrong-model loop would otherwise
    /// keep every blob eviction-proof for the anchor duration — disk
    /// exhaustion by anchor). It also must not let a fingerprint-mismatched
    /// child mask a fingerprint-matched exact entry at the walk-end node
    /// (the mismatch classifies as Miss, so serving it is a lie). `None`
    /// disables the gate (tests, benches).
    pub fn lookup(
        &self,
        tokens: &[u32],
        fingerprint: Option<&ModelFingerprint>,
    ) -> Option<(IndexEntry, usize)> {
        use std::sync::atomic::Ordering::Relaxed;
        let fp_ok = |entry: &IndexEntry| fingerprint.is_none_or(|f| entry.meta.fingerprint == *f);
        let root = self.read_lock();
        let mut node: &Node = &root;
        let mut best: Option<(IndexEntry, usize)> = None;
        for (i, t) in tokens.iter().enumerate() {
            match node.children.get(t) {
                Some(child) => {
                    node = child;
                    if let Some(entry) = &node.entry {
                        if entry.state == CheckpointState::Published && fp_ok(entry) {
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
        // T22 end-anchored divergence: a published DIRECT child of the
        // deepest-reached node has key length depth+1; whether the walk
        // broke on a request token or consumed the request exactly, the
        // request agrees with such a child through depth = len(key) - 1 —
        // exactly the serve rule — PROVIDED the request is long enough
        // (checked below). Deeper descendants need LCP >= len(key)-1 >
        // depth: never qualify.
        //
        // The child is adoptable ONLY when the request is at least as long as
        // the child's key: the sidecar covers len(key)-1 positions and must
        // still receive the key's final token as part of the delta it feeds
        // (prefix_len > len(tokens) is rejected as unusable → scratch). A
        // shorter request therefore can NEVER be served by the child branch —
        // serving it would report a hit/partial while the adapter runs from
        // scratch (an accounting lie) and would mask a usable exact entry at
        // the walk-end node. The break shape satisfies len(key) <= len(tokens)
        // by construction; the walk-end shape never does, so the guard simply
        // makes the walk-end branch inert. Prefer the hottest eligible child
        // (recency tie-break: lowest token, deterministic); its coverage
        // (len(key)-1) beats any walk best, which ends at depth < len(key)-1.
        if let Some(child_entry) = Self::hottest_published_child(node, fingerprint) {
            // Touch through the reference BEFORE cloning: the clone's
            // last_used_millis is a copy, and storing into it would leave the
            // real trie entry cold (the reaper would then reap the base of a
            // live multi-turn chain).
            child_entry.last_used_millis.store(now_millis(), Relaxed);
            if tokens.len() >= child_entry.key.len() {
                let matched = child_entry.key.len();
                if matched >= 2 {
                    best = Some((child_entry.clone(), matched));
                }
            }
        }
        best
    }

    /// Hottest published direct child of `node` (T22 serve rule): the entry a
    /// request that diverges at the child key's last token may adopt.
    /// Selection is by recency (last_used_millis) with a lowest-token
    /// tie-break for determinism; every candidate covers the same positions,
    /// so any of them is sound and the pick only affects which blob stays
    /// warm. Returns the entry BORROWED from the trie so the caller can touch
    /// recency through the real entry before cloning (a returned clone would
    /// make the touch a no-op on the trie).
    fn hottest_published_child<'a>(
        node: &'a Node,
        fingerprint: Option<&ModelFingerprint>,
    ) -> Option<&'a IndexEntry> {
        let mut best: Option<(u64, u32, &IndexEntry)> = None;
        for (t, child) in &node.children {
            let Some(entry) = &child.entry else {
                continue;
            };
            if entry.state != CheckpointState::Published {
                continue;
            }
            if let Some(f) = fingerprint {
                if entry.meta.fingerprint != *f {
                    continue;
                }
            }
            let used = entry
                .last_used_millis
                .load(std::sync::atomic::Ordering::Relaxed);
            let better = match &best {
                None => true,
                Some((bu, bt, _)) => used > *bu || (used == *bu && t < bt),
            };
            if better {
                best = Some((used, *t, entry));
            }
        }
        best.map(|(_, _, entry)| entry)
    }

    /// Every Published entry, as an eviction candidate: its prefix tokens (the
    /// removal key), token count, last-use instant, and whether any *other*
    /// published entry extends it (an ancestor of a live chain — the ds4
    /// anchor rule: never reap the base a longer checkpoint was built on).
    pub fn eviction_candidates(&self) -> Vec<IndexCandidate> {
        use std::sync::atomic::Ordering::Relaxed;
        let root = self.read_lock();
        let mut out = Vec::new();
        // One DFS collects entries, marks anchors, and materializes removal
        // paths. The snapshot is taken under one read lock, so it is
        // consistent with itself.
        //
        // Two cost invariants (the previous implementation cloned the full
        // path for EVERY node — O(depth) each, quadratic on long chains —
        // and marked anchors with an all-pairs prefix scan, O(entries^2)):
        // (1) frames link to their parent by index; an entry's path is
        // materialized ONCE, by walking up (O(depth) per ENTRY, and entries
        // are the output itself, so no cheaper shape exists);
        // (2) "any published entry strictly below" propagates bottom-up in
        // the unwind instead of pairwise compares — O(nodes) total.
        struct Frame<'a> {
            node: &'a Node,
            parent: Option<usize>,
            /// The child token linking this frame to its parent (root: 0).
            token: u32,
            /// A published entry at or below this node (own entry folded in
            /// during the unwind).
            pub_below: bool,
        }
        let mut stack: Vec<Frame<'_>> = vec![Frame {
            node: &root,
            parent: None,
            token: 0,
            pub_below: false,
        }];
        // Flat frame arena, parents always at lower indices than children.
        let mut frames: Vec<Frame<'_>> = Vec::new();
        while let Some(f) = stack.pop() {
            let idx = frames.len();
            frames.push(f);
            for (t, child) in &frames[idx].node.children {
                stack.push(Frame {
                    node: child,
                    parent: Some(idx),
                    token: *t,
                    pub_below: false,
                });
            }
        }
        // Unwind children-before-parents (reverse index order): each frame
        // folds its published-entry flag into its parent, so by the time a
        // parent is visited, pub_below already covers its whole subtree.
        // The snapshot collects entries; paths materialize afterwards so the
        // flags are final when the anchor bit is read.
        struct Snapshot {
            frame_idx: usize,
            token_count: usize,
            last_used: std::time::Instant,
            blob_path: String,
            generation: u64,
        }
        let mut snapshots: Vec<Snapshot> = Vec::new();
        for i in (0..frames.len()).rev() {
            let mut below_self = frames[i].pub_below;
            if let Some(entry) = &frames[i].node.entry {
                if entry.state == CheckpointState::Published {
                    below_self = true;
                    let millis = entry.last_used_millis.load(Relaxed);
                    snapshots.push(Snapshot {
                        frame_idx: i,
                        token_count: entry.meta.token_count as usize,
                        last_used: last_used_instant(millis),
                        blob_path: entry.blob_path.clone(),
                        generation: entry.generation,
                    });
                }
            }
            if let Some(p) = frames[i].parent {
                frames[p].pub_below |= below_self;
            }
        }
        for s in snapshots {
            // Anchor rule (ds4): a published entry whose prefix a LONGER
            // published entry extends is a live chain's base. Its own node's
            // pub_below excludes the entry itself (the own-entry fold went to
            // the parent), so a true flag means a strict descendant holds one.
            let is_anchor = frames[s.frame_idx].pub_below;
            let mut path: Vec<u32> = Vec::new();
            let mut cur = Some(s.frame_idx);
            while let Some(i) = cur {
                let f = &frames[i];
                if f.parent.is_some() {
                    path.push(f.token);
                }
                cur = f.parent;
            }
            path.reverse();
            out.push(IndexCandidate {
                tokens: path,
                token_count: s.token_count,
                last_used: s.last_used,
                is_anchor,
                blob_path: s.blob_path,
                generation: s.generation,
            });
        }
        out
    }

    /// Remove the Published entry at exactly `tokens` if it is still the same
    /// blob + generation. Returns true when removed (the blob file should then
    /// be unlinked by the caller). Never touches quarantined entries — they are
    /// diagnostics, not space.
    ///
    /// The removal also RECLAIMS now-empty trie branches along the path
    /// (unique-prompt churn otherwise retains every node it ever published,
    /// forever, eviction or not). The prune is O(depth), not quadratic: the
    /// descent tracks the deepest "keep" node on the path — a node with an
    /// entry (any state) or two or more children — and everything strictly
    /// below it on this path is a linear chain of childless, entryless nodes
    /// (each holds exactly the path child). One `children.remove` at the keep
    /// node then drops the whole chain at once (Drop cascades). Both descents
    /// run under the same write lock, so a concurrent publish can never sneak
    /// a live entry into the reclaimed branch.
    pub fn remove_published(&self, tokens: &[u32], blob_path: &str, generation: u64) -> bool {
        if tokens.is_empty() {
            return false;
        }
        let mut root = self.write_lock();
        let mut node: &mut Node = &mut root;
        // Depth of the deepest node that must survive (root counts: it can
        // never be removed itself). A node on the path with an entry — any
        // state, quarantined included — or with a child OFF the path anchors
        // the kept prefix.
        let mut last_keep = 0usize;
        for (j, t) in tokens.iter().enumerate() {
            if node.entry.is_some() || node.children.len() >= 2 {
                last_keep = j;
            }
            match node.children.get_mut(t) {
                Some(child) => node = child,
                None => return false,
            }
        }
        let removed = match &mut node.entry {
            Some(entry)
                if entry.blob_path == blob_path
                    && entry.generation == generation
                    && entry.state == CheckpointState::Published =>
            {
                node.entry = None;
                true
            }
            _ => false,
        };
        // Prunable suffix: the removed entry's node must be childless (a
        // deeper chain anchored at it survives), and the suffix is non-empty
        // by construction (last_keep <= len-1 < the removed node's depth).
        if removed && node.children.is_empty() {
            let mut cur: &mut Node = &mut root;
            for t in &tokens[..last_keep] {
                cur = cur
                    .children
                    .get_mut(t)
                    .expect("path existed during descent");
            }
            // Drop cascades through the empty chain below this child.
            cur.children.remove(&tokens[last_keep]);
        }
        removed
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
            // The trie path IS the key (publish is the only constructor);
            // lookups that serve an end-divergent request report this key,
            // not the request's tokens (T22).
            key: tokens.to_vec(),
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

    fn fingerprint_other() -> ModelFingerprint {
        ModelFingerprint {
            model_id: "other-model".into(),
            ..fingerprint()
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
        assert!(index.lookup(&tokens, None).is_none());
        let _ = publish_entry(&index, &tokens, "blob-123");
        // Exact match
        let (entry, matched) = index.lookup(&tokens, None).unwrap();
        assert_eq!(matched, 3);
        assert_eq!(entry.blob_path, "blob-123");
        // Extension matches the same entry
        let (entry, matched) = index.lookup(&[1, 2, 3, 4, 5], None).unwrap();
        assert_eq!(matched, 3);
        assert_eq!(entry.blob_path, "blob-123");
        // Non-matching prefix
        assert!(index.lookup(&[9, 9], None).is_none());
    }

    #[test]
    fn quarantined_entries_not_served() {
        let index = PrefixIndex::new();
        let tokens = [1, 2, 3];
        let _ = publish_entry(&index, &tokens, "blob-123");
        let gen = index.lookup(&tokens, None).unwrap().0.generation;
        assert!(index.quarantine_blob("blob-123", gen, &tokens, || {}));
        assert!(index.lookup(&tokens, None).is_none());
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
        // Request extends the published prefix; looku