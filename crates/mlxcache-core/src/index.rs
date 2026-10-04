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

    /// Quarantine-candidate records, coldest first: every quarantined entry
    /// ordered by `last_used` ascending, capped at `max_keep`. A quarantine
    /// tombstone is diagnostic state, not a serveable asset: left unbounded, a
    /// daemon that quarantines pathological blobs for weeks grows this map
    /// forever (QA ISSUE-002 follow-up). The reaper calls this each pass and
    /// retires the coldest stones beyond the cap via `remove_quarantined`.
    /// Returns (key, blob_path, generation) triples — the key comes from the
    /// entry itself (a serve can key-diverge from the request, T22).
    pub fn quarantine_candidates(&self, max_keep: usize) -> Vec<(Vec<u32>, String, u64)> {
        let root = self.read_lock();
        let mut all: Vec<(u64, Vec<u32>, String, u64)> = Vec::new();
        let mut stack: Vec<(Vec<u32>, &Node)> = vec![(Vec::new(), &root)];
        while let Some((prefix, node)) = stack.pop() {
            if let Some(entry) = &node.entry {
                if entry.state == CheckpointState::Quarantined {
                    all.push((
                        entry
                            .last_used_millis
                            .load(std::sync::atomic::Ordering::Relaxed),
                        prefix.clone(),
                        entry.blob_path.clone(),
                        entry.generation,
                    ));
                }
            }
            for (t, child) in &node.children {
                let mut p = prefix.clone();
                p.push(*t);
                stack.push((p, child));
            }
        }
        // Coldest first (ascending last_used). The reaper removes stones while
        // the live count exceeds the cap, so the candidates are the EXCESS:
        // `len - max_keep` coldest stones, front = coldest first.
        all.sort_by_key(|(lu, _, _, _)| *lu);
        let excess = all.len().saturating_sub(max_keep);
        all.into_iter()
            .take(excess)
            .map(|(_, tokens, blob_path, generation)| (tokens, blob_path, generation))
            .collect()
    }

    /// Remove ONE quarantined entry by exact identity (tokens + blob name +
    /// generation), pruning the trie branch if it dies. Mirrors
    /// `remove_published`'s verification contract: never removes a published
    /// entry, an ancestor, or a republish that reused the name.
    pub fn remove_quarantined(&self, tokens: &[u32], blob_path: &str, generation: u64) -> bool {
        let mut root = self.write_lock();
        // Same O(depth) prune shape as remove_published: track the deepest
        // keep-node on the path, drop the childless suffix at it.
        let mut node: &mut Node = &mut root;
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
                    && entry.state == CheckpointState::Quarantined =>
            {
                node.entry = None;
                true
            }
            _ => false,
        };
        if removed && node.children.is_empty() {
            let mut cur: &mut Node = &mut root;
            for t in &tokens[..last_keep] {
                cur = cur
                    .children
                    .get_mut(t)
                    .expect("path existed during descent");
            }
            cur.children.remove(&tokens[last_keep]);
        }
        removed
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
        // Request extends the published prefix; lookup matches at depth 2.
        let request = [1, 2, 3, 4];
        let (entry, matched) = index.lookup(&request, None).unwrap();
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
            index.lookup(&request, None).is_none(),
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
        let deep = index.lookup(&[1, 2, 3, 4, 5], None).unwrap().0;
        assert_eq!(deep.blob_path, "deep");
        assert!(index.quarantine_blob(&deep.blob_path, deep.generation, &[1, 2, 3, 4], || {}));
        assert!(
            index.lookup(&[1, 2, 3, 4, 5], None).is_some(),
            "the shallower ancestor is still published"
        );
        let (entry, matched) = index.lookup(&[1, 2, 3, 4, 5], None).unwrap();
        assert_eq!(matched, 2);
        assert_eq!(entry.blob_path, "ancestor");
    }

    #[test]
    fn quarantine_candidates_and_remove_quarantined_bound_tombstones() {
        // QA ISSUE-002 follow-up: quarantine tombstones are diagnostic state
        // and must be reapable, not an unbounded map. Candidates are coldest
        // first and never include published entries; remove_quarantined retires
        // the exact stone (never a published entry or a republish) and prunes
        // the dead branch.
        let index = PrefixIndex::new();
        let _ = publish_entry(&index, &[1, 2, 3], "stone-a");
        let _ = publish_entry(&index, &[4, 5, 6], "stone-b");
        let _ = publish_entry(&index, &[7, 8, 9], "healthy");
        for name in ["stone-a", "stone-b"] {
            let tokens: &[u32] = if name == "stone-a" {
                &[1, 2, 3]
            } else {
                &[4, 5, 6]
            };
            let (entry, _) = index.lookup(tokens, None).unwrap();
            assert!(index.quarantine_blob(&entry.blob_path, entry.generation, tokens, || {}));
        }
        // Candidates are the EXCESS over the cap, coldest first — never a
        // published entry. Two stones at cap 10: no excess, nothing to reap.
        assert_eq!(index.quarantined_count(), 2, "quarantine_blob marked both");
        assert!(
            index.quarantine_candidates(10).is_empty(),
            "under-cap tombstones are keepers, not candidates"
        );
        // Cap 1: excess 1 — the COLDEST stone is the reaper's candidate.
        let cands = index.quarantine_candidates(1);
        assert_eq!(cands.len(), 1);
        assert!(cands.iter().all(|(_, name, _)| *name != "healthy"));
        // Removing by exact identity retires the stone; a PUBLISHED identity
        // can never be removed through this path.
        let (tokens, name, gen) = &cands[0];
        assert!(index.remove_quarantined(tokens, name, *gen));
        assert!(
            !index.remove_quarantined(tokens, name, *gen),
            "already gone"
        );
        let (entry, _) = index.lookup(&[7, 8, 9], None).unwrap();
        assert!(
            !index.remove_quarantined(&[7, 8, 9], &entry.blob_path, entry.generation),
            "remove_quarantined must never remove a published entry"
        );
        assert_eq!(index.quarantined_count(), 1);
        assert_eq!(index.published_count(), 1);
        // The pruned branch is gone: the removed stone's tokens no longer walk.
        let removed_tokens = cands[0].0.clone();
        assert!(index.lookup(&removed_tokens, None).is_none());
    }

    #[test]
    fn quarantine_blob_retires_only_the_exact_generation() {
        // A late failure must retire the publication it used, not a fresh
        // republish that reused the same deterministic blob name. Publish gen 1,
        // then republish the SAME name (gen 2); retiring gen 1 must be a no-op,
        // and retiring gen 2 must quarantine.
        let index = PrefixIndex::new();
        let _ = publish_entry(&index, &[1, 2, 3], "same-name");
        let gen1 = index.lookup(&[1, 2, 3], None).unwrap().0.generation;
        let _ = publish_entry(&index, &[1, 2, 3], "same-name");
        let gen2 = index.lookup(&[1, 2, 3], None).unwrap().0.generation;
        assert_ne!(gen1, gen2, "each publish bumps the generation");
        assert!(
            !index.quarantine_blob("same-name", gen1, &[1, 2, 3], || {}),
            "retiring a superseded generation must not touch the live entry"
        );
        assert!(
            index.lookup(&[1, 2, 3], None).is_some(),
            "the fresh republish is still served"
        );
        assert!(
            index.quarantine_blob("same-name", gen2, &[1, 2, 3], || {}),
            "retiring the live generation quarantines it"
        );
        assert!(
            index.lookup(&[1, 2, 3], None).is_none(),
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
        assert!(index.lookup(&[1, 2, 3], None).is_none());
    }

    #[test]
    fn quarantine_blob_descends_the_given_prefix_only() {
        // Keyed descent: a wrong prefix must not find (or touch) the entry, even
        // with the right name+generation. This is why the caller passes the
        // matched prefix; it also makes retirement O(prefix), not a full scan.
        let index = PrefixIndex::new();
        let _ = publish_entry(&index, &[1, 2, 3], "b");
        let gen = index.lookup(&[1, 2, 3], None).unwrap().0.generation;
        assert!(
            !index.quarantine_blob("b", gen, &[1, 2, 9], || {}),
            "a wrong prefix must not match the entry"
        );
        assert!(index.lookup(&[1, 2, 3], None).is_some());
        assert!(index.quarantine_blob("b", gen, &[1, 2, 3], || {}));
        assert!(index.lookup(&[1, 2, 3], None).is_none());
    }

    #[test]
    fn quarantine_callback_runs_only_on_match() {
        // The on_retired callback (blob deletion) must run only when the entry is
        // actually quarantined. A stale generation must not delete the live
        // entry's file.
        let index = PrefixIndex::new();
        let _ = publish_entry(&index, &[1, 2, 3], "b");
        let gen1 = index.lookup(&[1, 2, 3], None).unwrap().0.generation;
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
            index.lookup(&[42], None).is_none(),
            "a 1-token prefix must never match"
        );
    }

    #[test]
    fn end_divergent_request_serves_entry_t22() {
        // T22 serve rule: a request that diverges from the entry's key at the
        // key's LAST token is served (the blob covers tokens[:-1], and the
        // KV at every covered position depends only on the agreed prefix).
        // Trie-wise the walk breaks one edge short and the deepest-reached
        // node's direct child (the entry) qualifies.
        let index = PrefixIndex::new();
        let _ = publish_entry(&index, &[1, 2, 3], "turn1");
        // Divergence exactly at the key's last token (the multi-turn shape:
        // turn 2 replaced turn 1's closing bracket).
        let (entry, matched) = index.lookup(&[1, 2, 9], None).unwrap();
        assert_eq!(entry.blob_path, "turn1");
        assert_eq!(matched, 3, "matched = len(key) capped to the request");
        // A longer divergent request matches the same entry.
        let (entry, matched) = index.lookup(&[1, 2, 9, 9], None).unwrap();
        assert_eq!(entry.blob_path, "turn1");
        assert_eq!(matched, 3);
        // Walk-end one-short: a request SHORTER than the key is never served
        // by the child branch (the sidecar would reject it: prefix longer
        // than the request → scratch) — an honest miss, not a false hit.
        let index = PrefixIndex::new();
        let _ = publish_entry(&index, &[1, 2, 3, 4], "long");
        assert!(
            index.lookup(&[1, 2, 3], None).is_none(),
            "one-short walk-end request must miss: the child blob is unusable \
             for a request shorter than the key"
        );
    }

    #[test]
    fn exact_entry_not_masked_by_longer_sibling_t22() {
        // Published [1,2,3] AND [1,2,3,4]; request [1,2,3]. The longer child
        // is inadmissible (request shorter than its key), so the exact
        // walk-best entry must serve — the child must not mask it.
        let index = PrefixIndex::new();
        let _ = publish_entry(&index, &[1, 2, 3], "exact");
        let _ = publish_entry(&index, &[1, 2, 3, 4], "long");
        let (entry, matched) = index.lookup(&[1, 2, 3], None).unwrap();
        assert_eq!(entry.blob_path, "exact", "the usable exact entry serves");
        assert_eq!(matched, 3);
    }

    #[test]
    fn divergent_serve_touches_real_entry_recency_t22() {
        // The divergent-serve path must refresh the TRIE entry's recency
        // (the reaper reads eviction_candidates), not a clone's. Regression
        // for the clone-touch bug: storing into the clone left hot chain
        // bases cold and evictable.
        let index = PrefixIndex::new();
        let _ = publish_entry(&index, &[1, 2, 3], "turn1");
        let before = index.eviction_candidates()[0].last_used;
        std::thread::sleep(std::time::Duration::from_millis(15));
        let _ = index.lookup(&[1, 2, 9, 9], None);
        let after = index.eviction_candidates()[0].last_used;
        assert!(
            after > before,
            "a divergent serve must advance the real entry's last_used"
        );
    }

    #[test]
    fn mid_prefix_divergence_never_serves_t22() {
        // The serve rule is END-ANCHORED: divergence before the key's last
        // token means KV past the LCP is NOT identical for the request —
        // the trie geometry enforces it (the entry is not a direct child of
        // the deepest-reached node) and these pin it.
        let index = PrefixIndex::new();
        let _ = publish_entry(&index, &[1, 2, 3], "turn1");
        assert!(
            index.lookup(&[1, 9, 3], None).is_none(),
            "divergence at position 1 must not serve a key of length 3"
        );
        assert!(
            index.lookup(&[9, 2, 3], None).is_none(),
            "divergence at position 0 must not serve"
        );
        // A two-token entry serves a divergent second token (agreement 1 =
        // len(key)-1) but nothing below two agreed tokens.
        let index = PrefixIndex::new();
        let _ = publish_entry(&index, &[1, 2], "short");
        let (_, matched) = index.lookup(&[1, 9], None).unwrap();
        assert_eq!(matched, 2);
        assert!(
            index.lookup(&[9, 9], None).is_none(),
            "a request agreeing on zero tokens must not serve (root children \
             cannot hold entries: publish refuses <2-token keys)"
        );
    }

    #[test]
    fn one_token_request_never_serves_t22() {
        // A 1-token request has an empty covered prefix (matched would cap
        // to 1 < 2): never a hit/partial, mirroring publish's 2-token floor.
        let index = PrefixIndex::new();
        let _ = publish_entry(&index, &[1, 2], "short");
        assert!(index.lookup(&[1], None).is_none());
    }

    #[test]
    fn quarantine_keys_by_the_entry_key_after_divergent_serve_t22() {
        // Under end-anchored divergence the REQUEST's tokens do not spell
        // the entry's key. Retirement must use the entry's TRUE key (now
        // carried on IndexEntry): the entry stays servable to divergent
        // requests until quarantined by that key, and a divergent lookup
        // finds nothing after quarantine.
        let index = PrefixIndex::new();
        let _ = publish_entry(&index, &[1, 2, 3], "turn1");
        let (entry, _) = index.lookup(&[1, 2, 9], None).unwrap();
        assert_eq!(entry.key, vec![1, 2, 3], "the entry reports its true key");
        assert!(index.quarantine_blob("turn1", entry.generation, &entry.key, || {}));
        assert!(
            index.lookup(&[1, 2, 9], None).is_none(),
            "a quarantined entry must not serve divergent requests either"
        );
    }

    #[test]
    fn remove_published_prunes_empty_branches() {
        // Eviction removes the entry but must also reclaim the trie nodes:
        // unique-prompt churn otherwise retains every node forever. After a
        // removal, the removed key is gone for lookups AND a re-publish of
        // the same key rebuilds its branch cleanly.
        let index = PrefixIndex::new();
        let gen = publish_entry(&index, &[7, 7, 7], "solo");
        assert!(index.remove_published(&[7, 7, 7], "solo", gen));
        assert!(
            index.lookup(&[7, 7, 7], None).is_none(),
            "removed entry must not serve"
        );
        assert!(
            index.eviction_candidates().is_empty(),
            "pruned branch must not linger as a candidate"
        );
        // Re-publish rebuilds the branch.
        let _ = publish_entry(&index, &[7, 7, 7], "solo2");
        assert!(index.lookup(&[7, 7, 7], None).is_some());
    }

    #[test]
    fn remove_published_keeps_shared_and_anchored_branches() {
        // The prune must never eat a branch anything still uses: a shared
        // fork (sibling child), a walked-off prefix entry (any state), and a
        // deeper chain anchored at the removed node all survive.
        let index = PrefixIndex::new();
        // Shared fork: [1,2,3] and [1,2,9]; removing [1,2,3] keeps [1,2].
        let _ = publish_entry(&index, &[1, 2, 3], "a");
        let _ = publish_entry(&index, &[1, 2, 9], "b");
        let (e, _) = index.lookup(&[1, 2, 3], None).unwrap();
        let gen_a = e.generation;
        assert!(index.remove_published(&[1, 2, 3], "a", gen_a));
        assert!(
            index.lookup(&[1, 2, 9], None).is_some(),
            "fork sibling must survive"
        );
        // The removed entry is gone — but its sibling [1,2,9] legitimately
        // serves request [1,2,3]: they agree through len(key)-1 = 2 and
        // diverge exactly at the key's last token (the T22 serve rule).
        let (entry, _) = index.lookup(&[1, 2, 3], None).unwrap();
        assert_eq!(entry.blob_path, "b", "sibling serves the divergent request");
        // Chain base: [3,3] published, [3,3,4,4] published on top; removing
        // the DEEP one keeps the base; removing the base then must keep the
        // deep chain's nodes (it has children — not prunable).
        let index2 = PrefixIndex::new();
        let _ = publish_entry(&index2, &[3, 3], "base");
        let _ = publish_entry(&index2, &[3, 3, 4, 4], "deep");
        let (e, _) = index2.lookup(&[3, 3, 4, 4], None).unwrap();
        let gen_deep = e.generation;
        assert!(index2.remove_published(&[3, 3, 4, 4], "deep", gen_deep));
        assert!(
            index2.lookup(&[3, 3], None).is_some(),
            "chain base must survive"
        );
        // Base removal with no deeper chain: branch fully reclaimed.
        let (e, _) = index2.lookup(&[3, 3], None).unwrap();
        let gen_base = e.generation;
        assert!(index2.remove_published(&[3, 3], "base", gen_base));
        assert!(
            index2.eviction_candidates().is_empty(),
            "whole chain reclaimed"
        );
        // Quarantined entry on the path anchors its branch (diagnostics).
        let index3 = PrefixIndex::new();
        let _ = publish_entry(&index3, &[5, 5], "live");
        let _ = publish_entry(&index3, &[5, 5, 6, 6], "dead");
        let (e, _) = index3.lookup(&[5, 5, 6, 6], None).unwrap();
        let dead = e;
        assert!(index3.quarantine_blob("dead", dead.generation, &dead.key, || {}));
        let (e, _) = index3.lookup(&[5, 5], None).unwrap();
        let live = e;
        assert!(index3.remove_published(&[5, 5], "live", live.generation));
        assert!(
            index3.quarantined_count() == 1,
            "the path through a quarantined entry must survive (any-state entry anchors)"
        );
    }

    #[test]
    fn fingerprint_gates_serving_and_recency_t22() {
        // Adversarial F4: a wrong-model request must not serve a blob (the
        // daemon classifies the mismatch as Miss anyway) and must not
        // refresh the entry's anchor recency — otherwise a wrong-model loop
        // keeps every cold blob eviction-proof for the anchor window.
        let index = PrefixIndex::new();
        let _ = publish_entry(&index, &[1, 2, 3], "turn1");
        // Wrong fingerprint: no serve at any shape.
        assert!(index
            .lookup(&[1, 2, 3], Some(&fingerprint_other()))
            .is_none());
        assert!(index
            .lookup(&[1, 2, 9], Some(&fingerprint_other()))
            .is_none());
        // And no recency refresh from the mismatched traffic.
        let before = index.eviction_candidates()[0].last_used;
        std::thread::sleep(std::time::Duration::from_millis(15));
        let _ = index.lookup(&[1, 2, 9], Some(&fingerprint_other()));
        assert_eq!(
            index.eviction_candidates()[0].last_used,
            before,
            "fingerprint-mismatched lookups must not refresh recency"
        );
        // Matching fingerprint serves the exact and the divergent shape.
        let (entry, matched) = index.lookup(&[1, 2, 9], Some(&fingerprint())).unwrap();
        assert_eq!(entry.blob_path, "turn1");
        assert_eq!(matched, 3);
        // A fingerprint-mismatched child must not mask a fingerprint-matched
        // exact entry at the walk-end node.
        let index2 = PrefixIndex::new();
        let _ = publish_entry(&index2, &[1, 2, 3], "mine");
        let generation = index2.reserve_generation();
        assert!(index2.publish(
            &[1, 2, 3, 4],
            CheckpointMeta {
                fingerprint: fingerprint_other(),
                token_count: 4,
                tokens: vec![1, 2, 3, 4],
                format_version: 1,
            },
            "theirs".into(),
            generation,
            |_| {}
        ));
        let (entry, _) = index2.lookup(&[1, 2, 3], Some(&fingerprint())).unwrap();
        assert_eq!(
            entry.blob_path, "mine",
            "fp-matched exact entry must not be masked by an fp-mismatched child"
        );
    }

    #[test]
    fn anchor_marking_matches_prefix_scan() {
        // The bottom-up descendant-marking must agree with the definition:
        // an entry is an anchor iff a LONGER published entry extends its
        // prefix. Pins the eviction_candidates rewrite.
        let index = PrefixIndex::new();
        let _ = publish_entry(&index, &[1, 2], "base");
        let _ = publish_entry(&index, &[1, 2, 3, 4], "deep");
        let _ = publish_entry(&index, &[8, 8], "lonely");
        let cands = index.eviction_candidates();
        let by_path: std::collections::HashMap<&str, &crate::index::IndexCandidate> =
            cands.iter().map(|c| (c.blob_path.as_str(), c)).collect();
        assert!(by_path["base"].is_anchor, "extended base is an anchor");
        assert!(!by_path["deep"].is_anchor, "leaf chain tip is not");
        assert!(!by_path["lonely"].is_anchor, "unrelated singleton is not");
        assert_eq!(
            by_path["deep"].tokens,
            vec![1, 2, 3, 4],
            "materialized path"
        );
    }

    #[test]
    fn exact_depth_match_still_dominates_t22() {
        // Pre-T22 behavior is preserved verbatim when the request contains
        // the key exactly, and a deeper walked entry still beats the
        // hottest-child candidate (coverage grows with depth).
        let index = PrefixIndex::new();
        let _ = publish_entry(&index, &[1, 2], "base");
        let _ = publish_entry(&index, &[1, 2, 3], "deep");
        let (entry, matched) = index.lookup(&[1, 2, 3, 4], None).unwrap();
        assert_eq!(entry.blob_path, "deep");
        assert_eq!(matched, 3);
        // And the old shortest-request guard: a request that is a strict
        // prefix must not match a longer entry (the child branch cannot
        // serve it — see the one-short walk-end rule in
        // end_divergent_request_serves_entry_t22). A walk-end EXACT match
        // (request == key) still serves through the walk itself.
        let index = PrefixIndex::new();
        let _ = publish_entry(&index, &[1, 2], "base");
        let _ = publish_entry(&index, &[1, 2, 3], "deep");
        let (entry, matched) = index.lookup(&[1, 2, 3], None).unwrap();
        assert_eq!(entry.blob_path, "deep", "exact walk-end match serves");
        assert_eq!(matched, 3);
        let index = PrefixIndex::new();
        let _ = publish_entry(&index, &[1, 2, 3, 4, 5], "long");
        assert!(
            index.lookup(&[1, 2], None).is_none(),
            "short request must not match a longer entry"
        );
        // And the overlapping case: published [1,2], request [1,2,3] matches at 2.
        let index = PrefixIndex::new();
        let _ = publish_entry(&index, &[1, 2], "short");
        let (_, matched) = index.lookup(&[1, 2, 3], None).unwrap();
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
        let (entry, matched) = index.lookup(&[1, 2, 3], None).expect("recovered lookup");
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
        let (entry, matched) = index.lookup(&[1, 2, 3, 4], None).unwrap();
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
        let _ = index.lookup(&[1, 2, 3, 9], None);
        let after = index.eviction_candidates()[0].last_used;
        assert!(
            after > before,
            "lookup must advance the entry's last_used instant"
        );
    }
}
