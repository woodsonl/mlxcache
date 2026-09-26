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
}

impl PrefixIndex {
    pub fn new() -> Self {
        Self::default()
    }

    /// Longest published prefix match for `tokens`. Returns the entry and the
    /// number of tokens matched.
    pub fn lookup(&self, tokens: &[u32]) -> Option<(IndexEntry, usize)> {
        let root = self.root.read().expect("index lock poisoned");
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

    /// Publish an entry. Only callable for a prefix whose ancestors are consistent;
    /// the atomic blob rename must have completed BEFORE this call (R1-3).
    pub fn publish(&self, tokens: &[u32], meta: CheckpointMeta, blob_path: String) {
        let mut root = self.root.write().expect("index lock poisoned");
        let mut node: &mut Node = &mut root;
        for t in tokens {
            node = node.children.entry(*t).or_default();
        }
        node.entry = Some(IndexEntry {
            meta,
            blob_path,
            state: CheckpointState::Published,
        });
    }

    /// Quarantine an entry (R1-1): keep it visible for diagnostics but never serve it.
    pub fn quarantine(&self, tokens: &[u32]) -> bool {
        let mut root = self.root.write().expect("index lock poisoned");
        let mut node: &mut Node = &mut root;
        for t in tokens {
            match node.children.get_mut(t) {
                Some(child) => node = child,
                None => return false,
            }
        }
        match &mut node.entry {
            Some(entry) => {
                entry.state = CheckpointState::Quarantined;
                true
            }
            None => false,
        }
    }

    /// Count of published entries (for /stats).
    pub fn published_count(&self) -> usize {
        fn walk(node: &Node) -> usize {
            let mut n = usize::from(
                node.entry
                    .as_ref()
                    .map(|e| e.state == CheckpointState::Published)
                    .unwrap_or(false),
            );
            for child in node.children.values() {
                n += walk(child);
            }
            n
        }
        let root = self.root.read().expect("index lock poisoned");
        walk(&root)
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
        index.publish(&tokens, meta(), "blob-123".into());
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
        index.publish(&tokens, meta(), "blob-123".into());
        assert!(index.quarantine(&tokens));
        assert!(index.lookup(&tokens).is_none());
        assert_eq!(index.published_count(), 0);
    }

    #[test]
    fn publish_counts() {
        let index = PrefixIndex::new();
        index.publish(&[1], meta(), "a".into());
        index.publish(&[1, 2], meta(), "b".into());
        assert_eq!(index.published_count(), 2);
    }

    #[test]
    fn matched_never_exceeds_request_length() {
        // A request that is a strict prefix of a longer published entry must
        // NOT match the longer entry: the walk ends when request tokens run out,
        // so no entry beyond the request tail is ever seen. This guards the
        // invariant that `classify` can never see matched_tokens > request_tokens.
        let index = PrefixIndex::new();
        index.publish(&[1, 2, 3, 4, 5], meta(), "long".into());
        assert!(
            index.lookup(&[1, 2]).is_none(),
            "short request must not match a longer entry"
        );
        // And the overlapping case: published [1,2], request [1,2,3] matches at 2.
        let index = PrefixIndex::new();
        index.publish(&[1, 2], meta(), "short".into());
        let (_, matched) = index.lookup(&[1, 2, 3]).unwrap();
        assert_eq!(matched, 2);
        assert!(matched <= 3);
    }
}
