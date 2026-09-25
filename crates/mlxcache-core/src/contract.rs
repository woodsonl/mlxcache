//! Engine-agnostic cache contract.
//!
//! Binding rules (from docs/contract-spec.md):
//! - R1-1: fingerprint/tokenizer/dtype mismatch => miss + quarantine, never serve incompatible KV.
//! - R1-2: the adapter tokenizes; the daemon keys checkpoints by adapter-reported token-ID prefix.
//! - R1-3: single-flight on identical uncached prefixes; atomic publish (write-temp-then-rename).
//! - R1-4: daemon restart drops in-flight streams; persisted checkpoints survive.

use serde::{Deserialize, Serialize};

/// Fingerprint pinned in every checkpoint's metadata. Any mismatch is a miss (R1-1).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelFingerprint {
    pub model_id: String,
    /// Hash of the tokenizer artifact (file bytes), not a version string.
    pub tokenizer_hash: String,
    pub kv_dtype: String,
    pub kv_layout_version: u32,
}

/// Metadata header for a checkpoint blob. Stored alongside the KV tensors.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointMeta {
    pub fingerprint: ModelFingerprint,
    /// Number of tokens the KV state covers.
    pub token_count: u64,
    /// Format version of the checkpoint blob itself.
    pub format_version: u32,
}

/// The checkpoint lifecycle: absent -> in-flight -> published -> quarantined.
/// Invalid transition: published -> in-flight (never re-preflight a published prefix).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointState {
    Absent,
    InFlight,
    Published,
    Quarantined,
}

/// Errors surfaced by contract operations.
#[derive(Debug, thiserror::Error)]
pub enum ContractError {
    #[error("checkpoint format version {got} incompatible with {want}")]
    VersionMismatch { got: u32, want: u32 },
    #[error("fingerprint mismatch: {reason}")]
    FingerprintMismatch { reason: String },
    #[error("checkpoint corrupt: {reason}")]
    Corrupt { reason: String },
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}
