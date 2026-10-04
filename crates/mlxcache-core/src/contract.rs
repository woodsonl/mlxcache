//! Engine-agnostic cache contract.
//!
//! Binding rules (from docs/contract-spec.md):
//! - R1-1: fingerprint/tokenizer/dtype mismatch => miss + quarantine, never serve incompatible KV.
//! - R1-2: the adapter tokenizes; the daemon keys checkpoints by adapter-reported token-ID prefix.
//! - R1-3: single-flight on identical uncached prefixes; atomic publish (write-temp-then-rename).
//! - R1-4: daemon restart drops in-flight streams; persisted checkpoints survive.

use serde::{Deserialize, Serialize};

/// Fingerprint pinned in every checkpoint's metadata. Any mismatch is a miss (R1-1).
///
/// `kv_bits`/`kv_group_size` pin the KV quantization config (T12): an 8-bit
/// group-64 cache holds different bytes than the f16 cache of the same model,
/// so the two must never cross-serve. Zero bits means "not quantized" (f16)
/// and is also what pre-T12 blobs deserialize to via the serde defaults — old
/// f16 checkpoints remain loadable and servable, still distinct from any q8
/// checkpoint. Only a parity-proven config (8/64 on the target model) may be
/// enabled via the sidecar's MLXCACHE_KV_BITS knob; 4-bit breaks greedy parity.
/// Default for test/bench constructors: an unquantized f16 fingerprint
/// (`kv_bits: 0`, `kv_group_size: 0`), matching the pre-quantization era.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelFingerprint {
    pub model_id: String,
    /// Hash of the tokenizer artifact (file bytes), not a version string.
    pub tokenizer_hash: String,
    pub kv_dtype: String,
    pub kv_layout_version: u32,
    /// KV quantization bits; 0 = f16 (unquantized). Part of R1-1 equality.
    #[serde(default)]
    pub kv_bits: u8,
    /// Group size used with `kv_bits` (0 when unquantized). Two q8 configs
    /// with different group sizes are different formats, not variants.
    #[serde(default)]
    pub kv_group_size: u32,
}

/// Metadata header for a checkpoint blob. Stored alongside the KV tensors.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointMeta {
    pub fingerprint: ModelFingerprint,
    /// Number of tokens the KV state covers.
    pub token_count: u64,
    /// The exact token-ID prefix this checkpoint covers. Persisted so a daemon
    /// restart can rebuild the radix index from disk alone (R1-4: persisted
    /// checkpoints survive). Without this the index is keyed by tokens the blob
    /// does not record, and no persisted checkpoint could ever be found again.
    #[serde(default)]
    pub tokens: Vec<u32>,
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
