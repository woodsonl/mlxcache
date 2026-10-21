//! Engine-agnostic cache contract.
//!
//! Binding rules (from docs/contract-spec.md):
//! - R1-1: fingerprint/tokenizer/dtype mismatch => miss + quarantine, never serve incompatible KV.
//! - R1-2: the adapter tokenizes; the daemon keys checkpoints by adapter-reported token-ID prefix.
//! - R1-3: single-flight on identical uncached prefixes; atomic publish (write-temp-then-rename).
//! - R1-4: daemon restart drops in-flight streams; persisted checkpoints survive.

use serde::{Deserialize, Serialize};

/// The engine that owns the default key domain (connector protocol §4).
/// It contributes ZERO fold bytes, so every pre-namespacing key is
/// byte-identical to the current one.
pub const DEFAULT_ENGINE: &str = "mlx-lm";

/// The `CheckpointMeta::engine_id` value that records `engine`: `None` for the
/// default engine. Same zero-byte rule as the key fold, applied to the header:
/// a blob from the default engine is byte-identical to a pre-namespacing blob,
/// and every reader already resolves an absent engine_id to the default.
pub fn engine_id_meta(engine: &str) -> Option<String> {
    (engine != DEFAULT_ENGINE).then(|| engine.to_string())
}

/// Resolve an engine id (request, entry, or header) to its domain: an absent
/// id IS the default engine. Every adoption, sibling-scan, and rebuild
/// grouping must compare domains, never raw ids — a `None` entry and a
/// `"mlx-lm"` request are the same engine (legacy stores rely on this).
pub fn engine_domain(engine_id: Option<&str>) -> &str {
    engine_id.unwrap_or(DEFAULT_ENGINE)
}

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
    /// Signature of the model's CACHE COMPOSITION: the ordered tuple of cache
    /// entry class names the engine builds for this model (dense KV stack,
    /// CacheList of linear-attention state, ArraysCache of SSM state, ...).
    ///
    /// Distinct architectures can share a model id, tokenizer, dtype and
    /// quantization tier while their payloads are shaped differently — a
    /// dense stack's tensors cannot be loaded as a hybrid model's recurrent
    /// state, and the mismatch is not detectable from the bytes. Pinning the
    /// composition in the fingerprint makes cross-architecture serving
    /// impossible by construction instead of by payload inspection.
    ///
    /// `#[serde(default)]` because pre-B1 blobs have no such field: they read
    /// as `""`, the "composition unknown" value (see `crate::policy` for the
    /// serve rule). An empty layout contributes ZERO fold bytes, so every
    /// existing store keeps byte-identical keys.
    ///
    /// `skip_serializing_if` for the same reason: an empty layout means
    /// "unrecorded", and a header that says nothing about composition is the
    /// pre-B1 header byte-for-byte. Emitting `"cache_layout":""` would make
    /// every publish differ from the pinned cross-language wire golden while
    /// carrying no information.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub cache_layout: String,
    /// Digest of the request's NON-TEXT content — images today, any future
    /// modality by the same rule — so that two requests with identical token
    /// prefixes but different pixels can never share a checkpoint.
    ///
    /// The index keys on tokens alone, and a multimodal prompt renders its
    /// media as ordinary tokens (`<|image_pad|>` is a plain id, not a
    /// placeholder the tokenizer keeps distinct per image). Two different
    /// images therefore produce the SAME token sequence while their KV state
    /// differs at every position the media occupies. Keying on tokens alone
    /// would serve one image's KV to the other: silent, plausible-looking,
    /// wrong output.
    ///
    /// Empty means "this prefix carries no non-text content", which is the
    /// text-only case and the only case any checkpoint written before this
    /// field existed can represent. Like `cache_layout`, an empty digest
    /// contributes ZERO fold bytes, so text-only keys stay byte-identical,
    /// and the serve rule in `crate::policy` compares digests exactly — an
    /// unrecorded digest never matches a recorded one.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub content_digest: String,
    /// Named KV persistence tier (the TurboQuant tiers, e.g. `tq-k3v4-g64`).
    ///
    /// The scalar `kv_bits`/`kv_group_size` pair cannot express the TQ tiers:
    /// they are asymmetric (distinct key/value bit-widths) and their payload
    /// is a codec-specific frame, not mlx-lm's quantized cache layout. The
    /// tier label pins the format the same way the scalar pair pins q8 — two
    /// tiers never cross-serve, and a TQ blob is unreachable from the
    /// f16/q8 identities.
    ///
    /// Empty means "no named tier" (f16 or q8 via `kv_bits`), which is every
    /// blob written before tiers existed. The same zero-byte rule as the
    /// other defaulted fingerprint axes: empty contributes nothing to the
    /// fold, so pre-existing keys are unchanged; non-empty folds as one more
    /// length-prefixed field after `content_digest`, the last identity axis.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub kv_tier: String,
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
    /// sha256 (hex) of the payload bytes — D1 integrity (format_version 2).
    /// Stamped by the daemon at the single publish choke point, verified on
    /// the daemon's boot sweep (full read) and by the sidecar at first serve
    /// (once per blob per process). None on legacy version-1 blobs, which
    /// stay loadable but unverified.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub payload_sha256: Option<String>,
    /// Engine identity for namespaced keys (connector protocol §4): the
    /// engine family that produced and must consume this payload. Absent
    /// means the default engine `"mlx-lm"` — which contributes ZERO bytes
    /// to the key fold, so legacy stores keep their keys byte-identical.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engine_id: Option<String>,
    /// Resume granularity (connector protocol §3.3): 0/absent = ANY_PREFIX
    /// (radix, partial hits), 1 = WHOLE_CONTEXT (exact full-prompt match
    /// only). Matching semantics are enforced by the store's index, not the
    /// format; this field is the declaration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub granularity: Option<u8>,
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
