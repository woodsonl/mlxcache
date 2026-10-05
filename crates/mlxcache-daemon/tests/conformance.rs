//! Connector-protocol FORMAT conformance (B0.3) — the Rust half. The same
//! golden blob SHA is asserted by sidecar/tests/test_conformance.py: both
//! writers producing byte-identical files IS the cross-language round-trip
//! (protocol §9). Store-semantics rows (granularity matching, budget,
//! concurrency) land with the L1 client in B1.1.

use mlxcache_core::contract::CheckpointMeta;
use mlxcache_daemon::persistence::Persistence;
use sha2::{Digest, Sha256};

fn golden_meta() -> CheckpointMeta {
    CheckpointMeta {
        fingerprint: mlxcache_core::contract::ModelFingerprint {
            model_id: "conformance-model".into(),
            tokenizer_hash: "conf-tok".into(),
            kv_dtype: "f16".into(),
            kv_layout_version: 1,
            kv_bits: 0,
            kv_group_size: 0,
        },
        token_count: 4,
        tokens: vec![1, 2, 3, 4],
        // publish_atomic stamps format_version 2 + the payload digest; the
        // Python writer crafts the identical final bytes explicitly.
        format_version: 1,
        payload_sha256: None,
        engine_id: None,
        granularity: None,
    }
}

/// THE cross-language pin: the Rust daemon's on-disk bytes for the
/// canonical fixture are exactly the Python codec's. If this or the Python
/// twin changes, the two bindings have diverged on the wire.
#[test]
fn golden_blob_bytes_match_the_python_writer() {
    let dir = tempfile::tempdir().unwrap();
    let p = Persistence::new(dir.path()).unwrap();
    let path = p
        .publish_atomic(0xC0FFE, 1, golden_meta(), b"conformance-payload")
        .unwrap();
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(bytes.len(), 306, "wire length drifted");
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        "5da07a82f4336e3f8d1bed1bba3decfa582ce4dc0e1d92f2e6f8dd4b465c7204",
        "the Rust writer no longer matches the cross-language golden"
    );
    // And it loads back with the stamped v2 + digest.
    let (meta, payload) = p.load(&path).unwrap();
    assert_eq!(meta.format_version, 2);
    assert_eq!(payload, b"conformance-payload");
}

/// §2 unknown-fields: an unrecognized header field is ignored, never
/// corruption (the T12 rule).
#[test]
fn unknown_header_field_is_ignored_not_corruption() {
    let dir = tempfile::tempdir().unwrap();
    let p = Persistence::new(dir.path()).unwrap();
    let path = p.publish_atomic(0x1, 1, golden_meta(), b"kv").unwrap();
    let mut bytes = std::fs::read(&path).unwrap();
    // Splice `"future_field":42,` into the header JSON right after the
    // opening brace, then fix the u32 length prefix.
    let splice = b"\"future_field\":42,".len(); // net insert (the brace is reused)
    let mut out = Vec::with_capacity(bytes.len() + splice);
    out.extend_from_slice(&bytes[..4]);
    out.extend_from_slice(b"{\"future_field\":42,");
    out.extend_from_slice(&bytes[5..]);
    let len = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize + splice;
    out[0..4].copy_from_slice(&(len as u32).to_le_bytes());
    bytes = out;
    std::fs::write(&path, &bytes).unwrap();
    let (meta, payload) = p
        .load(&path)
        .expect("unknown fields are ignorable, and the digest still matches");
    assert_eq!(payload, b"kv");
    assert_eq!(meta.token_count, 4);
}

/// §2 [B0.2] fields: engine_id + granularity survive the wire and absent
/// means default.
#[test]
fn engine_id_and_granularity_round_trip_on_the_wire() {
    let dir = tempfile::tempdir().unwrap();
    let p = Persistence::new(dir.path()).unwrap();
    let mut meta = golden_meta();
    meta.engine_id = Some("llama-cpp".into());
    meta.granularity = Some(1);
    let path = p.publish_atomic(0x2, 1, meta, b"opaque-state").unwrap();
    let (m, payload) = p.load(&path).unwrap();
    assert_eq!(
        payload, b"opaque-state",
        "payload opacity: non-safetensors bytes pass"
    );
    assert_eq!(m.engine_id.as_deref(), Some("llama-cpp"));
    assert_eq!(m.granularity, Some(1));
}
