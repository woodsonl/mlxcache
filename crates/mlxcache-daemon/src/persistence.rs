//! Checkpoint persistence: atomic publish (write-temp-then-rename, R1-3).
//!
//! ENOSPC rescue (error registry): catch the write error, log path+bytes,
//! fail the warming path only — real requests proceed uncached.

use mlxcache_core::contract::CheckpointMeta;
#[cfg(test)]
use mlxcache_core::contract::ContractError;
use sha2::{Digest, Sha256};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Clone)]
pub struct Persistence {
    pub blob_dir: PathBuf,
}

/// Process-wide monotonic counter to make temp file names unique across
/// concurrent publishes within one process.
fn next_temp_seq() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    SEQ.fetch_add(1, Ordering::Relaxed)
}

#[derive(Debug, thiserror::Error)]
pub enum PersistError {
    #[error("disk full or unwritable at {path}: {source}")]
    DiskFull {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("checkpoint corrupt: {reason}")]
    Corrupt { reason: String },
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

impl Persistence {
    pub fn new(blob_dir: impl Into<PathBuf>) -> Result<Self, PersistError> {
        let dir = blob_dir.into();
        fs::create_dir_all(&dir)?;
        Ok(Self { blob_dir: dir })
    }

    /// Atomic publish: write to `<final>.tmp`, fsync, rename. A crash mid-write
    /// leaves only the temp file, never a partial final blob (R1-3).
    ///
    /// The final name embeds `generation`, so every publication is an immutable
    /// file: a republish of the same prefix writes a NEW name rather than
    /// overwriting an earlier generation's file. That is what makes retirement
    /// safe — a delayed quarantine of generation N unlinks only N's file and can
    /// never delete a replacement the daemon just wrote.
    pub fn publish_atomic(
        &self,
        prefix_hash: u128,
        generation: u64,
        mut meta: CheckpointMeta,
        payload: &[u8],
    ) -> Result<PathBuf, PersistError> {
        // D1 integrity (format_version 2), stamped at the single publish
        // choke point so every blob carries it: a framing-valid bit-rot flip
        // is otherwise undetectable — the wire checks catch bad headers and
        // bad framing, not wrong bytes. The version bump marks the new
        // on-disk contract; legacy version-1 blobs stay loadable, unverified.
        meta.format_version = 2;
        meta.payload_sha256 = Some(format!("{:x}", Sha256::digest(payload)));
        let final_path = self
            .blob_dir
            .join(format!("{:032x}-{:016x}.ckpt", prefix_hash, generation));
        // Unique temp name: two concurrent publishers for the same hash must not
        // interleave writes into one temp file (that yields a corrupt blob which
        // the rename then publishes as if complete). pid + counter keeps the
        // name unique per write; the `.ckpt.tmp` suffix family is still excluded
        // by list_blobs.
        let unique = format!(
            "{:032x}-{:016x}.{}.{}.ckpt.tmp",
            prefix_hash,
            generation,
            std::process::id(),
            next_temp_seq()
        );
        let tmp_path = self.blob_dir.join(unique);

        let header = serde_json::to_vec(&meta).map_err(|e| PersistError::Corrupt {
            reason: format!("meta serialize: {e}"),
        })?;
        // The on-disk header length is a u32; reject rather than silently
        // truncate (a header this large is pathological, but truncation would
        // corrupt every later read).
        let header_len_u32 = u32::try_from(header.len()).map_err(|_| PersistError::Corrupt {
            reason: format!("checkpoint header too large: {} bytes", header.len()),
        })?;
        let header_len = header_len_u32.to_le_bytes();

        let write_result = (|| -> std::io::Result<()> {
            let mut f = fs::File::create(&tmp_path)?;
            f.write_all(&header_len)?;
            f.write_all(&header)?;
            f.write_all(payload)?;
            f.sync_all()?;
            Ok(())
        })();

        if let Err(source) = write_result {
            // Best-effort temp cleanup; the final blob is untouched.
            let _ = fs::remove_file(&tmp_path);
            // ENOSPC and friends: surface as DiskFull with the path (registry row).
            return Err(PersistError::DiskFull {
                path: tmp_path,
                source,
            });
        }

        if let Err(source) = fs::rename(&tmp_path, &final_path) {
            // A failed rename (ENOSPC on the directory, EACCES, ...) must not
            // leak the temp file; classify it as DiskFull too so the ENOSPC
            // rescue path handles it rather than a generic Io.
            let _ = fs::remove_file(&tmp_path);
            return Err(PersistError::DiskFull {
                path: final_path,
                source,
            });
        }
        Ok(final_path)
    }

    /// Load a blob: parse header, return (meta, payload). Corruption or version
    /// mismatch returns Corrupt — the caller quarantines (R1-1).
    pub fn load(&self, blob_path: &Path) -> Result<(CheckpointMeta, Vec<u8>), PersistError> {
        let bytes = fs::read(blob_path)?;
        if bytes.len() < 4 {
            return Err(PersistError::Corrupt {
                reason: "truncated header length".into(),
            });
        }
        let header_len = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
        if 4 + header_len > bytes.len() {
            return Err(PersistError::Corrupt {
                reason: "header length exceeds blob".into(),
            });
        }
        let meta: CheckpointMeta =
            serde_json::from_slice(&bytes[4..4 + header_len]).map_err(|e| {
                PersistError::Corrupt {
                    reason: format!("meta parse: {e}"),
                }
            })?;
        // Version policy: 1 = legacy (no digest field exists — structural
        // checks only), 2 = current (digest required and verified here).
        // Anything else is a blob a daemon we cannot reason about wrote:
        // refuse loudly rather than misinterpret its layout.
        match meta.format_version {
            1 => {}
            2 => {
                let expected =
                    meta.payload_sha256
                        .as_deref()
                        .ok_or_else(|| PersistError::Corrupt {
                            reason: "format version 2 requires payload_sha256".into(),
                        })?;
                let actual = format!("{:x}", Sha256::digest(&bytes[4 + header_len..]));
                if !actual.eq_ignore_ascii_case(expected) {
                    return Err(PersistError::Corrupt {
                        reason: format!(
                            "payload digest mismatch (expected {expected}, got {actual})"
                        ),
                    });
                }
            }
            v => {
                return Err(PersistError::Corrupt {
                    reason: format!("format version {v}"),
                })
            }
        }
        Ok((meta, bytes[4 + header_len..].to_vec()))
    }

    /// Delete a published blob file on retirement (R1-1). Best-effort: a missing
    /// file is fine, and an I/O failure is logged by the caller, not fatal. This
    /// makes retirement durable — a blob left on disk would be re-indexed by the
    /// next startup rebuild and resurrect the poison.
    pub fn remove(&self, blob_name: &str) -> Result<(), PersistError> {
        match fs::remove_file(self.blob_dir.join(blob_name)) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(PersistError::Io(e)),
        }
    }

    /// Rebuild support (index corruption rescue): list all published blobs.
    /// Each blob's header carries its own token prefix (CheckpointMeta.tokens),
    /// so the index is rebuilt from blob metadata alone — blobs are the source
    /// of truth (see Orchestrator::rebuild_from_disk).
    pub fn list_blobs(&self) -> Result<Vec<PathBuf>, PersistError> {
        let mut out = Vec::new();
        for entry in fs::read_dir(&self.blob_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().map(|e| e == "ckpt").unwrap_or(false) {
                out.push(path);
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orchestrator::test_support::fp;

    fn meta(n: u64) -> CheckpointMeta {
        CheckpointMeta {
            fingerprint: fp("m"),
            token_count: n,
            tokens: vec![1, 2, 3],
            format_version: 1,
            payload_sha256: None,
            engine_id: None,
            granularity: None,
        }
    }

    fn tmpdir() -> tempfile::TempDir {
        tempfile::tempdir().unwrap()
    }

    #[test]
    fn atomic_publish_and_load() {
        let dir = tmpdir();
        let p = Persistence::new(dir.path()).unwrap();
        let path = p
            .publish_atomic(0xdeadbeef, 1, meta(10), b"kvbytes")
            .unwrap();
        assert!(path.exists());
        assert!(!path.to_string_lossy().ends_with(".tmp"));
        let (m, payload) = p.load(&path).unwrap();
        assert_eq!(m.token_count, 10);
        assert_eq!(payload, b"kvbytes");
    }

    #[test]
    fn remove_is_durable_and_idempotent() {
        // Retirement deletes the file so a rebuild cannot resurrect the poison.
        // Removing an already-missing file is fine (the request that retired it
        // may race another).
        let dir = tmpdir();
        let p = Persistence::new(dir.path()).unwrap();
        let path = p.publish_atomic(0x1, 1, meta(10), b"kv").unwrap();
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        p.remove(&name).unwrap();
        assert!(!path.exists(), "retired blob file must be gone");
        p.remove(&name).unwrap(); // idempotent
    }

    #[test]
    fn load_rejects_corrupt() {
        let dir = tmpdir();
        let p = Persistence::new(dir.path()).unwrap();
        let bad = dir.path().join("bad.ckpt");
        fs::write(&bad, b"xy").unwrap(); // < 4 bytes
        assert!(matches!(p.load(&bad), Err(PersistError::Corrupt { .. })));
    }

    #[test]
    fn publish_stamps_v2_with_verified_digest() {
        // D1: publish is the single choke point that stamps format_version 2
        // + the payload digest; load verifies it. Legacy v1 blobs (no digest)
        // stay loadable, unverified.
        let dir = tmpdir();
        let p = Persistence::new(dir.path()).unwrap();
        let path = p.publish_atomic(0x2, 1, meta(10), b"kv-payload").unwrap();
        let (m, _payload) = p.load(&path).unwrap();
        assert_eq!(m.format_version, 2);
        assert_eq!(
            m.payload_sha256.as_deref(),
            Some(&format!("{:x}", Sha256::digest(b"kv-payload"))[..]),
            "digest must be sha256 of exactly the payload bytes"
        );
    }

    #[test]
    fn tampered_payload_is_rejected_by_digest() {
        // Framing-valid corruption (bit rot, a torn rewrite that kept the
        // header intact) is exactly what the structural checks cannot catch.
        let dir = tmpdir();
        let p = Persistence::new(dir.path()).unwrap();
        let path = p.publish_atomic(0x2, 1, meta(10), b"kv-payload").unwrap();
        let mut bytes = fs::read(&path).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff;
        fs::write(&path, &bytes).unwrap();
        match p.load(&path) {
            Err(PersistError::Corrupt { reason }) => {
                assert!(reason.contains("digest mismatch"), "reason: {reason}");
            }
            other => panic!("expected Corrupt, got {other:?}"),
        }
    }

    #[test]
    fn v2_without_digest_and_legacy_v1_are_classified_apart() {
        // Hand-crafted wire blobs (publish always stamps v2, so the skew
        // paths must be built by hand): a v2 header without a digest is
        // corrupt (the field is part of the v2 contract); a legacy v1 blob
        // loads unverified (the boot sweep must not strand pre-D1 stores).
        let dir = tmpdir();
        let p = Persistence::new(dir.path()).unwrap();
        let mut m2 = meta(10);
        m2.format_version = 2;
        let header = serde_json::to_vec(&m2).unwrap();
        let mut wire = (header.len() as u32).to_le_bytes().to_vec();
        wire.extend_from_slice(&header);
        wire.extend_from_slice(b"kv");
        let v2_path = dir.path().join("v2-nodigest.ckpt");
        fs::write(&v2_path, &wire).unwrap();
        match p.load(&v2_path) {
            Err(PersistError::Corrupt { reason }) => {
                assert!(
                    reason.contains("requires payload_sha256"),
                    "reason: {reason}"
                );
            }
            other => panic!("expected Corrupt, got {other:?}"),
        }

        let mut m1 = meta(10);
        m1.format_version = 1;
        let header = serde_json::to_vec(&m1).unwrap();
        let mut wire = (header.len() as u32).to_le_bytes().to_vec();
        wire.extend_from_slice(&header);
        wire.extend_from_slice(b"kv");
        let v1_path = dir.path().join("v1-legacy.ckpt");
        fs::write(&v1_path, &wire).unwrap();
        assert!(p.load(&v1_path).is_ok(), "legacy v1 must stay loadable");
    }

    #[test]
    fn list_blobs_finds_published_only() {
        let dir = tmpdir();
        let p = Persistence::new(dir.path()).unwrap();
        p.publish_atomic(0x1, 1, meta(1), b"a").unwrap();
        p.publish_atomic(0x2, 1, meta(2), b"b").unwrap();
        fs::write(dir.path().join("stray.txt"), b"nope").unwrap();
        assert_eq!(p.list_blobs().unwrap().len(), 2);
    }

    #[test]
    fn concurrent_publish_same_hash_uses_unique_temps() {
        // Two publishers for the same hash must not share a temp file; each
        // rename must land a complete blob. Run them on threads to exercise the
        // real concurrency, then verify the final blob loads clean.
        let dir = tmpdir();
        let p = std::sync::Arc::new(Persistence::new(dir.path()).unwrap());
        let mut handles = Vec::new();
        for i in 0..8u8 {
            let p = p.clone();
            handles.push(std::thread::spawn(move || {
                p.publish_atomic(0x55, 1, meta(3), &[i; 4096]).unwrap();
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        // Exactly one final blob; no .tmp litter; it loads without corruption.
        assert_eq!(p.list_blobs().unwrap().len(), 1);
        let stray_tmp = fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().to_string_lossy().ends_with(".tmp"))
            .count();
        assert_eq!(stray_tmp, 0, "temp files must be renamed away, not leaked");
        let blob = p.list_blobs().unwrap().pop().unwrap();
        let (_m, payload) = p.load(&blob).unwrap();
        assert_eq!(payload.len(), 4096);
        assert!(
            payload.iter().all(|b| *b == payload[0]),
            "interleaved writes would mix payload bytes"
        );
    }

    // ENOSPC simulation: write to a full filesystem is hard to simulate
    // portably; we verify the DiskFull error path by pointing at an
    // unwritable directory.
    #[test]
    fn unwritable_dir_maps_to_diskfull() {
        let dir = tmpdir();
        let p = Persistence::new(dir.path()).unwrap();
        // Make the target dir read-only
        let mut perms = fs::metadata(dir.path()).unwrap().permissions();
        #[allow(clippy::permissions_set_readonly_false)]
        perms.set_readonly(true);
        fs::set_permissions(dir.path(), perms).unwrap();
        let result = p.publish_atomic(0x3, 1, meta(1), b"x");
        // Restore so tempdir cleanup works (explicit 0o755, not set_readonly(false))
        let mut perms = fs::metadata(dir.path()).unwrap().permissions();
        use std::os::unix::fs::PermissionsExt;
        perms.set_mode(0o755);
        fs::set_permissions(dir.path(), perms).unwrap();
        assert!(
            matches!(result, Err(PersistError::DiskFull { .. })),
            "unwritable dir must map to DiskFull (ENOSPC rescue path)"
        );
    }

    // ContractError integration smoke: version mismatch surfaces as Corrupt.
    #[test]
    fn contract_error_display() {
        let e = ContractError::VersionMismatch { got: 2, want: 1 };
        assert!(e.to_string().contains("incompatible"));
    }

    #[test]
    fn foreign_format_version_blob_is_rejected_with_reason() {
        // Wire-format skew (api-contract 2026-10-04): a blob written by a
        // NEWER daemon/sidecar carries a higher format_version (2 is CURRENT:
        // the D1 digest contract; use 3 here as a genuinely foreign one).
        // `load` must refuse it loudly (Corrupt, reason naming the version)
        // so a mixed rollout retires the foreign blob instead of mis-reading
        // it — the on-disk layout beyond the header is not ours to interpret.
        let dir = tempfile::tempdir().unwrap();
        let p = Persistence::new(dir.path()).unwrap();
        let mut meta = meta(6);
        meta.format_version = 3;
        let header = serde_json::to_vec(&meta).unwrap();
        let mut wire = (header.len() as u32).to_le_bytes().to_vec();
        wire.extend_from_slice(&header);
        wire.extend_from_slice(b"payload-bytes");
        let blob = dir.path().join("foreign.ckpt");
        fs::write(&blob, &wire).unwrap();

        let e = p.load(&blob).unwrap_err();
        match e {
            PersistError::Corrupt { reason } => {
                assert!(reason.contains("format version 3"), "reason: {reason}");
            }
            other => panic!("expected Corrupt, got {other:?}"),
        }
    }
}
