//! Checkpoint persistence: atomic publish (write-temp-then-rename, R1-3).
//!
//! ENOSPC rescue (error registry): catch the write error, log path+bytes,
//! fail the warming path only — real requests proceed uncached.

use crate::chunkstore::{ChunkStore, Manifest};
use mlxcache_core::contract::CheckpointMeta;
#[cfg(test)]
use mlxcache_core::contract::ContractError;
use sha2::{Digest, Sha256};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Publication mode (MLXCACHE_CHUNK_STORE; default Chunked = the shipped
/// v1 codec decision from REPORT.md).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkMode {
    Whole,
    Chunked,
}

impl ChunkMode {
    pub fn from_env() -> Self {
        match std::env::var("MLXCACHE_CHUNK_STORE").as_deref() {
            Ok("0") | Ok("false") => ChunkMode::Whole,
            _ => ChunkMode::Chunked,
        }
    }
}

#[derive(Clone)]
pub struct Persistence {
    pub blob_dir: PathBuf,
    /// The chunked store (format_version 3 publisher), when enabled. `None`
    /// = whole-blob v2 mode. Shared by reference; the store is
    /// mutex-interiored, so clones share one chunk store per process.
    chunk_store: Option<Arc<ChunkStore>>,
    mode: ChunkMode,
    /// Serializes chunked publication of one final name. The pair of
    /// renames (.ckm then .ckpt) is atomic only against other holders of
    /// this lock: two unlocked publishers for the same name can interleave
    /// their renames and leave a header from one publication atop a
    /// manifest from another — a blob that can never load. (The daemon's
    /// index lock already serializes same-prefix publications; this makes
    /// Persistence safe by construction, verified by the concurrent
    /// same-name test.)
    chunk_publish_lock: Arc<std::sync::Mutex<()>>,
}

/// Process-wide monotonic counter to make temp file names unique across
/// concurrent publishes within one process.
fn chunk_err(e: crate::chunkstore::ChunkError) -> PersistError {
    match e {
        crate::chunkstore::ChunkError::Disk(reason) => PersistError::DiskFull {
            path: PathBuf::from(&reason),
            source: std::io::Error::other(reason),
        },
        other => PersistError::Corrupt {
            reason: other.to_string(),
        },
    }
}

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
        Self::new_with_mode(blob_dir, ChunkMode::from_env())
    }

    /// `ChunkMode::Chunked` publishes format_version 3 chunked blobs and
    /// opens the shared chunk store; `Whole` keeps the v2 whole-blob layout
    /// (rollback knob: MLXCACHE_CHUNK_STORE=0). Both modes READ both
    /// layouts — a store may mix them freely.
    pub fn new_with_mode(
        blob_dir: impl Into<PathBuf>,
        mode: ChunkMode,
    ) -> Result<Self, PersistError> {
        let dir = blob_dir.into();
        fs::create_dir_all(&dir)?;
        // Boot is the degenerate age sweep (everything is stale to a daemon
        // that was not running to own it).
        Self::reap_stale_settle_staging_in(&dir, std::time::Duration::ZERO);
        // The chunk store opens in BOTH modes: v3 blobs must stay READABLE
        // under the MLXCACHE_CHUNK_STORE=0 rollback (a healthy checkpoint is
        // never corruption because the operator rolled back the writer), and
        // the boot rebuild must derive refs so their chunks are not
        // reclaimed out from under them. Publication mode gates WRITES only.
        let chunk_store = {
            // list first: the boot scan derives refcounts from live
            // .ckpt files (v3 headers), then the store reclaims
            // crash orphans deterministically.
            let live = Self::list_blobs_in(&dir)?;
            Some(Arc::new(ChunkStore::open(&dir, &live).map_err(|e| {
                PersistError::Corrupt {
                    reason: e.to_string(),
                }
            })?))
        };
        Ok(Self {
            blob_dir: dir,
            chunk_store,
            mode,
            chunk_publish_lock: Arc::new(std::sync::Mutex::new(())),
        })
    }

    /// Remove `.settle-*` staging/temp files older than `older_than`.
    ///
    /// Why recurring, not boot-only: the sidecar writes its staging file at
    /// the END of its own generation, which the daemon does not control. A
    /// request that times out, a stream that fails before its guard exists,
    /// or a client disconnect between the last token and the done frame all
    /// leave a file written AFTER the daemon's guard dropped — invisible to
    /// `list_blobs` (its filter is `.ckpt`) and, before this, waiting for the
    /// next restart. Each file can be up to the settle byte cap, so repeated
    /// timeouts can fill the blob disk. Called from the evict reaper with an
    /// age threshold above the sidecar timeout, so no live request's staging
    /// file is ever taken.
    pub fn reap_stale_settle_staging(&self, older_than: std::time::Duration) -> usize {
        Self::reap_stale_settle_staging_in(&self.blob_dir, older_than)
    }

    fn reap_stale_settle_staging_in(
        dir: &std::path::Path,
        older_than: std::time::Duration,
    ) -> usize {
        let mut reaped = 0;
        for entry in fs::read_dir(dir).into_iter().flatten().flatten() {
            let name = entry.file_name();
            let Some(name) = name.to_str() else { continue };
            let is_staging = name.starts_with(".settle-")
                && (name.ends_with(".staging") || name.ends_with(".part"));
            // Publication temps are reaped at BOOT only: a live publish task
            // owns its .ckpt.tmp until the rename, and a slow fsync can
            // legitimately outlast the recurring threshold — reaping it
            // mid-write would kill a publication the request already earned.
            // At boot no task is live, so every temp is garbage.
            let is_publish_temp =
                older_than.is_zero() && (name.ends_with(".ckpt.tmp") || name.ends_with(".ckm.tmp"));
            if !(is_staging || is_publish_temp) {
                continue;
            }
            let age = entry
                .metadata()
                .and_then(|m| m.modified())
                .ok()
                .and_then(|t| t.elapsed().ok());
            // At boot (threshold ZERO) nothing can be live, so a file whose
            // age cannot be computed (mtime skew, stat failure) is stale too;
            // under a recurring threshold, an unknown age conservatively
            // spares the file.
            let stale = match age {
                Some(a) => a >= older_than,
                None => older_than.is_zero(),
            };
            if !stale {
                continue;
            }
            match fs::remove_file(entry.path()) {
                Ok(()) => {
                    reaped += 1;
                    tracing::warn!(
                        path = %entry.path().display(),
                        "reaped stale settle staging file"
                    );
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => tracing::warn!(
                    path = %entry.path().display(),
                    error = %e,
                    "could not reap stale settle staging file"
                ),
            }
        }
        reaped
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
        meta: CheckpointMeta,
        payload: &[u8],
    ) -> Result<PathBuf, PersistError> {
        // D1 integrity (format_version 2), stamped at the single publish
        // choke point so every blob carries it: a framing-valid bit-rot flip
        // is otherwise undetectable — the wire checks catch bad headers and
        // bad framing, not wrong bytes. The version bump marks the new
        // on-disk contract; legacy version-1 blobs stay loadable, unverified.
        // The implementation is publish_from_staged with the payload at offset
        // zero: one writer for the stamp/digest/temp/rename/DiskFull
        // sequence, so a change to it cannot miss a call site.
        self.publish_from_staged(prefix_hash, generation, meta, payload, 0)
    }

    /// Read a staged settle file ONCE, keeping the single buffer.
    ///
    /// The settle publish path must not hold two payload copies (the staged
    /// file can be the full settle byte cap). This reads the file one time,
    /// parses the header from the front, and hands back the buffer plus the
    /// payload's start offset; every later step (digest, publish write)
    /// operates on slices of this one allocation.
    pub fn read_staged(
        &self,
        path: &Path,
    ) -> Result<(CheckpointMeta, Vec<u8>, usize), PersistError> {
        let bytes = fs::read(path)?;
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
        // Same version policy as load(): the staged file is untrusted engine
        // output, and a format the daemon cannot interpret must not publish.
        // (The v2 payload digest is NOT re-verified here — the caller
        // cross-checks the ENGINE-reported digest against the payload slice,
        // which is the stronger guarantee for staging: it binds the file to
        // this request, not just to itself.)
        match meta.format_version {
            1 | 2 => {}
            v => {
                return Err(PersistError::Corrupt {
                    reason: format!("format version {v}"),
                })
            }
        }
        Ok((meta, bytes, 4 + header_len))
    }

    /// Publish from the staged buffer: writes the daemon's header then the
    /// payload SLICE of the already-read staging bytes. One copy of the
    /// payload exists for the whole publish (the read in `read_staged`); this
    /// adds none. The digest is computed over the same slice it writes, so
    /// what is verified is exactly what lands on disk.
    /// write-temp-then-rename with fsync and cleanup: the shared tail of
    /// every publish path (whole-blob, chunk manifest, chunk header).
    fn tmp_publish(&self, tmp: &Path, final_path: &Path, bytes: &[u8]) -> Result<(), PersistError> {
        let write_result = (|| -> std::io::Result<()> {
            let mut f = fs::File::create(tmp)?;
            f.write_all(bytes)?;
            f.sync_all()?;
            Ok(())
        })();
        if let Err(source) = write_result {
            let _ = fs::remove_file(tmp);
            return Err(PersistError::DiskFull {
                path: tmp.to_path_buf(),
                source,
            });
        }
        if let Err(source) = fs::rename(tmp, final_path) {
            let _ = fs::remove_file(tmp);
            return Err(PersistError::DiskFull {
                path: final_path.to_path_buf(),
                source,
            });
        }
        Ok(())
    }

    pub fn publish_from_staged(
        &self,
        prefix_hash: u128,
        generation: u64,
        mut meta: CheckpointMeta,
        staged: &[u8],
        payload_start: usize,
    ) -> Result<PathBuf, PersistError> {
        let payload = &staged[payload_start..];
        meta.payload_sha256 = Some(format!("{:x}", Sha256::digest(payload)));
        if let (Some(store), ChunkMode::Chunked) = (&self.chunk_store, self.mode) {
            let _guard = self.chunk_publish_lock.lock().unwrap();
            return self.publish_chunked(store, prefix_hash, generation, meta, payload);
        }
        meta.format_version = 2;
        let final_path = self
            .blob_dir
            .join(format!("{:032x}-{:016x}.ckpt", prefix_hash, generation));
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
        let header_len_u32 = u32::try_from(header.len()).map_err(|_| PersistError::Corrupt {
            reason: format!("checkpoint header too large: {} bytes", header.len()),
        })?;
        let mut wire = header_len_u32.to_le_bytes().to_vec();
        wire.extend_from_slice(&header);
        wire.extend_from_slice(payload);
        self.tmp_publish(&tmp_path, &final_path, &wire)?;
        Ok(final_path)
    }

    /// Chunked publish (format_version 3): chunk the payload over its token
    /// grid, append new chunks to the packed segments (fsynced), write the
    /// manifest, THEN publish the header-only .ckpt by atomic rename — the
    /// rename is the visibility point, so a crash mid-publish leaves
    /// reclaimable orphans, never a half blob.
    fn publish_chunked(
        &self,
        store: &Arc<ChunkStore>,
        prefix_hash: u128,
        generation: u64,
        mut meta: CheckpointMeta,
        payload: &[u8],
    ) -> Result<PathBuf, PersistError> {
        // positions covered by the payload: the tokens[:-1] convention.
        let positions = meta.token_count.saturating_sub(1);
        // Build the header FIRST: after staging+registration every failure
        // must be an I/O error path that unregisters, and serde/len checks
        // are not — so they run before anything is registered.
        // payload_sha256 was stamped by the caller (publish_from_staged)
        // over the same slice; re-hashing GB payloads per publish is pure
        // cost. Only the version changes here.
        meta.format_version = 3;
        let header = serde_json::to_vec(&meta).map_err(|e| PersistError::Corrupt {
            reason: format!("meta serialize: {e}"),
        })?;
        let header_len = u32::try_from(header.len()).map_err(|_| PersistError::Corrupt {
            reason: format!("checkpoint header too large: {} bytes", header.len()),
        })?;
        // Stage AND register atomically (the eviction GC runs on another
        // thread): registering before the .ckpt rename means a concurrent
        // sweep that removes the LAST other blob referencing a shared chunk
        // cannot GC our just-staged segment out from under the publication.
        // A crash between staging and the rename leaves refs with no blob —
        // bounded (boot replay re-derives from live pairs), never a lost
        // publication. Every failure path below unregisters.
        let manifest = store
            .stage_and_register(payload, positions)
            .map_err(chunk_err)?;
        let final_path = self
            .blob_dir
            .join(format!("{:032x}-{:016x}.ckpt", prefix_hash, generation));
        let ckm_path = self
            .blob_dir
            .join(format!("{:032x}-{:016x}.ckm", prefix_hash, generation));
        // Republishing an existing NAME takes over its refs: read the
        // manifest we are about to replace and unregister it first, or its
        // chunks stay referenced forever (the daemon never republishes a
        // name — generations are monotonic — but Persistence must not leak
        // when a caller does).
        if let Ok(old) = fs::read(&ckm_path) {
            if let Ok(old_manifest) = Manifest::decode(&old) {
                store.unregister(&old_manifest);
            }
        }
        // Temp names end `.ckm.tmp`/`.ckpt.tmp` so the boot reaper (which
        // takes `.ckpt.tmp` at boot, `.ckm.tmp` alongside it) catches a
        // crash's stranded temps instead of stranding unbudgeted files.
        let tmp = self.blob_dir.join(format!(
            "{:032x}-{:016x}.{}.{}.ckm.tmp",
            prefix_hash,
            generation,
            std::process::id(),
            next_temp_seq()
        ));
        // Write the manifest first, fsync, rename; then the (tiny) header.
        let write_result = (|| -> std::io::Result<()> {
            let mut f = fs::File::create(&tmp)?;
            f.write_all(&manifest.encode())?;
            f.sync_all()?;
            Ok(())
        })();
        if let Err(source) = write_result {
            let _ = fs::remove_file(&tmp);
            store.unregister(&manifest);
            return Err(PersistError::DiskFull { path: tmp, source });
        }
        if let Err(source) = fs::rename(&tmp, &ckm_path) {
            let _ = fs::remove_file(&tmp);
            store.unregister(&manifest);
            return Err(PersistError::DiskFull {
                path: ckm_path.clone(),
                source,
            });
        }
        let header_tmp = self.blob_dir.join(format!(
            "{:032x}-{:016x}.{}.{}.ckpt.tmp",
            prefix_hash,
            generation,
            std::process::id(),
            next_temp_seq()
        ));
        let write_result = (|| -> std::io::Result<()> {
            let mut f = fs::File::create(&header_tmp)?;
            f.write_all(&header_len.to_le_bytes())?;
            f.write_all(&header)?;
            f.sync_all()?;
            Ok(())
        })();
        if let Err(source) = write_result {
            let _ = fs::remove_file(&header_tmp);
            let _ = fs::remove_file(&ckm_path);
            store.unregister(&manifest);
            return Err(PersistError::DiskFull {
                path: header_tmp,
                source,
            });
        }
        if let Err(source) = fs::rename(&header_tmp, &final_path) {
            let _ = fs::remove_file(&header_tmp);
            let _ = fs::remove_file(&ckm_path);
            store.unregister(&manifest);
            return Err(PersistError::DiskFull {
                path: final_path.clone(),
                source,
            });
        }
        // Visible. Refs were registered by stage_and_register (one lock
        // with the append); registering again here would double-count every
        // chunk and pin removed blobs' bytes until a boot replay.
        Ok(final_path)
    }

    /// Load a blob: parse header, return (meta, payload). Corruption or version
    /// mismatch returns Corrupt — the caller quarantines (R1-1).
    pub fn load(&self, blob_path: &Path) -> Result<(CheckpointMeta, Vec<u8>), PersistError> {
        let mut bytes = fs::read(blob_path)?;
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
            3 => {
                // Chunked blob: the payload lives in the chunk store.
                // Reassemble (byte-exact by construction) and verify the
                // SAME D1 digest contract — corruption anywhere in
                // segments or manifest produces wrong bytes here, and the
                // digest is what retires the blob.
                let Some(store) = &self.chunk_store else {
                    return Err(PersistError::Corrupt {
                        reason: "chunk store unavailable (no chunks/ directory \
                                 beside this store?)"
                            .into(),
                    });
                };
                let expected =
                    meta.payload_sha256
                        .as_deref()
                        .ok_or_else(|| PersistError::Corrupt {
                            reason: "format version 3 requires payload_sha256".into(),
                        })?;
                let ckm = blob_path.with_extension("ckm");
                let mbytes = fs::read(&ckm).map_err(|_| PersistError::Corrupt {
                    reason: "chunk manifest missing".into(),
                })?;
                let manifest =
                    Manifest::decode(&mbytes).map_err(|reason| PersistError::Corrupt { reason })?;
                let payload = store
                    .reassemble(&manifest)
                    .map_err(|e| PersistError::Corrupt {
                        reason: e.to_string(),
                    })?;
                let actual = format!("{:x}", Sha256::digest(&payload));
                if !actual.eq_ignore_ascii_case(expected) {
                    return Err(PersistError::Corrupt {
                        reason: format!(
                            "payload digest mismatch (expected {expected}, got {actual})"
                        ),
                    });
                }
                return Ok((meta, payload));
            }
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
        // Strip the header in place instead of copying the payload into a
        // second allocation: KV blobs are tens-to-hundreds of MB, and the
        // copy doubled peak memory per load for no benefit.
        let payload_start = 4 + header_len;
        bytes.drain(..payload_start);
        Ok((meta, bytes))
    }

    /// Delete a published blob file on retirement (R1-1). Best-effort: a missing
    /// file is fine, and an I/O failure is logged by the caller, not fatal. This
    /// makes retirement durable — a blob left on disk would be re-indexed by the
    /// next startup rebuild and resurrect the poison.
    pub fn remove(&self, blob_name: &str) -> Result<(), PersistError> {
        match fs::remove_file(self.blob_dir.join(blob_name)) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(PersistError::Io(e)),
        }
        // Chunked blob retirement: drop the manifest and its chunk refs so
        // shared chunks' lifetimes follow the last blob that cites them.
        // The generation-encoded name means a remove() can never hit a
        // different publication's manifest.
        if let Some(store) = &self.chunk_store {
            let ckm = blob_name
                .strip_suffix(".ckpt")
                .unwrap_or(blob_name)
                .to_string()
                + ".ckm";
            let path = self.blob_dir.join(&ckm);
            if let Ok(mbytes) = fs::read(&path) {
                if let Ok(manifest) = Manifest::decode(&mbytes) {
                    store.unregister(&manifest);
                }
            }
            let _ = fs::remove_file(path);
        }
        Ok(())
    }

    /// Rebuild support (index corruption rescue): list all published blobs.
    /// Each blob's header carries its own token prefix (CheckpointMeta.tokens),
    /// so the index is rebuilt from blob metadata alone — blobs are the source
    /// of truth (see Orchestrator::rebuild_from_disk).
    pub fn list_blobs(&self) -> Result<Vec<PathBuf>, PersistError> {
        Self::list_blobs_in(&self.blob_dir)
    }

    fn list_blobs_in(dir: &Path) -> Result<Vec<PathBuf>, PersistError> {
        let mut out = Vec::new();
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().map(|e| e == "ckpt").unwrap_or(false) {
                out.push(path);
            }
        }
        Ok(out)
    }

    /// True when the named blob is a header-only v3 checkpoint (its bytes
    /// live in the chunk store, budgeted there). Reads ONLY the header
    /// prefix (4-byte length + that many bytes), never the payload: the
    /// reaper calls this per candidate per pass, and v1/v2 payloads are
    /// tens-to-hundreds of MB.
    pub fn is_chunked_blob(&self, blob_name: &str) -> bool {
        if self.chunk_store.is_none() {
            return false;
        }
        let Ok(mut f) = fs::File::open(self.blob_dir.join(blob_name)) else {
            return false;
        };
        use std::io::Read as _;
        let mut len_bytes = [0u8; 4];
        if f.read_exact(&mut len_bytes).is_err() {
            return false;
        }
        let header_len = u32::from_le_bytes(len_bytes) as usize;
        if header_len > crate::chunkstore::MAX_HEADER_READ {
            return false; // not one of ours (the wire cap is far lower)
        }
        let mut header = vec![0u8; header_len];
        if f.read_exact(&mut header).is_err() {
            return false;
        }
        // peek_meta parses `u32 length + JSON`; rebuild the prefix in front
        // of the JSON bytes we read.
        let mut prefixed = len_bytes.to_vec();
        prefixed.extend_from_slice(&header);
        crate::chunkstore::peek_meta(&prefixed)
            .map(|m| m.format_version == 3)
            .unwrap_or(false)
    }

    /// The store's physical footprint for budget recounts: whole-blob .ckpt
    /// files (v1/v2 payload-bearing; v3 headers are counted with the chunk
    /// store below via manifests) + segments + manifests. Directory truth,
    /// no in-memory state, so a recount after GC is always honest.
    pub fn store_physical_bytes(&self) -> u64 {
        let mut total = 0u64;
        if let Ok(entries) = fs::read_dir(&self.blob_dir) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let Some(name) = name.to_str() else { continue };
                if name.ends_with(".ckpt") || name.ends_with(".ckm") {
                    total += entry.metadata().map(|m| m.len()).unwrap_or(0);
                }
            }
        }
        if let Some(store) = &self.chunk_store {
            total += store.physical_bytes();
        }
        total
    }

    /// GC the chunk store (delete all-dead segments; returns bytes freed).
    /// Called by the eviction reaper after removals so the byte budget
    /// reflects reclaimed segments, not just unlinked headers.
    pub fn gc_chunked(&self) -> u64 {
        self.chunk_store
            .as_ref()
            .map(|s| s.gc_empty_segments())
            .unwrap_or(0)
    }

    /// One blob's eviction-budget footprint. v1/v2: the file's size. v3:
    /// the manifest's gross chunk bytes (shared chunks count once per
    /// referencing blob, which is what eviction ordering wants — evicting
    /// this blob frees its UNIQUE bytes; the pass-level total adds the
    /// shared/overhead truth exactly once via `chunked_overhead_bytes`).
    pub fn blob_disk_bytes(&self, blob_name: &str) -> Result<u64, PersistError> {
        let path = self.blob_dir.join(blob_name);
        let meta_len = fs::metadata(&path)
            .map(|m| m.len())
            .map_err(PersistError::Io);
        if self.chunk_store.is_none() {
            return meta_len;
        }
        if !self.is_chunked_blob(blob_name) {
            return meta_len;
        }
        let ckm = blob_name
            .strip_suffix(".ckpt")
            .unwrap_or(blob_name)
            .to_string()
            + ".ckm";
        let mbytes = fs::read(self.blob_dir.join(&ckm)).map_err(PersistError::Io)?;
        let manifest =
            Manifest::decode(&mbytes).map_err(|reason| PersistError::Corrupt { reason })?;
        Ok(manifest.gross_bytes())
    }

    /// Segment + manifest bytes NOT attributable to any single blob (shared
    /// chunks, dead bytes, record headers): added once per eviction pass.
    /// Zero in whole-blob mode.
    pub fn chunked_overhead_bytes(&self) -> u64 {
        let Some(store) = self.chunk_store.as_ref() else {
            return 0;
        };
        let mut manifests = 0u64;
        if let Ok(entries) = fs::read_dir(&self.blob_dir) {
            for entry in entries.flatten() {
                if entry.file_name().to_string_lossy().ends_with(".ckm") {
                    manifests += entry.metadata().map(|m| m.len()).unwrap_or(0);
                }
            }
        }
        store.physical_bytes() + manifests
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orchestrator::test_support::fp;

    pub(super) fn meta(n: u64) -> CheckpointMeta {
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

    pub(super) fn tmpdir() -> tempfile::TempDir {
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
    fn publish_stamps_v2_with_verified_digest_in_whole_mode() {
        // D1: publish is the single choke point that stamps the format
        // version + payload digest; load verifies it. Whole mode (the
        // MLXCACHE_CHUNK_STORE=0 rollback) keeps the v2 layout exactly.
        let dir = tmpdir();
        let p = Persistence::new_with_mode(dir.path(), ChunkMode::Whole).unwrap();
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
    fn publish_stamps_v3_chunked_with_verified_digest_by_default() {
        // The shipped default: the .ckpt is a header-only file, the payload
        // lives chunked in segments, and load() reassembles + verifies the
        // SAME digest contract (D1 covers reassembly: corruption anywhere
        // in segments or manifest yields wrong bytes here).
        let dir = tmpdir();
        let p = Persistence::new(dir.path()).unwrap();
        let path = p.publish_atomic(0x2, 1, meta(10), b"kv-payload").unwrap();
        let (m, payload) = p.load(&path).unwrap();
        assert_eq!(m.format_version, 3);
        assert_eq!(payload, b"kv-payload");
        assert_eq!(
            m.payload_sha256.as_deref(),
            Some(&format!("{:x}", Sha256::digest(b"kv-payload"))[..])
        );
        // The header-only .ckpt carries no payload bytes.
        assert!(fs::metadata(&path).unwrap().len() < 512);
    }

    #[test]
    fn chunked_payload_corruption_in_segment_is_rejected_by_digest() {
        // The v3 analog of the framing-valid bit-rot test: flip a byte in a
        // SEGMENT (not the tiny header) and load must retire the blob.
        let dir = tmpdir();
        let p = Persistence::new(dir.path()).unwrap();
        let path = p
            .publish_atomic(0x9, 1, meta(64), &vec![7u8; 4096])
            .unwrap();
        // Tamper the first chunk segment record's body byte.
        let seg = dir.path().join("chunks").join("seg-0000000000.seg");
        let mut bytes = fs::read(&seg).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff;
        fs::write(&seg, &bytes).unwrap();
        match p.load(&path) {
            Err(PersistError::Corrupt { reason }) => {
                assert!(reason.contains("digest mismatch"), "reason: {reason}");
            }
            other => panic!("expected Corrupt, got {other:?}"),
        }
    }

    #[test]
    fn tampered_payload_is_rejected_by_digest() {
        // Framing-valid corruption (bit rot, a torn rewrite that kept the
        // header intact) is exactly what the structural checks cannot
        // catch. (Chunked corruption is covered by its own test above.)
        let dir = tmpdir();
        let p = Persistence::new_with_mode(dir.path(), ChunkMode::Whole).unwrap();
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
    fn read_staged_refuses_unknown_format_versions() {
        // The staged file is untrusted engine output: a version the daemon
        // cannot interpret must not reach publication (same policy as load).
        let dir = tmpdir();
        let p = Persistence::new(dir.path()).unwrap();
        let meta_bytes = br#"{"fingerprint":{"model_id":"m","tokenizer_hash":"h","kv_dtype":"f16","kv_layout_version":1},"token_count":2,"tokens":[1,2],"format_version":3}"#;
        let mut blob = Vec::new();
        blob.extend_from_slice(&(meta_bytes.len() as u32).to_le_bytes());
        blob.extend_from_slice(meta_bytes);
        blob.extend_from_slice(b"kv");
        let staged = dir.path().join(".settle-1-1-1.staging");
        fs::write(&staged, &blob).unwrap();
        let err = p.read_staged(&staged).unwrap_err();
        assert!(
            err.to_string().contains("format version 3"),
            "reason: {err}"
        );
    }

    #[test]
    fn age_based_reaper_takes_only_stale_staging_files() {
        // The recurring sweep (CodeRabbit #36): a staging file written by an
        // abandoned sidecar request after its guard dropped must not wait for
        // the next boot, but a file inside the age threshold belongs to a
        // possibly-live request and must be left alone.
        let dir = tmpdir();
        // Persistence FIRST: its constructor's boot sweep (threshold zero)
        // would take every staging file before the age logic runs.
        let p = Persistence::new(dir.path()).unwrap();
        let old = dir.path().join(".settle-1-1-1.staging");
        let fresh = dir.path().join(".settle-2-2-2.staging");
        let part = dir.path().join(".settle-3-3-3.staging.ABC.part");
        fs::write(&old, b"orphan").unwrap();
        fs::write(&fresh, b"live").unwrap();
        fs::write(&part, b"orphan").unwrap();
        // Backdate the old pair past any plausible threshold.
        let past = std::time::SystemTime::now() - std::time::Duration::from_secs(3600);
        let old_time = fs::File::options().append(true).open(&old).unwrap();
        old_time.set_modified(past).unwrap();
        let part_time = fs::File::options().append(true).open(&part).unwrap();
        part_time.set_modified(past).unwrap();

        let reaped = p.reap_stale_settle_staging(std::time::Duration::from_secs(600));
        assert_eq!(reaped, 2, "old .staging + old .part taken");
        assert!(!old.exists());
        assert!(!part.exists());
        assert!(fresh.exists(), "a file inside the threshold is live");
    }

    #[test]
    fn stale_settle_staging_files_are_reaped_at_boot() {
        // A daemon that died between the sidecar's temp write and its rename
        // strands staging files list_blobs can never see (its filter is
        // `.ckpt`). Each is up to the settle byte cap, so construction must
        // reap them; ordinary files and published blobs stay untouched.
        let dir = tmpdir();
        fs::write(dir.path().join(".settle-1-1-1.staging"), b"orphan").unwrap();
        fs::write(dir.path().join(".settle-1-1-1.staging.ABC.part"), b"orphan").unwrap();
        fs::write(dir.path().join("unrelated.txt"), b"keep").unwrap();
        let p = Persistence::new(dir.path()).unwrap();
        p.publish_atomic(0x1, 1, meta(1), b"a").unwrap();

        assert!(!dir.path().join(".settle-1-1-1.staging").exists());
        assert!(!dir.path().join(".settle-1-1-1.staging.ABC.part").exists());
        assert!(dir.path().join("unrelated.txt").exists());
        assert_eq!(p.list_blobs().unwrap().len(), 1);
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

#[cfg(test)]
mod chunk_rollback_tests {
    use super::tests::{meta, tmpdir};
    use super::{ChunkMode, Persistence};

    #[test]
    fn rollback_mode_reads_v3_blobs_instead_of_deleting_them() {
        // The documented rollback (MLXCACHE_CHUNK_STORE=0) must stop
        // WRITING v3, never stop READING it: a healthy checkpoint is not
        // corruption because the operator rolled the writer back, and the
        // boot rebuild would otherwise quarantine (delete) every v3 blob.
        let dir = tmpdir();
        let payload: Vec<u8> = (0..6000u32).map(|i| (i % 251) as u8).collect();
        let chunked = Persistence::new(dir.path()).unwrap();
        let path = chunked.publish_atomic(0x5, 1, meta(9), &payload).unwrap();
        drop(chunked);
        let rolled_back = Persistence::new_with_mode(dir.path(), ChunkMode::Whole).unwrap();
        let (m, got) = rolled_back.load(&path).expect("rollback must READ v3");
        assert_eq!(m.format_version, 3);
        assert_eq!(got, payload);
        // And it still does not WRITE v3.
        let path2 = rolled_back
            .publish_atomic(0x6, 1, meta(9), &payload)
            .unwrap();
        let (m2, _) = rolled_back.load(&path2).unwrap();
        assert_eq!(m2.format_version, 2);
    }

    #[test]
    fn undecodable_manifest_never_aborts_boot() {
        // A corrupt .ckm is BLOB corruption (load() -> Corrupt -> the
        // rebuild quarantines that one blob), never a store-wide boot
        // failure.
        let dir = tmpdir();
        let p = Persistence::new(dir.path()).unwrap();
        let payload: Vec<u8> = (0..3000u32).map(|i| (i % 249) as u8).collect();
        let _path = p.publish_atomic(0x7, 1, meta(9), &payload).unwrap();
        drop(p);
        let ckm = dir.path().join(
            std::fs::read_dir(dir.path())
                .unwrap()
                .find(|e| {
                    e.as_ref()
                        .map(|e| e.file_name().to_string_lossy().ends_with(".ckm"))
                        .unwrap_or(false)
                })
                .unwrap()
                .unwrap()
                .file_name(),
        );
        std::fs::write(&ckm, b"MCKM-garbage").unwrap();
        // Boot must succeed; the blob itself is corrupt on load.
        let p2 = Persistence::new(dir.path()).expect("boot must tolerate a corrupt manifest");
        let ckpt: std::path::PathBuf = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .find(|p| p.extension().map(|e| e == "ckpt").unwrap_or(false))
            .unwrap();
        assert!(p2.load(&ckpt).is_err(), "the blob itself is corrupt");
    }
}

#[cfg(test)]
mod is_chunked_probe {
    use super::tests::{meta, tmpdir};
    use super::Persistence;

    #[test]
    fn is_chunked_blob_is_true_for_v3_and_false_for_v2() {
        let dir = tmpdir();
        let p = Persistence::new(dir.path()).unwrap();
        let payload: Vec<u8> = (0..5000u32).map(|i| (i % 247) as u8).collect();
        let v3 = p.publish_atomic(0xa, 1, meta(9), &payload).unwrap();
        let p2 = Persistence::new_with_mode(dir.path(), super::ChunkMode::Whole).unwrap();
        let v2 = p2.publish_atomic(0xb, 1, meta(9), &payload).unwrap();
        assert!(p.is_chunked_blob(v3.file_name().unwrap().to_str().unwrap()));
        assert!(!p.is_chunked_blob(v2.file_name().unwrap().to_str().unwrap()));
    }
}

#[cfg(test)]
mod republish_takeover {
    use super::tests::{meta, tmpdir};
    use super::Persistence;

    #[test]
    fn republishing_a_name_does_not_leak_the_old_publications_refs() {
        // Same (hash, generation), different payloads: the name is the
        // claim, and taking it over must release the replaced manifest's
        // chunk refs (a leaked refcount pins dead bytes until a boot
        // replay). Prove it end-state: after the takeover AND a remove of
        // the surviving blob, every chunk is dead — boot GC reclaims the
        // whole store. A leaked ref from publication A keeps the segment
        // live and fails this.
        let dir = tmpdir();
        let p = Persistence::new(dir.path()).unwrap();
        let a: Vec<u8> = (0..4096u32).map(|i| (i % 253) as u8).collect();
        let b: Vec<u8> = (0..4096u32).map(|i| ((i * 7) % 251) as u8).collect();
        let path = p.publish_atomic(0x99, 5, meta(9), &a).unwrap();
        let path2 = p.publish_atomic(0x99, 5, meta(9), &b).unwrap();
        assert_eq!(path, path2, "same name both times");
        let (_m, payload) = p.load(&path).unwrap();
        assert_eq!(payload, b, "the surviving pair is b's");
        let name = path.file_name().unwrap().to_str().unwrap().to_string();
        p.remove(&name).unwrap();
        drop(p);
        let p2 = Persistence::new(dir.path()).unwrap();
        assert_eq!(
            p2.store_physical_bytes(),
            0,
            "boot GC must reclaim everything: a leaked ref from the replaced publication keeps segments live"
        );
    }
}
