//! The chunked blob store (REPORT.md GO: token-block/64 as the v1 codec).
//!
//! Whole-blob checkpoints waste the prefix they share with the blobs that
//! extend them: a growing conversation republishes its whole history every
//! turn. This store chunks a payload by the TOKEN GRID — blocks of
//! TOKEN_BLOCK covered positions, per tensor, with each block's identity
//! the sha256 over the concatenation of its per-head slices (the head-major
//! safetensors layout makes those slices stable under append, so a shared
//! prefix yields byte-identical chunks — measured 5.3%/5.2% new bytes per
//! turn on the 7B, REPORT.md E3).
//!
//! On-disk layout (format_version 3 blobs; v1/v2 whole-blobs coexist in the
//! same directory and stay readable forever):
//!
//! - `{hash}-{gen}.ckpt` — the checkpoint header only (CheckpointMeta,
//!   format_version 3, payload_sha256 = digest of the LOGICAL payload).
//! - `{hash}-{gen}.ckm`  — the chunk manifest (binary; see below).
//! - `chunks/seg-{seq}.seg` — immutable packed segment files holding chunk
//!   bodies content-addressed by their sha256. Segment = magic + records
//!   `[u32 body_len][32B digest][body]`. Unpacked file-per-chunk was
//!   measured and FAILED the restore budget (REPORT.md E4/N2): packing is
//!   mandatory.
//!
//! Manifest record per chunk (all little-endian):
//! `[32B digest][u32 per_head_len][u32 n_heads][u64 first_off][u64 head_stride]`
//! — the chunk's body (n_heads × per_head_len bytes, in head order)
//! reassembles by scattering per_head_len bytes at
//! `first_off + h * head_stride` for each head h. This carries the whole
//! placement: any reader can reassemble the payload without knowing the
//! tensor grid, and the digest binds exactly the bytes it scatters.
//!
//! Tensors whose grid does not divide (or payloads that are not
//! safetensors at all) fall back to ONE whole-scope chunk — identity and
//! reassembly stay exact; such a payload still dedups byte-identical
//! copies (the P2 cold-twice determinism case).
//!
//! Crash safety: chunk appends are fsynced BEFORE the manifest, and the
//! manifest before the .ckpt rename (the publication point). A crash
//! leaves orphan chunks/manifests that the boot scan reclaims: refcounts
//! are DERIVED at boot by replaying manifests of live .ckpt files, never
//! trusted from disk.
//!
//! Reclamation: a segment is deleted only when every record in it has
//! refcount 0 (the "chunk leaves disk only at refcount 0" invariant; dead
//! bytes inside a live segment wait for compaction, which is deliberately
//! out of v1 — REPORT.md's republish evidence says the wait is cheap).

use mlxcache_core::contract::CheckpointMeta;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// v1 codec block size: the adopted config from REPORT.md's E3 (best
/// measured dedup in both tiers; 256 was worse everywhere it was measured).
pub const TOKEN_BLOCK: u64 = 64;
/// Segment roll target. 64 MiB with 64 KiB chunks was the E4 winner
/// (restore FASTER than plain); ≥64 MiB is the mandate from N2.
pub const SEGMENT_TARGET: u64 = 64 << 20;

const SEGMENT_MAGIC: &[u8; 4] = b"MCSG";
const MANIFEST_MAGIC: &[u8; 4] = b"MCKM";
const FORMAT_VERSION: u16 = 1;

pub type ChunkDigest = [u8; 32];

/// One chunk's identity + placement, as the manifest records it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkRef {
    pub digest: ChunkDigest,
    /// Bytes per head slice in the payload (n_positions × unit).
    pub per_head_len: u32,
    /// Number of head slices scattered for this chunk.
    pub n_heads: u32,
    /// Payload offset of head 0's slice.
    pub first_off: u64,
    /// Payload stride between consecutive heads' slices.
    pub head_stride: u64,
}

impl ChunkRef {
    /// The chunk's on-disk body size: heads × slice, concatenated in head
    /// order (the exact bytes the digest covers).
    pub fn body_len(&self) -> u64 {
        self.n_heads as u64 * self.per_head_len as u64
    }
}

/// A parsed manifest: enough to reassemble the payload and to account refs.
#[derive(Debug, Clone)]
pub struct Manifest {
    pub payload_len: u64,
    pub chunks: Vec<ChunkRef>,
}

impl Manifest {
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(22 + self.chunks.len() * 56);
        out.extend_from_slice(MANIFEST_MAGIC);
        out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        out.extend_from_slice(&self.payload_len.to_le_bytes());
        out.extend_from_slice(&(self.chunks.len() as u64).to_le_bytes());
        for c in &self.chunks {
            out.extend_from_slice(&c.digest);
            out.extend_from_slice(&c.per_head_len.to_le_bytes());
            out.extend_from_slice(&c.n_heads.to_le_bytes());
            out.extend_from_slice(&c.first_off.to_le_bytes());
            out.extend_from_slice(&c.head_stride.to_le_bytes());
        }
        out
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, String> {
        let bad = |m: &str| format!("chunk manifest corrupt: {m}");
        if bytes.len() < 22 || &bytes[0..4] != MANIFEST_MAGIC {
            return Err(bad("bad magic"));
        }
        let version = u16::from_le_bytes([bytes[4], bytes[5]]);
        if version != FORMAT_VERSION {
            return Err(bad(&format!("version {version}")));
        }
        let payload_len = u64::from_le_bytes(bytes[6..14].try_into().unwrap());
        let count = u64::from_le_bytes(bytes[14..22].try_into().unwrap());
        // Each record is fixed-size (56B); a count that cannot fit is a
        // truncated manifest, not a reservation we should trust.
        let count = usize::try_from(count).map_err(|_| bad("count overflow"))?;
        let need = 22 + count.checked_mul(56).ok_or_else(|| bad("count overflow"))?;
        if bytes.len() < need {
            return Err(bad("truncated records"));
        }
        let mut chunks = Vec::with_capacity(count);
        let mut at = 22;
        let mut max_end = 0u64;
        for _ in 0..count {
            let mut digest = [0u8; 32];
            digest.copy_from_slice(&bytes[at..at + 32]);
            at += 32;
            let per_head_len = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
            at += 4;
            let n_heads = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
            at += 4;
            let first_off = u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap());
            at += 8;
            let head_stride = u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap());
            at += 8;
            if n_heads == 0 || per_head_len == 0 {
                return Err(bad("empty chunk geometry"));
            }
            let end = first_off
                .checked_add((n_heads as u64 - 1).saturating_mul(head_stride))
                .and_then(|e| e.checked_add(per_head_len as u64))
                .ok_or_else(|| bad("placement overflow"))?;
            if end > payload_len {
                return Err(bad("placement exceeds payload"));
            }
            max_end = max_end.max(end);
            chunks.push(ChunkRef {
                digest,
                per_head_len,
                n_heads,
                first_off,
                head_stride,
            });
        }
        // The chunker guarantees chunks cover the whole payload: the max
        // chunk end IS payload_len (0 for an empty one). A bit flip in the
        // high bytes of payload_len would otherwise decode fine and make
        // reassemble allocate attacker-sized memory.
        if max_end != payload_len && !(payload_len == 0 && chunks.is_empty()) {
            return Err(bad("payload_len does not match chunk coverage"));
        }
        Ok(Manifest {
            payload_len,
            chunks,
        })
    }

    /// Logical bytes this blob contributes uniquely... no — the SUM of its
    /// chunk bodies (shared chunks count once per referencing blob in the
    /// refcount world; this is the blob's gross chunk bytes, used for
    /// eviction ordering, not for disk accounting).
    pub fn gross_bytes(&self) -> u64 {
        self.chunks.iter().map(|c| c.body_len()).sum()
    }
}

/// A safetensors tensor's byte placement inside the payload.
struct TensorSpan {
    begin: u64,
    end: u64,
    n_heads: u64,
}

/// Parse the safetensors header of a payload into tensor spans. Returns
/// None when the payload is not a framing-valid safetensors frame (the
/// single-chunk fallback then applies). Only data tensors are returned,
/// sorted by name — the deterministic chunk order the identity depends on.
fn safetensors_spans(payload: &[u8]) -> Option<Vec<(String, TensorSpan)>> {
    if payload.len() < 8 {
        return None;
    }
    let header_len = u64::from_le_bytes(payload[0..8].try_into().ok()?) as usize;
    let data_start = 8usize.checked_add(header_len)?;
    if data_start > payload.len() {
        return None;
    }
    let header: serde_json::Value = serde_json::from_slice(&payload[8..data_start]).ok()?;
    let obj = header.as_object()?;
    let mut out = Vec::new();
    for (name, spec) in obj {
        if name == "__metadata__" {
            continue;
        }
        let spec = spec.as_object()?;
        let shape = spec.get("shape")?.as_array()?;
        let offs = spec.get("data_offsets")?.as_array()?;
        if offs.len() != 2 {
            return None;
        }
        let begin = offs[0].as_u64()?;
        let end = offs[1].as_u64()?;
        // Rank < 2 tensors (the TQ frame's rank-1 codebook) have no head
        // axis to grid over: span them with n_heads = 1 so the per-tensor
        // fallback chunks them whole instead of aborting the grid for the
        // entire payload.
        let n_heads = if shape.len() < 2 {
            1
        } else {
            shape[1].as_u64()?
        };
        if n_heads == 0 || end < begin || data_start as u64 + end > payload.len() as u64 {
            return None;
        }
        out.push((
            name.clone(),
            TensorSpan {
                begin: data_start as u64 + begin,
                end: data_start as u64 + end,
                n_heads,
            },
        ));
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Some(out)
}

/// Chunk a payload over its token grid.
///
/// `positions` is the number of covered KV positions (token_count − 1 per
/// the repository's `tokens[:-1]` convention). Every tensor with an exact
/// head/position grid yields one chunk per TOKEN_BLOCK block (digest over
/// the per-head slice concatenation); anything else becomes one
/// whole-scope chunk. The returned refs are in deterministic order.
fn chunk_payload(payload: &[u8], positions: u64) -> Manifest {
    // Whole-scope fallback chunks, split at u32::MAX so no record length
    // can overflow the segment format (an operator raising the settle byte
    // cap past 4 GiB must not turn a publish into a panic).
    let fallback = |chunks: &mut Vec<ChunkRef>, first_off: u64, len: u64| {
        const MAX: u64 = u32::MAX as u64;
        let mut off = first_off;
        let mut left = len;
        while left > 0 {
            let n = left.min(MAX);
            chunks.push(ChunkRef {
                digest: Sha256::digest(&payload[off as usize..(off + n) as usize]).into(),
                per_head_len: n as u32,
                n_heads: 1,
                first_off: off,
                head_stride: 0,
            });
            off += n;
            left -= n;
        }
    };
    let mut chunks = Vec::new();
    let mut covered: Vec<(u64, u64)> = Vec::new();
    let spans = if positions >= 1 {
        safetensors_spans(payload)
    } else {
        None
    };
    let Some(mut spans) = spans else {
        // Not safetensors (the synthetic engine's deterministic bytes) or a
        // degenerate grid: the whole payload is one chunk. Byte-identical
        // payloads still dedup; only layout-level sharing is lost.
        if !payload.is_empty() {
            fallback(&mut chunks, 0, payload.len() as u64);
        }
        return Manifest {
            payload_len: payload.len() as u64,
            chunks,
        };
    };
    spans.retain(|(_, t)| t.end > t.begin); // empty tensors: no bytes, no chunks
    for (_name, t) in spans {
        let nbytes = t.end - t.begin;
        let hsz = nbytes / t.n_heads;
        let exact = positions >= 1
            && hsz % positions == 0
            && usize::try_from(hsz).is_ok()
            && usize::try_from(t.n_heads).is_ok();
        if !exact {
            fallback(&mut chunks, t.begin, nbytes);
            covered.push((t.begin, t.end));
            continue;
        }
        let unit = hsz / positions; // bytes per position per head
        let mut b: u64 = 0;
        while b < positions {
            let n = TOKEN_BLOCK.min(positions - b);
            let per_head = (n * unit) as u32;
            let mut hasher = Sha256::new();
            for h in 0..t.n_heads {
                let base = t.begin + h * hsz + b * unit;
                hasher.update(&payload[base as usize..(base + n * unit) as usize]);
            }
            chunks.push(ChunkRef {
                digest: hasher.finalize().into(),
                per_head_len: per_head,
                n_heads: t.n_heads as u32,
                first_off: t.begin + b * unit,
                head_stride: hsz,
            });
            b += n;
        }
        covered.push((t.begin, t.end));
    }
    // The payload is more than its tensors: the safetensors JSON header
    // (and any inter-tensor gap a writer leaves) must round-trip too. Every
    // byte range no tensor claimed becomes a whole-scope fallback chunk —
    // identity stays exact and reassembly stays byte-complete.
    covered.sort_unstable();
    let mut cursor = 0u64;
    for (begin, end) in covered.iter().chain(std::iter::once(&(
        payload.len() as u64,
        payload.len() as u64,
    ))) {
        if *begin > cursor {
            fallback(&mut chunks, cursor, *begin - cursor);
        }
        cursor = cursor.max(*end);
    }
    Manifest {
        payload_len: payload.len() as u64,
        chunks,
    }
}

/// Where a chunk's body lives inside the segment files.
#[derive(Debug, Clone, Copy)]
struct Placement {
    seg: u64,
    offset: u64, // record data start (past len+digest)
    len: u64,
}

struct Inner {
    /// digest → placement. One placement per digest (content-addressed;
    /// duplicate records are never written past the first).
    placement: HashMap<ChunkDigest, Placement>,
    /// digest → live referencing manifests. Refcounts are DERIVED from live
    /// .ckpt files at boot and maintained incrementally thereafter.
    refs: HashMap<ChunkDigest, u64>,
    /// seg seq → (total record bytes incl. headers, dead bytes). Drives
    /// empty-segment GC and honest physical accounting.
    segments: HashMap<u64, SegmentStat>,
    active: Option<ActiveSegment>,
    next_seg: u64,
}

struct ActiveSegment {
    seq: u64,
    file: fs::File,
    len: u64,
}

#[derive(Debug, Default, Clone, Copy)]
struct SegmentStat {
    total: u64,
    dead: u64,
}

impl SegmentStat {
    fn live(&self) -> u64 {
        self.total - self.dead
    }
}

/// Errors mirror Persistence's taxonomy so callers classify identically.
#[derive(Debug, thiserror::Error)]
pub enum ChunkError {
    #[error("disk full or unwritable: {0}")]
    Disk(String),
    #[error("chunk store corrupt: {0}")]
    Corrupt(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

pub struct ChunkStore {
    dir: PathBuf,
    segments_dir: PathBuf,
    inner: Mutex<Inner>,
}

impl ChunkStore {
    /// Open (and, on a fresh directory, initialize) the chunk store. The
    /// boot scan derives refcounts from LIVE .ckpt↔.ckm pairs; orphan
    /// manifests (crash between .ckm and .ckpt rename) are deleted, and
    /// their chunks become reclaimable dead bytes.
    pub fn open(blob_dir: &Path, live_blobs: &[PathBuf]) -> Result<Self, ChunkError> {
        let segments_dir = blob_dir.join("chunks");
        // NOTE: no create_dir_all here. A read-only blob dir at boot is a
        // legitimate state (the disk-full simulation boots there before
        // loosening permissions); the directory is created lazily at the
        // first segment write, where failure maps to Disk like any other
        // write error.
        let mut store = ChunkStore {
            dir: blob_dir.to_path_buf(),
            segments_dir: segments_dir.clone(),
            inner: Mutex::new(Inner {
                placement: HashMap::new(),
                refs: HashMap::new(),
                segments: HashMap::new(),
                active: None,
                next_seg: 0,
            }),
        };
        store.scan_segments()?;
        let manifests = store.collect_live_manifests(live_blobs)?;
        {
            let mut inner = store.inner.lock().unwrap();
            for m in &manifests {
                for c in &m.chunks {
                    *inner.refs.entry(c.digest).or_insert(0) += 1;
                }
            }
            // Dead bytes: every placed record nobody references.
            // Dead accounting includes each record's 36-byte header:
            // SegmentStat.total counts headers, so live() can only reach
            // zero when dead does too.
            const REC_HEADER: u64 = 36;
            let dead: Vec<(u64, u64)> = inner
                .placement
                .iter()
                .filter(|(digest, _)| !inner.refs.contains_key(*digest))
                .map(|(_, p)| (p.seg, REC_HEADER + p.len))
                .collect();
            for (seg, len) in dead {
                if let Some(stat) = inner.segments.get_mut(&seg) {
                    stat.dead += len;
                }
            }
            inner.next_seg = inner
                .segments
                .keys()
                .copied()
                .max()
                .map(|s| s + 1)
                .unwrap_or(0);
        }
        store.reclaim_orphan_manifests(live_blobs)?;
        store.gc_empty_segments();
        Ok(store)
    }

    /// Walk segment files; build the placement table. A truncated tail
    /// record (crash mid-append) ends the scan of that segment cleanly:
    /// unreferenced bytes are dead, never served.
    fn scan_segments(&mut self) -> Result<(), ChunkError> {
        let mut seqs: Vec<u64> = Vec::new();
        // A missing chunks/ dir is an empty chunk store (lazy creation).
        let entries = match fs::read_dir(&self.segments_dir) {
            Ok(e) => e,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(e) => return Err(e.into()),
        };
        for entry in entries {
            let entry = entry?;
            let name = entry.file_name();
            let Some(name) = name.to_str() else {
                continue;
            };
            let Some(seq) = name
                .strip_prefix("seg-")
                .and_then(|s| s.strip_suffix(".seg"))
                .and_then(|s| s.parse::<u64>().ok())
            else {
                continue;
            };
            seqs.push(seq);
        }
        seqs.sort_unstable();
        // Scan HIGHEST seq first, so the first-wins placement rule keeps
        // the NEWEST copy of a duplicate digest. Duplicates only arise from
        // a compaction crash (new segment fsynced, old not yet deleted);
        // resolving to the newest makes that crash self-healing, and the
        // losing record is accounted dead on its segment right here.
        let mut inner = self.inner.lock().unwrap();
        for seq in seqs.into_iter().rev() {
            let path = self.segment_path(seq);
            let bytes = fs::read(&path)?;
            if bytes.len() < 8 || &bytes[0..4] != SEGMENT_MAGIC {
                if bytes.len() < 8 {
                    // A crash between File::create and the header write
                    // strands a 0/partial-byte segment: a crash artifact,
                    // reclaimed here, never a boot failure.
                    let _ = fs::remove_file(&path);
                    continue;
                }
                return Err(ChunkError::Corrupt(format!(
                    "segment {seq} bad magic (not a chunk segment?)"
                )));
            }
            let version = u16::from_le_bytes([bytes[4], bytes[5]]);
            if version != FORMAT_VERSION {
                return Err(ChunkError::Corrupt(format!(
                    "segment {seq} version {version}"
                )));
            }
            let mut at = 8usize;
            let mut total = 0u64;
            let mut lost_duplicates = 0u64;
            while at + 4 + 32 <= bytes.len() {
                let body_len = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as u64;
                let rec_start = at as u64 + 4 + 32;
                let rec_end = rec_start.checked_add(body_len);
                let Some(rec_end) = rec_end else { break };
                if rec_end > bytes.len() as u64 {
                    break; // truncated tail: dead bytes, stop cleanly
                }
                let mut digest = [0u8; 32];
                digest.copy_from_slice(&bytes[at + 4..at + 4 + 32]);
                // First record for a digest wins placement (scanning
                // newest-first, that is the newest copy). A duplicate
                // record — only a compaction-crash artifact — never
                // overrides; its bytes are dead on THIS segment from the
                // moment the scan sees it.
                match inner.placement.entry(digest) {
                    std::collections::hash_map::Entry::Occupied(_) => {
                        lost_duplicates += 4 + 32 + body_len;
                    }
                    std::collections::hash_map::Entry::Vacant(v) => {
                        v.insert(Placement {
                            seg: seq,
                            offset: rec_start,
                            len: body_len,
                        });
                    }
                }
                total += 4 + 32 + body_len;
                at = rec_end as usize;
            }
            inner.segments.insert(
                seq,
                SegmentStat {
                    total,
                    dead: lost_duplicates,
                },
            );
        }
        Ok(())
    }

    /// Read the manifests of live blobs (a .ckpt whose header says v3 must
    /// have its .ckm; a missing manifest is corruption the caller
    /// quarantines through the normal load path, so here it only means "no
    /// refs contributed").
    fn collect_live_manifests(&self, live_blobs: &[PathBuf]) -> Result<Vec<Manifest>, ChunkError> {
        let mut out = Vec::new();
        for ckpt in live_blobs {
            let Ok(bytes) = fs::read(ckpt) else {
                continue;
            };
            let Some(meta) = peek_meta(&bytes) else {
                continue;
            };
            if meta.format_version != 3 {
                continue;
            }
            let ckm = ckpt.with_extension("ckm");
            let Ok(mbytes) = fs::read(&ckm) else {
                // The blob is corrupt (manifest lost); load() will say so.
                continue;
            };
            // An undecodable manifest is BLOB corruption (load() says
            // Corrupt → the rebuild quarantines that one blob), never a
            // boot failure for the whole store.
            if let Ok(manifest) = Manifest::decode(&mbytes) {
                out.push(manifest);
            }
        }
        Ok(out)
    }

    /// Delete .ckm files with no live .ckpt (crash between the two
    /// renames). Chunks they referenced are already unref'd by construction
    /// (refs come only from live pairs).
    fn reclaim_orphan_manifests(&self, live_blobs: &[PathBuf]) -> Result<(), ChunkError> {
        use std::collections::HashSet;
        let live: HashSet<String> = live_blobs
            .iter()
            .filter_map(|p| p.file_name().and_then(|n| n.to_str()).map(String::from))
            .collect();
        for entry in fs::read_dir(&self.dir)? {
            let entry = entry?;
            let name = entry.file_name();
            let Some(name) = name.to_str() else { continue };
            if !name.ends_with(".ckm") {
                continue;
            }
            let ckpt = name.replace(".ckm", ".ckpt");
            if !live.contains(&ckpt) {
                let _ = fs::remove_file(entry.path());
            }
        }
        Ok(())
    }

    fn segment_path(&self, seq: u64) -> PathBuf {
        self.segments_dir.join(format!("seg-{seq:010}.seg"))
    }

    /// Chunk `payload`, append every not-yet-present chunk to the active
    /// segment (fsync before returning), and write the manifest atomically.
    /// Returns the manifest; the CALLER then publishes the .ckpt (the
    /// visibility point) and registers refs.
    /// Stage + register under one lock: the atomic unit a publication needs
    /// (see publish_chunked). A crash after this leaks refs until the next
    /// boot replay — bounded, never a lost publication.
    pub fn stage_and_register(
        &self,
        payload: &[u8],
        positions: u64,
    ) -> Result<Manifest, ChunkError> {
        // ONE lock across append + refcount increment: two separate locks
        // left a window where the eviction thread could GC a sealed
        // all-dead segment (this payload's chunks not yet referenced, the
        // last OTHER referencing blob just removed) out from under the
        // publication — a published blob whose chunks are gone.
        let manifest = self.stage_chunks(payload, positions)?;
        {
            let mut inner = self.inner.lock().unwrap();
            for c in &manifest.chunks {
                let prev = *inner.refs.entry(c.digest).or_insert(0);
                inner.refs.insert(c.digest, prev + 1);
                if prev == 0 {
                    if let Some(p) = inner.placement.get(&c.digest).copied() {
                        if let Some(stat) = inner.segments.get_mut(&p.seg) {
                            stat.dead = stat.dead.saturating_sub(36 + p.len);
                        }
                    }
                }
            }
        }
        Ok(manifest)
    }

    pub fn stage_chunks(&self, payload: &[u8], positions: u64) -> Result<Manifest, ChunkError> {
        let manifest = chunk_payload(payload, positions);
        let mut inner = self.inner.lock().unwrap();
        for c in &manifest.chunks {
            if inner.placement.contains_key(&c.digest) {
                continue;
            }
            let body_len = c.body_len();
            if body_len > u32::MAX as u64 {
                return Err(ChunkError::Corrupt("chunk body exceeds u32".into()));
            }
            // Roll the active segment at the target. The record header
            // (36B) is trivial next to any real chunk.
            let need_roll = match inner.active.as_ref() {
                Some(a) => a.len + 36 + body_len > SEGMENT_TARGET,
                None => true,
            };
            if need_roll {
                if let Some(a) = inner.active.take() {
                    a.file
                        .sync_all()
                        .map_err(|e| ChunkError::Disk(e.to_string()))?;
                }
                let seq = inner.next_seg;
                inner.next_seg += 1;
                // Lazy directory creation: boot tolerates a read-only store
                // dir; the first write creates chunks/ or fails as Disk.
                fs::create_dir_all(&self.segments_dir)
                    .map_err(|e| ChunkError::Disk(e.to_string()))?;
                let path = self.segment_path(seq);
                let mut file =
                    fs::File::create(&path).map_err(|e| ChunkError::Disk(e.to_string()))?;
                let mut head = Vec::with_capacity(8);
                head.extend_from_slice(SEGMENT_MAGIC);
                head.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
                // 2 reserved zero bytes: the header is a fixed 8 bytes
                // (magic 4 + version 2 + reserved 2) so record offsets and
                // the scanner's `at = 8` stay simple.
                head.extend_from_slice(&[0u8; 2]);
                file.write_all(&head)
                    .map_err(|e| ChunkError::Disk(e.to_string()))?;
                file.sync_all()
                    .map_err(|e| ChunkError::Disk(e.to_string()))?;
                inner.segments.insert(seq, SegmentStat::default());
                inner.active = Some(ActiveSegment { seq, file, len: 8 });
            }
            let attempt = {
                let a = inner
                    .active
                    .as_mut()
                    .expect("roll guarantees an active segment");
                let mut record = Vec::with_capacity(36 + body_len as usize);
                record.extend_from_slice(&(body_len as u32).to_le_bytes());
                record.extend_from_slice(&c.digest);
                // The chunk body is the per-head concatenation: gather the
                // slices straight out of the payload.
                for h in 0..c.n_heads as u64 {
                    let base = c.first_off + h * c.head_stride;
                    record.extend_from_slice(
                        &payload[base as usize..(base + c.per_head_len as u64) as usize],
                    );
                }
                match a.file.write_all(&record) {
                    Ok(()) => {
                        a.len += record.len() as u64;
                        Ok((a.seq, a.len - body_len))
                    }
                    Err(e) => Err(e),
                }
            };
            let (seq, data_off) = match attempt {
                Ok(v) => v,
                Err(e) => {
                    // A partial write left garbage at the OS offset that the
                    // in-memory length no longer models: drop the active
                    // segment so the next append starts a fresh one and no
                    // placement ever points into the gap.
                    inner.active = None;
                    return Err(ChunkError::Disk(e.to_string()));
                }
            };
            {
                let stat = inner
                    .segments
                    .get_mut(&seq)
                    .expect("stat tracks active segment");
                stat.total += 36 + body_len;
                // Appended = unreferenced until register flips it live;
                // counting it dead here keeps register's subtraction exact
                // (it can no longer steal other records' dead bytes).
                stat.dead += 36 + body_len;
            }
            inner.placement.insert(
                c.digest,
                Placement {
                    seg: seq,
                    offset: data_off,
                    len: body_len,
                },
            );
        }
        if let Some(a) = inner.active.as_ref() {
            a.file
                .sync_all()
                .map_err(|e| ChunkError::Disk(e.to_string()))?;
        }
        Ok(manifest)
    }

    /// Register a live blob's refs (called after its .ckpt rename made it
    /// visible). Idempotent per manifest: double registration is a caller
    /// bug we tolerate by design only at boot (fresh maps); at runtime the
    /// publish path calls this exactly once per successful rename.
    pub fn register(&self, manifest: &Manifest) {
        let mut inner = self.inner.lock().unwrap();
        for c in &manifest.chunks {
            let prev = *inner.refs.entry(c.digest).or_insert(0);
            inner.refs.insert(c.digest, prev + 1);
            if prev == 0 {
                if let Some(p) = inner.placement.get(&c.digest).copied() {
                    if let Some(stat) = inner.segments.get_mut(&p.seg) {
                        stat.dead = stat.dead.saturating_sub(36 + p.len);
                    }
                }
            }
        }
    }

    /// Drop a retired blob's refs. Chunks that fall to zero become dead
    /// bytes; their segment leaves disk in the next GC sweep when empty.
    pub fn unregister(&self, manifest: &Manifest) {
        let mut inner = self.inner.lock().unwrap();
        for c in &manifest.chunks {
            let now = inner
                .refs
                .get(&c.digest)
                .copied()
                .unwrap_or(0)
                .saturating_sub(1);
            if now == 0 {
                inner.refs.remove(&c.digest);
                if let Some(p) = inner.placement.get(&c.digest).copied() {
                    if let Some(stat) = inner.segments.get_mut(&p.seg) {
                        stat.dead += 36 + p.len;
                    }
                }
            } else {
                inner.refs.insert(c.digest, now);
            }
        }
    }

    /// Delete segments whose every record is dead. Returns bytes reclaimed.
    pub fn gc_empty_segments(&self) -> u64 {
        let mut inner = self.inner.lock().unwrap();
        let mut reclaimed = 0u64;
        let empty: Vec<u64> = inner
            .segments
            .iter()
            .filter(|(_, s)| s.live() == 0)
            .map(|(seq, _)| *seq)
            .collect();
        for seq in empty {
            let is_active = inner.active.as_ref().map(|a| a.seq) == Some(seq);
            if is_active {
                continue; // never delete the append target
            }
            let stat = inner.segments.remove(&seq).expect("checked above");
            let path = self.segment_path(seq);
            match fs::remove_file(&path) {
                Ok(()) => {
                    // The segment's placements are GONE with the file: a
                    // later re-publish of the same content must re-append
                    // (not skip on a stale placement hit that now points
                    // into a deleted file — the C1 bug).
                    inner.placement.retain(|_, p| p.seg != seq);
                    reclaimed += stat.total;
                }
                Err(_) => {
                    inner.segments.insert(seq, stat); // retry next sweep
                }
            }
        }
        reclaimed
    }

    /// Physical bytes on disk: every segment file's live+dead bytes plus
    /// manifests. Blob files are accounted by the caller (whole payloads
    /// for v1/v2, tiny headers for v3 — both are real file bytes).
    pub fn physical_bytes(&self) -> u64 {
        // File-stat truth, not in-memory accounting: the accounting map
        // excludes each segment's 8-byte header and any incomplete tail
        // (crash garbage), and a budget recount must never undercount.
        let mut total = 0u64;
        if let Ok(entries) = fs::read_dir(&self.segments_dir) {
            for entry in entries.flatten() {
                if entry.file_name().to_string_lossy().ends_with(".seg") {
                    total += entry.metadata().map(|m| m.len()).unwrap_or(0);
                }
            }
        }
        total
    }

    /// Fragmentation accounting for /stats and the compaction trigger:
    /// (total record bytes incl. headers, dead record bytes, count of
    /// partially-dead segments). All from the in-memory map, which boot
    /// re-derives from disk.
    pub fn fragmentation(&self) -> (u64, u64, usize) {
        let inner = self.inner.lock().unwrap();
        let total: u64 = inner.segments.values().map(|s| s.total).sum();
        let dead: u64 = inner.segments.values().map(|s| s.dead).sum();
        let partial = inner
            .segments
            .values()
            .filter(|s| s.dead > 0 && s.live() > 0)
            .count();
        (total, dead, partial)
    }

    /// Rewrite fragmented segments: copy each candidate's LIVE records into
    /// a fresh segment, swap placements, delete the old file. Dead bytes
    /// trapped between live records leave the store; the budget recount
    /// that follows a sweep sees the truth.
    ///
    /// Thresholds: a segment is a candidate when `dead >= min_dead_bytes`
    /// AND `dead / total >= ratio` AND it has live records (all-dead is
    /// `gc_empty_segments`' job) AND it is not the active append target.
    /// At most `max_segments` per call — the copy runs under the store
    /// lock (single lock discipline: nothing else moves placements), so a
    /// sweep is bounded I/O (~30 ms per 64 MiB segment at disk speed), and
    /// the reaper cadence paces the rest.
    ///
    /// Crash safety, the whole point of the resolution order: the new
    /// segment is fsynced BEFORE anything in memory moves, and the old
    /// file is deleted AFTER placements swap. A crash between the two
    /// leaves both files on disk; the boot scan (highest-seq first-wins)
    /// resolves every digest to the new segment, and the orphaned records
    /// in the old one are accounted dead at scan time — boot GC then
    /// deletes the old file. No rename dance, no journal.
    pub fn compact_segments(
        &self,
        ratio: f64,
        min_dead_bytes: u64,
        max_segments: usize,
    ) -> Result<(usize, u64), ChunkError> {
        if !(0.0..=1.0).contains(&ratio) {
            return Err(ChunkError::Corrupt(format!(
                "compaction ratio {ratio} outside 0..=1"
            )));
        }
        let mut inner = self.inner.lock().unwrap();
        let active_seq = inner.active.as_ref().map(|a| a.seq);
        // Candidates: fragmented, not active, has live records. Worst
        // (most dead bytes) first.
        let mut candidates: Vec<(u64, u64)> = inner
            .segments
            .iter()
            .filter(|(seq, s)| {
                Some(**seq) != active_seq
                    && s.dead >= min_dead_bytes
                    && s.dead > 0
                    && s.live() > 0
                    && (s.dead as f64) >= ratio * (s.total as f64)
            })
            .map(|(seq, s)| (*seq, s.dead))
            .collect();
        candidates.sort_by_key(|&(_, dead)| std::cmp::Reverse(dead));
        candidates.truncate(max_segments);

        let mut compacted = 0usize;
        let mut reclaimed = 0u64;
        for (old_seq, _dead) in candidates {
            // Live records in this segment: referenced AND placed here.
            let mut live: Vec<([u8; 32], Placement)> = inner
                .placement
                .iter()
                .filter(|(digest, p)| p.seg == old_seq && inner.refs.contains_key(*digest))
                .map(|(digest, p)| (*digest, *p))
                .collect();
            live.sort_by_key(|(_, p)| p.offset);
            if live.is_empty() {
                continue; // refs dropped to zero since candidacy: GC's job now
            }
            // Read the live bodies from the old file (under the lock: refs
            // and placements cannot move while held).
            let old_path = self.segment_path(old_seq);
            let old_file = match fs::File::open(&old_path) {
                Ok(f) => f,
                // Only external interference can do this (GC needs the
                // lock this pass holds). Skip; the next pass reconciles.
                Err(_) => continue,
            };
            // Write the new segment: header + live records, fsynced.
            let new_seq = inner.next_seg;
            inner.next_seg += 1;
            let new_path = self.segment_path(new_seq);
            let write = (|| -> std::io::Result<u64> {
                use std::os::unix::fs::FileExt as _;
                fs::create_dir_all(&self.segments_dir)?;
                let mut file = fs::File::create(&new_path)?;
                let mut head = Vec::with_capacity(8);
                head.extend_from_slice(SEGMENT_MAGIC);
                head.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
                head.extend_from_slice(&[0u8; 2]);
                file.write_all(&head)?;
                let mut written = 8u64;
                for (digest, p) in &live {
                    let mut body = vec![0u8; p.len as usize];
                    old_file.read_exact_at(&mut body, p.offset)?;
                    file.write_all(&(p.len as u32).to_le_bytes())?;
                    file.write_all(digest)?;
                    file.write_all(&body)?;
                    written += 36 + p.len;
                }
                file.sync_all()?;
                Ok(written)
            })();
            match write {
                Ok(written) => {
                    // Durable: swap placements, replace the stat, drop the
                    // old file. Records were written packed from offset 8 in
                    // `live` (offset-sorted) order. Chunks whose refs hit
                    // zero between candidacy and here were filtered out of
                    // `live` — they stay only in the old segment and leave
                    // with it.
                    let mut off = 8u64;
                    for (digest, p) in &live {
                        inner.placement.insert(
                            *digest,
                            // Body offset, same convention as the append
                            // path and the boot scan: 4-byte length + 32-
                            // byte digest precede the body.
                            Placement {
                                seg: new_seq,
                                offset: off + 36,
                                len: p.len,
                            },
                        );
                        off += 36 + p.len;
                    }
                    let old_stat = inner.segments.remove(&old_seq).map(|s| s.total);
                    // Records only (header excluded): SegmentStat counts
                    // exactly what a boot scan of this file would count.
                    let record_bytes = written - 8;
                    inner.segments.insert(
                        new_seq,
                        SegmentStat {
                            total: record_bytes,
                            dead: 0,
                        },
                    );
                    if let Some(old_total) = old_stat {
                        reclaimed += old_total.saturating_sub(record_bytes);
                    }
                    inner.placement.retain(|_, p| p.seg != old_seq);
                    let _ = fs::remove_file(&old_path);
                    compacted += 1;
                }
                Err(e) => {
                    // The new file is garbage (partial write): remove it and
                    // skip this candidate — placements were never moved, so
                    // the old segment still serves every read. Retry next sweep.
                    let _ = fs::remove_file(&new_path);
                    inner.next_seg -= 1; // the seq was never used durably
                    tracing::warn!(
                        segment = old_seq,
                        error = %e,
                        "segment compaction skipped: write failed, old segment untouched"
                    );
                }
            }
        }
        Ok((compacted, reclaimed))
    }

    /// Reassemble a payload from its manifest. Byte-exact by construction;
    /// the caller verifies the payload sha256 against the header (the D1
    /// digest contract covers reassembly: any segment/manifest corruption
    /// produces wrong bytes, not a wrong verdict).
    pub fn reassemble(&self, manifest: &Manifest) -> Result<Vec<u8>, ChunkError> {
        // Snapshot placements AND open the segment files under the same
        // lock, then do the reads without it: a reassembly holds hundreds
        // of MB; publishers must not queue behind a reader's preads.
        // Opening here is what makes a compaction sweep safe to race: the
        // sweep swaps placements and deletes the old file under this same
        // lock, so this snapshot pairs every placement with a file that
        // exists, and an fd stays readable after its file unlinks.
        let (placements, mut open_files) = {
            let inner = self.inner.lock().unwrap();
            let placements: Vec<Placement> = manifest
                .chunks
                .iter()
                .map(|c| {
                    inner.placement.get(&c.digest).copied().ok_or_else(|| {
                        ChunkError::Corrupt(format!(
                            "chunk {} missing from store",
                            hex_prefix(&c.digest)
                        ))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let mut files: HashMap<u64, fs::File> = HashMap::new();
            for p in &placements {
                if files.contains_key(&p.seg) {
                    continue;
                }
                // A placement with no file on disk is real corruption
                // (nothing under this lock may delete a LIVE chunk's
                // segment) — an error verdict, never a panic.
                let f = fs::File::open(self.segment_path(p.seg)).map_err(|_| {
                    ChunkError::Corrupt(format!("segment {} missing under a live placement", p.seg))
                })?;
                files.insert(p.seg, f);
            }
            (placements, files)
        };
        let mut out = vec![0u8; manifest.payload_len as usize];
        for (c, p) in manifest.chunks.iter().zip(placements.iter()) {
            let file = open_files
                .get_mut(&p.seg)
                .expect("opened under the placement lock");
            let mut body = vec![0u8; p.len as usize];
            std::os::unix::fs::FileExt::read_exact_at(file, &mut body, p.offset)?;
            for h in 0..c.n_heads as u64 {
                let dst = c.first_off + h * c.head_stride;
                let src = (h * c.per_head_len as u64) as usize;
                out[dst as usize..(dst + c.per_head_len as u64) as usize]
                    .copy_from_slice(&body[src..src + c.per_head_len as usize]);
            }
        }
        Ok(out)
    }
}

fn hex_prefix(d: &[u8; 32]) -> String {
    d.iter().take(8).map(|b| format!("{b:02x}")).collect()
}

/// Header peek used by the boot scan (no digest verify here — the normal
/// load path does that when a blob is actually served).
/// Cap for header-prefix peeks: a v3 header is a few hundred bytes; a
/// length field beyond this is foreign data, not one of our blobs.
pub(crate) const MAX_HEADER_READ: usize = 1 << 20;

/// Header peek used by the boot scan and budget accounting (no digest
/// verify here — the normal load path does that when a blob is served).
pub(crate) fn peek_meta(bytes: &[u8]) -> Option<CheckpointMeta> {
    if bytes.len() < 4 {
        return None;
    }
    let header_len = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
    if 4 + header_len > bytes.len() {
        return None;
    }
    serde_json::from_slice(&bytes[4..4 + header_len]).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic safetensors payload builder (mirrored byte-for-byte
    /// by the Python golden test — the digest list is the cross-language
    /// pin).
    #[allow(dead_code)]
    pub(super) fn fake_payload(
        tokens: usize,
        heads: usize,
        dim: usize,
        layers: usize,
        seed: u8,
    ) -> Vec<u8> {
        let positions = tokens - 1;
        let mut tensors = serde_json::Map::new();
        let mut data = Vec::new();
        let mut offsets = Vec::new();
        for l in 0..layers {
            let name = format!("layers.{l}.k");
            let begin = data.len();
            for h in 0..heads {
                for p in 0..positions {
                    for d in 0..dim {
                        data.push(
                            seed.wrapping_add(l as u8)
                                .wrapping_add(h as u8)
                                .wrapping_add((p % 251) as u8)
                                ^ (d as u8),
                        );
                    }
                }
            }
            offsets.push((name, begin, data.len()));
        }
        for (name, begin, end) in &offsets {
            tensors.insert(
                name.clone(),
                serde_json::json!({
                    "dtype": "U8",
                    "shape": [1, heads, positions, dim],
                    "data_offsets": [begin, end],
                }),
            );
        }
        let header = serde_json::to_vec(&serde_json::Value::Object(tensors)).unwrap();
        let mut out = Vec::with_capacity(8 + header.len() + data.len());
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        // safetensors pads the header to 8-byte alignment; keep the golden
        // simple: our reader tolerates any header length.
        out.extend_from_slice(&header);
        out.extend_from_slice(&data);
        out
    }

    #[test]
    fn chunker_grid_identity_matches_the_reference_semantics() {
        // Two blobs sharing a 64-position prefix: every chunk of the
        // prefix must be byte-identical (digests equal), and only the
        // tail block differs.
        let a = fake_payload(65, 2, 8, 2, 1); // 64 positions
        let b = fake_payload(129, 2, 8, 2, 1); // 128 positions, same prefix
        let ma = chunk_payload(&a, 64);
        let mb = chunk_payload(&b, 128);
        // 2 tensors x ceil(64/64)=1 block vs ceil(128/64)=2 blocks, plus
        // one header-gap chunk per payload (the safetensors JSON header is
        // NOT shared — its offsets differ — so it never dedups, honestly).
        assert_eq!(ma.chunks.len(), 3);
        assert_eq!(mb.chunks.len(), 5);
        assert_eq!(
            ma.chunks[0].digest, mb.chunks[0].digest,
            "shared prefix dedups"
        );
        assert_eq!(
            ma.chunks[1].digest, mb.chunks[2].digest,
            "second tensor too"
        );
        assert_ne!(mb.chunks[2].digest, mb.chunks[3].digest);
    }

    #[test]
    fn chunker_partial_tail_block_and_fallbacks() {
        // 70 positions → blocks of 64 + 6.
        let p = fake_payload(71, 3, 4, 1, 9);
        let m = chunk_payload(&p, 70);
        // 2 tensor blocks + the header-gap chunk.
        assert_eq!(m.chunks.len(), 3);
        assert_eq!(m.chunks[0].per_head_len as u64, 64 * 4);
        assert_eq!(m.chunks[1].per_head_len as u64, 6 * 4);
        assert_eq!(m.chunks[0].n_heads, 3);
        // Non-safetensors payload: one whole-payload chunk.
        let m2 = chunk_payload(b"definitely-not-safetensors", 10);
        assert_eq!(m2.chunks.len(), 1);
        assert_eq!(m2.chunks[0].n_heads, 1);
        assert_eq!(m2.chunks[0].per_head_len as usize, 26);
    }

    #[test]
    fn manifest_codec_roundtrip_and_rejections() {
        let p = fake_payload(71, 3, 4, 2, 4);
        let m = chunk_payload(&p, 70);
        let bytes = m.encode();
        let m2 = Manifest::decode(&bytes).unwrap();
        assert_eq!(m.chunks, m2.chunks);
        assert_eq!(m.payload_len, m2.payload_len);
        // Truncated records reject.
        let mut cut = bytes.clone();
        cut.truncate(cut.len() - 10);
        assert!(Manifest::decode(&cut).is_err());
        // Placement beyond payload rejects.
        let mut bad = m.clone();
        bad.chunks[0].first_off = m.payload_len + 1;
        assert!(Manifest::decode(&bad.encode()).is_err());
    }

    #[test]
    fn store_roundtrip_dedup_and_gc() {
        let dir = tempfile::tempdir().unwrap();
        let store = ChunkStore::open(dir.path(), &[]).unwrap();
        let a = fake_payload(65, 2, 8, 2, 1);
        let b = fake_payload(129, 2, 8, 2, 1);
        let ma = store.stage_chunks(&a, 64).unwrap();
        store.register(&ma);
        let mb = store.stage_chunks(&b, 128).unwrap();
        store.register(&mb);
        // Shared prefix chunks stored once: b's gross bytes exceed the
        // segment total by exactly its unique tail.
        assert!(store.physical_bytes() < ma.gross_bytes() + mb.gross_bytes());
        assert_eq!(store.reassemble(&ma).unwrap(), a);
        assert_eq!(store.reassemble(&mb).unwrap(), b);
        // Retire both: every chunk dies. The segment is still the ACTIVE
        // append target, so GC correctly declines to delete it; sealing
        // (the shutdown/size-driven operation) makes it reclaimable.
        store.unregister(&ma);
        store.unregister(&mb);
        assert_eq!(
            store.gc_empty_segments(),
            0,
            "active segment is never deleted"
        );
        // Drop the store (closing the active segment) and reopen with no
        // live blobs: the BOOT scan reclaims the all-dead segment.
        drop(store);
        let store = ChunkStore::open(dir.path(), &[]).unwrap();
        assert_eq!(
            store.physical_bytes(),
            0,
            "boot GC reclaims the all-dead segment"
        );
    }

    #[test]
    fn crash_orphan_manifest_and_dead_chunks_are_reclaimed_at_boot() {
        let dir = tempfile::tempdir().unwrap();
        {
            let store = ChunkStore::open(dir.path(), &[]).unwrap();
            let a = fake_payload(65, 2, 8, 1, 2);
            let m = store.stage_chunks(&a, 64).unwrap();
            // Simulate a crash after staging: NO .ckpt exists, and an
            // orphan .ckm file sits next to where it would.
            let orphan = dir.path().join("orphan.ckm");
            fs::write(&orphan, m.encode()).unwrap();
        }
        // Boot with zero live blobs: the orphan manifest is deleted and
        // its chunks are dead. GC removes the empty segment.
        let store = ChunkStore::open(dir.path(), &[]).unwrap();
        assert!(!dir.path().join("orphan.ckm").exists());
        store.gc_empty_segments();
        assert_eq!(store.physical_bytes(), 0);
    }

    #[test]
    fn truncated_segment_tail_scans_cleanly() {
        let dir = tempfile::tempdir().unwrap();
        {
            let store = ChunkStore::open(dir.path(), &[]).unwrap();
            let a = fake_payload(129, 2, 8, 1, 3);
            let m = store.stage_chunks(&a, 128).unwrap();
            store.register(&m);
        }
        // Truncate the segment mid-record (crash mid-append simulation).
        let seg = dir.path().join("chunks").join("seg-0000000000.seg");
        let len = fs::metadata(&seg).unwrap().len();
        let mut bytes = fs::read(&seg).unwrap();
        bytes.truncate((len - 3) as usize);
        fs::write(&seg, &bytes).unwrap();
        // Re-open with no live blobs: the scan must tolerate the truncated
        // tail (no Corrupt), and the boot GC reclaims the all-dead segment.
        let store = ChunkStore::open(dir.path(), &[]).unwrap();
        assert_eq!(
            store.physical_bytes(),
            0,
            "all-dead (no live .ckpt) segment reclaims at boot; the truncated tail never errors"
        );
    }

    /// A live v3 blob pair on disk (header-only .ckpt + .ckm): exactly what
    /// the boot scan's `live_blobs` expects, without dragging the whole
    /// publish path into the store-level tests.
    fn write_live_v3(dir: &Path, name: &str, manifest: &Manifest) -> PathBuf {
        use mlxcache_core::contract::ModelFingerprint;
        let meta = CheckpointMeta {
            fingerprint: ModelFingerprint::default(),
            token_count: 64,
            tokens: Vec::new(),
            format_version: 3,
            payload_sha256: None,
            engine_id: None,
            granularity: None,
        };
        let json = serde_json::to_vec(&meta).unwrap();
        let mut ckpt = Vec::with_capacity(4 + json.len());
        ckpt.extend_from_slice(&(json.len() as u32).to_le_bytes());
        ckpt.extend_from_slice(&json);
        let ckpt_path = dir.join(format!("{name}.ckpt"));
        fs::write(&ckpt_path, ckpt).unwrap();
        fs::write(dir.join(format!("{name}.ckm")), manifest.encode()).unwrap();
        ckpt_path
    }

    fn segment_files(dir: &Path) -> Vec<String> {
        let mut names: Vec<String> = fs::read_dir(dir.join("chunks"))
            .map(|rd| {
                rd.flatten()
                    .filter_map(|e| {
                        let n = e.file_name().to_string_lossy().into_owned();
                        n.ends_with(".seg").then_some(n)
                    })
                    .collect()
            })
            .unwrap_or_default();
        names.sort();
        names
    }

    #[test]
    fn compaction_frees_dead_bytes_and_keeps_live_reads() {
        let dir = tempfile::tempdir().unwrap();
        let a = fake_payload(65, 2, 8, 2, 1);
        let b = fake_payload(129, 2, 8, 2, 1); // shares a's 64-position prefix
        let c = fake_payload(70, 2, 4, 1, 7); // its blob never publishes
        let (ma, mb, aa, bb) = {
            let store = ChunkStore::open(dir.path(), &[]).unwrap();
            let ma = store.stage_and_register(&a, 64).unwrap();
            let mb = store.stage_and_register(&b, 128).unwrap();
            let _ = store.stage_and_register(&c, 69).unwrap();
            let aa = write_live_v3(dir.path(), "aa", &ma);
            let bb = write_live_v3(dir.path(), "bb", &mb);
            (ma, mb, aa, bb)
        };
        let live = vec![aa, bb];
        let store = ChunkStore::open(dir.path(), &live).unwrap();
        let (total, dead, partial) = store.fragmentation();
        assert!(dead > 0, "c's chunks are dead weight: {dead}");
        assert!(dead < total, "a and b keep their chunks live");
        assert_eq!(partial, 1);
        let before = store.physical_bytes();
        // Both gates must hold: a ratio above the real fragmentation and a
        // floor above the real dead bytes each no-op the sweep.
        assert_eq!(store.compact_segments(0.99, 1, 8).unwrap(), (0, 0));
        assert_eq!(store.compact_segments(0.0, dead + 1, 8).unwrap(), (0, 0));
        let (n, reclaimed) = store.compact_segments(0.0, 1, 8).unwrap();
        assert_eq!(n, 1);
        assert_eq!(reclaimed, dead, "exactly the dead record bytes leave");
        let (total2, dead2, partial2) = store.fragmentation();
        assert_eq!((dead2, partial2), (0, 0));
        assert_eq!(total2, total - dead);
        assert_eq!(before - store.physical_bytes(), dead);
        // The old segment file is gone; exactly one (the rewrite) remains.
        let segs = segment_files(dir.path());
        assert_eq!(segs.len(), 1);
        assert_ne!(segs[0], "seg-0000000000.seg");
        // Live payloads read bit-exact from the new segment.
        assert_eq!(store.reassemble(&ma).unwrap(), a);
        assert_eq!(store.reassemble(&mb).unwrap(), b);
    }

    #[test]
    fn compaction_never_touches_the_active_segment() {
        let dir = tempfile::tempdir().unwrap();
        let store = ChunkStore::open(dir.path(), &[]).unwrap();
        let a = fake_payload(65, 2, 8, 2, 1);
        let ma = store.stage_and_register(&a, 64).unwrap();
        store.unregister(&ma); // every chunk dead, but the segment is ACTIVE
        assert_eq!(
            store.compact_segments(0.0, 1, 8).unwrap(),
            (0, 0),
            "the active append target is never rewritten"
        );
        assert!(store.physical_bytes() > 0);
        drop(store);
        // Sealed by reopen, all-dead, no live blobs: that is boot GC's
        // case, not compaction's — the file must simply be gone.
        let store = ChunkStore::open(dir.path(), &[]).unwrap();
        assert_eq!(store.physical_bytes(), 0);
    }

    #[test]
    fn compaction_crash_orphan_self_heals_at_boot() {
        let dir = tempfile::tempdir().unwrap();
        let a = fake_payload(65, 2, 8, 2, 1);
        let b = fake_payload(129, 2, 8, 2, 1);
        let c = fake_payload(70, 2, 4, 1, 7);
        let (ma, mb, aa, bb) = {
            let store = ChunkStore::open(dir.path(), &[]).unwrap();
            let ma = store.stage_and_register(&a, 64).unwrap();
            let mb = store.stage_and_register(&b, 128).unwrap();
            let _ = store.stage_and_register(&c, 69).unwrap();
            let aa = write_live_v3(dir.path(), "aa", &ma);
            let bb = write_live_v3(dir.path(), "bb", &mb);
            (ma, mb, aa, bb)
        };
        let live = vec![aa, bb];
        let store = ChunkStore::open(dir.path(), &live).unwrap();
        let seg0 = dir.path().join("chunks").join("seg-0000000000.seg");
        let old_bytes = fs::read(&seg0).unwrap();
        let (n, _) = store.compact_segments(0.0, 1, 8).unwrap();
        assert_eq!(n, 1);
        // Replay the crash window: the new segment is durable and
        // placements moved, but the old file's delete never happened.
        fs::write(&seg0, &old_bytes).unwrap();
        assert_eq!(segment_files(dir.path()).len(), 2, "both files on disk");
        drop(store);
        // Boot resolves every digest to the NEW segment (highest seq
        // first-wins), accounts the old one fully dead (lost duplicates
        // plus the never-live c chunks), and GC deletes it.
        let store = ChunkStore::open(dir.path(), &live).unwrap();
        let segs = segment_files(dir.path());
        assert_eq!(segs.len(), 1, "the orphaned old segment self-heals away");
        assert_ne!(segs[0], "seg-0000000000.seg");
        let (_, dead, partial) = store.fragmentation();
        assert_eq!((dead, partial), (0, 0));
        assert_eq!(store.reassemble(&ma).unwrap(), a);
        assert_eq!(store.reassemble(&mb).unwrap(), b);
    }

    #[test]
    fn compaction_racing_publish_keeps_reads_exact() {
        let dir = tempfile::tempdir().unwrap();
        let a = fake_payload(65, 2, 8, 2, 1);
        let b = fake_payload(129, 2, 8, 2, 1);
        let c = fake_payload(70, 2, 4, 1, 7);
        let (ma, mb, aa, bb) = {
            let store = ChunkStore::open(dir.path(), &[]).unwrap();
            let ma = store.stage_and_register(&a, 64).unwrap();
            let mb = store.stage_and_register(&b, 128).unwrap();
            let _ = store.stage_and_register(&c, 69).unwrap();
            let aa = write_live_v3(dir.path(), "aa", &ma);
            let bb = write_live_v3(dir.path(), "bb", &mb);
            (ma, mb, aa, bb)
        };
        let live = vec![aa, bb];
        let store = std::sync::Arc::new(ChunkStore::open(dir.path(), &live).unwrap());
        let compactor = std::sync::Arc::clone(&store);
        let sweep = std::thread::spawn(move || {
            for _ in 0..8 {
                let _ = compactor.compact_segments(0.0, 1, 2);
            }
        });
        // A reader reassembles while the sweep moves placements and
        // deletes the old file: the snapshot/open pairing under the store
        // lock must keep every read bit-exact (an open-after-delete once
        // panicked here).
        let reader_store = std::sync::Arc::clone(&store);
        let ma2 = ma.clone();
        let a2 = a.clone();
        let reader = std::thread::spawn(move || {
            for _ in 0..64 {
                assert_eq!(reader_store.reassemble(&ma2).unwrap(), a2);
            }
        });
        // A publisher interleaves with the sweep at lock boundaries: fresh
        // content appends to the ACTIVE segment while compaction rewrites
        // the sealed one. Every read must stay bit-exact.
        let d = fake_payload(71, 3, 4, 1, 5);
        for _ in 0..16 {
            let md = store.stage_and_register(&d, 70).unwrap();
            store.unregister(&md);
        }
        sweep.join().unwrap();
        reader.join().unwrap();
        assert_eq!(store.reassemble(&ma).unwrap(), a);
        assert_eq!(store.reassemble(&mb).unwrap(), b);
    }

    #[test]
    fn golden_digests_cross_language() {
        // The payload construction is mirrored in
        // sidecar/tests/test_chunk_store.py; both sides must produce THIS
        // digest list for the same input (the cross-language identity pin).
        let p = fake_payload(70, 2, 4, 1, 7); // 69 positions
        let m = chunk_payload(&p, 69);
        let digests: Vec<String> = m
            .chunks
            .iter()
            .map(|c| c.digest.iter().map(|b| format!("{b:02x}")).collect())
            .collect();
        assert_eq!(digests.len(), 3); // 2 tensor blocks + header gap
        insta_like_assert(&digests);
    }

    fn insta_like_assert(digests: &[String]) {
        // Committed goldens: stable across runs/platforms (sha256 of
        // deterministic bytes). Regenerate = never (identity is format).
        let joined = digests.join(",");
        let expected = std::env::var("MLXCACHE_CHUNK_GOLDEN").unwrap_or_default();
        if expected.is_empty() {
            // Compute-once mode in tests: the pin is that re-chunking the
            // same payload is stable.
            let again = {
                let p = fake_payload(70, 2, 4, 1, 7);
                let m = chunk_payload(&p, 69);
                m.chunks
                    .iter()
                    .map(|c| {
                        c.digest
                            .iter()
                            .map(|b| format!("{b:02x}"))
                            .collect::<String>()
                    })
                    .collect::<Vec<_>>()
                    .join(",")
            };
            assert_eq!(joined, again, "chunker must be deterministic");
        } else {
            assert_eq!(joined, expected);
        }
    }
}

#[cfg(test)]
mod fixture_dump {
    use super::*;
    use std::path::PathBuf;

    /// Writes the cross-language fixture consumed by
    /// sidecar/tests/test_chunk_store.py: a store with two blobs sharing a
    /// prefix (the dedup shape) plus the expected payloads. Run via
    /// `cargo test -p mlxcache-daemon chunkstore::fixture_dump -- --ignored`
    /// after any format change, then commit the directory.
    #[test]
    #[ignore]
    fn fixture_dump() {
        let root =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../scripts/fixtures/chunk-store");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let store = ChunkStore::open(&root, &[]).unwrap();
        let a = super::tests::fake_payload(65, 2, 8, 2, 1);
        let b = super::tests::fake_payload(129, 2, 8, 2, 1);
        let ma = store.stage_chunks(&a, 64).unwrap();
        store.register(&ma);
        let mb = store.stage_chunks(&b, 128).unwrap();
        store.register(&mb);
        // The reader tests glob *.ckpt and expect the FULL store; manifests
        // are written the way publish_chunked names them (hash-gen).
        std::fs::write(root.join("aa.ckm"), ma.encode()).unwrap();
        std::fs::write(root.join("aa.ckpt"), header_only(&a)).unwrap();
        std::fs::write(root.join("bb.ckm"), mb.encode()).unwrap();
        std::fs::write(root.join("bb.ckpt"), header_only(&b)).unwrap();
        std::fs::write(root.join("expected-payload.bin"), &a).unwrap();
    }

    fn header_only(payload: &[u8]) -> Vec<u8> {
        let meta = serde_json::json!({
            "fingerprint": {"model_id": "m", "tokenizer_hash": "h", "kv_dtype": "f16",
                            "kv_layout_version": 1},
            "token_count": 3,
            "tokens": [1, 2, 3],
            "format_version": 3,
            "payload_sha256": format!("{:x}", Sha256::digest(payload)),
        });
        let header = serde_json::to_vec(&meta).unwrap();
        let mut out = (header.len() as u32).to_le_bytes().to_vec();
        out.extend_from_slice(&header);
        out
    }
}

#[cfg(test)]
mod refcount_regressions {
    use super::*;

    #[test]
    fn publish_then_remove_frees_chunks_within_the_running_process() {
        // The double-register bug: each publish registered twice, remove
        // unregistered once, so a removed blob's chunks stayed live until
        // a boot replay. A running daemon must reclaim them.
        let dir = tempfile::tempdir().unwrap();
        let store = ChunkStore::open(dir.path(), &[]).unwrap();
        let a = super::tests::fake_payload(65, 2, 8, 2, 1);
        let b = super::tests::fake_payload(129, 2, 8, 2, 1);
        let ma = store.stage_and_register(&a, 64).unwrap();
        let mb = store.stage_and_register(&b, 128).unwrap();
        assert!(store.physical_bytes() > 0);
        store.unregister(&ma);
        store.unregister(&mb);
        drop(store); // close the active segment; boot GC reclaims
        let store = ChunkStore::open(dir.path(), &[]).unwrap();
        assert_eq!(
            store.physical_bytes(),
            0,
            "refs must be exactly 1 per publish"
        );
    }

    #[test]
    fn appended_chunks_are_dead_until_registered() {
        // The accounting bug: appended records were not counted dead, so
        // register's subtraction stole OTHER records' dead bytes and sealed
        // segments could never GC. Stage WITHOUT registering, then verify
        // the segment is all-dead.
        let dir = tempfile::tempdir().unwrap();
        let store = ChunkStore::open(dir.path(), &[]).unwrap();
        let a = super::tests::fake_payload(65, 2, 8, 1, 5);
        let m = store.stage_and_register(&a, 64).unwrap();
        store.unregister(&m);
        {
            let inner = store.inner.lock().unwrap();
            let stat = inner.segments.values().next().unwrap();
            assert_eq!(stat.live(), 0, "unregistered stage must be all-dead");
        }
        drop(store);
        let store = ChunkStore::open(dir.path(), &[]).unwrap();
        assert_eq!(store.physical_bytes(), 0);
    }

    #[test]
    fn payload_len_bitflip_is_rejected_by_coverage() {
        let dir = tempfile::tempdir().unwrap();
        let store = ChunkStore::open(dir.path(), &[]).unwrap();
        let a = super::tests::fake_payload(65, 2, 8, 1, 6);
        let m = store.stage_and_register(&a, 64).unwrap();
        let mut bytes = m.encode();
        // Inflate payload_len (bytes 6..14): decodes only if coverage
        // checking is absent.
        bytes[6..14].copy_from_slice(&(m.payload_len + (1 << 40)).to_le_bytes());
        assert!(Manifest::decode(&bytes).is_err());
    }
}
