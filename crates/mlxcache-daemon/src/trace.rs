//! Trace capture (step 5, success criteria 3-4): JSONL one-record-per-request
//! exporter for offline replay scoring.
//!
//! The honesty criterion (D1: "count all traffic, including warming") means a
//! replay harness must see EVERY request the daemon served, in order, with the
//! verdict and reuse it actually got. `log_request` (tracing) is interleaved
//! with all other daemon events and best-effort; the trace file is dedicated
//! and strictly ordered, written by a single background thread through a
//! bounded channel: a slow/faulted sink can never block a request, and a
//! browse of /stats stays the live view while the file is the replay source.
//!
//! Shape (one JSON object per line):
//! `{"ts_ms": <unix ms>, "model": "...", "model_hash": ..., "n_tokens": N,
//!   "verdict": "hit|miss|partial", "matched_tokens": N, "prefill_from": N,
//!   "messages": [...the exact request payload...]}`
//!
//! `messages` embeds the OpenAI-style payload verbatim so the replay harness
//! re-sends the EXACT request: same text through the same tokenizer gives the
//! same tokens (deterministic), so a replay reproduces the workload faithfully.
//! PRIVACY: the file therefore contains full prompt text — an operator only
//! ~raises~ this via MLXCACHE_TRACE on files they own.
//!
//! Capture is opt-in via MLXCACHE_TRACE. Replay lives in
//! scripts/replay_trace.py (turns records back into requests, tallies what a
//! daemon run scored).

use mlxcache_core::policy::{covered_kv_tokens, CacheVerdict, PolicyDecision};
use serde::Serialize;
use std::io::Write;
use std::sync::mpsc::{Receiver, SyncSender};
use std::time::{SystemTime, UNIX_EPOCH};

/// Record depth: a burst of in-flight requests is bounded by the daemon's own
/// concurrency; 4096 is generous headroom. Memory bound is records, not bytes:
/// each record embeds the verbatim messages payload (a 20K-token conversation
/// is ~0.1–1 MB), so a stalled writer can buffer up to a few GB in the worst
/// case. Accepted: traces are an opt-in diagnostic (MLXCACHE_TRACE unset pays
/// nothing), the writer's only slow step is a userspace flush per record, and
/// byte-budgeting the channel would drop whole conversations mid-word —
/// a much worse replay story than late delivery.
const CHANNEL_BOUND: usize = 4096;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceRecord {
    pub ts_ms: u64,
    /// Raw model id — replay needs a servable value to POST.
    pub model: String,
    /// Fingerprint attribution: a multi-model trace stays attributable to the
    /// exact tokenizer/kv-config even after replay-side rename/serving.
    pub model_hash: u64,
    pub n_tokens: u32,
    /// hit | miss | partial
    pub verdict: TraceVerdict,
    pub matched_tokens: u32,
    /// covered KV the request actually reused (tokens[:-1] definition, the
    /// same value /stats and prefill_from report).
    pub prefill_from: u32,
    /// The exact request payload (OpenAI-style messages array, echoed as
    /// JSON) — the replay source. `Null` only for synthetic/unit constructs.
    pub messages: serde_json::Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraceVerdict {
    Hit,
    Miss,
    Partial,
}

impl TraceVerdict {
    fn as_str(self) -> &'static str {
        match self {
            TraceVerdict::Hit => "hit",
            TraceVerdict::Miss => "miss",
            TraceVerdict::Partial => "partial",
        }
    }
}

fn from_decision(v: CacheVerdict) -> TraceVerdict {
    match v {
        CacheVerdict::Hit => TraceVerdict::Hit,
        CacheVerdict::Miss => TraceVerdict::Miss,
        CacheVerdict::Partial => TraceVerdict::Partial,
    }
}

impl TraceRecord {
    pub fn from_decision(model_hash: u64, decision: &PolicyDecision) -> Self {
        TraceRecord {
            ts_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
            model: String::new(),
            model_hash,
            n_tokens: decision.request_tokens as u32,
            verdict: from_decision(decision.verdict),
            matched_tokens: decision.matched_tokens as u32,
            prefill_from: covered_kv_tokens(decision.verdict, decision.matched_tokens) as u32,
            messages: serde_json::Value::Null,
        }
    }

    /// Full record for live capture (step 5): model id + the exact request
    /// payload, so replay re-sends byte-equivalent requests.
    pub fn from_request(
        model_hash: u64,
        model: &str,
        messages: &serde_json::Value,
        decision: &PolicyDecision,
    ) -> Self {
        let mut rec = Self::from_decision(model_hash, decision);
        rec.model = model.to_string();
        // The messages array is echoed verbatim (NOT re-stringified as an
        // escaped string) so the replay harness reads it back as JSON and
        // rebuilds the request 1:1.
        rec.messages = messages.clone();
        rec
    }

    /// One compact JSON line via serde (no newline; caller appends). Field
    /// order is declaration order — the replay harness only relies on names,
    /// but the flat hand-rolled format() is gone because nested message JSON
    /// needs real escaping.
    pub fn to_json(&self) -> String {
        let json = TraceRecordJson {
            ts_ms: self.ts_ms,
            model: &self.model,
            model_hash: self.model_hash,
            n_tokens: self.n_tokens,
            verdict: self.verdict.as_str(),
            matched_tokens: self.matched_tokens,
            prefill_from: self.prefill_from,
            messages: &self.messages,
        };
        serde_json::to_string(&json).expect("TraceRecordJson serializes")
    }
}

#[derive(Serialize)]
struct TraceRecordJson<'a> {
    ts_ms: u64,
    #[serde(skip_serializing_if = "str::is_empty")]
    model: &'a str,
    model_hash: u64,
    n_tokens: u32,
    verdict: &'static str,
    matched_tokens: u32,
    prefill_from: u32,
    #[serde(skip_serializing_if = "serde_json::Value::is_null")]
    messages: &'a serde_json::Value,
}

pub enum TraceCommand {
    Record(TraceRecord),
    Flush,
}

/// Handle cheap enough to build per request: a try-send on a bounded channel.
/// Never blocks, never panics; a full or disconnected channel drops records
/// (counted, surfaced once on drop) — trace capture must degrade, not wedge.
pub struct TraceWriter {
    tx: Option<SyncSender<TraceCommand>>,
    dropped: std::sync::atomic::AtomicU64,
    err: std::sync::Arc<std::sync::atomic::AtomicU64>,
}

/// Spawn the writer thread over an open file. Returns the handle; the thread
/// drains until the channel disconnects (sender dropped) — the daemon never
/// joins on the request path, the file closes via the BufWriter drop.
fn spawn_writer(
    file: std::fs::File,
    rx: Receiver<TraceCommand>,
    err_counter: std::sync::Arc<std::sync::atomic::AtomicU64>,
) -> std::thread::JoinHandle<()> {
    std::thread::Builder::new()
        .name("mlxcache-trace".into())
        .spawn(move || {
            let mut w = std::io::BufWriter::with_capacity(64 * 1024, file);
            for cmd in rx {
                match cmd {
                    TraceCommand::Record(r) => {
                        // writeln + flush per record: a trace that loses its
                        // tail on a crash understates the load it measured.
                        // BufWriter->File flush is a userspace write (no
                        // fsync); the per-record cost is one syscall per
                        // request on the CAPTURE path only (MLXCACHE_TRACE
                        // unset pays nothing).
                        if writeln!(w, "{}", r.to_json())
                            .and_then(|_| w.flush())
                            .is_err()
                        {
                            err_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                    }
                    TraceCommand::Flush => {
                        if w.flush().is_err() {
                            err_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                    }
                }
            }
            let _ = w.flush();
        })
        .expect("trace writer thread")
}

impl TraceWriter {
    /// Build from MLXCACHE_TRACE (a file path). Truncates (a trace is a
    /// session's story; append-only across restarts would mix daemons).
    /// Invalid path/permission is a LOUD startup error: an operator who asked
    /// for traces must not silently run untraced.
    pub fn from_env() -> Result<Option<Self>, String> {
        match std::env::var("MLXCACHE_TRACE") {
            Ok(path) if !path.trim().is_empty() => Self::from_path(path).map(Some),
            _ => Ok(None),
        }
    }

    /// Explicit-path constructor (tests, multi-daemon drivers). Same truncat-
    /// ing, loud-on-failure contract as `from_env`.
    pub fn from_path(path: impl AsRef<std::path::Path>) -> Result<Self, String> {
        let path = path.as_ref();
        // 0600, not File::create's umask-dependent 0644: the trace embeds the
        // full request payload (user prompts), so it must be owner-readable
        // only. create+write+truncate keeps the from_env truncating contract
        // (a trace is a session's story, not an append log).
        #[cfg(unix)]
        let file = {
            use std::os::unix::fs::OpenOptionsExt;
            std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .mode(0o600)
                .open(path)
                .map_err(|e| format!("MLXCACHE_TRACE {}: {e}", path.display()))?
        };
        #[cfg(not(unix))]
        let file = std::fs::File::create(path)
            .map_err(|e| format!("MLXCACHE_TRACE {}: {e}", path.display()))?;
        let (tx, rx) = std::sync::mpsc::sync_channel(CHANNEL_BOUND);
        let err_counter = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        spawn_writer(file, rx, err_counter.clone());
        Ok(Self {
            tx: Some(tx),
            dropped: std::sync::atomic::AtomicU64::new(0),
            err: err_counter,
        })
    }

    pub fn record(&self, r: TraceRecord) {
        if let Some(tx) = &self.tx {
            if tx.try_send(TraceCommand::Record(r)).is_err() {
                self.dropped
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }

    /// Best-effort durability point for tests and graceful shutdown.
    pub fn flush(&self) {
        if let Some(tx) = &self.tx {
            let _ = tx.try_send(TraceCommand::Flush);
        }
    }

    /// Total records the writer could not accept (full channel) or write
    /// (I/O error). Zero in every sane run; nonzero means the trace file is a
    /// SAMPLE, and any replay score derived from it must say so.
    pub fn degraded(&self) -> u64 {
        self.dropped.load(std::sync::atomic::Ordering::Relaxed)
            + self.err.load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl Drop for TraceWriter {
    fn drop(&mut self) {
        // Detach the sender: the writer drains what it has and exits when the
        // channel disconnects. Warn loudly if anything was lost — a trace that
        // silently lost records would understate hit-rate.
        let lost = self.degraded();
        if lost > 0 {
            tracing::warn!(
                lost,
                "trace capture lost records; replay scores are lower bounds"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observability::tests::decision;

    /// Serializes every test that reads or writes MLXCACHE_TRACE: the env is
    /// process-global and the test harness runs this module's tests on
    /// parallel threads, so an unguarded mutation races any other test's
    /// `from_env` read (flaky pass/fail by interleaving).
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn record_shape_is_stable_jsonl() {
        let d = decision(CacheVerdict::Hit, 50, 60);
        let mut r = TraceRecord::from_decision(0xdeadbeef, &d);
        r.model = "mlx-community/Qwen2.5-7B-Instruct-4bit".into();
        r.messages = serde_json::json!([{"role": "user", "content": "hi"}]);
        let line = r.to_json();
        assert!(line.contains("\"verdict\":\"hit\""), "{line}");
        assert!(line.contains("\"n_tokens\":60"), "{line}");
        assert!(line.contains("\"prefill_from\":49"), "{line}");
        assert!(
            line.contains("\"model\":\"mlx-community/Qwen2.5-7B-Instruct-4bit\""),
            "{line}"
        );
        // The messages array is embedded verbatim as JSON (not escaped):
        assert!(line.contains("\"content\":\"hi\""), "{line}");
        // Single line: no embedded newlines.
        assert!(!line.contains('\n'));
        // Round-trips through serde_json (the consumer's parser).
        let v: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert_eq!(v["model_hash"], 0xdeadbeefu64 as f64);
        assert_eq!(v["matched_tokens"], 50);
        assert_eq!(v["messages"][0]["role"], "user");
    }

    #[test]
    fn messages_omitted_when_null() {
        // Synthetic/unit-only records (from_decision constructor) leave
        // messages Null; the replay harness tolerates both shapes, but a
        // capture-only consumer gets a compact line.
        let d = decision(CacheVerdict::Hit, 50, 60);
        let r = TraceRecord::from_decision(7, &d);
        let line = r.to_json();
        assert!(!line.contains("messages"), "{line}");
        assert!(!line.contains("\"model\":"), "{line}");
    }

    #[test]
    fn miss_and_partial_cover_zero_or_matched_minus_one() {
        let m = decision(CacheVerdict::Miss, 0, 60);
        let r = TraceRecord::from_decision(1, &m);
        assert_eq!(r.prefill_from, 0);
        assert!(r.to_json().contains("\"verdict\":\"miss\""));

        let p = decision(CacheVerdict::Partial, 30, 60);
        let r = TraceRecord::from_decision(1, &p);
        assert_eq!(r.prefill_from, 29);
        assert!(r.to_json().contains("\"verdict\":\"partial\""));
    }

    #[test]
    fn from_env_disabled_by_default() {
        // No MLXCACHE_TRACE -> no writer. The lock guards the env against the
        // other env-mutating tests on parallel threads.
        let _env = ENV_LOCK.lock().unwrap();
        let old = std::env::var("MLXCACHE_TRACE").ok();
        unsafe { std::env::remove_var("MLXCACHE_TRACE") };
        assert!(TraceWriter::from_env().unwrap().is_none());
        if let Some(v) = old {
            unsafe { std::env::set_var("MLXCACHE_TRACE", v) };
        }
    }

    #[test]
    fn writer_flushes_records_to_disk() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("trace.jsonl");
        let (tx, rx) = std::sync::mpsc::sync_channel(16);
        let err_counter = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let writer = spawn_writer(
            std::fs::File::create(&path).unwrap(),
            rx,
            err_counter.clone(),
        );
        let d = decision(CacheVerdict::Hit, 10, 11);
        tx.send(TraceCommand::Record(TraceRecord::from_decision(7, &d)))
            .unwrap();
        // Drop the sender FIRST so the writer's loop ends after draining,
        // then join: when join returns, BufWriter is dropped (flushed) and
        // the file contents are final — no race with the writer thread.
        drop(tx);
        writer.join().unwrap();
        // Reader side: the file must contain exactly one JSON line.
        let body = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = body.lines().collect();
        assert_eq!(lines.len(), 1, "{body}");
        let v: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(v["n_tokens"], 11);
    }

    #[test]
    fn trace_writer_end_to_end_via_env() {
        // Mutates MLXCACHE_TRACE: hold the env lock for the whole body so
        // parallel tests cannot observe or clobber the value mid-run.
        let _env = ENV_LOCK.lock().unwrap();
        let old = std::env::var("MLXCACHE_TRACE").ok();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("trace.jsonl");
        // SAFETY (tests only): the env lock above serializes every reader.
        unsafe { std::env::set_var("MLXCACHE_TRACE", &path) };
        let w = TraceWriter::from_env().unwrap().unwrap();
        let d = decision(CacheVerdict::Partial, 20, 40);
        w.record(TraceRecord::from_decision(9, &d));
        w.flush();
        // Poll instead of assume: the flush command takes the same bounded
        // channel as the record, so once the file shows the verdict the
        // ordering guarantee has caught up (both are FIFO).
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            let body = std::fs::read_to_string(&path).unwrap_or_default();
            if body.contains("\"verdict\":\"partial\"") {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "flush never landed in {}: {body}",
                path.display()
            );
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        unsafe { std::env::remove_var("MLXCACHE_TRACE") };
        // Restore whatever the environment had before this test touched it.
        if let Some(v) = old {
            unsafe { std::env::set_var("MLXCACHE_TRACE", v) };
        }
    }

    #[cfg(unix)]
    #[test]
    fn trace_file_is_owner_only() {
        // The trace embeds full prompts, so from_path must create 0600 —
        // File::create's umask-dependent 0644 would expose them to other
        // local users. Unique name; cleaned up after the assertion.
        use std::os::unix::fs::PermissionsExt;
        let path = std::env::temp_dir().join(format!(
            "mlxcache-trace-mode-{}-{}.jsonl",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let w = TraceWriter::from_path(&path).expect("trace writer builds");
        w.flush();
        drop(w);
        let mode = std::fs::metadata(&path).unwrap().permissions().mode();
        assert_eq!(
            mode & 0o777,
            0o600,
            "trace file must be owner-only: {:o}",
            mode
        );
        let _ = std::fs::remove_file(&path);
    }
}
