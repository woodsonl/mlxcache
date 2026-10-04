//! Concurrency + chaos suite (T7, R6).
//!
//! Covers the design doc's chaos requirements:
//! - single-flight race: N concurrent requests for the same prefix → exactly
//!   one prefill (R1-3)
//! - kill -9 mid-write: temp file left behind, final blob never partial
//!   (atomic rename)
//! - restart mid-stream: streams drop, checkpoints survive (R1-4)

use mlxcache_core::contract::CheckpointMeta;
use mlxcache_core::singleflight::{await_result, Role, SingleFlight};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

#[tokio::test]
async fn singleflight_race_one_winner() {
    let sf = Arc::new(SingleFlight::new());
    let prefill_count = Arc::new(AtomicUsize::new(0));

    let mut handles = Vec::new();
    for _ in 0..50 {
        let sf = sf.clone();
        let count = prefill_count.clone();
        handles.push(tokio::spawn(async move {
            match sf.enter(&[0xFFFF_0001]).await {
                Role::Leader(lead) => {
                    // Only the leader runs the prefill.
                    count.fetch_add(1, Ordering::SeqCst);
                    tokio::time::sleep(std::time::Duration::from_millis(5)).await;
                    lead.complete(Ok("blob".into()));
                }
                Role::Follower(mut f) => {
                    // Followers await the leader's published result.
                    let _ = await_result(&mut f).await;
                }
            }
        }));
    }
    for h in handles {
        h.await.unwrap();
    }
    assert_eq!(
        prefill_count.load(Ordering::SeqCst),
        1,
        "50 racing requests for one prefix must trigger exactly one prefill"
    );
}

#[tokio::test]
async fn singleflight_different_keys_run_parallel() {
    let sf = Arc::new(SingleFlight::new());
    let completed = Arc::new(AtomicUsize::new(0));

    let mut handles = Vec::new();
    for i in 0..10 {
        let sf = sf.clone();
        let c = completed.clone();
        handles.push(tokio::spawn(async move {
            match sf.enter(&[0xFFFF_0000 + i as u32]).await {
                Role::Leader(lead) => {
                    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                    c.fetch_add(1, Ordering::SeqCst);
                    lead.complete(Ok("blob".into()));
                }
                Role::Follower(mut f) => {
                    let _ = await_result(&mut f).await;
                }
            }
        }));
    }
    let start = std::time::Instant::now();
    for h in handles {
        h.await.unwrap();
    }
    // 10 x 20ms sleeps in parallel must NOT take 200ms sequential time.
    // Generous bound: under 150ms proves parallelism.
    assert!(
        start.elapsed() < std::time::Duration::from_millis(150),
        "different keys must not serialize ({:?})",
        start.elapsed()
    );
    assert_eq!(completed.load(Ordering::SeqCst), 10);
}

#[test]
fn kill_mid_write_leaves_temp_only() {
    // Simulate kill -9 between temp-file create and rename: a temp file
    // exists, no final blob. Restart must not treat the temp as published.
    let dir = tempfile::tempdir().unwrap();
    let tmp = dir.path().join("abc.ckpt.tmp");
    std::fs::write(&tmp, b"partial").unwrap();

    // The recovery scan (list_blobs) must find zero published blobs.
    let p = mlxcache_daemon::persistence::Persistence::new(dir.path()).unwrap();
    assert_eq!(
        p.list_blobs().unwrap().len(),
        0,
        "temp files must never count as published checkpoints"
    );
}

#[test]
fn crash_before_rename_never_serves_partial() {
    // Full atomicity proof: a blob that exists under its final name is
    // always complete (rename is atomic on APFS). Verify load() succeeds
    // on every published blob and every published blob round-trips.
    let dir = tempfile::tempdir().unwrap();
    let p = mlxcache_daemon::persistence::Persistence::new(dir.path()).unwrap();
    let meta = CheckpointMeta {
        fingerprint: mlxcache_daemon::orchestrator::test_support::fp("m"),
        token_count: 3,
        tokens: vec![1, 2, 3],
        format_version: 1,
        payload_sha256: None,
    };
    let path = p.publish_atomic(0xabc, 1, meta, b"complete-kv").unwrap();
    let (m, payload) = p.load(&path).unwrap();
    assert_eq!(m.token_count, 3);
    assert_eq!(payload, b"complete-kv");
}

#[test]
fn rebuild_skips_one_token_legacy_blob() {
    // Regression (adversarial F1): a legacy one-token checkpoint (nonempty KV,
    // token_count=1) must not be indexed on rebuild. Otherwise the daemon
    // reports a "hit" the adapter refuses to serve, so the client sees a hit
    // that silently ran from scratch.
    let dir = tempfile::tempdir().unwrap();
    let p = mlxcache_daemon::persistence::Persistence::new(dir.path()).unwrap();
    let meta = CheckpointMeta {
        fingerprint: mlxcache_daemon::orchestrator::test_support::fp("m"),
        token_count: 1,
        tokens: vec![7],
        format_version: 1,
        payload_sha256: None,
    };
    p.publish_atomic(0x1, 1, meta, b"nonempty-legacy-kv")
        .unwrap();

    let orch = mlxcache_daemon::orchestrator::Orchestrator::new();
    let report = orch.rebuild_from_disk(&p);
    assert_eq!(report.rebuilt, 0, "a one-token blob must not be indexed");
    assert_eq!(report.skipped, 1);
    let out = orch.route(&[7], &mlxcache_daemon::orchestrator::test_support::fp("m"));
    assert_eq!(
        out.decision.verdict,
        mlxcache_core::policy::CacheVerdict::Miss,
        "no indexed one-token entry, so the request is an honest miss"
    );
}

#[tokio::test]
async fn restart_drops_streams_checkpoints_survive() {
    // R1-4: a "restart" is a fresh Orchestrator; checkpoints published to
    // persistence before the restart must be re-publishable into the new
    // index (rebuild path) and still classify as hits.
    let dir = tempfile::tempdir().unwrap();
    let p = mlxcache_daemon::persistence::Persistence::new(dir.path()).unwrap();
    let meta = CheckpointMeta {
        fingerprint: mlxcache_daemon::orchestrator::test_support::fp("m"),
        token_count: 4,
        tokens: vec![1, 2, 3, 4],
        format_version: 1,
        payload_sha256: None,
    };
    let path = p.publish_atomic(0x777, 1, meta, b"kv").unwrap();

    // "Restart": fresh index, rebuild from disk via the real startup path.
    let orch = mlxcache_daemon::orchestrator::Orchestrator::new();
    let report = orch.rebuild_from_disk(&p);
    assert_eq!(
        report.rebuilt, 1,
        "the persisted checkpoint must be rebuilt"
    );
    assert_eq!(report.skipped, 0);
    let out = orch.route(
        &[1, 2, 3, 4],
        &mlxcache_daemon::orchestrator::test_support::fp("m"),
    );
    assert_eq!(
        out.decision.verdict,
        mlxcache_core::policy::CacheVerdict::Hit,
        "checkpoint must survive restart"
    );
    assert_eq!(
        out.blob.as_ref().map(|(name, _gen, _pfx)| name.as_str()),
        Some("00000000000000000000000000000777-0000000000000001.ckpt"),
        "rebuilt entry must point at the on-disk blob NAME (not an abs path)"
    );
    assert!(path.exists());
}

#[tokio::test]
async fn rebuild_indexes_extension_lookup_and_skips_corrupt() {
    // A restart must serve: (a) the exact persisted prefix, (b) an extension of
    // it as a partial hit, and (c) must not publish a corrupt/mis-keyed blob.
    let dir = tempfile::tempdir().unwrap();
    let p = mlxcache_daemon::persistence::Persistence::new(dir.path()).unwrap();
    let fp = mlxcache_daemon::orchestrator::test_support::fp("m");
    let good = CheckpointMeta {
        fingerprint: fp.clone(),
        token_count: 4,
        tokens: vec![1, 2, 3, 4],
        format_version: 1,
        payload_sha256: None,
    };
    p.publish_atomic(0x1, 1, good, b"kv").unwrap();

    // A corrupt blob (garbage bytes) and a blob with no recoverable prefix.
    std::fs::write(dir.path().join("deadbeef.ckpt"), b"not-a-blob").unwrap();
    let noprefix = CheckpointMeta {
        fingerprint: fp.clone(),
        token_count: 2,
        tokens: vec![],
        format_version: 1,
        payload_sha256: None,
    };
    p.publish_atomic(0x2, 1, noprefix, b"kv").unwrap();

    // A multi-token blob with an EMPTY payload: a truncated write the adapter
    // rejects at runtime. The file can outlive its retirement (a repaired
    // republish writes a different deterministic name), so rebuild must skip it
    // rather than resurrect the poison.
    let empty = CheckpointMeta {
        fingerprint: fp.clone(),
        token_count: 3,
        tokens: vec![7, 7, 7],
        format_version: 1,
        payload_sha256: None,
    };
    p.publish_atomic(0x3, 1, empty, b"").unwrap();

    let orch = mlxcache_daemon::orchestrator::Orchestrator::new();
    let report = orch.rebuild_from_disk(&p);
    assert_eq!(report.rebuilt, 1, "only the well-formed blob is indexed");
    assert_eq!(
        report.skipped, 3,
        "corrupt, no-prefix, and empty-payload blobs are skipped"
    );
    // Since the gauntlet wave (2026-10-04), a deterministically-corrupt blob
    // is RECLAIMED at the sweep (stranding it left the file invisible to the
    // eviction byte budget forever) instead of merely reported as an error.
    assert_eq!(report.reclaimed_corrupt, 1, "the corrupt blob is reclaimed");
    assert_eq!(
        report.errors.len(),
        0,
        "no errors: corruption is handled, not logged",
    );

    // Exact hit after rebuild.
    assert_eq!(
        orch.route(&[1, 2, 3, 4], &fp).decision.verdict,
        mlxcache_core::policy::CacheVerdict::Hit
    );
    // Extension -> partial hit at the persisted prefix.
    let ext = orch.route(&[1, 2, 3, 4, 5, 6], &fp);
    assert_eq!(
        ext.decision.verdict,
        mlxcache_core::policy::CacheVerdict::Partial
    );
    assert_eq!(ext.prefill_from, 3);
}

#[tokio::test]
async fn rebuild_keeps_the_highest_generation_and_reclaims_the_rest() {
    // Regression (Codex pass 7): immutable per-generation names mean a prefix can
    // have several files on disk. Directory order is arbitrary, so recovery must
    // pick the LATEST publication (highest generation) per prefix and must not
    // let an older file overwrite (and delete) the newer repair.
    let dir = tempfile::tempdir().unwrap();
    let p = mlxcache_daemon::persistence::Persistence::new(dir.path()).unwrap();
    let fp = mlxcache_daemon::orchestrator::test_support::fp("m");
    let meta = CheckpointMeta {
        fingerprint: fp.clone(),
        token_count: 4,
        tokens: vec![1, 2, 3, 4],
        format_version: 1,
        payload_sha256: None,
    };
    // Same prefix, generations 5 and 17. gen 17 is the repair; it must win.
    p.publish_atomic(0xabc, 5, meta.clone(), b"old").unwrap();
    let repair = p.publish_atomic(0xabc, 17, meta.clone(), b"new").unwrap();

    let orch = mlxcache_daemon::orchestrator::Orchestrator::new();
    let report = orch.rebuild_from_disk(&p);
    assert_eq!(report.rebuilt, 1, "one entry per prefix");
    let out = orch.route(&[1, 2, 3, 4], &fp);
    let name = out.blob.as_ref().map(|(n, _, _)| n.as_str());
    assert_eq!(
        name,
        repair.file_name().map(|n| n.to_string_lossy()).as_deref(),
        "the highest generation is indexed"
    );
    assert!(repair.exists(), "the repair file must survive recovery");
    assert_eq!(
        std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().ends_with(".ckpt"))
            .count(),
        1,
        "the superseded generation is reclaimed"
    );

    // Regression (Codex pass 8): after a restart the counter must be seeded above
    // the persisted generation, so a NEW publication is not given a lower one and
    // discarded by the next rebuild. Publish after rebuild; its name must encode
    // a generation above 17.
    let gen = orch.reserve_generation();
    assert!(
        gen > 17,
        "counter must resume above the persisted max, got {gen}"
    );
    let newer = p
        .publish_atomic(0xabc, gen, meta.clone(), b"newer")
        .unwrap();
    let orch2 = mlxcache_daemon::orchestrator::Orchestrator::new();
    let report2 = orch2.rebuild_from_disk(&p);
    assert_eq!(report2.rebuilt, 1);
    let out2 = orch2.route(&[1, 2, 3, 4], &fp);
    assert_eq!(
        out2.blob.as_ref().map(|(n, _, _)| n.as_str()),
        newer.file_name().map(|n| n.to_string_lossy()).as_deref(),
        "the newest publication survives a second restart"
    );
}

#[tokio::test]
async fn rebuild_seeds_generation_above_unloadable_files_too() {
    // Regression (Codex pass 10): the generation floor must come from FILENAMES,
    // not only files that load. A higher-generation file that is temporarily
    // unreadable must still raise the floor, or a replacement published now gets
    // a lower generation and is deleted once the higher one recovers.
    let dir = tempfile::tempdir().unwrap();
    let p = mlxcache_daemon::persistence::Persistence::new(dir.path()).unwrap();
    let fp = mlxcache_daemon::orchestrator::test_support::fp("m");
    // A structurally valid blob at generation 3.
    let meta = CheckpointMeta {
        fingerprint: fp.clone(),
        token_count: 4,
        tokens: vec![1, 2, 3, 4],
        format_version: 1,
        payload_sha256: None,
    };
    p.publish_atomic(0x1, 3, meta, b"kv").unwrap();
    // A higher-generation file whose body is unreadable (truncated header): it is
    // skipped by the load, but its NAME must still seed the floor above 17.
    std::fs::write(
        dir.path()
            .join("00000000000000000000000000000002-0000000000000011.ckpt"),
        b"xx",
    )
    .unwrap();

    let orch = mlxcache_daemon::orchestrator::Orchestrator::new();
    let report = orch.rebuild_from_disk(&p);
    assert_eq!(report.rebuilt, 1, "only the valid blob is indexed");
    let gen = orch.reserve_generation();
    assert!(
        gen > 17,
        "the floor must account for the unloadable generation-17 filename, got {gen}"
    );
}

#[cfg(unix)]
#[test]
fn sigterm_triggers_graceful_shutdown() {
    // The daemon must drain and exit 0 on SIGTERM, not die abruptly. A clean
    // shutdown logs the drain line. (Publishes are atomic, so this is about not
    // dropping in-flight requests, not data safety.)
    use std::io::Read;
    use std::process::{Command, Stdio};

    let port = portpicker::pick_unused_port().expect("pick port");
    let addr = format!("127.0.0.1:{port}");
    let blobs = tempfile::tempdir().unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_mlxcache-daemon"))
        .env("MLXCACHE_ADDR", &addr)
        .env("MLXCACHE_BLOBS", blobs.path())
        .env("MLXCACHE_MODELS", "e2e-model")
        // Pin the log filter: the test asserts on the INFO signal line, so an
        // inherited RUST_LOG=error must not suppress it.
        .env("RUST_LOG", "info")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn daemon");

    // Read pipes on threads so a verbose failure cannot fill the pipe buffer and
    // deadlock, and so we can wait with a bound below.
    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();
    let out_h = std::thread::spawn(move || {
        let mut s = String::new();
        let mut r = stdout;
        let _ = r.read_to_string(&mut s);
        s
    });
    let err_h = std::thread::spawn(move || {
        let mut s = String::new();
        let mut r = stderr;
        let _ = r.read_to_string(&mut s);
        s
    });

    // Ensure we never leak the daemon if an assertion below fails.
    struct KillOnDrop(std::process::Child);
    impl Drop for KillOnDrop {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
    let mut guard = KillOnDrop(child);

    // Wait until the port accepts connections, then let the serve loop poll the
    // graceful-shutdown future (installing signal handlers) before signalling.
    // A connect can succeed right after bind and before that first poll.
    let ready = (0..100).any(|_| {
        std::thread::sleep(std::time::Duration::from_millis(50));
        std::net::TcpStream::connect(&addr).is_ok()
    });
    assert!(ready, "daemon never started listening on {addr}");
    std::thread::sleep(std::time::Duration::from_millis(200));

    let pid = guard.0.id();
    let rc = unsafe { libc::kill(pid as libc::pid_t, libc::SIGTERM) };
    assert_eq!(rc, 0, "SIGTERM delivery failed");

    // Bounded wait: graceful shutdown must complete promptly, not hang.
    let status = wait_with_timeout(&mut guard.0, std::time::Duration::from_secs(10))
        .expect("daemon did not exit within 10s of SIGTERM");
    assert!(status.success(), "daemon must exit 0 on SIGTERM: {status}");

    let combined = format!("{}{}", err_h.join().unwrap(), out_h.join().unwrap());
    assert!(
        combined.contains("mlxcache daemon stopped"),
        "shutdown should log completion, got: {combined}"
    );
}

/// Wait for a child with a deadline, killing it if the deadline passes. Returns
/// None on timeout.
#[cfg(unix)]
fn wait_with_timeout(
    child: &mut std::process::Child,
    timeout: std::time::Duration,
) -> Option<std::process::ExitStatus> {
    let deadline = std::time::Instant::now() + timeout;
    loop {
        if let Some(status) = child.try_wait().expect("try_wait") {
            return Some(status);
        }
        if std::time::Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            return None;
        }
        std::thread::sleep(std::time::Duration::from_millis(25));
    }
}
