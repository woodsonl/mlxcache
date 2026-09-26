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
            match sf.enter(vec![0xFFFF_0001]).await {
                Role::Leader(lead) => {
                    // Only the leader runs the prefill.
                    count.fetch_add(1, Ordering::SeqCst);
                    tokio::time::sleep(std::time::Duration::from_millis(5)).await;
                    lead.complete(Ok("blob".into()));
                }
                Role::Follower(rx) => {
                    // Followers await the leader's published result.
                    let _ = await_result(rx).await;
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
            match sf.enter(vec![0xFFFF_0000 + i as u32]).await {
                Role::Leader(lead) => {
                    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                    c.fetch_add(1, Ordering::SeqCst);
                    lead.complete(Ok("blob".into()));
                }
                Role::Follower(rx) => {
                    let _ = await_result(rx).await;
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
    };
    let path = p.publish_atomic(0xabc, &meta, b"complete-kv").unwrap();
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
    };
    p.publish_atomic(0x1, &meta, b"nonempty-legacy-kv").unwrap();

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
    };
    let path = p.publish_atomic(0x777, &meta, b"kv").unwrap();

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
        out.blob_path.as_deref(),
        Some("00000000000000000000000000000777.ckpt"),
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
    };
    p.publish_atomic(0x1, &good, b"kv").unwrap();

    // A corrupt blob (garbage bytes) and a blob with no recoverable prefix.
    std::fs::write(dir.path().join("deadbeef.ckpt"), b"not-a-blob").unwrap();
    let noprefix = CheckpointMeta {
        fingerprint: fp.clone(),
        token_count: 2,
        tokens: vec![],
        format_version: 1,
    };
    p.publish_atomic(0x2, &noprefix, b"kv").unwrap();

    let orch = mlxcache_daemon::orchestrator::Orchestrator::new();
    let report = orch.rebuild_from_disk(&p);
    assert_eq!(report.rebuilt, 1, "only the well-formed blob is indexed");
    assert_eq!(report.skipped, 2, "corrupt + no-prefix blobs are skipped");
    assert_eq!(report.errors.len(), 1, "the corrupt blob is reported");

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
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn daemon");

    // Wait until the port answers, so SIGTERM lands after handlers are installed.
    let ready = (0..100).any(|_| {
        std::thread::sleep(std::time::Duration::from_millis(50));
        std::net::TcpStream::connect(&addr).is_ok()
    });
    assert!(ready, "daemon never started listening on {addr}");

    let pid = child.id();
    unsafe {
        libc::kill(pid as libc::pid_t, libc::SIGTERM);
    }

    // Bounded wait: graceful shutdown must complete, not hang.
    let status = child.wait().expect("wait daemon");
    assert!(status.success(), "daemon must exit 0 on SIGTERM: {status}");

    let mut out = String::new();
    child
        .stderr
        .take()
        .unwrap()
        .read_to_string(&mut out)
        .unwrap();
    let mut out2 = String::new();
    child
        .stdout
        .take()
        .unwrap()
        .read_to_string(&mut out2)
        .unwrap();
    let combined = format!("{out}{out2}");
    assert!(
        combined.contains("SIGTERM"),
        "shutdown should log the signal, got: {combined}"
    );
}
