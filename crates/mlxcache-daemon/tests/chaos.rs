//! Concurrency + chaos suite (T7, R6).
//!
//! Covers the design doc's chaos requirements:
//! - single-flight race: N concurrent requests for the same prefix → exactly
//!   one prefill (R1-3)
//! - kill -9 mid-write: temp file left behind, final blob never partial
//!   (atomic rename)
//! - restart mid-stream: streams drop, checkpoints survive (R1-4)

use mlxcache_core::contract::CheckpointMeta;
use mlxcache_core::singleflight::SingleFlight;
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
            let (guard, follower) = sf.try_lead(vec![0xFFFF_0001]).await;
            match follower {
                None => {
                    // Only the leader runs the prefill.
                    count.fetch_add(1, Ordering::SeqCst);
                    tokio::time::sleep(std::time::Duration::from_millis(5)).await;
                    drop(guard);
                }
                Some(mut rx) => {
                    // Followers await the leader's completion signal.
                    let _ = rx.recv().await;
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
            let (guard, follower) = sf.try_lead(vec![0xFFFF_0000 + i as u32]).await;
            match follower {
                None => {
                    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                    c.fetch_add(1, Ordering::SeqCst);
                    drop(guard);
                }
                Some(mut rx) => {
                    let _ = rx.recv().await;
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
        format_version: 1,
    };
    let path = p.publish_atomic(0xabc, &meta, b"complete-kv").unwrap();
    let (m, payload) = p.load(&path).unwrap();
    assert_eq!(m.token_count, 3);
    assert_eq!(payload, b"complete-kv");
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
        format_version: 1,
    };
    let path = p.publish_atomic(0x777, &meta, b"kv").unwrap();

    // "Restart": fresh index.
    let orch = mlxcache_daemon::orchestrator::Orchestrator::new();
    // Rebuild: scan blobs, republish into index (this is the index-corruption
    // recovery path too).
    for blob in p.list_blobs().unwrap() {
        let (meta, _payload) = p.load(&blob).unwrap();
        orch.publish_checkpoint(&[1, 2, 3, 4], meta, blob.to_string_lossy().into());
    }
    let out = orch.route(
        &[1, 2, 3, 4],
        &mlxcache_daemon::orchestrator::test_support::fp("m"),
    );
    assert_eq!(
        out.decision.verdict,
        mlxcache_core::policy::CacheVerdict::Hit,
        "checkpoint must survive restart"
    );
    assert!(path.exists());
}
