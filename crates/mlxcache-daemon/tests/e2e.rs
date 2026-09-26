//! End-to-end: daemon + sidecar over real HTTP (hermetic synthetic engine).
//!
//! Proves the full pipeline: request → daemon → sidecar tokenize → route
//! (miss) → sidecar prefill → atomic blob publish → generate; second request
//! with the same prompt → route (hit) → generate without prefill.

use axum::body::Body;
use http_body_util::BodyExt;
use std::sync::Arc;
use tower::util::ServiceExt;

use mlxcache_core::singleflight::SingleFlight;
use mlxcache_daemon::http::{router, AppState};
use mlxcache_daemon::orchestrator::Orchestrator;
use mlxcache_daemon::sidecar::{SidecarClient, SidecarConfig};

#[tokio::test]
async fn unusable_blob_is_quarantined_and_served_from_scratch() {
    // A published checkpoint whose blob vanished (deleted, disk fault) must not
    // 502 forever. The daemon quarantines the entry and retries from scratch;
    // the following identical request then misses cleanly (no 502).
    let Some((sidecar_url, mut child)) = spawn_sidecar().await else {
        eprintln!("skipping: sidecar unavailable (install uv + sync deps)");
        return;
    };
    let blobs = tempfile::tempdir().unwrap();
    let state = Arc::new(AppState {
        orchestrator: Orchestrator::new(),
        singleflight: SingleFlight::new(),
        stats: Arc::new(mlxcache_daemon::http::Stats::default()),
        served_models: vec!["e2e-model".into()],
        sidecar: Some(
            SidecarClient::new(SidecarConfig::new(sidecar_url, "e2e-model".into())).unwrap(),
        ),
        persistence: mlxcache_daemon::persistence::Persistence::new(blobs.path()).unwrap(),
    });
    let body = serde_json::json!({
        "model": "e2e-model",
        "messages": [{"role": "user", "content": "quarantine me"}],
        "stream": false,
    })
    .to_string();

    let post = |body: String| {
        let app = router(state.clone());
        async move {
            app.oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap()
        }
    };

    // Request 1: miss → publish a blob.
    let res = post(body.clone()).await;
    assert_eq!(res.status(), 200);
    assert_eq!(state.orchestrator.quarantined_count(), 0);

    // Delete the blob out from under the index (simulate a disk fault).
    let blob_path = state.persistence.list_blobs().unwrap().pop().unwrap();
    std::fs::remove_file(&blob_path).unwrap();

    // Request 2: hit points at a missing blob → adapter errors → daemon must
    // quarantine and retry from scratch, returning 200 (not 502).
    let res = post(body.clone()).await;
    assert_eq!(
        res.status(),
        200,
        "unusable blob must not wedge the request"
    );
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["mlxcache"]["verdict"], "miss", "retried from scratch");
    assert_eq!(
        state.orchestrator.quarantined_count(),
        1,
        "entry quarantined"
    );

    // Request 3: the quarantined entry is never served again → still a clean
    // miss, not a 502.
    let res = post(body).await;
    assert_eq!(res.status(), 200);
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["mlxcache"]["verdict"], "miss");

    child.kill().expect("kill sidecar");
    child.wait().expect("reap sidecar");
}

#[tokio::test]
async fn one_token_prompt_serves_and_publishes_no_blob() {
    // A one-token prompt caches nothing (empty adapter payload). The daemon must
    // serve it from scratch, publish no checkpoint, and never report a phantom
    // hit on the repeat request.
    let Some((sidecar_url, mut child)) =
        spawn_sidecar_with_env(&[("MLXCACHE_TOKENIZE_ONE", "1")]).await
    else {
        eprintln!("skipping: sidecar unavailable (install uv + sync deps)");
        return;
    };
    let blobs = tempfile::tempdir().unwrap();
    let state = Arc::new(AppState {
        orchestrator: Orchestrator::new(),
        singleflight: SingleFlight::new(),
        stats: Arc::new(mlxcache_daemon::http::Stats::default()),
        served_models: vec!["e2e-model".into()],
        sidecar: Some(
            SidecarClient::new(SidecarConfig::new(sidecar_url, "e2e-model".into())).unwrap(),
        ),
        persistence: mlxcache_daemon::persistence::Persistence::new(blobs.path()).unwrap(),
    });
    let body = serde_json::json!({
        "model": "e2e-model",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": false,
    })
    .to_string();

    for _ in 0..2 {
        let res = router(state.clone())
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(body.clone()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), 200);
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["mlxcache"]["verdict"], "miss", "nothing was cached");
        assert!(!v["generated_tokens"].as_array().unwrap().is_empty());
    }
    assert_eq!(
        state.persistence.list_blobs().unwrap().len(),
        0,
        "a one-token prompt must publish no blob"
    );
    assert_eq!(state.orchestrator.published_count(), 0);

    child.kill().expect("kill sidecar");
    child.wait().expect("reap sidecar");
}

#[tokio::test]
async fn concurrent_one_token_requests_all_run_from_scratch() {
    // The follower path for the empty-blob (no-publish) case: N concurrent
    // one-token requests coalesce on a leader whose payload is empty. The leader
    // signals "no blob" (empty name); every follower must run from scratch, all
    // must be served, and no checkpoint may be published.
    //
    // Coalescing is only guaranteed for requests that overlap in flight, so the
    // test holds the leader open (MLXCACHE_PREFILL_DELAY) and releases the rest
    // only after /stats confirms the leader is prefilling. A plain barrier on
    // entry is not enough: a task can be scheduled past the leader's completion
    // and legitimately become a second leader (observed prefill_count 2 on CI).
    let Some((sidecar_url, mut child)) = spawn_sidecar_with_env(&[
        ("MLXCACHE_TOKENIZE_ONE", "1"),
        ("MLXCACHE_PREFILL_DELAY", "1.5"),
    ])
    .await
    else {
        eprintln!("skipping: sidecar unavailable (install uv + sync deps)");
        return;
    };
    let blobs = tempfile::tempdir().unwrap();
    let state = Arc::new(AppState {
        orchestrator: Orchestrator::new(),
        singleflight: SingleFlight::new(),
        stats: Arc::new(mlxcache_daemon::http::Stats::default()),
        served_models: vec!["e2e-model".into()],
        sidecar: Some(
            SidecarClient::new(SidecarConfig::new(sidecar_url.clone(), "e2e-model".into()))
                .unwrap(),
        ),
        persistence: mlxcache_daemon::persistence::Persistence::new(blobs.path()).unwrap(),
    });
    let body = serde_json::json!({
        "model": "e2e-model",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": false,
    })
    .to_string();

    const N: usize = 6;
    // Task 0 starts immediately and becomes the leader, then sleeps 1.5s in the
    // sidecar prefill. The remaining tasks are held on a watch channel until
    // /stats shows prefill_count == 1, so they enter single-flight while the
    // leader is provably in flight. This makes coalescing deterministic instead
    // of dependent on scheduler timing.
    let (start_followers_tx, followers_rx) = tokio::sync::watch::channel(false);
    let mut handles = Vec::new();
    for i in 0..N {
        let app = router(state.clone());
        let body = body.clone();
        let mut start_followers = followers_rx.clone();
        handles.push(tokio::spawn(async move {
            if i > 0 {
                while !*start_followers.borrow() {
                    if start_followers.changed().await.is_err() {
                        break;
                    }
                }
            }
            let res = app
                .oneshot(
                    axum::http::Request::builder()
                        .method("POST")
                        .uri("/v1/chat/completions")
                        .header("content-type", "application/json")
                        .body(Body::from(body))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(res.status(), 200, "every request must be served");
            let bytes = res.into_body().collect().await.unwrap().to_bytes();
            let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            v["mlxcache"]["verdict"].as_str().unwrap().to_string()
        }));
    }
    // Wait until the leader is inside the sidecar prefill (count is incremented
    // at prefill entry, before the delay), then release the followers.
    for _ in 0..200 {
        let stats: serde_json::Value = reqwest::get(format!("{sidecar_url}/stats"))
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        if stats["prefill_count"].as_u64().unwrap_or(0) >= 1 {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    start_followers_tx.send_replace(true);
    for h in handles {
        assert_eq!(h.await.unwrap(), "miss", "nothing was cached");
    }
    // The delay makes a single leader observable: N identical requests must
    // coalesce to exactly one prefill, so the follower branch is exercised.
    let stats: serde_json::Value = reqwest::get(format!("{sidecar_url}/stats"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        stats["prefill_count"].as_u64(),
        Some(1),
        "N={N} one-token requests must coalesce to 1 prefill; got {stats}"
    );
    assert_eq!(
        state.persistence.list_blobs().unwrap().len(),
        0,
        "no checkpoint may be published for a one-token prompt"
    );
    assert_eq!(state.orchestrator.published_count(), 0);

    child.kill().expect("kill sidecar");
    child.wait().expect("reap sidecar");
}

#[tokio::test]
async fn one_token_streaming_serves_and_publishes_no_blob() {
    // The streaming path for the no-publish case: a one-token prompt must stream
    // a 200 SSE with a miss verdict and publish nothing.
    let Some((sidecar_url, mut child)) =
        spawn_sidecar_with_env(&[("MLXCACHE_TOKENIZE_ONE", "1")]).await
    else {
        eprintln!("skipping: sidecar unavailable (install uv + sync deps)");
        return;
    };
    let blobs = tempfile::tempdir().unwrap();
    let state = Arc::new(AppState {
        orchestrator: Orchestrator::new(),
        singleflight: SingleFlight::new(),
        stats: Arc::new(mlxcache_daemon::http::Stats::default()),
        served_models: vec!["e2e-model".into()],
        sidecar: Some(
            SidecarClient::new(SidecarConfig::new(sidecar_url, "e2e-model".into())).unwrap(),
        ),
        persistence: mlxcache_daemon::persistence::Persistence::new(blobs.path()).unwrap(),
    });
    let body = serde_json::json!({
        "model": "e2e-model",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": true,
    })
    .to_string();

    let res = router(state.clone())
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), 200);
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let text = String::from_utf8_lossy(&bytes);
    assert!(text.contains("\"verdict\":\"miss\""), "SSE meta: {text}");
    assert!(
        text.trim_end().ends_with("data: [DONE]"),
        "SSE tail: {text}"
    );
    assert_eq!(
        state.persistence.list_blobs().unwrap().len(),
        0,
        "no checkpoint may be published for a one-token prompt"
    );

    child.kill().expect("kill sidecar");
    child.wait().expect("reap sidecar");
}

#[tokio::test]
async fn end_to_end_miss_then_hit() {
    let Some((sidecar_url, mut child)) = spawn_sidecar().await else {
        eprintln!("skipping: sidecar unavailable (install uv + sync deps)");
        return;
    };
    let blobs = tempfile::tempdir().unwrap();

    let state = Arc::new(AppState {
        orchestrator: Orchestrator::new(),
        singleflight: SingleFlight::new(),
        stats: Arc::new(mlxcache_daemon::http::Stats::default()),
        served_models: vec!["e2e-model".into()],
        sidecar: Some(
            SidecarClient::new(SidecarConfig::new(sidecar_url, "e2e-model".into())).unwrap(),
        ),
        persistence: mlxcache_daemon::persistence::Persistence::new(blobs.path()).unwrap(),
    });
    let make_app = || router(state.clone());

    let body = serde_json::json!({
        "model": "e2e-model",
        "messages": [{"role": "user", "content": "hello world"}],
        "stream": false,
    })
    .to_string();

    // Request 1: miss → prefill → publish → generate
    let res = make_app()
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body.clone()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), 200);
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["mlxcache"]["verdict"], "miss");
    assert_eq!(v["status"], "ok");
    assert!(!v["generated_tokens"].as_array().unwrap().is_empty());
    // The synthetic tokenizer emits 8 tokens; the leader adopts the blob it
    // just published, whose KV covers tokens[:-1] = 7. prefill_from reports the
    // covered count, so it must be 7, not 8.
    assert_eq!(
        v["mlxcache"]["prefill_from"], 7,
        "leader reports covered KV tokens (len-1), not the full request length"
    );

    // Blob must be on disk now.
    assert_eq!(state.persistence.list_blobs().unwrap().len(), 1);

    // Request 2: same prompt → hit (index lookup finds the published prefix)
    let res = make_app()
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), 200);
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        v["mlxcache"]["verdict"], "hit",
        "same prompt must hit the published checkpoint"
    );
    assert_eq!(
        v["mlxcache"]["prefill_from"], 7,
        "hit reports covered KV tokens for the 8-token prefix (len-1)"
    );

    // Stats: 2 requests, 1 miss, 1 hit.
    let res = make_app()
        .oneshot(
            axum::http::Request::builder()
                .uri("/stats")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let stats: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(stats["requests"], 2);
    assert_eq!(stats["hits"], 1);
    assert_eq!(stats["misses"], 1);

    child.kill().expect("kill sidecar");
    child.wait().expect("reap sidecar");
}

/// Spawn the sidecar with extra environment variables (test knobs), e.g. a
/// prefill delay to widen the single-flight window, or an empty tokenizer.
#[allow(clippy::zombie_processes)]
async fn spawn_sidecar_with_env(env: &[(&str, &str)]) -> Option<(String, std::process::Child)> {
    let port = portpicker::pick_unused_port().expect("free port");
    let script = format!(
        "import sys; sys.path.insert(0, {root:?}); \
         from mlxcache_sidecar import server; \
         server.Handler.engine = server.make_engine('e2e-model'); \
         server.ThreadingHTTPServer(('127.0.0.1', {port}), server.Handler).serve_forever()",
        root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../sidecar"),
        port = port
    );
    let mut cmd = std::process::Command::new("uv");
    cmd.args(["run", "python", "-c", &script]);
    for (k, v) in env {
        cmd.env(k, v);
    }
    let child = match cmd.spawn() {
        Ok(c) => c,
        Err(_) => return None,
    };
    let url = format!("http://127.0.0.1:{port}");
    for _ in 0..100 {
        if reqwest::get(format!("{url}/health")).await.is_ok() {
            return Some((url, child));
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    let mut child = child;
    let _ = child.kill();
    let _ = child.wait();
    None
}

/// One sidecar, immediately ready, no test knobs.
async fn spawn_sidecar() -> Option<(String, std::process::Child)> {
    spawn_sidecar_with_env(&[]).await
}

#[tokio::test]
async fn two_models_same_tokens_do_not_share_a_blob() {
    // Two served models, same prompt (the synthetic engine derives tokens from
    // the prompt only, so both produce identical token ids). Their checkpoints
    // MUST land in separate blob files: sharing one file would let the second
    // model overwrite the first, whose index entry would then load foreign KV.
    let Some((sidecar_url, mut child)) = spawn_sidecar().await else {
        eprintln!("skipping: sidecar unavailable (install uv + sync deps)");
        return;
    };
    let blobs = tempfile::tempdir().unwrap();
    let state = Arc::new(AppState {
        orchestrator: Orchestrator::new(),
        singleflight: SingleFlight::new(),
        stats: Arc::new(mlxcache_daemon::http::Stats::default()),
        served_models: vec!["model-a".into(), "model-b".into()],
        sidecar: Some(
            SidecarClient::new(SidecarConfig::new(sidecar_url.clone(), "model-a".into())).unwrap(),
        ),
        persistence: mlxcache_daemon::persistence::Persistence::new(blobs.path()).unwrap(),
    });

    for model in ["model-a", "model-b"] {
        let body = serde_json::json!({
            "model": model,
            "messages": [{"role": "user", "content": "same prompt"}],
            "stream": false,
        })
        .to_string();
        let res = router(state.clone())
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), 200);
    }

    let ckpts: Vec<_> = std::fs::read_dir(blobs.path())
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".ckpt"))
        .collect();
    assert_eq!(ckpts.len(), 2, "each model needs its own blob: {ckpts:?}");

    child.kill().expect("kill sidecar");
    child.wait().expect("reap sidecar");
}

#[tokio::test]
async fn concurrent_identical_requests_share_one_prefill() {
    // The core R1-3 promise: N identical uncached requests must trigger ONE
    // prefill, and every follower must still be served (from the leader's
    // freshly published checkpoint), not re-prefill from scratch.
    //
    // Coalescing is guaranteed only for requests that overlap in flight. The
    // leader is held open (MLXCACHE_PREFILL_DELAY) and followers are released
    // only after /stats confirms it is prefilling; spawning all N at once lets a
    // task be scheduled past the leader's completion and legitimately become a
    // second leader (observed as extra "miss" verdicts on CI).
    let Some((sidecar_url, mut child)) =
        spawn_sidecar_with_env(&[("MLXCACHE_PREFILL_DELAY", "5.0")]).await
    else {
        eprintln!("skipping: sidecar unavailable (install uv + sync deps)");
        return;
    };
    let blobs = tempfile::tempdir().unwrap();
    let state = Arc::new(AppState {
        orchestrator: Orchestrator::new(),
        singleflight: SingleFlight::new(),
        stats: Arc::new(mlxcache_daemon::http::Stats::default()),
        served_models: vec!["e2e-model".into()],
        sidecar: Some(
            SidecarClient::new(SidecarConfig::new(sidecar_url.clone(), "e2e-model".into()))
                .unwrap(),
        ),
        persistence: mlxcache_daemon::persistence::Persistence::new(blobs.path()).unwrap(),
    });

    let body = serde_json::json!({
        "model": "e2e-model",
        "messages": [{"role": "user", "content": "concurrent hello"}],
        "stream": false,
    })
    .to_string();

    const N: usize = 8;
    // Task 0 is the leader and starts immediately; it sleeps 1.5s in the
    // sidecar prefill. Followers wait on a watch channel until /stats shows the
    // leader is in flight, so they enter single-flight deterministically.
    let (start_followers_tx, start_followers_rx) = tokio::sync::watch::channel(false);
    let mut handles = Vec::new();
    for i in 0..N {
        let app = router(state.clone());
        let body = body.clone();
        let mut gate = start_followers_rx.clone();
        handles.push(tokio::spawn(async move {
            if i != 0 {
                while !*gate.borrow() {
                    if gate.changed().await.is_err() {
                        break;
                    }
                }
            }
            let res = app
                .oneshot(
                    axum::http::Request::builder()
                        .method("POST")
                        .uri("/v1/chat/completions")
                        .header("content-type", "application/json")
                        .body(Body::from(body))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(res.status(), 200);
            let bytes = res.into_body().collect().await.unwrap().to_bytes();
            let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            v["mlxcache"]["verdict"].as_str().unwrap().to_string()
        }));
    }

    // Wait for the leader to be prefilling, then release the followers. The
    // leader holds its prefill for MLXCACHE_PREFILL_DELAY (5s), far longer than
    // this bounded poll, so it is provably still in flight when followers enter
    // single-flight (otherwise a slow poll could release them after the leader
    // finished and they would just hit normally, not coalesce). If the leader
    // never starts within the bound, fail loudly rather than release unlocked
    // (which would re-elect a follower as leader and mask the setup failure).
    let mut leader_started = false;
    for _ in 0..200 {
        let stats: serde_json::Value =
            match tokio::time::timeout(std::time::Duration::from_secs(2), async {
                reqwest::get(format!("{sidecar_url}/stats"))
                    .await?
                    .json()
                    .await
            })
            .await
            {
                Ok(Ok(v)) => v,
                _ => {
                    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                    continue;
                }
            };
        if stats["prefill_count"].as_u64().unwrap_or(0) >= 1 {
            leader_started = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    assert!(
        leader_started,
        "leader never began prefilling; cannot test coalescing"
    );
    start_followers_tx.send_replace(true);

    let mut verdicts = Vec::new();
    for h in handles {
        verdicts.push(h.await.unwrap());
    }

    // Sidecar must have run the prefill exactly once for all N requests.
    let stats: serde_json::Value = reqwest::get(format!("{sidecar_url}/stats"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        stats["prefill_count"].as_u64(),
        Some(1),
        "N={N} identical requests must coalesce to exactly 1 prefill; got {stats}"
    );

    // Exactly one request (the leader) experienced a miss — it ran the prefill.
    // Every follower adopted the leader's published checkpoint (hit/partial).
    let misses = verdicts.iter().filter(|v| *v == "miss").count();
    assert_eq!(
        misses, 1,
        "exactly one leader should see a miss: {verdicts:?}"
    );
    assert!(
        verdicts
            .iter()
            .all(|v| v == "hit" || v == "partial" || v == "miss"),
        "unexpected verdicts: {verdicts:?}"
    );

    // One blob on disk, not N.
    assert_eq!(state.persistence.list_blobs().unwrap().len(), 1);

    child.kill().expect("kill sidecar");
    child.wait().expect("reap sidecar");
}

#[tokio::test]
async fn end_to_end_restart_resumes_from_disk() {
    // R1-4 across the REAL HTTP path: request 1 publishes a checkpoint; a fresh
    // daemon (new AppState + rebuild_from_disk on the same blob dir) must
    // serve request 2 as a hit without re-prefilling.
    let Some((sidecar_url, mut child)) = spawn_sidecar().await else {
        eprintln!("skipping: sidecar unavailable (install uv + sync deps)");
        return;
    };
    let blobs = tempfile::tempdir().unwrap();
    let make_state = || {
        let state = Arc::new(AppState {
            orchestrator: Orchestrator::new(),
            singleflight: SingleFlight::new(),
            stats: Arc::new(mlxcache_daemon::http::Stats::default()),
            served_models: vec!["e2e-model".into()],
            sidecar: Some(
                SidecarClient::new(SidecarConfig::new(sidecar_url.clone(), "e2e-model".into()))
                    .unwrap(),
            ),
            persistence: mlxcache_daemon::persistence::Persistence::new(blobs.path()).unwrap(),
        });
        state.orchestrator.rebuild_from_disk(&state.persistence);
        state
    };

    let body = serde_json::json!({
        "model": "e2e-model",
        "messages": [{"role": "user", "content": "survive the restart"}],
        "stream": false,
    })
    .to_string();

    // Daemon instance 1: miss -> prefill -> publish.
    let state1 = make_state();
    let res = router(state1.clone())
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body.clone()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), 200);
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["mlxcache"]["verdict"], "miss");
    drop(state1); // "process exit"

    // Daemon instance 2: fresh index, rebuilt from the persisted blob.
    let state2 = make_state();
    let res = router(state2.clone())
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), 200);
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        v["mlxcache"]["verdict"], "hit",
        "a restart must resume the persisted checkpoint (R1-4)"
    );

    child.kill().expect("kill sidecar");
    child.wait().expect("reap sidecar");
}

#[tokio::test]
async fn empty_tokenization_is_rejected() {
    // A tokenizer that returns no tokens must yield a 400, not a prefill of an
    // empty cache or a shared zero-token blob.
    let Some((sidecar_url, mut child)) =
        spawn_sidecar_with_env(&[("MLXCACHE_TOKENIZE_EMPTY", "1")]).await
    else {
        eprintln!("skipping: sidecar unavailable (install uv + sync deps)");
        return;
    };
    let blobs = tempfile::tempdir().unwrap();
    let state = Arc::new(AppState {
        orchestrator: Orchestrator::new(),
        singleflight: SingleFlight::new(),
        stats: Arc::new(mlxcache_daemon::http::Stats::default()),
        served_models: vec!["e2e-model".into()],
        sidecar: Some(
            SidecarClient::new(SidecarConfig::new(sidecar_url, "e2e-model".into())).unwrap(),
        ),
        persistence: mlxcache_daemon::persistence::Persistence::new(blobs.path()).unwrap(),
    });
    let body = serde_json::json!({
        "model": "e2e-model",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": false,
    })
    .to_string();
    let res = router(state)
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        res.status(),
        400,
        "empty tokenization must be a client error"
    );

    child.kill().expect("kill sidecar");
    child.wait().expect("reap sidecar");
}

#[tokio::test]
async fn end_to_end_streaming_sse() {
    let Some((sidecar_url, mut child)) = spawn_sidecar().await else {
        eprintln!("skipping: sidecar unavailable (install uv + sync deps)");
        return;
    };
    let blobs = tempfile::tempdir().unwrap();
    let state = Arc::new(AppState {
        orchestrator: Orchestrator::new(),
        singleflight: SingleFlight::new(),
        stats: Arc::new(mlxcache_daemon::http::Stats::default()),
        served_models: vec!["e2e-model".into()],
        sidecar: Some(
            SidecarClient::new(SidecarConfig::new(sidecar_url, "e2e-model".into())).unwrap(),
        ),
        persistence: mlxcache_daemon::persistence::Persistence::new(blobs.path()).unwrap(),
    });
    let make_app = || router(state.clone());

    let body = serde_json::json!({
        "model": "e2e-model",
        "messages": [{"role": "user", "content": "stream me"}],
        "stream": true,
    })
    .to_string();

    let res = make_app()
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), 200);
    assert_eq!(
        res.headers().get("content-type").unwrap(),
        "text/event-stream"
    );
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let text = String::from_utf8_lossy(&bytes);
    // Must lead with the mlxcache meta frame, carry token frames, end with [DONE].
    assert!(text.contains("\"mlxcache\""), "meta frame missing: {text}");
    assert!(
        text.contains("\"verdict\":\"miss\""),
        "verdict missing: {text}"
    );
    assert!(text.contains("data: [DONE]"), "terminal frame missing");
    // [DONE] must appear exactly once and be the final frame, or SSE clients
    // hang waiting or double-terminate.
    assert_eq!(text.matches("data: [DONE]").count(), 1, "duplicate [DONE]");
    assert!(
        text.trim_end().ends_with("data: [DONE]"),
        "[DONE] must be the last frame: {text}"
    );
    let token_frames = text.matches("\"token\"").count();
    assert!(
        token_frames > 1,
        "expected multiple token frames, got {token_frames}"
    );
    // The real TTFT (first generated token) is reported before the terminator.
    let ttft_pos = text.find("\"ttft_ms\"").expect("ttft frame missing");
    let done_pos = text.find("data: [DONE]").unwrap();
    assert!(ttft_pos < done_pos, "ttft frame must precede [DONE]");

    child.kill().expect("kill sidecar");
    child.wait().expect("reap sidecar");
}
