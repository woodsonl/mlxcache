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
async fn concurrent_identical_requests_share_one_prefill() {
    // The core R1-3 promise: N identical uncached requests must trigger ONE
    // prefill, and every follower must still be served (from the leader's
    // freshly published checkpoint), not re-prefill from scratch.
    let Some((sidecar_url, mut child)) =
        spawn_sidecar_with_env(&[("MLXCACHE_PREFILL_DELAY", "0.5")]).await
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
    let mut handles = Vec::new();
    for _ in 0..N {
        let app = router(state.clone());
        let body = body.clone();
        handles.push(tokio::spawn(async move {
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

    child.kill().expect("kill sidecar");
    child.wait().expect("reap sidecar");
}
