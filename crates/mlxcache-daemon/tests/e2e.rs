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

/// Spawn the real sidecar server (synthetic engine) on an ephemeral port.
/// Returns None when `uv` or the sidecar package is unavailable, so the test
/// suite degrades to a skip instead of failing in environments without Python.
// The child is reaped by the test's kill+wait on success; the panic path
// (sidecar never ready) intentionally leaks it — test process exit cleans up.
#[allow(clippy::zombie_processes)]
async fn spawn_sidecar() -> Option<(String, std::process::Child)> {
    let port = portpicker::pick_unused_port().expect("free port");
    let script = format!(
        "import sys; sys.path.insert(0, {root:?}); \
         from mlxcache_sidecar import server; \
         server.Handler.engine = server.make_engine('e2e-model'); \
         server.ThreadingHTTPServer(('127.0.0.1', {port}), server.Handler).serve_forever()",
        root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../sidecar"),
        port = port
    );
    let child = match std::process::Command::new("uv")
        .args(["run", "python", "-c", &script])
        .spawn()
    {
        Ok(c) => c,
        Err(_) => return None, // uv not installed: skip
    };
    let url = format!("http://127.0.0.1:{port}");
    // Wait for readiness (uv may sync deps on first run).
    for _ in 0..100 {
        if reqwest::get(format!("{url}/health")).await.is_ok() {
            return Some((url, child));
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    // Sidecar never came up (missing deps?). Reap and skip rather than hang.
    let mut child = child;
    let _ = child.kill();
    let _ = child.wait();
    None
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
    let token_frames = text.matches("\"token\"").count();
    assert!(
        token_frames > 1,
        "expected multiple token frames, got {token_frames}"
    );

    child.kill().expect("kill sidecar");
    child.wait().expect("reap sidecar");
}
