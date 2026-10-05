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

/// Standard e2e harness state: one served model, a sidecar client for it, a
/// fresh blob store. Tests needing special clients (timeouts, trace) build
/// their own literal — everything else routes through here so the AppState
/// shape has exactly one canonical construction site.
fn app_state(sidecar_url: &str, blobs: &tempfile::TempDir) -> Arc<AppState> {
    Arc::new(AppState {
        orchestrator: Orchestrator::new(),
        singleflight: SingleFlight::new(),
        stats: Arc::new(mlxcache_daemon::http::Stats::default()),
        served_models: vec!["e2e-model".into()],
        sidecar: Some(
            SidecarClient::new(SidecarConfig::new(
                sidecar_url.to_owned(),
                "e2e-model".into(),
            ))
            .unwrap(),
        ),
        persistence: mlxcache_daemon::persistence::Persistence::new(blobs.path()).unwrap(),
        trace: None,
    })
}

#[tokio::test]
async fn unusable_blob_is_quarantined_and_served_from_scratch() {
    // A published checkpoint whose blob vanished (deleted, disk fault) must not
    // 502 forever. The daemon quarantines the entry and retries from scratch;
    // the following identical request then misses cleanly (no 502).
    let (sidecar_url, mut child) = spawn_sidecar().await;
    let blobs = tempfile::tempdir().unwrap();
    let state = app_state(&sidecar_url, &blobs);
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

#[cfg(unix)]
#[tokio::test]
async fn eviction_reaper_unlinks_and_stats_and_next_request_misses() {
    // Full-path eviction coverage: a real request publishes a checkpoint; a
    // reaper pass with cap 0 and window 0 must (1) remove it from the index,
    // (2) unlink the blob file (an orphan would resurrect on the next restart
    // rebuild), (3) surface in /stats, and (4) leave the next identical request
    // a clean miss served from scratch. This is the only place the reaper's
    // interaction with the live request path is proven end to end.
    let (sidecar_url, mut child) = spawn_sidecar().await;
    let blobs = tempfile::tempdir().unwrap();
    let state = app_state(&sidecar_url, &blobs);
    let body = serde_json::json!({
        "model": "e2e-model",
        "messages": [{"role": "user", "content": "evict me"}],
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
    let get_stats = || {
        let app = router(state.clone());
        async move {
            app.oneshot(
                axum::http::Request::builder()
                    .method("GET")
                    .uri("/stats")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap()
        }
    };

    // Request 1: miss → publish.
    let res = post(body.clone()).await;
    assert_eq!(res.status(), 200);
    assert_eq!(
        state.orchestrator.published_count(),
        1,
        "checkpoint published"
    );
    let blob = state.persistence.list_blobs().unwrap().pop().unwrap();

    // Reaper pass: cap 0 + window 0 → the just-published entry is NOT recency-
    // protected (window 0) and has no published extension → evicted. Goes
    // through AppState::evict_pass — the same single path the background
    // reaper uses — so the /stats counter is exercised here too.
    let evicted = state.evict_pass(0, 0, std::time::Duration::ZERO);
    assert_eq!(evicted, 1, "the single published entry must be reaped");
    assert_eq!(state.orchestrator.published_count(), 0);
    assert!(
        !blob.exists(),
        "evicted blob file must be unlinked, or the next restart resurrects it"
    );

    // /stats reflects the eviction.
    let res = get_stats().await;
    assert_eq!(res.status(), 200);
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["evictions"], 1, "/stats must report the eviction");
    assert_eq!(v["checkpoints_published"], 0);

    // Request 2: identical prompt → clean miss from scratch (no 502, no
    // reference to the unlinked blob).
    let res = post(body).await;
    assert_eq!(res.status(), 200);
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        v["mlxcache"]["verdict"], "miss",
        "evicted entry must not serve"
    );

    child.kill().expect("kill sidecar");
    child.wait().expect("reap sidecar");
}

#[cfg(unix)]
#[tokio::test]
async fn failed_startup_scan_writes_no_blob_and_serves_from_scratch() {
    // Regression (coverage audit): after a FAILED startup scan the generation
    // floor is unknown, so the handler must serve from scratch and write NO
    // checkpoint (writing first would leak an unindexed file on every request).
    use std::os::unix::fs::PermissionsExt;
    let (sidecar_url, mut child) = spawn_sidecar().await;
    let blobs = tempfile::tempdir().unwrap();
    let persistence = mlxcache_daemon::persistence::Persistence::new(blobs.path()).unwrap();
    // Make the dir unreadable so list_blobs fails, then run the real startup
    // scan: it sets recovery_failed and blocks publishing.
    let mut perms = std::fs::metadata(blobs.path()).unwrap().permissions();
    let orig = perms.mode();
    perms.set_mode(0o000);
    std::fs::set_permissions(blobs.path(), perms).unwrap();
    let orchestrator = Orchestrator::new();
    let report = orchestrator.rebuild_from_disk(&persistence);
    assert!(!report.errors.is_empty(), "scan must report the failure");
    assert!(!orchestrator.can_publish(), "publishing must be blocked");
    // Restore access so the handler can at least attempt (and refuse) a write.
    let mut perms = std::fs::metadata(blobs.path()).unwrap().permissions();
    perms.set_mode(orig);
    std::fs::set_permissions(blobs.path(), perms).unwrap();

    let state = Arc::new(AppState {
        orchestrator,
        singleflight: SingleFlight::new(),
        stats: Arc::new(mlxcache_daemon::http::Stats::default()),
        served_models: vec!["e2e-model".into()],
        sidecar: Some(
            SidecarClient::new(SidecarConfig::new(sidecar_url, "e2e-model".into())).unwrap(),
        ),
        persistence,
        trace: None,
    });
    let body = serde_json::json!({
        "model": "e2e-model",
        "messages": [{"role": "user", "content": "no cache please"}],
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
    assert_eq!(res.status(), 200, "request is still served");
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["mlxcache"]["verdict"], "miss");
    assert_eq!(
        state.orchestrator.published_count(),
        0,
        "no entry may be indexed while the floor is unknown"
    );
    let ckpts = std::fs::read_dir(blobs.path())
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().ends_with(".ckpt"))
        .count();
    assert_eq!(ckpts, 0, "no blob file may be written: {ckpts}");

    child.kill().expect("kill sidecar");
    child.wait().expect("reap sidecar");
}

#[tokio::test]
async fn one_token_prompt_serves_and_publishes_no_blob() {
    // A one-token prompt caches nothing (empty adapter payload). The daemon must
    // serve it from scratch, publish no checkpoint, and never report a phantom
    // hit on the repeat request.
    let (sidecar_url, mut child) = spawn_sidecar_with_env(&[("MLXCACHE_TOKENIZE_ONE", "1")]).await;
    let blobs = tempfile::tempdir().unwrap();
    let state = app_state(&sidecar_url, &blobs);
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
    let (sidecar_url, mut child) = spawn_sidecar_with_env(&[
        ("MLXCACHE_TOKENIZE_ONE", "1"),
        ("MLXCACHE_PREFILL_DELAY", "1.5"),
    ])
    .await;
    let blobs = tempfile::tempdir().unwrap();
    let state = app_state(&sidecar_url, &blobs);
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
async fn stream_leg_traces_the_settled_verdict() {
    // Testing review 2026-10-04: the RT#7 settle test covered only the JSON
    // leg. A stream whose blob is 422'd at open (corrupt ancestor) must
    // also trace the miss it became after quarantine + scratch retry.
    let (sidecar_url, mut child) =
        spawn_sidecar_with_env(&[("MLXCACHE_TOKENIZE_GROW", "grow me")]).await;
    let blobs = tempfile::tempdir().unwrap();
    let tracedir = tempfile::tempdir().unwrap();
    let trace_path = tracedir.path().join("trace.jsonl");
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
        trace: Some(mlxcache_daemon::trace::TraceWriter::from_path(&trace_path).unwrap()),
    });
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

    let res = post(
        serde_json::json!({
            "model": "e2e-model",
            "messages": [{"role": "user", "content": "grow me"}],
        })
        .to_string(),
    )
    .await;
    assert_eq!(res.status(), 200);

    let blob = state.persistence.list_blobs().unwrap().pop().unwrap();
    std::fs::write(&blob, b"JUNK").unwrap();

    let res = post(
        serde_json::json!({
            "model": "e2e-model",
            "messages": [{"role": "user", "content": "grow me streamed"}],
            "stream": true,
        })
        .to_string(),
    )
    .await;
    assert_eq!(
        res.status(),
        200,
        "corrupt ancestor must not fail the stream"
    );
    let _ = res.into_body().collect().await.unwrap().to_bytes();

    let trace = wait_for_trace_lines(&trace_path, 2).await;
    let rec2: serde_json::Value =
        serde_json::from_str(trace.lines().nth(1).unwrap_or_default()).unwrap_or_default();
    assert_eq!(
        rec2["verdict"], "miss",
        "the stream leg must trace the settled miss: {rec2}"
    );
    assert_eq!(rec2["prefill_from"].as_u64(), Some(0), "{rec2}");

    child.kill().expect("kill sidecar");
    child.wait().expect("reap sidecar");
}

#[tokio::test]
async fn zero_max_tokens_yields_empty_completion_in_sdk_shape() {
    // Some(0) is explicitly honored (a client asking for zero tokens gets
    // zero); pin the choices[] shape for it: empty content, zero-token
    // usage, finish_reason "stop" (the max_tokens>0 length rule does not
    // apply to a client-requested zero).
    let (sidecar_url, mut child) = spawn_sidecar().await;
    let blobs = tempfile::tempdir().unwrap();
    let state = app_state(&sidecar_url, &blobs);
    let res = router(state)
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::json!({
                        "model": "e2e-model",
                        "messages": [{"role": "user", "content": "nothing to add"}],
                        "max_tokens": 0,
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), 200);
    let v: serde_json::Value =
        serde_json::from_slice(&res.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(v["choices"][0]["message"]["content"], "", "{v}");
    assert_eq!(v["choices"][0]["finish_reason"], "stop", "{v}");
    assert_eq!(v["usage"]["completion_tokens"], 0, "{v}");

    child.kill().expect("kill sidecar");
    child.wait().expect("reap sidecar");
}
#[tokio::test]
async fn nonstream_response_decodes_with_stock_openai_sdk_shape() {
    // The choices[] compatibility layer: id/object/created/model/choices/usage
    // are REQUIRED by stock SDK response models (missing-required fails
    // validation; extras like `mlxcache` are allowed). A client doing
    // `response.choices[0].message.content` must get the sidecar's
    // detokenized text — identical to what the streaming path emits.
    let (sidecar_url, mut child) = spawn_sidecar().await;
    let blobs = tempfile::tempdir().unwrap();
    let state = app_state(&sidecar_url, &blobs);
    let res = router(state)
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::json!({
                        "model": "e2e-model",
                        "messages": [{"role": "user", "content": "sdk shape please"}],
                        "max_tokens": 4,
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), 200);
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["object"], "chat.completion", "{v}");
    assert!(
        v["id"].as_str().unwrap_or("").starts_with("chatcmpl-"),
        "{v}"
    );
    assert!(v["created"].as_u64().is_some(), "{v}");
    assert_eq!(v["model"], "e2e-model", "{v}");
    // The synthetic engine generates exactly max_tokens (4): OpenAI
    // semantics say a cap-truncated completion ends "length", not "stop".
    assert_eq!(v["choices"][0]["finish_reason"], "length", "{v}");
    assert_eq!(v["choices"][0]["message"]["role"], "assistant", "{v}");
    assert_eq!(
        v["choices"][0]["message"]["content"], "tok0 tok1 tok2 tok3 ",
        "decoded text must match the stream path's pieces: {v}"
    );
    assert_eq!(v["usage"]["completion_tokens"], 4, "{v}");
    assert!(v["usage"]["prompt_tokens"].as_u64().unwrap() > 0, "{v}");
    // Additive: the raw fields stay for existing consumers.
    assert!(v["generated_tokens"].as_array().unwrap().len() == 4, "{v}");
    assert!(v["mlxcache"]["verdict"].is_string(), "{v}");

    child.kill().expect("kill sidecar");
    child.wait().expect("reap sidecar");
}

#[tokio::test]
async fn stream_chunks_decode_with_stock_openai_sdk_shape() {
    // Every SSE frame a parser sees must be a valid chat.completion.chunk
    // (id/object/created/model/choices required) EXCEPT the error frames,
    // which follow OpenAI's own error-stream shape. Deltas accumulate into
    // the same text the non-stream body returns; the verdict rides as a
    // top-level extra on the first chunk.
    let (sidecar_url, mut child) = spawn_sidecar().await;
    let blobs = tempfile::tempdir().unwrap();
    let state = app_state(&sidecar_url, &blobs);
    let res = router(state)
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::json!({
                        "model": "e2e-model",
                        "messages": [{"role": "user", "content": "sdk shape stream"}],
                        "stream": true,
                        "max_tokens": 3,
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), 200);
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let text = String::from_utf8_lossy(&bytes).into_owned();

    // Walk every data: frame like an SDK would.
    let mut deltas = String::new();
    let mut saw_first = false;
    let mut saw_stop = false;
    let mut last_was_done = false;
    for line in text.lines() {
        let Some(payload) = line.strip_prefix("data: ") else {
            continue;
        };
        if payload.trim() == "[DONE]" {
            assert!(saw_stop, "finish_reason=length must precede [DONE]: {text}");
            last_was_done = true;
            continue;
        }
        let c: serde_json::Value = serde_json::from_str(payload)
            .unwrap_or_else(|e| panic!("every frame must be valid JSON ({e}): {payload}"));
        if c.get("error").is_some() {
            continue; // OpenAI's error-stream shape — allowed
        }
        assert_eq!(c["object"], "chat.completion.chunk", "{payload}");
        assert!(
            c["id"].as_str().unwrap_or("").starts_with("chatcmpl-"),
            "{payload}"
        );
        assert!(c["created"].as_u64().is_some(), "{payload}");
        assert_eq!(c["model"], "e2e-model", "{payload}");
        let choice = &c["choices"][0];
        assert!(choice.is_object(), "every chunk carries choices: {payload}");
        if !saw_first {
            saw_first = true;
            assert_eq!(choice["delta"]["role"], "assistant", "{payload}");
            assert!(
                c["mlxcache"]["verdict"].is_string(),
                "verdict rides the first chunk: {payload}"
            );
        }
        if let Some(piece) = choice["delta"]["content"].as_str() {
            deltas.push_str(piece);
        }
        if choice["finish_reason"] == "length" {
            // max_tokens=3 and 3 tokens generated: truncated by cap.
            saw_stop = true;
        }
    }
    assert!(last_was_done, "stream must end with [DONE]: {text}");
    assert_eq!(
        deltas, "tok0 tok1 tok2 ",
        "deltas must accumulate to the detokenized text: {text}"
    );

    child.kill().expect("kill sidecar");
    child.wait().expect("reap sidecar");
}

#[tokio::test]
async fn one_token_streaming_serves_and_publishes_no_blob() {
    // The streaming path for the no-publish case: a one-token prompt must stream
    // a 200 SSE with a miss verdict and publish nothing.
    let (sidecar_url, mut child) = spawn_sidecar_with_env(&[("MLXCACHE_TOKENIZE_ONE", "1")]).await;
    let blobs = tempfile::tempdir().unwrap();
    let state = app_state(&sidecar_url, &blobs);
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
    let (sidecar_url, mut child) = spawn_sidecar().await;
    let blobs = tempfile::tempdir().unwrap();

    let state = app_state(&sidecar_url, &blobs);
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
    // The synthetic tokenizer emits 8 tokens. A cold miss built its own KV; no
    // KV came from an earlier request, so prefill_from/tokens_cached report 0
    // (README: "0 on a miss"). Generation still resumes from the blob the leader
    // just wrote, but that is not cross-request reuse.
    assert_eq!(
        v["mlxcache"]["prefill_from"], 0,
        "a cold miss reused no prior KV"
    );
    assert_eq!(
        v["mlxcache"]["tokens_cached"], 0,
        "a cold miss cached nothing from a prior request"
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
    // tokens_cached must agree with prefill_from: both name the KV actually in
    // hand (covered = len-1 = 7), not the matched prefix length (8). A client
    // reading tokens_cached=8 would believe one more token of KV is cached than
    // the adapter can resume from.
    assert_eq!(
        v["mlxcache"]["tokens_cached"], 7,
        "tokens_cached must report covered KV, matching prefill_from"
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
    // tokens_cached accumulates covered KV (miss=0, hit=7 for the 8-token
    // prefix), matching the per-response field. Summing matched_tokens would
    // report 8 here and inflate hit_rate.
    assert_eq!(
        stats["tokens_cached"], 7,
        "/stats must accumulate covered KV, not matched prefix length"
    );

    child.kill().expect("kill sidecar");
    child.wait().expect("reap sidecar");
}

/// Spawn the sidecar with extra environment variables (test knobs), e.g. a
/// prefill delay to widen the single-flight window, or an empty tokenizer.
#[allow(clippy::zombie_processes)]
/// RAII wrapper: the sidecar dies even when a test panics mid-flight. The
/// tail-of-test kill+wait only runs on the happy path — failed sweeps leaked
/// one `uv run python` engine per failed test (64 strays held real RAM after
/// the 2026-10-04 sweeps; engines are GB-scale residents).
struct SidecarHandle {
    child: std::process::Child,
    // The child's stdout after the PORT line was consumed: held open and
    // drained so the child never blocks on a full pipe mid-test, and closed
    // by the Drop kill (EOF ends the drain).
    _stdout_drain: Option<std::thread::JoinHandle<()>>,
}

impl Drop for SidecarHandle {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(handle) = self._stdout_drain.take() {
            let _ = handle.join();
        }
    }
}

impl std::ops::Deref for SidecarHandle {
    type Target = std::process::Child;

    fn deref(&self) -> &std::process::Child {
        &self.child
    }
}

impl std::ops::DerefMut for SidecarHandle {
    fn deref_mut(&mut self) -> &mut std::process::Child {
        &mut self.child
    }
}

async fn spawn_sidecar_with_env(env: &[(&str, &str)]) -> (String, SidecarHandle) {
    // The health probe needs its own deadline: `reqwest::get` is a bare
    // client with NO timeout, so under a loaded machine (parallel e2e
    // spawns) a half-open connect hangs the poll forever and the test
    // wedges instead of failing. A per-attempt timeout converts that into
    // the intended "not up yet" retry.
    let probe = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(2))
        .timeout(std::time::Duration::from_secs(3))
        .build()
        .expect("probe client");
    // The child binds port 0 (kernel-assigned) and prints its actual port —
    // the URL is derived from OUR child's own bind, so a health answer on it
    // cannot come from another test's sidecar. The previous portpicker scheme
    // raced: two spawns milliseconds apart could take the same "unused" port,
    // and a foreign server answering health in the alive-check window made
    // the daemon 503 the moment that server's own test killed it
    // (issue #31).
    let script = format!(
        "import sys; sys.path.insert(0, {root:?}); \
         from mlxcache_sidecar import server; \
         server.Handler.engine = server.make_engine('e2e-model'); \
         srv = server.BurstServer(('127.0.0.1', 0), server.Handler); \
         print('PORT=%d' % srv.server_address[1], flush=True); \
         srv.serve_forever()",
        root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../sidecar"),
    );
    // Spawn the venv python DIRECTLY, not via `uv run`: uv keeps itself
    // resident and spawns python as a CHILD, so SIGKILL to the Child we
    // hold reaps uv and ORPHANS the server (the 64-stray leak held real
    // RAM across sweeps, 2026-10-04) — and 18 concurrent `uv run` starts
    // contend on uv's environment lock, the original parallel-e2e flake
    // source. One process, one kill, no lock.
    let python = concat!(env!("CARGO_MANIFEST_DIR"), "/../../.venv/bin/python");
    assert!(
        std::path::Path::new(python).exists(),
        "e2e needs the repo venv at {python:?} (uv sync)"
    );
    let mut cmd = std::process::Command::new(python);
    cmd.arg("-c").arg(&script);
    cmd.stdout(std::process::Stdio::piped());
    for (k, v) in env {
        cmd.env(k, v);
    }
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => panic!("sidecar spawn failed: {e}"),
    };
    // Read the PORT line with a deadline: the kernel-assigned port is the
    // only one that can truthfully answer for this child. The thread hands
    // the stdout back either way — the drain below needs the pipe open.
    let stdout = child.stdout.take().expect("piped stdout");
    let port = {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            use std::io::BufRead;
            let mut reader = std::io::BufReader::new(stdout);
            let mut line = String::new();
            let _ = reader.read_line(&mut line);
            let port = line.strip_prefix("PORT=").map(|p| p.trim().to_string());
            let _ = tx.send((port, reader.into_inner()));
        });
        match rx.recv_timeout(std::time::Duration::from_secs(30)) {
            Ok((port_opt, stdout)) => {
                let port = port_opt.unwrap_or_else(|| "<no PORT= line on stdout>".into());
                match port.parse::<u16>() {
                    Ok(p) => (p, stdout),
                    Err(_) => {
                        let _ = child.kill();
                        let _ = child.wait();
                        panic!("sidecar printed an unparsable port: {port:?}");
                    }
                }
            }
            Err(_) => {
                let _ = child.kill();
                let _ = child.wait();
                panic!("sidecar never printed its port within 30s");
            }
        }
    };
    let (port, stdout) = port;
    // Drain the rest of stdout in the background: nothing this child may
    // print to stdout may block it on a full pipe mid-test.
    let drain = std::thread::spawn(move || {
        use std::io::Read;
        let mut stdout = stdout;
        let mut buf = [0u8; 4096];
        while let Ok(n) = stdout.read(&mut buf) {
            if n == 0 {
                break;
            }
        }
    });
    let url = format!("http://127.0.0.1:{port}");
    let alive =
        |child: &mut std::process::Child| child.try_wait().map_or(true, |status| status.is_none());
    for attempt in 0..100 {
        // Trust a health answer ONLY while OUR child is alive, and re-check
        // alive AFTER the answer: a crash in between frees the port for an
        // instant rebinding, and accepting a foreign answer would poison the
        // whole test.
        if !alive(&mut child) {
            panic!("sidecar exited during startup (attempt {attempt})");
        }
        if probe.get(format!("{url}/health")).send().await.is_ok() {
            if !alive(&mut child) {
                panic!("sidecar exited between health answer and accept");
            }
            let handle = SidecarHandle {
                child,
                _stdout_drain: Some(drain),
            };
            return (url, handle);
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    // The child is typically STILL ALIVE here (slow or wedged startup) and
    // Child does not kill on drop — without this, the panic leaks a live
    // engine (port + RAM) and the drain thread blocks forever on the pipe.
    let _ = child.kill();
    let _ = child.wait();
    panic!("sidecar never became healthy after 100 probes at {url}");
}

/// One sidecar, immediately ready, no test knobs.
async fn spawn_sidecar() -> (String, SidecarHandle) {
    spawn_sidecar_with_env(&[]).await
}

#[tokio::test]
async fn two_models_same_tokens_do_not_share_a_blob() {
    // Two served models, same prompt (the synthetic engine derives tokens from
    // the prompt only, so both produce identical token ids). Their checkpoints
    // MUST land in separate blob files: sharing one file would let the second
    // model overwrite the first, whose index entry would then load foreign KV.
    let (sidecar_url, mut child) = spawn_sidecar().await;
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
        trace: None,
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
    // The index holds one entry per token prefix. Both models tokenize the same
    // prompt, so the second publish replaces the first at that node and reclaims
    // its immutable file — exactly one blob survives, with no generation leak.
    // The evicted model is a fingerprint miss (recomputed), never an error.
    assert_eq!(
        ckpts.len(),
        1,
        "the superseded generation must be reclaimed: {ckpts:?}"
    );

    child.kill().expect("kill sidecar");
    child.wait().expect("reap sidecar");
}

#[tokio::test]
async fn trace_capture_writes_one_jsonl_record_per_request() {
    // Step 5 (success criteria 3-4): with MLXCACHE_TRACE set, every served
    // request (warming included) lands as one JSONL record with the verdict
    // the request actually got — the ground truth the replay harness scores.
    // Proven end to end over the synthetic engine: miss -> hit -> another
    // miss = exactly 3 records, verdicts in serve order, prefill_from = the
    // single covered-KV definition.
    let (sidecar_url, mut child) = spawn_sidecar().await;
    let dir = tempfile::tempdir().unwrap();
    let trace_path = dir.path().join("trace.jsonl");
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
        trace: Some(mlxcache_daemon::trace::TraceWriter::from_path(&trace_path).unwrap()),
    });
    let post = |content: &str| {
        let body = serde_json::json!({
            "model": "e2e-model",
            "messages": [{"role": "user", "content": content}],
            "stream": false,
        })
        .to_string();
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

    // 1) miss (publishes), 2) hit, 3) miss (distinct prompt).
    for (res, want_verify) in [
        (post("trace me").await, "miss"),
        (post("trace me").await, "hit"),
        (post("different prompt").await, "miss"),
    ] {
        assert_eq!(res.status(), 200);
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["mlxcache"]["verdict"], want_verify);
    }

    // Wait for the writer thread to land all three (FIFO channel; the file is
    // the only side channel we can observe without a join handle).
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    let body = loop {
        let body = std::fs::read_to_string(&trace_path).unwrap_or_default();
        if body.lines().count() >= 3 {
            break body;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "trace file never reached 3 records: {body}"
        );
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    };

    let records: Vec<serde_json::Value> = body
        .lines()
        .map(|l| serde_json::from_str(l).expect("each trace line is valid JSON"))
        .collect();
    assert_eq!(records.len(), 3, "{body}");
    assert_eq!(records[0]["verdict"], "miss");
    assert_eq!(records[1]["verdict"], "hit");
    assert_eq!(records[2]["verdict"], "miss");
    // The covered-KV definition: hit covered = matched-1 (the checkpoint
    // caches tokens[:-1]), miss = 0 — the SAME value /stats and prefill_from
    // report, which is the no-drift guarantee the capture inherits.
    assert_eq!(records[0]["prefill_from"], 0);
    assert_eq!(
        records[1]["prefill_from"],
        records[0]["n_tokens"].as_i64().unwrap() - 1,
        "hit covers matched-1 (the tokens[:-1] checkpoint convention)"
    );
    assert_eq!(
        records[1]["matched_tokens"], records[0]["n_tokens"],
        "the hit matched the entire published prefix"
    );
    assert!(
        records[0]["ts_ms"].as_u64().unwrap() > 0,
        "records carry unix-ms timestamps for replay ordering"
    );
    // Step 5 fidelity: the record embeds the EXACT request payload so a
    // replay re-sends byte-equivalent requests (same text, same tokenizer,
    // same tokens).
    assert_eq!(
        records[0]["messages"][0]["content"], "trace me",
        "messages are captured verbatim for replay fidelity"
    );

    child.kill().expect("kill sidecar");
    child.wait().expect("reap sidecar");
}

#[tokio::test]
async fn q8_and_f16_blobs_never_share_a_file() {
    // T12 adoption: the same model+prompt under two KV tiers (q8 via the
    // sidecar's MLXCACHE_KV_BITS knob vs f16 knobless) must publish to
    // DIFFERENT blob files and never serve each other (R1-1: the fingerprint
    // pins the quantization config). Two sidecars over ONE blob dir — exactly
    // an operator flipping the knob between runs against persisted state.
    let (q8_url, mut q8_child) =
        spawn_sidecar_with_env(&[("MLXCACHE_KV_BITS", "8"), ("MLXCACHE_KV_GROUP_SIZE", "64")])
            .await;
    let (f16_url, mut f16_child) = spawn_sidecar_with_env(&[]).await;
    let blobs = tempfile::tempdir().unwrap();

    let body = serde_json::json!({
        "model": "e2e-model",
        "messages": [{"role": "user", "content": "quantized tier"}],
        "stream": false,
    })
    .to_string();
    let state = |url: String| {
        Arc::new(AppState {
            orchestrator: Orchestrator::new(),
            singleflight: SingleFlight::new(),
            stats: Arc::new(mlxcache_daemon::http::Stats::default()),
            served_models: vec!["e2e-model".into()],
            sidecar: Some(SidecarClient::new(SidecarConfig::new(url, "e2e-model".into())).unwrap()),
            persistence: mlxcache_daemon::persistence::Persistence::new(blobs.path()).unwrap(),
            trace: None,
        })
    };
    let post = |state: Arc<AppState>, body: String| async move {
        router(state)
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap()
    };
    let blob_names = || {
        std::fs::read_dir(blobs.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.ends_with(".ckpt"))
            .collect::<Vec<_>>()
    };

    let state_q8 = state(q8_url.clone());
    // 1) q8 run: miss → publish.
    let res = post(state_q8.clone(), body.clone()).await;
    assert_eq!(res.status(), 200);
    assert_eq!(state_q8.orchestrator.published_count(), 1, "q8 published");
    let q8_blobs = blob_names();
    assert_eq!(q8_blobs.len(), 1);

    // 2) same prompt through the f16 sidecar: the tier is a DIFFERENT
    //    fingerprint (kv_bits 0 vs 8 in /tokenize → blob_key fold) → the q8
    //    checkpoint must NOT serve it, and the publish must not reclaim the
    //    q8 blob file.
    let state_f16 = state(f16_url.clone());
    let res = post(state_f16.clone(), body.clone()).await;
    assert_eq!(res.status(), 200);
    assert_eq!(
        state_f16.orchestrator.published_count(),
        1,
        "the f16 tier must publish its own checkpoint, not serve the q8 one"
    );
    let both = blob_names();
    assert_eq!(
        both.len(),
        2,
        "two tiers = two blob files; a shared name would overwrite the other tier: {both:?}"
    );
    assert!(
        q8_blobs.iter().all(|b| both.contains(b)),
        "the q8 blob must survive the f16 publish (no reclaim ping-pong)"
    );

    // 3) the q8 daemon still hits ITS checkpoint: the other tier's publish
    //    changed nothing for it.
    let res = post(state_q8.clone(), body).await;
    assert_eq!(res.status(), 200);
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["mlxcache"]["verdict"], "hit", "same tier must hit");

    q8_child.kill().expect("kill q8 sidecar");
    q8_child.wait().expect("reap q8 sidecar");
    f16_child.kill().expect("kill f16 sidecar");
    f16_child.wait().expect("reap f16 sidecar");
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
    let (sidecar_url, mut child) =
        spawn_sidecar_with_env(&[("MLXCACHE_PREFILL_DELAY", "2.0")]).await;
    let blobs = tempfile::tempdir().unwrap();
    let state = app_state(&sidecar_url, &blobs);

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

    // Wait for the leader to begin prefilling, then release the followers, so
    // the requests genuinely overlap (with the leader held open by
    // MLXCACHE_PREFILL_DELAY). If the leader never starts, fail loudly rather
    // than release unlocked, which would re-elect a follower and mask the fault.
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

    // The user-visible contract: N identical uncached requests trigger one
    // prefill, and every request is served. Whether a follower takes the
    // internal Role::Follower path or adopts the blob after the leader publishes
    // is a scheduling detail (see the singleflight unit tests, which exercise
    // the follower path directly with a waiter-count assertion). Here we hold
    // the leader open so the requests genuinely overlap, then assert the outcome.
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

    // Exactly one request (the leader) ran the prefill; the rest were served
    // from its published checkpoint (hit/partial), so none re-prefilled.
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

#[cfg(unix)]
#[tokio::test]
async fn publish_failure_serves_leader_and_followers_from_scratch() {
    // OV6/D2 fault injection: the checkpoint WRITE fails (read-only blob dir →
    // EACCES, the same DiskFull path ENOSPC takes) while requests are in
    // flight. The contract: the LEADER continues from scratch (it already
    // prefilled), and every coalesced FOLLOWER is served too — never a 502.
    // Before the fix the leader signalled Err into single-flight and every
    // follower 502ed even though nothing was wrong with their request.
    use std::os::unix::fs::PermissionsExt;
    let (sidecar_url, mut child) =
        spawn_sidecar_with_env(&[("MLXCACHE_PREFILL_DELAY", "2.0")]).await;
    let blobs = tempfile::tempdir().unwrap();
    // Read-only BEFORE any request: prefill succeeds (sidecar-side), the
    // daemon-side publish fails on temp-file creation.
    let mut perms = std::fs::metadata(blobs.path()).unwrap().permissions();
    let orig = perms.mode();
    perms.set_mode(0o555);
    std::fs::set_permissions(blobs.path(), perms).unwrap();

    let state = app_state(&sidecar_url, &blobs);

    let body = serde_json::json!({
        "model": "e2e-model",
        "messages": [{"role": "user", "content": "write me if you can"}],
        "stream": false,
    })
    .to_string();

    const N: usize = 8;
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
            let status = res.status();
            let bytes = res.into_body().collect().await.unwrap().to_bytes();
            let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            (
                status,
                v["mlxcache"]["verdict"].as_str().unwrap().to_string(),
            )
        }));
    }

    // Release followers once the leader is mid-prefill (held open by
    // MLXCACHE_PREFILL_DELAY), so they coalesce behind it deterministically.
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
        "leader never began prefilling; cannot test the publish fault"
    );
    start_followers_tx.send_replace(true);

    let mut results = Vec::new();
    for h in handles {
        results.push(h.await.unwrap());
    }

    // EVERY request — leader and followers alike — must be served, from
    // scratch, with a miss verdict. Not one 502.
    let bad: Vec<_> = results.iter().filter(|(s, _)| *s != 200).collect();
    assert!(
        bad.is_empty(),
        "a failed checkpoint write must never fail a request: {bad:?}"
    );
    let leader_verdict = &results[0].1;
    assert_eq!(
        leader_verdict, "miss",
        "leader served from scratch after publish failure"
    );
    assert!(
        results.iter().all(|(_, v)| v == "miss"),
        "all requests report scratch: {results:?}"
    );

    // The prefill itself ran exactly once (the leader's), and nothing was
    // published or indexed.
    let stats: serde_json::Value = reqwest::get(format!("{sidecar_url}/stats"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(stats["prefill_count"].as_u64(), Some(1));
    assert_eq!(state.orchestrator.published_count(), 0);
    assert_eq!(state.orchestrator.quarantined_count(), 0);

    // Restore permissions so the tempdir can be cleaned up.
    let mut perms = std::fs::metadata(blobs.path()).unwrap().permissions();
    perms.set_mode(orig);
    std::fs::set_permissions(blobs.path(), perms).unwrap();

    child.kill().expect("kill sidecar");
    child.wait().expect("reap sidecar");
}

#[tokio::test]
async fn end_to_end_restart_resumes_from_disk() {
    // R1-4 across the REAL HTTP path: request 1 publishes a checkpoint; a fresh
    // daemon (new AppState + rebuild_from_disk on the same blob dir) must
    // serve request 2 as a hit without re-prefilling.
    let (sidecar_url, mut child) = spawn_sidecar().await;
    let blobs = tempfile::tempdir().unwrap();
    let make_state = || {
        let state = app_state(&sidecar_url, &blobs);
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
    let (sidecar_url, mut child) =
        spawn_sidecar_with_env(&[("MLXCACHE_TOKENIZE_EMPTY", "1")]).await;
    let blobs = tempfile::tempdir().unwrap();
    let state = app_state(&sidecar_url, &blobs);
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
    let (sidecar_url, mut child) = spawn_sidecar().await;
    let blobs = tempfile::tempdir().unwrap();
    let state = app_state(&sidecar_url, &blobs);
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
                .body(Body::from(body.clone()))
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
    // First request (miss): the leader prefills and publishes, but no prior KV
    // was reused, so tokens_cached and prefill_from are both 0 (README: "0 on a
    // miss"). They must agree with each other.
    assert!(
        text.contains("\"tokens_cached\":0") && text.contains("\"prefill_from\":0"),
        "cold-miss stream must report 0 covered KV: {text}"
    );

    // Second identical stream request: a hit. The meta frame must report
    // tokens_cached = prefill_from = covered KV (8-token prefix -> 7), not 8.
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
    let text = String::from_utf8_lossy(&bytes);
    assert!(
        text.contains("\"verdict\":\"hit\""),
        "second stream must be a hit: {text}"
    );
    assert!(
        text.contains("\"tokens_cached\":7") && text.contains("\"prefill_from\":7"),
        "hit stream must report covered KV (7), not matched prefix (8): {text}"
    );

    child.kill().expect("kill sidecar");
    child.wait().expect("reap sidecar");
}

#[tokio::test]
async fn partial_hit_delta_prefills_only_the_delta() {
    // OV3: a growing conversation must NOT re-prefill the whole prompt. Request
    // 2 extends request 1's token prefix, so the daemon passes the matched
    // ancestor blob to /prefill and the adapter prefills ONLY the uncovered
    // delta. Proven through the sidecar's phase accounting: the second prefill
    // processes 8 model steps (16 tokens, 7 covered by the ancestor, minus the
    // final token) instead of the 15 a scratch prefill would.
    let (sidecar_url, mut child) =
        spawn_sidecar_with_env(&[("MLXCACHE_TOKENIZE_GROW", "grow me")]).await;
    let blobs = tempfile::tempdir().unwrap();
    let state = app_state(&sidecar_url, &blobs);

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

    // Request 1: the base prompt — a full miss, publishes the 8-token ancestor.
    let res = post(
        serde_json::json!({
            "model": "e2e-model",
            "messages": [{"role": "user", "content": "grow me"}],
            "stream": false,
        })
        .to_string(),
    )
    .await;
    assert_eq!(res.status(), 200);
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["mlxcache"]["verdict"], "miss");
    assert_eq!(state.persistence.list_blobs().unwrap().len(), 1);

    // Request 2: the SAME conversation grown by 8 more tokens. Longest match is
    // the ancestor (8 tokens) → partial → delta prefill.
    let res = post(
        serde_json::json!({
            "model": "e2e-model",
            "messages": [{"role": "user", "content": "grow me more"}],
            "stream": false,
        })
        .to_string(),
    )
    .await;
    assert_eq!(res.status(), 200);
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        v["mlxcache"]["verdict"], "partial",
        "grown prompt must classify as a partial against the ancestor: {v}"
    );
    assert_eq!(
        v["mlxcache"]["prefill_from"], 7,
        "the leader reused the ancestor's 7 tokens of covered KV (matched 8 - 1): {v}"
    );
    assert_eq!(
        v["mlxcache"]["tokens_cached"], 7,
        "tokens_cached must equal prefill_from (the single covered-KV definition): {v}"
    );
    // Both the ancestor (8-token) and the new (16-token) checkpoint exist.
    assert_eq!(state.persistence.list_blobs().unwrap().len(), 2);

    // The sidecar's phase accounting proves the delta path ran: the second
    // prefill covered a 16-token prompt but only processed the 8-step delta
    // (16 - 7 covered - 1 final token), not the 15 steps of a scratch prefill.
    let stats: serde_json::Value = reqwest::get(format!("{sidecar_url}/stats"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(stats["prefill_count"].as_u64(), Some(2));
    assert_eq!(stats["last_prefill_tokens"].as_u64(), Some(16));
    assert_eq!(
        stats["last_prefill_delta_tokens"].as_u64(),
        Some(8),
        "delta prefill must process only the uncovered delta (8 steps, not 15): {stats}"
    );

    // Request 3: the grown prompt again — a full hit on the new checkpoint.
    let res = post(
        serde_json::json!({
            "model": "e2e-model",
            "messages": [{"role": "user", "content": "grow me more"}],
            "stream": false,
        })
        .to_string(),
    )
    .await;
    assert_eq!(res.status(), 200);
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["mlxcache"]["verdict"], "hit");
    assert_eq!(
        v["mlxcache"]["prefill_from"], 15,
        "hit on the 16-token checkpoint covers 15 tokens: {v}"
    );
    assert_eq!(
        state.stats.snapshot().partials,
        1,
        "exactly one partial (the delta-prefill leader) was recorded"
    );

    child.kill().expect("kill sidecar");
    child.wait().expect("reap sidecar");
}

#[tokio::test]
async fn transient_prefill_failure_keeps_ancestor_and_recovers() {
    // Fault path with an ADOPTED ancestor (api-contract 2026-10-04): the
    // delta prefill over the ancestor fails with a transient sidecar 500
    // (MLXCACHE_PREFILL_FAIL_AT=2 fails exactly the 2nd /prefill call).
    // Contract: a 500 is NOT a checkpoint rejection — no quarantine, the
    // ancestor stays published — the client gets 502 adapter_error, and the
    // next identical request retries the delta prefill and succeeds.
    let (sidecar_url, mut child) = spawn_sidecar_with_env(&[
        ("MLXCACHE_TOKENIZE_GROW", "grow me"),
        ("MLXCACHE_PREFILL_FAIL_AT", "2"),
    ])
    .await;
    let blobs = tempfile::tempdir().unwrap();
    let state = app_state(&sidecar_url, &blobs);

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

    // Request 1: full miss — prefill call #1 succeeds, publishes the ancestor.
    let res = post(
        serde_json::json!({
            "model": "e2e-model",
            "messages": [{"role": "user", "content": "grow me"}],
            "stream": false,
        })
        .to_string(),
    )
    .await;
    assert_eq!(res.status(), 200);
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["mlxcache"]["verdict"], "miss");
    assert_eq!(state.persistence.list_blobs().unwrap().len(), 1);

    // Request 2: grown prompt → partial → delta prefill is call #2 → 500.
    // The 502 envelope must carry the adapter_error kind (a genuine adapter
    // failure, not a transport-level 503), and the ancestor must survive.
    let res = post(
        serde_json::json!({
            "model": "e2e-model",
            "messages": [{"role": "user", "content": "grow me more"}],
            "stream": false,
        })
        .to_string(),
    )
    .await;
    assert_eq!(
        res.status(),
        502,
        "a transient sidecar 500 maps to 502 adapter_error"
    );
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        v["error"]["type"], "adapter_error",
        "500 is an adapter failure, not a checkpoint rejection: {v}"
    );
    assert!(
        v["error"]["message"]
            .as_str()
            .expect("message is a string")
            .contains("synthetic induced prefill failure"),
        "the induced failure must be surfaced verbatim: {v}"
    );
    assert_eq!(
        state.persistence.list_blobs().unwrap().len(),
        1,
        "no quarantine and no new publish: the ancestor stays exactly as-is"
    );

    // Request 3: identical to request 2 — prefill call #3 succeeds now, so
    // the delta prefill publishes and the request is served from the ancestor.
    let res = post(
        serde_json::json!({
            "model": "e2e-model",
            "messages": [{"role": "user", "content": "grow me more"}],
            "stream": false,
        })
        .to_string(),
    )
    .await;
    assert_eq!(
        res.status(),
        200,
        "the failure was transient: the next request must recover"
    );
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["mlxcache"]["verdict"], "partial", "recovered: {v}");
    assert_eq!(
        state.persistence.list_blobs().unwrap().len(),
        2,
        "ancestor + the recovered delta checkpoint"
    );

    child.kill().expect("kill sidecar");
    child.wait().expect("reap sidecar");
}

#[tokio::test]
async fn newline_free_flood_is_cut_by_the_partial_line_cap() {
    // MAX_PARTIAL_LINE (1 MiB): the idle budget only governs byte ARRIVALS,
    // so a fast flood with no newline would otherwise grow the bridge buffer
    // without bound (api-contract/red-team 2026-10-04). The sidecar floods
    // 2 MiB of newline-free bytes; the stream must end with the explicit
    // "unbounded line" upstream error frame + [DONE], not unbounded memory.
    let (sidecar_url, mut child) =
        spawn_sidecar_with_env(&[("MLXCACHE_STREAM_FLOOD", "2097152")]).await;
    let blobs = tempfile::tempdir().unwrap();
    let state = app_state(&sidecar_url, &blobs);

    let res = router(state)
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::json!({
                        "model": "e2e-model",
                        "messages": [{"role": "user", "content": "flood me"}],
                        "stream": true,
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        res.status(),
        200,
        "SSE headers arrive before the flood is cut"
    );
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let text = String::from_utf8_lossy(&bytes);
    assert!(
        text.contains("unbounded line"),
        "the flood must trip the explicit upstream error, got: {text}"
    );
    assert!(
        text.contains("data: [DONE]"),
        "the stream must terminate cleanly after the error: {text}"
    );

    child.kill().expect("kill sidecar");
    child.wait().expect("reap sidecar");
}

#[tokio::test]
async fn multi_turn_end_divergent_serves_t22() {
    // T22: a growing MULTI-TURN conversation whose turn-2 token stream
    // diverges from turn-1's checkpoint key at its LAST token must reuse
    // turn-1's KV. The wire shape: the daemon JSON-serializes the messages,
    // so turn r's serialization replaces turn r-1's closing bracket. Under
    // the diverge knob the turn streams share the first len(key)-1 tokens
    // and differ at the key's final (hinge) token — the exact measured
    // shape (Qwen2.5: turn-2 LCP 7454 vs turn-1 key 7455). Pre-T22 this
    // classified as a full miss and re-prefilled everything; the end-
    // anchored serve rule serves turn-1's checkpoint instead.
    let (sidecar_url, mut child) = spawn_sidecar_with_env(&[
        ("MLXCACHE_TOKENIZE_GROW", "turn one"),
        ("MLXCACHE_TOKENIZE_DIVERGE", "1"),
    ])
    .await;
    let blobs = tempfile::tempdir().unwrap();
    let state = app_state(&sidecar_url, &blobs);

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

    // Turn 1: a full miss, publishes the 9-token checkpoint (8 grow + 1
    // hinge token). Covered KV = 8 tokens.
    let res = post(
        serde_json::json!({
            "model": "e2e-model",
            "messages": [{"role": "user", "content": "turn one"}],
            "stream": false,
        })
        .to_string(),
    )
    .await;
    assert_eq!(res.status(), 200);
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["mlxcache"]["verdict"], "miss", "{v}");

    // Turn 2: the conversation grows (the serialization changes, the hinge
    // token differs), so the stream diverges from turn-1's key exactly at
    // its last token. The end-anchored rule serves turn-1's checkpoint as a
    // HIT: matched = 9 (the full request), covered KV = 8 (matched - 1),
    // and the adapter's feed derives from the blob meta (tokens[len-1:])
    // so it consumes the one uncovered token and the delta — identical to
    // a scratch run by construction.
    let turn2 = serde_json::json!({
        "model": "e2e-model",
        "messages": [
            {"role": "user", "content": "turn one"},
            {"role": "assistant", "content": "ans"},
            {"role": "user", "content": "turn two"},
        ],
        "stream": false,
    })
    .to_string();
    let res = post(turn2.clone()).await;
    assert_eq!(res.status(), 200);
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        v["mlxcache"]["verdict"], "hit",
        "turn 2 must reuse turn 1's checkpoint through the end-anchored rule: {v}"
    );
    assert_eq!(
        v["mlxcache"]["prefill_from"], 8,
        "turn-1's 8 tokens of covered KV reused (matched 9 - 1): {v}"
    );
    // Turn 2 did NOT republish: it served from turn-1's blob (no new
    // publication on the hit path).
    assert_eq!(
        state.persistence.list_blobs().unwrap().len(),
        1,
        "a hit must not publish a second checkpoint"
    );

    // Replay turn 2: the same hit, byte-identical routing.
    let res = post(turn2).await;
    assert_eq!(res.status(), 200);
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["mlxcache"]["verdict"], "hit", "{v}");
    assert_eq!(v["mlxcache"]["prefill_from"], 8, "{v}");

    // The sidecar saw exactly one prefill: turn 1. Turn 2 (and its replay)
    // resumed from the checkpoint — no prefill calls at all on the divergent
    // hit path (the generate path's uncovered tail is fed by the loader, not
    // a prefill).
    let stats: serde_json::Value = reqwest::get(format!("{sidecar_url}/stats"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        stats["prefill_count"].as_u64(),
        Some(1),
        "turn 2 must not prefill: its KV came from turn-1's checkpoint: {stats}"
    );

    child.kill().expect("kill sidecar");
    child.wait().expect("reap sidecar");
}

#[cfg(unix)]
#[tokio::test]
async fn corrupt_ancestor_quarantines_and_prefills_from_scratch() {
    // A partial hit whose ANCESTOR blob is corrupt must not wedge the request:
    // the adapter 422s the delta prefill, the daemon quarantines the ancestor
    // and retries the prefill from full scratch → 200 with a miss verdict.
    let (sidecar_url, mut child) =
        spawn_sidecar_with_env(&[("MLXCACHE_TOKENIZE_GROW", "grow me")]).await;
    let blobs = tempfile::tempdir().unwrap();
    let state = app_state(&sidecar_url, &blobs);
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

    // Base request publishes the ancestor.
    let res = post(
        serde_json::json!({
            "model": "e2e-model",
            "messages": [{"role": "user", "content": "grow me"}],
            "stream": false,
        })
        .to_string(),
    )
    .await;
    assert_eq!(res.status(), 200);

    // Corrupt the ancestor in place: a truncation the wire decoder rejects.
    let ancestor = state.persistence.list_blobs().unwrap().pop().unwrap();
    std::fs::write(&ancestor, b"JUNK").unwrap();

    // A different extension of the same base: its longest match IS the
    // (now corrupt) ancestor → delta prefill 422s → quarantine + scratch.
    let res = post(
        serde_json::json!({
            "model": "e2e-model",
            "messages": [{"role": "user", "content": "grow me differently"}],
            "stream": false,
        })
        .to_string(),
    )
    .await;
    assert_eq!(
        res.status(),
        200,
        "a corrupt ancestor must never fail the request"
    );
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        v["mlxcache"]["verdict"], "miss",
        "after the ancestor was rejected the request ran from scratch: {v}"
    );
    assert_eq!(
        state.orchestrator.quarantined_count(),
        1,
        "the corrupt ancestor must be quarantined"
    );

    // Three prefills ran across the whole test: the base publish (1), the
    // rejected delta attempt (2), and the scratch retry (3). The retry's
    // accounting shows a FULL 15-step prefill (16 tokens, nothing covered).
    let stats: serde_json::Value = reqwest::get(format!("{sidecar_url}/stats"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(stats["prefill_count"].as_u64(), Some(3), "{stats}");
    assert_eq!(stats["last_prefill_tokens"].as_u64(), Some(16), "{stats}");
    assert_eq!(
        stats["last_prefill_delta_tokens"].as_u64(),
        Some(15),
        "the scratch retry must have prefilled the whole prompt: {stats}"
    );

    child.kill().expect("kill sidecar");
    child.wait().expect("reap sidecar");
}

/// Wait for the async trace writer to land `want` records (bounded channel +
/// writer thread; a bare read races the last record's flush).
async fn wait_for_trace_lines(path: &std::path::Path, want: usize) -> String {
    for _ in 0..100 {
        if let Ok(text) = std::fs::read_to_string(path) {
            if text.lines().count() >= want {
                return text;
            }
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    std::fs::read_to_string(path).unwrap_or_default()
}

#[tokio::test]
async fn failed_request_gets_one_trace_record_and_stats_agree() {
    // Testing review 2026-10-04: a request that fails PAST routing used to
    // emit NO trace record while stats.record had already counted its
    // verdict — capture and /stats disagreed on exactly the failed requests
    // (the RT#7 invariant, reopened). Contract now: every routed request
    // gets exactly one record; a blob that was leaned on but never served
    // corrects its claim (correct_retire) and traces as the miss it was.
    let (sidecar_url, mut child) = spawn_sidecar_with_env(&[("MLXCACHE_GENERATE_FAIL", "1")]).await;
    let blobs = tempfile::tempdir().unwrap();
    let tracedir = tempfile::tempdir().unwrap();
    let trace_path = tracedir.path().join("trace.jsonl");
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
        trace: Some(mlxcache_daemon::trace::TraceWriter::from_path(&trace_path).unwrap()),
    });
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

    // Request 1 (miss whose generate fails): one record, a miss — stats and
    // trace agree with nothing to correct.
    let res = post(
        serde_json::json!({
            "model": "e2e-model",
            "messages": [{"role": "user", "content": "fail me once"}],
        })
        .to_string(),
    )
    .await;
    assert_eq!(
        res.status(),
        502,
        "the knob 500s generate; no blob leaned on"
    );

    // Request 2, identical: routes as a HIT (the checkpoint published by
    // request 1's prefill), generate 500s, the scratch retry 500s → 502.
    // The hit claim must be corrected and the record must say miss.
    let res = post(
        serde_json::json!({
            "model": "e2e-model",
            "messages": [{"role": "user", "content": "fail me once"}],
        })
        .to_string(),
    )
    .await;
    assert_eq!(res.status(), 502);

    let trace = wait_for_trace_lines(&trace_path, 2).await;
    let lines: Vec<&str> = trace.lines().collect();
    assert_eq!(
        lines.len(),
        2,
        "one record per request, failed or not: {trace}"
    );
    let rec1: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
    let rec2: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
    assert_eq!(rec1["verdict"], "miss", "{rec1}");
    assert_eq!(
        rec2["verdict"], "miss",
        "the failed hit must trace as the scratch run it became: {rec2}"
    );
    assert_eq!(rec2["prefill_from"].as_u64(), Some(0), "{rec2}");
    // And /stats agrees: the hit claim was corrected away.
    let s = state.stats.snapshot();
    assert_eq!(s.requests, 2, "{s:?}");
    assert_eq!(s.hits, 0, "the failed hit's claim must be corrected: {s:?}");

    child.kill().expect("kill sidecar");
    child.wait().expect("reap sidecar");
}

#[tokio::test]
async fn trace_records_the_settled_verdict_not_the_stale_route() {
    // RT#7 (red-team 2026-10-04): the trace record used to be emitted right
    // after routing — before the generate leg could still flip the outcome.
    // A request whose blob the adapter 422s quarantines it and retries from
    // scratch: /stats corrects the claim via correct_retire, but the trace
    // kept the stale hit. A capture and its /stats line must agree by
    // construction, so the record may only be emitted once the outcome has
    // SETTLED. (Would fail pre-fix with line 2 reading "hit".)
    let (sidecar_url, mut child) =
        spawn_sidecar_with_env(&[("MLXCACHE_TOKENIZE_GROW", "trace settle")]).await;
    let blobs = tempfile::tempdir().unwrap();
    let tracedir = tempfile::tempdir().unwrap();
    let trace_path = tracedir.path().join("trace.jsonl");
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
        trace: Some(mlxcache_daemon::trace::TraceWriter::from_path(&trace_path).unwrap()),
    });
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
    let body = serde_json::json!({
        "model": "e2e-model",
        "messages": [{"role": "user", "content": "trace settle"}],
        "stream": false,
    })
    .to_string();
    let body_grown = serde_json::json!({
        "model": "e2e-model",
        "messages": [{"role": "user", "content": "trace settle and grow the prompt"}],
        "stream": false,
    })
    .to_string();

    // Request 1: publishes the checkpoint (a genuine miss).
    let res = post(body).await;
    assert_eq!(res.status(), 200);

    // Corrupt the published blob in place (same shape as the quarantine
    // tests). The synthetic engine's /generate is token arithmetic and never
    // re-reads blob bytes, so the rejection must be driven through the
    // delta-PREFILL leg — the grown request's matched ancestor IS the corrupt
    // blob: the adapter 422s the prefill, the daemon quarantines and retries
    // from scratch → 200 with a miss body.
    let blob = state.persistence.list_blobs().unwrap().pop().unwrap();
    std::fs::write(&blob, b"JUNK").unwrap();

    // Request 2, grown: routes as a PARTIAL against the corrupt ancestor,
    // settles as a miss after quarantine + scratch retry.
    let res = post(body_grown).await;
    assert_eq!(res.status(), 200);
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["mlxcache"]["verdict"], "miss", "{v}");

    let trace = wait_for_trace_lines(&trace_path, 2).await;
    let lines: Vec<&str> = trace.lines().collect();
    assert_eq!(lines.len(), 2, "one record per request: {trace}");
    let rec2: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
    assert_eq!(
        rec2["verdict"], "miss",
        "the SETTLED verdict (scratch retry after quarantine), not the stale routing partial: {rec2}"
    );
    assert_eq!(rec2["prefill_from"].as_u64(), Some(0), "{rec2}");
    assert_eq!(rec2["matched_tokens"].as_u64(), Some(0), "{rec2}");

    child.kill().expect("kill sidecar");
    child.wait().expect("reap sidecar");
}

#[tokio::test]
async fn stalled_stream_is_cut_by_the_idle_budget_with_an_explicit_error() {
    // Streams carry NO total timeout (a decode's wall time is unbounded by
    // design). Instead the daemon enforces an idle budget between bytes: a
    // sidecar that stalls before the first token is cut with an explicit
    // upstream_error frame + [DONE], never a silent hang.
    let (sidecar_url, mut child) =
        spawn_sidecar_with_env(&[("MLXCACHE_FIRST_TOKEN_DELAY", "5")]).await;
    let blobs = tempfile::tempdir().unwrap();
    // stream_idle = 1s, far below the sidecar's 5s first-token stall.
    let sidecar_client =
        SidecarClient::with_timeouts(SidecarConfig::new(sidecar_url, "e2e-model".into()), 120, 1)
            .unwrap();
    let state = Arc::new(AppState {
        orchestrator: Orchestrator::new(),
        singleflight: SingleFlight::new(),
        stats: Arc::new(mlxcache_daemon::http::Stats::default()),
        served_models: vec!["e2e-model".into()],
        sidecar: Some(sidecar_client),
        persistence: mlxcache_daemon::persistence::Persistence::new(blobs.path()).unwrap(),
        trace: None,
    });
    let started = std::time::Instant::now();
    let res = router(state.clone())
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::json!({
                        "model": "e2e-model",
                        "messages": [{"role": "user", "content": "stall please"}],
                        "stream": true,
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        res.status(),
        200,
        "SSE headers arrive before the first token"
    );
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let elapsed = started.elapsed();
    let text = String::from_utf8_lossy(&bytes);
    assert!(
        elapsed < std::time::Duration::from_secs(4),
        "the idle budget (1s) must cut the stalled stream promptly, took {elapsed:?}"
    );
    assert!(
        text.contains("upstream idle timeout"),
        "an explicit upstream error frame must be emitted: {text}"
    );
    assert!(
        text.contains("data: [DONE]"),
        "the stream must still be terminated with [DONE]: {text}"
    );

    child.kill().expect("kill sidecar");
    child.wait().expect("reap sidecar");
}
