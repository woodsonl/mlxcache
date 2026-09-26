//! HTTP layer: OpenAI-compatible proxy endpoints + /stats.
//!
//! Request pipeline (design doc): validate → 404 unknown model BEFORE any
//! cache lookup (Section 3) → tokenize via sidecar (R1-2) → route/classify →
//! single-flight → respond. v0: sidecar is the only adapter; owner engine
//! replaces it as the native adapter (R4).

use crate::observability::log_request;
use crate::orchestrator::Orchestrator;
use crate::sidecar::{SidecarClient, SidecarError};
use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use mlxcache_core::contract::ModelFingerprint;
use mlxcache_core::policy::{CacheVerdict, PolicyDecision};
use mlxcache_core::singleflight::SingleFlight;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(default)]
    pub stream: bool,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Message {
    pub role: String,
    pub content: serde_json::Value,
}

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: ErrorBody,
}

#[derive(Debug, Serialize)]
pub struct ErrorBody {
    pub message: String,
    #[serde(rename = "type")]
    pub kind: String,
}

pub struct AppState {
    pub orchestrator: Orchestrator,
    pub singleflight: SingleFlight,
    pub stats: Arc<Stats>,
    /// Models this daemon serves. Requests for anything else 404 before lookup.
    pub served_models: Vec<String>,
    pub sidecar: Option<SidecarClient>,
    pub persistence: crate::persistence::Persistence,
}

/// Hit-rate counters (D3). Mutex over a small struct is fine at v1 scale.
#[derive(Debug, Default)]
pub struct Stats {
    inner: std::sync::Mutex<StatsInner>,
}

#[derive(Debug, Default)]
pub struct StatsInner {
    pub requests: u64,
    pub hits: u64,
    pub misses: u64,
    pub partials: u64,
    pub tokens_cached: u64,
    pub tokens_total: u64,
}

impl Stats {
    pub fn record(&self, d: &PolicyDecision) {
        let mut s = self.inner.lock().unwrap();
        s.requests += 1;
        match d.verdict {
            CacheVerdict::Hit => s.hits += 1,
            CacheVerdict::Miss => s.misses += 1,
            CacheVerdict::Partial => s.partials += 1,
        }
        s.tokens_cached += d.matched_tokens as u64;
        s.tokens_total += d.request_tokens as u64;
    }
}

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/stats", get(stats))
        .with_state(state)
}

fn err(status: StatusCode, message: &str, kind: &str) -> (StatusCode, Json<ErrorResponse>) {
    (
        status,
        Json(ErrorResponse {
            error: ErrorBody {
                message: message.into(),
                kind: kind.into(),
            },
        }),
    )
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatRequest>,
) -> Result<axum::response::Response, (StatusCode, Json<ErrorResponse>)> {
    // Validation order per error registry: malformed → 400; unknown model →
    // 404 BEFORE any cache lookup (Section 3 finding).
    if req.messages.is_empty() {
        return Err(err(
            StatusCode::BAD_REQUEST,
            "messages must not be empty",
            "invalid_request_error",
        ));
    }
    if !state.served_models.contains(&req.model) {
        return Err(err(
            StatusCode::NOT_FOUND,
            format!("model '{}' not served by this daemon", req.model).as_str(),
            "invalid_request_error",
        ));
    }

    let started = std::time::Instant::now();
    let prompt = serde_json::to_string(&req.messages)
        .map_err(|_| err(StatusCode::BAD_REQUEST, "unserializable messages", "invalid_request_error"))?;

    // Tokenize via adapter (R1-2). Sidecar unavailable → 503 naming it.
    let tokens = match &state.sidecar {
        Some(client) => match client.tokenize(&prompt).await {
            Ok(r) => r.tokens,
            Err(SidecarError::Unreachable { url, .. }) => {
                return Err(err(
                    StatusCode::SERVICE_UNAVAILABLE,
                    format!("adapter (sidecar) unreachable at {url}").as_str(),
                    "adapter_unavailable",
                ));
            }
            Err(e) => {
                return Err(err(
                    StatusCode::BAD_GATEWAY,
                    &e.to_string(),
                    "adapter_error",
                ));
            }
        },
        None => {
            return Err(err(
                StatusCode::SERVICE_UNAVAILABLE,
                "no adapter configured",
                "adapter_unavailable",
            ));
        }
    };

    // Route + classify (R1-1 fingerprint check inside classify).
    let fingerprint = ModelFingerprint {
        model_id: req.model.clone(),
        tokenizer_hash: "sidecar".into(), // real hash arrives with the sidecar's tokenize response
        kv_dtype: "f16".into(),
        kv_layout_version: 1,
    };
    let outcome = state.orchestrator.route(&tokens, &fingerprint);
    state.stats.record(&outcome.decision);
    let ttft_ms = started.elapsed().as_millis() as u64;
    log_request(&req.model, 0, &outcome.decision, ttft_ms);

    // Prefill on miss/partial (single-flight, R1-3), then persist. On hit the
    // published blob path is passed to the adapter. Shared by both modes.
    let client = state.sidecar.as_ref().expect("checked above");
    if outcome.decision.verdict != CacheVerdict::Hit {
        let (guard, follower) = state.singleflight.try_lead(tokens.clone()).await;
        match follower {
            None => {
                match client.prefill(&tokens).await {
                    Ok(blob) => {
                        let hash = prefix_hash(&tokens);
                        let meta = mlxcache_core::contract::CheckpointMeta {
                            fingerprint: fingerprint.clone(),
                            token_count: tokens.len() as u64,
                            format_version: 1,
                        };
                        if let Err(e) = state.persistence.publish_atomic(hash, &meta, &blob) {
                            // ENOSPC rescue (registry): log and continue uncached.
                            tracing::warn!(error = %e, "checkpoint write failed; continuing uncached");
                        } else {
                            state
                                .orchestrator
                                .publish_checkpoint(&tokens, meta, format!("{:016x}.ckpt", hash));
                        }
                        drop(guard);
                    }
                    Err(e) => {
                        drop(guard);
                        return Err(err(StatusCode::BAD_GATEWAY, &e.to_string(), "adapter_error"));
                    }
                }
            }
            Some(mut rx) => {
                // Follower: leader's prefill covers us.
                let _ = rx.recv().await;
            }
        }
    }

    // The adapter needs an absolute path to open the blob directly.
    let blob_abs = match &outcome.blob_path {
        Some(name) => state
            .persistence
            .blob_dir
            .join(name)
            .to_string_lossy()
            .into_owned(),
        None => String::new(),
    };
    let blob_arg = if outcome.blob_path.is_some() {
        Some(blob_abs.as_str())
    } else {
        None
    };

    if req.stream {
        return stream_response(state.clone(), client, tokens, outcome, blob_arg, ttft_ms).await;
    }

    // Generate: full context tokens, continuation from the request length.
    let generated = match client
        .generate(&tokens, outcome.prefill_from, 64, outcome.blob_path.as_deref())
        .await
    {
        Ok(t) => t,
        Err(e) => return Err(err(StatusCode::BAD_GATEWAY, &e.to_string(), "adapter_error")),
    };

    let body = serde_json::json!({
        "mlxcache": {
            "verdict": match outcome.decision.verdict {
                CacheVerdict::Hit => "hit",
                CacheVerdict::Partial => "partial",
                CacheVerdict::Miss => "miss",
            },
            "tokens_cached": outcome.decision.matched_tokens,
            "tokens_total": outcome.decision.request_tokens,
            "prefill_from": outcome.prefill_from,
            "ttft_ms": ttft_ms,
        },
        "generated_tokens": generated,
        "status": "ok",
    });
    Ok((StatusCode::OK, Json(body)).into_response())
}

/// FNV-1a over token ids — stable blob key for a token prefix.
fn prefix_hash(tokens: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for t in tokens {
        h ^= *t as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Open the sidecar's NDJSON generation stream and re-emit it as SSE.
/// Each sidecar line `{"token":..,"text":..}` becomes an OpenAI-style
/// `data: {...}` chunk; the terminal `{"done":true}` closes with `[DONE]`.
async fn stream_response(
    _state: Arc<AppState>,
    client: &SidecarClient,
    tokens: Vec<u32>,
    outcome: crate::orchestrator::RouteOutcome,
    blob_arg: Option<&str>,
    ttft_ms: u64,
) -> Result<axum::response::Response, (StatusCode, Json<ErrorResponse>)> {
    use futures_util::StreamExt;

    let verdict = match outcome.decision.verdict {
        CacheVerdict::Hit => "hit",
        CacheVerdict::Partial => "partial",
        CacheVerdict::Miss => "miss",
    };
    let meta_line = serde_json::json!({
        "mlxcache": {
            "verdict": verdict,
            "tokens_cached": outcome.decision.matched_tokens,
            "tokens_total": outcome.decision.request_tokens,
            "prefill_from": outcome.prefill_from,
            "ttft_ms": ttft_ms,
        }
    });

    let upstream = client
        .generate_stream(&tokens, outcome.prefill_from, 64, blob_arg)
        .await
        .map_err(|e| err(StatusCode::BAD_GATEWAY, &e.to_string(), "adapter_error"))?;

    // Split the upstream byte stream on newlines, then map each NDJSON line to
    // an SSE frame. A tiny state machine keeps partial lines across chunks.
    let meta_bytes = format!("data: {meta_line}\n\n").into_bytes();
    let first =
        futures_util::stream::once(async move { Ok::<Vec<u8>, std::io::Error>(meta_bytes) });

    let mut buf: Vec<u8> = Vec::new();
    let body_stream = upstream
        .bytes_stream()
        .flat_map(move |chunk| {
            let mut frames: Vec<Result<Vec<u8>, std::io::Error>> = Vec::new();
            match chunk {
                Ok(bytes) => {
                    buf.extend_from_slice(&bytes);
                    while let Some(pos) = buf.iter().position(|b| *b == b'\n') {
                        let line: Vec<u8> = buf.drain(..=pos).collect();
                        let line = String::from_utf8_lossy(&line);
                        let line = line.trim();
                        if line.is_empty() {
                            continue;
                        }
                        if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
                            if v.get("done").is_some() {
                                frames.push(Ok(b"data: [DONE]\n\n".to_vec()));
                            } else {
                                frames.push(Ok(format!("data: {v}\n\n").into_bytes()));
                            }
                        }
                    }
                }
                Err(e) => frames.push(Err(std::io::Error::other(e.to_string()))),
            }
            futures_util::stream::iter(frames)
        });

    let stream = first.chain(body_stream);
    let body = axum::body::Body::from_stream(stream);
    Ok(axum::response::Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .body(body)
        .expect("valid SSE response"))
}

async fn stats(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let s = state.stats.inner.lock().unwrap();
    Json(serde_json::json!({
        "requests": s.requests,
        "hits": s.hits,
        "misses": s.misses,
        "partials": s.partials,
        "tokens_cached": s.tokens_cached,
        "tokens_total": s.tokens_total,
        "hit_rate": if s.tokens_total > 0 {
            s.tokens_cached as f64 / s.tokens_total as f64
        } else {
            0.0
        },
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use http_body_util::BodyExt;
    use tower::util::ServiceExt;

    fn app() -> Router {
        let state = Arc::new(AppState {
            orchestrator: Orchestrator::new(),
            singleflight: SingleFlight::new(),
            stats: Arc::new(Stats::default()),
            served_models: vec!["test-model".into()],
            sidecar: None,
            persistence: crate::persistence::Persistence::new(
                tempfile::tempdir().unwrap().keep(),
            )
            .unwrap(),
        });
        router(state)
    }

    async fn post_json(router: Router, body: &str) -> axum::response::Response {
        router
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn empty_messages_rejected() {
        let res = post_json(app(), r#"{"model":"test-model","messages":[],"stream":false}"#).await;
        assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn unknown_model_404_before_lookup() {
        let res = post_json(
            app(),
            r#"{"model":"other-model","messages":[{"role":"user","content":"hi"}],"stream":false}"#,
        )
        .await;
        assert_eq!(res.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn no_adapter_is_503() {
        let res = post_json(
            app(),
            r#"{"model":"test-model","messages":[{"role":"user","content":"hi"}],"stream":false}"#,
        )
        .await;
        assert_eq!(res.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn stats_endpoint_counts() {
        let router = app();
        let _ = post_json(
            router.clone(),
            r#"{"model":"other-model","messages":[{"role":"user","content":"hi"}],"stream":false}"#,
        )
        .await; // 404 — must NOT count as a request (rejected before lookup)
        let res = router
            .oneshot(
                axum::http::Request::builder()
                    .uri("/stats")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let body = res.into_body().collect().await.unwrap().to_bytes();
        let stats: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(stats["requests"], 0, "404s must not enter the cache stats");
    }
}
