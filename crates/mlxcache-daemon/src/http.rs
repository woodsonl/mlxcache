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
    /// Recover a poisoned lock instead of panicking: counters are diagnostic,
    /// never worth failing a request over (error-registry principle).
    fn lock(&self) -> std::sync::MutexGuard<'_, StatsInner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn record(&self, d: &PolicyDecision) {
        let mut s = self.lock();
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
    let prompt = serde_json::to_string(&req.messages).map_err(|_| {
        err(
            StatusCode::BAD_REQUEST,
            "unserializable messages",
            "invalid_request_error",
        )
    })?;

    // Tokenize via adapter (R1-2). Sidecar unavailable → 503 naming it.
    let (tokens, tokenizer_hash, kv_dtype) = match &state.sidecar {
        Some(client) => match client.tokenize(&prompt).await {
            Ok(r) => (r.tokens, r.tokenizer_hash, r.kv_dtype),
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

    // A tokenizer can return zero tokens (e.g. an empty string for HF
    // tokenizers). An empty prefix has no KV to cache and its hash is a shared
    // constant, so every empty request would alias one blob. Reject it here.
    if tokens.is_empty() {
        return Err(err(
            StatusCode::BAD_REQUEST,
            "prompt tokenized to zero tokens",
            "invalid_request_error",
        ));
    }

    // Route + classify (R1-1 fingerprint check inside classify). The tokenizer
    // hash comes from the adapter (R1-2): a constant here would let checkpoints
    // from different tokenizers share a fingerprint and be served wrongly.
    let fingerprint = ModelFingerprint {
        model_id: req.model.clone(),
        tokenizer_hash,
        kv_dtype,
        kv_layout_version: 1,
    };
    let mut outcome = state.orchestrator.route(&tokens, &fingerprint);
    let started_ms = started.elapsed().as_millis() as u64;
    // Compute the prefix key once: reused by the log line and (on miss) the
    // blob filename. Hashing a long prefix twice is wasted hot-path work.
    let hash = prefix_hash(&tokens);

    // Prefill on miss/partial. Single-flight (R1-3): one leader runs the
    // prefill and publishes; followers await its result and re-route so they
    // adopt the leader's checkpoint instead of re-prefilling. The leader keeps
    // its miss/partial verdict because it did run the prefill, but records the
    // blob it published so the adapter can resume from it.
    let Some(client) = state.sidecar.as_ref() else {
        return Err(err(
            StatusCode::SERVICE_UNAVAILABLE,
            "no adapter configured",
            "adapter_unavailable",
        ));
    };
    if outcome.decision.verdict != CacheVerdict::Hit {
        match state.singleflight.enter(tokens.clone()).await {
            mlxcache_core::singleflight::Role::Leader(lead) => {
                match client.prefill(&tokens).await {
                    Ok(blob) => {
                        let meta = mlxcache_core::contract::CheckpointMeta {
                            fingerprint: fingerprint.clone(),
                            token_count: tokens.len() as u64,
                            tokens: tokens.clone(),
                            format_version: 1,
                        };
                        let blob_name = format!("{:032x}.ckpt", blob_key(&fingerprint, &tokens));
                        match state.persistence.publish_atomic(
                            blob_key(&fingerprint, &tokens),
                            &meta,
                            &blob,
                        ) {
                            // ENOSPC rescue (registry): log and continue uncached.
                            Err(e) => {
                                tracing::warn!(error = %e, "checkpoint write failed; continuing uncached");
                                lead.complete(Err(e.to_string()));
                            }
                            Ok(_) => {
                                state.orchestrator.publish_checkpoint(
                                    &tokens,
                                    meta,
                                    blob_name.clone(),
                                );
                                // The leader's own request resumes from the blob
                                // it just wrote (avoids re-prefilling the delta).
                                outcome.blob_path = Some(blob_name.clone());
                                outcome.prefill_from = tokens.len();
                                lead.complete(Ok(blob_name));
                            }
                        }
                    }
                    Err(e) => {
                        let msg = e.to_string();
                        lead.complete(Err(msg.clone()));
                        return Err(err(StatusCode::BAD_GATEWAY, &msg, "adapter_error"));
                    }
                }
            }
            mlxcache_core::singleflight::Role::Follower(rx) => {
                // Await the leader's outcome; None means the leader died before
                // publishing, so fall through to our own (uncached) path.
                match mlxcache_core::singleflight::await_result(rx).await {
                    Some(Err(msg)) => {
                        // Leader failed too; surface the same adapter error.
                        return Err(err(StatusCode::BAD_GATEWAY, &msg, "adapter_error"));
                    }
                    Some(Ok(_)) => {
                        // Re-route: the leader's publish is now visible, so this
                        // follower adopts it (hit/partial) instead of prefilling.
                        outcome = state.orchestrator.route(&tokens, &fingerprint);
                    }
                    None => {}
                }
            }
        }
    }

    // Record + log the verdict the request saw: the leader a miss/partial (it
    // prefilled), a coalesced follower the hit/partial it adopted. Recording
    // after single-flight keeps stats truthful.
    state.stats.record(&outcome.decision);
    log_request(&req.model, hash, &outcome.decision, started_ms);
    let ttft_ms = started_ms;

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
    // The adapter opens the blob directly, so it needs the absolute path.
    let generated = match client
        .generate(&tokens, outcome.prefill_from, 64, blob_arg)
        .await
    {
        Ok(t) => t,
        Err(e) => {
            return Err(err(
                StatusCode::BAD_GATEWAY,
                &e.to_string(),
                "adapter_error",
            ))
        }
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

/// Stable blob key for a token prefix: 128 bits from two FNV-1a passes with
/// different seeds and different primes, so a collision in one lane does not
/// correlate with the other. 64 bits is too thin as the store grows; a filename
/// collision would alias two distinct KV states. (The index is exact and keyed
/// by token ids; this only names the blob on disk.)
fn prefix_hash(tokens: &[u32]) -> u128 {
    fn fnv1a(seed: u64, prime: u64, tokens: &[u32]) -> u64 {
        let mut h = seed;
        for t in tokens {
            h ^= *t as u64;
            h = h.wrapping_mul(prime);
        }
        h
    }
    // Lane 1: FNV-1a 64 (basis 0xcbf29ce484222325, prime 0x100000001b3).
    // Lane 2: FNV-1 (no final fold) with the golden-ratio seed and the FNV-1
    // alternate prime 0x880355f21e6d1965, so the two lanes use different mixers.
    let lo = fnv1a(0xcbf29ce484222325, 0x00000100000001b3, tokens);
    let hi = fnv1a(0x9e3779b97f4a7c15, 0x880355f21e6d1965, tokens);
    ((hi as u128) << 64) | lo as u128
}

/// On-disk blob key: `prefix_hash(tokens)` folded with the model fingerprint.
///
/// The index key is the token prefix and the fingerprint is checked on every
/// lookup, but the filename must ALSO be fingerprint-specific. Two models can
/// share token ids; without the fingerprint in the name, the second model's
/// prefill would overwrite the first model's blob in place, and the first
/// model's still-valid index entry would load foreign KV on its next hit:
/// silent wrong output. Prepending the fingerprint keeps the two files apart.
fn blob_key(fingerprint: &mlxcache_core::contract::ModelFingerprint, tokens: &[u32]) -> u128 {
    let mut f = fingerprint.model_id.clone();
    f.push('\u{1f}');
    f.push_str(&fingerprint.tokenizer_hash);
    f.push('\u{1f}');
    f.push_str(&fingerprint.kv_dtype);
    f.push('\u{1f}');
    f.push_str(&fingerprint.kv_layout_version.to_string());
    // Fold the fingerprint into the token hash: hash the fingerprint bytes into
    // both lanes first, then continue with the tokens.
    let fp_bytes: Vec<u32> = f
        .as_bytes()
        .chunks(4)
        .map(|c| {
            let mut b = [0u8; 4];
            b[..c.len()].copy_from_slice(c);
            u32::from_le_bytes(b)
        })
        .collect();
    let mut all = fp_bytes;
    all.push(0xFFFF_FFFF); // domain separator so fp||tokens can't alias tokens
    all.extend_from_slice(tokens);
    prefix_hash(&all)
}

/// Map one raw NDJSON line from the sidecar into zero or more SSE frames.
/// `{"done":true}` becomes `[DONE]`; a token line becomes `data: {...}`; a
/// non-JSON line is a protocol violation and surfaces as a stream error.
fn push_frame(frames: &mut Vec<Result<Vec<u8>, std::io::Error>>, raw: &[u8]) {
    let line = String::from_utf8_lossy(raw);
    let line = line.trim();
    if line.is_empty() {
        return;
    }
    match serde_json::from_str::<serde_json::Value>(line) {
        Ok(v) if v.get("done").is_some() => frames.push(Ok(b"data: [DONE]\n\n".to_vec())),
        Ok(v) => frames.push(Ok(format!("data: {v}\n\n").into_bytes())),
        Err(_) => frames.push(Err(std::io::Error::other(
            "sidecar emitted a non-JSON stream line",
        ))),
    }
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
    // an SSE frame. A tiny state machine keeps partial lines across chunks, and
    // a final flush handles a trailing line with no newline plus a guaranteed
    // [DONE] terminator (SSE clients wait for it; an early close must not leave
    // them hanging or silently truncate the completion).
    let meta_bytes = format!("data: {meta_line}\n\n").into_bytes();
    let first =
        futures_util::stream::once(async move { Ok::<Vec<u8>, std::io::Error>(meta_bytes) });

    struct StreamState {
        upstream: std::pin::Pin<
            Box<dyn futures_util::Stream<Item = reqwest::Result<bytes::Bytes>> + Send>,
        >,
        buf: Vec<u8>,
        done: bool,
        terminated: bool,
    }
    let state = StreamState {
        upstream: Box::pin(upstream.bytes_stream()),
        buf: Vec::new(),
        done: false,
        terminated: false,
    };
    let body_stream = futures_util::stream::unfold(state, |mut st| async move {
        use futures_util::StreamExt;
        let mut frames: Vec<Result<Vec<u8>, std::io::Error>> = Vec::new();
        loop {
            if st.done {
                return None;
            }
            match st.upstream.next().await {
                Some(Ok(bytes)) => {
                    st.buf.extend_from_slice(&bytes);
                    while let Some(pos) = st.buf.iter().position(|b| *b == b'\n') {
                        let line: Vec<u8> = st.buf.drain(..=pos).collect();
                        push_frame(&mut frames, &line);
                        if frames.last().is_some_and(|f| {
                            f.as_ref().is_ok_and(|b| b.starts_with(b"data: [DONE]"))
                        }) {
                            st.terminated = true;
                        }
                    }
                    if !frames.is_empty() {
                        break;
                    }
                }
                Some(Err(e)) => {
                    frames.push(Err(std::io::Error::other(e.to_string())));
                    break;
                }
                None => {
                    // Upstream ended: flush a trailing line with no newline, then
                    // terminate with [DONE] unless the sidecar already sent it.
                    if !st.buf.is_empty() {
                        let line = std::mem::take(&mut st.buf);
                        push_frame(&mut frames, &line);
                        if frames.last().is_some_and(|f| {
                            f.as_ref().is_ok_and(|b| b.starts_with(b"data: [DONE]"))
                        }) {
                            st.terminated = true;
                        }
                    }
                    if !st.terminated {
                        frames.push(Ok(b"data: [DONE]\n\n".to_vec()));
                    }
                    st.done = true;
                    break;
                }
            }
        }
        Some((futures_util::stream::iter(frames), st))
    })
    .flatten();

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
    let s = state.stats.lock();
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

    #[test]
    fn push_frame_maps_ndjson_to_sse() {
        let mut frames = Vec::new();
        push_frame(&mut frames, b"{\"token\":5,\"text\":\"hi\"}\n");
        assert_eq!(frames.len(), 1);
        let text = String::from_utf8(frames[0].as_ref().unwrap().clone()).unwrap();
        assert!(text.starts_with("data: {"));
        assert!(text.ends_with("\n\n"));

        frames.clear();
        push_frame(&mut frames, b"{\"done\":true}\n");
        assert_eq!(frames[0].as_ref().unwrap(), b"data: [DONE]\n\n");

        // Blank lines and whitespace are ignored, not turned into frames.
        frames.clear();
        push_frame(&mut frames, b"\n");
        push_frame(&mut frames, b"   \n");
        assert!(frames.is_empty());

        // A non-JSON line is a protocol violation.
        frames.clear();
        push_frame(&mut frames, b"not json\n");
        assert!(frames[0].is_err());
    }

    fn app() -> Router {
        let state = Arc::new(AppState {
            orchestrator: Orchestrator::new(),
            singleflight: SingleFlight::new(),
            stats: Arc::new(Stats::default()),
            served_models: vec!["test-model".into()],
            sidecar: None,
            persistence: crate::persistence::Persistence::new(tempfile::tempdir().unwrap().keep())
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
        let res = post_json(
            app(),
            r#"{"model":"test-model","messages":[],"stream":false}"#,
        )
        .await;
        assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn malformed_json_rejected() {
        // Registry row: malformed JSON -> 400 (axum's Json extractor rejects
        // before the handler runs; the body must still be a 4xx, not a 500).
        let res = post_json(app(), r#"{"model": broken"#).await;
        assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn missing_required_field_rejected() {
        // model present, messages absent -> deserialization error -> 4xx.
        let res = post_json(app(), r#"{"model":"test-model"}"#).await;
        assert!(res.status().is_client_error(), "got {}", res.status());
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

    #[test]
    fn prefix_hash_distinguishes_similar_prefixes() {
        // A collision aliases two KV states; verify the hash separates
        // prefixes that differ only at the tail or by one token.
        let a = vec![1, 2, 3, 4];
        let b = vec![1, 2, 3, 5];
        let c = vec![1, 2, 3];
        assert_ne!(prefix_hash(&a), prefix_hash(&b), "tail difference aliased");
        assert_ne!(
            prefix_hash(&a),
            prefix_hash(&c),
            "length difference aliased"
        );
        // Deterministic: same input, same key.
        assert_eq!(prefix_hash(&a), prefix_hash(&[1, 2, 3, 4]));
        // Must exceed 64 bits (the 128-bit widening).
        assert!(prefix_hash(&a) > u64::MAX as u128);
    }

    #[test]
    fn blob_key_is_fingerprint_specific() {
        // Same tokens, two models: the on-disk key MUST differ, or one model's
        // prefill overwrites the other's blob and a stale index entry serves
        // foreign KV (silent wrong output).
        let tokens = vec![10, 20, 30];
        let a = mlxcache_core::contract::ModelFingerprint {
            model_id: "model-a".into(),
            tokenizer_hash: "tok-a".into(),
            kv_dtype: "f16".into(),
            kv_layout_version: 1,
        };
        let b = mlxcache_core::contract::ModelFingerprint {
            model_id: "model-b".into(),
            tokenizer_hash: "tok-a".into(),
            kv_dtype: "f16".into(),
            kv_layout_version: 1,
        };
        assert_ne!(blob_key(&a, &tokens), blob_key(&b, &tokens));
        // Tokenizer-only difference must also separate.
        let c = mlxcache_core::contract::ModelFingerprint {
            model_id: "model-a".into(),
            tokenizer_hash: "tok-b".into(),
            kv_dtype: "f16".into(),
            kv_layout_version: 1,
        };
        assert_ne!(blob_key(&a, &tokens), blob_key(&c, &tokens));
        // Deterministic and distinct from the bare token hash.
        assert_eq!(blob_key(&a, &tokens), blob_key(&a, &[10, 20, 30]));
        assert_ne!(blob_key(&a, &tokens), prefix_hash(&tokens));
    }

    #[test]
    fn prefix_hash_halves_are_independent() {
        // Both 64-bit lanes must react to a one-token change, so a collision in
        // one lane does not imply a collision in the other.
        let x = prefix_hash(&[7]);
        let y = prefix_hash(&[8]);
        assert_ne!(x as u64, y as u64, "low lane ignores input");
        assert_ne!(
            (x >> 64) as u64,
            (y >> 64) as u64,
            "high lane ignores input"
        );

        // The lanes must not collapse to the same value across a broad sweep.
        let same = (0..4096u32)
            .filter(|i| {
                let h = prefix_hash(&[*i]);
                (h >> 64) as u64 == h as u64
            })
            .count();
        assert!(same <= 1, "hash lanes collapsed: {same} identical halves");
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
