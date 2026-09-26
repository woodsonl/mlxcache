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
use mlxcache_core::policy::{covered_kv_tokens, CacheVerdict, PolicyDecision};
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
        // tokens_cached counts KV actually in hand (covered = matched-1), the
        // same value the response reports as tokens_cached/prefill_from. Counting
        // matched_tokens here would overstate reuse by one per hit.
        s.tokens_cached += covered_kv_tokens(d.verdict, d.matched_tokens) as u64;
        s.tokens_total += d.request_tokens as u64;
    }

    /// Correct a decision whose checkpoint was retired (quarantined) after
    /// `record`: no KV was actually reused, so drop the covered count and move
    /// the verdict to a miss. Otherwise /stats and hit_rate report reuse for a
    /// request that ran from scratch. The per-response body is already corrected
    /// on the retire path; this keeps the aggregate consistent with it.
    ///
    /// No-op for a decision already recorded as Miss: a cold-miss leader keeps
    /// its miss verdict while adopting the blob it just published, and retiring
    /// that blob must not count a second miss for the one request.
    pub fn correct_retire(&self, d: &PolicyDecision) {
        if d.verdict == CacheVerdict::Miss {
            return;
        }
        let mut s = self.lock();
        match d.verdict {
            CacheVerdict::Hit => s.hits = s.hits.saturating_sub(1),
            CacheVerdict::Partial => s.partials = s.partials.saturating_sub(1),
            CacheVerdict::Miss => {}
        }
        s.misses += 1;
        s.tokens_cached = s
            .tokens_cached
            .saturating_sub(covered_kv_tokens(d.verdict, d.matched_tokens) as u64);
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
                    // An empty payload means the adapter cached nothing (e.g. a
                    // one-token prompt). Publishing would index a checkpoint with
                    // no KV. The leader's publish result is an `Ok(blob_name)`, and
                    // an empty name is the sentinel for "nothing published"; the
                    // follower branch below treats an empty name as a no-blob miss
                    // and runs from scratch too.
                    //
                    // A <2-token prefix caches nothing even if a non-conforming
                    // adapter returned bytes: the index refuses to key it, so
                    // persisting and adopting it would write a blob no lookup can
                    // serve and hand the adapter a checkpoint covering zero KV.
                    Ok(blob) if blob.is_empty() || tokens.len() < 2 => {
                        // Force a scratch decision so this request and every
                        // follower report the same verdict: nothing was cached,
                        // so no blob may be resumed from even if a shorter
                        // ancestor had been matched.
                        outcome.decision = mlxcache_core::policy::PolicyDecision {
                            verdict: mlxcache_core::policy::CacheVerdict::Miss,
                            matched_tokens: 0,
                            request_tokens: tokens.len(),
                        };
                        outcome.blob_path = None;
                        outcome.prefill_from = 0;
                        lead.complete(Ok(String::new()));
                    }
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
                                // The leader's generation resumes from the blob it
                                // just wrote (avoids re-prefilling the delta), but
                                // that is not cross-request cache reuse: the KV was
                                // built by THIS request, not served from an earlier
                                // one. Keep the routed covered count (0 on a cold
                                // miss, matched-1 on a partial) so the response,
                                // /stats, the log, and README's "0 on a miss" all
                                // agree. Only blob_path drives generation.
                                outcome.blob_path = Some(blob_name.clone());
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
            mlxcache_core::singleflight::Role::Follower(mut follower) => {
                // Await the leader's outcome; None means the leader died before
                // publishing, so fall through to our own (uncached) path.
                match mlxcache_core::singleflight::await_result(&mut follower).await {
                    Some(Err(msg)) => {
                        // Leader failed too; surface the same adapter error.
                        return Err(err(StatusCode::BAD_GATEWAY, &msg, "adapter_error"));
                    }
                    Some(Ok(name)) if !name.is_empty() => {
                        // Re-route: the leader's publish is now visible, so this
                        // follower adopts it (hit/partial) instead of prefilling.
                        outcome = state.orchestrator.route(&tokens, &fingerprint);
                    }
                    Some(Ok(_)) => {
                        // Leader cached nothing (empty blob): run from scratch and
                        // report the same scratch verdict as the leader.
                        outcome.decision = mlxcache_core::policy::PolicyDecision {
                            verdict: mlxcache_core::policy::CacheVerdict::Miss,
                            matched_tokens: 0,
                            request_tokens: tokens.len(),
                        };
                        outcome.blob_path = None;
                        outcome.prefill_from = 0;
                    }
                    None => {}
                }
            }
        }
    }

    // Time from request start to a usable cache state: routing plus, on a
    // miss/partial, the leader's prefill and atomic publish (or the wait for a
    // coalesced follower). This is the cache's real contribution to TTFT.
    let lookup_ms = started.elapsed().as_millis() as u64;
    // Record + log the verdict the request saw: the leader a miss/partial (it
    // prefilled), a coalesced follower the hit/partial it adopted. Recording
    // after single-flight keeps stats truthful.
    state.stats.record(&outcome.decision);
    log_request(&req.model, hash, &outcome.decision, lookup_ms);

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
        return stream_response(state.clone(), client, tokens, outcome, blob_arg, started).await;
    }

    // Generate: full context tokens, continuation from the request length.
    // The adapter opens the blob directly, so it needs the absolute path.
    let mut retired_blob = false;
    let generated = match client.generate(&tokens, 64, blob_arg).await {
        Ok(t) => t,
        Err(e) => {
            // If this request leaned on a blob and the adapter could not use it,
            // the checkpoint is bad (deleted, truncated, disk fault). Quarantine
            // it (R1-1) so identical requests stop 502ing, then retry from
            // scratch. A scratch failure is a genuine adapter error.
            if let Some(name) = outcome.blob_path.clone() {
                state.orchestrator.quarantine_checkpoint(&tokens);
                retired_blob = true;
                // Correct the aggregate: the covered KV counted at record() was
                // never in hand.
                state.stats.correct_retire(&outcome.decision);
                // Correct the decision's claim: no KV was actually reused.
                tracing::warn!(
                    blob = %name,
                    error = %e,
                    kv_claimed = outcome.prefill_from,
                    "blob unusable; quarantined, retrying from scratch (effective prefill_from=0)"
                );
                match client.generate(&tokens, 64, None).await {
                    Ok(t) => t,
                    Err(e2) => {
                        return Err(err(
                            StatusCode::BAD_GATEWAY,
                            &e2.to_string(),
                            "adapter_error",
                        ))
                    }
                }
            } else {
                return Err(err(
                    StatusCode::BAD_GATEWAY,
                    &e.to_string(),
                    "adapter_error",
                ));
            }
        }
    };

    let total_ms = started.elapsed().as_millis() as u64;
    let body = serde_json::json!({
        "mlxcache": {
            "verdict": if retired_blob {
                "miss"
            } else {
                match outcome.decision.verdict {
                    CacheVerdict::Hit => "hit",
                    CacheVerdict::Partial => "partial",
                    CacheVerdict::Miss => "miss",
                }
            },
            "tokens_cached": if retired_blob { 0 } else { outcome.prefill_from },
            "tokens_total": outcome.decision.request_tokens,
            "prefill_from": if retired_blob { 0 } else { outcome.prefill_from },
            // Time to the cache decision (prefill+publish on a miss). The
            // non-streaming path has no first-token hook, so the full round trip
            // is reported separately as total_ms.
            "lookup_ms": lookup_ms,
            "total_ms": total_ms,
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
    state: Arc<AppState>,
    client: &SidecarClient,
    tokens: Vec<u32>,
    outcome: crate::orchestrator::RouteOutcome,
    blob_arg: Option<&str>,
    started: std::time::Instant,
) -> Result<axum::response::Response, (StatusCode, Json<ErrorResponse>)> {
    use futures_util::StreamExt;

    // Time from request start to the cache decision, known before the first
    // token exists. The true TTFT (first generated token) is only known once the
    // sidecar emits it, so it is reported in the terminal frame, not here.
    let lookup_ms = started.elapsed().as_millis() as u64;

    let mut blob_for_open = outcome.blob_path.clone();
    let mut prefill_from = outcome.prefill_from;
    let retired_blob = std::cell::Cell::new(false);
    let upstream = match client.generate_stream(&tokens, 64, blob_arg).await {
        Ok(u) => u,
        Err(e) if blob_for_open.is_some() => {
            // The blob could not be opened (deleted/truncated/disk fault).
            // Quarantine it and retry from scratch so the stream still starts.
            if let Some(name) = blob_for_open.take() {
                state.orchestrator.quarantine_checkpoint(&tokens);
                state.stats.correct_retire(&outcome.decision);
                tracing::warn!(
                    blob = %name,
                    error = %e,
                    kv_claimed = prefill_from,
                    "blob unusable; quarantined, retrying stream from scratch (effective prefill_from=0)"
                );
            }
            prefill_from = 0;
            retired_blob.set(true);
            client
                .generate_stream(&tokens, 64, None)
                .await
                .map_err(|e| err(StatusCode::BAD_GATEWAY, &e.to_string(), "adapter_error"))?
        }
        Err(e) => {
            return Err(err(
                StatusCode::BAD_GATEWAY,
                &e.to_string(),
                "adapter_error",
            ))
        }
    };

    // The meta frame reflects what this stream actually did: if the blob was
    // retired, it is a scratch run (miss), not the original hit/partial.
    // tokens_cached and prefill_from are the same quantity (covered KV), so both
    // come from the single `prefill_from` value, already zeroed on retire.
    let verdict = if retired_blob.get() {
        "miss"
    } else {
        match outcome.decision.verdict {
            CacheVerdict::Hit => "hit",
            CacheVerdict::Partial => "partial",
            CacheVerdict::Miss => "miss",
        }
    };
    let meta_line = serde_json::json!({
        "mlxcache": {
            "verdict": verdict,
            "tokens_cached": prefill_from,
            "tokens_total": outcome.decision.request_tokens,
            "prefill_from": prefill_from,
            "lookup_ms": lookup_ms,
        }
    });

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
        started: std::time::Instant,
        ttft_ms: Option<u64>,
    }
    let state = StreamState {
        upstream: Box::pin(upstream.bytes_stream()),
        buf: Vec::new(),
        done: false,
        terminated: false,
        started,
        ttft_ms: None,
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
                        let last_done = frames.last().is_some_and(|f| {
                            f.as_ref().is_ok_and(|b| b.starts_with(b"data: [DONE]"))
                        });
                        // The first real token line marks TTFT.
                        if st.ttft_ms.is_none() && !last_done {
                            let is_token = frames.last().is_some_and(|f| {
                                f.as_ref().is_ok_and(|b| b.starts_with(b"data: {"))
                            });
                            if is_token {
                                st.ttft_ms = Some(st.started.elapsed().as_millis() as u64);
                            }
                        }
                        if last_done {
                            // Insert the true-TTFT stats frame BEFORE [DONE] so the
                            // terminator stays the last frame. push_frame already
                            // appended [DONE]; re-emit it after the stats frame.
                            let done = frames.pop().expect("just pushed [DONE]");
                            if let Some(ttft) = st.ttft_ms {
                                let stats = serde_json::json!({"mlxcache": {"ttft_ms": ttft}});
                                frames.push(Ok(format!("data: {stats}\n\n").into_bytes()));
                            }
                            frames.push(done);
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
                        // No [DONE] yet: emit the TTFT stats frame then terminate.
                        if let Some(ttft) = st.ttft_ms {
                            let stats = serde_json::json!({"mlxcache": {"ttft_ms": ttft}});
                            frames.push(Ok(format!("data: {stats}\n\n").into_bytes()));
                        }
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
        "checkpoints_published": state.orchestrator.published_count(),
        "checkpoints_quarantined": state.orchestrator.quarantined_count(),
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

    #[test]
    fn stats_correct_retire_removes_claimed_reuse() {
        // A hit recorded, then quarantined, must end as a miss with zero claimed
        // KV, or /stats and hit_rate report reuse that never happened.
        let stats = Stats::default();
        let hit = PolicyDecision {
            verdict: CacheVerdict::Hit,
            matched_tokens: 8,
            request_tokens: 8,
        };
        stats.record(&hit);
        {
            let s = stats.lock();
            assert_eq!(s.hits, 1);
            assert_eq!(s.tokens_cached, 7);
        }
        stats.correct_retire(&hit);
        let s = stats.lock();
        assert_eq!(s.hits, 0, "retired hit must not remain a hit");
        assert_eq!(s.misses, 1, "retired hit is an effective miss");
        assert_eq!(s.tokens_cached, 0, "no KV was actually reused");
        assert_eq!(s.requests, 1, "the request still happened");
    }
}
