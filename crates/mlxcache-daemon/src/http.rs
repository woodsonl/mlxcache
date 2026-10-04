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
use std::sync::OnceLock;

/// Cap on the stream line buffer between newlines. The idle timeout governs
/// byte ARRIVALS, not their content, so a fast flood without a newline would
/// otherwise grow `st.buf` unboundedly; past this the stream is cut with an
/// explicit upstream error (no legitimate token frame approaches it).
const MAX_PARTIAL_LINE: usize = 1024 * 1024;

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
    /// Optional JSONL trace capture (MLXCACHE_TRACE, step 5). None when the
    /// env knob is unset — the hot path costs one `Option::None` check.
    pub trace: Option<crate::trace::TraceWriter>,
}

impl AppState {
    /// One eviction pass against this state: reap cold checkpoints, then
    /// record the count in /stats. Both the background reaper (main.rs) and
    /// tests go through this single path, so the counter can never drift from
    /// what the reaper actually removed.
    pub fn evict_pass(&self, max_entries: usize, anchor_window: std::time::Duration) -> usize {
        let evicted = self
            .orchestrator
            .evict_cold(&self.persistence, max_entries, anchor_window);
        if evicted > 0 {
            self.stats.record_evictions(evicted as u64);
        }
        evicted
    }
}

/// Hit-rate counters (D3). Per-counter atomics instead of a Mutex: `record`
/// runs on every request, and under concurrent streaming the lock is a
/// cross-core contention point on the hot path for no benefit — the counters
/// are independent monotonic (or correcting) totals that never need to be
/// read and written transactionally together.
#[derive(Debug, Default)]
pub struct Stats {
    requests: std::sync::atomic::AtomicU64,
    hits: std::sync::atomic::AtomicU64,
    misses: std::sync::atomic::AtomicU64,
    partials: std::sync::atomic::AtomicU64,
    /// Sum of covered KV across requests. On a retire, `correct_retire` subtracts
    /// the entry's covered count, so this reflects KV actually in hand.
    tokens_cached: std::sync::atomic::AtomicU64,
    /// Sum of request lengths. `hit_rate = tokens_cached / tokens_total` is a
    /// coarse reuse diagnostic, not a true per-request fraction: a retire lowers
    /// the numerator without lowering this, so the ratio can dip below the
    /// fraction of requests that hit. Read it as a trend, not an exact rate.
    tokens_total: std::sync::atomic::AtomicU64,
    /// Checkpoints removed by the eviction reaper (ds4 anchor policy). Purely
    /// cumulative observability: evictions are policy, not failures.
    evictions: std::sync::atomic::AtomicU64,
}

/// Point-in-time read of every counter, for /stats and tests.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct StatsSnapshot {
    pub requests: u64,
    pub hits: u64,
    pub misses: u64,
    pub partials: u64,
    pub tokens_cached: u64,
    pub tokens_total: u64,
    pub evictions: u64,
}

impl Stats {
    fn get(c: &std::sync::atomic::AtomicU64) -> u64 {
        c.load(std::sync::atomic::Ordering::Relaxed)
    }
    /// Decremented-by-one, saturating at zero (a retire racing a concurrent
    /// record can observe a transient zero; counters are diagnostic).
    fn sub_one(c: &std::sync::atomic::AtomicU64) {
        let _ = c.fetch_update(
            std::sync::atomic::Ordering::Relaxed,
            std::sync::atomic::Ordering::Relaxed,
            |v| v.checked_sub(1),
        );
    }
    fn sub(c: &std::sync::atomic::AtomicU64, n: u64) {
        let _ = c.fetch_update(
            std::sync::atomic::Ordering::Relaxed,
            std::sync::atomic::Ordering::Relaxed,
            |v| v.checked_sub(n),
        );
    }

    pub fn record(&self, d: &PolicyDecision) {
        use std::sync::atomic::Ordering::Relaxed;
        self.requests.fetch_add(1, Relaxed);
        match d.verdict {
            CacheVerdict::Hit => self.hits.fetch_add(1, Relaxed),
            CacheVerdict::Miss => self.misses.fetch_add(1, Relaxed),
            CacheVerdict::Partial => self.partials.fetch_add(1, Relaxed),
        };
        // tokens_cached counts KV actually in hand (covered = matched-1), the
        // same value the response reports as tokens_cached/prefill_from. Counting
        // matched_tokens here would overstate reuse by one per hit.
        self.tokens_cached.fetch_add(
            covered_kv_tokens(d.verdict, d.matched_tokens) as u64,
            Relaxed,
        );
        self.tokens_total
            .fetch_add(d.request_tokens as u64, Relaxed);
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
        use std::sync::atomic::Ordering::Relaxed;
        if d.verdict == CacheVerdict::Miss {
            return;
        }
        match d.verdict {
            CacheVerdict::Hit => Self::sub_one(&self.hits),
            CacheVerdict::Partial => Self::sub_one(&self.partials),
            CacheVerdict::Miss => {}
        }
        self.misses.fetch_add(1, Relaxed);
        Self::sub(
            &self.tokens_cached,
            covered_kv_tokens(d.verdict, d.matched_tokens) as u64,
        );
    }

    pub fn record_evictions(&self, n: u64) {
        self.evictions
            .fetch_add(n, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> StatsSnapshot {
        StatsSnapshot {
            requests: Self::get(&self.requests),
            hits: Self::get(&self.hits),
            misses: Self::get(&self.misses),
            partials: Self::get(&self.partials),
            tokens_cached: Self::get(&self.tokens_cached),
            tokens_total: Self::get(&self.tokens_total),
            evictions: Self::get(&self.evictions),
        }
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

    // Tokenize (R1-2). T13: when a native tokenizer is configured the daemon
    // encodes in-process — no HTTP hop. Before the first native encode, the
    // daemon proves the configured tokenizer.json IS the engine's tokenizer:
    // one probe prompt is encoded both ways and the ids must match, or the
    // daemon refuses to serve (a mismatched artifact would corrupt routing).
    // On match it adopts the sidecar-reported tokenizer_hash + kv_dtype (+
    // kv quantization config), so fingerprints are IDENTICAL to the
    // pure-sidecar path — native tokenize is a pure latency optimization,
    // invisible to the cache.
    let (identity, tokens) = if let Some((_path, native)) = crate::native_tokenizer::from_env() {
        match resolve_native_identity(&prompt, native, state.sidecar.as_ref()).await {
            // A failed native encode must surface (502, matching the sidecar
            // tokenize path's failure class), never degrade to an empty prompt.
            Ok(identity) => match native.encode(&prompt) {
                Some(tokens) => (identity, tokens),
                None => {
                    return Err(err(
                        StatusCode::BAD_GATEWAY,
                        "native tokenizer failed to encode the prompt",
                        "adapter_error",
                    ));
                }
            },
            Err(e) => return Err(e),
        }
    } else {
        match &state.sidecar {
            Some(client) => match client.tokenize(&prompt).await {
                Ok(r) => (
                    Identity {
                        tokenizer_hash: r.tokenizer_hash,
                        kv_dtype: r.kv_dtype,
                        kv_bits: r.kv_bits,
                        kv_group_size: r.kv_group_size,
                    },
                    r.tokens,
                ),
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

    // Route + classify (R1-1 fingerprint check inside classify). The identity
    // comes from the adapter (R1-2): a constant here would let checkpoints
    // from different tokenizers/dtypes/quantization share a fingerprint and
    // be served wrongly.
    let fingerprint = ModelFingerprint {
        model_id: req.model.clone(),
        tokenizer_hash: identity.tokenizer_hash,
        kv_dtype: identity.kv_dtype,
        kv_layout_version: 1,
        kv_bits: identity.kv_bits,
        kv_group_size: identity.kv_group_size,
    };
    let mut outcome = state.orchestrator.route(&tokens, &fingerprint);
    // The token-only prefix hash for the request log line. The blob filename
    // uses a DIFFERENT key (blob_key = fingerprint + tokens), computed in the
    // leader branch where the fingerprint is in scope; the two are not equal.
    let prefix_hash = prefix_hash(&tokens);

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
        match state.singleflight.enter(&tokens).await {
            mlxcache_core::singleflight::Role::Leader(lead) => {
                // Delta prefill (OV3): on a PARTIAL match the route already
                // found an ancestor checkpoint covering tokens[:matched-1] of
                // KV. Pass its absolute path so the adapter loads that KV and
                // prefills only the uncovered delta, instead of recomputing
                // the whole prefix (the single biggest TTFT lever on growing
                // conversations). A full miss has no ancestor.
                let ancestor = outcome.blob.as_ref().map(|(name, _, _)| {
                    state
                        .persistence
                        .blob_dir
                        .join(name)
                        .to_string_lossy()
                        .into_owned()
                });
                match client.prefill(&tokens, ancestor.as_deref()).await {
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
                        outcome.decision = mlxcache_core::policy::scratch_decision(tokens.len());
                        outcome.blob = None;
                        outcome.prefill_from = 0;
                        lead.complete(Ok(String::new()));
                    }
                    Ok(blob) => {
                        publish_leader_blob(
                            state.clone(),
                            lead,
                            &mut outcome,
                            &fingerprint,
                            &tokens,
                            blob,
                        )
                        .await?;
                    }
                    Err(e) => {
                        // A rejected ANCESTOR (the adapter 422s when the blob it
                        // was told to resume from is bad) must not wedge the
                        // partial path: quarantine the ancestor exactly like the
                        // generate path does, then retry once from full scratch.
                        // Any other error (or a failed scratch retry) is an
                        // adapter failure: 502, no quarantine — same contract
                        // as the generate path's open failure.
                        if e.is_checkpoint_rejected() {
                            if let Some((name, generation, prefix)) = outcome.blob.take() {
                                state.orchestrator.quarantine_checkpoint(
                                    &state.persistence,
                                    &name,
                                    generation,
                                    &prefix,
                                );
                                tracing::warn!(
                                    blob = %name,
                                    error = %e,
                                    "ancestor rejected by adapter during prefill; quarantined, retrying from scratch"
                                );
                            }
                            outcome.decision =
                                mlxcache_core::policy::scratch_decision(tokens.len());
                            outcome.prefill_from = 0;
                            // The retry below re-derives blob state from the
                            // scratch prefill.
                            match client.prefill(&tokens, None).await {
                                Ok(blob) if !blob.is_empty() && tokens.len() >= 2 => {
                                    publish_leader_blob(
                                        state.clone(),
                                        lead,
                                        &mut outcome,
                                        &fingerprint,
                                        &tokens,
                                        blob,
                                    )
                                    .await?;
                                }
                                Ok(_) => {
                                    outcome.blob = None;
                                    lead.complete(Ok(String::new()));
                                }
                                Err(e2) => {
                                    let msg = e2.to_string();
                                    lead.complete(Err(msg.clone()));
                                    return Err(err(
                                        StatusCode::BAD_GATEWAY,
                                        &msg,
                                        "adapter_error",
                                    ));
                                }
                            }
                        } else {
                            let msg = e.to_string();
                            lead.complete(Err(msg.clone()));
                            return Err(err(StatusCode::BAD_GATEWAY, &msg, "adapter_error"));
                        }
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
                        // Leader cached nothing (empty blob: a one-token prompt,
                        // a publish refusal, or a publish failure): run from
                        // scratch and report the same scratch verdict as the
                        // leader. D2: a full disk must never turn a follower's
                        // working completion into a 502.
                        outcome.decision = mlxcache_core::policy::scratch_decision(tokens.len());
                        outcome.blob = None;
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
    log_request(&req.model, prefix_hash, &outcome.decision, lookup_ms);
    // Trace capture (step 5): one JSONL record per request, off the hot path
    // (bounded channel + writer thread). model_hash folds the fingerprint so
    // a multi-model trace stays attributable; messages embed the exact
    // payload so replay re-sends byte-equivalent requests. The payload
    // serialization happens ONLY when capture is on.
    if let Some(tracer) = &state.trace {
        let messages_value = serde_json::to_value(&req.messages).unwrap_or(serde_json::Value::Null);
        tracer.record(crate::trace::TraceRecord::from_request(
            (blob_key(&fingerprint, &[]) >> 64) as u64,
            &req.model,
            &messages_value,
            &outcome.decision,
        ));
    }

    // The adapter needs an absolute path to open the blob directly.
    let blob_abs = match &outcome.blob {
        Some((name, _, _)) => state
            .persistence
            .blob_dir
            .join(name)
            .to_string_lossy()
            .into_owned(),
        None => String::new(),
    };
    let blob_arg = if outcome.blob.is_some() {
        Some(blob_abs.as_str())
    } else {
        None
    };

    if req.stream {
        return stream_response(state.clone(), client, tokens, outcome, blob_arg, started).await;
    }

    // Generate: full context tokens, continuation from the request length.
    // The adapter opens the blob directly, so it needs the absolute path.
    // True when this request leaned on a blob but ended up not using its KV
    // (rejected OR a transient failure). The response and aggregate must then
    // report a scratch run: no prior KV was reused. Quarantine is a separate
    // decision, taken only on an explicit adapter rejection.
    let mut blob_unused = false;
    let generated = match client.generate(&tokens, 64, blob_arg).await {
        Ok(t) => t,
        Err(e) => {
            // If this request leaned on a blob and the request failed, retry
            // from scratch. A checkpoint the adapter explicitly rejected (422)
            // is bad: quarantine it so identical requests stop failing. A mere
            // transport/decode failure must NOT retire a healthy checkpoint.
            if let Some((name, generation, prefix)) = outcome.blob.clone() {
                blob_unused = true;
                if e.is_checkpoint_rejected() {
                    // Quarantine marks the entry; the blob file is deleted under
                    // the index lock (durable retirement, race-free vs republish).
                    state.orchestrator.quarantine_checkpoint(
                        &state.persistence,
                        &name,
                        generation,
                        &prefix,
                    );
                    tracing::warn!(
                        blob = %name,
                        error = %e,
                        kv_claimed = outcome.prefill_from,
                        "blob rejected by adapter; quarantined, retrying from scratch (effective prefill_from=0)"
                    );
                } else {
                    tracing::warn!(
                        blob = %name,
                        error = %e,
                        "blob run failed with a non-rejection error; retrying from scratch without quarantine (effective prefill_from=0)"
                    );
                }
                match client.generate(&tokens, 64, None).await {
                    Ok(t) => {
                        // Correct the aggregate only after the scratch retry
                        // succeeds: the covered KV counted at record() was never
                        // in hand. A 502 must not be counted as a served miss.
                        state.stats.correct_retire(&outcome.decision);
                        t
                    }
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
            "verdict": if blob_unused {
                "miss"
            } else {
                match outcome.decision.verdict {
                    CacheVerdict::Hit => "hit",
                    CacheVerdict::Partial => "partial",
                    CacheVerdict::Miss => "miss",
                }
            },
            "tokens_cached": if blob_unused { 0 } else { outcome.prefill_from },
            "tokens_total": outcome.decision.request_tokens,
            "prefill_from": if blob_unused { 0 } else { outcome.prefill_from },
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

/// Publish the leader's freshly prefilled checkpoint (shared by the plain
/// prefill path and the rejected-ancestor scratch retry) and settle the
/// leader's route outcome.
///
/// Delta-prefill-aware accounting: when the leader adopted an ancestor's KV
/// (`adopted_from = Some(matched)`), the prefill DID reuse `matched - 1`
/// tokens of cached KV, so the Partial verdict and its prefill_from stand —
/// reporting a miss would hide real reuse from /stats and hit_rate. When no
/// ancestor was adopted, the leader ran a fresh full prefill and reports a
/// plain miss with zero covered KV (generation resuming from the blob just
/// written is not reuse). On a write/refusal failure the outcome degrades:
/// a leader that adopted an ancestor keeps its Partial (the ancestor's KV is
/// still published and reusable — the ENOSPC/refusal path completes
/// single-flight with the ancestor's name so followers re-route onto it),
/// and only a leader with no adopted ancestor is forced to scratch, with
/// single-flight completed with the empty-name sentinel so followers run from
/// scratch too (OV6: a failed publish must never 502 a follower whose request
/// was fine).
async fn publish_leader_blob(
    state: Arc<AppState>,
    lead: mlxcache_core::singleflight::Leadership,
    outcome: &mut crate::orchestrator::RouteOutcome,
    fingerprint: &ModelFingerprint,
    tokens: &[u32],
    blob: Vec<u8>,
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    let adopted_from = outcome.blob.as_ref().map(|(_, _, prefix)| prefix.len());
    let meta = mlxcache_core::contract::CheckpointMeta {
        fingerprint: fingerprint.clone(),
        token_count: tokens.len() as u64,
        tokens: tokens.to_vec(),
        format_version: 1,
    };
    let key = blob_key(fingerprint, tokens);
    // Refuse before writing if the generation floor is unknown (a failed
    // startup scan): writing first would leak an unindexed file on every
    // request. Serve from scratch.
    if !state.orchestrator.can_publish() {
        // Floor unknown: don't write. Report a plain miss so followers also
        // serve from scratch (an empty name is the "not cached" signal, not an
        // error). An adopted ancestor's KV was still real reuse, but the
        // recovery-blocked daemon reports conservatively: scratch.
        tracing::warn!("not caching: startup recovery incomplete");
        outcome.decision = mlxcache_core::policy::scratch_decision(tokens.len());
        outcome.blob = None;
        outcome.prefill_from = 0;
        lead.complete(Ok(String::new()));
        return Ok(());
    }
    // Reserve the generation BEFORE writing so the on-disk name is
    // generation-specific (immutable). A republish then never overwrites an
    // earlier generation's file, so a delayed retirement cannot delete a
    // healthy replacement.
    let generation = state.orchestrator.reserve_generation();
    let blob_name = format!("{:032x}-{:016x}.ckpt", key, generation);
    // The write + fsync + rename runs on the blocking pool, not a runtime
    // worker: an fsync is milliseconds of disk latency, and blocking a worker
    // here stalls every OTHER request scheduled on it, including live streams
    // (A3). The payload MOVES into the closure (nothing needs it after
    // publish) and meta is cloned (publish_checkpoint still consumes it
    // below) — one memcpy vs. an ms-scale runtime stall.
    let persistence = state.persistence.clone();
    let meta_for_write = meta.clone();
    let write_result = tokio::task::spawn_blocking(move || {
        persistence.publish_atomic(key, generation, &meta_for_write, &blob)
    })
    .await
    .unwrap_or_else(|join| {
        // A panic inside publish_atomic is a daemon bug, not a request error:
        // treat it exactly like a failed write (serve uncached) and log it at
        // error level, loudly.
        tracing::error!(error = %join, "checkpoint publish task panicked");
        Err(crate::persistence::PersistError::Io(std::io::Error::other(
            "checkpoint publish task panicked",
        )))
    });
    match write_result {
        // ENOSPC rescue (registry): log and continue uncached. The LEADER
        // continues AND the single-flight result is the empty-name sentinel,
        // so coalesced followers also run from scratch instead of 502ing (D2
        // serve-uncached must hold for every request the disk failed under,
        // not just the leader).
        //
        // One refinement (delta prefill): when this leader ADOPTED an
        // ancestor's KV, that KV was loaded and verified by the adapter
        // moments ago and the ancestor is still published — so the leader
        // keeps generating from it (real reuse) while followers re-route onto
        // the same ancestor. Only a leader with no adopted ancestor falls
        // back to a plain scratch miss.
        Err(e) => {
            tracing::warn!(error = %e, "checkpoint write failed; continuing uncached");
            if let Some(matched) = adopted_from {
                // The ancestor KV was loaded and verified by the adapter
                // moments ago and the ancestor is still published: the leader
                // keeps generating from it (real reuse), and completing with
                // its name lets followers RE-ROUTE onto it (a Partial hit)
                // instead of forcing them to scratch.
                let ancestor_name = outcome
                    .blob
                    .as_ref()
                    .map(|(name, _, _)| name.clone())
                    .unwrap_or_default();
                outcome.decision.verdict = CacheVerdict::Partial;
                outcome.decision.matched_tokens = matched;
                outcome.prefill_from = covered_kv_tokens(CacheVerdict::Partial, matched);
                lead.complete(Ok(ancestor_name));
            } else {
                outcome.decision = mlxcache_core::policy::scratch_decision(tokens.len());
                outcome.blob = None;
                outcome.prefill_from = 0;
                lead.complete(Ok(String::new()));
            }
        }
        Ok(_) => {
            let published = state.orchestrator.publish_checkpoint(
                &state.persistence,
                tokens,
                meta,
                blob_name.clone(),
                generation,
            );
            // publish_checkpoint is expected to succeed (we checked
            // can_publish), but if the index rejected it, remove the
            // now-unindexed file so it cannot leak.
            if !published {
                if let Err(e) = state.persistence.remove(&blob_name) {
                    tracing::warn!(error = %e, "could not remove unindexed blob");
                }
            }
            if adopted_from.is_some() {
                // Delta prefill reused the ancestor's KV: keep the Partial
                // verdict + its covered count (prefill_from = matched-1) so
                // stats/reporting reflect the reuse. Generation resumes from
                // the NEW blob (which covers tokens[:-1]).
                outcome.blob = Some((blob_name.clone(), generation, tokens.to_vec()));
            } else {
                // Fresh full prefill: report a miss with zero covered KV so
                // response, /stats, and the log agree on what happened.
                outcome.decision = mlxcache_core::policy::scratch_decision(tokens.len());
                outcome.prefill_from = 0;
                outcome.blob = Some((blob_name.clone(), generation, tokens.to_vec()));
            }
            lead.complete(Ok(blob_name));
        }
    }
    Ok(())
}

/// Adapter-reported cache identity (R1-1/R1-2): everything the fingerprint
/// pins that the sidecar owns. The native-tokenize path adopts it wholesale
/// after the parity probe so fingerprints stay byte-identical to the
/// pure-sidecar path.
#[derive(Debug, Clone)]
struct Identity {
    tokenizer_hash: String,
    kv_dtype: String,
    kv_bits: u8,
    kv_group_size: u32,
}

/// T13 identity resolution: prove the native tokenizer IS the engine's, once,
/// then reuse the verdict forever (the artifact cannot change mid-process).
///
/// The probe encodes a fixed prompt natively and via the sidecar and requires
/// exact id equality. On success the sidecar-reported (tokenizer_hash,
/// kv_dtype) is cached and adopted for every later request — fingerprints are
/// byte-identical to the pure-sidecar path. On mismatch the error is 503
/// (fail closed): serving with the wrong tokenizer would mis-route every
/// checkpoint.
async fn resolve_native_identity(
    prompt: &str,
    native: &crate::native_tokenizer::NativeTokenizer,
    sidecar: Option<&SidecarClient>,
) -> Result<Identity, (StatusCode, Json<ErrorResponse>)> {
    static VERIFIED: OnceLock<Result<Identity, String>> = OnceLock::new();
    if let Some(Ok(identity)) = VERIFIED.get() {
        return Ok(identity.clone());
    }
    if let Some(Err(reason)) = VERIFIED.get() {
        return Err(err(
            StatusCode::SERVICE_UNAVAILABLE,
            reason.as_str(),
            "adapter_unavailable",
        ));
    }

    let Some(client) = sidecar else {
        // No sidecar at all: there is nothing to verify against and no engine
        // to prefill with either, so the standard no-adapter error applies.
        return Err(err(
            StatusCode::SERVICE_UNAVAILABLE,
            "no adapter configured",
            "adapter_unavailable",
        ));
    };
    let probe = format!("mlxcache native-tokenizer parity probe: {prompt}");
    let reference = match client.tokenize(&probe).await {
        Ok(r) => r,
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
    };
    // A failed probe encode is parity-unprovable: treat it exactly like a
    // mismatch and fail closed (the verdict is cached either way).
    let local = native.encode(&probe);
    if local.as_ref() != Some(&reference.tokens) {
        let reason = format!(
            "native tokenizer does not match the engine's tokenizer \
             (probe encode: {n} native ids vs {m} sidecar ids) — refusing to serve; \
             fix MLXCACHE_NATIVE_TOKENIZER",
            n = local.as_ref().map_or(0, Vec::len),
            m = reference.tokens.len()
        );
        let _ = VERIFIED.set(Err(reason.clone()));
        return Err(err(
            StatusCode::SERVICE_UNAVAILABLE,
            &reason,
            "adapter_unavailable",
        ));
    }
    let identity = Identity {
        tokenizer_hash: reference.tokenizer_hash,
        kv_dtype: reference.kv_dtype,
        kv_bits: reference.kv_bits,
        kv_group_size: reference.kv_group_size,
    };
    let _ = VERIFIED.set(Ok(identity.clone()));
    Ok(identity)
}

/// The token-only prefix hash for the request log line: 128 bits from two
/// FNV-1a passes with different seeds and different primes, so a collision in
/// one lane does not correlate with the other. 64 bits is too thin as the
/// store grows; a collision would alias two distinct KV states in the log.
/// (The index is exact and keyed by token ids; this only names log lines.)
/// Public for the B1 criterion bench; the blob filename uses a DIFFERENT key
/// (`blob_key` = fingerprint + tokens) — the two are not equal.
pub fn prefix_hash(tokens: &[u32]) -> u128 {
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
/// silent wrong output. Prepending the fingerprint keeps the two files apart
/// — short of a 128-bit fold collision (birthday scale ~2⁻⁶⁴, accepted
/// residual risk in the D1 register): the lookup-time fingerprint check
/// reads the INDEX entry, not the file, so a collision would NOT be caught
/// downstream. Public for the B1 criterion bench.
pub fn blob_key(fingerprint: &mlxcache_core::contract::ModelFingerprint, tokens: &[u32]) -> u128 {
    // Length-prefix each field so no field's bytes can be re-split across the
    // boundaries: joining with a separator is ambiguous when model_id itself
    // (request-controlled) contains the separator, which would alias two
    // fingerprints to one key and one blob file.
    fn put(out: &mut String, field: &str) {
        out.push_str(&field.len().to_string());
        out.push(':');
        out.push_str(field);
    }
    let mut f = String::new();
    put(&mut f, &fingerprint.model_id);
    put(&mut f, &fingerprint.tokenizer_hash);
    put(&mut f, &fingerprint.kv_dtype);
    put(&mut f, &fingerprint.kv_layout_version.to_string());
    // KV quantization config (T12): an 8-bit checkpoint of the same model must
    // never share a filename with the f16 one — a republish at the same name
    // would unlink the other tier's blob (index replacement reclaims the old
    // file), ping-ponging the two tiers on every alternating request.
    put(&mut f, &fingerprint.kv_bits.to_string());
    put(&mut f, &fingerprint.kv_group_size.to_string());
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
///
/// Hot-path shape (P2b): the line is validated with a two-field typed struct —
/// serde skips every other field without allocating — and the frame is emitted
/// by forwarding the engine's OWN bytes. The old path built a full
/// `serde_json::Value` tree per token and re-serialized it; at 64+ tokens per
/// stream that is a tree alloc + string churn per token for zero gain. Key
/// order and spacing now come straight from the sidecar instead of serde's
/// normalization — both are valid JSON and clients parse by key, not layout.
fn push_frame(frames: &mut Vec<Result<bytes::Bytes, std::io::Error>>, raw: &[u8]) {
    /// A done marker may carry any NON-NULL JSON value (`true`, `1`, …):
    /// `Option<IgnoredAny>` maps a literal `null` to `None`, so presence of
    /// the key with a null value would NOT terminate the stream. Our sidecar
    /// pins `true` (server.py's done line), making the gap contractual
    /// rather than live (red-team 2026-10-04).
    #[derive(serde::Deserialize)]
    struct SidecarStreamFrame {
        #[serde(default)]
        done: Option<serde::de::IgnoredAny>,
    }
    let line = match std::str::from_utf8(raw) {
        Ok(s) => s.trim(),
        // Invalid UTF-8 from the sidecar is a protocol violation (it must be,
        // because the SSE body is a text/event-stream). Heap-building the
        // error is fine: this is a fatal one-time stream abort, not a
        // per-token cost.
        Err(_) => {
            frames.push(Err(std::io::Error::other(
                "sidecar emitted a non-UTF-8 stream line",
            )));
            return;
        }
    };
    if line.is_empty() {
        return;
    }
    match serde_json::from_str::<SidecarStreamFrame>(line) {
        Ok(f) if f.done.is_some() => {
            frames.push(Ok(bytes::Bytes::from_static(b"data: [DONE]\n\n")))
        }
        Ok(_) => {
            let mut frame = Vec::with_capacity(line.len() + 8);
            frame.extend_from_slice(b"data: ");
            frame.extend_from_slice(line.as_bytes());
            frame.extend_from_slice(b"\n\n");
            frames.push(Ok(bytes::Bytes::from(frame)));
        }
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

    let mut blob_for_open = outcome.blob.clone();
    let mut prefill_from = outcome.prefill_from;
    // True when the stream leaned on a blob but ran from scratch instead
    // (rejection OR transient failure): the meta frame and aggregate must then
    // report a miss. Quarantine stays conditional on an explicit rejection.
    let blob_unused = std::cell::Cell::new(false);
    let upstream = match client.generate_stream(&tokens, 64, blob_arg).await {
        Ok(u) => u,
        Err(e) if blob_for_open.is_some() => {
            // The stream could not be opened. Retry from scratch so the stream
            // still starts. Quarantine only on an explicit adapter rejection
            // (422): a transport/decode failure must not retire a healthy blob.
            if e.is_checkpoint_rejected() {
                if let Some((name, generation, prefix)) = blob_for_open.take() {
                    state.orchestrator.quarantine_checkpoint(
                        &state.persistence,
                        &name,
                        generation,
                        &prefix,
                    );
                    tracing::warn!(
                        blob = %name,
                        error = %e,
                        kv_claimed = prefill_from,
                        "blob rejected by adapter; quarantined, retrying stream from scratch (effective prefill_from=0)"
                    );
                }
            } else {
                tracing::warn!(
                    error = %e,
                    "stream open failed with a non-rejection error; retrying from scratch without quarantine (effective prefill_from=0)"
                );
            }
            prefill_from = 0;
            blob_unused.set(true);
            let upstream = client
                .generate_stream(&tokens, 64, None)
                .await
                .map_err(|e| err(StatusCode::BAD_GATEWAY, &e.to_string(), "adapter_error"))?;
            // Correct the aggregate only after the scratch retry succeeded: a
            // 502 must not be counted as a served miss.
            state.stats.correct_retire(&outcome.decision);
            upstream
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
    //
    // ponytail: the meta frame is sent BEFORE the first token so TTFT can be
    // reported. A mid-stream decode failure after a successful load truncates
    // the stream (the adapter cannot tell OOM from a corrupt-but-loadable blob)
    // and is signaled by an `upstream_error` frame, not a retirement: the design
    // forbids retiring a healthy checkpoint on a transient decode failure. If a
    // blob proves persistently undecodable, add a consecutive-failure counter
    // that retires after K and re-emit a corrected terminal frame.
    //
    // /stats semantics on truncation: the checkpoint DID load and its covered KV
    // WAS reused, so `hits`/`tokens_cached` are NOT corrected — they measure KV
    // reuse (a trend metric), while `upstream_error` frames measure requests that
    // completed. The two are distinct on purpose; do not "fix" one to match the
    // other without changing the documented meaning of the metric.
    let verdict = if blob_unused.get() {
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
    //
    // The line buffer itself is bounded (MAX_PARTIAL_LINE): a flood of bytes
    // containing no newline never trips the idle timeout (bytes ARE arriving),
    // so without a cap the buffer would grow unboundedly on a fast flood.
    let meta_bytes = bytes::Bytes::from(format!("data: {meta_line}\n\n"));
    let first =
        futures_util::stream::once(async move { Ok::<bytes::Bytes, std::io::Error>(meta_bytes) });

    struct StreamState {
        upstream: std::pin::Pin<
            Box<dyn futures_util::Stream<Item = reqwest::Result<bytes::Bytes>> + Send>,
        >,
        /// Line buffer. BytesMut, not Vec<u8>: `split_to` hands out a line as
        /// an O(1) view — the old `drain(..=pos).collect()` memmoved the whole
        /// remaining buffer once per token (quadratic over a stream).
        buf: bytes::BytesMut,
        done: bool,
        terminated: bool,
        started: std::time::Instant,
        ttft_ms: Option<u64>,
        /// Max gap with no bytes from the sidecar before the stream is cut
        /// with an explicit error (the sidecar stalling must not wedge a
        /// client forever now that streams carry no total timeout).
        idle: std::time::Duration,
    }
    let state = StreamState {
        upstream: Box::pin(upstream.bytes_stream()),
        buf: bytes::BytesMut::with_capacity(8 * 1024),
        done: false,
        terminated: false,
        started,
        ttft_ms: None,
        idle: client.stream_idle_timeout(),
    };
    let body_stream = futures_util::stream::unfold(state, |mut st| async move {
        use futures_util::StreamExt;
        let mut frames: Vec<Result<bytes::Bytes, std::io::Error>> = Vec::new();
        loop {
            if st.done {
                return None;
            }
            match tokio::time::timeout(st.idle, st.upstream.next()).await {
                // The sidecar stopped producing tokens mid-stream: emit the
                // same explicit-error + [DONE] treatment as a truncated
                // upstream so the client sees a failure, not a hang.
                Err(_elapsed) => {
                    if let Some(ttft) = st.ttft_ms {
                        let stats = serde_json::json!({"mlxcache": {"ttft_ms": ttft}});
                        frames.push(Ok(bytes::Bytes::from(format!("data: {stats}\n\n"))));
                    }
                    let err = serde_json::json!({
                        "error": {
                            "message": "upstream idle timeout: no tokens within the streaming budget",
                            "type": "upstream_error",
                        }
                    });
                    frames.push(Ok(bytes::Bytes::from(format!("data: {err}\n\n"))));
                    frames.push(Ok(bytes::Bytes::from_static(b"data: [DONE]\n\n")));
                    st.done = true;
                    break;
                }
                Ok(Some(Ok(bytes))) => {
                    st.buf.extend_from_slice(&bytes);
                    // Bound the partial-line buffer: a fast flood with no
                    // newline never fires the idle timeout, so the buffer
                    // needs its own cap. Frames are emitted (including the
                    // stats frame when TTFT is known), not silently dropped,
                    // matching the stream-abort shape above.
                    if !st.buf.contains(&b'\n') && st.buf.len() > MAX_PARTIAL_LINE {
                        if let Some(ttft) = st.ttft_ms {
                            let stats = serde_json::json!({"mlxcache": {"ttft_ms": ttft}});
                            frames.push(Ok(bytes::Bytes::from(format!("data: {stats}\n\n"))));
                        }
                        let err = serde_json::json!({
                            "error": {
                                "message": "upstream protocol error: unbounded line",
                                "type": "upstream_error",
                            }
                        });
                        frames.push(Ok(bytes::Bytes::from(format!("data: {err}\n\n"))));
                        frames.push(Ok(bytes::Bytes::from_static(b"data: [DONE]\n\n")));
                        st.done = true;
                        break;
                    }
                    while let Some(pos) = st.buf.iter().position(|b| *b == b'\n') {
                        let line = st.buf.split_to(pos + 1);
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
                                frames.push(Ok(bytes::Bytes::from(format!("data: {stats}\n\n"))));
                            }
                            frames.push(done);
                            st.terminated = true;
                        }
                    }
                    if !frames.is_empty() {
                        break;
                    }
                }
                Ok(Some(Err(e))) => {
                    frames.push(Err(std::io::Error::other(e.to_string())));
                    break;
                }
                Ok(None) => {
                    // Upstream ended: flush a trailing line with no newline, then
                    // terminate. A clean end is the sidecar's `{"done":true}` line
                    // (already mapped to `[DONE]` by push_frame). If EOF arrives
                    // WITHOUT that marker, the sidecar died mid-stream (decode
                    // error, OOM, crash): surface an error frame so the client
                    // does not read a silent truncation as a completed answer.
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
                        if let Some(ttft) = st.ttft_ms {
                            let stats = serde_json::json!({"mlxcache": {"ttft_ms": ttft}});
                            frames.push(Ok(bytes::Bytes::from(format!("data: {stats}\n\n"))));
                        }
                        // Truncated upstream: an explicit error, not a success.
                        let err = serde_json::json!({
                            "error": {
                                "message": "upstream stream ended without a completion marker",
                                "type": "upstream_error",
                            }
                        });
                        frames.push(Ok(bytes::Bytes::from(format!("data: {err}\n\n"))));
                        frames.push(Ok(bytes::Bytes::from_static(b"data: [DONE]\n\n")));
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
    let s = state.stats.snapshot();
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
        "evictions": s.evictions,
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
        let text = std::string::String::from_utf8(frames[0].as_ref().unwrap().to_vec()).unwrap();
        assert!(text.starts_with("data: {"));
        assert!(text.ends_with("\n\n"));

        frames.clear();
        push_frame(&mut frames, b"{\"done\":true}\n");
        assert_eq!(
            frames[0].as_ref().unwrap().as_ref(),
            b"data: [DONE]\n\n".as_slice()
        );

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
            trace: None,
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
            kv_bits: 0,
            kv_group_size: 0,
        };
        let b = mlxcache_core::contract::ModelFingerprint {
            model_id: "model-b".into(),
            tokenizer_hash: "tok-a".into(),
            kv_dtype: "f16".into(),
            kv_layout_version: 1,
            kv_bits: 0,
            kv_group_size: 0,
        };
        assert_ne!(blob_key(&a, &tokens), blob_key(&b, &tokens));
        // Tokenizer-only difference must also separate.
        let c = mlxcache_core::contract::ModelFingerprint {
            model_id: "model-a".into(),
            tokenizer_hash: "tok-b".into(),
            kv_dtype: "f16".into(),
            kv_layout_version: 1,
            kv_bits: 0,
            kv_group_size: 0,
        };
        assert_ne!(blob_key(&a, &tokens), blob_key(&c, &tokens));
        // Deterministic and distinct from the bare token hash.
        assert_eq!(blob_key(&a, &tokens), blob_key(&a, &[10, 20, 30]));
        assert_ne!(blob_key(&a, &tokens), prefix_hash(&tokens));
    }

    #[test]
    fn blob_key_folds_fields_unambiguously() {
        // Length-prefixed fields: model_id is request-controlled and could carry
        // the old separator, which would let two distinct fingerprints fold to
        // one key and alias one blob file. With length prefixes the split is
        // unique, so these two must differ.
        let tokens = vec![1, 2, 3];
        let a = mlxcache_core::contract::ModelFingerprint {
            model_id: "m\u{1f}tok".into(),
            tokenizer_hash: "x".into(),
            kv_dtype: "f16".into(),
            kv_layout_version: 1,
            kv_bits: 0,
            kv_group_size: 0,
        };
        let b = mlxcache_core::contract::ModelFingerprint {
            model_id: "m".into(),
            tokenizer_hash: "tok\u{1f}x".into(),
            kv_dtype: "f16".into(),
            kv_layout_version: 1,
            kv_bits: 0,
            kv_group_size: 0,
        };
        assert_ne!(blob_key(&a, &tokens), blob_key(&b, &tokens));
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
            let s = stats.snapshot();
            assert_eq!(s.hits, 1);
            assert_eq!(s.tokens_cached, 7);
        }
        stats.correct_retire(&hit);
        let s = stats.snapshot();
        assert_eq!(s.hits, 0, "retired hit must not remain a hit");
        assert_eq!(s.misses, 1, "retired hit is an effective miss");
        assert_eq!(s.tokens_cached, 0, "no KV was actually reused");
        assert_eq!(s.requests, 1, "the request still happened");
    }

    #[test]
    fn stats_correct_retire_partial_and_miss() {
        let stats = Stats::default();
        // A partial recorded, then retired: partials->misses, covered KV dropped.
        let partial = PolicyDecision {
            verdict: CacheVerdict::Partial,
            matched_tokens: 6,
            request_tokens: 10,
        };
        stats.record(&partial);
        {
            let s = stats.snapshot();
            assert_eq!(s.partials, 1);
            assert_eq!(s.tokens_cached, 5);
        }
        stats.correct_retire(&partial);
        {
            let s = stats.snapshot();
            assert_eq!(s.partials, 0, "retired partial must not remain a partial");
            assert_eq!(s.misses, 1);
            assert_eq!(s.tokens_cached, 0, "no KV was actually reused");
        }
        // correct_retire on an already-Miss decision is a no-op: a cold-miss
        // leader keeps its miss while adopting the blob it just published, and
        // retiring that blob must not count a second miss.
        let miss = PolicyDecision {
            verdict: CacheVerdict::Miss,
            matched_tokens: 0,
            request_tokens: 10,
        };
        stats.record(&miss);
        stats.correct_retire(&miss);
        let s = stats.snapshot();
        assert_eq!(s.misses, 2, "the miss is not double-counted");
    }

    #[test]
    fn stats_record_is_lock_free_and_concurrent() {
        // The atomics-backed counters must survive concurrent record/retire from
        // many tasks without deadlock, poisoning, or lost updates.
        let stats = Stats::default();
        let stats = std::sync::Arc::new(stats);
        let mut handles = Vec::new();
        for _ in 0..16 {
            let stats = stats.clone();
            handles.push(std::thread::spawn(move || {
                let hit = PolicyDecision {
                    verdict: CacheVerdict::Hit,
                    matched_tokens: 8,
                    request_tokens: 8,
                };
                for _ in 0..1000 {
                    stats.record(&hit);
                    // No retire here: retires would legitimately move verdicts
                    // and make totals order-dependent; record is the hot path.
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        let s = stats.snapshot();
        assert_eq!(s.requests, 16_000);
        assert_eq!(s.hits, 16_000);
        assert_eq!(s.tokens_cached, 112_000);
        assert_eq!(s.tokens_total, 128_000);
    }
}

/// Public surface for the allocation gate (`tests/alloc_gate.rs`). The gate
/// runs in its own test binary (a counting global allocator is process-wide)
/// and needs to exercise the REAL framer, not a copy of it.
pub mod test_support {
    pub fn push_frame_for_gate(frames: &mut Vec<Result<bytes::Bytes, std::io::Error>>, raw: &[u8]) {
        super::push_frame(frames, raw);
    }
}
