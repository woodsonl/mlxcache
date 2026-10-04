//! Sidecar HTTP client: the mlx-lm compatibility adapter (Python process).
//!
//! Protocol (docs/contract-spec.md): JSON envelopes over localhost HTTP;
//! KV payloads ride as binary frames (length-prefixed). The sidecar is never
//! the hot-path default once the owner engine ships.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct SidecarConfig {
    pub base_url: String,
    /// Model id the sidecar should have loaded.
    pub model_id: String,
}

impl SidecarConfig {
    pub fn new(base_url: String, model_id: String) -> Self {
        Self { base_url, model_id }
    }
}

#[derive(Debug, Serialize)]
pub struct TokenizeRequest {
    pub prompt: String,
}

#[derive(Debug, Deserialize)]
pub struct TokenizeResponse {
    pub tokens: Vec<u32>,
    /// Hash of the tokenizer artifact the sidecar used.
    pub tokenizer_hash: String,
    /// KV/compute dtype the engine caches in (e.g. "float16", "bfloat16"). Pin
    /// it in the fingerprint: a checkpoint from a different dtype holds
    /// different bytes and must not be served to this request.
    #[serde(default = "default_kv_dtype")]
    pub kv_dtype: String,
    /// KV quantization the engine applies before persisting (T12): 0 = f16.
    /// Defaults keep older sidecars (which never reported these) compatible —
    /// they were f16, and f16 is the 0/0 fingerprint.
    #[serde(default)]
    pub kv_bits: u8,
    /// Group size for `kv_bits` (0 when unquantized). Distinct group sizes are
    /// distinct formats for R1-1, not tunable variants.
    #[serde(default)]
    pub kv_group_size: u32,
}

fn default_kv_dtype() -> String {
    "unknown".to_string()
}

#[derive(Debug, thiserror::Error)]
pub enum SidecarError {
    #[error("sidecar unreachable at {url}: {source}")]
    Unreachable { url: String, source: reqwest::Error },
    #[error("sidecar returned {status}: {body}")]
    Http { status: u16, body: String },
    #[error("sidecar protocol violation: {0}")]
    Protocol(String),
    /// The sidecar rejected the checkpoint we asked it to resume from (HTTP
    /// 422): corrupt payload, undecodable header, or a prefix that disagrees
    /// with the request. Only this error retires the entry; a decode or
    /// transport failure must NOT quarantine a healthy checkpoint.
    #[error("checkpoint rejected by adapter: {body}")]
    CheckpointRejected { body: String },
    #[error("sidecar stream did not open within {secs}s at {url}")]
    StreamOpenTimeout { url: String, secs: u64 },
}

/// Default idle budget for a live generation stream (seconds): the longest
/// gap between bytes the daemon tolerates before declaring the sidecar wedged.
fn default_stream_idle_s() -> u64 {
    60
}

impl SidecarError {
    /// Whether this error means the checkpoint we passed is bad and must be
    /// quarantined. True only for the adapter's explicit rejection.
    pub fn is_checkpoint_rejected(&self) -> bool {
        matches!(self, SidecarError::CheckpointRejected { .. })
    }
}

/// Map a non-success sidecar response to the right error. HTTP 422 is the
/// adapter's explicit checkpoint rejection; everything else is a plain HTTP
/// error. Shared so every endpoint classifies identically.
async fn classify_http_error(resp: reqwest::Response) -> SidecarError {
    let status = resp.status().as_u16();
    let body = resp.text().await.unwrap_or_default();
    classify_status(status, body)
}

/// The status/body classification, pure so it is unit-testable.
fn classify_status(status: u16, body: String) -> SidecarError {
    if status == 422 {
        SidecarError::CheckpointRejected { body }
    } else {
        SidecarError::Http { status, body }
    }
}

/// Minimal async HTTP client for the sidecar. reqwest keeps the connection
/// pool warm; JSON bodies only (KV blobs stream through a separate path).
pub struct SidecarClient {
    config: SidecarConfig,
    /// JSON calls (tokenize/prefill/generate): a TOTAL request timeout is
    /// correct for them — each has a bounded, short-ish duration.
    http: reqwest::Client,
    /// Streaming generation: NO total timeout (a decode stream's wall time is
    /// unbounded by design). Connect is bounded by `connect_timeout`; the
    /// open phase (send → response headers) by `stream_open`; and idle gaps
    /// between body chunks are enforced per-read by the caller via
    /// [`Self::stream_idle_timeout`].
    stream_http: reqwest::Client,
    stream_open: std::time::Duration,
    stream_idle: std::time::Duration,
}

impl SidecarClient {
    pub fn new(config: SidecarConfig) -> Result<Self, SidecarError> {
        // A hung sidecar must not wedge the daemon (and, via single-flight, every
        // follower). Default: generous enough for a 50K-token prefill on the
        // measured slowest model (~10s), overridable via MLXCACHE_SIDECAR_TIMEOUT_S.
        let timeout_s: u64 = std::env::var("MLXCACHE_SIDECAR_TIMEOUT_S")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(120);
        let idle_s: u64 = std::env::var("MLXCACHE_STREAM_IDLE_TIMEOUT_S")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(60);
        Self::with_timeouts(config, timeout_s, idle_s)
    }

    /// Build with an explicit total timeout for JSON calls (testable without
    /// mutating global env); stream budgets keep their defaults.
    pub fn with_timeout(config: SidecarConfig, timeout_s: u64) -> Result<Self, SidecarError> {
        Self::with_timeouts(config, timeout_s, default_stream_idle_s())
    }

    /// Build with explicit JSON-total and stream-idle budgets (seconds).
    pub fn with_timeouts(
        config: SidecarConfig,
        timeout_s: u64,
        stream_idle_s: u64,
    ) -> Result<Self, SidecarError> {
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(timeout_s))
            .build()
            .map_err(|_e| SidecarError::Protocol("client build failed".into()))?;
        let stream_http = reqwest::Client::builder()
            // Bound connect attempts on streams; everything after the response
            // headers is governed by open/idle budgets, not a total deadline.
            .connect_timeout(std::time::Duration::from_secs(5))
            .build()
            .map_err(|_e| SidecarError::Protocol("stream client build failed".into()))?;
        Ok(Self {
            config,
            http,
            stream_http,
            // The open deadline covers blob load + stream setup on the sidecar:
            // same generous budget as a JSON prefill.
            stream_open: std::time::Duration::from_secs(timeout_s),
            stream_idle: std::time::Duration::from_secs(stream_idle_s.max(1)),
        })
    }

    /// Max wall time with NO bytes arriving from a live generation stream
    /// before the daemon terminates it with an explicit upstream error.
    pub fn stream_idle_timeout(&self) -> std::time::Duration {
        self.stream_idle
    }

    pub async fn prefill(
        &self,
        tokens: &[u32],
        ancestor_blob_path: Option<&str>,
    ) -> Result<Vec<u8>, SidecarError> {
        let url = format!("{}/prefill", self.config.base_url);
        let resp = self
            .http
            .post(&url)
            .json(&serde_json::json!({
                "tokens": tokens,
                // Delta prefill (OV3): the matched ancestor checkpoint to
                // adopt KV from. None = full scratch prefill.
                "ancestor_blob_path": ancestor_blob_path,
            }))
            .send()
            .await
            .map_err(|e| SidecarError::Unreachable {
                url: url.clone(),
                source: e,
            })?;
        let status = resp.status();
        if !status.is_success() {
            return Err(classify_http_error(resp).await);
        }
        resp.bytes()
            .await
            .map(|b| b.to_vec())
            .map_err(|e| SidecarError::Protocol(format!("bad prefill response: {e}")))
    }

    pub async fn generate(
        &self,
        tokens: &[u32],
        max_tokens: usize,
        blob_path: Option<&str>,
    ) -> Result<Vec<u32>, SidecarError> {
        let url = format!("{}/generate", self.config.base_url);
        let resp = self
            .http
            .post(&url)
            .json(&serde_json::json!({
                "tokens": tokens,
                "max_tokens": max_tokens,
                "blob_path": blob_path,
            }))
            .send()
            .await
            .map_err(|e| SidecarError::Unreachable {
                url: url.clone(),
                source: e,
            })?;
        let status = resp.status();
        if !status.is_success() {
            return Err(classify_http_error(resp).await);
        }
        #[derive(serde::Deserialize)]
        struct GenResponse {
            tokens: Vec<u32>,
        }
        resp.json::<GenResponse>()
            .await
            .map(|r| r.tokens)
            .map_err(|e| SidecarError::Protocol(format!("bad generate response: {e}")))
    }

    /// Start a streaming generation: returns a byte stream of NDJSON from the
    /// sidecar (one `{"token":..,"text":..}` per token, then `{"done":true}`).
    ///
    /// The stream-open phase (connect → response headers) is bounded by the
    /// same budget as a JSON request. After that the stream's wall time is
    /// unbounded — the CALLER enforces an idle budget between chunks.
    pub async fn generate_stream(
        &self,
        tokens: &[u32],
        max_tokens: usize,
        blob_path: Option<&str>,
    ) -> Result<reqwest::Response, SidecarError> {
        let url = format!("{}/generate", self.config.base_url);
        let open = self.stream_open;
        let fut = async {
            self.stream_http
                .post(&url)
                .json(&serde_json::json!({
                    "tokens": tokens,
                    "max_tokens": max_tokens,
                    "blob_path": blob_path,
                    "stream": true,
                }))
                .send()
                .await
        };
        let resp = match tokio::time::timeout(open, fut).await {
            // Elapsed before headers: the sidecar wedged opening the stream.
            Err(_elapsed) => {
                return Err(SidecarError::StreamOpenTimeout {
                    url,
                    secs: open.as_secs(),
                })
            }
            Ok(inner) => inner.map_err(|e| SidecarError::Unreachable {
                url: url.clone(),
                source: e,
            })?,
        };
        let status = resp.status();
        if !status.is_success() {
            return Err(classify_http_error(resp).await);
        }
        Ok(resp)
    }

    pub async fn tokenize(&self, prompt: &str) -> Result<TokenizeResponse, SidecarError> {
        let url = format!("{}/tokenize", self.config.base_url);
        let resp = self
            .http
            .post(&url)
            .json(&TokenizeRequest {
                prompt: prompt.to_string(),
            })
            .send()
            .await
            .map_err(|e| SidecarError::Unreachable {
                url: url.clone(),
                source: e,
            })?;
        let status = resp.status();
        if !status.is_success() {
            return Err(classify_http_error(resp).await);
        }
        resp.json::<TokenizeResponse>()
            .await
            .map_err(|e| SidecarError::Protocol(format!("bad tokenize response: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_shapes() {
        let c = SidecarConfig::new("http://127.0.0.1:8421".into(), "m".into());
        assert_eq!(c.base_url, "http://127.0.0.1:8421");
    }

    #[test]
    fn status_classification_is_422_only() {
        // Only 422 retires a checkpoint; every other failure must not, or a
        // transient 500 would quarantine a healthy entry.
        let rejected = classify_status(422, "bad header".into());
        assert!(rejected.is_checkpoint_rejected());
        assert!(matches!(rejected, SidecarError::CheckpointRejected { .. }));

        for status in [500u16, 503, 502, 400, 404, 200] {
            let e = classify_status(status, "x".into());
            assert!(
                !e.is_checkpoint_rejected(),
                "status {status} must not retire a checkpoint"
            );
            assert!(matches!(e, SidecarError::Http { status: s, .. } if s == status));
        }
    }

    #[tokio::test]
    async fn client_times_out_against_a_black_hole() {
        // A hung/unroutable sidecar must return an error rather than wedging the
        // daemon (regression: the builder previously had no timeout at all).
        // 203.0.113.0/24 (TEST-NET-3) is non-routable, so a connect never
        // completes; the client must give up within its configured timeout.
        // Uses with_timeout (not global env) so parallel tests are not raced.
        let c = SidecarClient::with_timeout(
            SidecarConfig::new("http://203.0.113.1:9".into(), "m".into()),
            1,
        )
        .expect("client builds");
        let start = std::time::Instant::now();
        let res = c.tokenize("x").await;
        assert!(res.is_err(), "black-hole sidecar must error, not hang");
        assert!(
            start.elapsed() < std::time::Duration::from_secs(10),
            "must respect the configured timeout, took {:?}",
            start.elapsed()
        );
    }
}
