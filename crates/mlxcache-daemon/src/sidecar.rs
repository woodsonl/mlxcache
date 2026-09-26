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
}

#[derive(Debug, thiserror::Error)]
pub enum SidecarError {
    #[error("sidecar unreachable at {url}: {source}")]
    Unreachable { url: String, source: reqwest::Error },
    #[error("sidecar returned {status}: {body}")]
    Http { status: u16, body: String },
    #[error("sidecar protocol violation: {0}")]
    Protocol(String),
}

/// Minimal async HTTP client for the sidecar. reqwest keeps the connection
/// pool warm; JSON bodies only (KV blobs stream through a separate path).
pub struct SidecarClient {
    config: SidecarConfig,
    http: reqwest::Client,
}

impl SidecarClient {
    pub fn new(config: SidecarConfig) -> Result<Self, SidecarError> {
        let http = reqwest::Client::builder()
            .build()
            .map_err(|_e| SidecarError::Protocol("client build failed".into()))?;
        Ok(Self { config, http })
    }

    pub async fn prefill(&self, tokens: &[u32]) -> Result<Vec<u8>, SidecarError> {
        let url = format!("{}/prefill", self.config.base_url);
        let resp = self
            .http
            .post(&url)
            .json(&serde_json::json!({ "tokens": tokens }))
            .send()
            .await
            .map_err(|e| SidecarError::Unreachable {
                url: url.clone(),
                source: e,
            })?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(SidecarError::Http {
                status: status.as_u16(),
                body,
            });
        }
        resp.bytes()
            .await
            .map(|b| b.to_vec())
            .map_err(|e| SidecarError::Protocol(format!("bad prefill response: {e}")))
    }

    pub async fn generate(
        &self,
        tokens: &[u32],
        prefill_from: usize,
        max_tokens: usize,
        blob_path: Option<&str>,
    ) -> Result<Vec<u32>, SidecarError> {
        let url = format!("{}/generate", self.config.base_url);
        let resp = self
            .http
            .post(&url)
            .json(&serde_json::json!({
                "tokens": tokens,
                "prefill_from": prefill_from,
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
            let body = resp.text().await.unwrap_or_default();
            return Err(SidecarError::Http {
                status: status.as_u16(),
                body,
            });
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
            let body = resp.text().await.unwrap_or_default();
            return Err(SidecarError::Http {
                status: status.as_u16(),
                body,
            });
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
}
