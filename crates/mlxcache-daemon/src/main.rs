//! mlxcache-daemon binary entry point.

use anyhow::Result;
use std::sync::Arc;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .json()
        .init();

    let served_models: Vec<String> = std::env::var("MLXCACHE_MODELS")
        .unwrap_or_else(|_| String::new())
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|s| s.trim().to_string())
        .collect();
    // Treat an empty MLXCACHE_SIDECAR_URL as unset: an empty value would build
    // a client with malformed relative URLs. Common launchd misconfiguration.
    let sidecar_base = std::env::var("MLXCACHE_SIDECAR_URL")
        .ok()
        .filter(|u| !u.trim().is_empty());
    let sidecar = match sidecar_base {
        Some(url) => {
            let model = served_models.first().cloned().unwrap_or_default();
            Some(
                mlxcache_daemon::sidecar::SidecarClient::new(
                    mlxcache_daemon::sidecar::SidecarConfig::new(url, model),
                )
                .map_err(|e| anyhow::anyhow!("sidecar client init failed: {e}"))?,
            )
        }
        None => None,
    };
    if served_models.is_empty() {
        tracing::warn!("MLXCACHE_MODELS is empty: every request will 404");
    }
    // Empty values fall back to the default rather than failing on an empty path.
    let blob_dir = std::env::var("MLXCACHE_BLOBS")
        .ok()
        .filter(|d| !d.trim().is_empty())
        .unwrap_or_else(|| "/tmp/mlxcache-blobs".into());
    let persistence = mlxcache_daemon::persistence::Persistence::new(blob_dir).map_err(|e| {
        // adapter#load rescue row: refuse at startup with a clear message.
        anyhow::anyhow!("blob dir init failed: {e}")
    })?;
    let state = Arc::new(mlxcache_daemon::http::AppState {
        orchestrator: mlxcache_daemon::orchestrator::Orchestrator::new(),
        singleflight: mlxcache_core::singleflight::SingleFlight::new(),
        stats: Arc::new(mlxcache_daemon::http::Stats::default()),
        served_models,
        sidecar,
        persistence,
    });
    // R1-4: rebuild the index from persisted checkpoints so a daemon restart
    // resumes from disk instead of re-prefilling everything.
    let report = state.orchestrator.rebuild_from_disk(&state.persistence);
    if report.rebuilt > 0 || report.skipped > 0 || !report.errors.is_empty() {
        tracing::info!(
            rebuilt = report.rebuilt,
            skipped = report.skipped,
            errors = report.errors.len(),
            "index rebuilt from persisted checkpoints"
        );
    }
    let app = mlxcache_daemon::http::router(state);
    let addr = std::env::var("MLXCACHE_ADDR").unwrap_or_else(|_| "127.0.0.1:8420".into());
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!(addr = %addr, "mlxcache daemon listening");
    axum::serve(listener, app).await?;
    Ok(())
}
