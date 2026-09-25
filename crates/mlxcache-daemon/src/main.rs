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
    let sidecar_base = std::env::var("MLXCACHE_SIDECAR_URL").ok();
    let sidecar = sidecar_base.and_then(|url| {
        let model = served_models.first().cloned().unwrap_or_default();
        mlxcache_daemon::sidecar::SidecarClient::new(
            mlxcache_daemon::sidecar::SidecarConfig::new(url, model),
        )
        .ok()
    });
    let state = Arc::new(mlxcache_daemon::http::AppState {
        orchestrator: mlxcache_daemon::orchestrator::Orchestrator::new(),
        singleflight: mlxcache_core::singleflight::SingleFlight::new(),
        stats: Arc::new(mlxcache_daemon::http::Stats::default()),
        served_models,
        sidecar,
    });
    let app = mlxcache_daemon::http::router(state);
    let addr = std::env::var("MLXCACHE_ADDR").unwrap_or_else(|_| "127.0.0.1:8420".into());
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!(addr = %addr, "mlxcache daemon listening");
    axum::serve(listener, app).await?;
    Ok(())
}
