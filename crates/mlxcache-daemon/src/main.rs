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
    // Drain in-flight requests on SIGINT/SIGTERM instead of dropping them.
    // Bounded: axum waits for active responses, which for a streaming (SSE)
    // request means its upstream sidecar stream ends — up to the sidecar client
    // timeout (120s default), longer than a supervisor's ~90s SIGKILL grace. So
    // cap the drain at MLXCACHE_SHUTDOWN_GRACE_S (default 30s): once it fires, a
    // held-open stream cannot delay exit past the supervisor's patience.
    // Publishes are atomic, so a forced exit loses nothing.
    let grace_s: u64 = std::env::var("MLXCACHE_SHUTDOWN_GRACE_S")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(30);
    let (signal_tx, signal_rx) = tokio::sync::oneshot::channel();
    let serve = axum::serve(listener, app).with_graceful_shutdown(async move {
        shutdown_signal().await;
        let _ = signal_tx.send(());
    });
    tokio::select! {
        result = serve => result?,
        () = async {
            let _ = signal_rx.await;
            tracing::info!(grace_s, "draining in-flight requests");
            tokio::time::sleep(std::time::Duration::from_secs(grace_s)).await;
        } => {
            tracing::warn!(grace_s, "shutdown grace elapsed; forcing exit");
        }
    }
    tracing::info!("mlxcache daemon stopped");
    Ok(())
}

/// Resolve on SIGINT or SIGTERM. Logs which signal so operators can tell an
/// intentional stop from a crash. A failed handler install is fatal-logged and
/// treated as a shutdown trigger rather than a panic.
async fn shutdown_signal() {
    let ctrl_c = async {
        match tokio::signal::ctrl_c().await {
            Ok(()) => tracing::info!("received SIGINT; draining"),
            Err(e) => tracing::error!(error = %e, "SIGINT handler failed; draining"),
        }
    };

    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut sig) => {
                sig.recv().await;
                tracing::info!("received SIGTERM; draining");
            }
            Err(e) => tracing::error!(error = %e, "SIGTERM handler failed; draining"),
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => {},
        () = terminate => {},
    }
}
