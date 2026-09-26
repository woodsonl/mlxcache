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
    // Drain in-flight requests on SIGINT/SIGTERM, bounded by
    // MLXCACHE_SHUTDOWN_GRACE_S (default 30s). The signal is received and the
    // deadline enforced on a dedicated OS thread, NOT the async runtime: a
    // blocked checkpoint write, a stalled log collector, or runtime teardown can
    // stall the async side, which would let the process outlive a supervisor's
    // ~90s SIGKILL grace. The watchdog exits with libc::_exit (no logging, no
    // allocator) and is armed for the whole lifetime: if the process has not
    // exited within the grace of the signal, it is forced. Publishes are atomic,
    // so a forced exit loses nothing.
    let grace_s: u64 = std::env::var("MLXCACHE_SHUTDOWN_GRACE_S")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(30);
    // Install handlers BEFORE announcing readiness: if this fails the daemon
    // cannot be shut down cleanly, so refuse to start (nonzero exit) rather than
    // run with no way to stop it.
    let (signal_tx, signal_rx) = tokio::sync::oneshot::channel::<()>();
    spawn_signal_watchdog(grace_s, signal_tx)?;
    tracing::info!(addr = %addr, "mlxcache daemon listening");
    let serve = axum::serve(listener, app).with_graceful_shutdown(async move {
        // Resolve only when the watchdog observed a real signal. A dropped
        // sender means setup failed, which we have already treated as fatal.
        // No logging here: a blocked log sink would stop this future from
        // resolving, leaving axum accepting connections during the block.
        let _ = signal_rx.await;
    });
    serve.await?;
    tracing::info!("mlxcache daemon stopped");
    Ok(())
}

/// Receive SIGINT/SIGTERM and enforce the shutdown deadline off the async
/// runtime. On the first signal, notify the async drain via `signal_tx`, then
/// force-exit with `libc::_exit` after `grace_s` if the process is still alive.
/// The watchdog is never disarmed: a blocked final log or runtime teardown
/// cannot defeat the deadline. `libc::_exit` is async-signal-safe (no logging,
/// no allocation). Returns an error if handlers cannot be installed.
fn spawn_signal_watchdog(
    grace_s: u64,
    signal_tx: tokio::sync::oneshot::Sender<()>,
) -> anyhow::Result<()> {
    #[cfg(unix)]
    {
        let mut signals = signal_hook::iterator::Signals::new([
            signal_hook::consts::SIGINT,
            signal_hook::consts::SIGTERM,
        ])
        .map_err(|e| anyhow::anyhow!("install signal handlers: {e}"))?;
        std::thread::spawn(move || {
            if signals.forever().next().is_none() {
                return;
            }
            // Wake the async graceful-shutdown future.
            let _ = signal_tx.send(());
            std::thread::sleep(std::time::Duration::from_secs(grace_s));
            // The process has not exited within the grace: force it. Never
            // touch logging or the allocator here.
            unsafe { libc::_exit(0) };
        });
    }
    #[cfg(not(unix))]
    {
        let _ = (grace_s, signal_tx);
    }
    Ok(())
}
