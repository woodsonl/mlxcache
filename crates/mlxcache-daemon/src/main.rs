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
    // Drain in-flight requests on SIGINT/SIGTERM, bounded by
    // MLXCACHE_SHUTDOWN_GRACE_S (default 30s). The signal is received and the
    // deadline enforced on a dedicated OS thread, NOT the async runtime: a
    // blocked checkpoint write or a stalled log collector can stall the runtime
    // (and its signal future), which would let the process outlive a supervisor's
    // ~90s SIGKILL grace. The watchdog exits with libc::_exit (no logging, no
    // allocator). Publishes are atomic, so a forced exit loses nothing.
    let grace_s: u64 = std::env::var("MLXCACHE_SHUTDOWN_GRACE_S")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(30);
    let drain_done = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let (signal_tx, signal_rx) = tokio::sync::oneshot::channel::<()>();
    spawn_signal_watchdog(grace_s, Arc::clone(&drain_done), signal_tx);
    let serve = axum::serve(listener, app).with_graceful_shutdown(async move {
        // Resolve when the watchdog has seen the signal.
        let _ = signal_rx.await;
        tracing::info!("draining in-flight requests");
    });
    serve.await?;
    // Tell the watchdog the drain finished so it exits instead of forcing.
    drain_done.store(true, std::sync::atomic::Ordering::SeqCst);
    tracing::info!("mlxcache daemon stopped");
    Ok(())
}

/// Receive SIGINT/SIGTERM and enforce the shutdown deadline off the async
/// runtime. On the first signal, notify the async drain via `signal_tx`, then
/// wait up to `grace_s` for `drain_done`; if the drain has not finished, force
/// exit with `libc::_exit` (async-signal-safe: no logging, no allocation).
fn spawn_signal_watchdog(
    grace_s: u64,
    drain_done: Arc<std::sync::atomic::AtomicBool>,
    signal_tx: tokio::sync::oneshot::Sender<()>,
) {
    use std::sync::atomic::Ordering;
    #[cfg(unix)]
    {
        let mut signals = match signal_hook::iterator::Signals::new([
            signal_hook::consts::SIGINT,
            signal_hook::consts::SIGTERM,
        ]) {
            Ok(s) => s,
            Err(e) => {
                tracing::error!(error = %e, "failed to install signal handlers");
                return;
            }
        };
        std::thread::spawn(move || {
            if signals.forever().next().is_none() {
                return;
            }
            // Wake the async graceful-shutdown future.
            let _ = signal_tx.send(());
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(grace_s);
            while !drain_done.load(Ordering::SeqCst) {
                if std::time::Instant::now() >= deadline {
                    // Force exit without touching logging or the allocator.
                    unsafe { libc::_exit(0) };
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
        });
    }
    #[cfg(not(unix))]
    {
        let _ = (grace_s, drain_done, signal_tx);
    }
}
