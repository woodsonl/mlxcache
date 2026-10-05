//! mlxcache-daemon binary entry point.

use anyhow::Result;
use std::sync::Arc;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<()> {
    // Non-blocking logging: log writes go to a bounded channel drained by a
    // background thread, so a stalled log sink can never block a request, the
    // shutdown future, or any synchronous tracing event on the exit path. Keep
    // the guard alive for the process lifetime; dropping it flushes the worker.
    let (log_writer, _log_guard) = tracing_appender::non_blocking(std::io::stdout());
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .json()
        .with_writer(log_writer)
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
        // A daemon with no served models rejects every completion request with
        // 404 while /stats stays 200 — a misconfigured deploy looks healthy
        // (QA ISSUE-001). Refuse at boot unless the operator explicitly opted
        // into a model-less run (dev/test harnesses that only exercise /stats
        // or the synthetic stack).
        if std::env::var("MLXCACHE_ALLOW_NO_MODELS").ok().as_deref() != Some("1") {
            anyhow::bail!(
                "MLXCACHE_MODELS is empty: this daemon would 404 every request \
                 while appearing healthy. Set MLXCACHE_MODELS (comma-separated \
                 model ids), or set MLXCACHE_ALLOW_NO_MODELS=1 to run model-less \
                 on purpose."
            );
        }
        tracing::warn!("MLXCACHE_MODELS is empty (allowed by MLXCACHE_ALLOW_NO_MODELS=1): every request will 404");
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
        trace: mlxcache_daemon::trace::TraceWriter::from_env()
            .map_err(|e| anyhow::anyhow!("{e}"))?,
    });
    // Warm the native-tokenizer OnceLock NOW, not on the first request: a
    // bad MLXCACHE_NATIVE_TOKENIZER must kill the boot (the module's
    // fail-closed contract), not panic mid-traffic inside get_or_init —
    // there it surfaces as a dropped connection on every request until
    // restart. When unset, this is a no-op.
    let _ = mlxcache_daemon::native_tokenizer::from_env();
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
    spawn_reaper(state.clone());
    let app = mlxcache_daemon::http::router(state.clone());
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
    // Graceful-shutdown trace safety net: records are queued on a bounded
    // channel and drained by the writer thread, and every record is already
    // flushed per-write on capture, so a queued record is at most one write
    // away from disk. This explicit flush is therefore a near-no-op — it only
    // matters for a record still in flight inside the writer between channel
    // receipt and file write when the server finished draining.
    if let Some(tracer) = &state.trace {
        tracer.flush();
    }
    tracing::info!("mlxcache daemon stopped");
    Ok(())
}

/// Background eviction reaper (T-eviction). Every `interval_s` seconds, evict
/// cold checkpoints until at most `max_entries` published blobs remain. Scoring
/// and anchor protection live in `Orchestrator::evict_cold` (ds4 anchor policy:
/// recently-served checkpoints and live chain bases are never evicted).
///
/// The reaper reads its knobs through [`EvictConfig::from_env`] (see
/// orchestrator.rs for the knob semantics and their tests).
fn spawn_reaper(state: Arc<mlxcache_daemon::http::AppState>) {
    let cfg = mlxcache_daemon::orchestrator::EvictConfig::from_env();
    if !cfg.enabled() {
        tracing::info!(
            interval_s = cfg.interval_s,
            entry_cap = ?cfg.entry_cap,
            max_bytes = cfg.max_bytes,
            "eviction reaper disabled (set MLXCACHE_EVICT_INTERVAL_S and a cap to enable; blob growth is unbounded)"
        );
        return;
    }
    tracing::info!(
        interval_s = cfg.interval_s,
        entry_cap = ?cfg.entry_cap,
        max_bytes = cfg.max_bytes,
        "eviction reaper armed (byte budget default 32 GiB; MLXCACHE_EVICT_MAX_BYTES=0 to disable)"
    );
    let anchor_window = std::time::Duration::from_secs(cfg.anchor_window_s);
    let (interval_s, entry_cap, max_bytes) = (cfg.interval_s, cfg.entry_cap, cfg.max_bytes);
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(std::time::Duration::from_secs(interval_s));
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        ticker.tick().await; // interval fires immediately; skip the no-op tick
        loop {
            ticker.tick().await;
            // A3: the sweep takes the index read lock and builds a candidate
            // snapshot over a trie that can hold ~20K entries — ms-scale
            // blocking work that must not stall other tasks scheduled on this
            // runtime worker, including live streams. The Arc is cloned per
            // pass so the blocking closure can own it.
            let state = state.clone();
            match tokio::task::spawn_blocking(move || {
                state.evict_pass(entry_cap, max_bytes, anchor_window)
            })
            .await
            {
                Ok(_) => {}
                // A silently-dead reaper repeats every interval with zero
                // signal — log it like the publish path logs a panicked task.
                Err(join) => {
                    tracing::error!(error = %join, "eviction pass panicked; sweep skipped this interval")
                }
            }
        }
    });
}
// (The armed log lives at the top of spawn_reaper; this trailing block was a
// duplicate from the knob refactor.)

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
