//! B1 benchmark: the daemon's request-side hot-path helpers — token-prefix
//! hashing (per request, for the log line), blob-key derivation (per
//! miss/partial publish), and the atomic stats record (per request).
//!
//! These pin the P3 micro-opts: prefix hashing stays allocation-free,
//! stats stay lock-free, and any regression that reintroduces an O(n) copy or
//! a Mutex shows up here as a wall-clock jump.
//!
//! Run: cargo bench -p mlxcache-daemon

use criterion::{criterion_group, criterion_main, Criterion};
use mlxcache_core::policy::{CacheVerdict, PolicyDecision};
use mlxcache_daemon::http::Stats;
use mlxcache_daemon::http::{blob_key, prefix_hash};
use std::hint::black_box;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

fn bench_prefix_hash(c: &mut Criterion) {
    let mut group = c.benchmark_group("prefix_hash");
    for n in [128usize, 2048, 32_768] {
        let tokens: Vec<u32> = (0..n as u32).collect();
        group.bench_function(format!("hash_{n}_tokens"), |b| {
            b.iter(|| black_box(prefix_hash(black_box(&tokens))))
        });
    }
    group.finish();
}

fn bench_blob_key(c: &mut Criterion) {
    let fingerprint = mlxcache_core::contract::ModelFingerprint {
        model_id: "bench-model".into(),
        tokenizer_hash: "0123456789abcdef".into(),
        kv_dtype: "float16".into(),
        kv_layout_version: 1,
    };
    let mut group = c.benchmark_group("blob_key");
    for n in [128usize, 2048, 32_768] {
        let tokens: Vec<u32> = (0..n as u32).collect();
        group.bench_function(format!("key_{n}_tokens"), |b| {
            b.iter(|| black_box(blob_key(black_box(&fingerprint), black_box(&tokens))))
        });
    }
    group.finish();
}

fn bench_stats_record(c: &mut Criterion) {
    let stats = Arc::new(Stats::default());
    let hit = PolicyDecision {
        verdict: CacheVerdict::Hit,
        matched_tokens: 2048,
        request_tokens: 2048,
    };
    let miss = PolicyDecision {
        verdict: CacheVerdict::Miss,
        matched_tokens: 0,
        request_tokens: 128,
    };
    let mut group = c.benchmark_group("stats");
    group.bench_function("record_hit", |b| {
        b.iter(|| stats.record(black_box(&hit)))
    });
    group.bench_function("record_miss", |b| {
        b.iter(|| stats.record(black_box(&miss)))
    });
    // Contended record: 4 threads hammering the counters. The atomic
    // implementation should stay within a small multiple of the single-thread
    // cost; a Mutex would show a superlinear degradation here.
    group.bench_function("record_hit_contended_4t", |b| {
        b.iter(|| {
            let barrier = Arc::new(std::sync::Barrier::new(5));
            let stop = Arc::new(AtomicU64::new(0));
            let mut handles = Vec::new();
            for _ in 0..4 {
                let stats = stats.clone();
                let barrier = barrier.clone();
                let stop = stop.clone();
                let hit = hit.clone();
                handles.push(std::thread::spawn(move || {
                    barrier.wait();
                    while stop.load(Ordering::Relaxed) == 0 {
                        stats.record(&hit);
                    }
                }));
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
            stop.store(1, Ordering::Relaxed);
            for h in handles {
                h.join().unwrap();
            }
            black_box(stats.snapshot().hits)
        })
    });
    group.finish();
}

criterion_group!(benches, bench_prefix_hash, bench_blob_key, bench_stats_record);
criterion_main!(benches);
