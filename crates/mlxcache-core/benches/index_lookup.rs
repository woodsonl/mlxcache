//! B1 benchmark: the prefix index's longest-prefix lookup — the daemon's
//! first hot-path data structure, hit on EVERY request before any adapter
//! call. Regression gate: lookup must stay O(prefix length) and scale with
//! the MATCHED prefix, not the index size or the request length beyond it.
//!
//! Run: cargo bench -p mlxcache-core

use criterion::{criterion_group, criterion_main, Criterion};
use mlxcache_core::contract::{CheckpointMeta, CheckpointState, ModelFingerprint};
use mlxcache_core::index::PrefixIndex;
use std::hint::black_box;

fn fingerprint() -> ModelFingerprint {
    ModelFingerprint {
        model_id: "bench-model".into(),
        tokenizer_hash: "bench".into(),
        kv_dtype: "float16".into(),
        kv_layout_version: 1,
        ..Default::default()
    }
}

/// Publish `count` checkpoints with distinct 8-token bodies (simulating an
/// index holding many unrelated conversations), plus one DEEP chain of
/// `chain_len` nested checkpoints (simulating a growing conversation).
fn build_index(count: usize, chain_len: usize) -> PrefixIndex {
    let index = PrefixIndex::new();
    let fp = fingerprint();
    for i in 0..count {
        let tokens: Vec<u32> = (0..8).map(|t| 0x7000_0000 + (i as u32) * 10 + t).collect();
        let meta = CheckpointMeta {
            fingerprint: fp.clone(),
            token_count: tokens.len() as u64,
            tokens: tokens.clone(),
            format_version: 1,
        };
        index.publish(
            &tokens,
            meta,
            format!("blob-{i}"),
            index.reserve_generation(),
            |_| {},
        );
    }
    // The nested chain: checkpoint k covers the first k+2 tokens of `base`,
    // so a lookup of the full base matches the deepest entry.
    let base: Vec<u32> = (0..chain_len as u32 + 2).collect();
    for k in 0..chain_len {
        let tokens = base[..k + 2].to_vec();
        let meta = CheckpointMeta {
            fingerprint: fp.clone(),
            token_count: tokens.len() as u64,
            tokens: tokens.clone(),
            format_version: 1,
        };
        index.publish(
            &tokens,
            meta,
            format!("chain-{k}"),
            index.reserve_generation(),
            |_| {},
        );
    }
    index
}

fn bench_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("prefix_index");
    // Tight budgets: this is a REGRESSION gate, not a publication dataset.
    group.warm_up_time(std::time::Duration::from_millis(300));
    group.measurement_time(std::time::Duration::from_millis(700));
    // Index shapes: 1k and 10k unrelated conversations, all with the same
    // 128-token query — lookup cost must NOT grow with index size.
    for count in [1_000usize, 10_000] {
        let index = build_index(count, 64);
        let query: Vec<u32> = (0..128).collect();
        group.bench_function(format!("lookup_128tok_index_{count}"), |b| {
            b.iter(|| {
                let (entry, matched) = index.lookup(black_box(&query)).expect("chain hit");
                black_box((entry.blob_path, matched))
            })
        });
    }
    // Match-length scaling: the SAME index, queries matched at 8 vs 64 tokens.
    let index = build_index(1_000, 64);
    for n in [8usize, 64] {
        let query: Vec<u32> = (0..n as u32).collect();
        group.bench_function(format!("lookup_match_{n}_tokens"), |b| {
            b.iter(|| {
                let (entry, matched) = index.lookup(black_box(&query)).expect("hit");
                black_box((entry.blob_path, matched))
            })
        });
    }
    // Full miss on a 128-token query (no shared prefix with anything).
    let miss_query: Vec<u32> = (0..128).map(|t| 0x7F00_0000 + t).collect();
    group.bench_function("lookup_128tok_full_miss", |b| {
        b.iter(|| black_box(index.lookup(black_box(&miss_query)).is_none()))
    });
    group.finish();
}

fn bench_publish(c: &mut Criterion) {
    let mut group = c.benchmark_group("prefix_index");
    group.warm_up_time(std::time::Duration::from_millis(300));
    group.measurement_time(std::time::Duration::from_millis(700));
    group.bench_function("publish_2048tok_checkpoint", |b| {
        b.iter_batched(
            PrefixIndex::new,
            |index| {
                let fp = fingerprint();
                let tokens: Vec<u32> = (0..2048).collect();
                let meta = CheckpointMeta {
                    fingerprint: fp,
                    token_count: tokens.len() as u64,
                    tokens: tokens.clone(),
                    format_version: 1,
                };
                assert!(index.publish(&tokens, meta, "b".into(), 1, |_| {}));
                assert_eq!(
                    index.lookup(&tokens).map(|(_, m)| m),
                    Some(2048),
                    "published entry must be immediately visible"
                );
                black_box(index);
            },
            criterion::BatchSize::PerIteration,
        );
    });
    group.finish();
}

/// Keep CheckpointState referenced so the import earns its place on all
/// toolchains (entry.state comparisons happen inside the index).
fn _state_is_used(_s: CheckpointState) {}

criterion_group!(benches, bench_lookup, bench_publish);
criterion_main!(benches);
