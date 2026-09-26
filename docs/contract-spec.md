# mlxcache Cache Contract Spec

Version: 1 (draft 1)
Status: BINDING for v1 adapters
Source of truth: docs/designs/mlx-kv-cache-daemon.md (eng review decisions R1-R7)

## Glossary (pinned — no synonyms allowed anywhere in this repo)

| Term | Meaning |
|---|---|
| **contract** | The engine-agnostic interface + rules defined in this document. |
| **adapter** | A per-engine implementation of the contract (Rust native for the owner engine; Python sidecar for mlx-lm). |
| **checkpoint** | A serialized KV blob for one token prefix, plus its metadata header. |
| **prefix** | A token-ID sequence identifying a checkpoint. |
| **fingerprint** | The ModelFingerprint struct pinning model identity, tokenizer, dtype, layout. |
| **sidecar** | The Python mlx-lm compatibility adapter process. Never the hot-path default. |
| **daemon** | The mlxcache Rust binary orchestrating proxy, policy, index, adapters. |

## Binding rules (verbatim from the design doc)

- **R1-1 (mismatch → miss):** on any fingerprint/tokenizer/dtype mismatch, the checkpoint is treated as a miss and the request falls through to normal prefill. Never serve incompatible KV. Mismatched blobs are quarantined, not deleted, for diagnostics.
- **R1-2 (tokenization ownership):** the engine's adapter tokenizes and reports token IDs; the daemon keys checkpoints by the adapter-reported token-ID prefix and pins the tokenizer artifact hash in checkpoint metadata. The daemon never re-tokenizes.
- **R1-3 (single-flight + atomic publish):** single-flight on identical uncached prefixes — a concurrent request for the same prefix blocks on the in-flight prefill and is served from its result. Checkpoint blob push/pull is serialized per-blob with an atomic publish (write-temp-then-rename). Index rows become visible only AFTER the rename succeeds.
- **R1-4 (restart semantics):** streaming connections drop on daemon restart; clients retry (standard HTTP semantics). Persisted checkpoints survive; in-flight generation state does not.

## Interface contract (language-agnostic)

The adapter interface, expressed as a Rust trait in mlxcache-core (`contract` module conceptually; native trait in code):

```
trait CacheAdapter {
    /// Tokenize a prompt. The daemon NEVER tokenizes (R1-2).
    fn tokenize(&self, prompt: &str) -> Result<Vec<u32>, AdapterError>;

    /// Model fingerprint for the loaded model (R1-1 matching key).
    fn fingerprint(&self) -> Result<ModelFingerprint, AdapterError>;

    /// Prefill `tokens[prefix_len..]` on top of adopted KV state, then
    /// generate. Streams tokens; single token latency is the hot path.
    fn generate_stream(&self, tokens: &[u32], prefix_len: usize, params: GenParams)
        -> Result<TokenStream, AdapterError>;

    /// Serialize current KV state for `tokens[..n]` into a checkpoint blob.
    /// MUST be zero-copy capture: reference live tensors, serialize only at
    /// publish time, off the streaming path (R7-spec-perf).
    fn capture_checkpoint(&self, tokens: &[u32]) -> Result<CheckpointBlob, AdapterError>;

    /// Load a checkpoint blob into the engine's KV state. Returns an error
    /// (not a fallback) on any incompatibility — the daemon decides miss/quarantine.
    fn adopt_checkpoint(&self, blob: &CheckpointBlob) -> Result<(), AdapterError>;
}
```

Wire format for non-Rust adapters (sidecar): HTTP + JSON envelopes over localhost; KV payloads ride as binary frames (length-prefixed). The sidecar IPC tax is measured by T2 and must never be on the default path once the owner engine ships.

## Checkpoint format

```
CheckpointBlob = {
  header: CheckpointMeta (JSON, length-prefixed),
  tensors: safetensors frame (kv dtype/layout per fingerprint),
}
CheckpointMeta = {
  fingerprint: { model_id, tokenizer_hash, kv_dtype, kv_layout_version },
  token_count: u64,
  tokens: [u32],         // the token-ID prefix this blob covers; makes the blob
                         // self-describing so the index is rebuildable after a
                         // daemon restart (R1-4). Empty prefix => not indexable.
  format_version: u32,   // bump on any layout change; R1-1 quarantines old versions
}
```

## Prefix index

- Structure: **radix tree over token IDs** (R7-spec-perf). O(prefix length) lookup; shared-prefix memory efficiency; matches mlx-lm LRUPromptCache trie semantics.
- Node states: absent → in-flight (single-flight lock) → published (post-rename) → quarantined.
- Invalid transition: published → in-flight. Prevented by index check before prefill.

## TTFT budget decomposition (R7-spec-perf)

`TTFT = t_index (µs) + t_checkpoint_load (ms, GB/s-bound) + t_delta_prefill (compute-bound)`

The R1-5 measurement gate records all three components separately so a budget miss is diagnosable.

## Eviction policy (v1)

- Anchors (checkpoints a live session can extend) are never evicted (ds4 lesson).
- Cold, non-extendable blobs evicted first, larger blobs first among equals.
- A/B against frequency/size-aware policies deferred to real traces (open question in design doc).

## Measurement gate (R1-5, unchanged)

Before the disk tier ships: bytes/token, serialize + deserialize wall time, peak memory (added by CEO review Section 7), TTFT decomposition — for one representative model (e.g. Qwen3-32B-4bit) at 50K tokens. If the round-trip fails the 2s budget, v1 ships memory-resident only.
