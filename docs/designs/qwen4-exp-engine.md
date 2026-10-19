# Qwen3.8-Flash-Next (`qwen4_exp`) engine

Status: implemented. The batches below (B2-B7) are built and measured; this
doc is the design record and the boundary statement for what is not loaded
(the bundled MTP head and vision tower).

## Scope

mlxcache serves `nvidia/Qwen3.8-Flash-Next-NVFP4` text input end to end on one
Apple Silicon machine: prefill, generation, persistent KV checkpoints, and
delta resume. No dependency on any other repository: the weight reader, the
architecture, and the n-gram embedding table all live here.

Scope boundary: the checkpoint also bundles a multi-token-prediction head
(`mtp.*`, 3,101 tensors) and a vision tower (`model.visual.*`, 333 tensors).
Neither contributes cache state — `weight_shapes()` enumerates 74,926 text
tensors and zero of either family — so neither is required to cache this model
correctly, and neither is implemented on this branch. The cache composition
(`make_cache()`) lives in the model, so a checkpoint that serves images
composes its own cache here with no change to the daemon, the wire format, or
the key. Adding either back is a feature decision with its own justification,
not a caching prerequisite.

Reference implementation: `transformers` `qwen4_exp` (Apache-2.0), read as the
authority for forward semantics. MLX has no implementation; this port is
line-by-line faithful at the tensor-op level and is verified against the
reference numerically where a comparison is possible.

## The checkpoint

`config.json` is a `Qwen4ExpConfig` wrapper; the language model is under
`text_config` (`Qwen4ExpTextConfig`, `model_type: qwen4_exp_text`). The
wrapper's `vision_config` block is not read.

Stored bytes, measured by `sidecar/probes/qwen4_inventory.py`: 132.6 GB.

| family | bytes | notes |
|---|---|---|
| experts (48 layers x 512) | 70.5 GB | NVFP4 packed, U8 + per-expert scales |
| PLE n-gram table (layer 1) | 51.2 GB | fp8_e4m3, 320M rows x 160, sharded |
| linear attention | 4.2 GB | GDN projections |
| other (norms, hyper-connections) | 2.4 GB | |
| MTP head | 2.7 GB | fc_embedding + hyper-connections + 1 layer |
| vision tower | ~0.9 GB | |
| embed_tokens + lm_head | 2.5 GB | bf16 |
| self attention | 1.3 GB | |

Residency on a 48 GB machine is the hard constraint and is why the loader is
lazy rather than eager: the resident set is the non-expert weights plus the
touched rows of the PLE table plus an LRU window of experts.

### NVFP4 expert weights

ModelOpt layout, per linear inside each expert:

- `weight` — U8, two fp4 values per byte, `[out, in/2]`.
- `weight_scale_2` — F32 scalar, the per-tensor global scale.
- `input_scale` — F32 scalar, the per-tensor input scale.
- `mlp.switch_mlp.{gate_proj,up_proj,down_proj}.global_scale` — F32 `[512]`,
  one per expert, in `nvfp4_global_scales.safetensors`.

Dequantization is handled by MLX itself: `mx.dequantize(w, scales, biases=None,
group_size=16, bits=4, mode="nvfp4", global_scale=...)`.

Three details are not obvious from the API and each silently produces wrong
weights, so they are recorded here with the measurement that settled them.

- **The packed matrix must be viewed as `uint32`, not `uint8`.** MLX rejects a
  `uint8` matrix for `nvfp4` ("The matrix should be given as a uint32"): each
  `uint32` word holds eight 4-bit values. From the stored `[out, in/2]` uint8
  tensor the view is `[out, in/8]` uint32.
- **The scales must stay `uint8` (raw e4m3 bytes).** Casting `weight_scale` to
  float32 is rejected ("Scale type must be uint8"): `nvfp4` decodes the scale
  bytes itself.
- **`global_scale` is the per-expert `global_scale`, not `weight_scale_2`.**
  `weight_scale_2` equals `global_scale / (6 * 448)`, so passing it directly
  makes the weights 2688x too small; omitting `global_scale` entirely makes
  them 20551x too large. Measured on layer 0 expert 0 `gate_proj`, comparing
  the RMS against the same layer's bf16 `shared_expert.gate_proj`:

  | `global_scale` | RMS | ratio to the bf16 shared expert |
  |---|---|---|
  | omitted | 204.0 | 20551 |
  | `weight_scale_2` | 5e-6 | 0.001 |
  | per-expert `global_scale` | 0.013489 | 1.359 |

  The per-expert value is also exactly reproducible: `6 * 448 *
  weight_scale_2` equals it for `gate_proj`, `up_proj` and `down_proj` (ratio
  1.0000 for each), so the two representations agree rather than one being
  unknown. The residual 1.359x against the shared expert is an expected
  property of routed versus shared experts, not a scaling error.


### PLE n-gram embedding (fp8, sharded)

`layer_multipliers`, `ngram_heads_offsets`, `ngram_heads_vocab_sizes` are
buffers in the shards. The embedding itself is stored as 128 shards of
`F8_E4M3` `[2500012, 160]` in `model-fp8-mtp-ple.safetensors`
(320,001,536 rows total, `row_width: 160`), with a bf16 per-row scale
(`ngram_embedding.weight_scale`, `scale_bf16_bits`). `ple_fp8_manifest.json`
records each shard's byte offset, so rows are addressable without loading a
shard: row `r` lives at `shard.offset + (r - shard.row_start) * 160`.

Access is a hash lookup, so the touch set per token is small (24 head lookups
per PLE layer). The loader keeps an LRU of decoded rows, not the table.

## Batch contracts

### B2 — architecture (`sidecar/mlxcache_qwen4/`)

`model.py` builds the text stack from a `Qwen4TextConfig` dataclass parsed out
of the checkpoint's `text_config`, and exposes:

```python
class Qwen4TextModel(nn.Module):
    def __init__(self, config: Qwen4TextConfig, weights: WeightSource): ...
    def make_cache(self) -> list          # mlx-lm cache-compatible entries
```

`WeightSource` is the ONLY coupling to the loader (B3):

```python
class WeightSource(Protocol):
    def get(self, name: str) -> mx.array          # dequantized, HF-style name
    def get_slice(self, name: str, rows: slice) -> mx.array
```

B2 is testable with a tiny random `WeightSource` over a small config, with no
checkpoint on disk. Names are HF-style (`model.language_model.layers.N....`),
stripped to `layers.N....` by the loader.

Cache entries, per the reference: linear-attention layers carry conv state
(3 tensors), a recurrent GDN state, and the PLE token context; full-attention
layers carry token KV plus the QSA indexer key blocks. All entries are
mlx-lm cache classes so the engine-generic save/load path applies.

### B3 — loader (`sidecar/mlxcache_qwen4/loader.py`)

Reads the checkpoint directory, resolves symlinked shards, and implements
`WeightSource`. Dequantizes NVFP4 experts lazily under a residency policy
(`MLXCACHE_Q4_EXPERT_RESIDENT_N`), reads fp8 PLE rows through the manifest with
an LRU row cache (`MLXCACHE_Q4_PLE_CACHE_MB`), and streams bf16 tensors.

### B4 — adapter (`sidecar/mlxcache_sidecar/engine_qwen4.py`)

Registers `MLXCACHE_ENGINE=qwen4-exp`. Mirrors `MlxLmEngine`'s surface exactly
(`tokenize`, `prefill`, `generate_with_text`, `stream`, `prepare_stream`,
`stream_prepared`, `kv_dtype`, `tokenizer_hash`, `kv_bits`, `kv_group_size`,
`engine_id`, `cache_layout`) so the daemon is unchanged apart from the
engine identity it now fingerprints.

### B5 — performance

TTFT and generation throughput on the real checkpoint, recorded in the
README's Measured section, with the resume budget held at 2 s for 20K-token
contexts on the resident configuration.

### B6 — settle

Checkpoint the post-generation state so a follow-up turn resumes with only its
new input tokens as the delta. Engine-generic; applies to mlx-lm too.

### B7 — docs and ops

README supported models, residency knobs and their tradeoffs in `ops.md`,
CHANGELOG, CI wiring for the hermetic suites.

## Cache identity (B1)

A checkpoint is only valid for the engine and cache composition that produced
it. `engine_id` namespaces keys (connector protocol section 4) and is now
reported by every adapter; `cache_layout` is a signature of the model's cache
composition, so a dense model and a hybrid model sharing a tokenizer can never
exchange blobs.

`cache_layout` is `mlx-lm`'s own class layout signature: the ordered tuple of
cache entry class names, folded into a short hash. It is computed by building
a prompt cache (which allocates nothing) and reading the entry classes, so it
is exact by construction rather than declared per architecture.
