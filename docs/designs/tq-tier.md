# The named KV tiers (TurboQuant persistence codecs)

Status: SHIPPED (opt-in). Implemented from arXiv:2504.19874 only — no code
from any third-party TurboQuant repository (GPL-3.0; this repo carries no
license and must stay uncontaminated).

## What the tier is

A named persistence tier (`MLXCACHE_KV_TIER` on the sidecar, `--kv-tier` on
the wrapper) compresses the KV checkpoint at capture time and rebuilds a
stock `KVCache` at adopt time. The engine never computes attention from
compressed state — this is the paper's storage-side algorithm, not its
live-compute path:

- **Keys**: per-(token,head) L2 norm in f32 (f16 storage would overflow on
  near-ceiling-magnitude keys) + a pinned random orthogonal
  rotation + a Lloyd-Max codebook on the rotated unit-vector coordinates
  (the paper's §3.1, solved for the exact Beta marginal; anchors match the
  paper: b=2 centroids ±0.453/±1.51, Theorem-1 MSE 0.36/0.117/0.03/0.009),
  bit-packed ⌊32/b⌋ indices per u32 word.
- **Values**: stock `mx.quantize` group quantization (the same mechanism
  `QuantizedKVCache` uses).
- **QJL residual sign bits: not included.** They exist to score attention
  directly from compressed keys; adopt always dequantizes, so the estimator
  has nothing to add on this tier.

The rotation matrix and codebook parameters travel inside every frame, so a
decoded cache depends on nothing but the frame itself. The frame is ordinary
safetensors: the existing structural validation and sha256 integrity
stampings apply unchanged.

## The correctness contract

Named tiers amend R1-5's exact-resume contract (see
[contract-spec.md](../contract-spec.md) "Tier correctness classes"):
determinism (same blob → identical tokens), a per-layer fidelity floor,
documented divergence from scratch, and fingerprint isolation — the tier
label is part of the fingerprint with exact-equality serving, so a lossy
blob can never serve an f16/q8 identity.

The f16 and q8 tiers keep the exact contract; q4/q3/q2 are reachable through
the existing `MLXCACHE_KV_BITS` plumbing with zero port cost and inherit
whatever parity the bits tiers carry. Named tiers are opt-in: no default
ever selects one.

## Measured: the bake-off (G1, G5)

`sidecar/probes/tq_quality_probe.py`, Qwen2.5-7B-Instruct-4bit, 1,500-token
prefix of the pinned T22 corpus, run against the FINAL shipped codec (f32
norms, norm compensation, codebook-in-frame). Fidelity = per-layer cosine of
quantize→dequantize KV against the f16 cache the engine actually held
(min over layers of min(cos_K, cos_V)); bytes = the wire payload; the f16
reference is the SERIALIZED frame (57,292 B/token), not allocation bytes.

| Tier | B/token | × vs f16 | min layer cos | mean layer cos |
|---|---:|---:|---:|---:|
| f16 (reference) | 57,292 | 1.00 | 1.0 | 1.0 |
| q8 (bits) | 30,474 | 1.88 | 0.99995 | 0.99997 |
| **tq-k4v4-g64** | **15,732** | **3.64** | **0.99234** | **0.99470** |
| q4 (bits) | 16,138 | 3.55 | 0.98995 | 0.99195 |
| tq-k3v4-g64 | 14,388 | 3.98 | 0.98106 | 0.98303 |
| q3 (bits) | 12,554 | 4.57 | 0.95983 | 0.96585 |
| tq-k3v2-g64 | 10,804 | 5.31 | 0.88506 | 0.90426 |
| q2 (bits) | 8,970 | 6.39 | 0.84858 | 0.87061 |

Readings that shaped the decision:

1. **tq-k4v4 strictly dominates q4** — smaller (15,732 vs 16,138 B/token)
   AND more faithful (0.992 vs 0.990). The port earns its place at the
   quality end of the frontier; tq-k4v4-g64 is the recommended default when
   a named tier is selected at all.
2. **The rotation buys fidelity, the uniform grid buys bytes.** At every
   matched compression band the TQ arm sits above the same-size bits arm on
   fidelity (k4v4 > q4, k3v4 > q3, k3v2 > q2), and the bits arms sit above
   on compression at the low end (q2 is the smallest payload measured).
   Both families are Pareto-frontier members; the tiers are complements,
   not replacements.
3. Real keys are slightly HARDER than the paper's worst case (iid): k3 keys
   measure 0.981–0.983 against the theoretical 0.985 floor — some layers'
   keys are less isotropic than random vectors. The probe's per-layer min is
   the number to watch per model.

## Measured: the remaining gates

- **G2 adopt budget** — TQ decode 11.4 ms vs mlx-lm f16 load 14.3 ms at the
  1,500-token prefix (both sides force full materialization): decode is
  faster than the f16 path. The R1-5 2 s budget holds with two orders of
  magnitude of headroom at this scale.
- **G4 append stability (real KV)** — encode(A) vs encode(A+B) over real
  engine caches: every prior token's packed bytes identical (element-wise
  codecs; proven again on the real tensors, not just synthetic ones). The
  future chunked store's block-identity property survives the tier.
- **G3 divergence (disclosure, not a pass/fail)** — greedy decode from a
  tq-k3v4-resumed cache vs scratch on three summarization prompts:
  agreement 0.062/0.062/0.042, first divergence at position 3/2/0 of 48.
  A lossy tier diverges from scratch EARLY; bounded-distortion is the
  contract, and the codec is deterministic so the divergence is stable
  across resumes of the same blob. If a workload needs token-exact resume,
  it must stay on f16/q8 — that is why the tier is opt-in and isolated in
  the key space.

The opt-in real gate (`test_roundtrip_real.py`, TQ arm) pins: codec
determinism, resume determinism, >3× compression vs f16, and the k4v4
per-layer fidelity floor (0.985, margin under the measured 0.992).

## Live-run disclosures (QA, real daemon + 7B)

Two behaviors of the lossy tier that the exact tiers do not have, both
verified on a live daemon (`tq-k4v4-g64`):

1. **The publishing request's own generation is a lossy continuation.** The
   daemon's design resumes /generate from the checkpoint the request just
   published (single-flight followers need this). On the TQ tier that means
   the FIRST request's output continues from quantize→dequantize KV —
   deterministic and reproducible (an identical re-request replays the same
   tokens from the same blob), but not bit-equal to the same prompt served
   without the daemon. A scratch retry that bypasses the blob (e.g. after a
   quarantine) can therefore generate a different tail than the blob-adopted
   run, and its settle lands under a different key (same fingerprint,
   different generated tokens).
2. **A quarantined entry costs one extra miss before the cache heals.** On
   f16 the scratch retry regenerates identical tokens and republishes under
   the original key immediately; on TQ the retry's divergent tail publishes
   a sibling entry instead, so the NEXT request takes an honest miss and
   republishes the shared-prefix node itself. Responses stay correct
   throughout; only hit rate dips once.

Neither is a correctness break — both follow from bounded distortion plus
the daemon's resume-from-own-checkpoint design — but an operator comparing
TQ outputs against a no-daemon run should expect token-level differences
with equivalent meaning.

## The live variant (phase 4): compressed-resident KV during decode

`MLXCACHE_TQ_LIVE=1` (sidecar engine, requires `MLXCACHE_KV_TIER`) swaps the
engine's own cache for `TurboQuantKVCache`: the RESIDENT state is the
compressed tier state; `update_and_fetch` quantizes new rows in place and
returns the dequantized history, so stock attention runs unchanged (the
cache-class API is the extension point — still no mlx-lm patches, and the
wrapper keeps its no-patch property).

Measured (Qwen2.5-7B, 2,000-token prefix, 32 decode tokens,
`sidecar/probes/tq_live_probe.py`):

| | f16 cache | live tq-k4v4-g64 | trade |
|---|---:|---:|---|
| resident KV | 117.4 MB | 32.1 MB | **3.66x smaller** |
| decode speed | 48.8 tok/s | 41.2 tok/s | 0.843x (18% slower) |

That is the paper's own hybrid-decode caveat, quantified on Metal: the
per-step dequant costs real compute, bought back as unified memory. Capacity
is step-rounded (256 positions) like mlx-lm's KVCache, so short contexts
carry rounded-up padding; long contexts see the full ratio.

**What the live tier does not promise** (measured, not assumed): token
identity with the persistence tier. Quantizing during the forward changes
the computation itself — every position's keys are computed attending
through quantize-then-dequantize history, recursively — while the
persistence tier's blob was computed by an f16 prefill and quantized
afterward. On the gate prompt the first token is a near-tie that flips
between the tiers, and the two quantized continuations then converge onto
the same stream. The pinned guarantees are per-run determinism, frame
standardness (a live-written blob is byte-identical to a persistence-written
one for the same input rows), and the tier's fidelity floors.

**Scoring from compressed keys** (`tq/score.py`): attention scores are
computable straight from the packed state — rotate the query once, then a
gather-dot over per-dimension centroid values — numerically equal to
dequantize-then-score. The QJL residual correction (1 sign bit per
dimension of the rotated residual) measurably reduces the score error and
is implemented but persisted by nothing: it costs d bits/token (the frame
grows ~a third at k3) and O(S·d) correction compute. The K-side bandwidth
of packed state is ~4x smaller than f16, so a FUSED unpack-gather-dot
kernel is the only way to realize a speed win — plain MLX ops materialize a
[S, d] u32 intermediate larger than the f16 it replaces (the probe measures
the tensor-op path anyway). Wiring any of this into the decode hot loop
means a custom Metal kernel or cache-aware attention in mlx-lm itself; the
math, its fidelity, and this boundary live measured in the module.

## Provenance and invariants

- Codebooks are DERIVED (Max-Lloyd on the paper's Lemma-1 density) and
  validated against the paper's own published anchors in
  `test_tq_codebook.py`. No codebook files or code were taken from any
  third-party implementation.
- The rotation seed is pinned (`rotation.PINNED_SEED`) so every blob this
  process writes uses one canonical basis; the matrix itself travels in the
  frame, so decode never depends on regenerating it. A future MLX whose RNG
  or QR differs only changes the bytes of NEW blobs, never their meaning.
- 3-bit key indices pack 10 per u32 word (2 bits wasted per word): the
  straddle-free layout keeps pack/unpack pure shift/mask tensor ops. b=2 and
  b=4 pack exactly.
- Dense KVCache stacks only (the same boot guard as the bits tiers; hybrid
  compositions refuse at boot on the sidecar and fail the persist
  best-effort on the wrapper).
- The tier namespaces the blob key (folded last, zero bytes when empty —
  see connector-protocol §4) and is compared with exact equality on every
  serving path (core `identity_compatible`, daemon `same_shape`, store
  client `identity_compatible`). Cross-tier serving is impossible by
  construction and covered by tests on both sides of the wire.
