# mlxcache Connector Protocol

Status: **Normative** for the mlxcache store. Version 1.0 of this spec.

Every clause is tagged with when it is enforceable: `[CURRENT]` is
verifiable against the code on `main` today; `[B0.2]` is introduced by the
format batch that follows this document and becomes `[CURRENT]` when that
batch merges. A conformance suite (Appendix A of the program plan) is the
executable form of this spec.

Audience: an engineer adding KV-cache persistence to an inference engine
(mlx-lm today, llama.cpp-class engines next) without adopting the mlxcache
daemon.

## 1. Model

An engine produces and consumes opaque KV state ("payload"). A **store**
persists payloads keyed by token prefix and engine identity. The store
NEVER interprets payload bytes `[CURRENT: persistence.rs treats payload as
&[u8]; framing checks live engine-side]`. Engines own encoding and
decoding. **Coverage is NOT engine-decided** (§3.2 is the exception carved
into this rule, binding all engines).

## 2. On-disk format (L0)

One checkpoint per file: `{key:032x}-{generation:016x}.ckpt` `[CURRENT]`,
relative to the store root, written temp-then-atomic-rename `[CURRENT]`.
File body:

```
| u32 LE header_len | JSON header (header_len bytes) | payload bytes |
```

JSON header fields `[CURRENT, plus B0.2 additions marked]`:

| field | type | notes |
|---|---|---|
| `fingerprint` | object | `model_id`, `tokenizer_hash`, `kv_dtype`, `kv_layout_version`, `kv_bits`, `kv_group_size`, `cache_layout`, `content_digest`, `kv_tier` `[CURRENT]`. `tokenizer_hash` is engine-specific, and an unavailable tokenizer fails CLOSED everywhere (the literal `"missing"` sentinel disables persistence — lookup and publication — rather than pinning an unverifiable identity). **Wrapper (L1, `mlxcache_serve`)**: sha256 of the model's `tokenizer.json` bytes, read symlink-safely: HF-cache snapshot symlinks are followed only while the target resolves inside the model dir or its own `blobs/` storage (fail-closed otherwise, directories read as absent), and the opened file's inode is reconciled against the checked path. **Sidecar connector (`mlxcache_sidecar`)**: canonically the UTF-8 SHA-256 digest of `repr(sorted(tokenizer.get_vocab().items()))`, truncated to the first 16 hexadecimal characters — implementations must reproduce this exact representation and truncation to produce compatible fingerprints and reuse checkpoints (an alternate serialization or truncation length yields a different hash and silently forfeits reuse); when `get_vocab()` is unavailable the hash is the `"missing"` sentinel and the sidecar refuses to settle (the daemon likewise refuses to publish under it) |
| `token_count` | u64 | cross-checked against `len(tokens)` at read `[CURRENT]` |
| `tokens` | [u32] | the exact token-ID prefix this entry is keyed by `[CURRENT]` |
| `format_version` | u32 | see §5 `[CURRENT]` |
| `payload_sha256` | hex string | sha256 of the payload bytes `[CURRENT]` |
| `engine_id` | string? | e.g. `"mlx-lm"`, `"llama-cpp"`; absent ⇒ `"mlx-lm"` `[CURRENT]` |
| `granularity` | u8? | 0 = `ANY_PREFIX` (default), 1 = `WHOLE_CONTEXT` `[CURRENT]` |

`fingerprint.cache_layout` is a short, stable signature of the model's
**cache composition**: the ordered tuple of cache-entry class names the
engine builds for it (a dense KV stack, a `CacheList` of linear-attention
state, an `ArraysCache` of SSM state, …), joined and hashed. Two
architectures can agree on model id, tokenizer, dtype and quantization tier
while their payloads have mutually unloadable shapes, so composition is
part of the identity that gates serving. A writer that cannot report a
composition MUST omit the field (it is never emitted empty); a reader
resolves an absent field to `""`, which means **composition unrecorded**
and is served only legacy-to-legacy (§3.3). An engine that reports no
`engine_id` is the default engine; writers MUST NOT name the default
engine explicitly, so default-engine headers stay byte-identical to
pre-namespacing ones.

`fingerprint.content_digest` is the identity of the request's **non-text
content**: a sha256 over the canonicalized non-text parts of the messages
(images today; the rule is "not a text part", so a future modality is
covered without a per-type table). The index keys on token ids, and media
renders as ordinary ids — every image is the same `<|image_pad|>` — so two
different images at one prompt produce one token sequence and two different
KV states. Without this axis one image's checkpoint would be served to the
other. An empty digest means **no non-text content**, which is a claim
about the request rather than a missing measurement, so the rule is exact
equality (§3.3) and a writer that records no digest MUST omit the field.

**Unknown fields**: readers MUST ignore header fields they do not
recognize; an unknown field is never corruption. `engine_id` and
`granularity` are additions WITHIN `format_version` 2, which writers MUST
still emit; `cache_layout` and `content_digest` are additions to the
fingerprint object (defaulted, omitted when empty). (Precedent: the T12 incident — strict
parsing quarantined every healthy ancestor when a newer daemon added a
field.) `[CURRENT: blob.py _known_fields strips unknowns; Rust serde
ignores unknown fields]`

### Chunked payloads (`format_version` 3) `[CURRENT]`

A v3 blob is a HEADER-ONLY `.ckpt` (the u32+JSON header above, no payload
bytes) plus two siblings, all written temp-then-rename with the `.ckpt`
rename as the visibility point:

- `{stem}.ckm` — a binary chunk manifest: magic `MCKM` + u16 version, u64
  `payload_len`, u64 `chunk_count`, then 56-byte records
  `digest[32] ‖ per_head_len u32 ‖ n_heads u32 ‖ first_off u64 ‖
  head_stride u64`. A chunk's body is the per-head slice concatenation (head
  order); reassembly scatters `per_head_len` bytes at
  `first_off + h·head_stride` per head.
- `chunks/seg-{seq:010}.seg` — immutable packed segments shared by every
  v3 blob in the store: magic `MCSG` + u16 version + 2 reserved bytes, then
  records `body_len u32 ‖ digest[32] ‖ body`. Chunk bodies are
  content-addressed (sha256); a digest already present is never re-written.
  A truncated tail record (crash mid-append) ends a segment scan cleanly.

The chunk codec is **token-block/64**: per safetensors tensor, blocks of 64
covered positions; a block's identity is the sha256 over the concatenation
of each KV head's byte slice for those positions (the head-major layout
makes those slices append-stable, so a shared token prefix yields
byte-identical chunks — the dedup the format exists for). Bytes no tensor
claims (the safetensors JSON header, inter-tensor gaps) and non-safetensors
payloads fall back to whole-scope single chunks: identity and reassembly
stay exact; byte-identical payloads still dedup.

`payload_sha256` covers the REASSEMBLED payload and is verified by every
reader (the D1 contract unchanged): corruption in a manifest or segment
produces wrong bytes and a digest mismatch — the same retire path as a
whole-blob flip. Refcounts are DERIVED at open from live `.ckpt`↔`.ckm`
pairs (never trusted from disk); a segment leaves disk only when every
record in it is unreferenced, and a crash between the two renames leaves
reclaimable orphans.

**Writer discipline**: ONE chunk-format writer per store directory — the
Rust daemon. The Python bindings READ v3 (the sidecar adopts chunked blobs
through the same reassembly; the L1 store verifies them at open and fetch)
but keep PUBLISHING whole-blob v2. Both layouts coexist in one directory
indefinitely; v1/v2 readers see a v3 blob as an uninterpretable version and
retire it loudly rather than misread it (mixed-version rollout means
upgrading the daemon and its sidecar together, which is how they ship).

**Deviation from REPORT.md's GO recommendation, recorded**: the GO decision
kept CDC (gear, 16 KiB, head scope) as the fallback codec for engines that
hand over opaque bytes. This build ships a **whole-scope single-chunk
fallback** instead: a payload that is not framing-valid safetensors becomes
one chunk covering its full length. Rationale: no production engine produces
opaque bytes (mlx-lm always emits safetensors; the only opaque producer is
the test-only synthetic engine), Python publishers stay whole-blob v2, and a
dead codec path is untested surface. The whole-scope fallback is exact and
still dedups byte-identical payloads (REPORT.md P2's cold-twice
determinism). If an engine that cannot report a tensor grid ever ships, CDC
remains the recorded design for it — port `chunk_stream`/`apply_envelope`
from `scripts/dedup/chunk_sim.py`, measured there at 5.6%/9.5% new bytes
per turn (f16/q8) against token-block's 5.3%/5.2%.

`MLXCACHE_CHUNK_STORE=0` on the daemon disables v3 publication (rollback
knob; reads still understand both).

## 3. Operations (L1)

Any store implementation MUST expose `[B0.2 for the standalone client;
the daemon's HTTP surface implements the same semantics today for
mlx-lm blobs]`:

```python
lookup(engine_id: str, fingerprint: Fingerprint, tokens: Sequence[int])
    -> (matched_len: int, ref: CacheRef | None)
fetch(ref: CacheRef) -> bytes
put(engine_id: str, fingerprint: Fingerprint, tokens: Sequence[int],
    payload: bytes, granularity: Granularity = ANY_PREFIX) -> CacheRef
invalidate(ref: CacheRef | None = None, prefix: Sequence[int] | None = None) -> int
```

### 3.1 CacheRef

A `CacheRef` is the blob **filename relative to the store root** (§2). It
is stable across process restarts and remains resolvable until the entry
is invalidated, evicted, or superseded; `fetch` on a ref whose file is
gone raises `Unavailable(io)`.

### 3.2 Coverage and matched_len (normative formulas)

- `put`'s payload MUST contain KV for **exactly `tokens[:-1]` positions**
  — the writer stores state for all but the final token. This binds ALL
  engines and BOTH granularities: `WHOLE_CONTEXT` restricts *matching*,
  never coverage. (KV at position j depends only on tokens `..=j`; the
  final token's KV is the decoder's, not the cache's.)
- `matched_len` is always `len(entry.tokens)` of the matched entry —
  never capped by the request length, never `matched_len - 1`.
- The engine MUST resume by feeding `request_tokens[matched_len - 1:]`
  (the payload covers `0..matched_len-1`; the request's token at
  `matched_len - 1` is the first one whose KV the payload does not
  already include).

### 3.3 Matching semantics

- `lookup` returns at most ONE ref: the longest match that is BOTH
  engine-compatible AND fingerprint-identical, honoring per-entry
  granularity. `ANY_PREFIX` entries match any request whose token stream
  extends the entry's `tokens` `[CURRENT in daemon index]`;
  `WHOLE_CONTEXT` entries match only when the request's FULL token
  stream equals the entry's `tokens` `[CURRENT]`.
- **Cache composition** `[CURRENT]`: an entry matches only when its
  `fingerprint.cache_layout` equals the caller's. `""` means composition
  unrecorded, and the two consequences are deliberate:
  - a request reporting a NON-EMPTY layout is NEVER served an
    unrecorded-layout entry — nothing proves that payload has the shape the
    caller's engine will deserialize, and deserializing the wrong shape is
    silent wrong output, not an error;
  - a request reporting `""` may be served a `""` entry, so a store written
    before compositions were recorded stays usable until the current engine
    rewrites its blobs (legacy-to-legacy only: the moment either side knows
    its composition, the comparison is exact).

  This is not a compatibility grace period to be relaxed later: an engine
  that reports no composition asserts it cannot prove compatibility, and
  exact equality is the only safe reading of that assertion.
- **Non-text content** `[CURRENT]`: an entry matches only when its
  `fingerprint.content_digest` equals the caller's, exactly. Unlike cache
  composition there is no legacy-tolerant branch: `""` asserts that the
  prefix carries no media, so it never matches a request that does and is
  never matched by one.
- **End-anchored rule** `[CURRENT in daemon]`: a request whose stream
  equals the entry's `tokens` through `len-1` but DIVERGES at the final
  token still matches — the payload covers exactly `0..matched_len-1`,
  all of which the diverging request shares.
- **Granularity fallback**: if the longest candidate is `WHOLE_CONTEXT`
  and the request is not its exact full stream, lookup falls back to the
  longest `ANY_PREFIX` match below it `[B0.2]`.
- Granularity is entry metadata, NOT part of the key: a `put` at the same
  `(engine_id, fingerprint, tokens)` as an existing entry supersedes it —
  last writer wins, generation increments `[B0.2]`.

### 3.4 Integrity semantics

- `fetch` verifies `payload_sha256` on every read; a mismatch raises
  `CorruptCache(digest)` and the entry is invalid (embedded callers fall
  back to scratch; the daemon quarantines and republishes). Verification
  lives in the READ layer (daemon `Persistence::load`, sidecar
  `read_wire_checkpoint`) — the wire codec itself only parses
  `[CURRENT]`.
- `put` is atomic (temp + rename), stamps the digest at write, and
  assigns a generation so a superseding publication never overwrites an
  earlier file `[CURRENT: daemon; B0.2 for L1 writers, which MUST use
  generation = (unix_ms << 16) | (pid & 0xffff) — practically unique
  across processes; the daemon's per-process monotonic floor remains its
  own scheme]`.
- `invalidate` by ref removes that entry; by prefix removes the exact
  entry AND every entry whose `tokens` extend the prefix (subtree), with
  the end-anchored rule applying to which entries count as extending
  `[B0.2 for L1 semantics]`. Removal re-verifies the exact generation
  before unlinking, so a late invalidation can never delete a fresh
  replacement `[CURRENT in daemon quarantine/eviction paths]`.

Error taxonomy (all bindings): `CorruptCache(digest | version | header)`,
`Unavailable(io)`, `Refused(budget | recovery)`. Header validation
(malformed JSON, wrong types, non-u32 token lists → `CorruptCache(header)`)
is performed at the read layer in every binding `[CURRENT in
read_wire_checkpoint's guards; blob.py codec-level hardening lands B1.1]`.

## 4. Namespacing and the key fold

Keys fold engine identity, fingerprint, and token prefix into 128 bits.
The fold is **normative by reference and by construction**:

- **Current (mlx-lm) fold** `[CURRENT: http.rs blob_key]`: length-prefixed
  fingerprint fields → u32-LE chunks → `0xFFFFFFFF` domain separator →
  token stream → dual-lane FNV-1a (lane seeds `0xcbf29ce484222325` /
  `0x9e3779b97f4a7c15`, primes `0x100000001b3` / `0x880355f21e6d1965`).
- **Engine namespacing**: a non-default `engine_id` is length-prefixed (same
  `:field` scheme as fingerprint fields) and folded BEFORE all fingerprint
  fields, **closed by its own `0xFFFFFFFF` domain separator**. The DEFAULT
  engine (`"mlx-lm"`, and an absent id) contributes ZERO bytes (no chunks,
  no separator) — keys of engine-less and default-engine entries are
  byte-identical to the engine-less fold, so existing stores keep their
  keys.
- **Cache composition** `[CURRENT]`: a NON-EMPTY
  `fingerprint.cache_layout` is one more length-prefixed field, folded
  AFTER `kv_group_size` and before the closing domain separator. An EMPTY
  layout contributes ZERO bytes, on the same rule as the default engine id:
  every blob whose composition is unrecorded keeps its previous key.
- **Non-text content** `[CURRENT]`: a NON-EMPTY
  `fingerprint.content_digest` folds as the LAST length-prefixed field,
  after `cache_layout`. Empty contributes ZERO bytes, so text-only keys are
  unchanged.
- **Named KV tier** `[CURRENT]`: a NON-EMPTY `fingerprint.kv_tier`
  (e.g. `tq-k3v4-g64`) folds as the last length-prefixed fingerprint field,
  after `content_digest`. Empty contributes ZERO bytes, so pre-tier keys
  are unchanged. A tier labels a payload format the scalar `kv_bits` pair
  cannot express (the TurboQuant codecs: asymmetric key/value widths, a
  codec-specific frame), so it namespaces exactly like `kv_bits` does for
  q8 and is compared with EXACT equality — never cross-served against the
  untiered identity or another tier.

Full order, with `?` marking the fields that fold only when non-default:

```
engine_chunks? ‖ SEP ‖ model_id ‖ tokenizer_hash ‖ kv_dtype
                       ‖ kv_layout_version ‖ kv_bits ‖ kv_group_size
                       ‖ cache_layout? ‖ content_digest? ‖ kv_tier? ‖ SEP ‖ tokens
```

Every field is length-prefixed, so no field's bytes can be re-split across
a boundary to alias another identity — including a `cache_layout` or
`content_digest` crafted to look like "previous field plus this one".

Cross-engine `lookup` returns `matched_len = 0` by construction (the
fingerprint check in §3.3 additionally re-verifies identity). `engine_id` is
a key domain, not a hint: because it folds ahead of the fingerprint behind its
own separator, two engines can never produce the same key for the same model
and tokens, so a blob written by one engine is unreachable to the other
regardless of how closely their fingerprints otherwise agree.

Engine registry (maintained here): `"mlx-lm"` (the default) and
`"llama-cpp"` (reserved).

## 5. Versioning

`format_version` policy `[CURRENT: persistence.rs load(), blob.py decode]`:

- `1` — legacy, pre-digest. Readable, NEVER verified. Writers MUST NOT
  emit it.
- `2` — whole-blob current. `payload_sha256` REQUIRED; a v2 header without
  it is `CorruptCache(version)`.
- `3` — chunked (§2 "Chunked payloads"): header-only `.ckpt`, payload in
  the chunk store, `payload_sha256` REQUIRED and verified over the
  REASSEMBLED bytes. Pre-v3 readers refuse loudly (a v3 blob is a foreign
  layout to them, never a misread).
- anything else — refuse with a reason naming the version. Never
  interpret a foreign layout.

Writers declare KV layout via `fingerprint.kv_layout_version`; readers
MUST refuse to serve an entry whose fingerprint differs from the caller's
(primary enforcement is §4 keying; implementations re-check).

`kv_layout_version` describes the PAYLOAD layout (how the bytes are
framed), not the cache composition. Adding composition to the identity is
therefore not a reason to bump it: the field lives in the fingerprint,
which every read already compares.

## 6. Trust boundary

- Every header field is untrusted input: malformed JSON, wrong types, and
  non-u32 token lists are rejected as `CorruptCache(header)` — never a
  crash, never a 500-loop `[CURRENT: server.py decode guards]`.
- The digest is the integrity primitive. **Framing verification is always
  performed by the ENGINE on the bytes returned by `fetch`** — the L1
  client never inspects payload structure; `check_safetensors` is an
  mlx-lm daemon-wire convenience outside this protocol `[CURRENT]`.
- A whole-store open (embedded) or boot sweep (daemon) verifies every
  entry; deterministically-corrupt entries are reclaimed (file deleted);
  transient I/O errors are retried on next open — a transient failure is
  not proof of corruption `[CURRENT: daemon rebuild; L1 mirrors B1.1]`.

## 7. Bounded storage

- **Budget basis**: the total size of all `*.ckpt` files PRESENT in the
  store directory (indexed by this process or not), recomputed by listing
  the directory at enforcement time `[B0.2 for L1; daemon today sums its
  indexed candidates — a known divergence the conformance suite pins]`.
- Default budget 32 GiB; 0 = unbounded `[CURRENT in daemon]`.
- Eviction removes non-anchor entries **ordered by `token_count`
  descending** (most bytes freed per unlink); recency and live extension
  confer anchor STATUS only — they never participate in ordering.
- An entry is an anchor if a live published child extends it OR it was
  used within the anchor window (default 900 s) `[CURRENT]`. Anchors are
  never evicted; an over-budget store held by anchors logs and continues.
  Eviction is a throughput cost, never a correctness one.
- **Recency clock** `[B0.2 for L1]`: per-process monotonic time; on first
  open of an existing store, every entry is seeded as just-used; the
  anchor window is measured from each process's own open time. (No
  on-disk timestamps; cross-process recency is not claimed.)
- Eviction may unlink any file present in the directory; a concurrent
  reader whose ref disappears gets `Unavailable(io)` and falls back to
  scratch.

## 8. Bindings

Identical semantics; transport only differs.

1. **Embedded library** — direct filesystem access; the L1 client
   (`mlxcache_store`, B1.1). Multiple processes MAY share one store
   directory: filenames are immutable per publication (per §3.4's
   generation rule), every read verifies its digest, and a vanished file
   is `Unavailable(io)` — worst case is a missed reuse, never wrong KV.
2. **Local service** — the mlxcache daemon: OpenAI-shaped client surface,
   `/stats`, eviction, quarantine republish, trace. Same L0 store.
3. **Remote** — future; this protocol over HTTP.

## 9. Conformance

An implementation conforms when it passes the conformance suite
(program plan Appendix A): cross-language golden round-trips, the
simulated foreign engine (namespacing, opacity, whole-context
granularity + fallback), the version matrix, tamper rejection,
concurrent writers, budget-basis-on-directory, and eviction ordering.
"No interpretation of payload bytes" is tested by storing non-safetensors
opaque bytes under a non-default engine.
