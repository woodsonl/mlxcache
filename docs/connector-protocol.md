# mlxcache Connector Protocol

Status: **Normative** for the mlxcache store. Version 1.1 of this spec
(v1.0 failed its own review gauntlet: matched_len, CacheRef, the key fold,
coverage, and the unknown-field policy were under-specified; v1.1 fixes
all ten findings).

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
| `fingerprint` | object | `model_id`, `tokenizer_hash`, `kv_dtype`, `kv_layout_version`, `kv_bits`, `kv_group_size` `[CURRENT]` |
| `token_count` | u64 | cross-checked against `len(tokens)` at read `[CURRENT]` |
| `tokens` | [u32] | the exact token-ID prefix this entry is keyed by `[CURRENT]` |
| `format_version` | u32 | see §5 `[CURRENT]` |
| `payload_sha256` | hex string | sha256 of the payload bytes `[CURRENT]` |
| `engine_id` | string? | e.g. `"mlx-lm"`, `"llama-cpp"`; absent ⇒ `"mlx-lm"` `[B0.2]` |
| `granularity` | u8? | 0 = `ANY_PREFIX` (default), 1 = `WHOLE_CONTEXT` `[B0.2]` |

**Unknown fields**: readers MUST ignore header fields they do not
recognize; an unknown field is never corruption. `engine_id` and
`granularity` are additions WITHIN `format_version` 2, which writers MUST
still emit. (Precedent: the T12 incident — strict parsing quarantined
every healthy ancestor when a newer daemon added a field.) `[CURRENT:
blob.py _known_fields strips unknowns; Rust serde ignores unknown fields]`

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
  stream equals the entry's `tokens` `[B0.2]`.
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
- **B0.2 extension**: non-default `engine_id` is length-prefixed and
  folded BEFORE all fingerprint fields, preceded by its own `0xFFFFFFFF`
  domain separator. The DEFAULT engine (`"mlx-lm"`) contributes ZERO
  bytes — keys of engine-less and default-engine entries are
  byte-identical to the current fold, so existing stores keep their keys.

Cross-engine `lookup` returns `matched_len = 0` by construction (the
fingerprint check in §3.3 additionally re-verifies identity).

Engine registry (maintained here): `"mlx-lm"`, `"llama-cpp"` (reserved).

## 5. Versioning

`format_version` policy `[CURRENT: persistence.rs load(), blob.py decode]`:

- `1` — legacy, pre-digest. Readable, NEVER verified. Writers MUST NOT
  emit it.
- `2` — current. `payload_sha256` REQUIRED; a v2 header without it is
  `CorruptCache(version)`.
- anything else — refuse with a reason naming the version. Never
  interpret a foreign layout.

Writers declare KV layout via `fingerprint.kv_layout_version`; readers
MUST refuse to serve an entry whose fingerprint differs from the caller's
(primary enforcement is §4 keying; implementations re-check).

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
