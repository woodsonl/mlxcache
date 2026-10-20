# Benchmarks

Two benchmark scripts, for two different questions:

| script | engine | question it answers |
|---|---|---|
| `scripts/bench_workloads.py` | synthetic (deterministic, no model) | does the request plumbing, streaming and accounting work end to end? |
| `scripts/bench_prefix.py` | **real mlx-lm** | what does prefix caching actually buy on real traffic? |

`bench_workloads.py` is a **plumbing harness**. Its synthetic tokenizer hashes a
whole prompt into 8 tokens, so it cannot express prefix reuse at all: every
number it prints about caching is meaningless by construction. It exists so the
HTTP/SSE path and the metrics arithmetic can be tested in CI without a model.
Cache claims come from `bench_prefix.py`, and only from it.

## What `bench_prefix.py` measures

Per request it records TTFT, inter-token latency (TPOT), end-to-end latency,
completed tokens, the cache verdict, and three separate reuse numbers:

- **declared** — the longest common token prefix this request shares with any
  earlier request in the run. This is the reuse the *traffic* offered.
- **served** — the daemon's own `prefill_from` (KV positions it actually
  adopted).
- **held** — the trace's `matched_tokens - 1`, i.e. the KV the adopted blob
  truly contains.

Reuse is reported as a fraction of requested tokens (how much of the work was
skipped) rather than as a hit rate, because a "hit" on a 12-token prompt is
worth nothing while a partial on a 4,000-token prompt is the whole point.

Three premises abort a run instead of printing a number:

1. **serialization parity** — the trace's `n_tokens` must equal the token
   count the engine reports for *our* serialization of the same messages.
   A mismatch means the daemon tokenizes a different string, so every declared
   prefix is fiction.
2. **served == held** — the response body and the trace must agree about the
   same request.
3. **served <= declared** — the daemon cannot serve reuse the traffic never
   offered.

There is also a disk preflight: a failed checkpoint publish is reported as a
plain miss (by design, so a healthy request never 502s), which makes a full
disk look exactly like a cache that never helps. The harness refuses to start
without enough free space rather than report that as a result.

## The serve rule these workloads encode

The index serves a checkpoint only when the request agrees with the entry key
through its **last token** (`crates/mlxcache-core/src/index.rs`). Diverging
mid-key means no reuse, because trimming quantized MLX KV at an arbitrary
position is not safe. The lookup also walks **one greedy path**, so a sibling
entry that agrees further along the request shadows a shallower entry that
would otherwise have served.

That has a direct, counter-intuitive consequence for the most common real
traffic shape — *one long system prompt plus a short varying question*:

- Each request publishes a key as long as the whole prompt. The next question
  diverges **17–20 tokens before that key's end**, so it is not served.
- If the prefix itself is published once as its own request, the follow-up
  question diverges exactly one token past that key's final token, which the
  end-anchored child rule *can* serve — about the whole prefix.
- Once one full-length question has published a deeper key on the same path,
  it shadows the prefix-only entry for later questions.

So the benefit depends on how the prefix got into the store, not just on how
much text is shared. `bench_prefix.py` therefore runs W1 **both ways**:
`W1` (primed: the prefix is published once as its own request) and `W1np`
(unprimed: the shapes a normal application actually sends).

## Workloads

| id | shape | what it isolates |
|---|---|---|
| `W1` | fixed long prefix + N short changing questions, primed with one prefix-only request | best-case benefit for shared-prefix traffic |
| `W1np` | the same traffic with **no** priming request | what an application gets without knowing the rule |
| `W2` | documents share a long opening, then diverge mid-document | near-miss / "similar document" traffic |
| `W3` | distinct short prompts, 256-token answers | decode-bound traffic, where reuse cannot help |
| `W4` | fixed prefix + k unrelated requests between repeats, under a tight entry cap | **does the benefit survive cache pressure?** |
| `W5` | the captured trace of the above, scored with `scripts/replay_trace.py` | trace and `/stats` must agree |

Model, hardware, concurrency (1) and output cap are held constant across
workloads; input length is held constant only WITHIN each workload's own
cold/warm comparison — W1, W1np and W2 share the same ~1,975-token prefix
while W3 deliberately uses short distinct prompts (decode-bound by design),
so cross-workload rows are not input-length-controlled and are not
comparable as such. Each workload in the cold/warm grid below (W1, W1np,
W2, W3) runs twice: **cold** (empty store, fresh process) and **warm**
(fresh process over the same store) — the cross-restart tier, which is what
a persistent cache is for. W4 (pressure) and W5 (trace/stat agreement) are
adversarial validation arms, not part of the cold/warm comparison.

The `direct` arm runs the same requests against plain `mlx_lm.server`, whose
`LRUPromptCache` is process-local and therefore cold after every restart. That
is the cross-process baseline, and it is not a straw man: its in-process repeat
leg measures the cache it actually has.

W4's two sub-arms are the pressure answer:

- `p_alone` publishes the prefix and nothing else. The reaper evicts
  biggest-first (`score = token_count`), so a long shared prefix is the **first**
  thing to go.
- `p_extended` also publishes a request that *extends* the prefix, which makes
  the prefix a structural anchor. The reaper takes the extension first and
  leaves the prefix published.

## Running it

The model must already be on disk; this harness never downloads one.

```bash
# full grid (model load + prefills; tens of minutes)
uv run python scripts/bench_prefix.py --model /abs/path/to/snapshot \
  --workloads W1,W1np,W2,W3 --arms daemon,direct

# the pressure sweep (fresh store and process per k, so it is the slow part)
uv run python scripts/bench_prefix.py --arms pressure --k 0,1,2,4,8,16
```

Output is a markdown table plus a JSON artifact containing every per-request
sample, the daemon `/stats` snapshot per leg, and the enforced trace/stats
cross-check. The premise arithmetic itself is pinned by
`sidecar/tests/test_bench_prefix.py` (what CI runs).

## Measured results

Provenance — run 2026-10-19 on Apple M3 Max (Darwin 27.0.0), `mlx-community/Qwen2.5-7B-Instruct-4bit` (mlx 0.32.2, mlx_lm 0.31.3), concurrency 1, `--prefix-tokens 2000` **calibrated at runtime against the real tokenizer** (27 tokens per filler repeat → a measured ~1,975-token shared prefix; earlier runs divided by a guessed 16 and silently benchmarked a ~3,250-token prefix under a 2,000 knob), 6 varying requests per workload, all premise guards green including the enforced trace/`/stats` cross-checks:

```bash
uv run python scripts/bench_prefix.py --workloads W1,W1np,W2,W3 --arms daemon,direct \
  --questions 6 --prefix-tokens 2000 --json /tmp/prefix-grid4.json
uv run python scripts/bench_prefix.py --arms pressure --k 0,1,2,4,8,16 \
  --prefix-tokens 2000 --json /tmp/prefix-pressure4.json
```

### Cross-restart reuse (the daemon's own cold vs warm)

| workload | leg | n | TTFT p50 (ms) | TPOT p50 (ms) | served % of requested | verdicts |
|---|---|---|---|---|---|---|
| W1 (primed) | daemon/cold | 7 | 3,178.3 | 13.12 | 14.17 | 6 miss, 1 partial |
| W1 (primed) | daemon/warm | 7 | 154.3 | 13.01 | 99.95 | 7 hit |
| W1np (unprimed) | daemon/cold | 6 | 3,183.4 | 13.38 | 0.0 | 6 miss |
| W1np (unprimed) | daemon/warm | 6 | 155.1 | 13.03 | 99.95 | 6 hit |
| W2 near-miss | daemon/cold | 6 | 2,020.6 | 13.30 | 0.0 | 6 miss |
| W2 near-miss | daemon/warm | 6 | 146.9 | 13.18 | 99.93 | 6 hit |
| W3 decode-heavy | daemon/cold | 6 | 164.2 | 12.21 | 0.0 | 6 miss |
| W3 decode-heavy | daemon/warm | 6 | 86.6 | 12.56 | 95.0 | 6 hit |

What the grid says:

1. **Cross-restart reuse is real**: warm TTFT drops per workload as the table shows — W1 3,178 → 154 ms (~21× lower), W1np 3,183 → 155 ms (~21×), W2 2,021 → 147 ms (~14×), W3 164 → 87 ms (~1.9×; short prompts, reuse cannot help much). W1, W1np and W2 serve 99.93–99.95% of requested prefix tokens after both processes were killed and restarted from disk (W3: 95%). No in-process cache can do this tier at all.
2. **TPOT is flat (~12–13 ms) cold vs warm everywhere.** Caching moves prefill, not decode; W3 exists to keep that measured rather than assumed.
3. **Mid-document divergence is never served**: W2's near-miss traffic (a long shared opening that diverges before either document's end) gets exactly 0% cold, as the end-anchored rule requires.
4. **Priming matters once**: W1's first question is served ~1,959 tokens because the prefix was published alone; W1np's first question gets 0. After each request publishes its own full-length key, both warm legs converge to ~99.95%.

### Against a raw local engine (the honest baseline)

The direct arm sends the daemon's exact serialization through `/v1/completions`, so both engines consume identical bytes (the daemon applies no chat template — it tokenizes `serde_json::to_string(messages)`).

| leg | TTFT p50 (ms) |
|---|---|
| direct inproc (mlx_lm LRU prompt cache) | 121 |
| daemon warm (blob load + KV adopt, cross-restart) | 154 |
| direct restart (cold prefill every time) | 189 |
| daemon cold (lookup + prefill + publish) | 3,178 |

At a true ~2,000-token prefix on this hardware the ordering flips from the
earlier (miscalibrated, ~3,250-token) run: the daemon's cross-restart warm
path (~154 ms) is now **faster than plain mlx_lm re-prefilling the same
bytes (~189 ms)**, and only the in-memory LRU cache of a live process stays
ahead (~121 ms) — at the cost of dying with the process. The cold path
still pays ~3.2 s, dominated by publishing a ~110 MB checkpoint (~57 MB per
1,000 tokens at 7B-4bit). Honest conclusion: at 2K prefixes the cached path
wins on latency AND survives restarts; the first request that publishes
still pays the serialization cost, so short-lived processes that never
repeat a prefix lose. The benchmark exists to keep that claim tied to
numbers, and the earlier "re-prefill is faster" claim was an artifact of
the miscalibrated prefix — a measured warning about measuring the workload
you think you measured.

### Pressure (W4): does the benefit survive unrelated traffic?

Store capped at 3 published entries, reaper every 1 s, probe = the prefix re-sent after k unrelated requests (quote the row's `repeat_declared`/`repeat_served` artifact fields for exact values; at this construction the prime-anchored probe serves its full declared LCP, while the extension-anchored arm serves seven below it once the extension anchors the chain):

| arm | k | evictions seen | probe verdict | probe served | probe TTFT (ms) |
|---|---|---|---|---|---|
| p_alone | 0 | 0 | partial | 1,959 of 1,959 declared | 766 |
| p_alone | 1–2 | 0 | partial | 1,959 of 1,959 | 706–711 |
| p_alone | 4 | 1 | miss | 0 of 1,959 | 3,097–3,122 |
| p_alone | 8 | 6 | miss | 0 | 3,097 |
| p_alone | 16 | 14 | miss | 0 | 3,160 |
| p_extended | 0–1 | 0 | miss | 0 of 1,966 | 3,155–3,235 |
| p_extended | 2 | 1 | partial | 1,959 of 1,966 | 897 |
| p_extended | 4 | 2 | miss | 0 | 3,099 |
| p_extended | 8 | 6 | miss | 0 | 3,063 |
| p_extended | 16 | 15 | miss | 0 | 3,219 |

- **p_alone**: the prefix is the longest entry, and the reaper scores biggest-first, so once the cap binds (k ≥ 4) the highest-value blob is the first to go. The benefit does not degrade gracefully; it collapses to a full cold prefill (~3.1 s vs ~0.7 s served).
- **p_extended**: publishing an extension makes the prefix an anchor. It buys survival exactly once the reaper has run (k = 2: the extension is evicted, the anchored prefix survives and still serves 1,959 of 1,966). At k ≥ 4 the chain breaks — with the extension gone the prefix loses its anchor, and the next sweep takes the prefix too.
- The k = 0/k = 1 p_extended probe misses are the shadowing rule, not eviction (0 evictions observed): the freshly published extension sits deeper on the greedy walk than the prefix-only entry, so the probe reads a miss until the first sweep removes the extension.
- Net answer to "does the benefit hold under cache pressure": **not under a tight cap**. The reaper's own scoring order targets exactly the blob the workload cares about, and anchoring delays but does not prevent the loss.
