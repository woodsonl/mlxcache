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
| `W2` | shared prefix that diverges in its first few tokens | near-miss / "similar document" traffic |
| `W3` | distinct short prompts, 256-token answers | decode-bound traffic, where reuse cannot help |
| `W4` | fixed prefix + k unrelated requests between repeats, under a tight entry cap | **does the benefit survive cache pressure?** |
| `W5` | the captured trace of the above, scored with `scripts/replay_trace.py` | trace and `/stats` must agree |

Model, hardware, concurrency (1), input length and output cap are held constant
across workloads. Every workload runs twice: **cold** (empty store, fresh
process) and **warm** (fresh process over the same store) — the cross-restart
tier, which is what a persistent cache is for.

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

# arithmetic only, no model, no disk (runs in CI)
uv run python scripts/bench_prefix.py --self-check
```

Output is a markdown table plus a JSON artifact containing every per-request
sample, the daemon `/stats` snapshot per leg, and the trace-replay cross-check.

## Measured results

Provenance — run 2026-10-19 on Apple M3 Max (Darwin 27.0.0), `mlx-community/Qwen2.5-7B-Instruct-4bit` (mlx 0.32.2, mlx_lm 0.31.3), concurrency 1, ~2,000-token shared prefix (~3,260-token serialized prompts), 6 varying requests per workload, premise guards green:

```bash
uv run python scripts/bench_prefix.py --workloads W1,W1np,W2,W3 --arms daemon,direct \
  --questions 6 --prefix-tokens 2000 --json /tmp/prefix-grid2.json
uv run python scripts/bench_prefix.py --arms pressure --k 0,1,2,4,8,16 \
  --prefix-tokens 2000 --json /tmp/prefix-pressure2.json
```

### Cross-restart reuse (the daemon's own cold vs warm)

| workload | leg | n | TTFT p50 (ms) | TPOT p50 (ms) | served % of requested | verdicts |
|---|---|---|---|---|---|---|
| W1 (primed) | daemon/cold | 7 | 5,317.5 | 13.99 | 14.2 | 6 miss, 1 partial |
| W1 (primed) | daemon/warm | 7 | 222.6 | 13.91 | 99.97 | 7 hit |
| W1np (unprimed) | daemon/cold | 6 | 5,138.4 | 13.09 | 0.0 | 6 miss |
| W1np (unprimed) | daemon/warm | 6 | 189.7 | 13.21 | 99.97 | 6 hit |
| W2 near-miss | daemon/cold | 6 | 5,086.2 | 13.00 | 0.0 | 6 miss |
| W2 near-miss | daemon/warm | 6 | 193.3 | 12.90 | 99.97 | 6 hit |
| W3 decode-heavy | daemon/cold | 6 | 153.7 | 11.89 | 0.0 | 6 miss |
| W3 decode-heavy | daemon/warm | 6 | 91.6 | 12.01 | 95.0 | 6 hit |

What the grid says:

1. **Cross-restart reuse is real**: warm TTFT is 24–27× the cold leg (5.1–5.3 s → 0.19–0.22 s) with 99.97% of requested prefix tokens served after both processes were killed and restarted from disk. No in-process cache can do this tier at all.
2. **TPOT is flat (~12–14 ms) cold vs warm everywhere.** Caching moves prefill, not decode; W3 exists to keep that measured rather than assumed.
3. **Early divergence is never served**: W2's near-miss traffic gets exactly 0% cold, as the end-anchored rule requires.
4. **Priming matters once**: W1's first question is served ~2,000 tokens because the prefix was published alone; W1np's first question gets 0. After each request publishes its own full-length key, both warm legs converge to 99.97%.

### Against a raw local engine (the honest baseline)

The direct arm sends the daemon's exact serialization through `/v1/completions`, so both engines consume identical bytes (the daemon applies no chat template — it tokenizes `serde_json::to_string(messages)`).

| leg | TTFT p50 (ms) |
|---|---|
| direct restart (cold prefill every time) | 154–172 |
| direct inproc (mlx_lm LRU prompt cache) | 101 |
| daemon cold (lookup + prefill + publish) | 5,318 |
| daemon warm (blob load + KV adopt, cross-restart) | 223 |

At a 2K-token prefix on this hardware, plain mlx_lm re-prefills the whole 3,260-token prompt in ~155 ms. The daemon's warm path costs ~223 ms — **slower than re-prefilling** — and its cold path pays ~5.3 s, dominated by publishing a ~190 MB checkpoint (~58 KB/token, measured 56.7 MB at 972 tokens). The honest conclusion: at short prefixes on a fast local engine, prefix caching here buys restart-proof reuse, not latency. It pays when prefill is expensive (much longer contexts, a contended or remote engine, cold weights), and the benchmark exists to keep that claim tied to numbers.

### Pressure (W4): does the benefit survive unrelated traffic?

Store capped at 3 published entries, reaper every 1 s, probe = the prefix re-sent after k unrelated requests (declared reuse 3,259 of 3,259 tokens):

| arm | k | evictions seen | probe verdict | probe served | probe TTFT (ms) |
|---|---|---|---|---|---|
| p_alone | 0–2 | 0 | partial | 3,259 | 932–1,142 |
| p_alone | 4 | 1 | miss | 0 | 5,004 |
| p_alone | 8 | 6 | miss | 0 | 5,093 |
| p_alone | 16 | 14 | miss | 0 | 6,517 |
| p_extended | 0–2 | 0–1 | partial | 3,259 | 906–1,306 |
| p_extended | 4 | 2 | miss | 0 | 5,154 |
| p_extended | 8–16 | 6–14 | miss | 0 | 5,104–5,227 |

- **p_alone**: the prefix is the longest entry, and the reaper scores biggest-first, so once the cap binds (k ≥ 4) the highest-value blob is the first to go. The benefit does not degrade gracefully; it collapses to a full cold prefill.
- **p_extended**: publishing an extension makes the prefix an anchor and buys survival through k = 2 (the reaper takes the extension instead). At k ≥ 4 the chain breaks — evicting the extension un-anchors the prefix, and the next sweep takes the prefix too.
- Net answer to "does the benefit hold under cache pressure": **not under a tight cap**. The reaper's own scoring order targets exactly the blob the workload cares about, and anchoring delays but does not prevent the loss.
