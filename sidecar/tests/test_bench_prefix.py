"""Premise guards for scripts/bench_prefix.py.

The prefix benchmark's whole claim is "declared reuse vs served reuse". Each
test here pins one way that claim can become fiction:

  * the declared prefix must be computed on the stream the daemon actually
    tokenizes (its serialization of the messages array), not on the raw text;
  * the arithmetic that turns samples into a rate;
  * the three premises that abort a run (serialization drift, served != held,
    served > declared);
  * the workload shapes themselves — W2 must diverge EARLY (a tail-only
    difference is a legitimate hit under the end-anchored serve rule, so a
    late divergence would silently turn the near-miss case into a hit).

No model, no subprocess, no GPU.
"""

from __future__ import annotations

import http.server
import importlib.util
import json
import os
import threading
import time
from pathlib import Path

import pytest


def _load(name: str):
    path = Path(__file__).resolve().parents[2] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bench = _load("bench_prefix")


# ---------------------------------------------------------------------------
# Serialization parity
# ---------------------------------------------------------------------------


def test_serialize_messages_is_compact_serde_shape():
    """The daemon tokenizes `serde_json::to_string(&messages)`: compact
    separators, no spaces, `"` escaped. Any other shape declares a prefix the
    daemon never sees."""
    out = bench.serialize_messages(
        [{"role": "user", "content": 'he said "hi"'}, {"role": "assistant", "content": "ok"}]
    )
    assert out == (
        '[{"role":"user","content":"he said \\"hi\\""},{"role":"assistant","content":"ok"}]'
    )


def test_serialize_messages_keeps_ascii_unwritten():
    """serde emits non-ASCII characters raw (no \\uXXXX), so escaping them here
    would shift every token after the first non-ASCII char."""
    out = bench.serialize_messages([{"role": "user", "content": "café"}])
    assert "café" in out
    assert "\\u" not in out


def test_serialized_prefix_is_a_byte_prefix_of_the_shared_text():
    """The whole W1 construction depends on the varying question being APPENDED
    last: a fixed prefix plus a short question must produce one serialized
    string that starts with the fixed part's bytes."""
    prefix = "shared policy text"
    a = bench.serialize_messages(
        [{"role": "system", "content": prefix}, {"role": "user", "content": "question one"}]
    )
    b = bench.serialize_messages(
        [{"role": "system", "content": prefix}, {"role": "user", "content": "question two"}]
    )
    common = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        common += 1
    # Shared through the prefix plus the JSON scaffolding that precedes the
    # question, minus nothing: the divergence is the question's first byte.
    assert common > len(prefix)
    assert a[:common] == b[:common]
    assert a[common] != b[common]


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------


def test_lcp_len_basic():
    assert bench.lcp_len([1, 2, 3], [1, 2, 9]) == 2
    assert bench.lcp_len([1, 2, 3], [1, 2, 3]) == 3
    assert bench.lcp_len([], [1]) == 0
    assert bench.lcp_len([5], [5, 6]) == 1


def test_tpot_divides_by_generated_tokens():
    """TPOT is (total - ttft) over the tokens ACTUALLY generated. A constant
    divisor drifts the moment max_tokens changes or the model stops at EOS."""
    assert bench.tpot_ms(1100.0, 100.0, 11) == 100.0
    # One token: no inter-token gap exists, so the whole decode time is it.
    assert bench.tpot_ms(1100.0, 100.0, 1) == 1000.0


def test_pct_is_none_on_empty_and_bounded():
    assert bench.pct([], 50) is None
    assert bench.pct([5.0], 50) == 5.0
    assert bench.pct([1.0, 2.0, 3.0], 99) == 3.0


def test_p95_is_withheld_below_the_sample_floor():
    """A p95 over 6 samples is the largest sample in a costume."""
    samples = [
        {
            "label": f"q{i}",
            "prompt_tokens": 100,
            "declared_lcp": 0,
            "ttft_ms": 10.0 + i,
            "total_ms": 100.0 + i,
            "tokens_out": 10,
            "tpot_ms": 10.0,
            "verdict": "miss",
            "served": 0,
            "daemon_tokens_total": 100,
            "lookup_ms": 1,
        }
        for i in range(6)
    ]
    row = bench.summarize("W-test", "daemon/cold", samples)
    assert row["ttft_p95_ms"] is None
    assert row["ttft_p50_ms"] is not None


# ---------------------------------------------------------------------------
# The premises that abort a run
# ---------------------------------------------------------------------------


def _sample(**over):
    base = {
        "label": "q",
        "prompt_tokens": 100,
        "declared_lcp": 90,
        "served": 89,
        "held": 89,
        "trace_n_tokens": 100,
    }
    base.update(over)
    return base


def test_valid_sample_passes_every_premise():
    assert bench.check_premises([_sample()]) == []


def test_served_greater_than_declared_is_a_violation():
    """The daemon cannot serve reuse the traffic never offered. If this fires,
    the declared side is under-counting (serialization drift) or the served
    side is inflated — either way the run is not a result."""
    problems = bench.check_premises([_sample(served=91, held=91)])
    assert any("reuse invented" in p for p in problems)


def test_served_must_equal_held():
    """`prefill_from` and `matched_tokens - 1` are the same quantity by
    definition (`covered_kv_tokens`). A split means the response body and the
    trace disagree about one request."""
    problems = bench.check_premises([_sample(held=88)])
    assert any("served 89 != held 88" in p for p in problems)


def test_serialization_drift_is_a_violation():
    """If the trace says the daemon tokenized a different number of tokens than
    our serialization produced, every declared prefix in the run is measured
    against the wrong stream."""
    problems = bench.check_premises([_sample(trace_n_tokens=101)])
    assert any("serialization drift" in p for p in problems)


def test_drift_check_can_be_relaxed_for_the_direct_arm():
    """The direct arm tokenizes with the model's chat template, so its stream
    legitimately differs from the daemon's JSON serialization."""
    assert bench.check_premises([_sample(trace_n_tokens=7)], strict_serialization=False) == []


# ---------------------------------------------------------------------------
# Workload shapes
# ---------------------------------------------------------------------------


def test_w1_prime_request_is_the_prefix_alone():
    """The serve rule only helps W1 if the prefix itself was published: the
    next request diverges one token past the prime's last token, which is the
    boundary the index's child rule serves. Change the prime shape and W1
    silently becomes W1np."""
    wl = bench.build_w1("P" * 50, 3, "W1")
    prime = wl["requests"][0]
    assert prime["messages"] == [{"role": "system", "content": "P" * 50}]
    # Every measured request carries the prefix AND a distinct question.
    questions = {r["messages"][-1]["content"] for r in wl["requests"][1:]}
    assert len(questions) == 3


def test_w1np_has_no_prime_request():
    """W1np is the honest production shape: the traffic never publishes the
    prefix by itself, so it must contain no prefix-only request."""
    wl = bench.build_w1_unprimed("P" * 50, 4)
    assert len(wl["requests"]) == 4
    for r in wl["requests"]:
        assert len(r["messages"]) == 2
        assert r["messages"][1]["role"] == "user"


def test_w2_shares_a_long_region_but_diverges_before_the_end():
    """A near miss must share a LONG region yet diverge well before either
    document's end: a tail-only difference is a legitimate hit under the
    end-anchored serve rule, while zero sharing (the old shape) measures
    disjoint traffic and calls it a near miss. Half the prefix shared,
    divergence mid-document — the shape the docs' 'similar text, different
    document' claim describes."""
    prefix = bench.filler_text(200, 26)
    wl = bench.build_w2(prefix, 2)
    a = wl["requests"][0]["messages"][0]["content"]
    b = wl["requests"][1]["messages"][0]["content"]
    assert a != b
    common = len(os.path.commonprefix([a, b]))
    assert common > len(a) // 3, f"near-miss shares only {common} of {len(a)} chars"
    assert common < len(a) - 40, "divergence must come before the document end"


def test_w3_is_decode_bound():
    """W3 answers must be long enough that decode dominates prefill; otherwise
    the row cannot separate a decode regression from a cache win."""
    wl = bench.build_w3(2)
    assert wl["max_tokens"] >= 128
    assert len(wl["requests"]) == 2


def test_w4_keeps_the_probe_out_of_the_noise_phase():
    """The reaper must land BETWEEN the noise and the measured repeat. If the
    probe rode along in `requests`, the row would measure the pre-eviction
    cache and label it a pressure result."""
    wl = bench.build_w4("P" * 50, 4, "p_alone")
    labels = [r["label"] for r in wl["requests"]]
    assert wl["probe"]["label"] not in labels
    assert labels[0].endswith(":prime")
    assert sum(1 for label in labels if ":noise" in label) == 4


def test_w4_extended_arm_publishes_an_extension_of_the_prefix():
    """`p_extended` only anchors the prefix if it also publishes a request that
    EXTENDS it — that is what makes the prefix a structural anchor and keeps
    the reaper off it."""
    alone = bench.build_w4("P" * 50, 2, "p_alone")
    extended = bench.build_w4("P" * 50, 2, "p_extended")
    assert len(extended["requests"]) == len(alone["requests"]) + 1
    ext = extended["requests"][1]
    assert ext["messages"][0]["content"] == "P" * 50
    assert len(ext["messages"]) == 2


# ---------------------------------------------------------------------------
# Trace joining
# ---------------------------------------------------------------------------


def test_join_trace_slices_from_the_leg_offset(tmp_path):
    """A restart leg reuses one trace file; joining from record 0 would attribute
    the previous process's requests to this one."""
    trace = tmp_path / "t.jsonl"
    records = [
        {"n_tokens": 5, "verdict": "miss", "matched_tokens": 0, "prefill_from": 0},
        {"n_tokens": 9, "verdict": "partial", "matched_tokens": 8, "prefill_from": 7},
    ]
    trace.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    samples = [{"label": "q", "prompt_tokens": 9, "declared_lcp": 8, "served": 7}]
    bench.join_trace(samples, str(trace), base_n=1)
    assert samples[0]["trace_n_tokens"] == 9
    assert samples[0]["held"] == 7


def test_join_trace_treats_a_miss_as_zero_held(tmp_path):
    """A miss has matched_tokens == 0; `held = matched - 1` would go negative
    and silently satisfy the served == held premise."""
    trace = tmp_path / "t.jsonl"
    trace.write_text(
        json.dumps({"n_tokens": 5, "verdict": "miss", "matched_tokens": 0, "prefill_from": 0})
        + "\n"
    )
    samples = [{"label": "q", "prompt_tokens": 5, "declared_lcp": 0, "served": 0}]
    bench.join_trace(samples, str(trace), base_n=0)
    assert samples[0]["held"] == 0
    assert bench.check_premises(samples) == []


def test_join_trace_refuses_a_short_capture(tmp_path):
    """A record still in flight when the leg ended must abort the join, not
    shift every subsequent sample onto the wrong record."""
    trace = tmp_path / "t.jsonl"
    trace.write_text(json.dumps({"n_tokens": 5, "verdict": "miss", "matched_tokens": 0}) + "\n")
    samples = [{"label": "a", "prompt_tokens": 5, "declared_lcp": 0, "served": 0}] * 2
    with pytest.raises(RuntimeError, match="trace has 1 records") as excinfo:
        bench.join_trace(samples, str(trace), base_n=0, timeout_s=0.2)
    assert "1 records for 2" in str(excinfo.value)


# ---------------------------------------------------------------------------
# The synthetic engine cannot express prefix reuse
# ---------------------------------------------------------------------------


def test_unknown_workload_and_arm_ids_are_refused(monkeypatch):
    """A typo like --workloads W5 (a cross-check, not a workload) or --arms
    bench must fail loudly: the old membership checks silently ran nothing and
    wrote an empty, green-looking artifact."""
    monkeypatch.setattr(
        bench,
        "resolve_model",
        lambda _m: pytest.fail("validation must run before resolving a model"),
    )
    monkeypatch.setattr("sys.argv", ["bench_prefix.py", "--workloads", "W5"])
    with pytest.raises(SystemExit, match=r"unknown workload\(s\) \['W5'\]"):
        bench.main()
    monkeypatch.setattr("sys.argv", ["bench_prefix.py", "--arms", "bench"])
    with pytest.raises(SystemExit, match="bench"):
        bench.main()


# ---------------------------------------------------------------------------
# Preflight: a full disk must not read as "the cache does nothing"
# ---------------------------------------------------------------------------


def test_disk_headroom_refuses_a_run_that_cannot_fit_its_checkpoints(monkeypatch):
    """A failed publish is reported as a plain miss, by design (OV6: a
    follower whose request was fine must never 502). That makes a full disk
    indistinguishable from a cache that simply never helps, so the run has to
    refuse up front instead of producing a confident zero."""
    monkeypatch.setattr(
        bench.shutil,
        "disk_usage",
        lambda _p: type("U", (), {"free": 100 * 1024 * 1024})(),
    )
    problems = bench.check_disk_headroom("/tmp", 3 * 1024**3)
    assert problems and "only 0.10 GB free" in problems[0]


def test_disk_headroom_allows_a_run_with_room(monkeypatch):
    monkeypatch.setattr(
        bench.shutil,
        "disk_usage",
        lambda _p: type("U", (), {"free": 100 * 1024**3})(),
    )
    assert bench.check_disk_headroom("/tmp", 3 * 1024**3) == []


# ---------------------------------------------------------------------------
# Stream parsing: the benchmark's servers speak three SSE dialects
# ---------------------------------------------------------------------------


class _SSEHandler(http.server.BaseHTTPRequestHandler):
    frames: tuple = ()

    def do_POST(self):
        payload = b"".join(f"data: {json.dumps(f)}\n\n".encode() for f in self.frames)
        payload += b"data: [DONE]\n\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


def _sse_base(frames):
    handler = type("handler", (_SSEHandler,), {"frames": frames})
    srv = http.server.HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{srv.server_port}", srv


def test_generated_text_recognizes_every_frame_dialect():
    """Daemon {"token"}, chat delta.content, completions text — and the two
    non-token frames (verdict extra, role-only delta) that must NOT count."""
    assert bench._generated_text({"token": "a"}) == "a"
    assert bench._generated_text({"choices": [{"delta": {"content": "He"}}]}) == "He"
    assert bench._generated_text({"choices": [{"text": "ab"}]}) == "ab"
    assert bench._generated_text({"choices": [{"delta": {"role": "assistant"}}]}) is None
    assert bench._generated_text({"mlxcache": {"verdict": "hit"}}) is None
    assert bench._generated_text({}) is None


def test_post_stream_counts_daemon_token_frames():
    """The daemon's dialect: a leading mlxcache verdict extra (merged into
    meta, not counted) followed by one frame per token."""
    base, srv = _sse_base([{"mlxcache": {"verdict": "hit"}}, {"token": "a"}, {"token": "b"}])
    try:
        timing, meta = bench._post_stream(base, "/v1/chat/completions", {}, 10)
        assert timing["tokens"] == 2
        assert meta == {"verdict": "hit"}
    finally:
        srv.shutdown()


def test_post_stream_counts_openai_chat_delta_frames():
    """The first benchmarked run counted only the daemon's frame shape, so
    every direct-arm row reported 0 tokens and its "TPOT" was really
    end-to-end minus TTFT. The role-only opener must not count as a token."""
    base, srv = _sse_base(
        [
            {"choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {"content": "He"}}]},
            {"choices": [{"delta": {"content": "llo"}}]},
        ]
    )
    try:
        timing, meta = bench._post_stream(base, "/v1/chat/completions", {}, 10)
        assert timing["tokens"] == 2
        assert timing["ttft_ms"] > 0
        assert meta is None
    finally:
        srv.shutdown()


def test_post_stream_counts_completions_text_frames():
    """The direct arm's raw /v1/completions baseline streams choices[].text."""
    base, srv = _sse_base([{"choices": [{"text": "ab"}]}, {"choices": [{"text": "cd"}]}])
    try:
        timing, _ = bench._post_stream(base, "/v1/completions", {}, 10)
        assert timing["tokens"] == 2
    finally:
        srv.shutdown()


def test_post_stream_raises_on_error_frame():
    base, srv = _sse_base([{"error": {"message": "boom"}}])
    try:
        with pytest.raises(RuntimeError, match="stream error frame"):
            bench._post_stream(base, "/v1/chat/completions", {}, 10)
    finally:
        srv.shutdown()


def test_pressure_flow_joins_the_trace_before_checking_premises(tmp_path):
    """The pressure arm once called check_premises on run_leg output before
    joining the trace, so every sample lacked 'held' and the whole arm died
    with KeyError. This mirrors the fixed order: run_leg shape -> join -> check."""
    trace = tmp_path / "trace.jsonl"
    records = [
        {"n_tokens": 10, "matched_tokens": 0, "verdict": "miss"},
        {"n_tokens": 12, "matched_tokens": 11, "verdict": "partial"},
    ]
    trace.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    samples = [
        {"label": "noise", "prompt_tokens": 10, "declared_lcp": 0, "served": 0, "verdict": "miss"},
        {
            "label": "probe",
            "prompt_tokens": 12,
            "declared_lcp": 11,
            "served": 10,
            "verdict": "partial",
        },
    ]
    bench.join_trace(samples, str(trace), 0)
    assert bench.check_premises(samples) == []


# ---------------------------------------------------------------------------
# Gate findings: run_leg, meta merge, exact-count join, served-required
# ---------------------------------------------------------------------------


def test_run_leg_declares_against_the_threaded_corpus():
    """run_leg produces every number the benchmark claims; its declared-LCP
    computation over a corpus threaded across legs is the denominator of the
    headline 'served 99.97% of declared' figure, so the corpus must grow in
    order and declared must be the max over corpus AND earlier in-leg
    requests."""
    base, srv = _sse_base([{"token": "a"}])

    class Stub:
        pass

    Stub.base = base
    Stub.completion_request = lambda self, messages, max_tokens: (
        "/v1/chat/completions",
        {"stream": True},
    )
    try:
        corpus = [[1, 2, 3]]  # a cold leg's streams
        wl = {
            "max_tokens": 1,
            "requests": [
                {"label": "a", "messages": [{"role": "user", "content": "x"}]},
                {"label": "b", "messages": [{"role": "user", "content": "y"}]},
            ],
        }
        toks = {"x": [1, 2, 3, 9], "y": [1, 2, 8]}
        samples = bench.run_leg(Stub(), wl, lambda m: toks[m[0]["content"]], corpus=corpus)
        assert samples[0]["declared_lcp"] == 3  # vs the cold-leg stream
        assert samples[1]["declared_lcp"] == 2  # max over corpus and in-leg
        assert corpus == [[1, 2, 3], [1, 2, 3, 9], [1, 2, 8]]
    finally:
        srv.shutdown()


def test_post_stream_merges_leading_verdict_and_trailing_timing_extras():
    """The daemon emits TWO mlxcache extras per stream; a last-write-wins
    regression would erase the verdict and report every request as n/a with
    premises green (None-skip)."""
    base, srv = _sse_base(
        [
            {"mlxcache": {"verdict": "partial", "prefill_from": 100}},
            {"token": "a"},
            {"mlxcache": {"ttft_ms": 5.0, "tokens_total": 101}},
        ]
    )
    try:
        timing, meta = bench._post_stream(base, "/v1/chat/completions", {}, 10)
        assert timing["tokens"] == 1
        assert meta == {
            "verdict": "partial",
            "prefill_from": 100,
            "ttft_ms": 5.0,
            "tokens_total": 101,
        }
    finally:
        srv.shutdown()


def test_post_stream_cross_checks_the_server_usage_count():
    """Every text-bearing frame carries at least one token, so the server's
    completion-token count may exceed the frame count (multi-token chunks)
    but never fall below it; where the server count exists it becomes the
    token count TPOT divides by."""
    base, srv = _sse_base(
        [
            {"choices": [{"delta": {"content": "He"}}]},
            {"choices": [{"delta": {"content": "llo"}}]},
            {"usage": {"completion_tokens": 3}},
        ]
    )
    try:
        timing, _ = bench._post_stream(base, "/v1/chat/completions", {}, 10)
        assert timing["tokens"] == 3  # server count wins over the 2 frames
    finally:
        srv.shutdown()

    base2, srv2 = _sse_base(
        [
            {"choices": [{"delta": {"content": "a"}}]},
            {"choices": [{"delta": {"content": "b"}}]},
            {"usage": {"completion_tokens": 1}},
        ]
    )
    try:
        with pytest.raises(RuntimeError, match="token accounting has drifted"):
            bench._post_stream(base2, "/v1/chat/completions", {}, 10)
    finally:
        srv2.shutdown()


def test_join_trace_refuses_an_overlength_capture(tmp_path):
    """An extra record would shift every later sample onto its predecessor's
    record (positional join): the count must match EXACTLY, not merely be at
    least as long."""
    trace = tmp_path / "t.jsonl"
    rec = {"n_tokens": 5, "matched_tokens": 0, "verdict": "miss"}
    trace.write_text(json.dumps(rec) + "\n" + json.dumps(rec) + "\n")
    samples = [
        {"label": "a", "prompt_tokens": 5, "declared_lcp": 0, "served": 0},
    ]
    with pytest.raises(RuntimeError, match="2 records for 1"):
        bench.join_trace(samples, str(trace), 0, timeout_s=0.2)


def test_join_trace_short_capture_is_fast(tmp_path):
    """Thread the timeout so the short-capture refusal does not burn the
    20s default on every CI run."""
    import time as _time

    trace = tmp_path / "t.jsonl"
    trace.write_text(json.dumps({"n_tokens": 5, "verdict": "miss", "matched_tokens": 0}) + "\n")
    samples = [
        {"label": "a", "prompt_tokens": 5, "declared_lcp": 0, "served": 0},
    ] * 2
    t0 = _time.monotonic()
    with pytest.raises(RuntimeError, match="1 records for 2"):
        bench.join_trace(samples, str(trace), 0, timeout_s=0.2)
    assert _time.monotonic() - t0 < 5


def test_trace_joined_sample_missing_served_meta_is_a_violation():
    """A daemon-leg sample that lost its meta extras would read as a
    premise-green 0% reuse — the confident-zero failure the disk preflight
    exists to refuse."""
    joined_no_meta = {
        "label": "a",
        "prompt_tokens": 10,
        "declared_lcp": 5,
        "trace_n_tokens": 10,
    }
    problems = bench.check_premises([joined_no_meta])
    assert any("lost its served meta" in p for p in problems)


def test_served_without_held_is_a_violation():
    bad = {"label": "a", "prompt_tokens": 10, "declared_lcp": 5, "served": 3}
    assert any("no held value" in p for p in bench.check_premises([bad]))


def test_direct_arm_samples_without_meta_pass_relaxed_premises():
    """The direct arm's samples carry no instrumentation: no trace_n_tokens,
    no served — the relaxation exists for exactly this shape."""
    plain = {"label": "a", "prompt_tokens": 10, "declared_lcp": 0, "served": None}
    assert bench.check_premises([plain], strict_serialization=False) == []
    assert bench.check_premises([plain]) == []


def test_summarize_reports_na_when_no_request_was_instrumented():
    """An uninstrumented leg (direct baseline) is NOT a measured zero: the
    reuse rates must be None, not 0.0, or the table invites a false
    daemon-vs-baseline reuse comparison."""
    plain = [
        {
            "label": "a",
            "prompt_tokens": 100,
            "declared_lcp": 90,
            "ttft_ms": 100.0,
            "total_ms": 200.0,
            "tokens_out": 10,
            "tpot_ms": 10.0,
            "verdict": None,
            "served": None,
        }
    ]
    row = bench.summarize("W1", "direct/restart1", plain)
    assert row["reuse_pct_of_requested"] is None
    assert row["served_pct_of_declared"] is None
    assert row["n_instrumented"] == 0


def test_score_trace_flags_a_served_disagreement(tmp_path):
    """The W5 cross-check's booleans must actually distinguish: a trace whose
    reuse totals differ from the bodies' must carry served_matches_samples
    False (the caller aborts on it)."""
    trace = tmp_path / "t.jsonl"
    trace.write_text(
        json.dumps(
            {
                "n_tokens": 10,
                "matched_tokens": 0,
                "verdict": "miss",
                "prefill_from": 0,
                "messages": [{"role": "user", "content": "q"}],
            }
        )
        + "\n"
    )
    samples = [{"label": "q", "prompt_tokens": 10, "served": 5}]  # bodies say 5 reused
    scored = bench.score_trace(str(trace), samples, offset=0)
    assert scored["samples_served"] == 5
    assert scored["served_matches_samples"] is False
    assert scored["tokens_total_matches_samples"] is True


def test_filler_text_calibrates_and_is_deterministic():
    assert bench.filler_text(16, 16) == bench._FILLER
    assert bench.filler_text(8, 16) == bench._FILLER  # floor of one repeat
    assert bench.filler_text(32, 16) == bench._FILLER * 2
    assert bench.filler_text(200, 26) == bench.filler_text(200, 26)
    with pytest.raises(ValueError, match="tokens_per_repeat"):
        bench.filler_text(100, 0)


def test_post_stream_enforces_its_total_deadline():
    """A stream dripping frames faster than the socket timeout but never
    ending must be stopped by the monotonic total deadline."""
    import http.server
    import threading

    class _Drip(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.close_connection = True
            self.end_headers()
            for _ in range(200):
                self.wfile.write(b'data: {"token": "a"}\n\n')
                self.wfile.flush()
                time.sleep(0.02)

        def log_message(self, *args):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), _Drip)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        with pytest.raises(RuntimeError, match="total budget"):
            bench._post_stream(f"http://127.0.0.1:{srv.server_port}", "/x", {}, timeout=0.3)
    finally:
        srv.shutdown()
