"""The /mlxcache/stats reuse counter.

A gate that only compares response text cannot prove the disk tier served a
request: at temperature 0 a full prefill and a disk resume are
token-identical. This route exposes the cache's counters so the harnesses
can assert observed reuse.
"""

from __future__ import annotations

import io
import json
import types

from mlxcache_serve.serve import _stats_payload, _write_stats


def test_stats_payload_reads_cache_counters():
    cache = types.SimpleNamespace(disk_hits=3, persisted=7, reuse_errors=1, persist_errors=0)
    gen = types.SimpleNamespace(prompt_cache=cache)
    assert _stats_payload(gen) == {
        "disk_hits": 3,
        "persisted": 7,
        "reuse_errors": 1,
        "persist_errors": 0,
    }


def test_stats_payload_nulls_when_no_persistent_cache():
    # A handler whose cache is not ours (or absent) must report nulls, not
    # crash — the route exists in every wrapped server.
    assert _stats_payload(types.SimpleNamespace())["disk_hits"] is None


class _FakeHandler:
    def __init__(self):
        self.response_generator = types.SimpleNamespace(
            prompt_cache=types.SimpleNamespace(
                disk_hits=5, persisted=2, reuse_errors=0, persist_errors=0
            )
        )
        self.wfile = io.BytesIO()
        self.sent: list[tuple] = []

    def send_response(self, code):
        self.sent.append(("status", code))

    def send_header(self, k, v):
        self.sent.append((k, v))

    def end_headers(self):
        pass


def test_write_stats_emits_200_json_body():
    h = _FakeHandler()
    _write_stats(h)
    assert ("status", 200) in h.sent
    body = json.loads(h.wfile.getvalue())
    assert body["disk_hits"] == 5
