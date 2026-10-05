"""The do_GET swap in mlxcache_serve.main.

The wrapper intercepts /mlxcache/stats by replacing mlx_lm.server.APIHandler.do_GET
and must (a) serve our path and (b) delegate every other path to the original,
(c) restore both module attributes on exit. Exercised against a fake
mlx_lm.server so no model or HTTP server is involved.
"""

from __future__ import annotations

import io
import json
import sys
import types

import pytest
from mlxcache_serve import main as serve_main


class _FakeHandler:
    def __init__(self, path, response_generator):
        self.path = path
        self.response_generator = response_generator
        self.wfile = io.BytesIO()
        self.calls: list[tuple] = []

    def send_response(self, code):
        self.calls.append(("status", code))

    def send_header(self, k, v):
        self.calls.append((k, v))

    def end_headers(self):
        pass


@pytest.fixture
def fake_mlx_server(monkeypatch):
    mod = types.ModuleType("mlx_lm.server")
    delegated: list[str] = []

    class APIHandler:
        def do_GET(self):  # noqa: N802 — matches mlx_lm.server.APIHandler's name
            delegated.append(self.path)

    class _Srv:
        pass

    def fake_main():
        # main() must have installed the swap by the time it runs; exercise
        # both a stats path and a normal one against the live class.
        h = _FakeHandler(
            "/mlxcache/stats",
            types.SimpleNamespace(
                prompt_cache=types.SimpleNamespace(
                    disk_hits=4, persisted=1, reuse_errors=0, persist_errors=0
                )
            ),
        )
        APIHandler.do_GET(h)
        assert json.loads(h.wfile.getvalue())["disk_hits"] == 4
        APIHandler.do_GET(_FakeHandler("/health", None))
        raise SystemExit(0)

    mod.LRUPromptCache = object()
    mod.APIHandler = APIHandler
    mod.main = fake_main
    monkeypatch.setitem(sys.modules, "mlx_lm.server", mod)
    monkeypatch.setitem(sys.modules, "mlx_lm", types.ModuleType("mlx_lm"))
    return mod, APIHandler, delegated, APIHandler.do_GET, mod.LRUPromptCache


def test_stats_route_served_and_other_paths_delegated(fake_mlx_server):
    mod, handler_cls, delegated, original, original_cache = fake_mlx_server
    with pytest.raises(SystemExit):
        serve_main(["--store-dir", "/tmp/does-not-matter-mlxcache"])
    # The original handler saw the non-stats path, not the stats one.
    assert "/health" in delegated and "/mlxcache/stats" not in delegated
    # Both module attributes restored (by identity) after main() exits.
    assert handler_cls.do_GET is original
    assert mod.LRUPromptCache is original_cache
