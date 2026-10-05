"""The mlxcache_serve CLI: mlx-lm's server with a persistent prompt cache.

Mechanism: swap the ``LRUPromptCache`` symbol in
``mlx_lm.server`` for a factory that wires our ``PersistentPromptCache``
around one shared ``Store``, then hand control to mlx-lm's own ``main``.
No mlx-lm source is modified; the swap is two module attributes (and
``sys.argv``) restored on exit.

The second swap wraps ``APIHandler.do_GET`` to serve ``/mlxcache/stats``:
the cache's reuse counters, so a caller can observe that the disk tier
actually served a request rather than only that the request was answered.
At temperature 0 a full prefill and a disk resume are token-identical, so
response equality cannot distinguish them; this counter can.

Known posture (documented trade-offs):
- ONE wrapper server per process: the symbol swaps are not reentrant (a
  second ``main()`` in the same interpreter would race the restores).
- The fingerprint pins the model DIRECTORY realpath + tokenizer bytes;
  in-place weight overwrites (HF snapshot refresh) keep the old identity.
- Persistence is synchronous on mlx-lm's generation thread (v1 posture);
  a background-writer queue is future work if serve-path stalls show up.
"""

from __future__ import annotations

import argparse
import json
import sys

from mlxcache_store import DEFAULT_BUDGET, Store

from .cache import PersistentPromptCache, default_fingerprint_for


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mlxcache_serve",
        description="mlx-lm server with persistent KV reuse (connector protocol L1). "
        "Unknown flags pass through to mlx_lm.server.",
        add_help=False,
        allow_abbrev=False,  # --model must NOT abbreviate to --model-id-override
    )
    p.add_argument("--store-dir", required=True, help="mlxcache store directory")
    p.add_argument(
        "--store-budget-bytes",
        type=int,
        default=DEFAULT_BUDGET,
        help=f"store byte budget (default {DEFAULT_BUDGET})",
    )
    p.add_argument(
        "--anchor-window-s",
        type=float,
        default=900.0,
        help="recency anchor window in seconds (protocol §7)",
    )
    p.add_argument("--trust-legacy", action="store_true", help="index digest-less v1 blobs")
    p.add_argument(
        "--kv-bits",
        type=int,
        default=0,
        help="KV tier label for the fingerprint (0 = f16); labels the store "
        "partition this server reads/writes — mlx-lm's server itself serves f16",
    )
    p.add_argument("--kv-group-size", type=int, default=0, help="KV quantization group size")
    p.add_argument("-h", "--help", action="help", help="show this help message and exit")
    return p


def _stats_payload(response_generator) -> dict:
    """The reuse counters of the live prompt cache, or nulls if the handler
    holds something that is not a PersistentPromptCache."""
    cache = getattr(response_generator, "prompt_cache", None)
    return {
        "disk_hits": getattr(cache, "disk_hits", None),
        "persisted": getattr(cache, "persisted", None),
        "reuse_errors": getattr(cache, "reuse_errors", None),
        "persist_errors": getattr(cache, "persist_errors", None),
    }


def _write_stats(handler) -> None:
    body = json.dumps(_stats_payload(handler.response_generator)).encode()
    # Match every other route on this server: without the CORS headers the
    # route is unreachable from a browser while /health and /v1/models are
    # not, and a caller debugging from a web console would see a silent CORS
    # failure rather than the counters.
    cors = getattr(handler, "_set_cors_headers", None)
    handler.send_response(200)
    handler.send_header("Content-type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    if cors is not None:
        cors()
    handler.end_headers()
    handler.wfile.write(body)
    handler.wfile.flush()


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:]) if argv is None else argv
    args, passthrough = build_parser().parse_known_args(argv)
    # Python 3.14's parse_known_args KEEPS a leading "--" in extras; mlx-lm's
    # strict parser would exit 2 on it. Strip exactly one separator of ours.
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    store = Store(
        args.store_dir,
        byte_budget=args.store_budget_bytes,
        anchor_window_s=args.anchor_window_s,
        trust_legacy=args.trust_legacy,
    )

    def fingerprint(model_key):
        # model_key is mlx-lm's (model_path, adapter_path, draft_model_path):
        # adapter + draft model fold into the model id (LoRA isolation).
        return default_fingerprint_for(
            model_key,
            kv_bits=args.kv_bits,
            kv_group_size=args.kv_group_size,
        )

    import mlx_lm.server as mlx_server

    original = mlx_server.LRUPromptCache
    original_do_get = mlx_server.APIHandler.do_GET
    saved_argv = sys.argv

    def factory(max_size=10, max_bytes=1 << 63):
        return PersistentPromptCache(store, fingerprint, max_size=max_size, max_bytes=max_bytes)

    def do_get(self):
        # A cache-reuse counter observable over HTTP. Without it a caller can
        # only see that a request was ANSWERED, not that the disk tier served
        # it: at temperature 0 a full prefill and a disk resume are
        # token-identical, so equality alone cannot tell the two apart.
        if self.path == "/mlxcache/stats":
            _write_stats(self)
            return
        return original_do_get(self)

    mlx_server.LRUPromptCache = factory
    mlx_server.APIHandler.do_GET = do_get
    try:
        sys.argv = ["mlx_lm.server", *passthrough]
        mlx_server.main()
        return 0
    finally:
        mlx_server.LRUPromptCache = original
        mlx_server.APIHandler.do_GET = original_do_get
        sys.argv = saved_argv


if __name__ == "__main__":
    raise SystemExit(main())
