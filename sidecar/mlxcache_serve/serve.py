"""The mlxcache_serve CLI: mlx-lm's server with a persistent prompt cache.

Mechanism (plan §D4, mechanism A): swap the ``LRUPromptCache`` symbol in
``mlx_lm.server`` for a factory that wires our ``PersistentPromptCache``
around one shared ``Store``, then hand control to mlx-lm's own ``main``.
No mlx-lm source is modified; the swap is one module attribute (and
``sys.argv``) restored on exit.

Known posture (documented trade-offs):
- ONE wrapper server per process: the symbol swap is not reentrant (a
  second ``main()`` in the same interpreter would race the restore).
- The fingerprint pins the model DIRECTORY realpath + tokenizer bytes;
  in-place weight overwrites (HF snapshot refresh) keep the old identity.
- Persistence is synchronous on mlx-lm's generation thread (v1 posture);
  a background-writer queue is future work if serve-path stalls show up.
"""

from __future__ import annotations

import argparse
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
    saved_argv = sys.argv

    def factory(max_size=10, max_bytes=1 << 63):
        return PersistentPromptCache(store, fingerprint, max_size=max_size, max_bytes=max_bytes)

    mlx_server.LRUPromptCache = factory
    try:
        sys.argv = ["mlx_lm.server", *passthrough]
        mlx_server.main()
        return 0
    finally:
        mlx_server.LRUPromptCache = original
        sys.argv = saved_argv


if __name__ == "__main__":
    raise SystemExit(main())
