"""mlxcache_serve — run any mlx-lm server with persistent KV reuse.

A drop-in wrapper around `mlx_lm.server` (protocol §8, tier L1): the
server's in-memory prompt cache becomes persistent, bounded, and
integrity-checked across restarts via `mlxcache_store`.

    python -m mlxcache_serve --store-dir ~/.mlxcache/store --model <model> [mlx-lm flags]

The blob convention matches the daemon exactly (a payload at token
stream T covers T[:-1]; resume feeds T[-1:]), and the default engine id
contributes zero fold bytes, so a directory is shareable with the
daemon's store.
"""

from .cache import (
    ENGINE_ID,
    PersistentPromptCache,
    default_fingerprint_for,
    fingerprint_for_model_dir,
)
from .serve import main

__all__ = [
    "ENGINE_ID",
    "PersistentPromptCache",
    "default_fingerprint_for",
    "fingerprint_for_model_dir",
    "main",
]
