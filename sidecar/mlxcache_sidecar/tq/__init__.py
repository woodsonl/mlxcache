"""TurboQuant persistence tier for mlxcache (paper: arXiv:2504.19874).

Implemented from the paper only — no code from any third-party TurboQuant
repository (their license is GPL-3.0; this repository carries no license and
must stay uncontaminated).

See ``codec`` for the frame format and the tier's deliberate scope
(persistence, not live compute), ``codebook`` for the Lloyd-Max derivation,
and ``rotation`` for the pinned-seed basis.
"""

from __future__ import annotations

from mlxcache_sidecar.tq.codebook import lloyd_max_codebook
from mlxcache_sidecar.tq.codec import (
    DEFAULT_GROUP_SIZE,
    FORMAT,
    decode_payload,
    encode_cache,
    parse_tier,
    tier_string,
)
from mlxcache_sidecar.tq.rotation import PINNED_SEED, build_rotation

__all__ = [
    "DEFAULT_GROUP_SIZE",
    "FORMAT",
    "PINNED_SEED",
    "build_rotation",
    "decode_payload",
    "encode_cache",
    "lloyd_max_codebook",
    "parse_tier",
    "tier_string",
]
