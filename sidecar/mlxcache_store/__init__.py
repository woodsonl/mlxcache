"""mlxcache_store — the L1 embedded store client (connector protocol §3).

An engine uses this package to gain persistent, cross-restart, bounded,
integrity-checked prefix KV reuse over an mlxcache store directory —
without the daemon. Same L0 format, same semantics, direct filesystem.

Protocol: docs/connector-protocol.md (v1.0). Conformance:
sidecar/tests/test_store.py + the format-tier suite in
sidecar/tests/test_conformance.py.
"""

from .errors import CorruptCache, Refused, Unavailable
from .fingerprint import Fingerprint, blob_key
from .store import Granularity, Store

__all__ = [
    "CorruptCache",
    "Refused",
    "Unavailable",
    "Fingerprint",
    "blob_key",
    "Granularity",
    "Store",
]
