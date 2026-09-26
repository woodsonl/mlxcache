"""mlxcache sidecar: mlx-lm compatibility adapter.

Speaks the daemon's sidecar protocol over HTTP. Never the hot-path default:
the owner engine implements the cache contract natively; this sidecar exists
so existing mlx-lm users get daemon benefits.
"""

from __future__ import annotations
