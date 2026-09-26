"""Sidecar tests (pytest, per R6 test-stack decision)."""


def test_package_imports():
    import mlxcache_sidecar
    import mlxcache_sidecar.server  # noqa: F401

    assert mlxcache_sidecar is not None
