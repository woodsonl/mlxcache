"""Sidecar tests (pytest, per R6 test-stack decision)."""

from mlxcache_sidecar import placeholder


def test_placeholder():
    assert placeholder() == 0
