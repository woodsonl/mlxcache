"""Version-source agreement.

The ship tooling bumps ``VERSION``; the crate and the Python package carry
their own declared versions. If they drift, a release can ship a binary and a
package that disagree about what they are, and only a reader noticing the
mismatch catches it. This test makes the drift loud at gate time.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _version_file() -> str:
    return (REPO / "VERSION").read_text().strip()


def test_version_file_exists_and_is_semver():
    v = _version_file()
    assert re.fullmatch(r"\d+\.\d+\.\d+", v), f"VERSION is not x.y.z: {v!r}"


def test_cargo_workspace_version_matches_version_file():
    data = tomllib.loads((REPO / "Cargo.toml").read_text())
    assert data["workspace"]["package"]["version"] == _version_file()


def test_pyproject_version_matches_version_file():
    data = tomllib.loads((REPO / "pyproject.toml").read_text())
    assert data["project"]["version"] == _version_file()
