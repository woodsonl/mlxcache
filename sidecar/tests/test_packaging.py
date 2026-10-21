"""Wheel packaging (CodeRabbit #36 regression guard).

The 0.1.8 wheel shipped only mlxcache_sidecar, so a wheel install could not
start a real engine. The packages list is now asserted from the
BUILT WHEEL, not from the pyproject source: the list can drift (or the hatch
target can change shape) while the declared config still looks right.
"""

import subprocess
import tempfile
import zipfile
from pathlib import Path

REQUIRED = {"mlxcache_sidecar", "mlxcache_store"}


def test_wheel_carries_every_runtime_package():
    repo = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as out:
        # `uv build`, not `python -m uv`: uv installs as a binary, and the
        # venv does not necessarily carry it as a module.
        import shutil

        uv = shutil.which("uv") or str(repo / ".venv" / "bin" / "uv")
        subprocess.run(
            [uv, "build", "--wheel", "--out-dir", out],
            cwd=repo,
            check=True,
            capture_output=True,
        )
        wheels = sorted(Path(out).glob("*.whl"))
        assert wheels, "no wheel was built"
        tops = {n.split("/")[0] for n in zipfile.ZipFile(wheels[-1]).namelist() if "/" in n}
        missing = REQUIRED - tops
        assert not missing, (
            f"wheel {wheels[-1].name} is missing runtime packages {missing}; "
            "a wheel install cannot start the engines that import them"
        )
