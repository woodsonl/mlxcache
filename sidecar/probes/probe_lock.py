"""Exclusive access to the NVFP4 checkpoint for timing probes.

The checkpoint is 132.6 GB and the engine streams it, so two probes running at
once do not merely share a CPU: they page the same weights through a machine
that cannot hold them, and the kernel starts swapping. Every duration produced
while that happens is measuring the filesystem, not the model. A run that
overlapped another measured 4.85 s/token where the same step costs 1.4 s alone,
and swap went from zero to 7 GB.

Timings are the product here, so contention has to be refused rather than
detected afterwards: a contaminated number can always be explained away, and
that is how a bad figure ends up in a README. The lock is advisory and
process-wide; it is released by the OS if the probe is killed, so a crashed run
cannot wedge the next one.

Token-identity checks are unaffected by sharing, but they are cheap to hold the
lock for and holding it keeps one rule for every probe that loads the model.

Set `MLXCACHE_Q4_PROBE_LOCK=off` to bypass, for a deliberately overlapped run.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import sys
from collections.abc import Iterator

ENV_KEY = "MLXCACHE_Q4_PROBE_LOCK"
LOCK_PATH = os.environ.get(
    "MLXCACHE_Q4_PROBE_LOCK_PATH",
    os.path.join(os.path.expanduser("~"), ".mlxcache", "q4-checkpoint.lock"),
)


@contextlib.contextmanager
def checkpoint_lock(what: str) -> Iterator[None]:
    """Hold the checkpoint exclusively, or explain why the probe must not run."""
    if os.environ.get(ENV_KEY, "").lower() in {"off", "0", "false", "no"}:
        print(f"lock bypassed ({ENV_KEY}=off); timings may be contaminated", flush=True)
        yield
        return

    # dirname('') is '' for a bare filename override, and makedirs('') raises
    # FileNotFoundError before any diagnostic names the bad path.
    lock_dir = os.path.dirname(LOCK_PATH)
    if lock_dir:
        os.makedirs(lock_dir, exist_ok=True)
    with open(LOCK_PATH, "a+") as fh:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fh.seek(0)
            holder = fh.read().strip() or "another probe"
            print(
                f"REFUSING to run {what}: the checkpoint is held by {holder}.\n"
                "Two probes streaming the same 132 GB checkpoint swap the machine "
                "and every duration they produce is meaningless.\n"
                f"Wait for it to finish, or set {ENV_KEY}=off to overlap on purpose.",
                file=sys.stderr,
                flush=True,
            )
            raise SystemExit(4) from None
        fh.seek(0)
        fh.truncate()
        fh.write(f"{what} (pid {os.getpid()})")
        fh.flush()
        yield
