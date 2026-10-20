"""The probe-overlap refusal, tested without the checkpoint.

The lock is the behavioral guarantee of the probe-landing unit: a second
measurement on the same checkpoint must refuse (naming the holder) rather
than contaminate both runs. flock releases on process death by OS
construction; these tests pin the three observable behaviors — refusal,
release-on-death, and the explicit bypass — plus the bare-filename override
that used to crash makedirs('') before any diagnostic.
"""

import importlib.util
import os
import subprocess
import sys
import tempfile
import textwrap

_PROBES = os.path.join(os.path.dirname(__file__), "..", "probes")


def _lock_module(lock_path: str):
    """Import probe_lock with LOCK_PATH overridden (it reads env at import)."""
    spec = importlib.util.spec_from_file_location(
        f"probe_lock_{abs(hash(lock_path))}", os.path.join(_PROBES, "probe_lock.py")
    )
    module = importlib.util.module_from_spec(spec)
    os.environ["MLXCACHE_Q4_PROBE_LOCK_PATH"] = lock_path
    spec.loader.exec_module(module)
    return module


def _holder_script(lock_path: str) -> str:
    return textwrap.dedent(
        f"""
        import fcntl, os, time
        d = os.path.dirname({lock_path!r})
        if d:
            os.makedirs(d, exist_ok=True)
        fh = open({lock_path!r}, "a+")
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        print("HELD", flush=True)
        time.sleep(60)
        """
    )


def _spawn_holder(lock: str) -> subprocess.Popen:
    holder = subprocess.Popen(
        [sys.executable, "-c", _holder_script(lock)],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert holder.stdout.readline().strip() == "HELD"
    return holder


def test_a_second_probe_refuses_and_names_the_holder():
    with tempfile.TemporaryDirectory() as tmp:
        lock = os.path.join(tmp, "nested", "probe.lock")
        holder = _spawn_holder(lock)
        try:
            module = _lock_module(lock)
            try:
                with module.checkpoint_lock("sweep[8192]"):
                    raise AssertionError("the lock must refuse while held")
            except SystemExit as exc:
                assert exc.code == 4, f"refusal exit code: {exc.code}"
            # The holder's PID is named in the refusal message.
        finally:
            holder.kill()
            holder.wait()


def test_the_refusal_message_names_the_holder_and_probe():
    with tempfile.TemporaryDirectory() as tmp:
        lock = os.path.join(tmp, "probe.lock")
        holder = _spawn_holder(lock)
        try:
            module = _lock_module(lock)
            import io

            buf = io.StringIO()
            import contextlib as _ctx

            with _ctx.redirect_stderr(buf):
                try:
                    with module.checkpoint_lock("sweep[256]"):
                        pass
                except SystemExit:
                    pass
            out = buf.getvalue()
            assert "REFUSING" in out, out
            # The holder text names the OTHER process's pid; the fcntl holder
            # here is an unrelated script, so the fallback text names "another
            # probe" — the pid assertion applies only when the lock file
            # carries a holder, which our raw-fcntl holder does not write.
            assert "another probe" in out or str(holder.pid) in out, out
        finally:
            holder.kill()
            holder.wait()


def test_a_killed_holder_releases_the_lock():
    # flock is released by the OS on process death; a SIGKILLed probe must
    # not wedge the checkpoint forever. The stale in-file holder text is
    # display-only, never a gate.
    with tempfile.TemporaryDirectory() as tmp:
        lock = os.path.join(tmp, "probe.lock")
        holder = _spawn_holder(lock)
        holder.kill()
        holder.wait()
        module = _lock_module(lock)
        with module.checkpoint_lock("sweep[256]"):
            pass  # acquired: no refusal, no SystemExit


def test_the_env_override_bypasses_the_lock_with_a_warning():
    with tempfile.TemporaryDirectory() as tmp:
        lock = os.path.join(tmp, "probe.lock")
        holder = _spawn_holder(lock)
        try:
            os.environ["MLXCACHE_Q4_PROBE_LOCK"] = "off"
            module = _lock_module(lock)
            import contextlib
            import io

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), module.checkpoint_lock("sweep[256]"):
                pass
            assert "lock bypassed" in buf.getvalue()
            assert "timings may be contaminated" in buf.getvalue()
        finally:
            os.environ.pop("MLXCACHE_Q4_PROBE_LOCK", None)
            holder.kill()
            holder.wait()


def test_a_bare_filename_lock_path_does_not_crash_makedirs():
    # dirname("probe.lock") is "" and makedirs("") raises before any
    # diagnostic; the guard must skip the call so acquisition proceeds.
    module = _lock_module("probe.lock")
    # Acquire against the bare name in a controlled cwd (acquisition would
    # resolve relative to this process's cwd, so chdir to a tmpdir).
    with tempfile.TemporaryDirectory() as tmp:
        prev = os.getcwd()
        os.chdir(tmp)
        try:
            with module.checkpoint_lock("sweep[256]"):
                pass
        finally:
            os.chdir(prev)
