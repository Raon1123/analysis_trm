"""Shared helpers for the harness hooks. stdlib only; kept 3.8-compatible; host python3 is 3.12."""
from __future__ import annotations

import json
import os
import subprocess
import sys


def project_dir() -> str:
    env = os.environ.get("CLAUDE_PROJECT_DIR")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", ".."))


def disabled() -> bool:
    if os.environ.get("HARNESS_HOOKS") == "0":
        return True
    try:
        return os.path.exists(os.path.join(project_dir(), ".claude", "harness_hooks.disabled"))
    except Exception:
        return False


def read_payload():
    try:
        data = json.loads(sys.stdin.read())
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def emit(obj) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=True))
    sys.stdout.write("\n")
    sys.stdout.flush()


def run_hook(fn) -> None:
    """Fail-open driver: fn(payload) -> dict. Any problem prints {} and exits 0."""
    out = {}
    try:
        if not disabled():
            payload = read_payload()
            if payload is not None:
                res = fn(payload)
                if isinstance(res, dict):
                    out = res
    except BaseException:  # noqa: BLE001 - fail open, always
        out = {}
    try:
        emit(out)
    except Exception:
        pass
    sys.exit(0)


def run_cmd(argv, timeout, cwd=None):
    """Return (returncode, stdout, stderr) or None on any failure/timeout."""
    try:
        p = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           timeout=timeout, cwd=cwd, universal_newlines=True)
        return p.returncode, p.stdout, p.stderr
    except Exception:
        return None


def _ancestors():
    pids = {os.getpid()}
    pid = os.getppid()
    for _ in range(64):
        if pid <= 1 or pid in pids:
            break
        pids.add(pid)
        try:
            with open("/proc/%d/stat" % pid) as f:
                s = f.read()
            pid = int(s.rsplit(")", 1)[1].split()[1])
        except Exception:
            break
    return pids


def _is_worker(pid: int) -> bool:
    """True if pid really runs queue_run.sh as a script (not status, not a grep for it)."""
    try:
        with open("/proc/%d/cmdline" % pid, "rb") as f:
            args = f.read().decode("utf-8", "replace").split("\0")
    except Exception:
        return True  # cannot verify; keep pgrep's verdict
    args = [a for a in args if a != ""]
    head = args[:3]
    idx = None
    for i, a in enumerate(head):
        if a.endswith("queue_run.sh"):
            idx = i
            break
    if idx is None:
        return False
    if idx == 1 and os.path.basename(head[0]) not in ("bash", "sh", "dash"):
        return False
    if idx == 2 and os.path.basename(head[0]) not in ("bash", "sh", "dash", "env"):
        return False
    return not (len(args) > idx + 1 and args[idx + 1] == "status")


def running_workers():
    """PIDs of running queue_run.sh workers, or None if detection failed.

    HARNESS_GUARD_FAKE_WORKERS (comma-separated pids, empty string = none) overrides.
    """
    fake = os.environ.get("HARNESS_GUARD_FAKE_WORKERS")
    if fake is not None:
        return [int(x) for x in fake.replace(" ", "").split(",") if x.isdigit()]
    r = run_cmd(["pgrep", "-f", "queue_run.sh"], 2)
    if r is None or r[0] not in (0, 1):
        return None
    skip = _ancestors()
    pids = [int(x) for x in r[1].split() if x.isdigit()]
    return [p for p in pids if p not in skip and _is_worker(p)]
