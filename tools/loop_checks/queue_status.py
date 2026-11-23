#!/usr/bin/env python3
"""AD-2: read-only `queue_run.sh status --json` equivalent.

Mirrors the `status` branch of scripts/queue_run.sh (JOBS_DIR/PROC_DIR/DONE_DIR/
FAIL_DIR/STOP_FILE under $QUEUE_DIR, default scripts/queue). Never writes.
Exit 0 normally; exit 2 if the queue directory is missing.
"""
import argparse
import datetime as dt
import json
import os
import sys

REPO = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def resolve_queue_dir(arg=None):
    q = arg or os.environ.get("QUEUE_DIR") or os.path.join(REPO, "scripts", "queue")
    return os.path.realpath(q)


def _ls(path):
    try:
        return sorted(os.listdir(path))
    except OSError:
        return []


def snapshot(queue_dir):
    """Return the status dict, or None if queue_dir is not a directory."""
    if not os.path.isdir(queue_dir):
        return None
    jobs = _ls(os.path.join(queue_dir, "jobs"))
    proc = []
    for name in _ls(os.path.join(queue_dir, "processing")):
        job, sep, gpu = name.rpartition(".gpu")
        proc.append({"job": job if sep else name, "gpu": gpu if sep else ""})
    failed = _ls(os.path.join(queue_dir, "failed"))
    return {
        "queue_dir": queue_dir,
        "jobs": jobs,
        "processing": proc,
        "done": len(_ls(os.path.join(queue_dir, "done"))),
        "failed": len(failed),
        "failed_names": failed,
        "stop_file": os.path.exists(os.path.join(queue_dir, "stop")),
        "snapshot_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def render_text(s):
    out = ["== queued (FIFO order) =="]
    out += s["jobs"]
    out.append("== running (suffix = gpu) ==")
    out += [f'{p["job"]}.gpu{p["gpu"]}' if p["gpu"] != "" else p["job"] for p in s["processing"]]
    out.append(f'== done: {s["done"]}  failed: {s["failed"]} ==')
    out += s["failed_names"]
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", action="store_true", help="JSON output")
    ap.add_argument("--queue-dir", help="override $QUEUE_DIR (default scripts/queue)")
    a = ap.parse_args(argv)
    qd = resolve_queue_dir(a.queue_dir)
    s = snapshot(qd)
    if s is None:
        print(f"queue dir missing: {qd}", file=sys.stderr)
        return 2
    print(json.dumps(s, ensure_ascii=False, indent=2) if a.json else render_text(s))
    return 0


if __name__ == "__main__":
    sys.exit(main())
