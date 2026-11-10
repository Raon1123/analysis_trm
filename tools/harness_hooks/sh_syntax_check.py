#!/usr/bin/env python3
"""PostToolUse(Write|Edit): run `bash -n` on written .sh files. Fail-open."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import run_cmd, run_hook  # noqa: E402


def handle(payload):
    ti = payload.get("tool_input")
    path = ti.get("file_path") if isinstance(ti, dict) else None
    if not isinstance(path, str) or not path.endswith(".sh") or not os.path.isfile(path):
        return {}
    r = run_cmd(["bash", "-n", path], 5)
    if r is None or r[0] == 0:
        return {}
    return {"decision": "block",
            "reason": "[harness:bash -n] %s has a syntax error:\n%s" % (path, (r[2] or "")[:600])}


if __name__ == "__main__":
    run_hook(handle)
