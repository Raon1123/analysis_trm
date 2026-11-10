#!/usr/bin/env python3
"""PreToolUse(Bash) guard: deny a few recurring mechanical mistakes. Fail-open.

Rules: rg (a typed `rg` runs as `rtk grep` here; only ripgrep-only options such as --files, -g, -t
fail or are silently ignored, so only those fire; plain `rg -n pat dir` passes), pytest-gpu (pytest without CUDA_VISIBLE_DEVICES=), queue-dup
(starting scripts/queue_run.sh while a worker runs). The former py38 rule (premise "bare
python3 is 3.8") was removed 2026-11-10: /usr/bin/python3 here is 3.12.3.

Design bias: false negatives are acceptable, false positives are not. Whenever quoting is
ambiguous (unbalanced quotes, or $( / ${ / backtick inside double quotes, e.g. a heredoc inside
a commit message), no rule fires. Commands longer than MAX_CMD_LEN skip all rules.
"""
from __future__ import annotations

import os
import re
import shlex
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import run_hook, running_workers  # noqa: E402

KEYWORDS = {"do", "then", "else", "elif", "!", "{", "time", "nohup", "env", "rtk"}
# Value-taking `uv run` options, checked against `uv run --help` (uv 2026-11): a package named
# pytest given as their value must not look like the command word. Flags (-m, -s, --script) are not here.
VALUE_OPTS_UV = {
    "--extra", "--no-extra", "--group", "--no-group", "--only-group", "--env-file", "-w", "--with",
    "--with-editable", "--with-requirements", "--package", "--python-platform", "--index",
    "--default-index", "-i", "--index-url", "--extra-index-url", "-f", "--find-links",
    "--index-strategy", "--keyring-provider", "-P", "--upgrade-package", "--resolution",
    "--prerelease", "--fork-strategy", "--exclude-newer", "--exclude-newer-package",
    "--reinstall-package", "--link-mode", "-C", "--config-setting", "--config-settings-package",
    "--no-build-isolation-package", "--no-build-package", "--no-binary-package", "--cache-dir",
    "--refresh-package", "-p", "--python", "--color", "--allow-insecure-host", "--directory",
    "--project", "--config-file",
}
# ripgrep-only options, each confirmed by a probe through `rtk grep` (see README table): error or ignored.
RG_ONLY_LONG = {
    "--files", "--glob", "--type", "--type-not", "--type-list", "--json", "--multiline", "--hidden",
    "--no-ignore", "--sort", "--vimgrep", "--smart-case", "--iglob",
}
RG_ONLY_SHORT = set("gtTSU")  # -g -t -T -S -U (also inside clusters and as -tpy / -g*.py)
RG_VALUE_OPTS = {"-e", "-f", "-A", "-B", "-C", "-m", "--regexp", "--file"}  # their value is not an option
MAX_CMD_LEN = 100000
READONLY_QUEUE_SUBCMDS = ("status", "dry-run")  # scripts/queue_run.sh: only these two run nothing
ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def split_segments(cmd, flags=None):
    """Quote-aware split on && || ; | & newline. Stops at first heredoc line.

    flags (dict, optional): flags["ambiguous"] = True when quoting cannot be trusted
    (a $( or backtick inside double quotes, or a quote left open at the end).
    """
    if flags is None:
        flags = {}
    segs, cur = [], []
    i, n = 0, len(cmd)
    quote = None
    while i < n:
        c = cmd[i]
        if quote:
            cur.append(c)
            if quote == '"' and (cmd.startswith("$(", i) or cmd.startswith("${", i) or c == "`"):
                flags["ambiguous"] = True
            if c == "\\" and quote == '"' and i + 1 < n:
                cur.append(cmd[i + 1])
                i += 1
            elif c == quote:
                quote = None
            i += 1
            continue
        if c == "\\" and i + 1 < n:
            cur.append(cmd[i:i + 2])
            i += 2
            continue
        if c in "'\"":
            quote = c
            cur.append(c)
            i += 1
            continue
        if c == "#" and (not cur or cur[-1] in " \t"):
            while i < n and cmd[i] != "\n":
                i += 1
            continue
        if cmd.startswith("<<<", i):
            cur.append("<<<")
            i += 3
            continue
        if cmd.startswith("<<", i):
            # heredoc: the rest of this line still counts, nothing after it does
            while i < n and cmd[i] != "\n":
                cur.append(cmd[i])
                i += 1
            segs.append("".join(cur))
            return segs
        if cmd.startswith("&&", i) or cmd.startswith("||", i):
            segs.append("".join(cur)); cur = []; i += 2
            continue
        if c in ";|\n":
            segs.append("".join(cur)); cur = []; i += 1
            continue
        if c == "&":
            prev = cur[-1] if cur else ""
            nxt = cmd[i + 1] if i + 1 < n else ""
            if prev and prev[-1] in "<>" or nxt == ">":
                cur.append(c)
            else:
                segs.append("".join(cur)); cur = []
            i += 1
            continue
        cur.append(c)
        i += 1
    if quote:
        flags["ambiguous"] = True
    segs.append("".join(cur))
    return segs


def tokenize(seg):
    try:
        return shlex.split(seg, posix=True)
    except ValueError:
        return seg.split()


def strip_wrappers(toks):
    toks = list(toks)
    while toks:
        t = toks[0]
        if ASSIGN.match(t):
            toks.pop(0)
        elif t == "rtk":
            toks.pop(0)
            if toks and toks[0] == "proxy":
                toks.pop(0)
        elif t in ("nohup", "time", "env", "do", "then", "else", "elif", "!", "{"):
            toks.pop(0)
        elif t == "timeout":
            toks.pop(0)
            while toks and toks[0].startswith("-"):
                o = toks.pop(0)
                if o in ("-k", "-s", "--kill-after", "--signal") and toks:
                    toks.pop(0)
            if toks:
                toks.pop(0)  # duration
        elif t == "stdbuf":
            toks.pop(0)
            while toks and toks[0].startswith("-"):
                toks.pop(0)
        else:
            break
    return toks


def uv_run_command(toks):
    """For `uv run [opts] CMD ...` return tokens starting at CMD, else None."""
    if len(toks) < 2 or toks[0] != "uv" or toks[1] != "run":
        return None
    rest = toks[2:]
    while rest and rest[0].startswith("-"):
        o = rest.pop(0)
        if o in VALUE_OPTS_UV and rest:
            rest.pop(0)
    return rest


def is_pytest(toks):
    if not toks:
        return False
    head = toks[0]
    if head == "pytest" or head.endswith("/pytest"):
        return True
    sub = uv_run_command(toks)
    if sub is not None:
        return is_pytest(sub) if sub else False
    if os.path.basename(head) in ("python", "python3") or re.match(r"^python3?\.\d+$", os.path.basename(head)):
        return len(toks) >= 3 and toks[1] == "-m" and toks[2] == "pytest"
    return False


def queue_start(toks):
    if not toks:
        return False
    head, rest = toks[0], toks[1:]
    if os.path.basename(head) in ("bash", "sh") and rest and rest[0].endswith("queue_run.sh"):
        rest = rest[1:]
    elif not head.endswith("queue_run.sh"):
        return False
    rest = [r for r in rest if r not in ("&", "2>&1") and not r.startswith(("2>", ">", "<"))]
    return not rest or rest[0] not in READONLY_QUEUE_SUBCMDS


def rg_only_flag(args):
    """First ripgrep-only option in the args of an rg segment, else None."""
    skip = False
    for t in args:
        if skip:
            skip = False
            continue
        if t == "--":
            return None
        if t in RG_VALUE_OPTS:
            skip = True
            continue
        if t.startswith("--"):
            if t.split("=", 1)[0] in RG_ONLY_LONG:
                return t.split("=", 1)[0]
        elif re.match(r"^-u{2,3}$", t):
            return t
        elif re.match(r"^-[A-Za-z]+", t) and set(t[1:]) & RG_ONLY_SHORT:
            return "-" + next(c for c in t[1:] if c in RG_ONLY_SHORT)
    return None


def is_rtk_rg(raw_toks):
    """True for `rtk rg ...` / `rtk proxy rg ...` (leading assignments/time/nohup/env allowed)."""
    toks = list(raw_toks)
    while toks and (ASSIGN.match(toks[0]) or toks[0] in ("nohup", "time", "env", "do", "then", "else", "elif", "!", "{")):
        toks.pop(0)
    if not toks or toks[0] != "rtk":
        return False
    toks.pop(0)
    if toks and toks[0] == "proxy":
        toks.pop(0)
    return bool(toks) and toks[0] == "rg"


def check(cmd, workers_fn=running_workers):
    """Return (rule, reason) or None."""
    if len(cmd) > MAX_CMD_LEN:
        return None
    flags = {}
    raw = split_segments(cmd, flags)
    if flags.get("ambiguous"):
        return None
    raw_toks = [tokenize(s) for s in raw]
    segs = [strip_wrappers(t) for t in raw_toks]
    segs = [s for s in segs if s]

    def marked(rule):
        return ("guard-ok:" + rule) in cmd

    if not marked("rg"):
        for t in raw_toks:
            if is_rtk_rg(t):
                return "rg", ("`rtk rg` / `rtk proxy rg` always fail on this host (no rg binary: rc 127 / rc 1). "
                              "Use `rtk grep` or the Grep tool. Bypass: add `# guard-ok:rg`.")
        for s in segs:
            if s[0] == "rg":
                flag = rg_only_flag(s[1:])
                if flag:
                    return "rg", ("an `rg` that starts a command is rewritten to `rtk grep`, which does not support ripgrep-only option `%s` "
                                  "(a pipe-sink `rg`, as in `x | rg ...`, is not rewritten and may run real ripgrep: bypass if so); "
                                  "use `git ls-files`, `grep -r --include=...`, or the Grep/Glob tools. "
                                  "Bypass: add `# guard-ok:rg`." % flag)
    if not marked("pytest-gpu") and "CUDA_VISIBLE_DEVICES=" not in cmd:
        for s in segs:
            if is_pytest(s):
                return "pytest-gpu", ('pytest without CUDA_VISIBLE_DEVICES: tests may touch shared GPUs. '
                                      'Prefix `CUDA_VISIBLE_DEVICES="" `. Bypass: add `# guard-ok:pytest-gpu`.')
    if not marked("queue-dup"):
        if any(queue_start(s) for s in segs):
            try:
                pids = workers_fn()
            except Exception:
                pids = None
            if pids:
                return "queue-dup", ("queue_run.sh workers already running (PIDs %s); a second start puts two jobs on each GPU. "
                                     "Check `scripts/queue_run.sh status`. Bypass: add `# guard-ok:queue-dup`."
                                     % ",".join(str(p) for p in pids))
    return None


def handle(payload, workers_fn=running_workers):
    if payload.get("tool_name") not in (None, "Bash"):
        return {}
    ti = payload.get("tool_input")
    cmd = ti.get("command") if isinstance(ti, dict) else None
    if not isinstance(cmd, str) or not cmd.strip():
        return {}
    res = check(cmd, workers_fn)
    if not res:
        return {}
    rule, reason = res
    return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                                   "permissionDecisionReason": "[harness-guard:%s] %s" % (rule, reason)}}


if __name__ == "__main__":
    run_hook(handle)
