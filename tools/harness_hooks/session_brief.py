#!/usr/bin/env python3
"""SessionStart: compact state brief (<= 6000 chars). Fail-open."""
from __future__ import annotations

import datetime
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import project_dir, run_cmd, run_hook, running_workers  # noqa: E402

LIMIT = 6000
STATE_REL = "lab/11_governance/11_lab-root/STATE.md"
STATUSES = ["needs-decisions", "ready-for-resume", "in-progress", "commit-pending", "DONE"]
POINTER = "전체 맥락: STATE.md §6이 가리키는 문서 1개만 더 읽는다. 이 brief는 요약이지 근거가 아니다."


def _utcnow():
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)


def safe(fn, default="n/a"):
    try:
        v = fn()
        return default if v is None else v
    except Exception:
        return default


def git_branch(root):
    r = run_cmd(["git", "-C", root, "rev-parse", "--abbrev-ref", "HEAD"], 2)
    return r[1].strip() if r and r[0] == 0 and r[1].strip() else None


def read_state(root):
    with open(os.path.join(root, STATE_REL), encoding="utf-8", errors="replace") as f:
        return f.read()


def freshness(root, text):
    p = os.path.join(root, STATE_REL)
    kb = os.path.getsize(p) / 1024.0
    m = re.search(r"^updated:\s*(.+?)\s*$", text, re.M)
    upd = m.group(1).strip().strip("\"'") if m else None
    flag = ""
    if upd:
        for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d"):
            try:
                dt = datetime.datetime.strptime(upd[:16] if "%H" in fmt else upd[:10], fmt)
                age = (_utcnow() - dt).total_seconds() / 3600.0
                flag = " STALE" if age > 48 else ""
                flag += " (age %.0fh)" % age
                break
            except ValueError:
                continue
    return "STATE.md: updated=%s size=%.1fKB%s" % (upd or "n/a", kb, flag)


def track_lines(text):
    lines = text.split("\n")
    start = None
    for i, l in enumerate(lines):
        if l.startswith("## 1."):
            start = i + 1
            break
    if start is None:
        return []
    out = []
    for l in lines[start:]:
        if l.startswith("## "):
            break
        if not l.startswith("|"):
            continue
        cells = [c.strip() for c in l.strip().strip("|").split("|")]
        first = cells[0] if cells else ""
        if not first or set(first) <= set("-: ") or first.lower() == "code" or "~~" in first:
            continue
        code = first.replace("*", "").replace("~", "").strip()
        status = "?"
        for span in re.findall(r"\*\*(.+?)\*\*", l):
            hit = next((s for s in STATUSES if s.lower() in span.lower()), None)
            if hit:
                status = hit
                break
        date = "n/a"
        if len(cells) > 2:
            m = re.search(r"\d\d-\d\d \d\d:\d\d", cells[2])
            date = m.group(0) if m else "n/a"
        m = re.search(r"31_handoffs/[^\s|`)]*?\.md", l)
        out.append("%s | %s | %s | %s" % (code, status, date, m.group(0) if m else "n/a"))
    return out


def queue_line(root):
    parts = []
    for d in ("jobs", "processing", "done", "failed"):
        p = os.path.join(root, "scripts", "queue", d)
        try:
            cnt = str(len([e for e in os.listdir(p) if not e.startswith(".")]))
        except Exception:
            cnt = "n/a"
        parts.append("%s=%s" % (d, cnt))
    w = safe(running_workers, None)
    wtxt = "n/a" if w is None else ("running (pids %s)" % ",".join(map(str, w)) if w else "none")
    return "queue: %s | worker: %s" % (" ".join(parts), wtxt)


def gpu_line():
    r = run_cmd(["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
                 "--format=csv,noheader,nounits"], 3)
    if not r or r[0] != 0:
        return "GPU 4-7: n/a"
    items = []
    for l in r[1].strip().splitlines():
        f = [x.strip() for x in l.split(",")]
        if len(f) == 3 and f[0] in ("4", "5", "6", "7"):
            items.append("%s:%s/%sMiB" % tuple(f))
    return "GPU 4-7: " + (" ".join(items) if items else "n/a")


def build(source, root):
    now = _utcnow().strftime("%Y-%m-%d %H:%M UTC")
    head = ["[session-brief] %s source=%s branch=%s" % (now, source, safe(lambda: git_branch(root)))]
    tracks = []
    text = safe(lambda: read_state(root), None)
    if text is None:
        head.append("STATE.md: n/a")
        head.append("tracks: n/a")
    else:
        head.append(safe(lambda: freshness(root, text)))
        head.append("tracks (code | status | date | handoff):")
        tracks = safe(lambda: track_lines(text), [])
        if not tracks:
            tracks = ["n/a"]
    tail = [safe(lambda: queue_line(root)), safe(gpu_line), POINTER]
    # Truncate the track table (never the closing pointer line) to fit LIMIT.
    kept = list(tracks)
    dropped = 0
    while True:
        extra = ["... (%d more tracks truncated)" % dropped] if dropped else []
        s = "\n".join(head + kept + extra + tail)
        if len(s) <= LIMIT or not kept:
            break
        kept.pop()
        dropped += 1
    return s if len(s) <= LIMIT else s[:LIMIT - len(POINTER) - 1] + "\n" + POINTER


def handle(payload):
    source = payload.get("source") or "n/a"
    brief = build(str(source), project_dir())
    return {"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": brief}}


if __name__ == "__main__":
    run_hook(handle)
