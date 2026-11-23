#!/usr/bin/env python3
"""AD-11: read-only memory dead-path / last-cited check.

Scans <memory>/*.md (not MEMORY.md): backticked repo paths that no longer exist, and
`last_cited` = newest mtime among lab/**/*.md containing the literal `[[<name>]]`.
Exit 0 if zero dead paths, else 1.
"""
import argparse
import datetime as dt
import json
import os
import re
import sys
import time

REPO = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
MEMORY = os.path.expanduser("~/.claude/projects/-home-ayp-project-trm/memory")
PREFIXES = ("lab/", "scripts/", "tools/", ".claude/", "analysis/", "reports/", "data/", "checkpoints/", "wandb/")
RESTRICTED = ["lab/11_governance", "lab/31_handoffs", "lab/24_audits", "lab/22_plans", "lab/92_daily"]
SKIP_DIRS = {"99_Archive", "before"}
SPAN = re.compile(r"`([^`\n]+)`")
LINK = re.compile(r"\[\[([^\[\]]+)\]\]")
PLACEHOLDER = re.compile(r"[<>{}$…]|\.\.\.|\bNN\b|YYYY|\bN\b")


def clean_path(span):
    tok = span.strip().split()[0] if span.strip() else ""
    if not tok.startswith(PREFIXES) or PLACEHOLDER.search(tok):
        return None
    if "*" in tok:
        pre = tok.split("*", 1)[0]
        tok = pre if pre.endswith("/") else os.path.dirname(pre)
    tok = tok.rstrip(".,;:)'\"")
    tok = re.sub(r":\d+(-\d+)?(,\d+)*$", "", tok)  # line anchors
    tok = tok.rstrip(".,;:)'\"").rstrip("/") if "*" in span else tok.rstrip(".,;:)'\"")
    return tok or None


def parse_memory(path):
    text = open(path, encoding="utf-8", errors="replace").read()
    name = os.path.splitext(os.path.basename(path))[0]
    m = re.match(r"---\n(.*?)\n---", text, re.S)
    if m:
        n = re.search(r"^name:\s*(.+?)\s*$", m.group(1), re.M)
        if n:
            name = n.group(1).strip("'\"")
    paths = []
    for span in SPAN.findall(text):
        p = clean_path(span)
        if p and p not in paths:
            paths.append(p)
    return name, paths


def scan_links(roots, names, deadline=None):
    """One pass; returns {name: (mtime, path)} or None if deadline exceeded."""
    best, seen = {}, set()
    for root in roots:
        for dp, dns, fns in os.walk(root, followlinks=True):
            if deadline and time.monotonic() > deadline:
                return None
            real = os.path.realpath(dp)
            if real in seen:
                dns[:] = []
                continue
            seen.add(real)
            dns[:] = [d for d in dns if d not in SKIP_DIRS]
            for fn in fns:
                if not fn.endswith(".md"):
                    continue
                fp = os.path.join(dp, fn)
                try:
                    txt = open(fp, encoding="utf-8", errors="replace").read()
                    mt = os.stat(fp).st_mtime
                except OSError:
                    continue
                if "[[" not in txt:
                    continue
                for n in set(LINK.findall(txt)) & names:
                    if n not in best or mt > best[n][0]:
                        best[n] = (mt, fp)
    return best


def run(memory_dir, repo, lab, budget_s=45.0):
    files = sorted(f for f in os.listdir(memory_dir) if f.endswith(".md") and f != "MEMORY.md")
    parsed = [(f,) + parse_memory(os.path.join(memory_dir, f)) for f in files]
    names = {n for _, n, _ in parsed}
    scope = "full lab/"
    best = scan_links([lab], names, time.monotonic() + budget_s)
    if best is None:
        scope = "restricted (full scan exceeded %ds): " % budget_s + ", ".join(RESTRICTED)
        roots = [os.path.join(os.path.dirname(lab), r.split("/", 1)[1]) for r in RESTRICTED]
        best = scan_links([r for r in roots if os.path.isdir(r)], names)
    rows, total_dead = [], 0
    for f, name, paths in parsed:
        dead = [p for p in paths if not os.path.exists(os.path.join(repo, p))]
        total_dead += len(dead)
        lc = best.get(name)
        rows.append({
            "file": f, "name": name, "paths_checked": len(paths), "dead_count": len(dead), "dead": dead,
            "last_cited": dt.datetime.fromtimestamp(lc[0], dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if lc else None,
            "last_cited_in": lc[1] if lc else None,
        })
    return {"memory_dir": memory_dir, "lab_scan_scope": scope, "files": len(rows),
            "total_dead": total_dead, "entries": rows}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--memory-dir", default=MEMORY)
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--lab", help="lab dir to scan (default <repo>/lab)")
    ap.add_argument("--budget-s", type=float, default=45.0)
    a = ap.parse_args(argv)
    if not os.path.isdir(a.memory_dir):
        print(f"memory dir missing: {a.memory_dir}", file=sys.stderr)
        return 2
    r = run(a.memory_dir, a.repo, a.lab or os.path.join(a.repo, "lab"), a.budget_s)
    if a.json:
        print(json.dumps(r, ensure_ascii=False, indent=2))
    else:
        print(f'# lab scan scope: {r["lab_scan_scope"]}; files={r["files"]} total_dead={r["total_dead"]}')
        for e in r["entries"]:
            print(f'{e["file"]}\tdead={e["dead_count"]}\tlast_cited={e["last_cited"]}')
            for d in e["dead"]:
                print(f"    DEAD {d}")
    return 1 if r["total_dead"] else 0


if __name__ == "__main__":
    sys.exit(main())
