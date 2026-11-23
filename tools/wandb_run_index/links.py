#!/usr/bin/env python3
"""Assign every indexed run exactly one wandb link tier (see common.assign_tier).

Run:  uv run python tools/wandb_run_index/links.py [--fetch] [--apply] [--root .]

  --fetch  READ-ONLY wandb API listing, one `wandb.Api().runs("raon1123/<project>")` per project that has
           local runs. Only id, name, state, url, created_at, project are cached, to
           _meta/api_listing.json (raw per-project listing + per-project ok/error/count).  Nothing is
           written, synced or deleted on wandb.  Without --fetch the cached listing is used.
  --apply  write _meta/links.jsonl, _meta/links_counts.json, _meta/queue_log_index.json

Queue-log evidence: every file under logs/queue/ (symlinks followed) is scanned for literal
`wandb.ai/<entity>/<project>/runs/<id>` URLs and for the run dir it reports ("Run data is saved locally
in .../wandb/run-..-<id>"), giving id -> log path.

Exit codes: 0 ok; 3 any per-run inconsistency (listed); 4 an API listing failed or returned 0 runs for a
project that has local online runs (never treated as "confirmed absent"); 2 empty inputs.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ENTITY, META_REL, TIERS, assign_tier  # noqa: E402

INDEX = "lab/21_experiments/11_lab-experiments/_wandb_runs_index.jsonl"
URL_RE = re.compile(rb"wandb\.ai/([A-Za-z0-9_-]+)/([A-Za-z0-9_.-]+)/runs/([a-z0-9]+)")
DIR_RE = re.compile(rb"wandb/(?:offline-)?run-\d{8}_\d{6}-([a-z0-9]+)")


def scan_queue_logs(root: Path):
    urls, dirs = collections.defaultdict(set), collections.defaultdict(set)
    files = sorted(glob.glob(str(root / "logs/queue/**/*"), recursive=True))
    n = 0
    for f in files:
        if not os.path.isfile(f):
            continue
        n += 1
        rel = os.path.relpath(f, root)
        try:
            b = open(f, "rb").read()
        except OSError:
            continue
        for ent, proj, rid in URL_RE.findall(b):
            e, p, r = ent.decode(), proj.decode(), rid.decode()
            urls[r].add((e, p, f"https://wandb.ai/{e}/{p}/runs/{r}", rel))
        for rid in DIR_RE.findall(b):
            dirs[rid.decode()].add(rel)
    return ({k: sorted(v) for k, v in urls.items()}, {k: sorted(v) for k, v in dirs.items()}, n)


def fetch_listing(projects):
    import wandb  # lazy; read-only listing only
    api = wandb.Api(timeout=60)
    listing, status = {}, {}
    for p in sorted(projects):
        try:
            rows = []
            for run in api.runs(f"{ENTITY}/{p}", per_page=500):
                rows.append({"id": run.id, "name": run.name, "state": run.state, "url": run.url,
                             "created_at": getattr(run, "created_at", None), "project": p})
            listing[p] = sorted(rows, key=lambda r: r["id"])
            status[p] = {"ok": True, "n": len(rows), "error": None}
        except Exception as e:  # noqa: BLE001  network / permission / missing project
            listing[p] = []
            status[p] = {"ok": False, "n": 0, "error": f"{type(e).__name__}: {str(e)[:200]}"}
        print(f"  api {p}: {status[p]}", flush=True)
    return listing, status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/ayp/project/trm")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    meta = root / META_REL
    rows = [json.loads(ln) for ln in open(root / INDEX)]
    if not rows:
        print("ERROR: empty index", file=sys.stderr)
        return 2
    online_projects = collections.Counter(r.get("wandb_project") for r in rows if r.get("mode") == "online")
    projects = {r.get("wandb_project") for r in rows if r.get("wandb_project") not in (None, "", "?", "unknown")}
    cache = meta / "api_listing.json"
    if a.fetch:
        listing, status = fetch_listing(projects)
        meta.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"entity": ENTITY, "status": status, "listing": listing},
                                    indent=1, sort_keys=True, default=str) + "\n")
        print("wrote", cache)
    c = json.loads(cache.read_text())
    status, listing = c["status"], c["listing"]
    api = {}
    for p, rs in listing.items():
        for r in rs:
            api[r["id"]] = r
    qurls, qdirs, nlogs = scan_queue_logs(root)
    print(f"queue logs scanned {nlogs}; ids with URL {len(qurls)}; ids with run-dir line {len(qdirs)}")
    if nlogs == 0:
        print("ERROR: 0 queue logs found", file=sys.stderr)
        return 2
    out, errs = [], []
    for r in sorted(rows, key=lambda x: x["id"]):
        t = assign_tier({"id": r["id"], "project": r.get("wandb_project"), "name": r.get("name"),
                         "mode": r.get("mode"), "synced_marker": r.get("synced_marker")}, api, qurls, status)
        t["local_project_arg"] = r.get("project")
        t["wandb_file"], t["wandb_sha256"] = r.get("wandb_file"), r.get("wandb_sha256")
        t["queue_logs"] = sorted(set(qdirs.get(r["id"], [])) | {x[3] for x in qurls.get(r["id"], [])})
        out.append(t)
        for e in t["errors"]:
            errs.append(f"{r['id']} {r.get('wandb_project')}/{r.get('name')}: {e}")
    idx_ids = {r["id"] for r in rows}
    api_only = sorted(i for i in api if i not in idx_ids)
    counts = collections.defaultdict(collections.Counter)
    for t in out:
        counts[t["project"] or "?"][t["tier"]] += 1
    mism = [t for t in out if t.get("name_mismatch")]
    listing_fail = sorted(p for p, n in online_projects.items()
                          if p not in (None, "", "?", "unknown") and not (status.get(p, {}).get("ok") and status[p]["n"] > 0))
    summary = {"entity": ENTITY, "n_runs": len(out), "tiers": dict(collections.Counter(t["tier"] for t in out)),
               "per_project": {p: {k: v[k] for k in TIERS if v[k]} for p, v in sorted(counts.items())},
               "name_mismatches": [{"id": t["id"], "local": next(r["name"] for r in rows if r["id"] == t["id"]),
                                    "api": t.get("api_name")} for t in mism],
               "api_status": status, "api_ids_not_local": api_only,
               "queue_log_not_api": sorted(t["id"] for t in out if t["tier"] == "queue-log"),
               "listing_failed_projects_with_online_runs": listing_fail,
               "inconsistencies": errs}
    print(json.dumps({k: summary[k] for k in ("n_runs", "tiers", "per_project", "listing_failed_projects_with_online_runs")}, indent=1))
    print(f"name mismatches {len(mism)}; api ids not local {len(api_only)}; inconsistencies {len(errs)}")
    for e in errs[:30]:
        print("  INCONSISTENT", e)
    if len({t['id'] for t in out}) != len(out) or any(t.get("tier") not in TIERS for t in out):
        print("ERROR: tier assignment not one-per-run", file=sys.stderr)
        return 3
    if a.apply:
        meta.mkdir(parents=True, exist_ok=True)
        with open(meta / "links.jsonl", "w") as fh:
            for t in out:
                fh.write(json.dumps(t, sort_keys=True, ensure_ascii=False) + "\n")
        (meta / "links_counts.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
        print("wrote", meta / "links.jsonl")
    if errs:
        return 3
    if listing_fail:
        print(f"ERROR: API listing failed/empty for projects with online runs: {listing_fail}", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
