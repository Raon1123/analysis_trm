#!/usr/bin/env python3
"""One full pass over every local .wandb file -> lab/21_experiments/12_run-cards/_meta/scan_cache.jsonl

Run with the trm-mcp venv (needs wandb's datastore + trm_mcp's metric allowlist):
  /home/ayp/project/trm-mcp/.venv/bin/python tools/wandb_run_index/scan.py [--root .] [--workers 32] [--apply]

Per run (keyed by run id; cached by (realpath, size, mtime_ns) so re-runs are cheap):
  * sha256 + size + mtime of the .wandb file (the local canonical record)
  * RunRecord header: entity, project, display_name, start time, and a sensitive-key-stripped config
    (only used for runs without files/wandb-metadata.json); environment git commit
  * exit record (exit_code, runtime) if present
  * FULL-RESOLUTION stats for every history key accepted by trm_mcp.wandb_reader.is_allowed_metric_key
    (incl. z/delta_step_<N>): n, first/last (+step), min/max (+step, ties -> earliest step), nonfinite count.
    NaN/inf are counted in `nonfinite` and excluded from min/max/first/last. Booleans and non-numeric
    values are ignored. No downsampling, no time/record budget: a scan error is recorded as
    stats status "scan-error: ..." and the stats are then not used by cards.
No network.  Nothing is written outside _meta/ (and only with --apply).
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, "/home/ayp/project/trm-mcp/src")
from trm_mcp.wandb_reader import _strip_sensitive, is_allowed_metric_key  # noqa: E402
from wandb.proto import wandb_internal_pb2 as pb  # noqa: E402
from wandb.sdk.internal import datastore  # noqa: E402

WANDB_DIRS = ["wandb", "/mnt/ayp/trm-offload/wandb"]
META = "lab/21_experiments/12_run-cards/_meta"
SCAN_VERSION = 1


def run_dirs(root: Path):
    seen = {}
    for base in WANDB_DIRS:
        base = base if os.path.isabs(base) else str(root / base)
        for d in sorted(glob.glob(base + "/run-*") + glob.glob(base + "/offline-run-*")):
            rid = d.rsplit("-", 1)[-1]
            if rid not in seen and os.path.isdir(d):
                seen[rid] = d
    return seen


def wandb_file(d: str):
    fs = sorted(f for f in glob.glob(d + "/run-*.wandb"))
    return fs[0] if fs else None


def _val(raw):
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return raw


def scan_one(args):
    rid, d = args
    f = wandb_file(d)
    out = {"id": rid, "run_dir_link": d, "run_dir": os.path.realpath(d), "scan_version": SCAN_VERSION,
           "synced_marker": bool(glob.glob(d + "/run-*.wandb.synced"))}
    if f is None:
        out.update(wandb_file=None, stats={"status": "no-wandb-file", "keys": {}})
        return out
    st = os.stat(f)
    out.update(wandb_file=os.path.realpath(f), wandb_size=st.st_size, wandb_mtime_ns=st.st_mtime_ns)
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    out["wandb_sha256"] = h.hexdigest()
    header, env, exit_rec = None, None, None
    keys: dict = {}
    last_step, n_hist, n_bad = None, 0, 0
    status = "ok"
    ds = datastore.DataStore()
    try:
        ds.open_for_scan(f)
    except Exception as e:  # noqa: BLE001
        out["stats"] = {"status": f"scan-error: open {type(e).__name__}: {e}"[:200], "keys": {}}
        return out
    while True:
        try:
            raw = ds.scan_data()
        except Exception as e:  # noqa: BLE001
            # a torn tail on a live run is common; earlier records are still valid
            status = f"scan-error: {type(e).__name__}: {str(e)[:120]}"
            break
        if raw is None:
            break
        rec = pb.Record()
        try:
            rec.ParseFromString(raw)
        except Exception:  # noqa: BLE001
            n_bad += 1
            continue
        kind = rec.WhichOneof("record_type")
        if kind == "run" and header is None:
            r = rec.run
            cfg = {}
            for it in r.config.update:
                k = it.key if it.key else "/".join(it.nested_key)
                if k and k != "_wandb":
                    cfg[k] = _val(it.value_json)
            header = {"entity": r.entity or None, "project": r.project or None, "display_name": r.display_name or None,
                      "run_group": r.run_group or None,
                      "start_time": (dt.datetime.fromtimestamp(r.start_time.seconds, dt.timezone.utc).isoformat()
                                     if r.start_time.seconds else None),
                      "config": _strip_sensitive(cfg)}
        elif kind == "environment" and env is None and rec.environment.git.commit:
            env = {"git_commit": rec.environment.git.commit}
        elif kind == "exit":
            exit_rec = {"exit_code": int(rec.exit.exit_code), "runtime": int(rec.exit.runtime)}
        elif kind == "history":
            n_hist += 1
            step, found = None, {}
            for it in rec.history.item:
                k = it.key if it.key else "/".join(it.nested_key)
                if k == "_step":
                    step = _val(it.value_json)
                elif is_allowed_metric_key(k):
                    found[k] = _val(it.value_json)
            if step is None and rec.history.HasField("step"):
                step = rec.history.step.num
            if isinstance(step, (int, float)) and not isinstance(step, bool):
                last_step = step if last_step is None else max(last_step, step)
            for k, v in found.items():
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    continue
                s = keys.setdefault(k, {"n": 0, "nonfinite": 0})
                s["n"] += 1
                if not math.isfinite(v):
                    s["nonfinite"] += 1
                    continue
                if "first" not in s:
                    s["first"], s["first_step"] = v, step
                s["last"], s["last_step"] = v, step
                if "min" not in s or v < s["min"]:
                    s["min"], s["min_step"] = v, step
                if "max" not in s or v > s["max"]:
                    s["max"], s["max_step"] = v, step
    out.update(header=header, env=env, exit=exit_rec, last_history_step=last_step,
               stats={"status": status, "keys": keys, "history_records": n_hist, "unparsed_records": n_bad})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/ayp/project/trm")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    meta = root / META
    cache_p = meta / "scan_cache.jsonl"
    old = {}
    if cache_p.exists():
        for ln in open(cache_p):
            r = json.loads(ln)
            old[r["id"]] = r
    snap = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    dirs = run_dirs(root)
    todo, keep = [], {}
    for rid, d in dirs.items():
        f = wandb_file(d)
        o = old.get(rid)
        if o and f and o.get("scan_version") == SCAN_VERSION and o.get("wandb_file") == os.path.realpath(f):
            st = os.stat(f)
            if st.st_size == o.get("wandb_size") and st.st_mtime_ns == o.get("wandb_mtime_ns"):
                keep[rid] = o
                continue
        todo.append((rid, d))
    print(f"run dirs {len(dirs)}  cached {len(keep)}  to scan {len(todo)}", flush=True)
    t0 = time.time()
    res = dict(keep)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(scan_one, sorted(todo, key=lambda x: x[0]), chunksize=1)):
            res[r["id"]] = r
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)
    bad = sum(1 for r in res.values() if r["stats"]["status"] != "ok")
    print(f"scanned {len(res)}  stats-not-ok {bad}  {time.time() - t0:.0f}s")
    if not res:
        print("ERROR: 0 runs scanned", file=sys.stderr)
        return 2
    if a.apply:
        meta.mkdir(parents=True, exist_ok=True)
        tmp = cache_p.with_suffix(".tmp")
        with open(tmp, "w") as fh:
            for rid in sorted(res):
                fh.write(json.dumps(res[rid], sort_keys=True, ensure_ascii=False, default=str) + "\n")
        os.replace(tmp, cache_p)
        snap_p = meta / "scan_snapshot.json"
        if not todo and snap_p.exists():  # nothing rescanned: keep the snapshot so cards stay byte-identical
            print("no run rescanned; scan_snapshot.json kept")
            return 0
        snap_p.write_text(json.dumps(
            {"snapshot_utc": snap, "n_runs": len(res), "stats_not_ok": bad,
             "wandb_dirs": WANDB_DIRS}, indent=1) + "\n")
        print("wrote", cache_p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
