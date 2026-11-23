#!/usr/bin/env python3
"""Extract allow-listed training-curve / z-dynamics series for one EXP into a long CSV.
Run with the trm-mcp venv:  /home/ayp/project/trm-mcp/.venv/bin/python tools/wandb_run_index/curves.py EXP-017 OUT.csv
Reads only trm_mcp's allowlisted keys (no sensitive fields); run list from _wandb_runs_index.jsonl."""
import csv, json, sys
from pathlib import Path
sys.path.insert(0, "/home/ayp/project/trm-mcp/src")
from trm_mcp.wandb_reader import get_metric_series

ROOT = Path("/home/ayp/project/trm")
exp, out = sys.argv[1], sys.argv[2]
KEYS = ["all/exact_accuracy", "all/lm_loss", "train/exact_accuracy", "train/lm_loss",
        "z/eff_rank", "phase/index", "probe/test_exact"]
runs = [json.loads(l) for l in open(ROOT / "lab/21_experiments/11_lab-experiments/_wandb_runs_index.jsonl")]
runs = [r for r in runs if exp in r["exps"]]
n = 0
with open(out, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["run_name", "run_id", "step", "key", "value"])
    for r in sorted(runs, key=lambda r: r["name"]):
        try:
            res = get_metric_series(ROOT, r["project"], r["id"], KEYS, max_points=60)
        except Exception as e:
            print("SKIP", r["name"], type(e).__name__, str(e)[:80], file=sys.stderr); continue
        pts = res.get("series") or res.get("points") or res.get("rows") or []
        if n == 0: print("keys:", list(res)[:8], "sample:", str(pts[:1])[:200], file=sys.stderr)
        for p in pts:
            step = p.get("step") or p.get("_step")
            for k in KEYS:
                if p.get(k) is not None:
                    w.writerow([r["name"], r["id"], step, k, p[k]]); n += 1
print("rows", n, "runs", len(runs))
