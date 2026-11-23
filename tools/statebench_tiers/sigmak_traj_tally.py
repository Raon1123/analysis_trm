"""C1: sigma^k trajectory re-tally with outcomes.py (design 03_DESIGN_DIALECTIC.md s2.1).

Cohort: runs named [H108N<d>_]fig1_tf_{z,noz}_iter_k<k>_s<seed>, config data_paths ==
[data/sigma_k_10/<k>], project_name in {Sigma_k_new, power_permutation}.
Read-only on wandb/ and checkpoints/. Output is deterministic (sorted, no timestamps).
usage: CUDA_VISIBLE_DEVICES="" uv run python tools/statebench_tiers/sigmak_traj_tally.py [out.json]
"""
import json
import os
import re
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import outcomes as oc  # noqa: E402

PROJECTS = ("Sigma_k_new", "power_permutation")
OUT = os.path.join(oc.REPO, "lab/21_experiments/11_lab-experiments/EXP-027_artifacts/c1_sigmak_traj_tally.json")
NAME_RE = re.compile(r"(H108N\d_)?fig1_tf_(z|noz)_iter_k(\d+)_s(\d+)")
DIFF_K, DIFF_VARIANT = 7, "z"
IGNORE_DIFF = {"_wandb"}


def scan(run_dir):
    try:
        cfg = oc.run_config(run_dir)
    except Exception:
        return None
    m = NAME_RE.fullmatch(cfg.get("run_name") or "")
    if not m or cfg.get("data_paths") != [f"data/sigma_k_10/{m.group(3)}"]:
        return None
    if cfg.get("project_name") not in PROJECTS:
        return None
    wf = oc._wandb_file(run_dir)
    if wf is None:
        return None
    row = dict(wandb_id=os.path.basename(run_dir).split("-")[-1], run_dir=os.path.basename(run_dir),
               run_name=cfg["run_name"], project=cfg["project_name"], variant=m.group(2),
               k=int(m.group(3)), seed=int(m.group(4)), prefix=m.group(1) or "")
    ck = os.path.join(oc.REPO, "checkpoints", row["project"], row["run_name"])
    row["checkpoint_path"] = f"checkpoints/{row['project']}/{row['run_name']}" if os.path.exists(ck) else None
    try:
        traj = oc.load_trajectory_file(wf)
    except Exception as e:  # recorded, not hidden
        row["error"] = repr(e)[:200]
        return row
    row.update(oc.summarize(traj))
    row["last3"] = [round(x, 3) for _, x in traj[-3:]]
    row["last_step"] = traj[-1][0]
    row["cfg"] = {k: v for k, v in cfg.items() if k not in IGNORE_DIFF}
    return row


def config_diff(rows):
    sel = {r["seed"]: r for r in rows if r["k"] == DIFF_K and r["variant"] == DIFF_VARIANT and "cfg" in r}
    base = sel[1]["cfg"]
    out = {}
    for s in sorted(sel):
        if s == 1:
            continue
        c = sel[s]["cfg"]
        d = {k: [base.get(k), c.get(k)] for k in sorted(set(base) | set(c)) if base.get(k) != c.get(k)}
        out[f"s1_vs_s{s}"] = dict(wandb_ids=[sel[1]["wandb_id"], sel[s]["wandb_id"]], projects=[sel[1]["project"], sel[s]["project"]], differing_keys=d)
    return out


def aggregate(rows):
    ok = [r for r in rows if "error" not in r]
    agg = dict(n_runs=len(rows), n_error=len(rows) - len(ok))
    low = [r for r in ok if r["final"] < 0.9]
    agg["final_lt_0.9"] = dict(n=len(low), z=sum(r["variant"] == "z" for r in low), noz=sum(r["variant"] == "noz" for r in low),
                               learned_then_collapsed=sum(r["Y"] == 1 for r in low), never_learned_Y0=sum(r["Y"] == 0 for r in low),
                               never_learned_max_le_027=sum(r["max"] <= 0.027 + 1e-6 for r in low if r["Y"] == 0))
    agg["collapse_flag_total"] = sum(r["collapse"] for r in ok)
    agg["Y_by_variant"] = {v: dict(n=sum(r["variant"] == v for r in ok), Y1=sum(r["Y"] for r in ok if r["variant"] == v)) for v in ("z", "noz")}
    # moderator criterion: >= 2 evals with E >= 0.90 (not necessarily consecutive); needs the raw count
    agg["per_k"] = {}
    for v in ("z", "noz"):
        for k in sorted({r["k"] for r in ok}):
            g = [r for r in ok if r["variant"] == v and r["k"] == k]
            ts = sorted(r["tau"] for r in g if r["tau"] is not None)
            agg["per_k"][f"{v}_k{k}"] = dict(n=len(g), Y1=sum(r["Y"] for r in g), tau_median=ts[len(ts) // 2] if len(ts) % 2 else (sum(ts[len(ts) // 2 - 1:len(ts) // 2 + 1]) / 2 if ts else None),
                                             tau_min=ts[0] if ts else None, tau_max=ts[-1] if ts else None,
                                             final_lt_09=sum(r["final"] < 0.9 for r in g), collapse=sum(r["collapse"] for r in g))
    return agg


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else OUT
    dirs = oc.wandb_run_dirs()
    with Pool(16) as p:
        rows = [r for r in p.map(scan, dirs) if r]
    rows.sort(key=lambda r: (r["variant"], r["k"], r["seed"], r["wandb_id"]))
    keys = {}
    for r in rows:
        keys.setdefault((r["variant"], r["k"], r["seed"]), []).append(r["wandb_id"])
    dups = {f"{v}_k{k}_s{s}": ids for (v, k, s), ids in keys.items() if len(ids) > 1}
    diff = config_diff(rows)
    for r in rows:
        r.pop("cfg", None)
    res = dict(script="tools/statebench_tiers/sigmak_traj_tally.py", thresholds=oc.THRESH.__dict__,
               aggregate=aggregate(rows), duplicate_cells=dups, config_diff_k7_z=diff, runs=rows)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=1, sort_keys=True)
    print(json.dumps(res["aggregate"], indent=1, sort_keys=True))
    print("dups", dups)


if __name__ == "__main__":
    main()
