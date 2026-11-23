"""P-C1 M-1 manifest (spec §7 step 3): terminal ``probe/test_exact`` + ``train_final`` of the 30 g7 / full cells.

Two phases, because the wandb reader needs the trm-mcp venv (no numpy / yaml there; repo venv lacks ``mcp``):

  (1) extract  [trm-mcp venv, stdlib only]
      /home/ayp/project/trm-mcp/.venv/bin/python3 analysis/ltf_decode/cells.py extract --out M1_raw.json
      Same reader as EXP-015 (E15:576-588): wr.list_runs(name_glob="pp_ltf_*") + RT.fetch_probe
      (probe/test_exact queried alone; X-4) + wr.get_run_summary. Canonical run per name = among runs whose
      summary ``_step`` >= 122050 the one with the latest ``started_at`` (exp015_extract.ltf_rows). noinj is NOT
      queried (not a P-C1 cell). Only public terminal scalars are written (no per-loop value exists here).
  (2) classify [repo venv]
      CUDA_VISIBLE_DEVICES="" uv run --no-sync python analysis/ltf_decode/cells.py classify --raw M1_raw.json \
          --out-csv M1_manifest.csv --out-json M1_manifest.json
      G/C/P/U from the parsed FROZEN_RULES class_rules (judge.outcome_class), family membership, comparison with
      ``families_expected`` (disagreements are REPORTED, never patched).

Terminal value = the ``probe/test_exact`` value at the latest logged eval step; that step must equal 122050 and
the series must not be downsampled, else the cell is flagged (``terminal_ok`` false).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

REPO = "/home/ayp/project/trm"
PROJECT = "power_permutation"
TERMINAL_STEP = 122050
K_LIST = (3, 5, 6, 7, 10)
SEEDS = (1, 2, 3)
ARMS = {"g7": "g7", "fullgrad": "full"}          # run tag -> arm label
FIELDS = ["run_name", "arm", "k", "seed", "run_id", "n_dirs_for_name", "summary_step", "probe_last_step",
          "probe_downsampled", "terminal_ok", "final_probe_test_exact", "train_final_proxy", "class",
          "tags"]


def require_cpu(device: str = "cpu") -> None:
    """Spec §4.1 CPU fail-closed: SystemExit(2) unless CUDA_VISIBLE_DEVICES == "" and device == "cpu" and (when torch
    is importable) torch.cuda.is_available() is False. The extract phase runs in the stdlib-only trm-mcp venv
    (no torch): there the third condition cannot be evaluated and is skipped (nothing is loaded on any device)."""
    problems = []
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        problems.append(f'CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")!r} (must be "")')
    if device != "cpu":
        problems.append(f"--device {device!r} (must be 'cpu')")
    try:
        import torch
        if torch.cuda.is_available():
            problems.append("torch.cuda.is_available() is True")
    except ImportError:
        pass
    if problems:
        print("CPU fail-closed: refusing: " + "; ".join(problems), file=sys.stderr)
        raise SystemExit(2)


def cell_names() -> list[tuple[str, str, int, int]]:
    """(run_name, arm, k, seed) for the 30 cells: g7 then fullgrad, k-major, seed-minor."""
    return [(f"pp_ltf_{tag}_lay2_k{k}_s{s}", arm, k, s) for tag, arm in ARMS.items() for k in K_LIST for s in SEEDS]


# ------------------------------------------------------------------ phase 1: extract (stdlib only)

def extract_raw(out_path: str) -> dict:
    sys.path.insert(0, os.path.join(REPO, "analysis"))
    import pp_campaign_retro_trajectory as RT  # noqa: E402  (same module exp015_extract.py imports)
    wr = RT.wr
    runs = wr.list_runs(RT.REPO, project=RT.PROJECT, name_glob="pp_ltf_*", limit=500)
    by_name: dict[str, list] = {}
    for r in runs:
        by_name.setdefault(r["run_name"], []).append(r)
    rows = []
    for name, arm, k, s in cell_names():
        cands = by_name.get(name, [])
        term = []
        for c in cands:
            st = wr.get_run_summary(RT.REPO, RT.PROJECT, c["run_id"])["summary"].get("_step")
            if st is not None and st >= TERMINAL_STEP:
                term.append(c)
        pick = max(term, key=lambda c: c["started_at"]) if term else None
        row = dict(run_name=name, arm=arm, k=k, seed=s, run_id=pick["run_id"] if pick else "",
                   n_dirs_for_name=len(cands), n_terminal_candidates=len(term))
        if pick:
            summ = wr.get_run_summary(RT.REPO, RT.PROJECT, pick["run_id"])["summary"]
            fp = RT.fetch_probe(pick["run_id"])
            test = fp["test"]
            row.update(summary_step=summ.get("_step"), train_final=summ.get("train/exact_accuracy"),
                       probe_downsampled=bool(fp["downsampled"]), n_probe_points=len(test),
                       probe_last_step=test[-1][0] if test else None,
                       final_probe_test_exact=test[-1][1] if test else None,
                       probe_last5=[[a, b] for a, b in test[-5:]])
        rows.append(row)
    out = {"source": "wandb local datastore via trm_mcp.wandb_reader (EXP-015 extraction path, E15:576-588)",
           "keys": ["probe/test_exact", "train/exact_accuracy (summary)"], "cells": rows}
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    return out


# ------------------------------------------------------------------ phase 2: classify (repo venv)

def classify_raw(raw: dict) -> list[dict]:
    from analysis.ltf_decode import judge
    out = []
    for r in raw["cells"]:
        fp = r.get("final_probe_test_exact")
        tr = r.get("train_final")
        tr = float(tr) if tr not in (None, "") else None
        terminal_ok = bool(r.get("run_id")) and r.get("probe_last_step") == TERMINAL_STEP and not r.get("probe_downsampled")
        cls = judge.outcome_class(None if fp is None else float(fp), tr) if r.get("run_id") else "MISSING"
        out.append(dict(run_name=r["run_name"], arm=r["arm"], k=r["k"], seed=r["seed"], run_id=r.get("run_id", ""),
                        n_dirs_for_name=r["n_dirs_for_name"], summary_step=r.get("summary_step"),
                        probe_last_step=r.get("probe_last_step"), probe_downsampled=r.get("probe_downsampled"),
                        terminal_ok=terminal_ok, final_probe_test_exact=fp, train_final_proxy=tr, **{"class": cls}))
    kboth = k_both(out)
    for r in out:
        r["tags"] = ";".join(membership(r, kboth))
    return out


def k_both(rows: list[dict]) -> list[int]:
    g = {arm: {r["k"] for r in rows if r["arm"] == arm and r["class"] == "G" and r["terminal_ok"]} for arm in ("g7", "full")}
    return sorted(g["g7"] & g["full"])


def membership(r: dict, kboth: list[int]) -> list[str]:
    """Family tags (final membership = M-1 class; gates are applied at run time)."""
    tags = []
    if r["class"] == "G" and r["terminal_ok"]:
        tags.append(f"{r['arm']}_G")
        if r["arm"] == "g7" and r["k"] in kboth:
            tags.append("g7_G_Kboth")
    elif r["class"] in ("C", "P", "U"):
        tags.append(f"{r['arm']}_{r['class']}_D4")       # D-4 train-correspondence layer (descriptive)
    return tags


def families(rows: list[dict]) -> dict[str, list[str]]:
    fam: dict[str, list[str]] = {"g7_G": [], "full_G": [], "g7_G_Kboth": []}
    for r in rows:
        for t in r["tags"].split(";") if isinstance(r["tags"], str) else r["tags"]:
            if t in fam:
                fam[t].append(r["run_name"])
    return fam


def compare_expected(rows: list[dict]) -> dict:
    """M-1 membership vs the prereg lists. g7_G / full_G: set equality. Descriptive lists: every listed run must
    exist in the manifest with a class in the allowed set (g7_P_reach_then_below_0p9 -> P; full_C_or_P -> C or P),
    and the union of all descriptive lists must equal the set of non-G cells."""
    from analysis.ltf_decode import judge
    fam = families(rows)
    exp = judge.rule("families_expected")
    cmp = {}
    for key in ("g7_G", "full_G"):
        want, got = set(exp[key]), set(fam[key])
        cmp[key] = {"match": want == got, "expected_not_in_M1": sorted(want - got), "M1_not_expected": sorted(got - want)}
    desc = judge.rule("descriptive_only")
    allowed = {"g7_P_reach_then_below_0p9": {"P"}, "full_C_or_P": {"C", "P"}}
    dp = {r["run_name"]: r["class"] for r in rows}
    union = set()
    for key, names in desc.items():
        ok_set = allowed.get(key)
        classes = {n: dp.get(n) for n in names}
        bad = sorted(n for n, c in classes.items() if ok_set is None or c not in ok_set)
        cmp[key] = {"match": not bad, "classes": classes, "outside_allowed": bad}
        union |= set(names)
    nong = {r["run_name"] for r in rows if r["class"] != "G"}
    cmp["descriptive_union_equals_non_G"] = {"match": union == nong, "listed_not_nonG": sorted(union - nong),
                                             "nonG_not_listed": sorted(nong - union)}
    return cmp


def write_manifest(rows: list[dict], csv_path: str, json_path: str | None = None) -> None:
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in FIELDS})
    if json_path:
        with open(json_path, "w") as f:
            json.dump({"cells": rows, "K_both": k_both(rows), "families": families(rows),
                       "vs_families_expected": compare_expected(rows)}, f, indent=1, default=str)


def read_manifest(path: str) -> list[dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["k"], r["seed"] = int(r["k"]), int(r["seed"])
        r["final_probe_test_exact"] = float(r["final_probe_test_exact"]) if r["final_probe_test_exact"] != "" else None
        r["terminal_ok"] = r["terminal_ok"] == "True"
    return rows


def counts_by_arm_class(rows: list[dict]) -> dict:
    out: dict = {}
    for r in rows:
        out.setdefault(r["arm"], {}).setdefault(r["class"], 0)
        out[r["arm"]][r["class"]] += 1
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--out", required=True)
    e.add_argument("--device", default="cpu")
    c = sub.add_parser("classify")
    c.add_argument("--raw", required=True)
    c.add_argument("--out-csv", required=True)
    c.add_argument("--out-json")
    c.add_argument("--device", default="cpu")
    a = ap.parse_args()
    require_cpu(a.device)
    if a.cmd == "extract":
        res = extract_raw(a.out)
        print("wrote", a.out, len(res["cells"]), "cells")
    else:
        if __package__ in (None, ""):
            sys.path.insert(0, REPO)
        with open(a.raw) as f:
            raw = json.load(f)
        rows = classify_raw(raw)
        write_manifest(rows, a.out_csv, a.out_json)
        print("wrote", a.out_csv, len(rows), "cells; counts", counts_by_arm_class(rows), "K_both", k_both(rows))
