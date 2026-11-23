"""EXP-027 P2: trivial-predictor floors of exact accuracy on the BUILT npy test splits.

Usage (CPU only):
  CUDA_VISIBLE_DEVICES="" uv run python tools/statebench_tiers/p2_floors.py \
      --root data/statebench_e027 --json lab/.../EXP-027_artifacts/p2_floors.json \
      [--compare-preview]

Predictors (all score exact = all m=5 ordinary registers right; token accuracy also reported):
  init       first literal written to each register (the initial assignment)
  lastlit    last literal written to each register (copies / increments ignored)
  mode       most frequent literal of the trace broadcast to all registers (ties -> smallest)
  modal_state  most frequent final 5-tuple of the TRAIN split, output for every test row
  modal_perm   (G only) most frequent net swap permutation of TRAIN; predict
               register i := initial value of register perm[i]
F_prereg = max exact over {init, lastlit, mode} (v2 s1.6; aperiodic `lastlit` excluded because it is
that tier's defining algorithm); F_all additionally includes modal_state and modal_perm.
A (tier, N) with F > THETA (0.20) leaves the N menu.  Exit 0.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp027_data_audit as AU  # noqa: E402

M, VAL0, INC0 = AU.M, AU.VAL0, AU.INC0
THETA = 0.20
FAMILY = {"G": "periodic", "S": "solvable", "A": "aperiodic"}


def literal_predictors(inp: np.ndarray):
    """Vectorised init / lastlit (rows, M) and mode (rows,) from the constant assignments."""
    rows, L = inp.shape
    n_op = L - M
    tg = inp[:, 0:n_op:2].astype(np.int64) - AU.REG0
    ag = inp[:, 1:n_op:2].astype(np.int64)
    is_const = (ag >= VAL0) & (ag < INC0)
    init = np.full((rows, AU.R), -1, dtype=np.int64)
    last = np.full((rows, AU.R), -1, dtype=np.int64)
    ar = np.arange(rows)
    for j in range(tg.shape[1]):
        c = is_const[:, j]
        t, v = tg[:, j], ag[:, j] - VAL0
        first = c & (init[ar, t] < 0)
        init[ar[first], t[first]] = v[first]
        last[ar[c], t[c]] = v[c]
    counts = np.zeros((rows, AU.Q), dtype=np.int32)
    for j in range(tg.shape[1]):
        c = is_const[:, j]
        np.add.at(counts, (ar[c], ag[c, j] - VAL0), 1)
    mode = counts.argmax(axis=1)  # argmax returns the smallest index on ties
    return init[:, :M], last[:, :M], mode


def score(pred: np.ndarray, truth: np.ndarray):
    hit = pred == truth
    return float(hit.all(axis=1).mean()), float(hit.mean())


def cell_floors(cell: Path, tier: str):
    tr, te = AU.load_split(cell, "train"), AU.load_split(cell, "test")
    n_op = te["inputs"].shape[1] - M
    truth_te = te["labels"][:, n_op:].astype(np.int64) - VAL0
    truth_tr = tr["labels"][:, n_op:].astype(np.int64) - VAL0
    init, last, mode = literal_predictors(te["inputs"])
    out = {}
    for name, pred in (("init", init), ("lastlit", last),
                       ("mode", np.repeat(mode[:, None], M, axis=1))):
        e, t = score(pred, truth_te)
        out[name] = {"exact": e, "token_acc": t}
    modal = Counter(map(bytes, np.ascontiguousarray(truth_tr))).most_common(1)[0][0]
    ms = np.frombuffer(modal, dtype=truth_tr.dtype)
    e, t = score(np.broadcast_to(ms, truth_te.shape), truth_te)
    out["modal_state"] = {"exact": e, "token_acc": t}
    if tier == "G":
        perms = Counter(AU.net_perm(AU.parse_swaps(r)[0]) for r in tr["inputs"])
        mp = perms.most_common(1)[0][0]
        e, t = score(init[:, list(mp)], truth_te)
        out["modal_perm"] = {"exact": e, "token_acc": t, "perm": list(mp),
                             "train_freq": perms[mp] / len(tr["inputs"])}
    prereg = [k for k in ("init", "lastlit", "mode") if not (tier == "A" and k == "lastlit")]
    f_pre = max(out[k]["exact"] for k in prereg)
    f_all = max(out[k]["exact"] for k in out if not (tier == "A" and k == "lastlit"))
    # train-split values of the literal predictors: a 100x larger sample, reported next to the
    # test-split floors to show how much of a test/preview gap is test-sampling noise
    ti, tl, tm = literal_predictors(tr["inputs"])
    train_ref = {name: score(pred, truth_tr)[0] for name, pred in
                 (("init", ti), ("lastlit", tl), ("mode", np.repeat(tm[:, None], M, axis=1)))}
    return {"train_split_exact": train_ref, "predictors": out, "F_prereg": f_pre, "F_all": f_all,
            "test_rows": int(te["inputs"].shape[0]), "train_rows": int(tr["inputs"].shape[0])}


def run_preview(Ns, samples, root: Path):
    """Run p2_floor_preview.py at c=0 (its --ap-const 0 0) and return its JSON rows."""
    tmp = root / "_preview_tmp.json"
    subprocess.run([sys.executable, str(Path(__file__).resolve().parent / "p2_floor_preview.py"),
                    "--Ns", *map(str, Ns), "--samples", str(samples), "--ap-const", "0", "0",
                    "--json", str(tmp)], check=True, stdout=subprocess.DEVNULL)
    rows = json.loads(tmp.read_text())
    tmp.unlink()
    return {r["N"]: r for r in rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default="data/statebench_e027")
    ap.add_argument("--json", required=True)
    ap.add_argument("--compare-preview", action="store_true")
    ap.add_argument("--preview-samples", type=int, default=20000)
    ap.add_argument("--tol", type=float, default=0.01)
    a = ap.parse_args(argv)
    root = Path(a.root)
    cells = sorted(p for p in root.iterdir() if p.is_dir() and "_N" in p.name)
    res = {}
    for c in cells:
        tier, N = c.name.split("_N")
        res[c.name] = {"tier": tier, "N": int(N), **cell_floors(c, tier)}
    exceed_pre = sorted(k for k, v in res.items() if v["F_prereg"] > THETA)
    exceed_all = sorted(k for k, v in res.items() if v["F_all"] > THETA)
    report = {"theta": THETA, "cells": res, "exceeding_theta_prereg_predictors": exceed_pre,
              "exceeding_theta_all_predictors": exceed_all}
    if "G_N12" in res:
        s = json.loads((Path(a.json).parent / "p3_audit_G_N12.json").read_text()) \
            if (Path(a.json).parent / "p3_audit_G_N12.json").exists() else None
        if s:
            ts = s["tier_stats"]
            report["G_N12_swap_overlap"] = {
                "frac_rows_with_overlapping_swaps": ts["frac_rows_with_overlapping_swaps"],
                "frac_rows_net_permutation_nontrivial": ts["frac_rows_net_permutation_nontrivial"],
                "swap_count_hist": ts["swap_count_hist"], "source": "p3_audit_G_N12.json (train split)"}
    if a.compare_preview:
        prev = run_preview(sorted({v["N"] for v in res.values()}), a.preview_samples, Path(a.json).parent)
        cmp_rows, worst = [], 0.0
        for k, v in res.items():
            p = prev[v["N"]]["floors"][FAMILY[v["tier"]]]
            for name in ("init", "lastlit", "mode"):
                d = abs(v["predictors"][name]["exact"] - p[name]["exact"])
                dt = abs(v["predictors"][name]["token_acc"] - p[name]["token_acc"])
                worst = max(worst, d)
                cmp_rows.append({"cell": k, "predictor": name, "built": v["predictors"][name]["exact"],
                                 "preview": p[name]["exact"], "train_split_exact": v["train_split_exact"][name],
                                 "abs_diff_vs_train_split": abs(v["train_split_exact"][name] - p[name]["exact"]),
                                 "abs_diff": d, "token_abs_diff": dt})
        report["preview_comparison"] = {"preview_samples": a.preview_samples, "preview_ap_const": [0, 0],
                                        "tolerance": a.tol, "max_abs_diff_exact": worst,
                                        "all_within_tolerance": worst <= a.tol,
                                        "max_abs_diff_exact_vs_train_split":
                                        max(r["abs_diff_vs_train_split"] for r in cmp_rows),
                                        "rows": cmp_rows}
    Path(a.json).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(f"{'cell':8} {'init':>7} {'lastlit':>8} {'mode':>7} {'modal_st':>9} {'modal_perm':>10} {'F_pre':>7} {'F_all':>7}")
    for k, v in res.items():
        pr = v["predictors"]
        print(f"{k:8} {pr['init']['exact']:7.4f} {pr['lastlit']['exact']:8.4f} {pr['mode']['exact']:7.4f} "
              f"{pr['modal_state']['exact']:9.4f} {pr.get('modal_perm', {}).get('exact', float('nan')):10.4f} "
              f"{v['F_prereg']:7.4f} {v['F_all']:7.4f}")
    print("exceeding theta (prereg predictors):", exceed_pre, "| (all predictors):", exceed_all)
    if a.compare_preview:
        pc = report["preview_comparison"]
        print("preview max abs diff (exact):", round(pc["max_abs_diff_exact"], 4), "within tol:", pc["all_within_tolerance"],
              "| max diff train-split vs preview:", round(pc["max_abs_diff_exact_vs_train_split"], 4))
    return 0


if __name__ == "__main__":
    sys.exit(main())
