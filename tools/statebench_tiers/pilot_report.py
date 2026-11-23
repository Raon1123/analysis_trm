#!/usr/bin/env python3
"""EXP-027 pilot report tooling: descriptive tables (D1-D13) and figures (F1-F5) from eval trajectories.

Status: DRAFT, author_model claude-sonnet-5-5.  NOT frozen, NOT part of the freeze witness.  No GPU.
Read-only on wandb/ and (with --extract-timings) on logs/queue/.  Output: figures and data only, no prose.

This module never decides anything.  It imports the frozen outcomes.py for every tau / Y / BSE / flag value and
exp027_judge.py only for pick_run (same run-selection logic), y_consec and SENSITIVITY.  It does not call
exp027_select.select / pool_rows or any rule of exp027_rules_v3.py.  Quantities that belong to the selection
(path, trace, late_flag, qualifiers, extension, ext_gate) are copied from the judge / ext_gate JSON files that the
caller passes in (--judge-json, --ext-gate-json).

Ordering gate (moderator's retrospective R1, list 4).  Every variant-split view (D13, variant colouring of
F1-F5, per-variant Y(T) lines) needs --selection-witness FILE --witness-sha256 HEX; the hash is verified and
recorded.  Without them figures are variant-blind (one neutral colour, seeds by line style) and no D13 file is
written.  The per-run table always carries the variant column, as the judge rows do.

Usage (default: the 17 pilot runs, e027_pilot_<tier>N<N>_<variant>_s<seed>, T_p = 41013 steps):
  CUDA_VISIBLE_DEVICES="" uv run python tools/statebench_tiers/pilot_report.py --t-max-steps 41013 \
      [--runs NAME ...] [--judge-json J.json] [--ext-gate-json G.json] [--smoke SMOKE2.json] \
      [--extract-timings [--log-dir logs/queue]] [--selection-witness W.json --witness-sha256 HEX] \
      [--out-dir reports/2026-11-12_exp027-pilot] [--wandb-root DIR] [--no-train]
  Test mode on other runs (refuses any e027 name):
      --names-override LABEL=WANDBID ...   (LABEL = fig1_tf_<z|noz>_iter_k<k>_s<seed>, WANDBID = 8-char wandb id)
      with --t-max-steps = last eval step and optionally --eval-step (default: inferred from the trajectories).
exit: 0 ok, 1 data missing / invalid, 2 usage, 4 a run is not judgeable yet (single incomplete dir; strict only).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import outcomes as O  # noqa: E402
import exp027_judge as J  # noqa: E402   (pick_run, y_consec, SENSITIVITY, RUN_NAME only)
import exp027_select as S  # noqa: E402  (EVAL_STEP, steps_for_epochs, pilot_run_specs, R0_MAX: constants only)

REPO = O.REPO
ART = os.path.join(REPO, "lab/21_experiments/11_lab-experiments/EXP-027_artifacts")
DEFAULT_OUT = os.path.join(REPO, "reports/2026-11-12_exp027-pilot")
DEFAULT_SMOKE = os.path.join(ART, "SMOKE2.json")
PLAN_PILOT17 = 111.773          # GPU-h of the 17 pilot runs at T_p = 420 epochs (ART/PLAN_pilot.arith.txt)
PLAN_WITH_SMOKE = 116.614       # the same plus the 4.841 GPU-h smoke runs (ART/PLAN_pilot.arith.txt)
TIER_RANK = {"A": 0, "S": 1, "G": 2, "K": 3}
SIGMAK_NAME = re.compile(r"^(?:H108N\d_)?fig1_tf_(z|noz)_iter_k(\d+)_s(\d+)$")
CLASSES = ("learned-stable", "learned-then-collapsed", "graded", "pretakeoff", "never-learned")
HIST_EDGES = [i / 10 for i in range(11)]
LOGIT_CLIP = 0.0005             # half of the 0.001 eval grid
FS_MIN = 9


# ---------------------------------------------------------------- names and metadata
def default_names() -> list:
    return [f"e027_pilot_{t}N{n}_{v}_s{s}" for (t, n, v, s) in S.pilot_run_specs()]


def parse_meta(name: str) -> dict:
    """e027 run name -> meta (stages pilot / ext only); sigma^k test names -> tier 'K', N = k, stage 'override'."""
    m = SIGMAK_NAME.match(name)
    if m:
        return {"stage": "override", "tier": "K", "N": int(m.group(2)), "variant": m.group(1),
                "seed": int(m.group(3))}
    meta = J.parse_name(name)
    if meta["stage"] not in ("pilot", "ext"):
        raise ValueError(f"{name}: pilot_report takes stages pilot / ext only")
    return meta


def cell_label(tier, N) -> str:
    return f"{tier}{N}"


# ---------------------------------------------------------------- rows
def classify(r: dict) -> str:
    """Five outcome-shape classes of one run row; precedence graded > pretakeoff > never-learned among Y=0."""
    if r["Y"]:
        return "learned-then-collapsed" if r["collapse"] else "learned-stable"
    if r["graded"]:
        return "graded"
    if r["pretakeoff"]:
        return "pretakeoff"
    return "never-learned"


def make_row(name: str, traj: list, meta: dict, eval_step: int, t_max: int) -> dict:
    """Per-run row.  All outcome values come from outcomes.summarize (plus J.y_consec for the sensitivity variants)."""
    O.check_finite(traj, name)
    s = O.summarize(traj)
    if s["BSE"] is None:
        raise ValueError(f"{name}: fewer than 2 evals, BSE undefined")
    tau = s["tau"]
    steps = [st for st, _ in traj]
    n_after = None if tau is None else sum(1 for st in steps if st > tau)
    r = {"run_name": name, **meta, "n_evals": s["n_evals"], "tau": tau,
         "tau_over_eval_step": None if tau is None else tau / eval_step,
         "tau_frac_Tp": None if tau is None else tau / t_max,
         "Y": s["Y"], "BSE": s["BSE"], "max": s["max"], "final": s["final"], "mean_last5": s["mean_last5"],
         "stability": s["stability"], "max_minus_final": s["max"] - s["final"], "n_evals_after_tau": n_after,
         "collapse": s["collapse"], "graded": s["graded"], "pretakeoff": s["pre_takeoff"],
         "Y_sens": {lab: J.y_consec(traj, lr, mm) for lab, lr, mm in J.SENSITIVITY}}
    r["class"] = classify(r)
    return r


# ---------------------------------------------------------------- collection (wandb or injected)
def infer_eval_step(trajs: list) -> int:
    d = sorted(b - a for tr in trajs for (a, _), (b, _) in zip(tr, tr[1:]))
    if not d:
        raise ValueError("no trajectory with two evals")
    return int(d[len(d) // 2])


def read_train_series(path: str, t_max: int, bins: int = 400) -> dict:
    """One pass over a .wandb file: binned train loss / exact accuracy / token accuracy up to t_max, the final
    train values (mean of the last 50 logged values at step <= t_max) and the total _runtime (seconds)."""
    sys.path.insert(0, os.path.join(REPO, "tools", "figures"))
    from wandb_ro import iter_history
    keys = ["train/lm_loss", "train/exact_accuracy", "train/accuracy", "_runtime"]
    rec = []
    runtime = None
    for st, v in iter_history(path, keys):
        if "_runtime" in v:
            runtime = max(runtime or 0.0, float(v["_runtime"]))
        if st <= t_max and "train/lm_loss" in v:
            rec.append((st, v["train/lm_loss"], v.get("train/exact_accuracy"), v.get("train/accuracy")))
    return summarize_train(rec, t_max, bins, runtime)


def summarize_train(rec: list, t_max: int, bins: int = 400, runtime=None) -> dict:
    """rec: [(step, loss, exact_acc|None, tok_acc|None)] -> binned series dict (see read_train_series)."""
    out = {"runtime_s": runtime, "steps": [], "loss": [], "acc": [], "tok_acc": [], "final_train_loss": None,
           "final_train_acc": None}
    if not rec:
        return out
    rec = sorted(rec, key=lambda x: x[0])
    last = rec[-50:]
    out["final_train_loss"] = math.fsum(x[1] for x in last) / len(last)
    acc = [x[2] for x in last if x[2] is not None]
    out["final_train_acc"] = math.fsum(acc) / len(acc) if acc else None
    width = max(1, math.ceil((t_max + 1) / bins))
    grp: dict = {}
    for st, lo, ac, ta in rec:
        grp.setdefault(st // width, []).append((st, lo, ac, ta))
    for b in sorted(grp):
        g = grp[b]
        out["steps"].append(math.fsum(x[0] for x in g) / len(g))
        out["loss"].append(math.fsum(x[1] for x in g) / len(g))
        for key, ix in (("acc", 2), ("tok_acc", 3)):
            vv = [x[ix] for x in g if x[ix] is not None]
            out[key].append(math.fsum(vv) / len(vv) if vv else None)
    return out


def collect(names: list, t_max: int, wandb_root=None, injected=None, with_train=True, id_of=None) -> dict:
    """{'runs': {name: {...}}, 'excluded': [...], 'waiting': [...]}.  Same-name dirs are resolved with
    exp027_judge.pick_run (the judge's rule).  injected: {name: traj | [traj per dir] | {'cands': [...], 'train':
    {...}, 'runtime_s': x, 'wandb_id': s}} replaces wandb (tests).  id_of: {name: wandb id} for the test mode."""
    cand: dict = {}
    extra: dict = {}
    if injected is not None:
        for n in names:
            v = injected[n]
            if isinstance(v, dict):
                extra[n] = v
                v = v["cands"]
            tr = J._as_candidates(v)
            cand[n] = [(f"run-injected-{n}-{i}", t) for i, t in enumerate(tr)]
    else:
        queries = [(id_of or {}).get(n, n) for n in names]
        got = O.load_candidates(queries, root=wandb_root)
        for n, q in zip(names, queries):
            cand[n] = got[q]
    res, excluded, waiting = {}, [], []
    for n in names:
        trs = [t for _, t in cand[n]]
        for k, t in enumerate(trs):
            O.check_finite(t, f"{n} (dir {k})")
        try:
            p = J.pick_run(trs, t_max)
        except J.NotJudgeable as e:
            waiting.append({"run_name": n, "reason": str(e)})
            continue
        if p["status"] != "complete":
            excluded.append({"run_name": n, "n_dirs": p["n_dirs"], "last_steps": p["last_steps"]})
            continue
        idx = max(i for i, t in enumerate(trs) if t and t[-1][0] >= t_max)
        run_dir = cand[n][idx][0]
        e = extra.get(n, {})
        info = {"traj": p["traj"], "run_dir": os.path.basename(run_dir), "n_dirs": p["n_dirs"],
                "wandb_id": e.get("wandb_id", os.path.basename(run_dir).split("-")[-1]),
                "train": e.get("train"), "runtime_s": e.get("runtime_s")}
        if injected is None and with_train:
            wf = O._wandb_file(run_dir)
            if wf:
                tr = read_train_series(wf, t_max)
                info["train"], info["runtime_s"] = tr, tr["runtime_s"]
        elif info["train"] is not None and info["runtime_s"] is None:
            info["runtime_s"] = info["train"].get("runtime_s")
        res[n] = info
    return {"runs": res, "excluded": excluded, "waiting": waiting}


# ---------------------------------------------------------------- tables
def state_y(n: int, n_y1: int) -> str:
    return "FLOOR" if n_y1 == 0 else ("CEIL" if n_y1 == n else "MIX")


def _med(x):
    x = sorted(x)
    if not x:
        return None
    m = len(x) // 2
    return float(x[m]) if len(x) % 2 else (x[m - 1] + x[m]) / 2.0


def hist(vals: list) -> list:
    """Counts in [0,.1) .. [.9,1.0] (last bin closed)."""
    out = [0] * 10
    for v in vals:
        out[min(int(v * 10 + 1e-9), 9)] += 1
    return out


def cell_keys(rows: list) -> list:
    return sorted({(r["tier"], r["N"]) for r in rows}, key=lambda c: (TIER_RANK.get(c[0], 9), c[1]))


def cell_table(rows: list) -> list:
    """D1 / D2 / D3 / D4 / D8 per (tier, N), variant-pooled."""
    out = []
    for (t, N) in cell_keys(rows):
        g = [r for r in rows if r["tier"] == t and r["N"] == N]
        n, n1 = len(g), sum(r["Y"] for r in g)
        c = {"cell": cell_label(t, N), "tier": t, "N": N, "n": n, "n_y1": n1, "f": n1 / n, "state_Y": state_y(n, n1),
             "classes": {k: sum(1 for r in g if r["class"] == k) for k in CLASSES},
             "n_collapse": sum(1 for r in g if r["collapse"]), "n_graded": sum(1 for r in g if r["graded"]),
             "n_pretakeoff_y0": sum(1 for r in g if r["pretakeoff"] and not r["Y"]),
             "BSE": sorted(r["BSE"] for r in g), "BSE_hist": hist([r["BSE"] for r in g]),
             "max": sorted(r["max"] for r in g), "max_hist": hist([r["max"] for r in g]),
             "tau_over_eval_step": sorted(r["tau_over_eval_step"] for r in g if r["Y"]),
             "tau_frac_Tp": sorted(r["tau_frac_Tp"] for r in g if r["Y"]),
             "n_censored_at_Tp": sum(1 for r in g if not r["Y"]),
             "final_median": _med([r["final"] for r in g]), "final_min": min(r["final"] for r in g),
             "final_max": max(r["final"] for r in g),
             "mean_last5_median": _med([r["mean_last5"] for r in g]),
             "mean_last5_min": min(r["mean_last5"] for r in g), "mean_last5_max": max(r["mean_last5"] for r in g),
             "max_median": _med([r["max"] for r in g])}
        sens = {}
        for lab, _lr, _m in J.SENSITIVITY:
            k1 = sum(r["Y_sens"][lab] for r in g)
            st = state_y(n, k1)
            sens[lab] = {"n_y1": k1, "state_Y": st, "changed_vs_base": st != c["state_Y"]}
        c["sensitivity"] = sens
        c["n_changed_variants"] = sum(1 for v in sens.values() if v["changed_vs_base"])
        out.append(c)
    return out


def per_variant_table(rows: list) -> list:
    """D13 (gated).  Per (tier, N, variant): k of n, BSE and tau lists.  Not a theta estimate."""
    out = []
    for (t, N) in cell_keys(rows):
        for v in ("z", "noz"):
            g = [r for r in rows if r["tier"] == t and r["N"] == N and r["variant"] == v]
            if g:
                out.append({"cell": cell_label(t, N), "tier": t, "N": N, "variant": v, "n": len(g),
                            "n_y1": sum(r["Y"] for r in g), "BSE": [r["BSE"] for r in g],
                            "tau": [r["tau"] for r in g], "seeds": [r["seed"] for r in g]})
    return out


def y_of_T(rows: list, trajs: dict, grid: list, split: bool = False) -> list:
    """D6b: per cell (and variant when split) the number of Y=1 runs at T in grid, via outcomes.Y(traj, T)."""
    out = []
    for (t, N) in cell_keys(rows):
        for var in (("z", "noz") if split else (None,)):
            g = [r for r in rows if r["tier"] == t and r["N"] == N and (var is None or r["variant"] == var)]
            if not g:
                continue
            for T in grid:
                out.append({"cell": cell_label(t, N), "variant": var or "pooled", "T": T, "n": len(g),
                            "n_y1": sum(O.Y(trajs[r["run_name"]], T) for r in g)})
    return out


def eval_grid(trajs: dict, unit: int) -> list:
    """T grid of D6b: every eval step of any trajectory from the second eval on (= unit * i, i >= 2 on a regular grid)."""
    steps = sorted({s for tr in trajs.values() for s, _ in tr})
    return [s for s in steps if s >= 2 * unit]


def data_properties(cells: list, art: str = ART) -> list:
    """D9: static data facts per cell from p3_audit_<tier>_N<N>.json and p2_floors.json (source paths in the row)."""
    try:
        floors = json.load(open(os.path.join(art, "p2_floors.json")))["cells"]
    except (OSError, ValueError, KeyError):
        floors = {}
    out = []
    for (t, N) in cells:
        p = os.path.join(art, f"p3_audit_{t}_N{N}.json")
        row = {"cell": cell_label(t, N), "audit_source": os.path.relpath(p, REPO) if os.path.exists(p) else None,
               "floors_source": "lab/21_experiments/11_lab-experiments/EXP-027_artifacts/p2_floors.json"}
        if os.path.exists(p):
            a = json.load(open(p))
            ts = a.get("tier_stats", {})
            row.update(seq_len=a.get("info", {}).get("train", {}).get("seq_len"),
                       swap_count_hist=ts.get("swap_count_hist"),
                       distinct_final_permutations=ts.get("distinct_final_permutations"),
                       frac_rows_with_overlapping_swaps=ts.get("frac_rows_with_overlapping_swaps"))
        row["F_all"] = floors.get(f"{t}_N{N}", {}).get("F_all")
        out.append(row)
    return out


def a_run_check(rows: list) -> dict:
    """D11: A-tier rows and the two terms of the R0 pipeline condition (S.R0_MAX is the frozen constant)."""
    a = [r for r in rows if r["tier"] == "A"]
    gs = [r for r in rows if r["tier"] in ("G", "S")]
    return {"A_rows": [{k: r[k] for k in ("run_name", "Y", "max", "BSE", "tau", "final")} for r in a],
            "R0_max_threshold": S.R0_MAX,
            "GS_max_over_runs": max((r["max"] for r in gs), default=None),
            "GS_n_runs_with_max_ge_threshold": sum(1 for r in gs if r["max"] >= S.R0_MAX),
            "term_all_GS_max_below_threshold": all(r["max"] < S.R0_MAX for r in gs) if gs else None,
            "A_sum_Y": sum(r["Y"] for r in a),
            "term_no_A_run_has_Y1": (sum(r["Y"] for r in a) == 0) if a else None}


def selection_record(judge: dict, ext_gate) -> dict:
    """D10 / D6a: copied from the judge JSON (not recomputed)."""
    sel = judge.get("selection", {})
    return {"selection": {k: sel.get(k) for k in ("path", "trace", "mode", "N_S", "N_G", "qualifiers", "extension",
                                                 "notes", "late_flag", "misfit", "sub", "stat") if k in sel},
            "late_flag": sel.get("late_flag"),
            "pooled": [{k: c.get(k) for k in ("tier", "N", "n", "n_y1", "n_graded_pilot", "n_pretakeoff_y0",
                                              "max_pilot")} for c in judge.get("pooled", [])],
            "infra_excluded": judge.get("infra_excluded"),
            "rows_n_dirs": {r["run_name"]: r.get("n_dirs") for r in judge.get("rows", [])},
            "ext_gate": ext_gate, "judge_t_max_steps": judge.get("t_max_steps"),
            "judge_code_sha256": judge.get("code_sha256")}


# ---------------------------------------------------------------- cost (F5 / D12)
def cost_rows(rows: list, runs: dict, t_max: int, eval_step: int, model=None, timings=None) -> list:
    out = []
    for r in rows:
        n = r["run_name"]
        rt = runs[n].get("runtime_s")
        pred = None
        if model is not None:
            try:
                pred = model.g_run(r["tier"], r["N"], r["variant"], t_max, max(1, t_max // eval_step))
            except (ValueError, KeyError):
                pred = None
        c = {"run_name": n, "tier": r["tier"], "N": r["N"], "variant": r["variant"], "seed": r["seed"],
             "measured_gpu_h_wandb_runtime": None if rt is None else rt / 3600.0, "predicted_gpu_h": pred}
        t = (timings or {}).get(n)
        if t:
            c.update(train_seconds=t.get("train_seconds"), eval_seconds=t.get("eval_seconds"),
                     startup_seconds=t.get("startup_seconds"),
                     measured_gpu_h_extract=(t["train_seconds"] + (t["eval_seconds"] * (r["n_evals"] - 1))
                                             + t["startup_seconds"]) / 3600.0)
        out.append(c)
    return out


def get_timings(names: list, log_dir: str, steps: int, n_evals: int, wandb_root=None) -> tuple:
    """exp027_cost.extract_timings per run (so one bad log does not hide the others)."""
    import exp027_cost as C
    res, err = {}, {}
    for n in names:
        try:
            d = C.extract_timings([n], log_dir, steps, n_evals, wandb_root)
            if d["runs"]:
                res[n] = d["runs"][0]
            else:
                err[n] = "infeasible: " + json.dumps(d["infeasible"])
        except (LookupError, ValueError, OSError) as e:
            err[n] = str(e)
    return res, err


# ---------------------------------------------------------------- figures
def _tokens():
    p = os.path.join(REPO, "lab/26_figure-pipeline/11_lab-figure-pipeline/25_style/viz_style.py")
    tok = {"ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "blue": "#184f95",
           "red": "#a83232", "warn": "#9a7400"}
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("viz_style", p)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        tok.update(ink=m.INK_PRIMARY, ink2=m.INK_SECONDARY, muted=m.INK_MUTED, grid=m.GRID_HAIRLINE,
                   blue=m.DIVERGING_BLUE, red=m.DIVERGING_RED)
    except Exception:
        pass
    return tok


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": ["Liberation Sans", "DejaVu Sans"],
        "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11, "xtick.labelsize": FS_MIN + 1,
        "ytick.labelsize": FS_MIN + 1, "legend.fontsize": FS_MIN + 1, "axes.spines.right": False,
        "axes.spines.top": False, "axes.linewidth": 1.2, "legend.frameon": False, "pdf.fonttype": 42,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"})
    return plt


LINESTYLES = ("-", "--", ":", "-.")


def _style_of(r: dict, split: bool, seed_rank: dict, tok: dict) -> dict:
    ls = LINESTYLES[seed_rank[r["seed"]] % len(LINESTYLES)]
    if split:
        return dict(color=tok["blue"] if r["variant"] == "z" else tok["red"], ls=ls)
    return dict(color=tok["ink2"], ls=ls)


def _legend_handles(plt, split: bool, seeds: list, tok: dict, extra=()):
    from matplotlib.lines import Line2D
    h = []
    if split:
        h += [Line2D([], [], color=tok["blue"], lw=2, label="z (trm)"),
              Line2D([], [], color=tok["red"], lw=2, label="noz (trm_singlez)")]
    for i, s in enumerate(seeds):
        h.append(Line2D([], [], color=tok["ink2"], ls=LINESTYLES[i % len(LINESTYLES)], lw=1.5, label=f"seed {s}"))
    return h + list(extra)


def _grid(plt, cells, ncols=3, w=3.5, h=2.8):
    n = len(cells)
    nc = min(ncols, n)
    nr = math.ceil(n / nc)
    fig, axes = plt.subplots(nr, nc, figsize=(w * nc, h * nr + 0.6), squeeze=False)
    flat = list(axes.flat)
    for ax in flat[n:]:
        ax.set_axis_off()
    return fig, flat[:n], nr, nc


def _save(fig, out_dir: str, stem: str, written: list):
    fig.tight_layout(pad=0.8, rect=(0, 0.07, 1, 1))
    for ext in ("png", "pdf"):
        p = os.path.join(out_dir, f"{stem}.{ext}")
        fig.savefig(p, dpi=200 if ext == "png" else None)
        written.append(os.path.relpath(p, out_dir))


def _kfmt(plt):
    from matplotlib.ticker import FuncFormatter
    return FuncFormatter(lambda x, p: f"{x / 1000:g}k")


def fig_curves(plt, rows, trajs, t_max, out_dir, written, split, tok, logit=False):
    from matplotlib.lines import Line2D
    cells = cell_keys(rows)
    seeds = sorted({r["seed"] for r in rows})
    rank = {s: i for i, s in enumerate(seeds)}
    fig, axes, _nr, _nc = _grid(plt, cells)
    lg = lambda e: math.log(min(max(e, LOGIT_CLIP), 1 - LOGIT_CLIP) / (1 - min(max(e, LOGIT_CLIP), 1 - LOGIT_CLIP)))
    for ax, (t, N) in zip(axes, cells):
        g = [r for r in rows if r["tier"] == t and r["N"] == N]
        for r in g:
            tr = trajs[r["run_name"]]
            x = [s for s, _ in tr]
            y = [lg(e) if logit else e for _, e in tr]
            st = _style_of(r, split, rank, tok)
            ax.plot(x, y, lw=1.4, **st)
            if r["tau"] is not None:
                e0 = dict(tr)[r["tau"]]
                ax.plot([r["tau"]], [lg(e0) if logit else e0], "o", ms=4, color=st["color"], mec="white", mew=0.6)
        ax.axhline(lg(0.9) if logit else 0.9, color=tok["muted"], lw=1.0, ls=(0, (4, 3)))
        ax.axvline(t_max, color=tok["muted"], lw=1.0)
        if logit:
            tk = [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999]
            ax.set_yticks([lg(v) for v in tk])
            ax.set_yticklabels([f"{v:g}" for v in tk])
            ax.set_ylim(lg(LOGIT_CLIP) - 0.4, lg(1 - LOGIT_CLIP) + 0.4)
        else:
            ax.set_ylim(-0.03, 1.03)
        ax.set_title(f"{cell_label(t, N)}  (n={len(g)})")
        ax.xaxis.set_major_formatter(_kfmt(plt))
        ax.set_xlim(0, t_max * 1.02)
    for ax in axes:
        ax.set_xlabel("optimizer step")
    for i in range(0, len(axes), _nc):
        axes[i].set_ylabel("exact accuracy (logit)" if logit else "eval exact accuracy")
    extra = [Line2D([], [], color=tok["muted"], ls=(0, (4, 3)), label="0.90"),
             Line2D([], [], color=tok["muted"], label=f"T_p = {t_max}"),
             Line2D([], [], marker="o", ls="", color=tok["ink2"], label="tau")]
    fig.legend(handles=_legend_handles(plt, split, seeds, tok, extra), loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, 0.0))
    _save(fig, out_dir, "F2_eval_logit_exploratory" if logit else "F1_eval_trajectories", written)
    plt.close(fig)


def fig_train(plt, rows, runs, t_max, out_dir, written, split, tok):
    cells = cell_keys(rows)
    seeds = sorted({r["seed"] for r in rows})
    rank = {s: i for i, s in enumerate(seeds)}
    for key, stem, ylab, logy in (("loss", "F3a_train_loss", "train lm loss", True),
                                  ("acc", "F3b_train_exact_accuracy", "train exact accuracy", False)):
        if not any((runs[r["run_name"]].get("train") or {}).get(key) for r in rows):
            continue
        fig, axes, _nr, _nc = _grid(plt, cells)
        for ax, (t, N) in zip(axes, cells):
            for r in [r for r in rows if r["tier"] == t and r["N"] == N]:
                tr = runs[r["run_name"]].get("train") or {}
                if not tr.get(key):
                    continue
                pts = [(s, v) for s, v in zip(tr["steps"], tr[key]) if v is not None and (v > 0 or not logy)]
                ax.plot([p[0] for p in pts], [p[1] for p in pts], lw=1.2, **_style_of(r, split, rank, tok))
            if logy:
                ax.set_yscale("log")
            else:
                ax.set_ylim(-0.03, 1.03)
            ax.set_title(cell_label(t, N))
            ax.xaxis.set_major_formatter(_kfmt(plt))
            ax.set_xlim(0, t_max * 1.02)
            ax.set_xlabel("optimizer step")
        for i in range(0, len(axes), _nc):
            axes[i].set_ylabel(ylab)
        fig.legend(handles=_legend_handles(plt, split, seeds, tok), loc="lower center", ncol=6,
                   bbox_to_anchor=(0.5, 0.0))
        _save(fig, out_dir, stem, written)
        plt.close(fig)


def fig_yT(plt, ycurve, rows, t_max, out_dir, written, split, tok):
    from matplotlib.lines import Line2D
    cells = cell_keys(rows)
    fig, axes, _nr, _nc = _grid(plt, cells)
    for ax, (t, N) in zip(axes, cells):
        lab = cell_label(t, N)
        for var, col, ls in ((("z", tok["blue"], "-"), ("noz", tok["red"], "--")) if split
                             else (("pooled", tok["ink2"], "-"),)):
            pts = [p for p in ycurve if p["cell"] == lab and p["variant"] == var]
            if not pts:
                continue
            ax.step([p["T"] for p in pts], [p["n_y1"] / p["n"] for p in pts], where="post", color=col, ls=ls, lw=1.8)
            ax.text(0.5, 0.62 if var in ("pooled", "z") else 0.48, f"{var}: {pts[-1]['n_y1']}/{pts[-1]['n']} at T_p",
                    transform=ax.transAxes, fontsize=FS_MIN + 1, color=col, va="top", ha="center")
        ax.axvline(t_max, color=tok["muted"], lw=1.0)
        ax.set_ylim(-0.04, 1.04)
        ax.set_yticks([0, 0.5, 1])
        ax.set_title(lab)
        ax.xaxis.set_major_formatter(_kfmt(plt))
        ax.set_xlim(0, t_max * 1.02)
        ax.set_xlabel("T (optimizer steps)")
    for i in range(0, len(axes), _nc):
        axes[i].set_ylabel("fraction with Y(T) = 1")
    h = [Line2D([], [], color=tok["blue"], lw=2, label="z"), Line2D([], [], color=tok["red"], ls="--", lw=2, label="noz")] \
        if split else [Line2D([], [], color=tok["ink2"], lw=2, label="variants pooled")]
    fig.legend(handles=h, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    _save(fig, out_dir, "F4_Y_of_T", written)
    plt.close(fig)


def fig_cost(plt, crow, plan, out_dir, written, tok):
    have = [c for c in crow if c["measured_gpu_h_wandb_runtime"] is not None]
    if not have:
        return
    pred = [c for c in have if c["predicted_gpu_h"] is not None]
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.4, 3.6), gridspec_kw={"width_ratios": [1.2, 1]})
    if pred:
        cells = sorted({cell_label(c["tier"], c["N"]) for c in pred})
        cmap = plt.get_cmap("tab10")
        for i, lab in enumerate(cells):
            pts = [c for c in pred if cell_label(c["tier"], c["N"]) == lab]
            a.scatter([c["predicted_gpu_h"] for c in pts], [c["measured_gpu_h_wandb_runtime"] for c in pts],
                      s=34, color=cmap(i % 10), label=lab, edgecolor="white", linewidth=0.5)
        lim = max(max(c["predicted_gpu_h"], c["measured_gpu_h_wandb_runtime"]) for c in pred) * 1.08
        a.plot([0, lim], [0, lim], color=tok["muted"], lw=1.0)
        a.set_xlim(0, lim)
        a.set_ylim(0, lim)
        a.set_xlabel("predicted GPU-h (SMOKE2 cost model)")
        a.set_ylabel("measured GPU-h (wandb _runtime)")
        a.legend(ncol=2, loc="upper left")
        tot_p = math.fsum(c["predicted_gpu_h"] for c in pred)
        tot_m = math.fsum(c["measured_gpu_h_wandb_runtime"] for c in pred)
        names, vals = ["predicted\n(model)", "measured\n(runtime)"], [tot_p, tot_m]
        if plan:
            names.append("plan\n(17 runs)")
            vals.append(plan)
        b.bar(names, vals, color=[tok["muted"], tok["blue"], tok["grid"]][:len(vals)], edgecolor=tok["ink"])
        for i, v in enumerate(vals):
            b.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=FS_MIN + 1)
        b.set_ylabel(f"GPU-h, sum over {len(pred)} runs")
    else:
        a.bar(range(len(have)), [c["measured_gpu_h_wandb_runtime"] for c in have], color=tok["ink2"])
        a.set_xticks(range(len(have)))
        a.set_xticklabels([c["run_name"].replace("fig1_tf_", "")[-20:] for c in have], rotation=70, ha="right", fontsize=FS_MIN)
        a.set_ylabel("measured GPU-h (wandb _runtime)")
        b.set_axis_off()
        b.text(0.0, 0.5, "no prediction for these runs", transform=b.transAxes)
    fig.tight_layout(pad=0.8)
    for ext in ("png", "pdf"):
        p = os.path.join(out_dir, f"F5_cost.{ext}")
        fig.savefig(p, dpi=200 if ext == "png" else None)
        written.append(os.path.relpath(p, out_dir))
    plt.close(fig)


# ---------------------------------------------------------------- output writers
def _csv(path, rows, cols):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow(["" if r.get(c) is None else (json.dumps(r[c]) if isinstance(r[c], (list, dict)) else r[c])
                        for c in cols])


def _json(path, obj):
    with open(path, "w") as f:
        f.write(json.dumps(obj, indent=1, sort_keys=True) + "\n")


def sha256_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


RUN_COLS = ("run_name", "tier", "N", "variant", "seed", "wandb_id", "run_dir", "n_dirs", "n_evals", "tau",
            "tau_over_eval_step", "tau_frac_Tp", "Y", "BSE", "max", "final", "mean_last5", "stability",
            "max_minus_final", "n_evals_after_tau", "collapse", "graded", "pretakeoff", "class", "Y_sens",
            "wall_clock_seconds", "final_train_loss", "final_train_acc")
CELL_COLS = ("cell", "tier", "N", "n", "n_y1", "f", "state_Y", "classes", "n_collapse", "n_graded", "n_pretakeoff_y0",
             "BSE_hist", "max_hist", "tau_over_eval_step", "tau_frac_Tp", "n_censored_at_Tp", "final_median",
             "final_min", "final_max", "mean_last5_median", "mean_last5_min", "mean_last5_max", "max_median",
             "sensitivity", "n_changed_variants")


def build_report(coll: dict, t_max: int, eval_step: int, out_dir: str, split: bool, judge=None, ext_gate=None,
                 model=None, timings=None, timing_errors=None, plan=PLAN_PILOT17, grid_check=True,
                 make_figs=True) -> dict:
    runs = coll["runs"]
    names = sorted(runs)
    if not names:
        raise ValueError("no judged run")
    trajs = {n: runs[n]["traj"] for n in names}
    if grid_check:
        for n, tr in trajs.items():
            bad = [s for s, _ in tr if s % eval_step]
            if bad:
                raise ValueError(f"{n}: eval steps {bad[:3]} are not multiples of {eval_step}")
    rows = []
    for n in names:
        r = make_row(n, trajs[n], parse_meta(n), eval_step, t_max)
        tr = runs[n].get("train") or {}
        rt = runs[n].get("runtime_s")
        r.update(wandb_id=runs[n]["wandb_id"], run_dir=runs[n]["run_dir"], n_dirs=runs[n]["n_dirs"],
                 wall_clock_seconds=rt, final_train_loss=tr.get("final_train_loss"),
                 final_train_acc=tr.get("final_train_acc"))
        rows.append(r)
    os.makedirs(out_dir, exist_ok=True)
    cells = cell_table(rows)
    grid = eval_grid(trajs, eval_step)
    ycurve = y_of_T(rows, trajs, grid, split=False)
    crow = cost_rows(rows, runs, t_max, eval_step, model, timings)
    written = []

    def put(name, obj, csv_cols=None, csv_rows=None):
        _json(os.path.join(out_dir, name + ".json"), obj)
        written.append(name + ".json")
        if csv_cols:
            _csv(os.path.join(out_dir, name + ".csv"), csv_rows if csv_rows is not None else obj, csv_cols)
            written.append(name + ".csv")

    put("per_run", rows, RUN_COLS)
    put("per_cell", cells, CELL_COLS)
    put("y_of_T", ycurve, ("cell", "variant", "T", "n", "n_y1"))
    put("cost", {"rows": crow, "timing_errors": timing_errors or {},
                 "plan_pilot17_gpu_h": plan, "plan_with_smoke_gpu_h": PLAN_WITH_SMOKE,
                 "predicted_sum": math.fsum(c["predicted_gpu_h"] for c in crow if c["predicted_gpu_h"] is not None),
                 "measured_sum_wandb_runtime": math.fsum(c["measured_gpu_h_wandb_runtime"] for c in crow
                                                         if c["measured_gpu_h_wandb_runtime"] is not None)})
    if any(c["tier"] != "K" for c in cells):
        put("data_properties", data_properties([(c["tier"], c["N"]) for c in cells if c["tier"] != "K"]))
    put("a_run_check", a_run_check(rows))
    if judge is not None:
        put("selection_record", selection_record(judge, ext_gate))
    if split:
        put("per_variant", per_variant_table(rows), ("cell", "variant", "n", "n_y1", "BSE", "tau", "seeds"))
        put("y_of_T_per_variant", y_of_T(rows, trajs, grid, split=True), ("cell", "variant", "T", "n", "n_y1"))
    if make_figs:
        plt = _plt()
        tok = _tokens()
        fig_curves(plt, rows, trajs, t_max, out_dir, written, split, tok)
        fig_curves(plt, rows, trajs, t_max, out_dir, written, split, tok, logit=True)
        fig_train(plt, rows, runs, t_max, out_dir, written, split, tok)
        fig_yT(plt, y_of_T(rows, trajs, grid, split=split) if split else ycurve, rows, t_max, out_dir, written, split,
               tok)
        fig_cost(plt, crow, plan, out_dir, written, tok)
    return {"rows": rows, "cells": cells, "written": written}


# ---------------------------------------------------------------- CLI
def main(argv=None, injected=None, model=None) -> int:
    """injected / model: test hooks (trajectories instead of wandb; a CostModel instead of --smoke)."""
    ap = argparse.ArgumentParser(prog="pilot_report.py")
    ap.add_argument("--runs", nargs="*")
    ap.add_argument("--names-override", nargs="*", help="LABEL=WANDBID entries (test mode; e027 names refused)")
    ap.add_argument("--t-max-steps", required=True, type=int)
    ap.add_argument("--eval-step", type=int)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--wandb-root")
    ap.add_argument("--no-train", action="store_true")
    ap.add_argument("--judge-json")
    ap.add_argument("--ext-gate-json")
    ap.add_argument("--smoke", default=None, help="SMOKE2.json for per-run cost predictions (default: if present)")
    ap.add_argument("--no-prediction", action="store_true")
    ap.add_argument("--extract-timings", action="store_true")
    ap.add_argument("--log-dir", default=os.path.join(REPO, "logs", "queue"))
    ap.add_argument("--selection-witness")
    ap.add_argument("--witness-sha256")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    try:
        a = ap.parse_args(argv)
    except SystemExit as x:
        return 2 if x.code else 0
    if a.t_max_steps <= 0:
        print("FAIL: --t-max-steps must be positive", file=sys.stderr)
        return 2
    split = False
    if a.selection_witness or a.witness_sha256:
        if not (a.selection_witness and a.witness_sha256):
            print("FAIL: variant-split views need both --selection-witness and --witness-sha256", file=sys.stderr)
            return 2
        try:
            got = sha256_file(a.selection_witness)
        except OSError as e:
            print(f"FAIL: {e}", file=sys.stderr)
            return 2
        if got != a.witness_sha256:
            print(f"REFUSED: sha256 of {a.selection_witness} is {got}, expected {a.witness_sha256}", file=sys.stderr)
            return 3
        split = True
    id_of = None
    if a.names_override:
        id_of = {}
        for ent in a.names_override:
            if "=" not in ent:
                print("FAIL: --names-override entries are LABEL=WANDBID", file=sys.stderr)
                return 2
            lab, wid = ent.split("=", 1)
            if "e027" in lab.lower() or "e027" in wid.lower():
                print("REFUSED: --names-override is for non-EXP-027 runs only", file=sys.stderr)
                return 3
            id_of[lab] = wid
        names = list(id_of)
    else:
        names = list(a.runs) if a.runs else default_names()
    if len(set(names)) != len(names):
        print("FAIL: duplicate run names", file=sys.stderr)
        return 2
    try:
        for n in names:
            parse_meta(n)
        if a.eval_step:
            eval_step = a.eval_step
        elif not a.names_override:
            eval_step = S.EVAL_STEP
        else:
            eval_step = None
        if not a.names_override and a.t_max_steps % S.EVAL_STEP:
            print(f"FAIL: --t-max-steps must be a multiple of {S.EVAL_STEP}", file=sys.stderr)
            return 2
        coll = collect(names, a.t_max_steps, a.wandb_root, injected, not a.no_train, id_of)
        if coll["waiting"]:
            print("NOT JUDGEABLE:\n  " + "\n  ".join(w["run_name"] + ": " + w["reason"] for w in coll["waiting"]),
                  file=sys.stderr)
            if a.strict:
                return 4
        if coll["excluded"]:
            print("INFRA-EXCLUDED: " + ", ".join(e["run_name"] for e in coll["excluded"]), file=sys.stderr)
        if eval_step is None:
            eval_step = infer_eval_step([r["traj"] for r in coll["runs"].values()])
        judge = json.load(open(a.judge_json)) if a.judge_json else None
        ext_gate = json.load(open(a.ext_gate_json)) if a.ext_gate_json else None
        if model is None and not a.no_prediction and not a.names_override:
            sm = a.smoke or (DEFAULT_SMOKE if os.path.exists(DEFAULT_SMOKE) else None)
            if sm:
                import exp027_cost as C
                model = C.CostModel(C.load_smoke(sm))
        timings = terr = None
        if a.extract_timings:
            timings, terr = get_timings(sorted(coll["runs"]), a.log_dir, a.t_max_steps,
                                        max(1, a.t_max_steps // eval_step), a.wandb_root)
        rep = build_report(coll, a.t_max_steps, eval_step, a.out_dir, split, judge, ext_gate, model, timings, terr,
                           grid_check=not a.names_override, make_figs=not a.no_figures)
        meta = {"t_max_steps": a.t_max_steps, "eval_step": eval_step, "n_planned": len(names),
                "n_judged": len(rep["rows"]), "waiting": coll["waiting"], "infra_excluded": coll["excluded"],
                "variant_split": split,
                "selection_witness": None if not split else {"path": a.selection_witness,
                                                              "sha256": a.witness_sha256},
                "class_precedence": "graded > pretakeoff > never-learned among Y=0",
                "code_sha256": {n: sha256_file(os.path.join(HERE, n))
                                for n in J.WITNESS_FILES + ("pilot_report.py",)},
                "files": sorted(rep["written"])}
        _json(os.path.join(a.out_dir, "meta.json"), meta)
        print(json.dumps({"n_judged": meta["n_judged"], "n_planned": meta["n_planned"], "variant_split": split,
                          "out_dir": a.out_dir}))
    except (LookupError, ValueError, OSError) as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
