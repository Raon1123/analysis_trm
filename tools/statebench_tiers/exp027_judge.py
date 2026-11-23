#!/usr/bin/env python3
"""EXP-027 judge runner: run names -> trajectories -> per-run table -> selection (pilot) or rule code (conf).

Status: DRAFT, author_model claude-sonnet-5-5.  Not frozen.  No GPU, read-only on wandb/.
All statistics float64; rng seeds are the ones fixed in exp027_rules_v3.py; the output JSON has no wall clock.

Usage:
  exp027_judge.py --mode pilot --t-max-steps T --runs NAME [NAME ...] --out PILOT.json
  exp027_judge.py --mode conf  --t-max-steps T --runs NAME [NAME ...] --out CONF.json --witness WITNESS.json
                  [--outcome Y|BSE]      (optional; must equal the witness selection mode)
  (--runs-file F reads one name per line instead.)

Run names must be e027_<stage>_<tier>N<N>_<variant>_s<seed>; pilot mode takes stages pilot and ext, conf mode
takes stage conf with exactly one N per tier (two tiers -> 4-cell code, G only -> 2-cell G code).

Incomplete-run rule (design-log 9.3).  T = --t-max-steps must lie on the eval grid (a multiple of
select.EVAL_STEP = 1953 (= select.block_steps(), batch select.GLOBAL_BATCH = 1024); select.steps_for_epochs gives T for an epoch count).  For each run name every wandb dir
carrying that name is a candidate; a candidate is complete when its last eval step is >= T.  The latest complete
candidate is judged (its trajectory truncated at step <= T).  If none is complete:
  exactly one candidate -> the run must be rerun once with the same seed (exit 4, nothing is judged), however
      far the first attempt got;
  two or more candidates and none complete -> the run is excluded and listed in the output ("infra_excluded"
      with its dir count and last eval steps).
A rerun keeps the run name; the older dir stays on disk and is ignored once a complete one exists.  A non-finite
eval value in any candidate fails the load (exit 1).  Conf mode: a cell of the witnessed design (4 cells for theta,
Gz/Gn for G-only) left with no judged run after exclusion fails (exit 1).  Pilot mode: a (tier, N) cell left with no
judged run is simply absent from the pooled table and the selection runs on the remaining cells.

Freeze witness (conf mode refuses without it): JSON
  {"sha256": {<each of WITNESS_FILES>: hex}, "selection": {"mode": "Y"|"BSE", "N_S": int|null, "N_G": int,
   "stat": "theta"|"G-only", "n": int (optional), "t_max_epochs": int (optional)}}.
Every recorded hash must equal the current file's sha256.  The selection block is the pilot result
(SelectionResult fields; see make_witness).  The conf runs must have the selected N per tier, the outcome is the
selection mode (a conflicting --outcome is refused), the tier set must match stat, and when n / t_max_epochs are
present the planned runs per cell must equal n and T must equal (t_max_epochs // 20) * EVAL_STEP.  The R6 path never confirms
(selection.stat other than theta / G-only is refused).

exit: 0 ok, 1 trajectories missing / data invalid, 2 usage, 3 witness refused or conf runs disagree with it,
      4 not judgeable yet (a run with a single incomplete dir needs a same-seed rerun).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import outcomes as O  # noqa: E402
import exp027_rules_v3 as R  # noqa: E402
import exp027_select as S  # noqa: E402

WITNESS_FILES = ("exp027_rules_v3.py", "outcomes.py", "exp027_select.py", "exp027_judge.py", "exp027_cost.py")
RUN_NAME = re.compile(r"^e027_(smoke|smoke2|pilot|ext|conf)_([ASG])N(\d+)_(z|noz)_s(\d+)$")
CELL_OF = {("S", "z"): "Sz", ("S", "noz"): "Sn", ("G", "z"): "Gz", ("G", "noz"): "Gn"}
# threshold sensitivity variants: (label, learn threshold, consecutive evals required)
SENSITIVITY = (("learn0.5", 0.5, 2), ("learn0.95", 0.95, 2), ("consec1", 0.9, 1), ("consec3", 0.9, 3))


class WitnessError(Exception):
    pass


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def check_witness(path: str, root: str = HERE) -> dict:
    """Return the current sha256 values if they match the witness; raise WitnessError otherwise."""
    try:
        w = json.load(open(path))
    except (OSError, ValueError) as e:
        raise WitnessError(f"cannot read witness {path!r}: {e}")
    rec = w.get("sha256") if isinstance(w, dict) else None
    if not isinstance(rec, dict):
        raise WitnessError("witness has no 'sha256' mapping")
    cur = {}
    for name in WITNESS_FILES:
        if name not in rec:
            raise WitnessError(f"witness lacks a hash for {name}")
        p = os.path.join(root, name)
        if not os.path.exists(p):
            raise WitnessError(f"{p} does not exist")
        cur[name] = sha256_file(p)
        if cur[name] != rec[name]:
            raise WitnessError(f"{name}: current sha256 {cur[name]} != witness {rec[name]}")
    return cur


SEL_KEYS = ("mode", "N_S", "N_G", "stat")


def make_witness(selection: dict, n=None, t_max_epochs=None) -> dict:
    """Build the conf freeze witness: current code hashes + the selection result (pilot output 'selection')."""
    sel = {k: selection[k] for k in SEL_KEYS}
    if n is not None:
        sel["n"] = int(n)
    if t_max_epochs is not None:
        sel["t_max_epochs"] = int(t_max_epochs)
    return {"sha256": {n_: sha256_file(os.path.join(HERE, n_)) for n_ in WITNESS_FILES}, "selection": sel}


def witness_selection(path: str) -> dict:
    """The selection block of a conf witness; WitnessError if absent or malformed."""
    try:
        sel = json.load(open(path)).get("selection")
    except (OSError, ValueError, AttributeError) as e:
        raise WitnessError(f"cannot read witness {path!r}: {e}")
    if not isinstance(sel, dict) or any(k not in sel for k in SEL_KEYS):
        raise WitnessError(f"conf witness needs a 'selection' block with {SEL_KEYS}")
    if sel["mode"] not in ("Y", "BSE"):
        raise WitnessError(f"witness selection mode {sel['mode']!r} is not Y/BSE")
    if sel["stat"] not in ("theta", "G-only"):
        raise WitnessError(f"witness selection stat {sel['stat']!r} is not judged here (theta / G-only only)")
    if sel["N_G"] is None or (sel["stat"] == "theta") != (sel["N_S"] is not None):
        raise WitnessError("witness selection N_S / N_G disagree with its stat")
    return sel


def check_conf_plan(names: list, sel: dict, t_max_steps: int) -> None:
    """Raise WitnessError unless the planned conf runs (all names, before any exclusion) match the witness."""
    metas = [parse_name(n) for n in names]
    want = {"G": sel["N_G"]}
    if sel["stat"] == "theta":
        want["S"] = sel["N_S"]
    for n, m in zip(names, metas):
        if m["stage"] != "conf":
            raise WitnessError(f"{n}: conf mode takes stage conf only")
        if m["tier"] not in want:
            raise WitnessError(f"{n}: tier {m['tier']} is not part of the selected design ({sorted(want)})")
        if m["N"] != want[m["tier"]]:
            raise WitnessError(f"{n}: N={m['N']} but the witness selection has N_{m['tier']}={want[m['tier']]}")
    cnt: dict = {}
    for m in metas:
        k = CELL_OF[(m["tier"], m["variant"])]
        cnt[k] = cnt.get(k, 0) + 1
    need = ("Sz", "Sn", "Gz", "Gn") if sel["stat"] == "theta" else ("Gz", "Gn")
    if any(k not in cnt for k in need):
        raise WitnessError(f"conf runs lack a cell of {need}")
    if "n" in sel and any(cnt[k] != sel["n"] for k in need):
        raise WitnessError(f"planned runs per cell {cnt} differ from the witness n={sel['n']}")
    if "t_max_epochs" in sel and t_max_steps != S.steps_for_epochs(int(sel["t_max_epochs"])):
        raise WitnessError(f"--t-max-steps {t_max_steps} != steps_for_epochs({sel['t_max_epochs']}) = "
                           f"{S.steps_for_epochs(int(sel['t_max_epochs']))}")


# ---------------------------------------------------------------- run selection among same-name dirs
class NotJudgeable(Exception):
    """A run needs a same-seed rerun (exit 4)."""


def pick_run(cands: list, t_max: int) -> dict:
    """cands: trajectories of every dir of one run name, oldest first.  Returns
    {'status': 'complete'|'infra', 'traj': truncated trajectory or None, 'n_dirs', 'last_steps'};
    raises NotJudgeable for a run whose only dir is incomplete (same-seed rerun needed)."""
    if not cands:
        raise ValueError("a run name needs at least one dir")
    last = [tr[-1][0] if tr else 0 for tr in cands]
    done = [i for i, ls in enumerate(last) if ls >= t_max]
    info = {"n_dirs": len(cands), "last_steps": last}
    if done:
        tr = [(s_, v) for s_, v in cands[done[-1]] if s_ <= t_max]
        return {"status": "complete", "traj": tr, **info}
    if len(cands) == 1:
        raise NotJudgeable(f"last eval step {last[0]} < T_max={t_max} and no other dir: rerun once with the same "
                           f"seed")
    return {"status": "infra", "traj": None, **info}


# ---------------------------------------------------------------- per-run table
def parse_name(name: str) -> dict:
    m = RUN_NAME.match(name)
    if not m:
        raise ValueError(f"run name {name!r} does not match e027_<stage>_<tier>N<N>_<variant>_s<seed>")
    stage, tier, N, var, seed = m.groups()
    return {"stage": stage, "tier": tier, "N": int(N), "variant": var, "seed": int(seed)}


def y_consec(traj, learn: float = 0.90, m: int = 2) -> int:
    """1 iff some m consecutive evals are all >= learn (m=2, learn=0.9 equals outcomes.Y).
    Uses the outcomes float32 tolerance (a logged 0.950 is 0.949999988 in float32)."""
    e = [x for _, x in traj]
    return int(any(all(O.ge(x, learn) for x in e[i:i + m]) for i in range(len(e) - m + 1)))


def build_rows(trajs: dict) -> list:
    """{run_name: trajectory} -> sorted list of per-run dicts (selection-compatible keys plus extras)."""
    rows = []
    for name in sorted(trajs):
        meta = parse_name(name)
        tr = trajs[name]
        O.check_finite(tr, name)
        s = O.summarize(tr)
        if s["BSE"] is None:
            raise ValueError(f"{name}: fewer than 2 evals, BSE undefined")
        flags = [f for f, k in (("graded", "graded"), ("collapse", "collapse"), ("pretakeoff", "pre_takeoff"))
                 if s[k]]
        rows.append({"run_name": name, **meta, "Y": s["Y"], "BSE": s["BSE"], "tau": s["tau"], "max": s["max"],
                     "final": s["final"], "mean_last5": s["mean_last5"], "stability": s["stability"],
                     "n_evals": s["n_evals"], "flags": flags,
                     "Y_sens": {lab: y_consec(tr, lr, mm) for lab, lr, mm in SENSITIVITY}})
    return rows


def _mean(x):
    return math.fsum(x) / len(x)


# ---------------------------------------------------------------- pilot mode
def _sel_rows(rows: list) -> list:
    return [{k: r[k] for k in ("tier", "N", "variant", "seed", "Y", "BSE", "tau", "flags", "max")}
            | {"stage": r["stage"]} for r in rows]


def states_at_T(trajs: dict, epochs_menu=(620, 420)) -> dict:
    """Pooled (variant-dropped) cell states when every pilot trajectory is cut at T = (e // 20) * EVAL_STEP steps,
    for the shorter T_max menu values of the n-gate (design s6).  Both modes are reported; the caller decides
    which one applies (the selection mode).  Cutting uses outcomes semantics: both evals within T."""
    out = {}
    for e in epochs_menu:
        T = S.steps_for_epochs(e)
        cut = {n: [(s_, v) for s_, v in tr if s_ <= T] for n, tr in trajs.items()}
        pooled = S.pool_rows(_sel_rows(build_rows(cut)))
        out[str(e)] = {"T_steps": T, "cells": [
            {"tier": c.tier, "N": c.N, "n": c.n, "n_y1": c.n_y1, "mean_BSE": S._mean_bse(c),
             "state_Y": S._state(c, "Y"), "state_BSE": S._state(c, "BSE")}
            for c in pooled.cells if c.tier in ("G", "S")]}
    return out


def judge_pilot(rows: list, trajs: dict = None) -> dict:
    """Selection on the variant-pooled table (R6 never confirms; there is no judgment-time gate flag)."""
    bad = [r["run_name"] for r in rows if r["stage"] not in ("pilot", "ext")]
    if bad:
        raise ValueError(f"pilot mode takes stages pilot/ext only: {bad}")
    pooled = S.pool_rows(_sel_rows(rows))
    res = S.select(pooled)
    out = {"mode": "pilot", "rows": rows,
           "pooled": [dataclasses.asdict(c) | {"tau": [None if math.isinf(t) else t for t in c.tau],
                                               "max_pilot": None if math.isinf(c.max_pilot) else c.max_pilot}
                      for c in pooled.cells],
           "selection": dataclasses.asdict(res)}
    if trajs is not None:
        out["states_at_shorter_T"] = states_at_T({r["run_name"]: trajs[r["run_name"]] for r in rows})
    return out


# ---------------------------------------------------------------- conf mode
def conf_cells(rows: list) -> dict:
    """{'Sz': [run rows], ...}.  Exactly one N per tier; tiers G or S+G; stage conf only."""
    bad = [r["run_name"] for r in rows if r["stage"] != "conf"]
    if bad:
        raise ValueError(f"conf mode takes stage conf only: {bad}")
    ns: dict = {}
    for r in rows:
        if r["tier"] not in ("S", "G"):
            raise ValueError(f"{r['run_name']}: tier {r['tier']} is not a confirmatory tier")
        ns.setdefault(r["tier"], set()).add(r["N"])
    for t, v in ns.items():
        if len(v) != 1:
            raise ValueError(f"tier {t} has several N in conf runs: {sorted(v)}")
    if "G" not in ns:
        raise ValueError("conf runs need a G tier")
    if "S" in ns and "G" not in ns:
        raise ValueError("S-only conf runs are not a defined design")
    cells: dict = {}
    for r in rows:
        cells.setdefault(CELL_OF[(r["tier"], r["variant"])], []).append(r)
    need = ("Sz", "Sn", "Gz", "Gn") if "S" in ns else ("Gz", "Gn")
    for k in need:
        if not cells.get(k):
            raise ValueError(f"cell {k} has no runs")
    return {k: cells[k] for k in need}


def _counts(cells: dict, key="Y"):
    return {k: (sum(r[key] for r in v), len(v)) for k, v in cells.items()}


def _did(means: dict):
    g = means["Gz"] - means["Gn"]
    if "Sz" in means:
        return {"theta": g - (means["Sz"] - means["Sn"]), "G_gain": g}
    return {"theta": None, "G_gain": g}


def judge_conf(rows: list, outcome: str = "Y") -> dict:
    if outcome not in ("Y", "BSE"):
        raise ValueError("outcome must be Y or BSE")
    cells = conf_cells(rows)
    four = "Sz" in cells
    if outcome == "Y":
        counts = _counts(cells)
        res = R.judge_theta(counts) if four else R.judge_g_only(counts)
    else:
        runs = {k: [r["BSE"] for r in v] for k, v in cells.items()}
        res = R.judge_bse(runs) if four else R.judge_bse_g_only(runs)

    def cell_mean(key):
        return {k: _mean([r[key] for r in v]) for k, v in cells.items()}

    sens = {}
    for lab, _lr, _m in SENSITIVITY:
        cs = {k: (sum(r["Y_sens"][lab] for r in v), len(v)) for k, v in cells.items()}
        rr = R.judge_theta(cs) if four else R.judge_g_only(cs)
        sens[lab] = {"counts": {k: list(v) for k, v in cs.items()}, "code": rr.code, "probs": rr.probs}
    desc = {
        "BSE_cell_means": cell_mean("BSE"), "BSE_DiD": _did(cell_mean("BSE")),
        "final_exact_cell_means": cell_mean("final"), "final_exact_DiD": _did(cell_mean("final")),
        "Y_rate_cell": {k: sum(r["Y"] for r in v) / len(v) for k, v in cells.items()},
        "collapse_counts": {k: sum(1 for r in v if "collapse" in r["flags"]) for k, v in cells.items()},
        "collapse_total": sum(1 for v in cells.values() for r in v if "collapse" in r["flags"]),
        "threshold_sensitivity": sens,
    }
    return {"mode": "conf", "outcome": outcome, "design": "4cell" if four else "G_only",
            "N": {t: sorted({r["N"] for r in rows if r["tier"] == t})[0] for t in ("S", "G")
                  if any(r["tier"] == t for r in rows)},
            "code": res.code, "probs": res.probs, "rule_inputs": res.inputs, "rule_constants": res.constants,
            "descriptive": desc, "rows": rows}


# ---------------------------------------------------------------- CLI
def _as_candidates(v):
    """Injected test value -> list of trajectories (a bare trajectory is one dir)."""
    if v and isinstance(v[0], list) and (not v[0] or isinstance(v[0][0], (list, tuple))):
        return [list(t) for t in v]
    return [list(v)]


def main(argv=None, trajs=None) -> int:
    """trajs: optional {run_name: trajectory | [trajectory per dir, oldest first]} injected instead of wandb (tests)."""
    ap = argparse.ArgumentParser(prog="exp027_judge.py")
    ap.add_argument("--mode", required=True, choices=("pilot", "conf"))
    ap.add_argument("--t-max-steps", required=True, type=int)
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--runs-file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--witness")
    ap.add_argument("--outcome", default=None, choices=("Y", "BSE"))
    ap.add_argument("--wandb-root")
    try:
        a = ap.parse_args(argv)
    except SystemExit as x:
        return 2 if x.code else 0
    names = list(a.runs)
    if a.runs_file:
        names += [ln.strip() for ln in open(a.runs_file) if ln.strip()]
    if not names or len(set(names)) != len(names):
        print("FAIL: need a non-empty list of distinct run names", file=sys.stderr)
        return 2
    if a.t_max_steps <= 0 or a.t_max_steps % S.EVAL_STEP:
        print(f"FAIL: --t-max-steps must be a positive multiple of {S.EVAL_STEP}", file=sys.stderr)
        return 2
    if a.mode == "pilot" and a.outcome:
        print("FAIL: --outcome is a conf-mode option and comes from the witness", file=sys.stderr)
        return 2
    witness_sha = sel = None
    if a.mode == "conf" or a.witness:
        if not a.witness:
            print("REFUSED: conf mode needs --witness (freeze witness file)", file=sys.stderr)
            return 3
        try:
            witness_sha = check_witness(a.witness)
            if a.mode == "conf":
                sel = witness_selection(a.witness)
                if a.outcome and a.outcome != sel["mode"]:
                    raise WitnessError(f"--outcome {a.outcome} conflicts with the witness selection mode {sel['mode']}")
                check_conf_plan(names, sel, a.t_max_steps)
        except WitnessError as e:
            print(f"REFUSED: {e}", file=sys.stderr)
            return 3
        except ValueError as e:        # unparsable run name
            print(f"FAIL: {e}", file=sys.stderr)
            return 1
    try:
        if trajs is not None:
            cand = {n: _as_candidates(trajs[n]) for n in names if n in trajs}
        else:
            cand = {n: [tr for _, tr in v] for n, v in O.load_candidates(names, root=a.wandb_root).items()}
        missing = [n for n in names if n not in cand]
        if missing:
            raise LookupError(f"no trajectory for {missing}")
        for n in names:                                  # a NaN in any dir fails the load
            for k, tr in enumerate(cand[n]):
                O.check_finite(tr, f"{n} (dir {k})")
        picked, infra, wait = {}, [], []
        for n in names:
            try:
                picked[n] = pick_run(cand[n], a.t_max_steps)
            except NotJudgeable as e:
                wait.append(f"{n}: {e}")
        if wait:
            print("NOT JUDGEABLE:\n  " + "\n  ".join(wait), file=sys.stderr)
            return 4
        infra = [{"run_name": n, "n_dirs": p["n_dirs"], "last_steps": p["last_steps"]}
                 for n, p in picked.items() if p["status"] == "infra"]
        used = {n: p["traj"] for n, p in picked.items() if p["status"] == "complete"}
        if not used:
            raise ValueError("every run is an infrastructure failure")
        rows = build_rows(used)
        for r in rows:
            r["n_dirs"] = picked[r["run_name"]]["n_dirs"]
        if a.mode == "pilot":
            out = judge_pilot(rows, used)
        else:
            # every cell of the witnessed design must keep >= 1 judged run after infrastructure exclusion;
            # otherwise a theta design whose S runs were all excluded would fall through to the G-only code
            need = ("Sz", "Sn", "Gz", "Gn") if sel["stat"] == "theta" else ("Gz", "Gn")
            have = {CELL_OF[(r["tier"], r["variant"])] for r in rows}
            lost = [k for k in need if k not in have]
            if lost:
                raise ValueError(f"cells {lost} have no judged run after infrastructure exclusion; the witnessed "
                                 f"design (stat={sel['stat']}) cannot be judged")
            out = judge_conf(rows, sel["mode"])
            out["witness_selection"] = sel
            planned: dict = {}
            for n in names:
                m = parse_name(n)
                k = CELL_OF[(m["tier"], m["variant"])]
                planned[k] = planned.get(k, 0) + 1
            out["planned_per_cell"] = planned
    except (LookupError, ValueError) as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    out["t_max_steps"] = a.t_max_steps
    out["n_planned"] = len(names)
    out["n_judged"] = len(rows)
    out["infra_excluded"] = infra
    out["witness_sha256"] = witness_sha
    out["code_sha256"] = {n: sha256_file(os.path.join(HERE, n)) for n in WITNESS_FILES}
    with open(a.out, "w") as f:
        f.write(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"mode": a.mode, "code": out.get("code"), "path": out.get("selection", {}).get("path"),
                      "n_planned": len(names), "n_judged": len(rows), "infra_excluded": len(infra)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
