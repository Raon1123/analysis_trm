#!/usr/bin/env python3
"""EXP-027 cost calculator: smoke timings -> c_step / c_eval / t0 -> G_run, pilot plan (design-log 9.2), C_set, n.

Status: DRAFT, author_model claude-sonnet-5-5.  Not frozen.  No GPU.  All arithmetic float64, no randomness,
no wall clock in outputs.

Smoke JSON (written by the director, or produced by the `extract` subcommand):
  {"steps": 3906, "n_evals": 2,
   "runs": [{"run_name": str, "tier": "G"|"S"|"A", "N": int, "variant": "z"|"noz",
             "train_seconds": float,      # training time for `steps` optimizer steps, evals excluded
             "eval_seconds": float,       # seconds per ONE eval (includes the checkpoint save inside the eval block)
             "startup_seconds": float}],  # process start until the first optimizer step plus teardown
   "infeasible": [{"run_name": str, "tier": ..., "N": int, "variant": ..., "reason": str}]}   # optional
  "infeasible" lists second-smoke runs that ran out of GPU memory; amendment 1.  They are not timing points.
  A crash without an out-of-memory message is not infeasible: `extract` fails on it (v3 rule unchanged).

Feasibility (amendment 1).  GPU memory grows with the token count 33 + 2N, so over all measured runs (both variants
pooled: noz carries a single z and is not heavier than z, and smoke2 measures noz only at N 16 and 32):
  (tier, N) is infeasible if some infeasible run has tokens <= tokens(N), or if tokens(N) exceeds the largest feasible
  measured token count (unmeasured; no extrapolation upward).  Interpolation of c_step / c_eval uses feasible points only.
  An infeasible extension option is reported as None in `ext_options`; ext_gate closes it as CLOSE_R7_EXT_INFEASIBLE.
  If a pilot cell (N 16 or 32, any tier, A included) is infeasible, plan() returns branch NO_PILOT_INFEASIBLE.

Model (design 03_DESIGN_DIALECTIC section 7):
  c_step(X, N) = train_seconds / steps / 3600         [GPU-h per optimizer step]
  c_eval(X, N) = eval_seconds / 3600                  [GPU-h per eval]
  t0           = per-variant mean of startup_seconds / 3600
  G_run(X, N)  = T_max * c_step + n_ev * c_eval + t0   [GPU-h]
c_step and c_eval are piecewise-linear in tokens 33 + 2N, separately for z and noz; extrapolated linearly outside
the measured range; a single measured token count is scaled proportionally to tokens.  Tier-specific points are
used for a (tier, variant) pair only when it has >= 2 distinct measured N; otherwise all tiers share the variant's
points (the smoke design measures S only at N=64 / z).

Usage:
  exp027_cost.py plan    --smoke SMOKE.json [--budget 130] [--ext-cap 60] [--out PLAN.json]   (prints the arithmetic)
  exp027_cost.py conf    --smoke SMOKE.json --cells S:16,G:16 --epochs EP [--out F.json]   (EP = T_max epochs)
  exp027_cost.py extract --runs RUN [RUN ...] [--log-dir logs/queue] [--out SMOKE.json] [--steps 3906]
                         [--n-evals 2] [--wandb-root DIR]
  exit: 0 ok, 1 missing/invalid data (fails loudly), 2 usage.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import exp027_select as S  # noqa: E402

# ---------------------------------------------------------------- constants
# Step counts come from exp027_select.steps_for_epochs: one eval block is S.EVAL_STEP = 1953 executed optimizer
# steps at global batch 1024 (820 / 620 / 420 epochs = 80073 / 60543 / 41013 steps, smoke 40 epochs = 3906).
T_P_MENU_EPOCHS = S.T_P_MENU_EPOCHS
PILOT_BUDGET = S.PILOT_BUDGET    # design-log 9.2: smoke + 17 runs
EXT_BUDGET = S.EXT_BUDGET        # design-log 9.2: extension (4 runs) has its own cap
B_CONF, N_CAP = 600.0, 30
N_GPUS = 4                       # GPUs 4..7
SMOKE_EPOCHS = 40
SMOKE_STEPS = S.steps_for_epochs(SMOKE_EPOCHS)       # 3906
SMOKE_N_EVALS = S.n_evals_for_epochs(SMOKE_EPOCHS)   # 2
PILOT_NS, PILOT_SEEDS, VARIANTS = S.PILOT_NS, S.PILOT_SEEDS, S.EXT_VARIANTS
EXT_OPTIONS = S.EXT_OPTIONS      # extension N by tier; 4 runs each (2 seeds x 2 variants)
EXT_RUNS = 4
steps_for = S.steps_for_epochs
n_evals_for = S.n_evals_for_epochs


def tokens(N: int) -> int:
    return 33 + 2 * N


# ---------------------------------------------------------------- smoke fit
def load_smoke(path_or_obj) -> dict:
    obj = path_or_obj if isinstance(path_or_obj, dict) else json.load(open(path_or_obj))
    steps = int(obj.get("steps", SMOKE_STEPS))
    nev = int(obj.get("n_evals", SMOKE_N_EVALS))
    runs = obj.get("runs")
    infeasible = [dict(r) for r in obj.get("infeasible", [])]
    for r in infeasible:
        for k in ("run_name", "tier", "N", "variant"):
            if k not in r:
                raise ValueError(f"infeasible record missing key {k!r}: {r}")
    if not runs:
        raise ValueError("smoke file has no feasible runs")
    for r in runs:
        for k in ("run_name", "tier", "N", "variant", "train_seconds", "eval_seconds", "startup_seconds"):
            if k not in r:
                raise ValueError(f"smoke run missing key {k!r}: {r}")
        if r["variant"] not in VARIANTS:
            raise ValueError(f"bad variant {r['variant']!r}")
        for k in ("train_seconds", "eval_seconds", "startup_seconds"):
            if not (isinstance(r[k], (int, float)) and r[k] >= 0 and math.isfinite(r[k])):
                raise ValueError(f"{r['run_name']}: {k} must be a finite number >= 0")
        if not r["train_seconds"] > 0:
            raise ValueError(f"{r['run_name']}: train_seconds must be > 0")
    if steps <= 0 or nev < 0:
        raise ValueError("bad steps / n_evals")
    return {"steps": steps, "n_evals": nev, "runs": [dict(r) for r in runs], "infeasible": infeasible}


def _interp(points, x: float) -> float:
    """points: [(tokens, value)]; per-token mean, piecewise linear, linear extrapolation, single point proportional."""
    by: dict = {}
    for px, py in points:
        by.setdefault(px, []).append(py)
    pts = sorted((px, math.fsum(v) / len(v)) for px, v in by.items())
    if not pts:
        raise ValueError("no smoke points")
    if len(pts) == 1:
        return pts[0][1] * x / pts[0][0]
    if x <= pts[0][0]:
        (x0, y0), (x1, y1) = pts[0], pts[1]
    elif x >= pts[-1][0]:
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
    else:
        i = max(j for j in range(len(pts) - 1) if pts[j][0] <= x)
        (x0, y0), (x1, y1) = pts[i], pts[i + 1]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


class CostModel:
    """Interpolating cost model built from a loaded smoke dict."""

    def __init__(self, smoke: dict):
        self.smoke = smoke
        self.steps = smoke["steps"]
        self.runs = smoke["runs"]
        self.infeasible = smoke.get("infeasible", [])
        for v in VARIANTS:
            if not any(r["variant"] == v for r in self.runs + self.infeasible):
                raise ValueError(f"no smoke run for variant {v!r}")
        self.smoke_cost = math.fsum(
            (r["train_seconds"] + smoke["n_evals"] * r["eval_seconds"] + r["startup_seconds"]) / 3600.0
            for r in self.runs)

    def max_feasible_tokens(self) -> int:
        return max(tokens(r["N"]) for r in self.runs)

    def infeasibility(self, N: int):
        """None if N fits, else the reason (pooled over tiers and variants; see module docstring)."""
        tk = tokens(N)
        bad = [r for r in self.infeasible if tokens(r["N"]) <= tk]
        if bad:
            r = min(bad, key=lambda r: tokens(r["N"]))
            return f"infeasible run {r['run_name']} at {tokens(r['N'])} tokens <= {tk}" + (
                f" ({r['reason']})" if r.get("reason") else "")
        if tk > self.max_feasible_tokens():
            return f"{tk} tokens exceed the largest feasible measured token count {self.max_feasible_tokens()} (unmeasured)"
        return None

    def feasible(self, N: int) -> bool:
        return self.infeasibility(N) is None

    def _points(self, tier: str, variant: str):
        own = [r for r in self.runs if r["variant"] == variant and r["tier"] == tier]
        if len({r["N"] for r in own}) >= 2:
            return own, "tier"
        return [r for r in self.runs if r["variant"] == variant], "shared"

    def c_step(self, tier: str, N: int, variant: str) -> float:
        pts, _ = self._points(tier, variant)
        return _interp([(tokens(r["N"]), r["train_seconds"] / self.steps / 3600.0) for r in pts], tokens(N))

    def c_eval(self, tier: str, N: int, variant: str) -> float:
        pts, _ = self._points(tier, variant)
        return _interp([(tokens(r["N"]), r["eval_seconds"] / 3600.0) for r in pts], tokens(N))

    def t0(self, variant: str) -> float:
        v = [r["startup_seconds"] / 3600.0 for r in self.runs if r["variant"] == variant]
        return math.fsum(v) / len(v)

    def source(self, tier: str, variant: str) -> str:
        return self._points(tier, variant)[1]

    def g_run(self, tier: str, N: int, variant: str, t_max_steps: int, n_evals: int) -> float:
        """G_run = T_max * c_step + n_ev * c_eval + t_0 in GPU-h.  Refuses an infeasible N."""
        why = self.infeasibility(N)
        if why:
            raise ValueError(f"N={N} is infeasible: {why}")
        return (t_max_steps * self.c_step(tier, N, variant) + n_evals * self.c_eval(tier, N, variant)
                + self.t0(variant))

    def g_run_epochs(self, tier, N, variant, epochs) -> float:
        return self.g_run(tier, N, variant, steps_for(epochs), n_evals_for(epochs))


# ---------------------------------------------------------------- pilot plan (design-log 9.2)
def pilot_specs() -> list:
    """(tier, N, variant, seed): G and S 2N x 2v x 2 seeds + A N=16 z first pilot seed = 17 runs."""
    return S.pilot_run_specs()


def lpt_makespan(durations_h, n_gpus: int = N_GPUS) -> float:
    """Longest-processing-time-first greedy schedule makespan (hours)."""
    loads = [0.0] * n_gpus
    for d in sorted(durations_h, reverse=True):
        i = loads.index(min(loads))
        loads[i] += d
    return max(loads)


def plan(model: CostModel, budget: float = PILOT_BUDGET, ext_cap: float = EXT_BUDGET) -> dict:
    """Design-log 9.2 decision (exp027_select.decide_budget) from a smoke-fitted model, plus wall-clock estimates."""
    ext_feas = {f"{tier}{N}": {"feasible": model.feasible(N), "reason": model.infeasibility(N)}
                for tier, ns in EXT_OPTIONS.items() for N in ns}
    feas_line = ("extension options feasible: "
                 + (", ".join(k for k, v in ext_feas.items() if v["feasible"]) or "none")
                 + "; infeasible: " + (", ".join(f"{k} ({v['reason']})" for k, v in ext_feas.items()
                                               if not v["feasible"]) or "none"))
    bad_pilot = sorted({(t, N) for (t, N, _v, _s) in pilot_specs() if not model.feasible(N)})
    if bad_pilot:
        reasons = {f"{t}{N}": model.infeasibility(N) for t, N in bad_pilot}
        lines = [f"pilot cells infeasible: {', '.join(f'{k} ({v})' for k, v in reasons.items())}",
                 feas_line, "-> NO_PILOT_INFEASIBLE: the pilot cells N 16 / 32 do not all fit; close with the smoke2 "
                 "measurement"]
        return {"branch": "NO_PILOT_INFEASIBLE", "epochs": None, "steps": None, "confirm_possible": False,
                "reason": reasons, "ext_feasibility": ext_feas, "smoke_cost": model.smoke_cost, "arithmetic": lines}
    cands, durs_by_e = [], {}
    for e in T_P_MENU_EPOCHS:
        durs = [model.g_run_epochs(t, N, v, e) for (t, N, v, _s) in pilot_specs()]
        durs_by_e[e] = durs
        opts = {f"{tier}{N}": (math.fsum(model.g_run_epochs(tier, N, v, e) for v in VARIANTS for _ in PILOT_SEEDS)
                               if ext_feas[f"{tier}{N}"]["feasible"] else None)
                for tier, ns in EXT_OPTIONS.items() for N in ns}
        cands.append({"epochs": e, "steps": steps_for(e), "n_evals": n_evals_for(e), "pilot_17": math.fsum(durs),
                      "ext_options": opts})
    g8 = [model.g_run_epochs("G", N, v, 420) for N in PILOT_NS for v in VARIANTS for _ in PILOT_SEEDS]
    out = S.decide_budget(model.smoke_cost, cands, math.fsum(g8), budget, ext_cap)
    out["ext_feasibility"] = ext_feas
    out["arithmetic"].append(feas_line)
    for r in out["candidates"]:
        r["wallclock_pilot_ideal_h"] = r["pilot_17"] / N_GPUS
        r["wallclock_pilot_lpt_h"] = lpt_makespan(durs_by_e[r["epochs"]])
    if out["branch"] == "FULL":
        r = next(c for c in out["candidates"] if c["epochs"] == out["epochs"])
        out["wallclock_pilot_lpt_h"] = r["wallclock_pilot_lpt_h"]
    elif out["branch"] == "G_ONLY_420":
        out["wallclock_pilot_lpt_h"] = lpt_makespan(g8)
    if "wallclock_pilot_lpt_h" in out:
        out["wallclock_pilot_lpt_days"] = out["wallclock_pilot_lpt_h"] / 24.0
    return out


# ---------------------------------------------------------------- confirmatory cells
def conf_cost(model: CostModel, cells, epochs: int,
              b_conf: float = B_CONF, n_cap: int = N_CAP) -> dict:
    """cells: [(tier, N)]; each expands to z and noz.  C_set = sum of G_run over the 2*len(cells) runs of one seed."""
    runs = [(t, N, v) for (t, N) in cells for v in VARIANTS]
    for (t, N) in cells:
        if not model.feasible(N):
            raise ValueError(f"cell {t}:{N} is infeasible: {model.infeasibility(N)}")
    per = {f"{t}N{N}_{v}": model.g_run_epochs(t, N, v, epochs) for (t, N, v) in runs}
    c_set = math.fsum(per.values())
    if not c_set > 0:
        raise ValueError("C_set must be positive")
    n = min(n_cap, int(math.floor(b_conf / c_set)))
    action = "ENQUEUE" if n >= 20 else ("ENQUEUE_WITH_QUALIFIER" if n >= 15 else "NO_CONFIRM")
    return {"epochs": epochs, "steps": steps_for(epochs), "cells": [list(c) for c in cells],
            "g_run_gpu_h": per, "C_set_gpu_h": c_set, "B_conf_gpu_h": b_conf, "n": n, "n_gate_action": action,
            "G_conf_gpu_h": n * c_set, "wallclock_ideal_days": n * c_set / N_GPUS / 24.0}


# ---------------------------------------------------------------- timing extraction (queue log + wandb _runtime)
_TQDM = re.compile(r"(\d+)/(\d+) \[(\d+(?::\d+)*)<")
RUN_NAME = re.compile(r"^e027_(smoke|smoke2|pilot|ext|conf)_([ASG])N(\d+)_(z|noz)_s(\d+)$")   # seed 0, 1000*i or 910x000


def _hms(s: str) -> int:
    t = 0
    for part in s.split(":"):
        t = t * 60 + int(part)
    return t


def parse_tqdm(text: str):
    """[(iter, elapsed_seconds)] with duplicate iters collapsed to the first occurrence, sorted by iter."""
    d: dict = {}
    for chunk in text.replace("\r", "\n").split("\n"):
        for m in _TQDM.finditer(chunk):
            d.setdefault(int(m.group(1)), _hms(m.group(3)))
    return sorted(d.items())


def split_train_eval(pairs, n_evals: int):
    """Total elapsed -> (train_seconds, [eval_seconds ...]).

    An eval blocks the progress bar, so it appears as an outsized gap between consecutive iterations.  The
    n_evals largest per-iteration gaps (minus the median per-iteration time) are taken as evals.  tqdm elapsed
    has 1 s resolution, so a per-eval value is accurate to about +-1 s."""
    if len(pairs) < 3:
        raise ValueError("too few tqdm lines")
    gaps = []
    for (i0, t0), (i1, t1) in zip(pairs, pairs[1:]):
        gaps.append(((t1 - t0) / max(i1 - i0, 1), i1, t1 - t0))
    per_iter = sorted(g[0] for g in gaps)
    med = per_iter[len(per_iter) // 2]
    big = sorted(gaps, key=lambda g: -g[2])[:n_evals]
    evals = [max(g[2] - med, 0.0) for g in big]
    total = pairs[-1][1]
    return float(total - sum(evals)), sorted(evals)


def find_log(run_name: str, log_dir: str) -> str:
    # the log is named after the job file: <prefix>_<NNNN>_<run>.log, or <existing stem>_z<NNNN>_<run>.log (--interleave)
    pat = re.compile(r"^[0-9A-Za-z][^/]*_" + re.escape(run_name) + r"\.log$")
    hits = [p for p in glob.glob(os.path.join(log_dir, "*.log")) if pat.match(os.path.basename(p))]
    if len(hits) != 1:
        raise LookupError(f"{run_name}: expected exactly 1 queue log in {log_dir}, found {len(hits)}")
    return hits[0]


def wandb_runtime(run_name: str, wandb_root=None) -> float:
    import outcomes
    sys.path.insert(0, os.path.join(REPO, "tools", "figures"))
    from wandb_ro import iter_history
    ds = outcomes.resolve_runs([run_name], wandb_root)[run_name]
    if len(ds) != 1:
        raise ValueError(f"{run_name}: {len(ds)} wandb dirs")
    f = outcomes._wandb_file(ds[0])
    if f is None:
        raise ValueError(f"{run_name}: no .wandb file")
    last = None
    for _s, v in iter_history(f, ["_runtime"]):
        last = v["_runtime"]
    if last is None:
        raise ValueError(f"{run_name}: no _runtime in wandb history")
    return float(last)


_OOM = re.compile(r"out of memory|OutOfMemoryError", re.I)
_CRASH = re.compile(r"Traceback \(most recent call last\)")


def failure_class(text: str):
    """'oom' / 'error' when the log shows a crash, None when it shows none (run unfinished or still running)."""
    if _OOM.search(text):
        return "oom"
    if _CRASH.search(text):
        return "error"
    return None


def extract_timings(run_names, log_dir, steps=SMOKE_STEPS, n_evals=SMOKE_N_EVALS, wandb_root=None,
                    runtime_fn=wandb_runtime) -> dict:
    """Smoke JSON from queue logs (+ wandb _runtime).  A missing log, or a log that stops early WITHOUT an
    out-of-memory message (still running, or crashed for another reason), raises: only out-of-memory makes a run
    infeasible (amendment 1, DL 11); any other crash keeps v3's rule (extract fails, v3 section 9).  A run whose
    log shows out of memory before reaching `steps`, or a Traceback with out of memory after the last tick (final
    eval / checkpoint save), does not abort: it is listed under "infeasible" with error_class 'oom'."""
    runs, infeasible = [], []
    for name in run_names:
        m = RUN_NAME.match(name)
        if not m:
            raise ValueError(f"run name {name!r} does not match e027_<stage>_<tier>N<N>_<variant>_s<seed>")
        _stage, tier, N, var, _seed = m.groups()
        path = find_log(name, log_dir)
        text = open(path, errors="replace").read()
        pairs = parse_tqdm(text)
        reached = bool(pairs) and pairs[-1][0] == steps
        cls = failure_class(text)
        if cls == "oom" and (not reached or _CRASH.search(text)):
            infeasible.append({"run_name": name, "tier": tier, "N": int(N), "variant": var,
                               "reason": "CUDA out of memory", "error_class": cls, "log": os.path.basename(path),
                               "last_iter": pairs[-1][0] if pairs else None})
            continue
        if not reached:
            raise ValueError(f"{name}: log ends at iter {pairs[-1][0] if pairs else None}, expected {steps} "
                             + ("(crashed without an out-of-memory message: not infeasible; extract fails as in v3)"
                                if cls == "error" else "(run unfinished or steps mismatch)"))
        # pretrain.py evaluates after the last step of each chunk, so the final eval follows the final tqdm tick
        # and leaves no gap; only n_evals - 1 evals are visible (checked: e026 trm_h16_k3_s3, 25 evals, 24 gaps).
        # The final eval's time therefore falls into startup_seconds (runtime - tqdm elapsed).
        train, evals = split_train_eval(pairs, max(n_evals - 1, 0))
        total_elapsed = pairs[-1][1]
        runtime = runtime_fn(name, wandb_root)
        runs.append({"run_name": name, "tier": tier, "N": int(N), "variant": var,
                     "train_seconds": train,
                     "eval_seconds": math.fsum(evals) / len(evals) if evals else 0.0,
                     "startup_seconds": max(runtime - total_elapsed, 0.0),
                     "log": os.path.basename(path), "wandb_runtime_seconds": runtime,
                     "tqdm_elapsed_seconds": total_elapsed})
    return {"steps": steps, "n_evals": n_evals, "runs": runs, "infeasible": infeasible}


# ---------------------------------------------------------------- CLI
def _parse_cells(text: str):
    out = []
    for tok in text.split(","):
        t, n = tok.split(":")
        if t not in ("S", "G"):
            raise ValueError(f"bad tier in cell {tok!r}")
        out.append((t, int(n)))
    return out


def _dump(obj, path):
    s = json.dumps(obj, indent=1, sort_keys=True)
    if path:
        with open(path, "w") as f:
            f.write(s + "\n")
    else:
        print(s)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="exp027_cost.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--smoke", required=True)
    p.add_argument("--budget", type=float, default=PILOT_BUDGET)
    p.add_argument("--ext-cap", type=float, default=EXT_BUDGET)
    p.add_argument("--out")
    c = sub.add_parser("conf")
    c.add_argument("--smoke", required=True)
    c.add_argument("--cells", required=True)
    c.add_argument("--epochs", type=int, required=True)      # T_max in epochs; no default (was a silent 820)
    c.add_argument("--b-conf", type=float, default=B_CONF)
    c.add_argument("--out")
    e = sub.add_parser("extract")
    e.add_argument("--runs", nargs="+", required=True)
    e.add_argument("--log-dir", default=os.path.join(REPO, "logs", "queue"))
    e.add_argument("--steps", type=int, default=SMOKE_STEPS)
    e.add_argument("--n-evals", type=int, default=SMOKE_N_EVALS)
    e.add_argument("--wandb-root")
    e.add_argument("--out")
    try:
        a = ap.parse_args(argv)
    except SystemExit as x:
        return 2 if x.code else 0
    try:
        if a.cmd == "extract":
            _dump(extract_timings(a.runs, a.log_dir, a.steps, a.n_evals, a.wandb_root), a.out)
        else:
            model = CostModel(load_smoke(a.smoke))
            if a.cmd == "plan":
                res = plan(model, a.budget, a.ext_cap)
                print("\n".join(res["arithmetic"]), file=sys.stderr)
                _dump(res, a.out)
            else:
                _dump(conf_cost(model, _parse_cells(a.cells), a.epochs, a.b_conf), a.out)
    except (LookupError, ValueError, OSError) as ex:
        print(f"FAIL: {ex}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
