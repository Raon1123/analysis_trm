"""EXP-027 P4 (v2, revised grid 2026-11-10): null firing and own-alternative power
of every §2 branch, by simulation (CPU only).  No EXP-027 value exists.

Revised design (orchestrator decisions 2026-11-10): tiers A (aperiodic), S (soluble
q=127), G (general finite), C (control: soluble q=5); N in {16, 32}; r-trivial and
N=8 are not trained.  delta = 0.10 (fixed in v1, prereg §2.3).

Outcome models (one run = (tier, N, variant, seed)):
  M1 bimodal mixture (PRIMARY): each run trains w.p. p -> E ~ U(0.95, 1.00), else
     fails -> E ~ floor + U(0, 0.05).  Reflects the seed-bimodal outcomes of this
     model family (2026-11-10 note §2.3/§4.5: one cell 0.013 in s1, 0.982 in s2).
  M2 unimodal: E = clip(mu + Normal(0, 0.15), 0, 1).  Shown for nulls only.
  "Aceil": aperiodic tier drawn U(0.995, 1.0) in BOTH variants (uninformative by the
     0.99 ceiling rule); exercises the S-a1 aperiodic-uninformative path.

Null families: H0-noz (no z effect anywhere), H0-noint (same z shift in every tier).
Own alternatives (M1, base p = 0.3, z shift added to trm's p per tier A/S/G/C):
  a1_large  0/0.2/0.5/0.2      a1_moder  0/0.1/0.3/0.1     (+ Aceil versions)
  b_own     0.4/0.4/0.4/0.4    c_own     0/0/0/0 (p 0.3 and 0.5)
  a2_own    0/0.4/0.4/0.4      d_own     0/0.4/0.4/0
Requirement: P(S-a1|S-a2|S-d) <= 0.10 over all null configs, P(S-b) <= 0.10 over H0-noz.
Usage: p4_null_firing.py --seeds 8 10 12 15 --json OUT [--power-only-seeds 20 30 40 50]
Exit 0.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp027_rules as R  # noqa: E402

TIERS = ("A", "S", "G", "C")
CODES = ("S-e", "S-c", "S-b", "S-a1", "S-a2", "S-d", "S-i", "UNCOVERED")
DELTA = 0.10
# P2 preview floors, c = 0 (p2_floor_preview.py; max exact over non-defining predictors)
FLOOR = {"A": {16: 0.0, 32: 0.0309}, "S": {16: 0.0, 32: 0.0309},
         "G": {16: 0.0, 32: 0.0309}, "C": {16: 0.0, 32: 0.0331}}


def draw(rng, model, base, shift, Ns, S, reps, aceil=False):
    E = {}
    for t in TIERS:
        E[t] = {}
        for N in Ns:
            E[t][N] = {}
            for v, add in (("trm_singlez", 0.0), ("trm", shift[t])):
                if aceil and t == "A":
                    x = rng.uniform(0.995, 1.0, (reps, S))
                elif model == "M1":
                    p = min(1.0, base[t] + add)
                    ok = rng.random((reps, S)) < p
                    x = np.where(ok, rng.uniform(0.95, 1.0, (reps, S)),
                                 FLOOR[t][N] + rng.uniform(0, 0.05, (reps, S)))
                else:
                    x = np.clip(base[t] + add + rng.normal(0, 0.15, (reps, S)), 0, 1)
                E[t][N][v] = x
    return E


def rates(E, Ns, W):
    q, inf = R.quantities(E, FLOOR, Ns, W)
    sc = R.scenarios(q, inf, DELTA)          # all branches kept: decidability is what we measure
    return {c: float((sc == c).mean()) for c in CODES}


def d4(vals):
    return dict(zip(TIERS, vals))


def null_configs():
    g3 = (0.2, 0.5, 0.8)
    for a, s, g in itertools.product(g3, repeat=3):
        yield "H0-noz", "M1", d4((a, s, g, s)), d4((0,) * 4), False
    for s, g in itertools.product(g3, repeat=2):
        yield "H0-noz", "M1", d4((0.5, s, g, s)), d4((0,) * 4), True
    for a, s, g in itertools.product((0.2, 0.5), repeat=3):
        yield "H0-noint", "M1", d4((a, s, g, s)), d4((0.3,) * 4), False
    for s, g in itertools.product((0.2, 0.5), repeat=2):
        yield "H0-noint", "M1", d4((0.5, s, g, s)), d4((0.3,) * 4), True
    for a, s, g in itertools.product((0.3, 0.7), repeat=3):
        yield "H0-noz", "M2", d4((a, s, g, s)), d4((0,) * 4), False
    for a, s, g in itertools.product((0.3, 0.5), repeat=3):
        yield "H0-noint", "M2", d4((a, s, g, s)), d4((0.2,) * 4), False


ALTS = {
    "a1_large": (d4((0.3,) * 4), d4((0, 0.2, 0.5, 0.2)), False, "S-a1"),
    "a1_large_Aceil": (d4((0.3,) * 4), d4((0, 0.2, 0.5, 0.2)), True, "S-a1"),
    "a1_moder": (d4((0.3,) * 4), d4((0, 0.1, 0.3, 0.1)), False, "S-a1"),
    "a1_moder_Aceil": (d4((0.3,) * 4), d4((0, 0.1, 0.3, 0.1)), True, "S-a1"),
    "b_own": (d4((0.3,) * 4), d4((0.4,) * 4), False, "S-b"),
    "c_own_p0.3": (d4((0.3,) * 4), d4((0,) * 4), False, "S-c"),
    "c_own_p0.5": (d4((0.5,) * 4), d4((0,) * 4), False, "S-c"),
    "a2_own": (d4((0.3,) * 4), d4((0, 0.4, 0.4, 0.4)), False, "S-a2"),
    "d_own": (d4((0.3,) * 4), d4((0, 0.4, 0.4, 0)), False, "S-d"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[8, 10, 12, 15])
    ap.add_argument("--power-only-seeds", type=int, nargs="*", default=[])
    ap.add_argument("--reps", type=int, default=4000)
    ap.add_argument("--confirm-reps", type=int, default=20000)
    ap.add_argument("--alt-reps", type=int, default=10000)
    ap.add_argument("--json", default=None)
    ap.add_argument("--rng", type=int, default=20261110)
    args = ap.parse_args()
    rng = np.random.default_rng(args.rng)
    plans = {"baseline": (16, 32), "reduced": (32,)}
    res = {"args": vars(args), "delta": DELTA, "plans": {}}
    for plan, Ns in plans.items():
        for S in args.seeds + args.power_only_seeds:
            W = R.bootstrap_weights(S)
            entry = {"Ns": Ns, "S": S}
            alts = {}
            for name, (base, shift, aceil, _) in ALTS.items():
                if S in args.power_only_seeds and not name.startswith("a1_large"):
                    continue
                alts[name] = rates(draw(rng, "M1", base, shift, Ns, S, args.alt_reps, aceil), Ns, W)
            entry["alternatives"] = alts
            if S in args.seeds:
                scan = []
                for fam, model, base, shift, aceil in null_configs():
                    r = rates(draw(rng, model, base, shift, Ns, S, args.reps, aceil), Ns, W)
                    scan.append({"family": fam, "model": model, "base": base, "shift": shift,
                                 "Aceil": aceil, "rates": r})
                worst = {}
                for code, fams in (("S-a1", ("H0-noz", "H0-noint")), ("S-a2", ("H0-noz", "H0-noint")),
                                   ("S-d", ("H0-noz", "H0-noint")), ("S-b", ("H0-noz",))):
                    for model in ("M1", "M2"):
                        cand = [x for x in scan if x["family"] in fams and x["model"] == model]
                        w = max(cand, key=lambda x: x["rates"][code])
                        conf = rates(draw(rng, w["model"], w["base"], w["shift"], Ns, S,
                                          args.confirm_reps, w["Aceil"]), Ns, W)[code]
                        worst[f"{code}|{model}"] = {"scan_max": w["rates"][code], "confirmed": conf,
                                                    "config": {k: w[k] for k in ("family", "base", "shift", "Aceil")}}
                entry["null_worst"] = worst
                entry["null_pass"] = all(v["confirmed"] <= 0.10 for v in worst.values())
                entry["scan"] = scan
                # null S-i rate at the most variable M1 null (p=0.5 everywhere, H0-noz)
                entry["null_Si_p05"] = next(x["rates"]["S-i"] for x in scan
                                            if x["model"] == "M1" and not x["Aceil"]
                                            and x["family"] == "H0-noz"
                                            and set(x["base"].values()) == {0.5})
            res["plans"][f"{plan}_S{S}"] = entry
            line = f"{plan:8s} S={S:2d} " + " ".join(
                f"{n}:{alts[n][ALTS[n][3]]:.3f}/Si{alts[n]['S-i']:.2f}" for n in alts)
            if "null_worst" in entry:
                line += " | null " + " ".join(f"{k}={v['confirmed']:.3f}" for k, v in entry["null_worst"].items())
                line += f" pass={entry['null_pass']}"
            print(line, flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
