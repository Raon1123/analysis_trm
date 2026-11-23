"""EXP-027 P4 supplement (CPU only), written AFTER p4_v2.json was seen:
(1) own-alternative firing of every branch under the unimodal model M2 (same ALTS as
    p4_null_firing.py, base/shift used as mu/shift), and
(2) M1 alternatives outside the pre-written mid-range p regime: both variants train on
    (nearly) every seed (c_own p 0.95 / 1.0), and a large uniform z benefit from a low
    base (b_own base 0.05, shift 0.9).
Reported beside the pre-registered M1 table; the decidable set is NOT re-chosen from it
(prereg §2.3).  All branches kept (decidability is what is measured).
Usage: p4_supplement_m2_highp.py --seeds 8 10 12 15 --reps 10000 --json OUT.  Exit 0.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p4_null_firing as P  # noqa: E402
import exp027_rules as R  # noqa: E402

d4 = P.d4
EXTRA_M1 = {
    "c_own_p0.95": (d4((0.95,) * 4), d4((0,) * 4), False, "S-c"),
    "c_own_p1.0": (d4((1.0,) * 4), d4((0,) * 4), False, "S-c"),
    "b_own_lowbase_big": (d4((0.05,) * 4), d4((0.9,) * 4), False, "S-b"),
}
EXTRA_M2 = {"c_own_mu0.9": (d4((0.9,) * 4), d4((0,) * 4), False, "S-c")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[8, 10, 12, 15])
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--json", default=None)
    ap.add_argument("--rng", type=int, default=20261112)
    a = ap.parse_args()
    rng = np.random.default_rng(a.rng)
    out = {"args": vars(a), "rows": []}
    for plan, Ns in (("baseline", (16, 32)), ("reduced", (32,))):
        for S in a.seeds:
            W = R.bootstrap_weights(S)
            jobs = [("M2", n, v) for n, v in {**P.ALTS, **EXTRA_M2}.items()]
            jobs += [("M1", n, v) for n, v in EXTRA_M1.items()]
            for model, name, (base, shift, aceil, own) in jobs:
                r = P.rates(P.draw(rng, model, base, shift, Ns, S, a.reps, aceil), Ns, W)
                row = {"plan": plan, "S": S, "model": model, "alt": name, "own": own,
                       "P_own": r[own], "rates": r}
                out["rows"].append(row)
                print(f"{plan:8s} S={S:2d} {model} {name:18s} P({own})={r[own]:.3f} "
                      f"S-i={r['S-i']:.3f} S-a1={r['S-a1']:.3f} S-e={r['S-e']:.3f} "
                      f"UNC={r['UNCOVERED']:.3f}", flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
