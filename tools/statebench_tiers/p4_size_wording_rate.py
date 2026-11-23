"""EXP-027 P4 supplement (CPU only): how often the binding S-a1 wording rule
(exp027_rules.size_wording, prereg §2.2) lets an S-a1 reading drop the qualifier
"operator size not excluded", under each simulated alternative.

Same draw()/FLOOR/DELTA/ALTS as p4_null_firing.py; decidable set = ("S-a1",).
size_excluded holds iff the bootstrap interval of sz = -D_CS lies strictly inside
(-|D_GS|, +|D_GS|) (vectorised copy of size_wording).
Usage: p4_size_wording_rate.py --seeds 10 --reps 10000 --json OUT.  Exit 0.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp027_rules as R  # noqa: E402
import p4_null_firing as P  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[10])
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--json", default=None)
    ap.add_argument("--rng", type=int, default=20261111)
    a = ap.parse_args()
    rng = np.random.default_rng(a.rng)
    out = {"args": vars(a), "rows": []}
    for plan, Ns in (("baseline", (16, 32)), ("reduced", (32,))):
        for S in a.seeds:
            W = R.bootstrap_weights(S)
            for name in ("a1_large", "a1_large_Aceil", "a1_moder", "d_own", "a2_own"):
                base, shift, aceil, _ = P.ALTS[name]
                E = P.draw(rng, "M1", base, shift, Ns, S, a.reps, aceil)
                q, inf = R.quantities(E, P.FLOOR, Ns, W)
                sc = R.scenarios(q, inf, P.DELTA, ("S-a1",))
                _, lo, hi = q["D_CS"]
                g = np.abs(q["D_GS"][0])
                excl = (-hi > -g) & (-lo < g)
                fire = sc == "S-a1"
                row = {"plan": plan, "S": S, "alt": name, "P_Sa1": float(fire.mean()),
                       "P_size_excluded_given_Sa1": (float(excl[fire].mean()) if fire.any() else None),
                       "P_Sa1_and_size_excluded": float((fire & excl).mean())}
                out["rows"].append(row)
                print(row, flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
