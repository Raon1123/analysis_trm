"""EXP-027 P1-realized / P2 preview: trace statistics and trivial-predictor floors.

PREVIEW ONLY: runs on the sb_generator.py re-implementation, not on the frozen
EXP-027 data builder output.  The binding P2 run is on the built npy files.

Per (tier, N) at m registers + TMP and value alphabet q, over --samples sources:
  * swaps per trace, constant resets per trace, scratch loads
  * distinct real-register permutations realised by the swap sequence
  * fraction of traces whose swap sequence contains two overlapping (hence
    non-commuting) transpositions
  * trivial predictors on the m ORDINARY registers (TMP excluded, EXP-027 label
    set), exact and token accuracy:
      init     = first literal written to each register (initial assignment)
      lastlit  = last literal written to each register (copies/increments ignored)
      mode     = most frequent literal of the trace, broadcast (ties -> smallest)
Exit 0.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sb_generator as G  # noqa: E402


def predictors(trace):
    R = trace.num_variables + 1
    init = [None] * R
    last = [None] * R
    lits = Counter()
    for kind, t, a in trace.operations:
        if kind == "const":
            if init[t] is None:
                init[t] = a
            last[t] = a
            lits[a] += 1
    top = max(lits.values())
    mode = min(v for v, c in lits.items() if c == top)
    return {"init": tuple(init), "lastlit": tuple(last), "mode": tuple([mode] * R)}


def swap_perm(trace):
    m = trace.num_variables
    perm = list(range(m))  # perm[i] = original register whose content now sits in i
    ops = trace.operations
    i = 0
    while i < len(ops):
        k, t, a = ops[i]
        if (k == "copy" and t == m and i + 2 < len(ops) and ops[i + 1][0] == "copy"
                and ops[i + 1][1] == a and ops[i + 2] == ("copy", ops[i + 1][2], m)):
            b = ops[i + 1][2]
            perm[a], perm[b] = perm[b], perm[a]
            i += 3
        else:
            i += 1
    return tuple(perm)


def swap_pairs(trace):
    m = trace.num_variables
    ops = trace.operations
    out = []
    i = 0
    while i < len(ops):
        k, t, a = ops[i]
        if (k == "copy" and t == m and i + 2 < len(ops) and ops[i + 1][0] == "copy"
                and ops[i + 1][1] == a and ops[i + 2] == ("copy", ops[i + 1][2], m)):
            out.append((a, ops[i + 1][2]))
            i += 3
        else:
            i += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=5)
    ap.add_argument("--q", type=int, default=127)
    ap.add_argument("--Ns", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--samples", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ap-const", type=int, nargs=2, default=[1, 64],
                    help="aperiodic_constants min max (repo default.yaml: 1 64; periodic_test.yaml: 0 0)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    out = []
    for N in args.Ns:
        rng = random.Random(args.seed * 1000 + N)
        stats = {t: {p: [0, 0] for p in ("init", "lastlit", "mode")} for t in G.TIERS}
        swaps, consts, perms, noncomm = Counter(), Counter(), set(), 0
        for _ in range(args.samples):
            src = G.periodic_source(m=args.m, q=args.q, n_extra_min=N, n_extra_max=N,
                                    ap_const_min=args.ap_const[0], ap_const_max=args.ap_const[1],
                                    rng=rng)
            fam = G.derive(src)
            ops = src.operations[args.m + 1:]
            n_const = sum(1 for o in ops if o[0] == "const")
            n_copy = len(ops) - n_const
            consts[n_const] += 1
            swaps[n_copy // 3] += 1
            perms.add(swap_perm(src))
            prs = swap_pairs(src)
            noncomm += int(any(set(a) & set(b) for a, b in zip(prs, prs[1:])
                               ) or any(set(prs[i]) & set(prs[j]) for i in range(len(prs))
                                        for j in range(i + 1, len(prs))))
            for tier, tr in fam.items():
                pr = predictors(tr)
                for p, pred in pr.items():
                    m = args.m
                    hit = sum(int(x == y) for x, y in zip(pred[:m], tr.final_state[:m]))
                    stats[tier][p][0] += int(hit == m)
                    stats[tier][p][1] += hit
        R = args.m + 1
        row = {"m": args.m, "q": args.q, "N": N, "samples": args.samples,
               "ap_const": args.ap_const,
               "seq_len_tokens(2/op + m answer slots)": 2 * (R + N) + args.m,
               "frac_traces_with_overlapping_swaps": noncomm / args.samples,
               "swaps_hist": dict(sorted(swaps.items())),
               "const_resets_hist": dict(sorted(consts.items())),
               "distinct_swap_permutations": len(perms),
               "floors": {t: {p: {"exact": v[0] / args.samples,
                                  "token_acc": v[1] / (args.samples * args.m)}
                              for p, v in d.items()} for t, d in stats.items()}}
        out.append(row)
        print(json.dumps({k: row[k] for k in ("N", "ap_const", "seq_len_tokens(2/op + m answer slots)",
                                              "swaps_hist", "const_resets_hist",
                                              "distinct_swap_permutations",
                                              "frac_traces_with_overlapping_swaps")}))
        for t in G.TIERS:
            print("  ", t, {p: (round(v["exact"], 4), round(v["token_acc"], 4))
                            for p, v in row["floors"][t].items()})
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
