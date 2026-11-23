"""P1 helper: check sb_generator.py against the pinned state-bench source.

Run with an isolated interpreter from OUTSIDE the cloned repo, e.g.

  CUDA_VISIBLE_DEVICES="" python3 -I tools/statebench_tiers/p1_equiv_check.py \
      --src <clone>/src --trials 3000

Only ``state_bench/model.py`` and ``state_bench/generation.py`` are loaded (the
package ``__init__`` is bypassed so ``semantics``/``io`` never execute).  For each
trial the same ``random.Random(seed)`` drives both generators; operations, query
and final state of all four families must match.  Exit 0 = all equal, 1 = mismatch.
"""

from __future__ import annotations

import argparse
import importlib
import random
import sys
import types
from pathlib import Path


def load_upstream(src: Path):
    pkg = types.ModuleType("state_bench")
    pkg.__path__ = [str(src / "state_bench")]
    sys.modules["state_bench"] = pkg
    model = importlib.import_module("state_bench.model")
    gen = importlib.import_module("state_bench.generation")
    return model, gen


def to_tuple(op, model):
    if isinstance(op, model.AssignConstant):
        return ("const", op.target, op.value)
    if isinstance(op, model.AssignVariable):
        return ("copy", op.target, op.source)
    return ("inc", op.target, op.delta)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--trials", type=int, default=3000)
    # D1 extension: restrict the grid and set aperiodic_constants (min, max).  Defaults keep
    # the original grid (m 4/5/8, q 100/127/5, N 8/16/32, constants 1..64) and its seeds.
    ap.add_argument("--m-list", type=int, nargs="+", default=[4, 5, 8])
    ap.add_argument("--q-list", type=int, nargs="+", default=[100, 127, 5])
    ap.add_argument("--N-list", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--c", type=int, nargs=2, default=[1, 64], metavar=("CMIN", "CMAX"))
    args = ap.parse_args()
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    import sb_generator as mine  # noqa: E402

    model, gen = load_upstream(args.src)
    fams = (model.Complexity.R_TRIVIAL, model.Complexity.APERIODIC,
            model.Complexity.SOLVABLE, model.Complexity.PERIODIC)
    names = {model.Complexity.R_TRIVIAL: "r-trivial", model.Complexity.APERIODIC: "aperiodic",
             model.Complexity.SOLVABLE: "solvable", model.Complexity.PERIODIC: "periodic"}
    bad = 0
    n = 0
    cmin, cmax = args.c
    ncell = len(args.m_list) * len(args.q_list) * len(args.N_list)
    for m in args.m_list:
        for q in args.q_list:
            for N in args.N_list:
                for t in range(args.trials // ncell + 1):
                    key = (m, q, N, t) if (cmin, cmax) == (1, 64) else (m, q, N, t, cmin, cmax)
                    seed = hash(key) & 0xFFFFFFFF
                    src_up = gen.generate_periodic_source(
                        num_variables=m, num_values=q, min_extra_assignments=N,
                        max_extra_assignments=N, min_aperiodic_constants=cmin,
                        max_aperiodic_constants=cmax, rng=random.Random(seed))
                    ups = dict(zip(fams, gen.derive_variants(src_up, families=fams)))
                    src_me = mine.periodic_source(m=m, q=q, n_extra_min=N, n_extra_max=N,
                                                  ap_const_min=cmin, ap_const_max=cmax,
                                                  rng=random.Random(seed))
                    mes = mine.derive(src_me)
                    for f in fams:
                        u, me = ups[f], mes[names[f]]
                        ok = (tuple(to_tuple(o, model) for o in u.operations) == me.operations
                              and tuple(u.query) == me.query
                              and tuple(u.final_state) == me.final_state
                              and mine.evaluate(me) == me.final_state)
                        n += 1
                        if not ok:
                            bad += 1
                            if bad <= 3:
                                print("MISMATCH", m, q, N, t, names[f])
    print(f"compared {n} (trace, family) pairs; mismatches {bad}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
