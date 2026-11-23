"""EXP-027 outcome extraction from eval trajectories (design 03_DESIGN_DIALECTIC.md s5.2 / s6).

A trajectory is a step-sorted list of (step, exact) pairs, E_i = test all/exact_accuracy.
All functions are pure except the loader section at the bottom (read-only wandb access).
All statistics are float64; no randomness, no wall clock.

Definitions (defaults in THRESH):
  tau      step of first eval i with E_i >= 0.90 and E_{i+1} >= 0.90, else None
  Y(T)     1 iff such a pair exists with BOTH evals at step <= T (T=None: whole trajectory)
  BSE      max_i min(E_i, E_{i+1})   (None if fewer than 2 evals)
  stability  fraction of evals strictly after tau with E >= 0.90 (None if tau None or no eval after)
  collapse   Y=1 and some eval strictly after tau has E < 0.5
  graded     Y=0 and BSE in [0.20, 0.90)
  pre_takeoff  Y=0, E_last >= 0.10, max(last 5 evals) > max(all evals before those last 5).
               With <= 5 evals there is no earlier eval, so the flag is False.

Float32 tolerance: pretrain.py computes exact_accuracy in float32 (metric sums / count on a float32
array), so a logged 0.900 is 0.8999999761581421 and a logged 0.950 is 0.949999988079071.  Every
threshold comparison against an eval value therefore uses tol = 1e-6: x ">= t" is x >= t - tol and
x "< t" is x < t - tol.  All thresholds lie on the 0.001 eval grid, distinct grid values differ by
>= 1e-3 and the float32 error is <= 6e-8, so the tolerance only removes the float32 rounding.
"""
from __future__ import annotations

import glob
import math
import os
import re
import sys
from dataclasses import dataclass

import numpy as np

KEY = "all/exact_accuracy"
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


@dataclass(frozen=True)
class Thresholds:
    learn: float = 0.90          # E >= learn counts as learned
    collapse_below: float = 0.5  # post-tau eval below this => collapse
    graded_lo: float = 0.20
    graded_hi: float = 0.90
    pretakeoff_min_last: float = 0.10
    pretakeoff_window: int = 5
    tol: float = 1e-6            # float32 rounding tolerance on every threshold comparison


THRESH = Thresholds()


def ge(x, t, th: Thresholds = THRESH) -> bool:
    """x >= t up to float32 rounding of the logged eval value."""
    return bool(x >= t - th.tol)


def lt(x, t, th: Thresholds = THRESH) -> bool:
    """x < t up to float32 rounding of the logged eval value."""
    return bool(x < t - th.tol)


def _arr(traj):
    steps = np.array([s for s, _ in traj], dtype=np.float64)
    e = np.array([x for _, x in traj], dtype=np.float64)
    return steps, e


def tau_index(traj, T=None, th: Thresholds = THRESH):
    """Index i of the first qualifying pair (both evals within T), else None."""
    steps, e = _arr(traj)
    for i in range(len(e) - 1):
        if T is not None and steps[i + 1] > T:
            break
        if ge(e[i], th.learn, th) and ge(e[i + 1], th.learn, th):
            return i
    return None


def tau(traj, T=None, th: Thresholds = THRESH):
    i = tau_index(traj, T, th)
    return None if i is None else int(traj[i][0])


def Y(traj, T=None, th: Thresholds = THRESH) -> int:
    return int(tau_index(traj, T, th) is not None)


def BSE(traj):
    _, e = _arr(traj)
    if len(e) < 2:
        return None
    return float(np.max(np.minimum(e[:-1], e[1:])))


def max_exact(traj):
    return float(max(x for _, x in traj)) if traj else None


def final(traj):
    return float(traj[-1][1]) if traj else None


def mean_last(traj, n=5):
    if not traj:
        return None
    return float(np.mean(np.array([x for _, x in traj[-n:]], dtype=np.float64)))


def stability(traj, th: Thresholds = THRESH):
    i = tau_index(traj, None, th)
    if i is None:
        return None
    after = [x for _, x in traj[i + 1:]]
    if not after:
        return None
    return float(np.mean(np.array(after, dtype=np.float64) >= th.learn - th.tol))


def collapse(traj, th: Thresholds = THRESH) -> bool:
    i = tau_index(traj, None, th)
    if i is None:
        return False
    return any(lt(x, th.collapse_below, th) for _, x in traj[i + 1:])


def graded(traj, th: Thresholds = THRESH) -> bool:
    if Y(traj, None, th):
        return False
    b = BSE(traj)
    return b is not None and ge(b, th.graded_lo, th) and lt(b, th.graded_hi, th)


def pre_takeoff(traj, th: Thresholds = THRESH) -> bool:
    if Y(traj, None, th) or not traj:
        return False
    e = [x for _, x in traj]
    w = th.pretakeoff_window
    if lt(e[-1], th.pretakeoff_min_last, th) or len(e) <= w:
        return False
    return max(e[-w:]) > max(e[:-w])


def summarize(traj, th: Thresholds = THRESH) -> dict:
    return dict(n_evals=len(traj), tau=tau(traj, None, th), Y=Y(traj, None, th), BSE=BSE(traj),
                max=max_exact(traj), final=final(traj), mean_last5=mean_last(traj, 5),
                stability=stability(traj, th), collapse=collapse(traj, th),
                graded=graded(traj, th), pre_takeoff=pre_takeoff(traj, th))


# ---------------------------------------------------------------- loader (read-only wandb)
def clean_trajectory(pairs):
    """Sort by step; for duplicate steps keep the last written value."""
    d = {}
    for s, v in pairs:
        d[int(s)] = float(v)
    return sorted(d.items())


def wandb_run_dirs(root=None):
    return sorted(glob.glob(os.path.join(root or os.path.join(REPO, "wandb"), "run-*")))


def run_config(run_dir):
    import yaml
    c = os.path.join(run_dir, "files", "config.yaml")
    cfg = yaml.safe_load(open(c))
    return {k: (v.get("value") if isinstance(v, dict) else v) for k, v in cfg.items()}


def _wandb_file(run_dir):
    f = glob.glob(os.path.join(run_dir, "run-*.wandb"))
    return f[0] if f else None


def load_trajectory_file(path, key=KEY):
    """Read one .wandb file; raises ValueError on an empty history."""
    sys.path.insert(0, os.path.join(REPO, "tools", "figures"))
    from wandb_ro import iter_history
    pairs = [(s, v[key]) for s, v in iter_history(path, [key]) if key in v]
    traj = clean_trajectory(pairs)
    if not traj:
        raise ValueError(f"empty {key} history: {path}")
    check_finite(traj, path)
    return traj


def check_finite(traj, where="trajectory"):
    """Raise ValueError on a non-finite eval value (a NaN would read as MIX in BSE mode)."""
    bad = [s for s, v in traj if not math.isfinite(v)]
    if bad:
        raise ValueError(f"non-finite {KEY} at steps {bad[:5]} in {where}")


def resolve_runs(names_or_ids, root=None):
    """Map run names (config run_name, exact) or 8-char wandb ids to run dirs.

    Returns {query: [run_dir, ...]}; a query matching nothing raises LookupError.
    A name may match several dirs (re-runs): the caller decides.
    """
    dirs = wandb_run_dirs(root)
    out = {}
    names = None
    for q in names_or_ids:
        hit = [d for d in dirs if d.endswith("-" + q)]
        if not hit:
            if names is None:
                names = {}
                for d in dirs:
                    try:
                        names.setdefault(run_config(d).get("run_name"), []).append(d)
                    except Exception:
                        continue
            hit = names.get(q, [])
        if not hit:
            raise LookupError(f"no wandb run for {q!r}")
        out[q] = hit
    return out


def load_trajectories(names_or_ids, root=None, key=KEY):
    """{query: trajectory}; fails loudly (LookupError / ValueError) on missing or empty runs.
    A query that matches several run dirs raises ValueError (ambiguous)."""
    res = {}
    for q, ds in resolve_runs(names_or_ids, root).items():
        if len(ds) != 1:
            raise ValueError(f"{q!r} matches {len(ds)} run dirs: {ds}")
        f = _wandb_file(ds[0])
        if f is None:
            raise ValueError(f"no .wandb file in {ds[0]}")
        res[q] = load_trajectory_file(f, key)
    return res


def load_candidates(names, root=None, key=KEY):
    """{name: [(run_dir, trajectory), ...]} with every wandb dir of the name, oldest first (dir names carry the
    start time).  A dir whose history is empty or has no .wandb file yields an empty trajectory [] (the run died
    before its first eval).  A name with no dir at all raises LookupError.  Non-finite values raise ValueError."""
    res = {}
    for q, ds in resolve_runs(names, root).items():
        cands = []
        for d in sorted(ds, key=os.path.basename):
            f = _wandb_file(d)
            if f is None:
                cands.append((d, []))
                continue
            try:
                cands.append((d, load_trajectory_file(f, key)))
            except ValueError as e:
                if str(e).startswith("empty"):
                    cands.append((d, []))
                else:
                    raise
        res[q] = cands
    return res


def main(argv=None):
    import json
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print("usage: outcomes.py <run_name_or_id>...", file=sys.stderr)
        return 2
    try:
        tr = load_trajectories(argv)
    except (LookupError, ValueError) as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    print(json.dumps({q: summarize(t) for q, t in tr.items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
