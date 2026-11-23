"""EXP-027 decision rules (§2 of the prereg).  Vectorised over replicate axis 0.

This module is the single implementation used by both the P4 null-firing
simulation and the post-run judge, so the rule that is frozen is the rule that
is simulated.  It is hashed into EXP-027_FREEZE.sha256 before any value exists.

Inputs
------
E[tier][N][variant] : float array (reps, S) of final test exact, one column per
                      seed index s (seed s of trm pairs with seed s of trm_singlez).
floor[tier][N]      : float, P2 trivial-predictor floor (max exact over the P2
                      predictor set for that tier/N, defining predictor excluded).
tiers               : "A" aperiodic, "S" solvable (q=127), "G" general finite,
                      "C" control = solvable with q=5.  r-trivial is never passed.
Ns                  : primary trace lengths (baseline {16,32}, reduced {32}).
delta               : effect threshold (fixed by P4; see prereg §2.3).

Outputs
-------
scenario code per replicate: "S-e", "S-c", "S-b", "S-a1", "S-a2", "S-d", "S-i",
"UNCOVERED".  ``permutation_check`` adds the judge-time permutation nulls.
"""

from __future__ import annotations

import itertools

import numpy as np

CEIL = 0.99
FLOOR_MARGIN = 0.01
PRIMARY = ("A", "S", "G")


def bootstrap_weights(S: int, max_exact: int = 100_000, n_random: int = 4000, seed: int = 0):
    """Row-stochastic resample matrix (B, S): exact enumeration of all S^S ordered
    resamples when S^S <= max_exact, else n_random fixed-seed resamples."""
    if S ** S <= max_exact:
        idx = np.array(list(itertools.product(range(S), repeat=S)), dtype=np.int64)
    else:
        idx = np.random.default_rng(seed).integers(0, S, size=(n_random, S))
    W = np.zeros((idx.shape[0], S), dtype=np.float32)
    for j in range(S):
        W[np.arange(idx.shape[0]), idx[:, j]] += 1.0 / S
    return W


def mean_ci(x: np.ndarray, W: np.ndarray, valid: np.ndarray):
    """x: (reps, S) per-seed values; returns mean, lo, hi (2.5/97.5 percentile of the
    seed bootstrap).  Rows with valid=False get NaN."""
    reps = x.shape[0]
    mean = np.full(reps, np.nan)
    lo = np.full(reps, np.nan)
    hi = np.full(reps, np.nan)
    if not valid.any():
        return mean, lo, hi
    xv = np.nan_to_num(x[valid].astype(np.float32))
    bm = xv @ W.T                                  # (r, B)
    B = bm.shape[1]
    k_lo = int(np.floor(0.025 * (B - 1)))
    k_hi = int(np.ceil(0.975 * (B - 1)))
    part = np.partition(bm, (k_lo, k_hi), axis=1)
    mean[valid] = xv.mean(axis=1)
    lo[valid] = part[:, k_lo]
    hi[valid] = part[:, k_hi]
    return mean, lo, hi


def classify(mean, lo, hi, delta):
    """'+', '-', '0', '?' per replicate (NaN -> '?')."""
    out = np.full(mean.shape, "?", dtype="<U1")
    ok = ~np.isnan(mean)
    pos = ok & (mean >= delta) & (lo > 0)
    neg = ok & (mean <= -delta) & (hi < 0)
    zero = ok & (lo > -delta) & (hi < delta)
    out[zero] = "0"
    out[pos] = "+"
    out[neg] = "-"
    return out


def informative(E, floor, tier, N):
    """(reps,) bool: cell (tier, N) carries information (not both-ceiling, not both-floor)."""
    mt = E[tier][N]["trm"].mean(axis=1)
    ms = E[tier][N]["trm_singlez"].mean(axis=1)
    both_ceil = (mt >= CEIL) & (ms >= CEIL)
    f = floor[tier][N] + FLOOR_MARGIN
    both_floor = (mt <= f) & (ms <= f)
    return ~(both_ceil | both_floor)


def per_seed_benefit(E, floor, tier, Ns):
    """B_T(s): mean over informative N of E(trm) - E(trm_singlez); (reps,S), NaN if none."""
    reps, S = E[tier][Ns[0]]["trm"].shape
    acc = np.zeros((reps, S))
    cnt = np.zeros((reps, 1))
    for N in Ns:
        inf = informative(E, floor, tier, N)[:, None]
        d = E[tier][N]["trm"] - E[tier][N]["trm_singlez"]
        acc += np.where(inf, d, 0.0)
        cnt += inf
    with np.errstate(invalid="ignore", divide="ignore"):
        b = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
    return b, (cnt[:, 0] > 0)


def quantities(E, floor, Ns, W):
    """Means/CIs for B_A, B_S, B_G, D_SA, D_GS, D_GA, D_CS. Returns dict name->(mean,lo,hi), inf."""
    B, inf = {}, {}
    for t in ("A", "S", "G", "C"):
        B[t], inf[t] = per_seed_benefit(E, floor, t, Ns)
    q = {}
    for t in ("A", "S", "G"):
        q[f"B_{t}"] = mean_ci(B[t], W, inf[t])
    for name, (t2, t1) in {"D_SA": ("S", "A"), "D_GS": ("G", "S"),
                           "D_GA": ("G", "A"), "D_CS": ("C", "S")}.items():
        q[name] = mean_ci(B[t2] - B[t1], W, inf[t2] & inf[t1])
    return q, inf


ALL_BRANCHES = ("S-c", "S-b", "S-a1", "S-a2", "S-d")


def scenarios(q, inf, delta, decidable=ALL_BRANCHES):
    """Ordered, mutually exclusive branches (prereg §2.2). Returns (reps,) array of codes.

    ``decidable``: branches kept as outcomes.  A branch not in it is "not decidable at
    this budget" (prereg §2.2) and its replicates are reported as S-i.  S-e, S-i and
    UNCOVERED are always present.  The frozen value is FROZEN_DECIDABLE below."""
    c = {k: classify(*v, delta) for k, v in q.items()}
    n_inf = inf["A"].astype(int) + inf["S"].astype(int) + inf["G"].astype(int)
    reps = n_inf.shape[0]
    out = np.full(reps, "UNCOVERED", dtype="<U9")

    s_e = n_inf < 2
    # S-c: every informative primary tier has B in {0,-}; uninformative tiers ignored
    nobenefit = np.ones(reps, bool)
    anyplus = np.zeros(reps, bool)
    for t in PRIMARY:
        cb = c[f"B_{t}"]
        nobenefit &= (~inf[t]) | np.isin(cb, ["0", "-"])
        anyplus |= inf[t] & (cb == "+")
    s_c = ~s_e & nobenefit
    s_b = ~s_e & anyplus & (c["D_SA"] == "0") & (c["D_GS"] == "0") & (c["D_GA"] == "0")
    # S-a1 also fires when the aperiodic tier is uninformative (e.g. both variants at ceiling):
    # then the soluble -> general-finite step alone carries it (D_GA, D_SA are '?').
    a_ok = (c["D_GA"] == "+") & (c["D_SA"] != "-")
    a_unk = ~inf["A"]
    s_a1_ = (~s_e & (c["B_G"] == "+") & (c["D_GS"] == "+") & (a_ok | a_unk))
    s_a2_ = (~s_e & (c["B_G"] == "+") & a_ok & (c["D_GS"] == "0")
             & np.isin(c["D_CS"], ["0", "+"]))
    s_d = (~s_e & (c["B_S"] == "+") & (c["D_GS"] == "0") & (c["D_CS"] == "-")
           & ((c["D_SA"] == "+") | (c["D_GA"] == "+")))
    s_a1 = s_a1_                         # step at soluble -> general finite
    s_a2 = s_a2_                         # step only at aperiodic -> {soluble, general finite}
    fired = s_e | s_c | s_b | s_a1 | s_a2 | s_d
    anyq = np.zeros(reps, bool)
    for k in ("B_A", "B_S", "B_G", "D_SA", "D_GS", "D_GA"):
        anyq |= c[k] == "?"
    s_i = ~fired & anyq                  # indeterminate: an interval too wide / tier missing
    stack = np.stack([s_e, s_c, s_b, s_a1, s_a2, s_d, s_i])
    if (stack.sum(axis=0) > 1).any():
        raise AssertionError("branches co-fired; rules are not mutually exclusive")
    for code, m in zip(("S-e", "S-c", "S-b", "S-a1", "S-a2", "S-d", "S-i"), stack):
        out[m] = code
    for code in ALL_BRANCHES:
        if code not in decidable:
            out[out == code] = "S-i"
    return out


def size_wording(q, reps_index=0):
    """Binding S-a1 wording rule (prereg §2.2).  Size effect on the z benefit is
    sz = B_S - B_C = -D_CS (soluble q=127 minus q=5).  'operator size not excluded'
    must be attached to any S-a1 reading unless the seed-bootstrap interval of sz lies
    strictly inside (-|D_GS|, +|D_GS|), i.e. it excludes a size effect as large as the
    observed tier step D_GS.  Returns dict with the estimate and the flag."""
    m, lo, hi = (float(x[reps_index]) for x in q["D_CS"])
    g = float(q["D_GS"][0][reps_index])
    sz, sz_lo, sz_hi = -m, -hi, -lo
    ok = not any(np.isnan([sz, g])) and (sz_lo > -abs(g)) and (sz_hi < abs(g))
    return {"size_effect": sz, "size_lo": sz_lo, "size_hi": sz_hi, "D_GS": g,
            "size_excluded": bool(ok),
            "required_qualifier": None if ok else "operator size not excluded"}


# P4 v2 (2026-11-10, S = 10, M1): own-alternative firing of S-b, S-c, S-a2, S-d = 0.000
# (< 0.5) -> folded into S-i.  S-a1 (0.342 baseline / 0.174 reduced, also < 0.5) is kept
# by the orchestrator's s = 10 fallback instruction, not by the 0.5 criterion (prereg §2.2).
# Set after p4_v2.json was produced; p4_null_firing.py keeps all branches by design.
FROZEN_DECIDABLE = ("S-a1",)


def judge(E, floor, Ns, delta, S, decidable=None):
    decidable = FROZEN_DECIDABLE if decidable is None else decidable
    W = bootstrap_weights(S)
    q, inf = quantities(E, floor, Ns, W)
    return scenarios(q, inf, delta, decidable), q, inf


# ------------------------------------------------------------ judge-time permutation nulls
TIERS_ALL = ("A", "S", "G", "C")


def _tile(E, n):
    return {t: {N: {v: np.repeat(E[t][N][v][:1], n, axis=0) for v in E[t][N]} for N in E[t]}
            for t in E}


def perm_noz(E, n_perm, rng):
    """Within every (tier, N, seed) cell swap the two variant labels with prob 1/2."""
    P = _tile(E, n_perm)
    for t in P:
        for N in P[t]:
            a, b = P[t][N]["trm"], P[t][N]["trm_singlez"]
            flip = rng.random(a.shape) < 0.5
            P[t][N]["trm"], P[t][N]["trm_singlez"] = np.where(flip, b, a), np.where(flip, a, b)
    return P


def perm_noint(E, n_perm, rng):
    """Within every (N, seed) block permute the per-cell z effect d = E_trm - E_singlez
    across the four tiers {A,S,G,C}; trm' = clip(E_singlez + d_perm, 0, 1).  Keeps each
    tier's trm_singlez level, destroys any tier dependence of the z effect."""
    P = _tile(E, n_perm)
    Ns = list(P["A"].keys())
    S = P["A"][Ns[0]]["trm"].shape[1]
    for N in Ns:
        D = np.stack([P[t][N]["trm"] - P[t][N]["trm_singlez"] for t in TIERS_ALL])  # (4, n, S)
        for s in range(S):
            order = np.argsort(rng.random((n_perm, 4)), axis=1)                     # (n, 4)
            D[:, :, s] = np.take_along_axis(D[:, :, s].T, order, axis=1).T
        for i, t in enumerate(TIERS_ALL):
            P[t][N]["trm"] = np.clip(P[t][N]["trm_singlez"] + D[i], 0.0, 1.0)
    return P


def permutation_check(E, floor, Ns, delta, S, n_perm=20000, seed=0, decidable=None):
    """Observed code (E has one replicate row) and its firing probability under the
    two judge-time permutation nulls.  Directional codes S-a1/S-a2/S-d are licensed only
    if P_noint(code) <= 0.10; S-b only if P_noz(S-b) <= 0.10 (prereg §2.3)."""
    decidable = FROZEN_DECIDABLE if decidable is None else decidable
    rng = np.random.default_rng(seed)
    W = bootstrap_weights(S)
    obs = scenarios(*quantities(E, floor, Ns, W), delta, decidable)[0]
    p_noz = scenarios(*quantities(perm_noz(E, n_perm, rng), floor, Ns, W), delta, decidable)
    p_noint = scenarios(*quantities(perm_noint(E, n_perm, rng), floor, Ns, W), delta, decidable)
    f_noz = float((p_noz == obs).mean())
    f_noint = float((p_noint == obs).mean())
    if obs in ("S-a1", "S-a2", "S-d"):
        licensed = f_noint <= 0.10
    elif obs == "S-b":
        licensed = f_noz <= 0.10
    else:
        licensed = None          # non-directional code; no licence needed
    return {"observed": str(obs), "P_noz": f_noz, "P_noint": f_noint, "directional_licensed": licensed}
