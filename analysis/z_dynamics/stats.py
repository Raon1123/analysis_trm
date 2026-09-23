"""EXP-017 cell statistics (C-Z-1 / C-Z-2 / C-Z-3 and descriptive D-1, D-5).

Every threshold is read from judge.FROZEN_RULES through judge.num()/rule(). The functions reproduce
EXP-017_null_check.py (v2) rules: cell_M / M_batch / shape_sub (C-Z-1), pooled_gamma + shuffled-call
permutation test + mixture qualifier + P_int + chance (C-Z-2), E_L (C-Z-3), exact stratified enumeration.

Exactness: a_j, v_j etc. are integer counts over a fixed denominator. R, maxdrop and maxrise are
formed from integer differences and divided once, so a boundary value such as R = 0.2 compares
equal to the frozen literal 0.20 ("all thresholds inclusive as written").
"""

from __future__ import annotations

import itertools
import math
import re
from collections import Counter

import numpy as np

from analysis.z_dynamics import judge
from analysis.z_dynamics.judge import num, rule
from analysis.z_dynamics.readouts import (UNDEF, SigmaInfo, is_valid, perm_from_cycle_type, trivial)


# ================================================================== C-Z-1

def classify_M(a1: float, R: float, maxdrop: float, maxrise: float) -> str:
    eps, rmin, pfa = num("czz1.EPS_DROP"), num("czz1.R_MIN"), num("czz1.PF_A1")
    if a1 >= pfa and maxdrop <= eps:
        return "PF"
    if R >= rmin and maxdrop <= eps:
        return "M+"
    if R <= -rmin and maxrise <= eps:
        return "M-"
    return "M0"


def cell_M(a) -> str:
    """Float series a_1..a_J (null_check.cell_M)."""
    a = np.asarray(a, float)
    R = a[-1] - a[0]
    maxdrop = float(np.max(np.maximum.accumulate(a) - a))
    maxrise = float(np.max(a - np.minimum.accumulate(a)))
    return classify_M(float(a[0]), float(R), maxdrop, maxrise)


def cell_M_counts(counts, denom: int) -> dict:
    """Integer agreement counts c_1..c_17 over `denom` positions -> exact R / maxdrop / maxrise and class."""
    c = np.asarray(counts, dtype=np.int64)
    R = int(c[-1] - c[0]) / denom
    maxdrop = int(np.max(np.maximum.accumulate(c) - c)) / denom
    maxrise = int(np.max(c - np.minimum.accumulate(c))) / denom
    a1 = int(c[0]) / denom
    M = classify_M(a1, R, maxdrop, maxrise)
    return {"a1": a1, "a17": int(c[-1]) / denom, "R": R, "maxdrop": maxdrop, "maxrise": maxrise, "M": M,
            "subclass": shape_sub(c) if M == "M+" else ""}


def shape_sub(a) -> str:
    """D-1 subclass of an M+ series (null_check.shape_sub): jump / staircase / gradual."""
    a = np.asarray(a, float)
    inc = np.diff(a)
    pos = np.clip(inc, 0, None)
    R = a[-1] - a[0]
    jf = num("czz1.subclass.jump_frac")
    sf = num("czz1.subclass.staircase_boundary_frac")
    b = judge.boundary_diff_indices()
    if R > 0 and inc.max() >= jf * R:
        return "jump"
    if pos.sum() > 0 and sum(pos[i] for i in b) >= sf * pos.sum():
        return "staircase"
    return "gradual"


def M_batch(A: np.ndarray) -> np.ndarray:
    """Vectorised cell_M: codes 2=PF, 1=M+, -1=M-, 0=M0 (null_check.M_batch)."""
    eps, rmin, pfa = num("czz1.EPS_DROP"), num("czz1.R_MIN"), num("czz1.PF_A1")
    R = A[:, -1] - A[:, 0]
    maxdrop = np.max(np.maximum.accumulate(A, axis=1) - A, axis=1)
    maxrise = np.max(A - np.minimum.accumulate(A, axis=1), axis=1)
    pf = (A[:, 0] >= pfa) & (maxdrop <= eps)
    up = ~pf & (R >= rmin) & (maxdrop <= eps)
    dn = ~pf & (R <= -rmin) & (maxrise <= eps)
    return np.where(pf, 2, np.where(up, 1, np.where(dn, -1, 0)))


# ================================================================== C-Z-2

def gamma_counts(M: np.ndarray) -> tuple[int, int]:
    """Concordant / discordant pair counts over j < j' within each sample; entries < 0 are undefined."""
    Mi = M[:, :, None]
    Mj = M[:, None, :]
    ok = (Mi >= 0) & (Mj >= 0)
    iu = np.triu(np.ones((M.shape[1], M.shape[1]), bool), 1)[None]
    C = int(np.sum(ok & iu & (Mi < Mj)))
    D = int(np.sum(ok & iu & (Mi > Mj)))
    return C, D


def pooled_gamma(M: np.ndarray) -> float:
    C, D = gamma_counts(M)
    return 0.0 if C + D == 0 else (C - D) / (C + D)  # gamma undefined -> 0


def shuffle_rows(M: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    idx = np.argsort(rng.random(M.shape), axis=1)
    return np.take_along_axis(M, idx, axis=1)


def perm_pvalue(M: np.ndarray, g_obs: float, n_perm: int | None = None, seed: int | None = None) -> float:
    """Shuffled-call test: permute the J readouts independently within each sample.

    p = (1 + #{γ_perm >= γ_obs}) / (1 + n_perm); a fresh Generator(PERM_SEED) per call.
    """
    n_perm = int(num("czz2.N_PERM")) if n_perm is None else n_perm
    seed = int(num("czz2.PERM_SEED")) if seed is None else seed
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n_perm):
        ge += pooled_gamma(shuffle_rows(M, rng)) >= g_obs
    return (1 + ge) / (1 + n_perm)


def chance_n_draw() -> int:
    s = str(rule("czz2.chance"))
    m = re.search(r"(\d+) draws per distinct cycle type", s)
    if not m:
        raise judge.FrozenRuleMissing(f"I-9: cannot parse draw count from czz2.chance = {s!r}")
    return int(m.group(1))


_OMEGA_CACHE: dict[tuple, float] = {}


def omega(ctype: tuple[int, ...], k: int, n_draw: int | None = None) -> float:
    """ω(σ): P(uniform π ∈ S_10 is 'proper' w.r.t. σ). Depends on σ only through its cycle type
    (conjugation invariance of Hamming and of the per-cycle mixture), so it is estimated once per
    (cycle type, k) on the canonical representative with a deterministic seed [PERM_SEED, k, *ctype]."""
    n_draw = chance_n_draw() if n_draw is None else n_draw
    key = (tuple(ctype), int(k), int(n_draw))
    if key in _OMEGA_CACHE:
        return _OMEGA_CACHE[key]
    s = SigmaInfo(perm_from_cycle_type(tuple(ctype)), k)
    if not s.A:
        val = float("nan")
    else:
        rng = np.random.default_rng([int(num("czz2.PERM_SEED")), int(k), *[int(x) for x in ctype]])
        pis = np.stack([rng.permutation(len(s.sig)) for _ in range(n_draw)])
        near = s.nearest(pis)
        val = float(s.proper(near).mean())
    _OMEGA_CACHE[key] = val
    return val


def pint_threshold(p0: float) -> float:
    return max(num("czz2.PINT_FLOOR"), num("czz2.PINT_MULT") * p0)


def classify_Z(g: float, p: float | None, pint: float, thr: float) -> str:
    if g >= num("czz2.GAMMA_MIN") and p is not None and p <= num("czz2.P_MAX"):
        return "stepwise" if pint >= thr else "jump"
    return "unordered"


# ================================================================== D-1 helpers

def shortest_chains(k: int) -> list[tuple[int, ...]]:
    """All shortest addition chains 1 = a_0 < ... < a_r = k (small k; BFS by length)."""
    if k == 1:
        return [(1,)]
    frontier = [(1,)]
    while True:
        nxt, done = [], []
        for ch in frontier:
            for x, y in itertools.combinations_with_replacement(ch, 2):
                s = x + y
                if s > ch[-1] and s <= k:
                    c2 = ch + (s,)
                    (done if s == k else nxt).append(c2)
        if done:
            return sorted(set(done))
        frontier = sorted(set(nxt))


def addition_chain_len(k: int) -> int:
    """ℓ(k): number of additions in a shortest addition chain."""
    return len(shortest_chains(k)[0]) - 1


def vis_class(vis: set[int], k: int) -> str:
    mid = ((k - 2) + (addition_chain_len(k) - 1)) / 2
    if len(vis) > mid:
        return "linear-like"
    if len(vis) >= 1:
        return "log-like"
    return "empty"


def vis_in_shortest_chain(vis: set[int], k: int) -> bool:
    return any(vis <= set(ch[1:-1]) for ch in shortest_chains(k))


# ================================================================== C-Z-3

def E_L(pr) -> float:
    """Excess path length of ln PR over j = 1..18 (primary C-Z-3 statistic)."""
    l = np.log(np.asarray(pr, float))
    return float(np.abs(np.diff(l)).sum() - abs(l[-1] - l[0]))


def E_L_from_logs(l) -> float:
    l = np.asarray(l, float)
    return float(np.abs(np.diff(l)).sum() - abs(l[-1] - l[0]))


def Theta_L(pr) -> float:
    """Tortuosity (descriptive only). floor 0.05 as frozen in czz3.descriptive_only."""
    s = str(rule("czz3.descriptive_only")[0])
    m = re.search(r"floor ([0-9.]+)", s)
    if not m:
        raise judge.FrozenRuleMissing("I-9: cannot parse Theta_L floor")
    floor = float(m.group(1))
    l = np.log(np.asarray(pr, float))
    return float(np.abs(np.diff(l)).sum() / max(abs(l[-1] - l[0]), floor))


def Lambda_L(pr) -> float:
    """mean_{j<=17} ln(PR_j / PR_18) (descriptive only)."""
    pr = np.asarray(pr, float)
    return float(np.mean(np.log(pr[:-1] / pr[-1])))


def exact_stratified_rates(sizes: dict[str, tuple[int, int]]) -> dict:
    """Exact enumeration of within-stratum G/C labelings (values distinct; the result is value-free)."""
    lab_sets = []
    for s, (nG, nC) in sizes.items():
        n = nG + nC
        lab_sets.append([(s, set(G)) for G in itertools.combinations(range(n), nG)])
    vals = {s: np.arange(nG + nC, dtype=float) for s, (nG, nC) in sizes.items()}
    tot = gt = lt = 0
    for combo in itertools.product(*lab_sets):
        tot += 1
        a = b = True
        for s, G in combo:
            v = vals[s]
            g = [v[i] for i in G]
            c = [v[i] for i in range(len(v)) if i not in G]
            a &= min(c) > max(g)
            b &= max(c) < min(g)
        gt += a
        lt += b
    return {"n_labelings": tot, "P_S3_1": gt / tot, "P_S3_2": lt / tot, "P_union": (gt + lt) / tot}


# ================================================================== per-cell decode analysis

def analyze_decodes(dec: np.ndarray, inputs_tok: np.ndarray, labels_tok: np.ndarray, k: int,
                    infos: list[SigmaInfo] | None = None, n_perm: int | None = None,
                    ref_dec: np.ndarray | None = None, j_labels: list[int] | None = None,
                    cell_stats: bool = True) -> dict:
    """Full per-cell analysis of one readout kind on one split.

    dec: (N, J_all, 10) decoded tokens (J_all = 18 for prelim/raw/lens; readouts j = 1..J_all).
    inputs_tok: (N, >=10) σ tokens; labels_tok: (N, 10) σ^k tokens.
    ref_dec: decodes used for a_self (defaults to dec[:, -1], i.e. ŷ_18).
    j_labels: readout index of each column (default 1..J_all; native readouts use [6, 12, 18]).
    cell_stats: False -> only per-j rows (native kind).
    Returns {"per_j": [...], "cell": {...}, "near": per-sample dicts} with statistics over j <= J_stat.
    """
    from analysis.z_dynamics.readouts import sigma_infos
    N, J_all, _ = dec.shape
    J = judge.j_stat()
    infos = sigma_infos(inputs_tok, k) if infos is None else infos
    ref = dec[:, -1] if ref_dec is None else ref_dec
    dec0 = dec.astype(np.int64) - 1
    inputs10 = np.asarray(inputs_tok)[:, :10]

    mstar = np.full((N, J_all), UNDEF, dtype=np.int64)
    ham_min = np.zeros((N, J_all), dtype=np.int64)
    tie = np.zeros((N, J_all), bool)
    none = np.zeros((N, J_all), bool)
    proper = np.zeros((N, J_all), bool)
    residues = []
    for i, s in enumerate(infos):
        near = s.nearest(dec0[i])
        mstar[i] = near["m"]
        ham_min[i] = near["ham_min"]
        tie[i] = near["tie"]
        none[i] = near["none"]
        proper[i] = s.proper(near)
        residues.append(s.residues(dec0[i]))
    informative = np.array([s.informative for s in infos])
    gcd1 = np.array([s.gcd == 1 for s in infos])
    valid = is_valid(dec)                                    # (N, J_all)
    triv = np.stack([trivial(dec[:, j], inputs10) for j in range(J_all)], 1)
    corr = (dec == labels_tok[:, None, :10])                 # (N, J_all, 10)

    jl = list(range(1, J_all + 1)) if j_labels is None else list(j_labels)
    per_j = []
    for gname, gmask in (("all", np.ones(N, bool)), ("1", gcd1), (">1", ~gcd1)):
        n_g = int(gmask.sum())
        for j in range(J_all):
            if n_g == 0:
                continue
            per_j.append({
                "j": jl[j], "gcd_class": gname, "n_rows": n_g,
                "a": int(corr[gmask, j].sum()) / (n_g * 10),
                "a_self": int((dec[gmask, j] == ref[gmask]).sum()) / (n_g * 10),
                "v": int(valid[gmask, j].sum()) / n_g,
                "trivial": int(triv[gmask, j].sum()) / n_g,
                "near_power_frac": int((mstar[gmask, j] >= 0).sum()) / n_g,
                "proper_partial_frac": int(proper[gmask, j].sum()) / n_g,
                "none_frac": int(none[gmask, j].sum()) / n_g,
                "tie_frac": int(tie[gmask, j].sum()) / n_g,
            })

    near = {"m_star": mstar, "ham_min": ham_min, "tie": tie, "none": none, "proper": proper,
            "residues": residues, "infos": infos}
    if not cell_stats:
        return {"per_j": per_j, "cell": None, "near": near}
    if J_all != judge.rule("readouts.j_all")[1]:
        raise ValueError(f"cell statistics need all {judge.rule('readouts.j_all')[1]} readouts, got {J_all}")

    # ---- C-Z-1 (j = 1..J)
    counts = corr[:, :J].sum(axis=(0, 2))
    cz1 = cell_M_counts(counts, N * 10)
    V = int(valid[:, :J].sum()) / (N * J)
    n_valid = int(valid[:, :J].sum())
    T = (int((valid[:, :J] & triv[:, :J]).sum()) / n_valid) if n_valid else float("nan")

    # ---- C-Z-2 (j = 1..J)
    on_path = np.where((mstar[:, :J] >= 0) & (mstar[:, :J] <= k), mstar[:, :J], UNDEF)
    g = pooled_gamma(on_path)
    p = perm_pvalue(on_path, g, n_perm=n_perm) if g >= num("czz2.GAMMA_MIN") else None
    n_inf = int(informative.sum())
    pint = float(proper[informative, :J].mean()) if n_inf else 0.0
    om = [omega(s.ctype, k) for s in infos if s.informative]
    p0 = float(np.mean(om)) if om else 0.0
    thr = pint_threshold(p0)
    Z_raw = classify_Z(g, p, pint, thr)
    Z = "PF" if cz1["M"] == "PF" else Z_raw
    mid = (mstar[:, :J] >= 2) & (mstar[:, :J] <= k - 1)
    n_mix_rej = int((mid & ~proper[:, :J]).sum())

    # ---- D-1 visited set, j*
    vis = set()
    if n_inf:
        vs = num("czz2.vis_support")
        for j in range(J):
            sel = informative
            for m in range(2, k):
                if (proper[sel, j] & (mstar[sel, j] == m)).sum() / n_inf >= vs:
                    vis.add(m)
    jstar = None
    for j in range(J_all):
        d = mstar[:, j][mstar[:, j] >= 0]
        if d.size:
            mode = Counter(d.tolist()).most_common()
            top = mode[0][1]
            modal = sorted(m for m, c in mode if c == top)
            if modal == [k]:
                jstar = j + 1
                break
    cell = dict(cz1)
    cell.update({
        "n_rows": N, "V": V, "T": T, "gamma": g, "p": p, "P_int": pint, "P_int0": p0, "thr": thr,
        "informative_frac": n_inf / N, "n_informative": n_inf, "n_mixture_rejected": n_mix_rej,
        "Z": Z, "Z_raw": Z_raw, "Vis": sorted(vis),
        "vis_class": vis_class(vis, k) if (Z == "stepwise" and k >= 5) else "",
        "vis_in_shortest_chain": vis_in_shortest_chain(vis, k) if (Z == "stepwise" and k >= 5) else "",
        "jstar": jstar,
    })
    return {"per_j": per_j, "cell": cell, "near": near}
