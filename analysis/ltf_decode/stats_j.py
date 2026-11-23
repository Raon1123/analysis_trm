"""P-C1 cell statistics with J as an argument (spec §4.1).

``analyze_decodes_j`` is the same computation as ``analysis.z_dynamics.stats.analyze_decodes`` (lines 258-374)
with the two EXP-017 hard-wirings made parameters:
    J     = number of statistic readouts (EXP-017: judge.j_stat() = 17; P-C1 primary: 20 = |T_stat|)
    J_all = number of readouts in ``dec`` that cell statistics need (EXP-017: 18; P-C1: 21)
Everything else is imported UNMODIFIED from analysis.z_dynamics (SigmaInfo, sigma_infos, is_valid, trivial,
pooled_gamma, perm_pvalue, omega, pint_threshold, classify_Z, cell_M_counts, vis_class, vis_in_shortest_chain);
those read their thresholds from EXP-017 FROZEN_RULES, which P-C1's ``czz2`` equals (judge.py checks deep-equality).

Output keys are identical to the EXP-017 function so that G-a item 5 can compare them exactly. In the
cell dict ``a17`` therefore means "a at the last statistic readout, a_J" and ``subclass`` is the EXP-017 D-1
shape subclass; P-C1 does not use the subclass (its boundary_diffs are TRM H-cycle boundaries), so
``with_subclass=False`` blanks it.

Also here: the synthetic ``genuine_sequential`` decode generator (port of the null-check generator
``dec_power_path`` / ``seq_path`` with an explicit seeded rng) used by G-a item 5 and by G-c.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from analysis.z_dynamics import judge as zjudge
from analysis.z_dynamics.judge import num
from analysis.z_dynamics.readouts import UNDEF, SigmaInfo, is_valid, sigma_infos, trivial
from analysis.z_dynamics.stats import (cell_M_counts, classify_Z, omega, perm_pvalue, pint_threshold,
                                       pooled_gamma, vis_class, vis_in_shortest_chain)


def analyze_decodes_j(dec: np.ndarray, inputs_tok: np.ndarray, labels_tok: np.ndarray, k: int, J: int,
                      J_all: int, infos: list[SigmaInfo] | None = None, n_perm: int | None = None,
                      ref_dec: np.ndarray | None = None, j_labels: list[int] | None = None,
                      cell_stats: bool = True, with_subclass: bool = True) -> dict:
    """dec: (N, W, 10) decoded tokens (1..10, 0 = pad). With cell_stats, W must equal J_all.

    Returns {"per_j": [...], "cell": {...} | None, "near": {...}} exactly as stats.analyze_decodes.
    """
    N, W, _ = dec.shape
    infos = sigma_infos(inputs_tok, k) if infos is None else infos
    ref = dec[:, -1] if ref_dec is None else ref_dec
    dec0 = dec.astype(np.int64) - 1
    J_cols = W

    mstar = np.full((N, J_cols), UNDEF, dtype=np.int64)
    ham_min = np.zeros((N, J_cols), dtype=np.int64)
    tie = np.zeros((N, J_cols), bool)
    none = np.zeros((N, J_cols), bool)
    proper = np.zeros((N, J_cols), bool)
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
    inputs10 = np.asarray(inputs_tok)[:, :10]
    valid = is_valid(dec)
    triv = np.stack([trivial(dec[:, j], inputs10) for j in range(J_cols)], 1)
    corr = (dec == labels_tok[:, None, :10])

    jl = list(range(1, J_cols + 1)) if j_labels is None else list(j_labels)
    per_j = []
    for gname, gmask in (("all", np.ones(N, bool)), ("1", gcd1), (">1", ~gcd1)):
        n_g = int(gmask.sum())
        for j in range(J_cols):
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
    if W != J_all:
        raise ValueError(f"cell statistics need all {J_all} readouts, got {W}")
    if not (1 <= J < J_all):
        raise ValueError(f"need 1 <= J < J_all, got J={J}, J_all={J_all}")

    # ---- C-Z-1 (readouts 1..J): only PF / V are used by P-C1
    counts = corr[:, :J].sum(axis=(0, 2))
    cz1 = cell_M_counts(counts, N * 10)
    if not with_subclass:
        cz1["subclass"] = ""
    V = int(valid[:, :J].sum()) / (N * J)
    n_valid = int(valid[:, :J].sum())
    T = (int((valid[:, :J] & triv[:, :J]).sum()) / n_valid) if n_valid else float("nan")

    # ---- C-Z-2 (readouts 1..J)
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


# ------------------------------------------------------------------ t*_i(k) (descriptive, spec §2.4)

def tstar_first_hit(dec: np.ndarray, labels_tok: np.ndarray, t_labels: list[int]) -> np.ndarray:
    """t*_i(k): first loop t at which the whole decode equals sigma^k; -1 if never. dec (N, W, 10)."""
    hit = (dec == labels_tok[:, None, :10]).all(-1)            # (N, W)
    first = np.where(hit.any(1), hit.argmax(1), -1)
    tl = np.asarray(t_labels)
    return np.where(first >= 0, tl[np.clip(first, 0, None)], -1)


# ------------------------------------------------------------------ synthetic decodes (G-a item 5, G-c)

def _path_cuts(rng: np.random.Generator, n_seg: int, width: int) -> list[int]:
    cuts = np.sort(rng.choice(np.arange(1, width), size=n_seg - 1, replace=False))
    return [0] + [int(c) for c in cuts] + [width]


def synth_genuine_sequential(sigs: list[SigmaInfo], k: int, width: int, rng: np.random.Generator,
                             hold_from: int | None = None) -> np.ndarray:
    """(N, total, 10) tokens: for each sample the sigma^1..sigma^k staircase over ``width`` columns with random
    cut points (port of the null-check ``dec_power_path(s, seq_path(k))``). If ``hold_from`` > width, the
    columns width..hold_from-1 repeat sigma^k (extrapolation columns; disclosed in G-c output)."""
    total = width if hold_from is None else hold_from
    out = np.empty((len(sigs), total, 10), dtype=np.int64)
    for i, s in enumerate(sigs):
        b = _path_cuts(rng, k, width)
        d = np.empty((total, 10), dtype=np.int64)
        for t in range(k):
            d[b[t]:b[t + 1]] = s.P[(t + 1) % s.o]
        if total > width:
            d[width:] = s.P[k % s.o]
        out[i] = d + 1
    return out
