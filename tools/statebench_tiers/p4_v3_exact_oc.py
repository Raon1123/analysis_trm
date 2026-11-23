"""EXP-027 C3 cross-check: exact operating characteristics of the 4-cell theta code (exp027_rules_v3.judge_theta).

Status: DRAFT, author_model claude-opus-5-5 (moderator review 2026-11-11).  No GPU, numpy only, no randomness.

Every outcome tuple (kSz, kSn, kGz, kGn) is enumerated with its binomial weight (no Monte Carlo over outcomes).
Posterior probabilities P(theta > 0), P(theta < 0.30), P(pGz > pGn) come from midpoint quadrature of the
Beta(1 + k, 1 + n - k) densities on an M = 2000 grid; differences live on the induced lattice, ties half-weighted.
Checked against the module: on 5,239 unique sampled n = 20 tuples the module code (200 000 draws, seed 20261111)
and the quadrature code differ on 1 tuple (mass 0.0003).  The module's own decisions near a threshold can differ
from these ideal-posterior decisions; `mass_within_0.002_of_threshold` reports how much weight is that close.

Usage: CUDA_VISIBLE_DEVICES="" uv run python tools/statebench_tiers/p4_v3_exact_oc.py --out OUT.json [--ns 15 20 ...]
"""
import itertools, json, sys
import numpy as np
from math import comb

M = 2000                     # grid points on [0,1]; margin 0.30 = 600 lattice steps exactly
S_MARGIN = 600
L = 2 * M - 1
NFFT = 4096


def beta_pmfs(n):
    u = (np.arange(M) + 0.5) / M
    P = np.empty((n + 1, M))
    for k in range(n + 1):
        lp = k * np.log(u) + (n - k) * np.log1p(-u)
        p = np.exp(lp - lp.max())
        P[k] = p / p.sum()
    return P


def diff_pmfs(n):
    """pmf of pz - pn on offsets -(M-1)..(M-1) for every (kz, kn); shape ((n+1)^2, L)."""
    P = beta_pmfs(n)
    A = np.fft.rfft(P, NFFT, axis=1)
    B = np.fft.rfft(P[:, ::-1], NFFT, axis=1)
    out = np.empty(((n + 1) ** 2, L))
    i = 0
    for kz in range(n + 1):
        out[i:i + n + 1] = np.fft.irfft(A[kz][None, :] * B, NFFT, axis=1)[:, :L]
        i += n + 1
    out = np.clip(out, 0.0, None)
    out /= out.sum(axis=1, keepdims=True)
    return out          # row index = kz*(n+1) + kn


def tables(n):
    D = diff_pmfs(n)
    C = np.cumsum(D, axis=1)
    Cm1 = np.concatenate([np.zeros((D.shape[0], 1)), C[:, :-1]], axis=1)
    S0 = Cm1 + 0.5 * D                       # P(Y < j) + .5 P(Y = j)
    Tm = 1.0 - C + 0.5 * D                   # P(Y > j) + .5 P(Y = j)
    # P(theta > 0) = sum_a pX[a] S0_Y[a]
    P_gt0 = D @ S0.T                         # rows: G pair (X), cols: S pair (Y)
    # P(theta < m) = sum_a pX[a] T_Y[a - s]; for a - s < 0, T = 1
    Tsh = np.concatenate([np.ones((D.shape[0], S_MARGIN)), Tm[:, :L - S_MARGIN]], axis=1)
    P_ltm = D @ Tsh.T
    zero = M - 1
    P_G = D[:, zero + 1:].sum(axis=1) + 0.5 * D[:, zero]
    return P_gt0, P_ltm, P_G


def oc(p, n, tabs, c_Ps=(0.975, 0.95), c_N=0.95, band=0.002):
    pSz, pSn, pGz, pGn = p
    P_gt0, P_ltm, P_G = tabs
    k = np.arange(n + 1)
    b = lambda q: np.array([comb(n, j) * q ** j * (1 - q) ** (n - j) for j in k])
    wS = np.outer(b(pSz), b(pSn)).ravel()
    wG = np.outer(b(pGz), b(pGn)).ravel()
    kz, kn = np.divmod(np.arange((n + 1) ** 2), n + 1)
    pool = (kz + kn) / (2 * n)
    okS = (pool >= 0.10) & (pool <= 0.90)
    okG = okS.copy()
    W = np.outer(wG, wS)
    ok = np.outer(okG, okS)
    res = {}
    for c_P in c_Ps:
        pos = (P_gt0 >= c_P) & (P_G[:, None] >= c_P)
        rev = (1.0 - P_gt0) >= c_P
        neg = P_ltm >= c_N
        code = np.full(W.shape, 4, dtype=np.int8)
        code[neg] = 3; code[rev] = 2; code[pos] = 1; code[~ok] = 0
        d = {nm: float(W[code == i].sum()) for i, nm in enumerate(("UNINF", "POS", "REV", "NEG", "INC"))}
        near = ok & ((np.abs(P_gt0 - c_P) < band) | (np.abs(1 - P_gt0 - c_P) < band) | (np.abs(P_ltm - c_N) < band)
                     | (np.abs(P_G[:, None] - c_P) < band))
        d["mass_within_%g_of_threshold" % band] = float(W[near].sum())
        res[str(c_P)] = d
    return res


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ns", type=int, nargs="*", default=list(range(10, 31)) + [40])
    a = ap.parse_args()
    rows = {}
    for pSn, pGn, dd in itertools.product([0.1, 0.3, 0.5, 0.7], [0.1, 0.3, 0.5, 0.7], [0.0, 0.2, 0.4]):
        if pSn + dd <= 0.95 and pGn + dd <= 0.95:
            rows[f"null_pSn{pSn}_pGn{pGn}_d{dd}"] = (round(pSn + dd, 3), pSn, round(pGn + dd, 3), pGn)
    rows["theta0.6"] = (0.5, 0.5, 0.8, 0.2)
    rows["reverse_theta-0.6"] = (0.8, 0.2, 0.5, 0.5)
    rows["theta0.3"] = (0.5, 0.5, 0.65, 0.35)
    out = {}
    summary = {}
    for n in a.ns:
        tabs = tables(n)
        out[n] = {nm: oc(p, n, tabs) for nm, p in rows.items()}
        nulls = {k: v for k, v in out[n].items() if k.startswith("null")}
        for c in ("0.975", "0.95"):
            mp = max(nulls.items(), key=lambda kv: kv[1][c]["POS"])
            mr = max(nulls.items(), key=lambda kv: kv[1][c]["REV"])
            print(f"n={n} c_P={c} maxPOS={mp[1][c]['POS']:.5f} ({mp[0]}) maxREV={mr[1][c]['REV']:.5f} ({mr[0]})")
            summary.setdefault(c, {})[str(n)] = {"max_null_POS": mp[1][c]["POS"], "row_POS": mp[0],
                                                 "max_null_REV": mr[1][c]["REV"], "row_REV": mr[0]}
        for nm in ("theta0.6", "reverse_theta-0.6", "theta0.3", "null_pSn0.5_pGn0.5_d0.0", "null_pSn0.5_pGn0.5_d0.4"):
            print("  ", nm, {c: {k: round(v, 5) for k, v in out[n][nm][c].items()} for c in ("0.975",)})
        sys.stdout.flush()
    res = {"meta": {"method": "exact enumeration of outcome tuples; Beta posteriors by midpoint quadrature",
                    "grid_M": M, "margin_lattice_steps": S_MARGIN, "c_N": 0.95, "guard": [0.10, 0.90],
                    "status": "DRAFT", "author_model": "claude-opus-5-5"},
           "worst_null": summary, "rows": {str(n): v for n, v in out.items()}}
    with open(a.out, "w") as f:
        f.write(json.dumps(res, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
