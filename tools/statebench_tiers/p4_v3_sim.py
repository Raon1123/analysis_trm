"""EXP-027 v3 pre-pilot simulations C3 (operating characteristics of the verdict codes) and C5 (pilot -> selection
-> confirmation, end to end).  Design: 03_DESIGN_DIALECTIC section 9; 00_DESIGN_LOG section 7.

Status: DRAFT, author_model claude-sonnet-5-5.  No GPU, numpy only.  Not freeze evidence.

Usage (always CPU only):
    CUDA_VISIBLE_DEVICES="" uv run python tools/statebench_tiers/p4_v3_sim.py --part c3 --out <json>
    CUDA_VISIBLE_DEVICES="" uv run python tools/statebench_tiers/p4_v3_sim.py --part c5 --out <json>
    add --smoke for a small-rep run.

Method notes
------------
* The decision logic is the one in exp027_rules_v3 (imported; constants imported, never edited).  200 000 posterior
  draws per decision times ~10^6 simulated decisions is too slow, so the Beta(1 + k, 1 + n - k) posterior is
  evaluated with M = POST_DRAWS draws taken from fixed per-(cell position, n, k) tables (common random numbers),
  vectorised over the unique outcome tuples.  `validate_fast_evaluator` compares the fast codes with
  `judge_theta` / `judge_g_only` (200 000 draws) on random tables; the result is stored in the C3 json.
* The BSE-bootstrap variant (R1) and its G-only version call `judge_bse` / `judge_bse_g_only` from the module
  directly, with n_boot reduced from 10 000 to BSE_BOOT (stated in the json).
* All random numbers come from numpy Generators seeded by [SIM_SEED, part, row, n]; no wall-clock value is stored.
  All statistics are float64.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp027_rules_v3 as R      # noqa: E402
import exp027_select as S        # noqa: E402

SIM_SEED = 20261113
POST_DRAWS = 10000               # posterior draws per decision in the fast evaluator (CRN tables)
BSE_BOOT = 1000                  # bootstrap resamples used in the simulation (module default is 10 000)
NS = (10, 15, 20, 25, 30, 40)
CPS = (0.975, 0.95)
CODES4 = ("UNINF", "POS", "REV", "NEG", "INC")
CODESG = ("UNINF-G", "POS-G", "REV-G", "NEG-G", "INC-G")


def _f(x) -> float:
    return float(round(float(x), 6))


# ======================================================================= fast posterior evaluator
class PostTables:
    """Per-(position, n) tables of Beta(1 + k, 1 + n - k) draws, k = 0..n, shape (n + 1, M), float64."""

    def __init__(self, m: int = POST_DRAWS, seed: int = SIM_SEED):
        self.m, self.seed, self._t = m, seed, {}

    def table(self, pos: int, n: int) -> np.ndarray:
        key = (pos, n)
        if key not in self._t:
            g = np.random.default_rng([self.seed, 77, pos, n])
            arr = np.empty((n + 1, self.m), dtype=np.float64)
            for k in range(n + 1):
                arr[k] = g.beta(1.0 + k, 1.0 + (n - k), size=self.m)
            self._t[key] = arr
        return self._t[key]


TABLES = PostTables()


def _chunks(t: int, size: int = 128):
    for i in range(0, t, size):
        yield slice(i, min(t, i + size))


def post_probs4(K: np.ndarray, n: int, tables: PostTables = TABLES) -> np.ndarray:
    """K: (T, 4) successes in cell order (Sz, Sn, Gz, Gn), n runs per cell.
    Returns (T, 4): P(theta>0), P(theta<0), P(theta<margin), P(pGz>pGn)."""
    tb = [tables.table(p, n) for p in range(4)]
    out = np.empty((K.shape[0], 4), dtype=np.float64)
    for sl in _chunks(K.shape[0]):
        k = K[sl]
        sz, sn, gz, gn = (tb[i][k[:, i]] for i in range(4))
        th = (gz - gn) - (sz - sn)
        out[sl, 0] = (th > 0.0).mean(axis=1)
        out[sl, 1] = (th < 0.0).mean(axis=1)
        out[sl, 2] = (th < R.MARGIN).mean(axis=1)
        out[sl, 3] = (gz > gn).mean(axis=1)
    return out


def post_probsG(K: np.ndarray, n: int, tables: PostTables = TABLES) -> np.ndarray:
    """K: (T, 2) successes (Gz, Gn).  Returns (T, 3): P(B>0), P(B<0), P(B<margin)."""
    tz, tn = tables.table(2, n), tables.table(3, n)
    out = np.empty((K.shape[0], 3), dtype=np.float64)
    for sl in _chunks(K.shape[0], 256):
        b = tz[K[sl, 0]] - tn[K[sl, 1]]
        out[sl, 0] = (b > 0.0).mean(axis=1)
        out[sl, 1] = (b < 0.0).mean(axis=1)
        out[sl, 2] = (b < R.MARGIN).mean(axis=1)
    return out


def _guard(k_pool: np.ndarray, n_pool: int) -> np.ndarray:
    r = k_pool / n_pool                      # float64 true division, same expression as the module
    return (r >= R.GUARD_LO) & (r <= R.GUARD_HI)


def codes4_from_probs(P: np.ndarray, K: np.ndarray, n: int, c_P: float, c_N: float = R.C_N) -> np.ndarray:
    """Index into CODES4 for each row (first match: UNINF > POS > REV > NEG > INC)."""
    ok = _guard(K[:, 0] + K[:, 1], 2 * n) & _guard(K[:, 2] + K[:, 3], 2 * n)
    code = np.full(K.shape[0], 4, dtype=np.int8)
    code[P[:, 2] >= c_N] = 3
    code[P[:, 1] >= c_P] = 2
    code[(P[:, 0] >= c_P) & (P[:, 3] >= c_P)] = 1
    code[~ok] = 0
    return code


def codesG_from_probs(P: np.ndarray, K: np.ndarray, n: int, c_P: float, c_N: float = R.C_N) -> np.ndarray:
    ok = _guard(K[:, 0] + K[:, 1], 2 * n)
    code = np.full(K.shape[0], 4, dtype=np.int8)
    code[P[:, 2] >= c_N] = 3
    code[P[:, 1] >= c_P] = 2
    code[P[:, 0] >= c_P] = 1
    code[~ok] = 0
    return code


def _unique_rows(K: np.ndarray):
    u, inv = np.unique(K, axis=0, return_inverse=True)
    return u, np.asarray(inv).reshape(-1)


def code_dist(K: np.ndarray, n: int, c_Ps=CPS, g_only: bool = False) -> dict:
    """K: (reps, 4) or (reps, 2) outcome tuples -> {c_P: {code: fraction}} (probabilities computed once per unique tuple)."""
    u, inv = _unique_rows(K)
    P = post_probsG(u, n) if g_only else post_probs4(u, n)
    names = CODESG if g_only else CODES4
    out = {}
    for c_P in c_Ps:
        cu = (codesG_from_probs if g_only else codes4_from_probs)(P, u, n, c_P)
        c = cu[inv]
        out[c_P] = {names[i]: _f(np.mean(c == i)) for i in range(5)}
    return out


def sample4(p, n: int, reps: int, g) -> np.ndarray:
    return g.binomial(n, np.asarray(p, dtype=np.float64), size=(reps, len(p)))


# ======================================================================= validation of the fast evaluator
def validate_fast_evaluator(n_tables: int = 120) -> dict:
    g = np.random.default_rng([SIM_SEED, 1, 0])
    max_dp, mism, near = 0.0, 0, 0
    mism_info = []
    for i in range(n_tables):
        n = int(g.choice(NS))
        k = g.integers(0, n + 1, size=4)
        cells = {c: (int(k[j]), n) for j, c in enumerate(R.CELL_KEYS)}
        for c_P in CPS:
            ref = R.judge_theta(cells, c_P=c_P)
            P = post_probs4(k[None, :], n)
            fast = CODES4[int(codes4_from_probs(P, k[None, :], n, c_P)[0])]
            rp = ref.probs
            if c_P == CPS[0]:
                d = max(abs(P[0, 0] - rp["P_theta_gt0"]), abs(P[0, 1] - rp["P_theta_lt0"]),
                        abs(P[0, 2] - rp["P_theta_lt_margin"]), abs(P[0, 3] - rp["P_G_gt"]))
                max_dp = max(max_dp, d)
            if fast != ref.code:
                mism += 1
                mism_info.append({"cells": {a: list(b) for a, b in cells.items()}, "c_P": c_P, "fast": fast,
                                  "module": ref.code,
                                  "module_probs": {a: _f(b) for a, b in rp.items() if a.startswith("P_")}})
        # G-only
        kg = k[2:]
        cg = {"Gz": (int(kg[0]), n), "Gn": (int(kg[1]), n)}
        refg = R.judge_g_only(cg)
        Pg = post_probsG(kg[None, :], n)
        fastg = CODESG[int(codesG_from_probs(Pg, kg[None, :], n, R.C_P)[0])]
        if fastg != refg.code:
            mism += 1
            mism_info.append({"cells": {a: list(b) for a, b in cg.items()}, "fast": fastg, "module": refg.code})
    return {"n_tables": n_tables, "n_decisions_compared": n_tables * 3, "post_draws_fast": POST_DRAWS,
            "post_draws_module": R.N_DRAWS, "max_abs_prob_diff_4cell": _f(max_dp), "code_mismatches": mism,
            "mismatch_detail": mism_info[:10]}


# ======================================================================= C3 rows
def null_grid() -> list:
    rows = []
    for pSn, pGn, d in itertools.product([0.1, 0.3, 0.5, 0.7], [0.1, 0.3, 0.5, 0.7], [0.0, 0.2, 0.4]):
        if pSn + d <= 0.95 and pGn + d <= 0.95:
            rows.append({"name": f"null_pSn{pSn}_pGn{pGn}_d{d}", "p": (round(pSn + d, 3), pSn, round(pGn + d, 3), pGn),
                         "theta": 0.0})
    return rows


def alt_rows() -> list:
    # p order: (pSz, pSn, pGz, pGn).  theta = (pGz - pGn) - (pSz - pSn)
    def th(t):
        return (0.5, 0.5, 0.5 + t / 2, 0.5 - t / 2)
    rows = [
        {"name": "theta0.3", "p": th(0.3), "kind": "alt"},
        {"name": "theta0.4", "p": th(0.4), "kind": "alt"},
        {"name": "theta0.6", "p": th(0.6), "kind": "alt"},
        {"name": "theta0.8", "p": th(0.8), "kind": "alt"},
        {"name": "reverse_theta-0.6", "p": (0.8, 0.2, 0.5, 0.5), "kind": "rev"},
        {"name": "z_hurts_S", "p": (0.2, 0.8, 0.5, 0.5), "kind": "zhurts"},
        {"name": "logit1.5_matched", "p": (0.7, 0.35, 0.7, 0.35), "kind": "logit"},
        {"name": "logit1.5_unmatched", "p": (0.913, 0.7, 0.658, 0.3), "kind": "logit"},
        {"name": "nearceil_null_S", "p": (0.95, 0.75, 0.6, 0.4), "kind": "nearceil"},
        {"name": "nearceil_alt0.6_S", "p": (0.95, 0.85, 0.8, 0.2), "kind": "nearceil"},
        {"name": "uninf_S_ceiling", "p": (1.0, 0.95, 0.8, 0.2), "kind": "nearceil"},
        {"name": "uninf_G_floor", "p": (0.5, 0.5, 0.08, 0.02), "kind": "nearceil"},
        {"name": "pool_low_null", "p": (0.3, 0.1, 0.3, 0.1), "kind": "nearceil"},
        {"name": "pool_high_null", "p": (0.9, 0.7, 0.9, 0.7), "kind": "nearceil"},
    ]
    for r in rows:
        p = r["p"]
        r["theta"] = _f((p[2] - p[3]) - (p[0] - p[1]))
    return rows


def run_c3_theta(reps: int) -> dict:
    out = {"rows": {}, "worst_null": {}}
    allrows = [dict(r, kind="null") for r in null_grid()] + alt_rows()
    names = [r["name"] for r in allrows]
    for n in NS:
        for ri, r in enumerate(allrows):
            g = np.random.default_rng([SIM_SEED, 3, ri, n])
            K = sample4(r["p"], n, reps, g)
            cd = code_dist(K, n)
            out["rows"].setdefault(r["name"], {"p": list(r["p"]), "theta": r["theta"], "kind": r["kind"], "by_n": {}})
            out["rows"][r["name"]]["by_n"][str(n)] = {str(c): cd[c] for c in CPS}
    for c_P in CPS:
        w = {}
        for n in NS:
            best_pos = max(((out["rows"][nm]["by_n"][str(n)][str(c_P)]["POS"], nm) for nm in names
                            if out["rows"][nm]["kind"] == "null"))
            best_rev = max(((out["rows"][nm]["by_n"][str(n)][str(c_P)]["REV"], nm) for nm in names
                            if out["rows"][nm]["kind"] == "null"))
            w[str(n)] = {"max_P_POS": best_pos[0], "row_POS": best_pos[1],
                         "max_P_REV": best_rev[0], "row_REV": best_rev[1]}
        out["worst_null"][str(c_P)] = w
    # pre-stated rule: 0.95 chosen only if every null row has P(POS) <= 0.05 and P(REV) <= 0.05 at every n
    ok95 = all(v["max_P_POS"] <= 0.05 and v["max_P_REV"] <= 0.05 for v in out["worst_null"]["0.95"].values())
    out["rule"] = {"statement": "c_P = 0.95 only if every null row has P(POS) <= 0.05 and P(REV) <= 0.05 at every n in "
                                "the table under 0.95; otherwise 0.975",
                   "null_rows_all_pass_at_0.95": bool(ok95), "selected_c_P": 0.95 if ok95 else 0.975,
                   "n_null_rows": len(null_grid()), "reps_per_row": reps}
    # re-estimate the near-0.05 null rows at c_P = 0.95 with 10x the replications (MC noise context for the rule)
    big = reps * 10
    rech = {}
    for n in NS:
        for ri, r in enumerate(allrows):
            if r["kind"] != "null":
                continue
            d0 = out["rows"][r["name"]]["by_n"][str(n)]["0.95"]
            if max(d0["POS"], d0["REV"]) >= 0.045:
                g = np.random.default_rng([SIM_SEED, 31, ri, n])
                cd = code_dist(sample4(r["p"], n, big, g), n, c_Ps=(0.95,))[0.95]
                rech[f"{r['name']}|n{n}"] = {"P_POS": cd["POS"], "P_REV": cd["REV"], "reps": big}
    out["recheck_c_P_0.95_rows_ge_0.045"] = rech
    out["recheck_max_P_POS"] = max((v["P_POS"] for v in rech.values()), default=None)
    out["recheck_max_P_REV"] = max((v["P_REV"] for v in rech.values()), default=None)
    # how close is the closest violation (MC noise context)
    out["rule"]["max_over_n_rows_at_0.95"] = {
        "P_POS": max(v["max_P_POS"] for v in out["worst_null"]["0.95"].values()),
        "P_REV": max(v["max_P_REV"] for v in out["worst_null"]["0.95"].values())}
    out["rule"]["max_over_n_rows_at_0.975"] = {
        "P_POS": max(v["max_P_POS"] for v in out["worst_null"]["0.975"].values()),
        "P_REV": max(v["max_P_REV"] for v in out["worst_null"]["0.975"].values())}
    return out


# ======================================================================= C3: G-only (R5)
def run_c3_gonly(reps: int) -> dict:
    rows = []
    for pGn in (0.1, 0.3, 0.5, 0.7, 0.9):
        rows.append({"name": f"null_BG0_pGn{pGn}", "p": (pGn, pGn), "BG": 0.0, "kind": "null"})
    rows += [{"name": "BG0.3", "p": (0.65, 0.35), "BG": 0.3, "kind": "alt"},
             {"name": "BG0.4", "p": (0.7, 0.3), "BG": 0.4, "kind": "alt"},
             {"name": "BG0.6", "p": (0.8, 0.2), "BG": 0.6, "kind": "alt"},
             {"name": "BG-0.6", "p": (0.2, 0.8), "BG": -0.6, "kind": "rev"},
             {"name": "G_floor", "p": (0.08, 0.02), "BG": 0.06, "kind": "guard"}]
    out = {"rows": {}, "worst_null": {}}
    for n in NS:
        for ri, r in enumerate(rows):
            g = np.random.default_rng([SIM_SEED, 4, ri, n])
            K = sample4(r["p"], n, reps, g)
            cd = code_dist(K, n, g_only=True)
            out["rows"].setdefault(r["name"], {"p": list(r["p"]), "BG": r["BG"], "kind": r["kind"], "by_n": {}})
            out["rows"][r["name"]]["by_n"][str(n)] = {str(c): cd[c] for c in CPS}
    for c_P in CPS:
        out["worst_null"][str(c_P)] = {
            str(n): {"max_P_POS-G": max(out["rows"][r["name"]]["by_n"][str(n)][str(c_P)]["POS-G"]
                                        for r in rows if r["kind"] == "null"),
                     "max_P_REV-G": max(out["rows"][r["name"]]["by_n"][str(n)][str(c_P)]["REV-G"]
                                        for r in rows if r["kind"] == "null")} for n in NS}
    return out


# ======================================================================= C3: BSE bootstrap (R1)
def _bse_runs(mean: float, kappa: float, n: int, g) -> np.ndarray:
    a, b = mean * kappa, (1.0 - mean) * kappa
    return g.beta(a, b, size=n)


def run_c3_bse(reps: int, ns=(10, 20, 30)) -> dict:
    """Null: BSE mean of z = mean of noz + d in both tiers (theta_BSE = 0).  Run BSE ~ Beta(m*kappa, (1-m)*kappa);
    kappa 2 is strongly bimodal/graded, kappa 6 is unimodal-broad."""
    rows = []
    for kappa in (2.0, 6.0):
        for mSn, mGn, d in itertools.product((0.3, 0.5, 0.7), (0.3, 0.5, 0.7), (0.0, 0.2)):
            rows.append({"name": f"null_k{kappa:g}_mSn{mSn}_mGn{mGn}_d{d}", "kappa": kappa,
                         "m": (mSn + d, mSn, mGn + d, mGn), "theta": 0.0, "kind": "null", "d": d})
    for kappa in (2.0, 6.0):
        rows.append({"name": f"alt_k{kappa:g}_theta0.6", "kappa": kappa, "m": (0.5, 0.5, 0.8, 0.2), "theta": 0.6,
                     "kind": "alt"})
        rows.append({"name": f"rev_k{kappa:g}_theta-0.6", "kappa": kappa, "m": (0.8, 0.2, 0.5, 0.5), "theta": -0.6,
                     "kind": "rev"})
    out = {"rows": {}, "n_boot": BSE_BOOT, "kappa_note": "run BSE ~ Beta(m*kappa,(1-m)*kappa)"}
    for n in ns:
        for ri, r in enumerate(rows):
            g = np.random.default_rng([SIM_SEED, 5, ri, n])
            cnt4, cntG = {c: 0 for c in CODES4}, {c: 0 for c in CODESG}
            for _ in range(reps):
                runs = {c: _bse_runs(r["m"][j], r["kappa"], n, g) for j, c in enumerate(R.CELL_KEYS)}
                cnt4[R.judge_bse(runs, n_boot=BSE_BOOT, rng=g).code] += 1
                cntG[R.judge_bse_g_only({"Gz": runs["Gz"], "Gn": runs["Gn"]}, n_boot=BSE_BOOT, rng=g).code] += 1
            out["rows"].setdefault(r["name"], {"m": list(r["m"]), "kappa": r["kappa"], "theta": r["theta"],
                                                "kind": r["kind"], "d": r.get("d"), "by_n": {}})
            out["rows"][r["name"]]["by_n"][str(n)] = {"4cell": {c: _f(v / reps) for c, v in cnt4.items()},
                                                      "G-only": {c: _f(v / reps) for c, v in cntG.items()}}
    wn = {}
    for n in ns:
        nr = [v for v in out["rows"].values() if v["kind"] == "null"]
        n0 = [v for v in nr if v["d"] == 0.0]      # G-only null is B_G = 0, i.e. d = 0 rows only
        wn[str(n)] = {"max_P_POS_4cell": max(v["by_n"][str(n)]["4cell"]["POS"] for v in nr),
                      "max_P_REV_4cell": max(v["by_n"][str(n)]["4cell"]["REV"] for v in nr),
                      "max_P_POS_Gonly_d0": max(v["by_n"][str(n)]["G-only"]["POS-G"] for v in n0),
                      "max_P_REV_Gonly_d0": max(v["by_n"][str(n)]["G-only"]["REV-G"] for v in n0)}
    out["worst_null"] = wn
    return out


# ======================================================================= C3: R6 log2 first-passage (port of ell_sim.py)
TMAX, EV = 40039, 977
SD_LOG2 = 0.45


def _ell_draw(g, n, q, mu, sd):
    learn = g.random(n) >= q
    lt = g.normal(mu, sd, n)
    tau = np.ceil(2 ** lt / EV) * EV
    tau = np.where(learn & (tau <= TMAX), tau, np.inf)
    return np.log2(np.minimum(tau, TMAX)), tau


def _boot_mean(g, x, B):
    return x[g.integers(0, len(x), (B, len(x)))].mean(1)


def _ell_rule(g, cells, m=0.5, B=BSE_BOOT):
    for T in "SG":
        tau = np.concatenate([cells[(T, "z")][1], cells[(T, "n")][1]])
        if np.isfinite(tau).mean() < 0.25 or np.median(tau) < 3 * EV:
            return "UNINF"
    bm = {k: _boot_mean(g, v[0], B) for k, v in cells.items()}
    BG = bm[("G", "n")] - bm[("G", "z")]
    BS = bm[("S", "n")] - bm[("S", "z")]
    D = BG - BS
    lo, hi = np.quantile(D, [0.025, 0.975])
    loG = np.quantile(BG, 0.025)
    if lo > 0 and loG > 0:
        return "POS"
    if hi < 0:
        return "REV"
    if hi < m:
        return "NEG"
    return "INC"


def _ell_P(muS, muG, bS, bG, qSn=0.0, qSz=0.0, qGn=0.0, qGz=0.0):
    sd = SD_LOG2
    return {("S", "n"): (qSn, muS, sd), ("S", "z"): (qSz, muS - bS, sd), ("G", "n"): (qGn, muG, sd),
            ("G", "z"): (qGz, muG - bG, sd)}


ELL_SCEN = {
    "null_b0_same": (_ell_P(14.0, 14.0, 0, 0), "null"),
    "null_b1_same": (_ell_P(14.0, 14.0, 1, 1), "null"),
    "null_b1_Seasy": (_ell_P(12.5, 14.5, 1, 1), "null"),
    "null_b1_censorG": (_ell_P(13.0, 15.0, 1, 1, qGn=0.3, qGz=0.3), "null"),
    "null_Sfast_b0.5": (_ell_P(11.5, 14.0, .5, .5), "null"),
    "null_censor_both_tiers": (_ell_P(14.0, 14.5, 1, 1, qSn=0.15, qSz=0.15, qGn=0.15, qGz=0.15), "null"),
    "null_Gfloor": (_ell_P(13.0, 15.5, 1, 1, qGn=0.8, qGz=0.8), "null"),
    "alt_D0.5": (_ell_P(14.0, 14.0, 0.5, 1.0), "alt"),
    "alt_D1.0": (_ell_P(14.0, 14.0, 0.5, 1.5), "alt"),
    "rev_D-1.0": (_ell_P(14.0, 14.0, 1.5, 0.5), "rev"),
    "zhurtsS_D+1": (_ell_P(14.0, 14.0, -1.0, 0.0), "zhurts"),
}


def run_c3_r6(reps: int, ns=(10, 20, 30)) -> dict:
    out = {"scenarios": {}, "note": "port of mod/ell_sim.py; sd(log2 tau)=0.45, T_max=40039, eval step 977, "
                                    f"bootstrap B={BSE_BOOT}, restricted log2 first-passage, censored at T_max"}
    for si, (name, (P, kind)) in enumerate(ELL_SCEN.items()):
        g0 = np.random.default_rng([SIM_SEED, 6, si, 0])
        # expected median pooled tau per tier (large sample) to evaluate the pilot gate 'median tau >= 8 eval'
        med = {}
        for T in "SG":
            big = np.concatenate([_ell_draw(g0, 20000, *P[(T, v)])[1] for v in "zn"])
            med[T] = float(np.median(big)) / EV if np.isfinite(np.median(big)) else float("inf")
        out["scenarios"][name] = {"kind": kind, "median_tau_eval_intervals": {t: _f(v) if np.isfinite(v) else "inf"
                                                                              for t, v in med.items()},
                                  "meets_pilot_gate_median_tau_ge_8": bool(min(med.values()) >= 8), "by_n": {}}
        for n in ns:
            g = np.random.default_rng([SIM_SEED, 6, si, n])
            cnt = {}
            for _ in range(reps):
                cells = {k: _ell_draw(g, n, *v) for k, v in P.items()}
                c = _ell_rule(g, cells)
                cnt[c] = cnt.get(c, 0) + 1
            out["scenarios"][name]["by_n"][str(n)] = {c: _f(cnt.get(c, 0) / reps) for c in
                                                      ("POS", "REV", "NEG", "INC", "UNINF")}
    nulls = {k: v for k, v in out["scenarios"].items() if v["kind"] == "null"}
    gated = {k: v for k, v in nulls.items() if v["meets_pilot_gate_median_tau_ge_8"]}

    def worst(d):
        return {str(n): {"max_P_POS": max(v["by_n"][str(n)]["POS"] for v in d.values()),
                         "max_P_REV": max(v["by_n"][str(n)]["REV"] for v in d.values())} for n in ns} if d else {}
    out["worst_null_all"] = worst(nulls)
    out["worst_null_meeting_pilot_gate"] = worst(gated)
    out["scenarios_meeting_pilot_gate"] = sorted(gated)
    ok = all(v["max_P_POS"] <= 0.05 and v["max_P_REV"] <= 0.05 for v in out["worst_null_all"].values())
    ok_g = bool(gated) and all(v["max_P_POS"] <= 0.05 and v["max_P_REV"] <= 0.05
                               for v in out["worst_null_meeting_pilot_gate"].values())
    out["gate_opens_all_null_scenarios"] = bool(ok)
    out["gate_opens_if_restricted_to_scenarios_meeting_pilot_median_tau_gate"] = bool(ok_g)
    out["verdict"] = ("R6 exception gate can open" if ok else "R6 exception gate does NOT open on the full null set; R6 stays "
                      "closed (default = no confirmation)")
    return out


# ======================================================================= C5: pilot -> selection -> confirmation
def expit(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    return math.log(p / (1.0 - p))


BOUNDARY = {"early": 3.6, "mid": 4.5, "late": 5.6}      # log2 N at which the noz success probability is 0.5
SHAPES = {"sharp": 0.15, "smooth": 0.50}                # logistic width in log2 N units
GAP_CFG = {  # (gap_S, gap_G): peak probability-scale z advantage over N
    "g0_0": (0.0, 0.0), "gS0_gG0.3": (0.0, 0.3), "gS0_gG0.6": (0.0, 0.6),
    "gS0.3_gG0.3": (0.3, 0.3), "gS0.6_gG0.6": (0.6, 0.6), "gS0.6_gG0": (0.6, 0.0)}


class Scenario:
    def __init__(self, name, bS, bG, w, gapS, gapG, pipe_broken=False):
        self.name, self.bS, self.bG, self.w, self.gapS, self.gapG = name, bS, bG, w, gapS, gapG
        self.pipe_broken = pipe_broken

    def p(self, tier: str, N: int, variant: str) -> float:
        b = self.bS if tier == "S" else self.bG
        gap = self.gapS if tier == "S" else self.gapG
        if variant == "z" and gap > 0.0:
            b = b + self.w * 4.0 * math.atanh(gap)
        v = float(expit(-(math.log2(N) - b) / self.w))
        return min(max(v, 0.0), 1.0)

    def pbar(self, tier, N):
        return 0.5 * (self.p(tier, N, "z") + self.p(tier, N, "noz"))

    def theta(self, NS, NG):
        return (self.p("G", NG, "z") - self.p("G", NG, "noz")) - (self.p("S", NS, "z") - self.p("S", NS, "noz"))


def scenario_grid() -> list:
    sc = []
    for (sh, w), (bs_n, bS), (bg_n, bG), (gn, (gS, gG)) in itertools.product(
            SHAPES.items(), BOUNDARY.items(), BOUNDARY.items(), GAP_CFG.items()):
        sc.append(Scenario(f"{sh}|S{bs_n}|G{bg_n}|{gn}", bS, bG, w, gS, gG))
    # fixed structural scenarios: all ceiling (R6 default), all floor (R7), pipeline broken (R0)
    sc.append(Scenario("allceil|g0_0", 9.0, 9.0, 0.15, 0.0, 0.0))
    sc.append(Scenario("allfloor|g0_0", 0.0, 0.0, 0.15, 0.0, 0.0))
    sc.append(Scenario("pipe_broken|g0_0", 0.0, 0.0, 0.15, 0.0, 0.0, pipe_broken=True))
    return sc


def _row(tier, N, variant, seed, y, stage):
    return {"tier": tier, "N": N, "variant": variant, "seed": seed, "Y": int(y), "BSE": 0.97 if y else 0.05,
            "tau": 1.0 if y else None, "flags": (), "max": 1.0 if y else 0.05, "stage": stage}


def pilot_rows(sc: Scenario, g, s_seeds=S.PILOT_SEEDS) -> list:
    rows = []
    for N in S.PILOT_NS:
        for v in ("z", "noz"):
            for s in S.PILOT_SEEDS:
                rows.append(_row("G", N, v, s, g.random() < sc.p("G", N, v), "pilot"))
            for s in s_seeds:
                rows.append(_row("S", N, v, s, g.random() < sc.p("S", N, v), "pilot"))
    rows.append(_row("A", 16, "z", 9101, False if sc.pipe_broken else True, "pilot"))
    return rows


def ext_rows(sc: Scenario, ext, g) -> list:
    rows = []
    for tier, N, runs in ext:
        for v, s in runs:
            rows.append(_row(tier, N, v, s, g.random() < sc.p(tier, N, v), "ext"))
    return rows


def simulate_selection(sc: Scenario, g, s_seeds=S.PILOT_SEEDS):
    rows = pilot_rows(sc, g, s_seeds)
    res = S.select_rows(rows)
    ext_req = None
    if res.path == "R2":
        ext_req = res.extension
        rows = rows + ext_rows(sc, res.extension, g)
        res = S.select_rows(rows)
    return res, ext_req


def label_path(res) -> str:
    return res.path if res.sub is None else f"{res.path}:{res.sub}"


def run_c5_scenario(sc: Scenario, si: int, reps: int, n_conf: int, s_seeds=S.PILOT_SEEDS, confirm=True) -> dict:
    g = np.random.default_rng([SIM_SEED, 7, si, n_conf, len(s_seeds)])
    paths, ext_n, ext_tiers = {}, 0, {}
    conf4, confG = [], []
    theta_sel, split_n = [], 0
    s_informative = any(0.2 <= sc.pbar("S", N) <= 0.8 for N in S.PILOT_NS)
    g_informative = any(0.2 <= sc.pbar("G", N) <= 0.8 for N in S.PILOT_NS)
    for _ in range(reps):
        res, ext = simulate_selection(sc, g, s_seeds)
        lp = label_path(res)
        paths[lp] = paths.get(lp, 0) + 1
        if ext is not None:
            ext_n += 1
            key = f"{ext[0][0]}{ext[0][1]}"
            ext_tiers[key] = ext_tiers.get(key, 0) + 1
        if confirm and res.confirm:
            if res.stat == "theta":
                p = [sc.p("S", res.N_S, "z"), sc.p("S", res.N_S, "noz"), sc.p("G", res.N_G, "z"), sc.p("G", res.N_G, "noz")]
                conf4.append(g.binomial(n_conf, p))
                theta_sel.append(sc.theta(res.N_S, res.N_G))
                split_n += int(res.N_S != res.N_G)
            elif res.stat == "G-only":
                p = [sc.p("G", res.N_G, "z"), sc.p("G", res.N_G, "noz")]
                confG.append(g.binomial(n_conf, p))
    out = {"scenario": sc.name, "reps": reps, "n_conf": n_conf, "pilot_S_seeds": len(s_seeds),
           "paths": {k: _f(v / reps) for k, v in sorted(paths.items())},
           "P_extension": _f(ext_n / reps), "extension_options": {k: _f(v / reps) for k, v in sorted(ext_tiers.items())},
           "S_informative_at_pilot_N": bool(s_informative), "G_informative_at_pilot_N": bool(g_informative),
           "P_confirm": _f((len(conf4) + len(confG)) / reps)}
    if confirm:
        final = {}
        if conf4:
            K = np.asarray(conf4)
            u, inv = _unique_rows(K)
            P = post_probs4(u, n_conf)
            cu = codes4_from_probs(P, u, n_conf, R.C_P)[inv]
            for i, nm in enumerate(CODES4):
                final[nm] = int(np.sum(cu == i))
            out["mean_true_theta_at_selected_cells"] = _f(np.mean(theta_sel))
            out["sd_true_theta_at_selected_cells"] = _f(np.std(theta_sel))
            out["P_split_N_given_theta_path"] = _f(split_n / len(conf4))
        if confG:
            K = np.asarray(confG)
            u, inv = _unique_rows(K)
            P = post_probsG(u, n_conf)
            cu = codesG_from_probs(P, u, n_conf, R.C_P)[inv]
            for i, nm in enumerate(CODESG):
                final[nm] = int(np.sum(cu == i))
        noconf = reps - len(conf4) - len(confG)
        final["NOCONF"] = noconf
        out["final_counts"] = final
        out["final"] = {k: _f(v / reps) for k, v in final.items() if v}
        n_c = len(conf4) + len(confG)
        out["P_UNINF_given_confirm"] = _f((final.get("UNINF", 0) + final.get("UNINF-G", 0)) / n_c) if n_c else None
    return out


def run_c5(reps: int, reps_pilot_only: int, extra_ns=(15, 30)) -> dict:
    scs = scenario_grid()
    out = {"scenarios": {}, "reps": reps, "reps_pilot_only": reps_pilot_only,
           "scenario_model": {"p(tier,N,variant)": "expit(-(log2 N - b_tier - shift_variant)/w)",
                              "boundary_log2N": BOUNDARY, "width_w": SHAPES, "gap_configs": GAP_CFG,
                              "z_shift": "z boundary moved by 4*w*atanh(gap) so the peak (over N) z-noz success-probability gap equals `gap`",
                              "run_outcome": "Y ~ Bernoulli(p) per run (graded/pretakeoff flags not simulated: R1 never fires here; "
                                             "Y=0 runs have max 0.05)",
                              "A_run": "Y=1 except in pipe_broken"}}
    for si, sc in enumerate(scs):
        out["scenarios"][sc.name] = run_c5_scenario(sc, si, reps, 20)
    # 2-seed vs 1-seed S pilot, path-only (no confirmation)
    cmp = {}
    for si, sc in enumerate(scs):
        r2 = run_c5_scenario(sc, si, reps_pilot_only, 20, S.PILOT_SEEDS, confirm=False)
        r1 = run_c5_scenario(sc, si, reps_pilot_only, 20, (9101,), confirm=False)
        cmp[sc.name] = {"S_informative": r2["S_informative_at_pilot_N"], "seeds2": r2["paths"], "seeds1": r1["paths"],
                        "P_ext_seeds2": r2["P_extension"], "P_ext_seeds1": r1["P_extension"]}
    out["seed_comparison"] = cmp
    # other confirmatory n for the structural subsets
    sub = [(si, sc) for si, sc in enumerate(scs) if sc.name.split("|")[-1] in ("g0_0", "gS0_gG0.6", "gS0.6_gG0")]
    out["by_n_conf"] = {}
    for n_c in extra_ns:
        out["by_n_conf"][str(n_c)] = {sc.name: run_c5_scenario(sc, si, reps, n_c) for si, sc in sub}
    return out


# ======================================================================= main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["c3", "c5"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    meta = {"sim_seed": SIM_SEED, "post_draws_fast": POST_DRAWS, "module_constants": {
        "C_P_default": R.C_P, "C_N": R.C_N, "MARGIN": R.MARGIN, "GUARD": [R.GUARD_LO, R.GUARD_HI],
        "BSE_NEG_MARGIN": R.BSE_NEG_MARGIN}, "float": "float64", "status": "DRAFT", "author_model": "claude-sonnet-5-5"}
    if a.part == "c3":
        reps = 300 if a.smoke else 4000
        res = {"meta": dict(meta, reps=reps, ns=list(NS), c_P_candidates=list(CPS),
                            mc_se_at_rate_0p05=_f(math.sqrt(0.05 * 0.95 / reps))),
               "validation": validate_fast_evaluator(20 if a.smoke else 120),
               "theta_codes": run_c3_theta(reps), "g_only": run_c3_gonly(reps),
               "bse_bootstrap": run_c3_bse(100 if a.smoke else 1000),
               "r6_log2_first_passage": run_c3_r6(100 if a.smoke else 1000)}
    else:
        reps = 100 if a.smoke else 4000
        res = {"meta": dict(meta, reps=reps, mc_se_max=_f(math.sqrt(0.25 / reps))),
               "c5": run_c5(reps, 100 if a.smoke else 4000)}
    Path(a.out).write_text(json.dumps(res, indent=1, sort_keys=True, default=str) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
