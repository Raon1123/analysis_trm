"""EXP-027 v3 decision codes (design 03_DESIGN_DIALECTIC section 4.3, rule R5 and rule R1 of section 6).

Status: DRAFT, author_model claude-sonnet-5-5.  Not frozen.  No GPU, numpy only.

Cells are (tier X in {S, G}) x (variant v in {z, noz}) and are keyed 'Sz', 'Sn', 'Gz', 'Gn'
(Sn = (S, noz), Gn = (G, noz)).  Each cell carries (k, n): k runs with Y = 1 out of n runs.

Codes (first match wins)
------------------------
4-cell theta code (section 4.3):
    UNINF  pooled Y rate of S or of G (z and noz pooled) outside [GUARD_LO, GUARD_HI]
    POS    P(theta > 0) >= c_P and P(pGz > pGn) >= c_P
    REV    P(theta < 0) >= c_P
    NEG    P(theta < MARGIN) >= c_N
    INC    otherwise
    with theta = (pGz - pGn) - (pSz - pSn) and independent Beta(1 + k, 1 + n - k) posteriors.
2-cell G-only code (rule R5):  UNINF-G > POS-G > REV-G > NEG-G > INC-G on B_G = pGz - pGn.
BSE bootstrap code (rule R1 MISFIT): same names as the 4-cell code, seed-bootstrap 95% interval.

Determinism: every call builds its own numpy Generator from a module-constant seed (unless the caller
passes `seed` or `rng` explicitly), so identical inputs give identical outputs.  All arithmetic is float64.
No wall-clock value enters any result.
"""
from __future__ import annotations

import dataclasses
from typing import Mapping, Sequence

import numpy as np

# ---------------------------------------------------------------- frozen constants
N_DRAWS = 200_000          # posterior draws per cell (simulations may override with n_draws=...)
SEED = 20261111            # posterior rng seed
C_P = 0.975                # positive / reversed posterior-probability threshold (design: tentative, fixed after C3)
C_N = 0.95                 # negative posterior-probability threshold
MARGIN = 0.30              # NEG margin Delta_NEG on theta (and on B_G for rule R5)
GUARD_LO = 0.10            # pooled-Y-rate guard, inclusive
GUARD_HI = 0.90
BOOT_B = 10_000            # seed-bootstrap resamples for the BSE variant
BOOT_SEED = 20261112
BOOT_ALPHA = 0.05          # two-sided 95% percentile interval -> 2.5% / 97.5%
BSE_NEG_MARGIN = 0.20      # section 6 R1: NEG = hi(theta_BSE) < 0.20
BSE_GUARD_TOL = 1e-6       # BSE values are float32 eval values (0.950 -> 0.949999988); guard on their mean is closed
                           # up to this tolerance, so a decimal mean of exactly 0.90 or 0.10 is inside
CELL_KEYS = ("Sz", "Sn", "Gz", "Gn")
CODES_4 = ("UNINF", "POS", "REV", "NEG", "INC")        # evaluation order
CODES_G = ("UNINF-G", "POS-G", "REV-G", "NEG-G", "INC-G")


@dataclasses.dataclass(frozen=True)
class RuleResult:
    code: str
    probs: dict          # posterior probabilities / interval ends (python floats)
    inputs: dict         # echo of the inputs
    constants: dict      # constants actually used

    def to_dict(self) -> dict:
        return {"code": self.code, "probs": dict(self.probs), "inputs": dict(self.inputs),
                "constants": dict(self.constants)}


# ---------------------------------------------------------------- helpers
def _check_cell(name: str, kn) -> tuple[int, int]:
    k, n = kn
    if int(k) != k or int(n) != n:
        raise ValueError(f"cell {name}: k and n must be integers, got {kn!r}")
    k, n = int(k), int(n)
    if n < 0 or k < 0 or k > n:
        raise ValueError(f"cell {name}: need 0 <= k <= n, got k={k}, n={n}")
    return k, n


def _cells(cells: Mapping, keys: Sequence[str]) -> dict:
    missing = [c for c in keys if c not in cells]
    extra = [c for c in cells if c not in keys]
    if missing or extra:
        raise ValueError(f"cells must be exactly {tuple(keys)}; missing={missing}, extra={extra}")
    return {c: _check_cell(c, cells[c]) for c in keys}


def _make_rng(seed: int, rng):
    return rng if rng is not None else np.random.default_rng(seed)


def draw_posteriors(kn: Sequence[tuple[int, int]], n_draws: int, rng) -> np.ndarray:
    """Independent Beta(1 + k, 1 + n - k) draws, shape (len(kn), n_draws), float64, in the given cell order."""
    out = np.empty((len(kn), n_draws), dtype=np.float64)
    for i, (k, n) in enumerate(kn):
        out[i] = rng.beta(1.0 + k, 1.0 + (n - k), size=n_draws)
    return out


def _guard_ok(k_pool: int, n_pool: int, lo: float, hi: float) -> bool:
    if n_pool <= 0:
        raise ValueError("a tier has zero runs; pooled Y rate undefined")
    r = k_pool / n_pool          # correctly rounded float64 division; 1/10 == 0.10 literal, 9/10 == 0.90 literal
    return lo <= r <= hi


# ---------------------------------------------------------------- 4-cell theta code
def judge_theta(cells: Mapping, *, c_P: float = C_P, c_N: float = C_N, margin: float = MARGIN,
                guard: tuple[float, float] = (GUARD_LO, GUARD_HI), n_draws: int = N_DRAWS,
                seed: int = SEED, rng=None) -> RuleResult:
    """cells: {'Sz': (k, n), 'Sn': (k, n), 'Gz': (k, n), 'Gn': (k, n)}."""
    c = _cells(cells, CELL_KEYS)
    kS, nS = c["Sz"][0] + c["Sn"][0], c["Sz"][1] + c["Sn"][1]
    kG, nG = c["Gz"][0] + c["Gn"][0], c["Gz"][1] + c["Gn"][1]
    okS, okG = _guard_ok(kS, nS, *guard), _guard_ok(kG, nG, *guard)
    g = _make_rng(seed, rng)
    post = draw_posteriors([c[x] for x in CELL_KEYS], n_draws, g)
    pSz, pSn, pGz, pGn = post
    theta = (pGz - pGn) - (pSz - pSn)
    probs = {
        "P_theta_gt0": float(np.mean(theta > 0.0)),
        "P_theta_lt0": float(np.mean(theta < 0.0)),
        "P_theta_lt_margin": float(np.mean(theta < margin)),
        "P_G_gt": float(np.mean(pGz > pGn)),
        "pooled_rate_S": kS / nS,
        "pooled_rate_G": kG / nG,
        "theta_mean": float(np.mean(theta)),
    }
    if not (okS and okG):
        code = "UNINF"
    elif probs["P_theta_gt0"] >= c_P and probs["P_G_gt"] >= c_P:
        code = "POS"
    elif probs["P_theta_lt0"] >= c_P:
        code = "REV"
    elif probs["P_theta_lt_margin"] >= c_N:
        code = "NEG"
    else:
        code = "INC"
    return RuleResult(code, probs, {k: list(v) for k, v in c.items()},
                      {"c_P": c_P, "c_N": c_N, "margin": margin, "guard": list(guard),
                       "n_draws": n_draws, "seed": seed if rng is None else None})


# ---------------------------------------------------------------- 2-cell G-only code (rule R5)
def judge_g_only(cells: Mapping, *, c_P: float = C_P, c_N: float = C_N, margin: float = MARGIN,
                 guard: tuple[float, float] = (GUARD_LO, GUARD_HI), n_draws: int = N_DRAWS,
                 seed: int = SEED, rng=None) -> RuleResult:
    """cells: {'Gz': (k, n), 'Gn': (k, n)}.  B_G = pGz - pGn."""
    c = _cells(cells, ("Gz", "Gn"))
    kG, nG = c["Gz"][0] + c["Gn"][0], c["Gz"][1] + c["Gn"][1]
    ok = _guard_ok(kG, nG, *guard)
    g = _make_rng(seed, rng)
    pGz, pGn = draw_posteriors([c["Gz"], c["Gn"]], n_draws, g)
    b = pGz - pGn
    probs = {
        "P_BG_gt0": float(np.mean(b > 0.0)),
        "P_BG_lt0": float(np.mean(b < 0.0)),
        "P_BG_lt_margin": float(np.mean(b < margin)),
        "pooled_rate_G": kG / nG,
        "BG_mean": float(np.mean(b)),
    }
    if not ok:
        code = "UNINF-G"
    elif probs["P_BG_gt0"] >= c_P:
        code = "POS-G"
    elif probs["P_BG_lt0"] >= c_P:
        code = "REV-G"
    elif probs["P_BG_lt_margin"] >= c_N:
        code = "NEG-G"
    else:
        code = "INC-G"
    return RuleResult(code, probs, {k: list(v) for k, v in c.items()},
                      {"c_P": c_P, "c_N": c_N, "margin": margin, "guard": list(guard),
                       "n_draws": n_draws, "seed": seed if rng is None else None})


# ---------------------------------------------------------------- BSE bootstrap variant (rule R1)
def _boot_means(x: np.ndarray, B: int, g) -> np.ndarray:
    idx = g.integers(0, x.size, size=(B, x.size))
    return x[idx].mean(axis=1)


def _interval(v: np.ndarray, alpha: float) -> tuple[float, float]:
    lo, hi = np.quantile(v, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def _bse_arrays(runs: Mapping, keys: Sequence[str]) -> dict:
    missing = [c for c in keys if c not in runs]
    extra = [c for c in runs if c not in keys]
    if missing or extra:
        raise ValueError(f"runs must be exactly {tuple(keys)}; missing={missing}, extra={extra}")
    out = {}
    for c in keys:
        a = np.asarray(runs[c], dtype=np.float64)
        if a.ndim != 1 or a.size < 1 or not np.all(np.isfinite(a)):
            raise ValueError(f"cell {c}: need a non-empty 1-d finite BSE array")
        out[c] = a
    return out


def judge_bse(runs: Mapping, *, margin: float = BSE_NEG_MARGIN,
              guard: tuple[float, float] = (GUARD_LO, GUARD_HI), n_boot: int = BOOT_B,
              alpha: float = BOOT_ALPHA, seed: int = BOOT_SEED, rng=None) -> RuleResult:
    """Rule R1 (MISFIT) 4-cell judge.  runs: {'Sz': [BSE per run], 'Sn': [...], 'Gz': [...], 'Gn': [...]}.

    Seeds are resampled independently within each cell (percentile bootstrap).  theta_BSE and B_G,BSE use cell means.
    UNINF: pooled mean BSE of S or of G (all runs of the tier) outside the guard.
    POS: lo(theta) > 0 and lo(B_G) > 0.  REV: hi(theta) < 0.  NEG: hi(theta) < `margin`.  Else INC.
    """
    a = _bse_arrays(runs, CELL_KEYS)
    mS = float(np.mean(np.concatenate([a["Sz"], a["Sn"]])))
    mG = float(np.mean(np.concatenate([a["Gz"], a["Gn"]])))
    g = _make_rng(seed, rng)
    bm = {c: _boot_means(a[c], n_boot, g) for c in CELL_KEYS}      # fixed order -> deterministic stream
    th = (bm["Gz"] - bm["Gn"]) - (bm["Sz"] - bm["Sn"])
    bg = bm["Gz"] - bm["Gn"]
    th_lo, th_hi = _interval(th, alpha)
    bg_lo, bg_hi = _interval(bg, alpha)
    probs = {
        "theta_BSE_point": float((a["Gz"].mean() - a["Gn"].mean()) - (a["Sz"].mean() - a["Sn"].mean())),
        "theta_BSE_lo": th_lo, "theta_BSE_hi": th_hi,
        "BG_BSE_point": float(a["Gz"].mean() - a["Gn"].mean()),
        "BG_BSE_lo": bg_lo, "BG_BSE_hi": bg_hi,
        "mean_BSE_S": mS, "mean_BSE_G": mG,
    }
    lo_g, hi_g = guard[0] - BSE_GUARD_TOL, guard[1] + BSE_GUARD_TOL
    inside = lo_g <= mS <= hi_g and lo_g <= mG <= hi_g
    if not inside:
        code = "UNINF"
    elif th_lo > 0.0 and bg_lo > 0.0:
        code = "POS"
    elif th_hi < 0.0:
        code = "REV"
    elif th_hi < margin:
        code = "NEG"
    else:
        code = "INC"
    return RuleResult(code, probs, {k: [float(x) for x in a[k]] for k in CELL_KEYS},
                      {"margin": margin, "guard": list(guard), "n_boot": n_boot, "alpha": alpha,
                       "seed": seed if rng is None else None})


def judge_bse_g_only(runs: Mapping, *, margin: float = BSE_NEG_MARGIN,
                     guard: tuple[float, float] = (GUARD_LO, GUARD_HI), n_boot: int = BOOT_B,
                     alpha: float = BOOT_ALPHA, seed: int = BOOT_SEED, rng=None) -> RuleResult:
    """Rule R1 + R5: G-only BSE bootstrap.  UNINF-G > POS-G (lo(B_G) > 0) > REV-G (hi < 0) > NEG-G (hi < margin) > INC-G."""
    a = _bse_arrays(runs, ("Gz", "Gn"))
    mG = float(np.mean(np.concatenate([a["Gz"], a["Gn"]])))
    g = _make_rng(seed, rng)
    bz, bn = _boot_means(a["Gz"], n_boot, g), _boot_means(a["Gn"], n_boot, g)
    lo, hi = _interval(bz - bn, alpha)
    probs = {"BG_BSE_point": float(a["Gz"].mean() - a["Gn"].mean()), "BG_BSE_lo": lo, "BG_BSE_hi": hi,
             "mean_BSE_G": mG}
    if not (guard[0] - BSE_GUARD_TOL <= mG <= guard[1] + BSE_GUARD_TOL):
        code = "UNINF-G"
    elif lo > 0.0:
        code = "POS-G"
    elif hi < 0.0:
        code = "REV-G"
    elif hi < margin:
        code = "NEG-G"
    else:
        code = "INC-G"
    return RuleResult(code, probs, {k: [float(x) for x in a[k]] for k in ("Gz", "Gn")},
                      {"margin": margin, "guard": list(guard), "n_boot": n_boot, "alpha": alpha,
                       "seed": seed if rng is None else None})
