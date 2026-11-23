"""Tests for tools/statebench_tiers/exp027_rules_v3.py and exp027_select.py (EXP-027 v3, DRAFT).

Run: CUDA_VISIBLE_DEVICES="" rtk uv run pytest tests/test_statebench_rules.py
"""
import dataclasses
import itertools
import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.statebench_tiers import exp027_rules_v3 as R  # noqa: E402
from tools.statebench_tiers import exp027_select as S  # noqa: E402

FAST = dict(n_draws=2000)


def cells(Sz, Sn, Gz, Gn, n):
    return {"Sz": (Sz, n), "Sn": (Sn, n), "Gz": (Gz, n), "Gn": (Gn, n)}


# ====================================================================== rules_v3: theta code
def test_constants_documented_defaults():
    assert (R.C_P, R.C_N, R.MARGIN) == (0.975, 0.95, 0.30)
    assert (R.GUARD_LO, R.GUARD_HI) == (0.10, 0.90)
    assert R.N_DRAWS > 0 and isinstance(R.SEED, int)


def _first_true(res, c_P=R.C_P, c_N=R.C_N):
    p = res.probs
    g = R.GUARD_LO <= p["pooled_rate_S"] <= R.GUARD_HI and R.GUARD_LO <= p["pooled_rate_G"] <= R.GUARD_HI
    conds = [("UNINF", not g),
             ("POS", p["P_theta_gt0"] >= c_P and p["P_G_gt"] >= c_P),
             ("REV", p["P_theta_lt0"] >= c_P),
             ("NEG", p["P_theta_lt_margin"] >= c_N),
             ("INC", True)]
    return [c for c, t in conds if t]


def test_codes_follow_declared_order_exhaustive_grid():
    """Over all (k_Sz, k_Sn, k_Gz, k_Gn) with n = 4: returned code == first true condition in the order
    UNINF > POS > REV > NEG > INC, and the codes seen cover every code (conditions do overlap)."""
    seen, overlap = set(), 0
    for ks in itertools.product(range(5), repeat=4):
        r = R.judge_theta(cells(*ks, 4), **FAST)
        true = _first_true(r)
        assert r.code == true[0]
        assert r.code in R.CODES_4
        seen.add(r.code)
        overlap += len(true) > 2  # more than {winner, INC}: ordering actually matters
    assert seen == set(R.CODES_4)
    assert overlap > 0


def test_pos_and_rev_are_never_both_true():
    for ks in itertools.product(range(0, 11, 2), repeat=4):
        r = R.judge_theta(cells(*ks, 10), **FAST)
        assert not (r.probs["P_theta_gt0"] >= R.C_P and r.probs["P_theta_lt0"] >= R.C_P)


def test_uninf_overrides_pos():
    # strong positive interaction but S pooled rate 0.05 (< 0.10)
    r = R.judge_theta(cells(1, 0, 20, 0, 20), **FAST)
    assert r.code == "UNINF"
    assert r.probs["P_theta_gt0"] >= R.C_P and r.probs["P_G_gt"] >= R.C_P  # POS condition itself held


def test_pos_overrides_neg():
    # large n, small positive theta: P(theta > 0) high and P(theta < 0.30) high
    r = R.judge_theta(cells(100, 100, 160, 100, 400), n_draws=20000)
    assert r.probs["P_theta_lt_margin"] >= R.C_N
    assert r.probs["P_theta_gt0"] >= R.C_P and r.probs["P_G_gt"] >= R.C_P
    assert r.code == "POS"


def test_pos_requires_g_gain_not_only_theta():
    # z hurts S (Sz low, Sn high), G flat: theta > 0 but pGz > pGn is a coin flip -> not POS
    r = R.judge_theta(cells(4, 36, 20, 20, 40), n_draws=20000)
    assert r.probs["P_theta_gt0"] >= R.C_P
    assert r.probs["P_G_gt"] < R.C_P
    assert r.code != "POS"


def test_rev_and_neg_and_inc_cases():
    assert R.judge_theta(cells(36, 4, 20, 20, 40), n_draws=20000).code == "REV"
    assert R.judge_theta(cells(20, 20, 20, 20, 40), n_draws=20000).code == "NEG"
    assert R.judge_theta(cells(5, 5, 5, 5, 10), n_draws=20000).code == "INC"  # n = 10: too little data


def test_guard_boundaries_inclusive():
    ok = {"Gz": (5, 10), "Gn": (5, 10)}
    assert R.judge_theta({"Sz": (1, 10), "Sn": (1, 10), **ok}, **FAST).code != "UNINF"      # 0.10 inside
    assert R.judge_theta({"Sz": (9, 10), "Sn": (9, 10), **ok}, **FAST).code != "UNINF"      # 0.90 inside
    assert R.judge_theta({"Sz": (1, 10), "Sn": (0, 10), **ok}, **FAST).code == "UNINF"      # 0.05
    assert R.judge_theta({"Sz": (10, 10), "Sn": (9, 10), **ok}, **FAST).code == "UNINF"     # 0.95
    gk = {"Sz": (5, 10), "Sn": (5, 10)}
    assert R.judge_theta({**gk, "Gz": (3, 30), "Gn": (0, 30)}, **FAST).code == "UNINF"      # 3/60 = 0.05
    assert R.judge_theta({**gk, "Gz": (6, 30), "Gn": (0, 30)}, **FAST).code != "UNINF"      # 6/60 = 0.10
    assert R.judge_theta({**gk, "Gz": (27, 30), "Gn": (27, 30)}, **FAST).code != "UNINF"    # 54/60 = 0.90


def test_guard_pooled_rate_uses_unequal_cell_n():
    r = R.judge_theta({"Sz": (1, 5), "Sn": (1, 15), "Gz": (5, 10), "Gn": (5, 10)}, **FAST)
    assert r.probs["pooled_rate_S"] == 2 / 20
    assert r.code != "UNINF"


def test_probability_threshold_boundaries_are_inclusive_ge():
    c = cells(2, 2, 18, 4, 20)
    r = R.judge_theta(c, n_draws=20000)
    assert r.code == "POS"
    p = min(r.probs["P_theta_gt0"], r.probs["P_G_gt"])
    assert R.judge_theta(c, n_draws=20000, c_P=p).code == "POS"                       # == threshold -> POS
    assert R.judge_theta(c, n_draws=20000, c_P=math.nextafter(p, 2.0)).code != "POS"  # just above -> not POS
    cr = cells(18, 4, 10, 10, 20)
    rr = R.judge_theta(cr, n_draws=20000)
    assert rr.code == "REV"
    p = rr.probs["P_theta_lt0"]
    assert R.judge_theta(cr, n_draws=20000, c_P=p).code == "REV"
    assert R.judge_theta(cr, n_draws=20000, c_P=math.nextafter(p, 2.0)).code != "REV"
    cn = cells(10, 10, 10, 10, 20)
    pn = R.judge_theta(cn, n_draws=20000, c_P=2.0).probs["P_theta_lt_margin"]
    assert R.judge_theta(cn, n_draws=20000, c_P=2.0, c_N=pn).code == "NEG"
    assert R.judge_theta(cn, n_draws=20000, c_P=2.0, c_N=math.nextafter(pn, 2.0)).code == "INC"


def test_margin_monotone():
    c = cells(10, 10, 14, 6, 20)
    a = R.judge_theta(c, margin=0.1, **FAST).probs["P_theta_lt_margin"]
    b = R.judge_theta(c, margin=0.5, **FAST).probs["P_theta_lt_margin"]
    assert a <= b


def test_determinism_and_seed_dependence():
    c = cells(7, 3, 12, 6, 20)
    a, b = R.judge_theta(c), R.judge_theta(c)
    assert a == b and a.to_dict() == b.to_dict()
    assert R.judge_theta(c, seed=1).probs != R.judge_theta(c, seed=2).probs
    rng1, rng2 = np.random.default_rng(5), np.random.default_rng(5)
    assert R.judge_theta(c, rng=rng1, **FAST) == R.judge_theta(c, rng=rng2, **FAST)


def test_float64_everywhere():
    post = R.draw_posteriors([(3, 10), (2, 10)], 100, np.random.default_rng(0))
    assert post.dtype == np.float64
    r = R.judge_theta(cells(3, 3, 8, 2, 10), **FAST)
    assert all(type(v) is float for v in r.probs.values())


def test_input_validation():
    with pytest.raises(ValueError):
        R.judge_theta({"Sz": (1, 10)})
    with pytest.raises(ValueError):
        R.judge_theta(cells(11, 0, 0, 0, 10))
    with pytest.raises(ValueError):
        R.judge_theta({"Sz": (0, 0), "Sn": (0, 0), "Gz": (1, 2), "Gn": (1, 2)})


# ====================================================================== rules_v3: G-only code
def test_g_only_codes_and_order():
    assert R.judge_g_only({"Gz": (18, 20), "Gn": (4, 20)}, n_draws=20000).code == "POS-G"
    assert R.judge_g_only({"Gz": (4, 20), "Gn": (18, 20)}, n_draws=20000).code == "REV-G"
    assert R.judge_g_only({"Gz": (10, 40), "Gn": (10, 40)}, n_draws=20000).code == "NEG-G"
    assert R.judge_g_only({"Gz": (5, 10), "Gn": (5, 10)}, n_draws=20000).code == "INC-G"
    assert R.judge_g_only({"Gz": (1, 20), "Gn": (0, 20)}, n_draws=20000).code == "UNINF-G"
    assert R.judge_g_only({"Gz": (20, 20), "Gn": (18, 20)}, n_draws=20000).code == "UNINF-G"


def test_g_only_exhaustive_order():
    for kz, kn in itertools.product(range(6), repeat=2):
        r = R.judge_g_only({"Gz": (kz, 5), "Gn": (kn, 5)}, **FAST)
        p = r.probs
        g = R.GUARD_LO <= p["pooled_rate_G"] <= R.GUARD_HI
        expect = ("UNINF-G" if not g else "POS-G" if p["P_BG_gt0"] >= R.C_P else "REV-G" if p["P_BG_lt0"] >= R.C_P
                  else "NEG-G" if p["P_BG_lt_margin"] >= R.C_N else "INC-G")
        assert r.code == expect


def test_g_only_boundary_and_determinism():
    c = {"Gz": (16, 20), "Gn": (6, 20)}
    r = R.judge_g_only(c, n_draws=20000)
    assert r.code == "POS-G"
    p = r.probs["P_BG_gt0"]
    assert R.judge_g_only(c, n_draws=20000, c_P=p).code == "POS-G"
    assert R.judge_g_only(c, n_draws=20000, c_P=math.nextafter(p, 2.0)).code != "POS-G"
    assert R.judge_g_only(c) == R.judge_g_only(c)
    with pytest.raises(ValueError):
        R.judge_g_only({"Gz": (1, 2)})


# ====================================================================== rules_v3: BSE bootstrap
def const(Sz, Sn, Gz, Gn, n=5):
    return {"Sz": [Sz] * n, "Sn": [Sn] * n, "Gz": [Gz] * n, "Gn": [Gn] * n}


def test_bse_codes():
    assert R.judge_bse(const(0.5, 0.5, 0.9, 0.1)).code == "POS"
    assert R.judge_bse(const(0.9, 0.1, 0.5, 0.5)).code == "REV"
    assert R.judge_bse(const(0.5, 0.5, 0.5, 0.5)).code == "NEG"      # theta = 0, hi = 0 < 0.20
    wide = {"Sz": [0.5] * 4, "Sn": [0.5] * 4, "Gz": [0.2, 0.8, 0.2, 0.8], "Gn": [0.5] * 4}
    assert R.judge_bse(wide).code == "INC"                           # interval straddles 0 and exceeds 0.20
    assert R.judge_bse(const(0.95, 0.95, 0.9, 0.1)).code == "UNINF"  # S mean 0.95
    assert R.judge_bse(const(0.5, 0.5, 0.02, 0.02)).code == "UNINF"  # G mean 0.02


def test_bse_pos_needs_b_g_lower_bound():
    # theta > 0 only because z hurts S; B_G = 0
    r = R.judge_bse(const(0.2, 0.8, 0.5, 0.5))
    assert r.probs["theta_BSE_lo"] > 0 and r.probs["BG_BSE_lo"] == 0.0
    assert r.code != "POS"


def test_bse_neg_boundary_strict():
    base = const(0.5, 0.5, 0.5, 0.5)
    assert R.judge_bse(base, margin=0.0).code == "INC"           # hi(theta) = 0 is not < 0
    assert R.judge_bse(base, margin=1e-9).code == "NEG"


def test_bse_noisy_determinism_and_float64():
    rng = np.random.default_rng(3)
    runs = {k: list(rng.uniform(0.2, 0.8, 6)) for k in R.CELL_KEYS}
    a, b = R.judge_bse(runs), R.judge_bse(runs)
    assert a == b
    assert R.judge_bse(runs, seed=1).probs != R.judge_bse(runs, seed=2).probs
    assert all(type(v) is float for v in a.probs.values())


def test_bse_g_only():
    assert R.judge_bse_g_only({"Gz": [0.9] * 4, "Gn": [0.1] * 4}).code == "POS-G"
    assert R.judge_bse_g_only({"Gz": [0.1] * 4, "Gn": [0.9] * 4}).code == "REV-G"
    assert R.judge_bse_g_only({"Gz": [0.5] * 4, "Gn": [0.5] * 4}).code == "NEG-G"
    assert R.judge_bse_g_only({"Gz": [0.01] * 4, "Gn": [0.01] * 4}).code == "UNINF-G"


# ====================================================================== select: synthetic pilot tables
SEEDS = (9101000, 9102000)
SLOTS = [("z", 9101000), ("z", 9102000), ("noz", 9101000), ("noz", 9102000)]


def mk(tier, N, k, *, stage="pilot", graded=0, pre=0, bse_y1=0.97, bse_y0=0.02, max_y1=1.0, max_y0=0.03,
       tau=5 * S.EVAL_STEP, graded_bse=0.5):
    """4 runs for (tier, N): the first k slots (z first) have Y=1; `graded` Y=0 runs are flagged graded with
    BSE `graded_bse`; `pre` Y=0 runs are flagged pretakeoff."""
    rows = []
    y0_seen = 0
    for i, (v, s) in enumerate(SLOTS):
        y = 1 if i < k else 0
        flags, bse, mx = [], (bse_y1 if y else bse_y0), (max_y1 if y else max_y0)
        if not y:
            if y0_seen < graded:
                flags.append("graded")
                bse, mx = graded_bse, 0.8
            elif y0_seen < graded + pre:
                flags.append("pretakeoff")
            y0_seen += 1
        rows.append(dict(tier=tier, N=N, variant=v, seed=s, Y=y, BSE=bse, tau=(tau if y else None), flags=flags,
                         max=mx, stage=stage))
    return rows


def pilot(G16, G32, S16, S32, A=1, **kw):
    rows = mk("G", 16, G16, **kw) + mk("G", 32, G32, **kw) + mk("S", 16, S16, **kw) + mk("S", 32, S32, **kw)
    if A is not None:
        rows.append(dict(tier="A", N=16, variant="z", seed=9101000, Y=A, BSE=0.97 if A else 0.02, tau=None,
                         flags=[], max=1.0 if A else 0.03, stage="pilot"))
    return rows


def test_pilot_size_and_menus():
    assert len(S.pilot_run_specs()) == 17
    assert S.G_MENU == (12, 16, 24, 32, 48) and S.S_MENU == (8, 16, 24, 32, 64)
    with pytest.raises(ValueError):
        S.pool_rows(mk("G", 64, 2))
    with pytest.raises(ValueError):
        S.pool_rows(mk("S", 12, 2))


def test_R0_pipeline():
    r = S.select_rows(pilot(0, 0, 0, 0, A=0))
    assert r.path == "R0" and not r.confirm and r.cells == ()
    # A learned -> pipeline fine -> goes on (all FLOOR -> R2 extension)
    assert S.select_rows(pilot(0, 0, 0, 0, A=1)).path == "R2"
    # A failed but some G run reached max 0.5 -> not R0
    rows = pilot(0, 0, 0, 0, A=0)
    rows[0]["max"] = 0.5
    assert S.select_rows(rows).path != "R0"


def test_R0_boundary_max_020():
    """max exactly 0.20 is not < 0.20 (kills the mutant `max_pilot <= 0.20`); just below is R0."""
    for a in (0, None):
        rows = pilot(0, 0, 0, 0, A=a)
        rows[0]["max"] = S.R0_MAX
        assert S.select_rows(rows).path != "R0"
        rows[0]["max"] = S.R0_MAX - 1e-9
        assert S.select_rows(rows).path == "R0"


def test_R0_without_A_row_decides_on_G_and_S_alone():
    # A row absent (or excluded as an infrastructure failure): G,S all below 0.20 -> R0
    assert S.select_rows(pilot(0, 0, 0, 0, A=None)).path == "R0"
    # A absent, a G run learned -> not R0
    assert S.select_rows(pilot(2, 0, 0, 0, A=None)).path != "R0"
    # A present and learned (Y = 1) still blocks R0 even if every G,S run is low
    assert S.select_rows(pilot(0, 0, 0, 0, A=1)).path != "R0"
    # no G,S pilot run at all (only extension rows) never reads as a pipeline check
    ext_only = [dict(r, stage="ext") for r in mk("G", 48, 0) + mk("S", 8, 0)]
    assert S.select_rows(ext_only).path != "R0"


def test_R1_misfit_threshold_and_bse_mode():
    # 3 of 16 graded (18.75 %) -> Y mode ; 4 of 16 (25 %) -> BSE mode
    base = mk("G", 16, 0, graded=3) + mk("G", 32, 2) + mk("S", 16, 2) + mk("S", 32, 2)
    assert S.select_rows(base).mode == "Y"
    four = mk("G", 16, 0, graded=4) + mk("G", 32, 2) + mk("S", 16, 2) + mk("S", 32, 2)
    r = S.select_rows(four)
    assert r.mode == "BSE" and S.QUAL_BSE in r.qualifiers
    # BSE-MIX picks G16 (mean 0.5) over G32 (mean 0.495) -> shared N whose summed |m - 0.5| is smallest
    rows = mk("G", 16, 0, graded=4) + mk("S", 16, 0, graded=4) + mk("G", 32, 4) + mk("S", 32, 4)
    r = S.select_rows(rows)
    assert (r.path, r.mode, r.N_S, r.N_G, r.confirm) == ("R3", "BSE", 16, 16, True)
    # the same table in Y mode has no MIX anywhere (would need an extension): confirms BSE mode changed the state
    rows_y = [dict(x, flags=[]) for x in rows]
    assert S.select_rows(rows_y).path == "R2"


def test_R2_extension_choices():
    # G both CEIL -> G48 ; S MIX
    r = S.select_rows(pilot(4, 4, 2, 2))
    assert r.path == "R2" and r.extension[0][:2] == ("G", 48) and len(r.extension[0][2]) == 4
    # G both FLOOR -> G12
    assert S.select_rows(pilot(0, 0, 2, 2)).extension[0][:2] == ("G", 12)
    # G one CEIL one FLOOR -> 24
    assert S.select_rows(pilot(4, 0, 2, 2)).extension[0][:2] == ("G", 24)
    # G MIX, S both CEIL -> S64 ; S both FLOOR -> S8 ; S mixed ceil/floor -> S24
    assert S.select_rows(pilot(2, 2, 4, 4)).extension[0][:2] == ("S", 64)
    assert S.select_rows(pilot(2, 2, 0, 0)).extension[0][:2] == ("S", 8)
    assert S.select_rows(pilot(2, 2, 4, 0)).extension[0][:2] == ("S", 24)
    # both lack MIX -> G only
    r = S.select_rows(pilot(4, 4, 0, 0))
    assert len(r.extension) == 1 and r.extension[0][:2] == ("G", 48)
    # extension seeds are 9101000, 9102000 for both variants
    assert set(r.extension[0][2]) == {("z", 9101000), ("z", 9102000), ("noz", 9101000), ("noz", 9102000)}


def test_R2_is_once_only():
    rows = pilot(4, 4, 2, 2) + mk("G", 48, 4, stage="ext")
    r = S.select_rows(rows)
    assert r.path == "R7" and r.extension is None      # G still has no MIX after the single extension


def test_R2_then_final_paths():
    r = S.select_rows(pilot(4, 4, 2, 2) + mk("G", 48, 2, stage="ext"))
    assert r.path == "R4" and r.N_G == 48 and r.N_S in (16, 32)


def test_R3_shared_and_ties():
    r = S.select_rows(pilot(2, 4, 2, 0))
    assert (r.path, r.N_S, r.N_G, r.confirm, r.stat) == ("R3", 16, 16, True, "theta")
    assert r.cells == (("S", 16, "z"), ("S", 16, "noz"), ("G", 16, "z"), ("G", 16, "noz"))
    # exact tie in score between N=16 and N=32 -> smaller N
    assert S.select_rows(pilot(2, 2, 2, 2)).N_G == 16
    # strictly better at 32: f = 0.25 at 16, 0.5 at 32 for both tiers
    r = S.select_rows(pilot(1, 2, 1, 2))
    assert (r.path, r.N_S, r.N_G) == ("R3", 32, 32)
    # shared-sum criterion: S best at 16, G best at 32, sums tie? S:(.25,.5)->0.25/0 ; choose pair minimising sum
    r = S.select_rows(pilot(1, 2, 2, 1))
    assert r.path == "R3" and r.N_S == r.N_G == 16     # sums tie (0.25 + 0 at both N) -> smaller N


def test_R4_split_and_tiebreak():
    r = S.select_rows(pilot(4, 2, 2, 0))               # G MIX only at 32, S MIX only at 16
    assert (r.path, r.N_S, r.N_G, r.confirm) == ("R4", 16, 32, True)
    assert S.QUAL_R4 in r.qualifiers
    r = S.select_rows(pilot(4, 4, 1, 3) + mk("G", 48, 2, stage="ext"))   # S: both |f-.5| = .25 -> smaller N
    assert (r.path, r.N_S, r.N_G) == ("R4", 16, 48)
    r = S.select_rows(pilot(1, 3, 4, 4) + mk("S", 64, 2, stage="ext"))   # G tie -> 16 ; S MIX only at 64
    assert (r.path, r.N_S, r.N_G) == ("R4", 64, 16)


def test_R5_after_s_extension():
    # S all CEIL -> R2 extends S to 64 ; then still CEIL -> R5 with sub SC
    assert S.select_rows(pilot(2, 2, 4, 4)).path == "R2"
    r = S.select_rows(pilot(2, 2, 4, 4) + mk("S", 64, 4, stage="ext"))
    assert (r.path, r.sub, r.stat, r.N_G, r.N_S, r.confirm) == ("R5", "SC", "G-only", 16, None, True)
    assert r.cells == (("G", 16, "z"), ("G", 16, "noz")) and S.QUAL_R5 in r.qualifiers
    r = S.select_rows(pilot(2, 2, 0, 0) + mk("S", 8, 0, stage="ext"))
    assert (r.path, r.sub) == ("R5", "SF")
    r = S.select_rows(pilot(2, 2, 4, 0) + mk("S", 24, 4, stage="ext"))
    assert (r.path, r.sub) == ("R5", "SX")
    # G tie among MIX Ns -> smaller N ; strictly better at 32 -> 32
    assert S.select_rows(pilot(2, 2, 4, 4) + mk("S", 64, 4, stage="ext")).N_G == 16
    assert S.select_rows(pilot(1, 2, 4, 4) + mk("S", 64, 4, stage="ext")).N_G == 32


def test_R5_both_missing_goes_through_g_extension_then_r5():
    r = S.select_rows(pilot(4, 4, 4, 4))
    assert r.path == "R2" and r.extension[0][:2] == ("G", 48)
    r = S.select_rows(pilot(4, 4, 4, 4) + mk("G", 48, 2, stage="ext"))
    assert (r.path, r.sub, r.N_G) == ("R5", "SC", 48)


def test_R6_all_ceiling_never_confirms():
    rows = pilot(4, 4, 4, 4) + mk("G", 48, 4, stage="ext")
    r = S.select_rows(rows)
    assert r.path == "R6" and r.sub == "default" and not r.confirm and r.stat is None and r.cells == ()
    # slow first passage does not open anything: the log2-tau exception is closed (design-log 9.3)
    slow = pilot(4, 4, 4, 4, tau=9 * S.EVAL_STEP) + mk("G", 48, 4, stage="ext", tau=9 * S.EVAL_STEP)
    r = S.select_rows(slow)
    assert (r.path, r.confirm, r.stat) == ("R6", False, None)
    with pytest.raises(TypeError):
        S.select_rows(slow, c3_log2tau_gate=True)
    for gone in ("QUAL_R6", "AL_N", "AL_TAU_MIN_EVALS", "_median"):
        assert not hasattr(S, gone)


def test_R7_catch_all():
    # S MIX, G all FLOOR even after extension
    r = S.select_rows(pilot(0, 0, 2, 2) + mk("G", 12, 0, stage="ext"))
    assert r.path == "R7" and not r.confirm and r.cells == ()


def test_all_paths_reached():
    reached = {
        S.select_rows(pilot(0, 0, 0, 0, A=0)).path,
        S.select_rows(mk("G", 16, 0, graded=4) + mk("S", 16, 0, graded=4) + mk("G", 32, 4) + mk("S", 32, 4)).path,
        S.select_rows(pilot(4, 4, 2, 2)).path,
        S.select_rows(pilot(2, 4, 2, 0)).path,
        S.select_rows(pilot(4, 2, 2, 0)).path,
        S.select_rows(pilot(2, 2, 4, 4) + mk("S", 64, 4, stage="ext")).path,
        S.select_rows(pilot(4, 4, 4, 4) + mk("G", 48, 4, stage="ext")).path,
        S.select_rows(pilot(0, 0, 2, 2) + mk("G", 12, 0, stage="ext")).path,
    }
    assert reached == {"R0", "R2", "R3", "R4", "R5", "R6", "R7"}
    # R1 is a mode switch, not a terminal rule: it shows up in `trace` and `misfit`
    r1 = S.select_rows(mk("G", 16, 0, graded=4) + mk("S", 16, 0, graded=4) + mk("G", 32, 4) + mk("S", 32, 4))
    assert r1.misfit and r1.trace == ("R1", "R3") and r1.path == "R3"


def test_R1_uses_pilot_stage_only_and_priority_over_r2():
    # MISFIT fires before any extension is requested
    rows = mk("G", 16, 4) + mk("G", 32, 4) + mk("S", 16, 0, graded=4) + mk("S", 32, 4)
    r = S.select_rows(rows)
    assert r.mode == "BSE"
    # ext-stage graded runs do not count toward the 25 %
    rows = pilot(4, 4, 2, 2) + mk("G", 48, 0, stage="ext", graded=4)
    assert S.select_rows(rows).mode == "Y"


def test_late_flag():
    rows = pilot(0, 0, 2, 2, pre=2)           # every G run has Y=0, 2 of 4 per cell flagged -> 50 %
    assert S.select_rows(rows).late_flag
    assert not S.select_rows(pilot(0, 0, 2, 2, pre=1)).late_flag
    assert not S.select_rows(pilot(4, 4, 2, 2)).late_flag    # no Y=0 G runs


def test_late_flag_does_not_change_path():
    a = S.select_rows(pilot(0, 0, 2, 2, pre=2))
    b = S.select_rows(pilot(0, 0, 2, 2, pre=0))
    assert a.path == b.path and a.extension == b.extension and a.late_flag and not b.late_flag


# ---------------------------------------------------------------- structural variant blindness
def _walk_fields(cls, seen=None):
    out = set()
    for f in dataclasses.fields(cls):
        out.add(f.name)
    return out


def test_pooled_table_has_no_variant_or_seed_field():
    names = _walk_fields(S.PooledCell) | _walk_fields(S.PooledTable)
    assert not ({"variant", "seed", "seeds", "variants"} & names)


def test_select_rejects_raw_rows():
    with pytest.raises(TypeError):
        S.select(pilot(2, 2, 2, 2))


def _permute_variants(rows, rnd):
    """Randomly permute the (variant, seed) labels among the runs of each (tier, N) group."""
    out = [dict(r) for r in rows]
    groups = {}
    for i, r in enumerate(out):
        groups.setdefault((r["tier"], r["N"]), []).append(i)
    for idx in groups.values():
        labels = [(out[i]["variant"], out[i]["seed"]) for i in idx]
        rnd.shuffle(labels)
        for i, (v, s) in zip(idx, labels):
            out[i]["variant"], out[i]["seed"] = v, s
    return out


@pytest.mark.parametrize("rows", [
    pilot(4, 2, 2, 0),
    pilot(2, 4, 2, 0),
    pilot(2, 2, 4, 4) + mk("S", 64, 4, stage="ext"),
    pilot(4, 4, 4, 4, tau=9 * S.EVAL_STEP) + mk("G", 48, 4, stage="ext", tau=9 * S.EVAL_STEP),
    pilot(0, 0, 0, 0, A=None),
    pilot(0, 0, 2, 2, pre=2),
    mk("G", 16, 0, graded=4) + mk("S", 16, 0, graded=4) + mk("G", 32, 4) + mk("S", 32, 4),
    pilot(1, 3, 3, 1),
])
def test_variant_label_permutation_invariance(rows):
    rnd = random.Random(0)
    base = S.select_rows(rows)
    base_pool = S.pool_rows(rows)
    for _ in range(25):
        perm = _permute_variants(rows, rnd)
        assert S.pool_rows(perm) == base_pool
        assert S.select_rows(perm) == base


def test_row_order_invariance():
    rows = pilot(4, 2, 2, 0)
    rnd = random.Random(1)
    base = S.select_rows(rows)
    for _ in range(10):
        sh = list(rows)
        rnd.shuffle(sh)
        assert S.select_rows(sh) == base


def test_variant_asymmetry_is_invisible():
    """z all succeed / noz all fail vs. the same counts spread evenly: pooled summary and output are identical."""
    a = mk("G", 16, 2) + mk("G", 32, 2) + mk("S", 16, 2) + mk("S", 32, 2)              # z succeeds, noz fails
    b = [dict(r) for r in a]
    for r in b:                                                                          # swap labels so noz succeeds
        r["variant"] = "noz" if r["variant"] == "z" else "z"
    assert S.select_rows(a) == S.select_rows(b)


def test_selection_deterministic_and_dataclass_equal():
    rows = pilot(4, 2, 2, 0)
    assert S.select_rows(rows) == S.select_rows(rows)
    assert S.select_rows(rows).to_dict() == S.select_rows(rows).to_dict()


# ---------------------------------------------------------------- n-gate
def test_n_gate_branches():
    r = S.n_gate(20.0, t_p_epochs=820)    # floor(600/20) = 30 (cap)
    assert (r.n, r.action, r.qualifier, r.t_max_epochs) == (30, "ENQUEUE", None, 820)
    r = S.n_gate(30.0, t_p_epochs=820)    # 20 exactly
    assert (r.n, r.action) == (20, "ENQUEUE")
    r = S.n_gate(30.5, t_p_epochs=820)    # 19
    assert (r.n, r.action, r.qualifier) == (19, "ENQUEUE_WITH_QUALIFIER", S.QUAL_NGATE)
    r = S.n_gate(40.0, t_p_epochs=820)    # 15
    assert (r.n, r.action) == (15, "ENQUEUE_WITH_QUALIFIER")
    r = S.n_gate(42.0, t_p_epochs=820)    # 14
    assert (r.n, r.action, r.t_max_epochs) == (14, "NO_CONFIRM", None)
    assert S.n_gate(1.0, t_p_epochs=820).n == 30          # cap
    assert S.n_gate(20.0, t_p_epochs=420).t_max_epochs == 420


def test_n_gate_t_p_epochs_required_and_on_menu():
    with pytest.raises(TypeError):
        S.n_gate(20.0)                    # no silent default
    for bad in (800, 40, None):
        with pytest.raises(ValueError):
            S.n_gate(20.0, t_p_epochs=bad)


def test_n_gate_t_max_reduction():
    kw = dict(c_set_by_epochs={620: 30.0, 420: 20.0}, c9_passed=True, t_p_epochs=820)
    # 820 -> n = 12 ; 620 same state -> n = 20 ; take 620
    r = S.n_gate(50.0, same_state_by_epochs={620: True, 420: True}, **kw)
    assert (r.n, r.t_max_epochs, r.reduced, r.action) == (20, 620, True, "ENQUEUE")
    # 620 state differs, 420 same -> 420
    r = S.n_gate(50.0, same_state_by_epochs={620: False, 420: True}, **kw)
    assert (r.n, r.t_max_epochs) == (30, 420)
    # no C9 -> closed path
    r = S.n_gate(50.0, t_p_epochs=820, same_state_by_epochs={620: True, 420: True},
                 c_set_by_epochs=kw["c_set_by_epochs"])
    assert r.action == "NO_CONFIRM"
    # reduced n still < 15 -> no confirmation
    r = S.n_gate(50.0, t_p_epochs=820, same_state_by_epochs={620: True, 420: True}, c9_passed=True,
                 c_set_by_epochs={620: 45.0, 420: 44.0})
    assert r.action == "NO_CONFIRM"
    # pilot ran at 620 -> only 420 considered
    r = S.n_gate(50.0, **{**kw, "t_p_epochs": 620}, same_state_by_epochs={620: True, 420: True})
    assert r.t_max_epochs == 420
    # pilot ran at 420 -> no shorter T_max exists; never "reduce" upward to 620
    r = S.n_gate(50.0, **{**kw, "t_p_epochs": 420}, same_state_by_epochs={620: True, 420: True})
    assert (r.action, r.t_max_epochs, r.reduced) == ("NO_CONFIRM", None, False)
    with pytest.raises(ValueError):
        S.n_gate(0.0, t_p_epochs=820)


# ---------------------------------------------------------------- pilot_plan
def table(scale, smoke, ns=(16, 32, 64), variants=("z", "noz")):
    pts = [dict(tier="G", N=N, variant=v, c_step=scale * (33 + 2 * N) * 1e-5, c_eval=0.0, t0=0.0)
           for N in ns for v in variants]
    return {"smoke_cost": smoke, "points": pts}


def test_run_cost_hand_value_and_interpolation():
    t = table(1.0, 0.0, ns=(16, 32))                 # only N = 16, 32 measured
    steps = S.steps_for_epochs(820)
    assert steps == 80073
    assert S.run_cost(t, "z", 16, 820) == pytest.approx(steps * 65e-5)
    assert S.run_cost(t, "noz", 24, 820) == pytest.approx(steps * 81e-5)       # interpolation
    assert S.run_cost(t, "z", 64, 820) == pytest.approx(steps * 161e-5)        # linear extrapolation
    t2 = dict(t, points=[dict(p, c_eval=0.01, t0=0.5) for p in t["points"]])
    assert S.run_cost(t2, "z", 16, 820) == pytest.approx(steps * 65e-5 + 41 * 0.01 + 0.5)
    assert S.steps_for_epochs(620) == 60543 and S.steps_for_epochs(420) == 41013


def test_pilot_plan_full_picks_largest_fitting_epoch():
    t = table(0.15, 2.0)
    p = S.pilot_plan(t)
    # hand arithmetic (body = smoke + 17 pilot runs): pilot token-runs 1361; extension is outside the body test
    c820 = 2.0 + 80073 * 1361e-5 * 0.15
    c620 = 2.0 + 60543 * 1361e-5 * 0.15
    c420 = 2.0 + 41013 * 1361e-5 * 0.15
    assert c820 > 130 >= c620 > c420
    assert (p["branch"], p["epochs"], p["steps"], p["confirm_possible"]) == ("FULL", 620, 60543, True)
    row = {r["epochs"]: r for r in p["candidates"]}
    assert row[620]["body"] == pytest.approx(c620) and row[820]["body"] == pytest.approx(c820)
    assert row[620]["ext_ub_option"] == "S64" and row[620]["ext_ub"] == pytest.approx(60543 * 644e-5 * 0.15)
    assert S.pilot_plan(table(0.05, 2.0))["epochs"] == 820     # body 56.4 at 820
    # the arithmetic is shown: one line per menu entry plus the decision
    assert len(p["arithmetic"]) == 4 and "T_p=820" in p["arithmetic"][0] and p["arithmetic"][-1].startswith("-> FULL")
    assert f"{c620:.3f}" in p["arithmetic"][1]


def test_pilot_plan_budget_inclusive():
    t = table(0.15, 2.0)
    c620 = 2.0 + 60543 * 1361e-5 * 0.15
    assert S.pilot_plan(t, budget=c620 + 1e-9)["epochs"] == 620
    assert S.pilot_plan(t, budget=c620 - 1e-6)["epochs"] == 420


def test_pilot_plan_extension_cap_is_separate():
    t = table(0.15, 2.0)
    p = S.pilot_plan(t)
    row = {r["epochs"]: r for r in p["candidates"]}
    # body fits at 620 although the S64 extension (60543 * 644e-5 * 0.15 = 38.97) is above a cap of 30; the cap only
    # flags the option, it does not change T_p
    q = S.pilot_plan(t, ext_cap=30.0)
    assert q["epochs"] == 620
    r620 = {r["epochs"]: r for r in q["candidates"]}[620]
    assert r620["ext_options_over_cap"] == ["G48", "S64"] and not r620["ext_ub_fits_cap"]
    assert row[620]["ext_ub_fits_cap"] and row[620]["ext_options_over_cap"] == []
    assert S.EXT_BUDGET == 60.0 and S.PILOT_BUDGET == 130.0
    # an extension is 4 runs for either tier
    assert len([1 for _ in S.EXT_VARIANTS for _ in S.EXT_SEEDS]) == 4


def test_ext_gate_over_cap_closes_as_r7_and_leaves_t_p():
    t = table(0.15, 2.0)
    q = S.pilot_plan(t, ext_cap=30.0)
    opts = {r["epochs"]: r for r in q["candidates"]}[q["epochs"]]["ext_options"]
    over = S.ext_gate("S", 64, opts, ext_cap=30.0)
    assert q["epochs"] == 620 and over["action"] == "CLOSE_R7_EXT_OVER_CAP" and over["cost_gpu_h"] > 30.0
    ok = S.ext_gate("G", 12, opts, ext_cap=30.0)
    assert ok["action"] == "ENQUEUE_EXT"
    # inclusive at the cap
    assert S.ext_gate("S", 64, {"S64": 60.0})["action"] == "ENQUEUE_EXT"
    assert S.ext_gate("S", 64, {"S64": 60.0 + 1e-9})["action"] == "CLOSE_R7_EXT_OVER_CAP"
    with pytest.raises(ValueError):
        S.ext_gate("S", 24 + 1, opts)


def test_pilot_plan_g_only_and_close():
    p = S.pilot_plan(table(0.25, 0.0))
    g8 = 41013 * 648e-5 * 0.25
    assert p["branch"] == "G_ONLY_420" and p["confirm_possible"] is False
    assert p["g_only_8"] == pytest.approx(g8) and p["epochs"] == 420 and p["steps"] == 41013
    assert p["arithmetic"][-1].startswith("-> G_ONLY_420") and "G8" in p["arithmetic"][-2]
    p = S.pilot_plan(table(0.5, 0.0))                # G8 = 132.8 > 130
    assert p["branch"] == "CLOSE_NO_PILOT" and p["epochs"] is None
    assert p["arithmetic"][-1].startswith("-> CLOSE_NO_PILOT")
    assert S.pilot_plan(table(0.5, 0.0), budget=200.0)["branch"] == "G_ONLY_420"


def test_pilot_plan_deterministic():
    t = table(0.2, 2.0)
    assert S.pilot_plan(t) == S.pilot_plan(t)


def test_seed_scheme():
    assert S.PILOT_SEEDS == (9101000, 9102000) and S.EXT_SEEDS == S.PILOT_SEEDS and S.SMOKE_SEED == 0
    assert S.conf_seeds(3) == (1000, 2000, 3000) and S.conf_seeds(30)[-1] == 30000
    assert [s_ for *_x, s_ in S.pilot_run_specs()] == [9101000, 9102000] * 8 + [9101000]


def test_pool_rows_rejects_nonfinite():
    for key in ("BSE", "max"):
        rows = mk("G", 16, 2) + mk("S", 16, 2)
        rows[0][key] = float("nan")
        with pytest.raises(ValueError):
            S.pool_rows(rows)


# ====================================================================== exp027_cost (appended)
import json as _json  # noqa: E402

from tools.statebench_tiers import exp027_cost as C  # noqa: E402
from tools.statebench_tiers import exp027_judge as J  # noqa: E402
from tools.statebench_tiers import outcomes as OUT  # noqa: E402


def _smoke_obj():
    # z: N=16 (65 tok) 0.5 s/step, eval 60 s; N=32 (97 tok) 0.9 s/step, eval 100 s; noz twice as cheap; startup 120 s
    def run(t, N, v, sps, ev):
        return {"run_name": f"e027_smoke2_{t}N{N}_{v}_s0", "tier": t, "N": N, "variant": v,
                "train_seconds": 3906 * sps, "eval_seconds": ev, "startup_seconds": 120.0}
    return {"steps": 3906, "n_evals": 2, "runs": [
        run("G", 16, "z", 0.5, 60.0), run("G", 32, "z", 0.9, 100.0),
        run("G", 16, "noz", 0.25, 30.0), run("G", 32, "noz", 0.45, 50.0), run("S", 64, "z", 1.7, 180.0)]}


def test_cost_g_run_hand_computed():
    m = C.CostModel(C.load_smoke(_smoke_obj()))
    # z, G, N=24 -> 81 tokens, halfway between 65 and 97: step 0.7 s, eval 80 s
    assert m.c_step("G", 24, "z") == pytest.approx(0.7 / 3600)
    assert m.c_eval("G", 24, "z") == pytest.approx(80.0 / 3600)
    steps = C.steps_for(820)
    assert steps == 80073 and C.n_evals_for(820) == 41 and C.steps_for is C.S.steps_for_epochs
    expect = (steps * 0.7 + 41 * 80.0 + 120.0) / 3600
    assert m.g_run("G", 24, "z", steps, 41) == pytest.approx(expect, rel=1e-12)
    # noz N=64 (161 tok) extrapolates the noz line: 0.25 + (0.45-0.25)*(161-65)/32 = 0.85 s
    assert m.c_step("S", 64, "noz") == pytest.approx(0.85 / 3600)
    # smoke total = sum(train + 2*eval + startup)/3600
    tot = sum(3906 * s + 2 * e + 120 for s, e in [(0.5, 60), (0.9, 100), (0.25, 30), (0.45, 50), (1.7, 180)]) / 3600
    assert m.smoke_cost == pytest.approx(tot)


def test_cost_plan_and_conf_branches():
    m = C.CostModel(C.load_smoke(_smoke_obj()))
    p = C.plan(m)
    assert len(C.pilot_specs()) == 17
    assert p["candidates"][0]["epochs"] == 820 and p["branch"] in ("FULL", "G_ONLY_420", "CLOSE_NO_PILOT")
    r = p["candidates"][0]
    assert r["body"] == pytest.approx(p["smoke_cost"] + r["pilot_17"]) and r["steps"] == 80073
    assert C.plan(m, budget=1e9)["epochs"] == 820 and C.plan(m, budget=1e9)["branch"] == "FULL"
    assert C.plan(m, budget=0.01)["branch"] == "CLOSE_NO_PILOT"
    assert p["arithmetic"] and "T_p=820" in p["arithmetic"][0]
    # cost and select take the same decision on the same per-run costs; no steps-per-epoch option remains
    assert C.main(["plan", "--smoke", "x", "--steps-per-epoch", "48.83"]) == 2
    c = C.conf_cost(m, [("G", 16)], 820)
    one = sum(m.g_run_epochs("G", 16, v, 820) for v in ("z", "noz"))
    assert c["C_set_gpu_h"] == pytest.approx(one) and c["n"] == min(30, int(600 // one))
    assert C.lpt_makespan([4, 3, 3, 2, 2, 2], 4) == 5.0


def test_cost_tqdm_split_and_missing_log(tmp_path):
    text = "".join(f"\r  5%|x| {i}/100 [00:{t:02d}<01:00, 1.00it/s]" for i, t in
                   [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 25), (6, 26), (7, 27)])
    pairs = C.parse_tqdm(text)
    train, evals = C.split_train_eval(pairs, 1)
    assert evals == [pytest.approx(20.0)] and train == pytest.approx(7.0)
    with pytest.raises(LookupError):
        C.find_log("e027_smoke_GN16_z_s0", str(tmp_path))
    assert C.main(["extract", "--runs", "e027_smoke_GN16_z_s0", "--log-dir", str(tmp_path)]) == 1


# ====================================================================== exp027_judge (appended)
def _traj(kind, n=12):
    # kind 'learn': reaches 1.0 at eval 4; 'never': stays low; 'collapse': learns then drops
    e = {"learn": [0.0] * 3 + [1.0] * (n - 3), "never": [0.02] * n,
         "collapse": [0.0] * 3 + [1.0] * 3 + [0.0] * (n - 6)}[kind]
    return [(S.EVAL_STEP * (i + 1), x) for i, x in enumerate(e)]


TMAX = 12 * S.EVAL_STEP   # 12 evals * 1953 = 23436 = (240 // 20) * EVAL_STEP
SEL_THETA = {"mode": "Y", "N_S": 16, "N_G": 16, "stat": "theta"}


def _write_witness(path, tamper=False, sel=None, n=4, t_max_epochs=240):
    w = J.make_witness(sel or SEL_THETA, n=n, t_max_epochs=t_max_epochs)
    if tamper:
        w["sha256"]["outcomes.py"] = "0" * 64
    path.write_text(_json.dumps(w))


def _conf_trajs():
    t, names = {}, []
    plan_ = {("S", "z"): "learn", ("S", "noz"): "learn", ("G", "z"): "learn", ("G", "noz"): "never"}
    for (tier, var), kind in plan_.items():
        for s in range(1, 5):
            nm = f"e027_conf_{tier}N16_{var}_s{s}"
            t[nm] = _traj("collapse" if (kind == "learn" and var == "z" and tier == "G" and s == 4) else kind)
            names.append(nm)
    return t, names


def test_judge_y_consec_matches_outcomes():
    for kind in ("learn", "never", "collapse"):
        tr = _traj(kind)
        assert J.y_consec(tr, 0.9, 2) == OUT.Y(tr)
    assert J.y_consec([(1, 1.0), (2, 0.0), (3, 1.0)], 0.9, 2) == 0 and J.y_consec([(1, 1.0), (2, 0.0)], 0.9, 1) == 1


def test_judge_conf_synthetic_and_witness(tmp_path):
    trajs, names = _conf_trajs()
    wit = tmp_path / "w.json"
    _write_witness(wit)
    out = tmp_path / "conf.json"
    assert J.main(["--mode", "conf", "--t-max-steps", str(TMAX), "--runs", *names, "--out", str(out), "--witness", str(wit)], trajs=trajs) == 0
    res = _json.loads(out.read_text())
    direct = R.judge_theta({"Sz": (4, 4), "Sn": (4, 4), "Gz": (4, 4), "Gn": (0, 4)})
    assert res["code"] == direct.code and res["probs"] == direct.probs and res["design"] == "4cell"
    assert res["descriptive"]["collapse_counts"]["Gz"] == 1 and res["descriptive"]["BSE_DiD"]["G_gain"] == pytest.approx(0.98)
    assert set(res["descriptive"]["threshold_sensitivity"]) == {"learn0.5", "learn0.95", "consec1", "consec3"}
    # deterministic: identical bytes on a second run
    out2 = tmp_path / "conf2.json"
    J.main(["--mode", "conf", "--t-max-steps", str(TMAX), "--runs", *names, "--out", str(out2), "--witness", str(wit)], trajs=trajs)
    assert out.read_text() == out2.read_text()
    # BSE outcome path: the outcome comes from the witness selection mode
    witb = tmp_path / "wb.json"
    _write_witness(witb, sel=dict(SEL_THETA, mode="BSE"))
    out3 = tmp_path / "conf3.json"
    assert J.main(["--mode", "conf", "--t-max-steps", str(TMAX), "--runs", *names, "--out", str(out3),
                   "--witness", str(witb)], trajs=trajs) == 0
    assert _json.loads(out3.read_text())["outcome"] == "BSE"
    assert _json.loads(out.read_text())["witness_selection"] == dict(SEL_THETA, n=4, t_max_epochs=240)


def test_judge_conf_witness_refusal(tmp_path):
    trajs, names = _conf_trajs()
    out = tmp_path / "o.json"
    argv = ["--mode", "conf", "--t-max-steps", str(TMAX), "--runs", *names, "--out", str(out)]
    assert J.main(argv, trajs=trajs) == 3                                  # no witness
    bad = tmp_path / "bad.json"
    _write_witness(bad, tamper=True)
    assert J.main(argv + ["--witness", str(bad)], trajs=trajs) == 3        # hash mismatch
    short = tmp_path / "short.json"
    short.write_text(_json.dumps({"sha256": {"outcomes.py": J.sha256_file(str(REPO / "tools/statebench_tiers/outcomes.py"))}}))
    assert J.main(argv + ["--witness", str(short)], trajs=trajs) == 3      # missing file hashes
    assert J.main(argv + ["--witness", str(tmp_path / "nope.json")], trajs=trajs) == 3
    assert not out.exists()


def test_judge_conf_missing_run_fails(tmp_path):
    trajs, names = _conf_trajs()
    wit = tmp_path / "w.json"
    _write_witness(wit)
    del trajs[names[0]]
    assert J.main(["--mode", "conf", "--t-max-steps", str(TMAX), "--runs", *names, "--out", str(tmp_path / "o.json"),
                   "--witness", str(wit)], trajs=trajs) == 1


def test_judge_pilot_synthetic(tmp_path):
    trajs, names = {}, []
    for tier in ("G", "S"):
        for N in (16, 32):
            for var in ("z", "noz"):
                for s in (9101000, 9102000):
                    nm = f"e027_pilot_{tier}N{N}_{var}_s{s}"
                    trajs[nm] = _traj("learn" if (var == "z" and N == 16) else "never")
                    names.append(nm)
    trajs["e027_pilot_AN16_z_s9101000"] = _traj("learn")
    names.append("e027_pilot_AN16_z_s9101000")
    assert len(names) == 17
    out = tmp_path / "p.json"
    assert J.main(["--mode", "pilot", "--t-max-steps", str(TMAX), "--runs", *names, "--out", str(out)], trajs=trajs) == 0
    res = _json.loads(out.read_text())
    assert len(res["rows"]) == 17 and res["selection"]["path"].startswith("R")
    # the selection saw only the pooled table: same path when variant labels are swapped
    sw = {n.replace("_z_", "_X_").replace("_noz_", "_z_").replace("_X_", "_noz_"): t for n, t in trajs.items()}
    out2 = tmp_path / "p2.json"
    assert J.main(["--mode", "pilot", "--t-max-steps", str(TMAX), "--runs", *sw, "--out", str(out2)], trajs=sw) == 0
    assert _json.loads(out2.read_text())["selection"] == res["selection"]
    # conf-stage names are refused in pilot mode
    assert J.main(["--mode", "pilot", "--t-max-steps", str(TMAX), "--runs", "e027_conf_GN16_z_s1", "--out", str(tmp_path / "x.json")],
                  trajs={"e027_conf_GN16_z_s1": _traj("learn")}) == 1


# ====================================================================== moderator review 2026-11-11: float32 / eval grid
def test_eval_step_matches_drop_last_chunks():
    # puzzle_dataset drops the last partial batch per 20-epoch chunk: evals at EVAL_STEP * j; amendment 1: global batch 1024 -> floor(20 * 100000 / 1024) = 1953
    assert S.GLOBAL_BATCH == 1024 and S.EVAL_STEP == 1953 == S.block_steps() == (20 * 100_000) // 1024
    assert S.block_steps(batch=2048) == 976                    # the v3 value, only for the record
    # one shared step definition: 820 / 620 / 420 epochs and the 40-epoch smoke run
    assert [S.steps_for_epochs(e) for e in (820, 620, 420, 40)] == [80073, 60543, 41013, 3906]
    assert [S.n_evals_for_epochs(e) for e in (820, 620, 420, 40)] == [41, 31, 21, 2]
    for bad in (0, -20, 25, 41, 820.0):
        with pytest.raises(ValueError):
            S.steps_for_epochs(bad)


def test_bse_mix_boundary_float32():
    f = lambda x: float(np.float32(x))
    rows = [dict(tier="G", N=16, variant=v, seed=s, Y=0, BSE=f(0.8), tau=None, flags=["graded"], max=0.9)
            for v, s in SLOTS]
    c = S.pool_rows(rows + mk("S", 16, 2)).get("G", 16)
    assert S._state(c, "BSE") == "MIX"           # mean float32(0.800) = 0.800000012 is on the closed bound
    rows_hi = [dict(r, BSE=f(0.801)) for r in rows]
    assert S._state(S.pool_rows(rows_hi + mk("S", 16, 2)).get("G", 16), "BSE") == "CEIL"


def test_bse_guard_float32_mean_on_bound():
    f = lambda x: float(np.float32(x))
    # decimal mean of G BSE is exactly 0.90 (0.95, 0.85); float32 mean is 0.9000000060 -> must be inside
    runs = {"Sz": [0.5, 0.5], "Sn": [0.5, 0.5], "Gz": [f(0.95), f(0.95)], "Gn": [f(0.85), f(0.85)]}
    assert R.judge_bse(runs, n_boot=200).code != "UNINF"
    assert R.judge_bse_g_only({"Gz": runs["Gz"], "Gn": runs["Gn"]}, n_boot=200).code != "UNINF-G"
    runs_out = dict(runs, Gz=[f(0.951), f(0.951)])
    assert R.judge_bse(runs_out, n_boot=200).code == "UNINF"


def test_judge_sensitivity_float32():
    f = lambda x: float(np.float32(x))
    tr = [(S.EVAL_STEP * (i + 1), x) for i, x in enumerate([0.1, f(0.95), f(0.95), 0.2])]
    assert J.y_consec(tr, 0.95, 2) == 1 and J.y_consec(tr, 0.9, 3) == 0


def test_smoke_steps_drop_last():
    assert C.SMOKE_STEPS == 3906


def test_extract_final_eval_invisible(tmp_path):
    # 3906 steps at 1 s each, eval 1 (30 s) visible after tick 1953; eval 2 runs after tick 3906 (no gap).
    lines, t = [], 0
    for it in range(0, 3907):
        if it == 1954:
            t += 30
        lines.append(f"{it}/3906 [{t // 60:02d}:{t % 60:02d}<00:00, 1.00it/s]")
        t += 1
    name = "e027_smoke_GN16_z_s0"
    (tmp_path / f"0800_0001_{name}.log").write_text("\r".join(lines))
    tq = 3906 + 30                                    # tqdm elapsed at the final tick
    out = C.extract_timings([name], str(tmp_path), runtime_fn=lambda n, r: tq + 30.0 + 100.0)
    r = out["runs"][0]
    assert r["eval_seconds"] == pytest.approx(30.0, abs=1.5)        # the one visible eval, not (30 + 1) / 2
    assert r["startup_seconds"] == pytest.approx(130.0, abs=1.5)    # final eval (30) + startup (100)


# ====================================================================== moderator review 2: infra rule, freeze integrity
def _cut(tr, frac):
    return [(s_, v) for s_, v in tr if s_ <= frac * TMAX]


def _judge_conf(tmp_path, trajs, names, sel=None, extra=(), tmax=TMAX, n=4, epochs=240):
    wit = tmp_path / "wj.json"
    _write_witness(wit, sel=sel, n=n, t_max_epochs=epochs)
    out = tmp_path / "cj.json"
    rc = J.main(["--mode", "conf", "--t-max-steps", str(tmax), "--runs", *names, "--out", str(out),
                 "--witness", str(wit), *extra], trajs=trajs)
    return rc, out


def test_pick_run_infra_rule():
    full, half, early = _traj("learn"), _cut(_traj("learn"), 0.6), _cut(_traj("learn"), 0.4)
    assert J.pick_run([full], TMAX)["status"] == "complete"
    # rerun after an early crash: the latest complete dir wins; older partial dir is ignored
    p = J.pick_run([early, full], TMAX)
    assert p["status"] == "complete" and p["traj"] == full and p["n_dirs"] == 2
    # a complete older dir followed by an incomplete newer one still counts (latest COMPLETE)
    assert J.pick_run([full, early], TMAX)["traj"] == full
    # trajectories longer than T_max are truncated at T_max
    assert J.pick_run([_traj("learn", 14)], TMAX)["traj"][-1][0] == TMAX
    # exactly one dir and incomplete -> same-seed rerun, at any progress (no 50 % line)
    for tr in (early, half, _cut(_traj("learn"), 0.5), _cut(_traj("learn"), 0.95), _cut(_traj("learn"), 0.1), []):
        with pytest.raises(J.NotJudgeable):
            J.pick_run([tr], TMAX)
    # two or more dirs and none complete -> excluded and listed, at any progress
    for cs in ([early, _cut(_traj("learn"), 0.2)], [early, half], [half, _cut(_traj("learn"), 0.95)], [[], []],
               [early, half, early]):
        p = J.pick_run(cs, TMAX)
        assert p["status"] == "infra" and p["traj"] is None and p["n_dirs"] == len(cs)
    # no 50 % constant remains
    assert not hasattr(J, "INFRA_FRAC")


def test_judge_conf_infra_exclusion_and_rerun(tmp_path):
    trajs, names = _conf_trajs()
    # one run crashed early twice -> excluded, counted; another crashed once then reran fine -> judged
    crashed = "e027_conf_GN16_noz_s1"
    trajs[crashed] = [_cut(_traj("never"), 0.3), _cut(_traj("never"), 0.1)]
    rerun = "e027_conf_GN16_noz_s2"
    trajs[rerun] = [_cut(_traj("never"), 0.2), _traj("never")]
    rc, out = _judge_conf(tmp_path, trajs, names)
    assert rc == 0
    res = _json.loads(out.read_text())
    assert [x["run_name"] for x in res["infra_excluded"]] == [crashed]
    assert res["n_planned"] == 16 and res["n_judged"] == 15 and res["planned_per_cell"]["Gn"] == 4
    assert res["rule_inputs"] is not None
    row = [r for r in res["rows"] if r["run_name"] == rerun][0]
    assert row["n_dirs"] == 2
    # the cell counts drive the code: Gn has 3 judged runs
    direct = R.judge_theta({"Sz": (4, 4), "Sn": (4, 4), "Gz": (4, 4), "Gn": (0, 3)})
    assert res["code"] == direct.code and res["probs"] == direct.probs
    # a single incomplete dir -> exit 4, nothing written, below or above 50 % of T_max
    trajs[crashed] = _cut(_traj("never"), 0.3)
    out.unlink()
    rc, out2 = _judge_conf(tmp_path, trajs, names)
    assert rc == 4 and not out2.exists()
    trajs[crashed] = _cut(_traj("never"), 0.8)
    assert _judge_conf(tmp_path, trajs, names)[0] == 4
    # two incomplete dirs, the second one at 80 %: excluded and listed (no undecided band)
    trajs[crashed] = [_cut(_traj("never"), 0.3), _cut(_traj("never"), 0.8)]
    rc, out3 = _judge_conf(tmp_path, trajs, names)
    assert rc == 0 and [x["run_name"] for x in _json.loads(out3.read_text())["infra_excluded"]] == [crashed]
    # a NaN in any dir of any run fails the load (exit 1), also in an older incomplete dir
    trajs[crashed] = [[(S.EVAL_STEP, float("nan"))], _cut(_traj("never"), 0.3)]
    assert _judge_conf(tmp_path, trajs, names)[0] == 1


def test_judge_conf_whole_tier_excluded_fails_not_g_only(tmp_path):
    # theta witness, every S run excluded (two incomplete dirs each): must exit 1, never fall through to the G-only code
    trajs, names = _conf_trajs()
    for nm in names:
        if "_SN16_" in nm:
            trajs[nm] = [_cut(_traj("learn"), 0.3), _cut(_traj("learn"), 0.5)]
    rc, out = _judge_conf(tmp_path, trajs, names)
    assert rc == 1 and not out.exists()
    # one theta cell (Sz) fully excluded: also exit 1
    trajs, names = _conf_trajs()
    for nm in names:
        if "_SN16_z_" in nm:
            trajs[nm] = [_cut(_traj("learn"), 0.3), _cut(_traj("learn"), 0.5)]
    assert _judge_conf(tmp_path, trajs, names)[0] == 1


def test_judge_pilot_whole_cell_excluded_continues(tmp_path):
    # pilot mode: a (tier, N) cell whose four runs are all excluded is absent from the pooled table (documented rule)
    trajs, names = {}, []
    for tier in ("G", "S"):
        for N in (16, 32):
            for var in ("z", "noz"):
                for s_ in (9101000, 9102000):
                    nm = f"e027_pilot_{tier}N{N}_{var}_s{s_}"
                    trajs[nm] = ([_cut(_traj("learn"), 0.2), _cut(_traj("learn"), 0.4)] if (tier, N) == ("G", 16)
                                 else _traj("learn" if s_ == 9101000 else "never"))
                    names.append(nm)
    out = tmp_path / "pw.json"
    assert J.main(["--mode", "pilot", "--t-max-steps", str(TMAX), "--runs", *names, "--out", str(out)], trajs=trajs) == 0
    res = _json.loads(out.read_text())
    assert len(res["infra_excluded"]) == 4 and ("G", 16) not in {(c["tier"], c["N"]) for c in res["pooled"]}


def test_cost_conf_epochs_required(tmp_path):
    sm = tmp_path / "smoke.json"
    sm.write_text(_json.dumps({"steps": 3906, "n_evals": 2, "runs": [
        {"run_name": f"e027_smoke_GN{N}_{v}_s0", "tier": "G", "N": N, "variant": v, "train_seconds": 100.0 + N,
         "eval_seconds": 5.0, "startup_seconds": 30.0} for N in (16, 32) for v in ("z", "noz")]}))
    assert C.main(["conf", "--smoke", str(sm), "--cells", "S:16,G:16"]) == 2
    assert C.main(["conf", "--smoke", str(sm), "--cells", "S:16,G:16", "--epochs", "420",
                   "--out", str(tmp_path / "c.json")]) == 0
    assert _json.loads((tmp_path / "c.json").read_text())["steps"] == 41013


def test_judge_pilot_infra_and_usage(tmp_path):
    trajs, names = {}, []
    for tier in ("G", "S"):
        for N in (16, 32):
            for var in ("z", "noz"):
                for s_ in (9101000, 9102000):
                    nm = f"e027_pilot_{tier}N{N}_{var}_s{s_}"
                    trajs[nm] = _traj("never")
                    names.append(nm)
    trajs["e027_pilot_GN16_z_s9101000"] = [_cut(_traj("learn"), 0.1), _cut(_traj("learn"), 0.3)]
    out = tmp_path / "p.json"
    argv = ["--mode", "pilot", "--t-max-steps", str(TMAX), "--runs", *names, "--out", str(out)]
    assert J.main(argv, trajs=trajs) == 0
    res = _json.loads(out.read_text())
    assert "c3_log2tau_gate" not in res and res["n_judged"] == 15 and len(res["infra_excluded"]) == 1
    assert set(res["states_at_shorter_T"]) == {"620", "420"}
    assert res["states_at_shorter_T"]["620"]["T_steps"] == S.steps_for_epochs(620) == 60543
    assert res["states_at_shorter_T"]["420"]["T_steps"] == S.steps_for_epochs(420) == 41013
    # usage errors: T_max off the EVAL_STEP grid, a removed free flag, conf-only option in pilot mode
    assert J.main(argv[:3] + [str(TMAX + 1)] + argv[4:], trajs=trajs) == 2
    assert J.main(argv + ["--c3-log2tau-gate"], trajs=trajs) == 2
    assert J.main(argv + ["--outcome", "BSE"], trajs=trajs) == 2
    assert J.main([a for a in argv if a != "--t-max-steps" and a != str(TMAX)], trajs=trajs) == 2


def test_judge_witness_integrity(tmp_path):
    trajs, names = _conf_trajs()
    # witness covers the judge and cost code too
    assert {"exp027_judge.py", "exp027_cost.py"} <= set(J.WITNESS_FILES)
    # --outcome conflicting with the witness selection mode is refused; the same value is accepted
    assert _judge_conf(tmp_path, trajs, names, extra=("--outcome", "BSE"))[0] == 3
    assert _judge_conf(tmp_path, trajs, names, extra=("--outcome", "Y"))[0] == 0
    # witness without a selection block, or with an unsupported stat / inconsistent N -> refused
    wit = tmp_path / "ns.json"
    w = J.make_witness(SEL_THETA)
    del w["selection"]
    wit.write_text(_json.dumps(w))
    argv = ["--mode", "conf", "--t-max-steps", str(TMAX), "--runs", *names, "--out", str(tmp_path / "x.json"),
            "--witness", str(wit)]
    assert J.main(argv, trajs=trajs) == 3
    assert _judge_conf(tmp_path, trajs, names, sel=dict(SEL_THETA, stat="AL-log2tau"))[0] == 3
    assert _judge_conf(tmp_path, trajs, names, sel=dict(SEL_THETA, N_S=None))[0] == 3
    # conf runs disagree with the selection: other N, missing tier, wrong n, wrong T_max
    assert _judge_conf(tmp_path, trajs, names, sel=dict(SEL_THETA, N_G=32))[0] == 3
    assert _judge_conf(tmp_path, trajs, names, n=5)[0] == 3
    assert _judge_conf(tmp_path, trajs, names, epochs=220)[0] == 3
    g_only = [n for n in names if "_GN16_" in n]
    assert _judge_conf(tmp_path, trajs, g_only)[0] == 3                                    # theta stat needs S runs
    ok = _judge_conf(tmp_path, trajs, g_only, sel=dict(SEL_THETA, N_S=None, stat="G-only"), n=4)
    assert ok[0] == 0 and _json.loads(ok[1].read_text())["design"] == "G_only"
    assert _judge_conf(tmp_path, trajs, names, sel=dict(SEL_THETA, N_S=None, stat="G-only"))[0] == 3  # S runs not selected
    # tampering with the judge hash is caught
    wt = tmp_path / "wt.json"
    w = J.make_witness(SEL_THETA, n=4, t_max_epochs=240)
    w["sha256"]["exp027_judge.py"] = "0" * 64
    wt.write_text(_json.dumps(w))
    assert J.main(argv[:-1] + [str(wt)], trajs=trajs) == 3


def test_nonfinite_eval_rejected():
    tr = [(S.EVAL_STEP, 0.1), (2 * S.EVAL_STEP, float("nan")), (3 * S.EVAL_STEP, 0.2)]
    with pytest.raises(ValueError):
        OUT.check_finite(tr)
    with pytest.raises(ValueError):
        J.build_rows({"e027_conf_GN16_z_s1": tr})


def test_qual_ngate_matches_exact_artifact():
    d = _json.loads((REPO / "lab/21_experiments/11_lab-experiments/EXP-027_artifacts/p4_v3_c3_exact.json").read_text())
    mid = [d["rows"][str(n)]["null_pSn0.5_pGn0.5_d0.0"]["0.975"]["NEG"] for n in range(15, 20)]
    allr = [x["0.975"]["NEG"] for n in range(15, 20) for k, x in d["rows"][str(n)].items() if k.startswith("null_")]
    assert (round(min(mid), 2), round(max(mid), 2)) == (0.32, 0.39)
    assert (round(min(allr), 2), round(max(allr), 2)) == (0.09, 0.49)
    assert "0.32–0.39" in S.QUAL_NGATE and "0.09–0.49" in S.QUAL_NGATE


def test_find_log_matches_prefix_and_interleaved_names(tmp_path):
    run = "e027_pilot_GN16_z_s9101000"
    other = "e027_pilot_GN16_z_s9102000"
    (tmp_path / f"0001_0003_{run}.log").write_text("x")
    (tmp_path / f"0000_0000_e027_smoke_GN16_z_s0_z0004_{other}.log").write_text("x")
    assert C.find_log(run, str(tmp_path)).endswith(f"0001_0003_{run}.log")
    assert "_z0004_" in C.find_log(other, str(tmp_path))
    assert C.RUN_NAME.match(run) and C.RUN_NAME.match("e027_conf_SN16_noz_s30000")
    assert J.parse_name(run)["seed"] == 9101000


# ====================================================================== amendment 1: batch 1024, smoke2, infeasible options
def test_block_steps_single_source():
    assert S.block_steps() == 1953 and S.EVAL_STEP == S.block_steps(S.EVAL_EPOCHS, S.TRAIN_N, S.GLOBAL_BATCH)
    assert [S.steps_for_epochs(e) for e in (820, 620, 420, 40)] == [41 * 1953, 31 * 1953, 21 * 1953, 2 * 1953]
    assert C.SMOKE_STEPS == 2 * 1953 and C.steps_for is C.S.steps_for_epochs


def test_enqueue_smoke2_names_cover_cost_regex():
    import tools.statebench_tiers.e027_enqueue as E
    for n in E.smoke2_run_names():
        assert C.RUN_NAME.match(n) and J.RUN_NAME.match(n)


def _smoke2_obj(infeasible=()):
    o = _smoke_obj()
    o["runs"].append({"run_name": "e027_smoke2_GN48_z_s0", "tier": "G", "N": 48, "variant": "z",
                      "train_seconds": 3906 * 1.3, "eval_seconds": 140.0, "startup_seconds": 120.0})
    bad = {(t, N, v) for t, N, v in infeasible}
    o["runs"] = [r for r in o["runs"] if (r["tier"], r["N"], r["variant"]) not in bad]
    o["infeasible"] = [{"run_name": f"e027_smoke2_{t}N{N}_{v}_s0", "tier": t, "N": N, "variant": v,
                        "reason": "CUDA out of memory"} for t, N, v in infeasible]
    return o


def test_feasibility_token_rule():
    m = C.CostModel(C.load_smoke(_smoke2_obj()))             # all six feasible, largest 161 tokens (N=64)
    assert all(m.feasible(N) for N in (8, 12, 16, 24, 32, 48, 64)) and not m.feasible(65)   # no upward extrapolation
    m = C.CostModel(C.load_smoke(_smoke2_obj([("S", 64, "z")])))      # N=64 out of memory
    assert m.feasible(48) and not m.feasible(64) and not m.feasible(80)
    assert "e027_smoke2_SN64_z_s0" in m.infeasibility(64)
    m = C.CostModel(C.load_smoke(_smoke2_obj([("G", 48, "z"), ("S", 64, "z")])))
    assert m.feasible(32) and not m.feasible(48) and not m.feasible(64)
    # unfeasible point also makes a LARGER N infeasible even when a larger point was measured feasible
    o = _smoke2_obj([("G", 48, "z")])
    m = C.CostModel(C.load_smoke(o))
    assert not m.feasible(64) and m.feasible(32)
    with pytest.raises(ValueError):
        m.g_run("S", 64, "z", 1953, 1)


def test_plan_prints_and_marks_infeasible_extension_options():
    m = C.CostModel(C.load_smoke(_smoke2_obj([("S", 64, "z")])))
    p = C.plan(m, budget=1e9)
    assert p["branch"] == "FULL"
    assert p["ext_feasibility"]["S64"]["feasible"] is False and p["ext_feasibility"]["G48"]["feasible"] is True
    assert p["candidates"][0]["ext_options"]["S64"] is None and p["candidates"][0]["ext_options"]["G48"] > 0
    assert p["candidates"][0]["ext_options_infeasible"] == ["S64"]
    last = p["arithmetic"][-1]
    assert last.startswith("extension options feasible:") and "G48" in last and "S64 (" in last
    assert "infeasible" in p["arithmetic"][0]
    # interpolation after dropping the infeasible N=64 point uses the three feasible z points only


def test_plan_no_pilot_infeasible():
    m = C.CostModel(C.load_smoke(_smoke2_obj([("G", 32, "z"), ("G", 32, "noz"), ("G", 48, "z"), ("S", 64, "z")])))
    p = C.plan(m)
    assert p["branch"] == "NO_PILOT_INFEASIBLE" and p["epochs"] is None and p["confirm_possible"] is False
    assert set(p["reason"]) == {"G32", "S32"} and "NO_PILOT_INFEASIBLE" in p["arithmetic"][-1]
    # a pilot cell above the largest measured token count (smoke2 stopped at N=16) is unmeasured -> infeasible too
    o = {"steps": 3906, "n_evals": 2, "runs": [r for r in _smoke_obj()["runs"] if r["N"] == 16]}
    assert C.plan(C.CostModel(C.load_smoke(o)))["branch"] == "NO_PILOT_INFEASIBLE"
    # conf cost of an infeasible cell refuses
    with pytest.raises(ValueError):
        C.conf_cost(m, [("G", 32)], 420)


def test_ext_gate_infeasible_closes_like_over_cap():
    opts = {"G48": 10.0, "S64": None}
    r = S.ext_gate("S", 64, opts)
    assert r["action"] == "CLOSE_R7_EXT_INFEASIBLE" and r["cost_gpu_h"] is None
    assert S.ext_gate("G", 48, opts)["action"] == "ENQUEUE_EXT"
    assert S.ext_gate("G", 48, {"G48": 61.0})["action"] == "CLOSE_R7_EXT_OVER_CAP"
    # decide_budget tolerates None and reports it
    rows = S.decide_budget(1.0, [{"epochs": 820, "steps": S.steps_for_epochs(820), "pilot_17": 5.0,
                                  "ext_options": opts}], 1.0)["candidates"][0]
    assert rows["ext_options_infeasible"] == ["S64"] and rows["ext_ub_option"] == "G48"


def test_extract_records_oom_run_as_infeasible(tmp_path):
    ok = "\r".join(f"{i}/3906 [00:{i // 100:02d}<00:00, 1.00it/s]" for i in range(0, 3900, 100))
    ok += "\r3906/3906 [00:39<00:00, 1.00it/s]"
    (tmp_path / "0010_0001_e027_smoke2_GN16_z_s0.log").write_text(ok)
    (tmp_path / "0010_0002_e027_smoke2_GN32_z_s0.log").write_text(
        "Traceback (most recent call last):\ntorch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB\n")
    (tmp_path / "0010_0003_e027_smoke2_GN32_noz_s0.log").write_text("   0%|  | 12/3906 [00:20<1:00:00, 1.0it/s]\n")
    names = ["e027_smoke2_GN16_z_s0", "e027_smoke2_GN32_z_s0"]
    out = C.extract_timings(names, str(tmp_path), runtime_fn=lambda n, r: 200.0)
    assert [r["run_name"] for r in out["runs"]] == [names[0]]
    assert out["infeasible"][0]["run_name"] == names[1] and out["infeasible"][0]["error_class"] == "oom"
    assert out["infeasible"][0]["N"] == 32 and out["infeasible"][0]["variant"] == "z"
    # a log that stops early with no crash message is still running -> fails loudly
    with pytest.raises(ValueError):
        C.extract_timings(["e027_smoke2_GN32_noz_s0"], str(tmp_path), runtime_fn=lambda n, r: 1.0)
    # a crash WITHOUT an out-of-memory message is not infeasible: extract fails (v3 rule), it is never pooled
    (tmp_path / "0010_0004_e027_smoke2_GN16_noz_s0.log").write_text(
        "   0%|  | 12/3906 [00:20<1:00:00, 1.0it/s]\nTraceback (most recent call last):\nRuntimeError: NCCL timeout\n")
    with pytest.raises(ValueError, match="not infeasible"):
        C.extract_timings(["e027_smoke2_GN16_noz_s0"], str(tmp_path), runtime_fn=lambda n, r: 1.0)
    # out of memory AFTER the last tick (final eval / checkpoint save) is still infeasible
    (tmp_path / "0010_0005_e027_smoke2_GN48_z_s0.log").write_text(
        ok + "\nEVALUATE\nTraceback (most recent call last):\ntorch.OutOfMemoryError: CUDA out of memory.\n")
    o2 = C.extract_timings(["e027_smoke2_GN48_z_s0"], str(tmp_path), runtime_fn=lambda n, r: 200.0)
    assert o2["runs"] == [] and o2["infeasible"][0]["error_class"] == "oom" and o2["infeasible"][0]["N"] == 48
    # the dict goes through load_smoke / CostModel only with at least one feasible run per variant seen
    assert C.load_smoke(out)["infeasible"] == out["infeasible"]
