"""Tests for analysis/z_dynamics (EXP-017). CPU only; the tiny models are randomly initialised.

No test reads a real checkpoint or computes a real cell statistic. Two tests read NAS or dataset files
and skip when those are absent: the prereg parse and the real D1 test σ rows (never sealed/).
"""

from __future__ import annotations

import copy
import json
import math
import os

import numpy as np
import pytest
import torch

from analysis.z_dynamics import judge, readouts, stats
from analysis.z_dynamics.instrumented_forward import (forward_trace, g1_equal, g2_equal, initial_carry,
                                                      native_logits)

REPO = "/home/ayp/project/trm"
NULL_OUT = "/mnt/ayp/agents/trm/21_experiments/11_lab-experiments/EXP-017_null_check.out.json"


# =============================================================== tiny models

def tiny_cfg(**kw):
    c = dict(batch_size=4, seq_len=11, puzzle_emb_ndim=32, num_puzzle_identifiers=1, vocab_size=11,
             H_cycles=3, L_cycles=6, H_layers=0, L_layers=2, hidden_size=32, expansion=2, num_heads=4,
             pos_encodings="rope", halt_max_steps=1, halt_exploration_prob=0.1, forward_dtype="bfloat16",
             mlp_t=False, puzzle_emb_len=4, no_ACT_continue=True)
    c.update(kw)
    return c


def tiny_inner(arch):
    torch.manual_seed(0)
    if arch == "trm":
        from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1 as M
    else:
        from models.recursive_reasoning.trm_singlez import TinyRecursiveReasoningModel_ACTV1 as M
    m = M(tiny_cfg())
    with torch.no_grad():  # the zero-initialised puzzle embedding would make the prefix trivial
        m.inner.puzzle_emb.weights.normal_()
    m.eval()
    return m.inner


def tiny_batch(B=4, seed=0):
    rng = np.random.default_rng(seed)
    inp = np.stack([np.r_[rng.permutation(10) + 1, 0] for _ in range(B)]).astype(np.int32)
    return {"inputs": torch.from_numpy(inp), "puzzle_identifiers": torch.zeros(B, dtype=torch.int32),
            "labels": torch.from_numpy(inp.copy())}


@pytest.mark.parametrize("arch", ["trm", "trm_singlez"])
def test_trace_captures_exactly_18_preliminary_states_and_is_bitwise_native(arch):
    inner = tiny_inner(arch)
    batch = tiny_batch()
    with torch.inference_mode():
        tr = forward_trace(inner, batch, arch)
        ref = native_logits(inner, batch)
    assert len(tr.zHp) == 18 and len(tr.S_L) == 18 and len(tr.logits) == 18 and len(tr.raw_logits) == 18
    assert len(tr.zHn) == 3
    assert tr.j_to_c == [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20]
    assert tr.logits[0].shape == (4, 11, 11)
    assert g1_equal(tr, ref)          # G1
    assert g2_equal(tr, 6)            # G2: zHp[6h] == zHn[h]
    # non-native preliminary states are genuinely different states
    assert not torch.equal(tr.zHp[0], tr.zHp[5])


def test_trace_counts_l_level_calls():
    inner = tiny_inner("trm")
    calls = []
    h = inner.L_level.register_forward_hook(lambda m, i, o: calls.append(1))
    with torch.inference_mode():
        forward_trace(inner, tiny_batch(), "trm")
    h.remove()
    assert len(calls) == 21 + 18


def test_initial_carry_is_H_L_init():
    inner = tiny_inner("trm")
    c = initial_carry(inner, 3)
    assert torch.equal(c.z_H[1, 5], inner.H_init) and torch.equal(c.z_L[2, 0], inner.L_init)


def test_pr_matches_pr_recompute_if_available():
    from analysis.z_dynamics.pr_traj import pr_of_latent
    z = torch.randn(64, 27, 16).to(torch.bfloat16)
    try:
        from analysis.pr_recompute import pr_of_latent as ref
    except Exception:
        pytest.skip("analysis.pr_recompute not importable (gitignored main-checkout module)")
    assert pr_of_latent(z) == ref(z)


# =============================================================== readouts

def test_validity_and_trivial():
    ident = np.arange(1, 11)
    sig = np.array([2, 1, 3, 4, 5, 6, 7, 8, 9, 10])
    toks = np.stack([ident, sig, np.r_[ident[:9], 0], np.r_[ident[:9], 9]])
    assert readouts.is_valid(toks).tolist() == [True, True, False, False]
    inp = np.stack([sig] * 4)
    assert readouts.trivial(toks, inp).tolist() == [True, True, False, False]


def _perm(cycles, n=10):
    p = np.arange(n)
    for c in cycles:
        for a, b in zip(c, c[1:] + c[:1]):
            p[a] = b
    return p


def test_nearest_power_tie_none_pad():
    s = readouts.SigmaInfo(_perm([[0, 1]]), k=1)
    assert s.o == 2
    dec = np.stack([
        np.arange(10),                          # identity -> m* = 0
        s.P[1],                                 # copy -> m* = 1
        np.r_[0, 0, np.arange(2, 10)],          # 1 mismatch to both -> tie
        np.full(10, -1),                        # all pad -> none (Ham 10 > 3)
        np.r_[-1, np.arange(1, 10)],            # pad at position 0 counts as a mismatch -> m* = 0, ham 1
    ])
    near = s.nearest(dec)
    assert near["m"].tolist() == [0, 1, -1, -1, 0]
    assert near["tie"].tolist() == [False, False, True, False, False]
    assert near["none"].tolist() == [False, False, False, True, False]
    assert near["ham_min"].tolist() == [0, 0, 1, 10, 1]


def test_hamming_boundary_ham_max():
    s = readouts.SigmaInfo(_perm([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]), k=3)  # 10-cycle, ord 10
    d3 = s.P[3].copy()
    d3[[0, 1, 2]] = d3[[1, 2, 0]]          # 3 mismatches to σ^3
    d4 = s.P[3].copy()
    d4[[0, 1, 2, 3]] = d4[[1, 2, 3, 0]]    # 4 mismatches
    near = s.nearest(np.stack([d3, d4]))
    assert near["ham_min"].tolist() == [3, 4]
    assert near["m"].tolist() == [3, -1]


def test_crt_mixture_is_not_proper():
    # σ = (0 1)(2 3 4), k = 5: identity on the 2-cycle + σ^k on the 3-cycle == σ^2 exactly (CRT).
    s = readouts.SigmaInfo(_perm([[0, 1], [2, 3, 4]]), k=5)
    dec = s.P[0].copy()
    dec[[2, 3, 4]] = s.P[5 % s.o][[2, 3, 4]]
    assert np.array_equal(dec, s.P[2])
    assert s.A == set() and not s.informative
    near = s.nearest(dec[None])
    assert near["m"][0] == 2 and not s.proper(near)[0]


def test_proper_partial_counts_only_when_better_than_mixture():
    # σ = (0 1 2 3)(4 5 6), ord 12, k = 5: A = {2, 3}
    s = readouts.SigmaInfo(_perm([[0, 1, 2, 3], [4, 5, 6]]), k=5)
    assert s.o == 12 and s.A == {2, 3}
    near = s.nearest(s.P[2][None])
    assert near["m"][0] == 2 and near["mix"][0] == 4 and s.proper(near)[0]
    mix = s.P[0].copy()
    mix[[4, 5, 6]] = s.P[5][[4, 5, 6]]                  # identity on the 4-cycle, answer on the 3-cycle
    near2 = s.nearest(mix[None])
    assert near2["m"][0] == 8 and near2["mix"][0] == 0 and not s.proper(near2)[0]
    res = s.residues(np.stack([s.P[2], mix]))
    assert res[0] == [2, 0] and res[1] == [2, 2]         # 4-cycle residues, 3-cycle residues


def test_labels_are_sigma_k_g6():
    sig = _perm([[0, 1, 2, 3], [4, 5, 6]])
    inp = (sig + 1)[None]
    lab = (readouts.perm_pow(sig, 5) + 1)[None]
    assert readouts.labels_are_sigma_k(inp, lab, 5)["pass"]
    assert not readouts.labels_are_sigma_k(inp, lab, 4)["pass"]


# =============================================================== stats vs the pre-freeze null check

CANON = {
    "linear": np.linspace(0.1, 1.0, 17),
    "staircase": np.array([0.1] * 6 + [0.5] * 6 + [0.95] * 5),
    "early_plateau": np.array([0.3, 0.7] + [1.0] * 15),
    "copy_to_answer_jump": np.array([0.1] * 12 + [1.0] * 5),
    "starts_at_ceiling": np.full(17, 0.97),
    "flat": np.full(17, 0.5),
    "dip_at_cycle_boundary": np.array([0.2, 0.4, 0.6, 0.8, 0.9, 1.0, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0] + [1.0] * 5),
    "decreasing": np.linspace(1.0, 0.1, 17),
}


def test_cell_M_reproduces_null_check_sensitivity_rows():
    if not os.path.isfile(NULL_OUT):
        pytest.skip("null-check output not mounted")
    ref = json.load(open(NULL_OUT))["C-Z-1"]["sensitivity"]
    for name, a in CANON.items():
        lab = stats.cell_M(a)
        assert lab == ref[name]["class"], name
        assert (stats.shape_sub(a) if lab == "M+" else None) == ref[name]["subclass"], name


def test_cell_M_counts_boundaries_exact():
    c = np.array([5000] * 16 + [7000])        # R = 0.2 exactly, maxdrop 0
    assert stats.cell_M_counts(c, 10000)["M"] == "M+"
    c = np.array([9000] + [9000] * 16)        # a_1 = 0.9 exactly, constant -> PF
    assert stats.cell_M_counts(c, 10000)["M"] == "PF"
    c = np.array([5000] * 17)                 # constant below ceiling -> M0
    assert stats.cell_M_counts(c, 10000)["M"] == "M0"


def test_exact_enumeration():
    three = {"A": (1, 2), "B": (2, 1), "C": (1, 1)}
    r = stats.exact_stratified_rates(three)
    assert r["n_labelings"] == 18 and math.isclose(r["P_S3_1"], 1 / 18) and math.isclose(r["P_S3_2"], 1 / 18)
    assert math.isclose(r["P_union"], 2 / 18)
    assert math.isclose(stats.exact_stratified_rates({"A": (1, 2), "B": (2, 1)})["P_S3_1"], 1 / 9)
    assert math.isclose(stats.exact_stratified_rates({"B": (2, 1), "C": (1, 1)})["P_S3_1"], 1 / 6)


def test_E_L_invariance_demo():
    assert stats.E_L_from_logs(np.linspace(1.0, 1.5, 18)) == pytest.approx(0.0, abs=1e-12)
    assert stats.E_L_from_logs(np.linspace(1.0, 3.5, 18)) == pytest.approx(0.0, abs=1e-12)
    osc = np.r_[np.linspace(1.0, 2.0, 6), np.linspace(1.8, 2.6, 6), np.linspace(2.4, 1.5, 6)]
    assert stats.E_L_from_logs(osc) == pytest.approx(2.6)
    assert stats.E_L(np.exp(osc)) == pytest.approx(2.6)


def test_gamma_and_permutation_test():
    assert stats.pooled_gamma(np.full((3, 17), -1)) == 0.0
    g = stats.pooled_gamma(np.tile(np.arange(17), (5, 1)))
    assert g == 1.0
    p = stats.perm_pvalue(np.tile(np.arange(17), (20, 1)), 1.0, n_perm=50)
    assert p == 1 / 51


def test_addition_chains():
    assert [stats.addition_chain_len(k) - 1 for k in (5, 6, 7, 10)] == [2, 2, 3, 3]
    assert stats.vis_in_shortest_chain({2, 4}, 5) and not stats.vis_in_shortest_chain({3, 4}, 10)


def _real_sigmas(k, n=200):
    p = f"{REPO}/data/power_permutation/d1/{k}/test/all__inputs.npy"
    if not os.path.isfile(p):
        pytest.skip("D1 test split not present")
    return np.load(p)[:n].astype(np.int64)


def _path_decodes(infos, path, rng, J=17):
    out = np.empty((len(infos), 18, 10), np.int64)
    for i, s in enumerate(infos):
        cuts = [0] + sorted(rng.choice(np.arange(1, J), size=len(path) - 1, replace=False).tolist()) + [J]
        for t in range(len(path)):
            out[i, cuts[t]:cuts[t + 1]] = s.P[path[t] % s.o] + 1
        out[i, 17] = s.P[s.k % s.o] + 1
    return out


def _cyclewise_decodes(infos, k, rng, J=17):
    out = np.empty((len(infos), 18, 10), np.int64)
    for i, s in enumerate(infos):
        for c in s.cyc:
            t = rng.integers(1, J)
            out[i][np.ix_(np.arange(0, t), c)] = s.P[1][c] + 1
            out[i][np.ix_(np.arange(t, J), c)] = s.P[k % s.o][c] + 1
        out[i, 17] = s.P[k % s.o] + 1
    return out


def test_cz2_decode_level_sensitivity_on_real_sigmas():
    """§2.4: genuine sequential -> stepwise; cycle-by-cycle copy->answer (CRT confound) -> jump."""
    k = 5
    inp = _real_sigmas(k)
    infos = readouts.sigma_infos(inp, k)
    lab = np.stack([s.P[k % s.o] + 1 for s in infos])
    rng = np.random.default_rng(1)
    seq = stats.analyze_decodes(_path_decodes(infos, list(range(1, k + 1)), rng), inp, lab, k,
                                infos=infos, n_perm=200)["cell"]
    assert seq["Z"] == "stepwise" and seq["thr"] == 0.1 and seq["informative_frac"] >= 0.965
    cyc = stats.analyze_decodes(_cyclewise_decodes(infos, k, rng), inp, lab, k, infos=infos, n_perm=200)["cell"]
    assert cyc["Z"] == "jump" and cyc["P_int"] == 0.0


def test_tuned_lens_runs_and_starts_at_identity():
    from analysis.z_dynamics import tuned_lens
    torch.manual_seed(0)
    ans = torch.randn(16, 18, 10, 8)
    w = torch.randn(11, 8)
    lens = tuned_lens.fit_lens(ans, w, steps=0)
    dec = tuned_lens.apply_lens(ans, lens, w)
    assert dec.shape == (16, 17, 10)
    assert torch.equal(dec, (ans[:, :17] @ w.T).argmax(-1))
    lens2 = tuned_lens.fit_lens(ans, w, steps=5)
    assert len(lens2) == 17 and lens2[0][0].abs().sum() > 0


# =============================================================== cells / gates

def test_manifest_agrees_with_frozen_rules():
    from analysis.z_dynamics import cells
    cells.check_manifest()
    here = os.path.join(os.path.dirname(cells.__file__), "EXP017_cells.csv")
    assert len(cells.read_csv(here)) == 41


def test_sealed_guard():
    from analysis.z_dynamics import gates
    with pytest.raises(gates.SealedAccess):
        gates.guard_path("data/power_permutation/d1/5/sealed/test")
    gates.guard_path("data/power_permutation/d1/5/test")


# =============================================================== judge

FROZEN = judge.frozen_cells()
M_VALS = {"PF": (0.95, 0.0, 0.0, 0.0), "M+": (0.2, 0.5, 0.05, 0.5), "M-": (0.9, -0.5, 0.5, 0.05),
          "M0": (0.3, 0.0, 0.3, 0.3)}
Z_VALS = {"stepwise": (0.8, 0.001, 0.3), "jump": (0.8, 0.001, 0.01), "unordered": (0.1, None, 0.3)}


def mk_rec(name, **over):
    fcls = FROZEN[name]
    arch = judge.arch_of(name)
    final = {"G": 1.0, "C": 0.02, "P": 0.5}[fcls]
    rec = {"run_name": name, "arch": arch, "status": "analysed", "final_probe_test_exact": final,
           "G0": True, "G1": True, "G2": True, "G6": True, "G7": True, "I3_ok": True, "I4_ok": True,
           "G3": {"pr_zH": 10.0, "pr_zH_snapshot": 10.0, "pr_zL": 5.0, "pr_zL_snapshot": 5.0} if arch == "trm" else None,
           "G4": {"test512_exact": final, "train512_exact": 1.0, "snapshot_train_exact": 1.0},
           "G7_values": {"arch.name": arch, "epochs": 1, "data_root": "d"}}
    rec.update(over)
    return rec


def mk_stat(name, M="M+", Z="stepwise", V=0.8, T=0.1, E_L=1.0):
    a1, R, md, mr = M_VALS[M]
    g, p, pint = Z_VALS[Z]
    row = {"run_name": name, "split": "test", "readout_kind": "prelim", "a1": a1, "R": R, "maxdrop": md,
           "maxrise": mr, "M": M, "V": V, "T": T, "gamma": g, "p": "" if p is None else p, "P_int": pint,
           "P_int0": 0.001, "thr": 0.1, "Z": "PF" if M == "PF" else Z}
    tr = {"run_name": name, "split": "train", "readout_kind": "prelim", "E_L": E_L}
    return [row, tr]


def world(M="M+", Z="stepwise", V=0.8, T=0.1, EL_G=1.0, EL_C=2.0, rec_over=None, stat_over=None):
    cells = {n: mk_rec(n, **(rec_over or {}).get(n, {})) for n in FROZEN}
    rows = []
    for n, c in FROZEN.items():
        kw = dict(M=M, Z=Z, V=V, T=T, E_L=EL_G if c == "G" else EL_C)
        kw.update((stat_over or {}).get(n, {}))
        rows += mk_stat(n, **kw)
    g5 = {"unit": "pp_base_tf_z_iter_k5_s1", "ran": True, "identical": True}
    return cells, rows, g5


def tokens(v):
    return ({f: x["token"] for f, x in v["C-Z-1"].items()}, {f: x["token"] for f, x in v["C-Z-2"].items()},
            v["C-Z-3"]["token"])


@pytest.mark.parametrize("kw,t1,t2", [
    (dict(M="M+", Z="stepwise", V=0.8, T=0.1), "S1-1", "S2-1"),
    (dict(M="M+", Z="jump", V=0.8, T=0.9), "S1-2", "S2-2"),
    (dict(M="M+", Z="unordered", V=0.2), "S1-3", "S2-3"),
    (dict(M="M+", V=0.01), "S1-4", "S2-1"),
    (dict(M="M0"), "S1-5", "S2-1"),
    (dict(M="M-"), "S1-6", "S2-1"),
    (dict(M="PF"), "S1-0", "S2-0"),
])
def test_judge_family_branches(kw, t1, t2):
    v, code = judge.judge_records(*world(**kw))
    c1, c2, c3 = tokens(v)
    assert set(c1.values()) == {t1} and set(c2.values()) == {t2}
    assert code == judge.EXIT_OK and c3 == "S3-1"
    if t1 == "S1-6":
        assert all(x.get("uncovered_mechanism") for x in v["C-Z-1"].values())


def test_judge_boundary_values_inclusive():
    # V median exactly 0.50 -> V+; T median exactly 0.50 -> trivial (S1-2)
    v, _ = judge.judge_records(*world(V=0.5, T=0.5))
    assert set(tokens(v)[0].values()) == {"S1-2"}


def test_judge_pf_minority_votes_over_non_pf():
    f1 = judge.rule("families")["F1_trm_D1"]
    over = {n: {"M": "PF"} for n in f1[:9]}      # 9 of 15 PF < ceil(30/3)=10 -> votes over n'=6
    v, _ = judge.judge_records(*world(M="M+", stat_over=over))
    x = v["C-Z-1"]["F1_trm_D1"]
    assert x["token"] == "S1-1" and x["n_PF"] == 9 and x["n_prime"] == 6 and x["need"] == 4


def test_judge_invalid_gate_and_empty():
    f1 = judge.rule("families")["F1_trm_D1"]
    f2 = judge.rule("families")["F2_singlez_D1"]
    over = {n: {"G1": False} for n in f1[:6]}                      # 6 > floor(15/3) = 5
    over.update({n: {"status": "excluded", "exclusion_reason": "I-3"} for n in f2})
    v, code = judge.judge_records(*world(rec_over=over))
    c1, c2, c3 = tokens(v)
    assert c1["F1_trm_D1"] == "INVALID-GATE" and c2["F1_trm_D1"] == "INVALID-GATE"
    assert c1["F2_singlez_D1"] == "INVALID-EMPTY" and c1["F3_singlez_D0"] == "S1-1"
    assert c3 == "S3-0"            # pp_base_tf_z_iter_k5_s1 (S_A G) is among the excluded
    assert code == judge.EXIT_EXCLUSIONS
    over5 = {n: {"G1": False} for n in f1[:5]}                     # exactly 5 -> still judged
    v5, _ = judge.judge_records(*world(rec_over=over5))
    assert v5["C-Z-1"]["F1_trm_D1"]["token"] == "S1-1" and v5["C-Z-1"]["F1_trm_D1"]["n"] == 10


@pytest.mark.parametrize("elg,elc,tok", [(1.0, 2.0, "S3-1"), (2.0, 1.0, "S3-2"), (1.0, 1.0, "S3-3"),
                                         (0.0, 0.0, "S3-3")])
def test_judge_cz3_branches(elg, elc, tok):
    v, _ = judge.judge_records(*world(EL_G=elg, EL_C=elc))
    assert tokens(v)[2] == tok


def test_judge_cz3_underfit_C_forces_S3_0():
    over = {"pp_seedext_tf_noz_iter_k7_s3": {"G4": {"test512_exact": 0.02, "train512_exact": 0.5,
                                                    "snapshot_train_exact": 0.5}}}
    v, code = judge.judge_records(*world(rec_over=over))
    assert tokens(v)[2] == "S3-0" and v["cells"]["pp_seedext_tf_noz_iter_k7_s3"]["class"] == "U"
    assert code == judge.EXIT_EXCLUSIONS


def test_judge_g3_g4_tolerances_from_frozen_rules():
    ok = {"pr_zH": 10.2, "pr_zH_snapshot": 10.0, "pr_zL": 5.0, "pr_zL_snapshot": 5.0}   # 0.02 exactly
    bad = {"pr_zH": 10.3, "pr_zH_snapshot": 10.0, "pr_zL": 5.0, "pr_zL_snapshot": 5.0}
    assert judge.g3_pass(ok, "trm") and not judge.g3_pass(bad, "trm") and judge.g3_pass(None, "trm_singlez")
    assert judge.g4_pass({"test512_exact": 0.995, "train512_exact": 1.0, "snapshot_train_exact": 0.992}, 1.0)
    assert not judge.g4_pass({"test512_exact": 0.98, "train512_exact": 1.0, "snapshot_train_exact": 1.0}, 1.0)


def test_judge_uncovered_on_nan():
    f3 = judge.rule("families")["F3_singlez_D0"]
    v, _ = judge.judge_records(*world(stat_over={f3[0]: {"V": float("nan")}}))
    assert v["C-Z-1"]["F3_singlez_D0"]["token"] == "UNCOVERED"


def test_judge_halts_g5_and_i11():
    cells, rows, g5 = world()
    v, code = judge.judge_records(cells, rows, dict(g5, identical=False))
    assert code == judge.EXIT_HALT and tokens(v)[2] == "HALT"
    v, code = judge.judge_records(cells, rows, None)
    assert code == judge.EXIT_HALT
    names = list(FROZEN)
    bad_g4 = {"test512_exact": 0.5, "train512_exact": 1.0, "snapshot_train_exact": 1.0}
    over4 = {n: {"G4": bad_g4} for n in names[:4]}
    _, code4 = judge.judge_records(*world(rec_over=over4))
    assert code4 == judge.EXIT_HALT                                  # 4 > 3
    over3 = {n: {"G4": bad_g4} for n in names[:3]}
    v3, code3 = judge.judge_records(*world(rec_over=over3))
    assert code3 == judge.EXIT_EXCLUSIONS and "HALT" not in v3


def test_judge_aborts_on_missing_record_and_label_mismatch():
    cells, rows, g5 = world()
    del cells["pp_base_tf_z_iter_k3_s1"]
    with pytest.raises(judge.JudgeAbort):
        judge.judge_records(cells, rows, g5)
    cells, rows, g5 = world()
    rows[0]["M"] = "M0"            # stats label disagrees with the frozen rule on the numbers
    with pytest.raises(judge.JudgeAbort):
        judge.judge_records(cells, rows, g5)


def test_judge_refuses_threshold_not_in_frozen_rules():
    with pytest.raises(judge.FrozenRuleMissing):
        judge.rule("czz1.NOT_A_THRESHOLD")
    with pytest.raises(judge.FrozenRuleMissing):
        judge.num("czz2.chance")                 # a string is not a numeric threshold
    rules = copy.deepcopy(judge.FROZEN_RULES)
    del rules["czz1"]["R_MIN"]
    with pytest.raises(judge.FrozenRuleMissing):
        judge.judge_records(*world(), rules=rules)
    rules = copy.deepcopy(judge.FROZEN_RULES)
    del rules["gates"]["G4_ABS_TOL"]
    with pytest.raises(judge.FrozenRuleMissing):
        judge.judge_records(*world(), rules=rules)


def test_judge_parses_numbers_from_rule_strings():
    assert judge.need_fraction("czz1.family_need") == (2, 3)
    assert judge.need_fraction("czz2.family_PF_precedence") == (2, 3)
    assert judge.exclusion_divisor() == 3 and judge.boundary_diff_indices() == [5, 11] and judge.j_stat() == 17
    assert [judge.ceil_frac(n, (2, 3)) for n in (1, 3, 8, 15)] == [1, 2, 6, 10]
    assert [judge.outcome_class(x, t) for x, t in [(0.9, 1.0), (0.05, 0.9), (0.05, 0.89), (0.0625, 1.0)]] == \
        ["G", "C", "U", "P"]


def test_judge_cli_end_to_end(tmp_path, monkeypatch):
    if not os.path.isfile(judge.DEFAULT_PREREG):
        pytest.skip("prereg not mounted")
    import csv
    cells, rows, g5 = world()
    (tmp_path / "gates.json").write_text(json.dumps({"cells": cells, "G5": g5}))
    (tmp_path / "run_manifest.json").write_text(json.dumps({"status": "complete", "smoke": False}))
    fields = sorted({k for r in rows for k in r})
    with open(tmp_path / "cell_stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    out = tmp_path / "verdict.json"
    assert judge.main(["--data", str(tmp_path), "--out", str(out)]) == judge.EXIT_OK
    v = json.loads(out.read_text())
    assert v["prereg"]["yaml_equal"] and v["C-Z-3"]["token"] == "S3-1"
    # a partial run manifest is never judged (I-8)
    (tmp_path / "run_manifest.json").write_text(json.dumps({"status": "running"}))
    assert judge.main(["--data", str(tmp_path), "--out", str(tmp_path / "v2.json")]) == judge.EXIT_ABORT
    assert not (tmp_path / "v2.json").exists()
    # a threshold missing from FROZEN_RULES -> refusal
    (tmp_path / "run_manifest.json").write_text(json.dumps({"status": "complete"}))
    broken = copy.deepcopy(judge.FROZEN_RULES)
    del broken["czz2"]["GAMMA_MIN"]
    monkeypatch.setattr(judge, "FROZEN_RULES", broken)
    assert judge.main(["--data", str(tmp_path), "--out", str(tmp_path / "v3.json")]) == judge.EXIT_ABORT


def test_prereg_block_equals_embedded_rules():
    if not os.path.isfile(judge.DEFAULT_PREREG):
        pytest.skip("prereg not mounted")
    info = judge.check_prereg(judge.DEFAULT_PREREG)
    assert info["yaml_equal"] and info["anchors_checked"] >= 8
    with pytest.raises(judge.JudgeAbort):
        judge.check_prereg(judge.DEFAULT_PREREG, expected_sha="0" * 64)


# =============================================================== run_zdyn integration (tiny random model)

def test_process_cell_end_to_end_on_tiny_random_model(tmp_path, monkeypatch):
    """Exercise run_zdyn.process_cell on a fake run dir holding a tiny random-init trm (no real checkpoint).
    It reads only the D1 k=5 test/train rows (never sealed/)."""
    import argparse
    import csv
    import io
    import shutil

    import yaml

    import models.recursive_reasoning.trm as mt
    from analysis.z_dynamics import cells, run_zdyn
    from measure_rho import build_model_config
    from models.losses import ACTLossHead

    data = "data/power_permutation/d1/5"
    if not os.path.isdir(os.path.join(REPO, data, "test")):
        pytest.skip("D1 data not present")
    monkeypatch.chdir(REPO)
    name = "pp_base_tf_z_iter_k5_s1"
    row = next(r for r in cells.rows() if r["run_name"] == name)
    run_dir = tmp_path / name
    (run_dir / "z_snapshots").mkdir(parents=True)
    arch_cfg = {k: v for k, v in tiny_cfg().items()
                if k not in ("batch_size", "seq_len", "vocab_size", "num_puzzle_identifiers")}
    arch_cfg["name"] = "recursive_reasoning.trm@TinyRecursiveReasoningModel_ACTV1"
    (run_dir / "all_config.yaml").write_text(yaml.safe_dump({"arch": arch_cfg, "data_paths": [data]}))
    shutil.copy(mt.__file__, run_dir / "trm.py")
    torch.manual_seed(0)
    cfg = build_model_config(str(run_dir), batch_size=8)
    wrapped = ACTLossHead(mt.TinyRecursiveReasoningModel_ACTV1(cfg), loss_type="stablemax_cross_entropy")
    torch.save({"_orig_mod." + k: v for k, v in wrapped.state_dict().items()}, run_dir / "step_122050")
    probe_n = 16
    labels = torch.from_numpy(np.load(os.path.join(REPO, data, "train", "all__labels.npy"))[:probe_n].astype(np.int32))
    labels[labels == 0] = -100
    torch.save({"z_H": torch.randn(probe_n, 15, 32).to(torch.bfloat16),
                "z_L": torch.randn(probe_n, 15, 32).to(torch.bfloat16),
                "labels": labels, "correct_mask": torch.zeros(probe_n, dtype=torch.bool)},
               run_dir / "z_snapshots" / "step_122050.pt")
    a = argparse.Namespace(ckpt_root=str(tmp_path), device="cpu", batch=8, test_n=20, probe_n=probe_n,
                           lens_train_n=24, n_perm=20)
    closeout = {name: {"run_id": row["run_id"], "final_probe_test_exact": str(row["final_probe_test_exact"]),
                       "traj_class": row["traj_class"]}}
    monkeypatch.setattr("analysis.z_dynamics.tuned_lens.LENS_STEPS", 3)
    out = run_zdyn.process_cell(row, a, closeout)
    rec = out["rec"]
    assert rec["status"] == "analysed" and rec["G0"] and rec["G1"] and rec["G2"] and rec["G6"] and rec["I3_ok"]
    assert rec["I4_ok"] and not rec["G7"]          # the tiny config deliberately violates the G7 key list
    kinds = {(r["split"], r["readout_kind"]) for r in out["cell"]}
    assert kinds == {("test", "prelim"), ("test", "raw_zL"), ("test", "lens"), ("train", "prelim"),
                     ("train", "raw_zL")}
    assert {r["readout_kind"] for r in out["per_call"]} == {"prelim", "raw_zL", "native", "lens"}
    assert len(out["near"]) == (20 + probe_n) * 18
    assert len([r for r in out["pr"] if r["split"] == "train" and r["stream"] == "z_L"]) == 18
    tr = next(r for r in out["cell"] if r["split"] == "train" and r["readout_kind"] == "prelim")
    assert math.isfinite(tr["E_L"])
    txt = run_zdyn.rows_to_text(out["cell"], run_zdyn.CELL_FIELDS)
    assert txt.count("\n") == 5
    # the judge's independent recomputation agrees with the stats labels on these rows
    for r in csv.DictReader(io.StringIO(",".join(run_zdyn.CELL_FIELDS) + "\n" + txt)):
        M = judge.recompute_M(r)
        assert M == r["M"] and judge.recompute_Z(r, M) == r["Z"]
