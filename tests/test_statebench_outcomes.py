"""Tests for tools/statebench_tiers/outcomes.py (synthetic trajectories + saved C1 JSON fixture)."""
import json
import os
import sys

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "tools", "statebench_tiers"))
import outcomes as oc  # noqa: E402

C1 = os.path.join(REPO, "lab/21_experiments/11_lab-experiments/EXP-027_artifacts/c1_sigmak_traj_tally.json")


def T(*es, d=100):
    return [((i + 1) * d, float(e)) for i, e in enumerate(es)]


def test_tau_basic_and_boundary_exact_090():
    t = T(0.1, 0.9, 0.9, 0.2)
    assert oc.tau(t) == 200 and oc.Y(t) == 1
    assert oc.tau(T(0.1, 0.899, 0.9)) is None  # next value below on the 0.001 eval grid


def test_float32_logged_values_hit_thresholds():
    # pretrain.py logs exact_accuracy in float32: 0.900 -> 0.8999999761581421, 0.950 -> 0.949999988079071
    import numpy as np
    f = lambda x: float(np.float32(x))
    assert f(0.9) < 0.9 and f(0.95) < 0.95          # the trap
    t = T(0.1, f(0.95), f(0.9), 0.2)
    assert oc.Y(t) == 1 and oc.tau(t) == 200
    assert oc.stability(T(f(0.9), f(0.9), f(0.9))) == 1.0
    assert oc.graded(T(f(0.9), f(0.9))) is False     # learned, not graded
    assert oc.graded(T(f(0.899), f(0.899))) is True
    assert oc.graded(T(f(0.2), f(0.2))) is True and oc.graded(T(f(0.199), f(0.199))) is False
    assert oc.collapse(T(0.95, 0.95, f(0.5))) is False and oc.collapse(T(0.95, 0.95, f(0.499))) is True
    assert oc.pre_takeoff(T(*([0.0] * 6), 0.0, 0.0, 0.0, 0.0, f(0.1))) is True
    assert oc.pre_takeoff(T(*([0.0] * 6), 0.0, 0.0, 0.0, 0.0, f(0.099))) is False
    assert oc.ge(f(0.9), 0.9) and not oc.ge(f(0.899), 0.9)


def test_c1_noz_k8_s3_float32_pair():
    # yz96drig (noz k8 s3) has an eval logged at exactly float32(0.900) at step 122050; with the float32
    # tolerance its first qualifying pair starts at 117168 (it was 126932 under a bare >= 0.9).
    if not os.path.exists(C1):
        pytest.skip("C1 json not generated")
    r = {x["wandb_id"]: x for x in json.load(open(C1))["runs"]}.get("yz96drig")
    if r is None:
        pytest.skip("run not in C1 json")
    assert r["tau"] == 117168 and r["Y"] == 1


def test_single_eval_above_at_last_eval():
    t = T(0.1, 0.2, 0.95)
    assert oc.tau(t) is None and oc.Y(t) == 0
    assert oc.BSE(t) == pytest.approx(0.2)


def test_non_consecutive_not_learned():
    t = T(0.95, 0.1, 0.95, 0.1)
    assert oc.Y(t) == 0 and oc.BSE(t) == pytest.approx(0.1)


def test_Y_T_cut_between_the_two_evals():
    t = T(0.1, 0.95, 0.95)  # pair at steps 200, 300
    assert oc.Y(t, T=300) == 1
    assert oc.Y(t, T=299) == 0
    assert oc.Y(t, T=200) == 0
    assert oc.Y(t, T=None) == 1
    assert oc.tau(t, T=299) is None


def test_bse_max_final_mean_last5():
    t = T(0.1, 0.5, 0.4, 0.9, 0.3, 0.2, 0.1)
    assert oc.BSE(t) == pytest.approx(0.4)  # min(0.5,0.4)=0.4 > min(0.4,0.9)=0.4 (tie) ; others lower
    assert oc.max_exact(t) == 0.9 and oc.final(t) == 0.1
    assert oc.mean_last(t, 5) == pytest.approx((0.4 + 0.9 + 0.3 + 0.2 + 0.1) / 5)
    assert oc.BSE(T(0.7)) is None
    assert oc.mean_last(T(0.2, 0.4)) == pytest.approx(0.3)


def test_stability():
    t = T(0.95, 0.95, 0.95, 0.2, 0.95, 0.1)  # tau=100; after: .95,.95,.2,.95,.1 -> 3/5
    assert oc.stability(t) == pytest.approx(0.6)
    assert oc.stability(T(0.1, 0.95, 0.95)) == 1.0  # E_{i+1} of the pair is always after tau
    assert oc.stability(T(0.95, 0.95)) == 1.0
    assert oc.stability(T(0.1, 0.2, 0.3)) is None


def test_collapse_flag():
    assert oc.collapse(T(0.95, 0.95, 0.4)) is True
    assert oc.collapse(T(0.95, 0.95, 0.5)) is False  # 0.5 is not < 0.5
    assert oc.collapse(T(0.95, 0.95, 0.7)) is False
    assert oc.collapse(T(0.1, 0.1, 0.0)) is False  # Y=0
    assert oc.collapse(T(0.2, 0.95, 0.95, 0.0)) is True
    assert oc.collapse(T(0.2, 0.0, 0.95, 0.95)) is False  # drop before tau does not count


def test_graded_flag():
    assert oc.graded(T(0.3, 0.3, 0.1)) is True  # BSE 0.3
    assert oc.graded(T(0.2, 0.2)) is True  # BSE exactly 0.20 included
    assert oc.graded(T(0.19, 0.19)) is False
    assert oc.graded(T(0.89, 0.89)) is True
    assert oc.graded(T(0.95, 0.95)) is False  # Y=1
    assert oc.graded(T(0.9, 0.9)) is False


def test_pre_takeoff_flag():
    base = [0.0] * 6
    assert oc.pre_takeoff(T(*base, 0.0, 0.0, 0.05, 0.1, 0.3)) is True
    assert oc.pre_takeoff(T(*base, 0.0, 0.0, 0.05, 0.1, 0.09)) is False  # last < 0.10
    assert oc.pre_takeoff(T(0.5, *base, 0.0, 0.0, 0.0, 0.0, 0.2)) is False  # earlier max higher
    assert oc.pre_takeoff(T(0.1, 0.2, 0.3, 0.4, 0.5)) is False  # <= 5 evals: no earlier eval
    assert oc.pre_takeoff(T(*base, 0.95, 0.95, 0.95, 0.95, 0.95)) is False  # Y=1


def test_thresholds_are_parameters():
    th = oc.Thresholds(learn=0.5)
    t = T(0.6, 0.6)
    assert oc.Y(t) == 0 and oc.Y(t, th=th) == 1
    assert oc.tau(t, th=th) == 100


def test_summarize_keys():
    s = oc.summarize(T(0.95, 0.95, 0.1))
    assert s["tau"] == 100 and s["Y"] == 1 and s["collapse"] is True and s["n_evals"] == 3
    assert set(s) == {"n_evals", "tau", "Y", "BSE", "max", "final", "mean_last5", "stability", "collapse", "graded", "pre_takeoff"}


def test_clean_trajectory_dedup_sort():
    assert oc.clean_trajectory([(300, 0.3), (100, 0.1), (100, 0.15), (200, 0.2)]) == [(100, 0.15), (200, 0.2), (300, 0.3)]


def test_loader_fails_loudly_on_unknown_run():
    with pytest.raises(LookupError):
        oc.load_trajectories(["no_such_run_name_zzz"])
    assert oc.main(["no_such_run_name_zzz"]) == 1


def test_loader_empty_history(tmp_path):
    # a file that is not a wandb datastore must raise (truncation/header error), never return []
    p = tmp_path / "x.wandb"
    p.write_bytes(b"")
    with pytest.raises(Exception):
        oc.load_trajectory_file(str(p))


@pytest.mark.skipif(not os.path.exists(C1), reason="C1 json not generated")
def test_c1_fixture_ccezqakd_and_internal_consistency():
    d = json.load(open(C1))
    by = {r["wandb_id"]: r for r in d["runs"]}
    r = by["ccezqakd"]
    assert (r["variant"], r["k"], r["seed"]) == ("z", 7, 2)
    assert r["tau"] == 107404 and r["Y"] == 1 and r["collapse"] is True
    assert r["final"] == pytest.approx(0.029, abs=1e-3)
    assert d["aggregate"]["n_runs"] == len(d["runs"])
    for r in d["runs"]:
        if "error" in r:
            continue
        assert r["Y"] == int(r["tau"] is not None)
        assert r["collapse"] is False or r["Y"] == 1
        assert not (r["graded"] and r["Y"])
