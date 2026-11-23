"""Tests for tools/statebench_tiers/pilot_report.py (synthetic trajectories injected; no wandb, no GPU)."""
import json
import os
import sys

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "tools", "statebench_tiers"))
import pilot_report as PR  # noqa: E402
import exp027_judge as J  # noqa: E402
import exp027_select as S  # noqa: E402
import outcomes as O  # noqa: E402

E = S.EVAL_STEP
TP = S.steps_for_epochs(420)


def traj(*es, d=E):
    return [((i + 1) * d, float(e)) for i, e in enumerate(es)]


def full(prefix, n=21, fill=0.0):
    es = list(prefix) + [fill] * (n - len(prefix))
    return traj(*es)


def synth():
    """17 pilot runs: G16 learns (z early, noz late), G32 never, S16 learns then collapses (one), S32 graded; A16 flat."""
    t = {}
    for tier, N, v, s in S.pilot_run_specs():
        nm = f"e027_pilot_{tier}N{N}_{v}_s{s}"
        if tier == "A":
            t[nm] = full([0.0] * 5)
        elif tier == "G" and N == 16:
            t[nm] = full([0.0, 0.1, 0.95, 0.96], 21, 0.97) if v == "z" else full([0.0] * 10 + [0.95, 0.95], 21, 0.95)
        elif tier == "G":
            t[nm] = full([0.0, 0.01, 0.02], 21, 0.01)
        elif tier == "S" and N == 16:
            t[nm] = full([0.0, 0.95, 0.95, 0.3], 21, 0.2) if s == 9101000 else full([0.95, 0.95], 21, 0.96)
        else:
            t[nm] = full([0.0, 0.3, 0.5, 0.45], 21, 0.5)
    return t


def test_default_names_are_the_17_pilot_runs():
    n = PR.default_names()
    assert len(n) == 17 and len(set(n)) == 17
    assert "e027_pilot_AN16_z_s9101000" in n
    for x in n:
        PR.parse_meta(x)


def test_parse_meta_refuses_conf_and_accepts_sigmak():
    with pytest.raises(ValueError):
        PR.parse_meta("e027_conf_GN16_z_s1000")
    m = PR.parse_meta("H108N2_fig1_tf_noz_iter_k7_s2")
    assert (m["tier"], m["N"], m["variant"], m["seed"]) == ("K", 7, "noz", 2)


def test_rows_match_judge_build_rows_and_outcomes():
    tr = synth()
    ref = {r["run_name"]: r for r in J.build_rows(tr)}
    for nm, t in tr.items():
        r = PR.make_row(nm, t, PR.parse_meta(nm), E, TP)
        j = ref[nm]
        for k in ("Y", "BSE", "tau", "max", "final", "mean_last5", "stability", "n_evals", "Y_sens"):
            assert r[k] == j[k], (nm, k)
        assert [f for f, k in (("graded", "graded"), ("collapse", "collapse"), ("pretakeoff", "pretakeoff"))
                if r[k]] == j["flags"]
        assert r["Y"] == O.Y(t) and r["tau"] == O.tau(t)


def test_classes_and_cell_table():
    tr = synth()
    rows = [PR.make_row(n, t, PR.parse_meta(n), E, TP) for n, t in sorted(tr.items())]
    cells = {c["cell"]: c for c in PR.cell_table(rows)}
    assert set(cells) == {"A16", "S16", "S32", "G16", "G32"}
    g16 = cells["G16"]
    assert (g16["n"], g16["n_y1"], g16["state_Y"]) == (4, 4, "CEIL")
    assert g16["n_censored_at_Tp"] == 0 and len(g16["tau_over_eval_step"]) == 4
    s16 = cells["S16"]
    assert s16["n"] == 4 and s16["n_y1"] == 4 and s16["n_collapse"] == 2
    assert s16["classes"]["learned-then-collapsed"] == 2 and s16["classes"]["learned-stable"] == 2
    assert cells["G32"]["state_Y"] == "FLOOR" and cells["G32"]["n_censored_at_Tp"] == 4
    assert cells["S32"]["classes"]["graded"] == 4 and cells["S32"]["state_Y"] == "FLOOR"
    for c in cells.values():
        assert sum(c["classes"].values()) == c["n"]
        assert sum(c["BSE_hist"]) == c["n"] and sum(c["max_hist"]) == c["n"]


def test_sensitivity_flags_state_change():
    # two evals at 0.92, 0.92 then 0.3: Y=1 at 0.9/consec2, 0.5; both pass.  A run with 0.93,0.5,... only consec1 passes
    rows = [PR.make_row("e027_pilot_GN16_z_s9101000", full([0.93, 0.5]), PR.parse_meta("e027_pilot_GN16_z_s9101000"), E, TP),
            PR.make_row("e027_pilot_GN16_noz_s9101000", full([0.0]), PR.parse_meta("e027_pilot_GN16_noz_s9101000"), E, TP)]
    c = PR.cell_table(rows)[0]
    assert c["state_Y"] == "FLOOR" and c["sensitivity"]["consec1"]["state_Y"] == "MIX"
    assert c["sensitivity"]["consec1"]["changed_vs_base"] and c["n_changed_variants"] >= 1


def test_y_of_T_matches_outcomes_Y():
    tr = synth()
    rows = [PR.make_row(n, t, PR.parse_meta(n), E, TP) for n, t in sorted(tr.items())]
    grid = PR.eval_grid(tr, E)
    assert grid == [E * i for i in range(2, 22)]
    yc = PR.y_of_T(rows, tr, grid)
    for p in yc:
        want = sum(O.Y(tr[r["run_name"]], p["T"]) for r in rows if PR.cell_label(r["tier"], r["N"]) == p["cell"])
        assert p["n_y1"] == want
    g16 = [p["n_y1"] for p in yc if p["cell"] == "G16"]
    assert g16 == sorted(g16) and g16[-1] == 4


def test_a_run_check_terms():
    tr = synth()
    rows = [PR.make_row(n, t, PR.parse_meta(n), E, TP) for n, t in sorted(tr.items())]
    a = PR.a_run_check(rows)
    assert a["A_sum_Y"] == 0 and a["term_no_A_run_has_Y1"] is True
    assert a["term_all_GS_max_below_threshold"] is False and a["R0_max_threshold"] == S.R0_MAX


def test_summarize_train_bins_and_final():
    rec = [(i, 1.0 / (i + 1), 0.5, 0.6) for i in range(1000)]
    s = PR.summarize_train(rec, 999, bins=10, runtime=7200.0)
    assert len(s["steps"]) == 10 and s["runtime_s"] == 7200.0
    assert s["final_train_loss"] == pytest.approx(sum(1.0 / (i + 1) for i in range(950, 1000)) / 50)


def test_collect_picks_complete_dir_and_reports_waiting_and_infra():
    ok = full([0.0, 0.95, 0.95])
    cut = ok[:5]
    inj = {"a": [cut, ok], "b": cut, "c": [cut, cut[:3]]}
    got = PR.collect(["a", "b", "c"], TP, injected=inj)
    assert set(got["runs"]) == {"a"} and got["runs"]["a"]["run_dir"].endswith("-1")
    assert [w["run_name"] for w in got["waiting"]] == ["b"]
    assert [e["run_name"] for e in got["excluded"]] == ["c"]


def _run(tmp_path, extra=(), names=None):
    tr = synth()
    inj = {n: {"cands": t, "train": PR.summarize_train([(i, 0.1, 0.9, 0.9) for i in range(0, TP, 50)], TP,
                                                      runtime=3600.0 * 5 + 7 * k)}
           for k, (n, t) in enumerate(tr.items())}
    out = str(tmp_path / "o")
    rc = PR.main(["--t-max-steps", str(TP), "--out-dir", out, "--no-prediction", *extra], injected=inj)
    return rc, out


def test_end_to_end_variant_blind_by_default(tmp_path):
    rc, out = _run(tmp_path)
    assert rc == 0
    files = set(os.listdir(out))
    assert {"per_run.csv", "per_cell.json", "y_of_T.csv", "cost.json", "a_run_check.json", "meta.json",
            "F1_eval_trajectories.png", "F1_eval_trajectories.pdf", "F2_eval_logit_exploratory.png",
            "F3a_train_loss.png", "F4_Y_of_T.png", "F5_cost.png", "data_properties.json"} <= files
    assert "per_variant.json" not in files and "y_of_T_per_variant.json" not in files
    meta = json.load(open(os.path.join(out, "meta.json")))
    assert meta["n_judged"] == 17 and meta["variant_split"] is False
    assert not [f for f in files if f.endswith((".md", ".txt"))]


def test_variant_split_needs_matching_sha(tmp_path):
    w = tmp_path / "w.json"
    w.write_text('{"x": 1}')
    sha = PR.sha256_file(str(w))
    assert _run(tmp_path, ["--selection-witness", str(w), "--witness-sha256", "0" * 64])[0] == 3
    assert _run(tmp_path, ["--selection-witness", str(w)])[0] == 2
    rc, out = _run(tmp_path, ["--selection-witness", str(w), "--witness-sha256", sha])
    assert rc == 0 and "per_variant.json" in os.listdir(out)
    pv = json.load(open(os.path.join(out, "per_variant.json")))
    assert all(p["variant"] in ("z", "noz") for p in pv)


def test_names_override_refuses_e027():
    assert PR.main(["--t-max-steps", "100", "--names-override", "e027_pilot_GN16_z_s1=abcd1234"]) == 3


def test_selection_record_copies_judge_fields():
    j = {"selection": {"path": "R3", "trace": ["R3"], "mode": "Y", "N_S": 16, "N_G": 16, "late_flag": False},
         "pooled": [{"tier": "G", "N": 16, "n": 4, "n_y1": 2, "n_graded_pilot": 0, "n_pretakeoff_y0": 0,
                     "max_pilot": 1.0}], "infra_excluded": [], "rows": [{"run_name": "x", "n_dirs": 1}]}
    r = PR.selection_record(j, {"action": "ENQUEUE_EXT"})
    assert r["selection"]["path"] == "R3" and r["late_flag"] is False and r["ext_gate"]["action"] == "ENQUEUE_EXT"
    assert r["rows_n_dirs"] == {"x": 1}
