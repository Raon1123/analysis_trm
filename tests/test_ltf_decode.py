"""Tests for analysis/ltf_decode (P-C1 pre-freeze instrument). CPU only, synthetic / random weights only.

No test opens any pp_ltf_* checkpoint weight or any pp_base_tf_* checkpoint weight. Data reads are limited to
the public D1 test/train rows (k = 3, 5, 10) and the all_config.yaml files named in the spec.
Run:  CUDA_VISIBLE_DEVICES="" uv run --no-sync python -m pytest tests/test_ltf_decode.py -q
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import shutil
import sys

import numpy as np
import pytest
import torch
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from analysis.ltf_decode import cells, gates, judge  # noqa: E402
from analysis.ltf_decode import run_ltf_decode as rld  # noqa: E402
from analysis.ltf_decode.stats_j import analyze_decodes_j, synth_genuine_sequential, tstar_first_hit  # noqa: E402
from analysis.ltf_decode.trace import build_ltf, native_forward, trace_loops  # noqa: E402
from analysis.z_dynamics import readouts  # noqa: E402

D1 = os.path.join(REPO, "data/power_permutation/d1")
needs_data = pytest.mark.skipif(not os.path.isdir(os.path.join(D1, "5", "test")), reason="D1 data not present")
PREREG = judge.DEFAULT_PREREG


@pytest.fixture(autouse=True)
def _cpu_env(monkeypatch):
    monkeypatch.setattr(judge, "ALLOW_DRAFT_TOKENS", True)           # draft mode refuses tokens unless a test opts in
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.chdir(REPO)


def tiny_cfg(**over):
    c = dict(gates.TINY_ARCH, batch_size=4, seq_len=11, vocab_size=11, num_puzzle_identifiers=1)
    c.update(over)
    return c


# ------------------------------------------------------------------ fail-closed CPU guard

def test_require_cpu_accepts_only_blank_cuda_env_cpu_device_no_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    rld.require_cpu("cpu")                                              # passes silently
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    with pytest.raises(SystemExit) as e:
        rld.require_cpu("cpu")                                          # unset -> fail closed
    assert e.value.code == 2
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(SystemExit):
        rld.require_cpu("cpu")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    for dev in ("cuda", "cuda:0", "CPU"):
        with pytest.raises(SystemExit) as e:
            rld.require_cpu(dev)
        assert e.value.code == 2
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(SystemExit):
        rld.require_cpu("cpu")


def test_load_ltf_refuses_non_cpu():
    from analysis.ltf_decode.trace import load_ltf
    with pytest.raises(ValueError):
        load_ltf("/nonexistent", tiny_cfg(), "cuda")


# ------------------------------------------------------------------ judge / rules

def test_rules_parse_and_czz2_deep_equal_to_exp017():
    from analysis.z_dynamics import judge as zj
    assert judge.FROZEN_RULES["czz2"] == zj.FROZEN_RULES["czz2"]
    assert judge.t_stat() == list(range(1, 21)) and judge.t_all() == list(range(1, 22))
    assert judge.dgrid() == [2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21]
    assert judge.extrapolation_range() == list(range(22, 43))
    assert judge.exclusion_divisor() == 3
    assert judge.need_fraction("czz2.family_need") == (2, 3)


@pytest.mark.skipif(not judge.DRAFT_MODE, reason="DRAFT-mode assertion; frozen tree uses the frozen-mode counterparts")
def test_czz2_mismatch_is_rejected_exit_code_2_path(tmp_path):
    txt = open(PREREG, encoding="utf-8").read().replace("GAMMA_MIN: 0.50", "GAMMA_MIN: 0.51", 1)
    p = tmp_path / "bad.md"
    p.write_text(txt, encoding="utf-8")
    with pytest.raises(judge.JudgeAbort):
        judge.load_rules(str(p))
    import subprocess
    r = subprocess.run([sys.executable, "-c", "import analysis.ltf_decode.judge"], cwd=REPO,
                       env=dict(os.environ, PC1_PREREG=str(p)), capture_output=True)
    assert r.returncode == 2                                          # import-time refusal, spec §2.3


def test_prereg_sha_check():
    assert len(judge.sha256_file(PREREG)) == 64          # draft may be edited; only the mechanism is checked
    with pytest.raises(judge.JudgeAbort):
        judge.parse_frozen_rules(PREREG, expected_sha256="0" * 64)


def test_missing_rule_raises():
    with pytest.raises(judge.FrozenRuleMissing):
        judge.rule("czz2.NOT_THERE")
    with pytest.raises(judge.FrozenRuleMissing):
        judge.num("class_rules.G")


@pytest.mark.parametrize("test,train,cls", [
    (1.0, 1.0, "G"), (0.9, 0.0, "G"), (0.8999, 1.0, "P"), (0.0501, 1.0, "P"), (0.05, 0.90, "C"),
    (0.05, 0.8999, "U"), (0.0, 0.95, "C"), (0.5, None, "P")])
def test_outcome_class_boundaries_inclusive(test, train, cls):
    assert judge.outcome_class(test, train) == cls


def _cell(name, Z, V=0.9, excluded=False):
    return {"run_name": name, "Z": Z, "V": V, "excluded": excluded}


def _fam(zs, n_frozen=None, vs=None, excl=()):
    cells_ = [_cell(f"c{i}", z, (vs[i] if vs else 0.9), i in excl) for i, z in enumerate(zs)]
    return judge.family_token(cells_, n_frozen or len(zs))


def test_family_tokens_and_slack():
    r = _fam(["jump"] * 8 + ["unordered"] * 3)                    # n'=11 need=8 -> S2-2 at threshold
    assert (r["token"], r["need"], r["slack"], r["at_threshold"]) == ("S2-2", 8, 0, True)
    assert "(at threshold: one cell flips the branch)" in r["annotations"]
    r = _fam(["jump"] * 9 + ["unordered"] * 2)
    assert (r["token"], r["slack"], r["at_threshold"]) == ("S2-2", 1, False)
    assert _fam(["stepwise"] * 6 + ["jump"] * 2)["token"] == "S2-1"          # n'=8 need=6
    assert _fam(["stepwise"] * 5 + ["jump"] * 3)["token"] == "S2-3"
    assert _fam(["PF"] * 6 + ["jump"] * 2)["token"] == "S2-0"                 # n=8 need_PF=6, at threshold
    assert _fam(["PF"] * 5 + ["jump"] * 3)["token"] != "S2-0"
    r = _fam(["PF"] * 5 + ["jump"] * 3)                                       # PF cells leave the vote
    assert (r["n_prime"], r["need"], r["token"]) == (3, 2, "S2-2")


def test_family_invalid_orders_and_gate_v():
    assert _fam([], n_frozen=0)["token"] == "INVALID-EMPTY"
    assert _fam(["jump"] * 3, n_frozen=3, excl=(0, 1, 2))["token"] == "INVALID-EMPTY"
    cells_ = [_cell(f"c{i}", "jump", excluded=i < 4) for i in range(11)]
    assert judge.family_token(cells_, 11)["token"] == "INVALID-GATE"          # 4 > floor(11/3)=3
    cells_ = [_cell(f"c{i}", "jump", excluded=i < 3) for i in range(11)]
    assert judge.family_token(cells_, 11)["token"] == "S2-2"                  # 3 == floor(11/3): not exceeded
    assert _fam(["jump"] * 8, vs=[0.01] * 8)["token"] == "V-FAIL"
    r = _fam(["jump"] * 8, vs=[0.3] * 8)
    assert r["token"] == "S2-2" and "(validity < 0.50)" in r["annotations"]
    assert _fam(["jump"] * 8, vs=[0.05] * 8)["Vfam"] == "V~"                  # V_MINUS inclusive
    assert _fam(["jump"] * 8, vs=[0.5] * 8)["Vfam"] == "V+"                   # V_PLUS inclusive
    with pytest.raises(judge.JudgeAbort):
        _fam(["jump"] * 3, vs=[float("nan")] * 3)


def test_keys_xt_e2_first_match():
    assert [judge.key_xt(t) for t in ("INVALID-EMPTY", "INVALID-GATE", "S2-0", "V-FAIL")] == ["XT-0"] * 4
    assert (judge.key_xt("S2-2"), judge.key_xt("S2-1"), judge.key_xt("S2-3")) == ("XT-a", "XT-b", "XT-c")
    assert judge.key_e2("S2-0", "S2-1") == "E2-0" and judge.key_e2("S2-1", "V-FAIL") == "E2-0"
    assert judge.key_e2("S2-1", "S2-1") == "E2-same" and judge.key_e2("S2-2", "S2-2") == "E2-same"
    assert judge.key_e2("S2-1", "S2-2") == "E2-diff" and judge.key_e2("S2-2", "S2-1") == "E2-diff"
    assert judge.key_e2("S2-3", "S2-3") == "E2-U" and judge.key_e2("S2-3", "S2-1") == "E2-U"


def test_czz1_used_is_deep_checked_against_exp017():
    from analysis.z_dynamics import judge as zj
    rules = copy.deepcopy(judge.FROZEN_RULES)
    judge.check_czz2(rules)
    for k in judge.CZZ1_USED_KEYS:
        bad = copy.deepcopy(rules)
        bad["czz1_used"][k] = rules["czz1_used"][k] + 0.01
        with pytest.raises(judge.JudgeAbort):
            judge.check_czz2(bad)
    bad = copy.deepcopy(rules)
    del bad["czz1_used"]["PF_A1"]
    with pytest.raises(judge.JudgeAbort):
        judge.check_czz2(bad)
    assert rules["czz1_used"]["PF_A1"] == zj.FROZEN_RULES["czz1"]["PF_A1"]


@pytest.mark.skipif(not judge.DRAFT_MODE, reason="DRAFT-mode assertion; frozen tree uses the frozen-mode counterparts")
def test_draft_mode_refuses_tokens_and_prints_banner(monkeypatch):
    assert judge.DRAFT_MODE and judge.FROZEN_PREREG_SHA256 is None and not hasattr(judge, "DRAFT_R7_SHA256")
    monkeypatch.setattr(judge, "ALLOW_DRAFT_TOKENS", False)
    with pytest.raises(judge.JudgeAbort):
        judge.family_token([_cell("a", "jump")], 1)
    with pytest.raises(judge.JudgeAbort):
        judge.key_xt("S2-2")
    import subprocess
    r = subprocess.run([sys.executable, "-c", "import analysis.ltf_decode.judge"], cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0 and "DRAFT MODE" in r.stderr


def _exec_judge_with_sha(sha):
    src = open(os.path.join(REPO, "analysis/ltf_decode/judge.py")).read()
    pat = re.compile(r"^FROZEN_PREREG_SHA256: str \| None = (None|\"[0-9a-f]{64}\")(?=[ \t]|$)", re.M)
    assert len(pat.findall(src)) == 1                  # works in DRAFT (None) and frozen (hex) trees
    return pat.sub(lambda m: f'FROZEN_PREREG_SHA256: str | None = "{sha}"', src)


def test_frozen_sha_enforced_at_module_load(tmp_path):
    import subprocess
    good = judge.sha256_file(PREREG)
    runner = "import sys; src=open(sys.argv[1]).read(); exec(compile(src, 'judge_x', 'exec'), {'__name__': 'jx'}); print('LOADED')"
    for sha, want_rc in ((good, 0), ("0" * 64, 2)):
        f = tmp_path / f"jx_{want_rc}.py"
        f.write_text(_exec_judge_with_sha(sha))
        r = subprocess.run([sys.executable, "-c", runner, str(f)], cwd=REPO, capture_output=True, text=True,
                           env=dict(os.environ, PC1_PREREG="/nonexistent"))     # env override ignored when frozen
        assert r.returncode == want_rc, r.stderr
        if want_rc == 0:
            assert "DRAFT MODE" not in r.stderr and "LOADED" in r.stdout


@pytest.mark.skipif(judge.DRAFT_MODE, reason="frozen-mode counterpart; DRAFT tree has no frozen sha")
def test_frozen_mode_counterparts(tmp_path, monkeypatch):
    import subprocess
    assert judge.FROZEN_PREREG_SHA256 == judge.sha256_file(PREREG)
    assert judge.EXP_ID != "PC1-DRAFT"
    # tokens are available without the test-only flag
    monkeypatch.setattr(judge, "ALLOW_DRAFT_TOKENS", False)
    judge.key_xt("S2-2")
    # a spec whose sha differs from the constant is refused at import (env override ignored)
    bad = tmp_path / "bad.md"
    bad.write_text(open(PREREG, encoding="utf-8").read() + "\n", encoding="utf-8")
    r = subprocess.run([sys.executable, "-c", "import analysis.ltf_decode.judge"], cwd=REPO,
                       env=dict(os.environ, PC1_PREREG=str(bad)), capture_output=True, text=True)
    assert r.returncode == 0 and "DRAFT MODE" not in r.stderr
    f = tmp_path / "jx_bad.py"
    f.write_text(_exec_judge_with_sha("0" * 64))
    r = subprocess.run([sys.executable, "-c", "import sys; exec(compile(open(sys.argv[1]).read(), 'jx', 'exec'), {'__name__': 'jx'})", str(f)],
                       cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 2


# ------------------------------------------------------------------ aggregator / I-8

M1 = os.path.join(ART_DIR := os.path.join(REPO, "lab/23_reports/11_lab-reports/2026-10-21_c1-prereg-draft_artifacts/pc1_impl"),
                  "M1_manifest.csv")


def _synth_manifest():
    rows = []
    for name, arm, k, s in cells.cell_names():
        G = (arm == "g7" and (k, s) in {(3, 1), (3, 2), (3, 3), (5, 1), (5, 2), (5, 3), (6, 1), (7, 1), (7, 3), (10, 2), (10, 3)}) or \
            (arm == "full" and (k in (3, 10) or (k, s) in {(6, 2), (6, 3)}))
        rows.append({"run_name": name, "arm": arm, "k": k, "seed": s, "class": "G" if G else "P", "terminal_ok": True,
                     "final_probe_test_exact": 1.0 if G else 0.3, "tags": ""})
    kb = sorted({r["k"] for r in rows if r["arm"] == "g7" and r["class"] == "G"} &
                {r["k"] for r in rows if r["arm"] == "full" and r["class"] == "G"})
    for r in rows:
        r["tags"] = ";".join(cells.membership(r, kb))
    return rows


def _write_run(tmp, rows, z="jump", drop=(), excl=(), status="complete", g5=True, g1_fail=(), row_over=None):
    import csv as _csv
    d = tmp / "run"
    d.mkdir(exist_ok=True)
    (d / "run_manifest.json").write_text(json.dumps({"status": status, "partial": False}))
    recs, cs = {}, []
    for r in rows:
        n = r["run_name"]
        if n in drop:
            continue
        if n in excl:
            recs[n] = {"status": "excluded", "exclusion_reason": "gate failure"}
            continue
        recs[n] = {"status": "analysed", "G1": n not in g1_fail, "G_P3": {"pass": True}}
        row = {"run_name": n, "split": "test", "readout_kind": "ltf", "a1": 0.1, "maxdrop": 0.0, "Z_raw": z, "Z": z, "V": 0.9}
        row.update((row_over or {}).get(n, {}))
        cs.append(row)
    (d / "gates.json").write_text(json.dumps({"cells": recs, "G5": {"ran": True, "pass": g5, "unit": "u"}}))
    with open(d / "cell_stats.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["run_name", "split", "readout_kind", "a1", "maxdrop", "Z_raw", "Z", "V"])
        w.writeheader()
        w.writerows(cs)
    return str(d)


def _manifest_csv(tmp, rows):
    p = tmp / "m.csv"
    cells.write_manifest([dict(r, run_id="x", n_dirs_for_name=1, summary_step=122050, probe_last_step=122050,
                               probe_downsampled=False, train_final_proxy=1.0) for r in rows], str(p))
    return str(p)


def test_synthetic_manifest_family_sizes():
    rows = _synth_manifest()
    fam = cells.families(rows)
    assert (len(fam["g7_G"]), len(fam["full_G"]), len(fam["g7_G_Kboth"])) == (11, 8, 6)


def test_aggregate_tokens_and_exit_codes(tmp_path):
    rows = _synth_manifest()
    mp = _manifest_csv(tmp_path, rows)
    d = _write_run(tmp_path, rows)
    out = tmp_path / "v.json"
    assert judge.main(["--data", d, "--manifest", mp, "--out", str(out)]) == 0
    v = json.loads(out.read_text())
    assert v["families"]["g7_G"]["token"] == "S2-2" and v["key_XT"] == "XT-a" and v["key_E2"] == "E2-same"
    assert v["families"]["g7_G"]["n"] == 11 and v["families"]["g7_G_Kboth"]["n"] == 6
    # one excluded cell -> exit 3
    ex = rows[0]["run_name"]
    d = _write_run(tmp_path, rows, excl=(ex,))
    assert judge.main(["--data", d, "--manifest", mp, "--out", str(out)]) == 3
    assert json.loads(out.read_text())["families"]["g7_G"]["n_excluded"] == 1
    # PF majority -> S2-0 -> key rows without reading
    d = _write_run(tmp_path, rows, z="jump", row_over={r["run_name"]: {"a1": 0.95, "Z": "PF"} for r in rows})
    assert judge.main(["--data", d, "--manifest", mp, "--out", str(out)]) == 0
    v = json.loads(out.read_text())
    assert v["families"]["g7_G"]["token"] == "S2-0" and v["key_XT"] == "XT-0" and v["key_E2"] == "E2-0"


@pytest.mark.parametrize("kw", [
    dict(drop=("pp_ltf_g7_lay2_k3_s1",)),                 # no record and no exclusion record (I-8)
    dict(status="running"),                               # incomplete run manifest
    dict(row_over={"pp_ltf_g7_lay2_k3_s1": {"a1": 0.95}}),    # stored Z disagrees with the PF label recomputed from czz1_used
])
def test_aggregate_i8_aborts_with_exit_2(tmp_path, kw):
    rows = _synth_manifest()
    mp = _manifest_csv(tmp_path, rows)
    d = _write_run(tmp_path, rows, **kw)
    assert judge.main(["--data", d, "--manifest", mp, "--out", str(tmp_path / "v.json")]) == 2
    assert not (tmp_path / "v.json").exists()


def test_aggregate_missing_files_and_cell_count_abort(tmp_path):
    rows = _synth_manifest()
    mp = _manifest_csv(tmp_path, rows)
    assert judge.main(["--data", str(tmp_path / "nope"), "--manifest", mp, "--out", str(tmp_path / "v.json")]) == 2
    mp2 = _manifest_csv(tmp_path, rows[:29])
    d = _write_run(tmp_path, rows[:29])
    assert judge.main(["--data", d, "--manifest", mp2, "--out", str(tmp_path / "v.json")]) == 2


def test_aggregate_halts_on_g5_failure_and_i11(tmp_path):
    rows = _synth_manifest()
    mp = _manifest_csv(tmp_path, rows)
    out = tmp_path / "v.json"
    d = _write_run(tmp_path, rows, g5=False)
    assert judge.main(["--data", d, "--manifest", mp, "--out", str(out)]) == 4
    assert json.loads(out.read_text())["halt_reason"] == "G5'" and "families" not in json.loads(out.read_text())
    names = [r["run_name"] for r in rows]
    d = _write_run(tmp_path, rows, g1_fail=tuple(names[:3]))
    assert judge.main(["--data", d, "--manifest", mp, "--out", str(out)]) == 0      # 3 failures: not more than 3
    d = _write_run(tmp_path, rows, g1_fail=tuple(names[:4]))
    assert judge.main(["--data", d, "--manifest", mp, "--out", str(out)]) == 4
    assert json.loads(out.read_text())["halt_reason"] == "I-11"


def _write_run_gp3(d, names_fail_gp3):
    gj = json.loads((d / "gates.json").read_text() if hasattr(d, "read_text") else open(os.path.join(d, "gates.json")).read())
    for n in names_fail_gp3:
        gj["cells"][n]["G_P3"] = {"pass": False}
    open(os.path.join(str(d), "gates.json"), "w").write(json.dumps(gj))


def test_halt_threshold_is_the_union_count_from_frozen_rules(tmp_path):
    rows = _synth_manifest()
    mp = _manifest_csv(tmp_path, rows)
    out = tmp_path / "v.json"
    names = [r["run_name"] for r in rows]
    assert judge.halt_rules() == (3, "halt")
    # 3 cells each failing BOTH G1' and G-P3 = union of 3 -> proceed
    d = _write_run(tmp_path, rows, g1_fail=tuple(names[:3]))
    _write_run_gp3(d, names[:3])
    assert judge.main(["--data", d, "--manifest", mp, "--out", str(out)]) == 0
    assert json.loads(out.read_text())["n_fail_G1_or_GP3"] == 3
    # 2 fail G1' + 2 other cells fail G-P3 = union of 4 -> halt
    d = _write_run(tmp_path, rows, g1_fail=tuple(names[:2]))
    _write_run_gp3(d, names[2:4])
    assert judge.main(["--data", d, "--manifest", mp, "--out", str(out)]) == 4
    assert json.loads(out.read_text())["n_fail_G1_or_GP3"] == 4
    # G5' failure -> halt
    d = _write_run(tmp_path, rows, g5=False)
    assert judge.main(["--data", d, "--manifest", mp, "--out", str(out)]) == 4


def test_halt_threshold_follows_the_rules_not_a_constant(tmp_path, monkeypatch):
    rows = _synth_manifest()
    mp = _manifest_csv(tmp_path, rows)
    names = [r["run_name"] for r in rows]
    d = _write_run(tmp_path, rows, g1_fail=tuple(names[:2]))
    monkeypatch.setitem(judge.FROZEN_RULES["halt_rules"], "G1p_or_GP3_failed_cells_max", 1)
    assert judge.main(["--data", d, "--manifest", mp, "--out", str(tmp_path / "v.json")]) == 4


@pytest.mark.parametrize("mutate", ["delete_key", "delete_max", "bad_policy", "bad_count"])
def test_missing_or_malformed_halt_rules_exit_2(tmp_path, monkeypatch, mutate):
    rows = _synth_manifest()
    mp = _manifest_csv(tmp_path, rows)
    d = _write_run(tmp_path, rows)
    hr = copy.deepcopy(judge.FROZEN_RULES["halt_rules"])
    if mutate == "delete_key":
        monkeypatch.delitem(judge.FROZEN_RULES, "halt_rules")
    else:
        if mutate == "delete_max":
            del hr["G1p_or_GP3_failed_cells_max"]
        elif mutate == "bad_policy":
            hr["G5p_fail"] = "continue"
        else:
            hr["count"] = "each gate separately"
        monkeypatch.setitem(judge.FROZEN_RULES, "halt_rules", hr)
    assert judge.main(["--data", d, "--manifest", mp, "--out", str(tmp_path / "v.json")]) == 2
    assert not (tmp_path / "v.json").exists()


def test_prereg_yaml_carries_halt_rules():
    assert judge.FROZEN_RULES["halt_rules"] == {"G1p_or_GP3_failed_cells_max": 3,
                                                "count": "union of cells failing G1prime or G-P3", "G5p_fail": "halt"}


@pytest.mark.skipif(not judge.DRAFT_MODE, reason="DRAFT-mode assertion; frozen tree uses the frozen-mode counterparts")
def test_run_ltf_decode_refuses_draft_mode_without_dry_flag_and_never_reads_repo_checkpoints(tmp_path):
    assert judge.DRAFT_MODE
    argv = ["--manifest", str(tmp_path / "m.csv"), "--gc-output", str(tmp_path / "g.txt"),
            "--out", str(tmp_path / "o")]
    assert rld.main(argv) == 2 and not (tmp_path / "o").exists()
    with pytest.raises(SystemExit) as e:           # flag given, but the checkpoint root is the repo's checkpoints/ tree
        rld.main(argv + ["--dry-synthetic", "--ckpt-root", os.path.join(REPO, "checkpoints/power_permutation")])
    assert e.value.code == 2
    row = {"run_name": "pp_ltf_g7_lay2_k5_s1", "arm": "g7", "k": 5, "seed": 1, "class": "G", "terminal_ok": True,
           "final_probe_test_exact": 1.0}
    a = argparse.Namespace(ckpt_root=os.path.join(REPO, "checkpoints/power_permutation"), device="cpu", batch=8,
                           test_n=8, probe_n=8, n_perm=2)
    with pytest.raises(SystemExit) as e:           # process_cell guard fires before any file is touched
        rld.process_cell(row, a, extrap=False)
    assert e.value.code == 2
    # outside the repo checkpoints tree the guard does not fire (tmp fake run dirs, tests)
    rld.draft_checkpoint_guard(str(tmp_path))


def test_g5_texts_include_near_power():
    near = [{"unit": "u", "t": 1, "residues": [0]}]
    assert rld.near_text(near) == '{"unit": "u", "t": 1, "residues": [0]}\n'
    assert not gates.g5_identical({"near_power": rld.near_text(near)}, {"near_power": rld.near_text(near + near)})["pass"]


def test_read_manifest_tolerates_missing_terminal_value(tmp_path):
    rows = cells.classify_raw({"cells": [{"run_name": "a", "arm": "g7", "k": 3, "seed": 1, "run_id": "",
                                          "n_dirs_for_name": 0}]})
    cells.write_manifest(rows, str(tmp_path / "m.csv"))
    back = cells.read_manifest(str(tmp_path / "m.csv"))
    assert back[0]["final_probe_test_exact"] is None and back[0]["terminal_ok"] is False and back[0]["class"] == "MISSING"


def test_check_czz2_missing_exp017_key_is_abort_not_keyerror(monkeypatch):
    from analysis.z_dynamics import judge as zj
    monkeypatch.delitem(zj.FROZEN_RULES["czz1"], "PF_A1")
    with pytest.raises(judge.JudgeAbort):
        judge.check_czz2(copy.deepcopy(judge.FROZEN_RULES))


@pytest.mark.skipif(not judge.DRAFT_MODE, reason="DRAFT-mode assertion; frozen tree uses the frozen-mode counterparts")
def test_aggregate_refuses_in_draft_mode(tmp_path, monkeypatch):
    rows = _synth_manifest()
    mp = _manifest_csv(tmp_path, rows)
    d = _write_run(tmp_path, rows)
    monkeypatch.setattr(judge, "ALLOW_DRAFT_TOKENS", False)
    assert judge.main(["--data", d, "--manifest", mp, "--out", str(tmp_path / "v.json")]) == 2


def test_cells_require_cpu_and_descriptive_comparison(monkeypatch):
    cells.require_cpu("cpu")
    with pytest.raises(SystemExit):
        cells.require_cpu("cuda")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    with pytest.raises(SystemExit):
        cells.require_cpu("cpu")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    with pytest.raises(SystemExit):
        cells.require_cpu("cpu")


def test_compare_expected_descriptive_lists_are_checked():
    rows = _synth_manifest()
    exp = judge.rule("families_expected")
    for r in rows:                                  # make the synthetic classes follow the prereg lists
        r["class"] = "G" if r["run_name"] in exp["g7_G"] + exp["full_G"] else "P"
    kb = cells.k_both(rows)
    for r in rows:
        r["tags"] = ";".join(cells.membership(r, kb))
    c = cells.compare_expected(rows)
    assert c["g7_G"]["match"] and c["full_G"]["match"] and c["g7_P_reach_then_below_0p9"]["match"]
    assert c["full_C_or_P"]["match"] and c["descriptive_union_equals_non_G"]["match"]
    rows[7]["class"] = "G"                          # a descriptive g7 P cell flips to G
    for r in rows:
        r["tags"] = ";".join(cells.membership(r, cells.k_both(rows)))
    c = cells.compare_expected(rows)
    assert not c["g7_P_reach_then_below_0p9"]["match"] and c["g7_P_reach_then_below_0p9"]["outside_allowed"]
    assert not c["g7_G"]["match"] and not c["descriptive_union_equals_non_G"]["match"]


# ------------------------------------------------------------------ M-1 cells

def _raw(name, arm, k, s, test, train, step=122050, last=122050):
    return {"run_name": name, "arm": arm, "k": k, "seed": s, "run_id": "rid", "n_dirs_for_name": 1,
            "summary_step": step, "probe_last_step": last, "probe_downsampled": False,
            "final_probe_test_exact": test, "train_final": train}


def test_cells_names_and_classification():
    names = cells.cell_names()
    assert len(names) == 30 and names[0][0] == "pp_ltf_g7_lay2_k3_s1" and names[-1][0] == "pp_ltf_fullgrad_lay2_k10_s3"
    assert all("noinj" not in n[0] for n in names)
    raw = {"cells": [_raw("a", "g7", 3, 1, 1.0, 1.0), _raw("b", "full", 3, 1, 0.9, 0.5), _raw("c", "full", 5, 1, 0.02, 0.97),
                     _raw("d", "full", 6, 1, 0.02, 0.5), _raw("e", "g7", 5, 1, 0.4, 1.0),
                     _raw("f", "g7", 6, 1, 1.0, 1.0, last=100000)]}
    rows = cells.classify_raw(raw)
    assert [r["class"] for r in rows] == ["G", "G", "C", "U", "P", "G"]
    assert [r["terminal_ok"] for r in rows] == [True, True, True, True, True, False]
    assert cells.k_both(rows) == [3]
    assert rows[0]["tags"] == "g7_G;g7_G_Kboth" and rows[2]["tags"] == "full_C_D4" and rows[5]["tags"] == ""


def test_manifest_roundtrip(tmp_path):
    rows = cells.classify_raw({"cells": [_raw("a", "g7", 3, 1, 1.0, 1.0)]})
    cells.write_manifest(rows, str(tmp_path / "m.csv"))
    back = cells.read_manifest(str(tmp_path / "m.csv"))
    assert back[0]["run_name"] == "a" and back[0]["class"] == "G" and back[0]["terminal_ok"] is True


ART = os.path.join(REPO, "lab/23_reports/11_lab-reports/2026-10-21_c1-prereg-draft_artifacts/pc1_impl")


@pytest.mark.skipif(not os.path.exists(os.path.join(ART, "M1_manifest.csv")), reason="M-1 manifest not generated yet")
def test_generated_manifest_matches_prereg_families():
    rows = cells.read_manifest(os.path.join(ART, "M1_manifest.csv"))
    assert len(rows) == 30 and all(r["terminal_ok"] for r in rows)
    cmp = cells.compare_expected(rows)
    assert cmp["g7_G"]["match"] and cmp["full_G"]["match"]
    assert cells.counts_by_arm_class(rows) == {"g7": {"G": 11, "P": 4}, "full": {"G": 8, "C": 6, "P": 1}}


# ------------------------------------------------------------------ gates

def test_g_p3_tolerance_inclusive():
    dec = np.tile(np.arange(1, 11), (100, 1))
    lab = dec.copy()
    lab[:10, 0] = 99                                                   # 10/100 wrong -> exact 0.90
    r = gates.g_p3(dec, lab, 0.905)
    assert r["pass"] and math.isclose(r["exact_t21_test512"], 0.9)
    assert not gates.g_p3(dec, lab, 0.9101)["pass"]
    assert gates.g_p3(dec, lab, 0.9)["pass"]
    assert gates.judge.num("gates.G_P3_abs_tol") == 0.01


def _cfg_dict(arm="g7"):
    arch = dict(gates.G7_ARCH_EXPECTED, loop_grad_cycles=gates.G7_LOOP_GRAD[arm])
    top = {k: i for i, k in enumerate(gates.G7_TOPLEVEL_KEYS)}
    return dict(top, arch=arch)


def test_g7_config():
    ref = _cfg_dict()
    assert gates.g7_config(_cfg_dict("g7"), ref, "g7")["pass"]
    assert gates.g7_config(_cfg_dict("full"), ref, "full")["pass"]
    assert not gates.g7_config(_cfg_dict("full"), ref, "g7")["pass"]            # loop_grad_cycles 0 vs 7
    bad = _cfg_dict()
    bad["arch"]["loops"] = 22
    assert "arch.loops" in gates.g7_config(bad, ref, "g7")["mismatches"]
    bad = _cfg_dict()
    bad["lr"] = 5
    assert "lr" in gates.g7_config(bad, ref, "g7")["mismatches"]
    assert not gates.g7_config(_cfg_dict(), None, "g7")["pass"]


def test_g7_constants_appear_in_spec_and_e15():
    spec = open(PREREG, encoding="utf-8").read()
    assert gates.G0_CODE_SHA256 in spec
    assert "H_layers 2, hidden 512, heads 12, expansion 4, rope, puzzle_emb_ndim 512, halt_max_steps 1, loops 21" in spec
    e15 = open(os.path.join(REPO, "lab/21_experiments/11_lab-experiments/EXP-015_looped-tf-direct-control.md"),
               encoding="utf-8").read()
    for k in gates.G7_TOPLEVEL_KEYS:
        assert f"`{k}`" in e15, k


def test_i3_i4_g5():
    assert gates.i3_keys([], [])["pass"] and not gates.i3_keys(["x"], [])["pass"]
    assert gates.i4_terminal(122050, [9764, 122050])["pass"]
    assert not gates.i4_terminal(122050, [9764, 122051])["pass"] and not gates.i4_terminal(87876, [87876])["pass"]
    assert gates.g5_identical({"a": "x"}, {"a": "x"})["pass"] and not gates.g5_identical({"a": "x"}, {"a": "y"})["pass"]


def test_sealed_guard_imported():
    with pytest.raises(gates.SealedAccess):
        gates.guard_path("/data/foo/sealed/test")


def test_g0_code_identity_on_repo_copy(tmp_path):
    import models.recursive_reasoning.transformers_baseline as m
    shutil.copy(m.__file__, tmp_path / "transformers_baseline.py")
    r = gates.g0_code_identity(str(tmp_path))
    assert r["pass"] and r["repo_sha256"] == gates.G0_CODE_SHA256
    (tmp_path / "transformers_baseline.py").write_text(open(m.__file__).read() + "\n# x\n")
    assert not gates.g0_code_identity(str(tmp_path))["pass"]


# ------------------------------------------------------------------ G-a (synthetic weights) -- spec §4.2

@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("lgc", [0, 7])
@pytest.mark.parametrize("inj", [True, False])
def test_ga1_tracer_t21_equals_native_tiny(dtype, lgc, inj):
    r = gates._eq_trace_native(tiny_cfg(loops=21, loop_grad_cycles=lgc, loop_input_injection=inj, forward_dtype=dtype))
    assert r["z_equal"] and r["logits_equal"]


def test_ga2_loops1_equals_single_h_level_call():
    cfg = tiny_cfg(loops=1, loop_grad_cycles=0, forward_dtype="float32")
    inner = gates._synthetic_model(cfg)
    batch = gates._synthetic_batch(cfg)
    with torch.inference_mode():
        tr = trace_loops(inner, batch, 1, keep_latents=True)
        z0 = inner.reset_carry(torch.ones(4, dtype=torch.bool), inner.empty_carry(4)).z_H
        emb = inner._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])
        ref = inner.H_level(z0, emb, cos_sin=inner.rotary_emb())
    assert torch.equal(tr.z[0], ref)


def test_ga3_extrapolation_42_equals_loops42_model():
    cfg21 = tiny_cfg(loops=21, loop_grad_cycles=7, forward_dtype="float32")
    m21, m42 = gates._synthetic_model(cfg21), gates._synthetic_model(dict(cfg21, loops=42))
    m42.load_state_dict(m21.state_dict())
    batch = gates._synthetic_batch(cfg21)
    with torch.inference_mode():
        tr = trace_loops(m21, batch, 42, keep_latents=True)
        z42, out42 = native_forward(m42, batch)
    assert torch.equal(tr.z[41], z42) and torch.equal(tr.logits[41], out42)


def test_tracer_is_sensitive_to_injection_rule_and_loop_count():
    cfg = tiny_cfg(loops=5, forward_dtype="float32")
    inner = gates._synthetic_model(cfg)
    batch = gates._synthetic_batch(cfg)
    with torch.inference_mode():
        a = trace_loops(inner, batch, 5, keep_latents=True)
        inner.config.loop_input_injection = False
        b = trace_loops(inner, batch, 5, keep_latents=True)
    assert torch.equal(a.z[0], b.z[0]) and not torch.equal(a.z[1], b.z[1])      # loop 0 always injects
    assert not torch.equal(a.z[3], a.z[4])


def test_ga4_puzzle_emb_len_is_1_for_the_exp015_arch():
    if not os.path.exists(os.path.join(REPO, gates.GA_CFG_DIR, "all_config.yaml")):
        pytest.skip("all_config.yaml not present")
    cfg = gates.ga_arch_configs()["full"]
    assert (cfg["hidden_size"], cfg["num_heads"], cfg["puzzle_emb_ndim"], cfg["loops"]) == (512, 12, 512, 21)
    inner = gates._synthetic_model(dict(cfg, forward_dtype="float32"))
    assert inner.puzzle_emb_len == 1


@needs_data
def test_ga5_stats_j_equals_exp017_stats_at_17_18():
    res = gates.ga_item5(n_rows=60, n_perm=30, ks=(3, 5))
    assert res and all(r["pass"] for r in res)
    assert any(r["p_computed"] for r in res)                     # the permutation path is exercised


@needs_data
def test_stats_j_j20_runs_and_forces_permutation_path():
    k = 5
    inp = np.load(os.path.join(D1, "5/test/all__inputs.npy"))[:80].astype(np.int64)
    lab = np.load(os.path.join(D1, "5/test/all__labels.npy"))[:80, :10].astype(np.int64)
    infos = readouts.sigma_infos(inp, k)
    dec = synth_genuine_sequential(infos, k, 21, np.random.default_rng(0))
    an = analyze_decodes_j(dec, inp, lab, k, J=20, J_all=21, infos=infos, n_perm=40, with_subclass=False)
    c = an["cell"]
    assert c["gamma"] >= 0.5 and c["p"] is not None and c["subclass"] == ""
    with pytest.raises(ValueError):
        analyze_decodes_j(dec[:, :20], inp, lab, k, J=20, J_all=21, infos=infos)       # needs all 21 readouts
    # statistics only see t = 1..20: changing column 21 changes nothing in the cell row
    dec2 = dec.copy()
    dec2[:, 20] = 1
    c2 = analyze_decodes_j(dec2, inp, lab, k, J=20, J_all=21, infos=infos, n_perm=40, with_subclass=False)["cell"]
    for key in ("gamma", "p", "P_int", "V", "Z", "a1"):
        assert c[key] == c2[key]


def test_tstar_first_hit():
    lab = np.tile(np.arange(1, 11), (3, 1))
    dec = np.zeros((3, 4, 10), np.int64)
    dec[0, 2:] = lab[0]
    dec[1, 0] = lab[1]
    dec[1, 1] = 5
    ts = tstar_first_hit(dec, lab, [1, 2, 3, 4])
    assert ts.tolist() == [3, 1, -1]


# ------------------------------------------------------------------ per-cell pipeline on a tiny random model

def _random_probe(n, k, seed):
    rng = np.random.default_rng(seed)
    sig, lab = [], []
    while len(sig) < n:
        p = rng.permutation(10)
        if readouts.order(p) > k:
            sig.append(p)
            lab.append(readouts.perm_pow(p, k))
    inp = np.zeros((n, 11), np.int32)
    lbl = np.zeros((n, 11), np.int32)
    inp[:, :10] = np.array(sig) + 1
    lbl[:, :10] = np.array(lab) + 1
    lbl[:, 10] = -100
    return {"inputs": torch.from_numpy(inp), "labels": torch.from_numpy(lbl),
            "puzzle_identifiers": torch.zeros(n, dtype=torch.int32)}


def test_analyze_cell_tiny_model_rows_and_determinism():
    k = 5
    cfg = tiny_cfg(loops=21, loop_grad_cycles=7, forward_dtype="float32")
    inner = gates._synthetic_model(cfg)
    test, train = _random_probe(40, k, 1), _random_probe(24, k, 2)
    base = {"exp_id": "T", "unit": "u", "run_name": "r", "arm": "g7", "k": k, "seed": 1, "frozen_class": "P"}

    def run():
        from analysis.z_dynamics import stats as zs
        zs._OMEGA_CACHE.clear()
        readouts._cached_info.cache_clear()
        return rld.analyze_cell(inner, k, test, train, 42, 16, 30, base, manifest_terminal_probe=0.0)

    r1 = run()
    assert r1["rec"]["G1"] and r1["rec"]["G1_test"] and r1["rec"]["G1_train"] and "G_P3" in r1["rec"]
    kinds = {(r["split"], r["readout_kind"]) for r in r1["cell"]}
    assert kinds == {("test", "ltf"), ("test", "dgrid"), ("train", "ltf"), ("train", "dgrid")}
    ts = sorted({r["t"] for r in r1["per_call"] if r["split"] == "test" and r["readout_kind"] == "ltf" and r["gcd_class"] == "all"})
    te = sorted({r["t"] for r in r1["per_call"] if r["split"] == "test" and r["readout_kind"] == "extrap" and r["gcd_class"] == "all"})
    assert ts == list(range(1, 22)) and te == list(range(22, 43))
    assert all(r["accuracy_protocol"] == rld.FOURTH for r in r1["per_call"])
    cl = next(r for r in r1["cell"] if r["split"] == "test" and r["readout_kind"] == "ltf")
    assert (cl["J"], cl["J_all"], cl["subclass"]) == (20, 21, "")
    dg = next(r for r in r1["cell"] if r["split"] == "test" and r["readout_kind"] == "dgrid")
    assert (dg["J"], dg["J_all"]) == (17, 18)
    assert len(r1["near"]) == (40 + 24) * 42 or len(r1["near"]) == (40 + 24) * 21
    r2 = run()
    for key, fields in (("per_call", rld.PER_CALL_FIELDS), ("cell", rld.CELL_FIELDS), ("tstar", rld.TSTAR_FIELDS)):
        assert rld.rows_to_text(r1[key], fields) == rld.rows_to_text(r2[key], fields)
    # DROP path: 21 loops, no extrapolation rows, no train split
    r3 = rld.analyze_cell(inner, k, test, None, 21, 16, 30, base, with_extrap=False)
    assert not [r for r in r3["per_call"] if r["readout_kind"] == "extrap"]
    assert {r["split"] for r in r3["cell"]} == {"test"}


def test_extrap_line_parser(tmp_path):
    p = tmp_path / "o.txt"
    p.write_text("x\nEXTRAP: DROP\n")
    assert rld.parse_extrap_line(str(p)) == "DROP"
    p.write_text("no decision here\n")
    with pytest.raises(SystemExit):
        rld.parse_extrap_line(str(p))
    p.write_text("EXTRAP: KEEP\nEXTRAP: DROP\n")
    with pytest.raises(SystemExit):
        rld.parse_extrap_line(str(p))


def test_cli_refuses_wrong_prereg_sha_and_non_cpu(tmp_path, monkeypatch):
    out = tmp_path / "gc.txt"
    out.write_text("EXTRAP: KEEP\n")
    argv = ["--manifest", str(tmp_path / "m.csv"), "--gc-output", str(out), "--expected-sha256", "0" * 64,
            "--out", str(tmp_path / "o")]
    assert rld.main(argv) == 2
    with pytest.raises(SystemExit) as e:
        rld.main(argv + ["--device", "cuda"])
    assert e.value.code == 2
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(SystemExit) as e:
        rld.main(argv)
    assert e.value.code == 2


@needs_data
def test_process_cell_io_on_fake_run_dir(tmp_path, monkeypatch):
    """IO path on a fake run dir holding a tiny random-init Model_ACTV2 (no real checkpoint)."""
    from measure_rho import build_model_config
    import models.recursive_reasoning.transformers_baseline as mb
    name = "pp_ltf_g7_lay2_k5_s1"
    run_dir = tmp_path / name
    run_dir.mkdir()
    arch_cfg = {k: v for k, v in gates.TINY_ARCH.items()}
    arch_cfg["name"] = "recursive_reasoning.transformers_baseline@Model_ACTV2"
    (run_dir / "all_config.yaml").write_text(yaml.safe_dump({"arch": arch_cfg, "data_paths": ["data/power_permutation/d1/5"]}))
    shutil.copy(mb.__file__, run_dir / "transformers_baseline.py")
    cfg = build_model_config(str(run_dir), batch_size=8)
    _w, inner = build_ltf(cfg)
    torch.save({"_orig_mod." + k: v for k, v in _w.state_dict().items()}, run_dir / "step_122050")
    a = argparse.Namespace(ckpt_root=str(tmp_path), device="cpu", batch=8, test_n=40, probe_n=24, n_perm=20)
    row = {"run_name": name, "arm": "g7", "k": 5, "seed": 1, "class": "P", "terminal_ok": True,
           "final_probe_test_exact": 0.0}
    out = rld.process_cell(row, a, extrap=True)
    rec = out["rec"]
    assert rec["I3"]["pass"] and rec["I4"]["pass"] and rec["G0"]["pass"] and rec["G6"]["pass"] and rec["G1"]
    assert not rec["G7"]["pass"] and rec["status"] == "excluded"            # tiny arch violates G7' (as intended)
    assert {r["split"] for r in out["cell"]} == {"test", "train"}           # P cell -> D-4 train layer
    row["class"] = "G"
    out = rld.process_cell(row, a, extrap=False)
    assert {r["split"] for r in out["cell"]} == {"test"}
    row["terminal_ok"] = False
    assert rld.process_cell(row, a, extrap=False)["rec"]["status"] == "excluded"


# ------------------------------------------------------------------ gc_timing protocol

from analysis.ltf_decode import gc_timing  # noqa: E402


def test_gc_decide_strict_inequalities():
    n = gc_timing.N_RUN
    # W_ext exactly 8 h and W_tot exactly 24 h -> KEEP
    sec_ext = 24 * 3600 / n
    d = gc_timing.decide(sec_ext, sec_ext - 8 * 3600 / n)
    assert math.isclose(d["W_ext"], 8.0) and math.isclose(d["W_tot"], 24.0)
    assert d["decision"] == "KEEP"
    assert gc_timing.decide(sec_ext * (1 + 1e-9), sec_ext - 8 * 3600 / n)["decision"] == "DROP"     # W_tot > 24
    assert gc_timing.decide(10.0, 10.0 - 8 * 3600 / n * (1 + 1e-9))["decision"] == "DROP"             # W_ext > 8
    assert gc_timing.decide(100.0, 50.0)["decision"] == "KEEP"
    assert (gc_timing.X_EXT_H, gc_timing.Y_TOT_H, gc_timing.N_RUN, gc_timing.REQUIRED_THREADS) == (8.0, 24.0, 31, 32)
    assert gc_timing.SEQUENCE == ["ext", "noext", "ext", "noext", "ext", "noext"]


def test_gc_wrong_thread_count_has_no_extrap_and_nonzero_rc(tmp_path, capsys):
    log, out = tmp_path / "gc_timing.log", tmp_path / "o.txt"
    rc = gc_timing.main(["--threads", "2", "--log", str(log), "--out", str(out)])
    assert rc == 3 and not out.exists()
    lines = log.read_text().splitlines()
    assert lines[0].startswith("START ") and "threads=2" in lines[0] and "argv_sha=" in lines[0]
    assert lines[1].endswith("rc=3 EXTRAP=none") and gc_timing.completed_run_in_log(str(log)) is None
    # not a completed run -> a rerun is allowed
    assert gc_timing.main(["--threads", "2", "--log", str(log), "--out", str(out)]) == 3
    assert len(log.read_text().splitlines()) == 4


@needs_data
def test_gc_dev_run_finish_precedes_decision_and_second_run_is_refused(tmp_path, monkeypatch):
    log, out = tmp_path / "gc_timing.log", tmp_path / "o.txt"
    seen = []

    class Spy:
        def __init__(self, real):
            self.real = real

        def write(self, s):
            if "EXTRAP: " in s:
                seen.append(log.read_text().splitlines()[-1] if log.exists() else "")
            return self.real.write(s)

        def flush(self):
            return self.real.flush()

    monkeypatch.setattr(sys, "stdout", Spy(sys.stdout))
    rc = gc_timing.main(["--dev", "--threads", "2", "--log", str(log), "--out", str(out)])
    monkeypatch.undo()
    assert rc == 0
    assert seen and seen[0].startswith("FINISH ") and ("EXTRAP=KEEP" in seen[0] or "EXTRAP=DROP" in seen[0])
    lines = log.read_text().splitlines()
    assert lines[0].startswith("START ") and lines[1].startswith("FINISH ") and len(lines) == 2
    text = out.read_text()
    assert text.count("\nEXTRAP: ") == 1 and text.rstrip().splitlines()[-1] in ("EXTRAP: KEEP", "EXTRAP: DROP")
    assert "raw_sec_ext=[" in text and text.count("raw_sec_ext") == 1 and "(medians of 3)" in text
    assert len(eval(text.split("raw_sec_ext=")[1].splitlines()[0])) == 3
    assert rld.parse_extrap_line(str(out)) in ("KEEP", "DROP")
    # the first completed run binds: refuse a second one, append nothing
    before = log.read_text()
    assert gc_timing.main(["--dev", "--threads", "2", "--log", str(log), "--out", str(tmp_path / "o2.txt")]) == 5
    assert log.read_text() == before


def test_gc_dev_refuses_default_log():
    assert gc_timing.main(["--dev", "--threads", "2"]) == 6


def test_gc_crash_writes_finish_none(tmp_path, monkeypatch):
    log = tmp_path / "gc_timing.log"
    monkeypatch.setattr(gc_timing, "build_inputs", lambda dev: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        gc_timing.main(["--dev", "--threads", "2", "--log", str(log), "--out", str(tmp_path / "o.txt")])
    assert log.read_text().splitlines()[-1].endswith("rc=4 EXTRAP=none")
    assert gc_timing.completed_run_in_log(str(log)) is None


# ------------------------------------------------------------------ G-b (code-vs-code), synthetic TRM only

def test_gb_compare_texts_counts_rows():
    from analysis.ltf_decode import gb_equivalence as gb
    assert gb.compare_texts("a\nb\nc\n", "a\nb\nc\n") == 0
    assert gb.compare_texts("a\nb\nc\n", "a\nX\nc\n") == 1
    assert gb.compare_texts("a\nb\n", "a\nb\nc\n") == 1
    import hashlib
    assert gb.sha256_text("x") == hashlib.sha256(b"x").hexdigest()


@needs_data
def test_gb_new_mode_equals_run_zdyn_process_cell_on_tiny_random_trm(tmp_path, monkeypatch):
    """G-b dress rehearsal: tiny RANDOM trm in a fake run dir; reference = run_zdyn.process_cell (imported, as
    tests/test_z_dynamics.py already does), new mode = gb_equivalence.new_mode_process_cell. Tolerance 0."""
    from analysis.ltf_decode import gb_equivalence as gb
    from analysis.z_dynamics import cells as zc, run_zdyn
    from measure_rho import build_model_config
    import models.recursive_reasoning.trm as mt
    from models.losses import ACTLossHead
    from tests.test_z_dynamics import tiny_cfg as ztiny
    name = "pp_base_tf_z_iter_k5_s1"
    row = next(r for r in zc.rows() if r["run_name"] == name)
    run_dir = tmp_path / name
    (run_dir / "z_snapshots").mkdir(parents=True)
    data = "data/power_permutation/d1/5"
    arch_cfg = {k: v for k, v in ztiny().items() if k not in ("batch_size", "seq_len", "vocab_size", "num_puzzle_identifiers")}
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
    torch.save({"z_H": torch.randn(probe_n, 15, 32).to(torch.bfloat16), "z_L": torch.randn(probe_n, 15, 32).to(torch.bfloat16),
                "labels": labels, "correct_mask": torch.zeros(probe_n, dtype=torch.bool)},
               run_dir / "z_snapshots" / "step_122050.pt")
    a = argparse.Namespace(ckpt_root=str(tmp_path), device="cpu", batch=8, test_n=20, probe_n=probe_n,
                           lens_train_n=0, n_perm=20)
    closeout = {name: {"run_id": row["run_id"], "final_probe_test_exact": str(row["final_probe_test_exact"]),
                       "traj_class": row["traj_class"]}}
    ref = run_zdyn.process_cell(row, a, closeout)
    new = gb.new_mode_process_cell(row, a)
    assert gb.PER_CALL_FIELDS == run_zdyn.PER_CALL_FIELDS and gb.CELL_FIELDS == run_zdyn.CELL_FIELDS
    assert gb.rows_to_text(new["per_call"], gb.PER_CALL_FIELDS) == run_zdyn.rows_to_text(ref["per_call"], run_zdyn.PER_CALL_FIELDS)
    assert gb.rows_to_text(new["cell"], gb.CELL_FIELDS) == run_zdyn.rows_to_text(ref["cell"], run_zdyn.CELL_FIELDS)
    assert gb.decodes_equal(new["dec"], gb.reference_decodes(row, a))
    # sensitivity: a perturbed row is detected
    perturbed = copy.deepcopy(new["cell"])
    perturbed[0]["gamma"] = (perturbed[0]["gamma"] or 0) + 1e-9
    assert gb.compare_texts(run_zdyn.rows_to_text(ref["cell"], run_zdyn.CELL_FIELDS),
                            gb.rows_to_text(perturbed, gb.CELL_FIELDS)) == 1
