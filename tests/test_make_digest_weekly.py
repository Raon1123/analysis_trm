"""WP3 tests: weekly HEA section, pinned material, lane fallback, --out-dir.
make_digest.py is a NAS module loaded by path. No real claude / API calls; output goes to tmp dirs."""
import importlib.util
import subprocess
from pathlib import Path

import pytest

MOD = Path("/mnt/ayp/agents/trm/93_monitoring/11_lab-monitoring/15_research-dashboard/make_digest.py")
SRC = "92_daily/11_canonical/x.md"

HEA = {
    "window": {"label": "2026-W45", "start": "2026-11-02", "end": "2026-11-08"},
    "cards": [
        {"exp_id": "EXP-025", "aliases": ["EXP-015"], "mention_only": False, "unextractable": [],
         "hypotheses": [{"id": "H-199", "statement": "s"}]},
        {"exp_id": "EXP-007", "aliases": [], "mention_only": False, "unextractable": ["x"],
         "hypotheses": [{"id": "H-A", "statement": "s"}]},
        {"exp_id": "EXP-001", "aliases": [], "mention_only": True, "unextractable": [],
         "hypotheses": [{"id": "H-022", "statement": "s"}]},
    ],
    "unattributed_detail": [],
}
TABLE = "| 가설 | 실험 |\n|---|---|\n| H-199 | EXP-025 |"


def good_body(extra=""):
    return ("# 2026-W45 주간 digest\n## 한 줄\n한 줄.\n## 가설 × 실험: 예상 대 실제\n{{HEA_TABLE}}\n"
            "- EXP-015(=EXP-025)는 결과가 있다.\n" + extra +
            "## 실험: 돌아간 것 / 끝난 것\n없음\n## 가설·주장: 움직인 것\n없음\n## 결정 대기\n없음\n"
            "## 다음 주 1수\n없음\n## 출처\n- `%s`\n" % SRC)


@pytest.fixture()
def md(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("make_digest_weekly_under_test", MOD)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    lab = tmp_path / "lab"
    (lab / "92_daily/11_canonical").mkdir(parents=True)
    (lab / SRC).write_text("x", encoding="utf-8")
    monkeypatch.setattr(m, "LAB", lab)
    m.DIGEST_DIR = lab / "92_daily/13_digest"
    m.REJECTED_DIR = m.DIGEST_DIR / "_rejected"
    monkeypatch.setattr(m, "collect", lambda *a: [("x", SRC, "text")])
    monkeypatch.setattr(m, "load_withheld", lambda: (
        lambda t, mm: (t, 0), type("M", (), {"matches": lambda s, x: False})(), "[열람 보류]"))
    monkeypatch.setattr(m, "load_hea", lambda start, end, withheld=None: (HEA, None))
    monkeypatch.setattr(m, "hea_table_md", lambda r: TABLE + "\n\n언급만 된 실험(표에서 제외, 1건): EXP-001")
    monkeypatch.setattr(m, "hea_material", lambda r: ("HEA 카드 발췌 (2개 실험)", "weekly_hea.py", "[EXP-025] cards"))
    return m


def run_main(md, monkeypatch, *argv):
    monkeypatch.setattr(md.sys, "argv", ["make_digest.py", *argv])
    return md.main()


def out_text(d, label="2026-W45"):
    return (d / (label + ".md")).read_text(encoding="utf-8")


def test_weekly_llm_inserts_table_and_frontmatter(md, monkeypatch, tmp_path):
    seen = {}

    def fake(prompt, material):
        seen["prompt"], seen["material"] = prompt, material
        return good_body(), None

    monkeypatch.setattr(md, "run_claude", fake)
    out = tmp_path / "out"
    assert run_main(md, monkeypatch, "--weekly", "2026-11-02", "--out-dir", str(out)) == 0
    t = out_text(out)
    assert 'mode: "llm"' in t and 'lane: "subscription"' in t and 'postcheck: "pass"' in t
    assert "hea_rows: 2" in t and "hea_mention_only: 1" in t and "hea_unextractable: 1" in t
    assert "{{HEA_TABLE}}" not in t and TABLE in t
    assert t.index("## 한 줄") < t.index("## 가설 × 실험: 예상 대 실제") < t.index(TABLE) \
        < t.index("EXP-015(=EXP-025)") < t.index("## 실험: 돌아간 것")
    assert "{{HEA_TABLE}}" in seen["prompt"] and "1500단어" in seen["prompt"]
    assert seen["material"].rstrip().endswith("[EXP-025] cards")          # pinned block is last
    assert not (md.LAB / "92_daily/13_digest").exists()                     # --out-dir: real dir untouched


def test_model_written_table_rows_are_replaced(md, monkeypatch, tmp_path):
    body = good_body().replace("{{HEA_TABLE}}\n", "| 가짜 | 표 |\n|---|---|\n| H-199 | EXP-025 지어냄 |\n")
    monkeypatch.setattr(md, "run_claude", lambda p, m: (body, None))
    out = tmp_path / "out"
    assert run_main(md, monkeypatch, "--weekly", "2026-11-02", "--out-dir", str(out)) == 0
    t = out_text(out)
    assert "지어냄" not in t and TABLE in t and 'mode: "llm"' in t


def test_unknown_id_in_hea_section_fails_postcheck_then_retry(md, monkeypatch, tmp_path):
    outs = iter([(good_body("- EXP-999도 봤다.\n"), None), (good_body(), None)])
    prompts = []

    def fake(prompt, material):
        prompts.append(prompt)
        return next(outs)

    monkeypatch.setattr(md, "run_claude", fake)
    out = tmp_path / "out"
    assert run_main(md, monkeypatch, "--weekly", "2026-11-02", "--out-dir", str(out)) == 0
    t = out_text(out)
    assert 'mode: "llm"' in t and "postcheck_retries: 1" in t and len(prompts) == 2
    assert "EXP-999" in prompts[1] and "EXP-999" not in t
    assert (out / "_rejected" / "2026-W45.md").exists()


def test_missing_hea_section_fails(md):
    body = good_body().replace("## 가설 × 실험: 예상 대 실제\n{{HEA_TABLE}}\n", "")
    assert any("섹션 없음" in f for f in md.postcheck_hea(body, {"EXP-025"}))


def test_alias_and_mention_only_ids_are_allowed(md):
    ids = md.hea_ids(HEA)
    assert {"EXP-025", "EXP-015", "EXP-007", "EXP-001", "H-199", "H-A", "H-022"} <= ids
    assert md.postcheck_hea(good_body("- EXP-001, H-A, H-022\n"), ids) == []


def test_fallback_week_still_has_table(md, monkeypatch, tmp_path):
    out = tmp_path / "out"
    assert run_main(md, monkeypatch, "--weekly", "2026-11-02", "--no-llm", "--out-dir", str(out)) == 0
    t = out_text(out)
    assert 'mode: "fallback"' in t and 'lane: "none"' in t and TABLE in t
    assert t.index("## 한 줄") < t.index("## 가설 × 실험: 예상 대 실제") < t.index("## 실험: 돌아간 것")


def test_hea_failure_does_not_block_digest(md, monkeypatch, tmp_path):
    monkeypatch.setattr(md, "load_hea", lambda s, e, withheld=None: (None, "ValueError: boom"))
    monkeypatch.setattr(md, "run_claude", lambda p, m: (good_body(), None))
    out = tmp_path / "out"
    assert run_main(md, monkeypatch, "--weekly", "2026-11-02", "--out-dir", str(out)) == 0
    t = out_text(out)
    assert "hea_error" in t and "HEA 표 생성 실패" in t


def test_real_load_hea_never_raises(md, monkeypatch):
    import importlib
    spec = importlib.util.spec_from_file_location("md_real", MOD)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    monkeypatch.setattr(m, "HERE", Path("/nonexistent-dir-for-test"))
    monkeypatch.setitem(__import__("sys").modules, "weekly_hea", None)       # import fails
    res, err = m.load_hea(None, None)
    assert res is None and err


# ---- lanes -------------------------------------------------------------------------------------------------------
def test_subscription_failure_falls_to_metered_weekly(md, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(md, "run_claude", lambda p, m: (None, "claude CLI rc=1: stderr=[] stdout_tail=[limit]"))

    def metered(prompt, material, label):
        calls.append(label)
        return good_body(), None

    monkeypatch.setattr(md, "run_metered", metered)
    out = tmp_path / "out"
    assert run_main(md, monkeypatch, "--weekly", "2026-11-02", "--out-dir", str(out)) == 0
    t = out_text(out)
    assert calls == ["2026-W45"] and 'lane: "metered"' in t and 'mode: "llm"' in t
    assert "subscription_failure" in t and "stdout_tail=[limit]" in t


def test_metered_retry_stays_on_metered(md, monkeypatch, tmp_path):
    sub, met = [], []
    monkeypatch.setattr(md, "run_claude", lambda p, m: (sub.append(1), (None, "rc=1"))[1])
    outs = iter([(good_body("- EXP-999\n"), None), (good_body(), None)])
    monkeypatch.setattr(md, "run_metered", lambda p, m, label: (met.append(1), next(outs))[1])
    out = tmp_path / "out"
    assert run_main(md, monkeypatch, "--weekly", "2026-11-02", "--out-dir", str(out)) == 0
    assert len(sub) == 1 and len(met) == 2 and 'lane: "metered"' in out_text(out)


def test_both_lanes_fail_gives_fallback_with_both_reasons(md, monkeypatch, tmp_path):
    monkeypatch.setattr(md, "run_claude", lambda p, m: (None, "claude CLI rc=1: A"))
    monkeypatch.setattr(md, "run_metered", lambda p, m, label: (None, "llm_metered rc=3: cap"))
    out = tmp_path / "out"
    assert run_main(md, monkeypatch, "--weekly", "2026-11-02", "--out-dir", str(out)) == 0
    t = out_text(out)
    assert 'mode: "fallback"' in t and "rc=1: A" in t and "rc=3: cap" in t and TABLE in t


def test_daily_never_uses_metered_and_keeps_old_contract(md, monkeypatch, tmp_path):
    monkeypatch.setattr(md, "run_claude", lambda p, m: (None, "claude CLI rc=1: A"))
    monkeypatch.setattr(md, "run_metered", lambda *a: pytest.fail("metered lane used for a daily digest"))
    out = tmp_path / "out"
    assert run_main(md, monkeypatch, "--daily", "2026-11-01", "--out-dir", str(out)) == 0
    t = out_text(out, "2026-11-01")
    assert 'mode: "fallback"' in t and "lane:" not in t and "가설 × 실험" not in t and "hea_" not in t


def test_run_metered_command_and_failure_reason(md, monkeypatch):
    seen = {}

    def fake(cmd, **kw):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 3, stdout=b"", stderr=b"llm_metered: rc=3 monthly cap\nreached")

    monkeypatch.setattr(md.subprocess, "run", fake)
    text, reason = md.run_metered("p", "m", "2026-W46")
    cmd = seen["cmd"]
    assert text is None and "rc=3" in reason and "\n" not in reason
    assert cmd[0].endswith("with-anthropic-key") and "claude" not in [Path(c).name for c in cmd]
    assert cmd[cmd.index("--with") + 1] == "anthropic==1.13.0" and cmd[-4:] == ["--kind", "weekly", "--label", "2026-W46"]


# ---- bundle ------------------------------------------------------------------------------------------------------
def test_pinned_block_is_never_trimmed(md):
    items = [("old%d" % i, "p", "x" * 20000) for i in range(6)]
    pin = ("HEA 카드 발췌", "weekly_hea.py", "PINNED-" + "y" * 50000)
    text, truncated = md.bundle(items, pinned=pin)
    assert "PINNED-" in text and text.rstrip().endswith("y" * 100) and truncated
    assert "old5" in text                                   # newest non-pinned item survives
    plain, tr2 = md.bundle(items)
    assert "PINNED-" not in plain and len(tr2) <= len(truncated)


# ---- final-verify r1 FAIL fixes (2026-11-10) -------------------------------------------------------------------------
def test_subscription_retry_call_failure_does_not_go_metered(md, monkeypatch, tmp_path):
    outs = iter([(good_body("- EXP-999\n"), None), (None, "claude CLI rc=1: later")])
    monkeypatch.setattr(md, "run_claude", lambda p, m: next(outs))
    monkeypatch.setattr(md, "run_metered", lambda *a: pytest.fail("retry left the subscription lane"))
    out = tmp_path / "out"
    assert run_main(md, monkeypatch, "--weekly", "2026-11-02", "--out-dir", str(out)) == 0
    t = out_text(out)
    assert 'mode: "fallback"' in t and 'lane: "none"' in t and "rc=1: later" in t


@pytest.mark.parametrize("bad", [
    {"cards": [{"mention_only": True}]},                                        # no exp_id
    {"cards": [{"exp_id": "EXP-001", "hypotheses": ["H-022", 3], "mention_only": False}]},
    {"cards": [{"exp_id": "EXP-001", "hypotheses": None, "aliases": 5}]},
    {"cards": "nope"},
    ["not", "a", "dict"],
    None,
])
def test_malformed_extractor_result_never_blocks_weekly(md, monkeypatch, tmp_path, bad):
    import sys
    import types
    fake = types.ModuleType("weekly_hea")
    fake.build_cards = lambda *a, **k: bad
    fake.render_table = lambda r: (_ for _ in ()).throw(KeyError("boom"))
    fake.material_block = lambda r: (_ for _ in ()).throw(KeyError("boom"))
    monkeypatch.setitem(sys.modules, "weekly_hea", fake)
    for name in ("load_hea", "hea_table_md", "hea_material"):                  # use the REAL helpers
        monkeypatch.undo() if False else None
    spec = importlib.util.spec_from_file_location("md_real_helpers", MOD)
    real = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(real)
    monkeypatch.setattr(md, "load_hea", real.load_hea)
    monkeypatch.setattr(md, "hea_table_md", real.hea_table_md)
    monkeypatch.setattr(md, "hea_material", real.hea_material)
    monkeypatch.setattr(md, "run_claude", lambda p, m: (good_body(), None))
    out = tmp_path / "out"
    assert run_main(md, monkeypatch, "--weekly", "2026-11-02", "--out-dir", str(out)) == 0
    assert (out / "2026-W45.md").exists()


def test_withheld_loader_any_exception_falls_back(md, monkeypatch, tmp_path):
    monkeypatch.setattr(md, "load_withheld", lambda: (_ for _ in ()).throw(OSError("nas gone")))
    out = tmp_path / "out"
    assert run_main(md, monkeypatch, "--weekly", "2026-11-02", "--out-dir", str(out)) == 0
    assert 'mode: "fallback"' in out_text(out) and "nas gone" in out_text(out)


def test_queue_lines_carry_run_names_not_queue_numbers(md):
    assert md._run_name("0696_fig1x2_tf_noz_iter_k7_s1.job") == "fig1x2_tf_noz_iter_k7_s1"
    assert md._run_name("fig1_plain") == "fig1_plain"


def test_hea_sane_drops_bad_cards_and_rejects_bad_shapes(md):
    good = {"cards": [{"exp_id": "EXP-001", "hypotheses": [{"id": "H-022"}, "junk", {"id": 3}], "aliases": 7},
                      {"mention_only": True}, "junk", {"exp_id": 12}]}
    out = md._hea_sane(good)
    assert [c["exp_id"] for c in out["cards"]] == ["EXP-001"]
    assert out["cards"][0]["hypotheses"] == [{"id": "H-022"}] and out["cards"][0]["aliases"] == []
    for bad in (None, [], {"cards": "x"}, {}):
        with pytest.raises(ValueError):
            md._hea_sane(bad)


def test_non_string_table_becomes_error_text_not_a_crash(md, monkeypatch):
    import sys
    import types
    spec = importlib.util.spec_from_file_location("md_real_tbl", MOD)
    real = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(real)
    fake = types.ModuleType("weekly_hea")
    fake.render_table = lambda r: None
    monkeypatch.setitem(sys.modules, "weekly_hea", fake)
    out = real.hea_table_md({"cards": [{"exp_id": "EXP-001", "mention_only": False}]})
    assert isinstance(out, str) and "렌더 실패" in out
    assert isinstance(real.insert_hea("# t\n## 가설 × 실험: 예상 대 실제\n{{HEA_TABLE}}\n## 출처\n", out), str)
