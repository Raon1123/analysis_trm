import json
from pathlib import Path

import pytest

from tools.api_batch import common
from tools.api_batch.common import Halt, sha256_file
from tools.api_batch.pack import PackBuilder, PackError, redact, select_sections

PROBES = common.PROBES_DIR
PINNED = Path(__file__).with_name("probe_sha256.json")


def test_probe_files_pinned():
    pins = json.loads(PINNED.read_text())
    assert set(pins) == {p.name for p in PROBES.glob("*.txt")}
    for n, h in pins.items():
        assert sha256_file(PROBES / n) == h, n


def test_deny_absent_halts_even_for_probes(tmp_path):
    with pytest.raises(Halt):
        PackBuilder.from_files([tmp_path / "missing.yaml"])


def test_probe_pack_ok_and_scoped(builder, tmp_path):
    p = builder.build_probe("p0_haiku_short.txt")
    assert p.kind == "probe" and p.sealed and p.mode == "verify-only"
    outside = tmp_path / "x.txt"
    outside.write_text("hello")
    for bad in ("../tests/conftest.py", str(outside), "p0_nonexistent.txt", "../probes/../__init__.py"):
        with pytest.raises(PackError):
            builder.build_probe(bad)


def write(lab, rel, text):
    p = lab / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return rel


def test_generation_redacts_blocks_and_residue_check(builder, lab):
    rel = write(lab, "11_governance/11_lab-root/STATE.md",
                "# State\n\n## 0. Now\nplain line\n\nEXP-013 produced numbers 0.91.\n\nrun fig1_mlp_z_iter_k3_s2 got 0.5\n\nA neutral paragraph.\n\n"
                "| run | acc |\n|---|---|\n| keep | 1 |\n| clean-r1 | 2 |\n")
    pk = builder.build_generation("ctx", [{"path": rel}])
    assert "0.91" not in pk.text and "fig1_mlp" not in pk.text and "clean-r1" not in pk.text
    assert "plain line" in pk.text and "| keep | 1 |" in pk.text
    assert pk.report["masked_blocks"] >= 3 and pk.sources[0]["sha256"] == sha256_file(lab / rel)


def test_run_name_only_paragraph_masked(builder, lab):
    rel = write(lab, "28_literature/n.md", "intro\n\nthe cohort cleanr1_d1_k4_sm2 reached 0.77 accuracy\n\nend\n")
    assert "0.77" not in builder.build_generation("c", [{"path": rel}]).text


LEAD = ("Results of EXP-012 are below:\n\n- first 0.11\n- second 0.22\n- third 0.33\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n"
        "After text stays.\n")


def test_lead_in_masks_all_consecutive_blocks_nb_r3_2(deny_files):
    from tools.api_batch import deny
    m = deny.load_deny(deny_files).matcher
    _, r = common.load_trm_mcp()
    alone, _n = r.mask_blocks(LEAD, m)
    assert "0.22" in alone and "0.33" in alone and "| 1 | 2 |" in alone       # mask_blocks alone leaves them (documented)
    out, n = redact(LEAD, m)
    for leaked in ("0.11", "0.22", "0.33", "| 1 | 2 |"):
        assert leaked not in out, leaked
    assert "After text stays." in out and n >= 5


def test_verify_only_refuses_never_masks(builder, lab):
    ok = builder.build_verify_only("seat_input", "r.md", "A RESULTS text with the word verdict and EXP-015.", None)
    assert ok.mode == "verify-only" and ok.text.startswith("A RESULTS")
    for txt in ("mentions EXP 013 here", "run fig1_tf_noz_noiter_k5_s3", "EXP-\n016 split"):
        with pytest.raises(PackError):
            builder.build_verify_only("seat_input", "r.md", txt, None)
    with pytest.raises(PackError):
        builder.build_spec_text("perspective-1", "look at EXP-016 specifically")


def test_whole_text_l3b_catches_paragraph_split(builder, lab):
    rel = write(lab, "28_literature/split.md", "see EXP-\n\n013 results\n")
    with pytest.raises(PackError):
        builder.build_generation("c", [{"path": rel}])


def test_l1_allow_list_default_deny(builder, lab):
    for rel in ("22_plans/20_DECISIONS_x.md", "99_misc/notes.md", "12_other/x.md"):
        write(lab, rel, "text")
        with pytest.raises(PackError):
            builder.build_generation("c", [{"path": rel}])
    with pytest.raises(PackError):
        builder.build_generation("c", [{"path": "/etc/hostname"}])


def test_l2_path_and_frontmatter(builder, lab):
    write(lab, "28_literature/per_call.csv", "x")
    with pytest.raises(PackError):
        builder.build_generation("c", [{"path": "28_literature/per_call.csv"}])
    rel = write(lab, "27_theory/11_lab-theory/H-900.md", "---\nexp_id: EXP-016\n---\nbody\n")
    with pytest.raises(PackError):
        builder.build_generation("c", [{"path": rel}])
    ok = write(lab, "27_theory/11_lab-theory/H-901.md", "---\nexp_id: EXP-015\n---\nbody\n")
    assert "body" in builder.build_generation("c", [{"path": ok}]).text


def test_l4_cap_and_secrets(builder, lab):
    rel = write(lab, "28_literature/big.md", "word " * 100000)
    with pytest.raises(PackError):
        builder.build_generation("c", [{"path": rel}])
    rel2 = write(lab, "28_literature/key.md", "token sk-ant-api03-abcdefgh12345\n")
    with pytest.raises(PackError):
        builder.build_generation("c", [{"path": rel2}])


def test_select_sections():
    t = "# T\n\n## 0. A\na\n\n### sub\nsa\n\n## 1. B\nb\n\n## 2. C\nc\n"
    s = select_sections(t, ["0", "2"])
    assert "sa" in s and "\nb" not in s and "c" in s
    with pytest.raises(PackError):
        select_sections(t, ["9"])


def test_derived_requires_l5_pass_and_inbox(builder, lab, tmp_path):
    bd = tmp_path / "root" / "11_batches" / "b1"
    (bd / "items").mkdir(parents=True)
    b = PackBuilder(builder.deny, lab_root=lab, batch_dir=bd)
    inbox = tmp_path / "root" / "INBOX.jsonl"
    inbox.write_text(json.dumps({"custom_id": "good"}) + "\n")
    (bd / "items" / "good.md").write_text("---\nstatus: DRAFT\nl5: pass\n---\nbody\n")
    (bd / "items" / "quar.md").write_text("---\nstatus: DRAFT\nl5: quarantined\n---\nbody\n")
    (bd / "items" / "unlisted.md").write_text("---\nstatus: DRAFT\nl5: pass\n---\nbody\n")
    assert b.build_derived(bd / "items" / "good.md", inbox).kind == "derived"
    for n in ("quar", "unlisted"):
        with pytest.raises(PackError):
            b.build_derived(bd / "items" / f"{n}.md", inbox)


def test_u1_data_wandb_binary_refused(builder, tmp_path):
    repo_like = PackBuilder(builder.deny, lab_root=builder.lab_root, repo_root=tmp_path)
    p = tmp_path / "wandb/run-1/files/wandb-summary.json"
    p.parent.mkdir(parents=True)
    p.write_text('{"acc": 0.5}')
    assert repo_like.build_verify_only("u1_data", "s", p.read_text(), origin=p).kind == "u1_data"
    q = tmp_path / "wandb/run-1/run-1.wandb"
    q.write_text("x")
    with pytest.raises(PackError):
        repo_like.build_verify_only("u1_data", "s", "x", origin=q)


def test_deny_absent_halts_probe_plan_through_cli(tmp_path, lab):
    """Probe path with no deny source: the builder cannot exist, so build_probe is unreachable. Through the CLI:
    rc 3, nothing written (no manifest, plan.json, pack dir, planned row)."""
    from tools.api_batch import cli
    from tools.api_batch.ledger import Ledger
    bd = tmp_path / "root" / "11_batches" / "x"
    bd.mkdir(parents=True)
    (bd / "spec.yaml").write_text("name: x\nuse: probe\nbudget: {cap_usd_hard_worst: 0.5}\nprobe_requests:\n"
                                  "  - {id: probe-haiku-s1, model: claude-haiku-5-5, effort: low, max_tokens: 1024, var: p0_haiku_short.txt}\n")
    rc = cli.main(["--batch-root", str(tmp_path / "root"), "--lab-root", str(lab), "--deny-file", str(tmp_path / "nope.yaml"), "plan", str(bd)])
    assert rc == 3
    assert sorted(p.name for p in bd.iterdir()) == ["spec.yaml"] and Ledger(tmp_path / "root" / "ledger.jsonl").rows() == []
    with pytest.raises(Halt):
        PackBuilder.from_files([tmp_path / "nope.yaml"]).build_probe("p0_haiku_short.txt")
