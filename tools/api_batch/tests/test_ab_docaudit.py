"""docaudit lane (tool-less document-audit reader seat): spec, per-file fail-closed selection, chunking, sharding,
deterministic ids, and the offline quote post-check. Synthetic lab only; no network."""
import json
import os

import pytest

from tools.api_batch import docaudit as DA
from tools.api_batch import manifest as M
from tools.api_batch import ops
from tools.api_batch.common import Halt
from tools.api_batch.ledger import Ledger
from tools.api_batch.pack import PackBuilder
from tools.api_batch.spec import SpecError, validate_spec

HEADER = "# Project context\n\nThe tree uses a numbered layout. Old flat paths are historical.\n"
SPEC = {
    "name": "da-test", "use": "docaudit", "models": ["claude-sonnet-5-5", "claude-opus-5-5"], "effort": "medium",
    "max_tokens": 4000, "cache_ttl": "5m", "header": "shared/header.md",
    "prompt": {"system_template": "tools/api_batch/prompts/docaudit_reader.md"}, "output_schema": "docaudit_findings",
    "files": {"list": "shared/files.txt"}, "shard": {"target_tokens": 400, "max_file_tokens": 600, "chunk_tokens": 300},
    "budget": {"cap_usd_hard_worst": 5.0},
}
BIG = "".join(f"line {i:03d} of the big file with some filler text to make it long enough\n" for i in range(1, 61))
FILES = {
    "11_gov/a.md": "# A\n\nupdated: 2026-11-10\n\nThe count is 8.\n",
    "11_gov/b.md": "# B\n\nThe count is 12.\n",
    "11_gov/big.md": BIG,
    "11_gov/deny_body.md": "see the table in per_call.csv for details\n",
    "11_gov/sealed/s.md": "nothing special\n",
    "11_gov/DECISIONS_2026.md": "a decision ledger\n",
    "11_gov/fm.md": "---\nexp_id: EXP-016\n---\nbody\n",
    "11_gov/secret.md": "key sk-ant-abcdefgh12345678 here\n",
    "11_gov/empty.md": "   \n",
    "23_rep/z.md": "# Z\n\nA report.\n",
}
LISTED = [*FILES, "11_gov/missing.md"]


@pytest.fixture
def env(tmp_path, deny_files, lab):
    for rel, text in FILES.items():
        p = lab / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    bd = tmp_path / "root" / "11_batches" / "2026-11-11_da"
    (bd / "shared").mkdir(parents=True)
    (bd / "shared/header.md").write_text(HEADER, encoding="utf-8")
    (bd / "shared/files.txt").write_text("# pilot list\n" + "\n".join(LISTED) + "\n", encoding="utf-8")
    (bd / "spec.yaml").write_text(json.dumps(SPEC), encoding="utf-8")          # JSON is YAML
    builder = PackBuilder.from_files(deny_files, lab_root=lab, batch_dir=bd)
    return type("E", (), dict(bd=bd, lab=lab, builder=builder, root=tmp_path / "root",
                              led=Ledger(lab / "38_api-batch" / "ledger.jsonl"), deny=deny_files))


def _bodies(bd):
    rows, _ = M.load_manifest(bd / "manifest.jsonl")
    return rows, [json.dumps(r["params"], ensure_ascii=False) for r in rows]


# ------------------------------------------------------------------------------------------------ spec
@pytest.mark.parametrize("patch", [
    {"models": []}, {"models": ["claude-sonnet-5-5"] * 2}, {"models": ["gpt-x"]},
    {"models": ["claude-sonnet-5-5", "claude-opus-5-5", "claude-haiku-5-5"]},
    {"cache_ttl": "none"}, {"header": "header.md"}, {"prompt": {"system_template": "x.md"}},
    {"files": {"list": "shared/files.txt", "roots": ["11_gov"]}}, {"files": {}}, {"files": {"list": "files.txt"}},
    {"files": {"roots": ["/abs"]}}, {"files": {"roots": ["a/../b"]}}, {"files": {"list": "shared/files.txt", "exclude": ["x"]}},
    {"shard": {"target_tokens": 0}}, {"shard": {"chunk_tokens": 900, "max_file_tokens": 600}}, {"shard": {"other": 1}},
    {"effort": "auto"}, {"max_tokens": True}, {"surprise": 1},
])
def test_spec_refusals(patch):
    with pytest.raises(SpecError):
        validate_spec({**SPEC, **patch})


def test_spec_ok_and_missing_key():
    assert validate_spec(dict(SPEC))["use"] == "docaudit"
    assert validate_spec({**SPEC, "files": {"roots": ["11_gov"], "exclude": ["*/big.md"]}})
    bad = dict(SPEC)
    del bad["files"]
    with pytest.raises(SpecError):
        validate_spec(bad)


# ------------------------------------------------------------------------------------------------ plan
def test_plan_skips_per_file_and_never_sends_denied_text(env):
    d = ops.plan(env.bd, env.builder, env.led)
    info = d["docaudit"]
    assert info["n_files_listed"] == len(LISTED)
    sent = {f.split("#")[0] for s in info["shards"] for f in s["files"]}
    assert sent == {"11_gov/a.md", "11_gov/b.md", "11_gov/big.md", "23_rep/z.md"}
    assert info["n_files_sent"] == 4
    layers = {s["rel"]: s["layer"] for s in info["skipped"]}
    assert layers == {"11_gov/deny_body.md": "L3b", "11_gov/sealed/s.md": "L2", "11_gov/DECISIONS_2026.md": "L1",
                      "11_gov/fm.md": "L2", "11_gov/secret.md": "L4", "11_gov/empty.md": "empty",
                      "11_gov/missing.md": "missing"}
    assert info["n_skipped"] == 7 and sum(info["skipped_by_layer"].values()) == 7
    rows, bodies = _bodies(env.bd)
    for needle in ("per_call.csv", "EXP-016", "sk-ant-", "decision ledger", "nothing special"):
        assert not any(needle in b for b in bodies), needle
    # nothing of a skipped file reaches the pack directory either
    packed = "".join(p.read_text(encoding="utf-8") for p in (env.bd / "pack").glob("*.txt"))
    assert "per_call.csv" not in packed and "sk-ant-" not in packed
    assert d["n_requests"] == info["n_requests"] == info["n_shards"] * 2 == len(rows)
    assert d["est_usd_hard_worst"] > 0 and env.led.last("planned")["n_requests"] == len(rows)


def test_request_shape_and_cache_prefix(env):
    ops.plan(env.bd, env.builder, env.led)
    rows, _ = _bodies(env.bd)
    systems = {r["params"]["system"][0]["text"] for r in rows}
    assert len(systems) == 1
    for r in rows:
        content = r["params"]["messages"][0]["content"]
        assert content[0]["type"] == "document" and content[0]["source"]["data"] == HEADER
        assert content[0]["cache_control"] == {"type": "ephemeral"}
        assert all(b["type"] == "text" and "cache_control" not in b for b in content[1:])
        assert r["params"]["model"] == r["meta"]["model"] and r["params"]["max_tokens"] == 4000
        assert r["params"]["output_config"]["format"]["type"] == "json_schema"
        for b, f in zip(content[1:], r["meta"]["files"]):
            assert b["text"].startswith(f"===== FILE {f['rel']} (") and b["text"].endswith(f"===== END FILE {f['rel']} =====\n")
    assert {r["meta"]["model"] for r in rows} == set(SPEC["models"])
    # both models get exactly the same shards
    by = {}
    for r in rows:
        by.setdefault(r["meta"]["shard"], set()).add(r["meta"]["model"])
    assert all(v == set(SPEC["models"]) for v in by.values())


def test_shards_do_not_cross_top_level_and_chunks_cover_the_file(env):
    d = ops.plan(env.bd, env.builder, env.led)
    info = d["docaudit"]
    for s in info["shards"]:
        assert len({f.split("/", 1)[0] for f in s["files"]}) == 1
    assert [c["rel"] for c in info["chunked"]] == ["11_gov/big.md"] and info["chunked"][0]["parts"] >= 2
    rows, _ = _bodies(env.bd)
    one_model = [r for r in rows if r["meta"]["model"] == SPEC["models"][0]]
    spans, text = [], ""
    for r in one_model:
        for f, b in zip(r["meta"]["files"], r["params"]["messages"][0]["content"][1:]):
            if f["rel"] == "11_gov/big.md":
                spans.append(tuple(f["lines"]))
                text += b["text"].split("\n", 1)[1].rsplit("\n===== END FILE", 1)[0] + "\n"
    assert spans[0][0] == 1 and spans[-1][1] == 60
    assert all(spans[i][1] + 1 == spans[i + 1][0] for i in range(len(spans) - 1))
    assert text == BIG                                             # every line once, in order, byte-identical


def test_replan_is_deterministic_and_tracks_file_changes(env):
    ops.plan(env.bd, env.builder, env.led)
    ids1 = [r["custom_id"] for r in _bodies(env.bd)[0]]
    ops.plan(env.bd, env.builder, env.led)
    assert ids1 == [r["custom_id"] for r in _bodies(env.bd)[0]] and len(set(ids1)) == len(ids1)
    (env.lab / "11_gov/a.md").write_text(FILES["11_gov/a.md"] + "one more line\n", encoding="utf-8")
    ops.plan(env.bd, env.builder, env.led)
    ids2 = [r["custom_id"] for r in _bodies(env.bd)[0]]
    assert ids1 != ids2 and len(ids1) == len(ids2)


def test_deny_hit_in_header_refuses_the_whole_plan(env):
    (env.bd / "shared/header.md").write_text(HEADER + "see per_call.csv\n", encoding="utf-8")
    with pytest.raises(Halt):
        ops.plan(env.bd, env.builder, env.led)
    assert not (env.bd / "manifest.jsonl").exists()


def test_nothing_to_send_and_cap(env):
    (env.bd / "shared/files.txt").write_text("11_gov/deny_body.md\n11_gov/missing.md\n", encoding="utf-8")
    with pytest.raises(DA.DocAuditError):
        ops.plan(env.bd, env.builder, env.led)
    (env.bd / "shared/files.txt").write_text("11_gov/a.md\n", encoding="utf-8")
    (env.bd / "spec.yaml").write_text(json.dumps({**SPEC, "budget": {"cap_usd_hard_worst": 0.0001}}), encoding="utf-8")
    with pytest.raises(Halt):
        ops.plan(env.bd, env.builder, env.led)


def test_list_file_errors(env):
    for body in ("\n# only a comment\n", "11_gov/a.md\n11_gov/a.md\n", "/etc/passwd\n", "11_gov/../x.md\n"):
        (env.bd / "shared/files.txt").write_text(body, encoding="utf-8")
        with pytest.raises(DA.DocAuditError):
            ops.plan(env.bd, env.builder, env.led)


def test_roots_walk_skips_generated_backup_archive_and_symlinks(env):
    for rel in ("11_gov/before/old.md", "11_gov/x_backups/o.md", "11_gov/99_Archive/o.md", "21_exp/12_run-cards/P/c.md",
                "11_gov/backup_retune/o.md", "11_gov/notes.txt", "11_gov/sub/keep.md"):
        p = env.lab / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x\n", encoding="utf-8")
    os.symlink(env.lab / "23_rep", env.lab / "11_gov" / "link_to_rep")
    rels = DA.walk_roots(env.lab, ["11_gov", "21_exp" if (env.lab / "21_exp").exists() else "11_gov"], ["*/DECISIONS_*", "11_gov/secret.md"])
    assert "11_gov/sub/keep.md" in rels and "11_gov/a.md" in rels
    assert not any(x in r for r in rels for x in ("before/", "_backups/", "99_Archive/", "12_run-cards/", "backup_retune/",
                                                  "link_to_rep", "DECISIONS", "secret.md", ".txt"))
    with pytest.raises(DA.DocAuditError):
        DA.walk_roots(env.lab, ["nope"], [])


# ----------------------------------------------------------------------------------------------- units
def test_chunk_lines_covers_every_line_once():
    text = "".join(f"row {i}\n" for i in range(1, 101)) + "last line without newline"
    parts = DA.chunk_lines(text, 20)
    assert parts[0][0] == 1 and parts[-1][1] == 101 and len(parts) > 5
    assert "\n".join(p[2] for p in parts) == text
    assert all(parts[i][1] + 1 == parts[i + 1][0] for i in range(len(parts) - 1))
    long = "x" * 500 + "\nshort\n"
    assert [(a, b) for a, b, _ in DA.chunk_lines(long, 10)] == [(1, 1), (2, 2)]


def test_locate():
    t = "alpha\nbeta  gamma\n  delta epsilon\nzeta\n"
    assert DA.locate(t, "beta  gamma") == (2, "exact")
    assert DA.locate(t, "gamma delta epsilon") == (2, "normalised")
    assert DA.locate(t, "delta   epsilon") == (3, "normalised")
    assert DA.locate(t, "not here") == (0, "not-found")
    assert DA.locate(t, "  ") == (0, "empty")


def test_model_tag_and_wrap():
    assert DA.model_tag("claude-opus-5-5") == "opus55" and DA.model_tag("claude-sonnet-5-5") == "sonnet55"
    assert DA.wrap("a/b.md", 1, 3, 3, "x").startswith("===== FILE a/b.md (whole file) =====\n")
    assert "(lines 4-9 of 20)" in DA.wrap("a/b.md", 4, 9, 20, "x")


# ---------------------------------------------------------------------------------------------- report
def _item(body):
    return "---\nstatus: DRAFT (api-batch output — not evidence)\nresult_status: succeeded\n---\n" + body + "\n"


def test_report_resolves_quotes_and_flags_fabrications(env):
    ops.plan(env.bd, env.builder, env.led)
    rows, _ = _bodies(env.bd)
    target = next(r for r in rows if any(f["rel"] == "11_gov/a.md" for f in r["meta"]["files"]))
    others = [r for r in rows if r is not target]
    f = lambda **kw: {"file": "11_gov/a.md", "quote": "", "category": "inconsistent", "basis": "internal", "severity": "low",  # noqa: E731
                      "frozen": False, "claim": "c", "counter_file": "11_gov/b.md", "counter_quote": "The count is 12.", "check": "", **kw}
    out = {"files_read": [{"file": "11_gov/a.md", "last_line_read": 5}], "notes": "",
           "findings": [f(quote="The count is 8."), f(quote="The count is 9."), f(quote="The   count is 8."),
                        f(file="11_gov/other.md", quote="x", basis="needs-check")]}
    (env.bd / "items").mkdir()
    (env.bd / "items" / f"{target['custom_id']}.md").write_text(_item(json.dumps(out, ensure_ascii=False)), encoding="utf-8")
    (env.bd / "items" / f"{others[0]['custom_id']}.md").write_text(_item("not json"), encoding="utf-8")
    (env.bd / "quarantine").mkdir()
    (env.bd / "quarantine" / f"{others[1]['custom_id']}.md").write_text("q", encoding="utf-8")
    inbox = [{"custom_id": target["custom_id"], "result_status": "succeeded"},
             {"custom_id": others[0]["custom_id"], "result_status": "succeeded"},
             {"custom_id": others[2]["custom_id"], "result_status": "max_tokens"}]
    (env.bd / "items" / f"{others[2]['custom_id']}.md").write_text(_item("{}"), encoding="utf-8")
    (env.root / "INBOX.jsonl").write_text("".join(json.dumps(r) + "\n" for r in inbox), encoding="utf-8")
    summ = DA.report(env.bd, lab_root=env.lab)
    got = [json.loads(ln) for ln in (env.bd / "docaudit/findings.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(g["line"], g["quote_match"], g["quote_found"]) for g in got] == [
        (5, "exact", True), (0, "not-found", False), (5, "normalised", True), (0, "file-not-in-request", False)]
    assert got[0]["file_changed_since_plan"] is False and got[3]["file_in_request"] is False
    st = summ["by_status"]
    assert st["succeeded"] == 1 and st["bad-json"] == 1 and st["quarantined"] == 1 and st["max_tokens"] == 1
    assert st.get("no-item", 0) == len(rows) - 4
    m = summ["by_model"][target["meta"]["model"]]
    assert (m["findings"], m["quote_found"], m["quote_not_found"]) == (4, 2, 2) and m["basis"] == {"internal": 3, "needs-check": 1}
    assert (env.bd / "docaudit/REPORT.md").read_text(encoding="utf-8").startswith("---\nstatus: DRAFT")
    # a file edited after the plan is flagged
    (env.lab / "11_gov/a.md").write_text("changed\n", encoding="utf-8")
    DA.report(env.bd, lab_root=env.lab)
    got = [json.loads(ln) for ln in (env.bd / "docaudit/findings.jsonl").read_text(encoding="utf-8").splitlines()]
    assert got[0]["file_changed_since_plan"] is True and got[0]["quote_found"] is False


def test_report_needs_collect(env):
    ops.plan(env.bd, env.builder, env.led)
    with pytest.raises(DA.DocAuditError):
        DA.report(env.bd, lab_root=env.lab)
    assert DA.main(["nope"]) == 2


# ------------------------------------------------------------------- VERIFY-BATCH-r1 minors (M1-M3, M6-M8, M10)
def test_hit_only_in_the_sent_text_skips_the_file_not_the_plan(env):
    # raw text passes every layer; the marker line appended by wrap() completes the L4 pattern `ANTHROPIC_API_KEY\s*=`
    (env.lab / "11_gov/tail.md").write_text("how to export ANTHROPIC_API_KEY", encoding="utf-8")
    (env.bd / "shared/files.txt").write_text("11_gov/a.md\n11_gov/tail.md\n", encoding="utf-8")
    info = ops.plan(env.bd, env.builder, env.led)["docaudit"]
    assert [(s["rel"], s["layer"]) for s in info["skipped"]] == [("11_gov/tail.md", "L4")]
    assert info["skipped"][0]["reason"].startswith("sent-text scan: ") and info["n_files_sent"] == 1
    assert not any("ANTHROPIC_API_KEY" in b for b in _bodies(env.bd)[1])


def test_aliases_are_sent_once_and_undecodable_files_are_skipped(env):
    os.symlink(env.lab / "11_gov/a.md", env.lab / "11_gov/a_alias.md")
    (env.lab / "11_gov/bin.md").write_bytes(b"\xff\xfe\x00bad")
    (env.bd / "shared/files.txt").write_text("11_gov/a.md\n11_gov/a_alias.md\n./11_gov/a.md\n11_gov/bin.md\n", encoding="utf-8")
    info = ops.plan(env.bd, env.builder, env.led)["docaudit"]
    assert {s["rel"]: s["layer"] for s in info["skipped"]} == {"11_gov/a_alias.md": "duplicate", "./11_gov/a.md": "duplicate",
                                                             "11_gov/bin.md": "decode"}
    assert info["n_files_sent"] == 1 and info["n_pieces"] == 1


def test_symlink_to_a_file_outside_lab_is_refused_by_l1(env, tmp_path):
    out = tmp_path / "outside.md"
    out.write_text("outside the lab tree\n", encoding="utf-8")
    os.symlink(out, env.lab / "11_gov/out_alias.md")
    (env.bd / "shared/files.txt").write_text("11_gov/a.md\n11_gov/out_alias.md\n", encoding="utf-8")
    info = ops.plan(env.bd, env.builder, env.led)["docaudit"]
    assert [(s["rel"], s["layer"]) for s in info["skipped"]] == [("11_gov/out_alias.md", "L1")]
    assert not any("outside the lab tree" in b for b in _bodies(env.bd)[1])


def test_roots_mode_through_plan_and_root_outside_lab(env, tmp_path):
    spec = {**SPEC, "files": {"roots": ["11_gov", "23_rep"], "exclude": ["*/big.md"]}}
    (env.bd / "spec.yaml").write_text(json.dumps(spec), encoding="utf-8")
    info = ops.plan(env.bd, env.builder, env.led)["docaudit"]
    sent = {f for s in info["shards"] for f in s["files"]}
    assert sent == {"11_gov/a.md", "11_gov/b.md", "23_rep/z.md"} and info["roots"] == ["11_gov", "23_rep"]
    assert "11_gov/sealed/s.md" in {s["rel"] for s in info["skipped"]}
    (tmp_path / "elsewhere").mkdir()
    os.symlink(tmp_path / "elsewhere", env.lab / "ext")
    with pytest.raises(DA.DocAuditError):
        DA.walk_roots(env.lab, ["ext"], [])


def test_report_survives_out_of_contract_items_and_checks_the_counter_quote(env):
    (env.lab / "11_gov/crlf.md").write_bytes(b"first line\r\nThe count is 7.\r\n")
    (env.bd / "shared/files.txt").write_text("11_gov/a.md\n11_gov/b.md\n11_gov/crlf.md\n", encoding="utf-8")
    (env.bd / "spec.yaml").write_text(json.dumps({**SPEC, "models": ["claude-sonnet-5-5"], "shard": {"target_tokens": 4000}}), encoding="utf-8")
    ops.plan(env.bd, env.builder, env.led)
    (row,) = _bodies(env.bd)[0]
    g = {"file": "11_gov/a.md", "quote": "The count is 8.", "category": "inconsistent", "basis": "internal", "severity": "low",
         "frozen": False, "claim": "c", "counter_file": "11_gov/b.md", "counter_quote": "The count is 12.", "check": ""}
    out = {"files_read": [{"file": "11_gov/a.md", "last_line_read": 5}, {"file": "11_gov/b.md", "last_line_read": 1}, "junk"],
           "notes": "", "findings": [g, {**g, "counter_quote": "The count is 13."}, {**g, "counter_file": "11_gov/zzz.md"},
                                     "not an object", {**g, "quote": 7, "file": ["x"]},
                                     {**g, "file": "11_gov/crlf.md", "quote": "The count is 7.", "counter_quote": ""}]}
    (env.bd / "items").mkdir()
    (env.bd / "items" / f"{row['custom_id']}.md").write_text(_item(json.dumps(out)), encoding="utf-8")
    (env.root / "INBOX.jsonl").write_text(json.dumps({"custom_id": row["custom_id"], "result_status": "succeeded"}) + "\n", encoding="utf-8")
    summ = DA.report(env.bd, lab_root=env.lab)
    got = [json.loads(ln) for ln in (env.bd / "docaudit/findings.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r.get("counter_found") for r in got] == [True, False, False, None, True, None]
    assert got[2]["counter_match"] == "file-not-in-request" and got[0]["counter_line"] == 3
    assert got[3]["bad_finding"] is True and got[3]["quote_match"] == "bad-finding"
    assert got[4]["quote_match"] == "file-not-in-request"
    assert (got[5]["line"], got[5]["quote_found"], got[5]["file_changed_since_plan"]) == (2, True, False)   # CRLF: not "changed"
    m = summ["by_model"]["claude-sonnet-5-5"]
    assert (m["findings"], m["counter_checked"], m["counter_not_found"], m["requests_with_unread_parts"]) == (6, 4, 2, 1)
    req = json.loads((env.bd / "docaudit/summary.json").read_text(encoding="utf-8"))["requests"][0]
    assert req["parts_not_read_to_end"] == ["11_gov/b.md", "11_gov/crlf.md"]      # b.md has 3 lines, reader said 1


def test_main_exit_codes(env, capsys):
    ops.plan(env.bd, env.builder, env.led)
    assert DA.main(["report", str(env.bd)]) == 3 and "REFUSED" in capsys.readouterr().err


def test_report_provenance_cannot_be_overwritten_by_the_model(env):          # VERIFY-BATCH-r2 N2
    (env.bd / "shared/files.txt").write_text("11_gov/a.md\n", encoding="utf-8")
    (env.bd / "spec.yaml").write_text(json.dumps({**SPEC, "models": ["claude-sonnet-5-5"]}), encoding="utf-8")
    ops.plan(env.bd, env.builder, env.led)
    (row,) = _bodies(env.bd)[0]
    f = {"file": "11_gov/a.md", "quote": "The count is 8.", "category": "wrong", "basis": "needs-check", "severity": "low",
         "frozen": False, "claim": "c", "counter_file": "", "counter_quote": "", "check": "x",
         "model": "claude-opus-5-5", "custom_id": "forged", "shard": "sh9999", "fid": "forged-00"}
    (env.bd / "items").mkdir()
    (env.bd / "items" / f"{row['custom_id']}.md").write_text(_item(json.dumps({"files_read": [], "notes": "", "findings": [f]})), encoding="utf-8")
    (env.root / "INBOX.jsonl").write_text(json.dumps({"custom_id": row["custom_id"], "result_status": "succeeded"}) + "\n", encoding="utf-8")
    summ = DA.report(env.bd, lab_root=env.lab)
    (got,) = [json.loads(ln) for ln in (env.bd / "docaudit/findings.jsonl").read_text(encoding="utf-8").splitlines()]
    assert (got["model"], got["custom_id"], got["shard"], got["fid"]) == ("claude-sonnet-5-5", row["custom_id"], row["meta"]["shard"], f"{row['custom_id']}-00")
    assert summ["by_model"] == {"claude-sonnet-5-5": summ["by_model"]["claude-sonnet-5-5"]} and summ["by_model"]["claude-sonnet-5-5"]["findings"] == 1
