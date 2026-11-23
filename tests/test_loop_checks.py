"""Tests for tools/loop_checks (AD-2/5/7/11). Read-only tools; tmp_path fixtures."""
import datetime as dt
import json
import os
import subprocess
import sys

import pytest

REPO = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
TOOLS = os.path.join(REPO, "tools", "loop_checks")
sys.path.insert(0, TOOLS)
import freshness  # noqa: E402
import ledger_regen  # noqa: E402
import memory_last_cited  # noqa: E402
import queue_status  # noqa: E402

FIXTURE = os.path.join(REPO, "tests", "fixtures", "loop_checks", "synthetic_results.md")
EXP016 = os.path.join(REPO, "lab/21_experiments/11_lab-experiments/EXP-016_RESULTS.md")


def mkqueue(root, jobs=(), proc=(), done=(), failed=(), stop=False):
    for d, names in (("jobs", jobs), ("processing", proc), ("done", done), ("failed", failed)):
        os.makedirs(root / d)
        for n in names:
            (root / d / n).write_text("")
    if stop:
        (root / "stop").write_text("")
    return str(root)


# ---- tool 1 ----
def test_queue_status_snapshot(tmp_path):
    q = mkqueue(tmp_path / "q", jobs=["0002_b.job", "0001_a.job"], proc=["0000_x.job.gpu5"],
                done=["d1", "d2"], failed=["0009_f.job"], stop=True)
    s = queue_status.snapshot(q)
    assert s["jobs"] == ["0001_a.job", "0002_b.job"]
    assert s["processing"] == [{"job": "0000_x.job", "gpu": "5"}]
    assert (s["done"], s["failed"], s["failed_names"], s["stop_file"]) == (2, 1, ["0009_f.job"], True)
    assert s["snapshot_utc"].endswith("Z")
    txt = queue_status.render_text(s)
    assert "== queued (FIFO order) ==" in txt and "0000_x.job.gpu5" in txt and "done: 2  failed: 1" in txt


def test_queue_status_cli_and_missing(tmp_path, capsys):
    q = mkqueue(tmp_path / "q")
    assert queue_status.main(["--json", "--queue-dir", q]) == 0
    assert json.loads(capsys.readouterr().out)["queue_dir"] == os.path.realpath(q)
    assert queue_status.main(["--queue-dir", str(tmp_path / "nope")]) == 2


def test_queue_status_follows_symlink(tmp_path):
    q = mkqueue(tmp_path / "real", jobs=["a.job"])
    os.symlink(q, tmp_path / "link")
    assert queue_status.snapshot(queue_status.resolve_queue_dir(str(tmp_path / "link")))["jobs"] == ["a.job"]


# ---- tool 2 ----
def write_events(path, events):
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n")


def iso(h_ago, now):
    return (now - dt.timedelta(hours=h_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")


def setup_fresh(tmp_path, state_age_h, mon_age_h, running):
    now = dt.datetime.now(dt.timezone.utc)
    st = tmp_path / "STATE.md"
    st.write_text("x")
    t = now.timestamp() - state_age_h * 3600
    os.utime(st, (t, t))
    ev = tmp_path / "events.jsonl"
    write_events(ev, [{"ts_utc": iso(mon_age_h, now), "stage": "monitor"},
                      {"ts_utc": iso(100, now), "stage": "monitor"},
                      {"ts_utc": iso(0.5, now), "stage": "digest"}])
    q = mkqueue(tmp_path / "q", proc=["0001_a.job.gpu4"] if running else [])
    return ["--state", str(st), "--events", str(ev), "--queue-dir", q, "--json"]


def test_freshness_fresh(tmp_path, capsys):
    assert freshness.main(setup_fresh(tmp_path, 1, 1, True)) == 0
    r = json.loads(capsys.readouterr().out)
    assert r["monitor_status"] == "fresh" and r["newest_event_stage"] == "digest"


def test_freshness_state_stale(tmp_path):
    assert freshness.main(setup_fresh(tmp_path, 49, 1, True)) == 1


def test_freshness_monitor_stale_only_when_running(tmp_path, capsys):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    assert freshness.main(setup_fresh(tmp_path / "a", 1, 7, True)) == 1
    assert json.loads(capsys.readouterr().out)["monitor_status"] == "stale"
    assert freshness.main(setup_fresh(tmp_path / "b", 1, 7, False)) == 0
    assert json.loads(capsys.readouterr().out)["monitor_status"] == "n/a"


def test_freshness_missing(tmp_path):
    q = mkqueue(tmp_path / "q")
    assert freshness.main(["--state", str(tmp_path / "no"), "--events", str(tmp_path / "no2"),
                           "--queue-dir", q]) == 2


# ---- tool 3 ----
def test_ledger_dry_run_fixture(capsys):
    assert ledger_regen.main([FIXTURE, "--json"]) == 0
    r = json.loads(capsys.readouterr().out)
    assert r["env"]["WORK"] == "/tmp/work"
    assert [x["class"] for x in r["rows"]] == ["pure", "pure", "gpu", "writes", "pure"]
    assert r["rows"][0]["command"] == "printf 'abc' | sha256sum"
    assert r["rows"][0]["expected_hash_prefix"] == "ba7816bf"
    assert r["rows"][4]["command"] == "echo /tmp/work"
    assert all(x["result"] is None for x in r["rows"])


def test_ledger_execute_fixture(capsys):
    # row 2 has a deliberately wrong expected prefix -> mismatch -> exit 1
    assert ledger_regen.main([FIXTURE, "--execute", "--json"]) == 1
    rows = json.loads(capsys.readouterr().out)["rows"]
    assert rows[0]["result"]["status"] == "match"
    assert rows[1]["result"]["status"] == "mismatch"
    assert rows[2]["result"] is None and rows[3]["result"] is None  # gpu/writes never run
    assert rows[4]["result"]["status"] == "ran_no_hash"


def test_ledger_execute_all_match(tmp_path, capsys):
    f = tmp_path / "r.md"
    f.write_text("## Claim ledger\n\n| # | claim | regeneration command |\n|---|---|---|\n"
                 "| 1 | sha256 `ba7816bf` | `printf 'abc' \\| sha256sum` |\n")
    assert ledger_regen.main([str(f), "--execute"]) == 0
    assert "match" in capsys.readouterr().out


def test_ledger_no_table(tmp_path):
    f = tmp_path / "r.md"
    f.write_text("# nothing here\n")
    assert ledger_regen.main([str(f)]) == 2


@pytest.mark.skipif(not os.path.exists(EXP016), reason="EXP-016 RESULTS absent")
def test_ledger_dry_run_exp016(capsys):
    assert ledger_regen.main([EXP016, "--json"]) == 0  # dry-run only, never --execute
    r = json.loads(capsys.readouterr().out)
    assert r["rows"] and len(r["rows"]) >= 20
    assert any(x["class"] == "writes" for x in r["rows"])
    assert all(x["result"] is None for x in r["rows"])


# ---- tool 4 ----
def test_memory_last_cited(tmp_path, capsys):
    repo, mem, lab = tmp_path / "repo", tmp_path / "mem", tmp_path / "repo" / "lab"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "ok.sh").write_text("")
    (lab / "11_governance").mkdir(parents=True)
    (lab / "99_Archive").mkdir()
    (lab / "11_governance" / "a.md").write_text("see [[alpha]] and [[beta]]")
    (lab / "99_Archive" / "b.md").write_text("[[gamma]]")
    mem.mkdir()
    (mem / "MEMORY.md").write_text("- `lab/skipped`")
    (mem / "f1.md").write_text("---\nname: alpha\n---\n`scripts/ok.sh:12` `scripts/gone.sh.` "
                               "`lab/zzz/**` `data/x_<k>` `scripts/ok.sh --flag`\n")
    (mem / "f2.md").write_text("---\nname: gamma\n---\nplain `ls`\n")
    argv = ["--memory-dir", str(mem), "--repo", str(repo), "--json"]
    assert memory_last_cited.main(argv) == 1
    r = json.loads(capsys.readouterr().out)
    e = {x["name"]: x for x in r["entries"]}
    assert "MEMORY.md" not in [x["file"] for x in r["entries"]]
    assert e["alpha"]["dead"] == ["scripts/gone.sh", "lab/zzz"]
    assert e["alpha"]["last_cited"] is not None
    assert e["gamma"]["last_cited"] is None and e["gamma"]["dead_count"] == 0
    (mem / "f1.md").write_text("---\nname: alpha\n---\n`scripts/ok.sh`\n")
    assert memory_last_cited.main(argv) == 0


def test_cli_help():
    for t in ("queue_status", "freshness", "ledger_regen", "memory_last_cited"):
        p = subprocess.run([sys.executable, os.path.join(TOOLS, t + ".py"), "--help"],
                           capture_output=True, text=True)
        assert p.returncode == 0 and "usage" in p.stdout.lower()


# ---- tool 5 (AD-4) ----
import enqueue_guard  # noqa: E402


def mkcard(d, exp="EXP-900", fm="literature_link: lab/28_literature/x.md", freeze=True, witness=True):
    (d / f"{exp}_card.md").write_text(f"---\nexp_id: {exp}\n{fm}\n---\n# body\n")
    (d / f"{exp}_RESULTS.md").write_text("results")
    (d / "data.txt").write_text("hello")
    if freeze:
        import hashlib
        h = hashlib.sha256(b"hello").hexdigest()
        (d / f"{exp}_FREEZE.sha256").write_text(f"{h}  data.txt\n")
    if witness:
        (d / f"{exp}_FREEZE_WITNESS.md").write_text("w")


def test_guard_all_ok(tmp_path, capsys):
    mkcard(tmp_path)
    assert enqueue_guard.main(["EXP-900", "--expdir", str(tmp_path), "--strict", "--json"]) == 0
    r = json.loads(capsys.readouterr().out)
    assert r["card"].endswith("EXP-900_card.md") and r["literature_via"] == "frontmatter literature_link"
    assert r["freeze_check"] == "pass" and r["warnings"] == []


def test_guard_warn_only_and_strict(tmp_path, capsys):
    mkcard(tmp_path, fm="literature_link:", freeze=False, witness=False)
    assert enqueue_guard.main(["EXP-900", "--expdir", str(tmp_path)]) == 0
    assert capsys.readouterr().err.count("WARN:") == 3
    assert enqueue_guard.main(["EXP-900", "--expdir", str(tmp_path), "--strict"]) == 1


def test_guard_fail_and_fallback(tmp_path, capsys):
    mkcard(tmp_path, fm="x: 1")
    (tmp_path / "data.txt").write_text("tampered")
    card = tmp_path / "EXP-900_card.md"
    card.write_text(card.read_text() + "\n## 문헌\nsee lab/28_literature/a.md\n")
    assert enqueue_guard.main([str(card), "--strict", "--json"]) == 1
    r = json.loads(capsys.readouterr().out)
    assert r["freeze_check"] == "fail" and r["literature_via"] == "literature section"


def test_guard_not_found(tmp_path):
    assert enqueue_guard.main(["EXP-901", "--expdir", str(tmp_path)]) == 2
