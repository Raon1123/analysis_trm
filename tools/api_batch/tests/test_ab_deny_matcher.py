import re
import shutil
from pathlib import Path

import pytest

from tools.api_batch import common, deny
from tools.api_batch.common import Halt

REPO = common.REPO_ROOT


def test_deny_absent_halts(tmp_path, deny_files):
    with pytest.raises(Halt):
        deny.load_deny([deny_files[0], tmp_path / "absent.yaml"])     # second file absent
    with pytest.raises(Halt):
        deny.load_deny([])


@pytest.mark.parametrize("body", [
    "not: [yaml",                                   # unparseable
    "version: 1\n",                                 # no families
    "families: {}\n",                               # not a list
    "families:\n  - {id: a, status: weird, patterns: ['x']}\n",
    "families:\n  - {id: a, status: withheld}\n",   # nothing to match
    "families:\n  - {id: a, status: withheld, paths: ['**/x']}\n",   # paths-only is a format error (trm-mcp compat)
    "families:\n  - {id: a, status: withheld, patterns: ['(']}\n",   # regex compile failure
    "families:\n  - {id: a, status: withheld, kind: bogus, patterns: ['x']}\n",
    "families:\n  - {id: a, status: withheld, patterns: 'x'}\n",      # patterns not a list
])
def test_deny_malformed_halts(tmp_path, deny_files, body):
    bad = tmp_path / "bad.yaml"
    bad.write_text(body, encoding="utf-8")
    with pytest.raises(Halt):
        deny.load_deny([deny_files[0], bad])


def test_proposed_file_activates_cleanly(tmp_path, deny_files):
    """The user renamed the proposed file to API_BATCH_DENY.yaml (2026-10-27, before P1 prep); the shipped deny
    source must load or everything would halt. Tests the live file, or the .proposed one while it still exists.
    Its unblind_pending list is the user's to edit, so only its type is asserted, not its contents."""
    src = next(p for p in (common.GOV_DIR / "API_BATCH_DENY.proposed.yaml", common.API_BATCH_DENY_PATH) if p.exists())
    dst = tmp_path / "API_BATCH_DENY.yaml"
    shutil.copy(src, dst)
    d = deny.load_deny([deny_files[0], dst])
    # 2026-10-30 user: EXP-016/018/020, EXP-017 (RESULTS, per_call.csv), EXP-012/013 run names, EXP-023 unblocked.
    # sealed/, preds, z_snapshots stay denied (file-type bans).
    for t in ("EXP-016", "EXP 018", "EXP-020", "EXP-017_RESULTS_v2", "EXP-017_artifacts/run/per_call.csv"):
        assert not d.matcher.matches(t), t
    assert not d.path_hit("23_reports/x/EXP-017_RESULTS.md") and d.path_hit("a/sealed/b.json")
    assert not d.path_hit("EXP-017_artifacts/r/per_call.csv")
    assert isinstance(d.allow_files, list)
    assert not d.matcher.matches("the predicts of the model")           # 'preds' is bounded
    assert isinstance(d.unblind_pending, list)


def test_prose_family_excluded_and_pinned(deny_files):
    d = deny.load_deny(deny_files)
    assert not d.matcher.matches("the verdict was falsified, C-LTF refuted")   # verdict-prose off
    assert deny.PROSE_FAMILY_IDS == {"verdict-prose"}
    src = (REPO / "tools/hypothesis_trace/withheld.py").read_text(encoding="utf-8")   # text only: it needs sys.path tricks to import
    m = re.search(r"PROSE_FAMILY_IDS\s*=\s*(\{[^}]*\})", src)
    import ast
    assert ast.literal_eval(m.group(1)) == set(deny.PROSE_FAMILY_IDS)


def test_exp_id_pattern_pinned():
    w, _ = common.load_trm_mcp()
    assert w._exp_id_pattern("EXP-012") == r"(?<![A-Za-z0-9])EXP[-_ ]?0*12(?![0-9])"


SPELLINGS = ["EXP-012", "EXP 012", "EXP_012", "EXP012", "exp-12", "ＥＸＰ-012", "EXP‐012", "EXP\\_012"]


@pytest.mark.parametrize("s", SPELLINGS)
def test_spellings_caught(deny_files, s):
    assert deny.load_deny(deny_files).matcher.matches(f"see {s} for details")


@pytest.mark.parametrize("s", ["EXP-0120", "XEXP-012", "EXP-0121"])
def test_non_matches(deny_files, s):
    assert not deny.load_deny(deny_files).matcher.matches(f"see {s} for details")


def test_linebreak_split_nb_r3_1(deny_files):
    """NB-R3-1: the reused matcher does NOT catch 'EXP-' newline '012'; pack-side pre-normalization does."""
    w, _ = common.load_trm_mcp()
    d = deny.load_deny(deny_files)
    pat = re.compile(w._exp_id_pattern("EXP-012"), re.I)
    for raw in ("EXP-\n012", "EXP-\r\n  012", "EXP\n-012", "EXP-\n\n012"):
        assert not pat.search(w.normalize(raw)), raw            # documents the reused code's limit
        assert d.matcher.matches(raw), raw                      # pack-side join catches it
    assert not d.matcher.matches("EXP-\n0120")                  # still bounded
    # documented residual: a split across a paragraph boundary handled block-by-block is not joined by mask_blocks
    from tools.api_batch.pack import redact
    out, n = redact("see EXP-\n\n012 here", d.matcher)
    assert n == 0 and "012" in out                              # block-wise masking misses it; whole-text L3b stops it (test_pack)
