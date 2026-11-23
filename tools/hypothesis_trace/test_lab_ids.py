"""lab_ids: resolver + HTML autolinker (2026-10-23). Synthetic lab tree; no real lab file is read."""
from __future__ import annotations

import lab_ids as L
import pytest


def _touch(root, rel, text=""):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


@pytest.fixture()
def lab(tmp_path):
    _touch(tmp_path, "27_theory/11_lab-theory/H-023.md", "SEALED-BODY-MUST-NOT-BE-READ")
    _touch(tmp_path, "27_theory/11_lab-theory/H-199.md")
    _touch(tmp_path, "27_theory/11_lab-theory/H-199.status.md")
    _touch(tmp_path, "27_theory/11_lab-theory/x/12_hypotheses/H-A_scaffold.md")
    _touch(tmp_path, "13_reference/11_registries/HYPOTHESES.md", "| **H-001** | row |\n| **H-014** | row |\n")
    _touch(tmp_path, "13_reference/11_registries/EXPERIMENTS.md", "## EXP-009 — t\n")
    _touch(tmp_path, "21_experiments/11_lab-experiments/EXP-005_mlp.md")
    _touch(tmp_path, "21_experiments/11_lab-experiments/EXP-005_RESULTS.md")
    _touch(tmp_path, "21_experiments/11_lab-experiments/EXP-012_clean-r1-x.md")
    _touch(tmp_path, "13_reference/12_atoms/figures/FG-0001-curve.md")
    return tmp_path


def blocked(t):
    return "EXP-012" in t or "EXP-013" in t or "clean-r1" in t


def make(lab):
    return L.LabIds(lab, blocked, {"H-023", "H-108"})


def test_resolve_tiers(lab):
    r = make(lab)
    assert r.resolve("H-023")["doc_path"] == "27_theory/11_lab-theory/H-023.md"
    assert r.resolve("H-023")["has_evidence"]
    assert r.resolve("h-199")["status_path"].endswith("H-199.status.md")
    assert r.resolve("H-A")["doc_path"].endswith("12_hypotheses/H-A_scaffold.md")
    assert r.resolve("H-001")["doc_tier"] == "registry"
    assert r.resolve("H-108")["doc_tier"] == "evidence-only" and r.resolve("H-108")["doc_path"] is None
    assert r.resolve("EXP-005")["doc_path"].endswith("EXP-005_mlp.md")   # never RESULTS
    assert r.resolve("EXP-009")["doc_tier"] == "registry"
    assert r.resolve("FG-0001")["doc_path"].endswith("FG-0001-curve.md")
    assert r.resolve("H-777")["status"] == "unresolved"


def test_withheld_checked_before_lookup(lab):
    r = make(lab)
    for i in ("EXP-012", "EXP-013"):         # EXP-013 has no doc at all: still withheld, not unresolved
        res = r.resolve(i)
        assert res["status"] == "withheld" and res["doc_path"] is None
    inv, held = r.inventory()
    assert held == 1 and not any(x["id"] in ("EXP-012", "EXP-013") for x in inv)


def test_autolink_text_nodes_only(lab):
    r = make(lab)
    st = L.new_stats()
    doc = ('<p>see H-023 and EXP-005, EXP-012, H-777.</p>'
           '<a href="x">H-199</a><code>H-023</code><pre>EXP-005</pre>'
           '<script>var a="H-023";</script><style>/*H-023*/</style>'
           '<span title="H-023" data-x="EXP-005">t</span><svg><text>H-023</text></svg>'
           '<p>H-023.status.md EXP-005_RESULTS FG-0001</p>')
    out = L.autolink_html(doc, r, "evidence/", "index.html", st)
    assert '<p>see <a data-lab="27_theory/11_lab-theory/H-023.md" class="idlink"' in out
    assert 'href="evidence/H-023.html"' in out                       # secondary evidence link
    assert 'EXP-005</a>' in out
    assert "EXP-012" in out and 'EXP-012</a>' not in out             # withheld stays plain
    assert 'H-777</a>' not in out
    assert '<a href="x">H-199</a>' in out and "<code>H-023</code>" in out and "<pre>EXP-005</pre>" in out
    assert '<script>var a="H-023";</script>' in out and "<style>/*H-023*/</style>" in out
    assert 'title="H-023" data-x="EXP-005"' in out and "<svg><text>H-023</text></svg>" in out
    assert "<p>H-023.status.md EXP-005_RESULTS " in out               # filenames untouched
    assert st["withheld_suppressed"] == 1 and st["unresolved"] == 1 and st["unresolved_ids"] == {"H-777"}
    assert "http" not in out.replace("http-equiv", "")                # no external URL introduced


def test_no_self_evidence_link(lab):
    r = make(lab)
    out = L.autolink_html("<p>H-023</p>", r, "", "H-023.html")
    assert "H-023.html" not in out and 'data-lab="27_theory/11_lab-theory/H-023.md"' in out
    out = L.autolink_html("<p>H-108</p>", r, "", "H-108.html")           # evidence-only, on its own page
    assert "<a" not in out


def test_resolver_never_opens_docs(lab, monkeypatch):
    real = L.Path.read_text
    seen = []
    monkeypatch.setattr(L.Path, "read_text", lambda self, *a, **k: (seen.append(self.name), real(self, *a, **k))[1])
    make(lab).inventory()
    assert set(seen) <= {"HYPOTHESES.md", "EXPERIMENTS.md"}
