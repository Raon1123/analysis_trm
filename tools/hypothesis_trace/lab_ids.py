"""H-NNN / EXP-NNN / FG-NNNN -> lab document resolver + HTML autolinker (2026-10-23).

Shared by the 8992 dashboard (15_research-dashboard/build_dashboard.py, loaded by file path) and
mirrored in trm-mcp (`trm_mcp/lab_ids.py`, tool `resolve_lab_id`).  Standard library only.

Rules
* Resolution is by DIRECTORY LISTING only.  No file under 21_experiments/, 27_theory/ or the figure atoms is
  opened; the only files read are the two registries (HYPOTHESES.md rows `| **H-NNN** |`, EXPERIMENTS.md
  headings `## EXP-NNN`).  Sealed/RESULTS/FREEZE/VERIFY artefacts are never a link target: an EXP doc must be
  `EXP-NNN_<lowercase slug>.md`.
* The withheld guard is injected as `blocked(text) -> bool` and is consulted BEFORE any lookup, on the id
  and on the candidate doc file name.  A blocked id has status "withheld" and no doc_path.
* Doc priority for H: card in 27_theory/11_lab-theory/ (top level, then `*/12_hypotheses/`) > registry row
  (HYPOTHESES.md) > evidence page only.  EXP: EXP doc file > registry heading.  FG: atom file name.
"""
from __future__ import annotations

import html as _html
import os
import re
from pathlib import Path

THEORY_REL = "27_theory/11_lab-theory"
EXP_REL = "21_experiments/11_lab-experiments"
FG_REL = "13_reference/12_atoms/figures"
HYP_REGISTRY_REL = "13_reference/11_registries/HYPOTHESES.md"
EXP_REGISTRY_REL = "13_reference/11_registries/EXPERIMENTS.md"

ID_PATTERN = r"(?:H-(?:\d{3}|A|RET)|EXP-\d{3}|FG-\d{4})"
# not glued to a word/path/identifier on either side; "H-023.status.md" / "EXP-015_RESULTS" stay untouched
ID_RE = re.compile(r"(?<![A-Za-z0-9_/\-.])(" + ID_PATTERN + r")(?![A-Za-z0-9_\-]|\.[A-Za-z0-9_])")

_H_CARD = re.compile(r"^(H-(?:\d{3}|[A-Z]+))(?:_[^/]*)?\.md$")
_EXP_DOC = re.compile(r"^(EXP-\d{3})_([a-z0-9][^/]*)\.md$")
_FG_DOC = re.compile(r"^(FG-\d{4})-[^/]*\.md$")
_H_ROW = re.compile(r"^\|\s*\*\*(H-[A-Z0-9]+)\*\*\s*\|", re.M)
_EXP_HEAD = re.compile(r"^##\s+(EXP-\d{3})\b", re.M)


def kind_of(i: str) -> str | None:
    return {"H": "H", "EXP": "EXP", "FG": "FG"}.get(i.split("-", 1)[0]) if ID_RE.fullmatch(i) else None


def normalize_id(raw: str) -> str:
    return re.sub(r"[\s_]+", "-", str(raw).strip()).upper()


def _listdir(p: Path) -> list[str]:
    try:
        return sorted(os.listdir(p))
    except OSError:
        return []


def _read(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


class LabIds:
    def __init__(self, lab_root, blocked, evidence_ids=()):
        self.root = Path(lab_root)
        self.blocked = blocked
        self.evidence_ids = set(evidence_ids)
        self._scan()

    # -------------------------------------------------------------- scanning (listing only)
    def _scan(self):
        theory = self.root / THEORY_REL
        dirs = [theory]
        for d in _listdir(theory):
            if d.startswith(".") or d.startswith("13_lean"):
                continue
            dirs.append(theory / d / "12_hypotheses")
        self.h_cards: dict[str, str] = {}
        self.h_status: dict[str, str] = {}
        for d in dirs:
            for n in _listdir(d):
                rel = f"{THEORY_REL}/{d.relative_to(theory)}/{n}".replace("/./", "/")
                m = _H_CARD.match(n)
                if m and not n.endswith(".status.md"):
                    self.h_cards.setdefault(m.group(1), rel)
                elif n.endswith(".status.md") and re.match(r"^H-(?:\d{3}|[A-Z]+)\.status\.md$", n):
                    self.h_status.setdefault(n[: -len(".status.md")], rel)
        self.exp_docs: dict[str, str] = {}
        for n in _listdir(self.root / EXP_REL):
            m = _EXP_DOC.match(n)
            if m and "_VERIFY" not in n:
                self.exp_docs.setdefault(m.group(1), f"{EXP_REL}/{n}")
        self.fg_docs: dict[str, str] = {}
        for n in _listdir(self.root / FG_REL):
            m = _FG_DOC.match(n)
            if m:
                self.fg_docs.setdefault(m.group(1), f"{FG_REL}/{n}")
        self.h_rows = set(_H_ROW.findall(_read(self.root / HYP_REGISTRY_REL)))
        self.exp_rows = set(_EXP_HEAD.findall(_read(self.root / EXP_REGISTRY_REL)))

    # -------------------------------------------------------------- resolution
    def resolve(self, raw_id: str) -> dict:
        i = normalize_id(raw_id)
        k = kind_of(i)
        res = {"id": i, "kind": k, "status": "unresolved", "doc_path": None, "doc_tier": None,
               "status_path": None, "has_evidence": False}
        if k is None:
            return res
        if self.blocked(i):
            res["status"] = "withheld"
            return res
        doc = tier = None
        if k == "H":
            if i in self.h_cards:
                doc, tier = self.h_cards[i], "card"
            elif i in self.h_rows:
                doc, tier = HYP_REGISTRY_REL, "registry"
            res["status_path"] = self.h_status.get(i)
            res["has_evidence"] = i in self.evidence_ids
        elif k == "EXP":
            if i in self.exp_docs:
                doc, tier = self.exp_docs[i], "doc"
            elif i in self.exp_rows:
                doc, tier = EXP_REGISTRY_REL, "registry"
        else:
            if i in self.fg_docs:
                doc, tier = self.fg_docs[i], "atom"
        if doc is not None and self.blocked(doc.rsplit("/", 1)[-1]):
            res["status"] = "withheld"
            return res
        if res["status_path"] and self.blocked(res["status_path"].rsplit("/", 1)[-1]):
            res["status_path"] = None
        if doc is None and not res["has_evidence"]:
            return res
        res.update(status="ok", doc_path=doc, doc_tier=tier or "evidence-only")
        return res

    def inventory(self):
        """(resolved ok list sorted by kind/id, number of withheld ids)."""
        ids = set(self.h_cards) | self.h_rows | self.evidence_ids | set(self.exp_docs) | self.exp_rows \
            | set(self.fg_docs)
        ids = {i for i in ids if ID_RE.fullmatch(i)}
        out, held = [], 0
        for i in sorted(ids, key=lambda s: (s.split("-")[0] != "H", s.split("-")[0] != "EXP", s)):
            r = self.resolve(i)
            if r["status"] == "ok":
                out.append(r)
            elif r["status"] == "withheld":
                held += 1
        return out, held


# ------------------------------------------------------------------ HTML autolinker
_TAG_RE = re.compile(r"<!--.*?-->|<![^>]*>|<(/?)([A-Za-z][A-Za-z0-9:-]*)(?:\"[^\"]*\"|'[^']*'|[^>\"'])*>", re.S)
SKIP_TAGS = {"a", "code", "pre", "kbd", "samp", "title", "textarea", "button", "option", "svg", "math"}
RAW_TAGS = {"script", "style"}


def new_stats():
    return {"linked": 0, "unresolved": 0, "withheld_suppressed": 0,
            "unresolved_ids": set()}


def _anchor(r, ev_prefix, here, cls="idlink"):
    parts = []
    label = _html.escape(r["id"])
    if r["doc_path"]:
        p = _html.escape(r["doc_path"], quote=True)
        parts.append('<a data-lab="%s" class="%s" title="%s">%s</a>' % (p, cls, p, label))
    if r["has_evidence"] and here != "%s.html" % r["id"]:
        ev = '<a href="%s%s.html" class="idev" title="증거 사슬">⛓</a>' % (ev_prefix, r["id"])
        parts.append(ev if r["doc_path"] else
                     '<a href="%s%s.html" class="%s" title="증거 사슬">%s</a>' % (ev_prefix, r["id"], cls, label))
    return "".join(parts) if parts else label


def autolink_html(doc: str, resolver: LabIds, ev_prefix: str = "evidence/", here: str = "",
                  stats: dict | None = None) -> str:
    """Link ids in text nodes only (not inside a/code/pre/kbd/samp/script/style/svg/title/..., not in tags)."""
    st = stats if stats is not None else new_stats()
    cache: dict[str, tuple] = {}

    def repl(m):
        i = m.group(1)
        if i not in cache:
            r = resolver.resolve(i)
            cache[i] = (r["status"], _anchor(r, ev_prefix, here) if r["status"] == "ok" else None)
        status, a = cache[i]
        if status == "ok":
            if a.startswith("<a"):
                st["linked"] += 1
            return a
        if status == "withheld":
            st["withheld_suppressed"] += 1
        else:
            st["unresolved"] += 1
            st["unresolved_ids"].add(i)
        return i

    out, pos, depth = [], 0, 0
    while True:
        m = _TAG_RE.search(doc, pos)
        text = doc[pos: m.start() if m else len(doc)]
        out.append(ID_RE.sub(repl, text) if depth == 0 else text)
        if not m:
            break
        out.append(m.group(0))
        pos = m.end()
        name = (m.group(2) or "").lower()
        if m.group(2) is None:        # comment / doctype
            continue
        if name in RAW_TAGS and not m.group(1):
            end = re.search(r"</%s\b" % name, doc[pos:], re.I)
            stop = pos + end.start() if end else len(doc)
            out.append(doc[pos:stop])
            pos = stop
        elif name in SKIP_TAGS:
            if m.group(1):
                depth = max(0, depth - 1)
            elif not m.group(0).endswith("/>"):
                depth += 1
    return "".join(out)
