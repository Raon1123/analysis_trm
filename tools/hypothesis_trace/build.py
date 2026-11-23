#!/usr/bin/env python
"""Hypothesis evidence-trace builder (plan R3 + R4 + R5).

    uv run python tools/hypothesis_trace/build.py [--out DIR] [--h H-199] [--paths-file PATHS.yaml]

Reads only files of record (registry, cards, sidecars, EXP docs, ACJ verdicts, atoms, events) and writes
one `H-NNN.trace.json` per hypothesis (schema trm.hypothesis_trace/0.1, design:
lab/22_plans/11_lab-plans/20_2026-10-19_autoresearch-rebuild/10_DASHBOARD_PLAN.md section 4), plus
index.json and README.md. Exit codes: 0 ok, 2 parse failure of a file it must read (file named on stderr).

Invariants (tests enforce them):
  * every edge has source {file, line}; a referenced path that does not exist becomes a gap, never an edge
  * status/claim cells are copied verbatim (R4)
  * no metric values; withheld run ids replaced by a count (R5); EXP-016/018 list-only
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import re
import sys
import traceback
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import frontmatter as fmod  # noqa: E402
from frontmatter import ParseError, load_frontmatter, load_yaml_file, read_text  # noqa: E402
from paths import DEFAULT_PATHS_FILE, Paths  # noqa: E402
from withheld import Withheld, redact_tree  # noqa: E402

SCHEMA = "trm.hypothesis_trace/0.1"
GENERATOR_PATH = "tools/hypothesis_trace/build.py"

# ARG-track experiments whose files are list-only: existence + sha, nothing is read (task R5, AUTORESEARCH lock).
LOCKED_EXPS = {"EXP-016", "EXP-018"}

GAP_CODES = {
    "REGISTRY_ROW_MISSING": "G4: hypothesis appears in atoms/EXPERIMENTS.md/cards but HYPOTHESES.md has no row (registry_row: null; '레지스트리 행 없음')",
    "NO_CARD": "no H card file in theory.cards for this registry row",
    "DEAD_PATH": "a path declared in a file does not exist (never turned into an edge)",
    "LEGACY_PATH": "path uses a pre-taxonomy prefix listed under PATHS.yaml legacy: (recognised, never read)",
    "UNRESOLVED_ROOT": "relative path whose root is unknown (e.g. vault-relative 20_Investigations/...); not stat-ed",
    "EXTERNAL_PATH": "absolute path outside lab_root and repo_root; not read",
    "REF_UNRESOLVED": "an H-/CL- id is referenced but is neither in the hypothesis set nor an atom",
    "EXP_DOC_MISSING": "EXP id is referenced but no EXP-NNN_<slug>.md exists in experiments.dir",
    "SCOUT_MISSING": "literature_link points to a file that does not exist",
    "ACJ_DIR_MISSING": "latest_acj.session names a directory that does not exist in audits.dir",
    "CLAIM_NO_STATUS_CELL": "claim id listed in the card but no status cell found in a sidecar",
    "STATUS_WORD_UNCONTROLLED": "registry status cell does not start with a controlled-vocabulary word (cell kept verbatim, word: null)",
    "STATUS_DIVERGE": "sidecar status word differs from the registry status word (both kept verbatim)",
    "FG_NO_RENDER": "G1: FG atom has no structured `render` field",
    "FG_RENDER_PENDING": "FG atom declares render: null with render_pending_reason",
    "FG_RENDER_MISSING_FILE": "FG render[].path does not exist",
    "FG_RENDER_SHA_MISMATCH": "FG render[].sha256 differs from the file on disk",
    "NO_FIGURE_ATOM": "G2: hypothesis has experiments but no FG atom",
    "NO_S7_EVENT": "G5: hypothesis has a status sidecar but events.jsonl has no S7 event for it",
    "NO_CLOSEOUT_MANIFEST": "G7: experiment has RESULTS but no atoms.builds/<EXP>-closeout.manifest.yaml",
    "LOCKED_LIST_ONLY": "EXP is ARG-locked/withheld: files listed by name+sha, contents deliberately not read",
    "VERIFY_NO_TOKEN": "VERIFY return file has no top-level `verdict`",
}

KIND_STAGE = {"H": "S1", "SCOUT": "S2", "ACJ": "S3", "EXP": "S4", "FREEZE": "S4", "VERIFY": "S4/S6",
              "RESULTS": "S6", "SIDE": "S7", "RETRO": "S8", "CL": None, "FG": None}
TIMELINE_KINDS = set(KIND_STAGE)
VERDICT_KINDS = {"verdict", "results-verified", "paper-claim-acj"}  # index last_ts uses these only; other events -> last_status_change_ts
NOHASH_SUFFIX = {".jsonl", ".json", ".parquet", ".npy", ".npz", ".pt", ".png", ".pdf"}
EXP_RE = re.compile(r"\bEXP-\d{3}\b")
H_RE = re.compile(r"\bH-[A-Za-z0-9]+\b")
H_NUM_RE = re.compile(r"\bH-\d{3}\b")
PLACEHOLDER_H = {"H-NNN"}


def utcnow():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def mtime_iso(p):
    return dt.datetime.fromtimestamp(os.stat(p).st_mtime, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


_SHA = {}


def sha256_file(p):
    if p not in _SHA:
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        _SHA[p] = h.hexdigest()
    return _SHA[p]


def _json_default(o):
    return o.isoformat() if hasattr(o, "isoformat") else str(o)


def safe_name(h):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", h)


def as_list(v):
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


# --------------------------------------------------------------------------------------------- world
class World:
    """Everything read once from files of record."""

    def __init__(self, paths_file=DEFAULT_PATHS_FILE):
        P = self.P = Paths(paths_file)
        self.W = Withheld(P.abs("governance.withheld_runs"))
        self.load_registry()
        self.load_experiments_md()
        self.load_cards()
        self.load_atoms()
        self.load_expfiles()
        self.load_acj_dirs()
        self.load_events()
        wdir = os.path.join(P.repo_root, P.repo_map.get("wandb.dir", "wandb"))
        self.wandb_names = set(os.listdir(wdir)) if os.path.isdir(wdir) else None
        self.exp_results_fm = {}
        self.exp_doc_fm = {}
        self.load_exp_frontmatters()
        self.set_withheld_doc_names()
        self.collect_h_set()

    def set_withheld_doc_names(self):
        """Exempt from the path guard only basenames of real documents enumerated in experiments.dir / theory.cards."""
        names = set()
        for d in (self.expdir, self.P.abs("theory.cards")):
            if os.path.isdir(d):
                for n in os.listdir(d):
                    if os.path.isfile(os.path.join(d, n)) and n.endswith((".md", ".yaml", ".sha256")):
                        names.add(n)
        self.W.set_doc_names(names)

    # -- registry
    def load_registry(self):
        P = self.P
        self.reg_path = P.abs("registry.hypotheses")
        text = read_text(self.reg_path)
        lines = text.split("\n")
        self.vocab = []
        for ln in lines:
            m = re.search(r"Status vocabulary \(controlled\)\*\*:\s*(.+)$", ln)
            if m:
                self.vocab = [t.strip() for t in m.group(1).split("/") if t.strip()]
                break
        if not self.vocab:
            raise ParseError(self.reg_path, "controlled status vocabulary line not found")
        hdr_i = next((i for i, ln in enumerate(lines) if ln.startswith("| H-id")), None)
        if hdr_i is None:
            raise ParseError(self.reg_path, "hypothesis table header '| H-id' not found")
        hdr = self.split_row(lines[hdr_i])
        want = {"id": "h-id", "statement": "statement", "status": "status", "canon": "canon-doc"}
        col = {}
        for k, pref in want.items():
            idx = [i for i, c in enumerate(hdr) if c.lower().startswith(pref)]
            if not idx:
                raise ParseError(self.reg_path, f"column {pref} missing in header")
            col[k] = idx[0]
        self.reg = {}
        self.reg_order = []
        i = hdr_i + 2
        while i < len(lines) and lines[i].startswith("|"):
            cells = self.split_row(lines[i])
            if len(cells) != len(hdr):
                raise ParseError(self.reg_path, f"line {i + 1}: {len(cells)} cells, header has {len(hdr)}")
            hid = cells[col["id"]].strip("*").strip()
            row = {"id": hid, "line": i + 1, "raw": lines[i]}
            for k in want:
                row[k] = cells[col[k]]
            self.reg[hid] = row
            self.reg_order.append(hid)
            i += 1
        if not self.reg:
            raise ParseError(self.reg_path, "no hypothesis rows parsed")

    @staticmethod
    def split_row(line):
        parts = re.split(r"(?<!\\)\|", line.strip())
        return [c.strip() for c in parts[1:-1]]

    # -- EXPERIMENTS.md
    def load_experiments_md(self):
        self.expmd_path = self.P.abs("registry.experiments")
        lines = read_text(self.expmd_path).split("\n")
        self.expmd = {}
        cur = None
        self.expmd_h_mentions = Counter()
        self.expmd_other_h = Counter()
        for i, ln in enumerate(lines, 1):
            m = re.match(r"## (EXP-\d{3})\b", ln)
            if m:
                cur = {"id": m.group(1), "line": i, "status": None, "h_line": None, "h_ids": [], "h_rel": {}}
                self.expmd[cur["id"]] = cur
            if cur and ln.startswith("|"):
                cells = self.split_row(ln)
                if len(cells) >= 2:
                    key = cells[0].strip("*").strip()
                    if key == "Status":
                        cur["status"] = {"cell": cells[1], "line": i}
                    elif key == "H-NNN tested":
                        cur["h_line"] = i
                        cur["h_ids"] = sorted(set(H_NUM_RE.findall(cells[1])))
                        cur["h_rel"] = {hid: self.tested_rel(cells[1], hid) for hid in cur["h_ids"]}
            for tok in H_RE.findall(ln):
                if tok in PLACEHOLDER_H:
                    continue
                (self.expmd_h_mentions if H_NUM_RE.fullmatch(tok) else self.expmd_other_h)[tok] += 1

    @staticmethod
    def tested_rel(cell, hid):
        """`tested-in` unless the cell itself weakens it: `feeds H-x` (same ;-segment) -> feeds;
        `context only` / `not a test` in the text following H-x -> context."""
        pos = [m.start() for m in re.finditer(re.escape(hid) + r"(?![A-Za-z0-9])", cell)]
        if not pos:
            return "tested-in"
        i = pos[0]
        seg_start = cell.rfind(";", 0, i) + 1
        if re.search(r"\bfeeds\b", cell[seg_start:i]):
            return "feeds"
        nxt = H_NUM_RE.search(cell, i + len(hid))
        tail = cell[i + len(hid): nxt.start() if nxt else len(cell)]
        if re.search(r"context only|not a test", tail):
            return "context"
        return "tested-in"

    # -- cards and sidecars
    def load_cards(self):
        d = self.P.abs("theory.cards")
        self.cards, self.sidecars = {}, {}
        self.card_path, self.side_path = {}, {}
        if not os.path.isdir(d):
            return
        for name in sorted(os.listdir(d)):
            m = re.fullmatch(r"(H-[A-Za-z0-9]+)\.status\.md", name)
            m2 = re.fullmatch(r"(H-[A-Za-z0-9]+)\.md", name)
            p = os.path.join(d, name)
            if m:
                self.sidecars[m.group(1)] = load_frontmatter(p, required=True)
                self.side_path[m.group(1)] = p
            elif m2:
                self.cards[m2.group(1)] = load_frontmatter(p, required=True)
                self.card_path[m2.group(1)] = p

    # -- atoms
    def load_atoms(self):
        self.atoms = {}
        for logical in ("atoms.claims", "atoms.figures"):
            d = self.P.abs(logical)
            if not os.path.isdir(d):
                continue
            for name in sorted(os.listdir(d)):
                if not name.endswith(".md"):
                    continue
                p = os.path.join(d, name)
                fm = load_frontmatter(p, required=True)
                aid = fm.data.get("atom_id")
                if not aid:
                    raise ParseError(p, "atom_id missing")
                self.atoms[aid] = {"id": aid, "fm": fm, "path": p, "kind": "FG" if aid.startswith("FG") else "CL"}
        self.atoms_by_h = {}
        for a in self.atoms.values():
            for hid in as_list(a["fm"].data.get("hypothesis_ids")):
                self.atoms_by_h.setdefault(str(hid), []).append(a)

    # -- experiment files by filename convention
    def load_expfiles(self):
        d = self.P.abs("experiments.dir")
        self.expdir = d
        self.expfiles = {}
        if not os.path.isdir(d):
            return
        for name in sorted(os.listdir(d)):
            m = re.match(r"(EXP-\d{3})_(.+)$", name)
            if not m or not os.path.isfile(os.path.join(d, name)):
                continue
            e, rest = m.group(1), m.group(2)
            f = self.expfiles.setdefault(e, {"verify": []})
            p = os.path.join(d, name)
            if rest == "FREEZE.sha256":
                f["freeze"] = p
            elif rest == "FREEZE_WITNESS.md":
                f["witness"] = p
            elif rest == "RESULTS.md":
                f["results"] = p
            elif rest.startswith("RETROFIT") and rest.endswith(".md"):
                f.setdefault("retro", []).append(p)
            else:
                mv = re.search(r"_VERIFY-r(\d+)\.return\.yaml$", name)
                if mv:
                    target = "results" if rest.startswith("RESULTS_VERIFY") else "prereg"
                    f["verify"].append((target, int(mv.group(1)), p))
                elif rest.endswith(".md") and "doc" not in f:
                    f["doc"] = p

    # -- ACJ dirs (verdict.md only; do not descend into staged/)
    def load_acj_dirs(self):
        d = self.P.abs("audits.dir")
        self.audits_dir = d
        self.acj_by_h = {}
        if not os.path.isdir(d):
            return
        for name in sorted(os.listdir(d)):
            m = re.fullmatch(r"acj_(H-[A-Za-z0-9]+)_(\d{8})", name)
            if m and os.path.isdir(os.path.join(d, name)):
                self.acj_by_h.setdefault(m.group(1), []).append(name)

    # -- events
    def load_events(self):
        self.events = []  # (kind, file, line, dict)
        for logical, basis in (("monitoring.events", "events"), ("monitoring.backfill_events", "backfill")):
            p = self.P.abs(logical)
            if not os.path.exists(p):
                continue
            for i, ln in enumerate(read_text(p).split("\n"), 1):
                if not ln.strip():
                    continue
                try:
                    ev = json.loads(ln)
                except json.JSONDecodeError as e:
                    raise ParseError(p, f"line {i}: {e}")
                self.events.append((basis, p, i, ev))

    # -- EXP frontmatters (non-locked, non-withheld only)
    def list_only(self, eid):
        return eid in LOCKED_EXPS or self.W.is_blind_exp(eid)

    def load_exp_frontmatters(self):
        self.exp_h_links = {}  # h -> [(eid, file, line, rel_basis)]
        for eid, f in self.expfiles.items():
            if self.list_only(eid):
                continue
            if "doc" in f:
                fm = load_frontmatter(f["doc"], required=True)
                self.exp_doc_fm[eid] = fm
                for j, item in enumerate(as_list(fm.data.get("hypotheses"))):
                    for h in H_RE.findall(str(item)):
                        self.exp_h_links.setdefault(h, []).append((eid, f["doc"], fm.line("hypotheses", j) if isinstance(fm.data.get("hypotheses"), list) else fm.line("hypotheses")))
            if "results" in f:
                fm = load_frontmatter(f["results"], required=True)
                self.exp_results_fm[eid] = fm
                for j, h in enumerate(as_list(fm.data.get("hypothesis_ids"))):
                    ln = fm.line("hypothesis_ids", j) if isinstance(fm.data.get("hypothesis_ids"), list) else fm.line("hypothesis_ids")
                    self.exp_h_links.setdefault(str(h), []).append((eid, f["results"], ln))

    # -- hypothesis set
    def collect_h_set(self):
        self.h_origin = {}
        for h in self.reg_order:
            self.h_origin[h] = "registry"
        for h in sorted(set(self.cards) | set(self.sidecars)):
            self.h_origin.setdefault(h, "card")
        for h in sorted(self.atoms_by_h):
            self.h_origin.setdefault(h, "atoms")
        for h in sorted(self.expmd_h_mentions):
            self.h_origin.setdefault(h, "EXPERIMENTS.md")
        self.h_ids = list(self.h_origin)
        self.h_like_unmapped = dict(self.expmd_other_h)


# --------------------------------------------------------------------------------------------- trace
class Trace:
    def __init__(self, w: World, h: str):
        self.w, self.P, self.h = w, w.P, h
        self.nodes, self.edges, self.gaps = {}, [], []
        self._ek = set()
        self.claims, self.figures = [], []
        self.pathnodes = {}
        self.exps = []
        self.exp_expanded = set()
        self.acjs = set()
        self.withheld_count = 0
        self.runs_seen = set()
        self.status = None
        self.registry_row = None
        self.pending_basis = []

    # primitives
    def src(self, abspath, line, basis):
        return {"file": self.P.display(abspath), "line": int(line), "basis": basis}

    def node(self, nid, kind, abspath=None, **extra):
        if nid in self.nodes:
            self.nodes[nid].update({k: v for k, v in extra.items() if v is not None and k not in self.nodes[nid]})
            return nid
        n = {"id": nid, "kind": kind, "path": None, "sha256": None, "mtime": None, "blind": False}
        if abspath:
            n["path"] = self.P.display(abspath)
            if os.path.isfile(abspath):
                n["mtime"] = mtime_iso(abspath)
                if os.path.splitext(abspath)[1] not in NOHASH_SUFFIX:
                    n["sha256"] = sha256_file(abspath)
            elif os.path.isdir(abspath):
                n["mtime"] = mtime_iso(abspath)
            self.pathnodes.setdefault(abspath, nid)
        n.update(extra)
        self.nodes[nid] = n
        return nid

    def edge(self, frm, to, rel, source):
        if not source or not source.get("file") or not isinstance(source.get("line"), int) or source["line"] < 1:
            raise AssertionError(f"edge {frm}->{to} ({rel}) without source")
        k = (frm, to, rel, source["file"], source["line"])
        if k in self._ek:
            return
        self._ek.add(k)
        self.edges.append({"from": frm, "to": to, "rel": rel, "source": source})

    def gap(self, code, what, where):
        assert code in GAP_CODES, code
        self.gaps.append({"code": code, "what": what, "where": where})

    def ref_gap(self, kind, ref, where):
        code = {"dead": "DEAD_PATH", "legacy": "LEGACY_PATH", "unresolved-root": "UNRESOLVED_ROOT",
                "external": "EXTERNAL_PATH"}[kind]
        self.gap(code, f"referenced path not an edge: {ref}", where)

    def guard(self, ref):
        """R5 on path-form references: a path with a withheld run-id component is counted, never emitted."""
        if self.w.W.match_path(str(ref)):
            self.withheld_count += 1
            return True
        return False

    def path_ref(self, frm, ref, rel, file_abs, line, basis, kind="DOC", nid=None):
        """Resolve a path declared at file:line; edge if it exists, gap otherwise."""
        where = f"{self.P.display(file_abs)}:{line}"
        if self.guard(ref):
            return None
        k, ab, disp = self.P.resolve(ref)
        if k != "ok":
            self.ref_gap(k, ref.strip(), where)
            return None
        n = self.pathnodes.get(ab) or self.node(nid or f"DOC:{disp}", kind, ab)
        if n != frm:  # a path that resolves to the node itself is not an edge
            src = self.src(file_abs, line, basis)
            if ref.strip().strip("`") != disp:  # keep the declared string (annotations included) for provenance
                src["declared"] = ref.strip()
            self.edge(frm, n, rel, src)
        return n

    # ---- hypothesis references
    def h_ref(self, frm, target, rel, file_abs, line, basis):
        where = f"{self.P.display(file_abs)}:{line}"
        if target in self.w.atoms:
            self.node(target, self.w.atoms[target]["kind"], self.w.atoms[target]["path"])
            self.edge(frm, target, rel, self.src(file_abs, line, basis))
        elif target in self.w.h_origin:
            self.node(target, "H", self.w.card_path.get(target))
            self.edge(frm, target, rel, self.src(file_abs, line, basis))
        else:
            self.gap("REF_UNRESOLVED", f"{target} is neither a hypothesis in the set nor an atom", where)

    # ---- experiments
    def ensure_exp(self, eid, expand=True):
        w = self.w
        f = w.expfiles.get(eid, {})
        if eid not in self.nodes:
            extra = {}
            if eid in w.expmd:
                st = w.expmd[eid]["status"]
                if st:
                    extra["registry_status_cell"] = {"cell": st["cell"], "source": self.src(w.expmd_path, st["line"], "table-cell")}
            lo = w.list_only(eid)
            if lo:
                extra["list_only"] = True
            self.node(eid, "EXP", f.get("doc"), blind=bool(w.W.is_blind_exp(eid)), **extra)
            if "doc" not in f:
                self.gap("EXP_DOC_MISSING", f"no EXP-NNN_<slug>.md for {eid}" + (" (registry section exists)" if eid in w.expmd else ""), eid)
            if lo:
                self.gap("LOCKED_LIST_ONLY", f"{eid}: files listed by name + sha only; contents not read", eid)
        if expand and eid not in self.exp_expanded:
            self.exp_expanded.add(eid)
            if f:
                self.expand_exp(eid, f)
        if expand and eid not in self.exps:
            self.exps.append(eid)
        return eid

    def expand_exp(self, eid, f):
        w = self.w
        lo = w.list_only(eid)
        blind = w.W.is_blind_exp(eid)
        doc_abs = f.get("doc")

        def fsrc(path, nread_pat=None):
            """source for a filename-convention edge: first line naming the target when readable."""
            if lo:
                return self.src(path, 1, "filename-convention")
            for i, ln in enumerate(read_text(path).split("\n"), 1):
                if nread_pat and re.search(nread_pat, ln):
                    return self.src(path, i, "file-line")
            return self.src(path, 1, "filename-convention")

        docname = os.path.basename(doc_abs) if doc_abs else eid
        if "freeze" in f:
            n = self.node(f"{eid}:FREEZE", "FREEZE", f["freeze"], blind=blind)
            self.edge(eid, n, "frozen-by", fsrc(f["freeze"], re.escape(docname)))
        if "witness" in f:
            n = self.node(f"{eid}:FREEZE_WITNESS", "FREEZE", f["witness"], blind=blind, role="witness")
            self.edge(eid, n, "frozen-by", fsrc(f["witness"], re.escape(eid)))
        for rp in f.get("retro", []):
            tag = re.sub(r"^EXP-\d{3}_", "", os.path.basename(rp)[:-3])
            n = self.node(f"{eid}:{tag}", "RETRO", rp, blind=blind)
            self.edge(eid, n, "retrofitted-by", fsrc(rp, re.escape(docname) if doc_abs else None))
        res_node = None
        if "results" in f:
            res_node = self.node(f"{eid}:RESULTS", "RESULTS", f["results"], blind=blind)
            if not lo and eid in w.exp_results_fm:
                self.expand_results(eid, res_node, f["results"], w.exp_results_fm[eid], doc_abs)
            else:
                self.edge(eid, res_node, "results-in", self.src(f["results"], 1, "filename-convention"))
            if not os.path.exists(os.path.join(w.P.abs("atoms.builds"), f"{eid}-closeout.manifest.yaml")):
                self.gap("NO_CLOSEOUT_MANIFEST", f"{eid} has RESULTS but no atoms.builds/{eid}-closeout.manifest.yaml", eid)
        for target, rnd, vp in sorted(f.get("verify", []), key=lambda t: (t[0], t[1])):
            nid = f"{eid}:VERIFY-{'results' if target == 'results' else 'prereg'}-r{rnd}"
            token, line, how = None, 1, "filename-convention"
            if not lo:
                vfm = load_yaml_file(vp)
                tv = vfm.data.get("verdict")
                token = None if tv is None else str(tv)
                if token is None:
                    self.gap("VERIFY_NO_TOKEN", f"{os.path.basename(vp)} has no top-level verdict", self.P.display(vp))
                for key in ("exp_id", "target", "audit_target"):
                    if key in vfm.data:
                        line, how = vfm.line(key), "frontmatter-field"
                        break
            self.node(nid, "VERIFY", vp, blind=blind, round=rnd, target=target, verifier_token=token)
            frm = res_node if (target == "results" and res_node) else eid
            self.edge(frm, nid, "verified-by", self.src(vp, line, how))

    def expand_results(self, eid, res_node, rp, fm, doc_abs):
        w = self.w
        d = fm.data
        node = self.nodes[res_node]
        if "status" in d:
            node["status_cell"] = {"cell": str(d["status"]), "source": self.src(rp, fm.line("status"), "frontmatter-field")}
        for key in ("prereg_sha256", "judge_sha256", "freeze_manifest_sha256", "per_run_csv_sha256"):
            if key in d:
                node[key] = str(d[key])
        pre = d.get("prereg")
        if pre and self.guard(pre):
            pre = None
            self.edge(eid, res_node, "results-in", self.src(rp, 1, "filename-convention"))
        elif pre:
            pre_abs = os.path.join(w.expdir, str(pre))
            self.edge(eid, res_node, "results-in", self.src(rp, fm.line("prereg"), "frontmatter-field"))
            if os.path.isfile(pre_abs) and d.get("prereg_sha256"):
                node["prereg_sha_matches"] = sha256_file(pre_abs) == str(d["prereg_sha256"])
        else:
            self.edge(eid, res_node, "results-in", self.src(rp, 1, "filename-convention"))
        for key, rel in (("judge", "judged-by"),):
            if d.get(key) and not self.guard(d[key]):
                ab = os.path.join(w.expdir, str(d[key]))
                if os.path.isfile(ab):
                    n = self.pathnodes.get(ab) or self.node(f"DOC:{self.P.display(ab)}", "DOC", ab)
                    self.edge(res_node, n, rel, self.src(rp, fm.line(key), "frontmatter-field"))
                else:
                    self.gap("DEAD_PATH", f"referenced path not an edge: {d[key]}", f"{self.P.display(rp)}:{fm.line(key)}")
        csvp = d.get("per_run_csv")
        if csvp and not self.guard(csvp):
            ab = os.path.join(w.expdir, str(csvp))
            if not os.path.isfile(ab):
                self.gap("DEAD_PATH", f"referenced path not an edge: {csvp}", f"{self.P.display(rp)}:{fm.line('per_run_csv')}")
                return
            n = self.node(f"DOC:{self.P.display(ab)}", "DOC", ab)
            self.edge(res_node, n, "per-run-data", self.src(rp, fm.line("per_run_csv"), "frontmatter-field"))
            self.read_run_ids(res_node, ab)

    def read_run_ids(self, res_node, csv_abs):
        """Only the project/run_id/run_name columns are read; no other column is touched."""
        w = self.w
        fmod.OPENED.append(csv_abs)
        with open(csv_abs, encoding="utf-8", newline="") as fh:
            rd = csv.DictReader(fh)
            cols = rd.fieldnames or []
            if "run_id" not in cols:
                self.gap("DEAD_PATH", "per-run CSV has no run_id column; RUN nodes not built", self.P.display(csv_abs))
                return
            for rowno, row in enumerate(rd, 2):
                rid, name, proj = row.get("run_id", ""), row.get("run_name", "") if "run_name" in cols else "", row.get("project", "") if "project" in cols else ""
                if w.W.match(rid) or (name and w.W.match(name)):
                    self.withheld_count += 1
                    continue
                self.add_run(res_node, proj, rid, name, "ran", self.src(csv_abs, rowno, "table-row"))

    def add_run(self, frm, project, rid, name, rel, source):
        w = self.w
        if w.W.match(rid) or (name and w.W.match(name)):
            self.withheld_count += 1
            return None
        nid = f"RUN:{project}/{rid}"
        if (project, rid) not in self.runs_seen:
            self.runs_seen.add((project, rid))
            verified = None
            if w.wandb_names is not None:
                verified = any(n.endswith("-" + rid) for n in w.wandb_names)
            self.node(nid, "RUN", None, project=project, run_id=rid, run_name=name or None, run_verified=verified)
        self.edge(frm, nid, rel, source)
        return nid

    # ---- ACJ
    def expand_acj(self, name):
        w = self.w
        if name in self.acjs:
            return f"ACJ:{name}"
        d = os.path.join(w.audits_dir, name)
        self.acjs.add(name)
        verdict = os.path.join(d, "verdict.md")
        nid = f"ACJ:{name}"
        extra = {"dir": self.P.display(d)}
        if os.path.isfile(verdict):
            fm = load_frontmatter(verdict, required=True)
            if "verdict" in fm.data:
                extra["verdict"] = {"cell": str(fm.data["verdict"]), "source": self.src(verdict, fm.line("verdict"), "frontmatter-field")}
            if "date" in fm.data:
                extra["date"] = str(fm.data["date"])
            self.node(nid, "ACJ", verdict, **extra)
            hline = fm.line("H_id") if "H_id" in fm.data else 1
            m = re.fullmatch(r"acj_(H-[A-Za-z0-9]+)_\d{8}", name)
            if m and str(fm.data.get("H_id")) == m.group(1):
                self.edge(m.group(1), nid, "adjudicated-by", self.src(verdict, hline, "frontmatter-field"))
        else:
            self.node(nid, "ACJ", d, **extra)
        for fn in sorted(os.listdir(d)):
            if re.fullmatch(r"VERIFY-.+\.return\.yaml", fn):
                vp = os.path.join(d, fn)
                vfm = load_yaml_file(vp)
                tv = vfm.data.get("verdict")
                line, how = 1, "filename-convention"
                for key in ("h_id", "H_id", "exp_id", "target", "audit_target"):
                    if key in vfm.data:
                        line, how = vfm.line(key), "frontmatter-field"
                        break
                vid = f"{nid}:{fn[:-len('.return.yaml')]}"
                self.node(vid, "VERIFY", vp, verifier_token=None if tv is None else str(tv), target="acj")
                self.edge(nid, vid, "verified-by", self.src(vp, line, how))
        return nid

    def acj_ref(self, frm, session, rel, file_abs, line):
        name = str(session)
        d = os.path.join(self.w.audits_dir, name)
        if os.path.isdir(d):
            nid = self.expand_acj(name)
            self.edge(frm, nid, rel, self.src(file_abs, line, "frontmatter-field"))
        else:
            self.gap("ACJ_DIR_MISSING", f"latest_acj.session '{name}' has no directory in audits.dir", f"{self.P.display(file_abs)}:{line}")

    # ---- card
    def do_card(self):
        w, h = self.w, self.h
        fm = w.cards.get(h)
        cp = w.card_path.get(h)
        self.node(h, "H", cp)
        if not fm:
            self.gap("NO_CARD", f"no {h}.md in theory.cards", h)
            return
        d = fm.data
        S = lambda *k: (cp, fm.line(*k))  # noqa: E731
        self.card_status = None
        if "status" in d:
            self.card_status = {"cell": str(d["status"]), "source": self.src(cp, fm.line("status"), "frontmatter-field")}

        def exp_ids_of(key, rel):
            v = d.get(key)
            if v is None:
                return
            items = v if isinstance(v, list) else [v]
            for j, it in enumerate(items):
                ln = fm.line(key, j) if isinstance(v, list) else fm.line(key)
                for e in EXP_RE.findall(str(it)):
                    # only the card's own experiment is expanded into FREEZE/VERIFY/RESULTS nodes;
                    # related experiments stay single nodes so the page does not show another H's evidence as its own
                    self.ensure_exp(e, expand=(rel == "pre-registered-in"))
                    self.edge(h, e, rel, self.src(cp, ln, "frontmatter-field"))

        exp_ids_of("exp_id", "pre-registered-in")
        exp_ids_of("exp", "pre-registered-in")
        exp_ids_of("related_experiments", "related-experiment")
        for j, it in enumerate(as_list(d.get("related_hypotheses"))):
            ln = fm.line("related_hypotheses", j) if isinstance(d.get("related_hypotheses"), list) else fm.line("related_hypotheses")
            for tok in re.findall(r"\b(?:H-[A-Za-z0-9]+|CL-\d{4})\b", str(it)):
                self.h_ref(h, tok, "related-to", cp, ln, "frontmatter-field")
        ll = d.get("literature_link")
        nv_cell, nv_line = d.get("novelty_verdict"), fm.line("novelty_verdict") if "novelty_verdict" in d else None
        if isinstance(ll, dict):  # mapping form: {scout_note, novelty_verdict, ...}; never stringified
            ll_line = fm.line("literature_link")
            if ll.get("novelty_verdict") is not None:
                nv_cell, nv_line = ll["novelty_verdict"], fm.line("literature_link", "novelty_verdict")
            if ll.get("scout_note"):
                ll = [ll["scout_note"]]
                lln = [fm.line("literature_link", "scout_note")]
            else:
                self.gap("SCOUT_MISSING", "literature_link mapping lacks key scout_note", f"{self.P.display(cp)}:{ll_line}")
                ll, lln = [], []
        else:
            lln = None
        for j, it in enumerate(as_list(ll)):
            ln = lln[j] if lln is not None else (fm.line("literature_link", j) if isinstance(ll, list) else fm.line("literature_link"))
            if self.guard(it):
                continue
            k, ab, disp = self.P.resolve(str(it))
            if k == "ok":
                n = self.pathnodes.get(ab) or self.node(f"SCOUT:{os.path.basename(ab)}", "SCOUT", ab)
                if nv_cell is not None:
                    self.nodes[n]["novelty_verdict"] = {"cell": str(nv_cell), "source": self.src(cp, nv_line, "frontmatter-field")}
                self.edge(h, n, "scouted-in", self.src(cp, ln, "frontmatter-field"))
            else:
                self.gap("SCOUT_MISSING" if k == "dead" else {"legacy": "LEGACY_PATH", "unresolved-root": "UNRESOLVED_ROOT", "external": "EXTERNAL_PATH"}[k],
                         f"literature_link not an edge: {str(it).strip()}", f"{self.P.display(cp)}:{ln}")
        for key, rel in (("north_star", "north-star"),):
            if isinstance(d.get(key), str):
                self.path_ref(h, d[key], rel, cp, fm.line(key), "frontmatter-field")
        for j, it in enumerate(as_list(d.get("related_investigations"))):
            self.path_ref(h, str(it), "related-investigation", cp, fm.line("related_investigations", j), "frontmatter-field")
        pe = d.get("pending_evidence")
        for j, it in enumerate(as_list(pe)):
            if isinstance(it, dict) and it.get("artifact"):
                self.path_ref(h, str(it["artifact"]), "pending-evidence", cp, fm.line("pending_evidence", j, "artifact"), "frontmatter-field")
        la = d.get("latest_acj")
        if isinstance(la, dict) and la.get("session"):
            self.acj_ref(h, la["session"], "adjudicated-by", cp, fm.line("latest_acj", "session"))
        # claims named in the card
        self.card_claim_ids = []
        cl = d.get("claims")
        for j, cid in enumerate(as_list(cl)):
            cid = str(cid)
            nid = f"{h}:{cid}"
            self.node(nid, "CLAIM", None, claim_id=cid)
            self.edge(h, nid, "claims", self.src(cp, fm.line("claims", j) if isinstance(cl, list) else fm.line("claims"), "frontmatter-field"))
            self.card_claim_ids.append(cid)

    # ---- registry row
    def do_registry(self):
        w, h = self.w, self.h
        row = w.reg.get(h)
        if not row:
            self.gap("REGISTRY_ROW_MISSING", "레지스트리 행 없음 — appears in " + w.h_origin[h], h)
            return
        reg = w.reg_path
        self.registry_row = {"file": self.P.display(reg), "line": row["line"], "status_cell": row["status"],
                             "statement_cell": row["statement"], "canon_doc_cell": row["canon"]}
        for ref in re.findall(r"`([^`]+)`", row["canon"]):
            if "/" not in ref and not re.search(r"\.(md|py|yaml|jsonl|csv)$", ref):
                continue
            self.path_ref(h, ref, "canon-doc", reg, row["line"], "registry-cell")

    # ---- sidecar
    def do_sidecar(self):
        w, h = self.w, self.h
        fm = w.sidecars.get(h)
        if not fm:
            return
        sp = w.side_path[h]
        d = fm.data
        sid = f"{h}.status"
        self.node(sid, "SIDE", sp)
        self.edge(h, sid, "status-sidecar", self.src(sp, fm.line("id"), "frontmatter-field"))
        for e in EXP_RE.findall(str(d.get("exp_id", ""))):
            self.ensure_exp(e)
        if d.get("supersedes_fields_of"):
            self.path_ref(sid, str(d["supersedes_fields_of"]), "supersedes-fields-of", sp, fm.line("supersedes_fields_of"), "frontmatter-field")
        la = d.get("latest_acj")
        if isinstance(la, dict) and la.get("session"):
            self.acj_ref(sid, la["session"], "adjudicated-by", sp, fm.line("latest_acj"))
        for j, it in enumerate(as_list(d.get("basis"))):
            self.pending_basis.append((sid, str(it), sp, fm.line("basis", j)))
        self.sidecar_status = None
        if "status" in d:
            self.sidecar_status = {"cell": str(d["status"]), "source": self.src(sp, fm.line("status"), "frontmatter-field")}
        # per-claim status cells: table rows "| C-XXX-n ... | <cell> |" (verbatim)
        for i, ln in enumerate(read_text(sp).split("\n"), 1):
            m = re.match(r"\|\s*(C-[A-Z0-9]+-\d+)\b", ln)
            if m and ln.count("|") >= 3:
                cells = World.split_row(ln)
                cid = m.group(1)
                nid = f"{h}:{cid}"
                self.node(nid, "CLAIM", None, claim_id=cid)
                self.edge(sid, nid, "states", self.src(sp, i, "table-row"))
                self.claims.append({"id": cid, "kind": "card-claim", "status_cell": cells[1], "claim_cell": cells[0],
                                    "source": self.src(sp, i, "table-row"), "evidence": []})

    def finish_basis(self):
        """Resolve sidecar basis paths after EXP/ACJ nodes exist so they attach to the same nodes."""
        basis_evidence = []
        for sid, item, sp, line in self.pending_basis:
            ref = re.split(r"\s+\(", item, 1)[0]
            ann = item[len(ref):].strip()
            if self.guard(ref):
                continue
            k, ab, disp = self.P.resolve(ref)
            where = f"{self.P.display(sp)}:{line}"
            if k != "ok":
                self.ref_gap(k, ref, where)
                continue
            if ab in self.pathnodes:
                n = self.pathnodes[ab]
            else:
                bn = os.path.basename(ab)
                kind = "VERIFY" if re.match(r"VERIFY-.*\.return\.yaml$", bn) else "DOC"
                n = self.node(f"DOC:{disp}", kind, ab)
            self.edge(sid, n, "basis", self.src(sp, line, "frontmatter-field"))
            tok = self.nodes[n].get("verifier_token")
            basis_evidence.append({"kind": "file", "path": disp, "line": line, "regen": None, "verifier_token": tok,
                                   "annotation": ann or None, "source": self.src(sp, line, "frontmatter-field")})
        if basis_evidence:
            for c in self.claims:
                if c["kind"] == "card-claim":
                    c["evidence"] = basis_evidence
                    c["evidence_scope"] = "sidecar basis (shared by the claims of this sidecar)"

    # ---- atoms
    def do_atoms(self):
        w, h = self.w, self.h
        for a in sorted(w.atoms_by_h.get(h, []), key=lambda a: a["id"]):
            aid, fm, ap = a["id"], a["fm"], a["path"]
            d = fm.data
            self.node(aid, a["kind"], ap)
            hl = fm.line("hypothesis_ids")
            self.edge(aid, h, "about-hypothesis", self.src(ap, hl, "frontmatter-field"))
            for j, e in enumerate(as_list(d.get("exp_ids"))):
                e = str(e)
                if EXP_RE.fullmatch(e):
                    self.ensure_exp(e)
                    self.edge(aid, e, "about-experiment", self.src(ap, fm.line("exp_ids", j) if isinstance(d.get("exp_ids"), list) else fm.line("exp_ids"), "frontmatter-field"))
            for j, dep in enumerate(as_list(d.get("depends_on"))):
                self.h_ref(aid, str(dep), "depends-on", ap, fm.line("depends_on", j), "frontmatter-field")
            ev_out = []
            for j, ev in enumerate(as_list(d.get("evidence"))):
                ln = fm.line("evidence", j)
                if isinstance(ev, dict) and "run_id" in ev:
                    rid, proj = str(ev["run_id"]), str(ev.get("project", ""))
                    nid = self.add_run(aid, proj, rid, None, "evidence", self.src(ap, ln, "frontmatter-field"))
                    if nid:
                        ev_out.append({"kind": "run", "project": proj, "run_id": rid, "line": ln, "regen": None, "verifier_token": None})
                    else:
                        ev_out.append({"kind": "run", "withheld": True, "line": ln})
                elif isinstance(ev, dict) and "path" in ev:
                    n = self.path_ref(aid, str(ev["path"]), "evidence", ap, ln, "frontmatter-field")
                    if n:
                        item = {"kind": "file", "path": self.nodes[n]["path"], "line": ln, "regen": None, "verifier_token": None}
                        m = re.search(r"#(.+)$|:([\w.,\-]+)$", str(ev["path"]).strip().strip("`"))
                        if m:
                            item["anchor"] = m.group(1) or m.group(2)  # "196-222" of trm.py:196-222, "§1" of x.md#§1; resolve() strips both
                        ev_out.append(item)
                    elif self.w.W.match_path(str(ev["path"])):
                        ev_out.append({"kind": "file", "withheld": True, "line": ln})
            if a["kind"] == "CL":
                self.claims.append({"id": aid, "kind": "atom-claim", "status_cell": str(d.get("verdict")), "review": d.get("review"),
                                    "source": self.src(ap, fm.line("verdict"), "frontmatter-field"), "evidence": ev_out})
            else:
                for j, s in enumerate(as_list(d.get("serves"))):
                    self.h_ref(aid, str(s), "serves", ap, fm.line("serves", j) if isinstance(d.get("serves"), list) else fm.line("serves"), "frontmatter-field")
                fig = {"atom": aid, "path": self.P.display(ap), "serves": [str(s) for s in as_list(d.get("serves"))],
                       "render": None, "review": d.get("review"), "verdict": str(d.get("verdict")),
                       "prohibited_upgrade": d.get("prohibited_upgrade"), "source": self.src(ap, fm.line("atom_id"), "frontmatter-field")}
                fs = d.get("figspec")
                if isinstance(fs, dict) and fs.get("output"):
                    fig["figspec_output_unstructured"] = fs["output"]
                if "render" in d and isinstance(d["render"], list):
                    out = []
                    for j, r in enumerate(d["render"]):
                        ln = fm.line("render", j)
                        rp = str(r.get("path", ""))
                        if self.guard(rp):
                            out.append({"path": None, "withheld": True, "line": ln})
                            continue
                        k, ab, disp = self.P.resolve(rp)
                        item = {"path": rp, "sha256_declared": r.get("sha256"), "rendered_at": str(r.get("rendered_at")), "exists": k == "ok", "sha256_matches": None, "line": ln}
                        if k == "ok":
                            fid = self.node(f"FIGFILE:{disp}", "FIGFILE", ab)
                            self.edge(aid, fid, "rendered-as", self.src(ap, ln, "frontmatter-field"))
                            item["sha256_matches"] = sha256_file(ab) == str(r.get("sha256"))
                            if not item["sha256_matches"]:
                                self.gap("FG_RENDER_SHA_MISMATCH", f"{rp} differs from declared sha256", f"{self.P.display(ap)}:{ln}")
                        else:
                            self.gap("FG_RENDER_MISSING_FILE", f"render path not found: {rp}", f"{self.P.display(ap)}:{ln}")
                        out.append(item)
                    fig["render"] = out
                elif "render" in d:
                    fig["render_pending_reason"] = d.get("render_pending_reason")
                    self.gap("FG_RENDER_PENDING", f"{aid}: render is null", self.P.display(ap))
                else:
                    self.gap("FG_NO_RENDER", f"{aid} has no structured render field", self.P.display(ap))
                self.figures.append(fig)

    # ---- link-in from experiment side
    def do_exp_links(self):
        w, h = self.w, self.h
        for eid, sec in sorted(w.expmd.items()):
            if h in sec["h_ids"]:
                self.ensure_exp(eid)
                self.edge(h, eid, sec["h_rel"].get(h, "tested-in"), self.src(w.expmd_path, sec["h_line"], "table-cell"))
        for eid, fpath, line in w.exp_h_links.get(h, []):
            self.ensure_exp(eid)
            self.edge(h, eid, "tested-in", self.src(fpath, line, "frontmatter-field"))
        for name in w.acj_by_h.get(h, []):
            self.expand_acj(name)

    # ---- status
    def do_status(self):
        w, h = self.w, self.h
        side = getattr(self, "sidecar_status", None)
        reg_word = None
        if self.registry_row:
            m = re.match(r"\*\*([^*]+)\*\*", self.registry_row["status_cell"])
            cand = m.group(1) if m else self.registry_row["status_cell"]
            if cand.lower() in [v.lower() for v in w.vocab] and (m or cand == self.registry_row["status_cell"]):
                reg_word = cand
            else:
                self.gap("STATUS_WORD_UNCONTROLLED", "registry status cell does not start with a controlled word; word: null", f"{self.registry_row['file']}:{self.registry_row['line']}")
        rsrc = self.src(w.reg_path, self.registry_row["line"], "table-cell") if self.registry_row else None
        out = {"word": None, "cell": None, "authority": None, "source": None,
               "registry": {"word": reg_word, "cell": self.registry_row["status_cell"], "source": rsrc} if self.registry_row else None,
               "sidecar": None, "card": getattr(self, "card_status", None)}
        if side:
            sw = side["cell"] if side["cell"].lower() in [v.lower() for v in w.vocab] else None
            if sw is None:  # e.g. H-200 sidecar "falsified-as-framed (C-ACT-1 ...)": qualified cell, kept verbatim, word null
                self.gap("STATUS_WORD_UNCONTROLLED", "sidecar status is not a bare controlled word; word: null", f"{side['source']['file']}:{side['source']['line']}")
            out["sidecar"] = {"word": sw, "cell": side["cell"], "source": side["source"]}
            out.update(word=sw, cell=side["cell"], authority="sidecar", source=side["source"])
            if reg_word and sw and reg_word.lower() != sw.lower():
                self.gap("STATUS_DIVERGE", f"sidecar '{sw}' vs registry '{reg_word}'", h)
        elif self.registry_row:
            out.update(word=reg_word, cell=self.registry_row["status_cell"], authority="registry", source=rsrc)
        elif out["card"]:
            out.update(cell=out["card"]["cell"], authority="card", source=out["card"]["source"])
        else:
            out["authority"] = None
        self.status = out

    # ---- claims bookkeeping
    def do_claims_gaps(self):
        have = {c["id"] for c in self.claims}
        for cid in getattr(self, "card_claim_ids", []):
            if cid not in have:
                nid = f"{self.h}:{cid}"
                self.claims.append({"id": cid, "kind": "card-claim", "status_cell": None, "source": None, "evidence": []})
                self.gap("CLAIM_NO_STATUS_CELL", f"{cid} named in card but no status cell in any sidecar", nid)

    # ---- timeline
    def do_timeline(self):
        w = self.w
        tl, with_event = [], set()
        by_node = {}
        for n in self.nodes.values():
            if n["kind"] in TIMELINE_KINDS and n["path"]:
                abs_ = os.path.join(w.P.lab_root, n["path"][4:]) if n["path"].startswith("lab/") else None
                rel = w.P.lab_rel(abs_) if abs_ else None
                if rel:
                    by_node[n["id"]] = rel
        for origin, ef, ln, ev in w.events:
            p = ev.get("path")
            hit = [nid for nid, rel in by_node.items() if p and (p == rel or p.startswith(rel.rstrip("/") + "/"))]
            if not hit and origin == "backfill" and self.h in (ev.get("h_ids") or []):
                hit = [self.h]  # registry-level writebacks (HYPOTHESES.md / EXPERIMENTS.md) are not timeline nodes
            for nid in hit:
                if ev.get("event", True):
                    with_event.add(nid)
                stage = ev.get("stage")
                ent = {"ts": ev.get("ts_utc") or ev.get("ts"), "stage": stage,
                       "what": f"{stage} {origin} event ({ev.get('code')}) on {os.path.basename(p or '')}",
                       "origin": origin, "ts_basis": ev.get("ts_basis", "event"), "node": nid, "event": ev.get("event", True),
                       "source": self.src(ef, ln, "file-line")}
                if origin == "backfill":
                    ent["event_kind"], ent["ts_kind"] = ev.get("event_kind"), ev.get("ts_kind")
                    ent["note"] = ev.get("note")  # status word as quoted by the backfill scanner (verbatim)
                    ent["backfill_source"] = ev.get("source")
                tl.append(ent)
        for nid, n in self.nodes.items():
            if nid in with_event or n["kind"] not in TIMELINE_KINDS or not n["mtime"]:
                continue
            tl.append({"ts": n["mtime"], "stage": KIND_STAGE.get(n["kind"]), "what": f"file mtime of {n['kind']} {os.path.basename(n['path'])}",
                       "origin": "mtime", "ts_basis": "mtime", "node": nid, "source": None})
        tl.sort(key=lambda t: (t["ts"] or "", t["node"]))
        self.timeline = tl
        if "SIDE" in {n["kind"] for n in self.nodes.values()} and not any(t["stage"] == "S7" and t["origin"] in ("events", "backfill") for t in tl):
            self.gap("NO_S7_EVENT", "events.jsonl / backfill has no S7 event for this hypothesis", self.h)

    def build(self):
        self.do_card()
        self.do_sidecar()
        self.do_exp_links()
        self.do_atoms()
        self.do_registry()
        self.finish_basis()
        self.do_claims_gaps()
        self.do_status()
        if any(n["kind"] == "EXP" for n in self.nodes.values()) and not self.figures:
            self.gap("NO_FIGURE_ATOM", "hypothesis has experiments but no FG atom", self.h)
        self.do_timeline()
        return self.to_json()

    def to_json(self):
        nodes = [self.nodes[k] for k in sorted(self.nodes)]
        edges = sorted(self.edges, key=lambda e: (e["from"], e["to"], e["rel"], e["source"]["file"], e["source"]["line"]))
        return {
            "schema": SCHEMA, "h_id": self.h, "generated_at": utcnow(),
            "generator": GENERATOR,
            "registry_row": self.registry_row,
            "registry_note": None if self.registry_row else "레지스트리 행 없음",
            "origin": self.w.h_origin[self.h],
            "primary_experiments": list(self.exps),
            "status": self.status,
            "claims": self.claims,
            "nodes": nodes, "edges": edges,
            "figures": self.figures,
            "timeline": self.timeline,
            "withheld_ref_count": self.withheld_count,
            "gaps": self.gaps,
        }


GENERATOR = {"path": GENERATOR_PATH, "sha256": None}


def set_generator():
    here = os.path.dirname(os.path.abspath(__file__))
    h = hashlib.sha256()
    for f in ("build.py", "frontmatter.py", "paths.py", "withheld.py"):
        with open(os.path.join(here, f), "rb") as fh:
            h.update(fh.read())
    GENERATOR["sha256"] = h.hexdigest()
    GENERATOR["covers"] = ["build.py", "frontmatter.py", "paths.py", "withheld.py"]


README = """# 16_evidence-trace

Generated by `tools/hypothesis_trace/build.py` (plan R3; design: 22_plans/.../20_2026-10-19_autoresearch-rebuild/10_DASHBOARD_PLAN.md section 4).
Do not edit by hand: re-run `cd /home/ayp/project/trm && uv run python tools/hypothesis_trace/build.py`.

## Files
- `H-NNN.trace.json` - one per hypothesis, schema `trm.hypothesis_trace/0.1`. File name = hypothesis id with characters outside
  `[A-Za-z0-9._-]` replaced by `_` (registry row `D1 suite` -> `D1_suite.trace.json`; the real id is the `h_id` field).
- `index.json` - one summary row per hypothesis (status cell, claims/figures/gaps counts, last timeline timestamp), gap totals by code,
  hypothesis-like tokens that are not H-NNN ids, generator hash. Only written by a full run (no `--h`).
- `README.md` - this file.
- `backfill_events.jsonl` (R6, written by `tools/hypothesis_trace/backfill_s7.py`, never by build.py): read after `events.jsonl`; timeline entries carry `origin` (`events` | `backfill` | `mtime`) and `ts_basis` (`event` | `stated` | `mtime`).

## Trace fields
`registry_row` (cells copied verbatim, null + `registry_note` when HYPOTHESES.md has no row), `status` (authority sidecar > registry > card; each
source kept verbatim next to the chosen one; `word` is non-null only when the cell is a controlled-vocabulary word), `claims`, `nodes`, `edges`,
`figures`, `timeline`, `withheld_ref_count`, `gaps`.
Node kinds: H SIDE CLAIM SCOUT ACJ EXP FREEZE VERIFY RESULTS RETRO CL FG FIGFILE RUN DOC.
Every edge has `source = {file, line, basis}` (basis: frontmatter-field | table-cell | table-row | registry-cell | file-line | filename-convention).
`filename-convention` means the declaring file could not be read (ARG-locked / withheld EXP) or names no target; line is then 1.
A path that does not exist never becomes an edge; it becomes a gap.
Backfill coverage: sidecars, EXP RESULTS status, every audits.dir session dir with a frontmatter `verdict:` plus `*_acj.md`, atoms, and dated HYPOTHESES/EXPERIMENTS status cells/changelog bullets; undated registry statuses and withheld-dropped rows are not backfilled. Only rows with `event: true` (stage S7; kinds verdict|writeback|review) count for the NO_S7_EVENT gap; `event: false` rows (created atoms, in-progress/under-test/enqueued mentions) are shown with `ts_kind` created|mention.
index.json `last_ts` = the latest non-mtime timeline row that is a verdict (backfill `event_kind` in verdict|results-verified|paper-claim-acj, or an events.jsonl S7 row), falling back to the latest mtime row (`last_ts_basis` says which); `last_status_change_ts` = the latest other non-mtime `event: true` row (writeback, review, ...).
Claim `evidence[]`: `line` is the line in the claim's declaring file (the item's own `source.file` when present, else the claim's `source.file`) where the item is listed, not a line inside `path`; `anchor` (when present) is the suffix the declaring file gave after `path:` or `path#` (e.g. `196-222`, `§1`), which path resolution strips.
Timeline: `origin` events (events.jsonl `path` equals the node path or a parent directory; `ts_basis` event; event `note` text is deliberately not copied) | backfill (backfill_events.jsonl; matched by `path`, else by `h_ids`; `ts_basis` stated | mtime; `note` = status word quoted verbatim, `backfill_source` = {file, line, ts_line, mtime}) | mtime (file mtime fallback).
Stage labels on `mtime` rows come from node kind (H S1, SCOUT S2, ACJ S3, EXP/FREEZE S4, VERIFY S4/S6, RESULTS S6, SIDE S7, RETRO S8) and are a label, not a claim.

## Blinding
Metric values are never copied. RUN nodes carry project/run_id/run_name only; ids matching an active WITHHELD_RUNS.yaml family are replaced by
the count `withheld_ref_count`. Path-form references (evidence paths, edge targets, frontmatter paths) any of whose components matches a withheld
pattern are dropped and counted there; the count therefore mixes runs and documents. The H-108 run family currently also matches report
documents whose slug begins with that hypothesis id, so those are dropped and counted too (fail-closed, pending the WITHHELD owner's ruling).
Only basenames of documents actually listed in experiments.dir / theory.cards are exempt from the path guard. The `verdict-prose` family (judgement words) is not a run-id family and is not applied. EXP-016 and EXP-018 (ARG-locked) and
withheld EXP ids (EXP-012, EXP-013) are list-only: path, sha256, mtime; contents are not read. Status cells are copied verbatim (R4) and may contain
counts that were written in the registry or sidecar themselves.
Generation-time redaction (R7-r1 NB4): after each trace is built, every string in it (and in index.json) has its withheld-pattern
matches replaced by `[열람 보류]`, including verbatim cells, file names and node paths (a list-only blind EXP node may therefore show a
redacted path). A bare hypothesis id (`H-108`) is kept; suffixed tokens are not. `redacted_strings` lists the JSON pointer and match count of
each replaced string; family ids are not written (they match the patterns themselves), and index.json carries `withheld_families_applied_n` only.

## Gap codes
"""


def write_readme(out):
    lines = [README]
    for c, d in GAP_CODES.items():
        lines.append(f"- `{c}` - {d}")
    with open(os.path.join(out, "README.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def last_verify(trace):
    best = None
    prim = set(trace["primary_experiments"])
    for n in trace["nodes"]:
        if n["kind"] == "VERIFY" and n.get("target") == "results" and n["id"].split(":")[0] in prim:
            if best is None or n["round"] > best["round"]:
                best = n
    return None if best is None else {"node": best["id"], "round": best["round"], "token": best.get("verifier_token")}


def run(out, only, paths_file):
    set_generator()
    w = World(paths_file)
    os.makedirs(out, exist_ok=True)
    targets = [only] if only else w.h_ids
    if only and only not in w.h_origin:
        raise ParseError("<cli>", f"--h {only} is not a hypothesis in the set")
    index, totals, gap_counts = [], Counter(), Counter()
    for h in targets:
        t = Trace(w, h).build()
        red = []
        t = redact_tree(w.W, t, "", red)
        t["redacted_strings"] = red  # JSON pointers whose string had a withheld-pattern match replaced by REDACTED
        fn = f"{safe_name(h)}.trace.json"
        with open(os.path.join(out, fn), "w", encoding="utf-8") as fh:
            json.dump(t, fh, ensure_ascii=False, indent=1, default=_json_default)
            fh.write("\n")
        totals.update(hypotheses=1, nodes=len(t["nodes"]), edges=len(t["edges"]), gaps=len(t["gaps"]))
        gap_counts.update(g["code"] for g in t["gaps"])
        live = [x for x in t["timeline"] if x.get("event", True)]
        ev = [x for x in live if x["ts_basis"] != "mtime"]
        verd = [x for x in ev if (x["origin"] == "backfill" and x.get("event_kind") in VERDICT_KINDS) or (x["origin"] == "events" and x["stage"] == "S7")]
        other = [x for x in ev if x not in verd]
        last = (verd or [x for x in t["timeline"] if x["origin"] == "mtime"])[-1:] or None
        last = last[0] if last else None
        lsc = other[-1] if other else None
        index.append({"h_id": h, "file": fn, "origin": t["origin"], "registry_row": t["registry_row"] is not None,
                      "status_cell": (t["registry_row"] or {}).get("status_cell"), "statement_cell": (t["registry_row"] or {}).get("statement_cell"),
                      "status_word": (t["status"] or {}).get("word"), "status_authority": (t["status"] or {}).get("authority"),
                      "n_claims": len(t["claims"]), "last_verify": last_verify(t), "n_figures": len(t["figures"]),
                      "n_nodes": len(t["nodes"]), "n_edges": len(t["edges"]), "n_gaps": len(t["gaps"]),
                      "last_ts": last["ts"] if last else None, "last_ts_basis": last["ts_basis"] if last else None,
                      "last_status_change_ts": lsc["ts"] if lsc else None})
    if not only:
        idx = {"schema": "trm.hypothesis_trace.index/0.1", "generated_at": utcnow(), "generator": GENERATOR,
               "totals": dict(totals), "gap_counts": dict(sorted(gap_counts.items())),
               "h_like_tokens_not_h_nnn": w.h_like_unmapped, "ignored_placeholders": sorted(PLACEHOLDER_H),
               "withheld_families_applied_n": len(w.W.families), "hypotheses": index}
        idx = redact_tree(w.W, idx)
        with open(os.path.join(out, "index.json"), "w", encoding="utf-8") as fh:
            json.dump(idx, fh, ensure_ascii=False, indent=1, default=_json_default)
            fh.write("\n")
        write_readme(out)
    return totals, gap_counts, w


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--h")
    ap.add_argument("--paths-file", default=DEFAULT_PATHS_FILE)
    a = ap.parse_args(argv)
    try:
        P = Paths(a.paths_file)
        out = a.out or P.abs("monitoring.evidence_trace")
        totals, gaps, _ = run(out, a.h, a.paths_file)
    except ParseError as e:
        print(f"PARSE FAILURE: {e.path}: {e.msg}", file=sys.stderr)
        return 2
    except Exception:
        traceback.print_exc()
        return 1
    print(f"out={out}")
    print(f"hypotheses={totals['hypotheses']} nodes={totals['nodes']} edges={totals['edges']} gaps={totals['gaps']}")
    for c, n in sorted(gaps.items()):
        print(f"  gap {c}: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
