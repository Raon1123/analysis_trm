#!/usr/bin/env python
"""S7 timestamp backfill (plan R6, gap G5; user decision D-3: separate file, events.jsonl is never touched).

    uv run python tools/hypothesis_trace/backfill_s7.py [--out FILE] [--paths-file PATHS.yaml]

Scans files of record for past S7 writebacks and writes one JSON row per event to
PATHS `monitoring.backfill_events` (fully derived; overwritten on every run; deterministic order).

Sources (code):
  STATUS     status sidecars  <theory.cards>/H-*.status.md       note = `status:` value, ts = `updated:`
  HYPSTATUS  HYPOTHESES.md table rows whose Status cell itself carries an ISO date
  REGLOG     HYPOTHESES.md dated changelog bullets `- **YYYY-MM-DD . heading**` whose heading names a status token
  EXPSTATUS  EXPERIMENTS.md `| **Status** | cell |` rows whose cell carries an ISO date
  RESULTS    EXP-*_RESULTS.md frontmatter `status:` (ts = date inside that status value, else `date:` line, else mtime)
  ACJ        <audits.dir>/*/verdict.md with frontmatter `verdict:` (undated -> mtime basis) and <audits.dir>/*_acj.md   note = `verdict:` value, ts = `date:`
  ATOM       atoms claims/figures   note = `verdict:` value (claims) or the `created:` line (figures), ts = `created:`

Row: {stage:"S7" (only when event:true; else null), event, event_kind (verdict|writeback|review|created|status-pending), ts_kind, code, h_ids, exp_ids, cl_ids, note, path (lab-relative, as in events.jsonl),
      source:{file, line, ts_line, mtime}, ts, ts_basis: stated|mtime, ts_precision: date|datetime|mtime}
  * `note` is a verbatim substring of source line `line`; the `ts` text is a verbatim substring of `ts_line`
    (ts_line == line when the date sits on the status line). No row without a file:line.
  * ts_basis `stated` = explicit date in the text; `mtime` = file mtime fallback (only for sources whose
    status line exists but states no date). Time of day is never invented: a date-only source gives `YYYY-MM-DD`.
  * Withheld guard (R5): ARG-locked / blind EXPs are not opened; files whose name matches a withheld pattern are
    skipped; rows whose note/path/file match a withheld pattern are dropped (counted).
  * No verdict language is produced: notes are quoted, never normalised.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml  # noqa: E402

from frontmatter import ParseError  # noqa: E402
from paths import DEFAULT_PATHS_FILE, Paths  # noqa: E402
from withheld import Withheld  # noqa: E402

LOCKED_EXPS = {"EXP-016", "EXP-018"}  # keep in sync with build.LOCKED_EXPS (asserted by the tests)
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}(?::\d{2})?Z?)?")
EXP_RE = re.compile(r"(?<![A-Za-z0-9])EXP-\d{3}(?!\d)")
H_RE = re.compile(r"(?<![A-Za-z0-9])H-\d{3}(?!\d)")
CL_RE = re.compile(r"(?<![A-Za-z0-9])CL-\d{4}(?!\d)")
MD_VERDICT_RE = re.compile(r"^(?:- )?\*\*[Vv]erdict:?\*\*:?\s*\*\*(.+?)\*\*|^## VERDICT: \*\*(.+?)\*\*")
MD_DATE_RE = re.compile(r"^(?:- )?\*\*[Dd]ate:?\*\*:?\s*(" + DATE_RE.pattern + ")")
LOG_RE = re.compile(r"^- \*\*(\d{4}-\d{2}-\d{2}) · (.+?)\*\*")
STATUS_TOKEN_RE = re.compile(r"UNRUN|PASS|FAIL|DRAFT|S[67]\b|verdict|판정|enqueued|VERIFIED|completed|falsified|supported|partial", re.I)
EVENT_KINDS = {"verdict", "results-verified", "paper-claim-acj", "writeback", "review"}  # stage S7 only for these; everything else is event:false, stage null
LEAD_VERDICT_RE = re.compile(r"^[*\s]*(completed|falsified|supported|refuted|partial|confirmed|weakened)", re.I)
VERDICT_TOKEN_RE = re.compile(r"PASS|VERIFIED")


def classify(text):
    """Writeback/verdict (leading controlled verdict word, or a PASS/VERIFIED token) vs dated status mention."""
    return "writeback" if (LEAD_VERDICT_RE.match(text) or VERDICT_TOKEN_RE.search(text)) else "status-pending"


BLOCK_SCALARS = {"", ">", "|", ">-", "|-", ">+", "|+"}


def mtime_iso(p):
    return dt.datetime.fromtimestamp(os.stat(p).st_mtime, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def split_row(line):
    parts = re.split(r"(?<!\\)\|", line.strip())
    return [c.strip() for c in parts[1:-1]]


def fm_range(lines):
    """(start, end) 0-based line indices of the frontmatter body, or None."""
    if not lines or lines[0].strip() != "---":
        return None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return (1, i)
    return None


def fm_key(lines, rng, key):
    """(0-based line index, raw value text) for a top-level `key:` in the frontmatter, else None."""
    for i in range(rng[0], rng[1]):
        m = re.match(re.escape(key) + r":[ \t]*(.*)$", lines[i])
        if m:
            return i, m.group(1).rstrip()
    return None


def unquote(v):
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        return v[1:-1]
    return v


class Scanner:
    def __init__(self, paths_file=DEFAULT_PATHS_FILE):
        self.P = P = Paths(paths_file)
        self.W = Withheld(P.abs("governance.withheld_runs"))
        names = set()
        for logical in ("experiments.dir", "theory.cards"):
            d = P.abs(logical)
            if os.path.isdir(d):
                names.update(n for n in os.listdir(d) if os.path.isfile(os.path.join(d, n)) and n.endswith((".md", ".yaml", ".sha256")))
        self.W.set_doc_names(names)
        self.rows = []
        self.unparsed = []  # (display path, reason)
        self.dropped_withheld = 0
        self.skipped_files = []  # (display path, reason)

    # -- helpers
    def read_lines(self, p):
        with open(p, encoding="utf-8") as fh:
            return fh.read().split("\n")

    def skip_file(self, p):
        base = os.path.basename(p)
        if self.W.match_path(base):
            self.skipped_files.append((self.P.display(p), "withheld name pattern"))
            return True
        for e in set(EXP_RE.findall(base)):
            if e in LOCKED_EXPS or self.W.is_blind_exp(e):
                self.skipped_files.append((self.P.display(p), f"{e} locked/blind: not opened"))
                return True
        return False

    def add(self, code, p, line_idx, note, ts_line_idx, ts_text, event_kind="status-pending", ts_kind=None, h_ids=(), exp_ids=(), cl_ids=()):
        """line_idx/ts_line_idx are 0-based; ts_text None -> mtime fallback."""
        rel = self.P.lab_rel(p)
        row = {
            "stage": "S7" if event_kind in EVENT_KINDS else None, "code": code,
            "event": event_kind in EVENT_KINDS, "event_kind": event_kind,
            "h_ids": sorted(set(h_ids)), "exp_ids": sorted(set(exp_ids)), "cl_ids": sorted(set(cl_ids)),
            "note": note, "path": rel if rel is not None else self.P.display(p),
            "source": {"file": self.P.display(p), "line": line_idx + 1, "ts_line": (ts_line_idx if ts_text else line_idx) + 1,
                       "mtime": mtime_iso(p)},
        }
        # ts_kind: writeback (date of the verdict/writeback) | review (atom review.date) | created (atom creation date, not a verdict)
        # | mention (date merely stated next to a non-verdict status)
        row["ts_kind"] = ts_kind or ("writeback" if row["event"] else "mention")
        if ts_text:
            row["ts"], row["ts_basis"] = ts_text, "stated"
            row["ts_precision"] = "datetime" if "T" in ts_text else "date"
        else:
            row["ts"], row["ts_basis"], row["ts_precision"] = row["source"]["mtime"], "mtime", "mtime"
        for s in (row["note"], row["path"], row["source"]["file"]):
            if self.W.match_path(s):
                self.dropped_withheld += 1
                return
        if any(e in LOCKED_EXPS or self.W.is_blind_exp(e) for e in row["exp_ids"]):
            self.dropped_withheld += 1
            return
        self.rows.append(row)

    def fm_ids(self, lines, rng, keys, p):
        text = "\n".join(lines[rng[0]:rng[1]])
        try:
            d = yaml.safe_load(text) or {}
        except yaml.YAMLError as e:
            self.unparsed.append((self.P.display(p), f"frontmatter yaml: {str(e).splitlines()[0]}"))
            return []
        out = []
        for k in keys:
            v = d.get(k) if isinstance(d, dict) else None
            for x in (v if isinstance(v, list) else [v]):
                if x is not None:
                    out.append(str(x))
        return out

    # -- frontmatter note + date sources
    def fm_event(self, code, p, note_key, date_keys, ids_keys, prefer_note_date=False, classify_note=False):
        lines = self.read_lines(p)
        rng = fm_range(lines)
        if not rng:
            if code == "ACJ":
                return self.md_acj(p, lines)
            self.unparsed.append((self.P.display(p), "no frontmatter"))
            return
        nk = fm_key(lines, rng, note_key)
        if not nk:
            self.unparsed.append((self.P.display(p), f"no `{note_key}:` line"))
            return
        ni, nraw = nk
        note = unquote(nraw)
        if nraw in BLOCK_SCALARS or not note:
            self.unparsed.append((self.P.display(p), f"`{note_key}:` is a block scalar/empty (line {ni + 1})"))
            return
        ts_text, ts_i = None, ni
        if prefer_note_date:
            m = DATE_RE.search(note)
            if m:
                ts_text = m.group(0)
        if ts_text is None:
            for dk in date_keys:
                k = fm_key(lines, rng, dk)
                if k:
                    m = DATE_RE.search(unquote(k[1]))
                    if m:
                        ts_text, ts_i = m.group(0), k[0]
                        break
        ids = self.fm_ids(lines, rng, ids_keys, p)
        base = os.path.basename(p)
        blob = " ".join(ids) + " " + base + " " + os.path.basename(os.path.dirname(p))
        h = set(H_RE.findall(blob)) | {"H-%s" % m.group(1) for m in re.finditer(r"(?<![A-Za-z0-9])h(\d{3})(?!\d)", base, re.I)}
        kind = "verdict"
        if classify_note:
            kind = "results-verified" if VERDICT_TOKEN_RE.search(note) else classify(note)
        elif code == "ACJ" and not h and not set(CL_RE.findall(blob)):
            kind = "paper-claim-acj"
        self.add(code, p, ni, note, ts_i, ts_text, event_kind=kind, h_ids=h, exp_ids=set(EXP_RE.findall(blob)), cl_ids=set(CL_RE.findall(blob)))

    def md_acj(self, p, lines):
        """Pre-frontmatter ACJ verdict files: bold `**Verdict**: **X**` / `## VERDICT: **X**` and `**Date**: D` in the first 60 lines."""
        ni = note = di = dtext = None
        for i, ln in enumerate(lines[:60]):
            m = MD_VERDICT_RE.match(ln)
            if m and ni is None:
                ni, note = i, (m.group(1) or m.group(2))
            m = MD_DATE_RE.match(ln)
            if m and di is None:
                di, dtext = i, m.group(1)
        if ni is None:
            self.unparsed.append((self.P.display(p), "no frontmatter and no bold Verdict line in first 60 lines"))
            return
        base = os.path.basename(p)
        h = set(H_RE.findall(lines[0])) | set(H_RE.findall(base)) | {"H-%s" % m.group(1) for m in re.finditer(r"(?<![A-Za-z0-9])h(\d{3})(?!\d)", base, re.I)}
        self.add("ACJ", p, ni, note, di if di is not None else ni, dtext, event_kind="verdict", h_ids=h, exp_ids=set(EXP_RE.findall(lines[0] + base)),
                 cl_ids=set(CL_RE.findall(lines[0])))

    # -- scanners
    def scan_sidecars(self):
        d = self.P.abs("theory.cards")
        for n in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            if re.fullmatch(r"H-[A-Za-z0-9]+\.status\.md", n):
                p = os.path.join(d, n)
                if not self.skip_file(p):
                    self.fm_event("STATUS", p, "status", ["updated"], ["id", "exp_id"])

    def scan_results(self):
        d = self.P.abs("experiments.dir")
        for n in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            if re.fullmatch(r"EXP-\d{3}_RESULTS\.md", n):
                p = os.path.join(d, n)
                if not self.skip_file(p):
                    self.fm_event("RESULTS", p, "status", ["date"], ["exp_id", "hypothesis_ids"], prefer_note_date=True, classify_note=True)

    def scan_acj(self):
        d = self.P.abs("audits.dir")
        if not os.path.isdir(d):
            return
        for root, dirs, files in os.walk(d):
            dirs[:] = sorted(x for x in dirs if x != "staged")
            if root != d and "verdict.md" in files:
                v = os.path.join(root, "verdict.md")
                rel = os.path.relpath(root, d)
                if not self.skip_file(v) and not self.W.match_path(rel):
                    lines = self.read_lines(v)
                    rng = fm_range(lines)
                    if rng and fm_key(lines, rng, "verdict"):
                        self.fm_event("ACJ", v, "verdict", ["date"], ["H_id", "h_id", "hypothesis_ids"])
        for n in sorted(os.listdir(d)):
            p = os.path.join(d, n)
            if os.path.isfile(p) and n.endswith("_acj.md") and not self.skip_file(p):
                self.fm_event("ACJ", p, "verdict", ["date"], ["H_id", "h_id", "hypothesis_ids"])

    def scan_atoms(self):
        for logical in ("atoms.claims", "atoms.figures"):
            d = self.P.abs(logical)
            for n in sorted(os.listdir(d)) if os.path.isdir(d) else []:
                if not n.endswith(".md"):
                    continue
                p = os.path.join(d, n)
                if self.skip_file(p):
                    continue
                lines = self.read_lines(p)
                rng = fm_range(lines)
                if not rng:
                    self.unparsed.append((self.P.display(p), "no frontmatter"))
                    continue
                ck = fm_key(lines, rng, "created")
                if not ck or not DATE_RE.search(ck[1]):
                    self.unparsed.append((self.P.display(p), "no dated `created:` line"))
                    continue
                vk = fm_key(lines, rng, "verdict")
                if vk and unquote(vk[1]) not in BLOCK_SCALARS:
                    ni, note = vk[0], unquote(vk[1])
                else:
                    ni, note = ck[0], lines[ck[0]].rstrip()
                ids = self.fm_ids(lines, rng, ["hypothesis_ids", "atom_id"], p)
                blob = " ".join(ids) + " " + n
                ts_i, ts_text, kind, ek = ck[0], DATE_RE.search(ck[1]).group(0), "created", "created"
                rk = fm_key(lines, rng, "review")
                if rk and re.search(r"\bstatus:\s*passed\b", rk[1]):
                    md = re.search(r"\bdate:\s*(" + DATE_RE.pattern + ")", rk[1])
                    if md:
                        ts_i, ts_text, kind, ek = rk[0], md.group(1), "review", "review"
                self.add("ATOM", p, ni, note, ts_i, ts_text, event_kind=ek, ts_kind=kind, h_ids=set(H_RE.findall(blob)),
                         exp_ids=set(EXP_RE.findall(blob)), cl_ids=set(CL_RE.findall(blob)))

    def scan_registries(self):
        # HYPOTHESES.md: status cell with a date, and dated changelog bullets
        hp = self.P.abs("registry.hypotheses")
        lines = self.read_lines(hp)
        hdr_i = next((i for i, ln in enumerate(lines) if ln.startswith("| H-id")), None)
        if hdr_i is None:
            raise ParseError(hp, "hypothesis table header '| H-id' not found")
        hdr = split_row(lines[hdr_i])
        sc = next((i for i, c in enumerate(hdr) if c.lower().startswith("status")), None)
        if sc is None:
            raise ParseError(hp, "status column missing")
        i = hdr_i + 2
        while i < len(lines) and lines[i].startswith("|"):
            cells = split_row(lines[i])
            if len(cells) == len(hdr):
                m = DATE_RE.search(cells[sc])
                if m:
                    self.add("HYPSTATUS", hp, i, cells[sc], i, m.group(0), event_kind=classify(cells[sc]), h_ids=H_RE.findall(cells[0]))
            i += 1
        for j, ln in enumerate(lines):
            m = LOG_RE.match(ln)
            if m and STATUS_TOKEN_RE.search(m.group(2)):
                head = m.group(2)
                self.add("REGLOG", hp, j, head, j, m.group(1), event_kind=classify(head), h_ids=H_RE.findall(head), exp_ids=EXP_RE.findall(head))
        # EXPERIMENTS.md
        ep = self.P.abs("registry.experiments")
        lines = self.read_lines(ep)
        cur, secs = None, []
        for j, ln in enumerate(lines):
            m = re.match(r"## (EXP-\d{3})\b", ln)
            if m:
                cur = {"id": m.group(1), "status": None, "h": []}
                secs.append(cur)
            if cur and ln.startswith("|"):
                cells = split_row(ln)
                if len(cells) >= 2:
                    key = cells[0].strip("*").strip()
                    if key == "Status":
                        cur["status"] = (j, cells[1])
                    elif key == "H-NNN tested":
                        cur["h"] = sorted(set(H_RE.findall(cells[1])))
        for s in secs:
            if s["id"] in LOCKED_EXPS or self.W.is_blind_exp(s["id"]) or not s["status"]:
                continue
            j, cell = s["status"]
            m = DATE_RE.search(cell)
            if m:
                self.add("EXPSTATUS", ep, j, cell, j, m.group(0), event_kind=classify(cell), h_ids=s["h"], exp_ids=[s["id"]])

    def run(self):
        self.scan_sidecars()
        self.scan_registries()
        self.scan_results()
        self.scan_acj()
        self.scan_atoms()
        self.rows.sort(key=lambda r: (r["ts"], r["source"]["file"], r["source"]["line"], r["code"]))
        return self.rows


def write_rows(rows, out):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--paths-file", default=DEFAULT_PATHS_FILE)
    a = ap.parse_args(argv)
    try:
        sc = Scanner(a.paths_file)
        out = a.out or sc.P.abs("monitoring.backfill_events")
        rows = sc.run()
        write_rows(rows, out)
    except ParseError as e:
        print(f"PARSE FAILURE: {e.path}: {e.msg}", file=sys.stderr)
        return 2
    print(f"out={out}\nrows={len(rows)}")
    from collections import Counter
    for (c, b), n in sorted(Counter((r["code"], r["ts_basis"]) for r in rows).items()):
        print(f"  {c} {b}: {n}")
    print(f"dropped_withheld={sc.dropped_withheld} skipped_files={len(sc.skipped_files)} unparsed={len(sc.unparsed)}")
    for f, why in sc.skipped_files:
        print(f"  skipped {f}: {why}", file=sys.stderr)
    for f, why in sc.unparsed:
        print(f"  unparsed {f}: {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
