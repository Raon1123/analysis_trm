"""Stage-1 deterministic extractor for the EXP-/H- chronological timeline.

Pure file reads. No LLM, no network, no GPU. Output is byte-identical across
runs on an unchanged lab/ tree (sorted traversal, fixed serialization, no run
timestamps). Judgement is never generated here: status/verdict fields are
verbatim tokens copied from the source, with the source quote that contains
them. Copies of documents (benchmark / snapshot / backup / scratch trees) are
written to a separate sidecar, not to the canonical event log.
See README.md and the EXTRACTION_NOTES.md of the report directory.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Iterable

import yaml

LAB_DEFAULT = Path("/home/ayp/project/trm/lab")
OUTPUT_DIR_NAME = "2026-10-31_exp-h-timeline"
OUT_DEFAULT = Path(
    "/home/ayp/project/trm/lab/23_reports/11_lab-reports/" + OUTPUT_DIR_NAME
)
REPO_PREFIX = ""  # source_path / date_source are lab-relative (user spec: "lab 상대경로")
LAB_TEXT_PREFIX = "lab/"  # how registry prose cites lab files (used only to build the join map)
TEXT_CAP = 200

REGISTRY = "13_reference/11_registries/EXPERIMENTS.md"
HYPOTHESES = "13_reference/11_registries/HYPOTHESES.md"
ALIASES = "13_reference/11_registries/ID_ALIASES.yaml"
WITHHELD = "11_governance/11_lab-root/WITHHELD_RUNS.yaml"
ATOMS = "13_reference/12_atoms/INDEX.jsonl"
AUTORESEARCH_MD = "11_governance/11_lab-root/AUTORESEARCH.md"
HANDOFF_INDEX = "31_handoffs/11_lab-handoffs/INDEX.md"
GOAL_LEDGER = "11_governance/13_goal-state/G-20260902-autoresearch-rls/goal-ledger.yaml"
EXP_DIR = "21_experiments/11_lab-experiments"
REPORT_DIR = "23_reports/11_lab-reports"
THEORY_DIR = "27_theory/11_lab-theory"
AR_DIRS = (
    "22_plans/11_lab-plans/16_2026-09-02_autoresearch-rls-seed",
    "22_plans/11_lab-plans/17_2026-09-02_arg-contract",
    "22_plans/11_lab-plans/20_2026-10-19_autoresearch-rebuild",
)
# paper-position campaign roots: every EXP-like token inside is ambiguous raw only.
PP_ROOTS = (
    "23_reports/11_lab-reports/15_2026-08-30_paper-position-campaign/",
    "22_plans/11_lab-plans/12_2026-08-30_paper-position-campaign/",
)

# Event types (schema). 'witness' = a timestamped witness line that is neither
# a freeze (freeze/frozen/동결 keyword), an enqueue record, nor a judge output.
EVENT_TYPES = {
    "register", "prereg", "freeze", "enqueue", "run-complete", "judge", "results",
    "verify", "verdict", "status-change", "decision", "tool", "plan", "handoff", "witness",
}
# Only these types may carry status_token (the event itself records a state/verdict).
# Everything else keeps the document's status in doc_status_at_read (no time attribution).
STATUS_OK_TYPES = {"status-change", "verify", "verdict"}
# status-change: only the registry Status cell, when the cell has exactly one ISO date and the
#   token sits in the same segment as that date (see registry_status_token rule, NOTES §5).
# verify: only the frontmatter `verdict:` key of a verify return (its own verdict at its own date).
# verdict: reserved (no source currently emits it). handoff/results/judge/witness: never a token.
LANES = {"experiment", "hypothesis", "autoresearch", "governance"}
DATE_BASES = {"in-doc-timestamp", "frontmatter", "registry-field", "filename", "mtime"}

NULL_VALUES = {"", "~", "null", "None"}
DATE_RE = re.compile(
    r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{2}):(\d{2})(?::(\d{2}))?)?(?!\d)"
)
TZ_NAME_RE = re.compile(r"\s?(Z|UTC|KST)(?![A-Za-z0-9])")
TZ_OFFSET_RE = re.compile(r"\s?([+-]\d{2}:?\d{2})(?![0-9])")
TS_LINE_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?Z?")
TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9])("
    r"EXP[-_ ]NEW-\d+|EXP[-_ ]\d+|N\d+[a-z]?-\d{4}-\d{2}-\d{2}"
    r"|H-\d+|H-[A-Z]+(?![A-Za-z])|CL-\d{4}|FG-\d{4}"
    r")"
)
ANGLE_RE = re.compile(r"\b(reachability|learnability|stability)\b", re.IGNORECASE)
FM_KEY_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_\-]*):(?:\s+(.*))?$")
ROW_RE = re.compile(r"^\|\s*\*\*([^*]+?)\*\*\s*\|(.*)\|\s*$")
FREEZE_WORD_RE = re.compile(r"freeze|frozen|동결", re.IGNORECASE)
# negated / pre-freeze wording: such a line is a witness, never a freeze act (NOTES §6)
NEG_FREEZE_RE = re.compile(
    r"not frozen|unfrozen|before (the )?freeze|after (the )?freeze|freeze manifest|미동결|동결 전",
    re.IGNORECASE)

# ---- date keys (frontmatter / top-level yaml). Priority and roles are documented in NOTES §3.
# Result-time keys: when one of these yields a date, the date-of-record for the file is it.
RESULT_KEY_TYPE = {
    "judged": "judge", "judged_at": "judge",
    "audit_date_utc": "verify", "audit_date": "verify", "date_utc": "verify",
    "audited_at": "verify", "verified_on": "verify",
    "run_date": "results",
}
RESULT_KEY_ORDER = ("judged", "judged_at", "audit_date_utc", "audit_date", "date_utc",
                    "audited_at", "verified_on", "run_date")
# Other keys (their events are kept unless they are the superseded doc date).
OTHER_KEY_TYPE = {
    "date_designed": "prereg", "registered_at": "prereg",
    "date": "plan",            # type may be overridden from the file name
    "created": "register", "updated": "status-change", "date_closed": "status-change",
    "date_drafted": "plan",    # draft date: fallback only (see fm_events)
}
OTHER_KEY_ORDER = ("date_designed", "registered_at", "date", "created", "updated",
                   "date_closed", "date_drafted")
NAME_TYPED_KEYS = ("date", "date_designed", "registered_at")
SUPERSEDABLE_KEYS = ("date", "date_drafted")  # not used when a result-time key is present
KNOWN_DATE_KEYS = set(RESULT_KEY_TYPE) | set(OTHER_KEY_TYPE)
UNKNOWN_DATE_KEY_RE = re.compile(r"date|_at$|_on$|_utc$|^judged|^run_|^as_of$")

ID_FM_KEYS = ("id", "exp_id", "exp_ids", "hypotheses", "hypothesis", "hypothesis_ids")
NULL_DATE_VALUES = NULL_VALUES
DASHES = str.maketrans({c: "-" for c in "‐‑‒–—―−"})


class FatalConfigError(SystemExit):
    """Raised for fail-closed conditions; exit code 3."""

    def __init__(self, message: str) -> None:
        print(f"build_timeline: FATAL: {message}", file=sys.stderr)
        super().__init__(3)


# ---------------------------------------------------------------- copy classes

def is_copy_path(rel: str) -> str | None:
    """Return the copy-class reason if a lab-relative path lies in a copy tree, else None.

    Rule: a directory component `benchmark`, `transcript_extracts`, `volatile_sources*`,
    `scratchpad_*` (incl. before/after under it), `backup*`, `*_snapshot`, `recovered`;
    or a file-name token (split on `-`, `_`, `.`) equal to `copy` (e.g. `01_RESULTS.r2-pass-copy.md`).
    The earlier `*_impl` + VERIFY clause was removed (r2 verifier: those are original verifier
    returns cited by FREEZE_WITNESS files, not copies). Names such as `copying-notes.md` do not match.
    """
    parts = rel.split("/")
    name = parts[-1]
    for p in parts:
        if p == "benchmark":
            return "copy: benchmark tree"
        if p == "transcript_extracts":
            return "copy: transcript extracts"
        if p.startswith("volatile_sources"):
            return "copy: volatile_sources tree"
        if p.startswith("scratchpad_"):
            return "copy: scratchpad tree (incl. before/after)"
        if p.startswith("backup"):
            return "copy: backup tree"
        if p.endswith("_snapshot"):
            return "copy: snapshot tree"
        if p == "recovered":
            return "copy: recovered tree"
    if "copy" in re.split(r"[-_.]", name.lower()):
        return "copy: file name token 'copy' (copy-marked file)"
    return None


# ---------------------------------------------------------------- primitives

def parse_first_date(text: str):
    """(matched_text, normalized, precision, tz) for the first valid ISO date in text.

    tz is the timezone marker immediately after the match in the source text
    (Z, UTC, KST, or a +HH:MM offset for minute/second precision), else None.
    """
    for m in DATE_RE.finditer(text or ""):
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            _dt.date(y, mo, d)
        except ValueError:
            continue
        hh, mi, ss = m.group(4), m.group(5), m.group(6)
        if ss is not None:
            prec = "second"
        elif hh is not None:
            prec = "minute"
        else:
            prec = "day"
        norm = f"{m.group(1)}-{m.group(2)}-{m.group(3)}T{hh or '00'}:{mi or '00'}:{ss or '00'}"
        rest = (text or "")[m.end():]
        tz = None
        tzm = TZ_NAME_RE.match(rest)
        if tzm:
            tz = tzm.group(1)
        elif prec != "day":
            tzm = TZ_OFFSET_RE.match(rest)
            if tzm:
                tz = tzm.group(1)
        return m.group(0), norm, prec, tz
    return None


def cap(raw: str, must: str | None = None) -> str:
    """Verbatim substring of raw, at most TEXT_CAP chars, keeping `must` if possible."""
    raw = raw.rstrip()
    if len(raw) <= TEXT_CAP:
        return raw
    if must and must in raw:
        pos = raw.index(must)
        start = min(max(0, pos - 20), len(raw) - TEXT_CAP)
        return raw[start:start + TEXT_CAP]
    return raw[:TEXT_CAP]


def clean_value(v: str) -> str:
    v = v.strip()
    if v[:1] in ("'", '"'):
        end = v.find(v[0], 1)
        if end > 0:
            return v[1:end]
    m = re.search(r"\s#", v)
    if m:
        v = v[:m.start()]
    return v.strip()


def split_token(value: str) -> str | None:
    """Verbatim leading token of a status/verdict value (before ' (', ' —', ';')."""
    tok = re.split(r"\s\(|\s—|;|,\s|\s\[", value)[0].strip()
    return tok or None


def status_cell_token(value: str) -> str | None:
    """Verbatim status token anchored at the START of a Status cell.

    Anchored bold span, else anchored backtick span, else the leading text before
    the first delimiter. Spans later in the cell are never used.
    """
    v = value.strip()
    m = re.match(r"\*\*(.+?)\*\*", v)
    if m and m.group(1).strip():
        return m.group(1).strip()
    m = re.match(r"`([^`]+)`", v)
    if m and m.group(1).strip():
        return m.group(1).strip()
    return split_token(v)


def status_date_text(value: str) -> str:
    """Status-cell text with cited paths removed (backtick spans and slash-bearing tokens)."""
    v = re.sub(r"`[^`]*`", " ", value)
    return re.sub(r"\S*/\S*", " ", v)


def token_in_date_segment(token: str, clean: str, date_text: str) -> bool:
    """True if `token` occurs in the same segment as `date_text` in a status cell.

    Segments are split on ';', ' — ' and '. ' (sentence/clause boundaries). If the date
    occurs in no segment, or in more than one, the answer is False (no time attribution).
    """
    segs = [s for s in re.split(r";|\s—\s|\.\s", clean) if date_text in s]
    return len(segs) == 1 and token in segs[0]


def read_fm(lines: list[str], is_yaml: bool) -> dict:
    out: dict = {}
    if is_yaml:
        rng: Iterable[int] = range(len(lines))
    else:
        if not lines or lines[0].strip() != "---":
            return out
        end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
        if end is None:
            return out
        rng = range(1, end)
    for i in rng:
        m = FM_KEY_RE.match(lines[i])
        if m and m.group(1) not in out:
            out[m.group(1)] = (i + 1, lines[i], clean_value(m.group(2) or ""))
    return out


def name_date(name: str):
    m = re.search(r"(?<!\d)(\d{4}-\d{2}-\d{2})(?!\d)", name)
    if not m:
        return None
    return parse_first_date(m.group(1))


def type_for_name(name: str, default: str) -> str:
    if "VERIFY" in name or name.endswith(".return.yaml"):
        return "verify"
    if "RESULTS" in name:
        return "results"
    if "DECISIONS" in name:
        return "decision"
    if "prereg" in name.lower():
        return "prereg"
    return default


def normalize_text(s: str) -> str:
    return unicodedata.normalize("NFKC", s).translate(DASHES)


# ---------------------------------------------------------------- builder

class Timeline:
    def __init__(self, lab: Path) -> None:
        self.lab = lab
        self.events: list[dict] = []        # canonical events
        self.copy_events: list[dict] = []   # copy-class events (sidecar)
        self.unparsed: list[list[str]] = []
        self.groups: dict[str, set[str]] = {}
        self.file_group: dict[str, str] = {}
        self.read_hash: dict[str, str] = {}      # every input file read -> sha256
        self.file_events: dict[str, int] = {}    # source path -> number of events
        self.file_reason: dict[str, str] = {}    # source path -> why no event
        self.reg_exp: set[str] = set()
        self.aliases: dict[str, list[dict]] = {}
        self.join: dict[str, set[str]] = {}
        self.families: list[dict] = []
        self.mtime_events = 0
        self.ambiguous_tokens = 0
        self._seen: set[str] = set()

    # -- io helpers
    def repo(self, rel: str) -> str:
        return REPO_PREFIX + rel

    def read(self, group: str, rel: str) -> list[str]:
        data = (self.lab / rel).read_bytes()
        self.read_hash[rel] = hashlib.sha256(data).hexdigest()
        self.file_group[rel] = group
        self.groups.setdefault(group, set()).add(self.repo(rel))
        return data.decode("utf-8").split("\n")

    def in_pp(self, rel: str) -> bool:
        return rel.startswith(PP_ROOTS)

    def _missing(self, rel: str, why: str) -> None:
        self.unparsed.append([self.repo(rel), f"source missing: {why}"])

    # -- configuration (fail-closed first)
    def load_withheld(self) -> None:
        path = self.lab / WITHHELD
        try:
            raw = path.read_bytes()
            data = yaml.safe_load(raw.decode("utf-8"))
        except (OSError, yaml.YAMLError, UnicodeDecodeError) as exc:
            raise FatalConfigError(f"withheld registry unreadable: {exc}") from exc
        self.read_hash[WITHHELD] = hashlib.sha256(raw).hexdigest()
        if not isinstance(data, dict) or not isinstance(data.get("families"), list):
            raise FatalConfigError("withheld registry: 'families' list missing")
        fams = []
        for fam in data["families"]:
            if not isinstance(fam, dict) or not isinstance(fam.get("id"), str):
                raise FatalConfigError("withheld registry: family without id")
            status = fam.get("status")
            if status not in ("released", "withheld"):
                raise FatalConfigError(f"withheld registry: bad status in {fam.get('id')}")
            exp_ids = fam.get("exp_ids") or []
            patterns = fam.get("patterns") or []
            if not isinstance(exp_ids, list) or not isinstance(patterns, list):
                raise FatalConfigError(f"withheld registry: bad lists in {fam['id']}")
            if status == "withheld" and not exp_ids and not patterns:
                raise FatalConfigError(f"withheld family {fam['id']} has no patterns/exp_ids")
            compiled = []
            for p in patterns:
                try:
                    compiled.append(re.compile(str(p), re.IGNORECASE))
                except re.error as exc:
                    raise FatalConfigError(f"withheld pattern invalid in {fam['id']}: {exc}") from exc
            numbers = []
            for e in exp_ids:
                mm = re.fullmatch(r"EXP-(\d+)", str(e))
                if not mm:
                    raise FatalConfigError(f"withheld exp_id malformed in {fam['id']}: {e}")
                numbers.append(int(mm.group(1)))
            fams.append({"id": fam["id"], "status": status, "patterns": compiled,
                         "exp_numbers": numbers})
        self.families = fams

    def load_aliases(self) -> None:
        raw = (self.lab / ALIASES).read_bytes()
        data = yaml.safe_load(raw.decode("utf-8"))
        self.read_hash[ALIASES] = hashlib.sha256(raw).hexdigest()
        self.file_group[ALIASES] = "aliases"
        self.groups.setdefault("aliases", set()).add(self.repo(ALIASES))
        for entry in data["entries"]:
            canon = entry["canonical"]
            for a in entry.get("aliases") or []:
                scope = a.get("scope", "") or ""
                win = re.search(r"(\d{4}-\d{2}-\d{2})\s*~\s*(\d{4}-\d{2}-\d{2})", scope)
                self.aliases.setdefault(a["string"], []).append({
                    "canonical": canon,
                    "pp": "paper-position" in scope,
                    "window": (win.group(1), win.group(2)) if win else None,
                    "physical": entry.get("physical_prefix"),
                })

    # -- id resolution
    def _alias_candidates(self, key: str, pp_src: bool, date_norm: str) -> list[dict]:
        out = []
        for e in self.aliases.get(key, []):
            if e["pp"] != pp_src:
                continue
            if e["window"] and not (e["window"][0] <= date_norm[:10] <= e["window"][1]):
                continue
            out.append(e)
        return out

    def resolve(self, text: str, src_rel: str, date_norm: str) -> dict:
        res = {"exp": set(), "h": set(), "atoms": set(), "other": set(),
               "ambiguous": set(), "aliases": set()}
        pp_src = self.in_pp(src_rel)
        for m in TOKEN_RE.finditer(text):
            raw = m.group(1)
            key = re.sub(r"[ _]", "-", raw) if raw.startswith("EXP") else raw
            if re.fullmatch(r"H-\d+", raw):
                res["h"].add(raw)
            elif re.fullmatch(r"H-[A-Z]+", raw):
                res["other"].add(raw)
            elif re.fullmatch(r"CL-\d{4}|FG-\d{4}", raw):
                res["atoms"].add(raw)
            elif pp_src and key.startswith("EXP"):
                res["ambiguous"].add(raw)
            elif key in self.reg_exp and key in self.aliases:
                # a registry canonical that is also declared as another entry's alias string
                # (e.g. EXP-019 under EXP-020): never bound automatically
                res["ambiguous"].add(raw)
            else:
                cands = self._alias_candidates(key, pp_src, date_norm)
                if len(cands) > 1:
                    physical = [c for c in cands if c["physical"]]
                    cands = physical if len(physical) == 1 else cands
                if len(cands) == 1:
                    canon = cands[0]["canonical"]
                    res["exp"].add(canon)
                    res["aliases"].add((raw, canon))
                elif not cands and key.startswith("EXP-") and re.fullmatch(r"EXP-\d+", key) \
                        and f"EXP-{int(key[4:]):03d}" in self.reg_exp:
                    res["exp"].add(f"EXP-{int(key[4:]):03d}")
                else:
                    res["ambiguous"].add(raw)
        return res

    # -- event emission
    def add(self, *, etype: str, date: tuple, src_rel: str, line: int | None, title: str | None,
            token: str | None, quote: str | None, id_text: str, basis: str,
            lane: str | None = None, angle: list | None = None, notes: list | None = None,
            join_registry: bool = True, doc_status: str | None = None,
            own_exp: str | None = None) -> None:
        if etype not in EVENT_TYPES or basis not in DATE_BASES:
            raise ValueError(f"bad type/basis {etype}/{basis}")
        if token is not None and etype not in STATUS_OK_TYPES:
            raise ValueError(f"status_token not allowed on type {etype} (use doc_status_at_read)")
        dtext, dnorm, prec, dtz = date
        if re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}(?::\d{2})?", dtext):
            dtext = dtext.replace(" ", "T", 1)  # ISO-8601 separator only; digits unchanged
        src_repo = self.repo(src_rel)
        date_source = f"{src_repo}:{line}" if line else src_repo
        res = self.resolve(id_text, src_rel, dnorm)
        if own_exp:
            res["exp"].add(own_exp)
            # the registry section's own canonical is explicit: its heading token is not ambiguous
            res["ambiguous"] = {t for t in res["ambiguous"] if t != own_exp}
        if join_registry:
            for canon in sorted(self.join.get(src_rel, ())):
                res["exp"].add(canon)
        if lane is None:
            lane = "experiment" if res["exp"] else ("hypothesis" if res["h"] else "governance")
        self.ambiguous_tokens += len(res["ambiguous"])
        ev = {
            "event_id": "",
            "date": dtext,
            "date_tz": dtz,
            "date_precision": prec,
            "date_basis": basis,
            "date_source": date_source,
            "type": etype,
            "lane": lane,
            "ids": {"exp": sorted(res["exp"]), "h": sorted(res["h"]),
                    "atoms": sorted(res["atoms"]), "other": sorted(res["other"])},
            "ids_ambiguous": sorted(res["ambiguous"]),
            "aliases_seen": [{"raw": r, "canonical": c} for r, c in sorted(res["aliases"])],
            "title": title,
            "status_token": token,
            "doc_status_at_read": doc_status,
            "quote": quote,
            "angle_tokens_in_doc": angle or [],
            "source_path": src_repo,
            "withheld": False,
            "notes": "; ".join(notes or []),
        }
        key = json.dumps([etype, src_repo, date_source, dtext, token, title or ""],
                         ensure_ascii=False)
        ev["event_id"] = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
        if ev["event_id"] in self._seen:
            raise RuntimeError(f"event_id collision for {key}")
        self._seen.add(ev["event_id"])
        self.file_events[src_rel] = self.file_events.get(src_rel, 0) + 1
        reason = is_copy_path(src_rel)
        if reason:
            ev["is_copy"] = True
            ev["copy_reason"] = reason
            self.copy_events.append(ev)
        else:
            self.events.append(ev)

    def angle_hits(self, lines: list[str], start: int, end: int) -> list:
        hits = []
        for idx in range(start, end):
            for m in ANGLE_RE.finditer(lines[idx]):
                hits.append([m.group(0), idx + 1])
        return hits

    # -- sources
    def scan_registry(self) -> None:
        lines = self.read("registry", REGISTRY)
        sections = []
        cur = None
        for i, line in enumerate(lines):
            if line.startswith("## "):
                cur = None
                m = re.match(r"^## EXP-(\d+)\b", line)
                if m:
                    cur = {"num": int(m.group(1)), "start": i, "end": len(lines),
                           "heading": line[3:].strip(), "rows": {}, "text": []}
                    sections.append(cur)
            if cur is not None:
                cur["text"].append(line)
                rm = ROW_RE.match(line)
                if rm:
                    cur["rows"].setdefault(rm.group(1).strip(), (i + 1, line, rm.group(2).strip()))
        for sec in sections:
            j = sec["start"] + 1
            while j < len(lines) and not lines[j].startswith("## "):
                j += 1
            sec["end"] = j
        for sec in sections:
            canon = f"EXP-{sec['num']:03d}"
            self.reg_exp.add(canon)
        for sec in sections:
            canon = f"EXP-{sec['num']:03d}"
            # join only from the Design-doc rows (any row whose key mentions 'design')
            for key, (_lno, _raw, value) in sec["rows"].items():
                if "design" not in key.lower():
                    continue
                for path in re.findall(r"lab/[A-Za-z0-9_./\-]+", value):
                    rel = path[len(LAB_TEXT_PREFIX):].rstrip(".")
                    self.join.setdefault(rel, set()).add(canon)
        for sec in sections:
            canon = f"EXP-{sec['num']:03d}"
            h_row = sec["rows"].get("H-NNN tested")
            h_text = h_row[1] if h_row else ""
            angle = self.angle_hits(lines, sec["start"], sec["end"])
            ident = f"{sec['heading']}\n{h_text}"
            date_row = sec["rows"].get("Date registered") or sec["rows"].get("Date")
            if date_row is None:
                self.unparsed.append([self.repo(REGISTRY), f"{canon}: no Date row"])
            else:
                lno, raw, value = date_row
                dt = parse_first_date(value)
                if dt is None:
                    self.unparsed.append([self.repo(REGISTRY) + f":{lno}",
                                          f"{canon}: Date row has no ISO date"])
                else:
                    self.add(etype="register", date=dt, src_rel=REGISTRY, line=lno,
                             title=sec["heading"], token=None, quote=cap(raw), id_text=ident,
                             basis="registry-field", lane="experiment", angle=angle,
                             join_registry=False, own_exp=canon)
            st_row = sec["rows"].get("Status")
            if st_row is None:
                self.unparsed.append([self.repo(REGISTRY), f"{canon}: no Status row"])
                continue
            lno, raw, value = st_row
            dt = parse_first_date(status_date_text(value))
            if dt is None:
                self.unparsed.append([self.repo(REGISTRY) + f":{lno}",
                                      f"{canon}: Status cell has no ISO date outside cited paths"])
                continue
            token = status_cell_token(value)
            if token and token not in raw:
                token = None
            # registry_status_token rule (NOTES §5): the token is a status_token only if the cell
            # has exactly one distinct ISO date and the token shares a segment with that date.
            # Otherwise the token is the read-time status: doc_status_at_read, no time attribution.
            clean = status_date_text(value)
            distinct = {m.group(0) for m in DATE_RE.finditer(clean)}
            tok_ok = bool(token) and len(distinct) == 1 and token_in_date_segment(token, clean, dt[0])
            self.add(etype="status-change", date=dt, src_rel=REGISTRY, line=lno,
                     title=sec["heading"], token=token if tok_ok else None,
                     doc_status=None if (tok_ok or not token) else token,
                     quote=cap(raw, must=token), id_text=ident, basis="registry-field",
                     lane="experiment", angle=angle, join_registry=False, own_exp=canon,
                     notes=None if (tok_ok or not token) else
                     ["status cell has %d distinct ISO dates or token not beside the date: "
                      "doc_status_at_read, not time-attributed" % len(distinct)])

    def scan_hypothesis_log(self) -> None:
        if not (self.lab / HYPOTHESES).is_file():
            return self._missing(HYPOTHESES, "HYPOTHESES.md")
        lines = self.read("hypotheses", HYPOTHESES)
        for i, line in enumerate(lines):
            m = re.match(r"^- \*\*(\d{4}-\d{2}-\d{2}) · (.+?)\*\*", line)
            if not m:
                continue
            dt = parse_first_date(m.group(1))
            if dt is None:
                self.unparsed.append([self.repo(HYPOTHESES) + f":{i + 1}", "bullet date unparsed"])
                continue
            self.add(etype="status-change", date=dt, src_rel=HYPOTHESES, line=i + 1,
                     title=m.group(2), token=None, quote=cap(line), id_text=line,
                     basis="in-doc-timestamp", lane="hypothesis",
                     angle=self.angle_hits(lines, i, i + 1))

    def scan_witness(self, rel: str, lines: list[str]) -> None:
        name = rel.rsplit("/", 1)[-1]
        found = False
        for i, raw in enumerate(lines):
            m = TS_LINE_RE.search(raw)
            if not m:
                continue
            dt = parse_first_date(m.group(0))
            if dt is None:
                self.unparsed.append([self.repo(rel) + f":{i + 1}", "witness timestamp invalid"])
                continue
            found = True
            low = raw.lower()
            # keyword test on text with file-like tokens and code spans removed, so that a
            # reference such as `EXP-023_FREEZE.sha256` or FREEZE_WITNESS.md is not a freeze act
            # only file-like tokens (containing / _ or .) are removed; hyphenated words such as
            # "re-freeze" are kept, so a header "## v2 re-freeze (...)" remains a freeze
            cleaned = re.sub(r"\S*[/_.]\S*", " ", re.sub(r"`[^`]*`", " ", raw))
            if "enqueue" in low:
                etype = "enqueue"
            elif FREEZE_WORD_RE.search(cleaned) and not NEG_FREEZE_RE.search(raw):
                etype = "freeze"
            else:
                etype = "witness"
            self.add(etype=etype, date=dt, src_rel=rel, line=i + 1,
                     title=raw.strip()[:120], token=None,
                     quote=cap(raw, must=m.group(0)), id_text=name + "\n" + raw,
                     basis="in-doc-timestamp", lane="experiment",
                     angle=self.angle_hits(lines, 0, len(lines)),  # whole document
                     notes=["first timestamp of line only"] if len(TS_LINE_RE.findall(raw)) > 1 else None)
        if not found:
            self.file_reason[rel] = "witness-no-timestamp"

    def fm_events(self, group: str, rel: str, *, lane: str | None, fallback: bool,
                  mtime: bool = False) -> bool:
        """Frontmatter/top-level-key dated events. Returns True if any event was emitted.

        Date precedence (NOTES §3):
          1. result-time keys (judged, judged_at, audit_date_utc, verified_on, run_date)
             each emit their own event (judge/verify/results). If any of them yields a
             date, the file's `date`/`date_drafted` values are NOT emitted as events
             (recorded in the notes of the result events instead).
          2. otherwise `date` (type from file name), and the other keys
             (date_designed, registered_at, created, updated, date_closed).
          3. `date_drafted` only if nothing else was emitted (draft fallback).
        Status: the first emitted event gets the status value. It becomes status_token
        only if its type is in STATUS_OK_TYPES; otherwise doc_status_at_read.
        Unknown date-like keys are never used; they go to unparsed.
        """
        lines = self.read(group, rel)
        is_yaml = rel.endswith(".yaml")
        fm = read_fm(lines, is_yaml)
        name = rel.rsplit("/", 1)[-1]
        title = fm["title"][2] if "title" in fm and fm["title"][2] else name
        id_lines = [name, title] + [fm[k][1] for k in ID_FM_KEYS if k in fm]
        id_text = "\n".join(id_lines)
        angle = self.angle_hits(lines, 0, len(lines))  # whole document

        # (line, raw line, token, key). `verdict` is tried first: only a verdict may become a token.
        status = None
        for key in ("verdict", "status"):
            if key in fm and fm[key][2] and split_token(fm[key][2]):
                status = (fm[key][0], fm[key][1], split_token(fm[key][2]), key)
                break

        unknown_logged = False
        for k, (lno, raw, value) in fm.items():
            if k not in KNOWN_DATE_KEYS and UNKNOWN_DATE_KEY_RE.search(k) and \
                    parse_first_date(status_date_text(value)):
                self.unparsed.append([self.repo(rel) + f":{lno}",
                                      f"unknown date-like key '{k}' not used"])
                unknown_logged = True

        def present(k: str) -> bool:
            return k in fm and fm[k][2] not in NULL_DATE_VALUES

        plan: list[tuple[str, str, tuple, int, str]] = []  # (key, etype, dt, line, value)
        superseded: list[str] = []
        for k in RESULT_KEY_ORDER:
            if not present(k):
                continue
            lno, raw, value = fm[k]
            dt = parse_first_date(status_date_text(value))  # path-embedded dates are not date values
            if dt is None:
                self.unparsed.append([self.repo(rel) + f":{lno}", f"{k}: no ISO date (value: {value[:60]})"])
                continue
            plan.append((k, RESULT_KEY_TYPE[k], dt, lno, value))
        result_present = bool(plan)
        for k in OTHER_KEY_ORDER:
            if not present(k):
                continue
            lno, raw, value = fm[k]
            if result_present and k in SUPERSEDABLE_KEYS:
                superseded.append(f"{k}={value}")
                continue
            dt = parse_first_date(status_date_text(value))
            if dt is None:
                self.unparsed.append([self.repo(rel) + f":{lno}", f"{k}: no ISO date"])
                continue
            if k == "date_drafted" and plan:
                superseded.append(f"{k}={value}")  # draft date: fallback only
                continue
            etype = OTHER_KEY_TYPE[k]
            if k in NAME_TYPED_KEYS:
                etype = type_for_name(name, etype)
            plan.append((k, etype, dt, lno, value))

        emitted = False
        for idx, (k, etype, dt, lno, value) in enumerate(plan):
            raw = fm[k][1]
            notes = []
            if k == "date_drafted":
                notes.append("draft date used as fallback")
            if idx == 0 and superseded:
                notes.append("not used (superseded by result-time key): " + ", ".join(superseded))
            token = None
            doc_status = None
            quote_line, quote_raw = lno, raw
            if idx == 0 and status is not None:
                slno, sraw, stok, skey = status
                if stok in sraw:
                    quote_line, quote_raw = slno, sraw
                    if etype == "verify" and skey == "verdict":
                        token = stok  # a verify return's own verdict, at its own date
                    else:
                        doc_status = stok
                        notes.append("doc_status_at_read = status as read, not as of this event date")
            self.add(etype=etype, date=dt, src_rel=rel, line=lno, title=title, token=token,
                     quote=cap(quote_raw, must=token or doc_status), id_text=id_text,
                     basis="frontmatter", lane=lane, angle=angle, doc_status=doc_status,
                     notes=notes or None)
            emitted = True
        if emitted:
            return True
        self.file_reason[rel] = self._no_event_reason(fm, is_yaml)

        nd = name_date(name)  # the file's own name only; dir dates get dir-level events
        if nd is not None and fallback:
            token = None
            doc_status = None
            quote = None
            if status is not None:
                slno, sraw, stok, skey = status
                if stok in sraw:
                    quote = cap(sraw, must=stok)
                    if type_for_name(name, "plan") == "verify" and skey == "verdict":
                        token = stok
                    else:
                        doc_status = stok
            self.add(etype=type_for_name(name, "plan"), date=nd, src_rel=rel, line=None,
                     title=name, token=token, quote=quote, id_text=id_text, basis="filename",
                     lane=lane, angle=angle, doc_status=doc_status,
                     notes=["date from file name; no frontmatter date"])
            self.file_reason.pop(rel, None)
            return True
        if mtime:
            st = (self.lab / rel).stat().st_mtime
            stamp = _dt.datetime.fromtimestamp(st, tz=_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            self.add(etype=type_for_name(name, "plan"), date=parse_first_date(stamp), src_rel=rel,
                     line=None, title=name, token=None, quote=None, id_text=id_text, basis="mtime",
                     lane=lane, notes=["mtime fallback: file copy/sync time, not a result time"])
            self.mtime_events += 1
            self.file_reason.pop(rel, None)
            return True
        if self._nested_unknown_dates(rel, lines) or unknown_logged:
            self.file_reason[rel] = "unknown-date-key (see unparsed)"
        return False

    def _nested_unknown_dates(self, rel: str, lines: list[str]) -> bool:
        """Zero-event files: indented (nested-mapping) date-like keys are not read.

        They are recorded in unparsed with their line number, never converted to events.
        """
        hit = False
        for i, line in enumerate(lines):
            m = re.match(r"^\s+([A-Za-z_][A-Za-z0-9_\-]*):\s*(.*)$", line)
            if not m:
                continue
            k = m.group(1)
            if not UNKNOWN_DATE_KEY_RE.search(k):
                continue
            if parse_first_date(clean_value(m.group(2))):
                # known key names too: a key under a mapping is never read as a top-level date
                self.unparsed.append([self.repo(rel) + f":{i + 1}",
                                      f"nested date-like key '{k}' under a mapping: not read"])
                hit = True
        return hit

    @staticmethod
    def _no_event_reason(fm: dict, is_yaml: bool) -> str:
        if not fm:
            return "no-frontmatter" if not is_yaml else "yaml-no-top-level-keys"
        if not any(k in fm for k in KNOWN_DATE_KEYS):
            return "no-date-key"
        return "date-key-null-or-unparsed"

    def scan_experiments(self) -> None:
        d = self.lab / EXP_DIR
        if not d.is_dir():
            return self._missing(EXP_DIR, "experiment docs dir")
        for p in sorted(d.iterdir()):
            if not p.is_file() or p.name.startswith("_") or p.name == "README.md":
                continue
            if p.suffix not in (".md", ".yaml"):
                continue
            rel = f"{EXP_DIR}/{p.name}"
            if p.name.endswith("FREEZE_WITNESS.md"):
                self.scan_witness(rel, self.read("experiments", rel))
            else:
                self.fm_events("experiments", rel, lane="experiment", fallback=True)

    def scan_reports(self) -> None:
        d = self.lab / REPORT_DIR
        if not d.is_dir():
            return self._missing(REPORT_DIR, "reports dir")
        for sub in sorted(d.iterdir()):
            if sub.name == OUTPUT_DIR_NAME:  # never ingest this extractor's own output
                continue
            nd = name_date(sub.name) if sub.is_dir() else None
            if nd is not None:
                rel = f"{REPORT_DIR}/{sub.name}"
                self.groups.setdefault("reports", set()).add(self.repo(rel))
                self.add(etype=type_for_name(sub.name, "plan"), date=nd, src_rel=rel, line=None,
                         title=sub.name, token=None, quote=None, id_text=sub.name,
                         basis="filename", lane=None,
                         notes=["directory-level event from dir name"])
        for p in sorted(d.rglob("*")):
            if not p.is_file() or p.suffix not in (".md", ".yaml") or p.name == "README.md":
                continue
            if any(part.startswith("_") for part in p.relative_to(d).parts):
                continue
            if p.relative_to(d).parts[0] == OUTPUT_DIR_NAME:
                continue
            rel = f"{REPORT_DIR}/{p.relative_to(d).as_posix()}"
            self.fm_events("reports", rel, lane=None, fallback=True)

    def scan_theory(self) -> None:
        d = self.lab / THEORY_DIR
        if not d.is_dir():
            return self._missing(THEORY_DIR, "theory dir")
        for p in sorted(d.iterdir()):
            if not re.fullmatch(r"H-\d+(\.status)?\.md", p.name):
                continue
            self.fm_events("theory", f"{THEORY_DIR}/{p.name}", lane="hypothesis", fallback=False)

    def scan_atoms(self) -> None:
        if not (self.lab / ATOMS).is_file():
            return self._missing(ATOMS, "atom INDEX")
        raw_lines = self.read("atoms", ATOMS)
        for i, raw in enumerate(raw_lines):
            if not raw.strip():
                continue
            row = json.loads(raw)
            dt = parse_first_date(str(row.get("created", "")))
            if dt is None:
                self.unparsed.append([self.repo(ATOMS) + f":{i + 1}", "atom created: no ISO date"])
                continue
            verdict = row.get("verdict")
            doc_status = None
            quote = cap(raw)
            if isinstance(verdict, str) and verdict:
                cand = f'"verdict": "{verdict}"'
                if cand in raw:
                    doc_status, quote = verdict, cand
            id_text = " ".join([str(row.get("atom_id", ""))] + [str(x) for x in row.get("exp_ids") or []]
                               + [str(x) for x in row.get("hypothesis_ids") or []])
            self.add(etype="register", date=dt, src_rel=ATOMS, line=i + 1,
                     title=str(row.get("atom_id")), token=None, quote=quote, id_text=id_text,
                     basis="registry-field", lane=None,
                     angle=[[mm.group(0), i + 1] for mm in ANGLE_RE.finditer(raw)],
                     doc_status=doc_status,
                     notes=["date is atom created; verdict (doc_status_at_read) has no date in INDEX"]
                     if doc_status else None)

    def scan_autoresearch(self) -> None:
        for dname in AR_DIRS:
            d = self.lab / dname
            if not d.is_dir():
                self.unparsed.append([self.repo(dname), "autoresearch dir missing"])
                continue
            nd = name_date(d.name)
            if nd is None:
                self.unparsed.append([self.repo(dname), "no date in dir name"])
            else:
                self.add(etype="plan", date=nd, src_rel=dname, line=None, title=d.name,
                         token=None, quote=None, id_text=d.name, basis="filename",
                         lane="autoresearch", notes=["directory-level event from dir name"])
            self.groups.setdefault("autoresearch", set()).add(self.repo(dname))
            for p in sorted(d.rglob("*")):
                if not p.is_file() or p.suffix not in (".md", ".yaml"):
                    continue
                rel = p.relative_to(self.lab).as_posix()
                self.fm_events("autoresearch", rel, lane="autoresearch", fallback=False,
                               mtime=nd is None)
        self._scan_goal_ledger()
        self._scan_autoresearch_md()
        self._scan_handoff_index()

    def _scan_goal_ledger(self) -> None:
        if not (self.lab / GOAL_LEDGER).is_file():
            return self._missing(GOAL_LEDGER, "goal ledger")
        lines = self.read("autoresearch", GOAL_LEDGER)
        goal = "G-20260902-autoresearch-rls"
        for i, line in enumerate(lines):
            if line.startswith("  created_at:") or line.startswith("created_at:"):
                dt = parse_first_date(line.split(":", 1)[1])
                if dt:
                    self.add(etype="plan", date=dt, src_rel=GOAL_LEDGER, line=i + 1, title=goal,
                             token=None, quote=cap(line, must=dt[0]), id_text=goal,
                             basis="in-doc-timestamp", lane="autoresearch")
        item = None
        for i, line in enumerate(lines):
            m = re.match(r"^- item_id: (\S+)", line)
            if m:
                item = m.group(1)
            rm = re.match(r"^\s+recorded_at: '(.+)'\s*$", line)
            if rm and item:
                dt = parse_first_date(rm.group(1))
                if dt:
                    self.add(etype="decision", date=dt, src_rel=GOAL_LEDGER, line=i + 1,
                             title=item, token=None, quote=cap(line, must=dt[0]),
                             id_text=f"{goal} {item}", basis="in-doc-timestamp",
                             lane="autoresearch", notes=["ledger item recorded_at"])

    def _scan_autoresearch_md(self) -> None:
        if not (self.lab / AUTORESEARCH_MD).is_file():
            return self._missing(AUTORESEARCH_MD, "AUTORESEARCH.md")
        lines = self.read("autoresearch", AUTORESEARCH_MD)
        for i, line in enumerate(lines):
            m = re.match(r"^> \*\*Rewritten (\d{4}-\d{2}-\d{2})\*\*", line)
            if m:
                dt = parse_first_date(m.group(1))
                self.add(etype="decision", date=dt, src_rel=AUTORESEARCH_MD, line=i + 1,
                         title="Rewritten", token=None, quote=cap(line, must=m.group(1)),
                         id_text=line, basis="in-doc-timestamp", lane="autoresearch",
                         angle=self.angle_hits(lines, i, i + 1))

    def _scan_handoff_index(self) -> None:
        if not (self.lab / HANDOFF_INDEX).is_file():
            return self._missing(HANDOFF_INDEX, "handoff INDEX")
        lines = self.read("autoresearch", HANDOFF_INDEX)
        for i, line in enumerate(lines):
            other = re.match(r"^- `\[([A-Z]+)\]` \[\d{4}-\d{2}-\d{2} \d{2}:\d{2}\]", line)
            if other and other.group(1) not in ("ARG", "ARB"):
                self.unparsed.append([self.repo(HANDOFF_INDEX) + f":{i + 1}",
                                      f"handoff row tag [{other.group(1)}] outside ARG/ARB scope: not extracted"])
                continue
            m = re.match(r"^- `\[(ARG|ARB)\]` \[(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\] \*\*([^*]+)\*\*", line)
            if not m:
                continue
            dt = parse_first_date(m.group(2))
            if dt is None:
                self.unparsed.append([self.repo(HANDOFF_INDEX) + f":{i + 1}", "handoff time unparsed"])
                continue
            tag = m.group(1)
            self.add(etype="handoff", date=dt, src_rel=HANDOFF_INDEX, line=i + 1,
                     title=f"[{tag}]", token=None, doc_status=m.group(3).strip(),
                     quote=cap(line, must=m.group(3)),
                     id_text=line, basis="in-doc-timestamp", lane="autoresearch",
                     angle=self.angle_hits(lines, i, i + 1),
                     notes=["handoff row; tag restricted to ARG/ARB; bold word is doc_status_at_read "
                            "(rows of the same tag are re-marked by later rows; not time-attributed)"])

    def scan_all(self) -> None:
        self.load_withheld()       # fail-closed before anything is emitted
        self.load_aliases()
        self.scan_registry()
        self.scan_hypothesis_log()
        self.scan_experiments()
        self.scan_reports()
        self.scan_theory()
        self.scan_atoms()
        self.scan_autoresearch()
        self.apply_withheld()

    def apply_withheld(self) -> None:
        active = [f for f in self.families if f["status"] == "withheld"]
        for ev in self.events + self.copy_events:
            text = normalize_text(" ".join(filter(None, [ev["title"], ev["quote"], ev["source_path"]])))
            hit = False
            for fam in active:
                if any(p.search(text) for p in fam["patterns"]):
                    hit = True
                for n in fam["exp_numbers"]:
                    if f"EXP-{n:03d}" in ev["ids"]["exp"]:
                        hit = True
                    if re.search(rf"(?<![A-Za-z0-9])EXP[-_ ]?0*{n}(?![0-9])", text, re.IGNORECASE):
                        hit = True
            if hit:
                ev["withheld"] = True
                ev["title"] = None
                ev["quote"] = None
                ev["status_token"] = None
                ev["doc_status_at_read"] = None


# ---------------------------------------------------------------- output

def _norm_key(ev: dict) -> str:
    return parse_first_date(ev["date"])[1]


def _sort(events: list[dict]) -> list[dict]:
    return sorted(events, key=lambda e: (_norm_key(e), e["event_id"]))


def _write_jsonl(path: Path, events: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for ev in events:
            fh.write(json.dumps(ev, sort_keys=True, ensure_ascii=False) + "\n")


def build(lab: Path, out_dir: Path) -> dict:
    tl = Timeline(lab)
    tl.scan_all()
    events = _sort(tl.events)
    copies = _sort(tl.copy_events)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "timeline_events.jsonl", events)
    _write_jsonl(out_dir / "timeline_copy_events.jsonl", copies)
    stats = make_stats(tl, events, copies)
    (out_dir / "timeline_stats.json").write_text(
        json.dumps(stats, sort_keys=True, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    no_event = sorted(p for p in tl.read_hash if tl.file_events.get(p, 0) == 0)
    (out_dir / "timeline_no_event_files.txt").write_text(
        "".join(f"{p}\t{tl.file_reason.get(p, 'no-event-rule')}\t{tl.file_group.get(p, '')}\n"
                for p in no_event), encoding="utf-8")
    copy_files = sorted({e["source_path"] for e in copies})
    lines = []
    for p in copy_files:
        reason = next(e["copy_reason"] for e in copies if e["source_path"] == p)
        lines.append(f"{p}\t{reason}\t{sum(1 for e in copies if e['source_path'] == p)}")
    (out_dir / "timeline_copy_files.txt").write_text("".join(l + "\n" for l in lines), encoding="utf-8")
    return stats


def _dup_groups(events: list[dict]) -> list[dict]:
    groups: dict[tuple, int] = {}
    for e in events:
        key = (e["type"], e["ids"]["exp"][0] if e["ids"]["exp"] else "-", e["date"][:10])
        groups[key] = groups.get(key, 0) + 1
    return [{"type": t, "exp": x, "date": d, "n": n}
            for (t, x, d), n in sorted(groups.items()) if n >= 2]


def make_stats(tl: Timeline, events: list[dict], copies: list[dict]) -> dict:
    counts = lambda evs, key: dict(sorted(_count(e[key] for e in evs).items()))  # noqa: E731
    norms = [_norm_key(e) for e in events]
    all_sources = sorted({p for s in tl.groups.values() for p in s})
    read_files = sorted(tl.read_hash)
    with_event = [p for p in read_files if tl.file_events.get(p, 0) > 0]
    no_event = [p for p in read_files if tl.file_events.get(p, 0) == 0]
    group_rows: dict[str, dict] = {}
    for p in read_files:
        g = tl.file_group.get(p, "?")
        row = group_rows.setdefault(g, {"read": 0, "with_event": 0, "no_event": 0})
        row["read"] += 1
        row["with_event" if tl.file_events.get(p, 0) else "no_event"] += 1
    copy_files = sorted({e["source_path"] for e in copies})
    # false-positive candidates for human review: copy-class files whose NAME looks canonical
    fp = [p for p in copy_files
          if re.search(r"EXP-\d{3}|VERIFY|RESULTS|FREEZE|WITNESS|PREREG|DECISION", p.rsplit("/", 1)[-1])]
    mixed: dict = {}
    for e in events:
        key = (e["ids"]["exp"][0] if e["ids"]["exp"] else "-", e["date"][:10])
        mixed.setdefault(key, set()).add(e["date_precision"] == "day")
    same_day_mixed = sum(1 for v in mixed.values() if v == {True, False})
    tree_hash = hashlib.sha256(
        "\n".join(f"{p}\t{tl.read_hash[p]}" for p in read_files).encode("utf-8")).hexdigest()
    return {
        "same_day_mixed_precision_groups": same_day_mixed,
        "events_total": len(events),
        "copy_events_total": len(copies),
        "copy_files_total": len(copy_files),
        "copy_fp_candidates": fp,
        "by_type": counts(events, "type"),
        "by_lane": counts(events, "lane"),
        "by_date_basis": counts(events, "date_basis"),
        "by_date_tz": counts(events, "date_tz"),
        "copy_by_reason": counts(copies, "copy_reason"),
        "mtime_basis_events": sum(1 for e in events if e["date_basis"] == "mtime"),
        "withheld_events": sum(1 for e in events if e["withheld"]),
        "events_with_status_token": sum(1 for e in events if e["status_token"] is not None),
        "events_with_doc_status_at_read": sum(1 for e in events if e["doc_status_at_read"] is not None),
        "events_with_ids_ambiguous": sum(1 for e in events if e["ids_ambiguous"]),
        "ids_ambiguous_raw_tokens_total": tl.ambiguous_tokens,
        "events_with_alias_conversion": sum(1 for e in events if e["aliases_seen"]),
        "events_with_angle_tokens": sum(1 for e in events if e["angle_tokens_in_doc"]),
        "canonical_duplicate_groups_type_exp_date": _dup_groups(events),
        "unparsed": sorted(tl.unparsed),
        "unparsed_count": len(tl.unparsed),
        "files_read_total": len(read_files),
        "files_with_event_total": len(with_event),
        "files_no_event_total": len(no_event),
        "no_event_reason_counts": counts([{"r": tl.file_reason.get(p, "no-event-rule")} for p in no_event], "r"),
        "per_group_files": dict(sorted(group_rows.items())),
        "source_files_by_group": {g: len(v) for g, v in sorted(tl.groups.items())},
        "source_files_total": len(all_sources),
        "date_range_min": min(norms) if norms else None,
        "date_range_max": max(norms) if norms else None,
        "source_list_sha256": hashlib.sha256("\n".join(all_sources).encode("utf-8")).hexdigest(),
        "source_tree_hash": tree_hash,
        "registry_exp_headings": len(tl.reg_exp),
    }


def _count(values: Iterable) -> dict:
    out: dict = {}
    for v in values:
        key = "null" if v is None else str(v)
        out[key] = out.get(key, 0) + 1
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lab-root", type=Path, default=LAB_DEFAULT)
    ap.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args(argv)
    stats = build(args.lab_root, args.out_dir)
    print(f"events_total={stats['events_total']} copy_events_total={stats['copy_events_total']} "
          f"unparsed={stats['unparsed_count']} tree={stats['source_tree_hash'][:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
