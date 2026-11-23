#!/usr/bin/env python3
"""Deterministic post-check of run-card prose returned by the runcard batch lane (no LLM, exit-code gated).

  uv run --no-sync --with pyyaml python tools/wandb_run_index/prose_check.py <batch_dir> [--lab-root lab]

Reads <batch_dir>/{plan.json,manifest.jsonl,items/*.md}, re-reads each card (current bytes), and prints the verdicts; with --write it writes
<batch_dir>/postcheck.json = {accepted, quarantined, missing} (or to --report PATH); by default nothing is written into the batch dir. Nothing is injected here (prose_inject.py does that,
and re-runs this check itself). Exit: 0 all requests accepted; 3 the check RAN and some outputs are quarantined/missing; 2 ERROR (could not run: bad
plan/manifest, missing run index, any exception).

Rules (every failure is a reason string; one reason quarantines the output):
  R1 shape       exactly the schema fields, run_id == card id, size limits, no HTML-comment markers; every character of every
                 field is printable ASCII (or newline) after NFKC; no '<' '>' / HTML entities / markdown structure
  R2 numbers     each what_the_numbers_show entry is {key, stat, value, step} and value/step equal the stats-table
                 cell character for character; numeric literals in free text are whole-token members of the card's
                 numeric tokens (numbers_summary: of the entries' value/step plus the frontmatter's); no '%',
                 no comma-grouped numbers, no rounding words / derived-quantity words
  R3 ids/paths   identifier-like tokens (letters+digits, dates) and path-like tokens occur verbatim in the card
  R4 urls        no URL unless link.tier == api-confirmed and it equals link.url
  R5 forbidden   verdict / causal / judgement / hypothesis words, run-status words (crash, finish, final, ended ... allowed only
                 inside the script-supplied sentences or as the card's literal status value), EXP/H ids outside see_also
                 (exact card tokens such as `api-confirmed` are exempt). Word lists: prose_forbidden.txt
  (R2 also rejects spelled-out numbers and digits glued to letters that the card does not print: 244k, 3x, 3rd)
  (attribution is one of three fixed sentences, Card.attribution_sentence; prose_inject prints the ids itself)
  R6 completeness caveat kinds required by the card state (status != ended, no metadata, history_truncated, snapshot, multi_attribution, legacy cohort, duplicates,
                 link tier != api-confirmed); attribution says DRAFT when the card lists exps/hyps
  R7 provenance  item result_status succeeded, model_returned == model_requested, l5 pass, card bytes unchanged
                 since plan (blanked sha256 == manifest meta)
It cannot detect a wrong-but-present number placed under the wrong key, or prose that is misleading with true tokens;
it only enforces that every token comes from the card.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.api_batch import runcard as RC  # noqa: E402
from tools.api_batch.common import sha256_text  # noqa: E402

FIELDS = ("run_id", "what_this_run_is", "attribution", "numbers_summary", "what_the_numbers_show", "caveats", "see_also")
KINDS = ("status", "no_metadata", "truncated_history", "snapshot", "multi_attribution", "legacy_cohort", "duplicate_run_name",
         "link_tier", "other")
MAX_STR, MAX_SUMMARY, MAX_ENTRIES, MAX_CAVEATS, MAX_SEEALSO = 600, 700, 12, 10, 12

URL_RE = re.compile(r"https?://[^\s)\]>\"'`]+")
PATH_RE = re.compile(r"[\w.\-]+(?:/[\w.\-]+)+")
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}(?:T[\d:.]+Z?)?")
IDENT_RE = re.compile(r"(?<![\w.])[A-Za-z_][\w\-]*\d[\w\-]*")
NUM_RE = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?![\w]|\.\d)")
BAD_NUM_FORM = re.compile(r"\d\s*%|\d,\d{3}\b")
ARITH_RE = re.compile(r"%|\bpercent|\bfold\b|\bratio\b|\bdifference\b|\bdelta\b|\b(?:roughly|approximately|about|around|nearly|"
                      r"almost|over|under)\s+-?\d|\bapprox\w*|\bca\.|~\s*\d|\d\s*[x×]\b|\bhalf\b|\bdoubl|\btripl|\btwice\b", re.I)
FORBIDDEN_FILE = Path(__file__).with_name("prose_forbidden.txt")


def load_sections(path: Path = FORBIDDEN_FILE) -> dict[str, list[str]]:
    """section -> lines of the maintained list file (en / ko / exempt / status / numwords); validates every regex."""
    sec, out = None, {k: [] for k in ("en", "ko", "exempt", "status", "numwords")}
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        if ln.startswith("[") and ln.endswith("]"):
            sec = ln[1:-1]
            if sec not in out:
                raise ValueError(f"{path.name}: unknown section {ln}")
            continue
        if sec is None:
            raise ValueError(f"{path.name}: line before any section: {ln!r}")
        if sec in ("en", "status", "numwords"):
            re.compile(ln)
        out[sec].append(ln)
    if not (out["en"] and out["status"] and out["numwords"]):
        raise ValueError(f"{path.name}: empty en / status / numwords section")
    return out


def _alt(pats: list[str]) -> re.Pattern:
    return re.compile("|".join(f"(?:{p})" for p in pats), re.I)


def load_forbidden(path: Path = FORBIDDEN_FILE) -> tuple[re.Pattern, tuple[str, ...]]:
    """(compiled alternation of the [en] regexes + [ko] literals, exempt literal phrases) from the maintained list file."""
    sec = load_sections(path)
    return _alt(sec["en"] + [re.escape(k) for k in sec["ko"]]), tuple(sec["exempt"])


FORBIDDEN_RE, EXEMPT_PHRASES = load_forbidden()
_SECS = load_sections()
STATUS_RE, NUMWORD_RE = _alt(_SECS["status"]), _alt(_SECS["numwords"])
GLUE_RE = re.compile(r"(?<![\w.])\d[\d.]*[A-Za-z%][\w.%]*")          # digits glued to letters / %: 244k, 3x, 10M, 3rd
EXPH_RE = re.compile(r"\bEXP-?\d+|\bH-?\d", re.I)                       # experiment / hypothesis ids (see_also only)
DOMAIN_RE = re.compile(r"(?<![\w.@-])[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}(?![\w-])")      # a.b.tld / file.ext
ENTITY_RE = re.compile(r"&#?\w+;|[<>]")
MD_RE = re.compile(r"```|^\s*(?:#|---)|\]\(|!\[", re.M)
SEE_RE = re.compile(r"[\w./\-]{1,120}")
ID_RE = re.compile(r"[A-Za-z0-9_]+")
INDEX_REL = "21_experiments/11_lab-experiments/_wandb_runs_index.jsonl"
SLASH_OK = frozenset({"min/max", "max/min", "first/last", "last/first", "H/L", "L/H", "and/or"})   # ordinary word pairs


# ------------------------------------------------------------------------------------------------ rule sets
# A request is judged by the rules its prompt told the model. `meta.rules` of a manifest row names the rule set; rows that
# predate it are mapped through the prompt's sha256 (meta.prompt_sha256 of the AS-SENT manifest). Unknown prompt = error.
#   r4        status wording: only the script sentences or exactly "status <card status value>"
#   r4-asent  the prompt of manifest 015462f8d43a (run-cards-pilot, submitted 2026-11-10T13:12Z) said "the card's `status` value
#             quoted exactly as printed": the literal value is also accepted after "status [value|field|is|was|of|reads|...]"
#             or in backticks / quotes; and an entry `step` written with the table's leading "@" ("@292", the prompt said
#             "copied character for character" from `value (@step)`) equals the cell step (prose_inject renders it without the "@").
#             Everything else is identical to r4.
RULE_SETS = {"r4": {"status_loose": False, "step_at": False}, "r4-asent": {"status_loose": True, "step_at": True}}
PROMPT_RULESET = {
    "b4acc404f8790d79920ffc66a96eafd594b9a501feae3d36e83afb580ae8118d": "r4-asent",   # as-sent pilot prompt (015462f8d43a)
    "4591bd3ce7f1a86c914a5df89873f246ce3a7e558dfeaabebd495afefa5e5637": "r4",          # prompt after the 13:14 tweak (84fdb3fda9bd)
}
CURRENT_RULES = "r4"


def rules_of(meta: dict) -> str:
    r = meta.get("rules") or PROMPT_RULESET.get(meta.get("prompt_sha256"))
    if r not in RULE_SETS:
        raise ValueError(f"unknown rule set for request {meta.get('run_id')}: meta.rules={meta.get('rules')!r}, "
                         f"prompt_sha256={str(meta.get('prompt_sha256'))[:12]} (add it to PROMPT_RULESET deliberately)")
    return r


# ------------------------------------------------------------------------------------------------ card view
class Card:
    def __init__(self, text: str, rules: str = CURRENT_RULES):
        self.rules = rules
        self.status_loose = RULE_SETS[rules]["status_loose"]
        self.step_at = RULE_SETS[rules]["step_at"]
        self.text = RC.blanked_text(text)
        self.fm, _ = RC.split_frontmatter(self.text)
        self.stats = RC.stats_table(self.text)
        end = self.text.index("\n---\n", 4)
        self.fm_text = self.text[4:end]
        # whole-token numbers via the same tokenizer as the model side (hex digests, commit shas, paths and dates are
        # removed first, so they cannot contribute stray integer fragments)
        self.nums = set(_tokens(self.text)["num"])
        self.fm_nums = set(_tokens(self.fm_text)["num"])
        self.tier = (self.fm.get("link") or {}).get("tier")
        self.url = (self.fm.get("link") or {}).get("url")
        self.tokens = set(ID_RE.findall(self.text))
        self.special = self.fm.get("history_truncated") is True or self.fm.get("snapshot") is True

    def required_sentences(self) -> dict[str, str]:
        """kind -> the exact sentence a caveat of that kind must contain (script-built from the card's own fields)."""
        fm, out = self.fm, {}
        if fm.get("status") != "ended":
            out["status"] = f"The card's status value is {fm.get('status')}."
        if fm.get("history_truncated") is True:
            out["truncated_history"] = (f"Local history ends at step {fm.get('history_last_step')}, before the summary step "
                                        f"{fm.get('summary_last_step')}; the last values quoted here are not the run's final step.")
        if fm.get("snapshot") is True:
            out["snapshot"] = (f"Partial snapshot taken at {fm.get('snapshot_utc')}; the run may still have been in progress, "
                               "so these values are not final.")
        if fm.get("multi_attribution") is True:
            out["multi_attribution"] = ("The attribution map assigns this run to several experiments ("
                                        + ", ".join(str(e) for e in (fm.get("attribution") or {}).get("exps") or [])
                                        + "); this is unresolved here.")
        return out

    def attribution_sentence(self) -> str:
        """The one attribution sentence the model must copy (the script prints the ids / confidences at inject time)."""
        fm = self.fm
        at = fm.get("attribution") or {}
        if not (at.get("exps") or at.get("hyps")):
            return "The card lists no experiment or hypothesis attribution."
        if fm.get("multi_attribution") is True:
            return ("DRAFT attribution map, not evidence; the map lists several experiments and is unresolved here; "
                    "the script prints the listed ids below.")
        return "DRAFT attribution map, not evidence; the script prints the listed ids below."

    def required_caveats(self) -> set[str]:
        fm, req = self.fm, set()
        if fm.get("status") != "ended":
            req.add("status")
        if fm.get("has_metadata") is False:
            req.add("no_metadata")
        if fm.get("history_truncated") is True:
            req.add("truncated_history")
        if fm.get("snapshot") is True:
            req.add("snapshot")
        if fm.get("multi_attribution") is True:
            req.add("multi_attribution")
        if fm.get("cohort") == "legacy":
            req.add("legacy_cohort")
        if fm.get("duplicates"):
            req.add("duplicate_run_name")
        if self.tier != "api-confirmed":
            req.add("link_tier")
        return req


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _tokens(s: str) -> dict:
    """Split a free-text string into url / path / date / ident / number tokens (each removed before the next kind)."""
    out = {}
    out["url"] = [u.rstrip(".,;:") for u in URL_RE.findall(s)]
    s = URL_RE.sub(" ", s)
    out["path"] = PATH_RE.findall(s)
    s = PATH_RE.sub(" ", s)
    out["date"] = DATE_RE.findall(s)
    s = DATE_RE.sub(" ", s)
    out["ident"] = IDENT_RE.findall(s)
    s = IDENT_RE.sub(" ", s)
    out["num"] = NUM_RE.findall(s)
    out["rest"] = s
    return out


_CF: dict[int, set] = {}


def _idcf(known) -> set:
    if id(known[0]) not in _CF:
        _CF[id(known[0])] = {i.casefold() for i in known[0]}
    return _CF[id(known[0])]


def check_text(label: str, s: str, card: Card, allowed_nums: set[str], fails: list[str], known=None) -> None:
    if len(s) > (MAX_SUMMARY if label == "numbers_summary" else MAX_STR):
        fails.append(f"R1 {label}: {len(s)} chars exceeds the limit")
    if "<!--" in s or "-->" in s:
        fails.append(f"R1 {label}: contains an HTML comment marker")
    t = _tokens(s)
    for u in t["url"]:
        if not (card.tier == "api-confirmed" and u == card.url):
            fails.append(f"R4 {label}: URL {u!r} (link tier {card.tier!r}; only the api-confirmed card URL is allowed)")
    for k in ("path", "date", "ident"):
        for x in t[k]:
            if k == "path":
                x = x.rstrip(".,;:")                        # a sentence period glued to a path is punctuation
                if x in SLASH_OK:
                    continue
            if x not in card.text:
                fails.append(f"R3 {label}: {k} token {x!r} does not occur in the card")
    for pat in ("://", "www.", "mailto:"):
        if pat in s.lower() and not (pat == "://" and card.tier == "api-confirmed" and card.url and card.url in s):
            fails.append(f"R4 {label}: link-like text {pat!r}")
    s_dom = s
    for e in EXEMPT_PHRASES:                                    # label phrases the cards print themselves (e.g. "...against pretrain.py")
        s_dom = s_dom.replace(e, " ")
    for dm in DOMAIN_RE.findall(s_dom):
        if dm not in card.text:
            fails.append(f"R4 {label}: dotted domain / file name {dm!r} is not printed in the card")
    if BAD_NUM_FORM.search(s):
        fails.append(f"R2 {label}: percent or comma-grouped number")
    if ARITH_RE.search(s):
        fails.append(f"R2 {label}: rounding / derived-quantity wording ({ARITH_RE.search(s).group(0)!r})")
    for n in t["num"]:
        if n not in allowed_nums:
            fails.append(f"R2 {label}: number {n!r} is not an allowed token for this field")
    # foreign run ids / run names (R3), compared case-insensitively: any token equal to a known id or name that this card
    # does not print exactly; plus any 8-char letter+digit token that the card does not print
    toks = ID_RE.findall(s)
    for tk in toks:
        if (known and tk.casefold() in _idcf(known) and tk not in card.tokens) or (
                len(tk) == 8 and any(c.isdigit() for c in tk) and any(c.isalpha() for c in tk) and tk not in card.tokens):
            fails.append(f"R3 {label}: run id {tk!r} does not belong to this card")
    if known:
        scf = s.casefold()
        for nm in known[1]:
            if len(nm) >= 6 and nm.casefold() in scf and nm not in card.text and re.search(
                    r"(?<![\w])" + re.escape(nm) + r"(?![\w])", s, re.I):
                fails.append(f"R3 {label}: run name {nm!r} does not belong to this card")
    # digits glued to letters / spelled-out numbers (R2)
    for g in GLUE_RE.findall(NUM_RE.sub(" ", t["rest"])):
        g = g.rstrip(".")
        if not re.search(r"(?<![\w.])" + re.escape(g) + r"(?![\w])", card.text):
            fails.append(f"R2 {label}: digits glued to letters {g!r} are not printed in the card")
    m = NUMWORD_RE.search(s)
    if m:
        fails.append(f"R2 {label}: spelled-out number {m.group(0)!r} (use digits printed in the card)")
    if MD_RE.search(s):
        fails.append(f"R1 {label}: markdown structure (heading / rule / fence / link)")
    masked = s
    for e in EXEMPT_PHRASES:
        masked = masked.replace(e, " ")
    if card.tier != "api-confirmed":
        masked = masked.replace("not confirmed", " ")
    for sent in card.required_sentences().values():           # the script-supplied sentences are exempt from the word scans
        masked = masked.replace(sent, " ")
    for own in (card.fm.get("name"), card.fm.get("project")):   # the card's own run name / project are quoted verbatim, never scanned as prose
        if own:
            masked = masked.replace(str(own), " ")
    stv = str(card.fm.get("status") or "")
    if stv:                                                    # the card's literal status value, quoted as "status <value>", is the only status wording
        if card.status_loose:
            masked = re.sub(r"\bstatus\b(?:\s+(?:value|field|is|was|of|reads|listed as|recorded as|=|:))*\s*[`\"']?" + re.escape(stv)
                            + r"[`\"']?(?![\w-])|[`\"']" + re.escape(stv) + r"[`\"']", " ", masked, flags=re.I)
        else:
            masked = re.sub(r"\bstatus(?: value)?\s*(?:is|of|:|=)?\s*[`\"']?" + re.escape(stv) + r"[`\"']?(?![\w-])", " ", masked, flags=re.I)
    m = FORBIDDEN_RE.search(masked)
    if m:
        fails.append(f"R5 {label}: forbidden word {m.group(0)!r}")
    m = STATUS_RE.search(masked)
    if m:
        fails.append(f"R5 {label}: run-status wording {m.group(0)!r} (allowed only in the script sentences or as the card's status value)")
    m = EXPH_RE.search(masked)
    if m:
        fails.append(f"R5 {label}: experiment / hypothesis id {m.group(0)!r} outside see_also")


def bad_chars(s: str) -> str:
    """Characters that are not printable ASCII (or newline) after NFKC, plus angle brackets / HTML entities."""
    n = unicodedata.normalize("NFKC", s)
    bad = {c for c in s + n if not (0x20 <= ord(c) <= 0x7E or c == "\n")}
    return "".join(sorted(bad))


def check_output(card_text: str, out, known=None, rules: str = CURRENT_RULES) -> list[str]:
    """All failure reasons for one parsed model output against one card (empty list = accepted)."""
    fails: list[str] = []
    if not isinstance(out, dict) or set(out) != set(FIELDS):
        return [f"R1 fields are not exactly {list(FIELDS)}"]
    try:
        card = Card(card_text, rules)
    except RC.RunCardError as e:
        return [f"R1 card unreadable: {e}"]
    for f in ("run_id", "what_this_run_is", "attribution", "numbers_summary"):
        if not isinstance(out[f], str) or not out[f].strip():
            fails.append(f"R1 {f}: must be a non-empty string")
    for f, mx in (("what_the_numbers_show", MAX_ENTRIES), ("caveats", MAX_CAVEATS), ("see_also", MAX_SEEALSO)):
        if not isinstance(out[f], list):
            return fails + [f"R1 {f}: must be a list"]
        if len(out[f]) > mx:
            fails.append(f"R1 {f}: {len(out[f])} items > {mx}")
    if fails:
        return fails
    # character allow-list over EVERY string of the output (before any whitespace folding)
    strs = [("run_id", out["run_id"]), ("what_this_run_is", out["what_this_run_is"]), ("attribution", out["attribution"]),
            ("numbers_summary", out["numbers_summary"])]
    strs += [(f"what_the_numbers_show[{i}].{k}", v) for i, e in enumerate(out["what_the_numbers_show"]) if isinstance(e, dict)
             for k, v in e.items() if isinstance(v, str)]
    strs += [(f"caveats[{i}].{k}", v) for i, c in enumerate(out["caveats"]) if isinstance(c, dict) for k, v in c.items() if isinstance(v, str)]
    strs += [(f"see_also[{i}]", v) for i, v in enumerate(out["see_also"]) if isinstance(v, str)]
    for lab, v in strs:
        bc = bad_chars(v)
        if bc:
            fails.append(f"R1 {lab}: characters outside printable ASCII {bc[:8]!r} (invisible, look-alike, non-English)")
        if ENTITY_RE.search(v):
            fails.append(f"R1 {lab}: angle bracket or HTML entity")
    if fails:
        return fails
    if out["run_id"] != str(card.fm.get("id")):
        fails.append(f"R1 run_id {out['run_id']!r} != card id {card.fm.get('id')!r}")
    # R2 structured numbers
    entry_nums, seen = set(), set()
    if not out["what_the_numbers_show"]:
        fails.append("R2 what_the_numbers_show is empty")
    for i, e in enumerate(out["what_the_numbers_show"]):
        lab = f"what_the_numbers_show[{i}]"
        if not isinstance(e, dict) or set(e) != {"key", "stat", "value", "step"} or not all(isinstance(v, str) for v in e.values()):
            fails.append(f"R1 {lab}: not {{key, stat, value, step}} strings")
            continue
        row = card.stats.get(e["key"])
        if row is None:
            fails.append(f"R2 {lab}: key {e['key']!r} is not in the stats table")
            continue
        if e["stat"] not in RC.STATS:
            fails.append(f"R2 {lab}: stat {e['stat']!r} not in {list(RC.STATS)}")
            continue
        if (e["key"], e["stat"]) in seen:
            fails.append(f"R2 {lab}: duplicate ({e['key']}, {e['stat']})")
        seen.add((e["key"], e["stat"]))
        v, st = row[e["stat"]]
        if card.step_at and e["step"] == "@" + st:
            e = {**e, "step": st}                                  # r4-asent only: tolerated leading "@"
        if e["value"] != v or e["step"] != st:
            fails.append(f"R2 {lab}: ({e['value']!r} @{e['step']!r}) != card cell ({v!r} @{st!r}) for {e['key']} {e['stat']}")
        else:
            entry_nums |= {v, st}
    # free text fields
    texts = {"what_this_run_is": out["what_this_run_is"], "numbers_summary": out["numbers_summary"]}
    if norm(out["attribution"]) != card.attribution_sentence():
        fails.append(f"R6 attribution must be, word for word: {card.attribution_sentence()!r}")
    for i, c in enumerate(out["caveats"]):
        if not isinstance(c, dict) or set(c) != {"kind", "text"} or not isinstance(c["text"], str) or c["kind"] not in KINDS:
            fails.append(f"R1 caveats[{i}]: not {{kind in {list(KINDS)}, text}}")
            continue
        texts[f"caveats[{i}]"] = c["text"]
    for i, s in enumerate(out["see_also"]):
        if not isinstance(s, str) or not s.strip():
            fails.append(f"R1 see_also[{i}]: must be a non-empty string")
        elif not SEE_RE.fullmatch(s) or URL_RE.search(s):
            fails.append(f"R3 see_also[{i}]: {s!r} must be one id / file name / path token (no spaces, no URL)")
        elif s not in card.text:
            fails.append(f"R3 see_also[{i}]: {s!r} does not occur in the card")
    for lab, s in texts.items():
        allowed = (entry_nums | card.fm_nums) if lab == "numbers_summary" else card.nums
        check_text(lab, norm(s), card, allowed, fails, known)
    # R6 completeness
    kinds = {c["kind"] for c in out["caveats"] if isinstance(c, dict)}
    for k in sorted(card.required_caveats() - kinds):
        fails.append(f"R6 missing required caveat kind {k!r}")
    by_kind: dict[str, list[str]] = {}
    for c in out["caveats"]:
        if isinstance(c, dict) and isinstance(c.get("text"), str):
            by_kind.setdefault(c.get("kind"), []).append(norm(c["text"]))
    fm = card.fm
    for kind, sent in card.required_sentences().items():
        if not any(sent in t for t in by_kind.get(kind, [])):
            fails.append(f"R6 caveat kind {kind!r} must contain, verbatim: {sent!r}")
    if fm.get("history_truncated") is True and any(isinstance(e, dict) and e.get("stat") == "last" for e in out["what_the_numbers_show"]) \
            and "local history" not in out["numbers_summary"].lower():
        fails.append("R6 truncated card: numbers_summary must say the values come from the 'local history'")
    return fails


# ------------------------------------------------------------------------------------------------ items
def parse_item(path: Path) -> tuple[dict, str]:
    t = path.read_text(encoding="utf-8")
    if not t.startswith("---\n"):
        raise ValueError("item has no frontmatter")
    head, _, body = t[4:].partition("\n---\n")
    fm = {}
    for ln in head.splitlines():
        k, _, v = ln.partition(": ")
        fm[k] = v
    return fm, body


def load_known(lab_root: Path) -> tuple[set[str], set[str]]:
    """(all run ids, all run names) of the wandb run index (every run, not only the cards of this batch). Missing or
    empty index = error (the foreign-id rule must not be silently skipped)."""
    p = Path(lab_root) / INDEX_REL
    if not p.is_file():
        raise ValueError(f"run index {p} not found (needed for the foreign run id/name rule)")
    ids, names = set(), set()
    for ln in p.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            r = json.loads(ln)
            ids.add(str(r["id"]))
            if r.get("name"):
                names.add(str(r["name"]))
    if not ids:
        raise ValueError(f"run index {p} is empty")
    return ids, names


def _ledger_submitted(lab_root: Path) -> list[dict]:
    lp = Path(lab_root) / "38_api-batch" / "ledger.jsonl"
    if not lp.is_file():
        raise ValueError(f"ledger {lp} not found: cannot verify which manifest was sent (fail closed)")
    rows = []
    for ln in lp.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            r = json.loads(ln)
            if r.get("event") == "submitted":
                rows.append(r)
    return rows


def resolve_sent(bd: Path, lab_root: Path, batch_ids=()) -> tuple[list, dict, str, str]:
    """(rows, plan, manifest_sha256, source) of the manifest the batch was actually SENT with. FAILS CLOSED (ValueError -> exit 2):
    the ledger is always consulted; if it has a `submitted` row for this batch dir (canonical path) or for the items' batch ids,
    the record used - <bd>/as_sent/ when present, else the top-level manifest.jsonl / plan.json - must hash to that submitted
    manifest. A top-level manifest that differs from what was submitted (e.g. re-planned after submit) is never used."""
    from tools.api_batch import manifest as M
    from tools.api_batch.common import canon_batch_dir
    bd = Path(bd)
    d = bd / "as_sent" if (bd / "as_sent" / "manifest.jsonl").is_file() else bd
    plan = json.loads((d / "plan.json").read_text(encoding="utf-8"))
    if plan.get("use") != "runcard":
        raise ValueError(f"{bd.name}: plan use is {plan.get('use')!r}, not 'runcard'")
    rows, msha = M.load_manifest(d / "manifest.jsonl")
    if msha != plan["manifest_sha256"]:
        raise ValueError(f"{d.name}/manifest.jsonl does not match its plan.json")
    here = bd.resolve()
    subs = _ledger_submitted(lab_root)
    by_batch = {r.get("batch_id"): r.get("manifest_sha256") for r in subs}
    by_dir = [r for r in subs if r.get("batch_dir") and canon_batch_dir(r["batch_dir"]) == here]
    want = {by_batch.get(b) for b in set(batch_ids)} | ({by_dir[-1].get("manifest_sha256")} if by_dir else set())
    for b in set(batch_ids):
        if b not in by_batch:
            raise ValueError(f"batch {b} has no `submitted` ledger row (fail closed)")
    if len(want) > 1:
        raise ValueError(f"ledger submitted manifests disagree for {bd}: {sorted(x[:12] for x in want if x)}")
    if want and msha not in want:
        raise ValueError(f"{'as_sent' if d is not bd else 'top-level'} manifest {msha[:12]} != ledger submitted manifest "
                         f"{next(iter(want))[:12]}: the sent record is missing or mismatched (fail closed)")
    return rows, plan, msha, ("as_sent" if d is not bd else "top")


def item_batch_ids(bd: Path) -> set:
    return {parse_item(p)[0].get("batch_id") for p in (Path(bd) / "items").glob("*.md")}


def check_batch(batch_dir: Path, lab_root: Path, known=None) -> dict:
    bd = Path(batch_dir)
    known = known or load_known(lab_root)
    rows, plan, msha, src = resolve_sent(bd, lab_root, item_batch_ids(bd))
    res = {"batch_dir": str(bd), "manifest_sha256": msha, "manifest_source": src, "accepted": [], "quarantined": [], "missing": []}
    for r in rows:
        cid, meta = r["custom_id"], r["meta"]
        ip = bd / "items" / f"{cid}.md"
        base = {"custom_id": cid, "run_id": meta["run_id"], "card_rel": meta["card_rel"]}
        if not ip.is_file():
            res["missing"].append({**base, "reason": "no item (not collected, or L5-quarantined)"})
            continue
        fm, body = parse_item(ip)
        fails = []
        if fm.get("result_status") != "succeeded":
            fails.append(f"R7 result_status {fm.get('result_status')!r}")
        if fm.get("l5") != "pass":
            fails.append(f"R7 l5 {fm.get('l5')!r}")
        if fm.get("model_returned") != fm.get("model_requested") or fm.get("model_requested") != r["params"]["model"]:
            fails.append(f"R7 model_returned {fm.get('model_returned')!r} != requested {r['params']['model']!r}")
        cp = Path(lab_root) / meta["card_rel"]
        card_text = cp.read_text(encoding="utf-8") if cp.is_file() else None
        if card_text is None:
            fails.append(f"R7 card {meta['card_rel']} not found")
        else:
            try:
                if RC.card_sha(card_text) != meta["card_sha256"]:
                    fails.append("R7 card changed since plan (blanked sha256 differs); re-plan")
            except RC.RunCardError as e:
                fails.append(f"R7 {e}")
        out = None
        if not fails:
            try:
                out = json.loads(body)
            except Exception:  # noqa: BLE001
                fails.append("R1 item body is not JSON")
        if not fails:
            fails = check_output(card_text, out, known, rules_of(meta))
        if fails:
            res["quarantined"].append({**base, "reasons": fails})
        else:
            res["accepted"].append({**base, "item": f"items/{cid}.md", "batch_id": fm.get("batch_id"),
                                    "model": fm.get("model_returned"), "output": out})
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("batch_dir", type=Path)
    ap.add_argument("--lab-root", type=Path, default=REPO / "lab")
    ap.add_argument("--rerun-ids", type=Path, default=None, metavar="IDS_FILE",
                    help="also write the run ids of quarantined + missing outputs (never the accepted ones) to IDS_FILE, "
                         "ready for a `cards.ids_file` re-run batch")
    ap.add_argument("--write", action="store_true", help="write <batch_dir>/postcheck.json (default: nothing is written into the batch dir)")
    ap.add_argument("--report", type=Path, default=None, metavar="PATH", help="write the full result JSON to PATH instead")
    a = ap.parse_args(argv)
    try:
        res = check_batch(a.batch_dir, a.lab_root)
    except Exception as e:  # noqa: BLE001  any failure to run the check is an ERROR (2), distinct from "ran, some quarantined" (3)
        print(f"prose_check: error: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    if a.rerun_ids:
        ids = sorted({x["run_id"] for x in res["quarantined"] + res["missing"]})
        a.rerun_ids.parent.mkdir(parents=True, exist_ok=True)
        a.rerun_ids.write_text(f"# re-run list from {Path(a.batch_dir).name}: {len(ids)} quarantined/missing ids (accepted ids excluded)\n"
                               + "".join(f"{i}\n" for i in ids), encoding="utf-8")
        print(f"prose_check: wrote {len(ids)} re-run ids to {a.rerun_ids}")
    dumped = json.dumps(res, indent=1, ensure_ascii=False, sort_keys=True)
    if a.report:
        a.report.parent.mkdir(parents=True, exist_ok=True)
        a.report.write_text(dumped, encoding="utf-8")
    if a.write:
        (a.batch_dir / "postcheck.json").write_text(dumped, encoding="utf-8")
    if not (a.report or a.write):
        for q in res["quarantined"]:
            print(f"  quarantined {q['run_id']} {q['custom_id']}: " + " | ".join(r[:160] for r in q["reasons"][:4]))
    print(f"prose_check: accepted {len(res['accepted'])}  quarantined {len(res['quarantined'])}  missing {len(res['missing'])}"
          f"  (sha256 of postcheck inputs: manifest {res['manifest_sha256'][:12]})")
    return 0 if not (res["quarantined"] or res["missing"]) else 3


if __name__ == "__main__":
    sys.exit(main())
