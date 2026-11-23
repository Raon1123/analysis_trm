"""WITHHELD_RUNS.yaml matcher (R5).

Patterns are applied to RUN identifiers only (wandb run ids / run names), never to prose, hypothesis ids
or status cells. Reason: family `h108n` matches the hypothesis id "H-108" itself and family
`verdict-prose` matches "falsified|refuted|C-LTF|verdict"; the registry file says the latter is a
prose family that does not apply to run names. `released` families are history and never redact.
"""
from __future__ import annotations

import re
import unicodedata

import yaml

from frontmatter import ParseError

# Families whose patterns are judgement-prose, not run identifiers (see WITHHELD_RUNS.yaml comment).
PROSE_FAMILY_IDS = {"verdict-prose"}
_DASHES = "‐‑‒–—―−"


def normalise(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    for d in _DASHES:
        s = s.replace(d, "-")
    return s


class Withheld:
    def __init__(self, path):
        self.path = str(path)
        try:
            with open(path, encoding="utf-8") as fh:
                d = yaml.safe_load(fh)
        except (OSError, yaml.YAMLError) as e:
            raise ParseError(path, str(e))
        fams = (d or {}).get("families")
        if not isinstance(fams, list):
            raise ParseError(path, "no families list")
        self.families = []
        self.exp_ids: set[str] = set()
        for f in fams:
            if f.get("status") != "withheld":
                continue
            pats = f.get("patterns") or []
            exps = f.get("exp_ids") or []
            if not pats and not exps:
                raise ParseError(path, f"family {f.get('id')} is withheld with no patterns/exp_ids")
            self.exp_ids.update(exps)
            if f.get("id") in PROSE_FAMILY_IDS:
                continue
            try:
                self.families.append((f["id"], [re.compile(p, re.IGNORECASE) for p in pats]))
            except re.error as e:
                raise ParseError(path, f"bad regex in {f.get('id')}: {e}")

    def match(self, run_ident: str):
        """Family id if the run identifier is withheld, else None."""
        text = normalise(run_ident)
        for fid, regs in self.families:
            if any(r.search(text) for r in regs):
                return fid
        return None

    def set_doc_names(self, names):
        """Identity-based exemption list: basenames of documents the generator actually enumerates."""
        self.doc_names = {normalise(n) for n in names}

    def match_path(self, ref: str):
        """Family id if any component of a path-like string is a withheld run identifier, else None.

        A component is exempt only if it EQUALS the basename of a real document (experiments.dir / theory.cards
        listing, see set_doc_names): the withheld experiment's own document title (EXP-012_clean-r1-...md) is
        not a run identifier. Nothing is exempt by shape, so synthetic `EXP-013_<run>.csv` / `H-108N5.csv` are blocked."""
        exempt = getattr(self, "doc_names", frozenset())
        for comp in re.split(r"[/\\\s,]+", normalise(str(ref))):
            if not comp or comp in exempt:
                continue
            fid = self.match(comp)
            if fid:
                return fid
        return None

    def is_blind_exp(self, exp_id: str) -> bool:
        return exp_id in self.exp_ids


REDACTED = "[열람 보류]"


def redact_text(W, s: str):
    """(text, families) with every withheld-pattern match replaced by REDACTED (R5 at generation, plan 2.4).

    Applied to every string the trace writes, not only run identifiers, so that verbatim cells and
    file names carry no withheld run token. One exception: a match that is only a hypothesis id
    (h108n matching "H-108"/"H108" with no suffix) is kept, because the trace is keyed on hypothesis ids;
    suffixed tokens (H108N2, h108_n1_...) are redacted. Prose families are not applied (module note)."""
    norm, hit, spans = normalise(s), [], []
    for fid, regs in W.families:
        for r in regs:
            for m in r.finditer(norm):
                if fid == "h108n" and re.fullmatch(r"H[-_ ]?108", m.group(0), re.IGNORECASE):
                    continue
                hit.append(fid)
                spans.append(m.span())
    if not hit:
        return s, hit
    # splice into the ORIGINAL string when normalisation kept the length (dash/most NFKC cases), so text
    # outside the token keeps its original characters; otherwise the normalised string is the base.
    base = s if len(norm) == len(s) else norm
    merged = []
    for a, b in sorted(spans):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    out, pos = [], 0
    for a, b in merged:
        out.append(base[pos:a]); out.append(REDACTED); pos = b
    out.append(base[pos:])
    return "".join(out), hit


def redact_tree(W, obj, ptr="", log=None):
    """Copy of a JSON-like tree with redact_text applied to every string value; log gets {pointer, n}."""
    log = [] if log is None else log
    if isinstance(obj, dict):
        return {k: redact_tree(W, v, f"{ptr}/{k}", log) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact_tree(W, v, f"{ptr}/{i}", log) for i, v in enumerate(obj)]
    if isinstance(obj, str):
        out, fams = redact_text(W, obj)
        if fams:
            log.append({"pointer": ptr, "n": len(fams)})  # family ids are not logged: they match the patterns themselves
        return out
    return obj
