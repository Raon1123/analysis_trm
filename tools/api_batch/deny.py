"""Deny-source loader (§5.5). The deny source is user-managed; this module only reads and
validates it. It is the union of a list of YAML files (default: WITHHELD_RUNS.yaml +
API_BATCH_DENY.yaml). Fail-closed: a missing/unreadable/malformed file raises Halt. It never
returns a match-everything matcher (the pack builder must stop instead).

Format = WITHHELD_RUNS.yaml families format (id, status, exp_ids, patterns, note) plus
optional `paths` (glob list, applied to paths only) and `kind`
(locked | sealed-artifact | unblind_pending | withheld).
Validation mirrors trm_mcp.withheld.load_matcher: status in {withheld, released}; a withheld
family with nothing to match is an error; regex compile failure is an error. The verdict-prose
family (PROSE_FAMILY_IDS) is excluded, as tools/hypothesis_trace/withheld.py:17-18 does.
"""
from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .common import Halt, load_trm_mcp, sha256_bytes

ALLOW_LANES = frozenset({"seat-doclint"})        # lanes that may carry an allow_files exception
ALLOW_KEYS = ("path", "sha256", "lanes", "reason", "approved_by", "approved_on")
PROSE_FAMILY_IDS = frozenset({"verdict-prose"})   # pinned by a test against hypothesis_trace
KINDS = ("locked", "sealed-artifact", "unblind_pending", "withheld")
_JOIN_AFTER_HYPHEN = re.compile(r"(?i)(?<![A-Za-z0-9])(EXP)[ \t]*([-_])\s+(?=\d)")
_JOIN_BEFORE_HYPHEN = re.compile(r"(?i)(?<![A-Za-z0-9])(EXP)\s+([-_])(?=\d)")


def prenormalize(text: str) -> str:
    """NB-R3-1: normalize() turns 'EXP-<newline>012' into 'EXP- 012', which the reused id
    pattern (one optional separator) does NOT match. Join such splits after normalize().
    Limit (documented, tested): a split across a paragraph boundary inside one mask_blocks
    block pass is only seen where both halves are in the same block/text handed to matches()."""
    w, _ = load_trm_mcp()
    t = w.normalize(text)
    t = _JOIN_AFTER_HYPHEN.sub(r"\1\2", t)
    return _JOIN_BEFORE_HYPHEN.sub(r"\1\2", t)


class PackMatcher:
    """Duck-types trm_mcp.withheld.Matcher for mask_blocks, with pre-normalization."""
    fail_closed = False

    def __init__(self, patterns: list[re.Pattern]):
        self.patterns = patterns

    def matches(self, text) -> bool:
        if not text:
            return False
        t = prenormalize(text)
        return any(p.search(t) for p in self.patterns)

    def hits(self, text) -> list[str]:
        t = prenormalize(text or "")
        return [p.pattern for p in self.patterns if p.search(t)]


def glob_to_regex(g: str) -> re.Pattern:
    i, out = 0, ""
    while i < len(g):
        if g.startswith("**/", i):
            out += "(?:.*/)?"; i += 3
        elif g.startswith("**", i):
            out += ".*"; i += 2
        elif g[i] == "*":
            out += "[^/]*"; i += 1
        elif g[i] == "?":
            out += "[^/]"; i += 1
        else:
            out += re.escape(g[i]); i += 1
    return re.compile(out + r"\Z", re.I)


@dataclass
class DenySource:
    matcher: PackMatcher
    path_globs: list[tuple[str, re.Pattern]] = field(default_factory=list)
    unblind_pending: list[str] = field(default_factory=list)
    sources: list[tuple[str, str]] = field(default_factory=list)   # (path, sha256)
    allow_files: list[dict] = field(default_factory=list)          # sha-pinned verify-only exceptions

    def allow_entry(self, rel: str, file_sha256: str, lane: str | None) -> tuple[dict | None, str | None]:
        """(entry, why_not). An entry applies only when lane, path AND the current file sha256 all match.
        why_not is set when an entry names this path (or this sha) but is not usable, so the refusal says so."""
        stale = None
        for e in self.allow_files:
            if e["path"] != rel:
                continue
            if lane not in e["lanes"]:
                stale = f"allow_files entry for {rel!r} does not cover lane {lane!r}; refusal kept"
            elif e["sha256"] != file_sha256:
                stale = f"allow_files entry for {rel!r} pins a different sha256 (file changed); refusal kept"
            else:
                return e, None
        return None, stale

    @property
    def sha256(self) -> str:
        return sha256_bytes("\n".join(f"{p}={s}" for p, s in self.sources).encode())

    def path_hit(self, rel: str) -> str | None:
        rel = rel.replace("\\", "/")
        for g, rx in self.path_globs:
            if rx.match(rel) or rx.match(rel.rsplit("/", 1)[-1]):
                return g
        return None


def _as_list(v, what, fid):
    if v is None:
        return []
    if not isinstance(v, list) or not all(isinstance(x, (str, int)) for x in v):
        raise Halt(f"family {fid!r}: {what} must be a list of strings")
    return [str(x) for x in v]


def _parse_allow_files(raw, where: str) -> list[dict]:
    """Validate the optional top-level `allow_files:` list (fail-closed: any defect halts)."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise Halt(f"{where}: allow_files must be a list")
    out = []
    for i, e in enumerate(raw):
        try:
            if not isinstance(e, dict) or set(e) != set(ALLOW_KEYS):
                raise ValueError(f"entry must have exactly the keys {list(ALLOW_KEYS)}")
            if not isinstance(e["path"], str) or not e["path"].strip() or e["path"].startswith("/") or ".." in e["path"].split("/"):
                raise ValueError("path must be a non-empty relative posix path (as the plan records it; no '..')")
            if not isinstance(e["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", e["sha256"]):
                raise ValueError("sha256 must be 64 lowercase hex chars (sha256sum of the file bytes)")
            ln = e["lanes"]
            if not isinstance(ln, list) or not ln or not all(isinstance(x, str) and x in ALLOW_LANES for x in ln):
                raise ValueError(f"lanes must be a non-empty list drawn from {sorted(ALLOW_LANES)}")
            if not isinstance(e["reason"], str) or not e["reason"].strip():
                raise ValueError("reason must be a non-empty string")
            if e["approved_by"] != "user":
                raise ValueError("approved_by must be 'user'")
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(e["approved_on"])):
                raise ValueError("approved_on must be YYYY-MM-DD")
        except ValueError as ex:
            raise Halt(f"{where}: allow_files[{i}]: {ex}")
        out.append({**e, "approved_on": str(e["approved_on"]), "lanes": list(e["lanes"])})
    return out


def load_deny(files: list[Path]) -> DenySource:
    if not files:
        raise Halt("no deny source files configured")
    w, _ = load_trm_mcp()
    pats: list[re.Pattern] = []
    globs: list[tuple[str, re.Pattern]] = []
    unblind: list[str] = []
    sources: list[tuple[str, str]] = []
    allow: list[dict] = []
    for f in files:
        f = Path(f)
        try:
            raw = f.read_bytes()
            doc = yaml.safe_load(raw.decode("utf-8"))
            fams = doc["families"]
            if not isinstance(fams, list):
                raise ValueError("families is not a list")
        except Exception as e:  # noqa: BLE001
            raise Halt(f"deny source {f} unusable ({type(e).__name__}: {str(e)[:120]})")
        sources.append((str(f), sha256_bytes(raw)))
        allow += _parse_allow_files(doc.get("allow_files"), str(f))
        for fam in fams:
            try:
                if not isinstance(fam, dict):
                    raise ValueError("family is not a mapping")
                fid = fam.get("id")
                status = fam.get("status")
                if status not in ("withheld", "released"):
                    raise ValueError(f"family {fid!r}: unknown status {status!r}")
                kind = fam.get("kind")
                if kind is not None and kind not in KINDS:
                    raise ValueError(f"family {fid!r}: unknown kind {kind!r}")
                pt = _as_list(fam.get("patterns"), "patterns", fid)
                ex = _as_list(fam.get("exp_ids"), "exp_ids", fid)
                pa = _as_list(fam.get("paths"), "paths", fid)
                if status != "withheld" or fid in PROSE_FAMILY_IDS:
                    continue
                # trm-mcp parser compatibility: every withheld family keeps >=1 pattern/exp_id
                if not (pt or ex):
                    raise ValueError(f"withheld family {fid!r} has no patterns or exp_ids")
                pats += [re.compile(p, re.I) for p in pt]
                pats += [re.compile(w._exp_id_pattern(e), re.I) for e in ex]
                globs += [(g, glob_to_regex(g)) for g in pa]
                if kind == "unblind_pending":
                    unblind += ex
            except Halt:
                raise
            except Exception as e:  # noqa: BLE001
                raise Halt(f"deny source {f}: {type(e).__name__}: {str(e)[:160]}")
    return DenySource(PackMatcher(pats), globs, unblind, sources, allow)


def default_deny_files() -> list[Path]:
    from .common import API_BATCH_DENY_PATH, WITHHELD_RUNS_PATH
    return [WITHHELD_RUNS_PATH, API_BATCH_DENY_PATH]
