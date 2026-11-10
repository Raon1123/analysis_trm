"""Context pack builder and scanner (§5.5): the ONLY path by which lab/repo data and human-written
instructions reach a request body. Six classes of Pack; two scan modes.

  redact       (generation packs): L3a masks matching blocks, then L3b re-checks the residue.
  verify-only  (byte-fixed inputs): any L2/L3b hit refuses the run; nothing is masked.

Layers: L1 allow-list (default deny), L2 hard stop (path/name/frontmatter exp_id vs deny source),
L3a block masking (trm_mcp mask_blocks + lead-in supplement), L3b residue check, L4 size/secrets.
(L5 is the output post-check in normalize.py.) Any failure raises Halt; no partial Pack survives.
Reuses trm-mcp's normalize/_exp_id_pattern/mask_blocks unmodified through a stub package.
"""
from __future__ import annotations

import base64
import json
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .common import (LAB_ROOT, PROBES_DIR, REPO_ROOT, Halt, est_tokens, load_trm_mcp, sha256_bytes,
                     sha256_text)
from .deny import DenySource, load_deny

CLASSES = ("generation", "seat_input", "u1_data", "derived", "spec_text", "probe", "figqa_image")
SECRET_PATTERNS = [re.compile(p) for p in (r"sk-ant-[A-Za-z0-9_\-]{8,}", r"ANTHROPIC_API_KEY\s*=",
                                           r"(?i)\bBearer\s+[A-Za-z0-9._~+/=\-]{16,}")]
_SEAL = object()
PNG_SIG = b"\x89PNG\r\n\x1a\n"
FIGQA_ORIGIN_RE = re.compile(r"^(25_figures|23_reports)/.+\.(pdf|png)$")


def png_dims(b: bytes) -> tuple[int, int]:
    """(width, height) from the IHDR chunk; PackError unless b is a PNG with IHDR first."""
    if len(b) < 24 or b[:8] != PNG_SIG or b[12:16] != b"IHDR":
        raise PackError("figqa image: not a PNG (signature/IHDR)")
    w, h = struct.unpack(">II", b[16:24])
    if w <= 0 or h <= 0:
        raise PackError("figqa image: zero PNG dimensions")
    return w, h


def image_tokens(w: int, h: int) -> int:
    """Image token rule: ceil(width*height/750)."""
    return -(-(w * h) // 750)


def image_tokens_b64(data: str) -> int:
    """Image tokens of a base64 PNG (head bytes only: the IHDR chunk)."""
    w, h = png_dims(base64.b64decode(data[:64] + "=" * (-len(data[:64]) % 4)))
    return image_tokens(w, h)


class PackError(Halt):
    pass


@dataclass(frozen=True)
class Pack:
    """A scanned Pack. Only PackBuilder constructs one with the seal."""
    kind: str
    name: str
    text: str
    mode: str                      # redact | verify-only
    sources: tuple = ()
    report: dict = field(default_factory=dict)
    _seal: object = None

    @property
    def pack_sha256(self) -> str:
        return sha256_text(self.text)

    @property
    def sealed(self) -> bool:
        return self._seal is _SEAL

    def manifest_entry(self) -> dict:
        return {"kind": self.kind, "name": self.name, "mode": self.mode, "pack_sha256": self.pack_sha256,
                "sources": list(self.sources), "report": self.report}


# ----------------------------------------------------------------------------------- helpers
def split_frontmatter(text: str) -> str:
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            return text[3:end]
    return ""


def select_sections(text: str, sections: list[str]) -> str:
    """Markdown sections by label ('0' matches '## 0. ...' / '## §0 ...') or exact heading text."""
    lines = text.splitlines()
    heads, in_fence = [], False
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s.startswith("```") or s.startswith("~~~"):
            in_fence = not in_fence
        if not in_fence and re.match(r"#{1,6}\s", s):
            heads.append((i, len(s) - len(s.lstrip("#")), s.lstrip("#").strip()))
    out = []
    for lab in sections:
        lab = str(lab)
        for k, (i, lvl, title) in enumerate(heads):
            if title == lab or re.match(rf"§?\s*{re.escape(lab)}(?![0-9A-Za-z])", title):
                j = len(lines)
                for (i2, lvl2, _t) in heads[k + 1:]:
                    if lvl2 <= lvl:
                        j = i2
                        break
                out.append("\n".join(lines[i:j]))
                break
        else:
            raise PackError(f"section {lab!r} not found")
    return "\n\n".join(out)


def redact(text: str, matcher) -> tuple[str, int]:
    """L3a: trm_mcp mask_blocks, then the lead-in supplement: after a masked prose block, ALL
    consecutive item/table blocks (blank blocks allowed between) are masked, until a heading,
    another prose block or a fence (NB-R3-2). mask_blocks alone leaves them (documented limit)."""
    w, r = load_trm_mcp()
    masked, n = r.mask_blocks(text, matcher)
    out, armed, extra = [], False, 0
    for b in r.split_blocks(masked):
        k = b["kind"]
        if k == "blank":
            out += b["lines"]
        elif k == "prose":
            armed = "\n".join(b["lines"]).strip() == w.WITHHELD_TEXT
            out += b["lines"]
        elif k in ("item", "table") and armed:
            body = "\n".join(b["lines"]).replace(w.WITHHELD_TEXT, "")
            if re.sub(r"[\s|\-*+#.)0-9]", "", body) == "" and body.count("\n") == 0 and k == "item":
                out += b["lines"]           # already a bare masked item
            elif k == "table" and all(ln.strip() == f"|{w.WITHHELD_TEXT}|" for ln in b["lines"]):
                out += b["lines"]
            else:
                out += r._masked(b)
                extra += 1
        else:
            armed = False
            out += b["lines"]
    return "\n".join(out), n + extra


# ----------------------------------------------------------------------------------- builder
class PackBuilder:
    def __init__(self, deny: DenySource, lab_root: Path = LAB_ROOT, repo_root: Path = REPO_ROOT,
                 batch_dir: Path | None = None, probes_dir: Path = PROBES_DIR, cap_tokens: int = 60000):
        self.deny, self.lab_root, self.repo_root = deny, Path(lab_root), Path(repo_root)
        self.batch_dir = Path(batch_dir) if batch_dir else None
        self.probes_dir, self.cap_tokens = Path(probes_dir), cap_tokens

    @classmethod
    def from_files(cls, deny_files, **kw) -> "PackBuilder":
        """Fail-closed constructor: Halt if any deny source is missing or malformed."""
        return cls(load_deny(list(deny_files)), **kw)

    # -- paths
    def _rel(self, path: Path) -> tuple[str, str]:
        """(root tag, relative posix path) of a file under lab/, the repo, or the batch dir."""
        p = Path(path).resolve()
        for tag, root in (("batch", self.batch_dir), ("lab", self.lab_root), ("repo", self.repo_root)):
            if root is None:
                continue
            try:
                return tag, p.relative_to(Path(root).resolve()).as_posix()
            except ValueError:
                continue
        raise PackError(f"L1: {path} is outside lab/, the repo and the batch directory")

    def _l1(self, kind: str, tag: str, rel: str) -> None:
        name = rel.rsplit("/", 1)[-1]
        if "DECISIONS" in name:
            raise PackError(f"L1: decision ledgers are excluded by default ({rel})")
        ok = False
        if kind == "generation":
            ok = (tag == "batch" and rel.startswith("user/")) or (tag == "lab" and (
                rel == "11_governance/11_lab-root/STATE.md"
                or re.fullmatch(r"27_theory/11_lab-theory/H-[^/]*\.md", rel)
                or re.fullmatch(r"13_reference/11_registries/(HYPOTHESES|EXPERIMENTS)\.md", rel)
                or re.match(r"13_reference/12_atoms/(claims|figures)/.+", rel)
                or rel.startswith("28_literature/")
                or (rel.startswith("22_plans/") and rel.endswith(".md"))))   # spec names the file
        elif kind == "seat_input":
            ok = tag == "lab" and rel.endswith((".md", ".txt"))
        elif kind == "u1_data":
            ok = (tag == "repo" and re.fullmatch(r"wandb/run-[^/]+/files/wandb-summary\.json", rel)) or (
                tag == "lab" and rel.endswith((".csv", ".json"))) or (
                # 2026-10-27 user reply: the curve CSVs the data appendix cites (reports/<date>_<slug>-data/curves/*.csv)
                tag == "repo" and re.fullmatch(r"reports/[0-9]{4}-[0-9]{2}-[0-9]{2}_[A-Za-z0-9._-]+-data/curves/[^/]+\.csv", rel))
            if rel.endswith(".wandb"):
                ok = False
        elif kind == "derived":
            ok = (tag == "batch" and re.search(r"(^|/)items/[^/]+\.md$|(^|/)(INBOX|decisions)\.jsonl$", rel)) or (
                # a reduce batch reads the collected items of its map batch (same lab/38_api-batch tree)
                tag == "lab" and re.fullmatch(r"38_api-batch/11_batches/[^/]+/items/[^/]+\.md", rel))
        elif kind == "spec_text":
            ok = (tag == "batch" and (rel.endswith("spec.yaml") or re.fullmatch(r"shared/[^/]+\.md", rel))) or (
                tag == "repo" and re.fullmatch(r"tools/api_batch/prompts/[^/]+\.md", rel))
        if not ok:
            raise PackError(f"L1: {rel!r} is not an allowed source for class {kind!r}")

    # -- scan layers
    def _l2(self, rel: str, text: str) -> None:
        g = self.deny.path_hit(rel)
        if g:
            raise PackError(f"L2: path {rel!r} matches deny glob {g!r}")
        m = self.deny.matcher
        if m.matches(rel):
            raise PackError(f"L2: path/name {rel!r} matches the deny source")
        fm = split_frontmatter(text)
        if fm:
            try:
                d = yaml.safe_load(fm) or {}
                vals = [str(v) for k, v in d.items() if str(k) in ("exp_id", "exp_ids", "exp")] if isinstance(d, dict) else []
            except Exception:  # noqa: BLE001
                vals = []
            if m.matches(fm) or any(m.matches(v) for v in vals):
                raise PackError(f"L2: frontmatter of {rel!r} names a denied experiment")

    def _l3b(self, label: str, text: str) -> None:
        if self.deny.matcher.matches(text):
            raise PackError(f"L3b: residual deny-source match in {label}")

    def _l4(self, label: str, text: str, cap: bool) -> int:
        for sp in SECRET_PATTERNS:
            if sp.search(text):
                raise PackError(f"L4: secret-like pattern in {label}")
        tok = est_tokens(text)
        if cap and tok > self.cap_tokens:
            raise PackError(f"L4: {label} is ~{tok} tokens > cap_tokens {self.cap_tokens}")
        return tok

    def _issue(self, kind, name, text, mode, sources, report) -> Pack:
        report = {**report, "deny_sha256": self.deny.sha256}
        return Pack(kind, name, text, mode, tuple(sources), report, _SEAL)

    # -- constructors
    def build_generation(self, name: str, sources: list[dict]) -> Pack:
        """sources: [{path, sections?}] (path absolute, or relative to lab/ / batch dir)."""
        if not sources:
            raise PackError("generation pack with no sources")
        parts, srcs, masked_total = [], [], 0
        for s in sources:
            p = Path(s["path"])
            if not p.is_absolute():
                p = (self.batch_dir / p) if (self.batch_dir and (self.batch_dir / p).exists()) else self.lab_root / p
            tag, rel = self._rel(p)
            self._l1("generation", tag, rel)
            raw = p.read_bytes()
            full = raw.decode("utf-8")
            self._l2(rel, full)
            body = select_sections(full, s["sections"]) if s.get("sections") else full
            masked, n = redact(body, self.deny.matcher)
            self._l3b(rel, masked)
            masked_total += n
            parts.append(f"===== SOURCE {rel} =====\n{masked}")
            srcs.append({"path": rel, "sha256": sha256_bytes(raw), "sections": list(s.get("sections") or []), "masked": n})
        text = "\n\n".join(parts) + "\n"
        tok = self._l4(name, text, cap=True)
        return self._issue("generation", name, text, "redact", srcs, {"masked_blocks": masked_total, "est_tokens": tok})

    def build_verify_only(self, kind: str, name: str, text: str, origin: Path | None = None,
                          lane: str | None = None) -> Pack:
        """Byte-fixed input: refuse on any hit, never mask. `origin` (a file) is checked by L1/L2.

        allow_files exception (user-approved, sha-pinned): only when `lane` is given, `origin` is a file whose
        lab-relative path AND current byte sha256 match an `allow_files` entry listing that lane. It skips ONLY the
        L2/L3b deny refusals (L1 allow-list and L4 secrets/size still apply); every skipped hit is recorded in
        report["allowed_hits"] and the entry in report["allowed_by"], so nothing is silent."""
        if kind not in CLASSES or kind in ("generation", "probe", "figqa_image"):
            raise PackError(f"verify-only does not apply to class {kind!r}")
        srcs, allowed_by, allowed_hits = [], None, []
        if origin is not None:
            tag, rel = self._rel(origin)
            self._l1(kind, tag, rel)
            if lane is not None:
                raw = Path(origin).read_bytes()
                entry, why = self.deny.allow_entry(rel, sha256_bytes(raw), lane)
                if entry is not None and raw == text.encode("utf-8"):
                    allowed_by = dict(entry)
                elif why:
                    try:
                        self._l2(rel, text); self._l3b(name, text)
                    except PackError as e:
                        raise PackError(f"{e}; {why}")
            if allowed_by is None:
                self._l2(rel, text)
            else:
                allowed_hits += self._collect_hits("L2", rel, text, origin_rel=rel)
            srcs.append({"path": rel, "sha256": sha256_text(text), "sections": [], "masked": 0})
        else:
            self._l2(name, text)
        if allowed_by is None:
            self._l3b(name, text)
        else:
            allowed_hits += self._collect_hits("L3b", name, text)
        tok = self._l4(name, text, cap=False)
        rep = {"masked_blocks": 0, "est_tokens": tok}
        if allowed_by is not None:
            rep["allowed_by"] = allowed_by
            rep["allowed_hits"] = allowed_hits
        return self._issue(kind, name, text, "verify-only", srcs, rep)

    def _collect_hits(self, layer: str, label: str, text: str, origin_rel: str | None = None) -> list[dict]:
        """The deny hits an allow_files entry lets through (recorded, never silent)."""
        out = []
        if layer == "L2":
            g = self.deny.path_hit(origin_rel or label)
            if g:
                out.append({"layer": "L2", "kind": "path-glob", "hit": g})
            for pat in self.deny.matcher.hits(origin_rel or label):
                out.append({"layer": "L2", "kind": "path-name", "hit": pat})
            fm = split_frontmatter(text)
            for pat in (self.deny.matcher.hits(fm) if fm else []):
                out.append({"layer": "L2", "kind": "frontmatter", "hit": pat})
        else:
            for pat in self.deny.matcher.hits(text):
                out.append({"layer": "L3b", "kind": "body", "hit": pat})
        return out

    def build_u1_cell(self, name: str, text: str, parents: list[Pack], desc: dict) -> Pack:
        """A per-cell slice of already-scanned U1 data (verify-only). Parents are the sealed u1_data Packs of the whole
        source files (each passed L1/L2/L3b on its own); the slice is rescanned (L2 on its name, L3b, L4) and lists
        every parent source with its sha256, so a request body traces to file bytes."""
        if not parents or not all(isinstance(p, Pack) and p.sealed and p.kind == "u1_data" and p.sources for p in parents):
            raise PackError("u1 cell slice needs sealed u1_data parent Packs")
        self._l2(name, text)
        self._l3b(name, text)
        tok = self._l4(name, text, cap=False)
        srcs = [dict(p.sources[0]) for p in parents]
        return self._issue("u1_data", name, text, "verify-only", srcs,
                           {"masked_blocks": 0, "est_tokens": tok, "slice": desc})

    def build_probe(self, filename: str) -> Pack:
        """Synthetic probe: only tools/api_batch/probes/*.txt, nothing else."""
        p = (self.probes_dir / filename).resolve()
        if p.parent != self.probes_dir.resolve() or p.suffix != ".txt" or not p.is_file():
            raise PackError(f"L1: probe {filename!r} is not a file directly under {self.probes_dir}")
        text = p.read_text(encoding="utf-8")
        self._l2(p.name, text)
        self._l3b(p.name, text)
        tok = self._l4(p.name, text, cap=False)
        return self._issue("probe", p.name, text, "verify-only",
                           [{"path": f"probes/{p.name}", "sha256": sha256_bytes(p.read_bytes()), "sections": [], "masked": 0}],
                           {"masked_blocks": 0, "est_tokens": tok})

    def build_derived(self, item_path: Path, inbox_path: Path) -> Pack:
        """Batch-derived input: only items that passed L5 and are listed in INBOX.jsonl."""
        item = Path(item_path)
        text = item.read_text(encoding="utf-8")
        hdr = text.split("\n---", 1)[0]
        if not re.search(r"(?m)^l5:\s*pass\s*$", hdr):
            raise PackError(f"derived: {item.name} has no 'l5: pass' header (quarantined or unchecked)")
        listed = set()
        try:
            for ln in Path(inbox_path).read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    listed.add(json.loads(ln).get("custom_id"))
        except Exception as e:  # noqa: BLE001
            raise PackError(f"derived: INBOX unreadable ({type(e).__name__})")
        if item.stem not in listed:
            raise PackError(f"derived: {item.stem} is not listed in INBOX.jsonl")
        return self.build_verify_only("derived", f"derived-{item.stem}", text, origin=item)

    def build_spec_text(self, name: str, text: str) -> Pack:
        """Spec strings (perspectives, instructions): refuse, never mask, if they name a denied target."""
        return self.build_verify_only("spec_text", name, text, origin=None)

    def build_figqa_image(self, name: str, png_path: Path, origin_rel: str, exp_id: str) -> Pack:
        """fig-qa figure (§5.5 class 2): the neutral-name PNG copy under <batch>/user/, base64 as the Pack text.
        Pixels cannot be scanned, so the figure's canonical origin path and owning exp_id go through L1/L2 instead."""
        if self.batch_dir is None:
            raise PackError("L1: figqa image needs a batch directory")
        user = (self.batch_dir / "user").resolve()
        pp = Path(png_path).resolve()
        try:
            png_rel = "user/" + pp.relative_to(user).as_posix()
        except ValueError:
            raise PackError(f"L1: figqa png {png_path} is not under the batch directory's user/")
        if not isinstance(origin_rel, str) or not FIGQA_ORIGIN_RE.match(origin_rel) or ".." in origin_rel.split("/"):
            raise PackError(f"L1: figqa origin {origin_rel!r} must match {FIGQA_ORIGIN_RE.pattern}")
        op = self.lab_root / origin_rel
        if not op.is_file():
            raise PackError(f"L1: figqa origin {origin_rel!r} does not exist under lab/")
        if not isinstance(exp_id, str) or not exp_id:
            raise PackError("L1: figqa exp_id must be a non-empty string")
        g = self.deny.path_hit(origin_rel)
        if g:
            raise PackError(f"L2: path {origin_rel!r} matches deny glob {g!r}")
        if self.deny.matcher.matches(origin_rel):
            raise PackError(f"L2: path/name {origin_rel!r} matches the deny source")
        if self.deny.matcher.matches(exp_id):
            raise PackError(f"L2: exp_id {exp_id!r} matches the deny source")
        self._l4(name, name, cap=False)
        raw = pp.read_bytes()
        w, h = png_dims(raw)
        text = base64.b64encode(raw).decode("ascii")
        srcs = [{"path": origin_rel, "sha256": sha256_bytes(op.read_bytes())}, {"path": png_rel, "sha256": sha256_bytes(raw)}]
        return self._issue("figqa_image", name, text, "verify-only", srcs,
                           {"masked_blocks": 0, "est_tokens": image_tokens(w, h), "width": w, "height": h, "exp_id": exp_id})
