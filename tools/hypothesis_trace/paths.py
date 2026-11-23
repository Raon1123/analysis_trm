"""PATHS.yaml loader and reference resolver. No lab path is hard-coded; everything joins on `lab_root`."""
from __future__ import annotations

import glob
import os
import re

import yaml

from frontmatter import ParseError

DEFAULT_PATHS_FILE = "/home/ayp/project/trm/lab/11_governance/11_lab-root/PATHS.yaml"


class Paths:
    def __init__(self, paths_file=DEFAULT_PATHS_FILE):
        self.paths_file = paths_file
        try:
            with open(paths_file, encoding="utf-8") as fh:
                d = yaml.safe_load(fh)
        except (OSError, yaml.YAMLError) as e:
            raise ParseError(paths_file, str(e))
        for key in ("lab_root", "repo_root", "paths"):
            if key not in (d or {}):
                raise ParseError(paths_file, f"missing key {key}")
        self.lab_root = d["lab_root"].rstrip("/")
        self.repo_root = d["repo_root"].rstrip("/")
        self.map = d["paths"]
        self.repo_map = d.get("repo_paths", {})
        self.not_yet_created = set(d.get("not_yet_created", []))
        self.legacy = list(d.get("legacy", []))
        self._lab_top = set(os.listdir(self.lab_root)) if os.path.isdir(self.lab_root) else set()
        self._repo_top = set(os.listdir(self.repo_root)) if os.path.isdir(self.repo_root) else set()

    def abs(self, name: str) -> str:
        if name not in self.map:
            raise ParseError(self.paths_file, f"logical name {name} not in paths")
        return os.path.join(self.lab_root, self.map[name])

    def exists(self, name: str) -> bool:
        return os.path.exists(self.abs(name))

    def display(self, abspath: str) -> str:
        """Stable path string used in the trace: `lab/<rel>` for lab files, repo-relative for repo files."""
        if abspath == self.lab_root:
            return "lab"
        if abspath.startswith(self.lab_root + "/"):
            return "lab/" + abspath[len(self.lab_root) + 1:]
        if abspath.startswith(self.repo_root + "/"):
            return abspath[len(self.repo_root) + 1:]
        return abspath

    def lab_rel(self, abspath: str):
        """Path relative to lab_root (the form used by events.jsonl `path`), or None."""
        if abspath.startswith(self.lab_root + "/"):
            return abspath[len(self.lab_root) + 1:]
        return None

    def is_legacy(self, s: str) -> bool:
        return any(s == p or s.startswith(p.rstrip("/") + "/") for p in self.legacy)

    def resolve(self, ref: str):
        """Classify a path-like reference. Returns (kind, abspath|None, display).

        kind: ok | dead | legacy | unresolved-root | external
        - `lab/...` -> lab_root; legacy prefixes (PATHS.yaml `legacy:`) are reported without stat
        - absolute paths under lab_root/repo_root are accepted; other absolute paths are external
        - bare paths: first component must exist under repo_root or lab_root, else the root is unknown
          (vault-relative strings such as 20_Investigations/...) and nothing is stat-ed.
        """
        s = ref.strip().strip("`").strip()
        s = s.split("#", 1)[0].strip()
        s = re.sub(r"\s*\(.*$", "", s).strip()  # trailing "(annotation)"
        s = re.sub(r",\s.*$", "", s).strip()  # trailing ", cell E5" style annotation
        s = re.sub(r"\s+§.*$", "", s).strip()  # trailing " §6-§7" section pointer
        s = re.sub(r":[\w.,\-]+$", "", s) if not re.fullmatch(r"[A-Za-z]:.*", s) else s  # file.py:196-222 / file.py:func
        if not s:
            return ("dead", None, ref)
        if self.is_legacy(s):
            return ("legacy", None, s)
        if s.startswith(self.lab_root + "/") or s == self.lab_root:
            cand = s
        elif s.startswith("lab/"):
            cand = os.path.join(self.lab_root, s[4:])
        elif s.startswith("/"):
            if s.startswith(self.repo_root + "/"):
                cand = s
            else:
                return ("external", None, s)
        else:
            first = s.split("/", 1)[0]
            if first in self._repo_top:
                cand = os.path.join(self.repo_root, s)
            elif first in self._lab_top:
                cand = os.path.join(self.lab_root, s)
            else:
                return ("unresolved-root", None, s)
        # brace / glob expansion
        cands = _expand_braces(cand)
        for c in cands:
            if "*" in c:
                hits = glob.glob(c)
                if hits:
                    return ("ok", hits[0], self.display(hits[0]))
            elif os.path.exists(c):
                return ("ok", c, self.display(c))
        return ("dead", None, self.display(cands[0]) if cands else s)


def _expand_braces(s: str):
    m = re.search(r"\{([^{}]*)\}", s)
    if not m:
        return [s]
    out = []
    for alt in m.group(1).split(","):
        out.extend(_expand_braces(s[:m.start()] + alt + s[m.end():]))
    return out
