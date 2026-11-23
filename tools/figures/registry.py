"""Declarative per-figure registry: one YAML file per figure in a directory.

Run only as `python -m tools.figures.registry --dir <dir> --list` from the repository root.
"""
import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

KEYS = ("id", "module", "title", "langs", "print_width_in", "inputs", "protocol", "cohorts", "report_refs")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_MODULE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")


class RegistryError(ValueError):
    pass


@dataclass(frozen=True)
class Entry:
    id: str
    module: str
    title: str
    langs: tuple
    print_width_in: float
    inputs: tuple
    protocol: dict
    cohorts: tuple
    report_refs: tuple
    file: str = ""


def _fail(path, msg):
    raise RegistryError(f"{path}: {msg}")


def _check_inputs(path, inputs):
    if not isinstance(inputs, list):
        _fail(path, "inputs must be a list")
    for i, inp in enumerate(inputs):
        if not isinstance(inp, dict) or "role" not in inp:
            _fail(path, f"inputs[{i}] must be a mapping with a role")
        role = inp["role"]
        want = {"cache": {"role", "family", "file"}, "file": {"role", "path"}}.get(role)
        if want is None:
            _fail(path, f"inputs[{i}].role must be cache or file, got {role!r}")
        if set(inp) != want:
            _fail(path, f"inputs[{i}] keys {sorted(inp)} != {sorted(want)}")
        for k in want - {"role"}:
            if not isinstance(inp[k], str) or not inp[k]:
                _fail(path, f"inputs[{i}].{k} must be a non-empty string")
        if role == "file" and not Path(inp["path"]).is_absolute():
            _fail(path, f"inputs[{i}].path must be absolute")


def load_entry(path):
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        d = yaml.safe_load(f)
    if not isinstance(d, dict):
        _fail(path, "top level must be a mapping")
    unknown, missing = set(d) - set(KEYS), set(KEYS) - set(d)
    if unknown:
        _fail(path, f"unknown keys {sorted(unknown)}")
    if missing:
        _fail(path, f"missing keys {sorted(missing)}")
    if not isinstance(d["id"], str) or not _ID_RE.match(d["id"]):
        _fail(path, f"bad id {d['id']!r}")
    if d["id"] != path.stem:
        _fail(path, f"id {d['id']!r} != file stem {path.stem!r}")
    if not isinstance(d["module"], str) or not _MODULE_RE.match(d["module"]):
        _fail(path, f"bad module {d['module']!r}")
    if not isinstance(d["title"], str) or not d["title"].strip():
        _fail(path, "title must be a non-empty string")
    langs = d["langs"]
    if not isinstance(langs, list) or not langs or not set(langs) <= {"en", "ko"} or len(set(langs)) != len(langs):
        _fail(path, f"langs must be a non-empty subset of [en, ko], got {langs!r}")
    w = d["print_width_in"]
    if isinstance(w, bool) or not isinstance(w, (int, float)) or w <= 0:
        _fail(path, f"print_width_in must be a positive number, got {w!r}")
    _check_inputs(path, d["inputs"])
    proto = d["protocol"]
    if not isinstance(proto, dict) or (proto and set(proto) != {"metric_test", "metric_train"}):
        _fail(path, "protocol must be {} or have exactly metric_test and metric_train")
    for k in ("cohorts", "report_refs"):
        if not isinstance(d[k], list) or not all(isinstance(x, str) for x in d[k]):
            _fail(path, f"{k} must be a list of strings")
    return Entry(
        id=d["id"], module=d["module"], title=d["title"], langs=tuple(langs),
        print_width_in=float(w), inputs=tuple(d["inputs"]), protocol=dict(proto),
        cohorts=tuple(d["cohorts"]), report_refs=tuple(d["report_refs"]), file=str(path),
    )


def load_registry(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise RegistryError(f"registry directory missing: {directory}")
    entries = [load_entry(p) for p in sorted(directory.glob("*.yaml"))]
    return sorted(entries, key=lambda e: e.id)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m tools.figures.registry")
    ap.add_argument("--dir", required=True)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)
    try:
        entries = load_registry(a.dir)
    except (RegistryError, OSError, yaml.YAMLError) as e:
        print(f"registry error: {e}", file=sys.stderr)
        return 2
    if a.list:
        for e in entries:
            print(f"{e.id}\t{','.join(e.langs)}\t{e.print_width_in:g} in\t{e.module}\t{e.title}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
