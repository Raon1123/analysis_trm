"""Input checking and per-figure output writing for tools.figures.

Run only as `python -m tools.figures.<module>` from the repository root, never as a script
file from inside this directory (this file would shadow the standard library's `io`).
"""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd

from . import style

VALUES_COLUMNS = [
    "figure_id", "panel", "series", "run_name", "wandb_id", "project",
    "cohort", "metric", "step", "value", "source_ref",
]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_cache_csv(cache_dir, file):
    """Read cache_dir/file after checking its sha256 against MANIFEST.files.csv."""
    cache_dir = Path(cache_dir)
    manifest = cache_dir / "MANIFEST.files.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"cache manifest missing: {manifest}")
    path = cache_dir / file
    if not path.is_file():
        raise FileNotFoundError(f"cache file missing: {path}")
    with open(manifest, newline="", encoding="utf-8") as f:
        rows = {r["file"]: r for r in csv.DictReader(f)}
    if file not in rows:
        raise ValueError(f"{file} is not listed in {manifest}")
    want, got = rows[file]["sha256"], sha256_file(path)
    if want != got:
        raise ValueError(f"sha256 mismatch for {path}: manifest {want}, file {got}")
    return pd.read_csv(path)


def _write_once(path, data: bytes):
    """Language-independent files are written once; a second write must be identical."""
    path = Path(path)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"{path.name} differs between languages; it must be language-independent")
        return path
    path.write_bytes(data)
    return path


def _distinct(df, col):
    if col not in df.columns:
        return []
    vals = df[col].dropna().astype(str)
    return sorted({v for v in vals if v.strip()})


class FigureContext:
    def __init__(self, fig_id, lang, out_dir, cache_dir, entry, registry_file=None):
        self.fig_id = fig_id
        self.lang = lang
        self.out_dir = Path(out_dir)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.entry = entry
        self.registry_file = Path(registry_file) if registry_file else None
        self._values = None

    def _p(self, suffix):
        return self.out_dir / f"{self.fig_id}.{suffix}"

    def load_cache(self, file):
        return load_cache_csv(self.cache_dir, file)

    def save(self, fig):
        """finalize (7 pt floor) then write <id>.<lang>.png/.pdf."""
        style.finalize(fig, self.entry.print_width_in)
        return style.save(fig, self.out_dir / f"{self.fig_id}.{self.lang}")

    def write_values(self, df):
        missing = [c for c in VALUES_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"values frame lacks required columns: {missing}")
        if len(df) == 0:
            raise ValueError("values frame has no rows")
        cols = VALUES_COLUMNS + [c for c in df.columns if c not in VALUES_COLUMNS]
        out = df[cols]
        data = out.to_csv(index=False, lineterminator="\n").encode("utf-8")
        self._values = out
        return _write_once(self._p("values.csv"), data)

    def write_extra_csv(self, name, df):
        data = df.to_csv(index=False, lineterminator="\n").encode("utf-8")
        return _write_once(self._p(f"{name}.csv"), data)

    def write_source(self, extra=None):
        extra = dict(extra or {})
        if self._values is None:
            raise RuntimeError("write_values must be called before write_source")
        rows_used = extra.pop("rows_used", {}) or {}
        inputs = []
        for inp in self.entry.inputs:
            if inp["role"] == "cache":
                path = self.cache_dir / inp["file"]
                key = inp["file"]
            else:
                path = Path(inp["path"])
                key = str(path)
            inputs.append({
                "path": str(path),
                "sha256": sha256_file(path),
                "rows_used": rows_used.get(key),
            })
        spec = importlib.util.find_spec(self.entry.module)
        if spec is None or not spec.origin:
            raise RuntimeError(f"cannot locate module {self.entry.module}")
        src = {
            "figure_id": self.fig_id,
            "module": self.entry.module,
            "module_sha256": sha256_file(spec.origin),
            "registry_sha256": sha256_file(self.registry_file) if self.registry_file else None,
            "inputs": inputs,
            "protocol": self.entry.protocol,
            "cohorts": list(self.entry.cohorts),
            "run_names": _distinct(self._values, "run_name"),
            "wandb_ids": _distinct(self._values, "wandb_id"),
            "n_values_rows": int(len(self._values)),
        }
        clash = set(extra) & set(src)
        if clash:
            raise ValueError(f"extra keys collide with standard source keys: {sorted(clash)}")
        src.update(extra)
        data = (json.dumps(src, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
        return _write_once(self._p("source.json"), data)
