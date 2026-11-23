"""THE ONE COMMAND that builds every registered figure.

cd /home/ayp/project/trm && CUDA_VISIBLE_DEVICES="" uv run --frozen --no-sync python \
    -m tools.figures.build_all --slug <slug> --cache-dir <dir> (--lab-copy-root <dir> | --no-lab-copy) [...]

Exit codes: 0 all good; 1 a figure, the extractor or the lab copy failed; 2 usage error;
3 the output directory (or lab copy destination) already exists and is not empty.
Nothing here deletes or overwrites: a build only ever writes into a directory it created.
"""
import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

from . import io as fio
from . import registry

REPO = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = REPO / "tools" / "figures" / "registry"
DEFAULT_OUT_ROOT = REPO / "reports" / "figures"
DEFAULT_EXTRACTOR = [sys.executable, "-m", "tools.figures.extract_curves"]


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _child_env(**extra):
    env = dict(os.environ)
    old = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(REPO) + (os.pathsep + old if old else "")
    env.update(extra)
    return env


def figure_env():
    return _child_env(CUDA_VISIBLE_DEVICES="", MPLBACKEND="Agg")


def extractor_env():
    d = tempfile.mkdtemp(prefix="figures-extract-")
    return _child_env(
        CUDA_VISIBLE_DEVICES="", WANDB_MODE="disabled", WANDB_DIR=d,
        WANDB_CACHE_DIR=d, WANDB_CONFIG_DIR=d, WANDB_DATA_DIR=d,
    )


def _distinct_col(df, col):
    if col not in df.columns:
        return []
    return sorted({str(v) for v in df[col].dropna() if str(v).strip()})


def figure_index_entry(files, values_csv, source_json, cache_manifest_sha256):
    """One FIGURE_INDEX.json figure entry; lists come from the values CSV, protocol from source.json."""
    df = pd.read_csv(values_csv, dtype=str, keep_default_na=False)
    with open(source_json, encoding="utf-8") as f:
        src = json.load(f)
    return {
        "files": sorted(files),
        "cache_manifest_sha256": cache_manifest_sha256,
        "run_names": _distinct_col(df, "run_name"),
        "wandb_ids": _distinct_col(df, "wandb_id"),
        "projects": _distinct_col(df, "project"),
        "cohorts": _distinct_col(df, "cohort"),
        "protocol": src.get("protocol", {}),
        "values_csv": Path(values_csv).name,
        "source_json": Path(source_json).name,
    }


def _expected_outputs(entry):
    names = [f"{entry.id}.values.csv", f"{entry.id}.source.json"]
    for lang in entry.langs:
        names += [f"{entry.id}.{lang}.png", f"{entry.id}.{lang}.pdf"]
    return names


def run_figure(entry, out_dir, cache_dir):
    cmd = [sys.executable, "-m", "tools.figures.run_one", "--registry-file", entry.file,
           "--out", str(out_dir), "--cache-dir", str(cache_dir)]
    t0 = time.time()
    p = subprocess.run(cmd, cwd=str(REPO), env=figure_env(), capture_output=True, text=True)
    secs = round(time.time() - t0, 2)
    rc = p.returncode
    tail = p.stderr.splitlines()[-20:]
    missing = [n for n in _expected_outputs(entry)
               if not (out_dir / n).is_file() or (out_dir / n).stat().st_size == 0]
    if rc == 0 and missing:
        rc = 1
        tail = tail + [f"build_all: missing or empty outputs: {missing}"]
    files = sorted(p_.name for p_ in out_dir.iterdir() if p_.name.startswith(entry.id + "."))
    outputs = [{"file": n, "sha256": fio.sha256_file(out_dir / n), "bytes": (out_dir / n).stat().st_size}
               for n in files]
    return {"id": entry.id, "rc": rc, "seconds": secs, "outputs": outputs, "stderr_tail": tail}


def _tree_hashes(root):
    return {str(p.relative_to(root)): fio.sha256_file(p)
            for p in sorted(root.rglob("*")) if p.is_file() and p.name != "LAB_COPY.sha256"}


def lab_copy(out_dir, dest):
    """Copy out_dir to dest (must not exist), compare sha256 file by file, write LAB_COPY.sha256 in both."""
    shutil.copytree(out_dir, dest)
    a, b = _tree_hashes(out_dir), _tree_hashes(dest)
    ok = a == b and len(a) > 0
    text = "".join(f"{h}  {n}\n" for n, h in sorted(a.items()))
    if ok:
        (out_dir / "LAB_COPY.sha256").write_text(text, encoding="utf-8")
        (dest / "LAB_COPY.sha256").write_text(text, encoding="utf-8")
    return ok, a, b


def _status_table(results, extra_lines):
    lines = [f"{'figure':<34}{'rc':>4}{'seconds':>10}  outputs"]
    for r in results:
        lines.append(f"{r['id']:<34}{r['rc']:>4}{r['seconds']:>10}  {len(r['outputs'])}")
    lines += extra_lines
    print("\n".join(lines))


def main(argv=None, *, extractor_cmd=None):
    ap = argparse.ArgumentParser(prog="python -m tools.figures.build_all")
    ap.add_argument("--slug", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--registry-dir", default=str(DEFAULT_REGISTRY))
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--refresh-cache", action="store_true")
    ap.add_argument("--families")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--only", nargs="+")
    ap.add_argument("--lab-copy-root")
    ap.add_argument("--no-lab-copy", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)
    cmdline = " ".join(sys.argv if argv is None else ["build_all"] + list(argv))

    if bool(a.lab_copy_root) == bool(a.no_lab_copy):
        print("exactly one of --lab-copy-root and --no-lab-copy is required", file=sys.stderr)
        return 2
    if a.refresh_cache and not a.families:
        print("--refresh-cache needs --families", file=sys.stderr)
        return 2
    if "/" in a.slug or a.slug in ("", ".", ".."):
        print(f"bad slug {a.slug!r}", file=sys.stderr)
        return 2
    try:
        entries = registry.load_registry(a.registry_dir)
    except Exception as e:  # noqa: BLE001
        print(f"registry error: {e}", file=sys.stderr)
        return 2
    if a.only:
        known = {e.id for e in entries}
        unknown = [i for i in a.only if i not in known]
        if unknown:
            print(f"unknown figure ids: {unknown}", file=sys.stderr)
            return 2
        entries = [e for e in entries if e.id in set(a.only)]
    if not entries:
        print("no figures selected (empty registry)", file=sys.stderr)
        return 2
    if a.list:
        for e in entries:
            print(f"{e.id}\t{','.join(e.langs)}\t{e.title}")
        return 0

    out_dir = Path(a.out_root) / a.slug
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        print(f"output directory exists and is not empty: {out_dir}", file=sys.stderr)
        return 3
    dest = Path(a.lab_copy_root) / a.slug if a.lab_copy_root else None
    if dest is not None and dest.exists():
        print(f"lab copy destination exists: {dest}", file=sys.stderr)
        return 3

    start = _now()
    cache_dir = Path(a.cache_dir)
    if a.refresh_cache:
        cmd = list(extractor_cmd or DEFAULT_EXTRACTOR) + [
            "--families", a.families, "--out", str(cache_dir), "--jobs", str(a.jobs)]
        p = subprocess.run(cmd, cwd=str(REPO), env=extractor_env())
        if p.returncode != 0:
            print(f"cache extraction failed (rc={p.returncode}); no figure was built", file=sys.stderr)
            return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    results = [run_figure(e, out_dir, cache_dir) for e in entries]
    cm = cache_dir / "MANIFEST.files.csv"
    cache_sha = fio.sha256_file(cm) if cm.is_file() else None

    index = {"slug": a.slug, "cache_manifest_sha256": cache_sha, "figures": {}}
    for e, r in zip(entries, results):
        if r["rc"] == 0:
            index["figures"][e.id] = figure_index_entry(
                [o["file"] for o in r["outputs"]], out_dir / f"{e.id}.values.csv",
                out_dir / f"{e.id}.source.json", cache_sha)
    failed = [r["id"] for r in results if r["rc"] != 0]
    exit_code = 1 if failed else 0

    def write_manifest(code, lab_status):
        manifest = {"slug": a.slug, "command": cmdline, "start": start, "end": _now(),
                    "cache_dir": str(cache_dir), "cache_manifest_sha256": cache_sha,
                    "figures": results, "lab_copy": lab_status, "exit_code": code}
        (out_dir / "build_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    (out_dir / "FIGURE_INDEX.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    extra = []
    if dest is None:
        lab_status = "disabled"
    elif exit_code != 0:
        lab_status = "skipped (a figure failed)"
    else:
        lab_status = f"requested {dest} (see LAB_COPY.sha256 for the comparison)"
    write_manifest(exit_code, lab_status)

    if dest is not None and exit_code == 0:
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            ok, _, _ = lab_copy(out_dir, dest)
        except Exception as e:  # noqa: BLE001
            ok = False
            extra.append(f"lab copy error: {e}")
        if ok:
            lab_status = f"matched {dest}"
        else:
            lab_status, exit_code = f"MISMATCH {dest}", 1
            write_manifest(exit_code, lab_status)
    extra.append(f"failed figures: {failed or 'none'}; lab copy: {lab_status}; exit {exit_code}")
    _status_table(results, extra)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
