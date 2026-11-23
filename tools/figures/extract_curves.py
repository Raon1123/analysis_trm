"""Extract training curves from the local wandb datastore into tidy long CSVs.

Usage (from the repository root):

    CUDA_VISIBLE_DEVICES="" uv run --frozen --no-sync python -m tools.figures.extract_curves \
        --families tools/figures/families/pilot.yaml --out <new or empty dir> \
        [--jobs 4] [--verify-after] [--max-out-mb 200] [--only <family id> ...]

Read-only: every ``*.wandb`` file is opened with mode "rb" (hashing and
scanning through ``tools.figures.wandb_ro``). Nothing is written outside --out.

Exit codes: 0 ok; 1 a loud failure (message names the runs); 2 usage or input
error; 3 --out exists and is not empty; 4 SOURCE CHANGED.
"""

from __future__ import annotations

from tools.figures import wandb_ro  # sets WANDB_MODE=disabled before wandb loads

import argparse
import concurrent.futures as cf
import csv
import datetime as dt
import glob
import hashlib
import io
import json
import multiprocessing as mp
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
DEFAULT_WANDB_ROOT = REPO / "wandb"
DEFAULT_PROTOCOLS = Path(__file__).resolve().parent / "protocols.yaml"

CURVE_COLUMNS = ["run_name", "wandb_id", "project", "family", "cohort", "data_path",
                 "started_at", "arch", "k", "seed", "step", "metric", "value"]
CONFIG_COLUMNS = ["run_name", "wandb_id", "project", "family", "key", "value"]
MANIFEST_COLUMNS = ["family", "run_name", "wandb_id", "project", "run_dir", "is_symlink", "role",
                    "reason", "started_at", "cohort", "wandb_file", "wandb_bytes",
                    "wandb_mtime_ns_before", "wandb_mtime_ns_after", "wandb_sha256_before",
                    "wandb_sha256_after", "last_step", "expected_last_step", "n_eval_points",
                    "n_train_points"]
FILES_COLUMNS = ["file", "sha256", "n_rows", "n_runs"]

RECENT_WRITE_SECONDS = 30 * 60
HASH_CHUNK = 8 * 1024 * 1024


class LoudFailure(Exception):
    def __init__(self, msg: str, code: int = 1):
        super().__init__(msg)
        self.code = code


# ----------------------------------------------------------------------------- helpers

def sha256_file(path: str) -> str:
    """Streamed sha256 of a file opened read-only ("rb")."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(HASH_CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def parse_args_list(args: list) -> list[tuple[str, str]]:
    """Hydra overrides -> [(key, value)], split at the first '=', leading '+' removed."""
    out = []
    for a in args or []:
        a = str(a)
        if "=" not in a:
            continue
        k, v = a.split("=", 1)
        out.append((k.lstrip("+"), v))
    return out


def first_data_path(v: str) -> str:
    v = (v or "").strip()
    if v.startswith("[") and v.endswith("]"):
        v = v[1:-1]
    v = v.split(",")[0].strip().strip("'\"")
    return v.rstrip("/")


def parse_utc(s: str) -> dt.datetime:
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    d = dt.datetime.fromisoformat(s)
    if d.tzinfo is None:
        d = d.replace(tzinfo=dt.timezone.utc)
    return d.astimezone(dt.timezone.utc)


def _under(path: str, prefix: str) -> bool:
    prefix = prefix.rstrip("/")
    return path == prefix or path.startswith(prefix + "/")


def cohort_for(data_path: str, started_at: str, fam_cfg: dict) -> str:
    cutoff = parse_utc(fam_cfg["rebuild_cutoff_utc"])
    default = "other"
    for rule in fam_cfg.get("cohort_rule", []):
        if "default" in rule:
            default = rule["default"]
            continue
        if _under(data_path, rule["prefix"]):
            if "cohort" in rule:
                return rule["cohort"]
            if not started_at:
                return default
            return rule["before_cutoff"] if parse_utc(started_at) < cutoff else rule["after_cutoff"]
    return default


def fmt_value(v):
    if isinstance(v, bool):
        return repr(float(int(v)))
    if isinstance(v, (int, float)):
        return repr(float(v))
    return None


def csv_bytes(columns: list[str], rows: list[list]) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(columns)
    w.writerows(rows)
    return buf.getvalue().encode("utf-8")


# ----------------------------------------------------------------------------- discovery

def read_identity(run_dir: str) -> dict | None:
    """Identity from files/wandb-metadata.json args (fallback files/config.yaml)."""
    meta_p = Path(run_dir, "files", "wandb-metadata.json")
    overrides: list[tuple[str, str]] = []
    started_at = ""
    meta = None
    if meta_p.is_file():
        try:
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
        except Exception:
            meta = None
    if isinstance(meta, dict):
        overrides = parse_args_list(meta.get("args") or [])
        started_at = meta.get("startedAt") or ""
    kv = {}
    for k, v in overrides:
        kv.setdefault(k, v)
    project = kv.get("project_name")
    run_name = kv.get("run_name")
    data_path = first_data_path(kv.get("data_paths", ""))
    seed = kv.get("seed", "")
    arch = kv.get("arch", "")
    source = "metadata"
    if not project or not run_name:
        cfg_p = Path(run_dir, "files", "config.yaml")
        cfg = None
        if cfg_p.is_file():
            try:
                cfg = yaml.safe_load(cfg_p.read_text(encoding="utf-8"))
            except Exception:
                cfg = None
        if not isinstance(cfg, dict):
            return None

        def cv(key):
            x = cfg.get(key)
            return x.get("value") if isinstance(x, dict) and "value" in x else x

        project = project or cv("project_name")
        run_name = run_name or cv("run_name")
        if not project or not run_name:
            return None
        dp = cv("data_paths")
        if not data_path and isinstance(dp, list) and dp:
            data_path = str(dp[0]).rstrip("/")
        if seed == "" and cv("seed") is not None:
            seed = str(cv("seed"))
        if not arch:
            a = cv("arch")
            arch = a.get("name", "") if isinstance(a, dict) else (a or "")
        source = "config.yaml"
    if not started_at:
        m = re.search(r"run-(\d{8}_\d{6})-", os.path.basename(run_dir))
        if m:
            started_at = dt.datetime.strptime(m.group(1), "%Y%m%d_%H%M%S").replace(
                tzinfo=dt.timezone.utc).isoformat().replace("+00:00", "Z")
    wid = os.path.basename(run_dir.rstrip("/")).rsplit("-", 1)[-1]
    return {"project": str(project), "run_name": str(run_name), "data_path": data_path,
            "seed": str(seed), "arch": str(arch), "started_at": started_at,
            "wandb_id": wid, "overrides": overrides, "identity_source": source}


def discover(wandb_root: str) -> tuple[list[str], list[str]]:
    """Every entry matching *run-* that is a directory after following symlinks."""
    dirs = []
    for e in sorted(os.listdir(wandb_root)):
        if not re.fullmatch(r".*run-.*", e):
            continue
        p = os.path.join(wandb_root, e)
        if os.path.isdir(p):  # follows symlinks
            dirs.append(p)
    return dirs, []


# ----------------------------------------------------------------------------- scanning

def scan_one(task: dict) -> dict:
    """Hash, scan, re-stat (and optionally re-hash) one run directory."""
    out = dict(task)
    files = sorted(glob.glob(os.path.join(task["run_dir"], "*.wandb")))
    if len(files) != 1:
        out.update(error=f"expected one *.wandb file, found {len(files)}", wandb_file="")
        return out
    f = files[0]
    st0 = os.stat(f)
    out.update(wandb_file=f, wandb_bytes=st0.st_size, mtime_before=st0.st_mtime_ns,
               size_before=st0.st_size)
    out["sha_before"] = sha256_file(f)
    eval_keys = set(task["eval_metrics"])
    train_keys = set(task["train_metrics"])
    stride = int(task["train_stride"])
    eval_rows: list[tuple[int, str, object]] = []
    eval_steps: set[int] = set()
    train_rows: list[tuple[int, str, object]] = []
    last_train: tuple[int, dict] | None = None
    stats: dict = {}
    truncated = None
    try:
        for step, values in wandb_ro.iter_history(f, eval_keys | train_keys, stats):
            ev = {k: v for k, v in values.items() if k in eval_keys}
            if ev:
                eval_steps.add(step)
                for k, v in ev.items():
                    eval_rows.append((step, k, v))
            tr = {k: v for k, v in values.items() if k in train_keys}
            if tr:
                if step % stride == 0:
                    for k, v in tr.items():
                        train_rows.append((step, k, v))
                if last_train is None or step >= last_train[0]:
                    last_train = (step, tr)
    except wandb_ro.TruncatedRun as exc:
        truncated = {"bytes_read": exc.bytes_read, "file_size": exc.file_size, "msg": str(exc)}
    if last_train is not None and last_train[0] % stride != 0:
        for k, v in last_train[1].items():
            train_rows.append((last_train[0], k, v))
    st1 = os.stat(f)
    out.update(mtime_after=st1.st_mtime_ns, size_after=st1.st_size)
    out["sha_after"] = sha256_file(f) if task.get("verify_after") else ""
    out.update(last_step=stats.get("last_step"), truncated=truncated,
               eval_rows=eval_rows, train_rows=train_rows, n_eval_points=len(eval_steps),
               n_train_points=len({s for s, _, _ in train_rows}))
    return out


def run_scans(tasks: list[dict], jobs: int) -> list[dict]:
    if jobs <= 1 or len(tasks) <= 1:
        return [scan_one(t) for t in tasks]
    ctx = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as ex:
        return list(ex.map(scan_one, tasks))


# ----------------------------------------------------------------------------- main

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m tools.figures.extract_curves",
                                description=__doc__.split("\n\n")[0])
    p.add_argument("--families", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--verify-after", action="store_true")
    p.add_argument("--max-out-mb", type=float, default=200)
    p.add_argument("--only", nargs="+", default=None)
    p.add_argument("--wandb-root", default=str(DEFAULT_WANDB_ROOT), help=argparse.SUPPRESS)
    p.add_argument("--protocols", default=str(DEFAULT_PROTOCOLS), help=argparse.SUPPRESS)
    return p


def git_head() -> str:
    try:
        r = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                           text=True, timeout=30)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def extract(a, argv: list[str]) -> int:
    t_start = time.time()
    start_iso = dt.datetime.now(dt.timezone.utc).isoformat()
    out_dir = Path(a.out)
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise LoudFailure(f"refusing: --out {out_dir} exists and is not empty", 3)

    fam_path = Path(a.families)
    fam_bytes = fam_path.read_bytes()
    fam_cfg = yaml.safe_load(fam_bytes.decode("utf-8"))
    families = fam_cfg["families"]
    if a.only:
        unknown = set(a.only) - {f["id"] for f in families}
        if unknown:
            raise LoudFailure(f"unknown family id(s) in --only: {sorted(unknown)}", 2)
        families = [f for f in families if f["id"] in a.only]
    prot_path = Path(a.protocols)
    prot_bytes = prot_path.read_bytes()
    protocols = yaml.safe_load(prot_bytes.decode("utf-8"))

    eval_metrics = list(fam_cfg["eval_metrics"])
    required = list(fam_cfg.get("required_eval_metrics", []))
    train_metrics = list(fam_cfg["train_metrics"])
    stride = int(fam_cfg["train_stride"])

    run_dirs, _ = discover(a.wandb_root)
    unidentified = []
    tasks = []
    for d in run_dirs:
        ident = read_identity(d)
        if ident is None:
            unidentified.append(os.path.basename(d))
            continue
        for fam in families:
            if ident["project"] == fam["project"] and re.fullmatch(fam["regex"], ident["run_name"]):
                tasks.append({
                    "family": fam["id"], "run_dir": d, "is_symlink": os.path.islink(d),
                    "ident": ident,
                    "cohort": cohort_for(ident["data_path"], ident["started_at"], fam_cfg),
                    "eval_metrics": eval_metrics, "train_metrics": train_metrics,
                    "train_stride": stride, "verify_after": bool(a.verify_after),
                    "expected_last_step": int(fam["expected_last_step"]),
                })
    print(f"discovered {len(run_dirs)} run dirs, {len(unidentified)} unidentified, "
          f"{len(tasks)} selected", file=sys.stderr)

    results = run_scans(tasks, a.jobs)

    failures: list[str] = []
    changed: list[str] = []
    for r in results:
        if r.get("error"):
            failures.append(f"{r['family']}: {r['ident']['run_name']} ({r['ident']['wandb_id']}): "
                            f"{r['error']}")
            continue
        if (r["mtime_before"] != r["mtime_after"] or r["size_before"] != r["size_after"]
                or (a.verify_after and r["sha_before"] != r["sha_after"])):
            changed.append(f"{r['ident']['run_name']} ({r['ident']['wandb_id']}) {r['wandb_file']}")
    if changed:
        raise LoudFailure("SOURCE CHANGED: " + "; ".join(changed), 4)

    # primary selection per (family, project, run name)
    groups: dict[tuple, list[dict]] = {}
    for r in results:
        if r.get("error"):
            continue
        groups.setdefault((r["family"], r["ident"]["project"], r["ident"]["run_name"]), []).append(r)
    now = time.time()
    for key, rs in groups.items():
        rs.sort(key=lambda r: (r["last_step"] if r["last_step"] is not None else -1,
                               parse_utc(r["ident"]["started_at"]).timestamp()
                               if r["ident"]["started_at"] else 0.0),
                reverse=True)
        prim = rs[0]
        prim["role"], prim["reason"] = "primary", ""
        for o in rs[1:]:
            o["role"] = "superseded"
            if o["truncated"]:
                o["reason"] = (f"TruncatedRun at byte {o['truncated']['bytes_read']} of "
                               f"{o['truncated']['file_size']}, last step {o['last_step']}; "
                               f"primary {prim['ident']['wandb_id']}")
            elif (o["last_step"] or -1) < (prim["last_step"] or -1):
                o["reason"] = (f"truncated at step {o['last_step']}; primary "
                               f"{prim['ident']['wandb_id']} reaches {prim['last_step']}")
            else:
                o["reason"] = (f"same last step {o['last_step']}, started earlier than primary "
                               f"{prim['ident']['wandb_id']}")
        rn = f"{key[0]}: {key[2]} ({prim['ident']['wandb_id']})"
        if prim["truncated"]:
            failures.append(f"{rn}: primary raised {prim['truncated']['msg']}")
        if prim["last_step"] != prim["expected_last_step"]:
            failures.append(f"{rn}: last step {prim['last_step']} != expected "
                            f"{prim['expected_last_step']}")
        for m in required:
            if not any(mm == m for _, mm, _ in prim["eval_rows"]):
                failures.append(f"{rn}: zero points for required metric {m}")
        if now - prim["mtime_before"] / 1e9 < RECENT_WRITE_SECONDS:
            failures.append(f"{rn}: {prim['wandb_file']} modified within the last 30 minutes "
                            f"(still being written)")
    for fam in families:
        names = {k[2] for k in groups if k[0] == fam["id"]}
        if len(names) < int(fam["expected_names"]):
            failures.append(f"{fam['id']}: matched {len(names)} distinct run names, expected "
                            f"{fam['expected_names']} (found: {sorted(names)})")
    if failures:
        raise LoudFailure("LOUD FAILURE:\n  " + "\n  ".join(failures), 1)

    # ------------------------------------------------------------------ build outputs in memory
    outputs: dict[str, bytes] = {}
    files_meta: list[list] = []
    primaries = sorted((r for r in results if r.get("role") == "primary"),
                       key=lambda r: (r["ident"]["run_name"], r["ident"]["project"],
                                      r["ident"]["wandb_id"]))

    def ident_cols(r):
        i = r["ident"]
        k = i["data_path"].rstrip("/").split("/")[-1] if i["data_path"] else ""
        return [i["run_name"], i["wandb_id"], i["project"], r["family"], r["cohort"],
                i["data_path"], i["started_at"], i["arch"], k, i["seed"]]

    for fam in families:
        for kind in ("eval", "train"):
            rows = []
            for r in primaries:
                if r["family"] != fam["id"]:
                    continue
                base = ident_cols(r)
                for step, metric, value in r[f"{kind}_rows"]:
                    sv = fmt_value(value)
                    if sv is None:
                        continue
                    rows.append(base + [step, metric, sv])
            rows.sort(key=lambda x: (x[0], x[11], x[10]))  # run_name, metric, step (stable)
            rows = [x[:10] + [str(x[10])] + x[11:] for x in rows]
            name = f"{fam['id']}.{kind}.csv"
            if not rows:
                raise LoudFailure(f"output {name} would be empty", 1)
            outputs[name] = csv_bytes(CURVE_COLUMNS, rows)
            files_meta.append([name, len(rows), len({(x[2], x[0], x[1]) for x in rows})])

    cfg_rows = []
    for r in primaries:
        i = r["ident"]
        for k, v in i["overrides"]:
            cfg_rows.append([i["run_name"], i["wandb_id"], i["project"], r["family"], k, v])
    outputs["configs.csv"] = csv_bytes(CONFIG_COLUMNS, cfg_rows)
    files_meta.append(["configs.csv", len(cfg_rows), len({(x[2], x[0], x[1]) for x in cfg_rows})])

    man_rows = []
    for r in sorted(results, key=lambda r: (r["family"], r["ident"]["run_name"],
                                            r.get("role", "error") != "primary",
                                            r["ident"]["started_at"], r["run_dir"])):
        i = r["ident"]
        man_rows.append([
            r["family"], i["run_name"], i["wandb_id"], i["project"], r["run_dir"],
            str(bool(r["is_symlink"])), r.get("role", "error"), r.get("reason", r.get("error", "")),
            i["started_at"], r["cohort"], r.get("wandb_file", ""), r.get("wandb_bytes", ""),
            r.get("mtime_before", ""), r.get("mtime_after", ""), r.get("sha_before", ""),
            r.get("sha_after", ""), "" if r.get("last_step") is None else r["last_step"],
            r["expected_last_step"], r.get("n_eval_points", ""), r.get("n_train_points", ""),
        ])
    outputs["MANIFEST.csv"] = csv_bytes(MANIFEST_COLUMNS, man_rows)
    files_meta.append(["MANIFEST.csv", len(man_rows), len({(x[3], x[1], x[2]) for x in man_rows})])

    outputs["PROTOCOLS.json"] = (json.dumps({"protocols": protocols,
                                             "sha256": sha256_bytes(prot_bytes),
                                             "source": str(prot_path)},
                                            indent=1, sort_keys=True, ensure_ascii=False)
                                 + "\n").encode("utf-8")
    files_meta.append(["PROTOCOLS.json", "", ""])

    outputs["MANIFEST.files.csv"] = csv_bytes(
        FILES_COLUMNS, [[n, sha256_bytes(outputs[n]), nr, nu] for n, nr, nu in files_meta])

    counts = {
        "run_dirs_discovered": len(run_dirs),
        "unidentified": len(unidentified),
        "selected_dirs": len(results),
        "per_family": {
            fam["id"]: {
                "dirs": sum(1 for r in results if r["family"] == fam["id"]),
                "primaries": sum(1 for r in primaries if r["family"] == fam["id"]),
                "superseded": sum(1 for r in results
                                  if r["family"] == fam["id"] and r.get("role") == "superseded"),
                "eval_rows": sum(len(r["eval_rows"]) for r in primaries if r["family"] == fam["id"]),
                "train_rows": sum(len(r["train_rows"]) for r in primaries
                                  if r["family"] == fam["id"]),
            } for fam in families},
    }
    meta = {
        "sha256": {
            "tools/figures/extract_curves.py": sha256_file(__file__),
            "tools/figures/wandb_ro.py": sha256_file(wandb_ro.__file__),
            "families": sha256_bytes(fam_bytes),
            "protocols": sha256_bytes(prot_bytes),
        },
        "families_file": str(fam_path),
        "git_head": git_head(),
        "start_utc": start_iso,
        "end_utc": None,
        "wall_seconds": None,
        "counts": counts,
        "unidentified_count": len(unidentified),
        "unidentified_dirs": unidentified,
        "command_line": [sys.executable, "-m", "tools.figures.extract_curves"] + list(argv),
        "verify_after": bool(a.verify_after),
        "jobs": a.jobs,
    }
    meta_draft = json.dumps(meta, indent=1, sort_keys=True).encode("utf-8")
    total = sum(len(b) for b in outputs.values()) + len(meta_draft) + 256
    if total > a.max_out_mb * 1024 * 1024:
        raise LoudFailure(f"output size {total} bytes exceeds --max-out-mb {a.max_out_mb}", 1)

    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()):
        raise LoudFailure(f"refusing: --out {out_dir} exists and is not empty", 3)
    for name in sorted(outputs):
        (out_dir / name).write_bytes(outputs[name])
    meta["end_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    meta["wall_seconds"] = round(time.time() - t_start, 1)
    (out_dir / "EXTRACT_META.json").write_bytes(
        (json.dumps(meta, indent=1, sort_keys=True) + "\n").encode("utf-8"))
    print(json.dumps(counts, indent=1), file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        a = build_parser().parse_args(argv)
    except SystemExit as e:
        return 2 if e.code else 0
    try:
        return extract(a, argv)
    except LoudFailure as e:
        print(str(e), file=sys.stderr)
        return e.code


if __name__ == "__main__":
    sys.exit(main())
