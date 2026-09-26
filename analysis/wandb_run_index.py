#!/usr/bin/env python
"""wandb_run_index.py — one organised row per wandb run_name from the LOCAL datastore.

Why: 1000+ `wandb/run-*` dirs, spread over 8 projects, are only reachable today
through ad-hoc scanners (rs_extract_sigma_k_new.py, fig_F3_from_wandb.py,
scripts/monitor/build_dashboard.py, live_run_probe.py) that each re-derive
run identity. This script is the single index they can all start from.

Sources (per run dir, in priority order):
    files/config.yaml          full resolved config (project/run_name/run_group/cell_id/arch.*)
    files/wandb-metadata.json  CLI args (fallback when config.yaml is absent), host, git, start time
    files/wandb-summary.json   final logged values (probe/test_exact, all/*, train/*, _step, _runtime)
    run-<id>.wandb             binary history — read ONLY with --history (~100 MB/run)

Outputs (default under analysis/data/):
    wandb_run_index.csv        flat table, one row per run dir (dup run_names flagged)
    wandb_run_index.jsonl      same rows + full flattened config/summary
    wandb_run_index.md         grouped project → run_group → run_name (human index)
    wandb_history/<project>/<run_name>.csv   with --history: long table step,metric,value

Usage:
    uv run python analysis/wandb_run_index.py                       # index everything
    uv run python analysis/wandb_run_index.py --project power_permutation --md lab/.../index.md
    uv run python analysis/wandb_run_index.py --cloud               # add cloud state (finished/crashed/running)
    uv run python analysis/wandb_run_index.py --history --project power_permutation --group ltf \
        --metrics probe/test_exact all/exact_accuracy train/lm_loss

Provenance note: `probe/test_exact` is the EMA-512 probe metric logged during
training; it is NOT the sealed eval (see memory: test-exact-metric-provenance).
The index reports what wandb logged; it never recomputes.
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import glob
import json
import os
import sys
import time
from collections import Counter, defaultdict
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WANDB_ROOT = os.path.join(REPO, "wandb")
OUT_DIR = os.path.join(REPO, "analysis", "data")

# summary keys we lift into the flat CSV (flattened with "/")
SUMMARY_COLS = [
    "_step", "_runtime", "num_params",
    "probe/test_exact", "probe/train_exact",
    "all/exact_accuracy", "all/accuracy", "all/lm_loss",
    "train/exact_accuracy", "train/accuracy", "train/lm_loss",
    "phase/index", "z/eff_rank",
]
# config keys lifted into the flat CSV
CFG_COLS = [
    "arch/name", "arch/H_cycles", "arch/L_cycles", "arch/H_layers", "arch/L_layers",
    "arch/loops", "arch/loop_grad_cycles", "arch/loop_input_injection",
    "arch/halt_max_steps", "arch/hidden_size", "arch/mlp_t", "arch/z_carry",
    "epochs", "eval_interval", "lr", "weight_decay", "global_batch_size", "ema",
    "seed", "k", "log_z_dynamics", "load_checkpoint",
]
LIVE_WINDOW_S = 15 * 60  # .wandb mtime newer than this => "running? (local heuristic)"


# --------------------------------------------------------------------------- helpers
def flatten(d: Any, prefix: str = "", out: dict[str, Any] | None = None) -> dict[str, Any]:
    out = {} if out is None else out
    if isinstance(d, dict):
        for k, v in d.items():
            flatten(v, f"{prefix}{k}/", out)
    else:
        out[prefix[:-1]] = d
    return out


def load_config_yaml(path: str) -> dict[str, Any]:
    """wandb config.yaml stores every key as {value: ...}; unwrap it."""
    if yaml is None or not os.path.isfile(path):
        return {}
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    cfg = {}
    for k, v in raw.items():
        if k == "_wandb":
            continue
        # new format {value: X}; old format {desc: ..., value: X}
        cfg[k] = v.get("value") if isinstance(v, dict) and "value" in v else v
    for key in ("project_name", "run_name", "run_group", "cell_id"):
        if cfg.get(key) is not None and not isinstance(cfg[key], str):
            cfg[key] = json.dumps(cfg[key], sort_keys=True)
    return cfg


def parse_cli_args(args: list[str]) -> dict[str, Any]:
    """Hydra `+key=value` / `key=value` args → nested dict (fallback source)."""
    cfg: dict[str, Any] = {}
    for a in args:
        if "=" not in a:
            continue
        k, v = a.split("=", 1)
        k = k.lstrip("+~")
        v = v.strip().strip('"')
        if v.startswith("[") and v.endswith("]"):
            v = [x.strip() for x in v[1:-1].split(",") if x.strip()]
        else:
            for cast in (int, float):
                try:
                    v = cast(v)
                    break
                except ValueError:
                    pass
            if v in ("True", "true"):
                v = True
            elif v in ("False", "false"):
                v = False
        node = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            nxt = node.get(p)
            if not isinstance(nxt, dict):
                # `arch=trm` (preset string) followed by `arch.H_cycles=3`
                nxt = {"preset": nxt} if nxt is not None else {}
                node[p] = nxt
            node = nxt
        if isinstance(node.get(parts[-1]), dict) and not isinstance(v, dict):
            node[parts[-1]]["preset"] = v
        else:
            node[parts[-1]] = v
    # `arch=trm` is the arch *preset* name; keep it addressable
    if isinstance(cfg.get("arch"), str):
        cfg["arch"] = {"preset": cfg["arch"]}
    return cfg


def infer_n_k_from_data(data_paths: Any) -> tuple[Any, Any]:
    """data/sigma_k_10/3 → (10, 3); data/sigma_k/5 → (20, 5); data/power_permutation/d1/10 → (10, 10)."""
    if isinstance(data_paths, list) and data_paths:
        p = str(data_paths[0])
    elif isinstance(data_paths, str):
        p = data_paths
    else:
        return None, None
    parts = p.rstrip("/").split("/")
    try:
        if "sigma_k_10" in parts:
            return 10, int(parts[-1])
        if "sigma_k" in parts:
            return 20, int(parts[-1])
        if "power_permutation" in parts:
            return 10, int(parts[-1])
    except ValueError:
        pass
    return None, None


def run_id_from_dir(d: str) -> str:
    return os.path.basename(d).rsplit("-", 1)[-1]


# --------------------------------------------------------------------------- scan
def scan_run_dir(d: str) -> dict[str, Any] | None:
    files = os.path.join(d, "files")
    meta_p = os.path.join(files, "wandb-metadata.json")
    if not os.path.isfile(meta_p):
        return None
    try:
        meta = json.load(open(meta_p))
    except json.JSONDecodeError:
        return None

    cfg = load_config_yaml(os.path.join(files, "config.yaml"))
    cfg_src = "config.yaml"
    if not cfg:
        cfg = parse_cli_args(meta.get("args") or [])
        cfg_src = "metadata.args"
    fcfg = flatten(cfg)
    # arch preset (trm / trm_singlez / transformers_baseline) is only in the CLI args
    preset = next((a.split("=", 1)[1] for a in meta.get("args") or [] if a.startswith("arch=")), None)

    summary: dict[str, Any] = {}
    sum_p = os.path.join(files, "wandb-summary.json")
    if os.path.isfile(sum_p):
        try:
            summary = flatten(json.load(open(sum_p)))
        except json.JSONDecodeError:
            summary = {}

    wandb_bin = glob.glob(os.path.join(d, "*.wandb"))
    bin_mtime = max((os.path.getmtime(p) for p in wandb_bin), default=None)
    bin_size = sum(os.path.getsize(p) for p in wandb_bin)
    age = (time.time() - bin_mtime) if bin_mtime else None
    if age is not None and age < LIVE_WINDOW_S:
        status = "running?"
    elif summary:
        status = "ended"
    else:
        status = "no-summary"

    n, k_data = infer_n_k_from_data(cfg.get("data_paths"))
    dp = cfg.get("data_paths")
    row: dict[str, Any] = {
        "run_name": cfg.get("run_name") or os.path.basename(d),
        "project": cfg.get("project_name") or "",
        "run_group": cfg.get("run_group") or "",
        "cell_id": cfg.get("cell_id") or "",
        "wandb_run_id": run_id_from_dir(d),
        "run_dir": os.path.relpath(d, REPO),
        "mode": "offline" if os.path.basename(d).startswith("offline") else "online",
        "status_local": status,
        "started_at": meta.get("startedAt"),
        "host": meta.get("host"),
        "git_commit": (meta.get("git") or {}).get("commit", "")[:12],
        "arch_preset": preset,
        "data_paths": ";".join(map(str, dp)) if isinstance(dp, list) else dp,
        "n": n,
        "k_from_data": k_data,
        "config_source": cfg_src,
        "wandb_bin_mb": round(bin_size / 1e6, 1),
    }
    for c in CFG_COLS:
        row["cfg:" + c] = fcfg.get(c)
    for c in SUMMARY_COLS:
        row["sum:" + c] = summary.get(c)
    row["_config"] = fcfg
    row["_summary"] = summary
    return row


def mark_duplicates(rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(r["project"], r["run_name"])].append(r)
    for g in groups.values():
        g.sort(key=lambda r: ((r["sum:_step"] or -1), r["started_at"] or ""), reverse=True)
        for i, r in enumerate(g):
            r["dup_count"] = len(g)
            r["is_primary"] = i == 0


def add_cloud_state(rows: list[dict[str, Any]], entity: str) -> None:
    import wandb  # lazy: only when asked

    api = wandb.Api(timeout=60)
    by_id = {r["wandb_run_id"]: r for r in rows}
    for project in sorted({r["project"] for r in rows if r["project"]}):
        try:
            for run in api.runs(f"{entity}/{project}", per_page=500):
                r = by_id.get(run.id)
                if r is not None:
                    r["cloud_state"] = run.state
                    r["cloud_url"] = run.url
        except Exception as e:  # network / permission
            print(f"[cloud] {project}: {e}", file=sys.stderr)
    for r in rows:
        r.setdefault("cloud_state", "")
        r.setdefault("cloud_url", "")


# --------------------------------------------------------------------------- history
def dump_history(rows: list[dict[str, Any]], metrics: list[str], out_root: str) -> int:
    from wandb.proto import wandb_internal_pb2 as pb
    from wandb.sdk.internal import datastore

    want = set(metrics)
    n_written = 0
    for r in rows:
        bins = glob.glob(os.path.join(r["_wandb_root"], os.path.basename(r["run_dir"]), "*.wandb"))
        if not bins:
            continue
        ds = datastore.DataStore()
        ds.open_for_scan(bins[0])
        by_step: dict[int, dict[str, Any]] = {}
        while True:
            try:
                rec = ds.scan_data()
            except Exception:
                break
            if rec is None:
                break
            pr = pb.Record()
            pr.ParseFromString(rec)
            if pr.WhichOneof("record_type") != "history":
                continue
            d = {}
            for it in pr.history.item:
                key = it.key if it.key else "/".join(it.nested_key)
                if key == "_step" or key in want:
                    d[key] = it.value_json
            if "_step" in d and len(d) > 1:
                s = int(json.loads(d["_step"]))
                by_step.setdefault(s, {}).update({k: json.loads(v) for k, v in d.items() if k != "_step"})
        out_dir = os.path.join(out_root, r["project"] or "_noproject")
        os.makedirs(out_dir, exist_ok=True)
        suffix = "" if r.get("is_primary", True) else f"__{r['wandb_run_id']}"
        out_p = os.path.join(out_dir, f"{r['run_name']}{suffix}.csv")
        with open(out_p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "metric", "value"])
            for s in sorted(by_step):
                for m, v in by_step[s].items():
                    w.writerow([s, m, v])
        n_written += 1
        print(f"[history] {out_p} ({len(by_step)} steps)")
    return n_written


# --------------------------------------------------------------------------- outputs
def fmt(v: Any) -> str:
    if v is None or v == "":
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 1e4 else f"{v:.3g}"
    return str(v)


def write_markdown(rows: list[dict[str, Any]], path: str, stats: Counter) -> None:
    by_proj: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_proj[r["project"] or "(no project)"][r["run_group"] or "(ungrouped)"].append(r)
    lines = [
        "# wandb run index (local datastore)",
        "",
        f"생성: {time.strftime('%Y-%m-%d %H:%M %z')} · 생성기 `analysis/wandb_run_index.py` · "
        "근거 `wandb/*run-*/files/{config.yaml,wandb-metadata.json,wandb-summary.json}`",
        "",
        "| 기호 | 정의 |",
        "|---|---|",
        "| `test_exact` | `probe/test_exact` — 훈련 중 EMA-512 probe의 exact-match 정확도(sealed eval 아님) |",
        "| `all/exact` | `all/exact_accuracy` — eval 루프 전체 test set exact-match |",
        "| `step` | `_step` — wandb summary의 마지막 로그 step |",
        "| `status` | `ended`(summary 있음) / `running?`(.wandb mtime < 15 min) / `no-summary` |",
        "| `dup` | 같은 (project, run_name)의 run dir 수; `*`=primary(최대 step) |",
        "",
        "## 요약",
        "",
        f"- run dirs 스캔 {stats['dirs']} · 인덱스 {len(rows)} · 메타데이터 없음 {stats['nometa']} · "
        f"중복 run_name {stats['dup_names']}",
        "",
        "| project | runs | groups | ended | running? | no-summary |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    order = sorted(by_proj.items(), key=lambda kv: -sum(len(v) for v in kv[1].values()))
    for p, groups in order:
        rs = [r for g in groups.values() for r in g]
        c = Counter(r["status_local"] for r in rs)
        lines.append(f"| {p} | {len(rs)} | {len(groups)} | {c['ended']} | {c['running?']} | {c['no-summary']} |")
    lines.append("")
    for p, groups in order:
        lines += [f"## project `{p}`", ""]
        for g, rs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            lines += [
                f"### group `{g}` ({len(rs)} runs)",
                "",
                "| run_name | arch | k | seed | cell_id | step | test_exact | all/exact | status | dup | started | id |",
                "|---|---|---:|---:|---|---:|---:|---:|---|---|---|---|",
            ]
            rs = sorted(rs, key=lambda r: (r["run_name"], not r.get("is_primary", True)))
            for r in rs:
                k = r["cfg:k"] if r["cfg:k"] is not None else r["k_from_data"]
                arch = r["arch_preset"] or (str(r["cfg:arch/name"]).split("@")[-1] if r["cfg:arch/name"] else "")
                dup = f"{r['dup_count']}{'*' if r['is_primary'] else ''}" if r["dup_count"] > 1 else "1"
                lines.append(
                    f"| {r['run_name']} | {fmt(arch)} | {fmt(k)} | {fmt(r['cfg:seed'])} | {fmt(r['cell_id'])} | "
                    f"{fmt(r['sum:_step'])} | {fmt(r['sum:probe/test_exact'])} | {fmt(r['sum:all/exact_accuracy'])} | "
                    f"{r['status_local']} | {dup} | {(r['started_at'] or '')[:10]} | {r['wandb_run_id']} |"
                )
            lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wandb-root", default=WANDB_ROOT)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--md", default=None, help="markdown index path (default <out-dir>/wandb_run_index.md)")
    ap.add_argument("--project", action="append", help="restrict to project(s)")
    ap.add_argument("--group", action="append", help="restrict to run_group(s)")
    ap.add_argument("--run-name", default=None, help="glob on run_name, e.g. 'pp_ltf_*'")
    ap.add_argument("--cloud", action="store_true", help="merge cloud run state/url via wandb.Api (needs ~/.netrc)")
    ap.add_argument("--entity", default="raon1123")
    ap.add_argument("--history", action="store_true", help="dump per-run metric history from .wandb (slow: ~100MB/run)")
    ap.add_argument("--metrics", nargs="+", default=["probe/test_exact", "all/exact_accuracy", "train/lm_loss"])
    ap.add_argument("--primary-only", action="store_true", help="with --history: skip duplicate (non-primary) run dirs")
    args = ap.parse_args()

    stats: Counter = Counter()
    rows: list[dict[str, Any]] = []
    for d in sorted(glob.glob(os.path.join(args.wandb_root, "*run-*"))):
        if os.path.islink(d) or not os.path.isdir(d):
            continue
        stats["dirs"] += 1
        r = scan_run_dir(d)
        if r is None:
            stats["nometa"] += 1
            continue
        r["_wandb_root"] = args.wandb_root
        rows.append(r)
    if not rows:
        sys.exit(f"FATAL: no runs indexed under {args.wandb_root}")
    mark_duplicates(rows)  # on the full set, so dup flags are global
    stats["dup_names"] = len({(r["project"], r["run_name"]) for r in rows if r["dup_count"] > 1})

    if args.project:
        rows = [r for r in rows if r["project"] in args.project]
    if args.group:
        rows = [r for r in rows if r["run_group"] in args.group]
    if args.run_name:
        rows = [r for r in rows if fnmatch.fnmatch(r["run_name"], args.run_name)]
    if not rows:
        sys.exit("FATAL: filter matched 0 runs")
    if args.cloud:
        add_cloud_state(rows, args.entity)

    os.makedirs(args.out_dir, exist_ok=True)
    flat_cols = [c for c in rows[0] if not c.startswith("_")]
    csv_p = os.path.join(args.out_dir, "wandb_run_index.csv")
    with open(csv_p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=flat_cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["project"], r["run_group"], r["run_name"], r["started_at"] or "")):
            w.writerow(r)
    jsonl_p = os.path.join(args.out_dir, "wandb_run_index.jsonl")
    with open(jsonl_p, "w") as f:
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items() if k != "_wandb_root"}, default=str) + "\n")
    md_p = args.md or os.path.join(args.out_dir, "wandb_run_index.md")
    os.makedirs(os.path.dirname(os.path.abspath(md_p)), exist_ok=True)
    write_markdown(rows, md_p, stats)

    c = Counter(r["status_local"] for r in rows)
    print(f"indexed {len(rows)} runs from {stats['dirs']} dirs (no metadata: {stats['nometa']}); "
          f"ended {c['ended']} / running? {c['running?']} / no-summary {c['no-summary']}; "
          f"dup run_names {stats['dup_names']}")
    print(f"  csv   {csv_p}\n  jsonl {jsonl_p}\n  md    {md_p}")

    if args.history:
        hrows = [r for r in rows if r["is_primary"]] if args.primary_only else rows
        n = dump_history(hrows, args.metrics, os.path.join(args.out_dir, "wandb_history"))
        print(f"history dumped for {n} runs → {os.path.join(args.out_dir, 'wandb_history')}")
        if n == 0:
            sys.exit("FATAL: --history wrote 0 files")


if __name__ == "__main__":
    main()
