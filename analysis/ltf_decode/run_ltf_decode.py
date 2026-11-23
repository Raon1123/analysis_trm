"""P-C1 CLI: per-loop decode of the g7 / full final checkpoints (spec §4.1). CPU only, fail-closed.

  CUDA_VISIBLE_DEVICES="" uv run --no-sync python -m analysis.ltf_decode.run_ltf_decode --device cpu \
      --manifest <M1_manifest.csv> --gc-output <gc_timing.out> --prereg <frozen prereg> --expected-sha256 <sha> \
      --ckpt-root checkpoints/power_permutation --threads 32 --out <empty dir>

NOT RUN before the freeze (I-12): this file is exercised only on synthetic random weights in tests.

Per cell: tracer over n_loops (42 if the G-c line says ``EXTRAP: KEEP`` else 21) on test 1000 rows (+ train first
512 rows for the D-4 cells = every non-G cell of the manifest), G1' against the native forward at t = 21, G-P3,
cell statistics over T_stat = 1..20 (J = 20, J_all = 21), descriptive D-grid cell statistics (J = 17, J_all = 18),
descriptive t = 22..42 per-loop rows, t*_i(k), D-9 residues (near_power.jsonl). All accuracies are "fourth protocol"
(decode-1000 / loop t); only G-P3 reports the P3 protocol.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import re
import sys
import time

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(os.path.dirname(PKG_DIR))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from analysis.ltf_decode import cells as cellsmod  # noqa: E402
from analysis.ltf_decode import gates, judge  # noqa: E402
from analysis.ltf_decode.stats_j import analyze_decodes_j, tstar_first_hit  # noqa: E402
from analysis.ltf_decode.trace import load_ltf, native_forward, trace_loops  # noqa: E402
from analysis.z_dynamics import readouts, stats as zstats  # noqa: E402

EXP = judge.EXP_ID   # single constant in judge.py; "PC1-DRAFT" until the EXP id is allocated
PER_CALL_FIELDS = ["exp_id", "unit", "run_name", "split", "readout_kind", "t", "accuracy_protocol", "a", "a_self",
                   "v", "trivial", "near_power_frac", "proper_partial_frac", "none_frac", "tie_frac", "gcd_class",
                   "n_rows"]
CELL_FIELDS = ["exp_id", "unit", "run_name", "arm", "k", "seed", "frozen_class", "split", "readout_kind", "J", "J_all",
               "n_rows", "a1", "a17", "R", "maxdrop", "maxrise", "M", "subclass", "V", "T", "gamma", "p", "P_int",
               "P_int0", "thr", "informative_frac", "n_informative", "n_mixture_rejected", "Z", "Z_raw", "Vis",
               "vis_class", "vis_in_shortest_chain", "jstar"]
TSTAR_FIELDS = ["exp_id", "unit", "run_name", "split", "gcd_class", "n_rows", "undefined_frac", "median", "q25", "q75"]
HASHED = ("judge.py", "stats_j.py", "trace.py", "gates.py", "cells.py", "run_ltf_decode.py")
FOURTH = "fourth protocol (decode-1000 / loop t)"


# ------------------------------------------------------------------ fail-closed CPU guard (spec §4.1)

def require_cpu(device: str) -> None:
    """SystemExit(2) unless CUDA_VISIBLE_DEVICES == "" AND device == "cpu" AND torch.cuda.is_available() is False."""
    problems = []
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        problems.append(f'CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")!r} (must be "")')
    if device != "cpu":
        problems.append(f"--device {device!r} (must be 'cpu')")
    if torch.cuda.is_available():
        problems.append("torch.cuda.is_available() is True")
    if problems:
        print("CPU fail-closed: refusing: " + "; ".join(problems), file=sys.stderr)
        raise SystemExit(2)


# ------------------------------------------------------------------ small helpers

def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def fmt(v):
    if v is None:
        return ""
    if isinstance(v, (list, tuple, set)):
        return ";".join(str(x) for x in sorted(v))
    if isinstance(v, np.floating):
        return repr(float(v))
    return v


def rows_to_text(rows: list[dict], fields: list[str]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    for r in rows:
        w.writerow({k: fmt(r.get(k)) for k in fields})
    return buf.getvalue()


def parse_extrap_line(path: str) -> str:
    """Read the G-c output file; the decision is its ``EXTRAP: KEEP|DROP`` line (never a human choice)."""
    with open(path) as f:
        found = re.findall(r"^EXTRAP: (KEEP|DROP)\b", f.read(), flags=re.M)
    if len(found) != 1:
        raise SystemExit(f"{path}: expected exactly one 'EXTRAP: KEEP|DROP' line, found {found}")
    return found[0]


def load_split(data_path: str, split: str, n: int):
    from utils.z_logging import _load_probe_tensors
    sd = gates.guard_path(os.path.join(data_path, split))
    with open(os.path.join(sd, "dataset.json")) as f:
        ignore = json.load(f)["ignore_label_id"]
    probe, fp = _load_probe_tensors(data_path, split, n, ignore)
    if probe["inputs"].shape[0] != n:
        raise RuntimeError(f"{data_path}/{split}: only {probe['inputs'].shape[0]} rows, need {n}")
    return probe, fp


# ------------------------------------------------------------------ forward -> decodes

def decode_loops(inner, probe: dict, n_loops: int, batch_size: int, check_g1_at: int | None = None):
    """(dec (N, n_loops, 10) int64 tokens, g1 bool). g1 compares the tracer's loop-``check_g1_at`` logits with the
    native forward (torch.equal) per batch."""
    N = probe["inputs"].shape[0]
    dec = np.zeros((N, n_loops, 10), np.int64)
    g1 = True
    with torch.inference_mode():
        for s in range(0, N, batch_size):
            e = min(N, s + batch_size)
            batch = {kk: v[s:e] for kk, v in probe.items()}
            tr = trace_loops(inner, batch, n_loops)
            if check_g1_at is not None:
                _z, ref = native_forward(inner, batch)
                g1 &= gates.g1_prime(tr.logits[check_g1_at - 1], ref)
            for t in range(n_loops):
                dec[s:e, t] = readouts.decode(tr.logits[t])
            del tr
    return dec, bool(g1)


# ------------------------------------------------------------------ per-split statistics (shared with gc_timing)

def analyze_split(dec: np.ndarray, inp: np.ndarray, lab: np.ndarray, k: int, infos: list, n_perm: int | None,
                  base: dict, split: str, with_dgrid: bool = True, with_extrap: bool = True) -> dict:
    """dec: (N, W, 10), W = 21 (+21 when extrapolating). Rows for per_call / cell / tstar, plus the near-power
    structure of the main readouts. Thresholds from judge (parsed FROZEN_RULES)."""
    t_all = judge.t_all()
    n_all, J = len(t_all), len(judge.t_stat())
    main = dec[:, :n_all]
    ref = main[:, -1]
    out = {"per_call": [], "cell": [], "tstar": []}
    an = analyze_decodes_j(main, inp, lab, k, J=J, J_all=n_all, infos=infos, n_perm=n_perm, ref_dec=ref,
                           j_labels=t_all, with_subclass=False)
    for pj in an["per_j"]:
        out["per_call"].append(dict(base, split=split, readout_kind="ltf", t=pj["j"], accuracy_protocol=FOURTH,
                                    **{kk: v for kk, v in pj.items() if kk != "j"}))
    out["cell"].append(dict(base, split=split, readout_kind="ltf", J=J, J_all=n_all, **an["cell"]))
    out["near"] = an["near"]
    if with_dgrid:
        dg = judge.dgrid()
        an_d = analyze_decodes_j(main[:, [t - 1 for t in dg]], inp, lab, k, J=len(dg) - 1, J_all=len(dg), infos=infos,
                                 n_perm=n_perm, ref_dec=ref, j_labels=dg, with_subclass=False)
        out["cell"].append(dict(base, split=split, readout_kind="dgrid", J=len(dg) - 1, J_all=len(dg), **an_d["cell"]))
    ext = judge.extrapolation_range()
    if with_extrap and ext is not None and dec.shape[1] > n_all:
        an_e = analyze_decodes_j(dec[:, n_all:n_all + len(ext)], inp, lab, k, J=1, J_all=1, infos=infos,
                                 ref_dec=ref, j_labels=ext, cell_stats=False)
        for pj in an_e["per_j"]:
            out["per_call"].append(dict(base, split=split, readout_kind="extrap", t=pj["j"], accuracy_protocol=FOURTH,
                                        **{kk: v for kk, v in pj.items() if kk != "j"}))
    # t*_i(k): first loop (over every traced loop) whose whole decode equals sigma^k; descriptive only
    tl = list(range(1, dec.shape[1] + 1))
    ts = tstar_first_hit(dec, lab, tl)
    gcd1 = np.array([s.gcd == 1 for s in infos])
    for gname, mask in (("all", np.ones(len(ts), bool)), ("1", gcd1), (">1", ~gcd1)):
        if not mask.any():
            continue
        v = ts[mask]
        d = v[v >= 0]
        q = (np.percentile(d, [25, 50, 75]) if d.size else [float("nan")] * 3)
        out["tstar"].append(dict(base, split=split, gcd_class=gname, n_rows=int(mask.sum()),
                                 undefined_frac=float((v < 0).mean()), median=float(q[1]), q25=float(q[0]),
                                 q75=float(q[2])))
    return out


def near_text(lines: list[dict]) -> str:
    """near_power.jsonl text exactly as main() writes it (G5' compares it too)."""
    return "".join(json.dumps(line) + "\n" for line in lines)


def near_lines(dst: list, base: dict, split: str, near: dict, k: int) -> None:
    ms, hm, tie, none, prop = near["m_star"], near["ham_min"], near["tie"], near["none"], near["proper"]
    for i, s in enumerate(near["infos"]):
        res = near["residues"][i]
        for t in range(ms.shape[1]):
            m = int(ms[i, t])
            dst.append({"unit": base["unit"], "split": split, "i": i, "t": t + 1, "m_star": None if m < 0 else m,
                        "ham_min": int(hm[i, t]), "tie": bool(tie[i, t]), "none": bool(none[i, t]),
                        "proper": bool(prop[i, t]), "ord": s.o, "cycle_type": list(s.ctype), "gcd": s.gcd,
                        "informative": s.informative, "residues": [r[t] for r in res]})


# ------------------------------------------------------------------ one cell (pure part: model + arrays)

def analyze_cell(inner, k: int, test: dict, train: dict | None, n_loops: int, batch: int, n_perm: int | None,
                 base: dict, manifest_terminal_probe: float | None = None, with_extrap: bool = True) -> dict:
    """Decode + gates G1'/G-P3 + statistics for one cell. ``train`` None -> test only (non-D-4 cell)."""
    out = {"per_call": [], "cell": [], "tstar": [], "near": []}
    rec = {}
    splits = [("test", test)] + ([("train", train)] if train is not None else [])
    for split, probe in splits:
        dec, g1 = decode_loops(inner, probe, n_loops, batch, check_g1_at=inner.config.loops)
        rec[f"G1_{split}"] = g1
        inp = probe["inputs"].numpy()[:, :10]
        lab = probe["labels"].numpy()[:, :10]
        if split == "test" and manifest_terminal_probe is not None:
            rec["G_P3"] = gates.g_p3(dec[:512, inner.config.loops - 1], lab[:512], manifest_terminal_probe)
        infos = readouts.sigma_infos(inp, k)
        res = analyze_split(dec, inp, lab, k, infos, n_perm, base, split, with_extrap=with_extrap)
        for key in ("per_call", "cell", "tstar"):
            out[key] += res[key]
        near_lines(out["near"], base, split, res["near"], k)
        if split == "test":
            rec["test_inputs_sha"] = hashlib.sha256(inp.tobytes()).hexdigest()[:12]
    rec["G1"] = all(v for kk, v in rec.items() if kk.startswith("G1_"))
    out["rec"] = rec
    return out


# ------------------------------------------------------------------ one cell (IO part)

def draft_checkpoint_guard(ckpt_root: str) -> None:
    """DRAFT mode (judge.FROZEN_PREREG_SHA256 is None) must never load a checkpoint from the repo's checkpoints/ tree."""
    if judge.DRAFT_MODE:
        real = os.path.realpath(ckpt_root)
        repo_ck = os.path.realpath(os.path.join(PKG_ROOT, "checkpoints"))
        if real == repo_ck or real.startswith(repo_ck + os.sep):
            print(f"DRAFT mode: refusing to read checkpoints under {repo_ck} (I-12)", file=sys.stderr)
            raise SystemExit(2)


def process_cell(row: dict, a, extrap: bool) -> dict:
    draft_checkpoint_guard(a.ckpt_root)
    name, arm, k = row["run_name"], row["arm"], int(row["k"])
    unit = f"{EXP}__{name}__step_{cellsmod.TERMINAL_STEP}"
    run_dir = os.path.join(a.ckpt_root, name)
    ck = os.path.join(run_dir, f"step_{cellsmod.TERMINAL_STEP}")
    rec = {"run_name": name, "unit": unit, "arm": arm, "k": k, "status": "excluded", "exclusion_reason": None}
    out = {"rec": rec, "per_call": [], "cell": [], "tstar": [], "near": []}
    if not row["terminal_ok"]:
        rec["exclusion_reason"] = "manifest M-1 terminal value not ok"
        return out
    steps = sorted(int(d.split("_")[1]) for d in os.listdir(run_dir)
                   if re.fullmatch(r"step_\d+", d))
    rec["I4"] = gates.i4_terminal(cellsmod.TERMINAL_STEP, steps)
    for p in (ck, os.path.join(run_dir, "all_config.yaml")):
        if not os.path.exists(p):
            rec["exclusion_reason"] = f"missing {p}"
            return out
    if not rec["I4"]["pass"]:
        rec["exclusion_reason"] = "I-4 non-terminal checkpoint"
        return out

    from measure_rho import build_model_config, load_all_config
    all_cfg = load_all_config(run_dir)
    ref_dir = os.path.join(a.ckpt_root, f"pp_base_tf_z_iter_k{k}_s{row['seed']}")
    ref_cfg = load_all_config(ref_dir) if os.path.isfile(os.path.join(ref_dir, "all_config.yaml")) else None
    rec["G7"] = gates.g7_config(all_cfg, ref_cfg, arm)
    data_path = all_cfg["data_paths"][0]
    gates.guard_path(data_path)
    if int(os.path.basename(data_path)) != k:
        raise RuntimeError(f"{name}: data path {data_path} disagrees with manifest k")
    rec["G0"] = gates.g0_code_identity(run_dir)
    cfg = build_model_config(run_dir, batch_size=a.batch)
    _w, inner, missing, unexpected = load_ltf(ck, cfg, a.device)
    rec["I3"] = gates.i3_keys(missing, unexpected)
    if not rec["I3"]["pass"]:
        rec["exclusion_reason"] = "I-3 state-dict key mismatch"
        return out

    d4 = row["class"] != "G"
    test, test_fp = load_split(data_path, "test", a.test_n)
    train, train_fp = (load_split(data_path, "train", a.probe_n) if d4 else (None, None))
    rec["probe_fingerprints"] = {"test": test_fp, "train": train_fp}
    g6t = gates.g6_data(test["inputs"].numpy()[:, :10], test["labels"].numpy()[:, :10], k)
    g6r = (gates.g6_data(train["inputs"].numpy()[:, :10], train["labels"].numpy()[:, :10], k) if d4 else None)
    rec["G6"] = {"test": g6t, "train": g6r, "pass": bool(g6t["pass"] and (g6r is None or g6r["pass"]))}
    base = {"exp_id": EXP, "unit": unit, "run_name": name, "arm": arm, "k": k, "seed": row["seed"],
            "frozen_class": row["class"]}
    n_loops = 42 if extrap else 21
    res = analyze_cell(inner, k, test, train, n_loops, a.batch, a.n_perm, base,
                       manifest_terminal_probe=float(row["final_probe_test_exact"]), with_extrap=extrap)
    rec.update(res["rec"])
    for key in ("per_call", "cell", "tstar", "near"):
        out[key] = res[key]
    rec["pass_gates"] = bool(rec["G0"]["pass"] and rec["G7"]["pass"] and rec["G6"]["pass"] and rec["G1"]
                             and rec["G_P3"]["pass"])
    rec["status"] = "analysed" if rec["pass_gates"] else "excluded"
    if not rec["pass_gates"]:
        rec["exclusion_reason"] = "gate failure (see gates.json)"
    return out


# ------------------------------------------------------------------ main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="M-1 manifest CSV (cells.py classify)")
    ap.add_argument("--gc-output", required=True, help="G-c output file holding the EXTRAP: KEEP|DROP line")
    ap.add_argument("--prereg", default=judge.DEFAULT_PREREG)
    ap.add_argument("--expected-sha256", default=judge.FROZEN_PREREG_SHA256, help="sha256 of the FROZEN prereg (I-7); default judge.FROZEN_PREREG_SHA256")
    ap.add_argument("--ckpt-root", default="checkpoints/power_permutation")
    ap.add_argument("--repo-root", default=os.getcwd())
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=250)
    ap.add_argument("--test-n", type=int, default=1000)
    ap.add_argument("--probe-n", type=int, default=512)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--only", default="")
    ap.add_argument("--no-g5", action="store_true")
    ap.add_argument("--dry-synthetic", action="store_true",
                    help="DRAFT mode only: allow a run on NON-checkpoint (synthetic / tmp) inputs, e.g. tests")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    require_cpu(a.device)
    if judge.DRAFT_MODE:
        if not a.dry_synthetic:
            print("DRAFT mode (judge.FROZEN_PREREG_SHA256 is None): refusing to run without --dry-synthetic", file=sys.stderr)
            return 2
        draft_checkpoint_guard(a.ckpt_root)
    if a.expected_sha256 is None or (judge.FROZEN_PREREG_SHA256 and a.expected_sha256 != judge.FROZEN_PREREG_SHA256):
        print("refusing (I-7): a sha256 equal to judge.FROZEN_PREREG_SHA256 is required", file=sys.stderr)
        return 2
    try:
        rules = judge.load_rules(a.prereg, a.expected_sha256)
    except judge.JudgeAbort as e:
        print(f"refusing (I-7): {e}", file=sys.stderr)
        return 2
    judge.FROZEN_RULES.clear()
    judge.FROZEN_RULES.update(rules)       # the run uses exactly the rules of the sha-checked prereg (I-9)

    a.out, a.manifest, a.gc_output = os.path.abspath(a.out), os.path.abspath(a.manifest), os.path.abspath(a.gc_output)
    os.chdir(a.repo_root)
    a.ckpt_root = os.path.abspath(a.ckpt_root)
    if a.threads:
        torch.set_num_threads(a.threads)
    frozen_args = {"n_perm": int(judge.num("czz2.N_PERM")), "test_n": 1000, "probe_n": 512}
    dev = {kk: getattr(a, kk) for kk, v in frozen_args.items() if getattr(a, kk) != v}
    if dev:
        raise SystemExit(f"arguments deviate from the frozen protocol: {dev} (expected {frozen_args})")
    extrap = parse_extrap_line(a.gc_output) == "KEEP"
    rows = cellsmod.read_manifest(a.manifest)
    if a.only:
        want = [x.strip() for x in a.only.split(",") if x.strip()]
        unknown = set(want) - {r["run_name"] for r in rows}
        if unknown:
            raise SystemExit(f"--only names not in manifest: {sorted(unknown)}")
        rows = [r for r in rows if r["run_name"] in want]
    if os.path.isdir(a.out) and os.listdir(a.out):
        raise SystemExit(f"--out {a.out} is not empty; I-8 requires a from-scratch run")
    os.makedirs(a.out, exist_ok=True)
    g5_unit = judge.rule("gates.G5_unit")
    rows = sorted(rows, key=lambda r: r["run_name"] != g5_unit)

    manifest = {"exp_id": EXP, "status": "running", "partial": bool(a.only), "args": vars(a), "extrap": extrap,
                "prereg_sha256": sha256(a.prereg), "manifest_sha256": sha256(a.manifest),
                "gc_output_sha256": sha256(a.gc_output),
                "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "sha256": {f: sha256(os.path.join(PKG_DIR, f)) for f in HASHED},
                "torch": torch.__version__, "python": platform.python_version(), "device": a.device,
                "threads": torch.get_num_threads(), "host": platform.node(), "cell_seconds": {}}
    man_path = os.path.join(a.out, "run_manifest.json")

    def write_manifest():
        with open(man_path, "w") as f:
            json.dump(manifest, f, indent=1, default=str)

    write_manifest()
    files = {"per_call": ("per_call.csv", PER_CALL_FIELDS), "cell": ("cell_stats.csv", CELL_FIELDS),
             "tstar": ("tstar.csv", TSTAR_FIELDS)}
    for key, (fn, fields) in files.items():
        with open(os.path.join(a.out, fn), "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields, lineterminator="\n").writeheader()
    open(os.path.join(a.out, "near_power.jsonl"), "w").close()
    recs, g5_first = {}, None
    try:
        for n, row in enumerate(rows):
            t0 = time.time()
            res = process_cell(row, a, extrap)
            dt = time.time() - t0
            manifest["cell_seconds"][row["run_name"]] = dt
            recs[row["run_name"]] = res["rec"]
            for key, (fn, fields) in files.items():
                with open(os.path.join(a.out, fn), "a", newline="") as f:
                    f.write(rows_to_text(res[key], fields))
            with open(os.path.join(a.out, "near_power.jsonl"), "a") as f:
                for line in res["near"]:
                    f.write(json.dumps(line) + "\n")
            if row["run_name"] == g5_unit:
                g5_first = {kk: rows_to_text(res[kk], files[kk][1]) for kk in ("per_call", "cell", "tstar")}
                g5_first["near_power"] = near_text(res["near"])
            print(f"[{n + 1}/{len(rows)}] {row['run_name']} status={res['rec']['status']} {dt:.0f}s", flush=True)
            write_manifest()
        g5 = {"unit": g5_unit, "ran": False, "pass": None}
        if not a.no_g5 and g5_first is not None:
            zstats._OMEGA_CACHE.clear()              # cold re-run (G5' end-to-end)
            readouts._cached_info.cache_clear()
            again = process_cell(next(r for r in rows if r["run_name"] == g5_unit), a, extrap)
            second = {kk: rows_to_text(again[kk], files[kk][1]) for kk in ("per_call", "cell", "tstar")}
            second["near_power"] = near_text(again["near"])
            g5 = dict(gates.g5_identical(g5_first, second), unit=g5_unit, ran=True)
        with open(os.path.join(a.out, "gates.json"), "w") as f:
            json.dump({"exp_id": EXP, "cells": recs, "G5": g5}, f, indent=1, default=str)
        manifest["status"] = "partial" if a.only else "complete"
        manifest["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        write_manifest()
    except BaseException:
        manifest["status"] = "failed"
        write_manifest()
        raise
    print(f"done -> {a.out} (status {manifest['status']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
