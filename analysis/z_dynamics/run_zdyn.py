"""EXP-017 CLI orchestrator: instrumented forward over the 41 frozen cells -> data files for judge.py.

Pre-registered command (EXP-017 §4.5; run from the repository root, CPU):
  uv run python analysis/z_dynamics/run_zdyn.py --cells analysis/z_dynamics/EXP017_cells.csv \
      --ckpt-root checkpoints/power_permutation --device cpu --batch 250 \
      --test-n 1000 --probe-n 512 --lens-train-n 2048 --n-perm 2000 --seed 20260923 \
      --out reports/figures/${TODAY}_z-dynamics/data

Outputs (data only): gates.json, per_call.csv, near_power.jsonl, soft_mass.csv, pr_traj.csv,
cell_stats.csv, run_manifest.json (status "complete" is written last; judge.py refuses anything else, I-8).

Smoke (loading + hook shapes only; computes NO statistic, accuracy or PR):
  uv run python analysis/z_dynamics/run_zdyn.py --smoke --only pp_base_tf_z_iter_k5_s1 --smoke-n 8 \
      --out /home/ayp/.claude/jobs/69e3c574/tmp/zdyn_smoke

Order: the G5 unit is processed first so its wall time can be extrapolated before the full run
(VERIFY-r1 N10). It is re-run at the end and compared for identical outputs (G5).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import time

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(os.path.dirname(PKG_DIR))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from analysis.z_dynamics import cells as cellsmod  # noqa: E402
from analysis.z_dynamics import gates, judge, readouts, stats, tuned_lens  # noqa: E402
from analysis.z_dynamics.instrumented_forward import (forward_trace, g1_equal, g2_equal, load_inner,  # noqa: E402
                                                      native_logits)
from analysis.z_dynamics.pr_traj import pr_of_latent, pr_rows  # noqa: E402

EXP = "EXP-017"
PER_CALL_FIELDS = ["exp_id", "unit", "run_name", "split", "readout_kind", "j", "c", "native", "a", "a_self", "v",
                   "trivial", "near_power_frac", "proper_partial_frac", "none_frac", "tie_frac", "gcd_class", "n_rows"]
PR_FIELDS = ["exp_id", "unit", "run_name", "split", "stream", "j", "h", "pr", "n_probe"]
SOFT_FIELDS = ["exp_id", "unit", "run_name", "split", "readout_kind", "j", "m", "mass"]
CELL_FIELDS = ["exp_id", "unit", "run_name", "arch", "protocol", "k", "frozen_class", "split", "readout_kind",
               "n_rows", "a1", "a17", "R", "maxdrop", "maxrise", "M", "subclass", "V", "T", "gamma", "p", "P_int",
               "P_int0", "thr", "informative_frac", "n_informative", "n_mixture_rejected", "Z", "Z_raw", "Vis",
               "vis_class", "vis_in_shortest_chain", "jstar", "E_L", "Theta_L", "Lambda_L", "lens_M", "lens_Z"]
HASHED = ("judge.py", "stats.py", "readouts.py")


# ------------------------------------------------------------------ helpers

def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def git_info() -> dict:
    def run(*a):
        try:
            return subprocess.run(["git", "-C", PKG_DIR, *a], capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception as e:  # pragma: no cover
            return f"<git unavailable: {e}>"
    return {"git_sha": run("rev-parse", "HEAD"), "git_dirty_z_dynamics": run("status", "--porcelain", "--", PKG_DIR)}


def load_split(data_path: str, split: str, n: int):
    from utils.z_logging import _load_probe_tensors
    sd = gates.guard_path(os.path.join(data_path, split))
    with open(os.path.join(sd, "dataset.json")) as f:
        ignore = json.load(f)["ignore_label_id"]
    probe, fp = _load_probe_tensors(data_path, split, n, ignore)
    if probe["inputs"].shape[0] != n:
        raise RuntimeError(f"{data_path}/{split}: only {probe['inputs'].shape[0]} rows, need {n}")
    return probe, fp


def refuse_smoke_out(out: str):
    p = os.path.abspath(out)
    parts = p.split(os.sep)
    if "reports" in parts or "lab" in parts or p.startswith("/mnt/ayp/agents"):
        raise SystemExit(f"--smoke output must not be under reports/ or lab/: {p}")


def fmt(v):
    if v is None:
        return ""
    if isinstance(v, (list, tuple, set)):
        return ";".join(str(x) for x in sorted(v))
    if isinstance(v, (np.floating,)):
        return repr(float(v))
    return v


def rows_to_text(rows: list[dict], fields: list[str]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    for r in rows:
        w.writerow({k: fmt(r.get(k)) for k in fields})
    return buf.getvalue()


# ------------------------------------------------------------------ forward over one split

def run_split(inner, arch: str, probe: dict, n_decode: int, n_pr: int, keep_ans: bool, B: int, k: int,
              infos: list) -> dict:
    N = probe["inputs"].shape[0]
    cfg = inner.config
    J = cfg.H_cycles * cfg.L_cycles
    pl = inner.puzzle_emb_len
    dec_p = np.zeros((N, J, 10), np.int64)
    dec_r = np.zeros((N, J, 10), np.int64)
    dec_n = np.zeros((N, cfg.H_cycles, 10), np.int64)
    soft = np.zeros((J, k + 1), np.float64)
    keep_SL, keep_zHp, keep_zHn, ans = [], [], [], []
    g1 = g2 = True
    j_to_c = None
    with torch.inference_mode():
        for s in range(0, N, B):
            e = min(N, s + B)
            batch = {kk: v[s:e] for kk, v in probe.items()}
            tr = forward_trace(inner, batch, arch)
            ref = native_logits(inner, batch)
            g1 &= g1_equal(tr, ref)
            g2 &= g2_equal(tr, cfg.L_cycles)
            j_to_c = tr.j_to_c
            for j in range(J):
                dec_p[s:e, j] = readouts.decode(tr.logits[j])
                dec_r[s:e, j] = readouts.decode(tr.raw_logits[j])
            for h in range(cfg.H_cycles):
                dec_n[s:e, h] = readouts.decode(tr.native_logits[h])
            nd = max(0, min(e, n_decode) - s)
            if nd:
                for j in range(J):
                    probs = F.softmax(tr.logits[j][:nd, :10].float(), dim=-1).numpy()
                    soft[j] += readouts.soft_mass(probs, infos[s:s + nd], k)
            npr = max(0, min(e, n_pr) - s)
            if npr:
                keep_SL.append([z[:npr].clone() for z in tr.S_L])
                keep_zHp.append([z[:npr].clone() for z in tr.zHp])
                keep_zHn.append([z[:npr].clone() for z in tr.zHn])
            if keep_ans:
                ans.append(torch.stack([z[:, pl:pl + 10] for z in tr.zHp], 1).clone())
            del tr, ref
    cat = lambda lst, n: [torch.cat([c[t] for c in lst], 0) for t in range(n)] if lst else []  # noqa: E731
    return {"dec_p": dec_p, "dec_r": dec_r, "dec_n": dec_n, "soft": soft / max(n_decode * 10, 1),
            "S_L": cat(keep_SL, J), "zHp": cat(keep_zHp, J), "zHn": cat(keep_zHn, cfg.H_cycles),
            "ans": torch.cat(ans, 0) if ans else None, "G1": bool(g1), "G2": bool(g2), "j_to_c": j_to_c}


def exact_rate(dec18: np.ndarray, labels10: np.ndarray) -> float:
    return float((dec18 == labels10).all(-1).mean())


# ------------------------------------------------------------------ one cell

def process_cell(row: dict, a, closeout: dict) -> dict:
    name, step, arch, proto, k = row["run_name"], row["step"], row["arch"], row["protocol"], int(row["k"])
    unit = cellsmod.unit_name(name, step)
    run_dir = os.path.join(a.ckpt_root, name)
    ck = os.path.join(run_dir, f"step_{step}")
    rec = {"run_name": name, "unit": unit, "arch": arch, "protocol": proto, "k": k, "step": step,
           "status": "excluded", "exclusion_reason": None}
    out = {"rec": rec, "per_call": [], "pr": [], "cell": [], "near": [], "soft": []}

    co = closeout.get(name)
    if co is None or co["run_id"] != row["run_id"] or abs(float(co["final_probe_test_exact"]) - float(row["final_probe_test_exact"])) > 1e-9:
        raise RuntimeError(f"{name}: closeout CSV row disagrees with the frozen manifest")
    rec["final_probe_test_exact"] = float(co["final_probe_test_exact"])
    rec["traj_class"] = co["traj_class"]

    # I-4 terminal step
    steps_on_disk = sorted(int(d.split("_")[1]) for d in os.listdir(run_dir)
                           if d.startswith("step_") and d.split("_")[1].isdigit() and len(d.split("_")) == 2)
    term = judge.rule("gates.TERMINAL_STEP")[proto]
    rec["I4_ok"] = bool(step == term and steps_on_disk and steps_on_disk[-1] == step)
    rec["steps_on_disk_max"] = steps_on_disk[-1] if steps_on_disk else None
    for p in (ck, os.path.join(run_dir, "all_config.yaml"), os.path.join(run_dir, "z_snapshots", f"step_{step}.pt")):
        if not os.path.exists(p):
            rec["exclusion_reason"] = f"missing {p}"
            return out
    if not rec["I4_ok"]:
        rec["exclusion_reason"] = "I-4 non-terminal checkpoint"
        return out

    from measure_rho import build_model_config, load_all_config
    all_cfg = load_all_config(run_dir)
    g7 = gates.g7_config(all_cfg)
    rec["G7"], rec["G7_detail"], rec["G7_values"] = g7["pass"], g7["mismatches"], g7["values"]
    data_path = all_cfg["data_paths"][0]
    gates.guard_path(data_path)
    if gates.data_root(data_path) != cellsmod.DATA_ROOT[proto] or int(os.path.basename(data_path)) != k:
        raise RuntimeError(f"{name}: data path {data_path} disagrees with manifest protocol/k")

    cfg = build_model_config(run_dir, batch_size=a.batch)
    wrapped, inner, missing, unexpected = load_inner(ck, cfg, a.device, arch)
    rec["I3_ok"] = not missing and not unexpected
    rec["I3"] = {"missing": missing, "unexpected": unexpected}
    if not rec["I3_ok"]:
        rec["exclusion_reason"] = "I-3 state-dict key mismatch"
        return out
    g0 = gates.g0_code_identity(run_dir, arch)
    rec["G0"], rec["G0_detail"] = g0["pass"], g0

    fam = [f for f, names in judge.rule("families").items() if name in names]
    do_lens = a.lens_train_n > 0 and fam == ["F1_trm_D1"]
    n_train = max(a.probe_n, a.lens_train_n if do_lens else 0)
    test, test_fp = load_split(data_path, "test", a.test_n)
    train, train_fp = load_split(data_path, "train", n_train)
    rec["probe_fingerprints"] = {"test": test_fp, "train": train_fp}
    t_in, t_lab = test["inputs"].numpy(), test["labels"].numpy()[:, :10]
    r_in, r_lab = train["inputs"].numpy(), train["labels"].numpy()[:, :10]
    g6t = readouts.labels_are_sigma_k(t_in, t_lab, k)
    g6r = readouts.labels_are_sigma_k(r_in, r_lab, k)
    rec["G6"], rec["G6_detail"] = bool(g6t["pass"] and g6r["pass"]), {"test": g6t, "train": g6r}

    infos_t = readouts.sigma_infos(t_in, k)
    infos_r = readouts.sigma_infos(r_in[:a.probe_n], k)
    frozen_cls = row["class"]
    base = {"exp_id": EXP, "unit": unit, "run_name": name}
    res = {}
    for split, probe, nd, infos, keep_ans in (("test", test, a.test_n, infos_t, do_lens),
                                              ("train", train, a.probe_n, infos_r, do_lens)):
        res[split] = run_split(inner, arch, probe, nd, a.probe_n, keep_ans, a.batch, k,
                               infos + [None] * (probe["inputs"].shape[0] - nd))
    rec["G1"] = res["test"]["G1"] and res["train"]["G1"]
    rec["G2"] = res["test"]["G2"] and res["train"]["G2"]
    j_to_c = res["test"]["j_to_c"]
    native_j = judge.rule("readouts.native_j")

    # ---- G3 / G4
    snap = gates.snapshot_reference(run_dir, step)
    if not torch.equal(snap["labels"][: a.probe_n].to(torch.int64), train["labels"][: a.probe_n].to(torch.int64)):
        raise RuntimeError(f"{name}: snapshot labels differ from the train probe (probe identity broken)")
    pr_zH = pr_of_latent(res["train"]["zHn"][-1])
    pr_zL = pr_of_latent(res["train"]["S_L"][-1])
    rec["G3"] = ({"pr_zH": pr_zH, "pr_zH_snapshot": snap["pr_zH_snapshot"],
                  "pr_zL": pr_zL, "pr_zL_snapshot": snap["pr_zL_snapshot"]} if arch == "trm" else None)
    rec["G4"] = {"test512_exact": exact_rate(res["test"]["dec_p"][:512, -1], t_lab[:512]),
                 "train512_exact": exact_rate(res["train"]["dec_p"][: a.probe_n, -1], r_lab[: a.probe_n]),
                 "snapshot_train_exact": snap["snapshot_train_exact"]}
    rec.update(gates.informational_flags(rec))

    # ---- decode analyses
    lens_cell = None
    for split in ("test", "train"):
        nd = a.test_n if split == "test" else a.probe_n
        inp = (t_in if split == "test" else r_in)[:nd]
        lab = (t_lab if split == "test" else r_lab)[:nd]
        infos = infos_t if split == "test" else infos_r
        R = res[split]
        dp, dr, dn = R["dec_p"][:nd], R["dec_r"][:nd], R["dec_n"][:nd]
        prl = pr_rows(unit, name, split, R["S_L"], R["zHp"], R["zHn"], a.probe_n)
        out["pr"] += prl
        zl = [r["pr"] for r in prl if r["stream"] == "z_L"]
        pr_stats = {"E_L": stats.E_L(zl), "Theta_L": stats.Theta_L(zl), "Lambda_L": stats.Lambda_L(zl)}
        kinds = [("prelim", dp), ("raw_zL", dr)]
        if split == "test" and do_lens:
            lm_w = inner.lm_head.weight
            lens = tuned_lens.fit_lens(res["train"]["ans"][: a.lens_train_n], lm_w)
            ldec = tuned_lens.apply_lens(res["test"]["ans"][:nd], lens, lm_w).numpy()
            kinds.append(("lens", np.concatenate([ldec, dp[:, -1:]], 1)))
        cell_rows = {}
        for kind, dec in kinds:
            an = stats.analyze_decodes(dec, inp, lab, k, infos=infos, n_perm=a.n_perm, ref_dec=dp[:, -1])
            for pj in an["per_j"]:
                out["per_call"].append(dict(base, split=split, readout_kind=kind, c=j_to_c[pj["j"] - 1],
                                            native=pj["j"] in native_j, **pj))
            c = dict(an["cell"])
            crow = dict(base, arch=arch, protocol=proto, k=k, frozen_class=frozen_cls, split=split,
                        readout_kind=kind, **c)
            if kind == "prelim":
                crow.update(pr_stats)
                _near_lines(out["near"], base, split, kind, an["near"], k)
                for j in range(dec.shape[1]):
                    for m in range(k + 1):
                        out["soft"].append(dict(base, split=split, readout_kind=kind, j=j + 1, m=m,
                                                mass=float(R["soft"][j, m])))
            cell_rows[kind] = crow
        an_n = stats.analyze_decodes(dn, inp, lab, k, infos=infos, ref_dec=dp[:, -1],
                                     j_labels=native_j, cell_stats=False)
        for pj in an_n["per_j"]:
            h = native_j.index(pj["j"]) + 1
            out["per_call"].append(dict(base, split=split, readout_kind="native",
                                        c=(inner.config.L_cycles + 1) * h, native=True, **pj))
        if "lens" in cell_rows:
            lens_cell = cell_rows["lens"]
            cell_rows["prelim"]["lens_M"], cell_rows["prelim"]["lens_Z"] = lens_cell["M"], lens_cell["Z"]
        out["cell"] += list(cell_rows.values())
    rec["status"] = "analysed"
    del wrapped, inner, res
    return out


def _near_lines(dst: list, base: dict, split: str, kind: str, near: dict, k: int):
    ms, hm, tie, none, prop = near["m_star"], near["ham_min"], near["tie"], near["none"], near["proper"]
    for i, s in enumerate(near["infos"]):
        ct = list(s.ctype)
        res = near["residues"][i]
        for j in range(ms.shape[1]):
            m = int(ms[i, j])
            dst.append({"exp_id": EXP, "unit": base["unit"], "split": split, "readout_kind": kind, "i": i,
                        "j": j + 1, "m_star": None if m < 0 else m, "ham_min": int(hm[i, j]),
                        "tie": bool(tie[i, j]), "none": bool(none[i, j]), "proper": bool(prop[i, j]),
                        "ord": s.o, "cycle_type": ct, "gcd": s.gcd, "informative": s.informative,
                        "residues": [r[j] for r in res]})


# ------------------------------------------------------------------ smoke

def smoke(a, rows):
    refuse_smoke_out(a.out)
    os.makedirs(a.out, exist_ok=True)
    from measure_rho import build_model_config, load_all_config
    report = {"exp_id": EXP, "smoke": True, "note": "loading + hook shapes only; no statistic, accuracy or PR computed",
              "cells": []}
    for row in rows:
        name, step, arch = row["run_name"], row["step"], row["arch"]
        run_dir = os.path.join(a.ckpt_root, name)
        all_cfg = load_all_config(run_dir)
        data_path = gates.guard_path(all_cfg["data_paths"][0])
        cfg = build_model_config(run_dir, batch_size=a.smoke_n)
        _w, inner, missing, unexpected = load_inner(os.path.join(run_dir, f"step_{step}"), cfg, a.device, arch)
        probe, _fp = load_split(data_path, "test", a.smoke_n)
        with torch.inference_mode():
            tr = forward_trace(inner, probe, arch)
            ref = native_logits(inner, probe)
        report["cells"].append({
            "run_name": name, "arch": arch, "missing_keys": missing, "unexpected_keys": unexpected,
            "n_rows": a.smoke_n, "n_S_L": len(tr.S_L), "n_zHp": len(tr.zHp), "n_zHn": len(tr.zHn),
            "n_prelim_logits": len(tr.logits), "S_L_shape": list(tr.S_L[0].shape), "zHp_shape": list(tr.zHp[0].shape),
            "logits_shape": list(tr.logits[0].shape), "dtype": str(tr.zHp[0].dtype), "j_to_c": tr.j_to_c,
            "G1_final_logits_torch_equal": g1_equal(tr, ref), "G2_native_readouts_torch_equal": g2_equal(tr, inner.config.L_cycles),
            "G0_code_identity": gates.g0_code_identity(run_dir, arch)["pass"],
        })
    with open(os.path.join(a.out, "smoke.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))
    ok = all(c["G1_final_logits_torch_equal"] and c["G2_native_readouts_torch_equal"] and c["n_zHp"] == 18
             and not c["missing_keys"] and not c["unexpected_keys"] for c in report["cells"])
    return 0 if ok else 1


# ------------------------------------------------------------------ main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", default=os.path.join(PKG_DIR, "EXP017_cells.csv"))
    ap.add_argument("--ckpt-root", default="checkpoints/power_permutation")
    ap.add_argument("--closeout", default="reports/pp_campaign_retro_trajectory/per_run_trajectory.csv")
    ap.add_argument("--repo-root", default=os.getcwd(), help="checkpoint/data paths are relative to this (default cwd)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=250)
    ap.add_argument("--test-n", type=int, default=1000)
    ap.add_argument("--probe-n", type=int, default=512)
    ap.add_argument("--lens-train-n", type=int, default=2048, help="0 disables the D-7 tuned lens")
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--only", default="", help="comma-separated run names (debug; the manifest is then 'partial')")
    ap.add_argument("--no-g5", action="store_true", help="skip the G5 re-run (judge will HALT on G5)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-n", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    a.out = os.path.abspath(a.out)
    a.cells = os.path.abspath(a.cells)
    os.chdir(a.repo_root)
    a.ckpt_root = os.path.abspath(a.ckpt_root)
    if a.threads:
        torch.set_num_threads(a.threads)
    rows = cellsmod.read_csv(a.cells)
    if a.only:
        want = [x.strip() for x in a.only.split(",") if x.strip()]
        unknown = set(want) - {r["run_name"] for r in rows}
        if unknown:
            raise SystemExit(f"--only names not in manifest: {sorted(unknown)}")
        rows = [r for r in rows if r["run_name"] in want]
    if a.smoke:
        if len(rows) != 1:
            raise SystemExit("--smoke needs exactly one cell via --only")
        return smoke(a, rows)

    # frozen run parameters (a deviation would silently change the analysis)
    frozen_args = {"n_perm": int(judge.num("czz2.N_PERM")), "seed": int(judge.num("czz2.PERM_SEED")),
                   "test_n": 1000, "probe_n": 512}
    dev = {k: getattr(a, k) for k, v in frozen_args.items() if getattr(a, k) != v}
    if dev:
        raise SystemExit(f"arguments deviate from the frozen protocol: {dev} (expected {frozen_args})")
    if os.path.isdir(a.out) and os.listdir(a.out):
        raise SystemExit(f"--out {a.out} is not empty; I-8 requires a from-scratch run")
    os.makedirs(a.out, exist_ok=True)

    g5_unit = judge.rule("gates.G5_UNIT")
    rows = sorted(rows, key=lambda r: (r["run_name"] != g5_unit, r["idx"]))
    closeout = gates.closeout_lookup(a.closeout)
    manifest = {"exp_id": EXP, "status": "running", "smoke": False, "partial": bool(a.only),
                "args": vars(a), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                **git_info(), "sha256": {f: sha256(os.path.join(PKG_DIR, f)) for f in sorted(os.listdir(PKG_DIR))
                                         if f.endswith(".py") or f.endswith(".csv")},
                "torch": torch.__version__, "python": platform.python_version(), "device": a.device,
                "threads": torch.get_num_threads(), "host": platform.node(), "cell_seconds": {}}
    man_path = os.path.join(a.out, "run_manifest.json")

    def write_manifest():
        with open(man_path, "w") as f:
            json.dump(manifest, f, indent=1, default=str)

    write_manifest()
    files = {"per_call": ("per_call.csv", PER_CALL_FIELDS), "pr": ("pr_traj.csv", PR_FIELDS),
             "cell": ("cell_stats.csv", CELL_FIELDS), "soft": ("soft_mass.csv", SOFT_FIELDS)}
    for key, (fn, fields) in files.items():
        with open(os.path.join(a.out, fn), "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields, lineterminator="\n").writeheader()
    open(os.path.join(a.out, "near_power.jsonl"), "w").close()

    recs, g5_first = {}, None
    try:
        for n, row in enumerate(rows):
            t0 = time.time()
            res = process_cell(row, a, closeout)
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
                g5_first = {k2: rows_to_text(res[k2], files[k2][1]) for k2 in ("per_call", "pr", "cell")}
                print(f"[timing] G5 unit {g5_unit}: {dt:.0f} s -> extrapolated {dt * len(rows) / 3600:.1f} h "
                      f"for {len(rows)} cells (+ G5 re-run)", flush=True)
            print(f"[{n + 1}/{len(rows)}] {row['run_name']} status={res['rec']['status']} {dt:.0f}s", flush=True)
            write_manifest()

        g5 = {"unit": g5_unit, "ran": False, "identical": None}
        if not a.no_g5 and g5_first is not None:
            t0 = time.time()
            again = process_cell(next(r for r in rows if r["run_name"] == g5_unit), a, closeout)
            second = {k2: rows_to_text(again[k2], files[k2][1]) for k2 in ("per_call", "pr", "cell")}
            diffs = {k2: g5_first[k2] != second[k2] for k2 in second}
            g5 = {"unit": g5_unit, "ran": True, "identical": not any(diffs.values()), "differs": diffs,
                  "seconds": time.time() - t0}
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
