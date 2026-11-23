"""P-C1 G-b: code-vs-code equivalence of ``analysis/ltf_decode`` (TRM reference mode) with the EXP-017 instrument.
(spec §4.3). IMPLEMENTED ONLY: this file must NOT be executed on real checkpoints before the G-c review.

  CUDA_VISIBLE_DEVICES="" uv run --no-sync python -m analysis.ltf_decode.gb_equivalence --device cpu --threads <N> \
      --scratch <empty dir outside reports/ and lab/>

Procedure (real run):
  1. reference: ``analysis/z_dynamics/run_zdyn.py --only pp_base_tf_z_iter_k5_s1,pp_base_tf_z_iter_k10_s2 --device cpu
     --threads N --lens-train-n 0 --no-g5 --out <scratch>/ref`` (subprocess, CUDA_VISIBLE_DEVICES="").
  2. new mode: the SAME forward (analysis.z_dynamics.instrumented_forward.forward_trace, imported unmodified) and
     ``stats_j.analyze_decodes_j(J = 17, J_all = 18)`` produce per_call / cell rows in the run_zdyn schema.
  3. compare with tolerance 0: decode arrays (test + train; run_zdyn does not persist decodes, so the reference
     decodes are regenerated in-process with ``run_zdyn.run_split``), per_call.csv text, cell_stats.csv text.
Output: PASS/FAIL, the sha256 of the four files, the number of differing rows, decode-equality booleans.
No per-call value is printed and nobody opens the scratch CSVs (spec §4.3 exposure disclosure).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
import subprocess
import sys

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(os.path.dirname(PKG_DIR))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from analysis.ltf_decode.run_ltf_decode import require_cpu  # noqa: E402
from analysis.ltf_decode.stats_j import analyze_decodes_j  # noqa: E402
from analysis.z_dynamics import cells as zcells  # noqa: E402
from analysis.z_dynamics import gates as zgates  # noqa: E402
from analysis.z_dynamics import judge as zjudge  # noqa: E402
from analysis.z_dynamics import readouts, stats as zstats  # noqa: E402
from analysis.z_dynamics.instrumented_forward import forward_trace, load_inner, native_logits  # noqa: E402
from analysis.z_dynamics.pr_traj import pr_rows  # noqa: E402

CELLS = ("pp_base_tf_z_iter_k5_s1", "pp_base_tf_z_iter_k10_s2")
EXP = "EXP-017"        # run_zdyn's base id; the rows must be byte-identical, so the same id is used
# schema copied from run_zdyn.py:49-56; at real run time main() asserts equality with the imported module's constants
PER_CALL_FIELDS = ["exp_id", "unit", "run_name", "split", "readout_kind", "j", "c", "native", "a", "a_self", "v",
                   "trivial", "near_power_frac", "proper_partial_frac", "none_frac", "tie_frac", "gcd_class", "n_rows"]
CELL_FIELDS = ["exp_id", "unit", "run_name", "arch", "protocol", "k", "frozen_class", "split", "readout_kind",
               "n_rows", "a1", "a17", "R", "maxdrop", "maxrise", "M", "subclass", "V", "T", "gamma", "p", "P_int",
               "P_int0", "thr", "informative_frac", "n_informative", "n_mixture_rejected", "Z", "Z_raw", "Vis",
               "vis_class", "vis_in_shortest_chain", "jstar", "E_L", "Theta_L", "Lambda_L", "lens_M", "lens_Z"]


def _fmt(v):
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
        w.writerow({k: _fmt(r.get(k)) for k in fields})
    return buf.getvalue()


def header_text(fields: list[str]) -> str:
    buf = io.StringIO()
    csv.DictWriter(buf, fieldnames=fields, lineterminator="\n").writeheader()
    return buf.getvalue()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _split_decode(inner, arch: str, probe: dict, n_pr: int, B: int):
    """Same forward as run_zdyn.run_split (decodes + the first n_pr rows of the latents for the PR columns)."""
    N = probe["inputs"].shape[0]
    cfg = inner.config
    J = cfg.H_cycles * cfg.L_cycles
    dec_p = np.zeros((N, J, 10), np.int64)
    dec_r = np.zeros((N, J, 10), np.int64)
    dec_n = np.zeros((N, cfg.H_cycles, 10), np.int64)
    keep_SL, keep_zHp, keep_zHn = [], [], []
    with torch.inference_mode():
        for s in range(0, N, B):
            e = min(N, s + B)
            batch = {kk: v[s:e] for kk, v in probe.items()}
            tr = forward_trace(inner, batch, arch)
            _ref = native_logits(inner, batch)
            for j in range(J):
                dec_p[s:e, j] = readouts.decode(tr.logits[j])
                dec_r[s:e, j] = readouts.decode(tr.raw_logits[j])
            for h in range(cfg.H_cycles):
                dec_n[s:e, h] = readouts.decode(tr.native_logits[h])
            npr = max(0, min(e, n_pr) - s)
            if npr:
                keep_SL.append([z[:npr].clone() for z in tr.S_L])
                keep_zHp.append([z[:npr].clone() for z in tr.zHp])
                keep_zHn.append([z[:npr].clone() for z in tr.zHn])
            del tr
    cat = lambda lst, n: [torch.cat([c[t] for c in lst], 0) for t in range(n)] if lst else []  # noqa: E731
    return {"dec_p": dec_p, "dec_r": dec_r, "dec_n": dec_n, "S_L": cat(keep_SL, J), "zHp": cat(keep_zHp, J),
            "zHn": cat(keep_zHn, cfg.H_cycles)}


def new_mode_process_cell(row: dict, a) -> dict:
    """TRM reference mode: per_call / cell rows in the run_zdyn schema, via forward_trace + stats_j(17, 18)."""
    from measure_rho import build_model_config, load_all_config
    from analysis.ltf_decode.run_ltf_decode import load_split
    name, step, arch, proto, k = row["run_name"], row["step"], row["arch"], row["protocol"], int(row["k"])
    unit = zcells.unit_name(name, step)
    run_dir = os.path.join(a.ckpt_root, name)
    ck = os.path.join(run_dir, f"step_{step}")
    all_cfg = load_all_config(run_dir)
    data_path = all_cfg["data_paths"][0]
    zgates.guard_path(data_path)
    cfg = build_model_config(run_dir, batch_size=a.batch)
    _w, inner, missing, unexpected = load_inner(ck, cfg, a.device, arch)
    if missing or unexpected:
        raise RuntimeError(f"{name}: state-dict keys missing={missing} unexpected={unexpected}")
    test, _ = load_split(data_path, "test", a.test_n)
    train, _ = load_split(data_path, "train", a.probe_n)
    t_in, t_lab = test["inputs"].numpy(), test["labels"].numpy()[:, :10]
    r_in, r_lab = train["inputs"].numpy(), train["labels"].numpy()[:, :10]
    infos_t = readouts.sigma_infos(t_in, k)
    infos_r = readouts.sigma_infos(r_in[:a.probe_n], k)
    J, J_all = zjudge.j_stat(), int(zjudge.rule("readouts.j_all")[1])
    native_j = zjudge.rule("readouts.native_j")
    base = {"exp_id": EXP, "unit": unit, "run_name": name}
    out = {"per_call": [], "cell": [], "dec": {}}
    res = {"test": _split_decode(inner, arch, test, a.probe_n, a.batch),
           "train": _split_decode(inner, arch, train, a.probe_n, a.batch)}
    j_to_c = [(inner.config.L_cycles + 1) * (j // inner.config.L_cycles) + j % inner.config.L_cycles + 1
              for j in range(inner.config.H_cycles * inner.config.L_cycles)]
    for split in ("test", "train"):
        nd = a.test_n if split == "test" else a.probe_n
        inp = (t_in if split == "test" else r_in)[:nd]
        lab = (t_lab if split == "test" else r_lab)[:nd]
        infos = infos_t if split == "test" else infos_r
        R = res[split]
        dp, dr, dn = R["dec_p"][:nd], R["dec_r"][:nd], R["dec_n"][:nd]
        out["dec"][split] = {"dec_p": R["dec_p"], "dec_r": R["dec_r"], "dec_n": R["dec_n"]}
        prl = pr_rows(unit, name, split, R["S_L"], R["zHp"], R["zHn"], a.probe_n)
        zl = [r["pr"] for r in prl if r["stream"] == "z_L"]
        pr_stats = {"E_L": zstats.E_L(zl), "Theta_L": zstats.Theta_L(zl), "Lambda_L": zstats.Lambda_L(zl)}
        cell_rows = {}
        for kind, dec in (("prelim", dp), ("raw_zL", dr)):
            an = analyze_decodes_j(dec, inp, lab, k, J=J, J_all=J_all, infos=infos, n_perm=a.n_perm, ref_dec=dp[:, -1])
            for pj in an["per_j"]:
                out["per_call"].append(dict(base, split=split, readout_kind=kind, c=j_to_c[pj["j"] - 1],
                                            native=pj["j"] in native_j, **pj))
            crow = dict(base, arch=arch, protocol=proto, k=k, frozen_class=row["class"], split=split,
                        readout_kind=kind, **an["cell"])
            if kind == "prelim":
                crow.update(pr_stats)
            cell_rows[kind] = crow
        an_n = analyze_decodes_j(dn, inp, lab, k, J=J, J_all=J_all, infos=infos, ref_dec=dp[:, -1],
                                 j_labels=native_j, cell_stats=False)
        for pj in an_n["per_j"]:
            h = native_j.index(pj["j"]) + 1
            out["per_call"].append(dict(base, split=split, readout_kind="native",
                                        c=(inner.config.L_cycles + 1) * h, native=True, **pj))
        out["cell"] += list(cell_rows.values())
    return out


def compare_texts(ref: str, new: str) -> int:
    """Number of differing rows (positionally) + |row-count difference|. 0 == identical."""
    a, b = ref.splitlines(), new.splitlines()
    return sum(1 for x, y in zip(a, b) if x != y) + abs(len(a) - len(b))


def reference_decodes(row: dict, a) -> dict:
    """Regenerate the reference decodes in-process with run_zdyn.run_split (unmodified; run_zdyn persists none)."""
    from analysis.z_dynamics import run_zdyn
    from measure_rho import build_model_config, load_all_config
    from analysis.ltf_decode.run_ltf_decode import load_split
    name, step, arch, k = row["run_name"], row["step"], row["arch"], int(row["k"])
    run_dir = os.path.join(a.ckpt_root, name)
    all_cfg = load_all_config(run_dir)
    cfg = build_model_config(run_dir, batch_size=a.batch)
    _w, inner, _m, _u = load_inner(os.path.join(run_dir, f"step_{step}"), cfg, a.device, arch)
    out = {}
    for split, n in (("test", a.test_n), ("train", a.probe_n)):
        probe, _ = load_split(all_cfg["data_paths"][0], split, n)
        infos = readouts.sigma_infos(probe["inputs"].numpy(), k)
        r = run_zdyn.run_split(inner, arch, probe, n, a.probe_n, False, a.batch, k, infos)
        out[split] = {"dec_p": r["dec_p"], "dec_r": r["dec_r"], "dec_n": r["dec_n"]}
    return out


def decodes_equal(new_dec: dict, ref_dec: dict) -> bool:
    return all(np.array_equal(new_dec[s][kk], ref_dec[s][kk]) for s in ("test", "train")
               for kk in ("dec_p", "dec_r", "dec_n"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--ckpt-root", default="checkpoints/power_permutation")
    ap.add_argument("--repo-root", default=PKG_ROOT)
    ap.add_argument("--batch", type=int, default=250)
    a = ap.parse_args(argv)
    require_cpu(a.device)
    a.test_n, a.probe_n, a.n_perm = 1000, 512, int(zjudge.num("czz2.N_PERM"))
    os.chdir(a.repo_root)
    a.ckpt_root = os.path.abspath(a.ckpt_root)
    scratch = os.path.abspath(a.scratch)
    parts = scratch.split(os.sep)
    if "reports" in parts or "lab" in parts or os.path.isdir(scratch) and os.listdir(scratch):
        raise SystemExit("--scratch must be empty and outside reports/ and lab/")
    os.makedirs(scratch, exist_ok=True)
    torch.set_num_threads(a.threads)

    from analysis.z_dynamics import run_zdyn
    if run_zdyn.PER_CALL_FIELDS != PER_CALL_FIELDS or run_zdyn.CELL_FIELDS != CELL_FIELDS:
        raise SystemExit("schema drift between run_zdyn and gb_equivalence")
    ref_dir = os.path.join(scratch, "ref")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    cmd = [sys.executable, "analysis/z_dynamics/run_zdyn.py", "--only", ",".join(CELLS), "--device", "cpu",
           "--threads", str(a.threads), "--lens-train-n", "0", "--no-g5", "--out", ref_dir]
    rc = subprocess.run(cmd, env=env, cwd=a.repo_root, stdout=subprocess.DEVNULL).returncode
    if rc != 0:
        print(f"GB: FAIL (reference run_zdyn rc={rc})")
        return 1
    rows = {r["run_name"]: r for r in zcells.rows()}
    order = [rows[n] for n in CELLS]      # G5 unit first, as run_zdyn orders it
    new_pc = header_text(PER_CALL_FIELDS)
    new_cs = header_text(CELL_FIELDS)
    dec_ok = True
    for row in order:
        res = new_mode_process_cell(row, a)
        new_pc += rows_to_text(res["per_call"], PER_CALL_FIELDS)
        new_cs += rows_to_text(res["cell"], CELL_FIELDS)
        dec_ok &= decodes_equal(res["dec"], reference_decodes(row, a))
    with open(os.path.join(scratch, "new_per_call.csv"), "w") as f:
        f.write(new_pc)
    with open(os.path.join(scratch, "new_cell_stats.csv"), "w") as f:
        f.write(new_cs)
    ref_pc = open(os.path.join(ref_dir, "per_call.csv")).read()
    ref_cs = open(os.path.join(ref_dir, "cell_stats.csv")).read()
    d_pc, d_cs = compare_texts(ref_pc, new_pc), compare_texts(ref_cs, new_cs)
    ok = bool(dec_ok and d_pc == 0 and d_cs == 0)
    print(f"GB: {'PASS' if ok else 'FAIL'}")
    print(f"sha256 ref_per_call={sha256_text(ref_pc)} new_per_call={sha256_text(new_pc)}")
    print(f"sha256 ref_cell_stats={sha256_text(ref_cs)} new_cell_stats={sha256_text(new_cs)}")
    print(f"differing_rows per_call={d_pc} cell_stats={d_cs} decodes_equal={dec_ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
