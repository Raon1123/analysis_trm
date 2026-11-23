"""P-C1 G-c: CPU timing + extrapolation cost rule (spec §4.4, r7 protocol). Random init, blind values: 0.

  CUDA_VISIBLE_DEVICES="" uv run --no-sync python -m analysis.ltf_decode.gc_timing --device cpu --threads 32

Protocol (all constants below are the spec's):
  * ``Model_ACTV2`` is randomly initialised (torch.manual_seed(0)) from the ``arch`` block of
    checkpoints/power_permutation/pp_ltf_g7_lay2_k5_s1/all_config.yaml; NO checkpoint file is read.
  * D1 k = 5 data rows (test 1000, train 512) go through the real tracer + native forward in batches of 250.
    The statistics layer runs on SYNTHETIC ``genuine_sequential`` decodes over the real sigma rows (rng seed 0),
    forcing the J = 20 shuffled-call permutation path (N_PERM = 2000). No accuracy is computed.
  * sec_ext: tracer 42 loops + t = 22..42 descriptive rows (+ native 21 loop G1' check); sec_noext: tracer 21 loops.
    Six measurements alternate ext, noext, ext, noext, ext, noext; the decision uses the medians. Caches
    (omega, SigmaInfo) are cleared before every measurement (cold, as the G5 precedent run_zdyn.py:460-461).
  * threads must be exactly 32, else FINISH rc=3 EXTRAP=none and exit before any measurement.
  * gc_timing.log: ``START <UTC> threads=<n> argv_sha=<sha256>`` at start; ``FINISH <UTC> rc=<rc> EXTRAP=<KEEP|DROP|none>``
    is appended, flushed and fsync'ed BEFORE the decision line is printed or written to the output file.
  * Decision: N_RUN = 31; W_tot = N_RUN*sec_ext/3600; W_ext = N_RUN*(sec_ext-sec_noext)/3600; DROP iff
    W_ext > X_EXT (8 h) or W_tot > Y_TOT (24 h), strict; else KEEP. The first completed run (a FINISH line with
    KEEP/DROP) binds: if one exists in the log this program refuses to start (exit 5, nothing appended).
  * ``--dev`` (tiny model/data, any thread count, must use a non-default --log) exists only for tests.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import statistics
import sys
import time
from datetime import datetime, timezone

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(os.path.dirname(PKG_DIR))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from analysis.ltf_decode import judge  # noqa: E402
from analysis.ltf_decode.run_ltf_decode import analyze_split, decode_loops, load_split, require_cpu  # noqa: E402
from analysis.ltf_decode.stats_j import synth_genuine_sequential  # noqa: E402
from analysis.ltf_decode.trace import build_ltf  # noqa: E402
from analysis.z_dynamics import readouts, stats as zstats  # noqa: E402

REQUIRED_THREADS = 32
X_EXT_H = 8.0
Y_TOT_H = 24.0
N_RUN = 31                      # 30 cells + G5' re-run (spec §4.4 "결정 수치")
N_PROJ = 33                     # 30 cells + 2 G-b cells + G5' re-run (spec "출력")
SEED = 0
CFG_DIR = "checkpoints/power_permutation/pp_ltf_g7_lay2_k5_s1"     # only all_config.yaml is read
DATA_PATH = "data/power_permutation/d1/5"
K = 5
BATCH = 250
SEQUENCE = ["ext", "noext"] * 3
ART = "/home/ayp/project/trm/lab/23_reports/11_lab-reports/2026-10-21_c1-prereg-draft_artifacts/pc1_impl"
DEFAULT_LOG = os.path.join(ART, "gc_timing.log")
DEFAULT_OUT = os.path.join(ART, "gc_timing.out")


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def append_flush(path: str, line: str) -> None:
    with open(path, "a") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def completed_run_in_log(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    for ln in open(path):
        if ln.startswith("FINISH ") and ("EXTRAP=KEEP" in ln or "EXTRAP=DROP" in ln):
            return ln.strip()
    return None


def decide(sec_ext: float, sec_noext: float) -> dict:
    w_tot = N_RUN * sec_ext / 3600
    w_ext = N_RUN * (sec_ext - sec_noext) / 3600
    w_noext = N_RUN * sec_noext / 3600
    drop = (w_ext > X_EXT_H) or (w_tot > Y_TOT_H)       # strict inequalities; boundary values are KEEP
    return {"W_tot": w_tot, "W_ext": w_ext, "W_noext": w_noext, "decision": "DROP" if drop else "KEEP",
            "proj33_ext_h": N_PROJ * sec_ext / 3600, "proj33_noext_h": N_PROJ * sec_noext / 3600}


def build_inputs(dev: bool):
    from measure_rho import build_model_config
    cfg = build_model_config(CFG_DIR, batch_size=BATCH)          # reads all_config.yaml + dataset.json only
    if dev:
        cfg.update(hidden_size=32, num_heads=2, expansion=2, puzzle_emb_ndim=32)
    torch.manual_seed(SEED)
    _w, inner = build_ltf(cfg, "cpu")
    n_test, n_train = (120, 60) if dev else (1000, 512)
    test, _ = load_split(DATA_PATH, "test", n_test)
    train, _ = load_split(DATA_PATH, "train", n_train)
    return inner, test, train


def one_measurement(ext: bool, inner, test, train, n_perm: int) -> tuple[float, bool]:
    """Wall seconds for one cell-equivalent; second value: the permutation path was taken on every split."""
    zstats._OMEGA_CACHE.clear()
    readouts._cached_info.cache_clear()
    rng = np.random.default_rng(SEED)
    n_loops = 42 if ext else 21
    t0 = time.perf_counter()
    perm_taken = True
    for split, probe in (("test", test), ("train", train)):
        _dec, _g1 = decode_loops(inner, probe, n_loops, BATCH, check_g1_at=inner.config.loops)
        inp = probe["inputs"].numpy()[:, :10]
        lab = probe["labels"].numpy()[:, :10]
        infos = readouts.sigma_infos(inp, K)
        syn = synth_genuine_sequential(infos, K, len(judge.t_all()), rng, hold_from=(42 if ext else None))
        res = analyze_split(syn, inp, lab, K, infos, n_perm, {"split": split}, split, with_dgrid=False,
                            with_extrap=ext)
        cell = res["cell"][0]
        perm_taken &= cell["p"] is not None
    return time.perf_counter() - t0, perm_taken


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=REQUIRED_THREADS)
    ap.add_argument("--log", default=DEFAULT_LOG)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--dev", action="store_true", help="tests only: tiny model/data, any thread count, non-default log")
    ap.add_argument("--repo-root", default=PKG_ROOT)
    a = ap.parse_args(argv)
    require_cpu(a.device)
    if a.dev and (os.path.abspath(a.log) == DEFAULT_LOG or os.path.abspath(a.out) == DEFAULT_OUT):
        print("--dev must not use the default log/out paths", file=sys.stderr)
        return 6
    prior = completed_run_in_log(a.log)
    if prior:
        print(f"refusing: a completed run already binds (first completed run decides): {prior}", file=sys.stderr)
        return 5
    os.chdir(a.repo_root)
    torch.set_num_threads(a.threads)
    n_thr = torch.get_num_threads()
    argv_sha = hashlib.sha256("\0".join(sys.argv[1:] if argv is None else argv).encode()).hexdigest()
    append_flush(a.log, f"START {utc()} threads={n_thr} argv_sha={argv_sha}")
    required = a.threads if a.dev else REQUIRED_THREADS
    if n_thr != required:
        append_flush(a.log, f"FINISH {utc()} rc=3 EXTRAP=none")
        print(f"threads={n_thr} != {required}: no measurement", file=sys.stderr)
        return 3
    try:
        n_perm = 20 if a.dev else int(judge.num("czz2.N_PERM"))
        inner, test, train = build_inputs(a.dev)
        raw = {"ext": [], "noext": []}
        perm_ok = True
        for i, kind in enumerate(SEQUENCE, 1):
            sec, taken = one_measurement(kind == "ext", inner, test, train, n_perm)
            raw[kind].append(sec)
            perm_ok &= taken
            print(f"measurement {i}/6 ({kind}) done", flush=True)
        if not perm_ok:
            raise RuntimeError("permutation path was not taken on every synthetic split (gamma < GAMMA_MIN)")
        sec_ext, sec_noext = statistics.median(raw["ext"]), statistics.median(raw["noext"])
        d = decide(sec_ext, sec_noext)
    except BaseException as e:
        append_flush(a.log, f"FINISH {utc()} rc=4 EXTRAP=none")
        print(f"crash: {type(e).__name__}: {e}", file=sys.stderr)
        raise
    # FINISH (with EXTRAP) is durable BEFORE anything derived from the measurement is printed or written
    append_flush(a.log, f"FINISH {utc()} rc=0 EXTRAP={d['decision']}")
    lines = [
        f"gc_timing threads={n_thr} seed={SEED} N_run={N_RUN} X_ext_h={X_EXT_H} Y_tot_h={Y_TOT_H} dev={a.dev}",
        f"raw_sec_ext={raw['ext']}", f"raw_sec_noext={raw['noext']}",
        f"sec_ext={sec_ext:.3f} sec_noext={sec_noext:.3f} (medians of 3)",
        f"W_tot_h={d['W_tot']:.4f} W_ext_h={d['W_ext']:.4f} W_noext_h={d['W_noext']:.4f}",
        f"projection_33_cells_h: ext={d['proj33_ext_h']:.4f} noext={d['proj33_noext_h']:.4f} "
        f"(30 cells + 2 G-b cells + G5' re-run)",
        "disclosures: caches cleared before each measurement; every cell gets test 1000 + train 512 rows; "
        "synthetic decodes hold sigma^k for t = 22..42; D-grid statistics not timed; random init, no accuracy computed",
        f"EXTRAP: {d['decision']}",
    ]
    text = "\n".join(lines) + "\n"
    with open(a.out, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    sys.stdout.write(text)
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
