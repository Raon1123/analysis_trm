"""EXP-017 §5 figures. Only Z1 is implemented so far; Z2-Z10 are later work and each one would be added here.

Z1: a_j (agreement with σ^k) and v_j (valid-permutation rate) against the readout index j = 1..18,
for the preliminary readout ẑ_H(j) on the test split. It is drawn per family (F1/F2/F3), with one
column per k: thin lines for each cell and a thick line for the median over cells. Native readouts
j ∈ {6, 12, 18} are marked, and j = 18 is shaded as "trained", the only readout lm_head saw in
training. House style: /fig-style (rcParams preset, 300 dpi, png + pdf, viz_style tokens).

  uv run python analysis/z_dynamics/make_figures.py --data reports/figures/${TODAY}_z-dynamics/data \
      --out reports/figures/${TODAY}_z-dynamics
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from analysis.z_dynamics import judge  # noqa: E402

VIZ_STYLE_DIR = "/mnt/ayp/agents/trm/26_figure-pipeline/11_lab-figure-pipeline/25_style"
FS_MIN = 7

PUBLICATION_RCPARAMS = {
    "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 15,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 2,
    "legend.frameon": False,
    "svg.fonttype": "none",
}


def load_viz_style():
    if VIZ_STYLE_DIR not in sys.path:
        sys.path.insert(0, VIZ_STYLE_DIR)
    import viz_style  # read-only canonical palette (fig-style §4)
    return viz_style


def apply_publication_style(style=None):
    plt.rcParams.update(PUBLICATION_RCPARAMS if style is None else style)


def finalize_figure(fig, out_path, formats=("png", "pdf"), dpi=300, close=True, pad=1):
    fig.tight_layout(pad=pad)
    paths = []
    for ext in formats:
        p = f"{out_path}.{ext}"
        fig.savefig(p, dpi=dpi, facecolor="white")
        paths.append(p)
    if close:
        plt.close(fig)
    return paths


def read_per_call(data_dir: str) -> list[dict]:
    with open(os.path.join(data_dir, "per_call.csv"), newline="") as f:
        return list(csv.DictReader(f))


def z1_series(rows: list[dict], names: list[str]) -> dict:
    """run_name -> {"a": [18], "v": [18]} for test / prelim / gcd_class=all."""
    out: dict[str, dict] = defaultdict(lambda: {"a": [np.nan] * 18, "v": [np.nan] * 18})
    want = set(names)
    for r in rows:
        if r["run_name"] in want and r["split"] == "test" and r["readout_kind"] == "prelim" and r["gcd_class"] == "all":
            j = int(r["j"])
            out[r["run_name"]]["a"][j - 1] = float(r["a"])
            out[r["run_name"]]["v"][j - 1] = float(r["v"])
    return dict(out)


def k_of(name: str) -> int:
    return int(name.split("_k")[1].split("_")[0])


def make_z1(data_dir: str, out_dir: str) -> list[str]:
    vs = load_viz_style()
    apply_publication_style()
    rows = read_per_call(data_dir)
    native = judge.rule("readouts.native_j")
    trained = judge.rule("readouts.trained_j")
    paths = []
    for fam, names in judge.rule("families").items():
        ser = z1_series(rows, names)
        ks = sorted({k_of(n) for n in names})
        fig, axes = plt.subplots(2, len(ks), figsize=(4.2 * len(ks) + 1, 7.5), squeeze=False, sharey="row")
        j = np.arange(1, 19)
        for col, k in enumerate(ks):
            cell_names = [n for n in names if k_of(n) == k and n in ser]
            for row_i, (metric, lab) in enumerate((("a", "agreement $a_j$"), ("v", "valid perm. $v_j$"))):
                ax = axes[row_i, col]
                ax.axvspan(trained - 0.5, trained + 0.5, color=vs.STATUS_MUTED_LIGHT, zorder=0)
                for x in native:
                    ax.axvline(x, color=vs.GRID_HAIRLINE, lw=1.2, zorder=0)
                Y = np.array([ser[n][metric] for n in cell_names]) if cell_names else np.full((1, 18), np.nan)
                for y in Y:
                    ax.plot(j, y, color=vs.SEQUENTIAL_BLUE_STEPS[2], lw=1.2, alpha=0.8)
                if cell_names:
                    ax.plot(j, np.nanmedian(Y, 0), color=vs.DIVERGING_BLUE, lw=3, label="median over cells")
                ax.set_ylim(-0.02, 1.02)
                ax.set_xticks([1, 6, 12, 18])
                if row_i == 0:
                    ax.set_title(f"k = {k}  (n = {len(cell_names)})", fontsize=max(FS_MIN, 15))
                if row_i == 1:
                    ax.set_xlabel("readout j (native: 6, 12, 18; shaded: trained)", fontsize=max(FS_MIN, 12))
                if col == 0:
                    ax.set_ylabel(lab, fontsize=max(FS_MIN, 15))
        fig.suptitle(f"EXP-017 Z1 — {fam}: preliminary readout decoded through lm_head (test)",
                     fontsize=max(FS_MIN, 15), color=vs.INK_PRIMARY)
        paths += finalize_figure(fig, os.path.join(out_dir, f"Z1_{fam}"))
    return paths


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    man = json.load(open(os.path.join(a.data, "run_manifest.json")))
    if man.get("status") != "complete":
        print(f"warning: run_manifest status={man.get('status')!r} (figures from a non-complete run)", file=sys.stderr)
    os.makedirs(a.out, exist_ok=True)
    for p in make_z1(a.data, a.out):
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
