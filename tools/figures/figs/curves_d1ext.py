"""RF-04: second-generation-data (d1) trainings, with and without the halting bundle, k = 5, 6, 7.

Plots probe/test_exact (solid) and probe/train_exact (dashed) only; three seeds per panel.
"""
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from .. import style
from ..summaries import PROBE_TEST, PROBE_TRAIN, check_panel, wrap_em
from .curves_budget import (
    CELLS_COLUMNS, THETA, cells_frame, cells_row, config_names, config_of, input_path, kfmt,
    load_names, only_probe, protocol_extra, run_series, values_rows,
)

BUDGET = 122050
STEP_MAX = 185516
KS = (5, 6, 7)
ROWS = (("tf_noz_iter", "CFG-tf_noz_iter"), ("act_noz_iter_h16", "CFG-act_noz_iter_h16"))
FIG_W = 13.2
SEED_COLOURS = {1: style.DIVERGING_BLUE, 2: style.STATUS_WARNING_EDGE, 3: style.SEQUENTIAL_BLUE_STEPS[2]}

LOCAL = {
    "en": {
        "seed": "seed", "leg_test": "test exact match (solid)", "leg_train": "train exact match (dashed)",
        "leg_budget": "budget of the second-generation data (122,050 steps)", "leg_thr": "0.9 line",
    },
    "ko": {
        "seed": "seed", "leg_test": "시험 정확도(실선)", "leg_train": "훈련 정확도(점선)",
        "leg_budget": "2세대 데이터의 예산(122,050 step)", "leg_thr": "0.9 선",
    },
}


def build(ctx):
    lang = ctx.lang
    L = LOCAL[lang]
    names = load_names(input_path(ctx, "NAMES.csv"))
    ev = only_probe(ctx.load_cache("d1_ext.eval.csv"))
    ev["config"] = [config_of(n) for n in ev["run_name"]]
    fig, axes = plt.subplots(len(ROWS), len(KS), figsize=(FIG_W, 4.6 * len(ROWS) + 1.2), squeeze=False,
                             layout="constrained")
    values, cells = [], []
    for ri, (_, cfg) in enumerate(ROWS):
        for ci, k in enumerate(KS):
            ax = axes[ri][ci]
            d = ev[(ev["config"] == cfg) & (ev["k"] == k)]
            if d.empty:
                raise ValueError(f"no runs for {cfg}, k={k}")
            panel = f"{cfg}_k{k}"
            seeds = sorted(d["seed"].unique())
            check_panel(metrics=sorted(d["metric"].unique()), cohorts=d["cohort"].unique(), tagged=True)
            ax.axhline(THETA, color=style.INK_MUTED, lw=1.5, ls=(0, (1, 2)), zorder=1)
            ax.axvline(BUDGET, color=style.INK_SECONDARY, lw=1.5, zorder=1)
            for s in seeds:
                dr = d[d["seed"] == s]
                if dr["run_name"].nunique() != 1 or dr["wandb_id"].nunique() != 1:
                    raise ValueError(f"{panel} seed {s}: not exactly one run")
                for metric, ls in ((PROBE_TRAIN, (0, (3.5, 2))), (PROBE_TEST, "-")):
                    st, va = run_series(dr, metric)
                    ax.plot(st, va, color=SEED_COLOURS[int(s)], lw=2.4, ls=ls, zorder=3, solid_capstyle="butt",
                            label=f"{L['seed']} {int(s)}" if metric == PROBE_TEST else None)
                values += values_rows(ctx.fig_id, panel, "ext", dr, "d1_ext.eval.csv")
                cells.append(cells_row(ctx.fig_id, panel, "ext", dr, names, BUDGET))
            ax.text(BUDGET, 0.05, style.word(lang, "budget") + " ", rotation=90, ha="right", va="bottom",
                    fontsize=14, color=style.INK_SECONDARY, zorder=5,
                    bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.85))
            for sp in ax.spines.values():
                sp.set_linewidth(1.5)
            ax.tick_params(labelsize=14, width=1.5, length=5)
            ax.set_xlim(0, STEP_MAX)
            ax.set_ylim(-0.03, 1.03)
            ax.set_yticks([0, 0.5, 1])
            ax.set_yticklabels(["0", "0.5", "1"])
            ax.set_xticks([0, 50000, 100000, 150000])
            ax.xaxis.set_major_formatter(plt.FuncFormatter(kfmt))
            if ri == len(ROWS) - 1:
                ax.set_xlabel(style.word(lang, "step"), fontsize=15)
            if ci == 0:
                ax.set_ylabel(style.word(lang, "exact match"), fontsize=15)
            _, name_en, name_ko = config_names(names, d["run_name"].iloc[0])
            name = name_en if lang == "en" else name_ko
            lines = wrap_em(name, 4.0 * 72 / 15 * (1.05 if lang == 'en' else 1.2))
            ax.set_title("\n".join(lines) + f"\nk = {k}", fontsize=15, pad=14 * 1.3 + 6)
            ax.annotate(style.word(lang, "d1"), xy=(0.5, 1.0), xytext=(0, 4), xycoords="axes fraction",
                        textcoords="offset points", ha="center", va="bottom", fontsize=14,
                        color=style.INK_SECONDARY)
    handles = [Line2D([], [], color=SEED_COLOURS[s], lw=2.4, label=f"{L['seed']} {s}") for s in (1, 2, 3)]
    handles += [
        Line2D([], [], color=style.INK_PRIMARY, lw=2.0, ls="-", label=L["leg_test"]),
        Line2D([], [], color=style.INK_PRIMARY, lw=2.0, ls=(0, (3.5, 2)), label=L["leg_train"]),
        Line2D([], [], color=style.INK_SECONDARY, lw=1.5, label=L["leg_budget"]),
        Line2D([], [], color=style.INK_MUTED, lw=1.5, ls=(0, (1, 2)), label=L["leg_thr"]),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=14, frameon=False)
    ctx.save(fig)
    ctx.write_values(pd.DataFrame(values))
    ctx.write_extra_csv("cells", cells_frame(cells))
    ctx.write_source({"rows_used": {"d1_ext.eval.csv": len(values)}, **protocol_extra(ctx),
                      "budget_steps": BUDGET, "threshold": THETA})


__all__ = ["build", "CELLS_COLUMNS"]
