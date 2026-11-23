"""RF-03: the nine twice-the-steps (2x) trainings against their same-named once-the-steps (1x) trainings.

A 2x run and a 1x run are different trainings: every line starts at step 0 and nothing joins them.
Plots probe/test_exact (solid) and probe/train_exact (dashed) only.
"""
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from .. import io as fio
from .. import style
from ..summaries import PROBE_TEST, PROBE_TRAIN, check_panel, summarize_run, wrap_em

BUDGET = 244100
STEP_MAX = 488200
THETA = 0.9
METRICS = (PROBE_TEST, PROBE_TRAIN)
FIG_W = 13.2
PANEL_W = 4.0

CELLS_COLUMNS = [
    "figure_id", "panel", "series", "run_name", "wandb_id", "project", "cohort", "cohort_word_en",
    "cohort_word_ko", "config", "config_name_en", "config_name_ko", "k", "seed", "budget",
    "first_reach_step", "sustained_start", "final_value", "max_value", "n_evals", "n_at_or_above",
    "value_at_budget", "reached_by_budget", "train_final",
]

# (record_id in NAMES.csv, substring of the run name); exactly one substring must match.
CONFIGS = (
    ("CFG-tf_z_iter", "tf_z_iter"),
    ("CFG-tf_noz_iter", "tf_noz_iter"),
    ("CFG-act_noz_iter_h16", "act_noz_iter_h16"),
)

LOCAL = {
    "en": {
        "run2": "2x run", "run1": "1x run", "run2_long": "2x run (twice the steps)",
        "single": "1x runs", "seed": "seed",
        "line_test": "test", "line_train": "train",
        "leg_test": "test exact match (solid)", "leg_train": "train exact match (dashed)",
        "leg_budget": "budget of the 1x runs (244,100 steps)", "leg_thr": "0.9 line",
        "leg_1l": "1x run, legacy data", "leg_1c": "1x run, clean data (d0)", "leg_2": "2x run",
    },
    "ko": {
        "run2": "2배 훈련", "run1": "1배 훈련", "run2_long": "2배 훈련(step 두 배)",
        "single": "1배 훈련", "seed": "seed",
        "line_test": "시험", "line_train": "훈련",
        "leg_test": "시험 정확도(실선)", "leg_train": "훈련 정확도(점선)",
        "leg_budget": "1배 훈련의 예산(244,100 step)", "leg_thr": "0.9 선",
        "leg_1l": "1배 훈련, 옛 데이터", "leg_1c": "1배 훈련, 고친 데이터", "leg_2": "2배 훈련",
    },
}


def input_path(ctx, name):
    for inp in ctx.entry.inputs:
        if inp["role"] == "file" and Path(inp["path"]).name == name:
            return Path(inp["path"])
    raise RuntimeError(f"registry entry {ctx.fig_id} lists no file input named {name}")


def load_names(path):
    with open(path, newline="", encoding="utf-8") as f:
        return {r["record_id"]: r for r in csv.DictReader(f)}


def config_of(run_name):
    hit = [cid for cid, sub in CONFIGS if sub in run_name]
    if len(hit) != 1:
        raise ValueError(f"run name {run_name!r} matches {len(hit)} of the three configurations")
    return hit[0]


def config_names(names, run_name):
    cid = config_of(run_name)
    if cid not in names:
        raise ValueError(f"NAMES.csv has no row for configuration {cid}")
    r = names[cid]
    if not r["name_en"].strip() or not r["name_ko"].strip():
        raise ValueError(f"NAMES.csv row {cid} has an empty name")
    return cid, r["name_en"], r["name_ko"]


def only_probe(df):
    out = df[df["metric"].isin(METRICS)].copy()
    for c in ("run_name", "wandb_id", "project", "cohort", "metric"):
        out[c] = out[c].astype(str)
    return out


def run_series(df, metric):
    d = df[df["metric"] == metric].sort_values("step")
    return d["step"].tolist(), d["value"].tolist()


def cells_row(fig_id, panel, series, run_df, names, budget):
    r0 = run_df.iloc[0]
    run_name = str(r0["run_name"])
    cid, name_en, name_ko = config_names(names, run_name)
    cohort = str(r0["cohort"])
    st, va = run_series(run_df, PROBE_TEST)
    s = summarize_run(st, va, theta=THETA, budget=budget)
    _, tr = run_series(run_df, PROBE_TRAIN)
    return {
        "figure_id": fig_id, "panel": panel, "series": series, "run_name": run_name,
        "wandb_id": str(r0["wandb_id"]), "project": str(r0["project"]), "cohort": cohort,
        "cohort_word_en": style.word("en", cohort), "cohort_word_ko": style.word("ko", cohort),
        "config": cid, "config_name_en": name_en, "config_name_ko": name_ko,
        "k": int(r0["k"]), "seed": int(r0["seed"]), "budget": int(budget),
        "first_reach_step": s["first_reach_step"], "sustained_start": s["sustained_start"],
        "final_value": s["final_value"], "max_value": s["max_value"], "n_evals": s["n_evals"],
        "n_at_or_above": s["n_at_or_above"], "value_at_budget": s["value_at_budget"],
        "reached_by_budget": "true" if s["reached_by_budget"] else "false",
        "train_final": tr[-1] if tr else None,
    }


def cells_frame(rows):
    df = pd.DataFrame(rows, columns=CELLS_COLUMNS)
    return df.astype(object).where(df.notna(), None)


def values_rows(fig_id, panel, series, run_df, source_file):
    out = []
    for r in run_df.itertuples(index=False):
        out.append({
            "figure_id": fig_id, "panel": panel, "series": series, "run_name": r.run_name,
            "wandb_id": r.wandb_id, "project": r.project, "cohort": r.cohort, "metric": r.metric,
            "step": int(r.step), "value": float(r.value),
            "source_ref": f"{source_file}:{r.run_name}/{r.wandb_id}",
        })
    return out


def protocol_extra(ctx):
    path = ctx.cache_dir / "PROTOCOLS.json"
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    metrics = d["protocols"]["metrics"]
    block = {}
    for m in METRICS:
        e = metrics[m]
        block[m] = {k: e[k] for k in ("eval_set", "n_examples", "weights", "status", "plain_en", "plain_ko")}
    return {"protocol_file": str(path), "protocol_file_sha256": fio.sha256_file(path), "protocol_metrics": block}


def style_axes(ax):
    for sp in ax.spines.values():
        sp.set_linewidth(1.5)
    ax.tick_params(labelsize=14, width=1.5, length=5)
    ax.set_xlim(0, STEP_MAX)
    ax.set_ylim(-0.03, 1.03)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["0", "0.5", "1"])


def kfmt(x, _pos=None):
    return "0" if x == 0 else f"{x / 1000:.0f}k"


def title_for(lang, name, k, seed, width_em):
    lines = wrap_em(name, width_em)
    return "\n".join(lines) + f"\nk = {k}, {LOCAL[lang]['seed']} {seed}", len(lines) + 1


def build(ctx):
    lang = ctx.lang
    L = LOCAL[lang]
    names = load_names(input_path(ctx, "NAMES.csv"))
    ev2 = only_probe(ctx.load_cache("ext_2x.eval.csv"))
    evl = only_probe(ctx.load_cache("legacy_1x.eval.csv"))
    evc = only_probe(ctx.load_cache("clean_1x.eval.csv"))
    order = {cid: i for i, (cid, _) in enumerate(CONFIGS)}
    runs2 = sorted(
        ev2.groupby("run_name").first().reset_index().itertuples(index=False),
        key=lambda r: (order[config_of(r.run_name)], int(r.k), int(r.seed), r.run_name),
    )
    n = len(runs2)
    ncol = 3
    nrow = math.ceil(n / ncol)
    rows_h = 4.3
    fig, axes = plt.subplots(nrow, ncol, figsize=(FIG_W, rows_h * nrow + 1.3), squeeze=False,
                             layout="constrained")
    colour = {"2x": style.STATUS_WARNING_EDGE, "1x_legacy": style.INK_MUTED, "1x_clean": style.DIVERGING_BLUE}
    values, cells, used = [], [], {"ext_2x.eval.csv": 0, "legacy_1x.eval.csv": 0, "clean_1x.eval.csv": 0}
    for i, r2 in enumerate(runs2):
        ax = axes[i // ncol][i % ncol]
        panel = r2.run_name
        d2 = ev2[ev2["run_name"] == panel]
        if d2["wandb_id"].nunique() != 1:
            raise ValueError(f"{panel}: more than one wandb id in ext_2x")
        if not panel.startswith("fig1x2_"):
            raise ValueError(f"unexpected 2x run name {panel!r}")
        one = "fig1_" + panel[len("fig1x2_"):]
        parts = [("2x", "ext_2x.eval.csv", d2)]
        for ser, fname, ev in (("1x_legacy", "legacy_1x.eval.csv", evl), ("1x_clean", "clean_1x.eval.csv", evc)):
            d1 = ev[ev["run_name"] == one]
            if len(d1):
                if d1["wandb_id"].nunique() != 1:
                    raise ValueError(f"{one}: more than one wandb id in {fname}")
                parts.append((ser, fname, d1))
        cohorts = [str(p[2]["cohort"].iloc[0]) for p in parts]
        check_panel(metrics=sorted({m for p in parts for m in p[2]["metric"].unique()}), cohorts=cohorts, tagged=True)
        ax.axhline(THETA, color=style.INK_MUTED, lw=1.5, ls=(0, (1, 2)), zorder=1)
        ax.axvline(BUDGET, color=style.INK_SECONDARY, lw=1.5, zorder=1)
        for ser, fname, d in sorted(parts, key=lambda p: p[0] == "2x"):
            thick = ser == "2x"
            for metric, ls in ((PROBE_TRAIN, (0, (3.5, 2))), (PROBE_TEST, "-")):
                st, va = run_series(d, metric)
                ax.plot(st, va, color=colour[ser], lw=2.6 if thick else 1.4, ls=ls,
                        zorder=4 if thick else 3, solid_capstyle="butt")
            values += values_rows(ctx.fig_id, panel, ser, d, fname)
            used[fname] += len(d)
            cells.append(cells_row(ctx.fig_id, panel, ser, d, names, BUDGET))
        ax.text(BUDGET, 0.05, style.word(lang, "budget") + " ", rotation=90, ha="right", va="bottom",
                fontsize=14, color=style.INK_SECONDARY, zorder=5,
                bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.85))
        style_axes(ax)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(kfmt))
        ax.set_xticks([0, 100000, 200000, 300000, 400000])
        if i // ncol == nrow - 1 or i + ncol >= n:
            ax.set_xlabel(style.word(lang, "step"), fontsize=15)
        if i % ncol == 0:
            ax.set_ylabel(style.word(lang, "exact match"), fontsize=15)
        cid, name_en, name_ko = config_names(names, panel)
        width_em = PANEL_W * 72 / 15 * (1.05 if lang == 'en' else 1.2)
        title, _ = title_for(lang, name_en if lang == "en" else name_ko, int(r2.k), int(r2.seed), width_em)
        cw = lambda c: style.word(lang, c)  # noqa: E731
        singles = [cw(str(p[2]["cohort"].iloc[0])) for p in parts if p[0] != "2x"]
        tag1 = f"{L['run2']}: {cw(cohorts[0])}"
        tag2 = f"{L['single']}: " + "; ".join(singles) if singles else ""
        tags = "\n".join(t for t in (tag1, tag2) if t)
        ax.set_title(title, fontsize=15, pad=14 * 1.3 * (1 + (1 if tag2 else 0)) + 6)
        ax.annotate(tags, xy=(0.5, 1.0), xytext=(0, 4), xycoords="axes fraction", textcoords="offset points",
                    ha="center", va="bottom", fontsize=14, color=style.INK_SECONDARY, linespacing=1.25)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    handles = [
        Line2D([], [], color=colour["2x"], lw=2.6, label=L["leg_2"]),
        Line2D([], [], color=colour["1x_legacy"], lw=1.4, label=L["leg_1l"]),
        Line2D([], [], color=colour["1x_clean"], lw=1.4, label=L["leg_1c"]),
        Line2D([], [], color=style.INK_PRIMARY, lw=2.0, ls="-", label=L["leg_test"]),
        Line2D([], [], color=style.INK_PRIMARY, lw=2.0, ls=(0, (3.5, 2)), label=L["leg_train"]),
        Line2D([], [], color=style.INK_SECONDARY, lw=1.5, label=L["leg_budget"]),
        Line2D([], [], color=style.INK_MUTED, lw=1.5, ls=(0, (1, 2)), label=L["leg_thr"]),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=14, frameon=False)
    ctx.save(fig)
    ctx.write_values(pd.DataFrame(values))
    ctx.write_extra_csv("cells", cells_frame(cells))
    ctx.write_source({"rows_used": used, **protocol_extra(ctx), "budget_steps": BUDGET, "threshold": THETA})
