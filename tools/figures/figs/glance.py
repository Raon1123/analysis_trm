"""RF-02: the at-a-glance grid. One line per change; four facts per line; the scoring cell is coloured by class."""
import csv

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

from .. import style
from ..summaries import wrap_em
from .curves_budget import input_path

FIG_W = 13.4
EN_DASH = "–"
FONT = 14
LINE_IN = FONT * 1.3 / 72
PAD_IN = 0.09
COL_KEYS = ("change", "data_generation", "runs_per_setting", "compute_matched", "scoring")
COL_W = (4.5, 2.4, 1.5, 2.9, 2.1)  # inches, sums to FIG_W

HEADERS = {
    "en": ("Change tested", "Data generation", "Trainings per setting", "Compared at equal compute?", "Scored today"),
    "ko": ("바꾼 것", "데이터 세대", "설정당 훈련 수", "같은 계산량으로 비교?", "현재 평가"),
}

# One state colour per scoring class (palette tokens only; a state key, not an accuracy ramp).
CLASS_FILL = {
    "observation": style.SEQUENTIAL_BLUE_STEPS[0],
    "association": style.STATUS_WARNING,
    "non-move": style.STATUS_MUTED,
    "matched-contrast": style.SEQUENTIAL_BLUE_STEPS[2],
    "mixed": style.STATUS_WARNING_EDGE,
    "other": style.DIVERGING_MID,
    "blank": style.SURFACE,
}

REQUIRED = ("row_no change_en change_ko data_generation runs_per_setting compute_matched_en compute_matched_ko "
            "scoring_word scoring_class scoring_source registry_token_verbatim registry_source seen_en seen_ko").split()


def _read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def scoring_phrase(gloss, word, lang):
    w = (word or "").strip()
    if not w:
        return EN_DASH
    hit = [g for g in gloss if g["kind"].strip() == "scoring" and g["token"].strip() == w]
    if not hit:
        raise ValueError(f"GLOSS.csv has no scoring row for scoring_word {word!r}")
    if len(hit) > 1:
        raise ValueError(f"GLOSS.csv has {len(hit)} scoring rows for scoring_word {word!r}")
    return hit[0][f"plain_{lang}"]


def data_word(names, gen, lang):
    g = (gen or "").strip()
    if g == "none" or not g:
        return EN_DASH
    r = names.get(g)
    if r is None:
        raise ValueError(f"NAMES.csv has no row for data generation {gen!r}")
    return r[f"name_{lang}"]


def cell_texts(row, names, gloss, lang):
    return {
        "change": row[f"change_{lang}"],
        "data_generation": data_word(names, row["data_generation"], lang),
        "runs_per_setting": row["runs_per_setting"],
        "compute_matched": row[f"compute_matched_{lang}"],
        "scoring": scoring_phrase(gloss, row["scoring_word"], lang),
    }


def class_of(row):
    c = (row["scoring_class"] or "").strip() or "blank"
    if c not in CLASS_FILL:
        raise ValueError(f"unknown scoring_class {c!r} in row {row['row_no']}")
    return c


def build(ctx):
    lang = ctx.lang
    gpath = input_path(ctx, "GLANCE.csv")
    rows = _read(gpath)
    if not rows:
        raise ValueError("GLANCE.csv has no rows")
    missing = [c for c in REQUIRED if c not in rows[0]]
    if missing:
        raise ValueError(f"GLANCE.csv lacks columns {missing}")
    gloss = _read(input_path(ctx, "GLOSS.csv"))
    names = {r["record_id"]: r for r in _read(input_path(ctx, "NAMES.csv"))}

    wrapped = []  # per row: list of wrapped line lists per column
    for r in rows:
        t = cell_texts(r, names, gloss, lang)
        wrapped.append([wrap_em(t[k], (COL_W[j] - 2 * PAD_IN) * 72 / FONT) for j, k in enumerate(COL_KEYS)])
    head = [wrap_em(h, (COL_W[j] - 2 * PAD_IN) * 72 / FONT) for j, h in enumerate(HEADERS[lang])]

    def height(lines_per_col):
        return max(len(c) for c in lines_per_col) * LINE_IN + 2 * PAD_IN

    heights = [height(head)] + [height(w) for w in wrapped]
    total_h = sum(heights)
    fig = plt.figure(figsize=(FIG_W, total_h))
    fig.set_layout_engine("none")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(total_h, 0)
    ax.axis("off")
    xs = [sum(COL_W[:j]) for j in range(len(COL_W) + 1)]
    texts = []

    def cell(x0, x1, y0, y1, lines, fill, tc, bold=False, ha="left"):
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=fill, ec=style.INK_MUTED, lw=1.0))
        tx = x0 + PAD_IN if ha == "left" else (x0 + x1) / 2
        t = ax.text(tx, (y0 + y1) / 2, "\n".join(lines), ha=ha, va="center", fontsize=FONT, color=tc,
                    fontweight="bold" if bold else "normal", linespacing=1.3)
        texts.append((t, x0, x1, y0, y1))

    y = 0.0
    for j in range(5):
        cell(xs[j], xs[j + 1], y, y + heights[0], head[j], style.INK_PRIMARY, "white", bold=True)
    y += heights[0]
    values = []
    for i, (r, w) in enumerate(zip(rows, wrapped)):
        h = heights[i + 1]
        cls = class_of(r)
        for j, k in enumerate(COL_KEYS):
            if k == "scoring":
                fill = CLASS_FILL[cls]
                tc = style.text_color_for(mcolors.to_rgba(fill), threshold=0.5)
            else:
                fill, tc = style.SURFACE, style.INK_PRIMARY
            cell(xs[j], xs[j + 1], y, y + h, w[j], fill, tc)
            values.append({
                "figure_id": ctx.fig_id, "panel": "grid", "series": k, "run_name": "", "wandb_id": "",
                "project": "", "cohort": "", "metric": "", "step": "", "value": "",
                "source_ref": f"GLANCE:{r['row_no']}", "column": k,
                "text": cell_texts(r, names, gloss, "en")[k], "text_ko": cell_texts(r, names, gloss, "ko")[k],
                "scoring_class": cls if k == "scoring" else "",
            })
        y += h
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    for t, x0, x1, y0, y1 in texts:
        bb = t.get_window_extent(rend)
        (bx0, by0), (bx1, by1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        if bx1 > x1 - 0.02 or bx0 < x0 + 0.02 or max(by0, by1) > y1 or min(by0, by1) < y0:
            raise ValueError(f"cell text overflows its cell: {t.get_text()[:40]!r}")
    ctx.save(fig)
    ctx.write_values(pd.DataFrame(values))
    ctx.write_source({"rows_used": {str(gpath): len(rows)}, "n_cells": len(values)})
