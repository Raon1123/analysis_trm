"""RF-11: reader timeline (Korean). One marker per experiment, rows in the order and numbering of
READER_TIMELINE_ko.csv, phases as light horizontal bands titled from PHASES_ko.csv.

Printed-size rule: every text element is >= 7 pt at the printed width (asserted here and by ctx.save).
"""
import csv
import datetime as dt
import re
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.text import Text

from .. import style

FIG_W = 12.0
FONT = 14
ROW_H = 0.30
X_MIN = dt.date(2026, 6, 1)
X_MAX = dt.date(2026, 11, 20)
MONTHS = {6: "6월", 7: "7월", 8: "8월", 9: "9월", 10: "10월", 11: "11월"}
_RE_CROSS = re.compile(r"(\d+)월 (\d+)일~(\d+)월 (\d+)일")
_RE_SAME = re.compile(r"(\d+)월 (\d+)~(\d+)일")
_RE_ONE = re.compile(r"(\d+)월 (\d+)일")
LEGEND = {"dot": "훈련을 시작한 날", "bar": "훈련들이 시작된 기간", "rerun": "앞 실험의 재훈련"}
# r3: a row number is either a main number "N" or a continuation "Nb"/"Nc" (a later re-run of experiment N,
# placed at its own start date and drawn with a hollow marker)
_RE_NO = re.compile(r"^(\d+)([a-z]?)$")


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def input_file(ctx, name):
    for inp in ctx.entry.inputs:
        if inp["role"] == "file" and Path(inp["path"]).name == name:
            return Path(inp["path"])
    raise RuntimeError(f"registry entry {ctx.fig_id} lists no file input named {name}")


def first_period(when_run, first_iso):
    """(start, end) of the first contiguous run period in the Korean 'when_run' text; start must equal first_iso."""
    seg = when_run.split(";")[0]
    year = int(first_iso[:4])
    m = _RE_CROSS.search(seg)
    if m:
        a, b = (int(m.group(1)), int(m.group(2))), (int(m.group(3)), int(m.group(4)))
    else:
        m = _RE_SAME.search(seg)
        if m:
            a, b = (int(m.group(1)), int(m.group(2))), (int(m.group(1)), int(m.group(3)))
        else:
            m = _RE_ONE.search(seg)
            if not m:
                raise ValueError(f"cannot read a date from when_run {when_run!r}")
            a = b = (int(m.group(1)), int(m.group(2)))
    start, end = dt.date(year, *a), dt.date(year, *b)
    if start.isoformat() != first_iso:
        raise ValueError(f"when_run {when_run!r} starts {start}, run_first_iso says {first_iso}")
    if end < start:
        raise ValueError(f"period ends before it starts: {when_run!r}")
    return start, end


def load_slots(reader_rows, phase_rows, label_rows):
    """Row slots top to bottom: for each phase a header slot, then its experiments in CSV order."""
    parsed = []
    for r in reader_rows:
        m = _RE_NO.match(r["no"].strip())
        if not m:
            raise ValueError(f"bad experiment number {r['no']!r}")
        parsed.append((int(m.group(1)), m.group(2)))
    mains = [n for n, suf in parsed if not suf]
    if mains != list(range(1, len(mains) + 1)):
        raise ValueError(f"experiment numbers are not 1..N in order: {mains}")
    for i, (n, suf) in enumerate(parsed):
        if suf and (n, "") not in parsed[:i]:
            raise ValueError(f"continuation row {n}{suf} comes before its experiment {n}")
    keys = [r["no"].strip() for r in reader_rows]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate experiment numbers")
    labels = {r["no"].strip(): r["label_ko"] for r in label_rows}
    if sorted(labels) != sorted(keys):
        raise ValueError("FIGLABELS numbers differ from the reader table")
    phases = {int(p["phase"]): p for p in phase_rows}
    ph_of = [int(r["phase"]) for r in reader_rows]
    if ph_of != sorted(ph_of):
        raise ValueError("phases are not in run order in the reader table")
    if sorted(set(ph_of)) != sorted(phases):
        raise ValueError("phase ids in the reader table and PHASES differ")
    slots = []
    for ph in sorted(phases):
        slots.append({"kind": "phase", "phase": ph, "no": None, "label": f"단계 {ph} · " + phases[ph]["title_ko"].strip()})
        for r in reader_rows:
            if int(r["phase"]) == ph:
                start, end = first_period(r["when_run"], r["run_first_iso"])
                key = r["no"].strip()
                rerun = not key.isdigit()
                slots.append({"kind": "exp", "phase": ph, "no": key if rerun else int(key), "rerun": rerun,
                              "label": f"{key}. {labels[key]}",
                              "start": start, "end": end, "timeline_id": r["timeline_id"], "when_run": r["when_run"]})
    return slots


def min_printed_pt(fig, print_width_in):
    scale = print_width_in / fig.get_figwidth()
    sizes = [t.get_fontsize() * scale for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    return min(sizes), scale


def build(ctx):
    if ctx.lang != "ko":
        raise ValueError("RF-11 is Korean only")
    rows = read_csv(input_file(ctx, "READER_TIMELINE_ko.csv"))
    slots = load_slots(rows, read_csv(input_file(ctx, "PHASES_ko.csv")), read_csv(input_file(ctx, "FIGLABELS_ko.csv")))
    n = len(slots)
    fig, ax = plt.subplots(figsize=(FIG_W, ROW_H * n + 1.3), layout="constrained")
    x = mdates.date2num
    blue = style.DIVERGING_BLUE
    # phase bands (header slot + its experiments), alternating light tints
    tints = (style.SEQUENTIAL_BLUE_STEPS[0], style.GRID_HAIRLINE)
    top = None
    for i, s in enumerate(slots):
        if s["kind"] == "phase":
            top = i
        last = i + 1 == n or slots[i + 1]["kind"] == "phase"
        if last:
            ax.axhspan(top - 0.5, i + 0.5, color=tints[(s["phase"] - 1) % 2], alpha=0.55, lw=0, zorder=0)
    values, order = [], []
    for i, s in enumerate(slots):
        order.append({"slot": i, "kind": s["kind"], "no": s["no"], "phase": s["phase"], "label": s["label"]})
        if s["kind"] != "exp":
            continue
        if s["end"] > s["start"]:
            ax.plot([x(s["start"]), x(s["end"])], [i, i], color=blue, lw=7, alpha=0.45, zorder=2, solid_capstyle="butt")
        if s.get("rerun"):
            ax.plot([x(s["start"])], [i], marker="o", ms=10, mfc="white", mec=blue, mew=2.2, zorder=5, ls="")
        else:
            ax.plot([x(s["start"])], [i], marker="o", ms=10, color=blue, mec="none", zorder=5, ls="")
        for series, d in (("run_first", s["start"]), ("run_period_end", s["end"])):
            values.append({
                "figure_id": ctx.fig_id, "panel": s["no"], "series": series, "run_name": "", "wandb_id": "",
                "project": "", "cohort": "", "metric": "date", "step": "", "value": "",
                "source_ref": f"READER_TIMELINE_ko.csv:no={s['no']}:{s['timeline_id']}", "date": d.isoformat(),
            })
    ax.set_xlim(x(X_MIN), x(X_MAX))
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels([s["label"] for s in slots], fontsize=FONT)
    for tl, s in zip(ax.get_yticklabels(), slots):
        if s["kind"] == "phase":
            tl.set_fontweight("bold")
    ax.tick_params(axis="y", length=0)
    ax.set_xticks([x(dt.date(2026, m, 1)) for m in MONTHS])
    ax.set_xticklabels(list(MONTHS.values()), fontsize=FONT)
    ax.tick_params(axis="x", width=1.5, length=5)
    ax.grid(axis="x", color="white", lw=1.5, zorder=1)
    ax.set_axisbelow(True)
    ax.set_xlabel("2026년", fontsize=FONT)
    for sp in ax.spines.values():
        sp.set_linewidth(1.5)
    handles = [
        Line2D([], [], marker="o", ms=10, color=blue, mec="none", ls="", label=LEGEND["dot"]),
        Line2D([], [], color=blue, lw=7, alpha=0.45, label=LEGEND["bar"]),
    ]
    if any(s.get("rerun") for s in slots):
        handles.append(Line2D([], [], marker="o", ms=10, mfc="white", mec=blue, mew=2.2, ls="", label=LEGEND["rerun"]))
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), fontsize=FONT, frameon=False)
    fig.canvas.draw()
    lo, scale = min_printed_pt(fig, ctx.entry.print_width_in)
    assert lo >= style.MIN_PRINT_PT, f"smallest text prints at {lo:.2f} pt (< {style.MIN_PRINT_PT} pt)"
    ctx.save(fig)
    ctx.write_values(pd.DataFrame(values))
    ctx.write_extra_csv("order", pd.DataFrame(order))
    ctx.write_source({
        "rows_used": {str(input_file(ctx, "READER_TIMELINE_ko.csv")): len(rows)},
        "n_experiments": sum(1 for r in rows if r["no"].strip().isdigit()),
        "n_reruns": sum(1 for r in rows if not r["no"].strip().isdigit()), "n_phases": len({s["phase"] for s in slots}),
        "figure_width_in": FIG_W, "print_width_in": ctx.entry.print_width_in,
        "print_scale": round(scale, 4), "min_printed_pt": round(lo, 3),
    })
