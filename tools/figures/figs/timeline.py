"""RF-01: one lane per numbered row of the experiment timeline, with six dated separators."""
import csv
import datetime as dt
import json
import re
import textwrap

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from .. import style
from .curves_budget import input_path

X_MIN = dt.date(2026, 5, 1)
X_MAX = dt.date(2026, 11, 15)
LITERALS = ("not recorded", "not applicable (post-hoc)", "no verdict recorded")
DATE_FIELDS = (("date_prereg", "plan"), ("date_run", "run_start"), ("date_verdict", "verdict"))
_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}$")
FIG_W = 13.4

LOCAL = {
    "en": {
        "xlabel": "date (2026)",
        "months": ["May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov"],
        "plan": "plan date", "run_start": "start of the runs", "verdict": "verdict date",
        "run_latest": "latest run date (no verdict recorded)",
        "wrap": 17,
    },
    "ko": {
        "xlabel": "날짜 (2026년)",
        "months": ["5월", "6월", "7월", "8월", "9월", "10월", "11월"],
        "plan": "계획일", "run_start": "훈련 시작", "verdict": "판정일",
        "run_latest": "마지막 훈련일(판정 기록 없음)",
        "wrap": 9,
    },
}


def parse_date(value, row_id, field):
    """Closed format of the three date fields; returns a date, or None for the three literals."""
    if not isinstance(value, str):
        raise ValueError(f"row {row_id} field {field}: not a string: {value!r}")
    s = value.strip()
    if s in LITERALS:
        return None
    if s.startswith("about "):
        s = s[len("about "):]
    parts = s.split("..")
    if len(parts) > 2 or not all(_ISO.match(p) for p in parts):
        raise ValueError(f"row {row_id} field {field}: unrecognised date value {value!r}")
    try:
        return dt.date.fromisoformat(parts[0])
    except ValueError as e:
        raise ValueError(f"row {row_id} field {field}: invalid date {value!r}") from e


def _names(path):
    with open(path, newline="", encoding="utf-8") as f:
        return {r["record_id"]: r for r in csv.DictReader(f)}


def plain_name(names, rec_id, lang):
    r = names.get(rec_id)
    if r is None or not r.get(f"name_{lang}", "").strip():
        raise ValueError(f"NAMES.csv has no name_{lang} for id {rec_id}")
    return r[f"name_{lang}"]


def collect(rows, names, lang):
    """Returns (lanes, separators). Lane: dict(id, no, label, plan, run_start, verdict, hollow)."""
    lanes, seps = [], []
    for r in rows:
        rid = r["id"]
        if r.get("separator"):
            sd = r.get("separator_date")
            if not sd or not isinstance(sd, str) or not _ISO.match(sd):
                raise ValueError(f"separator row {rid} has no ISO separator_date: {sd!r}")
            seps.append({"id": rid, "date": dt.date.fromisoformat(sd), "label": plain_name(names, rid, lang)})
        if r.get("main_row_no") is None:
            continue
        no = int(r["main_row_no"])
        d = {f: parse_date(r[f], rid, f) for f, _ in DATE_FIELDS}
        lane = {
            "id": rid, "no": no, "label": f"{no}. {plain_name(names, rid, lang)}",
            "plan": d["date_prereg"], "run_start": d["date_run"], "verdict": d["date_verdict"],
            "hollow": None,
        }
        if not r.get("verdict_recorded"):
            lane["verdict"] = None
            lane["hollow"] = d["date_run"]
            lane["run_start"] = None
        lanes.append(lane)
    return lanes, sorted(seps, key=lambda s: s["date"])


def build(ctx):
    lang = ctx.lang
    L = LOCAL[lang]
    tl_path = input_path(ctx, "TIMELINE.json")
    names = _names(input_path(ctx, "NAMES.csv"))
    with open(tl_path, encoding="utf-8") as f:
        rows = json.load(f)["rows"]
    lanes, seps = collect(rows, names, lang)
    if not lanes:
        raise ValueError("timeline has no numbered rows")
    n = len(lanes)
    fig, ax = plt.subplots(figsize=(FIG_W, 0.37 * n + 4.4), layout="constrained")
    x = mdates.date2num
    blue, ink = style.DIVERGING_BLUE, style.INK_PRIMARY
    for i in range(n):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color=style.GRID_HAIRLINE, alpha=0.45, lw=0, zorder=0)
    values = []

    def val(panel, series, rid, date):
        values.append({
            "figure_id": ctx.fig_id, "panel": panel, "series": series, "run_name": "", "wandb_id": "",
            "project": "", "cohort": "", "metric": "date", "step": "", "value": "",
            "source_ref": f"TIMELINE:{rid}", "date": date.isoformat(),
        })

    for i, ln in enumerate(lanes):
        a, b = ln["run_start"], ln["verdict"]
        if a and b:
            ax.plot([x(a), x(b)], [i, i], color=blue, lw=3.0, zorder=2, solid_capstyle="butt")
        if ln["plan"]:
            ax.plot([x(ln["plan"])], [i], marker="^", ms=10, color=style.INK_SECONDARY, mec="none", zorder=4, ls="")
            val(ln["no"], "plan", ln["id"], ln["plan"])
        if a:
            ax.plot([x(a)], [i], marker="o", ms=9, color=blue, mec="none", zorder=5, ls="")
            val(ln["no"], "run_start", ln["id"], a)
        if b:
            ax.plot([x(b)], [i], marker="D", ms=9, color=blue, mec="none", zorder=5, ls="")
            val(ln["no"], "verdict", ln["id"], b)
        if ln["hollow"]:
            ax.plot([x(ln["hollow"])], [i], marker="o", ms=9, mfc="white", mec=ink, mew=1.8, zorder=5, ls="")
            val(ln["no"], "run_latest", ln["id"], ln["hollow"])
    for s in seps:
        ax.axvline(x(s["date"]), color=style.INK_SECONDARY, lw=1.5, ls=(0, (4, 3)), zorder=1, clip_on=False)
        val("separator", "separator", s["id"], s["date"])
    # separator labels above the lanes, rotated; the last two sit on opposite sides of their lines
    close = [i for i in range(1, len(seps)) if (seps[i]["date"] - seps[i - 1]["date"]).days < 10]
    for i, s in enumerate(seps):
        ha = "right" if (i + 1) in close else "left"
        txt = "\n".join(textwrap.wrap(s["label"], L["wrap"]))
        off = -3 if ha == "right" else 3
        ax.annotate(txt, xy=(x(s["date"]), 1), xycoords=("data", "axes fraction"), xytext=(off, 4),
                    textcoords="offset points", rotation=90, ha=ha, va="bottom", fontsize=14,
                    color=ink, linespacing=1.15, annotation_clip=False)
    ax.set_xlim(x(X_MIN), x(X_MAX))
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels([ln["label"] for ln in lanes], fontsize=14)
    ax.tick_params(axis="y", length=0)
    month_starts = [dt.date(2026, m, 1) for m in range(5, 12)]
    ax.set_xticks([x(d) for d in month_starts])
    ax.set_xticklabels(L["months"], fontsize=14)
    ax.tick_params(axis="x", width=1.5, length=5)
    ax.grid(axis="x", color=style.GRID_HAIRLINE, lw=1.0, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel(L["xlabel"], fontsize=15)
    for sp in ax.spines.values():
        sp.set_linewidth(1.5)
    handles = [
        Line2D([], [], marker="^", ms=10, color=style.INK_SECONDARY, mec="none", ls="", label=L["plan"]),
        Line2D([], [], marker="o", ms=9, color=blue, mec="none", ls="", label=L["run_start"]),
        Line2D([], [], marker="D", ms=9, color=blue, mec="none", ls="", label=L["verdict"]),
        Line2D([], [], marker="o", ms=9, mfc="white", mec=ink, mew=1.8, ls="", label=L["run_latest"]),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=14, frameon=False)
    ctx.save(fig)
    df = pd.DataFrame(values)
    ctx.write_values(df)
    ctx.write_source({"rows_used": {str(tl_path): len(rows)}, "n_lanes": n, "n_separators": len(seps)})
