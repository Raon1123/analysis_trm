"""Tiny self-test figure (id SELFTEST), drawn from numbers in this module. Tests only."""
import os

import matplotlib.pyplot as plt
import pandas as pd

from .. import style

STEPS = [0, 100, 200, 300, 400]
VALUES = [0.05, 0.30, 0.62, 0.88, 0.95]


def build(ctx):
    lang = ctx.lang
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(STEPS, VALUES, color=style.DIVERGING_BLUE, lw=2.5, label=style.word(lang, "test"))
    ax.axhline(0.9, color=style.INK_MUTED, lw=1.5, ls="--", label=style.word(lang, "threshold 0.9"))
    ax.set_xlabel(style.word(lang, "step"))
    ax.set_ylabel(style.word(lang, "exact match"))
    ax.legend(loc="lower right")
    ctx.save(fig)
    rows = [
        {"figure_id": ctx.fig_id, "panel": "a", "series": "selftest", "run_name": "", "wandb_id": "",
         "project": "", "cohort": "", "metric": "selftest_value", "step": s, "value": v,
         "source_ref": "selftest:_selftest_fig.py VALUES"}
        for s, v in zip(STEPS, VALUES)
    ]
    ctx.write_values(pd.DataFrame(rows))
    ctx.write_source({"cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"]})
