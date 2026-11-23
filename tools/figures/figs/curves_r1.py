"""RF-13 / RF-14: the pilot curve figures (RF-03 / RF-04) re-drawn with the r1 Korean vocabulary.

The pilot modules are reused unchanged; this wrapper swaps, for the duration of one build, the Korean
words, the figure width (so that 14 pt text prints at >= 7 pt at 6.3 in) and the panel-title k wording.
Same data, same values CSV as the pilot.
"""
import contextlib

from . import curves_budget as cb
from . import curves_d1ext as cd

FIG_W = 12.4
PANEL_W = 3.6
SEED = "무작위 초기값"
KO_BUDGET = {
    "run2": "2배 훈련", "run1": "1배 훈련", "run2_long": "2배 훈련(예산 두 배)",
    "single": "1배 훈련", "seed": SEED,
    "line_test": "시험", "line_train": "훈련",
    "leg_test": "시험 정확도(실선)", "leg_train": "훈련 정확도(파선)",
    "leg_budget": "1배 훈련의 예산 244,100 step(세로선)", "leg_thr": "0.9 선(점선)",
    "leg_1l": "1배 훈련, 옛 데이터", "leg_1c": "1배 훈련, 고친 데이터", "leg_2": "2배 훈련(예산 두 배)",
}
KO_D1 = {
    "seed": SEED, "leg_test": "시험 정확도(실선)", "leg_train": "훈련 정확도(파선)",
    "leg_budget": "2세대 데이터의 예산 122,050 step(세로선)", "leg_thr": "0.9 선(점선)",
}


@contextlib.contextmanager
def patched(mod, **attrs):
    old = {k: getattr(mod, k) for k in attrs}
    try:
        for k, v in attrs.items():
            setattr(mod, k, v)
        yield
    finally:
        for k, v in old.items():
            setattr(mod, k, v)


class RetitleCtx:
    """Proxy: before saving, panel titles 'k = 5' become '깊이 k = 5'."""

    def __init__(self, ctx):
        self._ctx = ctx

    def __getattr__(self, name):
        return getattr(self._ctx, name)

    def save(self, fig):
        for ax in fig.axes:
            t = ax.get_title()
            if "\nk = " in t:
                ax.title.set_text(t.replace("\nk = ", "\n깊이 k = "))
        return self._ctx.save(fig)


def _check(ctx):
    if ctx.lang != "ko":
        raise ValueError("RF-13 / RF-14 are Korean only")


def build_budget(ctx):
    _check(ctx)
    local = {"en": cb.LOCAL["en"], "ko": KO_BUDGET}
    with patched(cb, LOCAL=local, FIG_W=FIG_W, PANEL_W=PANEL_W):
        cb.build(RetitleCtx(ctx))


def build_d1ext(ctx):
    _check(ctx)
    local = {"en": cd.LOCAL["en"], "ko": KO_D1}
    with patched(cd, LOCAL=local, FIG_W=FIG_W):
        cd.build(RetitleCtx(ctx))
