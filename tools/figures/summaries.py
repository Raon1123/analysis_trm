"""Descriptive per-run summaries and the protocol guard shared by the curve figures.

These are descriptive statistics of one evaluation series. They are not the verdict of any experiment.
"""

THETA = 0.9
_EPS = 1e-12

PROBE_TEST = "probe/test_exact"
PROBE_TRAIN = "probe/train_exact"
# Every measure of test accuracy that exists in the pipeline (never two of them in one panel).
TEST_MEASURES = (PROBE_TEST, "all/exact_accuracy", "checkpoint_recompute")


def summarize_run(steps, values, theta=THETA, budget=None):
    """Summary of one evaluation series (steps ascending after sorting, one value per step).

    first_reach_step: first evaluation at or above theta, or None.
    sustained_start: first step of the final unbroken stretch at or above theta that lasts to
        the last evaluation, or None (also None when the last evaluation is below theta).
    final_value, max_value, n_evals, n_at_or_above.
    value_at_budget: value of the last evaluation at or before budget (None without budget or
        when no evaluation is that early).
    reached_by_budget: an evaluation at or above theta exists at a step <= budget
        (any step when budget is None).
    """
    pairs = sorted(zip((int(s) for s in steps), (float(v) for v in values)), key=lambda p: p[0])
    n = len(pairs)
    out = {
        "first_reach_step": None, "sustained_start": None, "final_value": None, "max_value": None,
        "n_evals": n, "n_at_or_above": 0, "value_at_budget": None, "reached_by_budget": False,
    }
    if n == 0:
        return out
    hit = [s for s, v in pairs if v >= theta - _EPS]
    out["n_at_or_above"] = len(hit)
    if hit:
        out["first_reach_step"] = hit[0]
    out["final_value"] = pairs[-1][1]
    out["max_value"] = max(v for _, v in pairs)
    if pairs[-1][1] >= theta - _EPS:
        i = n - 1
        while i > 0 and pairs[i - 1][1] >= theta - _EPS:
            i -= 1
        out["sustained_start"] = pairs[i][0]
    if budget is not None:
        early = [(s, v) for s, v in pairs if s <= budget]
        if early:
            out["value_at_budget"] = early[-1][1]
        out["reached_by_budget"] = bool(hit) and hit[0] <= budget
    else:
        out["reached_by_budget"] = bool(hit)
    return out


def check_panel(metrics, cohorts, tagged):
    """Protocol guard for one panel of a curve figure.

    metrics: the metric names drawn in the panel; cohorts: the data generation of every drawn run;
    tagged: True when every run carries its own data-generation tag in the panel.
    Raises ValueError for two different test measures, for any metric other than the two probe
    metrics, and for runs of different data generations without per-run tags.
    """
    ms = set(metrics)
    tests = ms & set(TEST_MEASURES)
    if len(tests) > 1:
        raise ValueError(f"two different test measures in one panel: {sorted(tests)}")
    other = ms - {PROBE_TEST, PROBE_TRAIN}
    if other:
        raise ValueError(f"only {PROBE_TEST} and {PROBE_TRAIN} may be plotted, got {sorted(other)}")
    if len(set(cohorts)) > 1 and not tagged:
        raise ValueError(f"panel mixes data generations {sorted(set(cohorts))} without per-run tags")
    return True


def cells_row_stats(steps, values, budget):
    """summarize_run with theta 0.9 and the figure's budget (the cells CSV fields)."""
    return summarize_run(steps, values, theta=THETA, budget=budget)


def _em(ch):
    """Width of one character in em (full-width CJK glyphs 1.0, everything else about 0.55)."""
    o = ord(ch)
    if 0x1100 <= o <= 0x11FF or 0x2E80 <= o <= 0xD7FF or 0xF900 <= o <= 0xFAFF or 0xFF00 <= o <= 0xFFEF:
        return 1.0
    return 0.55


def text_em(text):
    return sum(_em(c) for c in text)


def wrap_em(text, width_em):
    """Greedy word wrap of text into lines no wider than width_em (in em); returns a list of lines."""
    lines, cur = [], ""
    for w in text.split(" "):
        cand = w if not cur else cur + " " + w
        if cur and text_em(cand) > width_em:
            lines.append(cur)
            cur = w
        else:
            cur = cand
    if cur:
        lines.append(cur)
    return lines
