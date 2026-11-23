"""Tests for the r1 reader timeline (RF-11) and the r1 curve wrappers. Synthetic inputs, no NAS."""
import csv
import datetime as dt

import pytest

from tools.figures.figs import curves_r1, timeline_reader as tr

READER = [
    {"no": "1", "phase": "1", "when_run": "6월 7~22일", "run_first_iso": "2026-06-07", "timeline_id": "EXP-001"},
    {"no": "2", "phase": "1", "when_run": "6월 22일~7월 13일", "run_first_iso": "2026-06-22", "timeline_id": "EXP-005"},
    {"no": "3", "phase": "2", "when_run": "7월 2일; 재훈련 11월 3일", "run_first_iso": "2026-07-02", "timeline_id": "EXP-006"},
]
PHASES = [{"phase": "1", "title_ko": "가"}, {"phase": "2", "title_ko": "나"}]
LABELS = [{"no": str(i), "label_ko": f"라벨{i}"} for i in (1, 2, 3)]


def test_first_period_forms():
    assert tr.first_period("6월 7~22일", "2026-06-07") == (dt.date(2026, 6, 7), dt.date(2026, 6, 22))
    assert tr.first_period("10월 27일~11월 2일", "2026-10-27")[1] == dt.date(2026, 11, 2)
    assert tr.first_period("7월 13일 시작(깊이 9); 11월 5~6일", "2026-07-13")[1] == dt.date(2026, 7, 13)
    assert tr.first_period("9월 3일 무렵", "2026-09-03")[0] == dt.date(2026, 9, 3)


def test_first_period_start_mismatch_raises():
    with pytest.raises(ValueError, match="run_first_iso"):
        tr.first_period("6월 7~22일", "2026-06-08")


def test_slots_order_and_headers():
    s = tr.load_slots(READER, PHASES, LABELS)
    assert [x["kind"] for x in s] == ["phase", "exp", "exp", "phase", "exp"]
    assert [x["no"] for x in s if x["kind"] == "exp"] == [1, 2, 3]
    assert s[0]["label"] == "단계 1 · 가"


def test_slots_reject_gaps_and_unordered_phases():
    with pytest.raises(ValueError):
        tr.load_slots([READER[0], READER[2]], PHASES, LABELS[:1] + LABELS[2:])
    bad = [dict(READER[0], phase="2"), dict(READER[1], phase="1"), READER[2]]
    with pytest.raises(ValueError, match="run order"):
        tr.load_slots(bad, PHASES, LABELS)


def test_slots_continuation_rows():
    """r3: a later re-run of experiment N is a row 'Nb' after N; it is kept as a string and flagged."""
    rows = READER + [{"no": "1b", "phase": "2", "when_run": "11월 3~5일", "run_first_iso": "2026-11-03",
                      "timeline_id": "EXP-001"}]
    s = tr.load_slots(rows, PHASES, LABELS + [{"no": "1b", "label_ko": "재훈련"}])
    exp = [x for x in s if x["kind"] == "exp"]
    assert [x["no"] for x in exp] == [1, 2, 3, "1b"]
    assert [x["rerun"] for x in exp] == [False, False, False, True]
    bad = [dict(rows[-1], no="4b")] + rows[:-1]
    with pytest.raises(ValueError):
        tr.load_slots(bad, PHASES, LABELS + [{"no": "4b", "label_ko": "x"}])


def test_curve_ko_vocabulary():
    for d in (curves_r1.KO_BUDGET, curves_r1.KO_D1):
        assert "파선" in d["leg_train"] and "점선" in d["leg_thr"] and "실선" in d["leg_test"]
        assert "점선" not in d["leg_train"]
        assert d["seed"] == "무작위 초기값"
