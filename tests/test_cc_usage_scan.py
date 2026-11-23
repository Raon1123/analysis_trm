"""cc_usage_scan: request dedupe, break classification, and the 5m-TTL counterfactual rules."""
import json

from tools.harness_metrics import cc_usage_scan as S


def _row(rid, t, model="claude-sonnet-5-5", effort="medium", inp=0, cw=0, cr=0, out=0, cw1h=None):
    return json.dumps({"type": "assistant", "requestId": rid, "timestamp": t, "effort": effort,
                       "message": {"model": model, "usage": {
                           "input_tokens": inp, "cache_creation_input_tokens": cw, "cache_read_input_tokens": cr,
                           "output_tokens": out,
                           "cache_creation": {"ephemeral_1h_input_tokens": cw if cw1h is None else cw1h}}}})


def _write(path, rows):
    path.write_text("\n".join(rows) + "\n")
    return str(path)


def test_dedupe_keeps_last_row_per_request(tmp_path):
    p = _write(tmp_path / "a.jsonl", [
        _row("r1", "2026-11-10T00:00:00Z", cw=100, out=1),
        _row("r1", "2026-11-10T00:00:01Z", cw=100, out=7),          # same request, final usage
        json.dumps({"type": "user", "message": {"usage": {}}}),      # not an assistant row
        "not json",
        _row("r2", "2026-11-10T00:01:00Z", cr=100, cw=10, out=2)])
    rs = S.read_requests(p, 0.0)
    assert [r["out"] for r in rs] == [7, 2]
    assert S.prompt(rs[1]) == 110


def test_break_classification():
    base = {"t": 0.0, "model": "m", "effort": "medium", "inp": 0, "cw": 0, "cr": 100_000, "out": 0, "cw1h": 0}
    nxt = dict(base, t=60.0, cr=0, cw=100_000)
    assert S.classify_break(base, dict(nxt, model="n")) == "model-switch"
    assert S.classify_break(base, dict(nxt, cw=10_000)) == "compact/shrink"
    assert S.classify_break(base, dict(nxt, t=4000.0)) == "idle>1h"
    assert S.classify_break(base, dict(nxt, effort="high")) == "effort-change"
    assert S.classify_break(base, nxt) == "other"
    assert S.classify_break(base, dict(nxt, cr=90_000, cw=10_000)) is None      # still a hit
    assert S.classify_break(dict(base, cr=5_000), nxt) is None                  # previous prompt too small


def test_5m_counterfactual_gap_and_first_request_rules():
    m = "claude-sonnet-5-5"                     # $2 in, read $0.10, 1h write $4, 5m write $2.50 per Mtok
    def r(t, cw=0, cr=0):
        return {"t": t, "model": m, "effort": "medium", "inp": 0, "cw": cw, "cr": cr, "out": 0, "cw1h": cw}
    files = {
        "a": ("worker", [r(0.0, cw=1_000_000),                 # cold first request: a write under both TTLs
                         r(100.0, cr=1_000_000),               # 100 s later: read under both
                         r(1000.0, cr=1_000_000)]),            # 900 s gap: read at 1h, re-written at 5m
        "b": ("worker", [r(1100.0, cr=1_000_000)]),            # first request 100 s after a's last: stays warm
        "c": ("worker", [r(3000.0, cr=1_000_000)]),            # first request 1900 s after anyone: lost at 5m
    }
    g = S.summarize(files)["worker"]
    assert g["spawns"] == 3 and g["reqs"] == 5 and g["gaps_5m"] == 1
    assert abs(g["cost"] - (4.0 + 4 * 0.10)) < 1e-9
    assert abs(g["cost_by_model"]["sonnet-5-5@medium"] - g["cost"]) < 1e-9
    assert abs(g["cost_5m"] - (2.5 + 0.10 + 2.5 + 0.10 + 2.5)) < 1e-9


def test_gap_over_one_hour_is_not_charged_to_5m():
    m = "claude-opus-5-5"
    a = {"t": 0.0, "model": m, "effort": "medium", "inp": 0, "cw": 1000, "cr": 0, "out": 0, "cw1h": 1000}
    b = dict(a, t=7200.0, cw=1000, cr=0)       # after >1h the transcript already shows a write
    g = S.summarize({"x": ("main", [a, b])})["main"]
    assert g["gaps_5m"] == 0
    assert g["cost_5m"] < g["cost"]            # only the write multiplier differs


def test_main_exit_codes(tmp_path, capsys):
    assert S.main(["--root", str(tmp_path / "missing")]) == 2
    assert S.main(["--root", str(tmp_path)]) == 1                    # empty window is a failure, not a zero


def test_advisor_iterations_are_counted_separately(tmp_path):
    row = json.loads(_row("r1", "2026-11-10T00:00:00Z", cw=1_000_000))
    row["message"]["usage"]["iterations"] = [
        {"type": "message", "input_tokens": 5, "output_tokens": 5},
        {"type": "advisor_message", "model": "claude-fable-5-2", "input_tokens": 1_000_000, "output_tokens": 100_000}]
    p = _write(tmp_path / "a.jsonl", [json.dumps(row)])
    rs = S.read_requests(p, 0.0)
    assert rs[0]["adv"] == [("claude-fable-5-2", 1_000_000, 100_000)]
    assert abs(S.cost_advisor(rs[0]) - (8.0 + 4.0)) < 1e-9
    g = S.summarize({"x": ("main", rs)})["main"]
    assert g["advisor_calls"] == 1 and abs(g["advisor_cost"] - 12.0) < 1e-9
    assert abs(g["cost"] - 4.0) < 1e-9          # sonnet 1h write of 1M tokens; the advisor is not added in

