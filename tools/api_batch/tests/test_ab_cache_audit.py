"""cache_audit: offline simulation of the prompt cache over manifest rows (no Packs, no network)."""
import base64
import json
import struct
import zlib

from tools.api_batch import cache_audit as CA
from tools.api_batch import cli

HDR = "H" * 4000          # ~1000 tok low estimate: above the 512 minimum
SMALL = "s" * 400         # ~100 tok: below the minimum
SONNET = "claude-sonnet-5-5"


def doc(text, title="d", cc=None):
    b = {"type": "document", "title": title, "source": {"type": "text", "media_type": "text/plain", "data": text}}
    if cc:
        b["cache_control"] = {"type": "ephemeral"}
    return b


def row(cid, content, model=SONNET, system="SYS " * 50, effort="high", fmt=None):
    oc = {"effort": effort}
    if fmt:
        oc["format"] = {"type": "json_schema", "schema": fmt}
    p = {"model": model, "max_tokens": 100, "system": [{"type": "text", "text": system}],
         "messages": [{"role": "user", "content": content}], "output_config": oc}
    return {"custom_id": cid, "params": p}


def write_manifest(d, rows):
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return d / "manifest.jsonl"


def codes(res):
    return [f["code"] for f in res["findings"]]


def test_map_shape_breakpoint_on_shared_header(tmp_path):
    rows = [row(f"m-{i}", [doc(HDR, "header", cc=True), {"type": "text", "text": f"cell {i} " * 300}]) for i in range(5)]
    res = CA.audit_manifest(write_manifest(tmp_path / "b", rows))
    assert res["ideal"]["read"] > 0 and res["ideal"]["coverage"] == res["potential"]["coverage"]
    assert "F3" not in codes(res)
    # 1 write then 4 reads of the same prefix
    assert res["ideal"]["read"] == 4 * (res["ideal"]["write"])


def test_reduce_shape_without_breakpoint_on_shared_header_is_missed_cache(tmp_path):
    # the lane's spec turned caching off because the last document varies: no breakpoint anywhere
    rows = [row(f"r-{i}", [doc(HDR, "header"), doc(f"memo {i} " * 400, "memo"),
                           {"type": "text", "text": f"task {i}"}]) for i in range(4)]
    res = CA.audit_manifest(write_manifest(tmp_path / "b", rows))
    assert res["ideal"]["coverage"] == 0.0 and res["potential"]["coverage"] > 0.1
    f3 = [f for f in res["findings"] if f["code"] == "F3" and "missed_prefix_tokens" in f]
    assert f3 and f3[0]["shared_until"] == [(1, "document")]
    assert res["input_usd"]["gap"] > 0


def test_breakpoint_below_minimum_flags_f2_and_never_reads(tmp_path):
    rows = [row(f"x-{i}", [doc(SMALL, cc=True), {"type": "text", "text": f"q{i}"}], system="sys") for i in range(3)]
    res = CA.audit_manifest(write_manifest(tmp_path / "b", rows))
    assert "F2" in codes(res) and res["ideal"]["read"] == 0


def test_models_and_settings_do_not_share_and_f5(tmp_path):
    rows = [row("a-1", [doc(HDR, cc=True), {"type": "text", "text": "q1"}]),
            row("a-2", [doc(HDR, cc=True), {"type": "text", "text": "q2"}], model="claude-opus-5-5"),
            row("a-3", [doc(HDR, cc=True), {"type": "text", "text": "q3"}], effort="low")]
    res = CA.audit_manifest(write_manifest(tmp_path / "b", rows))
    assert res["n_groups"] == 3 and res["ideal"]["read"] == 0
    assert "F5" in codes(res)


def _png(w=64, h=64):
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
    raw = b"".join(b"\x00" + b"\x00" * (w * 3) for _ in range(h))
    data = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + \
        chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    return base64.b64encode(data).decode()


def test_f4_unique_image_ahead_of_shared_context(tmp_path):
    imgs = [_png(64, 64 + i) for i in range(3)]
    rows = [row(f"f-{i}", [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": imgs[i]}},
                           doc(HDR, "problem_context"), {"type": "text", "text": "task"}]) for i in range(3)]
    res = CA.audit_manifest(write_manifest(tmp_path / "b", rows))
    assert codes(res).count("F4") == 3
    # the same image shared by two requests (b1/b2 readers on one model) is not an F4
    same = [row(f"g-{i}", [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": imgs[0]}},
                           doc(HDR, "problem_context"), {"type": "text", "text": f"task {i}"}]) for i in range(2)]
    assert "F4" not in codes(CA.audit_manifest(write_manifest(tmp_path / "c", same)))


def test_measured_coverage_counts_uncached_input(tmp_path):
    rows = [row(f"m-{i}", [doc(HDR, "header", cc=True), {"type": "text", "text": f"cell {i}"}]) for i in range(2)]
    d = tmp_path / "b"
    write_manifest(d, rows)
    (d / "raw").mkdir()
    us = [{"input_tokens": 100, "output_tokens": 5, "cache_creation_input_tokens": 1000, "cache_read_input_tokens": 0},
          {"input_tokens": 100, "output_tokens": 5, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 1000}]
    (d / "raw" / "results.jsonl").write_text("".join(json.dumps(
        {"custom_id": f"m-{i}", "result": {"type": "succeeded", "message": {"usage": u}}}) + "\n" for i, u in enumerate(us)))
    m = CA.audit_manifest(d / "manifest.jsonl")["measured"]
    assert m["coverage"] == round(1000 / 2200, 4) and m["r_hit_ledger_def"] == 0.5 and m["n"] == 2


def test_draft_sync_raw_layout(tmp_path):
    rows = [row("draft-sec1", [doc(HDR), {"type": "text", "text": "p"}])]
    d = tmp_path / "b"
    write_manifest(d, rows)
    (d / "raw").mkdir()
    (d / "raw" / "sec1.json").write_text(json.dumps({"usage": {"input_tokens": 900, "output_tokens": 1}}))
    res = CA.audit_manifest(d / "manifest.jsonl")
    assert res["measured"]["input"] == 900 and res["measured"]["coverage"] == 0.0
    assert "draft-*" in res["input_usd"]["basis"]


def test_cli_skips_dir_without_manifest_and_errors_on_missing_file(tmp_path, capsys):
    write_manifest(tmp_path / "ok", [row("m-1", [doc(HDR, cc=True)])])
    (tmp_path / "nomanifest").mkdir()
    assert cli.main(["cache-audit", str(tmp_path / "ok"), str(tmp_path / "nomanifest")]) == 0
    out = capsys.readouterr()
    assert "| ok |" in out.out and "skip" in out.err
    assert CA.main([str(tmp_path / "missing.jsonl")]) == 2


def _sim(rows, every=False):
    reqs = [CA.render(r) for r in rows]
    return CA.simulate(reqs, every_boundary=every), reqs


def test_read_write_input_partition_total_in_both_modes():
    rows = [row(f"p-{i}", [doc(HDR, "h", cc=True), doc(f"mid {i % 2} " * 300, "m", cc=True),
                           {"type": "text", "text": f"tail {i} " * 50}]) for i in range(5)]
    for every in (False, True):
        t, reqs = _sim(rows, every)
        assert t["read"] + t["write"] + t["input"] == t["total"] == sum(r.total for r in reqs)


def test_two_breakpoints_hand_computed():
    # A=[sys,H*,M0*,t0] B=[sys,H*,M0*,t1] C=[sys,H*,M1*,t2]: A writes through M0, B reads through M0,
    # C reads through H and writes M1 only
    mids = ["X" * 3000, "X" * 3000, "Y" * 3000]
    rows = [row(f"t-{i}", [doc(HDR, "h", cc=True), doc(mids[i], "m", cc=True), {"type": "text", "text": f"q{i}"}],
                system="") for i in range(3)]
    t, reqs = _sim(rows)
    h, m = reqs[0].cum[1], reqs[0].cum[2]          # cum through header / through mid (system "" is block 0)
    assert t["read"] == m + h
    assert t["write"] == m + (reqs[2].cum[2] - h)


def test_minimum_uses_byte_under_estimate_boundary():
    # bytes/4: 2000 B -> 500 tok < 512 (never cached); 2100 B -> 525 tok >= 512 (cached)
    for every in (False, True):
        for n, cached in ((2000, False), (2100, True)):
            rows = [row(f"b-{i}", [doc("Z" * n, "", cc=True), {"type": "text", "text": f"q{i}"}], system="") for i in range(2)]
            t, reqs = _sim(rows, every)
            assert (t["read"] > 0) is cached, (n, every)


def test_potential_writes_only_shared_levels_and_counts_breakpoints():
    rows = [row(f"s-{i}", [doc(HDR, "h"), {"type": "text", "text": f"unique tail {i} " * 300}], system="") for i in range(3)]
    t, reqs = _sim(rows, every=True)
    h = reqs[0].cum[1]
    assert t["write"] == h and t["read"] == 2 * h and t["max_breakpoints_needed"] == 1
    assert t["input"] == sum(r.total - h for r in reqs)


def test_draft_sync_rows_priced_at_twice_batch(tmp_path):
    base = [doc(HDR, "h", cc=True), {"type": "text", "text": "q"}]
    a = CA.audit_manifest(write_manifest(tmp_path / "a", [row("x-1", base)]))
    b = CA.audit_manifest(write_manifest(tmp_path / "b", [row("draft-x1", base)]))
    assert b["input_usd"]["ideal"] == round(2 * a["input_usd"]["ideal"], 4) > 0


def _usage_dir(tmp_path, rows, usages):
    d = tmp_path / "u"
    write_manifest(d, rows)
    (d / "raw").mkdir()
    (d / "raw" / "results.jsonl").write_text("".join(json.dumps(
        {"custom_id": r["custom_id"], "result": {"type": "succeeded", "message": {"usage": u}}}) + "\n"
        for r, u in zip(rows, usages)))
    return d / "manifest.jsonl"


def test_f6_fires_on_duplicate_writes_and_calibration_scales_dollars(tmp_path):
    rows = [row(f"c-{i}", [doc(HDR, "h", cc=True), {"type": "text", "text": f"q{i}"}], system="") for i in range(4)]
    est = CA.render(rows[0]).cum[1]          # through the header (block 0 is the empty system block)
    # every request wrote the header (4 concurrent starts) instead of 1 write + 3 reads
    us = [{"input_tokens": 1, "cache_creation_input_tokens": est, "cache_read_input_tokens": 0}] * 4
    res = CA.audit_manifest(_usage_dir(tmp_path, rows, us))
    assert res["measured"]["write_over_ideal"] > 2 and "F6" in codes(res)
    cal = res["measured"]["calibration"]
    plain = CA.audit_manifest(write_manifest(tmp_path / "plain", rows))
    assert abs(res["input_usd"]["ideal"] - cal * plain["input_usd"]["ideal"]) < 1e-3


def test_no_f6_when_writes_match_ideal(tmp_path):
    rows = [row(f"c-{i}", [doc(HDR, "h", cc=True), {"type": "text", "text": f"q{i}"}], system="") for i in range(4)]
    est = CA.render(rows[0]).cum[1]          # through the header (block 0 is the empty system block)
    us = [{"input_tokens": 1, "cache_creation_input_tokens": est, "cache_read_input_tokens": 0}] + \
         [{"input_tokens": 1, "cache_creation_input_tokens": 0, "cache_read_input_tokens": est}] * 3
    assert "F6" not in codes(CA.audit_manifest(_usage_dir(tmp_path, rows, us)))


def test_unknown_model_without_usage_reports_no_dollars(tmp_path):
    rows = [row(f"k-{i}", [doc(HDR, "h", cc=True), {"type": "text", "text": "q"}], model="claude-future-9") for i in range(2)]
    res = CA.audit_manifest(write_manifest(tmp_path / "k", rows))
    assert "error" in res["input_usd"] and res["measured"] is None
    assert res["ideal"]["read"] == 0     # ~1000-token header is below the unknown-model floor (4096): fails toward "won't cache"


def test_dollars_scale_with_calibration(tmp_path):
    rows = [row(f"z-{i}", [doc(HDR, "h", cc=True), {"type": "text", "text": f"q{i}"}], system="") for i in range(2)]
    tot = CA.render(rows[0]).total
    us = [{"input_tokens": tot // 4, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}] * 2
    res = CA.audit_manifest(_usage_dir(tmp_path, rows, us))
    cal = res["measured"]["calibration"]
    assert cal < 0.3
    plain = CA.audit_manifest(write_manifest(tmp_path / "plain", rows))
    assert res["input_usd"]["ideal"] == round(cal * plain["input_usd"]["ideal"], 4) < plain["input_usd"]["ideal"]


def test_one_sharing_level_spanning_two_boundaries_needs_one_breakpoint():
    rows = [row(f"l-{i}", [doc(HDR, "h1"), doc(HDR + "2", "h2"), {"type": "text", "text": f"q{i}"}], system="") for i in range(3)]
    t, _ = _sim(rows, every=True)
    assert t["max_breakpoints_needed"] == 1


def test_f7_when_potential_needs_more_than_four_breakpoints(tmp_path):
    # nested sharing levels: request i shares blocks 0..i with request i+1 -> 5 distinct levels in one request
    levels = [doc(f"{c}" * 3000, c) for c in "ABCDEF"]
    rows = [row(f"n-{k}", levels[:k] + [{"type": "text", "text": f"tail {k}"}], system="") for k in range(1, 7)]
    rows += [row(f"n-{k}b", levels[:k] + [{"type": "text", "text": f"other {k}"}], system="") for k in range(1, 7)]
    res = CA.audit_manifest(write_manifest(tmp_path / "n", rows))
    assert res["potential"]["max_breakpoints_needed"] > 4 and "F7" in codes(res)


def test_f8_when_request_carries_more_than_four_breakpoints(tmp_path):
    rows = [row("e-1", [doc(f"{c}" * 3000, c, cc=True) for c in "ABCDE"])]
    assert "F8" in codes(CA.audit_manifest(write_manifest(tmp_path / "e", rows)))
