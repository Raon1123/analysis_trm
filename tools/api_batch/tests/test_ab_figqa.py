"""seat-figqa: blind readers B1/B2/B3 through the batch path (fig-qa.md; design §4.2, §5.5)."""
import base64
import dataclasses
import json
import math
import struct
import zlib

import pytest

from tools.api_batch import manifest as M
from tools.api_batch import ops, prices
from tools.api_batch.common import Halt, REPO_ROOT, est_tokens, sha256_bytes, sha256_text
from tools.api_batch.fake_client import FakeClient
from tools.api_batch.ledger import Ledger
from tools.api_batch.pack import PackBuilder, PackError
from tools.api_batch.spec import SpecError, validate_spec

W, H = 30, 20


def make_png(w=W, h=H, shade=0):
    def chunk(t, d):
        c = struct.pack(">I", len(d)) + t + d
        return c + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    raw = b"".join(b"\x00" + bytes([shade]) * (3 * w) for _ in range(h))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + \
        chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


SPEC = """name: figqa-t
use: seat-figqa
models: {b1: claude-sonnet-5-5, b2: claude-sonnet-5-5, b3: claude-haiku-5-5}
effort: medium
max_tokens: 4096
cache_ttl: none
prompt: {system_template: tools/api_batch/prompts/figqa_reader.md}
context: tools/api_batch/prompts/figqa_context_sigma-k-min.md
output_schema: figqa_reading
figures:
  - {slug: grid, exp_id: EXP-015, origin: 25_figures/grid.png, png: user/fig_under_test.png, caption: 23_reports/grid_caption.md}
budget: {cap_usd_hard_worst: 1.0}
"""
CAPTION = "Rows are cohorts; columns are depth k. Hatched cells were not run.\n"
READING = {"encodings": [{"channel": "color", "decoded": "accuracy"}], "main_claim": "c", "secondary_observations": [],
           "cannot_determine": [], "misreading_traps": [], "confidence": [{"claim": "c", "level": "low"}]}


@pytest.fixture
def env(tmp_path, deny_files, lab):
    (lab / "25_figures").mkdir()
    (lab / "25_figures/grid.png").write_bytes(make_png(shade=9))
    (lab / "23_reports").mkdir()
    (lab / "23_reports/grid_caption.md").write_text(CAPTION, encoding="utf-8")
    root = tmp_path / "root"
    bd = root / "11_batches" / "2026-11-04_figqa"
    (bd / "user").mkdir(parents=True)
    png = make_png()
    (bd / "user/fig_under_test.png").write_bytes(png)
    (bd / "spec.yaml").write_text(SPEC, encoding="utf-8")
    builder = PackBuilder.from_files(deny_files, lab_root=lab, batch_dir=bd)
    return type("E", (), dict(root=root, bd=bd, builder=builder, led=Ledger(root / "ledger.jsonl"), lab=lab, png=png))


def rows(env):
    return [json.loads(ln) for ln in (env.bd / "manifest.jsonl").read_text().splitlines()]


# ----------------------------------------------------------------------------------- C1 spec
def sp(**kw):
    import yaml
    d = yaml.safe_load(SPEC)
    for k, v in kw.items():
        if v is None:
            d.pop(k, None)
        else:
            d[k] = v
    return d


def test_spec_ok_and_invalid_variants():
    assert validate_spec(sp())["use"] == "seat-figqa"
    f0 = sp()["figures"][0]
    bad = [dict(figures=None), dict(figures=[]),
           dict(models={"b1": "claude-sonnet-5-5", "b2": "claude-sonnet-5-5", "b3": "claude-nope-1"}),
           dict(models={"b1": "claude-sonnet-5-5", "b2": "claude-sonnet-5-5"}),
           dict(cache_ttl="2h"), dict(refusal_fallback="resend"), dict(effort="turbo"),
           dict(context="lab/x.md"), dict(context="tools/api_batch/prompts/sub/x.md"),
           dict(figures=[f0, dict(f0)]), dict(figures=[{**f0, "png": "/abs/x.png"}]), dict(figures=[{**f0, "png": "fig.png"}]),
           dict(figures=[{**f0, "caption": "23_reports/cap.txt"}]), dict(figures=[{**f0, "extra": "x"}]),
           dict(figures=[{k: v for k, v in f0.items() if k != "exp_id"}]), dict(max_tokens=0)]
    for kw in bad:
        with pytest.raises(SpecError):
            validate_spec(sp(**kw))


# ----------------------------------------------------------------------------------- C2 pack
def test_build_figqa_image_fields(env):
    p = env.builder.build_figqa_image("fig_under_test", env.bd / "user/fig_under_test.png", "25_figures/grid.png", "EXP-015")
    assert p.kind == "figqa_image" and p.mode == "verify-only" and p.sealed
    assert p.text == base64.b64encode(env.png).decode() and "\n" not in p.text
    assert p.report["width"] == W and p.report["height"] == H and p.report["exp_id"] == "EXP-015"
    assert p.report["est_tokens"] == math.ceil(W * H / 750)
    assert list(p.sources) == [{"path": "25_figures/grid.png", "sha256": sha256_bytes((env.lab / "25_figures/grid.png").read_bytes())},
                               {"path": "user/fig_under_test.png", "sha256": sha256_bytes(env.png)}]
    big = env.bd / "user/big.png"
    big.write_bytes(make_png(1500, 1000))
    assert env.builder.build_figqa_image("big", big, "25_figures/grid.png", "ad-hoc").report["est_tokens"] == 2000


def test_figqa_image_l1_refusals(env):
    b, png = env.builder, env.bd / "user/fig_under_test.png"
    for bad in ("22_plans/x.png", "25_figures/grid.txt", "25_figures/missing.png", "25_figures/../23_reports/grid_caption.md"):
        with pytest.raises(PackError, match="L1"):
            b.build_figqa_image("n", png, bad, "EXP-015")
    outside = env.lab / "25_figures/grid.png"                       # a png not under <batch>/user/
    with pytest.raises(PackError, match="L1"):
        b.build_figqa_image("n", outside, "25_figures/grid.png", "EXP-015")
    notpng = env.bd / "user/notpng.png"
    notpng.write_bytes(b"GIF89a" + b"\0" * 40)
    with pytest.raises(PackError, match="not a PNG"):
        b.build_figqa_image("n", notpng, "25_figures/grid.png", "EXP-015")
    with pytest.raises(PackError, match="verify-only"):              # binary class never goes through build_verify_only
        b.build_verify_only("figqa_image", "n", "abc")


def test_figqa_image_l2_refusals(env):
    b, png = env.builder, env.bd / "user/fig_under_test.png"
    (env.lab / "25_figures/sealed").mkdir()
    (env.lab / "25_figures/sealed/x.png").write_bytes(make_png())
    (env.lab / "25_figures/fig1_tf_z_iter_k3_s2.png").write_bytes(make_png())
    with pytest.raises(PackError, match="L2: path .* deny glob"):
        b.build_figqa_image("n", png, "25_figures/sealed/x.png", "EXP-015")
    with pytest.raises(PackError, match="L2: path/name"):
        b.build_figqa_image("n", png, "25_figures/fig1_tf_z_iter_k3_s2.png", "EXP-015")
    with pytest.raises(PackError, match="L2: exp_id"):
        b.build_figqa_image("n", png, "25_figures/grid.png", "EXP-016")
    with pytest.raises(PackError, match="L2: exp_id"):
        b.build_figqa_image("n", png, "25_figures/grid.png", "EXP-013")


# ----------------------------------------------------------------------------------- C3 manifest
def img_req(env, mutate=None, with_seat_pack=False):
    b = env.builder
    img = b.build_figqa_image("f", env.bd / "user/fig_under_test.png", "25_figures/grid.png", "EXP-015")
    seat = b.build_verify_only("spec_text", "t", "task words")
    r = M.build_request("figqa-abcd1234-b1-s1", "claude-sonnet-5-5", "low", 10, None, [], [seat], image_packs=[img])
    return r, img, seat


def test_image_block_pack_backed_accepted(env):
    r, img, _ = img_req(env)
    M.validate_request(r)
    assert img in r.packs


def test_unbacked_image_block_refused(env):
    r, img, seat = img_req(env)
    blk = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(make_png(3, 3)).decode()}}
    params = {**r.params, "messages": [{"role": "user", "content": [blk, {"type": "text", "text": seat.text}]}]}
    with pytest.raises(M.ManifestError, match="image block"):
        M.validate_request(dataclasses.replace(r, params=params))


def test_wrong_media_type_refused(env):
    r, img, seat = img_req(env)
    blk = {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img.text}}
    params = {**r.params, "messages": [{"role": "user", "content": [blk, {"type": "text", "text": seat.text}]}]}
    with pytest.raises(M.ManifestError, match="image block"):
        M.validate_request(dataclasses.replace(r, params=params))


def test_image_with_other_pack_kind_data_refused(env):
    b = env.builder
    cap = b.build_verify_only("seat_input", "c", CAPTION, origin=env.lab / "23_reports/grid_caption.md")
    seat = b.build_verify_only("spec_text", "t", "task words")
    blk = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": cap.text}}
    r = M.Request("figqa-abcd1234-b1-s1", {"model": "m", "max_tokens": 5, "messages": [{"role": "user", "content": [
        blk, {"type": "text", "text": seat.text}]}]}, (cap, seat))
    with pytest.raises(M.ManifestError, match="image block"):
        M.validate_request(r)
    # and a figqa_image Pack's bytes may not ride in a text segment
    img = b.build_figqa_image("f", env.bd / "user/fig_under_test.png", "25_figures/grid.png", "EXP-015")
    r2 = M.Request("figqa-abcd1234-b1-s1", {"model": "m", "max_tokens": 5, "messages": [{"role": "user", "content": [
        {"type": "text", "text": img.text}]}]}, (img,))
    with pytest.raises(M.ManifestError):
        M.validate_request(r2)


# ----------------------------------------------------------------------------------- C4/C6 plan
def test_plan_three_requests_per_figure(env):
    d = ops.plan(env.bd, env.builder, env.led)
    rs = rows(env)
    assert d["n_requests"] == 3 and len(rs) == 3
    by = {r["meta"]["cond"]: r for r in rs}
    assert set(by) == {"b1", "b2", "b3"}
    pre = "figqa-" + sha256_bytes(env.png)[:8]
    assert {r["custom_id"] for r in rs} == {f"{pre}-b1-s1", f"{pre}-b2-s1", f"{pre}-b3-s1"}
    assert by["b3"]["params"]["model"] == "claude-haiku-5-5" and by["b2"]["params"]["model"] == "claude-sonnet-5-5"
    ctx = (REPO_ROOT / "tools/api_batch/prompts/figqa_context_sigma-k-min.md").read_text(encoding="utf-8")
    sysp = (REPO_ROOT / "tools/api_batch/prompts/figqa_reader.md").read_text(encoding="utf-8")
    imgblk = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(env.png).decode()}}
    for c, r in by.items():
        p = r["params"]
        content = p["messages"][0]["content"]
        assert content[0] == imgblk
        assert [b["type"] for b in content] == (["image", "document", "text"] if c == "b1" else ["image", "document", "document", "text"])
        assert content[1]["source"]["data"] == ctx
        if c == "b1":
            assert content[-1]["text"] == "Condition B1: no caption is provided."
        else:
            assert content[2]["source"]["data"] == CAPTION and content[-1]["text"] == "Condition B2."
        assert p["system"] == [{"type": "text", "text": sysp}]
        assert p["output_config"]["format"]["type"] == "json_schema"
        assert "cache_control" not in json.dumps(p)
        assert r["meta"]["png_sha256"] == sha256_bytes(env.png) and r["meta"]["prompt_sha256"] == sha256_text(sysp)
        assert r["meta"]["image_tokens"] == math.ceil(W * H / 750) and r["meta"]["slug"] == "grid" and r["meta"]["exp_id"] == "EXP-015"
        assert ("caption_sha256" in r["meta"]) == (c != "b1")
        # hard-worst includes the image token term
        docs_t = sum(est_tokens(b["source"]["data"]) for b in content if b["type"] == "document")
        other_t = est_tokens(content[-1]["text"]) + est_tokens(sysp)
        want = prices.hard_worst_request(p["model"], docs_t, other_t + r["meta"]["image_tokens"], p["max_tokens"], "none")
        got = {x["custom_id"]: x["hard_worst_usd"] for x in d["requests"]}[r["custom_id"]]
        assert got == round(want, 6)
        assert want > prices.hard_worst_request(p["model"], docs_t, other_t, p["max_tokens"], "none")
    f = d["figqa"]
    assert f["n_figures"] == 1 and f["figures"][0] == {"slug": "grid", "exp_id": "EXP-015", "png_sha256": sha256_bytes(env.png),
                                                       "width": W, "height": H, "image_tokens": 1,
                                                       "custom_ids": f["figures"][0]["custom_ids"]}
    assert sorted(f["figures"][0]["custom_ids"]) == sorted(r["custom_id"] for r in rs)
    # pack/ holds the image pack as base64 text, listed with kind figqa_image
    pm = json.loads((env.bd / "pack/pack_manifest.json").read_text())
    ent = [e for e in pm if e["kind"] == "figqa_image"]
    assert len(ent) == 1 and ent[0]["report"]["width"] == W
    assert (env.bd / "pack" / f"figqa_image-{ent[0]['pack_sha256'][:8]}.txt").read_text() == base64.b64encode(env.png).decode()


def test_plan_image_tokens_scale_cost(env):
    big = make_png(1500, 1000)
    d1 = ops.plan(env.bd, env.builder, env.led)
    (env.bd / "user/fig_under_test.png").write_bytes(big)
    d2 = ops.plan(env.bd, env.builder, env.led)
    assert d2["figqa"]["figures"][0]["image_tokens"] == 2000 and d2["est_usd_hard_worst"] > d1["est_usd_hard_worst"]
    assert d2["figqa"]["figures"][0]["custom_ids"] != d1["figqa"]["figures"][0]["custom_ids"]     # re-rendered PNG -> new ids


def test_plan_refuses_denied_exp_id(env):
    (env.bd / "spec.yaml").write_text(SPEC.replace("exp_id: EXP-015", "exp_id: EXP-016"), encoding="utf-8")
    with pytest.raises(Halt):
        ops.plan(env.bd, env.builder, env.led)
    assert not (env.bd / "manifest.jsonl").exists()


# ----------------------------------------------------------------------------------- C5 end to end
def test_end_to_end_with_refusal_b3(env):
    d = ops.plan(env.bd, env.builder, env.led)
    ops.approve(env.bd, 1.0, env.led, env={})                       # env={} bypasses the Claude Code session guard
    fake = FakeClient(seed=3)
    bid = ops.submit(env.bd, env.led, fake)["batch_id"]
    sent = fake.batches[bid]["requests"]
    assert all(r["params"]["messages"][0]["content"][0]["type"] == "image" for r in sent)
    ids = {r["meta"]["cond"]: r["custom_id"] for r in rows(env)}
    fake.finish(bid, {ids["b3"]: "refusal"}, texts={ids["b1"]: json.dumps(READING), ids["b2"]: json.dumps(READING)})
    out = ops.collect(env.bd, env.led, fake, env.root, env.builder)
    assert out["counts"] == {"succeeded": 2, "refusal": 1} and out["quarantined"] == 0
    assert not (env.bd / "quarantine").exists()
    it = {c: (env.bd / "items" / f"{cid}.md").read_text() for c, cid in ids.items()}
    assert "figqa_condition: b3" in it["b3"] and "figqa_reader: REFUSED" in it["b3"] and "result_status: refusal" in it["b3"]
    for c in ("b1", "b2"):
        assert f"figqa_condition: {c}" in it[c] and "figqa_reader" not in it[c]
        assert "status: DRAFT (api-batch output — not evidence)" in it[c] and "l5: pass" in it[c]
    inbox = [json.loads(ln) for ln in (env.root / "INBOX.jsonl").read_text().splitlines()]
    assert len(inbox) == 3 and {r["use"] for r in inbox} == {"seat-figqa"}


def test_tampered_image_changes_plan_hash_refusal(env):
    ops.plan(env.bd, env.builder, env.led)
    ops.approve(env.bd, 1.0, env.led, env={})
    mp = env.bd / "manifest.jsonl"
    mp.write_text(mp.read_text().replace("image/png", "image/jpeg", 1))
    with pytest.raises(Halt):
        ops.submit(env.bd, env.led, FakeClient())


def test_plan_cache_ttl_5m_uses_ctx_first_v2_layout(env):
    (env.bd / "spec.yaml").write_text(SPEC.replace("cache_ttl: none", "cache_ttl: 5m"), encoding="utf-8")
    d = ops.plan(env.bd, env.builder, env.led)
    assert d["cost_formula"] == 2
    for r in rows(env):
        c = r["params"]["messages"][0]["content"]
        want = ["document", "image", "text"] if r["meta"]["cond"] == "b1" else ["document", "image", "document", "text"]
        assert [b["type"] for b in c] == want
        assert c[0]["title"] == "problem_context" and c[0]["cache_control"] == {"type": "ephemeral"}
        assert c[1]["cache_control"] == {"type": "ephemeral"}
        assert not any("cache_control" in b for b in c[2:])
        assert r["meta"]["layout"] == "ctx-first-v2" and r["meta"]["cache_after"] == ["problem_context", "fig_under_test"]


def test_formula1_plan_json_still_rederives_and_formula_is_enforced(env):
    """plan.json written before cost formula 2 has no `cost_formula`; _load_plan must re-derive it with formula 1.
    A plan claiming formula 2 with a formula-1 figure is refused."""
    (env.bd / "spec.yaml").write_text(SPEC.replace("cache_ttl: none", "cache_ttl: 5m"), encoding="utf-8")
    ops.plan(env.bd, env.builder, env.led)
    pj = env.bd / "plan.json"
    d = json.loads(pj.read_text(encoding="utf-8"))
    rs = rows(env)
    w1, w2 = ops._rows_hard_worst(rs, 1), ops._rows_hard_worst(rs, 2)
    assert w2 > w1 and d["est_usd_hard_worst"] == round(w2, 6)       # system+image now at the write price
    d.update(est_usd_hard_worst=round(w1, 6))
    pj.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(ops.Refused, match="edited after plan"):
        ops._load_plan(env.bd)
    d.pop("cost_formula")
    pj.write_text(json.dumps(d), encoding="utf-8")
    assert ops._load_plan(env.bd)["est_usd_hard_worst"] == round(w1, 6)
