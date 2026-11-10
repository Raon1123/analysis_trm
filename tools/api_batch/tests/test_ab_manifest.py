import dataclasses

import pytest

from tools.api_batch import manifest as M
from tools.api_batch import prices
from tools.api_batch.common import canon_json, sha256_text
from tools.api_batch.pack import Pack


def pk(builder, kind_text="hello", name="a"):
    return builder.build_verify_only("seat_input", name, kind_text, None)


def test_request_without_pack_refused():
    with pytest.raises(M.ManifestError):
        M.validate_request(M.Request("x-1", {"model": "m", "max_tokens": 5, "messages": [{"role": "user", "content": "hi"}]}, ()))


def test_unsealed_pack_refused(builder):
    fake = Pack("generation", "n", "hello", "redact")                      # no seal: not built by PackBuilder
    r = M.Request("x-1", {"model": "m", "max_tokens": 5, "messages": [{"role": "user", "content": "hello"}]}, (fake,))
    with pytest.raises(M.ManifestError):
        M.validate_request(r)


def test_built_request_validates_and_text_swap_refused(builder):
    sysp, docp, var = pk(builder, "system words", "s"), pk(builder, "doc words", "d"), pk(builder, "var words", "v")
    r = M.build_request("u3-abcd1234-v01-s1", "claude-sonnet-5-5", "high", 100, sysp, [docp], [var], "5m")
    M.validate_request(r)
    assert r.params["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    bad = dataclasses.replace(r, params={**r.params, "messages": [{"role": "user", "content": [
        {"type": "text", "text": "hand-written lab text appended"}]}]})
    with pytest.raises(M.ManifestError):
        M.validate_request(bad)
    extra = dataclasses.replace(r, packs=r.packs + (pk(builder, "unused", "u"),))
    with pytest.raises(M.ManifestError):                                   # a referenced Pack not present in params
        M.validate_request(extra)


def test_cache_ttl_1h_block(builder):
    r = M.build_request("p-1", "claude-sonnet-5-5", "low", 10, None, [pk(builder, "long prefix")], [pk(builder, "v", "v")], "1h")
    assert r.params["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert "system" not in r.params


@pytest.mark.parametrize("bad", ["a:b", "x" * 65, "", "sp ace"])
def test_custom_id_regex(builder, bad):
    p = pk(builder)
    r = M.build_request(bad, "claude-sonnet-5-5", "low", 10, None, [], [p])
    with pytest.raises(M.ManifestError):
        M.validate_request(r)


def test_custom_id_builder():
    cid = M.make_custom_id("U3", "pack bytes", "V03", 2)
    assert cid.startswith("u3-") and cid.endswith("-v03-s2") and len(cid) <= 64
    assert M.make_custom_id("seat-doclint", b"x", "o", 1).startswith("lint-")


def test_forbidden_params_and_unregistered_schema(builder):
    p = pk(builder)
    r = M.build_request("x-1", "claude-sonnet-5-5", "low", 10, None, [], [p], extra={"stream": True})
    with pytest.raises(M.ManifestError):
        M.validate_request(r)
    s = {"type": "object", "properties": {}}
    r2 = M.build_request("x-2", "claude-sonnet-5-5", "low", 10, None, [], [p], schema=s)
    with pytest.raises(M.ManifestError):
        M.validate_request(r2)
    M.validate_request(r2, schema_shas=frozenset({sha256_text(canon_json(s))}))
    r3 = M.build_request("x-3", "claude-sonnet-5-5", "low", 0, None, [], [p])
    with pytest.raises(M.ManifestError):
        M.validate_request(r3)


def test_duplicate_ids_and_manifest_roundtrip(builder, tmp_path):
    p = pk(builder)
    a = M.build_request("x-1", "claude-sonnet-5-5", "low", 10, None, [], [p])
    with pytest.raises(M.ManifestError):
        M.manifest_rows([a, a])
    sha = M.write_manifest([a], tmp_path / "m.jsonl")
    rows, sha2 = M.load_manifest(tmp_path / "m.jsonl")
    assert sha == sha2 and rows[0]["pack_sha256s"] == [p.pack_sha256]
    (tmp_path / "m.jsonl").write_text((tmp_path / "m.jsonl").read_text().replace('"max_tokens":10', '"max_tokens":11'))
    with pytest.raises(M.ManifestError):
        M.load_manifest(tmp_path / "m.jsonl")


def test_doclint_path_uses_frozen_build_params(builder):
    res, pre = pk(builder, "RESULTS body text", "r"), pk(builder, "PREREG body text", "p")
    r = M.build_doclint_request("lint-12345678-o-s1", "claude-opus-5-5", res, pre)
    assert r.params["thinking"] == {"type": "adaptive"} and len(r.packs) == 2
    # params text other than the frozen template must equal the Packs' bytes
    tampered = dataclasses.replace(r, packs=(pk(builder, "RESULTS body TEXT", "r2"), pre))
    with pytest.raises(M.ManifestError):
        M.validate_request(tampered, frozen_texts=frozenset(), schema_shas=frozenset())
    from tools.api_batch.doclint_bridge import doc_lint_module
    dl = doc_lint_module()
    with pytest.raises(M.ManifestError):                                     # without the frozen set the template text is refused
        M.validate_request(r, frozen_texts=frozenset(), schema_shas=frozenset({sha256_text(canon_json(dl.SCHEMA))}))


def test_cost_pin_p0_probe_batch():
    h = prices.hard_worst_request
    hai = h("claude-haiku-5-5", 0, 500, 1024, None)
    son = 2 * h("claude-sonnet-5-5", 1200, 100, 2048, "1h")
    fab = h("claude-fable-5-1", 0, 1000, 4096, None)
    assert round(hai, 4) == 0.0003 and round(son, 4) == 0.0255 and round(fab, 4) == 0.1074
    assert abs((hai + son + fab) - 0.1332) < 1e-4
    assert abs(h("claude-haiku-5-5", 0, 300, 256, None) - 0.00008) < 1e-5
    with pytest.raises(prices.UnknownModel):
        h("claude-nonexistent", 1, 1, 1, None)
    assert prices.price("claude-haiku-5-5", 100001)[0] == 0.25
    assert abs(3 * (45000 * 5 + 32000 * 20) / 1e6 - prices.standard_hard_worst("claude-opus-5-5", 45000, 0, 32000, "5m", n=3)) < 1e-9


# ------------------------------------------------------------------ multi-breakpoint (plan api-joyful-island B)
def test_cache_after_marks_exactly_the_named_blocks(builder):
    sysp, hdr, memo, var = pk(builder, "system words", "s"), pk(builder, "header", "h"), pk(builder, "memo", "m"), pk(builder, "v", "v")
    r = M.build_request("r-1", "claude-sonnet-5-5", "high", 10, sysp, [hdr, memo], [var], "5m", cache_after=[sysp, hdr])
    M.validate_request(r)
    assert r.params["system"][0]["cache_control"] == {"type": "ephemeral"}
    c = r.params["messages"][0]["content"]
    assert [("cache_control" in b) for b in c] == [True, False, False]
    assert r.meta["cache_after"] == ["s", "h"]


def test_cache_after_ignored_without_ttl_and_legacy_default_unchanged(builder):
    hdr, memo, var = pk(builder, "header", "h"), pk(builder, "memo", "m"), pk(builder, "v", "v")
    none = M.build_request("r-1", "claude-sonnet-5-5", "high", 10, None, [hdr, memo], [var], "none", cache_after=[hdr])
    assert "cache_control" not in canon_json(none.params) and "cache_after" not in none.meta
    legacy = M.build_request("r-2", "claude-sonnet-5-5", "high", 10, None, [hdr, memo], [var], "5m")
    assert [("cache_control" in b) for b in legacy.params["messages"][0]["content"]] == [False, True, False]
    assert "cache_after" not in legacy.meta


def test_cache_after_refusals(builder):
    ps = [pk(builder, f"d{i}", f"d{i}") for i in range(5)]
    v = pk(builder, "v", "v")
    with pytest.raises(M.ManifestError, match="> 4"):
        M.build_request("r-1", "claude-sonnet-5-5", "high", 10, None, ps, [v], "5m", cache_after=ps)
    with pytest.raises(M.ManifestError, match="matches 0"):
        M.build_request("r-2", "claude-sonnet-5-5", "high", 10, None, ps[:1], [v], "5m", cache_after=[ps[1]])
    with pytest.raises(M.ManifestError, match="images_after"):
        M.build_request("r-3", "claude-sonnet-5-5", "high", 10, None, ps[:1], [v], "5m", images_after=ps[1])


def test_cache_split_formula2():
    sys_b = {"type": "text", "text": "S" * 300}
    doc = {"type": "document", "title": "h", "source": {"type": "text", "media_type": "text/plain", "data": "H" * 3000},
           "cache_control": {"type": "ephemeral", "ttl": "1h"}}
    tail = {"type": "text", "text": "T" * 150}
    p = {"system": [sys_b], "messages": [{"role": "user", "content": [doc, tail]}]}
    from tools.api_batch.common import est_tokens
    assert M.cache_split(p) == (est_tokens("S" * 300) + est_tokens("H" * 3000), est_tokens("T" * 150), "1h")
    doc.pop("cache_control")
    assert M.cache_split(p) == (0, est_tokens("S" * 300) + est_tokens("H" * 3000) + est_tokens("T" * 150), "none")


def test_validate_refuses_bad_cache_control(builder):
    d, v = pk(builder, "doc", "d"), pk(builder, "v", "v")
    r = M.build_request("r-1", "claude-sonnet-5-5", "high", 10, None, [d], [v], "5m")
    r.params["messages"][0]["content"][0]["cache_control"] = {"type": "ephemeral", "ttl": "2h"}
    with pytest.raises(M.ManifestError, match="ephemeral"):
        M.validate_request(r)
    r.params["messages"][0]["content"][0]["cache_control"] = {"type": "ephemeral"}
    r.params["messages"][0]["content"][1]["cache_control"] = {"type": "ephemeral", "ttl": "1h"}
    with pytest.raises(M.ManifestError, match="longer TTLs"):
        M.validate_request(r)


def test_cache_split_uses_the_last_of_several_breakpoints():
    from tools.api_batch.common import est_tokens
    cc = {"type": "ephemeral"}
    p = {"system": [{"type": "text", "text": "S" * 300, "cache_control": cc}],
         "messages": [{"role": "user", "content": [
             {"type": "document", "title": "a", "source": {"type": "text", "media_type": "text/plain", "data": "A" * 1500}, "cache_control": cc},
             {"type": "document", "title": "b", "source": {"type": "text", "media_type": "text/plain", "data": "B" * 900}, "cache_control": cc},
             {"type": "document", "title": "c", "source": {"type": "text", "media_type": "text/plain", "data": "C" * 600}},
             {"type": "text", "text": "T" * 30}]}]}
    # est_tokens = ceil(bytes / 1.5): 200 + 1000 + 600 written; 400 + 20 uncached
    assert (est_tokens("S" * 300), est_tokens("A" * 1500), est_tokens("B" * 900)) == (200, 1000, 600)
    assert M.cache_split(p) == (1800, 420, "5m")
    assert M.cache_split({"system": "S" * 300, "messages": [{"role": "user", "content": "x"}]}) == (0, 201, "none")
