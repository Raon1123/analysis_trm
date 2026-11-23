"""Prompt-cache provenance of scripts/judge_openai.py (offline; _post is stubbed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pytest  # noqa: E402

import judge_openai as jo  # noqa: E402
from judge_contract import NoulQuestion  # noqa: E402

TODAY_KEYS = {"model", "messages", "temperature", "top_p", "max_tokens", "seed",
              "logprobs", "top_logprobs"}
QS = {"a": NoulQuestion(instructions="q1", criteria_true="t", criteria_false="f"),
      "b": NoulQuestion(instructions="q2", criteria_true="t", criteria_false="f")}


def _resp(**extra):
    r = {"model": "m", "choices": [{"message": {"content": "Yes"}}]}
    r.update(extra)
    return r


def _run(monkeypatch, responses, **kw):
    sent = []
    it = iter(responses)

    def fake_post(self, payload):
        sent.append(payload)
        return next(it)

    monkeypatch.setattr(jo.OpenAICompatJudge, "_post", fake_post)
    monkeypatch.delenv("OPENAI_COMPAT_CACHE_PROMPT", raising=False)
    j = jo.OpenAICompatJudge(model="m", base_url="http://127.0.0.1:1/v1", **kw)
    return j.ask({"claim": "x"}, QS), sent


def test_server_default_payload_unchanged(monkeypatch):
    res, sent = _run(monkeypatch, [_resp(), _resp()])
    assert all(set(p) == TODAY_KEYS for p in sent)
    assert res.usage["cache_prompt_requested"] == "server-default"


@pytest.mark.parametrize("opt,val", [("off", False), ("on", True)])
def test_explicit_switch(monkeypatch, opt, val):
    res, sent = _run(monkeypatch, [_resp(), _resp()], cache_prompt=opt)
    assert all(p["cache_prompt"] is val for p in sent)
    assert set(sent[0]) == TODAY_KEYS | {"cache_prompt"}
    assert res.usage["cache_prompt_requested"] == opt


def test_env_option(monkeypatch):
    monkeypatch.setattr(jo.OpenAICompatJudge, "_post", lambda self, p: _resp())
    monkeypatch.setenv("OPENAI_COMPAT_CACHE_PROMPT", "off")
    j = jo.OpenAICompatJudge(model="m", base_url="http://127.0.0.1:1/v1")
    assert j.cache_prompt == "off" and j.describe()["cache_prompt_requested"] == "off"


def test_source_usage_details(monkeypatch):
    r = _resp(usage={"prompt_tokens_details": {"cached_tokens": 7}})
    res, _ = _run(monkeypatch, [r, r])
    assert res.usage["cached_prompt_tokens"] == 14
    assert res.usage["cached_prompt_tokens_source"] == "usage.prompt_tokens_details.cached_tokens"


def test_source_tokens_cached(monkeypatch):
    res, _ = _run(monkeypatch, [_resp(tokens_cached=5), _resp(tokens_cached=0)])
    assert res.usage["cached_prompt_tokens"] == 5
    assert res.usage["cached_prompt_tokens_source"] == "tokens_cached"


def test_source_timings(monkeypatch):
    r = _resp(timings={"cache_n": 3})
    res, _ = _run(monkeypatch, [r, r])
    assert res.usage["cached_prompt_tokens"] == 6
    assert res.usage["cached_prompt_tokens_source"] == "timings.cache_n"


def test_not_reported_is_none_not_zero(monkeypatch):
    res, _ = _run(monkeypatch, [_resp(usage={"prompt_tokens": 9}), _resp()])
    assert res.usage["cached_prompt_tokens"] is None
    assert res.usage["cached_prompt_tokens_source"] == "none"


def test_reported_zero_is_zero(monkeypatch):
    r = _resp(usage={"prompt_tokens_details": {"cached_tokens": 0}})
    res, _ = _run(monkeypatch, [r, r])
    assert res.usage["cached_prompt_tokens"] == 0


def test_pyes_unaffected(monkeypatch):
    a, _ = _run(monkeypatch, [_resp(), _resp()])
    b, _ = _run(monkeypatch, [_resp(tokens_cached=4), _resp()], cache_prompt="off")
    assert a.answers == b.answers
