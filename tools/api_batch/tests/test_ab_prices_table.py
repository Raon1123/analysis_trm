"""Price/min-cache table pins (batch $/MTok = 50% of standard). fable-5-2 row: docs fetched 2026-11-10."""
import pytest

from tools.api_batch import cache_audit, cli, prices


def test_fable_5_2_batch_prices():
    assert prices.price("claude-fable-5-2") == (4.00, 20.00, 0.10, 5.00, 8.00)
    assert prices.price("claude-fable-5-2", 900_000) == prices.PRICES["claude-fable-5-2"]   # no long-context tier
    # standard 8/40, cw5m 10, cw1h 16, cr 0.20 halved
    assert prices.price("claude-fable-5-2") == tuple(x / 2 for x in (8.0, 40.0, 0.20, 10.0, 16.0))


def test_other_rows_unchanged():
    assert prices.PRICES["claude-sonnet-5-5"] == (1.00, 5.00, 0.05, 1.25, 2.00)
    assert prices.PRICES["claude-opus-5-5"] == (2.00, 10.00, 0.10, 2.50, 4.00)
    assert prices.PRICES["claude-fable-5-1"] == (5.00, 25.00, 0.125, 6.25, 10.00)
    assert prices.PRICES["claude-haiku-5-5"] == (0.05, 0.25, 0.005, 0.0625, 0.10)
    assert prices.HAIKU_LONG == (0.25, 1.25, 0.025, 0.3125, 0.50)
    assert set(prices.PRICES) == {"claude-sonnet-5-5", "claude-opus-5-5", "claude-fable-5-1",
                                  "claude-fable-5-2", "claude-haiku-5-5"}


def test_unknown_model_still_refused():
    for m in ("claude-fable-5-3", "claude-fable-5-2-x", "claude-nope-9"):
        with pytest.raises(prices.UnknownModel):
            prices.price(m)


def test_min_cache_tokens_fable_5_2():
    assert cache_audit.MIN_CACHE_TOKENS["claude-fable-5-2"] == 512
    assert cache_audit.MIN_CACHE_TOKENS["claude-fable-5-1"] == 512
    assert cache_audit.MIN_CACHE_UNKNOWN == 4096


def test_min_cache_tokens_other_models_pinned():
    for m in ("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"):
        assert cache_audit.MIN_CACHE_TOKENS[m] == 512


def test_skeleton_defaults_to_fable_5_2():
    assert "model: claude-fable-5-2\n" in cli.SKELETON_U3
    assert cli.SKELETON_U3.split("model: ")[1].split()[0] in prices.PRICES
