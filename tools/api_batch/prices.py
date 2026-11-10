"""Batch price table ($/MTok) and cost helpers. Source: 60_BATCH_API_PIPELINE_DESIGN.md §3 F3/F8 and
60_costmodel.py (docs fetched 2026-10-23: platform.claude.com/docs/en/build-with-claude/batch-processing,
.../prompt-caching). Estimates, not billing (Console is the ledger of record). Unknown models are refused.
"""
from __future__ import annotations

SOURCE = ("60_costmodel.py BATCH table; batch-processing + prompt-caching docs fetched 2026-10-23; "
          "claude-fable-5-2 row: pricing/batch-processing/prompt-caching docs fetched 2026-11-10")
# model alias: (p_in, p_out, p_cr, p_cw_5m, p_cw_1h)
PRICES = {
    "claude-sonnet-5-5": (1.00, 5.00, 0.05, 1.25, 2.00),
    "claude-opus-5-5":   (2.00, 10.00, 0.10, 2.50, 4.00),
    "claude-fable-5-1":  (5.00, 25.00, 0.125, 6.25, 10.00),
    # fable-5-2 (docs fetched 2026-11-10): std 8/40, cw5m 10, cw1h 16, cr 0.20 -> batch = 50%; no long-context tier
    "claude-fable-5-2":  (4.00, 20.00, 0.10, 5.00, 8.00),
    "claude-haiku-5-5":  (0.05, 0.25, 0.005, 0.0625, 0.10),   # prompts <= 100K tokens
}
HAIKU_LONG = (0.25, 1.25, 0.025, 0.3125, 0.50)                  # prompts > 100K tokens (F3)


class UnknownModel(ValueError):
    pass


def price(model: str, prompt_tokens: int = 0):
    if model not in PRICES:
        raise UnknownModel(f"model {model!r} is not in prices.py (refused at plan)")
    if model == "claude-haiku-5-5" and prompt_tokens > 100_000:
        return HAIKU_LONG
    return PRICES[model]


def hard_worst_request(model: str, t_ctx: int, t_var: int, max_tokens: int, ttl: str | None) -> float:
    """Every request writes the cache (TTL '5m'|'1h'; None/'none' = uncached full input) and uses
    all of max_tokens (G5)."""
    pin, pout, _pcr, cw5, cw1 = price(model, t_ctx + t_var)
    pcw = {"5m": cw5, "1h": cw1}.get(ttl or "none", pin)
    return (t_ctx * pcw + t_var * pin + max_tokens * pout) / 1e6


def expected_request(model: str, t_ctx: int, t_var: int, max_tokens: int, ttl: str | None,
                     r_hit: float = 0.5, out_frac: float = 0.5) -> float:
    """Planning figure only: assumed hit rate r_hit and out_frac*max_tokens output tokens."""
    pin, pout, pcr, cw5, cw1 = price(model, t_ctx + t_var)
    if (ttl or "none") == "none":
        ctx = t_ctx * pin
    else:
        pcw = cw5 if ttl == "5m" else cw1
        ctx = t_ctx * (r_hit * pcr + (1 - r_hit) * pcw)
    return (ctx + t_var * pin + out_frac * max_tokens * pout) / 1e6


def cost_from_usage(model: str, usage: dict) -> float:
    """Batch-price estimate of one finished request from its usage block."""
    u = usage or {}
    g = lambda k: int(u.get(k) or 0)  # noqa: E731
    pin, pout, pcr, cw5, cw1 = price(model, g("input_tokens") + g("cache_read_input_tokens"))
    cc = u.get("cache_creation") or {}
    w1 = int(cc.get("ephemeral_1h_input_tokens") or 0)
    w5 = int(cc.get("ephemeral_5m_input_tokens") or 0)
    if not cc:                                    # only the aggregate is present: treat as 5m
        w5 = g("cache_creation_input_tokens")
    return (g("input_tokens") * pin + g("output_tokens") * pout + g("cache_read_input_tokens") * pcr
            + w5 * cw5 + w1 * cw1) / 1e6


# Standard (synchronous) prices $/MTok = 2x batch (F3: batch is 50%). Used for probe-sync / doclint-sync estimates.
def standard_hard_worst(model: str, t_ctx: int, t_var: int, max_tokens: int, ttl: str | None, n: int = 1) -> float:
    return 2.0 * n * hard_worst_request(model, t_ctx, t_var, max_tokens, ttl)
