"""models/recursive_reasoning/transformers_baseline_skip.py (R3 / g7s arm).

The frozen transformers_baseline.py is untouched; the subclass adds only the injection
schedule `loop_input_injection_skip_period` (0 = base behaviour; p>=2 skips loops i%p==p-1).
CPU, float32, tiny config -- no GPU, no data files.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pydantic
import pytest
import torch
import yaml

from models.recursive_reasoning.transformers_baseline import Model_ACTV2 as Base
from models.recursive_reasoning.transformers_baseline_skip import Model_ACTV2 as Skip

B = 2
REPO = Path(__file__).resolve().parents[1]
FROZEN_SHA = "a8a4c5da76b86a7c3bf595da3c7980e0aca45a5d786a041a3f2a5556b8e23ba6"


def _cfg(**over):
    cfg = dict(batch_size=B, seq_len=11, puzzle_emb_ndim=32, num_puzzle_identifiers=1, vocab_size=12,
               H_cycles=1, H_layers=1, hidden_size=32, expansion=4, num_heads=2, pos_encodings="rope",
               halt_max_steps=1, halt_exploration_prob=0.1, forward_dtype="float32")
    cfg.update(over)
    return cfg


def _build(cls, seed=0, **over):
    torch.manual_seed(seed)
    return cls(_cfg(**over))


def _batch():
    g = torch.Generator().manual_seed(123)
    return {"inputs": torch.randint(1, 12, (B, 11), generator=g, dtype=torch.int32),
            "puzzle_identifiers": torch.zeros(B, dtype=torch.int32)}


def _inner_call(model):
    inner = model.inner
    carry = inner.reset_carry(torch.ones(B, dtype=torch.bool), inner.empty_carry(B))
    return inner(carry, _batch())


def _run(model):
    new_carry, logits, _q = _inner_call(model)
    logits.float().square().mean().backward()
    grads = {n: (None if p.grad is None else p.grad.clone()) for n, p in model.named_parameters()}
    return new_carry.z_H, logits.detach(), grads


def _injection_flags(model):
    seen = []
    orig = model.inner.H_level.forward

    def spy(hidden_states, input_injection, **kw):
        seen.append(input_injection is not None)
        return orig(hidden_states, input_injection, **kw)

    model.inner.H_level.forward = spy
    _inner_call(model)
    return seen


@pytest.mark.parametrize("train", [True, False])
@pytest.mark.parametrize("grad_cycles", [0, 7])
def test_period0_bitwise_identical_to_frozen_base(train, grad_cycles):
    over = dict(loops=21, loop_grad_cycles=grad_cycles)
    m_base = _build(Base, **over).train(train)
    m_p0 = _build(Skip, loop_input_injection_skip_period=0, **over).train(train)
    assert list(m_base.state_dict().keys()) == list(m_p0.state_dict().keys())
    for (na, a), (nb, b) in zip(m_base.state_dict().items(), m_p0.state_dict().items()):
        assert na == nb and torch.equal(a, b), na  # same seed -> same init draws
    z0, l0, g0 = _run(m_base)
    z1, l1, g1 = _run(m_p0)
    assert torch.equal(z0, z1) and torch.equal(l0, l1)
    assert g0.keys() == g1.keys()
    for k in g0:
        if g0[k] is None:
            assert g1[k] is None, k
        else:
            assert torch.equal(g0[k], g1[k]), k


@pytest.mark.parametrize("grad_cycles", [0, 7])
def test_period7_injects_at_18_loops_skips_6_13_20(grad_cycles):
    m = _build(Skip, loops=21, loop_grad_cycles=grad_cycles, loop_input_injection_skip_period=7)
    seen = _injection_flags(m)
    assert len(seen) == 21 and sum(seen) == 18
    assert [i for i, s in enumerate(seen) if not s] == [6, 13, 20]


@pytest.mark.parametrize("over", [
    dict(loops=21, loop_input_injection_skip_period=1),
    dict(loops=1, loop_input_injection_skip_period=1),
    dict(loops=21, loop_input_injection_skip_period=-1),
    dict(loops=20, loop_input_injection_skip_period=7),
    dict(loops=21, loop_input_injection=False, loop_input_injection_skip_period=7),
])
def test_invalid_settings_raise(over):
    with pytest.raises(ValueError):
        Skip(_cfg(**over))
    with pytest.raises(pydantic.ValidationError):
        Skip(_cfg(**over))


def test_period1_message_is_clear():
    with pytest.raises(ValueError, match=r"skip_period=1.*0 \(off\) or >= 2"):
        Skip(_cfg(loops=21, loop_input_injection_skip_period=1))


def test_period2_accepted():
    Skip(_cfg(loops=20, loop_input_injection_skip_period=2))


def test_period7_grad_window_reaches_input_embedding():
    m = _build(Skip, loops=21, loop_grad_cycles=7, loop_input_injection_skip_period=7).train()
    _z, _l, grads = _run(m)
    names = [n for n in grads if "embed_tokens" in n]
    assert names
    for n in names:
        g = grads[n]
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, n


def test_param_count_and_state_dict_keys_equal_base():
    mb = _build(Base, loops=21, loop_grad_cycles=7)
    ms = _build(Skip, loops=21, loop_grad_cycles=7, loop_input_injection_skip_period=7)
    assert sum(p.numel() for p in mb.parameters()) == sum(p.numel() for p in ms.parameters())
    sb, ss = mb.state_dict(), ms.state_dict()
    assert list(sb.keys()) == list(ss.keys())
    assert all(sb[k].shape == ss[k].shape for k in sb)


def test_frozen_file_sha_and_yaml_differ_only_in_name_and_skip_key():
    src = REPO / "models/recursive_reasoning/transformers_baseline.py"
    assert hashlib.sha256(src.read_bytes()).hexdigest() == FROZEN_SHA
    base = yaml.safe_load((REPO / "config/arch/transformers_baseline.yaml").read_text())
    new = yaml.safe_load((REPO / "config/arch/transformers_baseline_skip.yaml").read_text())
    assert base["name"] == "recursive_reasoning.transformers_baseline@Model_ACTV2"
    assert new["name"] == "recursive_reasoning.transformers_baseline_skip@Model_ACTV2"
    assert new["loop_input_injection_skip_period"] == 7
    assert "loop_input_injection_skip_period" not in base
    b2 = {k: v for k, v in base.items() if k != "name"}
    n2 = {k: v for k, v in new.items() if k not in ("name", "loop_input_injection_skip_period")}
    assert b2 == n2
