"""Weight-tied loop option for arch=transformers_baseline (looped-transformer control).

Config fields under test (models/recursive_reasoning/transformers_baseline.py):
  loops                 number of weight-tied passes of the whole H_layers stack
  loop_input_injection  re-add input embeddings at every loop (else loop 0 only)
  loop_grad_cycles      0 = full backprop; n>0 = grad only through the last n loops

CPU, float32, tiny config -- no GPU, no data files.
"""

from __future__ import annotations

import pydantic
import pytest
import torch

from models.recursive_reasoning.transformers_baseline import Model_ACTV2

B = 4


def _cfg(**over):
    cfg = dict(
        batch_size=B,
        seq_len=11,
        puzzle_emb_ndim=32,
        num_puzzle_identifiers=1,
        vocab_size=12,
        H_cycles=1,
        H_layers=2,
        hidden_size=32,
        expansion=4,
        num_heads=2,
        pos_encodings="rope",
        halt_max_steps=1,
        halt_exploration_prob=0.1,
        forward_dtype="float32",
    )
    cfg.update(over)
    return cfg


def _build(seed: int = 0, **over) -> Model_ACTV2:
    torch.manual_seed(seed)
    return Model_ACTV2(_cfg(**over))


def _batch():
    g = torch.Generator().manual_seed(123)
    return {
        "inputs": torch.randint(1, 12, (B, 11), generator=g, dtype=torch.int32),
        "puzzle_identifiers": torch.zeros(B, dtype=torch.int32),
    }


def _inner_forward(model: Model_ACTV2, batch):
    """Run the inner model once from the H_init-reset carry; return (z_H carry, logits)."""
    inner = model.inner
    carry = inner.reset_carry(torch.ones(B, dtype=torch.bool), inner.empty_carry(B))
    new_carry, logits, _q = inner(carry, batch)
    return new_carry.z_H, logits, carry


def _block_params(model: Model_ACTV2):
    return list(model.inner.H_level.parameters())


def test_loops1_bit_identical_to_single_h_level_call():
    model = _build(loops=1).eval()
    batch = _batch()
    with torch.no_grad():
        z_H, logits, carry = _inner_forward(model, batch)
        inner = model.inner
        cos_sin = inner.rotary_emb()
        emb = inner._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])
        ref = inner.H_level(carry.z_H, emb, cos_sin=cos_sin)
        ref_logits = inner.lm_head(ref)[:, inner.puzzle_emb_len:]
    assert torch.equal(z_H, ref)
    assert torch.equal(logits, ref_logits)


def test_loops3_differs_and_param_count_unchanged():
    m1 = _build(seed=0, loops=1).eval()
    m3 = _build(seed=0, loops=3).eval()
    n1 = sum(p.numel() for p in m1.parameters())
    n3 = sum(p.numel() for p in m3.parameters())
    assert n1 == n3
    # same init (same seed, identical module construction order)
    for (a, pa), (b, pb) in zip(m1.state_dict().items(), m3.state_dict().items()):
        assert a == b and torch.equal(pa, pb)
    batch = _batch()
    with torch.no_grad():
        z1, _, _ = _inner_forward(m1, batch)
        z3, _, _ = _inner_forward(m3, batch)
    assert not torch.allclose(z1, z3)


def test_truncated_grad_reaches_all_block_params_finite():
    model = _build(loops=3, loop_grad_cycles=1).train()
    batch = _batch()
    _z, logits, _ = _inner_forward(model, batch)
    logits.float().square().mean().backward()
    for name, p in model.inner.H_level.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, name


def test_truncated_grad_matches_last_loop_only():
    """loop_grad_cycles=1, loops=3 == run 2 loops under no_grad, then 1 loop with grad."""
    model = _build(loops=3, loop_grad_cycles=1).eval()
    batch = _batch()
    _z, logits, carry = _inner_forward(model, batch)
    logits.float().square().mean().backward()
    g_model = [p.grad.clone() for p in _block_params(model)]

    model.zero_grad(set_to_none=True)
    inner = model.inner
    cos_sin = inner.rotary_emb()
    emb = inner._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])
    with torch.no_grad():
        z = inner.H_level(carry.z_H, emb, cos_sin=cos_sin)
        z = inner.H_level(z, emb, cos_sin=cos_sin)
    z = inner.H_level(z, emb, cos_sin=cos_sin)
    inner.lm_head(z)[:, inner.puzzle_emb_len:].float().square().mean().backward()
    for a, p in zip(g_model, _block_params(model)):
        assert torch.allclose(a, p.grad)


def test_no_input_injection_runs_and_differs():
    batch = _batch()
    m_inj = _build(seed=0, loops=3, loop_input_injection=True).eval()
    m_no = _build(seed=0, loops=3, loop_input_injection=False).eval()
    with torch.no_grad():
        z_inj, _, _ = _inner_forward(m_inj, batch)
        z_no, logits_no, _ = _inner_forward(m_no, batch)
    assert torch.isfinite(logits_no).all()
    assert not torch.allclose(z_inj, z_no)


@pytest.mark.parametrize(
    "over",
    [
        dict(loops=3, loop_grad_cycles=4),
        dict(loops=0),
        dict(loops=2, loop_grad_cycles=-1),
    ],
)
def test_invalid_loop_config_raises(over):
    with pytest.raises(pydantic.ValidationError):
        Model_ACTV2(_cfg(**over))


def test_full_act_wrapper_forward_loops():
    """Smoke: the ACT wrapper path (as pretrain.py drives it) runs with loops>1."""
    model = _build(loops=4, loop_grad_cycles=2).train()
    batch = _batch()
    carry = model.initial_carry(batch)
    _carry, out = model(carry, batch)
    assert out["logits"].shape == (B, 11, 12)
    assert torch.isfinite(out["logits"]).all()
