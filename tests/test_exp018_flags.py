"""EXP-018 code flags for arch=trm / arch=trm_singlez (+ per_cycle_loss in losses.ACTLossHead).

CPU, tiny bfloat16 models, synthetic batches -- no GPU, no data, no queue, no checkpoints.

1. Default flags reproduce the pre-flag code bit-identically (torch.equal on logits, loss, carry,
   every gradient, every state_dict entry). The reference is the code at the branch point
   (`git show HEAD:...` of the unmodified files, loaded as throw-away modules).
2. Every flag on: shapes, finite loss, backward, and the intended structural change (hooks on L_level).
"""

from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
B, S_TOK, PLEN, D = 4, 11, 4, 32
S = S_TOK + PLEN  # sequence length inside the recurrent state (puzzle prefix + tokens)
H, L = 3, 6
N_CALLS = H * (L + 1)


# ---------------------------------------------------------------- reference (pre-flag) modules
def _load_ref(relpath: str, name: str):
    src = subprocess.run(["git", "-C", str(REPO), "show", f"HEAD:{relpath}"], check=True,
                         capture_output=True, text=True).stdout
    mod = types.ModuleType(name)
    mod.__dict__["__name__"] = name
    sys.modules[name] = mod  # dataclasses / pydantic look the defining module up by name
    exec(compile(src, f"<HEAD:{relpath}>", "exec"), mod.__dict__)
    return mod


@pytest.fixture(scope="module")
def ref_mods():
    return {
        "trm": _load_ref("models/recursive_reasoning/trm.py", "ref_trm"),
        "trm_singlez": _load_ref("models/recursive_reasoning/trm_singlez.py", "ref_trm_singlez"),
        "losses": _load_ref("models/losses.py", "ref_losses"),
    }


def _new_mod(arch):
    if arch == "trm":
        from models.recursive_reasoning import trm as m
    else:
        from models.recursive_reasoning import trm_singlez as m
    return m


def _cfg(**kw):
    c = dict(batch_size=B, seq_len=S_TOK, puzzle_emb_ndim=D, num_puzzle_identifiers=1, vocab_size=11,
             H_cycles=H, L_cycles=L, H_layers=0, L_layers=2, hidden_size=D, expansion=2, num_heads=4,
             pos_encodings="rope", halt_max_steps=2, halt_exploration_prob=0.1, forward_dtype="bfloat16",
             mlp_t=False, puzzle_emb_len=PLEN, no_ACT_continue=True)
    c.update(kw)
    return c


def _batch():
    g = torch.Generator().manual_seed(7)
    inp = torch.stack([torch.cat([torch.randperm(10, generator=g) + 1, torch.zeros(1, dtype=torch.long)])
                       for _ in range(B)]).to(torch.int32)
    lab = torch.stack([torch.cat([torch.randperm(10, generator=g) + 1, torch.zeros(1, dtype=torch.long)])
                       for _ in range(B)]).to(torch.int32)
    return {"inputs": inp, "puzzle_identifiers": torch.zeros(B, dtype=torch.int32), "labels": lab}


def _build(model_mod, loss_mod, cfg, seed=0):
    torch.manual_seed(seed)
    model = model_mod.TinyRecursiveReasoningModel_ACTV1(cfg)
    with torch.no_grad():  # zero-initialised puzzle embedding would make the prefix trivial
        model.inner.puzzle_emb.weights.normal_()
    head = loss_mod.ACTLossHead(model, "stablemax_cross_entropy")
    head.train()
    return head


def _two_steps(head, seed=1):
    """Two ACT steps (second one continues from the carry), backward each; return everything."""
    batch = _batch()
    carry = head.initial_carry(batch)
    torch.manual_seed(seed)
    res = []
    for _ in range(2):
        carry, loss, metrics, outs, _ = head(return_keys=["logits"], carry=carry, batch=batch)
        loss.backward()
        res.append((loss.detach().clone(), outs["logits"].clone(), metrics))
    grads = {n: (None if p.grad is None else p.grad.clone()) for n, p in head.named_parameters()}
    ic = carry.inner_carry
    carry_t = {k: getattr(ic, k).clone() for k in ("z_H", "z_L") if hasattr(ic, k)}
    return res, grads, carry_t, {k: v.clone() for k, v in head.state_dict().items()}


# ---------------------------------------------------------------- 1. default == pre-flag, bitwise
@pytest.mark.parametrize("arch", ["trm", "trm_singlez"])
def test_default_flags_bit_identical_to_pre_flag_code(arch, ref_mods):
    ref = _two_steps(_build(ref_mods[arch], ref_mods["losses"], _cfg(), seed=3))
    from models import losses as new_losses
    new = _two_steps(_build(_new_mod(arch), new_losses, _cfg(), seed=3))
    (r_res, r_g, r_c, r_sd), (n_res, n_g, n_c, n_sd) = ref, new
    assert r_sd.keys() == n_sd.keys()
    for k in r_sd:
        assert torch.equal(r_sd[k], n_sd[k]), k  # same init, same dtypes, buffers stay buffers
    for (rl, rlog, rm), (nl, nlog, nm) in zip(r_res, n_res):
        assert torch.equal(rl, nl) and torch.equal(rlog, nlog)
        assert rm.keys() == nm.keys()
        for k in rm:
            assert torch.equal(rm[k], nm[k]), k
    assert r_g.keys() == n_g.keys()
    for k in r_g:
        assert (r_g[k] is None) == (n_g[k] is None), k
        if r_g[k] is not None:
            assert torch.equal(r_g[k], n_g[k]), k
    for k in r_c:
        assert torch.equal(r_c[k], n_c[k]), k
    assert any(g is not None and g.abs().sum() > 0 for g in r_g.values())  # the comparison is not vacuous


@pytest.mark.parametrize("arch", ["trm", "trm_singlez"])
def test_default_init_vectors_are_buffers_not_parameters(arch):
    from models import losses as new_losses
    inner = _build(_new_mod(arch), new_losses, _cfg()).model.inner
    names = {n for n, _ in inner.named_parameters()}
    assert "L_init" not in names and "H_init" not in names and "step_tok" not in names
    assert not hasattr(inner, "step_tok") and not hasattr(inner, "rotary_emb_tok")


# ---------------------------------------------------------------- helpers for structural checks
class Trace:
    """Records every L_level call: (input hidden_states, injection or None, output, grad_enabled)."""

    def __init__(self, inner):
        self.calls = []
        self.h1 = inner.L_level.register_forward_pre_hook(self._pre, with_kwargs=True)
        self.h2 = inner.L_level.register_forward_hook(self._post, with_kwargs=True)

    def _pre(self, mod, args, kwargs):
        self.calls.append({"hs": args[0], "inj": args[1] if len(args) > 1 else None,
                           "grad_enabled": torch.is_grad_enabled(), "hs_requires_grad": args[0].requires_grad})

    def _post(self, mod, args, kwargs, out):
        self.calls[-1]["out"] = out

    def close(self):
        self.h1.remove()
        self.h2.remove()


def _one_forward(arch, **flags):
    """One inner forward from the reset carry; returns (head, trace, x_emb, carry_in, outputs-tuple)."""
    from models import losses as new_losses
    head = _build(_new_mod(arch), new_losses, _cfg(**flags), seed=5)
    inner = head.model.inner
    batch = _batch()
    carry_in = inner.reset_carry(torch.ones(B, dtype=torch.bool), inner.empty_carry(B))
    x = inner._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])
    tr = Trace(inner)
    out = inner(carry_in, batch)
    tr.close()
    return head, tr, x, carry_in, out


def _full_step(arch, **flags):
    """Full ACTLossHead step + backward (loss finite, grads exist)."""
    from models import losses as new_losses
    head = _build(_new_mod(arch), new_losses, _cfg(**flags), seed=5)
    batch = _batch()
    carry = head.initial_carry(batch)
    carry, loss, metrics, outs, _ = head(return_keys=["logits"], carry=carry, batch=batch)
    assert torch.isfinite(loss)
    assert outs["logits"].shape == (B, S_TOK, 11)
    loss.backward()
    return head, loss, metrics


TRM_FLAGS = ["inject_H", "zH_every_call", "zH_freeze", "zL_reset", "zH_reset",
             "full_grad_cycles", "per_cycle_loss", "learnable_init"]
SZ_FLAGS = ["inject_last", "step_token", "full_grad_cycles", "per_cycle_loss", "learnable_init",
            "boundary_copy", "x_copy"]


@pytest.mark.parametrize("flag", TRM_FLAGS)
def test_trm_flag_on_runs_and_backprops(flag):
    head, loss, metrics = _full_step("trm", **{flag: True})
    assert all(torch.isfinite(p.grad).all() for p in head.parameters() if p.grad is not None)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.model.inner.L_level.parameters())


@pytest.mark.parametrize("flag", SZ_FLAGS)
def test_singlez_flag_on_runs_and_backprops(flag):
    head, loss, metrics = _full_step("trm_singlez", **{flag: True})
    assert all(torch.isfinite(p.grad).all() for p in head.parameters() if p.grad is not None)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.model.inner.L_level.parameters())


# ---------------------------------------------------------------- 2a. trm structure
def _zh_idx(n_per=L + 1):
    return [h * n_per + L for h in range(H)]


def test_trm_default_trace_shape():
    _, tr, x, c, _ = _one_forward("trm")
    assert len(tr.calls) == N_CALLS
    for i in _zh_idx():  # z_H <- f(z_H, z_L): injection is exactly the previous (z_L) output
        assert torch.equal(tr.calls[i]["inj"], tr.calls[i - 1]["out"])


def test_trm_inject_H_adds_x_to_zH_update():
    _, tr, x, c, _ = _one_forward("trm", inject_H=True)
    assert len(tr.calls) == N_CALLS
    for i in _zh_idx():
        assert torch.equal(tr.calls[i]["inj"], tr.calls[i - 1]["out"] + x)
    # both the no_grad cycles (0,1) and the grad cycle (2) are covered
    assert [tr.calls[i]["grad_enabled"] for i in _zh_idx()] == [False, False, True]
    # z_L updates unchanged: injection is z_H + x
    assert torch.equal(tr.calls[0]["inj"], c.z_H + x)


def test_trm_zH_every_call_updates_zH_after_each_zL_call():
    _, tr, x, c, _ = _one_forward("trm", zH_every_call=True)
    assert len(tr.calls) == H * L * 2
    for j in range(H * L):
        zl, zh = tr.calls[2 * j], tr.calls[2 * j + 1]
        assert torch.equal(zh["inj"], zl["out"])           # z_H <- f(z_H, z_L) right after every z_L call
        if j + 1 < H * L:                                  # next z_L call sees the freshly updated z_H
            assert torch.equal(tr.calls[2 * j + 2]["inj"], zh["out"] + x)


def test_trm_zH_freeze_holds_zH_through_early_cycles():
    _, tr, x, c, _ = _one_forward("trm", zH_freeze=True)
    assert len(tr.calls) == H * L + 1                      # only the final readout update remains
    for i in range(H * L):                                 # every z_L call sees the carried (initial) z_H
        assert torch.equal(tr.calls[i]["inj"], c.z_H + x)
    assert torch.equal(tr.calls[-1]["hs"], c.z_H)          # readout update starts from the same frozen value


def test_trm_zL_reset_restores_L_init_each_cycle():
    head, tr, x, c, _ = _one_forward("trm", zL_reset=True)
    inner = head.model.inner
    assert isinstance(inner.L_init, torch.nn.Parameter) and inner.L_init.requires_grad
    li = inner.L_init.to(inner.forward_dtype)
    for h in range(H):
        hs = tr.calls[h * (L + 1)]["hs"]
        assert torch.equal(hs, li.expand_as(hs))
    # without the flag, cycle 1 starts from cycle 0's z_H output instead
    _, tr0, *_ = _one_forward("trm")
    assert not torch.equal(tr0.calls[L + 1]["hs"], tr.calls[L + 1]["hs"])


def test_trm_zH_reset_restores_H_init_each_cycle():
    head, tr, x, c, _ = _one_forward("trm", zH_reset=True)
    inner = head.model.inner
    assert isinstance(inner.H_init, torch.nn.Parameter)
    hi = inner.H_init.to(inner.forward_dtype)
    for h in range(H):
        first = tr.calls[h * (L + 1)]
        assert torch.equal(first["inj"], hi.expand_as(first["hs"]) + x)
        assert torch.equal(tr.calls[h * (L + 1) + L]["hs"], hi.expand_as(first["hs"]))  # z_H update input


def test_trm_learnable_init_alone_gets_no_gradient_under_truncation():
    head, loss, _ = _full_step("trm", learnable_init=True)
    inner = head.model.inner
    assert inner.H_init.grad is None and inner.L_init.grad is None  # truncated cycles cut the path
    head, loss, _ = _full_step("trm", learnable_init=True, full_grad_cycles=True)
    assert head.model.inner.L_init.grad.abs().sum() > 0


def test_trm_learnable_init_params_get_gradient_and_keep_state_dict_keys():
    head, loss, _ = _full_step("trm", learnable_init=True, zL_reset=True, zH_reset=True)
    inner = head.model.inner
    assert isinstance(inner.H_init, torch.nn.Parameter) and isinstance(inner.L_init, torch.nn.Parameter)
    assert inner.L_init.grad is not None and inner.L_init.grad.abs().sum() > 0
    assert inner.H_init.grad is not None
    sd = head.state_dict()
    assert "model.inner.H_init" in sd and "model.inner.L_init" in sd


def test_trm_full_grad_and_per_cycle_flags():
    _, tr, *_ = _one_forward("trm")
    assert [c["grad_enabled"] for c in tr.calls] == [False] * (2 * (L + 1)) + [True] * (L + 1)
    _, trf, *_ = _one_forward("trm", full_grad_cycles=True)
    assert all(c["grad_enabled"] for c in trf.calls)
    assert trf.calls[L + 1]["hs_requires_grad"]            # gradient flows across the cycle boundary
    head, trp, x, c, out = _one_forward("trm", per_cycle_loss=True)
    assert all(cc["grad_enabled"] for cc in trp.calls)
    assert not trp.calls[L + 1]["hs_requires_grad"] and not trp.calls[2 * (L + 1)]["hs_requires_grad"]  # detached
    assert len(out) == 4 and len(out[3]) == H - 1 and out[3][0].shape == (B, S_TOK, 11)


def test_per_cycle_loss_adds_cycle_terms_to_loss():
    head, loss, metrics = _full_step("trm", per_cycle_loss=True)
    expect = metrics["lm_loss"] + 0.5 * metrics["q_halt_loss"] + metrics["cycle_lm_loss"]
    assert torch.allclose(loss.detach().double(), expect.double(), rtol=1e-5)
    assert metrics["cycle_lm_loss"] > 0
    _, _, m0 = _full_step("trm")
    assert "cycle_lm_loss" not in m0


# ---------------------------------------------------------------- 2b. singlez structure
def test_singlez_default_trace_shape():
    _, tr, x, c, _ = _one_forward("trm_singlez")
    assert len(tr.calls) == N_CALLS
    assert torch.equal(tr.calls[0]["hs"], c.z_L + x)
    assert torch.equal(tr.calls[L]["hs"], tr.calls[L - 1]["out"])  # 7th step: no x


def test_singlez_inject_last_adds_x_at_7th_step():
    _, tr, x, c, _ = _one_forward("trm_singlez", inject_last=True)
    assert len(tr.calls) == N_CALLS
    for h in range(H):
        i = h * (L + 1) + L
        assert torch.equal(tr.calls[i]["hs"], tr.calls[i - 1]["out"] + x)
        assert torch.equal(tr.calls[i - 1]["hs"], tr.calls[i - 2]["out"] + x) if L > 1 else True


def test_singlez_step_token_only_at_7th_step_and_stripped():
    head, tr, x, c, out = _one_forward("trm_singlez", inject_last=True, step_token=True)
    inner = head.model.inner
    assert len(tr.calls) == N_CALLS
    for i, call in enumerate(tr.calls):
        if i % (L + 1) == L:
            assert call["hs"].shape == (B, S + 1, D) and call["out"].shape == (B, S + 1, D)
            tok = inner.step_tok.to(call["hs"].dtype)
            assert torch.equal(call["hs"][:, 0], tok.expand(B, D))               # token prepended at position 0
            assert torch.equal(call["hs"][:, 1:], tr.calls[i - 1]["out"] + x)    # rest = z + x (SINGLE_FULL)
            if i + 1 < len(tr.calls):                                            # stripped: next step sees length S
                assert torch.equal(tr.calls[i + 1]["hs"], call["out"][:, 1:] + x)
        else:
            assert call["hs"].shape == (B, S, D)
    new_carry, logits, _q = out
    assert new_carry.z_L.shape == (B, S, D) and logits.shape == (B, S_TOK, 11)
    head.zero_grad()
    # token is trainable
    batch = _batch()
    c2 = head.initial_carry(batch)
    _, loss, *_ = head(return_keys=[], carry=c2, batch=batch)
    loss.backward()
    assert inner.step_tok.grad is not None and inner.step_tok.grad.abs().sum() > 0


def test_singlez_step_token_off_has_no_extra_state():
    head, *_ = _one_forward("trm_singlez")
    assert not hasattr(head.model.inner, "step_tok")


def test_singlez_boundary_copy_reinjects_cycle_start_state():
    _, tr, x, c, _ = _one_forward("trm_singlez", boundary_copy=True)
    for h in range(H):
        base = h * (L + 1)
        bnd = c.z_L if h == 0 else tr.calls[base - 1]["out"]   # state at the cycle boundary
        inj = x + bnd
        assert torch.equal(tr.calls[base]["hs"], bnd + inj)
        for s in range(1, L):
            assert torch.equal(tr.calls[base + s]["hs"], tr.calls[base + s - 1]["out"] + inj)
        assert torch.equal(tr.calls[base + L]["hs"], tr.calls[base + L - 1]["out"])  # 7th step untouched


def test_singlez_x_copy_control_reinjects_second_x():
    _, tr, x, c, _ = _one_forward("trm_singlez", x_copy=True)
    for h in range(H):
        base = h * (L + 1)
        z0 = c.z_L if h == 0 else tr.calls[base - 1]["out"]
        assert torch.equal(tr.calls[base]["hs"], z0 + (x + x))
        assert torch.equal(tr.calls[base + L]["hs"], tr.calls[base + L - 1]["out"])


def test_singlez_learnable_init_and_cycle_flags():
    head, loss, _ = _full_step("trm_singlez", learnable_init=True)
    inner = head.model.inner
    assert isinstance(inner.L_init, torch.nn.Parameter)
    assert inner.L_init.grad is None  # FINDING: under the default no_grad truncation the init gets no gradient
    assert "model.inner.L_init" in head.state_dict()
    head, loss, _ = _full_step("trm_singlez", learnable_init=True, full_grad_cycles=True)
    assert head.model.inner.L_init.grad is not None and head.model.inner.L_init.grad.abs().sum() > 0
    def _links(tr):  # is cycle-1's first input differentiable w.r.t. cycle-0's last output?
        g = torch.autograd.grad(tr.calls[L + 1]["hs"].float().sum(), tr.calls[L]["out"], allow_unused=True)[0]
        return g is not None
    _, tr, *_ = _one_forward("trm_singlez", full_grad_cycles=True)
    assert all(c["grad_enabled"] for c in tr.calls) and _links(tr)
    head, trp, x, c, out = _one_forward("trm_singlez", per_cycle_loss=True)
    assert all(cc["grad_enabled"] for cc in trp.calls)
    assert not _links(trp)  # state detached between cycles
    assert len(out) == 4 and len(out[3]) == H - 1
    _, loss, metrics = _full_step("trm_singlez", per_cycle_loss=True)
    expect = metrics["lm_loss"] + 0.5 * metrics["q_halt_loss"] + metrics["cycle_lm_loss"]
    assert torch.allclose(loss.detach().double(), expect.double(), rtol=1e-5)


def test_invalid_flag_combinations_raise():
    from models import losses as new_losses
    with pytest.raises(ValueError):
        _build(_new_mod("trm_singlez"), new_losses, _cfg(boundary_copy=True, x_copy=True))
    with pytest.raises(ValueError):
        _build(_new_mod("trm"), new_losses, _cfg(zH_every_call=True, zH_freeze=True))
    with pytest.raises(ValueError):
        _build(_new_mod("trm_singlez"), new_losses, _cfg(step_token=True, mlp_t=True))


# ---------------------------------------------------------------- G4 CAP: config only
def test_cap_hidden_size_is_config_only():
    head, loss, _ = _full_step("trm_singlez", hidden_size=64, puzzle_emb_ndim=64, num_heads=4)
    assert head.model.inner.L_init.shape == (64,)


# ---------------------------------------------------------------- yaml defaults
def test_yaml_flag_defaults_are_off_and_match_config_fields():
    yaml = pytest.importorskip("yaml")
    for arch, flags in (("trm", TRM_FLAGS), ("trm_singlez", SZ_FLAGS)):
        cfg = yaml.safe_load((REPO / "config" / "arch" / f"{arch}.yaml").read_text())
        fields = _new_mod(arch).TinyRecursiveReasoningModel_ACTV1Config.model_fields
        for f in flags:
            assert cfg[f] is False, (arch, f)
            assert f in fields and fields[f].default is False
        assert cfg["hidden_size"] == 512 and cfg["H_cycles"] == 3 and cfg["L_cycles"] == 6  # existing values kept
