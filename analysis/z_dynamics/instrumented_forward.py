"""EXP-017 §4.3 instrumented re-execution of the TRM forward loop, plus the checkpoint loader.

forward_trace(inner, batch, arch) replays exactly the native call sequence of
``TinyRecursiveReasoningModel_ACTV1_Inner.forward`` (trm.py / trm_singlez.py) and additionally
evaluates one analysis-only preliminary L_level call per L-step:

  trm:          zL = L_level(zL, zH + x)          native call c = 7(h-1)+l ; S_L[j] = zL
                zHp[j] = L_level(zH, zL)          preliminary ẑ_H(j) (analysis only)
                zH = L_level(zH, zL)              native call c = 7h ; zHn[h] = zH
  trm_singlez:  z = L_level(z + x)                native call c = 7(h-1)+l ; S_L[j] = z
                zHp[j] = L_level(z)               preliminary = free step without injection
                z = L_level(z)                    native call c = 7h ; zHn[h] = z

Argument structure is identical to the native forward (``L_level(zL, zH + x)`` lets the module add
the injection internally; ``L_level(z + x)`` for singlez), so the native states are bitwise the
states the model computes. G1 checks the final logits with torch.equal against ``inner(carry0,
batch)``; G2 checks zHp[6h] == zHn[h].

Loader: ``load_inner`` mirrors ``analysis/pr_recompute.load_wrapped`` step for step (model class ->
ACTLossHead(stablemax) -> torch.load(weights_only) -> strip ``_orig_mod.`` -> load_state_dict(strict=False)
-> eval) but RETURNS the missing / unexpected key lists, which load_wrapped only prints; I-3 needs them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

ARCH_TRM = "trm"
ARCH_SINGLEZ = "trm_singlez"


@dataclass
class Trace:
    S_L: list[torch.Tensor] = field(default_factory=list)     # 18 × (B, S, D)  z_L^{(j)}
    zHp: list[torch.Tensor] = field(default_factory=list)     # 18 × (B, S, D)  ẑ_H(j)
    zHn: list[torch.Tensor] = field(default_factory=list)     # 3 × (B, S, D)   native z_H after c = 7h
    logits: list[torch.Tensor] = field(default_factory=list)  # 18 × (B, seq_len, V)  lm_head(ẑ_H(j)) answer slice
    raw_logits: list[torch.Tensor] = field(default_factory=list)  # 18 × lm_head(S_L[j])  (D-2)
    native_logits: list[torch.Tensor] = field(default_factory=list)  # 3 × lm_head(zHn[h])
    final_logits: torch.Tensor | None = None                  # lm_head(zHn[-1]) — the model's output
    j_to_c: list[int] = field(default_factory=list)           # native call index of L-step j


def initial_carry(inner, batch_size: int):
    """Fresh eval carry: reset_carry(all True, empty_carry(B)) -> H_init / L_init (EXP-017 §4.2)."""
    return inner.reset_carry(torch.ones(batch_size, dtype=torch.bool), inner.empty_carry(batch_size))


def arch_of_inner(inner) -> str:
    return ARCH_TRM if hasattr(inner, "H_init") else ARCH_SINGLEZ


def forward_trace(inner, batch: dict, arch: str | None = None, keep_latents: bool = True) -> Trace:
    """Instrumented forward per EXP-017 §4.3. Call under torch.inference_mode() with inner.eval()."""
    arch = arch_of_inner(inner) if arch is None else arch
    cfg = inner.config
    B = batch["inputs"].shape[0]
    seq = dict(cos_sin=inner.rotary_emb() if hasattr(inner, "rotary_emb") else None)
    x = inner._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])
    carry = initial_carry(inner, B)
    pl = inner.puzzle_emb_len
    head = lambda z: inner.lm_head(z)[:, pl:]  # noqa: E731
    tr = Trace()
    if arch == ARCH_TRM:
        zH, zL = carry.z_H, carry.z_L
        for h in range(cfg.H_cycles):
            for l in range(cfg.L_cycles):
                zL = inner.L_level(zL, zH + x, **seq)
                zp = inner.L_level(zH, zL, **seq)
                _record(tr, zL, zp, head, keep_latents)
                tr.j_to_c.append((cfg.L_cycles + 1) * h + l + 1)
            zH = inner.L_level(zH, zL, **seq)
            tr.zHn.append(zH if keep_latents else None)
            tr.native_logits.append(head(zH))
        tr.final_logits = head(zH)
    elif arch == ARCH_SINGLEZ:
        z = carry.z_L
        for h in range(cfg.H_cycles):
            for l in range(cfg.L_cycles):
                z = inner.L_level(z + x, **seq)
                zp = inner.L_level(z, **seq)
                _record(tr, z, zp, head, keep_latents)
                tr.j_to_c.append((cfg.L_cycles + 1) * h + l + 1)
            z = inner.L_level(z, **seq)
            tr.zHn.append(z if keep_latents else None)
            tr.native_logits.append(head(z))
        tr.final_logits = head(z)
    else:
        raise ValueError(f"unknown arch {arch!r}")
    return tr


def _record(tr: Trace, zl, zp, head, keep):
    tr.S_L.append(zl if keep else None)
    tr.zHp.append(zp if keep else None)
    tr.logits.append(head(zp))
    tr.raw_logits.append(head(zl))


def native_logits(inner, batch: dict) -> torch.Tensor:
    """The model's own forward: inner(carry0, batch) logits (G1 reference)."""
    B = batch["inputs"].shape[0]
    _carry, out, _q = inner(initial_carry(inner, B), batch)
    return out


def g1_equal(tr: Trace, ref: torch.Tensor) -> bool:
    return tr.final_logits is not None and tr.final_logits.dtype == ref.dtype and torch.equal(tr.final_logits, ref)


def g2_equal(tr: Trace, L_cycles: int) -> bool:
    ok = True
    for h, zn in enumerate(tr.zHn):
        j = L_cycles * (h + 1) - 1  # 0-based index of j = 6(h+1)
        ok &= torch.equal(tr.zHp[j], zn)
    return bool(ok)


# ------------------------------------------------------------------ loader

def load_inner(checkpoint_path: str, config: dict, device: str, arch: str):
    """Returns (wrapped ACTLossHead model, inner, missing_keys, unexpected_keys)."""
    from measure_rho import _strip_orig_mod
    from models.losses import ACTLossHead
    from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1 as TRM
    from models.recursive_reasoning.trm_singlez import TinyRecursiveReasoningModel_ACTV1 as TRM_SZ

    cls = TRM if arch == ARCH_TRM else TRM_SZ
    model = cls(config).to(device)
    wrapped = ACTLossHead(model, loss_type="stablemax_cross_entropy").to(device)
    sd = torch.load(checkpoint_path, map_location=device, weights_only=True)
    sd = _strip_orig_mod(sd)
    missing, unexpected = wrapped.load_state_dict(sd, strict=False)
    wrapped.eval()
    return wrapped, wrapped.model.inner, list(missing), list(unexpected)
