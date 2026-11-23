"""P-C1 looped-TF per-loop tracer + loader (spec §4.1; analysis/ltf_decode, CPU only).

``trace_loops`` replays ``Model_ACTV2_Inner.forward`` (models/recursive_reasoning/transformers_baseline.py:
263-273) call for call:

    z = carry.z_H                                   # H_init-reset carry
    for l in range(n_loops):
        injection = x if (l == 0 or loop_input_injection) else None
        z = inner.H_level(z, injection, cos_sin=rotary)
        record lm_head(z)[:, puzzle_emb_len:]       # answer slice, all seq_len positions

The native forward splits the loops into a no_grad prefix and a grad suffix when loop_grad_cycles > 0;
under ``torch.inference_mode()`` that split is numerically irrelevant, so the tracer has no grad split.
n_loops may exceed ``config.loops`` (extrapolation t = 22..42): same weights, same injection rule.

``load_ltf`` mirrors analysis/z_dynamics/instrumented_forward.load_inner (model class -> ACTLossHead
(stablemax) -> torch.load(weights_only=True, map_location="cpu") -> strip ``_orig_mod.`` ->
load_state_dict(strict=False) -> eval) and RETURNS the missing / unexpected key lists (I-3).

Nothing in this module reads a checkpoint unless ``load_ltf`` is called with a path.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class LoopTrace:
    z: list = field(default_factory=list)        # n_loops x (B, S, D) (None entries if keep_latents=False)
    logits: list = field(default_factory=list)   # n_loops x (B, seq_len, V) answer slice after each loop
    n_loops: int = 0


def initial_carry(inner, batch_size: int):
    """Fresh eval carry: reset_carry(all True, empty_carry(B)) -> H_init (same as native eval)."""
    return inner.reset_carry(torch.ones(batch_size, dtype=torch.bool), inner.empty_carry(batch_size))


def trace_loops(inner, batch: dict, n_loops: int, keep_latents: bool = False) -> LoopTrace:
    """Per-loop replay of the looped-TF forward. Call under torch.inference_mode() with inner.eval()."""
    if n_loops < 1:
        raise ValueError(f"n_loops must be >= 1, got {n_loops}")
    cfg = inner.config
    B = batch["inputs"].shape[0]
    seq_info = dict(cos_sin=inner.rotary_emb() if hasattr(inner, "rotary_emb") else None)
    x = inner._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])
    z = initial_carry(inner, B).z_H
    pl = inner.puzzle_emb_len
    tr = LoopTrace(n_loops=n_loops)
    for l in range(n_loops):
        injection = x if (l == 0 or cfg.loop_input_injection) else None
        z = inner.H_level(z, injection, **seq_info)
        tr.z.append(z if keep_latents else None)
        tr.logits.append(inner.lm_head(z)[:, pl:])
    return tr


def native_forward(inner, batch: dict):
    """The model's own forward: inner(carry0, batch) -> (new carry z_H, logits). G1' / G-a reference."""
    B = batch["inputs"].shape[0]
    carry, out, _q = inner(initial_carry(inner, B), batch)
    return carry.z_H, out


def g1_equal(tr: LoopTrace, ref_logits: torch.Tensor, t: int | None = None) -> bool:
    """G1': tracer logits after loop t (default: last traced loop) torch.equal the native forward logits."""
    t = tr.n_loops if t is None else t
    mine = tr.logits[t - 1]
    return mine.dtype == ref_logits.dtype and torch.equal(mine, ref_logits)


# ------------------------------------------------------------------ loader

def _strip_orig_mod(sd: dict) -> dict:
    if any(k.startswith("_orig_mod.") for k in sd):
        return {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
    return sd


def build_ltf(config: dict, device: str = "cpu"):
    """Random-init Model_ACTV2 wrapped in ACTLossHead(stablemax). Returns (wrapped, inner)."""
    from models.losses import ACTLossHead
    from models.recursive_reasoning.transformers_baseline import Model_ACTV2

    model = Model_ACTV2(config).to(device)
    wrapped = ACTLossHead(model, loss_type="stablemax_cross_entropy").to(device)
    wrapped.eval()
    return wrapped, wrapped.model.inner


def load_ltf(checkpoint_path: str, config: dict, device: str = "cpu"):
    """Returns (wrapped, inner, missing_keys, unexpected_keys). Refuses non-CPU devices."""
    if device != "cpu":
        raise ValueError("P-C1 is CPU only (spec §4.1)")
    wrapped, inner = build_ltf(config, device)
    sd = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    sd = _strip_orig_mod(sd)
    missing, unexpected = wrapped.load_state_dict(sd, strict=False)
    wrapped.eval()
    return wrapped, inner, list(missing), list(unexpected)
