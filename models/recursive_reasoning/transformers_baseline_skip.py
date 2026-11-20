"""Looped transformer with a TRM-style input-injection SCHEDULE (R3 / g7s arm).

Subclasses the FROZEN `transformers_baseline` module (which must stay byte-identical:
analysis/ltf_decode/gates.py pins its sha256).  The only behavioural difference is
the input-injection schedule inside the weight-tied loop:

  `loop_input_injection_skip_period` p (0 = off, identical to the base model;
  allowed 0 or >= 2; must divide `loops`; requires `loop_input_injection=True`).
  For p > 0 the input embedding is NOT added at loops i with i % p == p - 1
  (p = 7, loops = 21 -> loops 6, 13, 20 skipped, 18 injected).

Parameters, state_dict keys and shapes equal the base model's.  Arch target:
`recursive_reasoning.transformers_baseline_skip@Model_ACTV2`.
"""
from typing import Dict, Tuple

import torch
from torch import nn
from pydantic import model_validator

from models.recursive_reasoning.transformers_baseline import (
    Model_ACTV2Config as _BaseConfig,
    Model_ACTV2_Inner as _BaseInner,
    Model_ACTV2 as _BaseModel,
    Model_ACTV2InnerCarry,
)


class Model_ACTV2Config(_BaseConfig):
    loop_input_injection_skip_period: int = 0

    @model_validator(mode="after")
    def _check_skip_period(self):
        p = self.loop_input_injection_skip_period
        if p < 0:
            raise ValueError(f"loop_input_injection_skip_period must be 0 or >= 2, got {p}")
        if p == 1:
            raise ValueError(
                "loop_input_injection_skip_period=1 skips the input at every loop (no injection at all); "
                "allowed values are 0 (off) or >= 2"
            )
        if p > 1:
            if not self.loop_input_injection:
                raise ValueError("loop_input_injection_skip_period > 0 requires loop_input_injection=True")
            if self.loops % p != 0:
                raise ValueError(
                    f"loops must be divisible by loop_input_injection_skip_period, "
                    f"got loops={self.loops}, loop_input_injection_skip_period={p}"
                )
        return self


class Model_ACTV2_Inner(_BaseInner):
    def _inject(self, i: int, input_embeddings: torch.Tensor):
        """Base rule (loop 0 always, later loops iff loop_input_injection), minus skipped loops."""
        injection = input_embeddings if (i == 0 or self.config.loop_input_injection) else None
        p = self.config.loop_input_injection_skip_period
        if p > 0 and i % p == p - 1:
            return None
        return injection

    # Copy of _BaseInner.forward; ONLY the two `injection = ...` lines differ.
    def forward(
        self, carry: Model_ACTV2InnerCarry, batch: Dict[str, torch.Tensor]
    ) -> Tuple[Model_ACTV2InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        loops = self.config.loops
        n_nograd = loops - self.config.loop_grad_cycles if self.config.loop_grad_cycles > 0 else 0
        z_H = carry.z_H
        if n_nograd > 0:
            with torch.no_grad():
                for _i in range(n_nograd):
                    injection = self._inject(_i, input_embeddings)
                    z_H = self.H_level(z_H, injection, **seq_info)
        for _i in range(n_nograd, loops):
            injection = self._inject(_i, input_embeddings)
            z_H = self.H_level(z_H, injection, **seq_info)

        # LM Outputs
        new_carry = Model_ACTV2InnerCarry(
            z_H=z_H.detach(),
        )  # New carry no grad
        output = self.lm_head(z_H)[:, self.puzzle_emb_len :]

        # Q head
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)

        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class Model_ACTV2(_BaseModel):
    """ACT wrapper; the base ACT forward is inherited unchanged."""

    def __init__(self, config_dict: dict):
        # Do NOT call _BaseModel.__init__: it would build a base inner (extra RNG draws, double memory).
        nn.Module.__init__(self)
        self.config = Model_ACTV2Config(**config_dict)
        self.inner = Model_ACTV2_Inner(self.config)
