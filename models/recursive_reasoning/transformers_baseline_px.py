"""Looped transformer with a configurable PREFIX length (`puzzle_emb_len`) on top of the
injection-skip subclass (R6 / "comb" arm).

Subclasses `transformers_baseline_skip` (whose sha256 is pinned and which in turn wraps the
frozen `transformers_baseline`); neither file is touched.  This module adds ONE config field:

  `puzzle_emb_len` (int, default 1 = what the frozen looped model uses today for
  puzzle_emb_ndim == hidden_size: ceil(puzzle_emb_ndim / hidden_size)).

For a larger value the model is built exactly the way `trm_singlez` builds its prefix:
  * the learned puzzle embedding (puzzle_emb_ndim wide) fills the first prefix position;
    the remaining prefix positions are zero padding (F.pad in the inherited
    `_input_embeddings`), then everything is scaled by sqrt(hidden_size);
  * RoPE covers seq_len + puzzle_emb_len positions (the prefix is positioned, not skipped);
  * the carry, `empty_carry`, and the output slice `[:, puzzle_emb_len:]` follow
    `self.puzzle_emb_len`, which the inherited code already reads dynamically.
The only code here is therefore the config field + validation and re-creating the
(parameter-free, RNG-free) RotaryEmbedding for the longer sequence.  Parameters, state_dict
keys and shapes equal the base model's; same-seed initial parameters are unchanged.

Constraints: puzzle_emb_len >= 1 (ValueError otherwise); if puzzle_emb_ndim == 0 there is no
prefix at all, so only the default 1 is accepted (and ignored); puzzle_emb_len must be >=
ceil(puzzle_emb_ndim / hidden_size); a non-default prefix requires pos_encodings == "rope"
(the "learned" table is sized in the base __init__ and is not rebuilt here).

Arch target: `recursive_reasoning.transformers_baseline_px@Model_ACTV2`.
"""
import torch
from torch import nn
from pydantic import model_validator

from models.layers import RotaryEmbedding
from models.recursive_reasoning.transformers_baseline_skip import (
    Model_ACTV2Config as _SkipConfig,
    Model_ACTV2_Inner as _SkipInner,
    Model_ACTV2 as _SkipModel,
)

DEFAULT_PUZZLE_EMB_LEN = 1


class Model_ACTV2Config(_SkipConfig):
    puzzle_emb_len: int = DEFAULT_PUZZLE_EMB_LEN

    @model_validator(mode="after")
    def _check_puzzle_emb_len(self):
        n = self.puzzle_emb_len
        if n < 1:
            raise ValueError(f"puzzle_emb_len must be >= 1, got {n}")
        if self.puzzle_emb_ndim <= 0:
            if n != DEFAULT_PUZZLE_EMB_LEN:
                raise ValueError(
                    f"puzzle_emb_len={n} needs puzzle_emb_ndim > 0 (no puzzle embedding -> no prefix)"
                )
            return self
        derived = -(self.puzzle_emb_ndim // -self.hidden_size)  # ceil div, as the base model
        if n < derived:
            raise ValueError(
                f"puzzle_emb_len={n} is smaller than ceil(puzzle_emb_ndim/hidden_size)={derived}"
            )
        if n != derived and self.pos_encodings != "rope":
            raise ValueError("a prefix length other than the derived one requires pos_encodings='rope'")
        return self


class Model_ACTV2_Inner(_SkipInner):
    def __init__(self, config: Model_ACTV2Config) -> None:
        super().__init__(config)
        if config.puzzle_emb_ndim > 0 and config.puzzle_emb_len != self.puzzle_emb_len:
            self.puzzle_emb_len = config.puzzle_emb_len
            # RotaryEmbedding holds no parameters and draws no random numbers, so re-creating it
            # leaves the RNG stream (and hence same-seed initial parameters) unchanged.
            self.rotary_emb = RotaryEmbedding(
                dim=config.hidden_size // config.num_heads,
                max_position_embeddings=config.seq_len + self.puzzle_emb_len,
                base=config.rope_theta,
            )


class Model_ACTV2(_SkipModel):
    """ACT wrapper; forward/initial_carry are inherited unchanged."""

    def __init__(self, config_dict: dict):
        # As the skip wrapper: do not call a parent __init__ that would build a second inner.
        nn.Module.__init__(self)
        self.config = Model_ACTV2Config(**config_dict)
        self.inner = Model_ACTV2_Inner(self.config)
