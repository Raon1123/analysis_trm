"""EXP-017 §4.4 tuned lens (D-7; secondary, interpretive only; never feeds judge.py).

For each readout j <= 17 we fit an affine translator h' = ẑ + ẑ W_jᵀ + b_j (W_j, b_j initialised at 0)
at the 10 answer positions. The objective is the mean KL(softmax(lm_head(ẑ_H(18))) || softmax(lm_head(h'))),
using the float32 lm_head weight. Settings: the first 2048 train rows; AdamW lr 1e-3, weight decay 0, 300
full-batch steps, seed 20260923. These §4.4 settings are interpretive-only and are not in FROZEN_RULES,
because the lens classes never change a verdict.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

LENS_LR = 1e-3            # §4.4
LENS_WEIGHT_DECAY = 0.0   # §4.4
LENS_STEPS = 300          # §4.4
LENS_SEED = 20260923      # §4.4


def fit_lens(ans: torch.Tensor, lm_w: torch.Tensor, steps: int | None = None, seed: int = LENS_SEED) -> list:
    """ans: (N, 18, 10, D) preliminary states at the answer positions (any float dtype).
    lm_w: (V, D) lm_head weight. Returns [(W_j, b_j)] for j = 1..17 (float32, detached)."""
    steps = LENS_STEPS if steps is None else steps
    lm_w = lm_w.detach().float().clone()
    with torch.inference_mode(False):
        target = F.log_softmax(ans[:, -1].float().clone() @ lm_w.T, dim=-1)  # (N, 10, V)
        p_t = target.exp()
        out = []
        D = ans.shape[-1]
        for j in range(ans.shape[1] - 1):
            torch.manual_seed(seed)
            z = ans[:, j].float().clone()                                     # (N, 10, D)
            W = torch.zeros(D, D, requires_grad=True)
            b = torch.zeros(D, requires_grad=True)
            opt = torch.optim.AdamW([W, b], lr=LENS_LR, weight_decay=LENS_WEIGHT_DECAY)
            for _ in range(steps):
                opt.zero_grad(set_to_none=True)
                h = z + z @ W.T + b
                logq = F.log_softmax(h @ lm_w.T, dim=-1)
                loss = (p_t * (target - logq)).sum(-1).mean()
                loss.backward()
                opt.step()
            out.append((W.detach().clone(), b.detach().clone()))
    return out


def apply_lens(ans: torch.Tensor, lens: list, lm_w: torch.Tensor) -> torch.Tensor:
    """Lens decodes for j = 1..17 -> (N, 17, 10) int64 tokens (argmax over the vocabulary)."""
    lm_w = lm_w.detach().float()
    decs = []
    with torch.inference_mode():
        for j, (W, b) in enumerate(lens):
            z = ans[:, j].float()
            decs.append(((z + z @ W.T + b) @ lm_w.T).argmax(-1))
    return torch.stack(decs, 1).to(torch.int64)
