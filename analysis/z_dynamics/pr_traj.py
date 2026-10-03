"""EXP-017 per-call participation ratio (PR) trajectories.

``pr_of_latent`` is the same composition as ``analysis/pr_recompute.pr_of_latent`` (which lives only in
the gitignored main-checkout analysis/; VERIFY-r1 N9): bf16 round trip -> utils.z_logging._mean_pool_z
(mean over all 27 positions) -> _pca (covariance eigenvalues) -> _effective_rank ((Σλ)²/Σλ²).
tests/test_z_dynamics.py asserts equality with analysis.pr_recompute.pr_of_latent when that module is
importable.
"""

from __future__ import annotations

import torch

from utils.z_logging import _effective_rank, _mean_pool_z, _pca


def pr_of_latent(z: torch.Tensor | None) -> float | None:
    if z is None:
        return None
    z = z.to(torch.bfloat16).cpu()  # bf16 round trip, as z_logging snapshots store it
    X = _mean_pool_z(z)
    eig, _ = _pca(X)
    return _effective_rank(eig)


def pr_rows(unit: str, run_name: str, split: str, S_L: list, zHp: list, zHn: list, n_probe: int) -> list[dict]:
    """Rows for pr_traj.csv: stream in {z_L, zH_prelim, zH_native}; j for per-L-step streams, h for native."""
    rows = []
    for stream, seq, key in (("z_L", S_L, "j"), ("zH_prelim", zHp, "j"), ("zH_native", zHn, "h")):
        for t, z in enumerate(seq):
            rows.append({"exp_id": "EXP-017", "unit": unit, "run_name": run_name, "split": split, "stream": stream,
                         "j": t + 1 if key == "j" else "", "h": t + 1 if key == "h" else "",
                         "pr": pr_of_latent(z), "n_probe": n_probe})
    return rows
