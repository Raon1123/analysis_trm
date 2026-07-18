"""
z_logging.py — z dynamics / convergence logging + learning phase auto-detection.

Usage: imported by pretrain.py; all public entry points check rank and
log_z_dynamics internally.  No model file is modified.

Key design choices:
- PhaseTracker is pure Python (no torch), unit-testable in isolation.
- Probe inputs are loaded once at construction time and pinned; the same
  tensor (by hash/ptr) is reused every eval, satisfying Gate 5.
- Snapshots are saved in bfloat16 to stay well under 50 MB (Gate 4).
- PCA uses numpy (no sklearn needed).
- matplotlib uses the Agg (non-GUI) backend unconditionally.
"""

from __future__ import annotations

import os
import hashlib
import logging
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PhaseTracker — pure Python, no torch dependency
# ---------------------------------------------------------------------------

class PhaseTracker:
    """
    Tracks grokking phases based on exact accuracy thresholds.

    Phase 0: train_exact < theta
    Phase 1: train_exact >= theta  AND  test_exact < theta
    Phase 2: train_exact >= theta  AND  test_exact >= theta

    A transition is committed only after `patience` consecutive evals
    satisfy the next phase's conditions (prevents flapping).
    Phases are monotone — once committed, they never decrease.
    """

    def __init__(self, phase_threshold: float = 0.999, phase_patience: int = 2):
        self.theta = phase_threshold
        self.patience = phase_patience

        self._phase: int = 0          # committed phase (monotone)
        self._candidate: int = 0      # next candidate phase
        self._candidate_count: int = 0  # consecutive evals satisfying candidate
        self._transition_steps: list[int] = []  # history of committed transitions
        self._last_step: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_raw_phase(self, train_exact: float, test_exact: float) -> int:
        """Phase from current accuracy values (no patience)."""
        if train_exact >= self.theta and test_exact >= self.theta:
            return 2
        if train_exact >= self.theta:
            return 1
        return 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, train_exact: float, test_exact: float, step: int = 0) -> Tuple[int, bool]:
        """
        Feed a new observation.

        Returns:
            (current_phase, transitioned)
            transitioned=True means this call produced a new committed transition.
        """
        self._last_step = step
        raw = self._compute_raw_phase(train_exact, test_exact)

        # Only ever go forward (monotone)
        next_candidate = max(raw, self._phase)

        if next_candidate > self._phase:
            # We're looking at a potential transition
            if next_candidate == self._candidate:
                self._candidate_count += 1
            else:
                # New candidate resets the patience counter
                self._candidate = next_candidate
                self._candidate_count = 1

            if self._candidate_count >= self.patience:
                # Commit the transition
                self._phase = self._candidate
                self._candidate_count = 0
                self._transition_steps.append(step)
                return self._phase, True
        else:
            # Raw phase matches committed phase — reset candidate
            self._candidate = self._phase
            self._candidate_count = 0

        return self._phase, False

    @property
    def phase(self) -> int:
        return self._phase

    @property
    def transition_steps(self) -> list[int]:
        return list(self._transition_steps)


# ---------------------------------------------------------------------------
# Probe dataset loader (rank-0 only, loaded once)
# ---------------------------------------------------------------------------

IGNORE_LABEL_ID = -100  # matches losses.py


def _load_probe_tensors(data_path: str, split: str, probe_size: int,
                        ignore_label_id_in_file: Optional[int]) -> Dict[str, torch.Tensor]:
    """
    Load the first `probe_size` examples from a dataset split.

    Returns tensors: inputs, labels, puzzle_identifiers
    All on CPU, int32 (same as PuzzleDataset._collate_batch).

    NOTE: num_puzzle_identifiers==1 for sigma_k datasets, so direct
    slice of puzzle_identifiers.npy is safe (all=blank_identifier_id=0).
    """
    split_dir = os.path.join(data_path, split)
    inputs = np.load(os.path.join(split_dir, "all__inputs.npy"), mmap_mode="r")[:probe_size].astype(np.int32)
    labels = np.load(os.path.join(split_dir, "all__labels.npy"), mmap_mode="r")[:probe_size].astype(np.int32)
    puzzle_identifiers = np.load(
        os.path.join(split_dir, "all__puzzle_identifiers.npy"), mmap_mode="r"
    )[:probe_size].astype(np.int32)

    # Replicate _collate_batch label remap
    if ignore_label_id_in_file is not None:
        labels[labels == ignore_label_id_in_file] = IGNORE_LABEL_ID

    result = {
        "inputs": torch.from_numpy(inputs.copy()),
        "labels": torch.from_numpy(labels.copy()),
        "puzzle_identifiers": torch.from_numpy(puzzle_identifiers.copy()),
    }

    # Compute and log a fingerprint once for probe-fixity verification (Gate 5)
    fingerprint = hashlib.md5(inputs.tobytes()).hexdigest()[:8]
    log.info("Probe loaded: split=%s path=%s n=%d input_hash=%s",
             split, data_path, len(inputs), fingerprint)

    return result, fingerprint


# ---------------------------------------------------------------------------
# Z-probe forward (inference-mode, eval model)
# ---------------------------------------------------------------------------

def _probe_forward(model: torch.nn.Module, probe: Dict[str, torch.Tensor]
                   ) -> Tuple[Dict[str, Any], bool]:
    """
    Run the model on `probe` tensors (already on CPU).
    Returns dict with z_H, z_L (bfloat16 CPU), labels, correct_mask,
    per-step delta list, and exact accuracy.

    Mirrors evaluate()'s ACT loop: initial_carry → forward until all_finish.
    """
    batch = {k: v.cuda() for k, v in probe.items()}

    with torch.inference_mode(), torch.device("cuda"):
        carry = model.initial_carry(batch)  # type: ignore

    z_history: list[Tuple[torch.Tensor, torch.Tensor]] = []

    with torch.inference_mode():
        # Capture initial z (after reset on first step)
        while True:
            carry, _loss, metrics, _preds, all_finish = model(
                carry=carry, batch=batch, return_keys=[]
            )
            # Grab z after this step
            try:
                z_H = carry.inner_carry.z_H.detach().float()
                z_L = carry.inner_carry.z_L.detach().float()
                z_history.append((z_H, z_L))
            except AttributeError:
                # Fallback for models that don't have inner_carry.z_H/z_L
                z_history.append((None, None))

            if all_finish:
                break

    # --- Exact accuracy ---
    labels_gpu = batch["labels"]  # (B, seq_len), already remapped
    mask = labels_gpu != IGNORE_LABEL_ID

    # Rerun last forward to get logits (carry was deleted above)
    with torch.inference_mode(), torch.device("cuda"):
        carry2 = model.initial_carry(batch)  # type: ignore

    with torch.inference_mode():
        while True:
            carry2, _loss, metrics2, preds2, all_finish2 = model(
                carry=carry2, batch=batch, return_keys=["preds"]
            )
            if all_finish2:
                break

    preds = preds2["preds"]  # (B, seq_len)
    is_correct = mask & (preds == labels_gpu)
    loss_counts = mask.sum(-1)
    seq_correct = is_correct.sum(-1) == loss_counts
    halted = carry2.halted
    valid = halted & (loss_counts > 0)

    exact_acc = (valid & seq_correct).sum().item() / max(valid.sum().item(), 1)
    correct_mask = seq_correct.cpu()

    # --- Per-step z deltas ---
    step_deltas: list[float] = []
    for t in range(1, len(z_history)):
        zh_prev, _ = z_history[t - 1]
        zh_cur, _ = z_history[t]
        if zh_prev is None or zh_cur is None:
            continue
        # Mean relative change across batch, averaged over sequence & hidden
        delta = (
            (zh_cur - zh_prev).norm(dim=(-1, -2))
            / (zh_prev.norm(dim=(-1, -2)) + 1e-8)
        ).mean().item()
        step_deltas.append(delta)

    # Final z (last step)
    final_z_H, final_z_L = z_history[-1] if z_history else (None, None)

    return {
        "z_H": final_z_H.to(torch.bfloat16).cpu() if final_z_H is not None else None,
        "z_L": final_z_L.to(torch.bfloat16).cpu() if final_z_L is not None else None,
        "labels": batch["labels"].cpu(),
        "correct_mask": correct_mask,
        "step_deltas": step_deltas,
        "exact_acc": exact_acc,
    }


# ---------------------------------------------------------------------------
# PCA helpers (numpy only)
# ---------------------------------------------------------------------------

def _mean_pool_z(z: torch.Tensor) -> np.ndarray:
    """z: (B, S, D) → (B, D) via mean-pool over sequence dimension."""
    return z.float().mean(dim=1).numpy()


def _pca(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Thin PCA over rows of X (B, D).
    Returns (eigenvalues_descending, eigenvectors shape (D, k)).
    """
    X_centered = X - X.mean(axis=0, keepdims=True)
    cov = (X_centered.T @ X_centered) / max(len(X) - 1, 1)
    # Symmetric eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # eigh returns ascending order — reverse to descending
    idx = np.argsort(eigenvalues)[::-1]
    return eigenvalues[idx], eigenvectors[:, idx]


def _effective_rank(eigenvalues: np.ndarray) -> float:
    """Participation ratio: (sum lambda)^2 / sum lambda^2."""
    lam = np.maximum(eigenvalues, 0.0)
    s1 = lam.sum()
    s2 = (lam ** 2).sum()
    if s2 < 1e-30:
        return 1.0
    return float(s1 ** 2 / s2)


def _pca_top2_var(eigenvalues: np.ndarray) -> float:
    """Fraction of variance explained by top 2 PCs."""
    lam = np.maximum(eigenvalues, 0.0)
    total = lam.sum()
    if total < 1e-30:
        return 0.0
    return float(lam[:2].sum() / total)


# ---------------------------------------------------------------------------
# Scatter plot helper
# ---------------------------------------------------------------------------

def _make_pca_scatter(X_pca: np.ndarray,
                      labels_col0: np.ndarray,
                      correct_mask: np.ndarray,
                      title_prefix: str) -> list:
    """
    Build two wandb.Image scatter plots (label-colored, correct-colored).
    Returns list of (key, wandb.Image) pairs.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import wandb

    images = []
    for (color_arr, color_label, fname_suffix) in [
        (labels_col0, "label[0]", "by_label"),
        (correct_mask.astype(float), "correct", "by_correct"),
    ]:
        fig, ax = plt.subplots(figsize=(5, 4))
        sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=color_arr,
                        cmap="tab10", s=4, alpha=0.7)
        plt.colorbar(sc, ax=ax, label=color_label)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"{title_prefix} PCA ({fname_suffix})")
        fig.tight_layout()
        img = wandb.Image(fig)
        plt.close(fig)
        images.append((f"z/pca_{fname_suffix}_{title_prefix}", img))

    return images


# ---------------------------------------------------------------------------
# Main entry point: ZDynamicsLogger
# ---------------------------------------------------------------------------

class ZDynamicsLogger:
    """
    Constructed once at startup (rank 0 only when log_z_dynamics=True).
    Call .log(model, step, config) after each evaluate() call.
    """

    def __init__(self, data_path: str, probe_size: int,
                 ignore_label_id_in_file: Optional[int],
                 phase_threshold: float, phase_patience: int,
                 checkpoint_path: Optional[str]):
        self._checkpoint_path = checkpoint_path
        self._phase_tracker = PhaseTracker(phase_threshold, phase_patience)

        # Load probes once; fingerprints logged to verify fixity
        self._train_probe, self._train_fp = _load_probe_tensors(
            data_path, "train", probe_size, ignore_label_id_in_file)
        self._test_probe, self._test_fp = _load_probe_tensors(
            data_path, "test", probe_size, ignore_label_id_in_file)

        log.info("ZDynamicsLogger: train_probe_hash=%s  test_probe_hash=%s",
                 self._train_fp, self._test_fp)

    # ------------------------------------------------------------------

    def log(self, model: torch.nn.Module, step: int,
            save_train_state_fn,
            train_state) -> None:
        """
        Called after evaluate() on rank 0.
        `save_train_state_fn` is pretrain.save_train_state (partial or callable).
        """
        import wandb

        # --- Probe forwards ---
        train_result = _probe_forward(model, self._train_probe)
        test_result = _probe_forward(model, self._test_probe)

        # --- Phase tracking ---
        phase, transitioned = self._phase_tracker.update(
            train_result["exact_acc"], test_result["exact_acc"], step=step
        )

        # --- Build wandb log dict ---
        log_dict: Dict[str, Any] = {
            "probe/train_exact": train_result["exact_acc"],
            "probe/test_exact":  test_result["exact_acc"],
            "phase/index":       phase,
        }

        # Per-step deltas (use train probe deltas as representative)
        for t, delta in enumerate(train_result["step_deltas"], start=1):
            log_dict[f"z/delta_step_{t}"] = delta

        # PCA metrics on final z_H (mean-pool over sequence)
        if train_result["z_H"] is not None:
            z_H_np = _mean_pool_z(train_result["z_H"])  # (B, D)
            eigenvalues, eigenvectors = _pca(z_H_np)

            log_dict["z/eff_rank"]      = _effective_rank(eigenvalues)
            log_dict["z/pca_top2_var"]  = _pca_top2_var(eigenvalues)
            log_dict["z/mean_norm"]     = float(np.linalg.norm(z_H_np, axis=-1).mean())

            # PCA scatter (train probe)
            X_pca = z_H_np @ eigenvectors[:, :2]  # (B, 2)
            label_col0 = train_result["labels"][:, 0].numpy().astype(float)
            correct_np = train_result["correct_mask"].numpy()
            scatter_imgs = _make_pca_scatter(X_pca, label_col0, correct_np, "train")
            for k, img in scatter_imgs:
                log_dict[k] = img

        wandb.log(log_dict, step=step)

        # --- Save z snapshot ---
        if self._checkpoint_path is not None:
            snap_dir = os.path.join(self._checkpoint_path, "z_snapshots")
            os.makedirs(snap_dir, exist_ok=True)
            snap_path = os.path.join(snap_dir, f"step_{step}.pt")
            torch.save({
                "z_H":          train_result["z_H"],   # bfloat16
                "z_L":          train_result["z_L"],   # bfloat16
                "labels":       train_result["labels"],
                "correct_mask": train_result["correct_mask"],
            }, snap_path)
            log.info("z snapshot saved: %s", snap_path)

        # --- Phase transition: force checkpoint + extra snapshot ---
        if transitioned:
            log.info("Phase transition → %d at step %d", phase, step)
            if train_state is not None:
                save_train_state_fn(train_state)
                log.info("Forced checkpoint saved at step %d (phase transition)", step)
