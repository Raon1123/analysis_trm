"""Build the sigma^k dataset — Format D (full function I/O).

Task:
    Given a random permutation sigma on S = {1,...,n},
    predict the full function sigma^k: output[i] = sigma^k(i) for all i in S.

Input sequence (length SEQ_LEN = MAX_N + 1):
    [sigma(1), sigma(2), ..., sigma(n), PAD, ..., PAD]
    - positions 0..n-1 : permutation sigma (1-indexed)
    - positions n..    : PAD (0)

Label sequence (length SEQ_LEN):
    [sigma^k(1), sigma^k(2), ..., sigma^k(n), PAD, ..., PAD]
    - positions 0..n-1 : sigma^k values (all prediction targets, 1-indexed)
    - positions n..    : PAD (0 = ignore_label_id, masked out by loss)

Encoding: 1-indexed throughout (PAD = 0, elements 1..n).
    vocab_size = MAX_N + 1.

Deduplication: each permutation sigma appears at most once per k-dataset.
    Train and test splits are completely disjoint (no shared permutations).

Format: identical to build_sudoku_dataset.py for pipeline compatibility.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm

from common import PuzzleDatasetMetadata


cli = ArgParser()

MAX_N: int = 20
SEQ_LEN: int = MAX_N + 1       # 21: permutation (up to MAX_N positions) + 1 trailing PAD
VOCAB_SIZE: int = MAX_N + 1    # 21: PAD=0, elements 1..MAX_N


class DataConfig(BaseModel):
    output_dir: str = "data/sigma_k"
    k: int = 2           # composition depth k (fixed per dataset)
    n: int = 20         # permutation size |S| = {1,...,n}, must be <= MAX_N
    train_size: int = 5000
    test_size: int = 1000
    seed: int = 42


# ---------------------------------------------------------------------------
# Permutation utilities (0-indexed internally)
# ---------------------------------------------------------------------------

def perm_order(sigma: np.ndarray) -> int:
    """Compute the order of a permutation (smallest t s.t. sigma^t = id)."""
    n = len(sigma)
    visited = np.zeros(n, dtype=bool)
    ord_ = 1
    for i in range(n):
        if not visited[i]:
            cycle = 0
            j = i
            while not visited[j]:
                visited[j] = True
                j = sigma[j]
                cycle += 1
            ord_ = ord_ * cycle // math.gcd(ord_, cycle)  # lcm
    return ord_


def apply_sigma_k(sigma: np.ndarray, k: int) -> np.ndarray:
    """Return sigma^k as an array: result[i] = sigma^k(i) (0-indexed)."""
    result = np.arange(len(sigma))
    for _ in range(k):
        result = sigma[result]
    return result


def sample_unique_permutations(n: int, total: int, rng: np.random.Generator,
                                seen: set[bytes]) -> list[np.ndarray]:
    """
    Sample `total` permutations of [0..n-1] that are absent from `seen`.
    Each sampled permutation is added to `seen` before returning.

    Guarantee: every returned permutation is distinct and not in `seen` on entry.
    The caller controls which slice goes to train vs. test, making disjointness
    a structural property of the call site rather than an implicit side effect.
    """
    result: list[np.ndarray] = []
    while len(result) < total:
        sigma = rng.permutation(n)
        key = sigma.tobytes()
        if key not in seen:
            seen.add(key)
            result.append(sigma.copy())
    return result


# ---------------------------------------------------------------------------
# Example generation
# ---------------------------------------------------------------------------

def make_example(n: int, k: int, sigma: np.ndarray):
    """
    Build (input, label) from a given 0-indexed permutation sigma.

    Input:  [sigma(1), ..., sigma(n), PAD, ...]  1-indexed, length SEQ_LEN
    Label:  [sigma^k(1), ..., sigma^k(n), PAD, ...]  1-indexed, length SEQ_LEN
    PAD positions (>= n) remain 0 = ignore_label_id, masked out by loss.
    """
    sigma_k = apply_sigma_k(sigma, k)

    # Build input (1-indexed; positions n.. stay 0 = PAD)
    inp = np.zeros(SEQ_LEN, dtype=np.int32)
    inp[:n] = sigma + 1

    # Build label (1-indexed; PAD positions stay 0 = ignore_label_id)
    lbl = np.zeros(SEQ_LEN, dtype=np.int32)
    lbl[:n] = sigma_k + 1

    return inp, lbl


# ---------------------------------------------------------------------------
# Split builder
# ---------------------------------------------------------------------------

def build_split(split_name: str, config: DataConfig,
                sigmas: list[np.ndarray]):
    """
    Build one split from a pre-sampled list of unique permutations.

    Permutations are supplied externally (already deduplicated and split),
    so this function has no sampling logic and no shared state.
    """
    inputs_list, labels_list = [], []

    for sigma in tqdm(sigmas, desc=f"[{split_name}] k={config.k}"):
        inp, lbl = make_example(config.n, config.k, sigma)
        inputs_list.append(inp)
        labels_list.append(lbl)

    inputs = np.stack(inputs_list)  # [N, SEQ_LEN]
    labels = np.stack(labels_list)  # [N, SEQ_LEN]
    N = len(inputs)

    # Index arrays (1 example per puzzle, 1 puzzle per group)
    puzzle_indices     = np.arange(N + 1, dtype=np.int32)
    group_indices      = np.arange(N + 1, dtype=np.int32)
    puzzle_identifiers = np.zeros(N, dtype=np.int32)

    metadata = PuzzleDatasetMetadata(
        seq_len=SEQ_LEN,
        vocab_size=VOCAB_SIZE,
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=N,
        mean_puzzle_examples=1.0,
        total_puzzles=N,
        sets=["all"],
    )

    save_dir = os.path.join(config.output_dir, str(config.k), split_name)
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f, indent=2)

    for name, arr in [
        ("inputs",             inputs),
        ("labels",             labels),
        ("puzzle_identifiers", puzzle_identifiers),
        ("puzzle_indices",     puzzle_indices),
        ("group_indices",      group_indices),
    ]:
        np.save(os.path.join(save_dir, f"all__{name}.npy"), arr)

    print(f"  saved {N} examples → {save_dir}/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@cli.command(singleton=True)
def build(config: DataConfig):
    assert 1 <= config.n <= MAX_N, f"n must be in [1, {MAX_N}], got {config.n}"
    assert config.k >= 1,          f"k must be >= 1, got {config.k}"

    print(f"sigma^k dataset (Format D)  n={config.n}  "
          f"train={config.train_size}  test={config.test_size}")

    os.makedirs(config.output_dir, exist_ok=True)
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)

    for k in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20]:
        config_k = config.model_copy(update={"k": k})
        print(f"\nBuilding k={k} dataset...")
        rng = np.random.default_rng(config.seed)

        # Sample train + test together from a single unique pool.
        # Slicing is the only mechanism that assigns permutations to splits,
        # so train ∩ test = ∅ is guaranteed by construction.
        seen: set[bytes] = set()
        total = config.train_size + config.test_size
        all_sigmas = sample_unique_permutations(config.n, total, rng, seen)

        train_sigmas = all_sigmas[:config.train_size]
        test_sigmas  = all_sigmas[config.train_size:]

        build_split("train", config_k, train_sigmas)
        build_split("test",  config_k, test_sigmas)

    print("Done.")


if __name__ == "__main__":
    cli()
