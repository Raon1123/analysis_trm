"""Build the sigma^k dataset.

Task:
    Given a random permutation sigma on S = {1,...,n} and a query element x in S,
    predict sigma^k(x) — the result of applying sigma exactly k times.

Input sequence (length SEQ_LEN = MAX_N + 1 = 1001):
    [x, sigma(1), sigma(2), ..., sigma(n), PAD, ..., PAD]
    - position 0      : query x (1-indexed)
    - positions 1..n  : permutation sigma (1-indexed)
    - positions n+1.. : PAD (0)

Label sequence (length SEQ_LEN):
    [sigma^k(x), 0, 0, ..., 0]
    - position 0 only is the prediction target (ignore_label_id=0 masks the rest)

Encoding: 1-indexed throughout (PAD = 0, elements 1..n).
    No +1 shift needed at save time since values are already 1-indexed.
    vocab_size = MAX_N + 1 = 1001.

Permutation filter: only keep sigma with ord(sigma) > k,
    ensuring effective composition depth equals k.

Format: identical to build_sudoku_dataset.py for pipeline compatibility.
"""

from __future__ import annotations

import json
import math
import os
from typing import Optional

import numpy as np
from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm

from common import PuzzleDatasetMetadata


cli = ArgParser()

MAX_N: int = 1000
SEQ_LEN: int = MAX_N + 1       # 1001: query + full permutation (padded)
VOCAB_SIZE: int = MAX_N + 1    # 1001: PAD=0, elements 1..MAX_N


class DataConfig(BaseModel):
    output_dir: str = "data/sigma_k"
    k: int = 2           # composition depth k (fixed per dataset)
    n: int = 100         # permutation size |S| = {1,...,n}, must be <= MAX_N
    train_size: int = 50000
    test_size: int = 5000
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


def apply_k_times(sigma: np.ndarray, x: int, k: int) -> int:
    """Apply sigma to x exactly k times (0-indexed)."""
    for _ in range(k):
        x = int(sigma[x])
    return x


# ---------------------------------------------------------------------------
# Example generation
# ---------------------------------------------------------------------------

def make_example(n: int, k: int, rng: np.random.Generator):
    """
    Sample one (input, label) pair.

    Returns two int32 arrays of length SEQ_LEN.
    """
    # Rejection-sample until ord(sigma) > k (guarantees effective depth = k).
    while True:
        sigma = rng.permutation(n)        # 0-indexed
        if perm_order(sigma) > k:
            break

    x = int(rng.integers(0, n))          # 0-indexed query
    answer = apply_k_times(sigma, x, k)  # 0-indexed answer

    # Build input (1-indexed; pad with 0)
    inp = np.zeros(SEQ_LEN, dtype=np.int32)
    inp[0] = x + 1                # query: 1-indexed
    inp[1:n + 1] = sigma + 1      # permutation: 1-indexed; positions n+1.. stay 0

    # Build label (only position 0 is the prediction target)
    lbl = np.zeros(SEQ_LEN, dtype=np.int32)
    lbl[0] = answer + 1           # 1-indexed answer

    return inp, lbl


# ---------------------------------------------------------------------------
# Split builder
# ---------------------------------------------------------------------------

def build_split(split_name: str, size: int, config: DataConfig, rng: np.random.Generator):
    inputs_list, labels_list = [], []

    for _ in tqdm(range(size), desc=f"[{split_name}] n={config.n} k={config.k}"):
        inp, lbl = make_example(config.n, config.k, rng)
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

    save_dir = os.path.join(config.output_dir, split_name)
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

    print(f"sigma^k dataset  n={config.n}  k={config.k}  "
          f"train={config.train_size}  test={config.test_size}")

    os.makedirs(config.output_dir, exist_ok=True)
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)

    rng = np.random.default_rng(config.seed)
    build_split("train", config.train_size, config, rng)
    build_split("test",  config.test_size,  config, rng)

    print("Done.")


if __name__ == "__main__":
    cli()
