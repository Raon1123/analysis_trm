"""EXP-017 readouts: decode, validity, agreement, powers / order, nearest power, mixture Hamming, D-9, soft mass.

Conventions (EXP-017 §0, §4.3):
* Token form: a permutation row is [σ(1)+1, ..., σ(10)+1] (pad token 0). Internally permutations are
  0-based int arrays p with p[x] = σ(x+1)-1; a decode ŷ is converted with ``tokens - 1`` so pad -> -1
  (it then mismatches every power, i.e. it counts in Hamming).
* σ^m is the m-fold functional power; σ^0 = identity.
* Nearest power m*(i,j): unique argmin over m in 0..ord(σ_i)-1 of Hamming(ŷ_j(i), σ_i^m), provided the
  minimum is <= HAM_MAX (FROZEN_RULES czz2.HAM_MAX). Otherwise undefined: "none" (min > HAM_MAX) or "tie".
* Ham_mix(ŷ, σ) = Σ over cycles c of σ of min_{e in {0,1,k}} Hamming on c's positions to σ^e.
* A_i = {m in 2..k-1 : some cycle length L of σ_i has m mod L not in {0, 1 mod L, k mod L}}.
* proper(i,j) = m* defined ∧ m* ∈ A_i ∧ Hamming(ŷ, σ^m*) < Ham_mix(ŷ).
Thresholds come from judge.FROZEN_RULES only.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from analysis.z_dynamics.judge import num

N_PERM_SIZE = 10  # n = 10 fixed (EXP-017 §0)
UNDEF = -1        # m* undefined


# ------------------------------------------------------------------ permutation algebra (0-based)

def perm_pow(p: np.ndarray, m: int) -> np.ndarray:
    q = np.arange(len(p))
    for _ in range(m):
        q = p[q]
    return q


def cycles_of(p: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros(len(p), bool)
    cyc = []
    for i in range(len(p)):
        if not seen[i]:
            c, j = [], i
            while not seen[j]:
                seen[j] = True
                c.append(j)
                j = int(p[j])
            cyc.append(np.array(c, dtype=np.int64))
    return cyc


def order(p: np.ndarray) -> int:
    L = 1
    for c in cycles_of(p):
        L = L * len(c) // math.gcd(L, len(c))
    return L


def cycle_type(p: np.ndarray) -> tuple[int, ...]:
    return tuple(sorted((len(c) for c in cycles_of(p)), reverse=True))


def identifiable(p: np.ndarray, k: int) -> list[int]:
    lens = [len(c) for c in cycles_of(p)]
    return [m for m in range(2, k) if any((m % L) not in {0 % L, 1 % L, k % L} for L in lens)]


def perm_from_cycle_type(ct: tuple[int, ...]) -> np.ndarray:
    """Canonical representative: consecutive positions per cycle, x -> x+1 within a cycle."""
    p = np.empty(sum(ct), dtype=np.int64)
    s = 0
    for L in ct:
        for t in range(L):
            p[s + t] = s + (t + 1) % L
        s += L
    return p


def tokens_to_perm0(tokens: np.ndarray) -> np.ndarray:
    """Token row(s) (value+1, pad 0) -> 0-based (pad -> -1)."""
    return np.asarray(tokens, dtype=np.int64) - 1


# ------------------------------------------------------------------ per-sample power structure

class SigmaInfo:
    """Power structure of one σ (0-based) for a composition depth k."""

    def __init__(self, sig: np.ndarray, k: int):
        self.sig = np.asarray(sig, dtype=np.int64)
        self.k = int(k)
        self.o = order(self.sig)
        self.P = np.stack([perm_pow(self.sig, m) for m in range(self.o)])  # (o, 10)
        self.cyc = cycles_of(self.sig)
        self.ctype = cycle_type(self.sig)
        self.A = set(identifiable(self.sig, self.k))
        self.informative = len(self.A) > 0
        self.gcd = math.gcd(self.k, self.o)

    def mix_ham(self, dec: np.ndarray) -> np.ndarray:
        """dec: (..., 10) 0-based. Best per-cycle mixture of σ^0, σ^1, σ^k (Hamming)."""
        tot = 0
        for c in self.cyc:
            tot = tot + np.min(np.stack(
                [(dec[..., c] != self.P[e % self.o][c]).sum(-1) for e in (0, 1, self.k)]), 0)
        return np.asarray(tot)

    def nearest(self, dec: np.ndarray, ham_max: int | None = None) -> dict:
        """dec: (J, 10) 0-based -> dict of arrays over J: m (UNDEF if none/tie), ham_min, tie, none, mix."""
        hm = int(num("czz2.HAM_MAX")) if ham_max is None else ham_max
        ham = (dec[:, None, :] != self.P[None, :, :]).sum(-1)  # (J, o)
        mn = ham.min(1)
        uniq = (ham == mn[:, None]).sum(1) == 1
        am = ham.argmin(1)
        none = mn > hm
        tie = (~none) & (~uniq)
        ok = uniq & ~none
        return {"m": np.where(ok, am, UNDEF), "ham_min": mn, "tie": tie, "none": none,
                "mix": self.mix_ham(dec)}

    def proper(self, near: dict) -> np.ndarray:
        m = near["m"]
        inA = np.isin(m, list(self.A)) if self.A else np.zeros_like(m, dtype=bool)
        return (m >= 0) & inA & (near["ham_min"] < near["mix"])

    def residues(self, dec: np.ndarray) -> list[list[int]]:
        """D-9: per cycle c and readout j, the residue e in Z_{L_c} with dec == σ^e on c's positions (-1 if none).

        Returns [n_cycles][J]. Residues are unique because σ^e restricted to a cycle of length L has L
        distinct values for e mod L.
        """
        out = []
        for c in self.cyc:
            L = len(c)
            cand = np.stack([self.P[e % self.o][c] for e in range(L)])  # (L, |c|)
            eq = (dec[:, None, c] == cand[None, :, :]).all(-1)            # (J, L)
            out.append([int(np.argmax(r)) if r.any() else -1 for r in eq])
        return out


# ------------------------------------------------------------------ decode-level metrics

def decode(logits) -> np.ndarray:
    """ŷ = argmax over the 11 tokens at answer positions 0..9. logits: (B, >=10, V) torch or numpy."""
    try:
        import torch
        if isinstance(logits, torch.Tensor):
            return logits[:, :N_PERM_SIZE].argmax(-1).to(torch.int64).cpu().numpy()
    except ImportError:  # pragma: no cover
        pass
    return np.asarray(logits)[:, :N_PERM_SIZE].argmax(-1).astype(np.int64)


def is_valid(tokens: np.ndarray) -> np.ndarray:
    """1[sorted(ŷ) == 1..10] per row; tokens (..., 10)."""
    return (np.sort(tokens, axis=-1) == np.arange(1, N_PERM_SIZE + 1)).all(-1)


def agreement_counts(tokens: np.ndarray, labels_tok: np.ndarray) -> int:
    """Number of (row, position) matches; divide by rows*10 for a_j."""
    return int((tokens == labels_tok).sum())


def trivial(tokens: np.ndarray, inputs_tok: np.ndarray) -> np.ndarray:
    """ŷ ∈ {identity, σ_i} per row (token form)."""
    ident = np.arange(1, N_PERM_SIZE + 1)
    return (tokens == ident).all(-1) | (tokens == inputs_tok).all(-1)


@lru_cache(maxsize=None)
def _cached_info(sig_key: tuple[int, ...], k: int) -> SigmaInfo:
    return SigmaInfo(np.array(sig_key), k)


def sigma_infos(inputs_tok: np.ndarray, k: int) -> list[SigmaInfo]:
    """One SigmaInfo per input row (cached by σ)."""
    return [_cached_info(tuple(int(x) for x in tokens_to_perm0(r)), int(k)) for r in inputs_tok[:, :N_PERM_SIZE]]


def labels_are_sigma_k(inputs_tok: np.ndarray, labels_tok: np.ndarray, k: int) -> dict:
    """G6: label == σ^k on all rows and ord(σ) > k."""
    n_lab = n_ord = 0
    for r, lab in zip(inputs_tok[:, :N_PERM_SIZE], labels_tok[:, :N_PERM_SIZE]):
        s = tokens_to_perm0(r)
        if sorted(s.tolist()) != list(range(N_PERM_SIZE)):
            continue
        n_lab += int(np.array_equal(perm_pow(s, k) + 1, lab))
        n_ord += int(order(s) > k)
    n = len(inputs_tok)
    return {"n": n, "label_eq_sigma_k": n_lab, "ord_gt_k": n_ord, "pass": n_lab == n and n_ord == n}


def soft_mass(probs: np.ndarray, infos: list[SigmaInfo], k: int) -> np.ndarray:
    """D-8: sum over (i, pos) of softmax[pos, σ_i^m(pos)+1] for m in 0..k. probs: (B, 10, V). Returns (k+1,) sums."""
    out = np.zeros(k + 1, dtype=np.float64)
    pos = np.arange(N_PERM_SIZE)
    for i, s in enumerate(infos):
        for m in range(k + 1):
            out[m] += float(probs[i, pos, s.P[m % s.o] + 1].sum())
    return out
