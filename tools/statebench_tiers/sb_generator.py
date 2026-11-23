"""Clean-room re-implementation of the allenai/state-bench trace generator.

Source read (not imported): allenai/state-bench @ 5b333b8d426638d8825f40efe8a508c20eb85730,
``src/state_bench/generation.py`` (``_generate_swap_periodic_source`` l.238-349,
``derive_variants`` l.119-204, helpers l.352-461) and ``src/state_bench/model.py``
(operation records l.34-70, ``StateTrackingInstance`` l.73-98).

Only the semantic trace is reproduced (no formatters, filler, distractors or
instance ids).  RNG call order follows the source line by line so that, for the
same ``random.Random`` state, this module emits the same operations, query and
final state as the source.  ``tools/statebench_tiers/p1_equiv_check.py`` checks
that claim against the pinned source (EXP-027 prereg, P1).

Operation tuples: ("const", target, value) | ("copy", target, source) |
("inc", target, delta).  Registers 0..m-1 are ordinary variables; register m is
the scratch register TMP (source: model.py l.77-81, generation.py l.273).
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass

TIERS = ("r-trivial", "aperiodic", "solvable", "periodic")


@dataclass(frozen=True)
class Trace:
    tier: str
    num_variables: int          # m (ordinary registers; TMP is register m)
    num_values: int             # q (value alphabet Z_q)
    operations: tuple           # tuple of op tuples, length m+1+N
    query: tuple                # registers queried (query_mode=all -> 0..m)
    final_state: tuple          # length m+1


def _sample_constant_count(lo: int, hi: int, rng: random.Random) -> int:
    # generation.py l.385-396
    if lo == hi:
        return lo
    sampled = math.exp(rng.uniform(math.log(lo), math.log(hi + 1)))
    return min(hi, int(sampled))


def _initial_values(m: int, q: int, rng: random.Random) -> list[int]:
    # generation.py l.399-414, distinct=False branch (configs use distinct_initial_values: false)
    diversity = min(m, q)
    vocab = rng.sample(range(q), diversity)
    values = [vocab[i % diversity] for i in range(m)]
    rng.shuffle(values)
    return values


def _balanced_pairs(m: int, count: int, rng: random.Random) -> list[tuple[int, int]]:
    # generation.py l.417-445
    part = [0] * m
    pairs = []
    for _ in range(count):
        mn = min(part)
        left = rng.choice([v for v, c in enumerate(part) if c == mn])
        omn = min(c for v, c in enumerate(part) if v != left)
        right = rng.choice([v for v, c in enumerate(part) if v != left and c == omn])
        part[left] += 1
        part[right] += 1
        pairs.append((left, right))
    return pairs


def _injection_counts(constant_count: int, update_count: int) -> Counter:
    # generation.py l.352-358
    if not constant_count:
        return Counter()
    cutoff = max(1, min(update_count, round(0.8 * update_count)))
    return Counter(i * cutoff // constant_count for i in range(constant_count))


def _choose_reset(state, m, q, future, resets, rng):
    # generation.py l.361-382
    best = max(future)
    cands = [v for v, c in enumerate(future) if c == best]
    few = min(resets[v] for v in cands)
    cands = [v for v in cands if resets[v] == few]
    target = rng.choice(cands)
    live = set(state[:m])
    novel = [v for v in range(q) if v not in live]
    choices = novel or [v for v in range(q) if v != state[target]]
    return ("const", target, rng.choice(choices))


def _apply(op, state):
    kind, t, a = op
    if kind == "const":
        state[t] = a
    elif kind == "copy":
        state[t] = state[a]
    else:
        raise ValueError(op)


def periodic_source(*, m: int, q: int, n_extra_min: int, n_extra_max: int,
                    ap_const_min: int = 1, ap_const_max: int = 64,
                    rng: random.Random) -> Trace:
    """generation.py l.238-349 (query_mode=all, distinct_initial_values=False)."""
    extra = rng.randint(max(3, n_extra_min), n_extra_max)
    requested = _sample_constant_count(ap_const_min, ap_const_max, rng)
    n_const = min(requested, extra - 3)
    n_ref = extra - n_const
    n_swap = n_ref // 3
    n_scratch = n_ref - 3 * n_swap
    if n_swap < 1:
        raise ValueError("need at least one swap")
    tmp = m
    init = _initial_values(m, q, rng)
    state = [*init, rng.randrange(q)]
    order = list(range(m + 1))
    rng.shuffle(order)
    ops = [("const", v, state[v]) for v in order]
    pairs = _balanced_pairs(m, n_swap, rng)
    future = [0] * m
    for a, b in pairs:
        future[a] += 1
        future[b] += 1
    for _ in range(n_scratch):
        op = ("copy", tmp, rng.randrange(m))
        ops.append(op)
        _apply(op, state)
    inj = _injection_counts(n_const, n_swap)
    resets = [0] * m
    for i, (a, b) in enumerate(pairs):
        for _ in range(inj[i]):
            op = _choose_reset(state, m, q, future, resets, rng)
            ops.append(op)
            _apply(op, state)
            resets[op[1]] += 1
        for op in (("copy", tmp, a), ("copy", a, b), ("copy", b, tmp)):
            ops.append(op)
            _apply(op, state)
        future[a] -= 1
        future[b] -= 1
    query = tuple(range(m + 1))  # _choose_query with ALL (l.455-457)
    return Trace("periodic", m, q, tuple(ops), query, tuple(state))


def derive(source: Trace) -> dict[str, Trace]:
    """generation.py l.119-204: same targets/positions, tier-specific rewrite."""
    q = source.num_values
    R = source.num_variables + 1
    state = [None] * R
    first = [None] * R
    ap, so, rt = [], [], []
    for op in source.operations:
        kind, t, a = op
        if kind == "const":
            value = a
            so.append(op)
        elif kind == "copy":
            value = state[a]
            so.append(("inc", t, (value - state[t]) % q))
        else:
            raise ValueError(op)
        ap.append(("const", t, value))
        if first[t] is None:
            first[t] = value
        rt.append(("const", t, first[t]))
        state[t] = value
    out = {"periodic": source}
    for tier, ops, fs in (("aperiodic", ap, source.final_state),
                          ("solvable", so, source.final_state),
                          ("r-trivial", rt, tuple(first))):
        out[tier] = Trace(tier, source.num_variables, q, tuple(ops), source.query, fs)
    return out


def evaluate(trace: Trace) -> tuple:
    """Execute a trace from an empty register file (every register's first op is a const)."""
    R = trace.num_variables + 1
    q = trace.num_values
    state = [None] * R
    for kind, t, a in trace.operations:
        if kind == "const":
            state[t] = a
        elif kind == "copy":
            state[t] = state[a]
        else:
            state[t] = (state[t] + a) % q
    return tuple(state)
