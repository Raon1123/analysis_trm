"""EXP-027 P1: transformation-monoid census of the StateBench tiers (CPU only).

For each tier the generator's operation alphabet (sb_generator.py / pinned source
generation.py) is closed under composition and the maximal subgroups (group
H-classes) of the resulting transformation monoid are measured.

Faithful symbolic representation (valid for any value alphabet with >= 2 values):
  * copy/constant tiers: an element is a tuple t of length R=m+1; t[i] in 0..R-1
    means "register i receives the old value of register t[i]", t[i] = R+c means
    "register i receives constant c".  Every word over {a<-b, a<-c} is such a map,
    and distinct tuples are distinct maps on V^R when |V| >= 2.
  * solvable tier: per register an element is "+d" (0<=d<q) or "=c"; registers
    never read each other (generation.py l.161-170), so the monoid is the direct
    product of the one-register monoids.

What is ENUMERATED (exact closure + all group H-classes):
  periodic letter alphabet  m in {4,5}, |V|=2   (copies a<-b for every ordered pair
                                                 of registers incl. TMP, constants)
  periodic swap alphabet    m in {4,5}, |V|=2   (macro TMP<-a;a<-b;b<-TMP, scratch
                                                 loads TMP<-a, constants)
  aperiodic                 m in {4,5}, |V|=2   (constants only)
  solvable                  one register, q in {5,100,127}; R in {2,3} at q=4
What is ARGUED (not enumerated): m=8 periodic (|T_9| = 9^9 is too large) and the
solvable tier at R=m+1 registers with q=127 (direct-product argument, checked at
small R).  See the EXP-027 prereg §P1 for the argument text.

Output: JSON to --json and a table on stdout.  Exit 0 on success.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import deque


# ----------------------------------------------------------------- generic monoid
def closure(gens, identity, compose):
    """All words over gens (BFS by right multiplication). compose(s,t) = s then t."""
    seen = {identity}
    dq = deque([identity])
    while dq:
        x = dq.popleft()
        for g in gens:
            y = compose(x, g)
            if y not in seen:
                seen.add(y)
                dq.append(y)
    return seen


def group_hclasses(monoid, compose):
    """Map idempotent e -> list of group elements x with x^omega = e (|H_e| = len)."""
    classes = {}
    for x in monoid:
        powers = [x]
        p = x
        while True:
            p = compose(p, x)
            if p in powers:
                break
            powers.append(p)
        # p == x^(len(powers)+1); x is a group element iff the cycle returns to x itself
        if p != x:
            continue
        # x lies in a cyclic group of order len(powers); its identity is the idempotent power
        e = next(y for y in powers if compose(y, y) == y)
        classes.setdefault(e, []).append(x)
    return classes


def derived_series(group, compose):
    """Sizes of G > G' > G'' ... until stable. Group given as a list of elements."""
    gset = set(group)
    e = next(y for y in group if compose(y, y) == y)

    def inv(x):
        for y in gset:
            if compose(x, y) == e:
                return y
        raise RuntimeError("no inverse")

    sizes = [len(gset)]
    cur = gset
    while True:
        invs = {x: inv(x) for x in cur}
        comms = {compose(compose(compose(a, b), invs[a]), invs[b]) for a in cur for b in cur}
        sub = closure(list(comms), e, compose)
        sizes.append(len(sub))
        if len(sub) == len(cur):
            break
        cur = sub
        # inverse lookup restricted to the subgroup
        gset = cur
    return sizes


def prime_factors(n):
    out, d = [], 2
    while d * d <= n:
        while n % d == 0:
            if d not in out:
                out.append(d)
            n //= d
        d += 1
    if n > 1 and n not in out:
        out.append(n)
    return out


def summarize(monoid, compose, label, extra):
    classes = group_hclasses(monoid, compose)
    sizes = sorted({len(v) for v in classes.values()}, reverse=True)
    largest = max(classes.values(), key=len)
    ds = derived_series(largest, compose) if len(largest) <= 1000 else None
    # composition-factor primes of each distinct maximal-subgroup size (via derived series of one rep)
    reps = {}
    for v in classes.values():
        reps.setdefault(len(v), v)
    factors = {}
    any_nonsolvable = False
    for sz, grp in sorted(reps.items(), reverse=True):
        if sz == 1:
            factors[sz] = {"derived_series": [1], "abelian_layer_primes": [], "perfect_core": 1}
            continue
        if sz > 1000:
            factors[sz] = {"derived_series": None, "note": "too large for derived series"}
            continue
        d = derived_series(grp, compose)
        primes = []
        for a, b in zip(d, d[1:]):
            if a != b:
                primes += [p for p in prime_factors(a // b) if p not in primes]
        core = d[-1]
        if core > 1:
            any_nonsolvable = True
        factors[sz] = {"derived_series": d, "abelian_layer_primes": primes, "perfect_core": core}
    return {
        "label": label, **extra,
        "monoid_size": len(monoid),
        "num_idempotents_with_group": len(classes),
        "maximal_subgroup_orders": sizes,
        "largest_maximal_subgroup_order": len(largest),
        "largest_derived_series": ds,
        "per_order_structure": {str(k): v for k, v in factors.items()},
        "all_groups_trivial (aperiodic)": sizes == [1],
        "solvable": not any_nonsolvable,
    }


# ------------------------------------------------------- copy/constant representation
def make_copy_compose():
    def compose(s, t):  # apply s, then t
        R = len(s)
        return tuple(s[x] if x < R else x for x in t)
    return compose


def copy_letter(R, target, source):
    return tuple(source if i == target else i for i in range(R))


def const_letter(R, target, c):
    return tuple(R + c if i == target else i for i in range(R))


def periodic_letter_gens(m, V):
    R = m + 1
    gens = [const_letter(R, a, c) for a in range(R) for c in range(V)]
    # generation.py emits TMP<-a (scratch load / swap step 1), a<-b (step 2), b<-TMP (step 3):
    # i.e. every ordered pair (target, source) of distinct registers occurs as a letter.
    gens += [copy_letter(R, a, b) for a in range(R) for b in range(R) if a != b]
    return gens


def periodic_swap_gens(m, V):
    R = m + 1
    tmp = m
    comp = make_copy_compose()
    gens = [const_letter(R, a, c) for a in range(R) for c in range(V)]
    gens += [copy_letter(R, tmp, a) for a in range(m)]          # scratch loads
    for a, b in itertools.permutations(range(m), 2):
        sw = comp(comp(copy_letter(R, tmp, a), copy_letter(R, a, b)), copy_letter(R, b, tmp))
        gens.append(sw)
    return gens


def aperiodic_gens(m, V):
    R = m + 1
    return [const_letter(R, a, c) for a in range(R) for c in range(V)]


# --------------------------------------------------------------- solvable representation
def make_solv_compose(q):
    # element: tuple per register of ints; 0..q-1 = "+d", q..2q-1 = "=c"
    def compose(s, t):
        out = []
        for x, y in zip(s, t):
            if y >= q:
                out.append(y)
            elif x >= q:
                out.append(q + ((x - q) + y) % q)
            else:
                out.append((x + y) % q)
        return tuple(out)
    return compose


def solvable_gens(R, q):
    gens = []
    for a in range(R):
        for d in range(q):
            gens.append(tuple(d if i == a else 0 for i in range(R)))
        for c in range(q):
            gens.append(tuple(q + c if i == a else 0 for i in range(R)))
    return gens


def solvable_unit_group_abelian(R, q):
    """Exhaustive commutativity check on the unit group when it is small, else per-register."""
    comp = make_solv_compose(q)
    units = [tuple(x) for x in itertools.product(range(q), repeat=R)]
    if len(units) > 2000:
        units = [tuple(d if i == 0 else 0 for i in range(R)) for d in range(q)]
    return all(comp(a, b) == comp(b, a) for a in units for b in units)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--V", type=int, default=2, help="value alphabet size for symbolic enumeration")
    args = ap.parse_args()
    V = args.V
    rows = []
    comp = make_copy_compose()
    for m in (4, 5):
        R = m + 1
        ident = tuple(range(R))
        for label, gfun in (("periodic/letter", periodic_letter_gens),
                            ("periodic/swap-macro", periodic_swap_gens),
                            ("aperiodic", aperiodic_gens)):
            M = closure(gfun(m, V), ident, comp)
            rows.append(summarize(M, comp, label, {"m": m, "R": R, "V": V}))
            print(f"[enumerated] {label:22s} m={m} V={V}: |M|={len(M)} "
                  f"max subgroup orders={rows[-1]['maximal_subgroup_orders'][:6]} "
                  f"solvable={rows[-1]['solvable']}", flush=True)
    for R, q in ((1, 5), (1, 100), (1, 127), (2, 4), (3, 4), (2, 5)):
        compq = make_solv_compose(q)
        M = closure(solvable_gens(R, q), tuple([0] * R), compq)
        classes = group_hclasses(M, compq)
        orders = sorted({len(v) for v in classes.values()}, reverse=True)
        row = {"label": "solvable", "R": R, "q": q, "monoid_size": len(M),
               "expected_direct_product_size_(2q)^R": (2 * q) ** R,
               "maximal_subgroup_orders": orders,
               "largest_maximal_subgroup_order": max(orders),
               "unit_group_abelian": solvable_unit_group_abelian(R, q),
               "abelian_layer_primes": prime_factors(q),
               "solvable": True}
        rows.append(row)
        print(f"[enumerated] solvable R={R} q={q}: |M|={len(M)} (2q)^R={(2*q)**R} "
              f"max subgroup orders={orders[:6]} abelian={row['unit_group_abelian']}", flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)
            f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
