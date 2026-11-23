"""EXP-027 P1b: swap-support ladder 사전 점검 (CPU only, GPU 미사용).

---
title: P1b swap-support census (swap-support ladder 후보 통제의 monoid·trace 점검)
date: 2026-11-11
type: script
status: DRAFT
author_model: claude-sonnet-5-5
---

기호표 (사용 전 정의):
  m      일반(ordinary) 레지스터 수. 레지스터 0..m-1 이 일반, 레지스터 m 이 TMP. R = m+1.
  j      swap 에 참여할 수 있는 일반 레지스터의 수. 지지집합 S = {0..j-1}. 2 <= j <= m.
  V      기호적 열거에 쓰는 값 알파벳 크기 (V = 2; P1 과 같음, |V| >= 2 이면 충실).
  원소   길이 R 튜플 t. t[i] < R 이면 "레지스터 i 가 레지스터 t[i] 의 옛 값을 받는다",
         t[i] = R + c 이면 "레지스터 i 가 상수 c 를 받는다" (p1_monoid_census.py l.5-13 과 동일).
  letter     알파벳 = 상수 a<-c (R 개 전부) + TMP<-a + a<-b (a,b in S) + b<-TMP (b in S).
  macro      알파벳 = 상수 + TMP<-a + 매크로 (TMP<-a; a<-b; b<-TMP) (a,b in S, a != b).
  atomic     알파벳 = 상수 + 일반 레지스터 위의 호환 (a b) 를 TMP 부작용 없는 순열로 취급.
  scratch=all      scratch load TMP<-a 가 a in {0..m-1} 전부에서 가능 (원본 l.299-303 그대로).
  scratch=support  scratch load 도 a in S 로 제한 (민감도 변형).
  unit group       모노이드에서 가역인 원소(R 개 레지스터 전체를 순열하는 원소)의 군.
  H-class          idempotent e 에 대한 최대 부분군 H_e (P1 과 같은 정의).
  S_k              k 개 점 위의 대칭군, 위수 k!.

출처 (파일 + 줄):
  모델링·알파벳: tools/statebench_tiers/p1_monoid_census.py l.153-187 (copy/const 표현, letter/macro 알파벳)
  생성기: tools/statebench_tiers/sb_generator.py l.102-143 (periodic_source), l.56-68 (_balanced_pairs),
          l.79-89 (_choose_reset), l.125-128 (scratch load), l.137-141 (swap triple)
  원본: allenai/state-bench @ 5b333b8d426638d8825f40efe8a508c20eb85730
        src/state_bench/generation.py l.238-349, l.352-382, l.417-446 (이 세션에서 clone 을 직접 읽음)

판정 어휘 없음: 이 스크립트는 열거한 위수·derived series·카운트만 기록한다.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p1_monoid_census as P1  # noqa: E402
import sb_generator as SBG  # noqa: E402


# ----------------------------------------------------------------------------- alphabets
def support(j):
    return list(range(j))


def gens_letter(m, j, V, scratch):
    R, tmp, S = m + 1, m, support(j)
    gens = [P1.const_letter(R, a, c) for a in range(R) for c in range(V)]
    loads = range(m) if scratch == "all" else S
    gens += [P1.copy_letter(R, tmp, a) for a in loads]
    gens += [P1.copy_letter(R, a, b) for a in S for b in S if a != b]
    gens += [P1.copy_letter(R, b, tmp) for b in S]
    return gens


def gens_macro(m, j, V, scratch):
    R, tmp, S = m + 1, m, support(j)
    comp = P1.make_copy_compose()
    gens = [P1.const_letter(R, a, c) for a in range(R) for c in range(V)]
    loads = range(m) if scratch == "all" else S
    gens += [P1.copy_letter(R, tmp, a) for a in loads]
    for a, b in itertools.permutations(S, 2):
        gens.append(comp(comp(P1.copy_letter(R, tmp, a), P1.copy_letter(R, a, b)),
                         P1.copy_letter(R, b, tmp)))
    return gens


def gens_atomic(m, j, V, scratch):
    R, S = m + 1, support(j)
    gens = [P1.const_letter(R, a, c) for a in range(R) for c in range(V)]
    for a, b in itertools.combinations(S, 2):
        t = list(range(R))
        t[a], t[b] = b, a
        gens.append(tuple(t))
    return gens


GENS = {"letter": gens_letter, "macro": gens_macro, "atomic": gens_atomic}


# ------------------------------------------------------------------ vectorised monoid ops
def comp_arr(S, T, R):
    """Row-wise compose(s, t) = s then t, matching p1_monoid_census.make_copy_compose."""
    idx = np.minimum(T, R - 1).astype(np.intp)
    out = np.take_along_axis(S, idx, axis=1)
    return np.where(T < R, out, T).astype(np.int8)


def closure_np(gens, R, B):
    pw = (B ** np.arange(R)).astype(np.int64)
    visited = np.zeros(B ** R, dtype=bool)
    ident = np.arange(R, dtype=np.int8)
    visited[int((ident.astype(np.int64) * pw).sum())] = True
    frontier = ident[None, :]
    garr = [np.array(g, dtype=np.int8) for g in gens]
    while frontier.shape[0]:
        collect = []
        for g in garr:
            idx = np.minimum(g, R - 1).astype(np.intp)
            Y = np.where(g < R, frontier[:, idx], g).astype(np.int64)
            codes = Y @ pw
            codes = codes[~visited[codes]]
            if codes.size:
                codes = np.unique(codes)
                visited[codes] = True
                collect.append(codes)
        if not collect:
            break
        new = np.concatenate(collect)
        frontier = ((new[:, None] // pw) % B).astype(np.int8)
    allc = np.flatnonzero(visited)
    return ((allc[:, None] // pw) % B).astype(np.int8), pw


def power_arr(X, L, R):
    res = np.broadcast_to(np.arange(R, dtype=np.int8), X.shape).copy()
    base = X.copy()
    while L:
        if L & 1:
            res = comp_arr(res, base, R)
        base = comp_arr(base, base, R)
        L >>= 1
    return res


def lcm_upto(n):
    out = 1
    for k in range(1, n + 1):
        out = out * k // math.gcd(out, k)
    return out


def sympy_group(rows, I):
    """Faithful permutation representation of an H-class on the e-fixed registers I."""
    from sympy.combinatorics import Permutation, PermutationGroup
    pos = {r: k for k, r in enumerate(I)}
    perms = []
    for x in rows:
        img = [int(x[i]) for i in I]
        if any(v not in pos for v in img) or len(set(img)) != len(I):
            return None
        perms.append(tuple(pos[v] for v in img))
    if len(set(perms)) != len(rows):
        return None
    gens = [Permutation(list(p)) for p in perms[:10]]
    G = PermutationGroup(gens)
    if G.order() != len(rows):
        G = PermutationGroup([Permutation(list(p)) for p in perms])
    return G if G.order() == len(rows) else None


def derived_sizes(G):
    sizes = [int(G.order())]
    cur = G
    while True:
        d = cur.derived_subgroup()
        sizes.append(int(d.order()))
        if d.order() == cur.order():
            break
        cur = d
    return sizes


def analyse_cell(m, j, level, scratch, V=2):
    t0 = time.time()
    R = m + 1
    B = R + V
    gens = GENS[level](m, j, V, scratch)
    X, _ = closure_np(gens, R, B)
    n = X.shape[0]
    ident = np.arange(R)
    is_unit = ((X < R).all(1)) & (np.sort(X, axis=1) == ident).all(1)
    n_units = int(is_unit.sum())
    L = lcm_upto(R)
    P = power_arr(X, L, R)
    isgrp = (comp_arr(X, P, R) == X).all(1)
    Xg, Pg = X[isgrp], P[isgrp]
    pw = (B ** np.arange(R)).astype(np.int64)
    ecodes = Pg.astype(np.int64) @ pw
    order = np.argsort(ecodes, kind="stable")
    ecodes_s, Xg_s, Pg_s = ecodes[order], Xg[order], Pg[order]
    uniq, start, counts = np.unique(ecodes_s, return_index=True, return_counts=True)
    order_hist = Counter(int(c) for c in counts)
    largest = int(counts.max())
    # structure of the largest H-class and of every H-class with order >= 60
    # (groups of order < 60 are solvable: smallest non-solvable group is A_5, order 60)
    big = [(int(c), int(s)) for c, s in zip(counts, start) if c >= 60]
    top = int(np.argmax(counts))
    if int(counts[top]) < 60:
        big.append((int(counts[top]), int(start[top])))
    nonsolv_orders = Counter()
    solv_orders = Counter()
    rep_series = {}
    failed = 0
    names = {}
    for c, s in big:
        rows = Xg_s[s:s + c]
        e = Pg_s[s]
        I = [i for i in range(R) if int(e[i]) == i]
        G = sympy_group(rows, I)
        if G is None:
            failed += 1
            continue
        if c >= 60:
            (solv_orders if G.is_solvable else nonsolv_orders)[c] += 1
        if c not in rep_series:
            rep_series[c] = derived_sizes(G)
            moved = set()
            for g in G.generators:
                moved |= set(g.support())
            k = len(moved)
            names[c] = (f"S_{k}" if k is not None and math.factorial(k) == c else None, k)
    # sanity: ordering of small classes via P1.derived_series on one representative per order < 60
    small_series = {}
    for c in sorted({int(c) for c in counts if c < 60}, reverse=True):
        s = int(start[int(np.argmax(counts == c))])
        rows = [tuple(int(v) for v in r) for r in Xg_s[s:s + c]]
        small_series[c] = P1.derived_series(rows, P1.make_copy_compose())
    lg = largest
    return {
        "m": m, "j": j, "level": level, "scratch": scratch, "V": V, "R": R,
        "generators": len(gens),
        "monoid_size": int(n),
        "n_units_bijective_tuples": n_units,
        "n_group_elements": int(isgrp.sum()),
        "n_H_classes": int(len(uniq)),
        "H_class_order_histogram": {str(k): v for k, v in sorted(order_hist.items(), reverse=True)},
        "largest_maximal_subgroup_order": lg,
        "largest_derived_series": rep_series.get(lg) or small_series.get(lg),
        "largest_group_name": (names.get(lg) or (None,))[0],
        "largest_group_moved_points": (names.get(lg) or (None, None))[1],
        "classes_with_order_ge_60": int(sum(1 for c in counts if c >= 60)),
        "nonsolvable_class_orders": {str(k): v for k, v in sorted(nonsolv_orders.items())},
        "solvable_class_orders_ge_60": {str(k): v for k, v in sorted(solv_orders.items())},
        "perm_rep_check_failed": failed,
        "all_maximal_subgroups_solvable": (not nonsolv_orders) and failed == 0,
        "derived_series_by_order_ge_60": {str(k): v for k, v in sorted(rep_series.items()) if k >= 60},
        "elapsed_s": round(time.time() - t0, 1),
    }


def _worker(conn, args):
    try:
        conn.send(("ok", analyse_cell(*args)))
    except Exception as exc:  # noqa: BLE001
        conn.send(("err", f"{type(exc).__name__}: {exc}"))
    conn.close()


def run_cells(cells, timeout, maxproc):
    ctx = mp.get_context("fork")
    pending = list(cells)
    running = []
    results = {}
    while pending or running:
        while pending and len(running) < maxproc:
            c = pending.pop(0)
            a, b = ctx.Pipe(duplex=False)
            p = ctx.Process(target=_worker, args=(b, c))
            p.start()
            running.append((c, p, a, time.time()))
        for item in list(running):
            c, p, a, t0 = item
            if a.poll():
                status, val = a.recv()
                p.join()
                results[c] = val if status == "ok" else {"error": val}
                running.remove(item)
            elif time.time() - t0 > timeout:
                p.terminate()
                p.join()
                results[c] = {"error": f"timeout>{timeout}s"}
                running.remove(item)
            elif not p.is_alive():
                results[c] = {"error": "worker died"}
                running.remove(item)
        time.sleep(0.2)
    return results


# --------------------------------------------------------------------- trace-level analysis
def support_source(*, m, j, q, n_extra, c_min, c_max, rng):
    """periodic_source with swap pairs drawn from the first j registers only.

    Same RNG call order as sb_generator.periodic_source; the only change is that
    _balanced_pairs receives j instead of m.  future / resets stay length m.
    For j == m this is identical to sb_generator.periodic_source (checked in main()).
    """
    extra = rng.randint(max(3, n_extra), n_extra)
    requested = SBG._sample_constant_count(c_min, c_max, rng)
    n_const = min(requested, extra - 3)
    n_ref = extra - n_const
    n_swap = n_ref // 3
    n_scratch = n_ref - 3 * n_swap
    tmp = m
    init = SBG._initial_values(m, q, rng)
    state = [*init, rng.randrange(q)]
    order = list(range(m + 1))
    rng.shuffle(order)
    ops = [("const", v, state[v]) for v in order]
    pairs = SBG._balanced_pairs(j, n_swap, rng)
    future = [0] * m
    for a, b in pairs:
        future[a] += 1
        future[b] += 1
    for _ in range(n_scratch):
        op = ("copy", tmp, rng.randrange(m))
        ops.append(op)
        SBG._apply(op, state)
    inj = SBG._injection_counts(n_const, n_swap)
    resets = [0] * m
    for i, (a, b) in enumerate(pairs):
        for _ in range(inj[i]):
            op = SBG._choose_reset(state, m, q, future, resets, rng)
            ops.append(op)
            SBG._apply(op, state)
            resets[op[1]] += 1
        for op in (("copy", tmp, a), ("copy", a, b), ("copy", b, tmp)):
            ops.append(op)
            SBG._apply(op, state)
        future[a] -= 1
        future[b] -= 1
    return {"ops": tuple(ops), "pairs": pairs, "init": init, "state": tuple(state),
            "n_swap": n_swap, "n_scratch": n_scratch, "n_const": n_const}


def trace_stats(m, j, N, cfg, ntr, q=127):
    c_min, c_max = cfg
    perm_ctr = Counter()
    perms = []
    outside_reset = total_reset = 0
    init_pos_hit = np.zeros(m)
    outside_changed = 0
    n_swaps = Counter()
    part = np.zeros(m)
    resets_per_trace = []
    for t in range(ntr):
        rng = random.Random(1000003 * m + 9176 * j + 131 * N + 17 * c_max + t)
        tr = support_source(m=m, j=j, q=q, n_extra=N, c_min=c_min, c_max=c_max, rng=rng)
        n_swaps[tr["n_swap"]] += 1
        for a, b in tr["pairs"]:
            part[a] += 1
            part[b] += 1
        n_init_block = m + 1
        nr = 0
        for kind, tgt, _ in tr["ops"][n_init_block:]:
            if kind == "const":
                total_reset += 1
                nr += 1
                if tgt >= j:
                    outside_reset += 1
        resets_per_trace.append(nr)
        st = tr["state"]
        for i in range(m):
            init_pos_hit[i] += (st[i] == tr["init"][i])
        for i in range(j, m):
            if st[i] != tr["init"][i]:
                outside_changed += 1
        if c_max == 0:
            p = list(range(m))
            for a, b in tr["pairs"]:
                p[a], p[b] = p[b], p[a]
            tp = tuple(p)
            perm_ctr[tp] += 1
            perms.append(tp)
    out = {
        "m": m, "j": j, "N": N, "c_range": [c_min, c_max], "traces": ntr, "q": q,
        "n_swap_distribution": {str(k): v for k, v in sorted(n_swaps.items())},
        "mean_swaps_per_supported_register": float(part[:j].mean() / ntr),
        "outside_registers_in_any_swap": float(part[j:].sum()),
        "reset_ops_total": total_reset,
        "reset_ops_on_outside_registers": outside_reset,
        "mean_resets_per_trace": float(np.mean(resets_per_trace)),
        "outside_register_final_differs_from_initial_count": outside_changed,
        "init_predictor_token_acc_per_position": [round(float(x / ntr), 4) for x in init_pos_hit],
        "init_predictor_token_acc_mean_over_m": round(float(init_pos_hit.sum() / ntr / m), 4),
    }
    if c_max == 0:
        modal, mc = perm_ctr.most_common(1)[0]
        parity_even = 0
        for tp in perm_ctr:
            seen, sgn = set(), 1
            for s0 in range(m):
                if s0 in seen:
                    continue
                k, x = 0, s0
                while x not in seen:
                    seen.add(x)
                    x = tp[x]
                    k += 1
                sgn *= -1 if k % 2 == 0 else 1
            if sgn == 1:
                parity_even += perm_ctr[tp]
        mt = float(np.mean([sum(1 for i in range(m) if p[i] == modal[i]) / m for p in perms]))
        out.update({
            "distinct_final_permutations": len(perm_ctr),
            "max_possible_group_order_j_factorial": math.factorial(j),
            "fraction_even_permutations": round(parity_even / ntr, 4),
            "modal_permutation_frequency": round(mc / ntr, 4),
            "modal_permutation_token_acc": round(mt, 4),
            "identity_permutation_frequency": round(perm_ctr[tuple(range(m))] / ntr, 4),
        })
    return out


def _trace_worker(args):
    return trace_stats(*args)


def regression_checks():
    out = {}
    bad = 0
    n = 0
    for m in (4, 5, 6):
        for q in (127, 5):
            for N in (8, 16, 32):
                for cfg in ((0, 0), (1, 64)):
                    for t in range(40):
                        seed = (m * 1000 + q) * 7919 + N * 101 + cfg[1] * 13 + t
                        a = SBG.periodic_source(m=m, q=q, n_extra_min=N, n_extra_max=N,
                                                ap_const_min=cfg[0], ap_const_max=cfg[1],
                                                rng=random.Random(seed))
                        b = support_source(m=m, j=m, q=q, n_extra=N, c_min=cfg[0], c_max=cfg[1],
                                           rng=random.Random(seed))
                        n += 1
                        if a.operations != b["ops"] or a.final_state != b["state"]:
                            bad += 1
    out["j_equals_m_vs_sb_generator"] = {"compared": n, "mismatches": bad}
    return out


def uniform_pair_probe(j, n_swap, ntr=5000):
    """Counterfactual: pairs drawn uniformly (no balance rule); distinct final permutations of S_j."""
    ctr = Counter()
    for t in range(ntr):
        rng = random.Random(77 * j + 13 * n_swap + t)
        p = list(range(j))
        for _ in range(n_swap):
            a, b = rng.sample(range(j), 2)
            p[a], p[b] = p[b], p[a]
        ctr[tuple(p)] += 1
    return {"j": j, "n_swap": n_swap, "traces": ntr, "distinct_final_permutations": len(ctr),
            "j_factorial": math.factorial(j)}


def build_rows(res, tstats):
    ts = {(t["m"], t["j"], t["N"]): t for t in tstats if t["c_range"] == [0, 0]}
    rows = []
    for m in (5, 6):
        for j in range(2, m + 1):
            L = res[(m, j, "letter", "all")]
            Mc = res[(m, j, "macro", "all")]
            A = res[(m, j, "atomic", "all")]
            t16, t32 = ts[(m, j, 16)], ts[(m, j, 32)]
            ser = L["largest_derived_series"]
            note = (
                f"letter |M|={L['monoid_size']}, macro |M|={Mc['monoid_size']} (|V|=2 기호적, scratch=all); "
                f"가역 원소(모든 R={m+1}개 레지스터를 순열하는 튜플) letter={L['n_units_bijective_tuples']}, "
                f"macro={Mc['n_units_bijective_tuples']}, atomic={A['n_units_bijective_tuples']}; "
                f"최대 부분군 위수 {L['largest_maximal_subgroup_order']} = {j}!, derived series {ser}; "
                f"위수 60 이상 H-class {L['classes_with_order_ge_60']}개 전부 "
                f"{'가해' if L['all_maximal_subgroups_solvable'] else '비가해 포함 (위수 ' + ','.join(L['nonsolvable_class_orders']) + ')'}; "
                f"c=0 실현 서로 다른 최종 순열 N=16: {t16['distinct_final_permutations']}, "
                f"N=32: {t32['distinct_final_permutations']} (10,000 trace)."
            )
            rows.append({
                "m": m, "j": j, "basis": "enumerated",
                "group": f"S_{j}",
                "group_order": str(L["largest_maximal_subgroup_order"]),
                "solvable": "yes" if L["all_maximal_subgroups_solvable"] else "no",
                "largest_group_is_S_j": bool(L["largest_group_name"] == f"S_{j}"),
                "derived_series_largest": ser,
                "unit_group_order_letter_macro": L["n_units_bijective_tuples"],
                "unit_group_order_atomic": A["n_units_bijective_tuples"],
                "monoid_size_letter": L["monoid_size"], "monoid_size_macro": Mc["monoid_size"],
                "monoid_size_atomic": A["monoid_size"],
                "realised_final_perms_N16_c0": t16["distinct_final_permutations"],
                "realised_final_perms_N32_c0": t32["distinct_final_permutations"],
                "note": note,
            })
    return rows


GENERATOR_SUPPORT = (
    "기존 파라미터로는 불가능. sb_generator.periodic_source(m, q, n_extra_min, n_extra_max, ap_const_min, ap_const_max, rng)에는 "
    "swap 지지집합 인자가 없고(l.102-104), 원본 allenai/state-bench @ 5b333b8d의 _generate_swap_periodic_source "
    "(generation.py l.238-349)도 마찬가지다. _balanced_swap_pairs의 호출처는 generation.py l.287 한 곳이고 "
    "num_variables를 그대로 넘긴다(저장소 src/ 전체 grep, 'support' 인자 없음). config.py의 condition에도 "
    "num_variables 외에 swap 범위를 정하는 필드가 없다. "
    "최소 추가(구현하지 않음): (a) 인자 swap_support: int | None = None. None이면 현재 동작. 값 j이면 "
    "_balanced_pairs(j, n_swap, rng)로 호출한다(쌍은 레지스터 0..j-1 안에서만 나온다). future(길이 m)·resets(길이 m)·"
    "_initial_values(m, q)·scratch load rng.randrange(m)·초기 const 블록은 m 그대로 둔다. j = m이면 RNG 호출 순서가 "
    "원본과 같아야 하며 이 스크립트의 support_source가 j = m에서 sb_generator.periodic_source와 1,440쌍 중 불일치 0임을 "
    "확인했다(regression). (b) 상수 reset을 바깥 레지스터에도 주려면 별도 인자 reset_scope = 'all'이 필요하다: "
    "_choose_reset이 max(future)인 레지스터만 고르므로(sb_generator.py l.81-85) 바깥 레지스터(future = 0)는 "
    "reset 대상이 될 수 없고, 측정에서도 reset 76,000-120,000개 중 바깥 레지스터 대상은 0개였다. c = 0이면 reset이 "
    "아예 없으므로 EXP-027 설정에서는 (b)가 필요 없다. (c) 선택: _balanced_pairs의 균형 규칙이 j가 짝수일 때 "
    "한 round를 완전 매칭으로 만들어 실현 곱을 좁힌다(caveat 2). 실현 군을 키우려면 pair_policy = 'uniform' 같은 별도 인자가 "
    "필요하다(uniform_pair_probe 참고). p1_equiv_check.py는 swap_support = None에서 불일치 0을 유지해야 한다."
)

CAVEATS = [
    "[가역 원소의 정의] P1 모델(TMP 부작용을 포함한 letter·macro)에서 가역 원소는 항등원 하나뿐이다(열거: 모든 m, j, 모든 알파벳에서 bijective 튜플 1개). 추측문의 'group of units = S_j'는 swap을 TMP 부작용 없는 순열로 취급하는 atomic 모델에서만 문자 그대로 성립하고(열거: 2, 6, 24, 120, 720), P1 모델에서 S_j는 가장 큰 최대 부분군(idempotent의 H-class)이다(EXP-027 §1.4 l.140이 m = 4에서 이미 같은 구분을 한다). 이 구분은 군의 크기·가해성 값에는 영향이 없다.",
    "[실현 군 대 알파벳 군] 연산자 크기 2, 6, 24, 120, 720은 알파벳이 만들 수 있는 군이고 trace가 쓰는 군이 아니다. 저장소의 균형 쌍 규칙(sb_generator.py l.56-68, 원본 generation.py l.417-446)은 j가 짝수이면 j/2개 swap마다 완전 매칭(서로 겹치지 않는 호환들)을 만든다(j = 4면 두 swap마다). c = 0, j = 4에서 N = 32(swap 10개)의 최종 순열은 4개뿐이고(Klein 4-군, 가환) N = 16(swap 5개)은 12개다. 실현된 서로 다른 최종 순열 수(10,000 trace, c = 0): j = 2: 1/1, j = 3: 3/3, j = 4: 12/4, j = 5: 30/60 (m = 5, 6 동일), j = 6: 240/360 (N = 16/32). 따라서 j = 4 대 j = 5 구간은 '가해 S_4 대 비가해 S_5'만 바꾸는 것이 아니라 실현 곱이 가환(V4 계열)에서 A_5 전체로 바뀌는 구간이기도 하다. 이것이 이 사다리가 보기보다 나쁜 통제인 가장 큰 이유다.",
    "[j = 2, 3은 사실상 상수 라벨] c = 0에서 swap 수는 N으로 정해지고(N = 16: 5개, N = 32: 10개) j = 2의 쌍은 항상 (0,1)이라 최종 순열이 N만의 함수다(실현 순열 1개, 최빈 순열 예측기의 token 정확도 1.0). j = 3은 순열이 3개(모두 같은 홀짝)이고 최빈 순열 예측기의 지지 위치 정확도는 약 1/3이다. 이 예측기는 연산열을 읽지 않는다. 즉 j = 2에서는 연산을 읽지 않고 N만 알아도 정답이 나온다.",
    "[바깥 레지스터는 한 번만 쓰인다] 지지집합 밖 레지스터는 초기 const 한 번 뒤에 읽히기만 한다. 최종값 = 초기값이므로 'init'(첫 대입 복사) 예측기가 바깥 위치에서 정확도 1.0이다(측정: 최종값이 초기값과 다른 바깥 레지스터 0건). 따라서 (i) m개 위치 모두 맞춰야 하는 exact 지표는 j개 위치의 exact로 환원되지만, chance 수준은 m이 아니라 j 기준으로 바뀐다; (ii) token 정확도 바닥이 (m − j)/m만큼 올라간다(init 예측기 평균: m = 5, N = 16, c = 0에서 j = 2: 0.60, j = 5: 0.20); (iii) j를 키울 때 출력 길이(m)는 그대로이므로 '출력이 길어져서 어렵다'는 요인은 분리되지만 '쉬운 위치의 비율'이 j와 함께 움직인다.",
    "[reset은 바깥 레지스터에 가지 않는다] '상수 reset은 모든 레지스터에' 라는 조건은 현재 생성기 로직과 맞지 않는다. _choose_reset이 앞으로 swap이 가장 많이 남은 레지스터를 고르므로(l.81-85) 지지집합 밖(future = 0)은 대상이 못 된다. 측정: c = 1-64에서 reset 약 76,000-120,000개 중 바깥 대상 0개. c = 0(EXP-027 설정)에서는 reset 자체가 없다.",
    "[letter 알파벳에서 바깥 레지스터가 TMP로 흐르는 경로는 실제 trace에서 비어 있다] scratch load TMP<-u(u 바깥)는 모든 swap 앞에 오고 swap의 첫 단계 TMP<-a가 TMP를 덮어쓴다. 그래서 바깥 레지스터의 값은 어떤 일반 레지스터에도 도달하지 않는다. scratch=all과 scratch=support의 letter monoid 크기는 다르지만(예: m = 6, j = 4: 515,250 대 150,192) 최대 부분군은 같다. 따라서 monoid 크기는 이 통제의 의미 있는 척도가 아니다.",
    "[4 -> 5 단계는 여러 요인을 한 번에 움직인다] 연산자 크기 24 -> 120, 가해성, 실현 곱(가환 -> A_5), 서로 다른 swap 쌍 수(6 -> 10), 레지스터당 swap 참여 횟수(N = 16: 2.5 -> 2.0)가 동시에 바뀐다. m = 5에서 비가해 칸은 j = 5 하나뿐이고 그 칸은 주 실험 G arm 자체다(사다리가 새 정보를 더하는 것은 j <= 4의 가해 칸들이다). m = 6은 j = 5, 6 두 비가해 칸이 있으나 EXP-027의 m = 5 설계와 R(레지스터 수·출력 길이)이 달라진다.",
    "[유한 N 한계] 가해·비가해는 점근적 구분이다. N <= 32에서 swap은 최대 10개라 어떤 칸의 trace도 유한 길이 합성이며, 열거한 monoid 구조가 짧은 N에서 쓰이는 정도는 실현 순열 수(위 caveat 2)로만 알 수 있다. 연산자 크기와 실현 군 크기는 j = 3, 5에서 같은 방향이지만 j = 4에서 어긋난다.",
    "[열거 방식의 한계] |V| = 2 기호적 열거이며 q = 127의 monoid 크기가 아니다(P1과 같은 전제, EXP-027 §1.4 l.106). 최대 부분군의 위수·가해성은 |V| >= 2에서 |V|에 의존하지 않는다는 P1의 논증을 그대로 받아 썼다. 위수 60 미만 H-class는 '위수 60 미만의 군은 가해'라는 사실로 가해로 처리하고 derived series는 위수별 대표 하나에서만 계산했다. 위수 60 이상 H-class는 sympy로 전수 검사했다(perm_rep_check_failed = 0이면 전부 검사됨). m >= 7은 열거하지 않았다.",
    "[재현 및 범위] 이 JSON은 기호적 monoid와 시뮬레이션한 trace 통계만 담고 모델 학습 결과는 담지 않는다. 사다리의 어느 칸이 학습 가능한지는 이 점검으로 알 수 없다.",
]

SOURCES = [
    "tools/statebench_tiers/p1_monoid_census.py l.153-187 (copy/const 기호 표현, letter·macro 알파벳), l.54-97 (group H-class, derived series)",
    "tools/statebench_tiers/sb_generator.py l.56-68 (_balanced_pairs), l.79-89 (_choose_reset), l.102-143 (periodic_source: scratch l.125-128, swap triple l.137-141)",
    "lab/21_experiments/11_lab-experiments/EXP-027_statebench-tier-ladder-two-state.md §1.4 (P1 표, l.97-157), §1.3 l.86-88 (생성기·동등성)",
    "lab/21_experiments/11_lab-experiments/EXP-027_artifacts/p1_monoid_census.json (m = 4, 5 기준값; 이 스크립트의 m = 5, j = 5 칸이 261,425 / 40,507 / [120, 60, 60]을 재현함)",
    "allenai/state-bench @ 5b333b8d426638d8825f40efe8a508c20eb85730, src/state_bench/generation.py l.238-349, l.352-382, l.417-446 (원본 clone을 이 세션에서 직접 읽음; HEAD 해시 일치 확인)",
]

SYMBOLS = {
    "m": "일반 레지스터 수 (TMP 제외), R = m + 1",
    "j": "swap에 참여할 수 있는 일반 레지스터 수, 지지집합 S = {0..j-1}",
    "|V|": "기호적 열거에 쓴 값 알파벳 크기 (= 2)",
    "letter / macro / atomic": "알파벳 수준: 복사 글자 / TMP 부작용을 가진 swap 매크로 / TMP 부작용 없는 순열 호환",
    "scratch=all / support": "scratch load TMP<-a의 원천을 모든 일반 레지스터 / 지지집합으로 제한",
    "H-class": "idempotent e의 최대 부분군 H_e",
    "S_k": "k개 점의 대칭군 (위수 k!)",
    "N": "extra assignment 수 (EXP-027 용어; N = 16 -> swap 5개, N = 32 -> swap 10개)",
    "c": "상수 reset 범위 (c = 0: reset 없음, c = 1-64: 저장소 기본)",
}


# ------------------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--procs", type=int, default=24)
    ap.add_argument("--traces", type=int, default=10000)
    ap.add_argument("--skip-heavy", action="store_true", help="skip letter/macro cells with m=6")
    args = ap.parse_args()
    t_start = time.time()

    cells = []
    for m in (5, 6):
        for j in range(2, m + 1):
            for level in ("atomic", "macro", "letter"):
                for scratch in (("all",) if level == "atomic" else ("all", "support")):
                    if args.skip_heavy and m == 6 and level != "atomic":
                        continue
                    cells.append((m, j, level, scratch))
    # largest first so the long jobs start early
    cells.sort(key=lambda c: -(c[0] * 10 + c[1]))
    res = run_cells(cells, args.timeout, args.procs)
    for c in sorted(res):
        r = res[c]
        if "error" in r:
            print(f"[{c}] ERROR {r['error']}", flush=True)
        else:
            print(f"[enumerated] m={c[0]} j={c[1]} {c[2]:6s} scratch={c[3]:7s} |M|={r['monoid_size']} "
                  f"units={r['n_units_bijective_tuples']} largest={r['largest_maximal_subgroup_order']} "
                  f"{r['largest_group_name']} series={r['largest_derived_series']} "
                  f"all_solvable={r['all_maximal_subgroups_solvable']} ({r['elapsed_s']}s)", flush=True)

    # trace-level
    jobs = [(m, j, N, cfg, args.traces) for m in (5, 6) for j in range(2, m + 1)
            for N in (16, 32) for cfg in ((0, 0), (1, 64))]
    with mp.get_context("fork").Pool(min(args.procs, len(jobs))) as pool:
        tstats = pool.map(_trace_worker, jobs)
    reg = regression_checks()

    probe = [uniform_pair_probe(j, ns) for j in range(2, 7) for ns in (5, 10)]
    rows = build_rows(res, tstats)
    out = {
        "meta": {
            "title": "P1b swap-support census",
            "date": "2026-11-11",
            "type": "artifact",
            "status": "DRAFT",
            "author_model": "claude-sonnet-5-5",
            "script": "tools/statebench_tiers/p1b_swap_support_census.py",
            "reproduce": 'CUDA_VISIBLE_DEVICES="" uv run python tools/statebench_tiers/p1b_swap_support_census.py --json <out>',
            "elapsed_s": round(time.time() - t_start, 1),
        },
        "symbols": SYMBOLS,
        "sources": SOURCES,
        "rows": rows,
        "generator_support": GENERATOR_SUPPORT,
        "caveats": CAVEATS,
        "uniform_pair_probe_c0": probe,
        "cells": {f"m{c[0]}_j{c[1]}_{c[2]}_scratch-{c[3]}": res[c] for c in sorted(res)},
        "trace_stats": tstats,
        "regression": reg,
    }
    Path(args.json).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {args.json} in {out['meta']['elapsed_s']}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
