"""Moderator unfrozen CPU sim of the difficulty-matched binary DiD rule (03_DESIGN_DIALECTIC §5-6).
Not freeze evidence. CUDA not used."""
import numpy as np, itertools, json, sys
rng = np.random.default_rng(20261111)
D = 4000  # posterior draws
def rule(k, n, m=0.30, cP=0.975, cN=0.95, guard=(0.10, 0.90)):
    # k: successes (Sz, Sn, Gz, Gn), n per cell
    a = 1 + np.array(k); b = 1 + n - np.array(k)
    post = rng.beta(a[:, None], b[:, None], size=(4, D))
    th = (post[2] - post[3]) - (post[0] - post[1])
    pS = (k[0] + k[1]) / (2 * n); pG = (k[2] + k[3]) / (2 * n)
    if not (guard[0] <= pS <= guard[1] and guard[0] <= pG <= guard[1]):
        return 'UNINF'
    Ppos = (th > 0).mean(); PG = (post[2] > post[3]).mean(); Prev = (th < 0).mean(); Pneg = (th < m).mean()
    if Ppos >= cP and PG >= cP:
        return 'POS'
    if Prev >= cP:
        return 'REV'
    if Pneg >= cN:
        return 'NEG'
    return 'INC'
def run(p, n, reps):
    out = {}
    for _ in range(reps):
        k = rng.binomial(n, p)
        c = rule(k, n); out[c] = out.get(c, 0) + 1
    return {c: v / reps for c, v in sorted(out.items())}
codes = ['POS', 'REV', 'NEG', 'INC', 'UNINF']
def fmt(r): return ' '.join(f"{c}={r.get(c,0):.3f}" for c in codes)
reps = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
ns = [10, 15, 20, 25, 30, 40]
# null grid: theta = 0
nulls = []
for pSn, pGn, d in itertools.product([0.1, 0.3, 0.5, 0.7], [0.1, 0.3, 0.5, 0.7], [0.0, 0.2, 0.4]):
    if pSn + d <= 0.95 and pGn + d <= 0.95:
        nulls.append((round(pSn + d, 2), pSn, round(pGn + d, 2), pGn))
alts = {'th0.3': (0.5, 0.5, 0.65, 0.35), 'th0.4': (0.5, 0.5, 0.7, 0.3), 'th0.6': (0.5, 0.5, 0.8, 0.2), 'th0.8': (0.5, 0.5, 0.9, 0.1),
        'null_mid': (0.5, 0.5, 0.5, 0.5), 'null_d0.4': (0.7, 0.3, 0.7, 0.3), 'th0.2': (0.5, 0.5, 0.6, 0.4),
        'rev0.6': (0.8, 0.2, 0.5, 0.5), 'zhurtsS': (0.2, 0.8, 0.5, 0.5),
        'null_Snearceil': (0.95, 0.75, 0.6, 0.4), 'null_pool0.2': (0.3, 0.1, 0.3, 0.1), 'null_pool0.8': (0.9, 0.7, 0.9, 0.7),
        'logit1.5_unmatched': (0.913, 0.7, 0.658, 0.3), 'logit1.5_matched': (0.7, 0.35, 0.7, 0.35),
        'alt0.6_Snearceil': (0.95, 0.85, 0.8, 0.2), 'alt0.6_pool0.3': (0.3, 0.3, 0.6, 0.0),
        'uninf_Sceil': (1.0, 0.95, 0.8, 0.2), 'uninf_Gfloor': (0.5, 0.5, 0.08, 0.02)}
res = {}
for n in ns:
    worst = max(((run(p, n, reps // 4).get('POS', 0) + run(p, n, reps // 4).get('POS-SMALL', 0)) / 2, p) for p in nulls)
    res[n] = {'worst_null_POS': worst}
    print(f"n={n} worst null P(POS)={worst[0]:.3f} at {worst[1]}")
    for name, p in alts.items():
        r = run(p, n, reps); res[n][name] = r
        print(f"  {name:10s} {p}  {fmt(r)}")
json.dump({str(k): v for k, v in res.items()}, open(sys.argv[2] if len(sys.argv) > 2 else '/dev/null', 'w'), indent=1, default=str)
