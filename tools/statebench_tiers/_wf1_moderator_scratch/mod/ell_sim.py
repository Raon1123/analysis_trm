"""Moderator unfrozen CPU sim: DiD of restricted log2 first-passage (ell) between tiers.
Per run: never-learn with prob q[T,v] -> ell = log2(Tmax); else log2 tau ~ N(mu[T,v], sd^2), rounded up to the
eval grid, capped at log2(Tmax). B_T = mean ell(noz) - mean ell(z); D = B_G - B_S.
Rule: percentile bootstrap (cells resampled independently, B=1000). POS: lo(D)>0 and lo(B_G)>0 (2.5% one-sided each);
REV: hi(D)<0; NEG: hi(D)<m and not POS/REV; INC otherwise; UNINF if a tier fails the guard
(pooled learned fraction <0.25, or pooled median tau < 3 eval intervals). Not freeze evidence."""
import numpy as np, sys, json
rng = np.random.default_rng(20261111)
TMAX, EV = 40039, 977
LT = np.log2(TMAX)
def draw(n, q, mu, sd):
    learn = rng.random(n) >= q
    lt = rng.normal(mu, sd, n)
    tau = np.ceil(2 ** lt / EV) * EV
    tau = np.where(learn & (tau <= TMAX), tau, np.inf)
    ell = np.log2(np.minimum(tau, TMAX))
    return ell, tau
def boot_mean(x, B):
    idx = rng.integers(0, len(x), (B, len(x)))
    return x[idx].mean(1)
def rule(cells, m=0.5, B=1000):
    # cells: dict (T,v)->(ell, tau)
    for T in 'SG':
        tau = np.concatenate([cells[(T, 'z')][1], cells[(T, 'n')][1]])
        if np.isfinite(tau).mean() < 0.25 or np.median(tau) < 3 * EV:
            return 'UNINF'
    bm = {k: boot_mean(v[0], B) for k, v in cells.items()}
    BG = bm[('G', 'n')] - bm[('G', 'z')]; BS = bm[('S', 'n')] - bm[('S', 'z')]; D = BG - BS
    lo, hi = np.quantile(D, [0.025, 0.975]); loG = np.quantile(BG, 0.025)
    if lo > 0 and loG > 0: return 'POS'
    if hi < 0: return 'REV'
    if hi < m: return 'NEG'
    return 'INC'
def scen(n, p, reps):
    out = {}
    for _ in range(reps):
        cells = {k: draw(n, *v) for k, v in p.items()}
        c = rule(cells); out[c] = out.get(c, 0) + 1
    return {c: round(v / reps, 3) for c, v in sorted(out.items())}
sd = 0.45
def P(muS, muG, bS, bG, qSn=0., qSz=0., qGn=0., qGz=0.):
    # mu = noz mean log2 tau; z is faster by b (log2 units)
    return {('S', 'n'): (qSn, muS, sd), ('S', 'z'): (qSz, muS - bS, sd), ('G', 'n'): (qGn, muG, sd), ('G', 'z'): (qGz, muG - bG, sd)}
S = {
 'null_b0_same':      P(14.0, 14.0, 0, 0),
 'null_b1_same':      P(14.0, 14.0, 1, 1),
 'null_b1_Seasy':     P(12.5, 14.5, 1, 1),
 'null_b1_censorG':   P(13.0, 15.0, 1, 1, qGn=0.3, qGz=0.3),
 'null_Sfast_b0.5':   P(11.5, 14.0, .5, .5),
 'alt_D0.5':          P(14.0, 14.0, 0.5, 1.0),
 'alt_D1.0':          P(14.0, 14.0, 0.5, 1.5),
 'alt_D1.0_Seasy':    P(12.5, 14.5, 0.5, 1.5),
 'alt_learnG_only':   P(13.0, 14.5, 0.5, 0.5, qGn=0.6, qGz=0.1),
 'rev_D-1.0':         P(14.0, 14.0, 1.5, 0.5),
 'zhurtsS_D+1':       P(14.0, 14.0, -1.0, 0.0),
 'Gfloor':            P(13.0, 15.5, 1, 1, qGn=0.8, qGz=0.8),
}
reps = int(sys.argv[1]) if len(sys.argv) > 1 else 600
res = {}
for n in [10, 15, 20]:
    res[n] = {}
    print('n =', n)
    for k, p in S.items():
        r = scen(n, p, reps); res[n][k] = r
        print(f'  {k:18s} {r}')
json.dump({str(k): v for k, v in res.items()}, open(sys.argv[2], 'w'), indent=1)
