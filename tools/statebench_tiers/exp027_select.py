"""EXP-027 v3 selection rule R0..R7, LATE flag, n-gate, and the pilot budget plan.

Status: DRAFT, author_model claude-sonnet-5-5.  Not frozen.  No GPU.

Sources: 03_DESIGN_DIALECTIC section 6 (rules), section 5 (pilot), amended by 00_DESIGN_LOG section 7
(S pilot has two seeds -> pilot = 17 runs; n-gate 20 / 15; no control arm), section 8 (seeds spaced by 1000,
executed step counts), section 9.2 (budget rule: body <= 130 GPU-h, extension cap 60 GPU-h) and 9.3
(R6 exception closed, R0 without an A row).

Seeds: pilot and extension seeds are 9101000 and 9102000, confirmatory seeds are 1000 * i (i = 1..n), the smoke
seed is 0.  The data order of a run is drawn from seed + (eval block index), at most 41 blocks, so seeds closer
than 42 apart share shifted data orders (SEED_MIN_GAP).

Structural variant blindness
----------------------------
The selection function accepts only a `PooledTable`.  `pool_rows` is the only constructor from run rows and it
validates and then DROPS the variant and seed columns: a pooled cell keeps run counts, sorted value tuples and
maxima, so no variant-level information can reach `select`.  `select(rows)` raises TypeError.

Input row (one per run): tier ('A'|'S'|'G'), N, variant ('z'|'noz'), seed, Y (0/1), BSE, tau (optimizer step,
None or inf if never), flags (iterable of 'graded' | 'collapse' | 'pretakeoff'; authoritative, produced by the
outcome extractor), max (max test exact over the run), optional stage ('pilot' default | 'ext').
"""
from __future__ import annotations

import dataclasses
import math
from typing import Iterable, Mapping, Optional

# ---------------------------------------------------------------- frozen constants
G_MENU = (12, 16, 24, 32, 48)
S_MENU = (8, 16, 24, 32, 64)
TIERS = ("A", "S", "G")
PILOT_NS = (16, 32)
EXT_LONG = {"G": 48, "S": 64}
EXT_SHORT = {"G": 12, "S": 8}
EXT_MID = 24
SEED_SPACING = 1000                              # design-log 8: seed spacing of confirmatory seeds
SEED_MIN_GAP = 42                                # eval blocks per run <= 41, so seeds must differ by >= 42
PILOT_SEEDS = (9101000, 9102000)
EXT_SEEDS = PILOT_SEEDS
SMOKE_SEED = 0
EXT_VARIANTS = ("z", "noz")


def conf_seeds(n: int) -> tuple:
    """Confirmatory seeds 1000 * i for i = 1..n."""
    return tuple(SEED_SPACING * i for i in range(1, n + 1))


GRADED_FRAC_NUM, GRADED_FRAC_DEN = 1, 4          # R1: >= 25 % of G,S pilot runs graded
R0_MAX = 0.20                                    # R0: all G,S runs max < 0.20
BSE_MIX_LO, BSE_MIX_HI = 0.20, 0.80              # R1: BSE-mean MIX interval
TIE_DIGITS = 12                                  # scores are rounded to 12 decimals before tie comparison
BOUNDARY_EPS = 1e-6                              # tolerance on the closed BSE-MIX interval; eval values are float32
                                                 # (0.800 is logged as 0.800000012), grid 0.001 -> 1e-6 is safe
TRAIN_N, EVAL_EPOCHS = 100_000, 20               # one eval per 20 epochs
GLOBAL_BATCH = 1024                              # EXP-027 amendment 1: every cell runs global_batch_size=1024
BATCH = GLOBAL_BATCH                             # alias kept for older callers
# puzzle_dataset._iter_train (puzzle_dataset.py:201-234) drops the last partial batch of every 20-epoch chunk, so
# one eval block runs floor(20 * 100000 / GLOBAL_BATCH) optimizer steps (1953 at 1024) and evals land at
# EVAL_STEP * j steps.  Amendment 1 replaces v3's 2048 / 976.


def block_steps(eval_epochs: int = EVAL_EPOCHS, train_rows: int = TRAIN_N, batch: int = GLOBAL_BATCH) -> int:
    """Optimizer steps of one eval block: floor(eval_epochs * train_rows / batch)."""
    return (eval_epochs * train_rows) // batch


EVAL_STEP = block_steps()                        # 1953 optimizer steps per eval block
N_CAP, B_CONF = 30, 600.0
N_CONFIRM, N_QUALIFIED = 20, 15
T_P_MENU_EPOCHS = (820, 620, 420)
PILOT_BUDGET = 130.0                             # design-log 9.2: smoke + 17 runs
EXT_BUDGET = 60.0                                # design-log 9.2: extension (4 runs) has its own cap


def steps_for_epochs(epochs: int) -> int:
    """Executed optimizer steps of a run of `epochs` epochs: one eval block (20 epochs) is EVAL_STEP = 1953 steps.
    820 / 620 / 420 epochs -> 80073 / 60543 / 41013 steps; the 40-epoch smoke run -> 3906.  The one place that
    converts epochs to steps; cost and judge import it."""
    if not isinstance(epochs, int) or epochs <= 0 or epochs % EVAL_EPOCHS:
        raise ValueError(f"epochs must be a positive multiple of {EVAL_EPOCHS}, got {epochs!r}")
    return (epochs // EVAL_EPOCHS) * EVAL_STEP


def n_evals_for_epochs(epochs: int) -> int:
    """Number of evals of a run of `epochs` epochs (one per 20 epochs)."""
    steps_for_epochs(epochs)
    return epochs // EVAL_EPOCHS

QUAL_R4 = "N 다름: S와 G의 확증 N이 서로 다르다 (단계와 N이 섞임)"
QUAL_R5 = "G 2셀(z, noz)만 확증한다; 상호작용 θ는 이 예산에서 측정하지 않는다"
QUAL_BSE = "MISFIT: 1차 결과를 Y 대신 BSE로 하고 판정은 seed bootstrap 95% 구간으로 한다"
# Source: EXP-027_artifacts/p4_v3_c3_exact.json (exact enumeration, c_P = 0.975): null_mid = both tiers p = 0.5,
# shift d = 0; n = 15..19 gives 0.324..0.389; across all 41 null rows 0.085..0.492.
QUAL_NGATE = ("null_mid 행(두 단계 p = 0.5, 공통 이동 0)에서 NEG 도달 확률 0.32–0.39 "
              "(n 15–19, c_P = 0.975, 정확 계산; null 41개 행 전체로는 0.09–0.49)")

_FLAGS = {"graded", "collapse", "pretakeoff"}


# ---------------------------------------------------------------- pooled table (variant is dropped here)
@dataclasses.dataclass(frozen=True)
class PooledCell:
    tier: str
    N: int
    n: int                    # runs, all stages
    n_y1: int
    bse: tuple                # sorted BSE values of all runs
    tau: tuple                # sorted tau of all runs (inf when never)
    n_y0: int
    n_pretakeoff_y0: int
    n_pilot: int              # pilot-stage runs
    n_graded_pilot: int
    max_pilot: float          # max over pilot-stage runs of run max (-inf when none)
    n_ext: int


@dataclasses.dataclass(frozen=True)
class PooledTable:
    cells: tuple              # tuple[PooledCell], sorted by (tier, N)

    def get(self, tier: str, N: int) -> Optional[PooledCell]:
        for c in self.cells:
            if c.tier == tier and c.N == N:
                return c
        return None

    def tier_cells(self, tier: str) -> list:
        return [c for c in self.cells if c.tier == tier]

    @property
    def has_ext(self) -> bool:
        return any(c.n_ext > 0 for c in self.cells)


def pool_rows(rows: Iterable[Mapping]) -> PooledTable:
    """Validate run rows and pool them by (tier, N).  Variant and seed are validated, then dropped."""
    groups: dict = {}
    for r in rows:
        tier, N = r["tier"], int(r["N"])
        if tier not in TIERS:
            raise ValueError(f"unknown tier {tier!r}")
        if tier == "G" and N not in G_MENU or tier == "S" and N not in S_MENU:
            raise ValueError(f"N={N} outside the {tier} menu")
        if r["variant"] not in ("z", "noz"):
            raise ValueError(f"unknown variant {r['variant']!r}")
        y = int(r["Y"])
        if y not in (0, 1):
            raise ValueError("Y must be 0 or 1")
        flags = set(r.get("flags") or ())
        if not flags <= _FLAGS:
            raise ValueError(f"unknown flags {flags - _FLAGS}")
        stage = r.get("stage", "pilot")
        if stage not in ("pilot", "ext"):
            raise ValueError(f"unknown stage {stage!r}")
        tau = r.get("tau")
        tau = math.inf if tau is None else float(tau)
        bse, mx = float(r["BSE"]), float(r["max"])
        if not (math.isfinite(bse) and math.isfinite(mx)):
            raise ValueError(f"non-finite BSE/max in a run row of ({tier}, {N}): BSE={bse}, max={mx}")
        g = groups.setdefault((tier, N), [])
        g.append((y, bse, tau, mx, flags, stage))
    cells = []
    for (tier, N) in sorted(groups):
        g = groups[(tier, N)]
        pil = [x for x in g if x[5] == "pilot"]
        y0 = [x for x in g if x[0] == 0]
        cells.append(PooledCell(
            tier=tier, N=N, n=len(g), n_y1=sum(x[0] for x in g),
            bse=tuple(sorted(x[1] for x in g)), tau=tuple(sorted(x[2] for x in g)),
            n_y0=len(y0), n_pretakeoff_y0=sum(1 for x in y0 if "pretakeoff" in x[4]),
            n_pilot=len(pil), n_graded_pilot=sum(1 for x in pil if "graded" in x[4]),
            max_pilot=max((x[3] for x in pil), default=-math.inf),
            n_ext=len(g) - len(pil)))
    return PooledTable(tuple(cells))


# ---------------------------------------------------------------- states
def _mean_bse(c: PooledCell) -> float:
    return math.fsum(c.bse) / c.n


def _state(c: PooledCell, mode: str) -> str:
    if mode == "Y":
        return "FLOOR" if c.n_y1 == 0 else ("CEIL" if c.n_y1 == c.n else "MIX")
    m = _mean_bse(c)
    if m > BSE_MIX_HI + BOUNDARY_EPS:
        return "CEIL"
    if m < BSE_MIX_LO - BOUNDARY_EPS:
        return "FLOOR"
    return "MIX"


def _dist(c: PooledCell, mode: str) -> float:
    f = c.n_y1 / c.n if mode == "Y" else _mean_bse(c)
    return abs(f - 0.5)


# ---------------------------------------------------------------- result
@dataclasses.dataclass(frozen=True)
class SelectionResult:
    path: str                         # terminal rule 'R0','R2'..'R7' (R2 = extension requested, rerun with ext rows pooled in)
    misfit: bool                      # R1 fired: mode switched to BSE; R1 is a modifier, never terminal
    trace: tuple                      # rules that fired in order, e.g. ('R1', 'R3')
    sub: Optional[str]                # R5: 'SC' | 'SF' | 'SX'; R6: 'default' (no confirmation)
    mode: str                         # 'Y' or 'BSE' (MISFIT)
    confirm: bool
    stat: Optional[str]               # 'theta' | 'G-only' | None
    N_S: Optional[int]
    N_G: Optional[int]
    cells: tuple                      # ((tier, N, variant), ...)
    qualifiers: tuple
    extension: Optional[tuple]        # ((tier, N, ((variant, seed), ...)), ...) or None
    late_flag: bool
    notes: tuple

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _cells4(NS: int, NG: int) -> tuple:
    return (("S", NS, "z"), ("S", NS, "noz"), ("G", NG, "z"), ("G", NG, "noz"))


def _cells2(NG: int) -> tuple:
    return (("G", NG, "z"), ("G", NG, "noz"))


def _pick_min(cells: list, mode: str, score) -> PooledCell:
    # deterministic tie-break: smaller N
    return min(cells, key=lambda c: (round(score(c), TIE_DIGITS), c.N))


# ---------------------------------------------------------------- selection
def select(pooled: PooledTable) -> SelectionResult:
    """Apply R0..R7 (first match) to a variant-pooled table.  R6 (every tested G,S cell at the ceiling) never
    confirms: the first-passage-time exception was closed (design-log 9.3)."""
    if not isinstance(pooled, PooledTable):
        raise TypeError("select() takes a PooledTable (use pool_rows); raw rows carry variant labels")
    gs = [c for c in pooled.cells if c.tier in ("G", "S")]
    if not pooled.tier_cells("G") or not pooled.tier_cells("S"):
        raise ValueError("table needs at least one G cell and one S cell")

    # LATE flag: >= 50 % of Y=0 G runs carry the pre-takeoff flag (informational)
    g_y0 = sum(c.n_y0 for c in pooled.tier_cells("G"))
    g_pre = sum(c.n_pretakeoff_y0 for c in pooled.tier_cells("G"))
    late = g_y0 > 0 and 2 * g_pre >= g_y0

    def out(path, **kw):
        base = dict(sub=None, mode="Y", confirm=False, stat=None, N_S=None, N_G=None, cells=(), qualifiers=(),
                    extension=None, late_flag=late, notes=())
        base.update(kw)
        misfit = base["mode"] == "BSE"
        return SelectionResult(path=path, misfit=misfit, trace=(("R1",) if misfit else ()) + (path,), **base)

    # R0 PIPE: every G,S pilot run has max < 0.20 and no A run has Y = 1.  An absent A row (or an A run excluded
    # for an infrastructure failure) leaves the G,S condition alone (design-log 9.3).
    a = pooled.tier_cells("A")
    gs_pilot = [c for c in gs if c.n_pilot > 0]
    if gs_pilot and sum(c.n_y1 for c in a) == 0 and all(c.max_pilot < R0_MAX for c in gs_pilot):
        return out("R0", notes=("pipeline check; no selection",))

    # R1 MISFIT: >= 25 % of G,S pilot runs graded -> primary outcome BSE
    n_p = sum(c.n_pilot for c in gs)
    n_gr = sum(c.n_graded_pilot for c in gs)
    mode = "BSE" if n_p > 0 and n_gr * GRADED_FRAC_DEN >= n_p * GRADED_FRAC_NUM else "Y"
    quals = (QUAL_BSE,) if mode == "BSE" else ()
    st = {(c.tier, c.N): _state(c, mode) for c in gs}
    has_mix = {t: any(st[(c.tier, c.N)] == "MIX" for c in pooled.tier_cells(t)) for t in ("G", "S")}

    # R2 extension (once): G has priority; both missing -> G only
    if not pooled.has_ext and not (has_mix["G"] and has_mix["S"]):
        tier = "G" if not has_mix["G"] else "S"
        states = [st[(c.tier, c.N)] for c in pooled.tier_cells(tier)]
        if all(s == "CEIL" for s in states):
            N_ext = EXT_LONG[tier]
        elif all(s == "FLOOR" for s in states):
            N_ext = EXT_SHORT[tier]
        else:
            N_ext = EXT_MID
        runs = tuple((v, s) for v in EXT_VARIANTS for s in EXT_SEEDS)
        return out("R2", mode=mode, qualifiers=quals, extension=((tier, N_ext, runs),),
                   notes=(f"extend {tier} to N={N_ext} (states {states})",))

    mix = {t: [c for c in pooled.tier_cells(t) if st[(c.tier, c.N)] == "MIX"] for t in ("G", "S")}
    dist = lambda c: _dist(c, mode)

    # R3 DM-shared
    shared = sorted({c.N for c in mix["G"]} & {c.N for c in mix["S"]})
    if shared:
        def sc(N):
            return round(dist(pooled.get("S", N)) + dist(pooled.get("G", N)), TIE_DIGITS)
        N = min(shared, key=lambda n: (sc(n), n))
        return out("R3", mode=mode, confirm=True, stat="theta", N_S=N, N_G=N, cells=_cells4(N, N),
                   qualifiers=quals)

    # R4 DM-split
    if mix["G"] and mix["S"]:
        cS, cG = _pick_min(mix["S"], mode, dist), _pick_min(mix["G"], mode, dist)
        return out("R4", mode=mode, confirm=True, stat="theta", N_S=cS.N, N_G=cG.N,
                   cells=_cells4(cS.N, cG.N), qualifiers=quals + (QUAL_R4,))

    # R5 SC / SF: S has no MIX, G has MIX
    if mix["G"]:
        cG = _pick_min(mix["G"], mode, dist)
        s_states = {st[(c.tier, c.N)] for c in pooled.tier_cells("S")}
        sub = "SC" if s_states == {"CEIL"} else ("SF" if s_states == {"FLOOR"} else "SX")
        return out("R5", sub=sub, mode=mode, confirm=True, stat="G-only", N_G=cG.N, cells=_cells2(cG.N),
                   qualifiers=quals + (QUAL_R5,))

    # R6: every tested G,S (tier, N) is CEIL (G has no MIX here, so S has none either); no confirmation
    if all(s == "CEIL" for s in st.values()):
        return out("R6", sub="default", mode=mode, notes=("all tested (tier, N) at ceiling; no confirmation",))

    # R7 GX
    return out("R7", mode=mode, notes=("no usable MIX configuration; close with a pilot description",))


def select_rows(rows: Iterable[Mapping]) -> SelectionResult:
    """Convenience: pool (drops variant/seed) then select."""
    return select(pool_rows(rows))


# ---------------------------------------------------------------- n-gate
@dataclasses.dataclass(frozen=True)
class NGateResult:
    n: int
    action: str                  # 'ENQUEUE' | 'ENQUEUE_WITH_QUALIFIER' | 'NO_CONFIRM'
    qualifier: Optional[str]
    t_max_epochs: Optional[int]  # epochs of the (possibly reduced) T_max; None if no confirmation
    reduced: bool


def _action(n: int) -> str:
    return "ENQUEUE" if n >= N_CONFIRM else ("ENQUEUE_WITH_QUALIFIER" if n >= N_QUALIFIED else "NO_CONFIRM")


def n_gate(c_set: float, *, t_p_epochs: int, b_conf: float = B_CONF, n_cap: int = N_CAP,
           c_set_by_epochs: Optional[Mapping[int, float]] = None, c9_passed: bool = False,
           same_state_by_epochs: Optional[Mapping[int, bool]] = None) -> NGateResult:
    """n = min(n_cap, floor(b_conf / c_set)).  n >= 20 enqueue; 15..19 enqueue with a fixed qualifier;
    n < 15: if c9_passed, try shorter T_max in the menu (620, then 420 epochs; only values below t_p_epochs) whose
    selected cells keep the same state; take the first that gives n >= 15; otherwise no confirmation.
    t_p_epochs (the pilot T_p in epochs) has no default: a silent 820 would let a 420-epoch pilot "reduce" to 620."""
    if t_p_epochs not in T_P_MENU_EPOCHS:
        raise ValueError(f"t_p_epochs must be one of {T_P_MENU_EPOCHS}, got {t_p_epochs!r}")
    if not c_set > 0:
        raise ValueError("c_set must be positive")
    n0 = min(n_cap, math.floor(b_conf / c_set))
    if n0 >= N_QUALIFIED:
        a = _action(n0)
        return NGateResult(n0, a, QUAL_NGATE if a == "ENQUEUE_WITH_QUALIFIER" else None, t_p_epochs, False)
    if c9_passed and c_set_by_epochs and same_state_by_epochs:
        for e in sorted((e for e in (620, 420) if e < t_p_epochs), reverse=True):
            if same_state_by_epochs.get(e) and e in c_set_by_epochs:
                cs = c_set_by_epochs[e]
                if not cs > 0:
                    raise ValueError("c_set must be positive")
                n = min(n_cap, math.floor(b_conf / cs))
                if n >= N_QUALIFIED:
                    a = _action(n)
                    return NGateResult(n, a, QUAL_NGATE if a == "ENQUEUE_WITH_QUALIFIER" else None, e, True)
    return NGateResult(n0, "NO_CONFIRM", None, None, False)


# ---------------------------------------------------------------- pilot budget plan (design-log 9.2)
def _interp(points: list, x: float) -> float:
    """Piecewise-linear in tokens through the per-token-count mean of the points; linear extrapolation from the
    nearest segment outside the range; one distinct point -> proportional to tokens."""
    by: dict = {}
    for px, py in points:
        by.setdefault(px, []).append(py)
    pts = sorted((px, math.fsum(v) / len(v)) for px, v in by.items())
    if not pts:
        raise ValueError("no smoke points for this variant")
    if len(pts) == 1:
        return pts[0][1] * x / pts[0][0]
    if x <= pts[0][0]:
        (x0, y0), (x1, y1) = pts[0], pts[1]
    elif x >= pts[-1][0]:
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
    else:
        i = max(j for j in range(len(pts) - 1) if pts[j][0] <= x)
        (x0, y0), (x1, y1) = pts[i], pts[i + 1]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def run_cost(cost_table: Mapping, variant: str, N: int, epochs: int) -> float:
    """G_run = T_max * c_step + n_ev * c_eval + t_0 in GPU-h; c_step, c_eval interpolated on tokens 33 + 2N per variant.
    T_max = steps_for_epochs(epochs) executed optimizer steps."""
    pts = [p for p in cost_table["points"] if p["variant"] == variant]
    tok = 33 + 2 * N
    c_step = _interp([(33 + 2 * p["N"], p["c_step"]) for p in pts], tok)
    c_eval = _interp([(33 + 2 * p["N"], p["c_eval"]) for p in pts], tok)
    t0 = math.fsum(p["t0"] for p in pts) / len(pts)
    return steps_for_epochs(epochs) * c_step + n_evals_for_epochs(epochs) * c_eval + t0


def _runs_cost(cost_table: Mapping, spec: Iterable, epochs: int) -> float:
    return math.fsum(run_cost(cost_table, v, N, epochs) for (_t, N, v) in spec)


def pilot_run_specs(with_s: bool = True, with_a: bool = True) -> list:
    """(tier, N, variant, seed) of the pilot: G 2N x 2v x 2 seeds, S same, A N=16 z first seed -> 17 runs."""
    out = [("G", N, v, s) for N in PILOT_NS for v in EXT_VARIANTS for s in PILOT_SEEDS]
    if with_s:
        out += [("S", N, v, s) for N in PILOT_NS for v in EXT_VARIANTS for s in PILOT_SEEDS]
    if with_a:
        out.append(("A", 16, "z", PILOT_SEEDS[0]))
    return out


EXT_OPTIONS = {"G": (12, 24, 48), "S": (8, 24, 64)}      # extension N by tier; 4 runs each (2 seeds x 2 variants)


def decide_budget(smoke: float, candidates: list, g8_420: float, budget: float = PILOT_BUDGET,
                  ext_cap: float = EXT_BUDGET) -> dict:
    """Design-log 9.2.  candidates: one dict per T_p in T_P_MENU_EPOCHS (descending) with keys epochs, steps,
    pilot_17 (GPU-h of the 17 pilot runs), ext_options ({'G12': GPU-h of 4 runs, ...}).
      body  = smoke + pilot_17;  the largest T_p with body <= budget is chosen (branch FULL);
      the extension (4 runs, once) has its own cap `ext_cap`, reported per option and not part of the body test
      (the cap binds at extension time through `ext_gate`, not through T_p);
      no T_p fits -> G only, 8 runs at 420 epochs (smoke + g8_420 <= budget -> G_ONLY_420, no confirmation);
      otherwise CLOSE_NO_PILOT.
    Returns the decision and `arithmetic`, one line per tested quantity."""
    rows, lines = [], []
    for c in candidates:
        body = smoke + c["pilot_17"]
        feas = {k: v for k, v in c["ext_options"].items() if v is not None}      # None = infeasible (amendment 1)
        infeas = sorted(k for k, v in c["ext_options"].items() if v is None)
        ub_key = max(sorted(feas), key=lambda k: feas[k]) if feas else None
        over = sorted(k for k, v in feas.items() if v > ext_cap)
        row = dict(c, body=body, fits=body <= budget, ext_ub=feas[ub_key] if feas else None, ext_ub_option=ub_key,
                   ext_options_over_cap=over, ext_options_infeasible=infeas, ext_ub_fits_cap=not over)
        rows.append(row)
        lines.append(f"T_p={c['epochs']} epochs ({c['steps']} steps): body = smoke {smoke:.3f} + pilot17 "
                     f"{c['pilot_17']:.3f} = {body:.3f} GPU-h {'<=' if row['fits'] else '>'} {budget:g}; "
                     + (f"extension worst case {ub_key} = {row['ext_ub']:.3f} GPU-h "
                        f"{'<=' if row['ext_ub_fits_cap'] else '>'} {ext_cap:g}" if feas else "no feasible extension option")
                     + (f" (options over the cap: {', '.join(over)})" if over else "")
                     + (f" (infeasible, memory: {', '.join(infeas)})" if infeas else ""))
    out = {"budget": budget, "ext_cap": ext_cap, "smoke_cost": smoke, "candidates": rows}
    for r in rows:
        if r["fits"]:
            lines.append(f"-> FULL: largest T_p with body <= {budget:g} is {r['epochs']} epochs")
            out.update(branch="FULL", epochs=r["epochs"], steps=r["steps"], confirm_possible=True, arithmetic=lines)
            return out
    total = smoke + g8_420
    lines.append(f"no T_p in the menu fits; G-only 8 runs at 420 epochs ({steps_for_epochs(420)} steps): smoke "
                 f"{smoke:.3f} + G8 {g8_420:.3f} = {total:.3f} GPU-h {'<=' if total <= budget else '>'} {budget:g}")
    out.update(g_only_8=g8_420, total=total, confirm_possible=False, arithmetic=lines)
    if total <= budget:
        lines.append("-> G_ONLY_420: no confirmation")
        out.update(branch="G_ONLY_420", epochs=420, steps=steps_for_epochs(420))
    else:
        lines.append("-> CLOSE_NO_PILOT: close with design result and smoke cost measurement")
        out.update(branch="CLOSE_NO_PILOT", epochs=None, steps=None)
    return out


def ext_gate(tier: str, N: int, ext_options: Mapping, ext_cap: float = EXT_BUDGET) -> dict:
    """Extension cap rule (EXP-027 v3 section 5.3).  R2 asked for the extension (tier, N); `ext_options` is the
    `ext_options` dict of the CHOSEN T_p candidate of decide_budget ({'G12': GPU-h of 4 runs, ...}).  The extension
    is enqueued only if its predicted cost <= ext_cap; otherwise it is not run and the selection closes as R7
    (pilot descriptive report).  T_p is not changed by this rule.  Returns action ENQUEUE_EXT or
    CLOSE_R7_EXT_OVER_CAP with the cost and the cap.  An option whose value is None is infeasible (out of memory in
    smoke2, or beyond the measured range; amendment 1): CLOSE_R7_EXT_INFEASIBLE, same effect as over-cap."""
    key = f"{tier}{N}"
    if key not in ext_options:
        raise ValueError(f"extension option {key} not in cost table")
    if ext_options[key] is None:
        return {"option": key, "cost_gpu_h": None, "ext_cap": ext_cap, "action": "CLOSE_R7_EXT_INFEASIBLE",
                "arithmetic": f"extension {key} infeasible (out of memory in smoke2 or no measurement); close at R7"}
    cost = float(ext_options[key])
    ok = cost <= ext_cap
    return {"option": key, "cost_gpu_h": cost, "ext_cap": ext_cap,
            "action": "ENQUEUE_EXT" if ok else "CLOSE_R7_EXT_OVER_CAP",
            "arithmetic": f"extension {key} = {cost:.3f} GPU-h {'<=' if ok else '>'} {ext_cap:g}"}


def pilot_plan(cost_table: Mapping, budget: float = PILOT_BUDGET, ext_cap: float = EXT_BUDGET) -> dict:
    """cost_table: {'smoke_cost': GPU-h of the 5 smoke runs, 'points': [{tier, N, variant, c_step, c_eval, t0}, ...]}
    (c_step in GPU-h per optimizer step).  Returns the design-log 9.2 decision and the arithmetic."""
    smoke = float(cost_table["smoke_cost"])
    cands = []
    for e in T_P_MENU_EPOCHS:
        opts = {f"{tier}{N}": _runs_cost(cost_table, [(tier, N, v) for v in EXT_VARIANTS for _ in EXT_SEEDS], e)
                for tier, ns in EXT_OPTIONS.items() for N in ns}
        cands.append({"epochs": e, "steps": steps_for_epochs(e), "n_evals": n_evals_for_epochs(e),
                      "pilot_17": _runs_cost(cost_table, [(t, N, v) for (t, N, v, _s) in pilot_run_specs()], e),
                      "ext_options": opts})
    g8 = _runs_cost(cost_table, [("G", N, v) for N in PILOT_NS for v in EXT_VARIANTS for _ in PILOT_SEEDS], 420)
    return decide_budget(smoke, cands, g8, budget, ext_cap)
