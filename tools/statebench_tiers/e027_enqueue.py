#!/usr/bin/env python3
"""EXP-027 job emitter (stdlib only).

Builds queue job files for the EXP-027 StateBench campaign.  Default is a
dry-run that only prints the grid.  Writing needs --commit and an absolute
--queue-dir, which must be the canonical queue (realpath comparison with
CANON_QUEUE_JOBS) unless --allow-other-queue.  This tool never touches
scripts/sigma_enqueue.sh and never renames or modifies an existing queue file.

Stages
------
smoke : SUPERSEDED by smoke2 (amendment 1: batch 2048 ran out of memory at N>=32).  Kept for the record; its
        commands now also carry global_batch_size=1024, so they no longer reproduce the jobs that actually ran.
smoke2: 6 jobs (G N16 z/noz, G N32 z/noz, G N48 z, S N64 z; all s0), epochs=40, global_batch_size=1024.
pilot : 17 jobs (G,S x N{16,32} x {z,noz} x seeds {9101000,9102000}; A N16 z s9101000).
ext   : --tier --N ; the one-time extension of design-log 9.2, 4 jobs for either tier
        (seeds 9101000,9102000 x z/noz), same stage name in run names (e027_ext_...).
conf  : --cells "S:16,G:16" --n K ; seeds 1000*i (i = 1..K) x z/noz per cell.

Seeds are spaced by 1000 (data order of a run is drawn from seed + eval block index, <= 41 blocks; design-log 8).

File names
----------
--prefix P            : P_<NNNN>_<run_name>.job (sorted position is set by P).
--interleave (pilot/ext/conf; replaces --prefix): list the *.job files of the queue dir in the order bash
  expands the glob (LC_ALL=en_US.UTF-8, as the worker does; bash is called, Python's sort is not used) and give the
  i-th new job the name <stem of the i-th existing job>_z<NNNN>_<run_name>.job, so it sorts directly after the
  i-th existing job (1:1 alternation).  New jobs beyond the last existing job continue after it in their own
  order.  The tag letter z is chosen because en_US collation ignores punctuation, so a tail starting with 'i' would
  sort before the existing '.job'; 'z' sorts after 'j' (and '_' after '.' in C).  The resulting order is re-listed with bash on a scratch directory of empty files and compared with the
  planned alternation; any mismatch aborts before anything is written.

Keys: wandb/checkpoint project is set with ``+project_name="StateBench_e027"``
(PretrainConfig field, pretrain.py:73; checkpoints/<project>/<run_name>).
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import exp027_select as S  # noqa: E402  (seeds and menus are shared with the selection rule)

REPO = os.path.dirname(os.path.dirname(HERE))
# Hard-coded: a git worktree of this repo has its own scripts/queue that no worker reads (memory
# worktree-queue-not-shared), so the canonical queue is never derived from __file__.
CANON_QUEUE_JOBS = "/home/ayp/project/trm/scripts/queue/jobs"
QUEUE_LOCALE = "en_US.UTF-8"
DEFAULT_DATA_ROOT = os.path.join(REPO, "data", "statebench_e027")
PROJECT = "StateBench_e027"
ARCH = {"z": "trm", "noz": "trm_singlez"}
TIERS = ("A", "S", "G")
EPOCH_MENU = (820, 620, 420)
PILOT_SEEDS = S.PILOT_SEEDS            # (9101000, 9102000)
# No submit-time queue cap exists in scripts/ (04_BUILD_SPEC s3.4); none enforced.


def _job(stage, tier, n, var, seed, epochs, ei):
    return dict(stage=stage, tier=tier, N=n, var=var, seed=seed, epochs=epochs, ei=ei,
                run_name=f"e027_{stage}_{tier}N{n}_{var}_s{seed}")


def grid_smoke():
    spec = [("G", 16, "z"), ("G", 16, "noz"), ("G", 32, "z"), ("G", 32, "noz"), ("S", 64, "z")]
    return [_job("smoke", t, n, v, S.SMOKE_SEED, 40, 20) for t, n, v in spec]


SMOKE2_SPEC = [("G", 16, "z"), ("G", 16, "noz"), ("G", 32, "z"), ("G", 32, "noz"), ("G", 48, "z"), ("S", 64, "z")]


def grid_smoke2():
    return [_job("smoke2", t, n, v, S.SMOKE_SEED, 40, 20) for t, n, v in SMOKE2_SPEC]


def smoke2_run_names():
    """The six second-smoke run names (e027_smoke2_GN16_z_s0, ...), for later tooling."""
    return [j["run_name"] for j in grid_smoke2()]


def grid_pilot(epochs, ei):
    out = []
    for t in ("G", "S"):
        for n in (16, 32):
            for v in ("z", "noz"):
                for s in PILOT_SEEDS:
                    out.append(_job("pilot", t, n, v, s, epochs, ei))
    out.append(_job("pilot", "A", 16, "z", PILOT_SEEDS[0], epochs, ei))
    return out


def grid_ext(tier, n, epochs, ei):
    return [_job("ext", tier, n, v, s, epochs, ei) for s in PILOT_SEEDS for v in ("z", "noz")]


def grid_conf(cells, nseeds, epochs, ei):
    out = []
    for tier, n in cells:
        for s in S.conf_seeds(nseeds):
            for v in ("z", "noz"):
                out.append(_job("conf", tier, n, v, s, epochs, ei))
    return out


def parse_cells(text):
    cells = []
    for part in text.split(","):
        t, _, n = part.strip().partition(":")
        if t not in TIERS or not n.isdigit():
            raise SystemExit(f"bad --cells entry {part!r} (want TIER:N, TIER in A,S,G)")
        cells.append((t, int(n)))
    return cells


def command(j):
    d = f"data/statebench_e027/{j['tier']}_N{j['N']}"
    cell = f"arm=e027_{j['stage']},cohort={j['var']},tier={j['tier']},N={j['N']},s={j['seed']}"
    lines = [
        f"uv run pretrain.py arch={ARCH[j['var']]} global_batch_size={S.GLOBAL_BATCH} "
        f"epochs={j['epochs']} eval_interval={j['ei']} "
        "lr=1e-4 puzzle_emb_lr=1e-4 \\",
        "    weight_decay=1.0 puzzle_emb_weight_decay=1.0 checkpoint_every_eval=False \\",
        "    +eval_save_outputs=[inputs,labels,preds,puzzle_identifiers] \\",
        "    arch.mlp_t=False arch.H_cycles=3 arch.L_cycles=6 arch.L_layers=2 arch.halt_max_steps=1 \\",
        '    evaluators="[]" \\',
        f'    data_paths="[{d}]" \\',
        f"    seed={j['seed']} \\",
        "    +k=0 \\",
        f'    +project_name="{PROJECT}" \\',
        f'    +run_group="e027_{j["stage"]}" \\',
        f'    +run_name="{j["run_name"]}" \\',
        f"    +cell_id='\"{cell}\"' \\",
        "    ema=True",
    ]
    return "\n".join(lines) + "\n"


def existing_run_names(queue_dir, run_name):
    """Paths in jobs/processing/done/failed whose job name ends in _<run_name>.job[.gpuN]."""
    base = os.path.dirname(os.path.abspath(queue_dir))
    hits = []
    for sub in ("jobs", "processing", "done", "failed"):
        d = os.path.join(base, sub)
        hits += glob.glob(os.path.join(glob.escape(d), f"*_{glob.escape(run_name)}.job"))
        hits += glob.glob(os.path.join(glob.escape(d), f"*_{glob.escape(run_name)}.job.gpu*"))
    return hits


def bash_glob_order(directory, pattern="*.job"):
    """File names matching `pattern` in `directory`, in the order bash expands the glob under QUEUE_LOCALE
    (the order the queue worker claims them).  Bash is called; nothing is sorted in Python."""
    r = subprocess.run(["bash", "-c", 'cd -- "$1" && for f in $2; do [[ -e $f ]] && printf "%s\\n" "$f"; done',
                        "_", directory, pattern], capture_output=True, text=True,
                       env=dict(os.environ, LC_ALL=QUEUE_LOCALE))
    if r.returncode != 0:
        raise SystemExit(f"cannot list {directory}: {r.stderr.strip()}")
    return [x for x in r.stdout.split("\n") if x]


def interleave_names(existing, run_names):
    """Names for `run_names` such that the i-th new name sorts directly after existing[i] (existing in bash
    order); extra new names continue after the last existing one.  Returns (new_names, planned_order)."""
    if not existing:
        raise SystemExit("--interleave needs at least one existing *.job file in the queue dir")
    new, order = [], []
    for i, rn in enumerate(run_names):
        anchor = existing[min(i, len(existing) - 1)]
        new.append(f"{anchor[:-len('.job')]}_z{i:04d}_{rn}.job")
    for i, x in enumerate(existing):
        order.append(x)
        if i < len(new):
            order.append(new[i])
    order += new[len(existing):]
    return new, order


def verify_interleave(existing, new, order):
    """Write empty files for existing + new into a scratch dir, list them with bash, require `order`."""
    clash = sorted(set(existing) & set(new))
    if clash:
        raise SystemExit(f"refusing to overwrite {clash[0]} (planned interleave name already exists)")
    with tempfile.TemporaryDirectory(prefix="e027_ilv_") as td:
        for name in list(existing) + list(new):
            open(os.path.join(td, name), "x").close()
        got = bash_glob_order(td)
    if got != order:
        bad = next((i for i, (a, b) in enumerate(zip(got, order)) if a != b), min(len(got), len(order)))
        raise SystemExit(f"interleave order check failed at position {bad}: bash lists {got[bad:bad + 2]}, "
                         f"planned {order[bad:bad + 2]}")


def build_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("smoke", "smoke2", "pilot", "ext", "conf"))
    p.add_argument("--prefix", help="file-name prefix controlling claim order (exclusive with --interleave)")
    p.add_argument("--interleave", action="store_true",
                   help="pilot/ext/conf: alternate 1:1 with the *.job files already in the queue dir")
    p.add_argument("--tier", choices=TIERS)
    p.add_argument("--N", type=int)
    p.add_argument("--cells")
    p.add_argument("--n", type=int, help="conf: number of seeds per cell")
    p.add_argument("--epochs", type=int)
    p.add_argument("--eval-interval", type=int, default=20)
    p.add_argument("--commit", action="store_true")
    p.add_argument("--queue-dir", help="absolute; with --interleave also the dir listed in a dry-run "
                                      "(default: the canonical queue)")
    p.add_argument("--allow-other-queue", action="store_true")
    p.add_argument("--data-root", default=DEFAULT_DATA_ROOT, help=argparse.SUPPRESS)
    return p.parse_args(argv)


def make_grid(a):
    if a.stage == "smoke":
        print("# WARNING: stage 'smoke' is superseded by 'smoke2' (amendment 1)", file=sys.stderr)
        return grid_smoke()
    if a.stage == "smoke2":
        return grid_smoke2()
    if a.epochs not in EPOCH_MENU:
        raise SystemExit(f"--epochs required for {a.stage}, one of {EPOCH_MENU}")
    if a.eval_interval != S.EVAL_EPOCHS:
        # EVAL_STEP steps per block, the seed spacing (<= 41 blocks) and the C9 prefix property all assume 20
        raise SystemExit(f"eval_interval is fixed at {S.EVAL_EPOCHS} for EXP-027 (got {a.eval_interval})")
    if a.epochs % a.eval_interval:
        raise SystemExit("eval_interval must divide epochs")
    if a.stage == "pilot":
        return grid_pilot(a.epochs, a.eval_interval)
    if a.stage == "ext":
        if not a.tier or not a.N:
            raise SystemExit("ext needs --tier and --N")
        return grid_ext(a.tier, a.N, a.epochs, a.eval_interval)
    if not a.cells or not a.n or a.n < 1:
        raise SystemExit("conf needs --cells and --n >= 1")
    return grid_conf(parse_cells(a.cells), a.n, a.epochs, a.eval_interval)


def main(argv=None):
    a = build_args(argv)
    if a.interleave:
        if a.prefix:
            raise SystemExit("--interleave replaces --prefix; give only one")
        if a.stage in ("smoke", "smoke2"):
            raise SystemExit("--interleave is for pilot / ext / conf")
    elif not a.prefix or "/" in a.prefix or " " in a.prefix:
        raise SystemExit("invalid --prefix")
    grid = make_grid(a)
    names = [j["run_name"] for j in grid]
    if len(set(names)) != len(names):
        raise SystemExit("duplicate run names inside grid")
    qd = os.path.normpath(a.queue_dir) if a.queue_dir else None
    if a.interleave:
        ldir = qd or CANON_QUEUE_JOBS
        if not os.path.isabs(ldir) or not os.path.isdir(ldir):
            raise SystemExit(f"cannot list queue dir {ldir!r} (absolute existing dir needed)")
        existing = bash_glob_order(ldir)
        files, order = interleave_names(existing, names)
        verify_interleave(existing, files, order)
        print(f"# interleave: {len(existing)} existing *.job in {ldir}, order verified with bash "
              f"LC_ALL={QUEUE_LOCALE}")
    else:
        files = [f"{a.prefix}_{i:04d}_{j['run_name']}.job" for i, j in enumerate(grid)]
    for f, j in zip(files, grid):
        print(f"== {f}\n   run_name={j['run_name']}")
        print(command(j))
    print(f"# {len(grid)} jobs ({'COMMIT' if a.commit else 'dry-run'})")
    if not a.commit:
        return 0
    if not a.queue_dir or not os.path.isabs(a.queue_dir):
        raise SystemExit("--commit needs an absolute --queue-dir")
    if os.path.realpath(qd) != os.path.realpath(CANON_QUEUE_JOBS) and not a.allow_other_queue:
        raise SystemExit(f"refusing queue dir {qd} (not {CANON_QUEUE_JOBS}); use --allow-other-queue")
    if not os.path.isdir(qd):
        raise SystemExit(f"queue dir {qd} does not exist")
    for j in grid:
        dd = os.path.join(a.data_root, f"{j['tier']}_N{j['N']}", "train", "dataset.json")
        if not os.path.isfile(dd):
            raise SystemExit(f"missing data: {dd}")
    for j in grid:
        hit = existing_run_names(qd, j["run_name"])
        if hit:
            raise SystemExit(f"run name already queued: {j['run_name']} -> {hit[0]}")
    for f in files:
        if os.path.exists(os.path.join(qd, f)):
            raise SystemExit(f"refusing to overwrite {f}")
    for f, j in zip(files, grid):
        with open(os.path.join(qd, f), "x") as fh:
            fh.write(command(j))
    print(f"wrote {len(files)} files to {qd}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
