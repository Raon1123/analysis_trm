"""Tests for tools/statebench_tiers/e027_enqueue.py (no GPU, tmp dirs only)."""
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools", "statebench_tiers"))
import e027_enqueue as E  # noqa: E402
import exp027_select as SEL  # noqa: E402


def grid(argv):
    return E.make_grid(E.build_args(argv))


def test_sizes():
    assert len(grid(["smoke", "--prefix", "p"])) == 5
    assert len(grid(["pilot", "--prefix", "p", "--epochs", "420"])) == 17
    assert len(grid(["ext", "--prefix", "p", "--epochs", "620", "--tier", "G", "--N", "24"])) == 4
    assert len(grid(["conf", "--prefix", "p", "--epochs", "820", "--cells", "S:16,G:16", "--n", "3"])) == 12


def test_smoke2_grid_and_batch_override():
    names = [j["run_name"] for j in grid(["smoke2", "--prefix", "p"])]
    assert names == [f"e027_smoke2_{x}_s0" for x in
                     ("GN16_z", "GN16_noz", "GN32_z", "GN32_noz", "GN48_z", "SN64_z")] == E.smoke2_run_names()
    assert SEL.GLOBAL_BATCH == 1024
    cmds = [E.command(j) for st in (["smoke2"], ["smoke"], ["pilot", "--epochs", "420"],
                                    ["ext", "--epochs", "620", "--tier", "G", "--N", "24"],
                                    ["conf", "--epochs", "820", "--cells", "S:16,G:16", "--n", "2"])
            for j in grid([*st, "--prefix", "p"])]
    for c in cmds:      # right after arch=..., once
        assert c.count("global_batch_size=1024") == 1 and "global_batch_size=2048" not in c
        assert " global_batch_size=1024 epochs=" in c.splitlines()[0]
        assert c.splitlines()[0].split()[3].startswith("arch=") and c.splitlines()[0].split()[4] == "global_batch_size=1024"
    with pytest.raises(SystemExit):
        E.main(["smoke2", "--interleave"])


def test_names_unique_and_format():
    g = grid(["pilot", "--prefix", "p", "--epochs", "420"])
    names = [j["run_name"] for j in g]
    assert len(set(names)) == len(names)
    assert "e027_pilot_GN16_z_s9101000" in names and "e027_pilot_AN16_z_s9101000" in names


def test_command_pins():
    for j in grid(["smoke", "--prefix", "p"]):
        c = E.command(j)
        for s in ("lr=1e-4", "puzzle_emb_lr=1e-4", "weight_decay=1.0", "puzzle_emb_weight_decay=1.0",
                  "checkpoint_every_eval=False", "+eval_save_outputs=[inputs,labels,preds,puzzle_identifiers]",
                  "arch.mlp_t=False", "arch.H_cycles=3", "arch.L_cycles=6", "arch.L_layers=2",
                  "arch.halt_max_steps=1", 'evaluators="[]"', "epochs=40", "eval_interval=20",
                  f'+project_name="{E.PROJECT}"', f'+run_name="{j["run_name"]}"', "ema=True",
                  "+cell_id='\"arm=e027_smoke,"):
            assert s in c, s
        assert "CUDA_VISIBLE_DEVICES" not in c and "log_z_dynamics" not in c
        assert c.rstrip().splitlines()[-1].strip() == "ema=True"
        assert ("arch=trm " in c) == (j["var"] == "z")


def test_epoch_validation():
    with pytest.raises(SystemExit):
        grid(["pilot", "--prefix", "p", "--epochs", "500"])
    with pytest.raises(SystemExit):
        grid(["pilot", "--prefix", "p", "--epochs", "420", "--eval-interval", "40"])
    for st, extra in (("pilot", ()), ("ext", ("--tier", "G", "--N", "24")), ("conf", ("--cells", "S:16,G:16", "--n", "2"))):
        with pytest.raises(SystemExit):     # 10 divides 420 but breaks the fixed eval block (EVAL_STEP steps) and the seed spacing
            grid([st, "--prefix", "p", "--epochs", "420", "--eval-interval", "10", *extra])


def _data(root, g):
    for j in g:
        d = root / f"{j['tier']}_N{j['N']}" / "train"
        d.mkdir(parents=True, exist_ok=True)
        (d / "dataset.json").write_text("{}")


def _argv(tmp_path, q, extra=()):
    return ["smoke", "--prefix", "0000", "--commit", "--queue-dir", str(q), "--allow-other-queue",
            "--data-root", str(tmp_path / "data"), *extra]


def test_commit_and_refusals(tmp_path):
    q = tmp_path / "queue" / "jobs"
    q.mkdir(parents=True)
    for s in ("processing", "done", "failed"):
        (tmp_path / "queue" / s).mkdir()
    g = grid(["smoke", "--prefix", "p"])
    # missing data
    with pytest.raises(SystemExit):
        E.main(_argv(tmp_path, q))
    _data(tmp_path / "data", g)
    # non-canonical queue without flag
    with pytest.raises(SystemExit):
        E.main(["smoke", "--prefix", "0000", "--commit", "--queue-dir", str(q),
                "--data-root", str(tmp_path / "data")])
    # relative queue dir
    with pytest.raises(SystemExit):
        E.main(["smoke", "--prefix", "0000", "--commit", "--queue-dir", "rel", "--allow-other-queue"])
    # dry-run writes nothing
    assert E.main(["smoke", "--prefix", "0000"]) == 0 and not list(q.iterdir())
    assert E.main(_argv(tmp_path, q)) == 0
    files = sorted(os.listdir(q))
    assert len(files) == 5 and files[0].startswith("0000_0000_e027_smoke_")
    # duplicate run names (same prefix and different prefix)
    with pytest.raises(SystemExit):
        E.main(_argv(tmp_path, q))
    with pytest.raises(SystemExit):
        E.main(["smoke", "--prefix", "0001", "--commit", "--queue-dir", str(q), "--allow-other-queue",
                "--data-root", str(tmp_path / "data")])


def test_dup_in_done_and_processing(tmp_path):
    q = tmp_path / "queue" / "jobs"
    q.mkdir(parents=True)
    (tmp_path / "queue" / "done").mkdir()
    (tmp_path / "queue" / "processing").mkdir()
    g = grid(["smoke", "--prefix", "p"])
    _data(tmp_path / "data", g)
    (tmp_path / "queue" / "processing" / f"0007_{g[2]['run_name']}.job.gpu5").write_text("x")
    with pytest.raises(SystemExit):
        E.main(_argv(tmp_path, q))
    assert not list(q.iterdir())


# ---------------------------------------------------------------- seeds (design-log 8)
def _stage_grids():
    return {"smoke": grid(["smoke", "--prefix", "p"]),
            "pilot": grid(["pilot", "--prefix", "p", "--epochs", "420"]),
            "ext": grid(["ext", "--prefix", "p", "--epochs", "420", "--tier", "S", "--N", "64"]),
            "conf": grid(["conf", "--prefix", "p", "--epochs", "420", "--cells", "S:16,G:16", "--n", "30"])}


def test_seed_values_per_stage():
    g = _stage_grids()
    assert {j["seed"] for j in g["smoke"]} == {0}
    assert {j["seed"] for j in g["pilot"]} == {9101000, 9102000}
    assert {j["seed"] for j in g["ext"]} == {9101000, 9102000} and len(g["ext"]) == 4
    assert {j["seed"] for j in g["conf"]} == {1000 * i for i in range(1, 31)}
    assert E.PILOT_SEEDS == SEL.PILOT_SEEDS
    assert {j["seed"] for j in grid(["ext", "--prefix", "p", "--epochs", "420", "--tier", "G", "--N", "12"])} \
        == {9101000, 9102000}


def test_no_two_seeds_in_a_stage_closer_than_42():
    for stage, jobs in _stage_grids().items():
        seeds = sorted({j["seed"] for j in jobs})
        gaps = [b - a for a, b in zip(seeds, seeds[1:])]
        assert all(gp >= SEL.SEED_MIN_GAP for gp in gaps), (stage, seeds)
    assert SEL.SEED_MIN_GAP == 42 and SEL.SEED_SPACING >= SEL.SEED_MIN_GAP
    # the pilot seeds and the largest conf seed also keep the gap to each other and to the smoke seed
    allseeds = sorted({0, *SEL.PILOT_SEEDS, *SEL.conf_seeds(30)})
    assert all(b - a >= 42 for a, b in zip(allseeds, allseeds[1:]))


# ---------------------------------------------------------------- canonical queue (hard-coded, realpath)
def test_canonical_queue_is_hard_coded(tmp_path):
    assert E.CANON_QUEUE_JOBS == "/home/ayp/project/trm/scripts/queue/jobs"
    src = open(E.__file__).read()
    assert 'os.path.join(REPO, "scripts", "queue"' not in src
    # a symlink to a scratch dir is not the canonical queue: refused without --allow-other-queue
    q = tmp_path / "queue" / "jobs"
    q.mkdir(parents=True)
    link = tmp_path / "lnk"
    link.symlink_to(q)
    g = grid(["smoke", "--prefix", "p"])
    _data(tmp_path / "data", g)
    with pytest.raises(SystemExit):
        E.main(["smoke", "--prefix", "0000", "--commit", "--queue-dir", str(link), "--data-root", str(tmp_path / "data")])
    assert not list(q.iterdir())


# ---------------------------------------------------------------- --interleave
REAL_NAMES = os.path.join(os.path.dirname(__file__), "fixtures", "e027_queue_names_20261111.txt")


def _bash_ls(d, loc="en_US.UTF-8"):
    r = subprocess.run(["bash", "-c", 'cd -- "$1" && for f in *.job; do printf "%s\\n" "$f"; done', "_", str(d)],
                       capture_output=True, text=True, env=dict(os.environ, LC_ALL=loc))
    assert r.returncode == 0
    return [x for x in r.stdout.split("\n") if x]


def _scratch_queue(tmp_path, names):
    q = tmp_path / "queue" / "jobs"
    q.mkdir(parents=True)
    for n in names:
        (q / n).write_text("")
    return q


def _real_names():
    return [x for x in open(REAL_NAMES).read().split("\n") if x]


def _dry_names(capsys):
    return [ln[3:] for ln in capsys.readouterr().out.splitlines() if ln.startswith("== ")]


def _expected_order(existing, new):
    out = []
    for i, x in enumerate(existing):
        out.append(x)
        if i < len(new):
            out.append(new[i])
    return out + new[len(existing):]


def test_interleave_alternates_one_to_one(tmp_path, capsys):
    names = _real_names()
    assert len(names) > 20                      # copies of the real queue names, taken 2026-11-11
    q = _scratch_queue(tmp_path, names)
    argv = ["conf", "--epochs", "420", "--cells", "S:16,G:16", "--n", "3"]
    assert E.main([*argv, "--interleave", "--queue-dir", str(q)]) == 0
    new = _dry_names(capsys)
    assert len(new) == 12 and not set(new) & set(names)
    existing = _bash_ls(q)
    assert set(existing) == set(names) and sorted(os.listdir(q)) == sorted(names)     # dry-run wrote nothing
    for n in new:                                # materialise the planned names as empty files, list with bash
        (q / n).write_text("")
    merged = _bash_ls(q)
    assert merged == _expected_order(existing, new)           # 1:1 alternation
    assert [x for x in merged if x in set(existing)] == existing    # existing order untouched
    runs = [j["run_name"] for j in grid(argv + ["--prefix", "p"])]
    assert all(n.endswith(f"_{r}.job") for n, r in zip(new, runs))


def test_interleave_more_new_than_existing_keeps_relative_order(tmp_path, capsys):
    names = _real_names()[:4]
    q = _scratch_queue(tmp_path, names)
    assert E.main(["pilot", "--epochs", "420", "--interleave", "--queue-dir", str(q)]) == 0
    new = _dry_names(capsys)
    assert len(new) == 17
    existing = _bash_ls(q)
    for n in new:
        (q / n).write_text("")
    merged = _bash_ls(q)
    assert merged == _expected_order(existing, new)
    assert merged[-13:] == new[4:]                            # the rest follow the last existing job, in order


def test_interleave_c_locale_and_existing_untouched(tmp_path, capsys):
    names = _real_names()
    q = _scratch_queue(tmp_path, names)
    before = {n: (q / n).stat().st_mtime_ns for n in names}
    assert E.main(["ext", "--epochs", "620", "--tier", "G", "--N", "24", "--interleave", "--queue-dir", str(q)]) == 0
    new = _dry_names(capsys)
    assert len(new) == 4
    assert {n: (q / n).stat().st_mtime_ns for n in names} == before and sorted(os.listdir(q)) == sorted(names)
    existing_c = _bash_ls(q, "C")
    for n in new:
        (q / n).write_text("")
    assert _bash_ls(q, "C") == _expected_order(existing_c, new)


def test_interleave_option_rules(tmp_path):
    q = _scratch_queue(tmp_path, ["0001_a.job"])
    with pytest.raises(SystemExit):                            # prefix and interleave together
        E.main(["pilot", "--epochs", "420", "--prefix", "p", "--interleave", "--queue-dir", str(q)])
    with pytest.raises(SystemExit):                            # neither
        E.main(["pilot", "--epochs", "420"])
    with pytest.raises(SystemExit):                            # smoke has no interleave
        E.main(["smoke", "--interleave", "--queue-dir", str(q)])
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit):                            # nothing to alternate with
        E.main(["pilot", "--epochs", "420", "--interleave", "--queue-dir", str(empty)])


def test_interleave_commit_writes_only_new_files(tmp_path, capsys):
    names = _real_names()[:6]
    q = _scratch_queue(tmp_path, names)
    for s_ in ("processing", "done", "failed"):
        (tmp_path / "queue" / s_).mkdir()
    g = grid(["conf", "--prefix", "p", "--epochs", "420", "--cells", "G:16", "--n", "2"])
    _data(tmp_path / "data", g)
    argv = ["conf", "--epochs", "420", "--cells", "G:16", "--n", "2", "--interleave", "--commit", "--queue-dir", str(q),
            "--allow-other-queue", "--data-root", str(tmp_path / "data")]
    assert E.main(argv) == 0
    now = _bash_ls(q)
    assert len(now) == 6 + 4 and set(names) <= set(now)
    with pytest.raises(SystemExit):                            # same run names again
        E.main(argv)
