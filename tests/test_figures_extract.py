"""Tests for tools/figures/wandb_ro.py and tools/figures/extract_curves.py.

CPU only, no NAS access: synthetic run directories are built in tmp_path and
their ``*.wandb`` datastores are written with wandb's own DataStore writer.
"""

from __future__ import annotations

import builtins
import csv
import inspect
import json
import os
import sys
import time
from pathlib import Path

import pytest

REPO = "/home/ayp/project/trm"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from tools.figures import extract_curves as ec  # noqa: E402
from tools.figures import wandb_ro  # noqa: E402

from wandb.proto import wandb_internal_pb2 as pb  # noqa: E402
from wandb.sdk.internal import datastore as wds  # noqa: E402

OLD = time.time() - 7200  # mtime two hours ago (not "still being written")

FAMILIES_YAML = """\
rebuild_cutoff_utc: "2026-07-20T13:18:00Z"
cohort_rule:
  - prefix: data/power_permutation/d1
    cohort: d1
  - prefix: data/sigma_k_10
    before_cutoff: legacy
    after_cutoff: clean_d0
  - default: other
eval_metrics: [probe/test_exact, probe/train_exact, all/exact_accuracy, phase/index]
required_eval_metrics: [probe/test_exact, probe/train_exact]
train_metrics: [train/exact_accuracy, train/lm_loss]
train_stride: 25
families:
  - id: famA
    project: ProjA
    regex: '^runA_k\\d+_s\\d+$'
    expected_names: {names}
    expected_last_step: {last}
"""


def _item(key: str, value, nested: bool = False):
    it = pb.HistoryItem()
    if nested:
        it.nested_key.extend(key.split("/"))
    else:
        it.key = key
    it.value_json = json.dumps(value)
    return it


def write_wandb(path: Path, last_step: int, eval_every: int = 10, *, train: bool = True,
                eval_metrics=("probe/test_exact", "probe/train_exact", "all/exact_accuracy",
                              "phase/index")) -> None:
    ds = wds.DataStore()
    ds.open_for_write(str(path))
    run = pb.Record()
    run.run.run_id = "x"
    ds.write(run)
    for step in range(1, last_step + 1):
        items = [_item("_step", step)]
        if train:
            items += [_item("train/exact_accuracy", step / (last_step + 1)),
                      _item("train/lm_loss", 1.0 / step, nested=True)]
        if step % eval_every == 0:
            for j, m in enumerate(eval_metrics):
                items.append(_item(m, round(step / (10 * last_step) + j * 0.01, 6),
                                   nested=(j % 2 == 0)))
        if len(items) == 1:
            continue
        rec = pb.Record()
        rec.history.item.extend(items)
        rec.history.step.num = step
        ds.write(rec)
    ds.close()


def make_run(root: Path, wid: str, *, project="ProjA", run_name="runA_k5_s1",
             data_path="data/sigma_k_10/5", started="2026-06-01T00:00:00Z", last_step=103,
             symlink_from: Path | None = None, truncate: bool = False, recent: bool = False,
             **kw) -> Path:
    ts = started.replace("-", "").replace(":", "").replace("T", "_")[:15]
    name = f"run-{ts}-{wid}"
    base = symlink_from if symlink_from is not None else root
    d = base / name
    (d / "files").mkdir(parents=True)
    args = ["arch=trm_singlez", f"data_paths=[{data_path}]", "seed=1", "ema=True",
            f"+project_name={project}", f"+run_name={run_name}"]
    (d / "files" / "wandb-metadata.json").write_text(
        json.dumps({"args": args, "startedAt": started, "program": "pretrain.py"}))
    f = d / f"run-{wid}.wandb"
    write_wandb(f, last_step, **kw)
    if truncate:
        os.truncate(f, f.stat().st_size - 5)
    if not recent:
        os.utime(f, (OLD, OLD))
    if symlink_from is not None:
        os.symlink(d, root / name)
    return d


@pytest.fixture
def env(tmp_path):
    root = tmp_path / "wandb"
    root.mkdir()

    def fam(names=1, last=103):
        p = tmp_path / f"fam_{names}_{last}.yaml"
        p.write_text(FAMILIES_YAML.format(names=names, last=last))
        return p

    def run(fam_path, out_name="out", extra=()):
        out = tmp_path / out_name
        argv = ["--families", str(fam_path), "--out", str(out), "--jobs", "1",
                "--wandb-root", str(root), "--verify-after", *extra]
        return ec.main(argv), out

    return root, fam, run, tmp_path


def read_csv(p: Path):
    with p.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


# --------------------------------------------------------------------------- read-only

def test_every_wandb_open_is_read_only(env, monkeypatch):
    root, fam, run, _ = env
    make_run(root, "aaaa0001")
    seen = []
    real_open = builtins.open

    def spy(file, mode="r", *a, **k):
        if str(file).endswith(".wandb"):
            seen.append(mode)
        return real_open(file, mode, *a, **k)

    monkeypatch.setattr(builtins, "open", spy)
    rc, _ = run(fam())
    monkeypatch.setattr(builtins, "open", real_open)
    assert rc == 0
    assert len(seen) == 3  # hash before, scan, hash after (--verify-after)
    assert all(m == "rb" for m in seen), seen


def test_reader_never_calls_parent_open_for_scan():
    src = inspect.getsource(wandb_ro.ReadOnlyDataStore.open_for_scan)
    assert "super(" not in src and "DataStore.open_for_scan" not in src
    assert wandb_ro.ReadOnlyDataStore.open_for_scan is not wds.DataStore.open_for_scan
    assert 'open(fname, "rb")' in src
    assert os.environ.get("WANDB_MODE") == "disabled"


def test_iter_history_clean_end_and_truncation(tmp_path):
    f = tmp_path / "run-x.wandb"
    write_wandb(f, 30)
    stats = {}
    got = list(wandb_ro.iter_history(str(f), ["probe/test_exact"], stats))
    assert [s for s, _ in got] == [10, 20, 30]
    assert stats["last_step"] == 30 and stats["bytes_read"] == f.stat().st_size
    os.truncate(f, f.stat().st_size - 5)
    with pytest.raises(wandb_ro.TruncatedRun) as ei:
        list(wandb_ro.iter_history(str(f), ["probe/test_exact"]))
    assert ei.value.file_size == f.stat().st_size and ei.value.bytes_read <= ei.value.file_size


# --------------------------------------------------------------------------- discovery / rows

def test_symlinked_run_directory_is_discovered(env):
    root, fam, run, tmp = env
    off = tmp / "offload"
    off.mkdir()
    make_run(root, "link0001", symlink_from=off)
    rc, out = run(fam())
    assert rc == 0
    man = read_csv(out / "MANIFEST.csv")
    assert [(m["wandb_id"], m["is_symlink"], m["role"]) for m in man] == [("link0001", "True",
                                                                          "primary")]


def test_every_eval_point_kept_and_train_follows_stride(env):
    root, fam, run, _ = env
    make_run(root, "aaaa0001", last_step=103, eval_every=10)
    rc, out = run(fam())
    assert rc == 0
    ev = read_csv(out / "famA.eval.csv")
    tr = read_csv(out / "famA.train.csv")
    pt = [int(r["step"]) for r in ev if r["metric"] == "probe/test_exact"]
    assert pt == list(range(10, 101, 10))  # all 10 evaluation points, none thinned
    for m in ("probe/train_exact", "all/exact_accuracy", "phase/index"):
        assert sum(r["metric"] == m for r in ev) == 10
    steps = sorted({int(r["step"]) for r in tr})
    assert steps == [25, 50, 75, 100, 103]  # stride multiples plus the last step
    assert list(ev[0].keys()) == ec.CURVE_COLUMNS
    assert ev[0]["k"] == "5" and ev[0]["cohort"] == "legacy" and ev[0]["arch"] == "trm_singlez"
    man = read_csv(out / "MANIFEST.csv")[0]
    assert man["n_eval_points"] == "10" and man["last_step"] == "103"
    assert man["wandb_sha256_before"] == man["wandb_sha256_after"] != ""
    assert man["wandb_mtime_ns_before"] == man["wandb_mtime_ns_after"]
    cfg = read_csv(out / "configs.csv")
    assert ("project_name", "ProjA") in {(c["key"], c["value"]) for c in cfg}


# --------------------------------------------------------------------------- loud failures

def _setup_failure(case, root, tmp):
    names, last, extra = 1, 103, []
    if case == "fewer_names":
        make_run(root, "aaaa0001")
        names = 2
    elif case == "wrong_last_step":
        make_run(root, "aaaa0001")
        last = 200
    elif case == "truncated_primary":
        make_run(root, "aaaa0001", truncate=True)
    elif case == "missing_required_metric":
        make_run(root, "aaaa0001", eval_metrics=("probe/test_exact", "all/exact_accuracy"))
    elif case == "empty_output":
        make_run(root, "aaaa0001", train=False, last_step=100)
        last = 100
    elif case == "recently_modified":
        make_run(root, "aaaa0001", recent=True)
    elif case == "over_size_cap":
        make_run(root, "aaaa0001")
        extra = ["--max-out-mb", "0.001"]
    return names, last, extra


@pytest.mark.parametrize("case", ["fewer_names", "wrong_last_step", "truncated_primary",
                                  "missing_required_metric", "empty_output",
                                  "recently_modified", "over_size_cap"])
def test_loud_failures_exit_nonzero(env, case, capsys):
    root, fam, run, tmp = env
    names, last, extra = _setup_failure(case, root, tmp)
    rc, out = run(fam(names, last), extra=extra)
    assert rc != 0, case
    err = capsys.readouterr().err
    reason = {"fewer_names": "distinct run names", "wrong_last_step": "!= expected",
              "truncated_primary": "TruncatedRun", "missing_required_metric":
              "zero points for required metric probe/train_exact",
              "empty_output": "would be empty", "recently_modified": "last 30 minutes",
              "over_size_cap": "max-out-mb"}[case]
    assert reason in err, (case, err)
    if case not in ("empty_output", "over_size_cap"):
        assert "aaaa0001" in err or "runA" in err, err
    assert not out.exists() or not any(out.iterdir())  # nothing written on failure


def test_size_cap_failure_message(env, capsys):
    root, fam, run, _ = env
    make_run(root, "aaaa0001")
    rc, _ = run(fam(), extra=["--max-out-mb", "0.001"])
    assert rc == 1 and "max-out-mb" in capsys.readouterr().err


def test_source_changed_detected(env, monkeypatch, capsys):
    root, fam, run, _ = env
    make_run(root, "aaaa0001")
    calls = {"n": 0}
    real = ec.sha256_file

    def flaky(path):
        if str(path).endswith(".wandb"):
            calls["n"] += 1
            if calls["n"] == 2:
                return "0" * 64
        return real(path)

    monkeypatch.setattr(ec, "sha256_file", flaky)
    rc, _ = run(fam())
    assert rc == 4 and "SOURCE CHANGED" in capsys.readouterr().err


def test_refuses_non_empty_out(env):
    root, fam, run, tmp = env
    make_run(root, "aaaa0001")
    (tmp / "out").mkdir()
    (tmp / "out" / "x.txt").write_text("x")
    rc, _ = run(fam())
    assert rc == 3


# --------------------------------------------------------------------------- duplicates / determinism

def test_partial_and_complete_directories_complete_is_primary(env):
    root, fam, run, _ = env
    make_run(root, "full0001", started="2026-07-01T00:00:00Z", last_step=103)
    make_run(root, "part0001", started="2026-07-05T00:00:00Z", last_step=50)
    make_run(root, "trnc0001", started="2026-07-06T00:00:00Z", last_step=40, truncate=True)
    rc, out = run(fam())
    assert rc == 0
    man = {m["wandb_id"]: m for m in read_csv(out / "MANIFEST.csv")}
    assert man["full0001"]["role"] == "primary"
    assert man["part0001"]["role"] == "superseded"
    assert "truncated at step 50" in man["part0001"]["reason"]
    assert man["trnc0001"]["role"] == "superseded"
    assert "TruncatedRun" in man["trnc0001"]["reason"]
    ev = read_csv(out / "famA.eval.csv")
    assert {r["wandb_id"] for r in ev} == {"full0001"}


def test_two_extractions_identical_bytes(env):
    root, fam, run, _ = env
    make_run(root, "aaaa0001", run_name="runA_k5_s1")
    make_run(root, "bbbb0002", run_name="runA_k7_s2", data_path="data/sigma_k_10/7")
    f = fam(2)
    rc1, o1 = run(f, "out1")
    rc2, o2 = run(f, "out2")
    assert rc1 == rc2 == 0
    names = sorted(p.name for p in o1.iterdir())
    assert names == sorted(p.name for p in o2.iterdir())
    for n in names:
        if n == "EXTRACT_META.json":
            continue
        assert (o1 / n).read_bytes() == (o2 / n).read_bytes(), n
    files = read_csv(o1 / "MANIFEST.files.csv")
    assert {r["file"] for r in files} >= {"famA.eval.csv", "famA.train.csv", "configs.csv",
                                         "MANIFEST.csv"}


def test_cohort_rule_cutoff_and_d1(tmp_path):
    import yaml
    cfg = yaml.safe_load(FAMILIES_YAML.format(names=1, last=1))
    assert ec.cohort_for("data/sigma_k_10/7", "2026-07-20T13:17:59Z", cfg) == "legacy"
    assert ec.cohort_for("data/sigma_k_10/7", "2026-07-20T13:18:00Z", cfg) == "clean_d0"
    assert ec.cohort_for("data/sigma_k_10/7", "2026-11-03T04:44:01.123Z", cfg) == "clean_d0"
    assert ec.cohort_for("data/power_permutation/d1/5", "2026-06-01T00:00:00Z", cfg) == "d1"
    assert ec.cohort_for("data/sigma_k/7", "2026-06-01T00:00:00Z", cfg) == "other"
    assert ec.cohort_for("data/sigma_k_100/7", "2026-06-01T00:00:00Z", cfg) == "other"


def test_same_run_name_in_two_projects_selects_family_project(env):
    root, fam, run, _ = env
    make_run(root, "mine0001", project="ProjA")
    make_run(root, "othr0001", project="ProjB")
    rc, out = run(fam())
    assert rc == 0
    man = read_csv(out / "MANIFEST.csv")
    assert [m["wandb_id"] for m in man] == ["mine0001"]
    assert {r["project"] for r in read_csv(out / "famA.eval.csv")} == {"ProjA"}
