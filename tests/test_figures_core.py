"""Core tests for tools.figures (registry, style, io, build_all). Everything runs in tmp_path."""
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tools.figures import build_all, registry, style  # noqa: E402
from tools.figures import io as fio  # noqa: E402

SELFTEST_YAML = REPO / "tools" / "figures" / "registry_selftest" / "SELFTEST.yaml"

BAD_MODULE = '''
def build(ctx):
    raise RuntimeError("boom from test")
'''
NOVALUES_MODULE = '''
import pandas as pd
import matplotlib.pyplot as plt
def build(ctx):
    fig, ax = plt.subplots(figsize=(8, 4.5)); ax.plot([0, 1], [0, 1]); ax.set_xlabel("step")
    ctx.save(fig)
    ctx.write_values(pd.DataFrame({"figure_id": [ctx.fig_id], "value": [1.0]}))
'''


def registry_dir(tmp_path, name="reg", extra=()):
    d = tmp_path / name
    d.mkdir()
    shutil.copy(SELFTEST_YAML, d / "SELFTEST.yaml")
    for fid, module in extra:
        text = SELFTEST_YAML.read_text().replace("SELFTEST", fid, 1).replace(
            "tools.figures.figs._selftest_fig", module)
        (d / f"{fid}.yaml").write_text(text)
    return d


def run_build(tmp_path, reg, extra_args=("--no-lab-copy",), slug="s1", **kw):
    return build_all.main(["--slug", slug, "--cache-dir", str(tmp_path / "cache"),
                           "--registry-dir", str(reg), "--out-root", str(tmp_path / "out"),
                           *extra_args], **kw)


def write_yaml(path, **over):
    d = {"id": path.stem, "module": "x.y", "title": "t", "langs": ["en"], "print_width_in": 5.5,
         "inputs": [], "protocol": {}, "cohorts": [], "report_refs": []}
    d.update(over)
    d = {k: v for k, v in d.items() if v is not None}
    import yaml
    path.write_text(yaml.safe_dump(d))


# ---------------------------------------------------------------- registry
def test_registry_rejects_unknown_key(tmp_path):
    write_yaml(tmp_path / "A-1.yaml", surprise=1)
    with pytest.raises(registry.RegistryError, match="unknown keys"):
        registry.load_registry(tmp_path)


def test_registry_rejects_missing_key_and_wrong_id(tmp_path):
    write_yaml(tmp_path / "A-1.yaml", title=None)
    with pytest.raises(registry.RegistryError, match="missing keys"):
        registry.load_entry(tmp_path / "A-1.yaml")
    write_yaml(tmp_path / "B-2.yaml", id="B-3")
    with pytest.raises(registry.RegistryError, match="file stem"):
        registry.load_entry(tmp_path / "B-2.yaml")


def test_registry_sorted_and_list_cli(tmp_path, capsys):
    write_yaml(tmp_path / "B-2.yaml")
    write_yaml(tmp_path / "A-1.yaml")
    assert [e.id for e in registry.load_registry(tmp_path)] == ["A-1", "B-2"]
    assert registry.main(["--dir", str(tmp_path), "--list"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 2 and lines[0].startswith("A-1")


# ---------------------------------------------------------------- build
def test_selftest_build_writes_everything(tmp_path):
    reg = registry_dir(tmp_path)
    assert run_build(tmp_path, reg) == 0
    out = tmp_path / "out" / "s1"
    for n in ["SELFTEST.en.png", "SELFTEST.en.pdf", "SELFTEST.ko.png", "SELFTEST.ko.pdf",
              "SELFTEST.values.csv", "SELFTEST.source.json", "build_manifest.json", "FIGURE_INDEX.json"]:
        assert (out / n).is_file() and (out / n).stat().st_size > 0, n
    src = json.loads((out / "SELFTEST.source.json").read_text())
    assert src["cuda_visible_devices"] == ""  # child saw CUDA_VISIBLE_DEVICES == ""
    assert src["n_values_rows"] == 5 and src["figure_id"] == "SELFTEST"
    with open(out / "SELFTEST.values.csv", newline="") as f:
        assert next(csv.reader(f))[:11] == fio.VALUES_COLUMNS
    man = json.loads((out / "build_manifest.json").read_text())
    assert man["exit_code"] == 0 and man["figures"][0]["rc"] == 0
    idx = json.loads((out / "FIGURE_INDEX.json").read_text())
    assert set(idx) == {"slug", "cache_manifest_sha256", "figures"}
    assert set(idx["figures"]["SELFTEST"]) == {
        "files", "cache_manifest_sha256", "run_names", "wandb_ids", "projects", "cohorts",
        "protocol", "values_csv", "source_json"}


def test_child_cuda_env_is_forced_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    assert run_build(tmp_path, registry_dir(tmp_path)) == 0
    src = json.loads((tmp_path / "out" / "s1" / "SELFTEST.source.json").read_text())
    assert src["cuda_visible_devices"] == ""
    assert build_all.figure_env()["MPLBACKEND"] == "Agg"


def test_failing_figure_exit1_other_still_built(tmp_path, monkeypatch):
    (tmp_path / "rfr_bad_mod.py").write_text(BAD_MODULE)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    reg = registry_dir(tmp_path, extra=[("BAD-1", "rfr_bad_mod")])
    assert run_build(tmp_path, reg) == 1
    out = tmp_path / "out" / "s1"
    man = json.loads((out / "build_manifest.json").read_text())
    rcs = {f["id"]: f for f in man["figures"]}
    assert rcs["BAD-1"]["rc"] != 0 and "boom from test" in "\n".join(rcs["BAD-1"]["stderr_tail"])
    assert rcs["SELFTEST"]["rc"] == 0 and (out / "SELFTEST.en.png").is_file()
    assert man["exit_code"] == 1
    assert "BAD-1" not in json.loads((out / "FIGURE_INDEX.json").read_text())["figures"]


def test_values_without_required_columns_fails(tmp_path, monkeypatch):
    (tmp_path / "rfr_novalues_mod.py").write_text(NOVALUES_MODULE)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    d = tmp_path / "reg"
    d.mkdir()
    shutil.copy(SELFTEST_YAML, d / "NV-1.yaml")
    (d / "NV-1.yaml").write_text(SELFTEST_YAML.read_text().replace("SELFTEST", "NV-1", 1).replace(
        "tools.figures.figs._selftest_fig", "rfr_novalues_mod"))
    assert run_build(tmp_path, d) == 1
    man = json.loads((tmp_path / "out" / "s1" / "build_manifest.json").read_text())
    assert man["figures"][0]["rc"] != 0
    assert "required columns" in "\n".join(man["figures"][0]["stderr_tail"])


def test_existing_nonempty_output_dir_exit3(tmp_path):
    out = tmp_path / "out" / "s1"
    out.mkdir(parents=True)
    keep = out / "keep.txt"
    keep.write_text("x")
    assert run_build(tmp_path, registry_dir(tmp_path)) == 3
    assert keep.read_text() == "x" and [p.name for p in out.iterdir()] == ["keep.txt"]


def test_lab_copy_matches_by_sha256(tmp_path):
    lab = tmp_path / "lab"
    rc = run_build(tmp_path, registry_dir(tmp_path), extra_args=("--lab-copy-root", str(lab)))
    assert rc == 0
    out, dest = tmp_path / "out" / "s1", lab / "s1"
    names = sorted(p.name for p in out.iterdir())
    assert names == sorted(p.name for p in dest.iterdir()) and "LAB_COPY.sha256" in names
    for n in names:
        assert fio.sha256_file(out / n) == fio.sha256_file(dest / n), n
    first = (out / "LAB_COPY.sha256").read_text().split()[0]
    assert len(first) == 64
    assert run_build(tmp_path, registry_dir(tmp_path, "reg2"), slug="s1",
                     extra_args=("--lab-copy-root", str(lab))) == 3  # never into existing dirs


def test_lab_copy_flags_exit2_and_nothing_written(tmp_path):
    reg = registry_dir(tmp_path)
    assert run_build(tmp_path, reg, extra_args=()) == 2
    assert run_build(tmp_path, reg, extra_args=("--no-lab-copy", "--lab-copy-root", str(tmp_path / "lab"))) == 2
    assert not (tmp_path / "out").exists() and not (tmp_path / "lab").exists()


def test_extractor_failure_exit1(tmp_path):
    fam = tmp_path / "families.yaml"
    fam.write_text("x: 1\n")
    rc = run_build(tmp_path, registry_dir(tmp_path),
                   extra_args=("--no-lab-copy", "--refresh-cache", "--families", str(fam)),
                   extractor_cmd=[sys.executable, "-c", "import sys; sys.exit(7)"])
    assert rc == 1 and not (tmp_path / "out").exists()


def test_extractor_gets_wandb_env_and_runs(tmp_path):
    fam = tmp_path / "families.yaml"
    fam.write_text("x: 1\n")
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import os, sys, pathlib\n"
        "d = {k: os.environ[k] for k in ['WANDB_MODE','WANDB_DIR','WANDB_CACHE_DIR','WANDB_CONFIG_DIR','WANDB_DATA_DIR']}\n"
        "assert d['WANDB_MODE']=='disabled' and len({d[k] for k in d if k!='WANDB_MODE'})==1\n"
        "out = pathlib.Path(sys.argv[sys.argv.index('--out')+1]); out.mkdir()\n"
        "(out/'MANIFEST.files.csv').write_text('file,sha256,n_rows,n_runs\\n')\n")
    rc = run_build(tmp_path, registry_dir(tmp_path),
                   extra_args=("--no-lab-copy", "--refresh-cache", "--families", str(fam)),
                   extractor_cmd=[sys.executable, str(probe)])
    assert rc == 0
    idx = json.loads((tmp_path / "out" / "s1" / "FIGURE_INDEX.json").read_text())
    assert idx["cache_manifest_sha256"] == fio.sha256_file(tmp_path / "cache" / "MANIFEST.files.csv")


def test_figure_index_projects_from_values_csv(tmp_path):
    vals = tmp_path / "F.values.csv"
    vals.write_text(
        ",".join(fio.VALUES_COLUMNS) + "\n"
        "F,a,s,runA,idA,ProjX,legacy,m,1,0.5,r\n"
        "F,a,s,runB,idB,ProjY,d1,m,2,0.6,r\n"
        "F,a,s,runB,idB,ProjY,d1,m,3,0.7,r\n"
        "F,b,s,,,,,m,1,0.1,timeline row 3\n")
    src = tmp_path / "F.source.json"
    src.write_text(json.dumps({"protocol": {"metric_test": "a", "metric_train": "b"}}))
    e = build_all.figure_index_entry(["F.en.png"], vals, src, "abc")
    assert e["projects"] == ["ProjX", "ProjY"]
    assert e["run_names"] == ["runA", "runB"] and e["wandb_ids"] == ["idA", "idB"]
    assert e["cohorts"] == ["d1", "legacy"] and e["protocol"]["metric_test"] == "a"
    assert e["values_csv"] == "F.values.csv" and e["source_json"] == "F.source.json"


# ---------------------------------------------------------------- style / io
def test_finalize_raises_below_7pt_and_passes_above():
    style.apply_style("en")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlabel("step")
    ax.plot([0, 1], [0, 1])
    assert style.finalize(fig, 5.5) == pytest.approx(5.5 / 8)  # 16 pt * 0.69 = 11 pt
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlabel("step", fontsize=8)  # 8 * 0.6875 = 5.5 pt
    with pytest.raises(ValueError, match="below 7"):
        style.finalize(fig, 5.5)
    plt.close(fig)


def test_word_and_palette():
    assert style.word("en", "step") == "step" and style.word("ko", "step") == "훈련 step"
    assert style.word("ko", "d1") == "2세대 데이터"
    with pytest.raises(KeyError):
        style.word("en", "nope")
    assert style.DIVERGING_BLUE == "#184f95"


def test_load_cache_csv_sha_mismatch(tmp_path):
    f = tmp_path / "a.csv"
    f.write_text("x,y\n1,2\n")
    man = tmp_path / "MANIFEST.files.csv"
    man.write_text(f"file,sha256,n_rows,n_runs\na.csv,{fio.sha256_file(f)},1,1\n")
    assert list(fio.load_cache_csv(tmp_path, "a.csv").columns) == ["x", "y"]
    f.write_text("x,y\n1,3\n")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        fio.load_cache_csv(tmp_path, "a.csv")
    with pytest.raises(FileNotFoundError):
        fio.load_cache_csv(tmp_path / "nowhere", "a.csv")


def test_sha256_file(tmp_path):
    f = tmp_path / "z"
    f.write_bytes(b"abc")
    assert fio.sha256_file(f) == hashlib.sha256(b"abc").hexdigest()
