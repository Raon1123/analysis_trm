"""Tests for tools/statebench_tiers/build_dataset.py (EXP-027 dataset builder, DRAFT).

Run: CUDA_VISIBLE_DEVICES="" rtk uv run pytest tests/test_statebench_data.py
"""
import filecmp
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.statebench_tiers import build_dataset as B  # noqa: E402

TRAIN, TEST = 200, 50
NPY = ("inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices")


def build(out, tier="G", N=16, **kw):
    args = dict(tier=tier, N=N, out=out, seed_train=B.SEED_TRAIN_DEFAULT, seed_test=B.SEED_TEST_DEFAULT,
                m=5, q=127, train=TRAIN, test=TEST, c_min=0, c_max=0, date="2026-11-11")
    args.update(kw)
    return B.build(**args)


@pytest.fixture(scope="module")
def cells(tmp_path_factory):
    root = tmp_path_factory.mktemp("cells")
    out = {}
    for tier, N in (("G", 16), ("S", 8), ("A", 16)):
        out[tier] = (root / tier, build(root / tier, tier=tier, N=N))
    return out


def load(d, split, name):
    return np.load(d / split / f"all__{name}.npy")


def test_determinism_byte_identical(tmp_path):
    build(tmp_path / "a")
    build(tmp_path / "b")
    for split in ("train", "test"):
        for p in sorted((tmp_path / "a" / split).iterdir()):
            assert filecmp.cmp(p, tmp_path / "b" / split / p.name, shallow=False), p.name
    assert filecmp.cmp(tmp_path / "a" / "build_manifest.json", tmp_path / "b" / "build_manifest.json",
                       shallow=False)


def test_seed_changes_data(tmp_path):
    build(tmp_path / "a")
    build(tmp_path / "b", seed_train=B.SEED_TRAIN_DEFAULT + 5)
    assert not np.array_equal(load(tmp_path / "a", "train", "inputs"), load(tmp_path / "b", "train", "inputs"))


@pytest.mark.parametrize("tier,N", [("G", 16), ("S", 8), ("A", 16)])
def test_shapes_dtypes_metadata(cells, tier, N):
    d, man = cells[tier]
    seq = 17 + 2 * N
    for split, rows in (("train", TRAIN), ("test", TEST)):
        meta = json.loads((d / split / "dataset.json").read_text())
        assert meta == {"pad_id": 0, "ignore_label_id": 0, "blank_identifier_id": 0, "vocab_size": 261,
                        "seq_len": seq, "num_puzzle_identifiers": 1, "total_groups": rows,
                        "mean_puzzle_examples": 1.0, "total_puzzles": rows, "sets": ["all"]}
        for name in NPY:
            a = load(d, split, name)
            assert a.dtype == np.int32
        assert load(d, split, "inputs").shape == (rows, seq)
        assert load(d, split, "labels").shape == (rows, seq)
        assert np.array_equal(load(d, split, "puzzle_indices"), np.arange(rows + 1))
        assert np.array_equal(load(d, split, "group_indices"), np.arange(rows + 1))
        assert not load(d, split, "puzzle_identifiers").any()
    assert man["seq_len"] == seq and man["rows"] == {"train": TRAIN, "test": TEST}


@pytest.mark.parametrize("tier", ["G", "S", "A"])
def test_label_layout_and_vocab_bound(cells, tier):
    d, _ = cells[tier]
    for split in ("train", "test"):
        inp, lab = load(d, split, "inputs"), load(d, split, "labels")
        n_op = inp.shape[1] - 5
        assert inp.min() >= 1 and inp.max() <= 260  # no PAD, within vocab 261
        assert (lab[:, :n_op] == 0).all()                       # op positions ignored
        assert (lab[:, n_op:] >= 7).all() and (lab[:, n_op:] <= 133).all()  # VAL tokens only
        assert (inp[:, n_op:] == np.arange(1, 6)).all()         # answer slots = REG_0..REG_4
        assert (inp[:, 0:n_op:2] >= 1).all() and (inp[:, 0:n_op:2] <= 6).all()  # targets REG/TMP
        # first 6 assignments are the initial constants: VAL arguments
        assert ((inp[:, 1:12:2] >= 7) & (inp[:, 1:12:2] <= 133)).all()


def test_tier_argument_token_ranges(cells):
    gi = load(cells["G"][0], "train", "inputs")[:, :-5]
    si = load(cells["S"][0], "train", "inputs")[:, :-5]
    ai = load(cells["A"][0], "train", "inputs")[:, :-5]
    assert (gi[:, 1::2] < 134).all()                 # G: const (VAL) or copy (REG), never INC
    assert (si[:, 1::2] >= 134).any() and not (si[:, 1::2] < 7).any()  # S: VAL or INC
    assert ((ai[:, 1::2] >= 7) & (ai[:, 1::2] <= 133)).all()  # A: constants only


def test_labels_match_final_state_by_execution(cells):
    # Re-execute G inputs from tokens and compare with the labels.
    d, _ = cells["G"]
    inp, lab = load(d, "train", "inputs"), load(d, "train", "labels")
    for r in range(TRAIN):
        st = [None] * 6
        for j in range(0, inp.shape[1] - 5, 2):
            t, a = int(inp[r, j]) - 1, int(inp[r, j + 1])
            st[t] = a - 7 if a >= 7 else st[a - 1]
        assert [v + 7 for v in st[:5]] == list(lab[r, -5:])


@pytest.mark.parametrize("tier", ["G", "S", "A"])
def test_train_test_disjoint_and_unique(cells, tier):
    d, man = cells[tier]
    tr = {bytes(r) for r in load(d, "train", "inputs")}
    te = {bytes(r) for r in load(d, "test", "inputs")}
    assert len(tr) == TRAIN and len(te) == TEST
    assert not tr & te


def test_dedupe_drops_are_recorded_and_enforced(tmp_path):
    # N = 8 at c = 0 has only 2 swaps, so collisions are likely at larger sizes; force them by
    # asking for many rows on a small space and check bookkeeping and disjointness.
    man = build(tmp_path / "s", tier="S", N=8, train=3000, test=500)
    st = man["dedupe"]
    assert st["train"]["drawn"] == 3000 + st["train"]["dropped_dup_within_split"]
    assert st["test"]["drawn"] == 500 + st["test"]["dropped_dup_within_split"] + st["test"]["dropped_dup_vs_other_split"]
    tr = {bytes(r) for r in load(tmp_path / "s", "train", "inputs")}
    te = {bytes(r) for r in load(tmp_path / "s", "test", "inputs")}
    assert len(tr) == 3000 and len(te) == 500 and not tr & te


def test_manifest_contents(cells):
    d, man = cells["G"]
    assert man["date"] == "2026-11-11"
    assert man["generator_sha256"] == B.sha256_file(Path(B.HERE) / "sb_generator.py")
    assert man["swap_count_distribution"]["train"] == {"5": TRAIN} or man["swap_count_distribution"]["train"] == {5: TRAIN}
    for rel, sha in man["files_sha256"].items():
        assert B.sha256_file(d / rel) == sha
    assert str(d) not in json.dumps(man)  # no absolute paths -> reproducible across dirs


def test_loadable_by_repo_puzzle_dataset_on_cpu(cells):
    pd = pytest.importorskip("puzzle_dataset")
    d, _ = cells["G"]
    cfg = pd.PuzzleDatasetConfig(seed=0, dataset_paths=[str(d)], global_batch_size=64, test_set_mode=False,
                                 epochs_per_iter=1, rank=0, num_replicas=1)
    train = pd.PuzzleDataset(cfg, "train")
    assert train.metadata.vocab_size == 261 and train.metadata.seq_len == 49
    set_name, batch, size = next(iter(train))
    assert batch["inputs"].shape == (64, 49) and batch["labels"].shape == (64, 49)
    labs = batch["labels"]
    assert ((labs == -100).sum(1) == 44).all()   # only the 5 answer slots are scored
    test_cfg = pd.PuzzleDatasetConfig(seed=0, dataset_paths=[str(d)], global_batch_size=64, test_set_mode=True,
                                      epochs_per_iter=1, rank=0, num_replicas=1)
    test = pd.PuzzleDataset(test_cfg, "test")
    n = sum(int(b["inputs"].shape[0]) for _, b, _ in test)
    assert n == 64  # TEST=50 rows padded to one batch of 64


# ---------------------------------------------------------------------------------------------
# P3 audit (exp027_data_audit.py) and P2 floors (p2_floors.py)
# ---------------------------------------------------------------------------------------------
import shutil  # noqa: E402

from tools.statebench_tiers import exp027_data_audit as AUD  # noqa: E402
from tools.statebench_tiers import p2_floors as FL  # noqa: E402


@pytest.fixture(scope="module")
def named(cells, tmp_path_factory):
    """Read-only copies of the built cells under the real naming scheme (<tier>_N<N>)."""
    root = tmp_path_factory.mktemp("named")
    out = {}
    for t, N in (("G", 16), ("S", 8), ("A", 16)):
        out[t] = root / f"{t}_N{N}"
        shutil.copytree(cells[t][0], out[t])
    return out


@pytest.fixture()
def scratch_cell(cells, tmp_path):
    """Copy of a built cell (named like the real ones) without its manifest, so a mutation has to be
    caught by the independent checks rather than by the manifest hashes."""
    def make(tier):
        dst = tmp_path / f"{tier}_N{16 if tier != 'S' else 8}"
        shutil.copytree(cells[tier][0], dst)
        (dst / "build_manifest.json").unlink()
        return dst
    return make


def _save(cell, split, name, arr):
    np.save(cell / split / f"all__{name}.npy", arr)


@pytest.mark.parametrize("tier", ["G", "S", "A"])
def test_audit_passes_on_clean_cell(named, tier):
    rep = AUD.audit(named[tier], siblings=False)
    assert rep["passed"], rep["failed_checks"]
    assert rep["tier"] == tier


def test_audit_cli_exit_codes(named, tmp_path):
    assert AUD.main([str(named["G"]), "--json", str(tmp_path / "r.json"), "--no-siblings"]) == 0
    assert json.loads((tmp_path / "r.json").read_text())["passed"] is True
    assert AUD.main([str(tmp_path / "nonexistent_N3")]) == 2
    assert AUD.main([str(tmp_path / "badname")]) == 2


@pytest.mark.parametrize("tier", ["G", "S"])
def test_audit_mutation_corrupt_label_exits_1(scratch_cell, tier):
    c = scratch_cell(tier)
    lab = load(c, "train", "labels").copy()
    lab[3, -1] = 7 + (lab[3, -1] - 7 + 1) % 127
    _save(c, "train", "labels", lab)
    assert AUD.main([str(c), "--no-siblings"]) == 1
    assert "train.labels_match_execution" in AUD.audit(c, siblings=False)["failed_checks"]


@pytest.mark.parametrize("tier", ["G", "S"])
def test_audit_mutation_corrupt_input_token_exits_1(scratch_cell, tier):
    c = scratch_cell(tier)
    inp = load(c, "train", "inputs").copy()
    j = next(j for j in range(0, 12, 2) if inp[0, j] != 6)   # initial constant of an ordinary register
    inp[0, j + 1] = 7 + (inp[0, j + 1] - 7 + 1) % 127
    _save(c, "train", "inputs", inp)
    assert AUD.main([str(c), "--no-siblings"]) == 1
    assert "train.labels_match_execution" in AUD.audit(c, siblings=False)["failed_checks"]


@pytest.mark.parametrize("tier", ["G", "S"])
def test_audit_mutation_test_row_duplicated_into_train_exits_1(scratch_cell, tier):
    c = scratch_cell(tier)
    for name in ("inputs", "labels"):   # an otherwise valid row, so only disjointness can catch it
        tr = load(c, "train", name).copy()
        tr[7] = load(c, "test", name)[0]
        _save(c, "train", name, tr)
    rep = AUD.audit(c, siblings=False)
    assert AUD.main([str(c), "--no-siblings"]) == 1
    assert rep["failed_checks"] == ["train_test_disjoint"]


def test_audit_mutation_duplicate_within_train_and_manifest_tamper(cells, tmp_path):
    c = tmp_path / "G_N16"
    shutil.copytree(cells["G"][0], c)
    inp = load(c, "train", "inputs").copy()
    lab = load(c, "train", "labels").copy()
    inp[9], lab[9] = inp[8], lab[8]
    _save(c, "train", "inputs", inp)
    _save(c, "train", "labels", lab)
    failed = AUD.audit(c, siblings=False)["failed_checks"]
    assert "train.rows_unique" in failed and "manifest_sha256" in failed


def test_audit_sibling_alignment_detects_misaligned_tier(cells, tmp_path):
    for t in ("G", "A"):
        shutil.copytree(cells[t][0], tmp_path / f"{t}_N16")
    assert AUD.audit(tmp_path / "G_N16")["passed"]
    lab = load(tmp_path / "A_N16", "test", "labels").copy()
    lab[[0, 1]] = lab[[1, 0]]          # rows swapped in A only: labels no longer match G row-for-row
    _save(tmp_path / "A_N16", "test", "labels", lab)
    assert "sibling_alignment.A_N16" in AUD.audit(tmp_path / "G_N16")["failed_checks"]


def test_audit_g_swap_statistics(named):
    st = AUD.audit(named["G"], siblings=False)["tier_stats"]
    assert st["swap_count_hist"] == {5: TRAIN}
    assert st["total_non_swap_non_scratch_ops"] == 0
    assert st["frac_rows_with_overlapping_swaps"] > 0.9 and st["distinct_final_permutations"] > 1


def test_independent_interpreter_matches_builder_encoding(cells):
    # the audit's own executor and the builder's generator agree on every tier (labels come from the generator)
    for t in ("G", "S", "A"):
        d = cells[t][0]
        inp, lab = load(d, "train", "inputs"), load(d, "train", "labels")
        final, bad = AUD.execute(inp)
        assert not bad.any() and (lab[:, -5:] == final[:, :5] + 7).all()


def test_p2_floors_on_tiny_cells(cells):
    g = FL.cell_floors(cells["G"][0], "G")
    a = FL.cell_floors(cells["A"][0], "A")
    assert a["predictors"]["lastlit"]["exact"] == 1.0          # aperiodic: last constant IS the answer
    assert a["F_prereg"] < 1.0 and a["F_all"] < 1.0           # ... and is excluded from F
    for v in (g, a):
        for p in v["predictors"].values():
            assert 0.0 <= p["exact"] <= p["token_acc"] <= 1.0
    assert "modal_perm" in g["predictors"] and "modal_perm" not in a["predictors"]
    # G has no resets at c = 0, so `init` and `lastlit` coincide
    assert g["predictors"]["init"] == g["predictors"]["lastlit"]
