"""Tests for tools/sigmak_data_audit.py and the sigma^k builder's cross-k guard."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import subprocess

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


audit_mod = _load("sigmak_data_audit", ROOT / "tools" / "sigmak_data_audit.py")


def _write_split(root: Path, k: int, split: str, sigmas: np.ndarray, labels: np.ndarray | None = None):
    n = sigmas.shape[1]
    seq = n + 1
    inp = np.zeros((len(sigmas), seq), dtype=np.int32)
    lbl = np.zeros((len(sigmas), seq), dtype=np.int32)
    inp[:, :n] = sigmas + 1
    lbl[:, :n] = (audit_mod.power(sigmas, k) if labels is None else labels) + 1
    d = root / str(k) / split
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "all__inputs.npy", inp)
    np.save(d / "all__labels.npy", lbl)


def _perms_with_order_gt(k: int, count: int, rng, n: int = 6):
    out, seen = [], set()
    while len(out) < count:
        s = rng.permutation(n).astype(np.int64)
        if audit_mod.order_of(audit_mod.cycle_type(s)) <= k or s.tobytes() in seen:
            continue
        seen.add(s.tobytes())
        out.append(s)
    return np.stack(out)


def test_clean_dataset_and_floors(tmp_path):
    rng = np.random.default_rng(0)
    for k in (2, 3):
        pool = _perms_with_order_gt(k, 30, rng)
        _write_split(tmp_path, k, "train", pool[:20])
        _write_split(tmp_path, k, "test", pool[20:])
    rep = audit_mod.audit(tmp_path, [2, 3])
    assert rep["integrity_errors"] == []
    for r in rep["per_k"]:
        f = r["test"]["floors"]
        # ord(sigma) > k  =>  sigma^k != id and sigma^k != sigma (k >= 2)
        assert f["identity_exact"] == 0.0 and f["copy_sigma_exact"] == 0.0
        assert r["train"]["min_order"] > r["k"]


def test_detects_label_mismatch_and_within_k_overlap(tmp_path):
    rng = np.random.default_rng(1)
    pool = _perms_with_order_gt(2, 10, rng)
    bad = audit_mod.power(pool[:5], 2).copy()
    bad[0] = bad[0][::-1]
    _write_split(tmp_path, 2, "train", pool[:5], labels=bad)
    _write_split(tmp_path, 2, "test", pool[4:])  # pool[4] shared with train
    rep = audit_mod.audit(tmp_path, [2])
    joined = " ".join(rep["integrity_errors"])
    assert "labels != sigma^k" in joined
    assert "train ∩ test = 1" in joined
    assert audit_mod.main([str(tmp_path)]) == 1


def test_cross_k_overlap_exit_codes(tmp_path):
    rng = np.random.default_rng(2)
    pool = _perms_with_order_gt(3, 40, rng)
    _write_split(tmp_path, 2, "train", pool[:20])
    _write_split(tmp_path, 2, "test", pool[20:30])
    _write_split(tmp_path, 3, "train", pool[20:30])  # = test(k=2)
    _write_split(tmp_path, 3, "test", pool[30:40])
    rep = audit_mod.audit(tmp_path, [2, 3])
    pair = {(c["train_k"], c["test_k"]): c["train_test_overlap"] for c in rep["cross_k"]}
    assert pair[(3, 2)] == 10 and pair[(2, 3)] == 0
    assert audit_mod.main([str(tmp_path)]) == 0
    assert audit_mod.main([str(tmp_path), "--fail-on-cross-k"]) == 2
    out = tmp_path / "r.json"
    audit_mod.main([str(tmp_path), "--json", str(out)])
    assert json.loads(out.read_text())["cross_k_overlap_total"] == 10


def test_order_filter_violation(tmp_path):
    ident_ish = np.array([[1, 0, 2, 3, 4, 5]], dtype=np.int64)  # ord 2
    other = _perms_with_order_gt(2, 3, np.random.default_rng(3))
    _write_split(tmp_path, 2, "train", ident_ish)
    _write_split(tmp_path, 2, "test", other)
    rep = audit_mod.audit(tmp_path, [2])
    assert any("order filter violated" in e for e in rep["integrity_errors"])
    assert audit_mod.audit(tmp_path, [2], order_filter=False)["integrity_errors"] == []


def test_missing_root_exit_3(tmp_path):
    assert audit_mod.main([str(tmp_path / "nope")]) == 3
    assert audit_mod.main([str(tmp_path)]) == 3  # no <k> dirs


# ---------------------------------------------------------------------------
# Builder guard (dataset/build_sigma_k_dataset.py)
# ---------------------------------------------------------------------------

BUILDER = ROOT / "dataset" / "build_sigma_k_dataset.py"


def _build(out: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(BUILDER), "--output-dir", str(out), *args],
        capture_output=True, text=True, cwd=ROOT,
    )


@pytest.mark.skipif(not (ROOT / "data/sigma_k_10/3/train/all__inputs.npy").exists(),
                    reason="canonical data/sigma_k_10/3 not present")
def test_default_build_is_byte_identical_to_canonical_k3(tmp_path):
    r = _build(tmp_path, "--only-k", "3")
    assert r.returncode == 0, r.stderr
    for split in ("train", "test"):
        for name in ("inputs", "labels"):
            got = np.load(tmp_path / "3" / split / f"all__{name}.npy")
            ref = np.load(ROOT / "data/sigma_k_10/3" / split / f"all__{name}.npy")
            assert np.array_equal(got, ref), (split, name)
    man = json.loads((tmp_path / "3" / "manifest.json").read_text())
    assert man["rng_seed"] == 42 and man["cross_k_disjoint_from"] == []
    assert man["order_range"]["train"][0] > 3 and man["train_test_overlap"] == 0


def test_cross_k_disjoint_build_has_zero_cross_k_overlap(tmp_path):
    # Sizes must be large enough that per-k seeds alone would collide; at 300/100 a
    # build with the shared seen-set removed still audits clean (verifier mutation m1).
    r = _build(tmp_path, "--cross-k-disjoint", "--train-size", "1500", "--test-size", "500")
    assert r.returncode == 0, r.stderr
    ks = audit_mod.discover_ks(tmp_path)
    assert ks == [3, 4, 5, 6, 7, 8, 10]
    rep = audit_mod.audit(tmp_path, ks)
    assert rep["integrity_errors"] == []
    assert rep["cross_k_overlap_total"] == 0
    assert all(c["train_train_overlap"] == 0 for c in rep["cross_k_train_train"])
    assert audit_mod.main([str(tmp_path), "--fail-on-cross-k"]) == 0
    man = json.loads((tmp_path / "10" / "manifest.json").read_text())
    assert man["cross_k_disjoint_from"] == [3, 4, 5, 6, 7, 8]


def test_legacy_small_build_does_overlap_across_k(tmp_path):
    # Documents the hazard the guard fixes: same seed stream reused per k.
    r = _build(tmp_path, "--train-size", "300", "--test-size", "100", "--no-write-manifest")
    assert r.returncode == 0, r.stderr
    rep = audit_mod.audit(tmp_path, audit_mod.discover_ks(tmp_path))
    assert rep["cross_k_overlap_total"] > 0
    assert not (tmp_path / "3" / "manifest.json").exists()


def test_cross_k_disjoint_rejects_single_k(tmp_path):
    r = _build(tmp_path, "--cross-k-disjoint", "--only-k", "3")
    assert r.returncode != 0
    assert "requires a full multi-k build" in (r.stderr + r.stdout)
    assert not (tmp_path / "identifiers.json").exists()  # rejected before any write


def test_json_write_failure_is_usage_error(tmp_path):
    rng = np.random.default_rng(4)
    pool = _perms_with_order_gt(2, 10, rng)
    _write_split(tmp_path, 2, "train", pool[:5])
    _write_split(tmp_path, 2, "test", pool[5:])
    assert audit_mod.main([str(tmp_path), "--json", str(tmp_path / "no" / "dir" / "r.json")]) == 3
