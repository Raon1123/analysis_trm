"""EXP-017 §6 gates G0..G7 (+ I-2 sealed guard, I-3 key check, I-4 terminal step).

gates.py records raw gate quantities. judge.g3_pass / judge.g4_pass decide pass/fail for the tolerance
gates (G3, G4), with tolerances from judge.FROZEN_RULES["gates"] (VERIFY-r1 N2), so the witnessed judge
is authoritative. The flags written here are informational copies of the same predicates.
"""

from __future__ import annotations

import csv
import filecmp
import os

import torch

from analysis.z_dynamics import judge
from analysis.z_dynamics.pr_traj import pr_of_latent


class SealedAccess(RuntimeError):
    """I-2: any read of a sealed/ or d0_sealed/ path invalidates the whole experiment."""


def guard_path(path: str) -> str:
    parts = os.path.normpath(os.path.abspath(path)).split(os.sep)
    if any(p in ("sealed", "d0_sealed") or p.endswith("_sealed") for p in parts):
        raise SealedAccess(f"I-2: refusing to read sealed path {path}")
    return path


def g0_code_identity(run_dir: str, arch: str) -> dict:
    """G0: plain byte comparison (filecmp shallow=False, i.e. cmp; never rtk diff) of the run's code copy
    against the model file this process actually imports."""
    import models.recursive_reasoning.trm as m_trm
    import models.recursive_reasoning.trm_singlez as m_sz
    mod = m_trm if arch == "trm" else m_sz
    fname = "trm.py" if arch == "trm" else "trm_singlez.py"
    run_copy = os.path.join(run_dir, fname)
    repo = os.path.abspath(mod.__file__)
    ok = os.path.isfile(run_copy) and filecmp.cmp(run_copy, repo, shallow=False)
    return {"pass": bool(ok), "run_copy": run_copy, "repo_file": repo}


def _get(cfg: dict, dotted: str):
    node = cfg
    for p in dotted.split("."):
        if not isinstance(node, dict) or p not in node:
            return "<absent>"
        node = node[p]
    return node


def data_root(data_path: str) -> str:
    return os.path.dirname(os.path.normpath(data_path))


def g7_config(all_cfg: dict) -> dict:
    """G7: the §6 key list (parsed from judge.FROZEN_RULES gates.G7_TEXT) must match exactly."""
    exp = judge.parse_g7_expected()
    mism = {}
    for key, want in exp.items():
        got = _get(all_cfg, key)
        if isinstance(want, bool) or isinstance(got, bool):
            same = got is want
        else:
            same = got == want
        if not same:
            mism[key] = {"want": want, "got": got}
    dp = all_cfg["data_paths"][0]
    values = {"arch.name": _get(all_cfg, "arch.name"), "epochs": _get(all_cfg, "epochs"), "data_root": data_root(dp)}
    return {"pass": not mism, "mismatches": mism, "values": values}


def snapshot_reference(run_dir: str, step: int) -> dict:
    """z_snapshots/step_<N>.pt: PR of z_H / z_L (trm; None latents for trm_singlez) and the train
    correct_mask mean (both archs)."""
    p = os.path.join(run_dir, "z_snapshots", f"step_{step}.pt")
    d = torch.load(p, map_location="cpu", weights_only=False)
    cm = d["correct_mask"]
    return {"path": p, "pr_zH_snapshot": pr_of_latent(d.get("z_H")), "pr_zL_snapshot": pr_of_latent(d.get("z_L")),
            "snapshot_train_exact": float(cm.float().mean().item()), "n": int(cm.numel()),
            "labels": d.get("labels")}


def closeout_lookup(path: str) -> dict[str, dict]:
    """reports/pp_campaign_retro_trajectory/per_run_trajectory.csv -> run_name -> row."""
    with open(path, newline="") as f:
        return {r["run_name"]: r for r in csv.DictReader(f)}


def informational_flags(rec: dict) -> dict:
    return {"G3_pass": judge.g3_pass(rec.get("G3"), rec["arch"]),
            "G4_pass": judge.g4_pass(rec.get("G4"), rec.get("final_probe_test_exact"))}
