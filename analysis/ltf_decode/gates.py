"""P-C1 per-cell gates (spec §2.5 table): G0', G1', G-P3, G5', G6, G7', I-2/I-3/I-4.

Tolerances / terminal step / G5 unit are read from the parsed FROZEN_RULES ``gates`` block (I-9). Items that
the spec states outside the YAML (the G0' code sha256, the explicit G7' arch values, the E15 §6 top-level key
list) are constants below; tests assert that each appears verbatim in the prereg / EXP-015 text.
"""

from __future__ import annotations

import filecmp
import os

import numpy as np
import torch

from analysis.ltf_decode import judge
from analysis.ltf_decode.trace import g1_equal  # noqa: F401  (G1' predicate lives next to the tracer)
from analysis.z_dynamics.gates import SealedAccess, guard_path  # noqa: F401  (I-2, imported unmodified)
from analysis.z_dynamics.readouts import labels_are_sigma_k

# spec §2.5 G0' row: sha256 of the repo transformers_baseline.py == the k5 s1 run copy (== E15:350 frozen sha)
G0_CODE_SHA256 = "a8a4c5da76b86a7c3bf595da3c7980e0aca45a5d786a041a3f2a5556b8e23ba6"

# spec §2.5 G7' row (arch block); loop_grad_cycles depends on the arm (g7 = 7, full = 0)
G7_ARCH_EXPECTED = {"H_layers": 2, "hidden_size": 512, "num_heads": 12, "expansion": 4, "pos_encodings": "rope",
                    "puzzle_emb_ndim": 512, "halt_max_steps": 1, "loops": 21, "loop_input_injection": True}
G7_LOOP_GRAD = {"g7": 7, "full": 0}

# EXP-015 §6 "Config parity" top-level key list (compared with the T reference all_config.yaml of the same (k, s))
G7_TOPLEVEL_KEYS = ["beta1", "beta2", "checkpoint_every_eval", "data_paths", "data_paths_test", "ema", "ema_rate",
                    "epochs", "eval_interval", "eval_save_outputs", "evaluators", "freeze_weights",
                    "global_batch_size", "k", "load_checkpoint", "log_z_dynamics", "lr", "lr_min_ratio",
                    "lr_warmup_steps", "min_eval_interval", "perf_benchmark", "perf_profiler", "phase_patience",
                    "phase_threshold", "project_name", "puzzle_emb_dense_update", "puzzle_emb_lr",
                    "puzzle_emb_weight_decay", "seed", "weight_decay", "z_probe_size", "z_seq_metrics",
                    "z_snapshot"]


def g0_code_identity(run_dir: str) -> dict:
    """G0': plain byte comparison (filecmp shallow=False, i.e. cmp; never rtk diff) of the run's code copy with
    the module this process imports, plus the spec sha256."""
    import hashlib
    import models.recursive_reasoning.transformers_baseline as m
    run_copy = os.path.join(run_dir, "transformers_baseline.py")
    repo = os.path.abspath(m.__file__)
    same = os.path.isfile(run_copy) and filecmp.cmp(run_copy, repo, shallow=False)
    with open(repo, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    return {"pass": bool(same and sha == G0_CODE_SHA256), "run_copy": run_copy, "repo_file": repo, "repo_sha256": sha}


def g1_prime(trace_logits_t21: torch.Tensor, native_logits: torch.Tensor) -> bool:
    return trace_logits_t21.dtype == native_logits.dtype and torch.equal(trace_logits_t21, native_logits)


def g_p3(dec_t21_test512: np.ndarray, labels512: np.ndarray, manifest_terminal_probe: float) -> dict:
    """G-P3: exact-match of y_21 on the test first 512 rows vs the manifest M-1 terminal probe/test_exact."""
    exact = float((dec_t21_test512 == labels512[:, :10]).all(-1).mean())
    tol = float(judge.num("gates.G_P3_abs_tol"))
    return {"exact_t21_test512": exact, "manifest_terminal_probe": manifest_terminal_probe, "abs_diff":
            abs(exact - manifest_terminal_probe), "tol": tol, "pass": abs(exact - manifest_terminal_probe) <= tol}


def g6_data(inputs_tok: np.ndarray, labels_tok: np.ndarray, k: int) -> dict:
    return labels_are_sigma_k(inputs_tok, labels_tok, k)


def _get(cfg: dict, dotted: str):
    node = cfg
    for p in dotted.split("."):
        if not isinstance(node, dict) or p not in node:
            return "<absent>"
        node = node[p]
    return node


def g7_config(all_cfg: dict, ref_cfg: dict | None, arm: str) -> dict:
    """G7': explicit arch values + (if ref_cfg is given) E15 §6 top-level keys equal to the T reference config."""
    exp = dict(G7_ARCH_EXPECTED, loop_grad_cycles=G7_LOOP_GRAD[arm])
    mism = {}
    for key, want in exp.items():
        got = _get(all_cfg, f"arch.{key}")
        same = (got is want) if isinstance(want, bool) or isinstance(got, bool) else (got == want)
        if not same:
            mism[f"arch.{key}"] = {"want": want, "got": got}
    if ref_cfg is None:
        mism["toplevel"] = "reference config missing"
    else:
        for key in G7_TOPLEVEL_KEYS:
            a, b = _get(all_cfg, key), _get(ref_cfg, key)
            if a != b:
                mism[key] = {"want": b, "got": a}
    return {"pass": not mism, "mismatches": mism}


def i3_keys(missing: list[str], unexpected: list[str]) -> dict:
    return {"pass": not missing and not unexpected, "missing": list(missing), "unexpected": list(unexpected)}


def i4_terminal(step: int, steps_on_disk: list[int]) -> dict:
    term = int(judge.num("gates.terminal_step"))
    return {"pass": bool(step == term and steps_on_disk and max(steps_on_disk) == step), "terminal": term}


def g5_identical(first: dict[str, str], second: dict[str, str]) -> dict:
    """G5': every output text (per_call / cell ...) identical between the first run and the cold re-run."""
    diffs = {k: first[k] != second[k] for k in first}
    return {"pass": not any(diffs.values()), "differs": diffs}


# =====================================================================================================
# G-a (spec §4.2): bit-equality of the tracer with the native forward on SYNTHETIC / random weights only.
# No checkpoint is read. The "full" arch is the arch block of pp_ltf_g7_lay2_k5_s1/all_config.yaml (config file
# only) with random init. Python:  python -m analysis.ltf_decode.gates --ga --out <json>
# =====================================================================================================

TINY_ARCH = dict(H_cycles=1, H_layers=2, hidden_size=32, expansion=4, num_heads=2, pos_encodings="rope",
                 puzzle_emb_ndim=32, halt_max_steps=1, halt_exploration_prob=0.1, act_enabled=True,
                 act_inference=False, loops=21, loop_input_injection=True, loop_grad_cycles=0)
GA_CFG_DIR = "checkpoints/power_permutation/pp_ltf_g7_lay2_k5_s1"      # only all_config.yaml is read
GA_B = 4


def _synthetic_model(cfg: dict, seed: int = 0):
    """Random-init Model_ACTV2 with a non-trivial (randomised) puzzle prefix; returns inner (eval)."""
    from analysis.ltf_decode.trace import build_ltf
    torch.manual_seed(seed)
    _w, inner = build_ltf(cfg, "cpu")
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        inner.puzzle_emb.weights.copy_(torch.randn(inner.puzzle_emb.weights.shape, generator=g))
    return inner


def _synthetic_batch(cfg: dict, seed: int = 123) -> dict:
    g = torch.Generator().manual_seed(seed)
    B, S, V = GA_B, cfg["seq_len"], cfg["vocab_size"]
    return {"inputs": torch.randint(0, V, (B, S), generator=g, dtype=torch.int32),
            "puzzle_identifiers": torch.zeros(B, dtype=torch.int32)}


def ga_arch_configs() -> dict:
    """{'tiny': cfg, 'full': cfg} model-config dicts (data fields: seq_len 11, vocab 11, 1 puzzle id)."""
    from measure_rho import build_model_config
    data = dict(batch_size=GA_B, seq_len=11, vocab_size=11, num_puzzle_identifiers=1)
    full = build_model_config(GA_CFG_DIR, batch_size=GA_B)
    return {"tiny": dict(TINY_ARCH, **data), "full": full}


def _eq_trace_native(cfg: dict, n_loops_trace: int | None = None, seed: int = 0) -> dict:
    from analysis.ltf_decode.trace import native_forward, trace_loops
    inner = _synthetic_model(cfg, seed)
    batch = _synthetic_batch(cfg)
    with torch.inference_mode():
        tr = trace_loops(inner, batch, cfg["loops"] if n_loops_trace is None else n_loops_trace, keep_latents=True)
        z_nat, out_nat = native_forward(inner, batch)
    return {"z_equal": bool(torch.equal(tr.z[-1], z_nat)), "logits_equal": bool(g1_equal_(tr.logits[-1], out_nat))}


def g1_equal_(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.dtype == b.dtype and torch.equal(a, b)


def _deep_equal(x, y) -> bool:
    import math
    if isinstance(x, dict):
        return isinstance(y, dict) and x.keys() == y.keys() and all(_deep_equal(x[k], y[k]) for k in x)
    if isinstance(x, (list, tuple)):
        return isinstance(y, (list, tuple)) and len(x) == len(y) and all(_deep_equal(a, b) for a, b in zip(x, y))
    if isinstance(x, np.ndarray):
        return isinstance(y, np.ndarray) and x.shape == y.shape and np.array_equal(x, y)
    if isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y):
        return True
    return bool(x == y)


def ga_item5(n_rows: int = 120, n_perm: int = 50, ks=(3, 5, 10)) -> list[dict]:
    """stats_j(J=17, J_all=18) == stats.analyze_decodes on synthetic decodes over real D1 sigma rows."""
    from analysis.ltf_decode.stats_j import analyze_decodes_j, synth_genuine_sequential
    from analysis.z_dynamics import stats as zstats
    from analysis.z_dynamics.readouts import sigma_infos
    res = []
    for k in ks:
        path = f"data/power_permutation/d1/{k}/test/all__inputs.npy"
        inputs = np.load(path)[:n_rows].astype(np.int64)
        labels = np.load(path.replace("inputs", "labels"))[:n_rows].astype(np.int64)
        infos = sigma_infos(inputs, k)
        rng = np.random.default_rng(k)
        decs = {"genuine_sequential": synth_genuine_sequential(infos, k, 18, rng),
                "uniform_random_tokens": rng.integers(1, 11, (n_rows, 18, 10)).astype(np.int64),
                "all_sigma_k": np.repeat(labels[:, None, :10], 18, 1)}
        for name, dec in decs.items():
            a = zstats.analyze_decodes(dec, inputs, labels[:, :10], k, infos=infos, n_perm=n_perm)
            b = analyze_decodes_j(dec, inputs, labels[:, :10], k, J=17, J_all=18, infos=infos, n_perm=n_perm)
            ok = (_deep_equal(a["per_j"], b["per_j"]) and _deep_equal(a["cell"], b["cell"])
                  and all(_deep_equal(a["near"][kk], b["near"][kk]) for kk in
                          ("m_star", "ham_min", "tie", "none", "proper", "residues")))
            res.append({"id": f"G-a.5[k={k},{name}]", "pass": bool(ok), "Z": a["cell"]["Z"], "p_computed": a["cell"]["p"] is not None})
    return res


def ga_report() -> dict:
    """Run every G-a check; returns {'checks': [...], 'pass': bool}. Raises nothing on a failed comparison."""
    checks = []
    cfgs = ga_arch_configs()
    for cname, base in cfgs.items():
        for dtype in ("float32", "bfloat16"):
            for lgc in (0, 7):
                for inj in (True, False):
                    cfg = dict(base, loops=21, loop_grad_cycles=lgc, loop_input_injection=inj, forward_dtype=dtype)
                    r = _eq_trace_native(cfg)
                    checks.append({"id": f"G-a.1[{cname},{dtype},lgc={lgc},inj={inj}]",
                                   "pass": r["z_equal"] and r["logits_equal"], **r})
            # item 2: loops = 1 equals a single H_level call (and the native forward)
            cfg1 = dict(base, loops=1, loop_grad_cycles=0, forward_dtype=dtype)
            from analysis.ltf_decode.trace import initial_carry, native_forward, trace_loops
            inner = _synthetic_model(cfg1)
            batch = _synthetic_batch(cfg1)
            with torch.inference_mode():
                tr = trace_loops(inner, batch, 1, keep_latents=True)
                z0 = initial_carry(inner, GA_B).z_H
                emb = inner._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])
                ref = inner.H_level(z0, emb, cos_sin=inner.rotary_emb())
                ref_logits = inner.lm_head(ref)[:, inner.puzzle_emb_len:]
                z_nat, out_nat = native_forward(inner, batch)
            ok = (torch.equal(tr.z[0], ref) and torch.equal(tr.logits[0], ref_logits)
                  and torch.equal(tr.z[0], z_nat) and torch.equal(tr.logits[0], out_nat))
            checks.append({"id": f"G-a.2[{cname},{dtype},loops=1]", "pass": bool(ok)})
            # item 3: extrapolation, tracer t = 42 == same weights in a loops = 42 model
            for lgc in (0, 7):
                cfg21 = dict(base, loops=21, loop_grad_cycles=lgc, forward_dtype=dtype)
                cfg42 = dict(cfg21, loops=42)
                m21, m42 = _synthetic_model(cfg21), _synthetic_model(cfg42)
                m42.load_state_dict(m21.state_dict())     # parameter shapes are loop independent
                batch = _synthetic_batch(cfg21)
                with torch.inference_mode():
                    tr = trace_loops(m21, batch, 42, keep_latents=True)
                    z42, out42 = native_forward(m42, batch)
                    z21, out21 = native_forward(m21, batch)
                ok = (torch.equal(tr.z[41], z42) and torch.equal(tr.logits[41], out42)
                      and torch.equal(tr.z[20], z21) and torch.equal(tr.logits[20], out21))
                checks.append({"id": f"G-a.3[{cname},{dtype},lgc={lgc},t=42]", "pass": bool(ok)})
    # item 4: puzzle_emb_len == 1 and the answer slice shape for the EXP-015 config
    inner = _synthetic_model(dict(cfgs["full"], forward_dtype="float32"))
    from analysis.ltf_decode.trace import trace_loops
    with torch.inference_mode():
        tr = trace_loops(inner, _synthetic_batch(cfgs["full"]), 1)
    checks.append({"id": "G-a.4[puzzle_emb_len==1, answer slice (B,11,11)]",
                   "pass": bool(inner.puzzle_emb_len == 1 and tuple(tr.logits[0].shape) == (GA_B, 11, 11)),
                   "puzzle_emb_len": inner.puzzle_emb_len, "logits_shape": list(tr.logits[0].shape)})
    checks += ga_item5()
    return {"checks": checks, "n_checks": len(checks), "n_failed": sum(1 for c in checks if not c["pass"]),
            "pass": all(c["pass"] for c in checks)}


if __name__ == "__main__":
    import argparse
    import json
    import sys

    from analysis.ltf_decode.run_ltf_decode import require_cpu
    ap = argparse.ArgumentParser()
    ap.add_argument("--ga", action="store_true", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    require_cpu(args.device)
    rep = ga_report()
    rep["torch"] = torch.__version__
    rep["threads"] = torch.get_num_threads()
    with open(args.out, "w") as f:
        json.dump(rep, f, indent=1)
    print(f"G-a: {'PASS' if rep['pass'] else 'FAIL'} ({rep['n_checks']} checks, {rep['n_failed']} failed)")
    sys.exit(0 if rep["pass"] else 1)
