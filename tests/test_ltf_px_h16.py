"""ICML27 addendum (cx27) pre-launch tests for transformers_baseline_px under the halt-16 ACT settings.
CPU only (CUDA_VISIBLE_DEVICES=""), no checkpoint, no wandb, no data files.  Job bodies are NOT read from the
queue: they are produced by running `scripts/sigma_enqueue.sh --dry-run` (DRY_RUN_SHOW_BODY=1) against an empty
scratch QUEUE_DIR, so the tests exercise the bodies the stages really emit.

(b) px under the R1 g7 halt-16 body (prefix 1 and 16, heads 12 and 8): finite loss over two consecutive
    supervision segments, parameter counts, carry passed across segments
(c) px(12, 1, 0) is bitwise identical to the frozen base class under the R1 g7 halt-16 body (two segments,
    outputs, halting outputs, carry, all parameter gradients; train and eval)
(d) eval mode: trm_singlez (X5 body) == px(8, 16, 7) (HPS body) bitwise over all 16 segments, two batches
(d') train mode: same pair, q_continue row of q_head forced to 0 on BOTH models so the two halting rules
    (singlez q_halt > 0, looped q_halt > q_continue) coincide: loss, logits, halting decisions, carry, gradients
    bitwise equal over three segments and equal state after one production-optimizer step; companion test:
    without forcing, training-mode behaviour diverges (the halting rule is the one remaining difference)
(e) all seven factorial bodies and the X5 and X7 bodies compose with Hydra and instantiate
"""
from __future__ import annotations

import os
import shlex
import subprocess
import tempfile
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import models.recursive_reasoning.trm_singlez as SZ
import models.recursive_reasoning.transformers_baseline_px as PX
from models.losses import ACTLossHead
from models.recursive_reasoning.transformers_baseline import Model_ACTV2 as Base

REPO = Path(__file__).resolve().parents[1]
N_PARAMS = {8: 6_828_034, 12: 6_795_266}
B, SEQ, VOC, NPID = 8, 11, 11, 1   # D1 n=10, k=5 shape
# cell tag -> (heads, prefix, period, seeds), the addendum table
CELLS = {"px12p1s7": (12, 1, 7, (74, 75, 76)), "px8p1s0": (8, 1, 0, (77, 78, 79)),
         "px8p1s7": (8, 1, 7, (81, 82, 83)), "px12p16s0": (12, 16, 0, (84, 85, 86)),
         "px12p16s7": (12, 16, 7, (87, 88, 89)), "px8p16s0": (8, 16, 0, (91, 92, 93)),
         "px8p16s7": (8, 16, 7, (94, 95, 96))}


# ------------------------------------------------------------------ job bodies from the real stages
@pytest.fixture(scope="module")
def bodies():
    """name -> override list, from sigma_enqueue.sh --dry-run output (scratch queue; nothing is written)."""
    out = {}
    with tempfile.TemporaryDirectory() as q:
        for sub in ("jobs", "processing", "done", "failed"):
            os.makedirs(os.path.join(q, sub))
        for st in ("cc27_r1", "cx27_xf", "cx27_x5", "cx27_x7", "cc27_r4"):
            env = dict(os.environ, QUEUE_DIR=q, STAGES=st, DRY_RUN_SHOW_BODY="1")
            env.pop("RUN_PREFIX", None)
            p = subprocess.run(["bash", str(REPO / "scripts/sigma_enqueue.sh"), "--dry-run"], cwd=REPO, env=env,
                               capture_output=True, text=True, timeout=300)
            assert p.returncode == 0, p.stderr
            name, buf = None, []
            for line in p.stdout.split("\n") + ["0000 END"]:
                if line[:4].isdigit() and line[4:5] == " ":
                    if name is not None:
                        toks = shlex.split(" ".join(buf).replace("\\", " "))
                        assert toks[:3] == ["uv", "run", "pretrain.py"]
                        out[name] = toks[3:]
                    name, buf = line[5:], []
                elif line.startswith("    | "):
                    buf.append(line[6:])
    assert len([n for n in out if n.startswith("cx27_px")]) == 63
    return out


def _cfg(overrides):
    with initialize_config_dir(config_dir=str(REPO / "config"), version_base=None):
        return OmegaConf.to_container(compose(config_name="cfg_pretrain", overrides=overrides), resolve=True)


def _model_dict(a):
    d = {k: v for k, v in a.items() if k not in ("name", "loss")}
    d.update(batch_size=B, vocab_size=VOC, seq_len=SEQ, num_puzzle_identifiers=NPID, causal=False)
    return d


def px_from_g7(bodies, heads, prefix, period):
    """parent R1 g7 halt-16 body, arch swapped to px, three factors explicit (after the last arch override)."""
    ov = ["arch=transformers_baseline_px" if o == "arch=transformers_baseline" else o
          for o in bodies["cc27_g7_h16_k5_s21"]]
    ov += [f"arch.num_heads={heads}", f"arch.puzzle_emb_len={prefix}",
           f"arch.loop_input_injection_skip_period={period}"]
    return _model_dict(_cfg(ov)["arch"])


def build(cls, d, seed=1234):
    torch.manual_seed(seed)
    return ACTLossHead(cls(d), loss_type="stablemax_cross_entropy")


def keymap(k):
    return k.replace("model.inner.L_level.", "model.inner.H_level.").replace("model.inner.L_init", "model.inner.H_init")


def randomise(m, seed=99, spread_halt=False, force_qc0=False):
    """puzzle_emb weights and q_head are zero/-5 initialised: make them non-zero so those paths carry signal.
    spread_halt: scale the q_halt row and zero its bias so q_halt is spread around 0 (halting varies per sample).
    force_qc0: q_continue row of q_head (weight row 1 and bias[1]) set to exactly 0 -> q_continue logit == 0."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        pe = m.model.inner.puzzle_emb.weights
        pe.copy_(torch.randn(pe.shape, generator=g) * 0.5)
        qh = m.model.inner.q_head
        qh.weight.copy_(torch.randn(qh.weight.shape, generator=g) * 0.05)
        if spread_halt:
            qh.weight[0].mul_(40.0)
            qh.bias[0] = 0.0
        if force_qc0:
            qh.weight[1].zero_()
            qh.bias[1] = 0.0


def copy_weights(src, dst):
    inv = {keymap(k): k for k in dst.state_dict()}
    dst.load_state_dict({inv[keymap(k)]: v.clone() for k, v in src.state_dict().items()}, strict=True)


def make_batch(seed, b=B):
    g = torch.Generator().manual_seed(seed)
    labels = torch.randint(1, VOC, (b, SEQ), generator=g)
    labels[torch.rand(b, SEQ, generator=g) < 0.1] = -100
    return {"inputs": torch.randint(1, VOC, (b, SEQ), generator=g), "labels": labels,
            "puzzle_identifiers": torch.zeros(b, dtype=torch.int32)}


def carry_z(c):
    ic = c.inner_carry
    return ic.z_L if hasattr(ic, "z_L") else ic.z_H


def segments(m, batches, train, seed0=5000):
    """consecutive supervision segments with the carry passed along; grads accumulate across segments
    (snapshot after each segment).  torch.manual_seed(seed0+s) before each segment (exploration draws)."""
    m.train(train)
    for p in m.parameters():
        p.grad = None
    carry = m.initial_carry(batches[0])
    recs = []
    for s, batch in enumerate(batches):
        torch.manual_seed(seed0 + s)
        with torch.set_grad_enabled(train):
            carry, loss, metrics, outs, fin = m(return_keys=["logits", "q_halt_logits", "q_continue_logits"],
                                                carry=carry, batch=batch)
        if train:
            loss.backward()
        rec = {"loss": loss.detach().clone(), "halted": carry.halted.clone(), "steps": carry.steps.clone(),
               "z": carry_z(carry).detach().clone(), "finished": fin.clone()}
        rec.update({f"out:{k}": v.clone() for k, v in outs.items()})
        rec.update({f"met:{k}": v.detach().clone() for k, v in metrics.items()})
        rec["grads"] = ({keymap(n): p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None}
                        if train else {})
        for n, bf in m.named_buffers():
            if train and bf.requires_grad and bf.grad is not None:
                rec["grads"]["BUF:" + keymap(n)] = bf.grad.detach().clone()
        recs.append(rec)
    return recs


def cmp_recs(ra, rb):
    """-> (bitwise_equal, max_abs_diff, n_grad_tensors, n_nonzero_grad_tensors, per-segment halted of a, of b)."""
    eq, worst, ng, nz = True, 0.0, 0, 0
    for a, b in zip(ra, rb):
        keys = sorted((set(a) & set(b)) - {"grads"})
        assert {"loss", "out:logits", "out:q_halt_logits", "halted", "steps", "z"} <= set(keys)
        for k in keys:
            x, y = a[k].float(), b[k].float()
            if x.shape != y.shape:
                return False, float("inf"), 0, 0
            eq &= torch.equal(x, y)
            worst = max(worst, float((x - y).abs().max()))
        assert set(a["grads"]) == set(b["grads"]), sorted(set(a["grads"]) ^ set(b["grads"]))
        for k in sorted(a["grads"]):
            x, y = a["grads"][k].float(), b["grads"][k].float()
            eq &= torch.equal(x, y)
            worst = max(worst, float((x - y).abs().max()))
    for k, v in ra[-1]["grads"].items():
        ng += 1
        nz += int(v.float().abs().max() > 0)
    return eq, worst, ng, nz


def count(m):
    return sum(p.numel() for p in m.parameters())


# ------------------------------------------------------------------ (b) px under halt-16 ACT settings
@pytest.mark.parametrize("heads", [12, 8])
@pytest.mark.parametrize("prefix", [1, 16])
def test_b_px_halt16_two_segments_finite(bodies, heads, prefix):
    cfg = _cfg(["arch=transformers_baseline_px" if o == "arch=transformers_baseline" else o
                for o in bodies["cc27_g7_h16_k5_s21"]] +
               [f"arch.num_heads={heads}", f"arch.puzzle_emb_len={prefix}", "arch.loop_input_injection_skip_period=7"])
    a = cfg["arch"]
    assert (a["halt_max_steps"], a["act_enabled"], a["act_inference"], a["halt_exploration_prob"]) == (16, True, False, 0.1)
    assert a["loops"] == 21 and a["loop_grad_cycles"] == 7
    m = build(PX.Model_ACTV2, _model_dict(a))
    assert count(m) == N_PARAMS[heads]
    assert m.model.inner.puzzle_emb_len == prefix
    randomise(m, spread_halt=True)
    batches = [make_batch(11), make_batch(12)]
    carry0 = m.initial_carry(batches[0])
    assert carry0.inner_carry.z_H.shape == (B, SEQ + prefix, 512)
    recs = segments(m, batches, train=True)
    assert all(bool(torch.isfinite(r["loss"])) and bool(torch.isfinite(r["out:logits"].float()).all()) for r in recs)
    assert recs[0]["steps"].tolist() == [1] * B
    # carry passed: rows not halted after segment 0 continue their step count in segment 1; halted rows restart at 1
    h0 = recs[0]["halted"]
    exp1 = torch.where(h0, torch.ones_like(recs[1]["steps"]), recs[0]["steps"] + 1)
    assert torch.equal(recs[1]["steps"], exp1)
    assert recs[1]["z"].shape == (B, SEQ + prefix, 512)
    print(f"B heads={heads} prefix={prefix}: params={count(m)} loss_seg0={float(recs[0]['loss']):.4f} "
          f"loss_seg1={float(recs[1]['loss']):.4f} halted_seg0={h0.int().tolist()} steps_seg1={recs[1]['steps'].tolist()}")


# ------------------------------------------------------------------ (c) px(12,1,0) == frozen base at halt 16
@pytest.mark.parametrize("mode", ["train", "eval"])
def test_c_px_12_1_0_bitwise_equals_frozen_base_halt16(bodies, mode):
    d_base = _model_dict(_cfg(bodies["cc27_g7_h16_k5_s21"])["arch"])
    d_px = px_from_g7(bodies, 12, 1, 0)
    assert d_base["halt_max_steps"] == d_px["halt_max_steps"] == 16 and "puzzle_emb_len" not in d_base
    m_a, m_b = build(Base, d_base), build(PX.Model_ACTV2, d_px)
    assert count(m_a) == count(m_b) == N_PARAMS[12]
    assert all(torch.equal(v, m_b.state_dict()[k]) for k, v in m_a.state_dict().items())   # same-seed init
    for m in (m_a, m_b):
        randomise(m, spread_halt=True)
    batches = [make_batch(21), make_batch(22)]
    ra, rb = segments(m_a, batches, train=(mode == "train")), segments(m_b, batches, train=(mode == "train"))
    eq, worst, ng, nz = cmp_recs(ra, rb)
    print(f"C {mode}: bitwise_equal={eq} max_abs_diff={worst} n_grads={ng} n_nonzero={nz} "
          f"halted_a={[r['halted'].int().tolist() for r in ra]} halted_b={[r['halted'].int().tolist() for r in rb]}")
    assert eq and worst == 0.0
    if mode == "train":
        assert ng >= 12 and nz == ng
        assert any(0 < int(r["halted"].sum()) < B for r in ra)   # halting varies -> reset path exercised


# ------------------------------------------------------------------ (d) / (d') singlez X5 vs px HPS
@pytest.fixture(scope="module")
def pair(bodies):
    x5 = _cfg(bodies["cx27_noz_h16_k5_s71"])
    hps = _cfg(bodies["cx27_px8p16s7_h16_k5_s94"])
    d_sz, d_px = _model_dict(x5["arch"]), _model_dict(hps["arch"])
    assert d_sz["halt_max_steps"] == d_px["halt_max_steps"] == 16 and d_sz["halt_exploration_prob"] == d_px["halt_exploration_prob"] == 0.1
    assert d_sz["no_ACT_continue"] is True
    assert (d_px["num_heads"], d_px["puzzle_emb_len"], d_px["loop_input_injection_skip_period"]) == (8, 16, 7)
    return dict(d_sz=d_sz, d_px=d_px, x5=x5, hps=hps)


def _make_pair(pair, mode):
    """mode 'plain'  : random q_head weights, q_head biases untouched (production init, both -5)    [(d) eval]
    mode 'forced' : q_continue row of q_head (weight[1] and bias[1]) set to exactly 0 on the singlez model BEFORE the
                    copy (so both models carry it); bias[0] shifted by minus the median q_halt of a no-grad probe of
                    the first batch, so about half of the samples halt in the first segment
    mode 'neg'    : q_head weights x0.001, bias[0] = -3, bias[1] left at the production init -5, so
                    q_continue (~-5) < q_halt (~-3) < 0 for every sample (looped rule halts, singlez rule does not)"""
    m_sz, m_px = build(SZ.TinyRecursiveReasoningModel_ACTV1, pair["d_sz"]), build(PX.Model_ACTV2, pair["d_px"])
    assert count(m_sz) == count(m_px) == N_PARAMS[8]
    sd_px = {keymap(k2): v2 for k2, v2 in m_px.state_dict().items()}
    assert all(torch.equal(v.float(), sd_px[keymap(k)].float()) for k, v in m_sz.state_dict().items())   # same-seed init
    randomise(m_sz)
    qh = m_sz.model.inner.q_head
    info = {}
    with torch.no_grad():
        if mode == "forced":
            qh.weight[1].zero_()
            qh.bias[1] = 0.0
            m_sz.eval()
            b0 = make_batch(41)
            _, _, _, outs, _ = m_sz(return_keys=["q_halt_logits"], carry=m_sz.initial_carry(b0), batch=b0)
            med = float(outs["q_halt_logits"].float().median())
            qh.bias[0] = qh.bias[0] - med
            info["median_shift"] = med
        elif mode == "neg":
            qh.weight.mul_(0.001)
            qh.bias[0] = -3.0
    copy_weights(m_sz, m_px)
    for m in (m_sz, m_px):
        q = m.model.inner.q_head
        info["q_head_bias"] = [float(q.bias[0]), float(q.bias[1])]
        info["q_continue_row_abs_max"] = float(q.weight[1].abs().max())
        if mode == "forced":
            assert float(q.weight[1].abs().max()) == 0.0 and float(q.bias[1]) == 0.0
    return m_sz, m_px, info


def test_d_eval_singlez_x5_equals_px_hps_16_segments(pair):
    """Evaluation mode, 16 consecutive segments, two random batches; documented key mapping
    L_level <-> H_level, L_init <-> H_init.  No forcing needed: eval halts only at the last step in both classes."""
    m_sz, m_px, _ = _make_pair(pair, "plain")
    sd_px = set(m_px.state_dict())
    assert len(sd_px) == 14 and {keymap(k) for k in m_sz.state_dict()} == sd_px
    for bseed in (31, 32):
        batches = [make_batch(bseed)] * 16
        ra, rb = segments(m_sz, batches, train=False), segments(m_px, batches, train=False)
        eq, worst, _, _ = cmp_recs(ra, rb)
        halted = [int(r["halted"].sum()) for r in ra]
        print(f"D eval batch_seed={bseed}: segments={len(ra)} bitwise_equal={eq} max_abs_diff={worst} halted_count_per_segment={halted}")
        assert len(ra) == 16 and eq and worst == 0.0
        assert halted[:15] == [0] * 15 and halted[15] == B   # all rows halt only at segment 16


def _opt_step(m, cfg):
    from adam_atan2_pytorch import AdamAtan2 as AdamATan2
    from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
    opts = [CastedSparseEmbeddingSignSGD_Distributed(m.model.puzzle_emb.buffers(), lr=1.0e-6,
                                                     weight_decay=cfg["puzzle_emb_weight_decay"], world_size=1,
                                                     dense_update=cfg["puzzle_emb_dense_update"]),
            AdamATan2(m.parameters(), lr=1.0e-10, weight_decay=cfg["weight_decay"], betas=(cfg["beta1"], cfg["beta2"]))]
    for o, lr in zip(opts, (cfg["puzzle_emb_lr"], cfg["lr"])):   # create_model's order and base lrs (flat after warmup)
        for g in o.param_groups:
            g["lr"] = lr
        o.step()
        o.zero_grad()


def test_dprime_train_singlez_x5_equals_px_hps_with_qcontinue_forced(pair):
    """FORCED: q_head.weight[1] = 0 and q_head.bias[1] = 0 on both models (q_continue logit == 0), so
    looped q_halt > q_continue is singlez q_halt > 0.  q_halt row scaled x40, bias[0]=0 so halting varies."""
    cfg = pair["x5"]
    m_sz, m_px, info = _make_pair(pair, "forced")
    batches = [make_batch(41), make_batch(42), make_batch(43)]
    ra, rb = segments(m_sz, batches, train=True), segments(m_px, batches, train=True)
    eq, worst, ng, nz = cmp_recs(ra, rb)
    ha, hb = [r["halted"].int().tolist() for r in ra], [r["halted"].int().tolist() for r in rb]
    print(f"D' train forced q_continue==0: segments={len(ra)} bitwise_equal={eq} max_abs_diff={worst} "
          f"n_grads={ng} n_nonzero={nz} halted_singlez={ha} halted_px={hb} loss_singlez={[float(r['loss']) for r in ra]} forced_state={info}")
    assert eq and worst == 0.0 and ha == hb
    assert ng >= 12 and nz >= ng - 1    # q_continue-row gradient is structurally absent/zero; all other tensors non-zero
    assert any(0 < sum(h) < B for h in ha)   # halting actually varies (reset + carry path exercised)
    pre = {keymap(k): v.clone() for k, v in m_sz.state_dict().items()}
    _opt_step(m_sz, cfg)
    _opt_step(m_px, cfg)
    sa = {keymap(k): v for k, v in m_sz.state_dict().items()}
    sb = {keymap(k): v for k, v in m_px.state_dict().items()}
    assert set(sa) == set(sb)
    bad = [k for k in sa if not torch.equal(sa[k], sb[k])]
    moved = [k for k in sa if not torch.equal(sa[k], pre[k])]
    print(f"D' optimizer step changed {len(moved)}/{len(sa)} state tensors vs the pre-step state (same set on both models: "
          f"{sorted(moved) == sorted(k for k in sb if not torch.equal(sb[k], pre[k]))})")
    print(f"D' after one production optimizer step (puzzle_emb SignSGD + AdamATan2, lr {cfg['lr']}/{cfg['puzzle_emb_lr']}, "
          f"wd {cfg['weight_decay']}/{cfg['puzzle_emb_weight_decay']}): {len(sa)} state tensors compared, differing={bad}")
    assert not bad and len(moved) >= 10   # the step is not a no-op


def test_dprime_companion_unforced_training_diverges(pair):
    """Companion (documenting): WITHOUT forcing q_continue (q_continue bias kept at its production init -5; q_halt
    pinned to about -3 with tiny weights) training-mode behaviour differs: the looped rule halts when q_halt > q_continue,
    the singlez rule only when q_halt > 0.  Measured values are printed; nothing here is interpreted."""
    m_sz, m_px, info = _make_pair(pair, "neg")
    batches = [make_batch(41), make_batch(42), make_batch(43)]
    ra, rb = segments(m_sz, batches, train=True), segments(m_px, batches, train=True)
    eq, worst, _, _ = cmp_recs(ra, rb)
    ha, hb = [r["halted"].int().tolist() for r in ra], [r["halted"].int().tolist() for r in rb]
    loss_d = [float((x["loss"] - y["loss"]).abs()) for x, y in zip(ra, rb)]
    print(f"D' companion UNFORCED: bitwise_equal={eq} max_abs_diff={worst} halted_singlez={ha} halted_px={hb} "
          f"abs_loss_diff_per_segment={loss_d} q_head_state={info}")
    assert not eq and worst > 0.0
    assert ha != hb    # the halting decisions themselves differ


# ------------------------------------------------------------------ (e) all bodies compose and instantiate
@pytest.mark.parametrize("tag", sorted(CELLS))
def test_e_factorial_bodies_compose_and_instantiate(bodies, tag):
    heads, prefix, period, seeds = CELLS[tag]
    for s in seeds[:1]:
        name = f"cx27_{tag}_h16_k5_s{s}"
        cfg = _cfg(bodies[name])
        a = cfg["arch"]
        assert a["name"] == "recursive_reasoning.transformers_baseline_px@Model_ACTV2"
        assert (a["num_heads"], a["puzzle_emb_len"], a["loop_input_injection_skip_period"]) == (heads, prefix, period)
        assert (a["halt_max_steps"], a["act_enabled"], a["act_inference"], a["loops"], a["loop_grad_cycles"]) == (16, True, False, 21, 7)
        assert cfg["seed"] == s and cfg["data_paths"] == ["data/power_permutation/d1/5"] and cfg["run_group"] == "cx27"
        m = build(PX.Model_ACTV2, _model_dict(a))
        assert count(m) == N_PARAMS[heads] and m.model.inner.puzzle_emb_len == prefix
        print(f"E {name}: params={count(m)} heads={heads} prefix={prefix} period={period}")


def test_e_x5_and_x7_bodies_compose_and_instantiate(bodies):
    x5, x7 = _cfg(bodies["cx27_noz_h16_k5_s71"]), _cfg(bodies["cx27_nozlr_h1_k5_s97"])
    for cfg, halt in ((x5, 16), (x7, 1)):
        assert cfg["arch"]["name"] == "recursive_reasoning.trm_singlez@TinyRecursiveReasoningModel_ACTV1"
        assert cfg["arch"]["halt_max_steps"] == halt and cfg["run_group"] == "cx27"
        m = build(SZ.TinyRecursiveReasoningModel_ACTV1, _model_dict(cfg["arch"]))
        assert count(m) == N_PARAMS[8]
    assert (x5["lr"], x5["puzzle_emb_lr"]) == (1e-4, 1e-4)
    assert (x7["lr"], x7["puzzle_emb_lr"], x7["weight_decay"], x7["puzzle_emb_weight_decay"]) == (3e-5, 3e-5, 1.0, 1.0)
    print(f"E X5 lr={x5['lr']} X7 lr={x7['lr']} puzzle_emb_lr={x7['puzzle_emb_lr']}")
