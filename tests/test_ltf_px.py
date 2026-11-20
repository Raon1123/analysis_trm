"""models/recursive_reasoning/transformers_baseline_px.py (R6 'comb' arm): prefix-length support
on top of the injection-skip subclass.  CPU only (run with CUDA_VISIBLE_DEVICES=""), no checkpoint,
no wandb, no data files except a tiny synthetic probe written to tmp_path.

(a) default prefix == frozen base / skip subclass, bitwise (outputs + every parameter gradient)
(b) IDENTITY: trm_singlez (halt-1 D1 reference body) == px(heads 8, prefix 16, period 7) on the g7 body
(c) negative controls: reverting any one of the three factors breaks equality
(d) eight factor combinations instantiate, run forward+backward, and report parameter counts
(e) sha256 pins of the frozen base and the skip module
(f) evaluation-side z logging path (utils/z_logging._probe_forward) on the new class
"""
from __future__ import annotations

import hashlib
import shlex
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import models.recursive_reasoning.trm_singlez as SZ
import models.recursive_reasoning.transformers_baseline_px as PX
import models.recursive_reasoning.transformers_baseline_skip as SK
from models.losses import ACTLossHead
from models.recursive_reasoning.transformers_baseline import Model_ACTV2 as Base

REPO = Path(__file__).resolve().parents[1]
BASE_SHA = "a8a4c5da76b86a7c3bf595da3c7980e0aca45a5d786a041a3f2a5556b8e23ba6"
SKIP_SHA = "408a2437f9fa3509f18c3470b89c6b43ac26eca583db0348a8a602b219f20d81"
SZ_JOB = REPO / "scripts/queue/done/0579_e016_fig1_tf_noz_iter_k5_s2.job"   # e016_fig1_tf_noz_iter_k5_s2
G7_JOB = REPO / "scripts/queue/done/0542_pp_ltf_g7_lay2_k5_s1.job"           # pp_ltf_g7_lay2_k5_s1
N_PARAMS = {8: 6_828_034, 12: 6_795_266}
B, SEQ, VOC, NPID = 8, 11, 11, 1   # D1 n=10, k=5 shape


# ------------------------------------------------------------------ helpers
def _job_overrides(path: Path):
    toks = shlex.split(path.read_text().replace("\\\n", " "))
    assert toks[:3] == ["uv", "run", "pretrain.py"], toks[:3]
    return toks[3:]


def _arch(overrides):
    with initialize_config_dir(config_dir=str(REPO / "config"), version_base=None):
        cfg = compose(config_name="cfg_pretrain", overrides=overrides)
    return OmegaConf.to_container(cfg.arch, resolve=True)


def _model_dict(a):
    d = {k: v for k, v in a.items() if k not in ("name", "loss")}
    d.update(batch_size=B, vocab_size=VOC, seq_len=SEQ, num_puzzle_identifiers=NPID, causal=False)
    return d


def px_dict(heads=8, prefix=16, period=7):
    """g7 reference body with arch=transformers_baseline_px and the three factors set explicitly."""
    ov = [o.replace("arch=transformers_baseline", "arch=transformers_baseline_px", 1)
          if o == "arch=transformers_baseline" else o for o in _job_overrides(G7_JOB)]
    ov += [f"arch.num_heads={heads}", f"arch.puzzle_emb_len={prefix}",
           f"arch.loop_input_injection_skip_period={period}"]
    return _model_dict(_arch(ov))


def sz_dict():
    return _model_dict(_arch(_job_overrides(SZ_JOB)))


def build(cls, d, seed=1234):
    torch.manual_seed(seed)
    return ACTLossHead(cls(d), loss_type="stablemax_cross_entropy")


def keymap(k):
    return k.replace("model.inner.L_level.", "model.inner.H_level.").replace("model.inner.L_init", "model.inner.H_init")


def randomise(m, seed=99):
    """puzzle_emb weights and q_head are zero-initialised; make them non-zero so those paths carry signal."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        pe = m.model.inner.puzzle_emb.weights
        pe.copy_(torch.randn(pe.shape, generator=g) * 0.5)
        qh = m.model.inner.q_head
        qh.weight.copy_(torch.randn(qh.weight.shape, generator=g) * 0.05)


def copy_weights(src, dst):
    inv = {keymap(k): k for k in dst.state_dict()}
    dst.load_state_dict({inv[keymap(k)]: v.clone() for k, v in src.state_dict().items()}, strict=True)


def make_batch(seed, b=B, seq=SEQ, voc=VOC):
    g = torch.Generator().manual_seed(seed)
    return {"inputs": torch.randint(1, voc, (b, seq), generator=g),
            "labels": torch.randint(1, voc, (b, seq), generator=g),
            "puzzle_identifiers": torch.zeros(b, dtype=torch.int32)}


def run(m, mode, batch):
    m.train(mode == "train")
    for p in m.parameters():
        p.grad = None
    for _, bf in m.named_buffers():
        if bf.requires_grad:
            bf.grad = None
    carry = m.initial_carry(batch)
    carry, loss, metrics, outs, fin = m(return_keys=["logits", "q_halt_logits", "q_continue_logits"],
                                        carry=carry, batch=batch)
    loss.backward()
    grads = {keymap(n): p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None}
    for n, bf in m.named_buffers():
        if bf.requires_grad and bf.grad is not None:
            grads["BUF:" + keymap(n)] = bf.grad.detach().clone()
    res = {"loss": loss.detach(), "halted": carry.halted.clone(), "steps": carry.steps.clone(),
           "finished": fin.clone()}
    res.update({f"out:{k}": v for k, v in outs.items()})
    res.update({f"met:{k}": v.detach() for k, v in metrics.items()})
    return res, grads


def diff(ra, ga, rb, gb):
    """-> (bitwise_equal, max_abs_diff, n_grads, n_nonzero_grads); compares every shared output/grad."""
    eq, worst = True, 0.0
    keys = sorted(set(ra) & set(rb))
    assert {"loss", "out:logits", "out:q_halt_logits"} <= set(keys)
    for k in keys:
        a, b = ra[k].float(), rb[k].float()
        if a.shape != b.shape:
            return False, float("inf"), 0, 0
        eq &= torch.equal(a, b)
        worst = max(worst, float((a - b).abs().max()))
    assert set(ga) == set(gb), (sorted(set(ga) ^ set(gb)))
    nz = 0
    for k in sorted(ga):
        a, b = ga[k].float(), gb[k].float()
        eq &= torch.equal(a, b)
        worst = max(worst, float((a - b).abs().max()))
        nz += int(a.abs().max() > 0)
    return eq, worst, len(ga), nz


def count(m):
    return sum(p.numel() for p in m.parameters())


# ------------------------------------------------------------------ (a) default prefix == frozen classes
def _tiny(**over):
    cfg = dict(batch_size=2, seq_len=11, puzzle_emb_ndim=32, num_puzzle_identifiers=1, vocab_size=12,
               H_cycles=1, H_layers=1, hidden_size=32, expansion=4, num_heads=2, pos_encodings="rope",
               halt_max_steps=1, halt_exploration_prob=0.1, forward_dtype="float32", loops=14,
               loop_input_injection=True)
    cfg.update(over)
    return cfg


def _tiny_run(cls, mode, cfg, seed=0):
    torch.manual_seed(seed)
    m = ACTLossHead(cls(cfg), loss_type="stablemax_cross_entropy")
    randomise(m)
    return run(m, mode, make_batch(5, b=2, seq=11, voc=12))


@pytest.mark.parametrize("mode", ["train", "eval"])
@pytest.mark.parametrize("grad_cycles", [0, 7])
def test_default_prefix_period0_bitwise_equals_frozen_base(mode, grad_cycles):
    cfg = _tiny(loop_grad_cycles=grad_cycles)
    ra, ga = _tiny_run(Base, mode, cfg)
    rb, gb = _tiny_run(PX.Model_ACTV2, mode, dict(cfg, loop_input_injection_skip_period=0))
    eq, worst, ng, nz = diff(ra, ga, rb, gb)
    assert eq and worst == 0.0 and nz > 5, (worst, ng, nz)


@pytest.mark.parametrize("mode", ["train", "eval"])
@pytest.mark.parametrize("grad_cycles", [0, 7])
def test_default_prefix_period7_bitwise_equals_skip(mode, grad_cycles):
    cfg = _tiny(loop_grad_cycles=grad_cycles, loop_input_injection_skip_period=7)
    ra, ga = _tiny_run(SK.Model_ACTV2, mode, cfg)
    rb, gb = _tiny_run(PX.Model_ACTV2, mode, dict(cfg, puzzle_emb_len=1))
    eq, worst, ng, nz = diff(ra, ga, rb, gb)
    assert eq and worst == 0.0 and nz > 5, (worst, ng, nz)


def test_tiny_prefix_changes_geometry_and_runs():
    for prefix in (1, 3, 16):
        m = build(PX.Model_ACTV2, _tiny(puzzle_emb_len=prefix))
        assert m.model.inner.puzzle_emb_len == prefix
        batch = make_batch(5, b=2, seq=11, voc=12)
        carry = m.initial_carry(batch)
        assert carry.inner_carry.z_H.shape == (2, 11 + prefix, 32)
        randomise(m)
        res, _ = run(m, "train", batch)
        assert res["out:logits"].shape == (2, 11, 12)   # prefix dropped from the output slice


@pytest.mark.parametrize("over,msg", [
    (dict(puzzle_emb_len=0), "must be >= 1"),
    (dict(puzzle_emb_len=-3), "must be >= 1"),
    (dict(puzzle_emb_ndim=0, puzzle_emb_len=16), "needs puzzle_emb_ndim > 0"),
    (dict(puzzle_emb_ndim=96, puzzle_emb_len=2), "smaller than ceil"),
    (dict(pos_encodings="learned", puzzle_emb_len=16), "requires pos_encodings='rope'"),
])
def test_validation_errors(over, msg):
    with pytest.raises(ValueError, match=msg):
        PX.Model_ACTV2(_tiny(**over))


def test_validation_accepts_defaults():
    PX.Model_ACTV2(_tiny())
    PX.Model_ACTV2(_tiny(puzzle_emb_ndim=0))            # no prefix at all, default 1 ignored
    PX.Model_ACTV2(_tiny(pos_encodings="learned"))      # derived prefix with learned pos table


# ------------------------------------------------------------------ (b) identity with the real configs
@pytest.fixture(scope="module")
def ident():
    d_sz, d_px = sz_dict(), px_dict(8, 16, 7)
    assert d_sz["num_heads"] == 8 and d_sz["puzzle_emb_len"] == 16 and d_sz["halt_max_steps"] == 1
    assert d_px["halt_max_steps"] == 1 and d_px["loops"] == 21 and d_px["loop_grad_cycles"] == 7
    m_sz = build(SZ.TinyRecursiveReasoningModel_ACTV1, d_sz)
    m_px = build(PX.Model_ACTV2, d_px)
    init_equal = {}
    sd_sz = {keymap(k): v for k, v in m_sz.state_dict().items()}
    sd_px = {keymap(k): v for k, v in m_px.state_dict().items()}
    init_equal = {k: torch.equal(sd_sz[k].float(), sd_px[k].float()) for k in sd_sz if k in sd_px}
    out = dict(d_sz=d_sz, d_px=d_px, m_sz=m_sz, m_px=m_px, sd_sz=sd_sz, sd_px=sd_px, init_equal=init_equal)
    randomise(m_sz)
    copy_weights(m_sz, m_px)
    out["results"] = {}
    for mode in ("train", "eval"):
        for seed in (7, 8):
            batch = make_batch(seed)
            out["results"][(mode, seed)] = diff(*run(m_sz, mode, batch), *run(m_px, mode, batch))
    return out


def test_identity_param_count(ident):
    assert count(ident["m_sz"]) == count(ident["m_px"]) == N_PARAMS[8]


def test_identity_state_dict_key_mapping(ident):
    """Documented mapping: singlez 'model.inner.L_level.*' -> px 'model.inner.H_level.*',
    singlez 'model.inner.L_init' -> px 'model.inner.H_init'; all other keys identical."""
    raw_sz, raw_px = set(ident["m_sz"].state_dict()), set(ident["m_px"].state_dict())
    assert len(raw_sz) == len(raw_px) == 14
    assert {keymap(k) for k in raw_sz} == raw_px
    assert {k for k in raw_sz if keymap(k) != k} == {k for k in raw_sz if ".L_level." in k or k.endswith(".L_init")}
    assert {k: tuple(v.shape) for k, v in ident["sd_sz"].items()} == {k: tuple(v.shape) for k, v in ident["sd_px"].items()}


def test_identity_same_seed_initial_parameters_equal(ident):
    assert ident["init_equal"] and all(ident["init_equal"].values()), ident["init_equal"]


@pytest.mark.parametrize("mode", ["train", "eval"])
@pytest.mark.parametrize("seed", [7, 8])
def test_identity_bitwise_equal_outputs_and_gradients(ident, mode, seed):
    eq, worst, ng, nz = ident["results"][(mode, seed)]
    print(f"IDENTITY {mode} batch_seed={seed}: bitwise_equal={eq} max_abs_diff={worst} n_grads={ng} n_nonzero={nz}")
    assert ng >= 12 and nz == ng, (ng, nz)   # every compared gradient is non-zero (signal on all paths)
    assert eq and worst == 0.0


# ------------------------------------------------------------------ (c) negative controls
@pytest.mark.parametrize("factor,kw", [("prefix1", dict(prefix=1)), ("period0", dict(period=0))])
def test_negative_control_weights_copied(ident, factor, kw):
    m_ctl = build(PX.Model_ACTV2, px_dict(**{"heads": 8, "prefix": 16, "period": 7, **kw}))
    copy_weights(ident["m_sz"], m_ctl)
    eq, worst, _, _ = diff(*run(ident["m_sz"], "eval", make_batch(7)), *run(m_ctl, "eval", make_batch(7)))
    print(f"NEGCTL {factor}: equal={eq} max_abs_diff={worst}")
    assert not eq and worst > 0.0


def test_negative_control_heads12(ident):
    """heads 12 changes qkv/o projection shapes (head_dim 42), so weights cannot be copied; equality is
    broken at parameter shape/count level, and the same-seed outputs differ as well."""
    m_ctl = build(PX.Model_ACTV2, px_dict(heads=12))
    assert count(m_ctl) == N_PARAMS[12] != count(ident["m_sz"])
    sd = {keymap(k): tuple(v.shape) for k, v in m_ctl.state_dict().items()}
    assert sd != {k: tuple(v.shape) for k, v in ident["sd_sz"].items()}
    randomise(m_ctl)
    eq, worst, _, _ = diff(*run(ident["m_sz"], "eval", make_batch(7))[:1], {}, *run(m_ctl, "eval", make_batch(7))[:1], {})
    print(f"NEGCTL heads12: equal={eq} max_abs_diff={worst}")
    assert not eq


# ------------------------------------------------------------------ (d) the eight factor combinations
COMBOS = [(h, p, s) for h in (12, 8) for p in (1, 16) for s in (0, 7)]


@pytest.mark.parametrize("heads,prefix,period", COMBOS)
def test_eight_combinations_instantiate_and_backprop(heads, prefix, period):
    m = build(PX.Model_ACTV2, px_dict(heads, prefix, period))
    inner = m.model.inner
    assert inner.puzzle_emb_len == prefix
    assert inner.config.num_heads == heads and inner.config.loop_input_injection_skip_period == period
    n = count(m)
    print(f"COMBO heads={heads} prefix={prefix} period={period} params={n} attn_seq_len={SEQ + prefix}")
    assert n == N_PARAMS[heads]
    randomise(m)
    batch = make_batch(7)
    assert m.initial_carry(batch).inner_carry.z_H.shape == (B, SEQ + prefix, 512)
    res, grads = run(m, "train", batch)
    assert torch.isfinite(res["loss"]) and torch.isfinite(res["out:logits"].float()).all()
    assert res["out:logits"].shape == (B, SEQ, VOC)
    g = grads["model.inner.embed_tokens.embedding_weight"]
    assert torch.isfinite(g.float()).all() and float(g.float().abs().max()) > 0.0


# ------------------------------------------------------------------ (e) frozen-file pins
def _sha(rel):
    return hashlib.sha256((REPO / rel).read_bytes()).hexdigest()


def test_frozen_base_and_skip_unchanged():
    assert _sha("models/recursive_reasoning/transformers_baseline.py") == BASE_SHA
    assert _sha("models/recursive_reasoning/transformers_baseline_skip.py") == SKIP_SHA


# ------------------------------------------------------------------ (f) evaluation-side z logging path
def _write_probe_dir(root: Path, n=8, seq=11, voc=12):
    rng = np.random.default_rng(0)
    for split in ("train", "test"):
        d = root / split
        d.mkdir(parents=True)
        np.save(d / "all__inputs.npy", rng.integers(1, voc, (n, seq)).astype(np.int32))
        np.save(d / "all__labels.npy", rng.integers(1, voc, (n, seq)).astype(np.int32))
        np.save(d / "all__puzzle_identifiers.npy", np.zeros(n, dtype=np.int32))


@pytest.mark.parametrize("prefix", [1, 16])
def test_z_logging_probe_runs_on_px_cpu(tmp_path, prefix):
    from utils import z_logging as ZL
    _write_probe_dir(tmp_path)
    logger = ZL.ZDynamicsLogger(data_path=str(tmp_path), probe_size=8, ignore_label_id_in_file=None,
                                phase_threshold=0.999, phase_patience=2, checkpoint_path=None, seq_metrics=True)
    m = build(PX.Model_ACTV2, _tiny(puzzle_emb_len=prefix, batch_size=8, loops=14, loop_grad_cycles=7,
                                    loop_input_injection_skip_period=7))
    m.eval()
    res = ZL._probe_forward(m, logger._train_probe, device="cpu", compute_extra=True)
    assert 0.0 <= res["exact_acc"] <= 1.0
    # Finding (frozen guard reads z_H and z_L together): the looped carry has no z_L -> no z/* keys.
    assert res["z_H"] is None and res["z_L"] is None and res["step_deltas"] == []
    extra = res["extra_metrics"]
    print(f"ZLOG prefix={prefix}: exact_acc={res['exact_acc']} z_H_is_None={res['z_H'] is None} "
          f"extra_status_ok={extra.get('zseq/status_ok')} latent_is_z_h={extra.get('zseq/latent_is_z_h')} "
          f"n_extra_keys={len(extra)}")


@pytest.mark.parametrize("prefix", [1, 16])
def test_z_logger_log_runs_on_px_cpu(tmp_path, monkeypatch, prefix):
    """Drive ZDynamicsLogger.log (the call pretrain.py makes) on CPU: _probe_forward is re-bound to
    device='cpu' (log() itself hard-wires the default 'cuda'), wandb.log is captured, snapshot saved."""
    import functools
    import wandb
    from utils import z_logging as ZL
    data = tmp_path / "data"
    data.mkdir()
    _write_probe_dir(data)
    snap = tmp_path / "ckpt"
    logger = ZL.ZDynamicsLogger(data_path=str(data), probe_size=8, ignore_label_id_in_file=None,
                                phase_threshold=0.999, phase_patience=2, checkpoint_path=str(snap), seq_metrics=True)
    monkeypatch.setattr(ZL, "_probe_forward", functools.partial(ZL._probe_forward, device="cpu"))
    logged = []
    monkeypatch.setattr(wandb, "log", lambda d, step=None, **kw: logged.append((step, d)))
    m = build(PX.Model_ACTV2, _tiny(puzzle_emb_len=prefix, batch_size=8, loops=14, loop_grad_cycles=7,
                                    loop_input_injection_skip_period=7))
    m.eval()
    logger.log(model=m, step=1, save_train_state_fn=lambda ts: None, train_state=None)
    assert len(logged) == 1
    keys = sorted(logged[0][1])
    print(f"ZLOG.log prefix={prefix}: wandb keys={keys} snapshot_written={(snap / 'z_snapshots' / 'step_1.pt').exists()}")
    assert "probe/train_exact" in keys and "probe/test_exact" in keys and "phase/index" in keys
    assert not any(k.startswith("z/") for k in keys)   # frozen guard: no z_L in the looped carry -> no z/* keys


def test_z_dynamics_arch_dispatch_misclassifies_px():
    """analysis/z_dynamics/instrumented_forward.py is NOT imported by utils/z_logging.py or pretrain.py; it is an
    offline-analysis tool.  It classifies by hasattr(inner,'H_init') -> ARCH_TRM, which the looped class satisfies."""
    from analysis.z_dynamics import instrumented_forward as IF
    m = build(PX.Model_ACTV2, _tiny(puzzle_emb_len=16))
    inner = m.model.inner
    assert IF.arch_of_inner(inner) == IF.ARCH_TRM
    assert not hasattr(inner, "z_L") and not hasattr(inner, "L_level")
    batch = {k: v for k, v in make_batch(5, b=2, seq=11, voc=12).items()}
    with pytest.raises(Exception) as ei:
        with torch.inference_mode():
            inner.eval()
            IF.forward_trace(inner, batch)
    print(f"FORWARD_TRACE on px inner raises {type(ei.value).__name__}: {ei.value}")


def test_arch_name_resolves_to_new_class_via_pretrain_loader():
    from utils.functions import load_model_class
    a = _arch(_job_overrides(G7_JOB) and [o.replace("arch=transformers_baseline", "arch=transformers_baseline_px", 1)
                                          if o == "arch=transformers_baseline" else o for o in _job_overrides(G7_JOB)])
    assert a["name"] == "recursive_reasoning.transformers_baseline_px@Model_ACTV2"
    assert load_model_class(a["name"]) is PX.Model_ACTV2
    assert a["puzzle_emb_len"] == 1 and a["loop_input_injection_skip_period"] == 0 and a["num_heads"] == 12
