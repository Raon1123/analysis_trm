import sys, yaml, torch
sys.path.insert(0, ".")
from utils.functions import load_model_class
assert not torch.cuda.is_available()
for arch in ("trm", "trm_singlez"):
    a = yaml.safe_load(open(f"config/arch/{arch}.yaml"))
    name = a.pop("name"); a.pop("loss")
    a.update(H_cycles=3, L_cycles=6, L_layers=2, halt_max_steps=1, mlp_t=False, forward_dtype="float32", puzzle_emb_ndim=512)
    for vocab, seq in ((11, 11), (261, 34), (261, 50), (261, 82)):
        cfg = dict(a, batch_size=8, vocab_size=vocab, seq_len=seq, num_puzzle_identifiers=1, causal=False)
        cls = load_model_class(name)
        m = cls(cfg)
        n = sum(p.numel() for p in m.parameters())
        print(arch, "vocab", vocab, "seq_len", seq, "params", n)
