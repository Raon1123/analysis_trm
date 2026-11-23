import json, glob, os, yaml, sys
pat = sys.argv[1:]  # substrings of run_name
out=[]
for f in sorted(glob.glob('wandb/run-*')):
    c=os.path.join(f,'files','config.yaml'); s=os.path.join(f,'files','wandb-summary.json'); m=os.path.join(f,'files','wandb-metadata.json')
    if not (os.path.exists(c) and os.path.exists(s)): continue
    try:
        cfg=yaml.safe_load(open(c)); sm=json.load(open(s))
    except Exception: continue
    g=lambda k: (cfg.get(k) or {}).get('value') if isinstance(cfg.get(k),dict) else None
    rn=g('run_name') or ''
    if not any(p in rn for p in pat): continue
    arch=g('arch') or {}
    gpu=None
    if os.path.exists(m):
        try:
            md=json.load(open(m)); gpu=(md.get('gpu'), md.get('gpu_count'), md.get('host'))
        except Exception: pass
    rt=sm.get('_runtime'); step=sm.get('_step')
    dp=g('data_paths')
    out.append((rn, os.path.basename(f)[-8:], arch.get('halt_max_steps'), arch.get('H_cycles'), arch.get('L_cycles'), arch.get('mlp_t'), g('global_batch_size'), dp, g('log_z_dynamics'), step, round(rt or 0), round(step/rt,3) if rt and step else None, gpu))
for r in out: print(r)
