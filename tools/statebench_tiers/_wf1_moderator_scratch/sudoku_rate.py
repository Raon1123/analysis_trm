import json, sys, glob, os, re
import yaml
rows=[]
for f in sorted(glob.glob('wandb/run-*')):
    c=os.path.join(f,'files','config.yaml'); s=os.path.join(f,'files','wandb-summary.json')
    if not (os.path.exists(c) and os.path.exists(s)): continue
    try:
        txt=open(c).read()
    except Exception: continue
    if 'Sudoku-extreme' not in txt and 'sudoku' not in txt.lower(): continue
    try:
        cfg=yaml.safe_load(txt); sm=json.load(open(s))
    except Exception: continue
    g=lambda k: (cfg.get(k) or {}).get('value') if isinstance(cfg.get(k),dict) else None
    arch=g('arch') or {}
    rt=sm.get('_runtime'); st=sm.get('train/count') ; step=sm.get('_step')
    rows.append((os.path.basename(f), g('run_name'), arch.get('halt_max_steps'), arch.get('H_cycles'), arch.get('L_cycles'), arch.get('L_layers'), g('global_batch_size'), step, rt, (step/rt if rt and step else None)))
for r in rows: print(r)
