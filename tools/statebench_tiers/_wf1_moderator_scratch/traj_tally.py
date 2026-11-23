"""Moderator CPU check (not frozen, not evidence for EXP-027): trajectory-based outcomes
for the sigma^k fig1 iter cohort (trm/trm_singlez, (3,6), h=1). Reads local wandb .wandb files read-only."""
import json, glob, os, re, sys, yaml
from multiprocessing import Pool
sys.path.insert(0, 'tools/figures')
from wandb_ro import iter_history

def find_runs():
    out=[]
    for f in sorted(glob.glob('wandb/run-*')):
        c=os.path.join(f,'files','config.yaml')
        if not os.path.exists(c): continue
        try: cfg=yaml.safe_load(open(c))
        except Exception: continue
        g=lambda k: (cfg.get(k) or {}).get('value') if isinstance(cfg.get(k),dict) else None
        rn=g('run_name') or ''
        m=re.fullmatch(r'(?:H108N\d_)?fig1_tf_(z|noz)_iter_k(\d+)_s(\d+)', rn)
        if not m: continue
        if g('data_paths')!=[f"data/sigma_k_10/{m.group(2)}"]: continue
        if g('project_name') not in ('Sigma_k_new','power_permutation'): continue
        wf=glob.glob(os.path.join(f,'run-*.wandb'))
        if not wf: continue
        out.append((m.group(1),int(m.group(2)),int(m.group(3)),os.path.basename(f)[-8:],wf[0]))
    return out

def one(r):
    v,k,s,rid,path=r
    pts=[]
    try:
        for step,vals in iter_history(path,['all/exact_accuracy']):
            if 'all/exact_accuracy' in vals: pts.append((step,vals['all/exact_accuracy']))
    except Exception as e:
        return (v,k,s,rid,'ERR',repr(e)[:80])
    pts.sort()
    ex=[e for _,e in pts]
    fp=next((st for st,e in pts if e>=0.9),None)
    n90=sum(e>=0.9 for e in ex)
    return dict(v=v,k=k,s=s,id=rid,n=len(pts),final=ex[-1] if ex else None,mx=max(ex) if ex else None,
                first90=fp,n90=n90,last3=[round(e,3) for e in ex[-3:]],
                frac90_after=(sum(e>=0.9 for st,e in pts if fp is not None and st>=fp)/max(1,sum(1 for st,e in pts if fp is not None and st>=fp))) if fp else None)

if __name__=='__main__':
    runs=find_runs()
    with Pool(16) as p:
        res=p.map(one,runs)
    json.dump(res,open(sys.argv[1],'w'),indent=1)
    for r in sorted(res,key=lambda d:(d['v'],d['k'],d['s']) if isinstance(d,dict) else ('',0,0)):
        print(r)
