# -*- coding: utf-8 -*-
"""Data-transmission-drop robustness and concurrent-load analysis for the remaining methods.
Drop: CLMDTFN, HMoE, Seq2Seq, DUNILM, FFSTT, AttentionNILM (full-sequence/multi-output) + Seq2Point (center point)
Concurrent: same methods (Seq2Seq already covered, skipped)
"""
import sys, os, json, warnings
warnings.filterwarnings('ignore')
CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE)
import numpy as np, torch
from baseline_experiments_v3 import (Seq2PointSeq2Seq, Seq2Seq, DUNILM, FFSTT, AttentionNILM,
                                     CLMDTFN, HMoE, NILMDataset, SEQ_LENGTH, APPLIANCE_NAMES)

DATA = os.environ.get('AMPDS2_PATH', os.path.join(CODE, 'data', 'ampds2_aligned.csv'))
DEV=torch.device('cuda'); TRAIN=VAL=40320; ON_TH=10.0
MODEL_DIR = os.path.join(CODE, 'trained_models')
dst = NILMDataset(DATA, seq_length=SEQ_LENGTH, split='test', train_size=TRAIN, val_size=VAL, appliance_columns=APPLIANCE_NAMES)
sel = np.arange(0, len(dst), SEQ_LENGTH)
sel60 = np.arange(0, len(dst), 60)

def iter_preds(ds, model, sel, drop_rate=0.0, seed=42, batch=64):
    rng = np.random.RandomState(seed)
    with torch.no_grad():
        for i in range(0, len(sel), batch):
            idxs = sel[i:i+batch]
            ins, raws, tgts, poss, days = [], [], [], [], []
            for j in idxs:
                a,b,c,d,e = ds[int(j)]; ins.append(a); raws.append(b); tgts.append(c); poss.append(d); days.append(e)
            ti=torch.stack(ins); tr=torch.stack(raws)
            if drop_rate>0:
                mask = rng.rand(ti.shape[0], SEQ_LENGTH, 1) > drop_rate
                ti = ti * torch.FloatTensor(mask)
            out,_,_,_ = model(ti.to(DEV), tr.to(DEV), torch.stack(poss).squeeze(-1).to(DEV))
            yield out.float().cpu().numpy(), torch.stack(tgts).numpy()

def load_multi(cls, name):
    m = cls(input_dim=1, n_appliances=8).to(DEV)
    st = torch.load(os.path.join(MODEL_DIR, f'ampds2_4w_{name}.pt'), map_location=DEV, weights_only=False)
    sd = st['model_state_dict'] if isinstance(st,dict) and 'model_state_dict' in st else st
    m.load_state_dict(sd, strict=False); m.eval(); return m

def load_perapp(cls, name, k):
    m = cls(input_dim=1, n_appliances=1).to(DEV)
    st = torch.load(os.path.join(MODEL_DIR, f'ampds2_4w_{name}_app{k}.pt'), map_location=DEV, weights_only=False)
    sd = st['model_state_dict'] if isinstance(st,dict) and 'model_state_dict' in st else st
    m.load_state_dict(sd, strict=False); m.eval(); return m

def mae_multi(model, drop_rate=0.0):
    sa=n=0
    for pred, tgt in iter_preds(dst, model, sel, drop_rate):
        for b in range(pred.shape[0]):
            valid=~np.isnan(tgt[b])
            sa+=np.abs(pred[b][valid]-tgt[b][valid]).sum(); n+=valid.sum()
    return sa/n

def mae_perapp(models, drop_rate=0.0, center=False):
    sa=n=0
    for k, m in models.items():
        s = sel60 if center else sel
        for pred, tgt in iter_preds(dst, m, s, drop_rate):
            for b in range(pred.shape[0]):
                if pred.ndim==2:
                    v=~np.isnan(tgt[b,SEQ_LENGTH//2,k])
                    if v: sa+=abs(pred[b,0]-tgt[b,SEQ_LENGTH//2,k]); n+=1
                elif center:
                    v=~np.isnan(tgt[b,SEQ_LENGTH//2,k])
                    if v: sa+=abs(pred[b,SEQ_LENGTH//2,0]-tgt[b,SEQ_LENGTH//2,k]); n+=1
                else:
                    v=~np.isnan(tgt[b,:,k])
                    sa+=np.abs(pred[b,:,0][v]-tgt[b,:,k][v]).sum(); n+=v.sum()
    return sa/n

def concurrent_multi(model):
    stats={}
    for pred, tgt in iter_preds(dst, model, sel):
        for b in range(pred.shape[0]):
            t=tgt[b]; p=pred[b]
            n_on=(t>ON_TH).sum(axis=1)
            for step in range(p.shape[0]):
                valid=~np.isnan(t[step])
                if not valid.any(): continue
                k=int(n_on[step])
                d=np.abs(p[step][valid]-t[step][valid])
                st=stats.setdefault(k,{'sa':0.0,'ss':0.0,'n':0})
                st['sa']+=d.sum(); st['ss']+=(d**2).sum(); st['n']+=valid.sum()
    return stats

def concurrent_perapp(models, center=False):
    stats={}
    for k, m in models.items():
        s = sel60 if center else sel
        for pred, tgt in iter_preds(dst, m, s):
            for b in range(pred.shape[0]):
                t=tgt[b]
                n_on=(t>ON_TH).sum(axis=1)
                if pred.ndim==2:
                    v=~np.isnan(t[SEQ_LENGTH//2,k])
                    if not v: continue
                    kk=int(n_on[SEQ_LENGTH//2])
                    d=abs(pred[b,0]-t[SEQ_LENGTH//2,k])
                    st=stats.setdefault(kk,{'sa':0.0,'ss':0.0,'n':0})
                    st['sa']+=d; st['ss']+=d**2; st['n']+=1
                elif center:
                    v=~np.isnan(t[SEQ_LENGTH//2,k])
                    if not v: continue
                    kk=int(n_on[SEQ_LENGTH//2])
                    d=abs(pred[b,SEQ_LENGTH//2,0]-t[SEQ_LENGTH//2,k])
                    st=stats.setdefault(kk,{'sa':0.0,'ss':0.0,'n':0})
                    st['sa']+=d; st['ss']+=d**2; st['n']+=1
                else:
                    for step in range(pred.shape[1]):
                        v=~np.isnan(t[step,k])
                        if not v: continue
                        kk=int(n_on[step])
                        d=abs(pred[b,step,0]-t[step,k])
                        st=stats.setdefault(kk,{'sa':0.0,'ss':0.0,'n':0})
                        st['sa']+=d; st['ss']+=d**2; st['n']+=1
    return stats

def summarize_stats(stats, label):
    def grp(lo,hi):
        ks=[k for k in stats if lo<=k<=hi] if hi is not None else [k for k in stats if k>=lo]
        sa=ss=n=0
        for k in ks: sa+=stats[k]['sa']; ss+=stats[k]['ss']; n+=stats[k]['n']
        return round(sa/n,3) if n else None, round((ss/n)**0.5,3) if n else None, n
    l,g,b = grp(2,3), grp(4,None), grp(0,None)
    print(f'{label}: light(2-3)={l[0]} busy(4+)={g[0]} overall={b[0]}')

results = {}
# ===== Drop robustness =====
print('=== Drop robustness (remaining methods) ===', flush=True)
drop_rates = [0.0, 0.05, 0.20]
for name, kind in [('CL-MDTFN','multi'),('HMoE','multi'),('Seq2Seq','perapp')]:
    print(f'-- {name} --', flush=True)
    mae0 = None
    for r in drop_rates:
        if kind=='multi':
            m = load_multi(CLMDTFN if 'CL' in name else HMoE, name)
            v = mae_multi(m, r)
        else:
            models = {k: load_perapp(Seq2Seq, 'Seq2Seq', k) for k in range(8)}
            v = mae_perapp(models, r)
        if mae0 is None: mae0 = v
        print(f'  {int(r*100)}%: {v:.2f}W ({(v/mae0-1)*100:+.1f}%)', flush=True)

for name in ['DU-NILM','FFSTT','Attention-NILM']:
    print(f'-- {name} --', flush=True)
    models = {k: load_perapp({'DU-NILM':DUNILM,'FFSTT':FFSTT,'Attention-NILM':AttentionNILM}[name], name, k) for k in range(8)}
    mae0 = None
    for r in drop_rates:
        v = mae_perapp(models, r)
        if mae0 is None: mae0 = v
        print(f'  {int(r*100)}%: {v:.2f}W ({(v/mae0-1)*100:+.1f}%)', flush=True)

print('-- Seq2Point (center) --', flush=True)
models = {k: load_perapp(Seq2PointSeq2Seq, 'Seq2Point', k) for k in range(8)}
mae0 = None
for r in drop_rates:
    v = mae_perapp(models, r, center=True)
    if mae0 is None: mae0 = v
    print(f'  {int(r*100)}%: {v:.2f}W ({(v/mae0-1)*100:+.1f}%)', flush=True)

# ===== Concurrent load =====
print('=== Concurrent load (remaining methods) ===', flush=True)
for name, kind in [('CL-MDTFN','multi'),('HMoE','multi')]:
    m = load_multi(CLMDTFN if 'CL' in name else HMoE, name)
    summarize_stats(concurrent_multi(m), name)
for name in ['DU-NILM','FFSTT','Attention-NILM']:
    models = {k: load_perapp({'DU-NILM':DUNILM,'FFSTT':FFSTT,'Attention-NILM':AttentionNILM}[name], name, k) for k in range(8)}
    summarize_stats(concurrent_perapp(models), name)
print('-- Seq2Point (center) --', flush=True)
models = {k: load_perapp(Seq2PointSeq2Seq, 'Seq2Point', k) for k in range(8)}
summarize_stats(concurrent_perapp(models, center=True), 'Seq2Point')
print('ALL DONE')
