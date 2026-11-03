# -*- coding: utf-8 -*-
"""Calibrated-threshold on/off metrics (F1/PR/ATP/ITP) for CL-MDTFN / HMoE."""
import sys, os, json, warnings
warnings.filterwarnings('ignore')
CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE)
import numpy as np, torch
from baseline_experiments_v3 import CLMDTFN, HMoE, NILMDataset, SEQ_LENGTH, APPLIANCE_NAMES

DATA = os.environ.get('AMPDS2_PATH', os.path.join(CODE, 'data', 'ampds2_aligned.csv'))
OUT_DIR = os.path.join(CODE, 'results')
os.makedirs(OUT_DIR, exist_ok=True)
DEV=torch.device('cuda'); TRAIN=VAL=40320; MIN_EV=5
THRESH_CANDS = np.arange(5, 101, 5)

dsv = NILMDataset(DATA, seq_length=SEQ_LENGTH, split='val', train_size=TRAIN, val_size=VAL, appliance_columns=APPLIANCE_NAMES)
dst = NILMDataset(DATA, seq_length=SEQ_LENGTH, split='test', train_size=TRAIN, val_size=VAL, appliance_columns=APPLIANCE_NAMES)

def iter_preds(ds, model, sel, batch=64):
    supports='day_type' in model.forward.__code__.co_varnames
    with torch.no_grad():
        for i in range(0, len(sel), batch):
            idxs = sel[i:i+batch]
            ins, raws, tgts, poss, days = [], [], [], [], []
            for j in idxs:
                a,b,c,d,e = ds[int(j)]; ins.append(a); raws.append(b); tgts.append(c); poss.append(d); days.append(e)
            ti=torch.stack(ins).to(DEV); tr=torch.stack(raws).to(DEV); sp=torch.stack(poss).squeeze(-1).to(DEV)
            if supports: out,_,_,_ = model(ti, tr, sp, torch.stack(days).to(DEV))
            else: out,_,_,_ = model(ti, tr, sp)
            yield out.float().cpu().numpy(), torch.stack(tgts).numpy()

def collect(ds, model, app_idx=None):
    sel = np.arange(0, len(ds), SEQ_LENGTH)
    P = {a: [] for a in APPLIANCE_NAMES}; T = {a: [] for a in APPLIANCE_NAMES}
    for pred, tgt in iter_preds(ds, model, sel):
        for b in range(pred.shape[0]):
            if app_idx is not None:
                a = APPLIANCE_NAMES[app_idx]
                P[a].append(pred[b,:,0]); T[a].append(tgt[b,:,app_idx])
            else:
                for k, a in enumerate(APPLIANCE_NAMES):
                    P[a].append(pred[b,:,k]); T[a].append(tgt[b,:,k])
    return {a: (np.concatenate(P[a]) if len(P[a]) else None) for a in APPLIANCE_NAMES}, \
           {a: (np.concatenate(T[a]) if len(T[a]) else None) for a in APPLIANCE_NAMES}

def f1_at(p, t, th):
    valid = ~np.isnan(t)
    p_on = (p>th)&valid; t_on = (t>th)&valid
    tp=(p_on&t_on).sum(); fp=(p_on&~t_on).sum(); fn=(~p_on&t_on).sum()
    prec = tp/(tp+fp) if tp+fp else 0; rec = tp/(tp+fn) if tp+fn else 0
    return 2*prec*rec/(prec+rec) if prec+rec else 0.0, int(tp), int(fp), int(fn)

def best_thresh(Pv, Tv):
    best = {}
    for a in APPLIANCE_NAMES:
        p, t = Pv[a], Tv[a]
        if p is None: best[a]=(10.0,0.0); continue
        bf, bth = -1, 10.0
        for th in THRESH_CANDS:
            f1,*_ = f1_at(p,t,th)
            if f1>bf: bf,bth=f1,th
        best[a]=(bth,bf)
    return best

def events(binary):
    out, s = [], None
    for i,v in enumerate(binary):
        if v and s is None: s=i
        elif not v and s is not None:
            if i-s>=MIN_EV: out.append((s,i))
            s=None
    if s is not None and len(binary)-s>=MIN_EV: out.append((s,len(binary)))
    return out
def ov(a,b): return max(0, min(a[1],b[1])-max(a[0],b[0]))

def atp_itp(Pt, Tt, ths):
    agg = {'tp':0,'fp':0,'fn':0,'atp':0,'itp':0,'miss':0,'fp_ev':0}
    for a in APPLIANCE_NAMES:
        p,t = Pt[a], Tt[a]
        if p is None: continue
        th,_ = ths[a]
        valid=~np.isnan(t)
        p_on=(p>th)&valid; t_on=(t>th)&valid
        tp=(p_on&t_on).sum(); fp=(p_on&~t_on).sum(); fn=(~p_on&t_on).sum()
        agg['tp']+=int(tp); agg['fp']+=int(fp); agg['fn']+=int(fn)
        g_ev=events(t_on.astype(int)); p_ev=events(p_on.astype(int))
        used=set()
        for g in g_ev:
            best,best_ov=None,0
            for pi,pe in enumerate(p_ev):
                if pi in used: continue
                o=ov(g,pe)
                if o>best_ov: best,best_ov=pi,o
            if best is not None and best_ov>0:
                ratio=best_ov/max(g[1]-g[0],p_ev[best][1]-p_ev[best][0])
                used.add(best)
                if ratio>=0.5: agg['atp']+=1
                else: agg['itp']+=1
            else: agg['miss']+=1
        agg['fp_ev']+=len(p_ev)-len(used)
    return agg

def run(name, cls, path):
    print(f'--- {name} ---', flush=True)
    m = cls(input_dim=1, n_appliances=8).to(DEV)
    m.load_state_dict(torch.load(path, map_location=DEV, weights_only=False)); m.eval()
    Pv,Tv = collect(dsv, m); Pt,Tt = collect(dst, m)
    ths = best_thresh(Pv,Tv)
    agg = atp_itp(Pt,Tt,ths)
    prec=agg['tp']/(agg['tp']+agg['fp']) if agg['tp']+agg['fp'] else 0
    rec=agg['tp']/(agg['tp']+agg['fn']) if agg['tp']+agg['fn'] else 0
    f1=2*prec*rec/(prec+rec) if prec+rec else 0
    atp_rate=agg['atp']/(agg['atp']+agg['itp']+agg['miss']) if agg['atp']+agg['itp']+agg['miss'] else 0
    print(f'  F1={f1:.3f} P={prec:.3f} R={rec:.3f} ATP={agg["atp"]} ITP={agg["itp"]} Miss={agg["miss"]} ATPrate={atp_rate:.3f}')
    return {'f1':round(f1,3),'precision':round(prec,3),'recall':round(rec,3),'atp_rate':round(atp_rate,3), **{k:agg[k] for k in ['atp','itp','miss']}}

res = {}
res['CL-MDTFN'] = run('CL-MDTFN', CLMDTFN, os.path.join(CODE,'trained_models/ampds2_4w_CL-MDTFN.pt'))
res['HMoE'] = run('HMoE', HMoE, os.path.join(CODE,'trained_models/ampds2_4w_HMoE.pt'))
json.dump(res, open(os.path.join(OUT_DIR, 'onoff_extended.json'), 'w'), ensure_ascii=False, indent=2)
print('Saved')
