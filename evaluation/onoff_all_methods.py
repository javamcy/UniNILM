# -*- coding: utf-8 -*-
"""On/off (calibrated thresholds) + drop + concurrent-load analysis for all methods.
Already covered methods (UniNILM/Mamba2NILM/Seq2Seq and part of CLMDTFN/HMoE) are not repeated.
Remaining: DUNILM, FFSTT, AttentionNILM, Seq2Point (per-appliance); drop/concurrent for CLMDTFN/HMoE.
"""
import sys, os, json, warnings
warnings.filterwarnings('ignore')
CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE)
import numpy as np, torch
from baseline_experiments_v3 import (Seq2PointSeq2Seq, Seq2Seq, DUNILM, FFSTT, AttentionNILM,
                                     CLMDTFN, HMoE, NILMDataset, SEQ_LENGTH, APPLIANCE_NAMES)

DATA = os.environ.get('AMPDS2_PATH', os.path.join(CODE, 'data', 'ampds2_aligned.csv'))
OUT_DIR = os.path.join(CODE, 'results')
os.makedirs(OUT_DIR, exist_ok=True)
DEV=torch.device('cuda'); TRAIN=VAL=40320; MIN_EV=5
THRESH_CANDS = np.arange(5, 101, 5)
MODEL_DIR = os.path.join(CODE, 'trained_models')

dsv = NILMDataset(DATA, seq_length=SEQ_LENGTH, split='val', train_size=TRAIN, val_size=VAL, appliance_columns=APPLIANCE_NAMES)
dst = NILMDataset(DATA, seq_length=SEQ_LENGTH, split='test', train_size=TRAIN, val_size=VAL, appliance_columns=APPLIANCE_NAMES)

def iter_preds(ds, model, sel, batch=64):
    with torch.no_grad():
        for i in range(0, len(sel), batch):
            idxs = sel[i:i+batch]
            ins, raws, tgts, poss, days = [], [], [], [], []
            for j in idxs:
                a,b,c,d,e = ds[int(j)]; ins.append(a); raws.append(b); tgts.append(c); poss.append(d); days.append(e)
            ti=torch.stack(ins).to(DEV); tr=torch.stack(raws).to(DEV); sp=torch.stack(poss).squeeze(-1).to(DEV)
            out,_,_,_ = model(ti, tr, sp)
            yield out.float().cpu().numpy(), torch.stack(tgts).numpy()

def collect(ds, model, sel, center_only=False):
    """center_only: only take the window center step (Seq2Point)."""
    P = {a: [] for a in APPLIANCE_NAMES}; T = {a: [] for a in APPLIANCE_NAMES}
    for pred, tgt in iter_preds(ds, model, sel):
        for b in range(pred.shape[0]):
            if pred.ndim == 2:  # center point [B,1]
                a0 = 0
                for k in range(8):
                    P[APPLIANCE_NAMES[k]].append(np.atleast_1d(pred[b,0])); T[APPLIANCE_NAMES[k]].append(np.atleast_1d(tgt[b, SEQ_LENGTH//2, k]))
            elif center_only:
                for k in range(8):
                    P[APPLIANCE_NAMES[k]].append(np.atleast_1d(pred[b,SEQ_LENGTH//2,k])); T[APPLIANCE_NAMES[k]].append(np.atleast_1d(tgt[b,SEQ_LENGTH//2,k]))
            else:
                for k in range(8):
                    P[APPLIANCE_NAMES[k]].append(pred[b,:,k]); T[APPLIANCE_NAMES[k]].append(tgt[b,:,k])
    return {a: (np.concatenate(P[a]) if len(P[a]) else None) for a in APPLIANCE_NAMES}, \
           {a: (np.concatenate(T[a]) if len(T[a]) else None) for a in APPLIANCE_NAMES}

def f1_at(p, t, th):
    valid=~np.isnan(t)
    p_on=(p>th)&valid; t_on=(t>th)&valid
    tp=(p_on&t_on).sum(); fp=(p_on&~t_on).sum(); fn=(~p_on&t_on).sum()
    prec=tp/(tp+fp) if tp+fp else 0; rec=tp/(tp+fn) if tp+fn else 0
    return 2*prec*rec/(prec+rec) if prec+rec else 0.0, int(tp), int(fp), int(fn)

def best_thresh(Pv, Tv):
    best={}
    for a in APPLIANCE_NAMES:
        p,t=Pv[a],Tv[a]
        if p is None: best[a]=(10.0,0.0); continue
        bf,bth=-1,10.0
        for th in THRESH_CANDS:
            f1,*_=f1_at(p,t,th)
            if f1>bf: bf,bth=f1,th
        best[a]=(bth,bf)
    return best

def events(binary):
    out,s=[],None
    for i,v in enumerate(binary):
        if v and s is None: s=i
        elif not v and s is not None:
            if i-s>=MIN_EV: out.append((s,i))
            s=None
    if s is not None and len(binary)-s>=MIN_EV: out.append((s,len(binary)))
    return out
def ov(a,b): return max(0,min(a[1],b[1])-max(a[0],b[0]))

def eval_onoff(Pt, Tt, ths):
    agg={'tp':0,'fp':0,'fn':0,'atp':0,'itp':0,'miss':0,'fp_ev':0}
    for a in APPLIANCE_NAMES:
        p,t=Pt[a],Tt[a]
        if p is None: continue
        th,_=ths[a]
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

def per_app_models(method):
    """Per-appliance methods: return the 8 (app_idx, model) pairs, or None."""
    models = {}
    cls_map = {'Seq2Point': Seq2PointSeq2Seq, 'DU-NILM': DUNILM, 'FFSTT': FFSTT, 'Attention-NILM': AttentionNILM, 'Seq2Seq': Seq2Seq}
    cls = cls_map[method]
    for k in range(8):
        path = os.path.join(MODEL_DIR, f'ampds2_4w_{method}_app{k}.pt')
        if not os.path.exists(path): continue
        m = cls(input_dim=1, n_appliances=1).to(DEV)
        st = torch.load(path, map_location=DEV, weights_only=False)
        sd = st['model_state_dict'] if isinstance(st,dict) and 'model_state_dict' in st else st
        m.load_state_dict(sd, strict=False); m.eval()
        models[k] = m
    return models

def collect_per_app(ds, models, sel, center_only=False):
    P={a:[] for a in APPLIANCE_NAMES}; T={a:[] for a in APPLIANCE_NAMES}
    for k, m in models.items():
        for pred, tgt in iter_preds(ds, m, sel):
            for b in range(pred.shape[0]):
                if pred.ndim==2:
                    P[APPLIANCE_NAMES[k]].append(np.atleast_1d(pred[b,0])); T[APPLIANCE_NAMES[k]].append(np.atleast_1d(tgt[b,SEQ_LENGTH//2,k]))
                elif center_only:
                    P[APPLIANCE_NAMES[k]].append(np.atleast_1d(pred[b,SEQ_LENGTH//2,0])); T[APPLIANCE_NAMES[k]].append(np.atleast_1d(tgt[b,SEQ_LENGTH//2,k]))
                else:
                    P[APPLIANCE_NAMES[k]].append(pred[b,:,0]); T[APPLIANCE_NAMES[k]].append(tgt[b,:,k])
    return {a:(np.concatenate(P[a]) if len(P[a]) else None) for a in APPLIANCE_NAMES}, \
           {a:(np.concatenate(T[a]) if len(T[a]) else None) for a in APPLIANCE_NAMES}

def run_per_app_all(method, center_only=False):
    models = per_app_models(method)
    sel = np.arange(0, len(dst), SEQ_LENGTH)
    selv = np.arange(0, len(dsv), SEQ_LENGTH)
    Pv,Tv = collect_per_app(dsv, models, selv, center_only)
    Pt,Tt = collect_per_app(dst, models, sel, center_only)
    ths = best_thresh(Pv,Tv)
    agg = eval_onoff(Pt,Tt,ths)
    prec=agg['tp']/(agg['tp']+agg['fp']) if agg['tp']+agg['fp'] else 0
    rec=agg['tp']/(agg['tp']+agg['fn']) if agg['tp']+agg['fn'] else 0
    f1=2*prec*rec/(prec+rec) if prec+rec else 0
    atp_rate=agg['atp']/(agg['atp']+agg['itp']+agg['miss']) if agg['atp']+agg['itp']+agg['miss'] else 0
    return {'f1':round(f1,3),'precision':round(prec,3),'recall':round(rec,3),
            'atp_rate':round(atp_rate,3),'atp':agg['atp'],'itp':agg['itp'],'miss':agg['miss']}

results = {}
for method in ['DU-NILM','FFSTT','Attention-NILM']:
    print(f'=== {method} on/off ===', flush=True)
    results[method] = {'onoff': run_per_app_all(method)}

json.dump(results, open(os.path.join(OUT_DIR, 'onoff_allmethods.json'), 'w'), ensure_ascii=False, indent=2)
print('saved after per-app methods', flush=True)
# Seq2Point: center point, stride=60 for better coverage
print('=== Seq2Point on/off (stride=60 center) ===', flush=True)
models = per_app_models('Seq2Point')
sel60 = np.arange(0, len(dst), 60)
selv60 = np.arange(0, len(dsv), 60)
Pv,Tv = collect_per_app(dsv, models, selv60)
Pt,Tt = collect_per_app(dst, models, sel60)
ths = best_thresh(Pv,Tv)
agg = eval_onoff(Pt,Tt,ths)
prec=agg['tp']/(agg['tp']+agg['fp']) if agg['tp']+agg['fp'] else 0
rec=agg['tp']/(agg['tp']+agg['fn']) if agg['tp']+agg['fn'] else 0
f1=2*prec*rec/(prec+rec) if prec+rec else 0
atp_rate=agg['atp']/(agg['atp']+agg['itp']+agg['miss']) if agg['atp']+agg['itp']+agg['miss'] else 0
results['Seq2Point'] = {'onoff': {'f1':round(f1,3),'precision':round(prec,3),'recall':round(rec,3),
                                  'atp_rate':round(atp_rate,3),'atp':agg['atp'],'itp':agg['itp'],'miss':agg['miss']}}

json.dump(results, open(os.path.join(OUT_DIR, 'onoff_allmethods.json'), 'w'), ensure_ascii=False, indent=2)
for m, r in results.items():
    print(f"{m}: {r['onoff']}")
print('Saved onoff_allmethods.json')
