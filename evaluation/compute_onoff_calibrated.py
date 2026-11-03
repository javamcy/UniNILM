# -*- coding: utf-8 -*-
"""On/off metrics with per-appliance optimal threshold calibration.
Thresholds are selected on the validation set (F1-maximizing) and evaluated on the test set.
UniNILM vs Mamba2NILM vs Seq2Seq."""
import sys, os, json, warnings
warnings.filterwarnings('ignore')
CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE)
import numpy as np, torch
from baseline_experiments_v3 import BiMamba, Mamba2NILM, Seq2Seq, NILMDataset, SEQ_LENGTH, APPLIANCE_NAMES

DATA = os.environ.get('AMPDS2_PATH', os.path.join(CODE, 'data', 'ampds2_aligned.csv'))
OUT_DIR = os.path.join(CODE, 'results')
os.makedirs(OUT_DIR, exist_ok=True)
DEV = torch.device('cuda')
TRAIN = VAL = 40320
THRESH_CANDS = np.arange(5, 101, 5)  # 5..100 W

def build_ds(split):
    return NILMDataset(DATA, seq_length=SEQ_LENGTH, split=split, train_size=TRAIN, val_size=VAL,
                       appliance_columns=APPLIANCE_NAMES)

def iter_preds(ds, model, sel, batch=64):
    supports_day = 'day_type' in model.forward.__code__.co_varnames
    with torch.no_grad():
        for i in range(0, len(sel), batch):
            idxs = sel[i:i+batch]
            ins, raws, tgts, poss, days = [], [], [], [], []
            for j in idxs:
                a,b,c,d,e = ds[int(j)]; ins.append(a); raws.append(b); tgts.append(c); poss.append(d); days.append(e)
            ti = torch.stack(ins).to(DEV); tr = torch.stack(raws).to(DEV); sp = torch.stack(poss).squeeze(-1).to(DEV)
            if supports_day:
                out,_,_,_ = model(ti, tr, sp, torch.stack(days).to(DEV))
            else:
                out,_,_,_ = model(ti, tr, sp)
            yield out.float().cpu().numpy(), torch.stack(tgts).numpy()

def collect_preds(ds, model, app_idx=None):
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
    p_on = (p > th) & valid; t_on = (t > th) & valid
    tp = (p_on & t_on).sum(); fp = (p_on & ~t_on).sum(); fn = (~p_on & t_on).sum()
    prec = tp/(tp+fp) if tp+fp else 0; rec = tp/(tp+fn) if tp+fn else 0
    return 2*prec*rec/(prec+rec) if prec+rec else 0.0, int(tp), int(fp), int(fn)

def best_thresh(Pv, Tv):
    best = {}
    for a in APPLIANCE_NAMES:
        p, t = Pv[a], Tv[a]
        if p is None: best[a] = (10.0, 0.0); continue
        bf, bth = -1, 10.0
        for th in THRESH_CANDS:
            f1, *_ = f1_at(p, t, th)
            if f1 > bf: bf, bth = f1, th
        best[a] = (bth, bf)
    return best

def eval_with(Pt, Tt, best):
    agg = {'tp':0,'fp':0,'fn':0}
    per = {}
    for a in APPLIANCE_NAMES:
        p, t = Pt[a], Tt[a]
        if p is None:
            per[a] = None; continue
        th, _ = best[a]
        f1, tp, fp, fn = f1_at(p, t, th)
        per[a] = {'th': th, 'f1': round(f1,3), 'tp': tp, 'fp': fp, 'fn': fn}
        agg['tp'] += tp; agg['fp'] += fp; agg['fn'] += fn
    prec = agg['tp']/(agg['tp']+agg['fp']) if agg['tp']+agg['fp'] else 0
    rec = agg['tp']/(agg['tp']+agg['fn']) if agg['tp']+agg['fn'] else 0
    f1 = 2*prec*rec/(prec+rec) if prec+rec else 0
    return {'overall_f1': round(f1,3), 'overall_prec': round(prec,3), 'overall_rec': round(rec,3), 'per': per}

print('loading data...', flush=True)
dsv, dst = build_ds('val'), build_ds('test')

def load_model(cls, path, n_app, **kw):
    m = cls(input_dim=1, n_appliances=n_app, **kw).to(DEV)
    m.load_state_dict(torch.load(path, map_location=DEV, weights_only=False))
    m.eval(); return m

results = {}
for name, cls, path, n_app, kw in [
    ('UniNILM', BiMamba, os.path.join(CODE,'trained_models/bimamba_daygate_ampds2/ampds2_week4_best.pt'), 8, {'d_model':64}),
    ('Mamba2NILM', Mamba2NILM, os.path.join(CODE,'trained_models/ampds2_4w_Mamba2NILM.pt'), 8, {}),
]:
    print(f'--- {name} ---', flush=True)
    m = load_model(cls, path, n_app, **kw)
    Pv, Tv = collect_preds(dsv, m); Pt, Tt = collect_preds(dst, m)
    best = best_thresh(Pv, Tv)
    r = eval_with(Pt, Tt, best)
    results[name] = r
    print(f'  overall F1={r["overall_f1"]} P={r["overall_prec"]} R={r["overall_rec"]}')
    for a in APPLIANCE_NAMES:
        print(f'    {a}: th={r["per"][a]["th"]}W F1={r["per"][a]["f1"]}')

# Seq2Seq per-appliance
print('--- Seq2Seq ---', flush=True)
seq_per = {}
for k in range(8):
    a = APPLIANCE_NAMES[k]
    m = load_model(Seq2Seq, os.path.join(CODE, f'trained_models/ampds2_4w_Seq2Seq_app{k}.pt'), 1)
    Pv, Tv = collect_preds(dsv, m, app_idx=k); Pt, Tt = collect_preds(dst, m, app_idx=k)
    best = best_thresh(Pv, Tv)
    r = eval_with(Pt, Tt, best)
    seq_per[a] = r['per'][a]
# Aggregate Seq2Seq
agg = {'tp':0,'fp':0,'fn':0}
for a in APPLIANCE_NAMES:
    d = seq_per[a]
    if d is None: continue
    for kk in agg: agg[kk] += d[kk]
prec = agg['tp']/(agg['tp']+agg['fp']) if agg['tp']+agg['fp'] else 0
rec = agg['tp']/(agg['tp']+agg['fn']) if agg['tp']+agg['fn'] else 0
f1 = 2*prec*rec/(prec+rec) if prec+rec else 0
results['Seq2Seq'] = {'overall_f1': round(f1,3), 'overall_prec': round(prec,3), 'overall_rec': round(rec,3), 'per': seq_per}
print(f'  overall F1={f1:.3f} P={prec:.3f} R={rec:.3f}')

with open(os.path.join(OUT_DIR, 'onoff_calibrated.json'), 'w') as f:
    json.dump(results, f, ensure_ascii=False, indent=2, default=lambda o: int(o) if isinstance(o, (np.integer,)) else str(o))
print('Saved -> onoff_calibrated.json')
