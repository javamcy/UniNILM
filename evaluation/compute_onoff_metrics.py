# -*- coding: utf-8 -*-
"""AMPds2 week4 on/off detection metrics: step-level F1/Precision/Recall + event-level ATP/ITP.
UniNILM vs Mamba2NILM vs Seq2Seq. Threshold 10 W, minimum event length 5 steps."""
import sys, os, json, warnings
warnings.filterwarnings('ignore')
CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE)
import numpy as np, torch
from baseline_experiments_v3 import BiMamba, Mamba2NILM, Seq2Seq, NILMDataset, SEQ_LENGTH, APPLIANCE_NAMES

DATA = os.environ.get('AMPDS2_PATH', os.path.join(CODE, 'data', 'ampds2_aligned.csv'))
OUT_DIR = os.path.join(CODE, 'results')
os.makedirs(OUT_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRAIN = VAL = 40320
TH = 10.0
MIN_EV = 5

ds = NILMDataset(DATA, seq_length=SEQ_LENGTH, split='test', train_size=TRAIN, val_size=VAL,
                 appliance_columns=APPLIANCE_NAMES)
sel = np.arange(0, len(ds), SEQ_LENGTH)
print(f'windows: {len(sel)}', flush=True)

def iter_preds(model, batch=64):
    with torch.no_grad():
        for i in range(0, len(sel), batch):
            idxs = sel[i:i+batch]
            ins, raws, tgts, poss, days = [], [], [], [], []
            for j in idxs:
                a, b, c, d, e = ds[int(j)]
                ins.append(a); raws.append(b); tgts.append(c); poss.append(d); days.append(e)
            out, _, _, _ = model(torch.stack(ins).to(DEVICE), torch.stack(raws).to(DEVICE),
                                 torch.stack(poss).squeeze(-1).to(DEVICE))
            yield out.float().cpu().numpy(), torch.stack(tgts).numpy()

def events(binary):
    out, s = [], None
    for i, v in enumerate(binary):
        if v and s is None: s = i
        elif not v and s is not None:
            if i - s >= MIN_EV: out.append((s, i))
            s = None
    if s is not None and len(binary) - s >= MIN_EV: out.append((s, len(binary)))
    return out

def overlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))

def onoff_1col(p, t):
    valid = ~np.isnan(t)
    if not valid.any():
        return None
    p_on = (p > TH) & valid
    t_on = (t > TH) & valid
    tp = (p_on & t_on).sum(); fp = (p_on & ~t_on).sum(); fn = (~p_on & t_on).sum()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    g_ev = events(t_on.astype(int)); p_ev = events(p_on.astype(int))
    atp = itp = miss = 0; used = set()
    for g in g_ev:
        best, best_ov = None, 0
        for pi, pe in enumerate(p_ev):
            if pi in used: continue
            ov = overlap(g, pe)
            if ov > best_ov: best, best_ov = pi, ov
        if best is not None and best_ov > 0:
            ratio = best_ov / max(g[1]-g[0], p_ev[best][1]-p_ev[best][0])
            used.add(best)
            if ratio >= 0.5: atp += 1
            else: itp += 1
        else:
            miss += 1
    fp_ev = len(p_ev) - len(used)
    return {'tp': int(tp), 'fp': int(fp), 'fn': int(fn),
            'precision': prec, 'recall': rec, 'f1': f1,
            'atp': atp, 'itp': itp, 'miss': miss, 'fp_ev': fp_ev}

def run(name, model, app_idx=None):
    print(f'--- {name} ---', flush=True)
    per = {a: {'tp':0,'fp':0,'fn':0,'atp':0,'itp':0,'miss':0,'fp_ev':0} for a in APPLIANCE_NAMES}
    for pred, tgt in iter_preds(model):
        if app_idx is not None:
            for b in range(pred.shape[0]):
                r = onoff_1col(pred[b, :, 0], tgt[b, :, app_idx])
                if r:
                    for kk in per[APPLIANCE_NAMES[app_idx]]: per[APPLIANCE_NAMES[app_idx]][kk] += r[kk]
        else:
            for b in range(pred.shape[0]):
                for k in range(8):
                    r = onoff_1col(pred[b, :, k], tgt[b, :, k])
                    if r:
                        for kk in per[APPLIANCE_NAMES[k]]: per[APPLIANCE_NAMES[k]][kk] += r[kk]
    return per

results = {}

m = BiMamba(input_dim=1, n_appliances=8, d_model=64).to(DEVICE)
m.load_state_dict(torch.load(os.path.join(CODE, 'trained_models/bimamba_daygate_ampds2/ampds2_week4_best.pt'),
                             map_location='cpu', weights_only=False))
results['UniNILM'] = run('UniNILM', m)

m = Mamba2NILM(input_dim=1, n_appliances=8).to(DEVICE)
m.load_state_dict(torch.load(os.path.join(CODE, 'trained_models/ampds2_4w_Mamba2NILM.pt'),
                             map_location='cpu', weights_only=False))
results['Mamba2NILM'] = run('Mamba2NILM', m)

seq = {a: {'tp':0,'fp':0,'fn':0,'atp':0,'itp':0,'miss':0,'fp_ev':0} for a in APPLIANCE_NAMES}
for k in range(8):
    mm = Seq2Seq(input_dim=1, n_appliances=1).to(DEVICE)
    mm.load_state_dict(torch.load(os.path.join(CODE, f'trained_models/ampds2_4w_Seq2Seq_app{k}.pt'),
                                  map_location='cpu', weights_only=False))
    r = run(f'Seq2Seq app{k}', mm, app_idx=k)
    seq[APPLIANCE_NAMES[k]] = r[APPLIANCE_NAMES[k]]
results['Seq2Seq'] = seq

def aggregate(per):
    agg = {'tp':0,'fp':0,'fn':0,'atp':0,'itp':0,'miss':0,'fp_ev':0}
    for a in APPLIANCE_NAMES:
        d = per[a]
        if d is None: continue
        for kk in agg: agg[kk] += d[kk]
    prec = agg['tp']/(agg['tp']+agg['fp']) if agg['tp']+agg['fp'] else 0
    rec = agg['tp']/(agg['tp']+agg['fn']) if agg['tp']+agg['fn'] else 0
    f1 = 2*prec*rec/(prec+rec) if prec+rec else 0
    atp_rate = agg['atp']/(agg['atp']+agg['itp']+agg['miss']) if agg['atp']+agg['itp']+agg['miss'] else 0
    return {'f1': round(f1,3), 'precision': round(prec,3), 'recall': round(rec,3),
            'atp': agg['atp'], 'itp': agg['itp'], 'miss': agg['miss'], 'fp_ev': agg['fp_ev'],
            'atp_rate': round(atp_rate,3)}

out = {name: aggregate(per) for name, per in results.items()}
print('\n===== Overall on/off metrics (AMPds2 week4) =====')
for name, d in out.items():
    print(f'  {name:12s}: F1={d["f1"]:.3f} P={d["precision"]:.3f} R={d["recall"]:.3f} '
          f'ATP={d["atp"]} ITP={d["itp"]} Miss={d["miss"]} FPev={d["fp_ev"]} ATPrate={d["atp_rate"]:.3f}')

with open(os.path.join(OUT_DIR, 'onoff_metrics.json'), 'w') as f:
    json.dump({'overall': out, 'per_appliance': results}, f, ensure_ascii=False, indent=2)
print('Saved -> onoff_metrics.json')
