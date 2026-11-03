# -*- coding: utf-8 -*-
"""Cross-dataset rerun: models trained on UKDALE house1 -> tested on house5
(compared with the legacy results)."""
import sys, os, json, warnings
warnings.filterwarnings('ignore')
CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE)
import numpy as np, torch
from baseline_experiments_v3 import (Mamba2NILM, CLMDTFN, HMoE, Seq2PointSeq2Seq,
                                     Seq2Seq, DUNILM, FFSTT, AttentionNILM, NILMDataset,
                                     SEQ_LENGTH, BiMamba)

H5 = os.environ.get('UKDALE_PATH', os.path.join(CODE, 'data', 'house_5_prepared.csv'))
OUT_DIR = os.path.join(CODE, 'results')
os.makedirs(OUT_DIR, exist_ok=True)
DEV=torch.device('cuda')
APPS = ['dishwasher','hairdryer','kettle','microwave','toaster']
MD = os.path.join(CODE, 'trained_models')

# Use the entire house5 data as the test set
ds = NILMDataset(H5, seq_length=SEQ_LENGTH, split='test', train_size=40320, val_size=0, appliance_columns=APPS)
sel = np.arange(0, len(ds), SEQ_LENGTH)
print(f'house5 windows: {len(sel)} (covers {len(sel)*SEQ_LENGTH} steps)', flush=True)

def iter_preds(model, batch=64):
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

def total_mae(model, app_cols=None):
    """app_cols=None: multi-output, all 5 columns; otherwise a single per-appliance column."""
    sa=n=0
    for pred, tgt in iter_preds(model):
        for b in range(pred.shape[0]):
            if app_cols is not None:
                k = app_cols[0]
                pcol = pred[b,:,0] if pred.ndim==3 else pred[b,0]
                if pred.ndim==2:
                    vv = ~np.isnan(tgt[b,SEQ_LENGTH//2,k])
                    if vv: sa+=abs(pcol-tgt[b,SEQ_LENGTH//2,k]); n+=1
                else:
                    v=~np.isnan(tgt[b,:,k])
                    sa+=np.abs(pcol[v]-tgt[b,:,k][v]).sum(); n+=v.sum()
            else:
                v=~np.isnan(tgt[b])
                sa+=np.abs(pred[b][v]-tgt[b][v]).sum(); n+=v.sum()
    return sa/n if n else float('nan')

def load(name, cls, n_app, kw=None, app_idx=None):
    path = os.path.join(MD, f'ukdale_{name}.pt' if app_idx is None else f'ukdale_{name}_app{app_idx}.pt')
    m = cls(input_dim=1, n_appliances=n_app, **(kw or {})).to(DEV)
    st = torch.load(path, map_location=DEV, weights_only=False)
    sd = st['model_state_dict'] if isinstance(st,dict) and 'model_state_dict' in st else st
    m.load_state_dict(sd, strict=False); m.eval(); return m

results = {}
# Multi-output methods
results['BiMamba'] = round(total_mae(load('BiMamba', BiMamba, 5, {'d_model':64})), 2)
results['Mamba2NILM'] = round(total_mae(load('Mamba2NILM', Mamba2NILM, 5)), 2)
results['CL-MDTFN'] = round(total_mae(load('CL-MDTFN', CLMDTFN, 5, {'d_model':64,'M':3,'N_R':2})), 2)
results['HMoE'] = round(total_mae(load('HMoE', HMoE, 5, {'hidden_dim':64})), 2)
# Per-appliance methods
for name, cls, kw in [('DU-NILM', DUNILM, {'base_channels':32,'N_segments':8}), ('Attention-NILM', AttentionNILM, {}), ('Seq2Point', Seq2PointSeq2Seq, {}),
                  ('Seq2Seq', Seq2Seq, {}), ('FFSTT', FFSTT, {})]:
    vals = []
    for k in range(5):
        m = load(name, cls, 1, kw, app_idx=k)
        v = total_mae(m, [k])
        if v == v: vals.append(v)  # skip NaN (missing appliance)
    results[name] = round(float(np.mean(vals)), 2) if vals else None
    print(f'{name}: {results[name]} (valid appliances={len(vals)})', flush=True)

print('\n===== Rerun results (house1->house5) =====')
for m, v in sorted(results.items(), key=lambda x: x[1]):
    print(f'  {m:15s}: MAE={v:.2f}')
json.dump(results, open(os.path.join(OUT_DIR, 'cross_household_rerun.json'), 'w'), ensure_ascii=False, indent=2)
print('\nComparison with legacy results:')
old = json.load(open(os.path.join(MD, 'ukdale_cross_dataset_results.json')))
for m in results:
    print(f'  {m:15s}: new={results[m]:6.2f}  legacy={old.get(m,{}).get("total_mae", "N/A") if isinstance(old.get(m),dict) else old.get(m,"N/A")}')
