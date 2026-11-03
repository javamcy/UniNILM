# -*- coding: utf-8 -*-
"""Verify per-appliance MAE of UniNILM on UKDALE week4 (against the paper table ukdale_app_4w_table.tex)."""
import sys, os, warnings
warnings.filterwarnings('ignore')
CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE)
import numpy as np, torch
from baseline_experiments_v3 import BiMamba, NILMDataset, SEQ_LENGTH

DATA = os.environ.get('UKDALE_PATH', os.path.join(CODE, 'data', 'house_1_prepared.csv'))
CKPT = os.path.join(CODE, 'trained_models/bimamba_daygate_ukdale/ukdale_week4_best.pt')
APPS = ['dishwasher', 'hairdryer', 'kettle', 'microwave', 'toaster']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRAIN = VAL = 1440 * 7 * 4
BATCH = 64

ds = NILMDataset(DATA, seq_length=SEQ_LENGTH, split='test',
                 train_size=TRAIN, val_size=VAL, appliance_columns=APPS)
sel = np.arange(0, len(ds), SEQ_LENGTH)
print(f'test windows: {len(sel)} (covers {len(sel)*SEQ_LENGTH} steps)', flush=True)

model = BiMamba(input_dim=1, n_appliances=5, d_model=64).to(DEVICE)
ck = torch.load(CKPT, map_location='cpu', weights_only=False)
sd = ck['model_state_dict'] if isinstance(ck, dict) and 'model_state_dict' in ck else ck
model.load_state_dict(sd)
model.eval()

app_abs = np.zeros(5); app_sq = np.zeros(5); app_n = np.zeros(5)
with torch.no_grad():
    for i in range(0, len(sel), BATCH):
        idxs = sel[i:i+BATCH]
        ins, raws, tgts, poss, days = [], [], [], [], []
        for j in idxs:
            a, b, c, d, e = ds[int(j)]
            ins.append(a); raws.append(b); tgts.append(c); poss.append(d); days.append(e)
        out, _, _, _ = model(torch.stack(ins).to(DEVICE), torch.stack(raws).to(DEVICE),
                             torch.stack(poss).squeeze(-1).to(DEVICE), torch.stack(days).to(DEVICE))
        pred = out.float().cpu().numpy()
        tgt = torch.stack(tgts).numpy()
        for b in range(pred.shape[0]):
            p, t = pred[b], tgt[b]
            valid = ~np.isnan(t)
            for k in range(5):
                if valid[:, k].any():
                    d = np.abs(p[valid[:, k], k] - t[valid[:, k], k])
                    app_abs[k] += d.sum(); app_sq[k] += (d**2).sum(); app_n[k] += valid[:, k].sum()

print('\n===== UKDALE week4 per-appliance MAE (non-overlapping window evaluation) =====')
total_mae = app_abs.sum() / app_n.sum()
for k, name in enumerate(APPS):
    mae = app_abs[k] / app_n[k] if app_n[k] else float('nan')
    rmse = (app_sq[k] / app_n[k]) ** 0.5 if app_n[k] else float('nan')
    print(f'  {name:12s}: MAE={mae:6.2f}W  RMSE={rmse:6.2f}W  n={int(app_n[k])}')
print(f'  Overall: MAE={total_mae:.2f}W')
