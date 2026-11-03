# -*- coding: utf-8 -*-
"""Data-transmission-drop simulation: randomly zero out timesteps of the input window
at test-time inference and compare MAE.
UniNILM AMPds2 week4. Drop rates: 0% / 5% / 10% / 20%."""
import sys, os, warnings
warnings.filterwarnings('ignore')
CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE)
import numpy as np, torch
from baseline_experiments_v3 import BiMamba, NILMDataset, SEQ_LENGTH, APPLIANCE_NAMES

DATA = os.environ.get('AMPDS2_PATH', os.path.join(CODE, 'data', 'ampds2_aligned.csv'))
DEV=torch.device('cuda'); TRAIN=VAL=40320
ds = NILMDataset(DATA, seq_length=SEQ_LENGTH, split='test', train_size=TRAIN, val_size=VAL, appliance_columns=APPLIANCE_NAMES)
sel = np.arange(0, len(ds), SEQ_LENGTH)
m = BiMamba(input_dim=1, n_appliances=8, d_model=64).to(DEV)
m.load_state_dict(torch.load(os.path.join(CODE,'trained_models/bimamba_daygate_ampds2/ampds2_week4_best.pt'), map_location=DEV, weights_only=False))
m.eval()

def eval_with_drop(drop_rate, seed=42):
    rng = np.random.RandomState(seed)
    sa=ss=n=0
    with torch.no_grad():
        for i in range(0, len(sel), 64):
            idxs = sel[i:i+64]
            ins, raws, tgts, poss, days = [], [], [], [], []
            for j in idxs:
                a,b,c,d,e = ds[int(j)]; ins.append(a); raws.append(b); tgts.append(c); poss.append(d); days.append(e)
            ti = torch.stack(ins); tr = torch.stack(raws)
            # Drop: randomly zero out timesteps of the input window (mask the forward features)
            if drop_rate > 0:
                B = ti.shape[0]
                mask = rng.rand(B, SEQ_LENGTH, 1) > drop_rate
                ti = ti * torch.FloatTensor(mask)
            out,_,_,_ = m(ti.to(DEV), tr.to(DEV), torch.stack(poss).squeeze(-1).to(DEV), torch.stack(days).to(DEV))
            p = out.float().cpu().numpy(); t = torch.stack(tgts).numpy()
            for b in range(p.shape[0]):
                valid = ~np.isnan(t[b])
                d_ = np.abs(p[b][valid]-t[b][valid])
                sa += d_.sum(); ss += (d_**2).sum(); n += valid.sum()
    return sa/n, (ss/n)**0.5

print('drop_rate | MAE    | RMSE  | MAE degradation (%)')
base_mae, base_rmse = eval_with_drop(0.0)
print(f'   0%     | {base_mae:6.2f} | {base_rmse:6.2f} |  -')
for rate in [0.05, 0.10, 0.20]:
    mae, rmse = eval_with_drop(rate)
    print(f'   {int(rate*100):3d}%   | {mae:6.2f} | {rmse:6.2f} | {(mae/base_mae-1)*100:+5.1f}%')
