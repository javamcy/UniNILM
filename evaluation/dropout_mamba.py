# -*- coding: utf-8 -*-
"""Data-transmission-drop simulation: Mamba2NILM baseline comparison
(UniNILM results are reported in the paper)."""
import sys, os, warnings
warnings.filterwarnings('ignore')
CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE)
import numpy as np, torch
from baseline_experiments_v3 import Mamba2NILM, NILMDataset, SEQ_LENGTH, APPLIANCE_NAMES

DATA = os.environ.get('AMPDS2_PATH', os.path.join(CODE, 'data', 'ampds2_aligned.csv'))
DEV=torch.device('cuda'); TRAIN=VAL=40320
ds = NILMDataset(DATA, seq_length=SEQ_LENGTH, split='test', train_size=TRAIN, val_size=VAL, appliance_columns=APPLIANCE_NAMES)
sel = np.arange(0, len(ds), SEQ_LENGTH)
m = Mamba2NILM(input_dim=1, n_appliances=8).to(DEV)
m.load_state_dict(torch.load(os.path.join(CODE,'trained_models/ampds2_4w_Mamba2NILM.pt'), map_location=DEV, weights_only=False))
m.eval()

def eval_drop(drop_rate, seed=42):
    rng = np.random.RandomState(seed)
    sa=ss=n=0
    with torch.no_grad():
        for i in range(0, len(sel), 64):
            idxs = sel[i:i+64]
            ins, raws, tgts, poss, days = [], [], [], [], []
            for j in idxs:
                a,b,c,d,e = ds[int(j)]; ins.append(a); raws.append(b); tgts.append(c); poss.append(d); days.append(e)
            ti = torch.stack(ins); tr = torch.stack(raws)
            if drop_rate > 0:
                mask = rng.rand(ti.shape[0], SEQ_LENGTH, 1) > drop_rate
                ti = ti * torch.FloatTensor(mask)
            out,_,_,_ = m(ti.to(DEV), tr.to(DEV), torch.stack(poss).squeeze(-1).to(DEV))
            p = out.float().cpu().numpy(); t = torch.stack(tgts).numpy()
            for b in range(p.shape[0]):
                valid = ~np.isnan(t[b])
                d_ = np.abs(p[b][valid]-t[b][valid])
                sa += d_.sum(); ss += (d_**2).sum(); n += valid.sum()
    return sa/n

print('Mamba2NILM under data-transmission drops:')
b = eval_drop(0.0); print(f'  0%: {b:.2f}W')
for r in [0.05, 0.10, 0.20]:
    v = eval_drop(r)
    print(f'  {int(r*100)}%: {v:.2f}W ({(v/b-1)*100:+.1f}%)')
