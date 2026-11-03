"""UniNILM test error grouped by the number of concurrent loads (AMPds2 week4 model).
Non-overlapping windows (stride=SEQ_LENGTH), so each timestep is counted once.
Groups: 0-1 / 2-3 / >=4 concurrently active appliances (threshold 10 W), corresponding to
the reviewers' isolated/light/busy regimes.
"""
import sys, os, json, warnings
warnings.filterwarnings('ignore')

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE)

import numpy as np
import torch
from baseline_experiments_v3 import BiMamba, NILMDataset, SEQ_LENGTH, APPLIANCE_NAMES

DATA_PATH = os.environ.get('AMPDS2_PATH', os.path.join(CODE, 'data', 'ampds2_aligned.csv'))
OUT_DIR = os.path.join(CODE, 'results')
os.makedirs(OUT_DIR, exist_ok=True)
CKPT = os.path.join(CODE, 'trained_models/bimamba_daygate_ampds2/ampds2_week4_best.pt')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRAIN_SIZE, VAL_SIZE = 40320, 40320   # week4 configuration
ON_THRESH = 10.0                       # same as the paper's F1 threshold
BATCH = 64

print(f'Device: {DEVICE}', flush=True)

ds = NILMDataset(DATA_PATH, seq_length=SEQ_LENGTH, split='test',
                 train_size=TRAIN_SIZE, val_size=VAL_SIZE, appliance_columns=APPLIANCE_NAMES)
print(f'test samples (overlapping): {len(ds)}', flush=True)

model = BiMamba(input_dim=1, n_appliances=8, d_model=64).to(DEVICE)
ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
sd = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
model.load_state_dict(sd)
model.eval()
print('model loaded', flush=True)

# Non-overlapping window indices
sel = np.arange(0, len(ds), SEQ_LENGTH)
print(f'non-overlapping windows: {len(sel)} (covers {len(sel)*SEQ_LENGTH} timesteps)', flush=True)

# Streaming accumulation: per-concurrent-count statistics
stats = {}  # k -> {sum_abs, sum_sq, n_steps, n_valid, sum_on}

def _acc(k, d_abs, d_sq, n_valid, n_on):
    s = stats.setdefault(k, {'sum_abs': 0.0, 'sum_sq': 0.0, 'n_steps': 0, 'n_valid': 0, 'sum_on': 0.0})
    s['sum_abs'] += float(d_abs)
    s['sum_sq'] += float(d_sq)
    s['n_steps'] += 1
    s['n_valid'] += int(n_valid)
    s['sum_on'] += float(n_on)

with torch.no_grad():
    for i in range(0, len(sel), BATCH):
        idxs = sel[i:i + BATCH]
        ins, raws, tgts, poss, days = [], [], [], [], []
        for j in idxs:
            a, b, c, d, e = ds[int(j)]
            ins.append(a); raws.append(b); tgts.append(c); poss.append(d); days.append(e)
        total_input = torch.stack(ins).to(DEVICE)
        total_raw = torch.stack(raws).to(DEVICE)
        target_raw = torch.stack(tgts).numpy()
        sample_positions = torch.stack(poss).squeeze(-1).to(DEVICE)
        day_type = torch.stack(days).to(DEVICE)

        out, _, _, _ = model(total_input, total_raw, sample_positions, day_type)
        pred = out.cpu().numpy()  # [B, 480, 8]

        for b in range(pred.shape[0]):
            p, t = pred[b], target_raw[b]  # [480, 8]
            n_on = (t > ON_THRESH).sum(axis=1)
            for step in range(p.shape[0]):
                valid = ~np.isnan(t[step])
                if not valid.any():
                    continue
                diff = np.abs(p[step][valid] - t[step][valid])
                _acc(int(n_on[step]), diff.sum(), (diff ** 2).sum(), valid.sum(), float(n_on[step]))
        if (i // BATCH) % 8 == 0:
            print(f'  processed {i+len(idxs)}/{len(sel)} windows', flush=True)

# Aggregation
def finalize(s):
    n = s['n_valid']
    mae = s['sum_abs'] / n if n else float('nan')
    rmse = (s['sum_sq'] / n) ** 0.5 if n else float('nan')
    return mae, rmse, s['n_steps'], s['n_valid'], s['sum_on'] / s['n_steps'] if s['n_steps'] else float('nan')

print('\n===== Fine-grained: per concurrent count (0-8) =====')
fine = {}
for k in sorted(stats):
    mae, rmse, steps, nvalid, avg_on = finalize(stats[k])
    fine[k] = {'mae': round(mae, 3), 'rmse': round(rmse, 3), 'n_steps': steps,
               'n_valid': nvalid, 'avg_on': round(avg_on, 2)}
    print(f'  concurrent={k}: MAE={mae:7.3f}W  RMSE={rmse:7.3f}W  steps={steps:7d}  valid={nvalid:8d}')

print('\n===== Three groups: 0-1 / 2-3 / >=4 =====')
def group(lo, hi):
    keys = [k for k in stats if lo <= k <= hi] if hi is not None else [k for k in stats if k >= lo]
    sa = ss = nv = ns = son = 0.0
    for k in keys:
        sa += stats[k]['sum_abs']; ss += stats[k]['sum_sq']
        nv += stats[k]['n_valid']; ns += stats[k]['n_steps']; son += stats[k]['sum_on']
    mae = sa / nv if nv else float('nan')
    rmse = (ss / nv) ** 0.5 if nv else float('nan')
    return {'mae': round(mae, 3), 'rmse': round(rmse, 3), 'n_steps': int(ns), 'n_valid': int(nv),
            'avg_on': round(son / ns, 2) if ns else float('nan')}

g0 = group(0, 1)   # isolated
g1 = group(2, 3)   # light
g2 = group(4, None)  # busy
print('  [isolated 0-1]:', json.dumps(g0, ensure_ascii=False))
print('  [light    2-3]:', json.dumps(g1, ensure_ascii=False))
print('  [busy    >=4 ]:', json.dumps(g2, ensure_ascii=False))

# Overall (sanity check against the paper's 4-week test setting)
all_k = list(stats.keys())
tot = group(0, None)
print('\n  [overall all]:', json.dumps(tot, ensure_ascii=False))

result = {
    'model': 'ampds2_week4_best.pt',
    'threshold_W': ON_THRESH,
    'fine': fine,
    'groups': {'isolated_0_1': g0, 'light_2_3': g1, 'busy_4plus': g2, 'overall': tot},
}
with open(os.path.join(OUT_DIR, 'concurrent_analysis.json'), 'w') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print('\nSaved -> concurrent_analysis.json')
