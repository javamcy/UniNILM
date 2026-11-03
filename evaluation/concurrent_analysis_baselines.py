"""Mamba2NILM / Seq2Seq error grouped by the number of concurrent loads
(AMPds2 week4, same windows as UniNILM).
Seq2Seq is per-appliance (8 models, app0-app7); Mamba2NILM is multi-output.
"""
import sys, os, json, warnings
warnings.filterwarnings('ignore')

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE)

import numpy as np
import torch
from baseline_experiments_v3 import Mamba2NILM, Seq2Seq, NILMDataset, SEQ_LENGTH, APPLIANCE_NAMES

DATA_PATH = os.environ.get('AMPDS2_PATH', os.path.join(CODE, 'data', 'ampds2_aligned.csv'))
OUT_DIR = os.path.join(CODE, 'results')
os.makedirs(OUT_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRAIN_SIZE, VAL_SIZE = 40320, 40320
ON_THRESH = 10.0
BATCH = 64
TMODEL = 'trained_models/ampds2_4w_Mamba2NILM.pt'
TSEQ = 'trained_models/ampds2_4w_Seq2Seq_app'

print(f'Device: {DEVICE}', flush=True)
ds = NILMDataset(DATA_PATH, seq_length=SEQ_LENGTH, split='test',
                 train_size=TRAIN_SIZE, val_size=VAL_SIZE, appliance_columns=APPLIANCE_NAMES)
sel = np.arange(0, len(ds), SEQ_LENGTH)
print(f'windows: {len(sel)}', flush=True)

def run_model(model, app_cols, name=''):
    """app_cols: None = all 8 columns; list = only count these columns (per-appliance)."""
    model.eval()
    stats = {}
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
            sp = torch.stack(poss).squeeze(-1).to(DEVICE)
            out, _, _, _ = model(total_input, total_raw, sp)
            pred = out.float().cpu().numpy()
            for b in range(pred.shape[0]):
                p, t = pred[b], target_raw[b]
                n_on = (t > ON_THRESH).sum(axis=1)
                cols = slice(None) if app_cols is None else app_cols[b % len(app_cols)]
                for step in range(p.shape[0]):
                    if app_cols is not None:
                        k_col = app_cols[0]
                        pk, tk = p[step, 0], t[step, k_col]  # per-appliance: output column 0 corresponds to appliance k_col
                        valid = ~np.isnan(tk)
                        if not valid:
                            continue
                        diff = np.abs(pk - tk)
                        st = stats.setdefault(int(n_on[step]), {'sum_abs': 0.0, 'sum_sq': 0.0, 'n_steps': 0, 'n_valid': 0, 'sum_on': 0.0})
                        st['sum_abs'] += float(diff)
                        st['sum_sq'] += float(diff ** 2)
                        st['n_steps'] += 1
                        st['n_valid'] += 1
                        st['sum_on'] += float(n_on[step])
                    else:
                        valid = ~np.isnan(t[step])
                        if not valid.any():
                            continue
                        diff = np.abs(p[step][valid] - t[step][valid])
                        st = stats.setdefault(int(n_on[step]), {'sum_abs': 0.0, 'sum_sq': 0.0, 'n_steps': 0, 'n_valid': 0, 'sum_on': 0.0})
                        st['sum_abs'] += float(diff.sum())
                        st['sum_sq'] += float((diff ** 2).sum())
                        st['n_steps'] += 1
                        st['n_valid'] += int(valid.sum())
                        st['sum_on'] += float(n_on[step])
    return stats

def summarize(stats, label):
    print(f'\n########## {label} ##########')
    def finalize(s):
        n = s['n_valid']
        return (s['sum_abs'] / n, (s['sum_sq'] / n) ** 0.5, s['n_steps'], n) if n else (float('nan'), float('nan'), s['n_steps'], n)
    fine = {}
    for k in sorted(stats):
        mae, rmse, steps, nv = finalize(stats[k])
        fine[k] = {'mae': round(mae, 3), 'rmse': round(rmse, 3), 'n_steps': steps, 'n_valid': nv}
    def grp(lo, hi):
        ks = [k for k in stats if lo <= k <= hi] if hi is not None else [k for k in stats if k >= lo]
        sa = ss = nv = ns = 0.0
        for k in ks:
            sa += stats[k]['sum_abs']; ss += stats[k]['sum_sq']; nv += stats[k]['n_valid']; ns += stats[k]['n_steps']
        return {'mae': round(sa / nv, 3) if nv else None, 'rmse': round((ss / nv) ** 0.5, 3) if nv else None,
                'n_steps': int(ns), 'n_valid': int(nv)}
    g = {'isolated_0_1': grp(0, 1), 'light_2_3': grp(2, 3), 'busy_4plus': grp(4, None), 'overall': grp(0, None)}
    print('fine:', json.dumps(fine))
    print('groups:', json.dumps(g))
    return {'fine': fine, 'groups': g}

results = {}

# 1) Mamba2NILM (multi-output)
m2n = Mamba2NILM(input_dim=1, n_appliances=8).to(DEVICE)
m2n.load_state_dict(torch.load(os.path.join(CODE, TMODEL), map_location='cpu', weights_only=False))
results['Mamba2NILM'] = summarize(run_model(m2n, None, 'Mamba2NILM'), 'Mamba2NILM')

# 2) Seq2Seq (per-appliance, 8 models)
seq_stats = {}
for k in range(8):
    m = Seq2Seq(input_dim=1, n_appliances=1).to(DEVICE)
    m.load_state_dict(torch.load(os.path.join(CODE, f'{TSEQ}{k}.pt'), map_location='cpu', weights_only=False))
    s = run_model(m, [k], f'Seq2Seq_app{k}')
    for kk, v in s.items():
        t = seq_stats.setdefault(kk, {'sum_abs': 0.0, 'sum_sq': 0.0, 'n_steps': 0, 'n_valid': 0, 'sum_on': 0.0})
        t['sum_abs'] += v['sum_abs']; t['sum_sq'] += v['sum_sq']
        t['n_steps'] += v['n_steps']; t['n_valid'] += v['n_valid']; t['sum_on'] += v['sum_on']
    print(f'  Seq2Seq app{k} done', flush=True)
results['Seq2Seq'] = summarize(seq_stats, 'Seq2Seq (per-appliance)')

with open(os.path.join(OUT_DIR, 'concurrent_baselines.json'), 'w') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print('\nSaved -> concurrent_baselines.json')
