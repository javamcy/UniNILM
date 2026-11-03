# -*- coding: utf-8 -*-
"""Event-level ATP/ITP on the test set under calibrated thresholds. UniNILM vs Mamba2NILM vs Seq2Seq."""
import sys, os, json, warnings
warnings.filterwarnings('ignore')
CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE)
import numpy as np, torch
from baseline_experiments_v3 import BiMamba, Mamba2NILM, Seq2Seq, NILMDataset, SEQ_LENGTH, APPLIANCE_NAMES

DATA = os.environ.get('AMPDS2_PATH', os.path.join(CODE, 'data', 'ampds2_aligned.csv'))
OUT_DIR = os.path.join(CODE, 'results')
os.makedirs(OUT_DIR, exist_ok=True)
DEV = torch.device('cuda'); TRAIN = VAL = 40320; MIN_EV = 5

ds = NILMDataset(DATA, seq_length=SEQ_LENGTH, split='test', train_size=TRAIN, val_size=VAL, appliance_columns=APPLIANCE_NAMES)
sel = np.arange(0, len(ds), SEQ_LENGTH)

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

def events(binary):
    out, s = [], None
    for i, v in enumerate(binary):
        if v and s is None: s=i
        elif not v and s is not None:
            if i-s >= MIN_EV: out.append((s,i))
            s=None
    if s is not None and len(binary)-s >= MIN_EV: out.append((s,len(binary)))
    return out

def ov(a,b): return max(0, min(a[1],b[1])-max(a[0],b[0]))

def atp_itp_for_app(model, k, th):
    atp=itp=miss=fpev=0
    for pred, tgt in iter_preds(model):
        for b in range(pred.shape[0]):
            p = pred[b,:,0] if pred.shape[2]==1 else pred[b,:,k]
            t = tgt[b,:,k]
            valid = ~np.isnan(t)
            p_on = (p>th)&valid; t_on = (t>th)&valid
            g_ev = events(t_on.astype(int)); p_ev = events(p_on.astype(int))
            used=set()
            for g in g_ev:
                best, best_ov = None, 0
                for pi,pe in enumerate(p_ev):
                    if pi in used: continue
                    o=ov(g,pe)
                    if o>best_ov: best,best_ov=pi,o
                if best is not None and best_ov>0:
                    ratio = best_ov/max(g[1]-g[0], p_ev[best][1]-p_ev[best][0])
                    used.add(best)
                    if ratio>=0.5: atp+=1
                    else: itp+=1
                else: miss+=1
            fpev += len(p_ev)-len(used)
    return atp,itp,miss,fpev

def run_all(model, ths, apps=None):
    apps = apps or APPLIANCE_NAMES
    per={}
    for k,a in enumerate(APPLIANCE_NAMES):
        if a not in apps: continue
        atp,itp,miss,fpev = atp_itp_for_app(model, k, ths[a])
        per[a]={'atp':atp,'itp':itp,'miss':miss,'fp_ev':fpev}
    return per

ths = json.load(open(os.path.join(OUT_DIR, 'onoff_calibrated.json')))
th_uni = {a: ths['UniNILM']['per'][a]['th'] for a in APPLIANCE_NAMES}
th_m2n = {a: ths['Mamba2NILM']['per'][a]['th'] for a in APPLIANCE_NAMES}
th_seq = {a: ths['Seq2Seq']['per'][a]['th'] for a in APPLIANCE_NAMES}

res={}
m = BiMamba(input_dim=1, n_appliances=8, d_model=64).to(DEV)
m.load_state_dict(torch.load(os.path.join(CODE,'trained_models/bimamba_daygate_ampds2/ampds2_week4_best.pt'), map_location=DEV, weights_only=False)); m.eval()
res['UniNILM']=run_all(m,th_uni); print('UniNILM done', flush=True)

m = Mamba2NILM(input_dim=1, n_appliances=8).to(DEV)
m.load_state_dict(torch.load(os.path.join(CODE,'trained_models/ampds2_4w_Mamba2NILM.pt'), map_location=DEV, weights_only=False)); m.eval()
res['Mamba2NILM']=run_all(m,th_m2n); print('Mamba2NILM done', flush=True)

seq={}
for k in range(8):
    a = APPLIANCE_NAMES[k]
    mm = Seq2Seq(input_dim=1, n_appliances=1).to(DEV)
    mm.load_state_dict(torch.load(os.path.join(CODE,f'trained_models/ampds2_4w_Seq2Seq_app{k}.pt'), map_location=DEV, weights_only=False)); mm.eval()
    seq[a] = run_all(mm, {a: th_seq[a]}, apps=[a])[a]
res['Seq2Seq']=seq; print('Seq2Seq done', flush=True)

json.dump(res, open(os.path.join(OUT_DIR, 'atp_itp_calibrated.json'), 'w'), ensure_ascii=False, indent=2)
for name in ['UniNILM','Mamba2NILM','Seq2Seq']:
    atp=sum(res[name][a]['atp'] for a in APPLIANCE_NAMES if res[name][a])
    itp=sum(res[name][a]['itp'] for a in APPLIANCE_NAMES if res[name][a])
    miss=sum(res[name][a]['miss'] for a in APPLIANCE_NAMES if res[name][a])
    fpev=sum(res[name][a]['fp_ev'] for a in APPLIANCE_NAMES if res[name][a])
    print(f'{name}: ATP={atp} ITP={itp} Miss={miss} FP_ev={fpev} ATPrate={atp/(atp+itp+miss):.3f}', flush=True)
print('Saved', flush=True)
