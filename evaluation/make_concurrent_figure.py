"""Generate the MAE vs. number of concurrent loads comparison figure (UniNILM / Mamba2NILM / Seq2Seq)."""
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(CODE, 'results')
os.makedirs(OUT_DIR, exist_ok=True)

# UniNILM fine-grained results (from concurrent_analysis.py output)
with open(os.path.join(OUT_DIR, 'concurrent_analysis.json')) as f:
    uni_data = json.load(f)
uninilm_fine = {int(k): v['mae'] for k, v in uni_data['fine'].items()}
with open(os.path.join(OUT_DIR, 'concurrent_baselines.json')) as f:
    base = json.load(f)
m2n_fine = {int(k): v['mae'] for k, v in base['Mamba2NILM']['fine'].items()}
seq_fine = {int(k): v['mae'] for k, v in base['Seq2Seq']['fine'].items()}

# bins with adequate support: 2-7 concurrent appliances
xs = [2, 3, 4, 5, 6, 7]
uni = [uninilm_fine[x] for x in xs]
m2n = [m2n_fine[x] for x in xs]
seq = [seq_fine[x] for x in xs]

# Number of timesteps for each concurrent count (from the UniNILM run)
steps = {int(k): v['n_steps'] for k, v in uni_data['fine'].items()}

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['legend.fontsize'] = 10.5
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11

fig, ax = plt.subplots(figsize=(6.2, 4.2))
ax.plot(xs, uni, 'o-', color='#1a9850', linewidth=2.2, markersize=7, label='UniNILM (ours)')
ax.plot(xs, m2n, 's--', color='#d73027', linewidth=1.8, markersize=6, label='Mamba2NILM')
ax.plot(xs, seq, '^:', color='#4575b4', linewidth=1.8, markersize=6, label='Seq2Seq')

# Annotate the number of samples at each point (optional, small font)
for x in xs:
    s = steps[x]
    ax.annotate(f'{s//1000:.0f}k' if s >= 1000 else str(s), (x, 2),
                textcoords='offset points', xytext=(0, -13), ha='center', fontsize=8, color='0.35')

ax.set_xlabel('Number of concurrently active appliances')
ax.set_ylabel('MAE (W)')
ax.set_xticks(xs)
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='upper left')
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'concurrent_analysis.pdf'))
fig.savefig(os.path.join(OUT_DIR, 'concurrent_analysis.png'), dpi=300)
print('saved concurrent_analysis.pdf / .png')

# Print the three-tier comparison (for the manuscript text)
print('\nLight/busy/overall MAE:')
for name, g in [('UniNILM', uni_data['groups']),
                ('Mamba2NILM', base['Mamba2NILM']['groups']),
                ('Seq2Seq', base['Seq2Seq']['groups'])]:
    print(f"  {name}: light={g['light_2_3']['mae']}, busy={g['busy_4plus']['mae']}, overall={g['overall']['mae']}")
