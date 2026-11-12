"""Figure 2 (participants): sex, age band, work type and BMI of the released sample by wave, from release/participants.csv only.
Same style as make_figures.py. Run: python make_fig_participants.py   Writes output/figures/Fig2_participants.png (400 dpi) and .pdf."""
import pandas as pd, numpy as np, sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.stdout.reconfigure(encoding='utf-8')
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__)); BASE = _os.path.join(_os.path.dirname(_HERE), 'data', ''); FIG = _os.path.join(_HERE, 'output', 'figures', '')
plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titlesize': 9, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left', 'legend.frameon': False})
BLUE, LIGHT = '#2b6cb0', '#a9c4e2'; WCOL = {1: BLUE, 2: LIGHT}
_os.makedirs(FIG, exist_ok=True)
P = pd.read_csv(BASE + "participants.csv", encoding='utf-8-sig')
fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.1), gridspec_kw={'width_ratios': [1, 1.2, 1.2, 1.5]})
def bars(ax, col, order, title):
    x = np.arange(len(order)); w = 0.38
    for k, wv in enumerate((1, 2)):
        n = P[P.wave == wv][col].value_counts().reindex(order).fillna(0).astype(int)
        b = ax.bar(x + (k - 0.5) * w, n.values, w, color=WCOL[wv], label=f'Wave {wv}')
        for xi, v in zip(x + (k - 0.5) * w, n.values): ax.text(xi, v + 1.5, str(v), ha='center', va='bottom', fontsize=6.5, color='#444')
    ax.set_xticks(x); ax.set_xticklabels(order); ax.set_title(title, pad=10); ax.set_ylim(0, P.groupby('wave')[col].value_counts().max() * 1.18)
bars(axs[0], 'sex', ['male', 'female'], 'a  Sex'); axs[0].set_ylabel('Participants'); axs[0].set_xlabel('Sex')
bars(axs[1], 'age_group', ['30-39', '40-49', '50+'], 'b  Age band'); axs[1].set_xlabel('Age band (years)')
bars(axs[2], 'work_type', ['day', 'shift', 'duty'], 'c  Work type'); axs[2].set_xlabel('Work type')
ax = axs[3]; edges = np.arange(17, 38, 1.0)
for wv in (1, 2):
    v = P.loc[P.wave == wv, 'bmi'].dropna()
    ax.hist(v, bins=edges, color=WCOL[wv], alpha=0.85 if wv == 1 else 0.9, label=f'Wave {wv}', histtype='step' if wv == 1 else 'bar', lw=1.3 if wv == 1 else 0)
    print(f"wave {wv}: bmi n={len(v)} median={v.median():.1f} IQR={v.quantile(.25):.1f}-{v.quantile(.75):.1f} >=25: {(v >= 25).mean() * 100:.0f}%")
ax.set_ylim(0, 30); ax.axvline(25, color='#bbb', lw=0.6, ls=(0, (3, 2)))
ax.set_xlabel('BMI (kg/m²)'); ax.set_ylabel('Participants'); ax.set_title('d  Body-mass index', pad=10); ax.legend(fontsize=6, loc='upper right', bbox_to_anchor=(1.04, 1.0), handlelength=1.0, handletextpad=0.5, borderaxespad=0.2)
axs[1].set_ylim(0, 100); axs[0].legend(fontsize=6.5, loc='upper right', bbox_to_anchor=(1.04, 0.98), handlelength=1.0, handletextpad=0.5, borderaxespad=0)
fig.tight_layout(w_pad=1.2)
fig.savefig(FIG + 'Fig2_participants.png', dpi=400, bbox_inches='tight'); fig.savefig(FIG + 'Fig2_participants.pdf', bbox_inches='tight'); print('saved Fig2_participants')
print("female share released:", (P.sex == 'female').mean().round(3), P.groupby('wave')['sex'].apply(lambda v: (v == 'female').mean()).round(3).to_dict())
