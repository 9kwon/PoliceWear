"""Figures 3-8 of the data descriptor and the numbers quoted in its text, drawn only from the release files
(../*.csv, ../ppg, ../acc) plus output/ppg_qc.csv from ppg_qc.py (run that first).
Run: python make_figures.py   (first run also caches per-recording wrist motion in output/acc_motion.csv, a few minutes)
Writes output/figures/Fig3..Fig8 (.png 400 dpi + .pdf) and output/summary.json."""
import pandas as pd, numpy as np, glob, os, sys, json
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
from scipy.stats import spearmanr, wilcoxon
sys.stdout.reconfigure(encoding='utf-8')
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__)); REL = _os.path.join(_os.path.dirname(_HERE), 'data', ''); EO = _os.path.join(_HERE, 'output', ''); FIG = _os.path.join(EO, 'figures', '')
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titlesize': 9, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left', 'legend.frameon': False})
BLUE, GREY, LIGHT = '#2b6cb0', '#8a8a8a', '#a9c4e2'
WCOL = {1: BLUE, 2: LIGHT}
S = {}  # numbers for the text

def save(fig, name):
    fig.savefig(FIG + name + '.png', dpi=400, bbox_inches='tight'); fig.savefig(FIG + name + '.pdf', bbox_inches='tight'); plt.close(fig)

def rd(f): return pd.read_csv(REL + f, encoding='utf-8-sig')
P, W, E, D, A = rd('participants.csv'), rd('watch_collections.csv'), rd('sleep_episodes.csv'), rd('sleep_daily.csv'), rd('activity_daily.csv')
mins = lambda day, t: (day - 1) * 1440 + t.str[:2].astype(int) * 60 + t.str[3:5].astype(int) + t.str[6:8].astype(int) / 60
W['hour'] = W['time_kst'].str[:2].astype(int); W['tmin'] = mins(W['day'], W['time_kst'])
Q = pd.read_csv(EO + 'ppg_qc.csv'); W = W.merge(Q[['pid', 'day', 'time_kst', 'sqi_med', 'good_frac', 'hr', 'hr_peak']], on=['pid', 'day', 'time_kst'], how='left')

# accelerometer motion per collection (std of |a| in g), cached
am_path = EO + 'acc_motion.csv'
if not os.path.exists(am_path):
    rows = []
    for f in sorted(glob.glob(REL + 'acc/wave*/*.csv.gz')):
        a = pd.read_csv(f, dtype={'pid': str, 'time_kst': str})
        a['mag'] = np.sqrt(a.acc_x ** 2 + a.acc_y ** 2 + a.acc_z ** 2) / 4096
        rows.append(a.groupby(['pid', 'day', 'time_kst'])['mag'].agg(motion_sd='std', mag_median='median').reset_index())
    pd.concat(rows).to_csv(am_path, index=False)
AM = pd.read_csv(am_path); W = W.merge(AM, on=['pid', 'day', 'time_kst'], how='left')

# sleep state of each collection (inside a Samsung Health sleep episode of the same participant)
E['s'] = mins(E['start_day'], E['start_time_kst']); E['e'] = mins(E['end_day'], E['end_time_kst'])
eps = {p: g[['s', 'e']].to_numpy() for p, g in E.groupby('pid')}
def asleep(r):
    ev = eps.get(r.pid)
    return np.nan if ev is None else float(((r.tmin >= ev[:, 0]) & (r.tmin < ev[:, 1])).any())
W['asleep'] = [asleep(r) for r in W[['pid', 'tmin']].itertuples()]
good = W['good_frac'] >= 0.5

# ---------- Fig 2: data availability ----------
fig = plt.figure(figsize=(7.2, 3.7)); gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.9], wspace=0.35)
for k, w in enumerate([1, 2]):
    ax = fig.add_subplot(gs[0, k]); x = W[W.wave == w].copy()
    x['dw'] = x.day - x.groupby('pid').day.transform('min') + 1  # day of wear: 1 = first watch recording of that participant
    order = x.groupby('pid').size().sort_values().index
    nd = int(x.dw.max()); M = np.zeros((len(order), nd))
    cnt = x.groupby(['pid', 'dw']).size()
    for i, p in enumerate(order):
        for d_, c in cnt.loc[p].items(): M[i, d_ - 1] = c
    im = ax.imshow(M, aspect='auto', cmap='Blues', vmin=0, vmax=24, interpolation='nearest', extent=[0.5, nd + 0.5, len(order), 0])
    ax.set_title(f'{"ab"[k]}  Wave {w}: collections per day'); ax.set_xlabel('Day of wear'); ax.set_ylabel('Participants' if k == 0 else '')
    ax.set_yticks([])
ax = fig.add_subplot(gs[0, 2])
items = [('Survey', P, None), ('PPG', W, None),
         ('Acc.', W[W.n_acc_samples.notna()], None), ('EMA', W[W.ema_stress.notna()], None), ('Sleep', E, None), ('Steps', A, None)]
y = np.arange(len(items))
for j, w in enumerate([1, 2]):
    v = [d[d.wave == w].pid.nunique() for _, d, _ in items]
    ax.barh(y + (j - 0.5) * 0.38, v, height=0.36, color=WCOL[w], label=f'Wave {w}')
    for yy, vv in zip(y, v): ax.text(vv + 3, yy + (j - 0.5) * 0.38, str(vv), va='center', fontsize=6.5, color='#333')
ax.set_yticks(y, [n for n, _, _ in items]); ax.invert_yaxis(); ax.set_xlabel('Participants'); ax.set_title('c  Participants per data type')
ax.set_xlim(0, 200)
bb, ba = ax.get_position(), fig.axes[0].get_position(); ax.set_position([bb.x0, ba.y0, bb.width, ba.height])  # same height as the heatmaps; no colour bar (scale given in the legend)
ax.set_ylim(len(items) - 0.4, -1.45)  # inverted axis; the strip above the first bars holds the legend
ax.legend(loc='upper left', ncol=2, fontsize=7, handlelength=1.4, columnspacing=1.2, borderaxespad=0.1)
save(fig, 'Fig3_data_availability')

# ---------- Fig 3: collection timing and EMA ----------
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.3), gridspec_kw={'wspace': 0.4})
for w in [1, 2]:
    x = W[W.wave == w]; npid = x.pid.nunique()
    for ct, ls in [('automatic', '-'), ('manual', '--')]:
        h = x[x.collection_type == ct].groupby('hour').size().reindex(range(24), fill_value=0) / npid / x.groupby('pid').day.nunique().median()
        axs[0].plot(h.index, h.values, ls, color=WCOL[w], lw=1.2, label=f'W{w} {ct}')
    r = x[x.collection_type == 'automatic'].groupby('hour').ema_stress.apply(lambda s: s.notna().mean())
    axs[1].plot(r.index, r.values * 100, color=WCOL[w], lw=1.2, label=f'Wave {w}')
axs[0].set_title('a  Collections by hour'); axs[0].set_xlabel('Hour of day (KST)'); axs[0].set_ylabel('Per participant-day'); axs[0].legend(fontsize=5.8, loc='center right', bbox_to_anchor=(1.04, 0.5), handlelength=1.4, borderaxespad=0, labelspacing=0.3)
axs[1].set_title('b  EMA response, automatic'); axs[1].set_xlabel('Hour of day (KST)'); axs[1].set_ylabel('Answered (%)'); axs[1].legend(fontsize=6)
for a in axs[:2]: a.set_xticks([0, 6, 12, 18, 23])
e = W[W.ema_stress.notna()]
tab = pd.crosstab([e.wave, e.collection_type], e.ema_stress, normalize='index') * 100
lab = [f'W{w}\n{c[:4]}.' for w, c in tab.index]; left = np.zeros(len(tab))
shades = [0.2, 0.45, 0.7, 1.0]
for k, s in enumerate(tab.columns):
    axs[2].barh(range(len(tab)), tab[s], left=left, color=BLUE, alpha=shades[k], label=str(int(s))); left += tab[s].values
axs[2].set_yticks(range(len(tab)), lab); axs[2].invert_yaxis(); axs[2].set_xlabel('Share of answers (%)'); axs[2].set_title('c  Momentary stress (0–3)')
axs[2].legend(ncol=4, fontsize=6, loc='upper center', bbox_to_anchor=(0.5, -0.18), handlelength=1)
save(fig, 'Fig4_collection_EMA')
for w in [1, 2]:
    x = W[W.wave == w]
    S[f'w{w}_ema_rate_auto'] = round(x[x.collection_type == 'automatic'].ema_stress.notna().mean() * 100, 1)
    S[f'w{w}_ema_rate_manual'] = round(x[x.collection_type == 'manual'].ema_stress.notna().mean() * 100, 1)

# ---------- Fig 4: signal quality ----------
fig = plt.figure(figsize=(7.2, 4.6)); gs = fig.add_gridspec(2, 3, hspace=0.6, wspace=0.4)
sos = butter(3, [0.5, 4], btype='band', fs=25, output='sos')
x2 = W[(W.wave == 2) & W.motion_sd.notna() & W.good_frac.notna()]
ex = [x2[(x2.good_frac == 1) & (x2.motion_sd < x2.motion_sd.quantile(0.2))].sample(1, random_state=3).iloc[0],
      x2[(x2.good_frac <= 0.25) & (x2.motion_sd > x2.motion_sd.quantile(0.8))].sample(1, random_state=3).iloc[0]]
for k, r in enumerate(ex):
    ax = fig.add_subplot(gs[0, k]); pp = pd.read_csv(f"{REL}ppg/wave2/{r.pid}.csv.gz"); aa = pd.read_csv(f"{REL}acc/wave2/{r.pid}.csv.gz")
    pp = pp[(pp.day == r.day) & (pp.time_kst == r.time_kst)]; aa = aa[(aa.day == r.day) & (aa.time_kst == r.time_kst)]
    t0 = pp.t_s.iloc[0] + 20; sel = (pp.t_s >= t0) & (pp.t_s < t0 + 15)
    yv = sosfiltfilt(sos, pp.ppg.values - pp.ppg.mean()); yv = yv / np.std(yv[sel.values])
    ax.plot(pp.t_s[sel] - t0, yv[sel.values], color=BLUE if k == 0 else GREY, lw=0.8)
    mg = np.sqrt(aa.acc_x ** 2 + aa.acc_y ** 2 + aa.acc_z ** 2) / 4096; sa = (aa.t_s >= t0) & (aa.t_s < t0 + 15)
    ax.plot(aa.t_s[sa] - t0, (mg[sa] - mg[sa].mean()) * 8 - 4.5, color='#333', lw=0.6)
    ax.set_yticks([]); ax.set_xlabel('Time (s)'); ax.spines['left'].set_visible(False)
    ax.set_title(f'{"ab"[k]}  {"Low" if k == 0 else "High"} motion', pad=14); ax.text(0, ax.get_ylim()[1], f'{r.good_frac:.0%} good windows', fontsize=6.5, color='#333', va='bottom')
    tp = (pp.t_s[sel] - t0).to_numpy(); xe = float(tp[-1]); av = ((mg[sa] - mg[sa].mean()) * 8 - 4.5).to_numpy(); ta = (aa.t_s[sa] - t0).to_numpy()
    ax.set_ylim(float(av.min()) - 1.4, None)  # room for the label under the accelerometer trace
    dy = 0.15 if k == 0 else 0.35  # under the last 2 s of each trace; a little lower in the high-motion panel
    ax.text(xe, float(yv[sel.values][tp > xe - 2].min()) - dy, 'PPG', fontsize=6.5, ha='right', va='top')
    ax.text(xe, float(av[ta > xe - 2].min()) - dy, 'Accel. |a|', fontsize=6.5, ha='right', va='top')
ax = fig.add_subplot(gs[0, 2])
for w in [1, 2]:
    v = W[(W.wave == w) & W.good_frac.notna()].good_frac
    ax.hist(v, bins=np.linspace(0, 1, 10), color=WCOL[w], alpha=0.9 if w == 1 else 0.8, label=f'Wave {w}', density=True, histtype='stepfilled' if w == 1 else 'step', lw=1.4)
ax.set_xlabel('Good 8 s windows per collection'); ax.set_ylabel('Density'); ax.set_title('c  PPG quality', pad=14); ax.legend(fontsize=6.5, loc='upper center')
ax = fig.add_subplot(gs[1, 0:2])
xa = W[W.motion_sd.notna() & W.good_frac.notna()]
for w in [1, 2]:
    xw = xa[xa.wave == w]; b = pd.qcut(xw.motion_sd, 10)
    g = xw.groupby(b, observed=True).good_frac.agg(['median', lambda s: s.quantile(.25), lambda s: s.quantile(.75)])
    mid = xw.groupby(b, observed=True).motion_sd.median()
    ax.errorbar(mid.values * 1000, g['median'], yerr=[g['median'] - g.iloc[:, 1], g.iloc[:, 2] - g['median']], fmt='o-', color=WCOL[w], ms=3, lw=1,
                capsize=2, label=f'Wave {w}')
    S[f'w{w}_rho_motion_quality'] = round(spearmanr(xw.motion_sd, xw.good_frac).correlation, 2)
ax.set_xscale('log'); ax.set_xlabel('Wrist motion: SD of |acceleration| (mg), deciles'); ax.set_ylabel('Good windows (median, IQR)')
ax.set_title('d  PPG quality falls with wrist motion'); ax.legend(fontsize=6.5)
S['rho_motion_quality'] = round(spearmanr(xa.motion_sd, xa.good_frac).correlation, 2)
ax = fig.add_subplot(gs[1, 2]); h2 = W[good & W.hr_peak.notna()]
ax.hexbin(h2.hr, h2.hr_peak, gridsize=40, cmap='Blues', mincnt=1, bins='log', extent=(40, 140, 40, 140))
ax.plot([40, 140], [40, 140], color=GREY, lw=0.8, ls='--'); ax.set_xlabel('HR, autocorrelation (bpm)'); ax.set_ylabel('HR, pulse peaks (bpm)')
ax.set_title('e  Two HR estimates')
S['hr_agree_within5'] = round(((h2.hr - h2.hr_peak).abs() <= 5).mean() * 100, 1)
save(fig, 'Fig5_signal_quality')
for w in [1, 2]:
    v = W[(W.wave == w) & W.good_frac.notna()]
    S[f'w{w}_goodfrac_median'] = round(v.good_frac.median(), 3); S[f'w{w}_share_any_good'] = round((v.good_frac > 0).mean() * 100, 1)
    S[f'w{w}_share_half_good'] = round((v.good_frac >= 0.5).mean() * 100, 1)

# ---------- Fig 5: physiological plausibility ----------
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4), gridspec_kw={'wspace': 0.45, 'width_ratios': [1.4, 1, 1]})
au = W[good & (W.collection_type == 'automatic')]
for w in [1, 2]:
    g = au[au.wave == w].groupby('hour').hr; m, lo, hi = g.median(), g.quantile(.25), g.quantile(.75)
    axs[0].plot(m.index, m.values, color=WCOL[w], lw=1.4, label=f'Wave {w}'); axs[0].fill_between(m.index, lo, hi, color=WCOL[w], alpha=0.15, lw=0)
    S[f'w{w}_hr_min_hour'] = int(m.idxmin())
axs[0].set_xticks([0, 6, 12, 18, 23]); axs[0].set_xlabel('Hour of day (KST)'); axs[0].set_ylabel('PPG heart rate (bpm)'); axs[0].legend(fontsize=6.5)
axs[0].set_title('a  Day–night heart-rate rhythm')
pp = au[au.asleep.notna()].groupby(['pid', 'asleep']).hr.median().unstack().dropna()
pp = pp[(au[au.asleep == 1].groupby('pid').size().reindex(pp.index) >= 3)]
for _, r in pp.iterrows(): axs[1].plot([0, 1], [r[0.0], r[1.0]], color=GREY, lw=0.3, alpha=0.5)
axs[1].plot([0, 1], [pp[0.0].median(), pp[1.0].median()], color=BLUE, lw=2, marker='o', ms=4)
axs[1].set_xticks([0, 1], ['Awake', 'Asleep']); axs[1].set_xlim(-0.3, 1.3); axs[1].set_ylabel('Median HR (bpm)')
axs[1].set_title('b  HR in Samsung sleep')
S['n_sleep_hr_pairs'] = len(pp); S['hr_awake_med'] = round(pp[0.0].median(), 1); S['hr_asleep_med'] = round(pp[1.0].median(), 1)
S['share_lower_asleep'] = round((pp[1.0] < pp[0.0]).mean() * 100, 1); S['wilcoxon_p'] = float(wilcoxon(pp[0.0], pp[1.0]).pvalue)
mo = W[W.motion_sd.notna() & W.asleep.notna()]
for k, (st, lab) in enumerate([(0.0, 'Awake'), (1.0, 'Asleep')]):
    for w in [1, 2]:
        v = mo[(mo.asleep == st) & (mo.wave == w)].motion_sd * 1000
        axs[2].boxplot(v, positions=[k + (w - 1.5) * 0.36], widths=0.3, showfliers=False, patch_artist=True,
                       boxprops=dict(facecolor=WCOL[w], alpha=0.6, color='#333'), medianprops=dict(color='#333'))
axs[2].set_yscale('log'); axs[2].set_xticks([0, 1], ['Awake', 'Asleep']); axs[2].set_ylabel('Wrist motion (mg)'); axs[2].set_title('c  Motion in sleep')
S['motion_awake_med_mg'] = round(mo[mo.asleep == 0].motion_sd.median() * 1000, 1); S['motion_asleep_med_mg'] = round(mo[mo.asleep == 1].motion_sd.median() * 1000, 1)
save(fig, 'Fig6_physiology')

# ---------- Fig 6: questionnaires and EMA ----------
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.5), gridspec_kw={'wspace': 0.4, 'width_ratios': [1.6, 1]})
sc = [('phq9', 'PHQ-9'), ('gad7', 'GAD-7'), ('pss', 'PSS-10'), ('fatigue', 'FSS')]
grp = ['healthy', 'stress', 'depression']; gcol = {'healthy': '#cfcfcf', 'stress': LIGHT, 'depression': BLUE}
for i, (c, lab) in enumerate(sc):
    for j, gname in enumerate(grp):
        v = P[P.group == gname][c].dropna(); mx = {'PHQ-9': 27, 'GAD-7': 21, 'PSS-10': 40, 'FSS': 63}[lab]
        axs[0].boxplot(v / mx * 100, positions=[i * 4 + j], widths=0.8, showfliers=False, patch_artist=True,
                       boxprops=dict(facecolor=gcol[gname], color='#333'), medianprops=dict(color='#333'))
axs[0].set_xticks([1, 5, 9, 13], [l for _, l in sc]); axs[0].set_ylabel('Score (% of maximum)'); axs[0].set_title('a  Self-report scores by group')
axs[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=gcol[g], ec='#333') for g in grp], labels=grp, fontsize=6.5, loc='upper center', ncol=3, handlelength=1, columnspacing=0.8); axs[0].set_ylim(-3, 118)
pm = W[W.ema_stress.notna()].groupby('pid').ema_stress.agg(['mean', 'size']); pm = pm[pm['size'] >= 10].join(P.set_index('pid')[['pss', 'wave']])
for w in [1, 2]:
    q = pm[pm.wave == w]; axs[1].scatter(q.pss, q['mean'], s=6, color=WCOL[w], label=f'Wave {w}', alpha=0.8, lw=0)
r = spearmanr(pm.pss, pm['mean'], nan_policy='omit'); S['rho_ema_pss'] = round(r.correlation, 2); S['p_ema_pss'] = float(r.pvalue); S['n_ema_pss'] = len(pm)
axs[1].set_xlabel('PSS-10'); axs[1].set_ylabel('Mean momentary stress'); axs[1].set_title(f'b  EMA vs PSS (ρ = {r.correlation:.2f})'); axs[1].legend(fontsize=6)
save(fig, 'Fig7_questionnaires_EMA')
for c, lab in sc:
    S[f'{lab}_median_by_group'] = P.groupby('group')[c].median().to_dict()

# ---------- Fig 7: sleep and steps ----------
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.3), gridspec_kw={'wspace': 0.4})
DD = D.merge(P[['pid', 'work_type']], on='pid'); wt = ['day', 'shift', 'duty']
for j, t in enumerate(wt):
    v = DD[DD.work_type == t].total_sleep_min / 60
    axs[0].boxplot(v, positions=[j], widths=0.6, showfliers=False, patch_artist=True, boxprops=dict(facecolor=BLUE if t == 'shift' else 'white', alpha=0.6, color='#333'), medianprops=dict(color='#333'))
axs[0].set_xticks(range(3), wt); axs[0].set_ylabel('Sleep per wake day (h)'); axs[0].set_title('a  Total sleep by work type')
on = DD.main_start_time_kst.str[:2].astype(int) + DD.main_start_time_kst.str[3:5].astype(int) / 60
for t in wt:
    axs[1].hist(on[DD.work_type == t], bins=np.arange(0, 24.5, 1), density=True, histtype='step', lw=1.3, color=BLUE if t == 'shift' else (GREY if t == 'day' else '#444'), label=t)
axs[1].set_xticks([0, 6, 12, 18, 24]); axs[1].set_xlabel('Main sleep onset (KST)'); axs[1].set_ylabel('Density'); axs[1].set_title('b  Sleep onset'); axs[1].legend(fontsize=6.5)
for w in [1, 2]:
    v = A[(A.wave == w) & A.steps_complete].steps
    axs[2].hist(v / 1000, bins=np.arange(0, 30, 1), density=True, histtype='stepfilled' if w == 1 else 'step', color=WCOL[w], lw=1.4, label=f'Wave {w}')
axs[2].set_xlabel('Daily steps (thousands)'); axs[2].set_ylabel('Density'); axs[2].set_title('c  Daily steps'); axs[2].legend(fontsize=6.5)
save(fig, 'Fig8_sleep_steps')
S['sleep_h_median_by_work'] = (DD.groupby('work_type').total_sleep_min.median() / 60).round(2).to_dict()
S['steps_median_by_wave'] = A[A.steps_complete].groupby('wave').steps.median().to_dict()
S['acc_mag_median_g'] = round(AM.mag_median.median(), 3)
json.dump(S, open(EO + 'summary.json', 'w'), indent=1, default=float)
print(json.dumps(S, indent=1, default=float, ensure_ascii=False))
