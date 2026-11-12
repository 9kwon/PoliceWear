"""Consistency and de-identification checks on the released files (../data/). Every check raises on failure.
Run: python check_release.py   (reads every PPG and accelerometer file; a few minutes). Writes output/check_release.txt.
The same checks were run on the release before publication; the internal version additionally compared the files against
the identifier lookup table, which is not released."""
import pandas as pd, numpy as np, os, re, glob, sys
sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__)); REL = os.path.join(os.path.dirname(HERE), 'data', ''); OUT = os.path.join(HERE, 'output', '')
os.makedirs(OUT, exist_ok=True)
QI = ['wave', 'sex', 'age_group', 'work_type']   # quasi-identifiers a colleague could know
K_MIN = 5
log = []
def ok(msg): log.append("PASS " + msg); print("PASS", msg)

EXPECT = {'participants.csv': ['pid', 'wave', 'sex', 'age_group', 'bmi', 'group', 'work_type', 'O_sum', 'C_sum', 'E_sum', 'A_sum', 'N_sum', 'brs_score',
                               'pss', 'phq9', 'gad7', 'fatigue'],
          'watch_collections.csv': ['pid', 'wave', 'day', 'time_kst', 'collection_type', 'on_shift', 'ema_stress', 'n_ppg_samples', 'n_acc_samples'],
          'sleep_episodes.csv': ['pid', 'wave', 'start_day', 'start_time_kst', 'end_day', 'end_time_kst', 'duration_min', 'source'],
          'sleep_daily.csv': ['pid', 'wave', 'day', 'total_sleep_min', 'main_sleep_min', 'n_episodes', 'main_start_day', 'main_start_time_kst',
                              'main_end_day', 'main_end_time_kst'],
          'activity_daily.csv': ['pid', 'wave', 'day', 'steps', 'steps_complete', 'source']}
PPG_COLS = ['pid', 'day', 'time_kst', 't_s', 'ppg']
ACC_COLS = ['pid', 'day', 'time_kst', 't_s', 'acc_x', 'acc_y', 'acc_z']

# 1. file inventory: the five tables, PPG and accelerometer files named by pid only
top = sorted(f for f in os.listdir(REL) if os.path.isfile(REL + f) and f.endswith('.csv'))
assert top == sorted(EXPECT), top
ppg = sorted(glob.glob(REL + 'ppg/wave*/*.csv.gz')); acc = sorted(glob.glob(REL + 'acc/wave*/*.csv.gz'))
assert all(re.fullmatch(r'P\d{4}\.csv\.gz', os.path.basename(f)) for f in ppg + acc)
ok(f"file inventory: {len(top)} tables + {len(ppg)} PPG files + {len(acc)} accelerometer files, file names carry pid only")

# 2. exact column whitelist
T = {f: pd.read_csv(REL + f, encoding='utf-8-sig', dtype=str) for f in EXPECT}
for f, cols in EXPECT.items(): assert list(T[f].columns) == cols, (f, list(T[f].columns))
ok("column whitelist matches for every table")

# 3. no direct identifier, calendar date, year or free text in any table cell
pats = {'internal id': r'prt\d', 'phone': r'01[016789]-?\d{3,4}-?\d{4}', 'uuid': r'[0-9a-f]{8}-[0-9a-f]{4}-', 'email': r'@',
        'calendar date': r'(?:19|20)\d\d[-./](?:0?[1-9]|1[0-2])', 'year': r'\b20[12]\d\b', 'hangul': r'[가-힣]'}
for f, d in T.items():
    vals = pd.Series(pd.unique(d.values.ravel())).dropna().astype(str)
    text = vals[~vals.str.fullmatch(r'-?\d+(\.\d+)?')]  # numbers such as 2021 steps are not years
    for k, p in pats.items():
        hit = text[text.str.contains(p, regex=True)]
        assert hit.empty, (f, k, hit.head().tolist())
ok("no internal id, phone, UUID, e-mail, calendar date, year or Korean free text in any table cell")

# 4. IDs: one wave per pid across files, every pid has a survey row
pw = pd.concat([d[['pid', 'wave']] for d in T.values()]).drop_duplicates()
assert pw['pid'].is_unique, "pid with two waves"
assert set(pw['pid']) == set(T['participants.csv']['pid']), "pid without survey"
ok(f"{len(pw)} pids, each in one wave ({pw['wave'].value_counts().sort_index().to_dict()}), all present in participants.csv")

# 5. k-anonymity of what a colleague could know, and l-diversity of the group label
p = T['participants.csv']; k = p.groupby(QI).size()
assert k.min() >= K_MIN
ok(f"k-anonymity on {'+'.join(QI)}: min k = {k.min()}, classes = {len(k)}")
l = p.groupby(QI)['group'].nunique()
assert l.min() >= 2, "a class reveals everyone's recruitment group"
ok(f"l-diversity of group within those classes: min l = {l.min()}")

# 6. time format: positive integer days, HH:MM:SS clock
for f, d in T.items():
    for c in [c for c in d if c == 'day' or c.endswith('_day')]:
        v = pd.to_numeric(d[c]); assert v.notna().all() and (v >= 1).all() and (v % 1 == 0).all(), (f, c)
    for c in [c for c in d if c.endswith('time_kst')]:
        assert d[c].dropna().str.fullmatch(r'([01]\d|2[0-3]):[0-5]\d:[0-5]\d').all(), (f, c)
ok("all day columns are integers >= 1 and all clock columns are HH:MM:SS")

# 7. PPG files: columns, dtypes, pid = file name, collections and sample counts equal watch_collections
wc = T['watch_collections.csv'].copy(); wc['n_ppg_samples'] = wc['n_ppg_samples'].astype(int)
exp = wc.groupby('pid').agg(n_col=('time_kst', 'size'), n_samp=('n_ppg_samples', 'sum'))
assert {os.path.basename(f)[:-7] for f in ppg} == set(exp.index)
tot = 0
for f in ppg:
    d = pd.read_csv(f, dtype={'pid': str, 'time_kst': str}); pid = os.path.basename(f)[:-7]
    assert list(d.columns) == PPG_COLS and (d['pid'] == pid).all(), f
    assert d['day'].dtype.kind == 'i' and d['ppg'].dtype.kind == 'i' and d['t_s'].dtype.kind == 'f' and (d['day'] >= 1).all(), f
    assert d['time_kst'].str.fullmatch(r'\d\d:\d\d:\d\d').all(), f
    assert len(d) == exp.at[pid, 'n_samp'] and len(d.drop_duplicates(['day', 'time_kst'])) == exp.at[pid, 'n_col'], f
    tot += len(d)
ok(f"PPG: {len(ppg)} files, {tot:,} samples; per-file collections and samples equal watch_collections")

# 8. accelerometer files: columns, dtypes, pid = file name, sample counts equal watch_collections.n_acc_samples
wa = wc[wc['n_acc_samples'].notna()].copy(); wa['n_acc_samples'] = wa['n_acc_samples'].astype(int)
expa = wa.groupby('pid').agg(n_col=('time_kst', 'size'), n_samp=('n_acc_samples', 'sum'))
assert {os.path.basename(f)[:-7] for f in acc} == set(expa.index)
tota = 0
for f in acc:
    d = pd.read_csv(f, dtype={'pid': str, 'time_kst': str}); pid = os.path.basename(f)[:-7]
    assert list(d.columns) == ACC_COLS and (d['pid'] == pid).all() and (d['day'] >= 1).all(), f
    assert all(d[c].dtype.kind == 'i' for c in ['day', 'acc_x', 'acc_y', 'acc_z']) and d['time_kst'].str.fullmatch(r'\d\d:\d\d:\d\d').all(), f
    assert len(d) == expa.at[pid, 'n_samp'] and len(d.drop_duplicates(['day', 'time_kst'])) == expa.at[pid, 'n_col'], f
    tota += len(d)
ok(f"accelerometer: {len(acc)} files, {tota:,} samples; per-file collections and samples equal watch_collections")

open(OUT + 'check_release.txt', 'w', encoding='utf-8').write("\n".join(log) + "\n")
print("all checks passed; report in", OUT + 'check_release.txt')
