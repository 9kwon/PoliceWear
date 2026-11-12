"""Per-collection PPG signal quality index used in the Technical Validation of the data descriptor (reads only the release files).
Run from anywhere: python ppg_qc.py   (about 10 min; reads ../ppg/wave*/*.csv.gz, writes output/ppg_qc.csv)
For each 64 s collection: band-pass 0.5-4 Hz, split into 8 s windows; window SQI = highest normalized autocorrelation at lags
0.33-1.5 s (40-180 bpm); good window = SQI >= 0.5; hr = median window heart rate over good windows; hr_peak from pulse troughs.
Output: output/ppg_qc.csv (one row per collection, keyed like watch_collections.csv)."""
import pandas as pd, numpy as np, glob, os, sys
from scipy.signal import butter, sosfiltfilt, find_peaks
sys.stdout.reconfigure(encoding='utf-8')
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__)); REL = _os.path.join(_os.path.dirname(_HERE), 'data', '') ; OUT = _os.path.join(_HERE, 'output', '')
os.makedirs(OUT, exist_ok=True)
FS = 25.0
SOS = butter(3, [0.5, 4.0], btype='band', fs=FS, output='sos')
LAG = np.arange(int(0.33 * FS), int(1.5 * FS) + 1)

WIN = int(8 * FS)   # 8 s windows: one motion burst spoils a window, not the whole 64 s
GOOD = 0.5          # window autocorrelation SQI threshold

def win_sqi(y):
    y = (y - y.mean()) / (y.std() + 1e-12)
    ac = np.array([np.dot(y[:-k], y[k:]) / len(y) for k in LAG])
    j = int(np.argmax(ac)); peak_inside = 0 < j < len(LAG) - 1  # a maximum on the lag edge is drift, not a pulse
    return (float(ac[j]) if peak_inside else 0.0), 60 * FS / LAG[j]

def qc(t, x):
    n = len(x)
    fs = (n - 1) / (t[-1] - t[0]) if t[-1] > t[0] else np.nan
    flat = float(np.mean(np.diff(x) == 0))
    if n < 2 * WIN or np.std(x) == 0:
        return dict(n=n, fs=fs, sqi_med=np.nan, good_frac=np.nan, hr=np.nan, hr_peak=np.nan, flat_frac=flat)
    y = sosfiltfilt(SOS, x - np.mean(x))
    w = [win_sqi(y[s:s + WIN]) for s in range(0, n - WIN + 1, WIN)]
    sq = np.array([a for a, _ in w]); hr = np.array([h for _, h in w])
    ok = sq >= GOOD
    pk, _ = find_peaks(-y / (np.median(np.abs(y)) + 1e-12), distance=int(0.33 * FS), prominence=1.0)  # green PPG dips with each pulse
    ibi = np.diff(pk) / FS; ibi = ibi[(ibi > 0.33) & (ibi < 1.5)]
    return dict(n=n, fs=fs, sqi_med=float(np.median(sq)), good_frac=float(ok.mean()),
                hr=float(np.median(hr[ok])) if ok.any() else np.nan,
                hr_peak=60 / np.median(ibi) if len(ibi) >= 10 else np.nan, flat_frac=flat)

rows = []
files = sorted(glob.glob(REL + 'ppg/wave*/*.csv.gz'))
for i, f in enumerate(files):
    d = pd.read_csv(f, dtype={'pid': str, 'time_kst': str})
    w = int(os.path.basename(os.path.dirname(f))[-1])
    for (day, tk), g in d.groupby(['day', 'time_kst'], sort=False):
        r = qc(g['t_s'].to_numpy(float), g['ppg'].to_numpy(float))
        r.update(pid=g['pid'].iat[0], wave=w, day=day, time_kst=tk); rows.append(r)
    if i % 50 == 0: print(i, len(files), flush=True)
q = pd.DataFrame(rows)[['pid', 'wave', 'day', 'time_kst', 'n', 'fs', 'sqi_med', 'good_frac', 'hr', 'hr_peak', 'flat_frac']]
q.to_csv(OUT + 'ppg_qc.csv', index=False)
print(q.describe().round(3).to_string())
