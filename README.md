# Code

Scripts that reproduce the checks, the signal-quality index and the figures of the data descriptor from the released files
only. Nothing here needs the internal working files; run them from this folder (they locate the data in ../data/).

**Data.** The data this code reads are the PoliceWear release on Zenodo, https://doi.org/10.5281/zenodo.24099823 (CC BY 4.0). Download and unpack it, then place this folder as `code/` inside the unpacked release, next to `data/` (the package already contains an identical copy), so that the scripts find the tables and `ppg/`, `acc/` in `../data/`.

| File | What it does | Output |
|---|---|---|
| quickstart.ipynb | Loads the tables, lists what each participant has, and plots one recording (PPG, accelerometer, stress rating). Start here. | shown in the notebook |
| check_release.py | Consistency and de-identification checks: file inventory, column whitelist, no identifiers or calendar dates, one wave per participant, k ≥ 5 on wave + sex + age band + work type, time formats, PPG and accelerometer sample counts against `watch_collections.csv`. Raises on any failure. | output/check_release.txt |
| ppg_qc.py | Per-recording PPG quality index used in the Technical Validation: 0.5–4 Hz band-pass, eight 8 s windows, window good when the normalised autocorrelation at 40–180 bpm lags is ≥ 0.5; also heart rate from autocorrelation and from pulse troughs. About 10 min. | output/ppg_qc.csv |
| make_figures.py | Figures 3–8 of the descriptor and the numbers quoted in its text (needs output/ppg_qc.csv; the first run caches wrist motion per recording in output/acc_motion.csv). | output/figures/, output/summary.json |
| make_fig_participants.py | Figure 2 (sex, age band, work type, BMI by wave). | output/figures/Fig2_participants.png |

Requirements: Python ≥ 3.10 with pandas, numpy, scipy and matplotlib (the notebook also needs jupyter). Tested with Python 3.10, pandas 2.3,
numpy 1.26, scipy 1.11 and matplotlib 3.10.

```bash
pip install pandas numpy scipy matplotlib jupyter
python check_release.py
python ppg_qc.py
python make_figures.py
python make_fig_participants.py
```

Licence: MIT (see `LICENSE` in this folder). The data themselves are CC BY 4.0; see *Licence, citation and contact* in `../README.md`.
