"""Synthetic eICU-like fixture for the TCD tests: patient + lab tables with lab PANELS drawn at shared offsets, rows before ICU admission and after
hour 24 (which must never matter), short stays, several labs per role, extra non-role labs, a lab that is never measured for training patients."""
import numpy as np
import pandas as pd

PANEL = ["creatinine", "bun", "sodium", "chloride", "bicarbonate", "platelets x 1000", "wbc x 1000", "glucose", "potassium", "hgb"]
BLOOD_GAS = ["pao2", "fio2", "paco2", "ph", "lactate"]
EXTRA = ["-lymphs", "magnesium", "calcium", "mch"]


def make_dataset(root, n_pat=260, seed=0, hospitals=(1, 2, 3, 4), post_exit_poison=False):
    rs = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    ids = np.arange(1000, 1000 + n_pat)
    los_h = np.where(rs.random(n_pat) < 0.15, rs.uniform(2, 23, n_pat), rs.uniform(26, 200, n_pat))
    pats = pd.DataFrame({"patientunitstayid": ids, "hospitalid": rs.choice(hospitals, n_pat), "unitdischargeoffset": np.round(los_h * 60).astype(int),
                         "unitdischargestatus": rs.choice(["Alive", "Expired"], n_pat, p=[.93, .07]),
                         "hospitaldischargestatus": rs.choice(["Alive", "Expired"], n_pat, p=[.9, .1])})
    rows = []
    for pid, exit_min in zip(ids, pats["unitdischargeoffset"]):
        sick = rs.random() < 0.3
        n_draw = rs.integers(0, 14)
        times = np.sort(rs.integers(-300, int(min(exit_min, 6000)), size=n_draw))       # before admission and far past hour 24 on purpose
        for t in times:
            for lab in PANEL:
                if rs.random() < 0.8:
                    base = {"creatinine": 2.2 if sick else 1.0, "bun": 40 if sick else 18, "sodium": 138, "chloride": 103, "bicarbonate": 21 if sick else 25,
                            "platelets x 1000": 120 if sick else 240, "wbc x 1000": 14 if sick else 8, "glucose": 150, "potassium": 4.2, "hgb": 11}[lab]
                    rows.append((pid, int(t), lab, round(float(rs.normal(base, abs(base) * 0.12)), 2)))
            if rs.random() < 0.35:
                for lab in BLOOD_GAS:
                    base = {"pao2": 70 if sick else 95, "fio2": rs.choice([21, 40, 60, 0.4]), "paco2": 44, "ph": 7.3 if sick else 7.4, "lactate": 3.0 if sick else 1.2}[lab]
                    rows.append((pid, int(t), lab, round(float(rs.normal(base, abs(base) * 0.05)), 2) if lab != "fio2" else float(base)))
            if rs.random() < 0.3:
                for lab in EXTRA:
                    rows.append((pid, int(t), lab, round(float(rs.normal(5, 1)), 2)))
            if rs.random() < 0.02:
                rows.append((pid, int(t), "creatinine", "not-a-number"))                   # unparseable result
    if post_exit_poison:     # rows after ICU exit but before minute 1440 (must be ignored): creatinine = 999 and a poisoned bun
        for pid, exit_min in zip(ids, pats["unitdischargeoffset"]):
            if exit_min <= 1400:
                rows += [(pid, int(exit_min) + 20, "creatinine", 999.0), (pid, int(exit_min) + 20, "bun", 999.0)]
    lab = pd.DataFrame(rows, columns=["patientunitstayid", "labresultoffset", "labname", "labresult"])
    lab.to_csv(root / "lab.csv", index=False)
    pats.to_csv(root / "patient.csv", index=False)
    return pats, lab


def make_manifest(pats, seed=7):
    from src.data.splitting.patient_split import SplitManifest
    rs = np.random.default_rng(seed)
    sp = rs.choice(["train", "val", "test"], len(pats), p=[.7, .15, .15])
    return SplitManifest(seed=1, train_frac=.7, val_frac=.15, test_frac=.15, stratify_on="none", sepsis_cohort=None,
                         patient_to_split={str(int(i)): s for i, s in zip(pats["patientunitstayid"], sp)})
