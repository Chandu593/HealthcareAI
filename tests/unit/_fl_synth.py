"""Synthetic FL clients built WITHOUT a cohort (pure arrays), for the FL unit tests."""
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.three_modality.fl_clients import FLClient


def make_client(hid, n_tr=60, n_va=24, n_te=24, d=6, seed=0, shift=0.0, signal=1.5, eligible=None, task="mortality", min_pos=2):
    rs = np.random.default_rng(seed * 1000 + hid)

    def part(n):
        Z = rs.normal(shift, 1.0, (n, d))
        y = ((signal * (Z[:, 0] - shift) + rs.normal(0, 1.0, n)) > 1.0).astype(np.int64)
        if n >= 2 * min_pos:
            y[:min_pos], y[min_pos:2 * min_pos] = 1, 0                      # both classes present
        return Z, y

    (Ztr, ytr), (Zva, yva), (Zte, yte) = part(n_tr), part(n_va), part(n_te)
    base = 100000 * hid
    c = FLClient(hospitalid=hid, task=task, counts={"train": {"n": n_tr, "pos": int(ytr.sum()), "neg": int((ytr == 0).sum())}, "val": {"n": n_va, "pos": int(yva.sum()), "neg": int((yva == 0).sum())},
                                                   "test": {"n": n_te}},
                 pid={"train": np.arange(base, base + n_tr), "val": np.arange(base + 1000, base + 1000 + n_va), "test": np.arange(base + 2000, base + 2000 + n_te)},
                 Z={"train": Ztr, "val": Zva, "test": Zte}, y={"train": ytr, "val": yva}, nonfl_eligible=(n_tr >= 20) if eligible is None else eligible, seed=seed * 7 + hid)
    c.scaler = StandardScaler().fit(Ztr)
    return c


def make_clients(sizes=(120, 80, 40, 25), **kw):
    return [make_client(11 + i, n_tr=n, n_va=max(8, n // 3), n_te=max(8, n // 3), **kw) for i, n in enumerate(sizes)]
