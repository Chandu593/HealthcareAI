"""Neural-network preprocessing. All statistics are TRAIN-rows-only and
PER HOSPITAL (hospital's own train rows), with a fallback to the
federated-aggregated train statistics (sums over hospitals' train rows --
sufficient statistics, no raw rows exchanged) for hospital/feature cells
with too few observations."""

from __future__ import annotations

import warnings

import numpy as np


def _slog(x):
    return np.sign(x) * np.log1p(np.abs(x))


def prepare_tabular(X: np.ndarray, hospital_ids: np.ndarray, train_mask: np.ndarray, ncfg: dict):
    """Returns (Xn float32 [no NaN], info). Columns: standardized continuous | binary | missing indicators."""
    N, F = X.shape
    tr = X[train_mask]
    is_bin = np.zeros(F, dtype=bool)
    for j in range(F):
        v = tr[:, j]; v = v[~np.isnan(v)]
        is_bin[j] = v.size > 0 and bool(np.isin(v, (0.0, 1.0)).all())
    cont = np.where(~is_bin)[0]; binc = np.where(is_bin)[0]
    Xc = _slog(X[:, cont]).astype(np.float32)
    trc = Xc[train_mask]
    lo, hi = ncfg["missing_indicator_range"]
    miss_rate = np.isnan(trc).mean(axis=0) if len(cont) else np.array([])
    ind_cols = np.where((miss_rate >= lo) & (miss_rate <= hi))[0]
    ind = np.isnan(Xc[:, ind_cols]).astype(np.float32)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        g_mean = np.nanmean(trc, axis=0); g_std = np.nanstd(trc, axis=0)
    g_mean = np.where(np.isnan(g_mean), 0.0, g_mean); g_std = np.where(np.isnan(g_std) | (g_std < 1e-6), 1.0, g_std)
    Z = np.zeros_like(Xc)
    min_n = int(ncfg["min_hospital_n_for_local_stats"]); clip = float(ncfg["clip"])
    for h in np.unique(hospital_ids):
        rows = np.where(hospital_ids == h)[0]; trows = rows[train_mask[rows]]
        mean = g_mean.copy(); std = g_std.copy()
        if len(trows):
            sub = Xc[trows]; n = (~np.isnan(sub)).sum(axis=0)
            with np.errstate(invalid="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                m = np.nanmean(sub, axis=0); s = np.nanstd(sub, axis=0)
            ok = (n >= min_n) & ~np.isnan(m) & ~np.isnan(s) & (s > 1e-6)
            mean[ok] = m[ok]; std[ok] = s[ok]
        z = (Xc[rows] - mean) / std
        Z[rows] = np.nan_to_num(np.clip(z, -clip, clip), nan=0.0)
    Xb = np.nan_to_num(X[:, binc], nan=0.0).astype(np.float32)
    Xn = np.concatenate([Z, Xb, ind], axis=1).astype(np.float32)
    return Xn, {"n_continuous": int(len(cont)), "n_binary": int(len(binc)), "n_missing_indicators": int(len(ind_cols)), "dim": int(Xn.shape[1])}


def prepare_sequence(seq: np.ndarray, hospital_ids: np.ndarray, train_mask: np.ndarray, ncfg: dict) -> np.ndarray:
    """(N,B,L) hourly lab means with NaN -> (N,B,2L): per-hospital/per-lab z-scored values (0 where missing) + observed mask."""
    N, B, L = seq.shape
    S = _slog(seq).astype(np.float32); mask = (~np.isnan(S)).astype(np.float32)
    tr = S[train_mask].reshape(-1, L)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        g_mean = np.nanmean(tr, axis=0); g_std = np.nanstd(tr, axis=0)
    g_mean = np.where(np.isnan(g_mean), 0.0, g_mean); g_std = np.where(np.isnan(g_std) | (g_std < 1e-6), 1.0, g_std)
    out = np.zeros_like(S); min_n = int(ncfg["min_hospital_n_for_local_stats"]); clip = float(ncfg["clip"])
    for h in np.unique(hospital_ids):
        rows = np.where(hospital_ids == h)[0]; trows = rows[train_mask[rows]]
        mean, std = g_mean.copy(), g_std.copy()
        if len(trows):
            sub = S[trows].reshape(-1, L); n = (~np.isnan(sub)).sum(axis=0)
            with np.errstate(invalid="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                m = np.nanmean(sub, axis=0); s = np.nanstd(sub, axis=0)
            ok = (n >= min_n) & ~np.isnan(m) & ~np.isnan(s) & (s > 1e-6)
            mean[ok] = m[ok]; std[ok] = s[ok]
        out[rows] = np.nan_to_num(np.clip((S[rows] - mean) / std, -clip, clip), nan=0.0)
    return np.concatenate([out, mask], axis=2).astype(np.float32)
