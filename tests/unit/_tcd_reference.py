"""Slow, loop-based REFERENCE implementation of the per-variable TCD features, written directly from the definitions in tcd_features.VAR_FEATURES
(no shared code with the vectorised implementation). Used only by the tests to prove the vectorised code computes what it documents."""
import math

import numpy as np


def ref_features(ts, vs, tcfg, rng):
    """ts: minutes (sorted, unique), vs: values. Returns {suffix: float or nan} for ONE patient/variable."""
    nan = float("nan")
    n = len(ts)
    W, half, mc, span = tcfg["window_minutes"], tcfg["half_split_minutes"], tcfg["min_counts"], tcfg["min_span_hours"]
    out = {}
    if n == 0:
        out.update({k: nan for k in ("mean", "median", "min", "max", "std", "iqr", "mad", "p10", "p90", "cv", "first", "last", "early_mean", "late_mean",
                                      "slope_per_h", "late_minus_early", "rate_first_last_per_h", "median_abs_change", "max_abs_change_6h", "mean_interval_h",
                                      "max_gap_h", "hours_since_last", "first_meas_h")})
        out.update({"n_meas": 0.0, "any_meas": 0.0, "hourly_coverage": 0.0})
        if rng is not None:
            out.update({k: nan for k in ("frac_abnormal", "n_abnormal", "n_abn_episodes", "abn_time_frac", "longest_abn_h", "first_abnormal", "last_abnormal", "recovered", "deteriorated")})
        return out
    v = np.array(vs, float); t = np.array(ts, float)

    def q(x, p):
        x = np.sort(np.array(x, float)); pos = p * (len(x) - 1); lo, hi = math.floor(pos), math.ceil(pos)
        return x[lo] * (1 - (pos - lo)) + x[hi] * (pos - lo)
    out["mean"], out["median"], out["min"], out["max"] = v.mean(), q(v, .5), v.min(), v.max()
    out["first"], out["last"] = v[0], v[-1]
    out["std"] = v.std(ddof=1) if n >= mc["std"] else nan
    out["cv"] = (v.std(ddof=1) / abs(v.mean())) if (n >= 2 and abs(v.mean()) > 1e-9) else nan
    out["iqr"] = q(v, .75) - q(v, .25) if n >= mc["quantiles"] else nan
    out["p10"] = q(v, .10) if n >= mc["quantiles"] else nan
    out["p90"] = q(v, .90) if n >= mc["quantiles"] else nan
    out["mad"] = q(np.abs(v - q(v, .5)), .5) if n >= mc["mad"] else nan
    e, l = v[t < half], v[t >= half]
    out["early_mean"] = e.mean() if len(e) else nan
    out["late_mean"] = l.mean() if len(l) else nan
    th = t / 60.0; sp = (t[-1] - t[0]) / 60.0          # whole-minute difference first: exact at the 2 h / 4 h gates
    if n >= mc["slope"] and sp >= span["slope"]:
        out["slope_per_h"] = np.polyfit(th, v, 1)[0]
    else:
        out["slope_per_h"] = nan
    out["late_minus_early"] = (l.mean() - e.mean()) if (len(e) and len(l)) else nan
    out["rate_first_last_per_h"] = (v[-1] - v[0]) / sp if (n >= 2 and sp >= span["rate"]) else nan
    dts, dvs = np.diff(t) / 60.0, np.diff(v)          # difference in whole minutes THEN /60: exact at the 0.5 h and 6 h boundaries (diff of t/60 is not)
    out["median_abs_change"] = q(np.abs(dvs), .5) if n >= mc["median_abs_change"] else nan
    lo_h, hi_h = tcfg["short_change_hours"]
    m = (dts >= lo_h) & (dts <= hi_h)
    out["max_abs_change_6h"] = np.abs(dvs)[m].max() if m.any() else nan
    out["n_meas"], out["any_meas"] = float(n), 1.0
    out["mean_interval_h"] = sp / (n - 1) if n >= 2 else nan
    out["max_gap_h"] = dts.max() if n >= 2 else nan
    out["hours_since_last"] = (W - t[-1]) / 60.0
    out["first_meas_h"] = t[0] / 60.0
    nb = W // 60
    out["hourly_coverage"] = len({min(int(x // 60), nb - 1) for x in t}) / nb
    if rng is not None:
        lo = -math.inf if rng[0] is None else rng[0]; hi = math.inf if rng[1] is None else rng[1]
        abn = [(x < lo) or (x > hi) for x in v]
        out["n_abnormal"] = float(sum(abn)); out["frac_abnormal"] = sum(abn) / n
        out["first_abnormal"], out["last_abnormal"] = float(abn[0]), float(abn[-1])
        eps = sum(1 for i in range(n) if abn[i] and (i == 0 or not abn[i - 1]))
        out["n_abn_episodes"] = float(eps)
        out["recovered"] = (float(not abn[-1]) if abn[0] else nan) if n >= 2 else nan
        out["deteriorated"] = (float(abn[-1]) if not abn[0] else nan) if n >= 2 else nan
        obs = ab = 0.0; best = cur = 0.0
        for i in range(n - 1):
            d = dts[i]
            if d <= tcfg["locf_max_gap_hours"]:
                obs += d
                if abn[i]:
                    ab += d; cur += d; best = max(best, cur)
                else:
                    cur = 0.0
            else:
                cur = 0.0
        out["abn_time_frac"] = ab / obs if obs > 0 else nan
        out["longest_abn_h"] = best if obs > 0 else nan
    return out
