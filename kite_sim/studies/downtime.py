"""
downtime.py - Downtime and continuous downtime of a design, from the hourly ERA5 wind of 2024
=============================================================================================

The kite model gives each design's cut-in and cut-out wind speed at its mean flying height z̄.
Here every hour of 2024 in every cell of the wind class is checked: the kite is down when the
wind at z̄ is below cut-in or above cut-out. Per cell this gives the downtime and the stretches
of consecutive down hours; the class value is the mean or median over its cells.

Wind at z̄ from the ERA5 wind at 10 and 100 m, as in windres (physics.convert_to_height):
    z̄ <= 100 m  linear in ln(z) between 10 and 100 m
    z̄ >  100 m  power law with the hourly shear exponent alpha = ln(V100 / V10) / ln(10);
                hours with V10 or V100 < 1 m/s, or alpha outside -0.2..0.8, use the cell's
                median valid alpha (1/7 if it has none).
Missing hours count as not down. Results are cached in DOWNTIME_CACHE, per class, z̄, cut-in
and cut-out, so only new designs read the hourly files (about 1 min for all 8,486 cells).
"""

import hashlib
import json
import pickle

import numpy as np
import pandas as pd

from .settings import DOWNTIME_CACHE, HOURLY_DIR

METRICS = ['downtime_h_yr', 'longest_downtime_h', 'longest_downtime_p90_h', 'mean_downtime_spell_h',
           'spells_ge_24h_per_yr']


def load_hourly(point_ids):
    """
    Hourly wind of the cells. Output: v10, v100 [m/s] and the shear exponent alpha [-] with the
    fallback already applied, each an array (n_cells, n_hours).
    """
    v10, v100 = [], []
    for p in point_ids:
        t = pd.read_parquet(HOURLY_DIR / p / '2024.parquet', columns=['u10', 'v10', 'u100', 'v100'])
        v10.append(np.hypot(t['u10'].to_numpy(float), t['v10'].to_numpy(float)))
        v100.append(np.hypot(t['u100'].to_numpy(float), t['v100'].to_numpy(float)))
    v10, v100 = np.array(v10), np.array(v100)
    ok = (v10 >= 1.0) & (v100 >= 1.0)
    alpha = np.full(v10.shape, np.nan)
    alpha[ok] = np.log(v100[ok] / v10[ok]) / np.log(10.0)
    valid = np.isfinite(alpha) & (alpha >= -0.2) & (alpha <= 0.8)
    median = np.array([np.median(a[m]) if m.any() else 1 / 7 for a, m in zip(alpha, valid)])
    return v10, v100, np.where(valid, alpha, median[:, None])


def speed_at(v10, v100, alpha, z):
    """Hourly wind speed [m/s] at height z [m] above ground (method in the module header)."""
    if z <= 100.0:
        return v10 + (v100 - v10) * np.log(z / 10.0) / np.log(10.0)
    return v100 * np.exp(alpha * np.log(z / 100.0))


def spell_metrics(down, valid):
    """
    Class values from a boolean array (n_cells, n_hours) of down hours.
    Output: dict with the keys of METRICS (per year = scaled to 8,760 h).
    """
    n_cells, n_hours = down.shape
    d = np.zeros((n_cells, n_hours + 2), dtype=np.int8)
    d[:, 1:-1] = down
    rows_s, cols_s = np.nonzero(np.diff(d, axis=1) == 1)        # starts of the down stretches
    _, cols_e = np.nonzero(np.diff(d, axis=1) == -1)            # ends, in the same order
    length = cols_e - cols_s
    longest = np.zeros(n_cells)
    np.maximum.at(longest, rows_s, length)
    count = np.bincount(rows_s, minlength=n_cells)
    mean_spell = np.bincount(rows_s, weights=length, minlength=n_cells) / np.maximum(count, 1)
    per_year = 8760.0 / np.maximum(valid.sum(axis=1), 1)
    return dict(downtime_h_yr=float(np.mean(down.sum(axis=1) * per_year)),
                longest_downtime_h=float(np.median(longest)),
                longest_downtime_p90_h=float(np.percentile(longest, 90)),
                mean_downtime_spell_h=float(np.mean(mean_spell)),
                spells_ge_24h_per_yr=float(np.mean(np.bincount(rows_s, weights=length >= 24,
                                                               minlength=n_cells) * per_year)))


def _design_key(r):
    v_out = r['v_out'] if np.isfinite(r['v_out']) else -1.0
    return f"{r['case']['cls']}|{r['zbar']:.6f}|{r['v_in']:.9g}|{v_out:.9g}"


def downtime_metrics(results, df):
    """
    Downtime metrics of every feasible design in results (dicts from run_case()).

    Inputs: results = list of run_case() outputs; df = classified data (cells per class).
    Output: DataFrame with one row per design: the key of its case (column 'key') and METRICS.
    """
    cells = {int(c): sorted(g['point_id']) for c, g in df.groupby('wind_class')}
    tag = hashlib.md5(json.dumps([str(HOURLY_DIR.name), cells]).encode()).hexdigest()[:12]
    cache = dict(tag=tag, metrics={})
    if DOWNTIME_CACHE.exists():
        with open(DOWNTIME_CACHE, 'rb') as fh:
            old = pickle.load(fh)
        if old.get('tag') == tag:
            cache = old
    todo = {}
    for r in results:
        if r['feasible'] and _design_key(r) not in cache['metrics']:
            todo.setdefault(r['case']['cls'], []).append(r)
    for cls, rs in sorted(todo.items()):
        print(f'  downtime: class {cls}, {len(cells[cls])} cells, {len(rs)} designs', flush=True)
        v10, v100, alpha = load_hourly(cells[cls])
        valid = np.isfinite(v100)
        for r in rs:
            v = speed_at(v10, v100, alpha, r['zbar'])
            v_out = r['v_out'] if np.isfinite(r['v_out']) else np.inf
            cache['metrics'][_design_key(r)] = spell_metrics((v < r['v_in']) | (v > v_out), valid)
        with open(DOWNTIME_CACHE, 'wb') as fh:
            pickle.dump(cache, fh)
    rows = [dict(cls=r['case']['cls'], **cache['metrics'][_design_key(r)]) for r in results
            if r['feasible']]
    return pd.DataFrame(rows, index=[json.dumps(r['case'], sort_keys=True) for r in results
                                     if r['feasible']])
