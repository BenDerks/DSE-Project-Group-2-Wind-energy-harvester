"""
distributions.py - Wind-speed distribution per class and height; bins for the kite model
========================================================================================
"""

import numpy as np
import pandas as pd

from .settings import BIN_CENTRES, BIN_WIDTH, DIST_HEIGHTS, ELEVATION_COLUMN, EXTRA_HEIGHTS, HEIGHTS


# =====================================================================================
# 3. WIND-SPEED DISTRIBUTION PER CLASS AND HEIGHT
# =====================================================================================
def weibull_bin_probabilities(A, k, centres=BIN_CENTRES, width=BIN_WIDTH):
    """
    Probability of each wind-speed bin for many Weibull distributions at once.

    Inputs
        A, k    : Weibull scale [m/s] and shape [-], one value per point (arrays)
        centres : [m/s] bin centres
        width   : [m/s] bin width

    Output
        array (n_points, n_bins): P(lower edge <= v < upper edge) per point and bin,
        from the Weibull CDF F(v) = 1 - exp(-(v/A)^k)
    """
    A = np.asarray(A, dtype=float)[:, None]
    k = np.asarray(k, dtype=float)[:, None]
    lower = np.maximum(centres - width / 2, 0.0)[None, :]
    upper = (centres + width / 2)[None, :]
    return np.exp(-(lower / A)**k) - np.exp(-(upper / A)**k)


def class_distributions(df, weight='points'):
    """
    Wind-speed distribution of each class at each data height.

    The kite model needs one distribution per class. Averaging the points' Weibull
    distributions (a mixture) keeps the spread between sites; averaging A and k instead
    would make the distribution too narrow.

    Inputs
        df     : DataFrame with 'wind_class' (from assign_classes)
        weight : 'points'     every point counts equally (default)
                 'population' points weighted by the people in the cell

    Output (dict)
        {class_id: array (len(DIST_HEIGHTS), len(BIN_CENTRES))} of bin probabilities;
        each row sums to ~1. Rows for EXTRA_HEIGHTS are extrapolated (see settings.py).
    """
    dist = {}
    for c in sorted(df['wind_class'].dropna().unique()):
        g = df[df['wind_class'] == c]
        w = np.ones(len(g)) if weight == 'points' else g['population'].to_numpy()
        w = w / w.sum()
        table = np.zeros((len(DIST_HEIGHTS), len(BIN_CENTRES)))
        for i, h in enumerate(HEIGHTS):
            p = weibull_bin_probabilities(g[f'h{h}_weibull_A_m_s'], g[f'h{h}_weibull_k'])
            table[i] = w @ p                     # weighted average over the points
        for i, h in enumerate(EXTRA_HEIGHTS, start=len(HEIGHTS)):
            A, k = _extrapolate_weibull(g, h)
            table[i] = w @ weibull_bin_probabilities(A, k)
        dist[int(c)] = table
    return dist


def _extrapolate_weibull(g, z):
    """
    Weibull A and k of every point at a height z above the data (which stops at 400 m).

    Method and source: EXTRA_HEIGHTS comment in settings.py.

    Inputs: g = DataFrame of points (needs h300/h400 Weibull columns), z [m] > 400
    Output: (A [m/s], k [-]) arrays, one value per point
    """
    A300, A400 = g['h300_weibull_A_m_s'].to_numpy(), g['h400_weibull_A_m_s'].to_numpy()
    alpha = np.log(A400 / A300) / np.log(400 / 300)          # point's own shear exponent
    return A400 * (z / 400) ** alpha, g['h400_weibull_k'].to_numpy()


def class_elevations(df, weight='points'):
    """
    Mean ground elevation above sea level of every wind class [m].

    The kite model uses it as site elevation for the air density (M3). Same weighting as class_distributions(), so wind and air density describe the same
    points. One value per class (not per point) is a simplification: density falls ~1% per
    100 m, so a few hundred metres of spread within a class changes the power a few percent.

    Inputs
        df     : DataFrame with 'wind_class'
        weight : 'points' or 'population' (as in class_distributions)

    Output (dict)
        {class_id: mean elevation [m]}; all 0.0 if ELEVATION_COLUMN is not in the data
    """
    classes = sorted(int(c) for c in df['wind_class'].dropna().unique())
    if ELEVATION_COLUMN not in df.columns:
        print(f'WARNING: no column {ELEVATION_COLUMN!r} in the data; sea level assumed for '
              f'the air density')
        return {c: 0.0 for c in classes}
    out = {}
    for c in classes:
        g = df[df['wind_class'] == c]
        w = np.ones(len(g)) if weight == 'points' else g['population'].to_numpy()
        z = g[ELEVATION_COLUMN].to_numpy(dtype=float)
        ok = np.isfinite(z)
        out[c] = float(np.average(z[ok], weights=w[ok])) if ok.any() else 0.0
    return out


def wind_bins(dist, class_id, z):
    """
    Wind-speed bins and probabilities of one class at altitude z (M4).

    The kite's mean altitude (geo['zbar']) usually lies between two data heights, so the
    probabilities are interpolated linearly in height and normalised to sum to 1.

    Inputs
        dist     : output of class_distributions()
        class_id : wind class number
        z        : [m] altitude, within DIST_HEIGHTS (lowest data height to highest
                   extrapolated height); above 400 m it is an estimate

    Output
        vs : [m/s] bin centres
        pj : [-]   probability per bin (sums to 1)
    """
    if not DIST_HEIGHTS[0] <= z <= DIST_HEIGHTS[-1]:
        raise ValueError(f'z = {z} m is outside {DIST_HEIGHTS[0]}-{DIST_HEIGHTS[-1]} m')
    table = dist[class_id]
    pj = np.array([np.interp(z, DIST_HEIGHTS, table[:, j]) for j in range(len(BIN_CENTRES))])
    return BIN_CENTRES.copy(), pj / pj.sum()


def distributions_to_table(dist):
    """Long table (class, height, bin centre, probability) for saving to CSV."""
    rows = []
    for c, table in dist.items():
        for i, h in enumerate(DIST_HEIGHTS):
            for v, p in zip(BIN_CENTRES, table[i]):
                rows.append(dict(wind_class=c, height_m=h, v_m_s=v, probability=p))
    return pd.DataFrame(rows)
