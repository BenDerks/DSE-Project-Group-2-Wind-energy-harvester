"""
figures_per_class.py - Figures with one value per wind class (baseline design)
==============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt

from .. import model as km
from .. import wind as da
from .settings import BASE, C_GREY, C_KITE, C_TETHER, S_MAX
from .cases import N_CLASSES, _key, cases_per_class
from .style import _base_note, _style, short_label


# =====================================================================================
# FIGURES
# =====================================================================================
def fig_size_mass_per_class(res, df=None):
    """
    Study 1: kite area and airborne + tether mass for every wind class, in one graph.

    Why: the two numbers per class that feed the volume and mass budget, side by side.
    Baseline design; classes where no kite up to S_MAX reaches the target are labelled.
    """
    rs = [res[_key(c)] for c in cases_per_class()]
    x = np.arange(1, N_CLASSES + 1)
    S = np.array([r['S'] for r in rs])
    M = np.array([r['m'] + r['mt_tot'] if r['feasible'] else np.nan for r in rs])
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.bar(x - 0.2, S, 0.4, color=C_KITE, label='projected kite area S [m²]')
    ax.bar(x + 0.2, M, 0.4, color=C_TETHER, label='kite + control unit + tether mass [kg]')
    for xi, s_, m_ in zip(x, S, M):
        if np.isfinite(s_):
            ax.text(xi - 0.2, s_, f'{s_:.0f}', ha='center', va='bottom', fontsize=7.5)
            ax.text(xi + 0.2, m_, f'{m_:.0f}', ha='center', va='bottom', fontsize=7.5)
    bad = [xi for xi, r in zip(x, rs) if not r['feasible']]
    if bad:
        which = f'C{min(bad)}' if min(bad) == max(bad) else f'C{min(bad)}-C{max(bad)}'
        ax.text(0.5 * (min(bad) + max(bad)), 0.05 * np.nanmax(np.r_[M, 1.0]),
                f'{which}:\nno kite up to\n{S_MAX:.0f} m²\nreaches 30 kW',
                ha='center', fontsize=7.5, color=C_GREY)
    ax.set_xlim(0.4, N_CLASSES + 0.6)
    ax.set_xticks(x, [short_label(c) for c in x], fontsize=8)
    _style(ax, 'Kite size and mass per wind class',
           f'wind class (mean wind speed at {da.CLASS_HEIGHT} m, m/s)', 'area [m²]  /  mass [kg]')
    ax.legend(frameon=False, fontsize=8, loc='upper right')
    LD = km.P['CLo'] / km.P['CDko']
    fig.text(0.01, 0.01, _base_note() + f'. Kite L/D {LD:.1f}, {km.P["w"]:.2f} kg/m². '
             'Tether mass for the full tether length.', fontsize=8, color=C_GREY)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def fig_power_curves(res, df=None):
    """Power curve of the kite designed for each class, and the wind distribution."""
    rs = [res[_key(c)] for c in cases_per_class()]
    feas = [(i + 1, r) for i, r in enumerate(rs) if r['feasible']]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                 gridspec_kw=dict(height_ratios=[2, 1]))
    cmap = plt.get_cmap('Blues')
    for j, (cls, r) in enumerate(feas):
        col = cmap(0.35 + 0.65 * j / max(1, len(feas) - 1))
        a1.plot(r['v'], np.array(r['Pc']) / 1e3, '-', color=col,
                label=f"{da.CLASS_NAMES[cls - 1]} (S = {r['S']:.0f} m²)")
        a2.plot(r['v'], 100 * np.array(r['pj']), '-', color=col)
    _style(a1, 'Cycle power of the kite designed for each class', ylabel='cycle power P_c [kW]')
    a1.legend(frameon=False, fontsize=8)
    _style(a2, 'Wind-speed distribution at the mean flying height', 'wind speed at z̄ [m/s]',
           'probability [%]')
    a2.set_xlim(0, 30)
    fig.text(0.01, 0.005, _base_note() + '. Mechanical power at the drum.', fontsize=8, color=C_GREY)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return fig


def fig_wind_classes(res, df=None):
    """
    Number of areas (ERA5 points) per mean wind speed, and the size of each class.

    Shows how much of the data each kite design would serve. Top: points per 0.25 m/s of
    mean wind speed at CLASS_HEIGHT, coloured by class. Bottom: each class's share of points,
    land area (cell area shrinks with latitude) and population.
    """
    summ = da.class_summary(df)
    v = df[f'h{da.CLASS_HEIGHT}_mean_speed_m_s']
    edges = np.arange(np.floor(v.min() * 4) / 4, np.ceil(v.max() * 4) / 4 + 0.25, 0.25)
    blues = plt.cm.Blues(np.linspace(0.3, 0.95, N_CLASSES))

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7.5), gridspec_kw=dict(height_ratios=[3, 2]))
    for c in range(1, N_CLASSES + 1):
        a1.hist(v[df['wind_class'] == c], bins=edges, color=blues[c - 1], label=None)
    for e in da.CLASS_EDGES[1:-1]:
        a1.axvline(e, color='#555555', lw=0.6, ls=':')
    top = a1.get_ylim()[1] * 1.15                  # room for the class labels above the bars
    a1.set_ylim(0, top)
    a1.set_xlim(0, np.percentile(v, 99.9) + 0.5)
    lows = [max(da.CLASS_EDGES[0], edges[0])] + da.CLASS_EDGES[1:-1]
    highs = da.CLASS_EDGES[1:-1] + [min(da.CLASS_EDGES[-1], edges[-1])]
    for c, (lo, hi) in enumerate(zip(lows, highs), start=1):
        share = summ.loc[summ['wind_class'] == c, 'share_points_pct'].iloc[0]
        a1.text(0.5 * (lo + hi) if c < N_CLASSES else lo + 0.6, top * 0.99,
                f'C{c}\n{share:.0f}%', ha='center', va='top', fontsize=8)
    _style(a1, f'Number of areas per mean wind speed at {da.CLASS_HEIGHT} m '
               f'({len(df):,} ERA5 cells)', f'mean wind speed at {da.CLASS_HEIGHT} m [m/s]',
           'number of cells')

    x = np.arange(1, N_CLASSES + 1)
    cols = [('share_points_pct', 'cells', C_KITE),
            ('share_area_pct', 'area', C_TETHER),
            ('share_pop_pct', 'population', '#d97706')]
    for i, (col, lab, colr) in enumerate(cols):
        a2.bar(x + (i - 1) * 0.27, summ[col], 0.27, color=colr, label=lab)
    a2.set_xticks(x, [short_label(c) for c in x], fontsize=8)
    _style(a2, 'Share of each wind class', f'wind class (mean wind speed at {da.CLASS_HEIGHT} m, m/s)',
           'share [%]')
    a2.legend(frameon=False, fontsize=8, ncol=3)
    fig.text(0.01, 0.005, 'Each ERA5 cell is one area of about 28 x 28 km at the equator '
             '(smaller towards the poles). Classes from data_analysis.CLASS_EDGES.',
             fontsize=8, color=C_GREY)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return fig


