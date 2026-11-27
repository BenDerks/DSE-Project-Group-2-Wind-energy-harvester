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
from .style import _base_note, _mark_infeasible, _style, short_label


# =====================================================================================
# FIGURES
# =====================================================================================
def fig_size_per_class(res, df=None):
    """Kite area and mass (kite + tether) for every wind class."""
    rs = [res[_key(c)] for c in cases_per_class()]
    x = np.arange(1, N_CLASSES + 1)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    S = [r['S'] for r in rs]
    A = [r['A'] for r in rs]
    a1.bar(x - 0.2, S, 0.4, color=C_KITE, label='projected area S')
    a1.bar(x + 0.2, A, 0.4, color=C_TETHER, label='flat area A')
    _mark_infeasible(a1, x, rs)
    _style(a1, 'Kite area needed for 30 kW mean power per wind class', ylabel='area [m²]')
    a1.legend(frameon=False, loc='upper left')

    m = [r['m'] for r in rs]
    mt = [r['mt_tot'] if r['feasible'] else np.nan for r in rs]
    a2.bar(x, m, 0.6, color=C_KITE, label='kite + control unit')
    a2.bar(x, mt, 0.6, bottom=m, color=C_TETHER, label='tether (full length)')
    for xi, mi, ti in zip(x, m, mt):
        if np.isfinite(mi):
            a2.text(xi, mi + ti, f'{mi + ti:.0f}', ha='center', va='bottom', fontsize=8)
    _mark_infeasible(a2, x, rs)
    _style(a2, 'Airborne mass and tether mass', ylabel='mass [kg]')
    a2.legend(frameon=False, loc='upper left')
    a2.set_xticks(x, [short_label(c) for c in x])
    a2.set_xlabel(f'wind class (mean wind speed at {da.CLASS_HEIGHT} m, m/s)')
    fig.text(0.01, 0.005, _base_note(), fontsize=8, color=C_GREY)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return fig


def fig_downtime_per_class(res, df=None):
    """Hours per year out of operation and cut-in wind speed for every class."""
    rs = [res[_key(c)] for c in cases_per_class()]
    x = np.arange(1, N_CLASSES + 1)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    T = [r['T_out'] for r in rs]
    ax.bar(x, T, 0.6, color=C_KITE)
    for xi, r in zip(x, rs):
        if r['feasible']:
            ax.text(xi, r['T_out'] + 80, f"{100 * r['T_out'] / 8760:.0f}%\ncut-in\n{r['v_in']:.1f} m/s",
                    ha='center', va='bottom', fontsize=7.5)
    _mark_infeasible(ax, x, rs)
    ax.set_ylim(0, 8760)
    _style(ax, 'Hours per year below cut-in or above cut-out', ylabel='hours out of operation per year')
    ax.set_xticks(x, [short_label(c) for c in x])
    ax.set_xlabel(f'wind class (mean wind speed at {da.CLASS_HEIGHT} m, m/s)')
    fig.text(0.01, 0.005, _base_note() + '. Cut-in is the wind speed at the mean flying height.',
             fontsize=8, color=C_GREY)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
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


def fig_stability_cf(res, df=None):
    """
    Power variability (stability criterion) and mean power as share of max cycle power.

    Not plotted: capacity factor against peak reel-out power (F_max * v_max); with fixed
    target and F_max it is the same for every class by construction.
    """
    rs = [res[_key(c)] for c in cases_per_class()]
    x = np.arange(1, N_CLASSES + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))
    a1.bar(x, [r['sigma_rel'] for r in rs], 0.6, color=C_KITE)
    _style(a1, 'Variability of the power, σ_P / mean', ylabel='σ_P / P̄ [-]')
    share = [100 * BASE['P_target'] / np.nanmax(r['Pc']) if r['feasible'] else np.nan for r in rs]
    a2.bar(x, share, 0.6, color=C_KITE)
    for xi, r, s in zip(x, rs, share):
        if r['feasible']:
            a2.text(xi, s + 0.5, f"max\n{np.nanmax(r['Pc']) / 1e3:.0f} kW", ha='center',
                    va='bottom', fontsize=7.5)
    _style(a2, 'Mean power as share of maximum cycle power', ylabel='P̄ / max P_c [%]')
    for ax in (a1, a2):
        ax.set_xticks(x, [short_label(c) for c in x], fontsize=8)
        ax.set_xlabel('wind class')
        _mark_infeasible(ax, x, rs)
    fig.text(0.01, 0.01, _base_note() + '. Variability over the wind distribution, not hourly.',
             fontsize=8, color=C_GREY)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
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


def fig_class_overview(res, df=None):
    """
    Kite area, airborne + tether mass and hours out of operation per class (baseline design).

    The three numbers that feed the trade-off (size, mass budget, downtime) side by side;
    one bar per class, value on top.
    """
    rs = [res[_key(c)] for c in cases_per_class()]
    x = np.arange(1, N_CLASSES + 1)
    panels = (
        ('Kite area for 30 kW mean', 'projected area S [m²]', [r['S'] for r in rs], '{:.0f}'),
        ('Airborne + tether mass', 'mass [kg]',
         [r['m'] + r['mt_tot'] if r['feasible'] else np.nan for r in rs], '{:.0f}'),
        ('Hours out of operation', 'hours per year', [r['T_out'] for r in rs], '{:,.0f}'),
    )
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, (title, unit, vals, fmt) in zip(axs, panels):
        ax.bar(x, vals, 0.6, color=C_KITE)
        top = np.nanmax(vals)
        for xi, v in zip(x, vals):
            if np.isfinite(v):
                ax.text(xi, v + 0.01 * top, fmt.format(v), ha='center', va='bottom', fontsize=7.5)
        ax.set_ylim(0, 1.12 * top)
        ax.set_xlim(0.4, N_CLASSES + 0.6)
        ax.set_xticks(x, [short_label(c) for c in x], fontsize=7)
        _style(ax, title, 'wind class (mean wind at 100 m, m/s)', unit)
        # One label spanning all classes without a design
        bad = [xi for xi, r in zip(x, rs) if not r['feasible']]
        if bad:
            ax.text(0.5 * (min(bad) + max(bad)), 0.06 * ax.get_ylim()[1],
                    f'C{min(bad)}-C{max(bad)}:\nno kite up to {S_MAX:.0f} m²\nreaches 30 kW',
                    ha='center', va='bottom', fontsize=8, color=C_GREY)
    LD = km.P['CLo'] / km.P['CDko']
    fig.text(0.01, 0.01, _base_note() + f'. Kite L/D {LD:.1f}, {km.P["w"]:.1f} kg/m². '
             f'Hours out of operation: wind below cut-in or above cut-out.', fontsize=8, color=C_GREY)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig
