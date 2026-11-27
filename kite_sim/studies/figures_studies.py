"""
figures_studies.py - Figures of the height, F_max, sensitivity and kite L/D studies
===================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory

from .. import model as km
from .. import wind as da
from .settings import (BASE, C_GREY, C_KITE, C_LINES, C_TETHER, DISPLAY_SCALE, FMAX_CLASSES,
                       FMAX_VALUES, HEIGHT_CLASSES, HEIGHT_WINDOWS, LD_RANGE, S_MAX, SENS_CLASS,
                       SENSITIVITY)
from .cases import N_CLASSES, _key, make_case
from .style import _base_note, _style, short_label


# =====================================================================================
# FIGURES
# =====================================================================================
def fig_size_vs_height(res, df=None):
    """Kite area, mass and tether against the average flying height."""
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2))
    for i, cls in enumerate(HEIGHT_CLASSES):
        rs = [res[_key(make_case(cls, zmin=a, zmax=b))] for a, b in HEIGHT_WINDOWS]
        z = [r['zbar'] for r in rs]
        col = C_LINES[i % len(C_LINES)]
        lab = da.CLASS_NAMES[cls - 1]
        axs[0].plot(z, [r['S'] for r in rs], 'o-', color=col, label=lab)
        axs[1].plot(z, [r['m'] + r['mt_tot'] if r['feasible'] else np.nan for r in rs],
                    'o-', color=col, label=lab)
        axs[2].plot(z, [r['T_out'] for r in rs], 'o-', color=col, label=lab)
    for ax in axs:     # wind above the highest data height is extrapolated
        ax.axvspan(da.HEIGHTS[-1], ax.get_xlim()[1], color='#f2f2f2', zorder=0)
        ax.text(da.HEIGHTS[-1] + 5, 0.98, 'wind\nextrapolated', transform=
                blended_transform_factory(ax.transData, ax.transAxes),
                va='top', fontsize=7, color=C_GREY)
    _style(axs[0], 'Kite area', 'average flying height z̄ [m]', 'projected area S [m²]')
    _style(axs[1], 'Airborne + tether mass', 'average flying height z̄ [m]', 'mass [kg]')
    _style(axs[2], 'Hours out of operation', 'average flying height z̄ [m]', 'hours per year')
    axs[0].legend(frameon=False, fontsize=8)
    fig.text(0.01, 0.01, f'Each point: heights z̄ ± 50 m, F_max {BASE["Fmax"] / 1e3:.0f} kN, '
             f'target {BASE["P_target"] / 1e3:.0f} kW. Missing points: no kite up to {S_MAX:.0f} m². '
             f'Tether mass grows from {rs[0]["mt_tot"]:.0f} kg (z̄ = {rs[0]["zbar"]:.0f} m) to '
             f'{rs[-1]["mt_tot"]:.0f} kg ({rs[-1]["zbar"]:.0f} m), the same for every class.',
             fontsize=8, color=C_GREY)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def fig_fmax_effect(res, df=None):
    """Kite area and mass against the maximum tether force."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for i, cls in enumerate(FMAX_CLASSES):
        rs = [res[_key(make_case(cls, Fmax=F))] for F in FMAX_VALUES]
        F = np.array(FMAX_VALUES) / 1e3
        col = C_LINES[i % len(C_LINES)]
        lab = da.CLASS_NAMES[cls - 1]
        a1.plot(F, [r['S'] for r in rs], 'o-', color=col, label=lab)
        a2.plot(F, [r['m'] + r['mt_tot'] if r['feasible'] else np.nan for r in rs], 'o-', color=col, label=lab)
    _style(a1, 'Kite area against maximum tether force', 'F_max [kN]', 'projected area S [m²]')
    _style(a2, 'Total airborne + tether mass', 'F_max [kN]', 'mass [kg]')
    a1.legend(frameon=False, fontsize=8)
    fig.text(0.01, 0.01, f'Heights {BASE["zmin"]:.0f}-{BASE["zmax"]:.0f} m, target '
             f'{BASE["P_target"] / 1e3:.0f} kW. Missing points: no kite up to {S_MAX:.0f} m² '
             '(F_max sets the ceiling on mean power).', fontsize=8, color=C_GREY)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def fig_sensitivity(res, df=None):
    """Tornado chart: change in kite area when one parameter goes to its low or high value."""
    base = res[_key(make_case(SENS_CLASS))]
    rows = []
    for key, lo, hi, label in SENSITIVITY:
        r_lo = res[_key(make_case(SENS_CLASS, **{key: lo}))]
        r_hi = res[_key(make_case(SENS_CLASS, **{key: hi}))]
        d_lo = 100 * (r_lo['S'] / base['S'] - 1)
        d_hi = 100 * (r_hi['S'] / base['S'] - 1)
        scale = DISPLAY_SCALE.get(key, 1.0)
        rows.append((label, lo / scale, hi / scale, d_lo, d_hi))
    rows.sort(key=lambda t: np.nanmax(np.abs([t[3], t[4]])) if np.any(np.isfinite([t[3], t[4]])) else 1e9)
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(rows) + 1.8))
    y = np.arange(len(rows))
    ax.barh(y, [r[3] for r in rows], color=C_TETHER, label='low value')
    ax.barh(y, [r[4] for r in rows], color=C_KITE, label='high value')
    for yi, (label, lo, hi, dl, dh) in zip(y, rows):
        # One label per row, right of the bars: low and high value and their effect
        fmt = lambda d: f'{d:+.0f}%' if np.isfinite(d) else 'not feasible'
        right = np.nanmax([0, dl, dh])
        ax.text(right + 1.5, yi, f'{lo:g} \u2192 {fmt(dl)},  {hi:g} \u2192 {fmt(dh)}',
                va='center', ha='left', fontsize=8)
    ax.axvline(0, color='#555555', lw=0.8)
    ax.set_yticks(y, [r[0] for r in rows])
    _style(ax, f'Sensitivity of the kite area, {da.CLASS_NAMES[SENS_CLASS - 1]} '
               f'(baseline S = {base["S"]:.1f} m²)', 'change in kite area [%]')
    ax.grid(axis='x', color='#e5e5e5', lw=0.8)
    ax.legend(frameon=False, loc='lower right')
    lim = np.nanmax(np.abs([[r[3], r[4]] for r in rows])) if rows else 10
    ax.set_xlim(-1.2 * lim, 1.9 * lim)
    fig.text(0.01, 0.01, _base_note() + '. Changes of about 2% or less are within the numerical '
             'noise of kite_model.FAST.', fontsize=8, color=C_GREY)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def fig_size_range_ld(res, df=None):
    """
    Kite area and hours out of operation per class: baseline L/D (point), LD_RANGE (whisker).

    Kite L/D is the most uncertain input with the largest effect on kite size. The whisker
    spans the measured peak L/D (smaller kite) to the lowest measured average (larger kite).
    """
    x = np.arange(1, N_CLASSES + 1)
    LD0 = km.P['CLo'] / km.P['CDko']
    base = [res[_key(make_case(c))] for c in x]
    alt = {LD: [res[_key(make_case(c, CDko=km.P['CLo'] / LD))] for c in x] for LD in LD_RANGE}
    lo_LD, hi_LD = max(LD_RANGE), min(LD_RANGE)          # high L/D -> small kite

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
    for ax, key, unit, title in ((a1, 'S', 'projected area S [m²]', 'Kite area for 30 kW mean'),
                                 (a2, 'T_out', 'hours per year', 'Hours out of operation')):
        for xi, rb, rl, rh in zip(x, base, alt[lo_LD], alt[hi_LD]):
            y0 = rb[key]
            y_lo = rl[key] if rl['feasible'] else np.nan
            y_hi = rh[key] if rh['feasible'] else np.nan
            if np.isfinite(y_lo) or np.isfinite(y_hi) or np.isfinite(y0):
                bottom = y_lo if np.isfinite(y_lo) else y0
                top = y_hi if np.isfinite(y_hi) else (S_MAX if key == 'S' else y0)
                if np.isfinite(bottom) and np.isfinite(top):
                    ax.plot([xi, xi], [bottom, top], color=C_TETHER, lw=2.5, solid_capstyle='butt')
                    ax.plot([xi - 0.12, xi + 0.12], [bottom, bottom], color=C_TETHER, lw=1.5)
                    if np.isfinite(y_hi):
                        ax.plot([xi - 0.12, xi + 0.12], [top, top], color=C_TETHER, lw=1.5)
                    elif key == 'S':
                        ax.annotate('', (xi, top + 12), (xi, top), arrowprops=dict(arrowstyle='->', color=C_TETHER))
            if np.isfinite(y0):
                ax.plot(xi, y0, 'o', color=C_KITE, ms=7, zorder=3)
                ax.text(xi + 0.18, y0, f'{y0:.0f}', va='center', fontsize=7.5)
            elif key == 'S' and np.isfinite(y_lo):
                ax.text(xi + 0.15, S_MAX * 0.9, f'L/D {LD0:.1f}:\n> {S_MAX:.0f} m²', va='top',
                        fontsize=7, color=C_KITE)
        _style(ax, title, ylabel=unit)
        ax.set_xlim(0.4, N_CLASSES + 0.6)
    a1.set_ylim(0, S_MAX * 1.12)
    a1.axhline(S_MAX, color=C_GREY, ls=':', lw=1)
    a1.text(N_CLASSES + 0.55, S_MAX, f'search limit {S_MAX:.0f} m²', ha='right', va='bottom', fontsize=7.5,
            color=C_GREY)
    for xi, rb, rl in zip(x, base, alt[lo_LD]):
        if not rl['feasible'] and not rb['feasible']:
            a1.text(xi, 15, f'> {S_MAX:.0f} m²\neven at\nL/D {lo_LD:g}', ha='center', fontsize=7, color=C_GREY)
    from matplotlib.lines import Line2D
    a2.legend([Line2D([], [], marker='o', ls='', color=C_KITE),
               Line2D([], [], color=C_TETHER, lw=2.5)],
              [f'baseline: kite L/D {LD0:.1f}\n(measured reel-out average)',
               f'range: L/D {lo_LD:g} (measured peak)\nto {hi_LD:g} (lowest measured average);\n'
               f'arrow = above {S_MAX:.0f} m²'], frameon=False, fontsize=8, loc='upper left')
    a2.set_xticks(x, [short_label(c) for c in x])
    a2.set_xlabel(f'wind class (mean wind speed at {da.CLASS_HEIGHT} m, m/s)')
    fig.text(0.01, 0.005, _base_note() + '. Kite L/D: Oehler & Schmehl 2019; van der Vlugt et al. 2019. '
             'C_L,o fixed at 0.9, C_D,k,o varied.', fontsize=7.5, color=C_GREY)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return fig
