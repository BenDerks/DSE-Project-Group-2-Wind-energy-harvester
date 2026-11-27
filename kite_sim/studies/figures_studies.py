"""
figures_studies.py - Figures of the height, kite L/D, F_max and sensitivity studies
===================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory

from .. import model as km
from .. import wind as da
from .settings import (BASE, C_GREY, C_KITE, C_LINES, C_TETHER, DISPLAY_SCALE, FMAX_CLASSES,
                       FMAX_VALUES, HEIGHT_CLASSES, HEIGHT_STROKES, HEIGHT_ZMAX_LIMIT, HEIGHT_ZMIN,
                       LD_CLASSES, LD_VALUES, S_MAX, SENS_CLASS, SENSITIVITY)
from .cases import N_CLASSES, _key, ld_case, make_case
from .style import _base_note, _style, short_label


# =====================================================================================
# FIGURES
# =====================================================================================
def _class_lines(ax, xs, rows_per_class):
    """
    One line per wind class (feasible points only). Returns the classes without any
    feasible point, so the figure can list them.
    """
    cmap = plt.get_cmap('viridis')
    none = []
    for cls, rows in rows_per_class:
        S = np.array([r['S'] if r['feasible'] else np.nan for r in rows])
        if np.all(np.isnan(S)):
            none.append(f'C{cls}')
            continue
        col = cmap(0.1 + 0.8 * (cls - 1) / max(1, N_CLASSES - 1))
        ax.plot(xs, S, 'o-', color=col, ms=4, label=da.CLASS_NAMES[cls - 1])
    return none


def fig_size_vs_height(res, df=None):
    """
    Study 2: kite area against the lowest flying height z_min, one line per stroke
    (z_max - z_min), one panel per class in HEIGHT_CLASSES. The baseline is marked.
    """
    fig, axs = plt.subplots(1, len(HEIGHT_CLASSES), figsize=(5 * len(HEIGHT_CLASSES), 4.6), sharey=True)
    axs = np.atleast_1d(axs)
    cols = ['#9ecae1', '#2b6cb0', '#08306b']
    for ax, cls in zip(axs, HEIGHT_CLASSES):
        for d, col in zip(HEIGHT_STROKES, cols):
            zs = [a for a in HEIGHT_ZMIN if a + d <= HEIGHT_ZMAX_LIMIT]
            rows = [res[_key(make_case(cls, zmin=a, zmax=a + d))] for a in zs]
            S = [r['S'] if r['feasible'] else np.nan for r in rows]
            ax.plot(zs, S, 'o-', color=col, label=f'stroke {d} m (z_max = z_min + {d} m)')
        base = res[_key(make_case(cls))]
        if base['feasible']:
            ax.plot(BASE['zmin'], base['S'], 's', color='#d97706', ms=8, zorder=3,
                    label=f"baseline {BASE['zmin']:.0f}-{BASE['zmax']:.0f} m")
        _style(ax, da.CLASS_NAMES[cls - 1], 'lowest flying height z_min [m]',
               'projected kite area S [m²]' if cls == HEIGHT_CLASSES[0] else None)
        ax.set_xticks(HEIGHT_ZMIN)
    axs[0].legend(frameon=False, fontsize=8)
    fig.text(0.01, 0.01, f'F_max {BASE["Fmax"] / 1e3:.0f} kN, target {BASE["P_target"] / 1e3:.0f} kW mean, '
             f'elevation 25° (tether length = 2.4 x height), z_max at most {HEIGHT_ZMAX_LIMIT} m. '
             f'Missing points: no kite up to {S_MAX:.0f} m² reaches the target.', fontsize=8, color=C_GREY)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def fig_size_vs_ld(res, df=None):
    """Study 3: kite area against the kite lift-to-drag ratio (sources in LD_VALUES), per class."""
    LDs = list(LD_VALUES)
    fig, ax = plt.subplots(figsize=(11, 5.8))
    none = _class_lines(ax, LDs, [(c, [res[_key(ld_case(c, LD))] for LD in LDs]) for c in LD_CLASSES])
    for LD in LDs:
        ax.axvline(LD, color='#eeeeee', lw=0.8, zorder=0)
    ax.set_xticks(LDs, [f'{LD:g}' for LD in LDs])
    ax.set_xlim(min(LDs) - 0.15, max(LDs) + 0.5)
    _style(ax, 'Kite area against kite lift-to-drag ratio', 'kite L/D (C_L,o = 0.9 fixed) [-]',
           'projected kite area S [m²]')
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc='upper right')
    src = '   '.join(f'{LD:g}: {txt}' for LD, txt in LD_VALUES.items())
    note = _base_note() + f'. Not feasible up to {S_MAX:.0f} m² at any L/D: {", ".join(none) or "none"}.'
    fig.text(0.01, 0.055, note, fontsize=7.5, color=C_GREY)
    fig.text(0.01, 0.012, 'Sources: ' + src, fontsize=6.5, color=C_GREY, wrap=True)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    return fig


def fig_size_vs_fmax(res, df=None):
    """Study 4: kite area against the maximum tether force, one line per class."""
    F = np.array(FMAX_VALUES) / 1e3
    fig, ax = plt.subplots(figsize=(11, 5.6))
    none = _class_lines(ax, F, [(c, [res[_key(make_case(c, Fmax=f))] for f in FMAX_VALUES])
                                for c in FMAX_CLASSES])
    top = ax.get_ylim()[1]
    ax.axvline(BASE['Fmax'] / 1e3, color=C_GREY, ls=':', lw=1)
    ax.text(BASE['Fmax'] / 1e3, top, ' baseline', fontsize=7.5, color=C_GREY, va='top')
    ax.axvline(34.3, color='#d97706', ls=':', lw=1)
    ax.text(34.3, top * 0.93, 'Kitepower\nFalcon 3.5 t ', fontsize=7.5, color='#d97706', va='top', ha='right')
    ax.set_xticks(F)
    ax.set_xlim(F.min() - 3, F.max() + 10)
    _style(ax, 'Kite area against maximum tether force', 'maximum tether force F_max [kN]',
           'projected kite area S [m²]')
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc='upper right')
    fig.text(0.01, 0.01, f'Heights {BASE["zmin"]:.0f}-{BASE["zmax"]:.0f} m, target {BASE["P_target"] / 1e3:.0f} kW '
             f'mean; tether diameter follows from F_max (Joshi et al. 2024). Not feasible up to {S_MAX:.0f} m² '
             f'at any F_max: {", ".join(none) or "none"}.', fontsize=8, color=C_GREY)
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


