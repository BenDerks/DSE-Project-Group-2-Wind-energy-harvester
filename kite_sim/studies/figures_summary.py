"""
figures_summary.py - Final summary: the best design per wind class, and the trade-off behind it
===============================================================================================

Which designs compete and how the best one is chosen depends on the objective (selection.py):
    area   smallest kite area                          -> RESULT_DIRS['area']
    multi  weighted score of kite area, downtime and
           longest continuous downtime                 -> RESULT_DIRS['multi']

Output in the objective's folder: best_designs.png + best_designs.csv, and for 'multi' also
tradeoff.png + all_designs.csv (every candidate with its goals, score and Pareto flag).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .. import wind as da
from .settings import BASE, C_GREY, C_KITE, C_TETHER, OBJECTIVE_WEIGHTS, RESULT_DIRS, RUN, S_MAX
from .cases import N_CLASSES, _key, cases_per_class
from .selection import best_multi, candidate_cases, label
from .style import _style, short_label


def cases_best():
    """Design cases that compete for the best design, for the current objective."""
    return candidate_cases(RUN['objective'])


def best_designs(res, df=None):
    """
    One row per wind class: the baseline kite area and the best design with its metrics.
    Objective 'area': smallest kite area. Objective 'multi': see selection.best_multi().
    Output: DataFrame; design columns are NaN for classes where no design reaches the target.
    """
    if RUN['objective'] == 'multi':
        return best_multi(res, df)[0]
    shares = da.class_summary(df).set_index('wind_class')['share_points_pct'] if df is not None else {}
    cases = cases_best()
    rows = []
    for c in range(1, N_CLASSES + 1):
        base = res[_key(cases_per_class()[c - 1])]
        options = [res[_key(k)] for k in cases if k['cls'] == c]
        feasible = [r for r in options if r['feasible']]
        row = dict(wind_class=da.CLASS_NAMES[c - 1], share_cells_pct=shares.get(c, np.nan),
                   designs_compared=len(options),
                   S_baseline_m2=base['S'] if base['feasible'] else np.nan)
        if feasible:
            b = min(feasible, key=lambda r: r['S'])
            row.update(z_min_m=b['case']['zmin'], z_max_m=b['case']['zmax'],
                       F_max_kN=b['case']['Fmax'] / 1e3, S_m2=b['S'], A_flat_m2=b['A'],
                       kite_mass_kg=b['m'], tether_mass_kg=b['mt_tot'],
                       total_mass_kg=b['m'] + b['mt_tot'], tether_diameter_mm=b['dt'] * 1e3,
                       tether_length_m=b['rmax'], v_in_m_s=b['v_in'], v_out_m_s=b['v_out'],
                       downtime_h_yr=b['T_out'], capacity_factor=b['CF'])
        rows.append(row)
    return pd.DataFrame(rows)


def fig_best_designs(res, df=None):
    """Final summary: kite area of the baseline and the best design per class, and a table."""
    multi = RUN['objective'] == 'multi'
    t = best_designs(res, df)
    out = RESULT_DIRS[RUN['objective']]
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / 'best_designs.csv', index=False)

    x = np.arange(1, N_CLASSES + 1)
    fig = plt.figure(figsize=(14 if multi else 13, 9))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.15, 1], hspace=0.28)
    ax, tab = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])

    base_label = (f'baseline design ({BASE["zmin"]:.0f}-{BASE["zmax"]:.0f} m, '
                  f'{BASE["Fmax"] / 1e3:.0f} kN)')
    ax.bar(x - 0.2, t['S_baseline_m2'], 0.4, color=C_TETHER, label=base_label)
    ax.bar(x + 0.2, t['S_m2'], 0.4, color=C_KITE,
           label='chosen design (weighted score)' if multi else 'best design studied')
    for xi, sb, s in zip(x, t['S_baseline_m2'], t['S_m2']):
        if np.isfinite(s):
            ax.text(xi + 0.2, s + 3, f'{s:.0f}', ha='center', va='bottom', fontsize=8,
                    fontweight='bold')
        if np.isfinite(sb):
            ax.text(xi - 0.2, sb + 3, f'{sb:.0f}', ha='center', va='bottom', fontsize=7.5,
                    color=C_GREY)
        else:
            ax.text(xi - 0.2, 6, f'> {S_MAX:.0f}', ha='center', va='bottom', fontsize=7,
                    color=C_GREY, rotation=90)
    ax.axhline(S_MAX, color=C_GREY, ls=':', lw=1)
    ax.text(N_CLASSES + 0.55, S_MAX, f'search limit {S_MAX:.0f} m²', ha='right', va='bottom',
            fontsize=7.5, color=C_GREY)
    ax.set_xlim(0.4, N_CLASSES + 0.6)
    ax.set_ylim(0, S_MAX * 1.12)
    ax.set_xticks(x, [short_label(c) for c in x], fontsize=8)
    w = OBJECTIVE_WEIGHTS
    title = (f'Best design per wind class: weighted score of kite area ({w["area"]:.2f}), downtime '
             f'({w["downtime"]:.2f}) and longest outage ({w["longest_downtime"]:.2f})' if multi else
             f'Best design per wind class: smallest kite for {BASE["P_target"] / 1e3:.0f} kW mean power')
    _style(ax, title, f'wind class (mean wind speed at {da.CLASS_HEIGHT} m, m/s)',
           'projected kite area S [m²]')
    ax.legend(frameon=False, fontsize=8.5, loc='center right')

    def fmt(v, f='{:.0f}'):
        return f.format(v) if np.isfinite(v) else '-'

    cols = ['Class', 'Share of\ncells [%]', 'Flying\nheights [m]', 'F_max\n[kN]',
            'Kite area\nS [m²]', 'vs\nbaseline', 'Kite + tether\nmass [kg]', 'Tether\nØ [mm]',
            'Cut-in\n[m/s]', 'Downtime\n[h/yr]'] + (['Longest\noutage [h]'] if multi else [])
    cells = []
    for c, r in zip(x, t.itertuples()):
        found = np.isfinite(getattr(r, 'S_m2', np.nan))
        gain = (f'{100 * (r.S_m2 / r.S_baseline_m2 - 1):+.0f}%'
                if found and np.isfinite(r.S_baseline_m2) else '-')
        cells.append([short_label(c).replace('\n', '  '), fmt(r.share_cells_pct, '{:.1f}'),
                      f'{r.z_min_m:.0f}-{r.z_max_m:.0f}' if found else 'no design',
                      fmt(r.F_max_kN), fmt(r.S_m2), gain, fmt(r.total_mass_kg),
                      fmt(r.tether_diameter_mm, '{:.1f}'), fmt(r.v_in_m_s, '{:.1f}'),
                      fmt(r.downtime_h_yr, '{:,.0f}')]
                     + ([fmt(r.longest_downtime_h)] if multi else []))
    tab.axis('off')
    table = tab.table(cellText=cells, colLabels=cols, loc='upper center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.55)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor('#e5e5e5')
        if row == 0:
            cell.set_text_props(fontweight='bold')
            cell.set_height(cell.get_height() * 1.5)
    for row, r in enumerate(t.itertuples(), start=1):      # highlight the chosen kite area
        table[row, 4].set_text_props(fontweight='bold', color=C_KITE)

    fig.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.03)
    return fig


def fig_tradeoff(res, df=None):
    """
    Objective 'multi': kite area against longest continuous downtime for every candidate, one
    panel per class. Pareto-optimal candidates are highlighted; the chosen design is starred.
    """
    t, d = best_multi(res, df)
    out = RESULT_DIRS['multi']
    out.mkdir(parents=True, exist_ok=True)
    d.drop(columns='key').to_csv(out / 'all_designs.csv', index=False)

    fig, axs = plt.subplots(2, 5, figsize=(18, 7.4), squeeze=False)
    axs = axs.ravel()
    for i, (ax, c) in enumerate(zip(axs, range(1, N_CLASSES + 1))):
        g = d[d['cls'] == c]
        ax.scatter(g['S_m2'], g['longest_downtime_h'], s=14, color='#cfd8e3', label='other candidates')
        p = g[g['pareto']].sort_values('S_m2')
        ax.plot(p['S_m2'], p['longest_downtime_h'], 'o-', color=C_KITE, ms=4.5, lw=1.2,
                label='Pareto-optimal (no design better on all three goals)')
        if len(g):
            b = g.loc[g['score'].idxmin()]
            ax.plot(b['S_m2'], b['longest_downtime_h'], '*', color='#d97706', ms=15, zorder=4,
                    label='chosen design (lowest weighted score)')
            ax.annotate(f"{label(b)}\n{b['downtime_h_yr']:,.0f} h/yr down",
                        (b['S_m2'], b['longest_downtime_h']), (6, 6), textcoords='offset points',
                        fontsize=7)
        _style(ax, da.CLASS_NAMES[c - 1], 'kite area S [m²]' if i >= 5 else None,
               'longest outage [h]' if i % 5 == 0 else None)
        ax.title.set_fontsize(10)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig
