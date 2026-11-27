"""
selection.py - Which designs compete, and which one is best per wind class
==========================================================================

Objective 'area'   candidates: the one-at-a-time studies (baseline, F_max study, every height
                   window) for every class, F_max at most FMAX_LIMIT. Best = smallest kite area.
Objective 'multi'  candidates: every flying-height window x every F_max (at most FMAX_LIMIT).
                   Three goals per candidate, all to be minimised:
                       kite area S [m2]
                       downtime [h/yr]                    wind below cut-in or above cut-out
                       longest continuous downtime [h]    longest stretch of consecutive down
                                                          hours in 2024, median over the cells
                   (downtime goals from the hourly wind: downtime.py). Each goal is scored as the
                   fraction worse than the best candidate of the class (0 = best, 0.2 = 20% worse);
                   the weighted sum with OBJECTIVE_WEIGHTS picks the design. A candidate is
                   Pareto-optimal when no other candidate is at least as good on all three goals
                   and better on one.
"""

import numpy as np
import pandas as pd

from .. import wind as da
from .settings import FMAX_LIMIT, OBJECTIVE_WEIGHTS
from .cases import N_CLASSES, _key, cases_fmax, cases_grid, cases_per_class, height_windows, make_case
from .downtime import downtime_metrics

GOALS = dict(area='S_m2', downtime='downtime_h_yr', longest_downtime='longest_downtime_h')


def candidate_cases(objective):
    """The design cases that compete for the best design of each class."""
    if objective == 'multi':
        return cases_grid()
    heights = [make_case(c, zmin=a, zmax=b) for c in range(1, N_CLASSES + 1)
               for a, b in height_windows()]
    cases = []
    for c in cases_per_class() + cases_fmax() + heights:
        if c not in cases and c['Fmax'] <= FMAX_LIMIT:
            cases.append(c)
    return cases


def label(row):
    """Short design label, e.g. '150-300 m, 50 kN'."""
    return f"{row['z_min_m']:.0f}-{row['z_max_m']:.0f} m, {row['F_max_kN']:.0f} kN"


def candidates_multi(res, df):
    """
    Every feasible candidate of objective 'multi' with its metrics, goal scores, total score and
    Pareto flag. Output: DataFrame, one row per candidate (column 'key' = case key).
    """
    rs = [res[_key(c)] for c in candidate_cases('multi')]
    rows = [dict(key=_key(r['case']), cls=r['case']['cls'], z_min_m=r['case']['zmin'],
                 z_max_m=r['case']['zmax'], F_max_kN=r['case']['Fmax'] / 1e3, S_m2=r['S'],
                 kite_mass_kg=r['m'], tether_mass_kg=r['mt_tot'], total_mass_kg=r['m'] + r['mt_tot'],
                 tether_diameter_mm=r['dt'] * 1e3, tether_length_m=r['rmax'], v_in_m_s=r['v_in'],
                 v_out_m_s=r['v_out'], downtime_model_h_yr=r['T_out'], capacity_factor=r['CF'])
            for r in rs if r['feasible']]
    d = pd.DataFrame(rows)
    d = d.join(downtime_metrics(rs, df).drop(columns='cls'), on='key')
    d['score'] = 0.0
    for goal, col in GOALS.items():
        d[f'{goal}_vs_best'] = d[col] / d.groupby('cls')[col].transform('min') - 1
        d['score'] += OBJECTIVE_WEIGHTS[goal] * d[f'{goal}_vs_best']
    d['pareto'] = False
    for _, g in d.groupby('cls'):
        x = g[list(GOALS.values())].to_numpy()
        for i, idx in enumerate(g.index):
            beaten = np.all(x <= x[i], axis=1) & np.any(x < x[i], axis=1)
            d.loc[idx, 'pareto'] = not beaten.any()
    d['design'] = d.apply(label, axis=1)
    return d


def best_multi(res, df):
    """
    One row per wind class: the design with the lowest score, and for comparison the best
    design for each goal on its own. Output: (table, all candidates).
    """
    d = candidates_multi(res, df)
    shares = da.class_summary(df).set_index('wind_class')['share_points_pct']
    rows = []
    for c in range(1, N_CLASSES + 1):
        base = res[_key(cases_per_class()[c - 1])]
        g = d[d['cls'] == c]
        row = dict(wind_class=da.CLASS_NAMES[c - 1], share_cells_pct=shares.get(c, np.nan),
                   designs_compared=len(g), S_baseline_m2=base['S'] if base['feasible'] else np.nan)
        if len(g):
            b = g.loc[g['score'].idxmin()]
            row.update({k: b[k] for k in ['z_min_m', 'z_max_m', 'F_max_kN', 'S_m2', 'total_mass_kg',
                                          'tether_diameter_mm', 'tether_length_m', 'v_in_m_s',
                                          'downtime_h_yr', 'longest_downtime_h',
                                          'longest_downtime_p90_h', 'mean_downtime_spell_h',
                                          'spells_ge_24h_per_yr', 'score']})
            for goal, col in GOALS.items():
                o = g.loc[g[col].idxmin()]
                row[f'best_for_{goal}'] = o['design']
                row[f'best_for_{goal}_value'] = o[col]
        rows.append(row)
    return pd.DataFrame(rows), d
