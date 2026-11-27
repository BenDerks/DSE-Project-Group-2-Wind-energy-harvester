"""
design_impact.py - What the verification changes do to the kite design
======================================================================

Sizes the kite for every wind class (baseline design inputs of plotting.py) with the first
version of kite_model (the three verification options off) and with the current version, and
plots kite area, mass and hours out of operation side by side. Uses plotting.py's case cache,
so cases it already ran are not repeated.

HOW TO RUN
    python verification/design_impact.py [--workers N]

OUTPUT
    outputs/verification/design_impact.png and design_impact.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import data_analysis as da
import plotting as pl

OUT = HERE.parent / 'outputs' / 'verification'

# The three options added during the verification, switched off (floats, as make_case wants)
FIRST_VERSION = dict(transition_energy=0.0, depower=0.0, optimise_elevation=0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=pl.WORKERS)
    args = ap.parse_args()

    df = da.assign_classes(da.load_data())
    dist = da.class_distributions(df)
    pl.SITE_ELEVATION.update(da.class_elevations(df))

    classes = list(range(1, pl.N_CLASSES + 1))
    new_cases = [pl.make_case(c) for c in classes]
    old_cases = [pl.make_case(c, **FIRST_VERSION) for c in classes]
    res = pl.run_cases(new_cases + old_cases, dist, workers=args.workers)
    new = [res[pl._key(c)] for c in new_cases]
    old = [res[pl._key(c)] for c in old_cases]

    rows = []
    for c, rn, ro in zip(classes, new, old):
        rows.append(dict(wind_class=da.CLASS_NAMES[c - 1],
                         S_first_m2=ro['S'], S_now_m2=rn['S'],
                         mass_first_kg=ro['m'] + ro['mt_tot'], mass_now_kg=rn['m'] + rn['mt_tot'],
                         T_out_first_h=ro['T_out'], T_out_now_h=rn['T_out'],
                         change_S_pct=100 * (rn['S'] / ro['S'] - 1)))
    table = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    table.round(2).to_csv(OUT / 'design_impact.csv', index=False)
    print(table.round(1).to_string(index=False))

    x = np.arange(1, len(classes) + 1)
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, (k_old, k_new, title, unit) in zip(axs, (
            ('S_first_m2', 'S_now_m2', 'Kite area for 30 kW mean', 'projected area S [m²]'),
            ('mass_first_kg', 'mass_now_kg', 'Airborne + tether mass', 'mass [kg]'),
            ('T_out_first_h', 'T_out_now_h', 'Hours out of operation', 'hours per year'))):
        ax.bar(x - 0.2, table[k_old], 0.4, color=pl.C_TETHER, label='first version')
        ax.bar(x + 0.2, table[k_new], 0.4, color=pl.C_KITE, label='after verification')
        ax.set_xticks(x, [pl.short_label(c) for c in classes], fontsize=7)
        pl._style(ax, title, 'wind class', unit)
    pl._mark_infeasible(axs[0], x, new)
    axs[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / 'design_impact.png', dpi=150)
    print(f'saved {OUT / "design_impact.png"}')


if __name__ == '__main__':
    main()
