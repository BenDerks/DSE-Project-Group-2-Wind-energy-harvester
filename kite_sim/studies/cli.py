"""
cli.py - The figure list and main() of  python plotting.py
==========================================================
"""

import argparse

import matplotlib.pyplot as plt

from .. import wind as da
from .settings import FIG_DIR, SITE_ELEVATION, WORKERS
from .cases import cases_fmax, cases_height, cases_ld, cases_per_class, cases_sensitivity
from .runner import run_cases
from .figures_per_class import fig_power_curves, fig_size_mass_per_class, fig_wind_classes
from .figures_studies import fig_sensitivity, fig_size_vs_fmax, fig_size_vs_height, fig_size_vs_ld


FIGURES = {
    'size_mass_per_class': (cases_per_class, fig_size_mass_per_class),
    'size_vs_height': (cases_height, fig_size_vs_height),
    'size_vs_ld': (cases_ld, fig_size_vs_ld),
    'size_vs_fmax': (cases_fmax, fig_size_vs_fmax),
    'power_curves': (cases_per_class, fig_power_curves),
    'sensitivity': (cases_sensitivity, fig_sensitivity),
    'wind_classes': (lambda: [], fig_wind_classes),
}


# =====================================================================================
# MAIN
# =====================================================================================
def main():
    parser = argparse.ArgumentParser(description='Kite design studies and figures')
    parser.add_argument('figures', nargs='*', default=list(FIGURES),
                        help=f'figures to make (default all): {", ".join(FIGURES)}')
    parser.add_argument('--workers', type=int, default=WORKERS, help='parallel processes')
    parser.add_argument('--fresh', action='store_true', help='ignore the cache and recompute')
    args = parser.parse_args()

    # Wind classes from the data (data_analysis.py)
    df = da.assign_classes(da.load_data())
    dist = da.class_distributions(df)
    SITE_ELEVATION.update(da.class_elevations(df))

    # Cases for the chosen figures; shared cases are run once
    cases = []
    for name in args.figures:
        for c in FIGURES[name][0]():
            if c not in cases:
                cases.append(c)
    res = run_cases(cases, dist, workers=args.workers, fresh=args.fresh)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for name in args.figures:
        fig = FIGURES[name][1](res, df)
        path = FIG_DIR / f'{name}.png'
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f'saved {path}')
