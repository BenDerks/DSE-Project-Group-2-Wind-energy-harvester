"""
cli.py - The figure list and main() of  python plotting.py
==========================================================
"""

import argparse
import time

import matplotlib.pyplot as plt

from .. import wind as da
from .settings import RESULT_DIRS, RUN, SITE_ELEVATION, WORKERS
from .cases import cases_fmax, cases_height, cases_ld, cases_per_class, cases_sensitivity
from .runner import run_cases
from .figures_per_class import fig_power_curves, fig_wind_classes
from .figures_studies import fig_sensitivity, fig_size_vs_fmax, fig_size_vs_height, fig_size_vs_ld
from .figures_summary import cases_best, fig_best_designs, fig_tradeoff


FIGURES = {
    'size_vs_height': (cases_height, fig_size_vs_height),
    'size_vs_ld': (cases_ld, fig_size_vs_ld),
    'size_vs_fmax': (cases_fmax, fig_size_vs_fmax),
    'power_curves': (cases_per_class, fig_power_curves),
    'sensitivity': (cases_sensitivity, fig_sensitivity),
    'wind_classes': (lambda: [], fig_wind_classes),
    'best_designs': (cases_best, fig_best_designs),
    'tradeoff': (cases_best, fig_tradeoff),
}
MULTI_ONLY = {'tradeoff'}                   # figures that only exist for --objective multi


def save(fig, path, tries=5):
    """Save a figure; retry briefly when another program holds the file (OSError on Windows)."""
    for i in range(tries):
        try:
            fig.savefig(path, dpi=150)
            return
        except OSError:
            if i == tries - 1:
                raise
            time.sleep(1.0)


# =====================================================================================
# MAIN
# =====================================================================================
def main():
    parser = argparse.ArgumentParser(description='Kite design studies and figures')
    parser.add_argument('figures', nargs='*',
                        help=f'figures to make (default all): {", ".join(FIGURES)}')
    parser.add_argument('--objective', choices=list(RESULT_DIRS), default='area',
                        help='best design: area = smallest kite; multi = weighted score of kite '
                             'area, downtime and longest outage (see settings.py)')
    parser.add_argument('--workers', type=int, default=WORKERS, help='parallel processes')
    parser.add_argument('--fresh', action='store_true', help='ignore the cache and recompute')
    args = parser.parse_args()
    RUN['objective'] = args.objective
    names = args.figures or [f for f in FIGURES if args.objective == 'multi' or f not in MULTI_ONLY]

    # Wind classes from the data (data_analysis.py)
    df = da.assign_classes(da.load_data())
    dist = da.class_distributions(df)
    SITE_ELEVATION.update(da.class_elevations(df))

    # Cases for the chosen figures; shared cases are run once
    cases = []
    for name in names:
        for c in FIGURES[name][0]():
            if c not in cases:
                cases.append(c)
    res = run_cases(cases, dist, workers=args.workers, fresh=args.fresh)

    out = RESULT_DIRS[args.objective]
    out.mkdir(parents=True, exist_ok=True)
    for name in names:
        fig = FIGURES[name][1](res, df)
        save(fig, out / f'{name}.png')
        plt.close(fig)
        print(f'saved {out / name}.png')
