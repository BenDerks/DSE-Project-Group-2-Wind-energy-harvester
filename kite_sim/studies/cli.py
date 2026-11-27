"""
cli.py - The figure list and main() of  python plotting.py
==========================================================
"""

import argparse

import matplotlib.pyplot as plt

from .. import wind as da
from .settings import FIG_DIR, SITE_ELEVATION, WORKERS
from .cases import cases_fmax, cases_height, cases_ld_range, cases_per_class, cases_sensitivity
from .runner import run_cases
from .figures_per_class import (fig_class_overview, fig_downtime_per_class, fig_power_curves,
                                fig_size_per_class, fig_stability_cf, fig_wind_classes)
from .figures_studies import fig_fmax_effect, fig_sensitivity, fig_size_range_ld, fig_size_vs_height


FIGURES = {
    'size_per_class': (cases_per_class, fig_size_per_class),
    'size_vs_height': (cases_height, fig_size_vs_height),
    'downtime_per_class': (cases_per_class, fig_downtime_per_class),
    'power_curves': (cases_per_class, fig_power_curves),
    'fmax_effect': (cases_fmax, fig_fmax_effect),
    'sensitivity': (cases_sensitivity, fig_sensitivity),
    'stability_cf': (cases_per_class, fig_stability_cf),
    'wind_classes': (lambda: [], fig_wind_classes),
    'size_range_ld': (cases_ld_range, fig_size_range_ld),
    'class_overview': (cases_per_class, fig_class_overview),
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
