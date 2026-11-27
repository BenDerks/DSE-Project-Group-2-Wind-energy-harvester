"""
kite_sim.studies - kite design studies: wind classes -> kite sizing -> figures
==============================================================================

Overview (figures, how to run, run time) in plotting.py, which re-exports everything here.
"""

import matplotlib
matplotlib.use('Agg')                       # files only, no window

from .settings import (REPO, FIG_DIR, CACHE_FILE, BASE, S_MAX, USE_SITE_ELEVATION,  # noqa: E402
                       SITE_ELEVATION, HEIGHT_ZMIN, HEIGHT_STROKES, HEIGHT_ZMAX_LIMIT,
                       HEIGHT_CLASSES, LD_VALUES, LD_CLASSES, FMAX_VALUES, FMAX_CLASSES,
                       SENS_CLASS, SENSITIVITY, DISPLAY_SCALE, WORKERS,
                       C_KITE, C_TETHER, C_GREY, C_LINES)
from .cases import (make_case, _code_tag, _key, load_cache, save_cache, N_CLASSES,  # noqa: E402
                    cases_per_class, height_windows, cases_height, ld_case, cases_ld, cases_fmax,
                    cases_sensitivity)
from .runner import run_case, run_cases                                              # noqa: E402
from .style import short_label, _style, _mark_infeasible, _base_note                # noqa: E402
from .figures_per_class import fig_size_mass_per_class, fig_power_curves, fig_wind_classes  # noqa: E402
from .figures_studies import (fig_size_vs_height, fig_size_vs_ld, fig_size_vs_fmax,  # noqa: E402
                              fig_sensitivity)
from .cli import FIGURES, main                                                        # noqa: E402

__all__ = ['REPO', 'FIG_DIR', 'CACHE_FILE', 'BASE', 'S_MAX', 'USE_SITE_ELEVATION',
           'SITE_ELEVATION', 'HEIGHT_ZMIN', 'HEIGHT_STROKES', 'HEIGHT_ZMAX_LIMIT', 'HEIGHT_CLASSES',
           'LD_VALUES', 'LD_CLASSES', 'FMAX_VALUES', 'FMAX_CLASSES', 'SENS_CLASS', 'SENSITIVITY',
           'DISPLAY_SCALE', 'WORKERS', 'C_KITE', 'C_TETHER', 'C_GREY', 'C_LINES',
           'make_case', '_code_tag', '_key', 'load_cache', 'save_cache', 'N_CLASSES',
           'cases_per_class', 'height_windows', 'cases_height', 'ld_case', 'cases_ld', 'cases_fmax',
           'cases_sensitivity', 'run_case', 'run_cases', 'short_label', '_style', '_mark_infeasible',
           '_base_note', 'fig_size_mass_per_class', 'fig_power_curves', 'fig_wind_classes',
           'fig_size_vs_height', 'fig_size_vs_ld', 'fig_size_vs_fmax', 'fig_sensitivity',
           'FIGURES', 'main']
