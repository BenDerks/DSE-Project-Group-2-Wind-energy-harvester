"""
kite_sim.studies - kite design studies: wind classes -> kite sizing -> figures
==============================================================================

Overview (figures, how to run, run time) in plotting.py, which re-exports everything here.
"""

import matplotlib
matplotlib.use('Agg')                       # files only, no window

from .settings import (REPO, FIG_DIR, CACHE_FILE, BASE, S_MAX, USE_SITE_ELEVATION,  # noqa: E402
                       SITE_ELEVATION, LD_RANGE, HEIGHT_CLASSES, FMAX_CLASSES, SENS_CLASS,
                       HEIGHT_WINDOWS, FMAX_VALUES, SENSITIVITY, DISPLAY_SCALE, WORKERS,
                       C_KITE, C_TETHER, C_GREY, C_LINES)
from .cases import (make_case, _code_tag, _key, load_cache, save_cache, N_CLASSES,  # noqa: E402
                    cases_per_class, cases_ld_range, cases_height, cases_fmax, cases_sensitivity)
from .runner import run_case, run_cases                                              # noqa: E402
from .style import short_label, _style, _mark_infeasible, _base_note                # noqa: E402
from .figures_per_class import (fig_size_per_class, fig_downtime_per_class,         # noqa: E402
                                fig_power_curves, fig_stability_cf, fig_wind_classes,
                                fig_class_overview)
from .figures_studies import (fig_size_vs_height, fig_fmax_effect, fig_sensitivity,  # noqa: E402
                              fig_size_range_ld)
from .cli import FIGURES, main                                                        # noqa: E402

__all__ = ['REPO', 'FIG_DIR', 'CACHE_FILE', 'BASE', 'S_MAX', 'USE_SITE_ELEVATION',
           'SITE_ELEVATION', 'LD_RANGE', 'HEIGHT_CLASSES', 'FMAX_CLASSES', 'SENS_CLASS',
           'HEIGHT_WINDOWS', 'FMAX_VALUES', 'SENSITIVITY', 'DISPLAY_SCALE', 'WORKERS',
           'C_KITE', 'C_TETHER', 'C_GREY', 'C_LINES',
           'make_case', '_code_tag', '_key', 'load_cache', 'save_cache', 'N_CLASSES',
           'cases_per_class', 'cases_ld_range', 'cases_height', 'cases_fmax', 'cases_sensitivity',
           'run_case', 'run_cases', 'short_label', '_style', '_mark_infeasible', '_base_note',
           'fig_size_per_class', 'fig_downtime_per_class', 'fig_power_curves', 'fig_stability_cf',
           'fig_wind_classes', 'fig_class_overview', 'fig_size_vs_height', 'fig_fmax_effect',
           'fig_sensitivity', 'fig_size_range_ld', 'FIGURES', 'main']
