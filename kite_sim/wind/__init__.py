"""
kite_sim.wind - wind classes for the kite (and turbine) design
==============================================================

The overview is in data_analysis.py, which re-exports everything listed here.
"""

from .settings import (REPO, DATA_FILE, OUTPUT_DIR, HEIGHTS, EXTRA_HEIGHTS, DIST_HEIGHTS,
                       CLASS_HEIGHT, CLASS_EDGES, _class_names, CLASS_NAMES, BIN_WIDTH,
                       BIN_CENTRES, CELL_AREA_EQUATOR_KM2, ELEVATION_COLUMN)
from .classes import load_data, mean_speed_at, assign_classes, class_summary
from .distributions import (weibull_bin_probabilities, class_distributions, _extrapolate_weibull,
                            class_elevations, wind_bins, distributions_to_table)

__all__ = ['REPO', 'DATA_FILE', 'OUTPUT_DIR', 'HEIGHTS', 'EXTRA_HEIGHTS', 'DIST_HEIGHTS',
           'CLASS_HEIGHT', 'CLASS_EDGES', '_class_names', 'CLASS_NAMES', 'BIN_WIDTH',
           'BIN_CENTRES', 'CELL_AREA_EQUATOR_KM2', 'ELEVATION_COLUMN',
           'load_data', 'mean_speed_at', 'assign_classes', 'class_summary',
           'weibull_bin_probabilities', 'class_distributions', '_extrapolate_weibull',
           'class_elevations', 'wind_bins', 'distributions_to_table']
