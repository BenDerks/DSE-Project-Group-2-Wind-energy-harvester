"""
settings.py - Wind analysis settings: data file, heights, wind classes, wind-speed bins
=======================================================================================
"""

from pathlib import Path
import numpy as np


# =====================================================================================
# SETTINGS - change these, not the code below
# =====================================================================================
REPO = Path(__file__).resolve().parents[2]          # repository root
DATA_FILE = REPO / 'wind_results_2024.csv'
OUTPUT_DIR = REPO / 'outputs'

# Heights with Weibull parameters in the data [m above ground level]
HEIGHTS = [40, 100, 200, 300, 400]

# Heights above the data, filled in by extrapolation so flying heights up to 500 m (mean
# altitude up to 450 m) can be sized. Per point: A(z) = A400 * (z / 400)**alpha, with alpha
# the point's own 300-400 m shear exponent; k is kept at its 400 m value.
# Power law: Manwell, McGowan & Rogers (2009), Wind Energy Explained, 2nd ed., Wiley, ch. 2
# (wind speed variation with height). Assumption: the 300-400 m shear holds up to 500 m
# (median alpha 0.21; 99% within 0.05-0.29). Treat results above 400 m as an estimate.
EXTRA_HEIGHTS = [500]
DIST_HEIGHTS = HEIGHTS + EXTRA_HEIGHTS      # rows of the class distribution tables

# Wind classes by annual MEAN wind speed at CLASS_HEIGHT; edges in m/s. 100 m is a common
# reference for both concepts (turbine hub at 40 m, kite at 100-400 m). The first class has
# no lower, the last no upper limit. 10 classes of 0.5 m/s span the data: 8,486 points (cells
# with >= 25% windy land), 4.4-13.2 m/s at 100 m, median 6.5 m/s. 7.5 m/s is an edge (lower
# bound of IEC wind class III, the market-analysis threshold). Kite runs skip empty classes.
CLASS_HEIGHT = 100
CLASS_EDGES = [0.0, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, np.inf]


def _class_names(edges):
    """Readable class names from the edges, e.g. 'C3: 5.5-6.0 m/s'."""
    names = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:]), start=1):
        if lo == 0:
            names.append(f'C{i}: < {hi} m/s')
        elif np.isinf(hi):
            names.append(f'C{i}: >= {lo} m/s')
        else:
            names.append(f'C{i}: {lo}-{hi} m/s')
    return names


CLASS_NAMES = _class_names(CLASS_EDGES)

# Wind-speed bins for the kite model: centres 0.5, 1.5, ... 34.5 m/s, width 1 m/s
# (probability above 35 m/s is negligible).
BIN_WIDTH = 1.0
BIN_CENTRES = np.arange(0.5, 35.0, BIN_WIDTH)

# ERA5 cell area at the equator, 0.25 deg grid [km2]; scales with cos(latitude).
# Check: equals the largest GWA area in the data.
CELL_AREA_EQUATOR_KM2 = 772.8

# Column with the ground elevation above sea level [m], for the air density at the kite's
# altitude (kite_model.geometry, z_site). Missing column: sea level is assumed, with a warning.
# Rename here if the data uses another name.
ELEVATION_COLUMN = 'elevation_m'
