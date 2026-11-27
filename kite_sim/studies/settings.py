"""
settings.py - Design study settings: baseline design, study ranges, colours
===========================================================================
"""

import os
from pathlib import Path


# =====================================================================================
# SETTINGS - change these to run other studies
# =====================================================================================
REPO = Path(__file__).resolve().parents[2]          # repository root
FIG_DIR = REPO / 'outputs' / 'figures'
CACHE_FILE = REPO / 'outputs' / 'kite_cache.pkl'

# Baseline design; each study changes one thing [m, m, N, W]
BASE = dict(zmin=100.0, zmax=200.0, Fmax=40e3, P_target=30e3)

# Largest kite area the sizing loop may use [m2]. A search bound for now; should become S_max
# from the volume budget. Classes that need more are shown as 'not feasible'.
S_MAX = 300.0

# Air density at the site's ground elevation (True) or at sea level (False)
USE_SITE_ELEVATION = True

# Mean ground elevation per class [m]; filled by main() from the data, read by make_case()
SITE_ELEVATION = {}

# Kite L/D range shown per class [-] (C_L,o fixed, C_D,k,o varied). Baseline 4.0 (kite_model.P);
# 5.0 = measured in-flight peak, TU Delft V3 kite (Oehler & Schmehl 2019);
# 3.6 = measured traction average (van der Vlugt et al. 2019)
LD_RANGE = [3.6, 5.0]

# Classes for the height, F_max and sensitivity studies. With the current CLASS_EDGES
# (data_analysis.py), at 100 m: C2 5.0-5.5, C3 5.5-6.0, C4 6.0-6.5, C5 6.5-7.0, C6 7.0-7.5,
# C7 7.5-8.0, C8 8.0-8.5 m/s
HEIGHT_CLASSES = [2, 4, 6, 8]
FMAX_CLASSES = [3, 5, 7]
SENS_CLASS = 7

# Height study windows (z_min, z_max) [m], all with a 100 m stroke. Wind above 400 m is
# extrapolated (data_analysis.EXTRA_HEIGHTS); that region is shaded in the figure.
HEIGHT_WINDOWS = [(50, 150), (100, 200), (150, 250), (200, 300), (250, 350), (300, 400),
                  (350, 450), (400, 500)]

# Maximum tether forces for the F_max study [N]
FMAX_VALUES = [20e3, 30e3, 40e3, 50e3, 60e3, 70e3, 80e3, 90e3, 100e3]

# Sensitivity study: (parameter, low, high, label); baseline values in kite_model.P
SENSITIVITY = [
    ('w',      1.6,  2.4,  'Kite mass per area w [kg/m2]'),
    ('CLo',    0.8,  1.0,  'Lift coefficient reel-out C_L,o'),
    ('CDko',   0.18, 0.25, 'Kite drag C_D,k,o (L/D 5.0 / 3.6)'),
    ('vmin',   1.0,  3.0,  'Minimum reeling speed v_min [m/s]'),
    ('vmax',   8.0,  12.0, 'Maximum reeling speed v_max [m/s]'),
    ('beta_o', 20.0, 30.0, 'Reel-out elevation beta_o [deg]'),
    ('sigma',  5.6e8, 8.4e8, 'Tether allowable stress sigma_t [MPa]'),
]
# Divisor for values shown in the sensitivity chart (sigma in MPa, not Pa)
DISPLAY_SCALE = {'sigma': 1e6}

WORKERS = max(1, (os.cpu_count() or 2) - 1)

# Colours: one accent for the kite, greys for context
C_KITE = '#2b6cb0'
C_TETHER = '#90a4b8'
C_GREY = '#9a9a9a'
C_LINES = ['#2b6cb0', '#d97706', '#2f855a', '#9b2c2c', '#6b46c1']
