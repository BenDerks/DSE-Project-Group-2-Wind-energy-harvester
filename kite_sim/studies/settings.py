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

# ---- Design objective and output folder:  python plotting.py --objective area | multi -------
#   area   best design = smallest kite area (one design input changed at a time)
#   multi  best design = lowest weighted score of kite area, downtime and longest continuous
#          downtime, over the full grid of flying-height windows x F_max values (selection.py).
#          Each goal is scored as the fraction worse than the best design of the class.
RESULT_DIRS = dict(area=REPO / 'outputs' / 'results_min_area',
                   multi=REPO / 'outputs' / 'results_multi_objective')
OBJECTIVE_WEIGHTS = dict(area=1 / 3, downtime=1 / 3, longest_downtime=1 / 3)
RUN = dict(objective='area')                        # set by main(), read by the summary figures
FIG_DIR = RESULT_DIRS['area']

# Hourly ERA5 wind of the windres project (one parquet per cell: u10, v10, u100, v100), used for
# the continuous downtime of objective multi. Reading it needs pyarrow (pip install pyarrow).
HOURLY_DIR = REPO.parent / 'windres' / 'data' / 'world_2024' / 'timeseries' / 'era5'
DOWNTIME_CACHE = REPO / 'outputs' / 'downtime_cache.pkl'
CACHE_FILE = REPO / 'outputs' / 'kite_cache.pkl'

# Baseline design; each study changes one thing [m, m, N, W]
BASE = dict(zmin=100.0, zmax=200.0, Fmax=50e3, P_target=30e3)

# Highest maximum tether force a design may use [N]: caps the F_max study (FMAX_VALUES) and the
# designs the best-design summary (figures_summary.py) can pick.
FMAX_LIMIT = 50e3

# Largest kite area the sizing loop may use [m2]. A search bound for now; should become S_max
# from the volume budget. Classes that need more are shown as 'not feasible'.
S_MAX = 300.0

# Air density at the site's ground elevation (True) or at sea level (False)
USE_SITE_ELEVATION = True

# Mean ground elevation per class [m]; filled by main() from the data, read by make_case()
SITE_ELEVATION = {}

# ---- Study 2: flying heights ---------------------------------------------------------
# Every combination of z_min and stroke (z_max - z_min), up to z_max = 300 m.
# Why at most 300 m: at the fixed 25 deg elevation the tether is z / sin(25 deg) = 2.4 z long,
# so 300 m needs a 710 m tether. That is within the 600-800 m tether of the SkySails PN-14,
# which specifies its performance at a flight altitude of 200 m (data sheet Venyo PN-14,
# 2025); Kitepower's 350 m tether (kitepower website, Falcon) reaches about 150 m at 25 deg.
# All heights stay inside the measured wind data (up to 400 m), so nothing is extrapolated.
HEIGHT_ZMIN = [50, 75, 100, 125, 150]              # [m]
HEIGHT_STROKES = [50, 100, 150]                    # [m] z_max - z_min
HEIGHT_ZMAX_LIMIT = 300                            # [m]
HEIGHT_CLASSES = list(range(1, 11))                # every wind class (C1-C10)

# ---- Study 3: kite lift-to-drag ratio (C_L,o fixed at 0.9, C_D,k,o = C_L,o / (L/D)) ----
# Value -> source. 4.5 is the baseline in kite_model.P.
LD_VALUES = {
    3.6: 'TU Delft V3 kite, traction average (van der Vlugt et al. 2019)',
    4.0: 'TU Delft V3 kite, reel-out average in flight (Oehler & Schmehl 2019)',
    4.5: 'TU Delft V3 kite parameters (Schelbergen et al. 2020, Table 4) - baseline',
    5.0: 'TU Delft V3 kite, measured peak (Oehler & Schmehl 2019)',
    6.0: 'small ram-air kites, 3-6 m2 (de Wachter 2008; Dadd et al. 2010; via Oehler & Schmehl '
         '2019); consistent with the SkySails PN-14 data sheet',
}
LD_CLASSES = list(range(1, 11))

# ---- Study 4: maximum tether force -----------------------------------------------------
# A design choice. For reference: Kitepower Falcon kite 3.5 t = 34 kN (Kitepower website).
# The tether diameter follows from F_max and the allowable stress (Joshi et al. 2024).
FMAX_VALUES = [F for F in (20e3, 30e3, 40e3, 50e3, 60e3, 80e3, 100e3) if F <= FMAX_LIMIT]  # [N]
FMAX_CLASSES = list(range(1, 11))

# ---- Extra: sensitivity of the kite area to the model parameters (one class) ------------
SENS_CLASS = 7

# Sensitivity study: (parameter, low, high, label); baseline values in kite_model.P
SENSITIVITY = [
    ('w',      1.15, 2.0,  'Kite mass per area w [kg/m2] (V3 kite / Cayon 2026)'),
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
