"""
data_analysis.py - Wind classes for the kite (and turbine) design
=================================================================

WHAT IT DOES
    1. Reads wind_results_2024.csv: one row per ERA5 grid point with Weibull A, k at 40, 100,
       200, 300 and 400 m above ground, population density and Global Wind Atlas (GWA)
       windy area. The ERA5 cell-average wind is used as is (no local correction): placement
       is also limited by e.g. population, so the cell average is representative. The file's
       outage statistics are not used; the kite model computes its own cut-in and downtime.
    2. Groups the points into wind classes by mean wind speed at a reference height and
       reports each class's share of the points, population and windy area.
    3. Builds one wind-speed distribution per class and height: the mixture (average) of the
       Weibull distributions of all points in the class.
    4. Gives the kite model wind-speed bins and probabilities per class at any altitude
       between 40 and 400 m (linear interpolation between data heights, M4).

USE IN THE KITE MODEL
        import data_analysis as da
        df   = da.load_data()
        df   = da.assign_classes(df)
        dist = da.class_distributions(df)
        vs, pj = da.wind_bins(dist, class_id=3, z=geo['zbar'])   # bins at the kite's altitude
        result = size_kite(z_min, z_max, F_max, 30e3, vs, pj)

OUTPUTS (python data_analysis.py)
    outputs/wind_class_summary.csv        per class: shares, mean speeds, Weibull k, countries
    outputs/wind_class_distributions.csv  long table: class, height, wind-speed bin, probability

CODE (re-exported here, so  import data_analysis as da  works as before)
    kite_sim/wind/settings.py       data file, heights, class edges, bins
    kite_sim/wind/classes.py        load_data(), assign_classes(), class_summary()
    kite_sim/wind/distributions.py  class_distributions(), class_elevations(), wind_bins()

Needs only numpy and pandas (no scipy).
"""

import numpy as np
import pandas as pd

from kite_sim.wind import *         # noqa: F401,F403  list in kite_sim/wind/__init__.py
from kite_sim.wind import (CLASS_HEIGHT, CLASS_NAMES, OUTPUT_DIR, assign_classes, class_distributions,
                           class_summary, distributions_to_table, load_data, wind_bins)


# =====================================================================================
# RUN: save the class summary and distributions
# =====================================================================================
if __name__ == '__main__':
    df = load_data()
    df = assign_classes(df)

    summary = class_summary(df)
    dist = class_distributions(df)

    OUTPUT_DIR.mkdir(exist_ok=True)
    summary.to_csv(OUTPUT_DIR / 'wind_class_summary.csv', index=False)
    distributions_to_table(dist).to_csv(OUTPUT_DIR / 'wind_class_distributions.csv', index=False)

    pd.set_option('display.width', 200)
    print(f'{len(df)} points, classified at {CLASS_HEIGHT} m\n')
    print(summary[['name', 'n_points', 'share_points_pct', 'share_pop_pct',
                   'share_windy_area_pct', 'mean_speed_100m', 'mean_speed_200m',
                   'weibull_k_100m']].round(1).to_string(index=False))
    top = len(CLASS_NAMES)
    print(f'\nExample: class {top} at 150 m (kite average altitude for 100-200 m flying heights)')
    vs, pj = wind_bins(dist, top, 150)
    print(f'  mean wind speed from the bins: {np.sum(vs * pj):.2f} m/s')
