"""
data_analysis.py - Wind classes for the kite (and turbine) design
================================================================

WHAT THIS FILE DOES
    1. Reads the wind-resource results (wind_results_2024.csv): one row per ERA5 grid point
       with Weibull parameters (A, k) at 40, 100, 200, 300 and 400 m above ground level,
       population density and Global Wind Atlas (GWA) windy-area values.
       The ERA5 cell-average wind is used as is (no local correction): placement is also
       limited by other constraints such as population, so the cell average is representative.
       The outage statistics in the file are not used; the kite model computes its own
       cut-in and time out of operation.
    2. Groups the points into wind classes by their mean wind speed at a reference height,
       and reports which share of the points (and of the population and windy area) falls
       into each class.
    3. Builds one wind-speed distribution per class and per height: the average of the
       Weibull distributions of all points in that class (a mixture distribution).
    4. Gives the kite model what it needs per class: wind-speed bins and their probabilities
       at any altitude between 40 and 400 m (linear interpolation between data heights, M4).

HOW THE KITE MODEL USES IT
        import data_analysis as da
        df   = da.load_data()
        df   = da.assign_classes(df)
        dist = da.class_distributions(df)
        vs, pj = da.wind_bins(dist, class_id=3, z=geo['zbar'])   # bins at the kite's altitude
        result = size_kite(z_min, z_max, F_max, 30e3, vs, pj)

OUTPUTS (written by running this file: python data_analysis.py)
    outputs/wind_class_summary.csv        one row per class: shares, mean speeds, Weibull k, countries
    outputs/wind_class_distributions.csv  long table: class, height, wind-speed bin, probability

ONLY numpy AND pandas are needed (no scipy).
"""

from pathlib import Path
import numpy as np
import pandas as pd


# =====================================================================================
# SETTINGS - change these, not the code below
# =====================================================================================
DATA_FILE = Path(__file__).parent / 'wind_results_2024.csv'
OUTPUT_DIR = Path(__file__).parent / 'outputs'

# Heights at which the data gives Weibull parameters [m above ground level]
HEIGHTS = [40, 100, 200, 300, 400]

# Wind classes: points are classified by their annual MEAN wind speed at CLASS_HEIGHT.
# 100 m is used as a common reference for both concepts (turbine hub at 40 m, kite at
# 100-400 m). Edges in m/s; the first class has no lower and the last no upper limit.
# 10 classes of 0.5 m/s; 7.5 m/s (lower bound of IEC wind class III, the threshold used in
# the market analysis) is the lower edge of the top class.
CLASS_HEIGHT = 100
CLASS_EDGES = [0.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, np.inf]


def _class_names(edges):
    """Readable class names from the edges, e.g. 'C3: 4.0-4.5 m/s'."""
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

# Wind-speed bins for the kite model: centres 0.5, 1.5, ... 34.5 m/s, width 1 m/s.
# The range covers the whole distribution (probability above 35 m/s is negligible).
BIN_WIDTH = 1.0
BIN_CENTRES = np.arange(0.5, 35.0, BIN_WIDTH)

# ERA5 grid cell area at the equator for a 0.25 deg grid [km2]; scales with cos(latitude).
# Check: the largest GWA area in the data (772.8 km2) equals this value.
CELL_AREA_EQUATOR_KM2 = 772.8

# Column with the site (ground) elevation above sea level [m], once it is added to the data.
# Needed for the air density at the kite's altitude. Not in the data yet.
ELEVATION_COLUMN = 'elevation_m'


# =====================================================================================
# 1. READ THE DATA
# =====================================================================================
def load_data(path=DATA_FILE, min_coverage=0.9, countries=None):
    """
    Read the wind-resource results and keep the points that are usable.

    Inputs
        path         : CSV file from the wind-resource analysis
        min_coverage : [-] minimum share of valid hours; points below are dropped
        countries    : optional list of country names to keep (target markets);
                       None keeps all points

    Output
        DataFrame with one row per grid point, plus two derived columns:
            cell_area_km2 : [km2] area of the grid cell (shrinks with cos(latitude))
            population    : [-]   people in the cell = population density x cell area
    """
    df = pd.read_csv(path)

    # Drop points with too little valid data
    df = df[df['data_coverage'] >= min_coverage].copy()

    # Optional: keep only target-market countries. A row can list several countries
    # ('Argentina; Chile'), so a point is kept if any of them is in the list.
    if countries is not None:
        wanted = set(countries)
        keep = df['country'].str.split('; ').apply(lambda names: bool(wanted & set(names)))
        df = df[keep].copy()

    # Cell area and number of people, used to weight the class shares
    df['cell_area_km2'] = CELL_AREA_EQUATOR_KM2 * np.cos(np.radians(df['lat']))
    df['population'] = df['population_per_km2'] * df['cell_area_km2']
    return df


# =====================================================================================
# 2. WIND CLASSES AND THEIR SHARES
# =====================================================================================
def assign_classes(df, height=CLASS_HEIGHT, edges=CLASS_EDGES):
    """
    Put every point into a wind class by its mean wind speed at the reference height.

    Inputs
        df     : DataFrame from load_data()
        height : [m]   reference height for classification (must be one of HEIGHTS)
        edges  : [m/s] class edges; class i holds edges[i] <= mean speed < edges[i+1]

    Output
        copy of df with a column 'wind_class' (1, 2, ... len(edges)-1)
    """
    df = df.copy()
    speed = df[f'h{height}_mean_speed_m_s']
    df['wind_class'] = pd.cut(speed, bins=edges, right=False, labels=False) + 1
    return df


def class_summary(df):
    """
    One row per wind class: how much of the dataset falls into it and what its wind is like.

    Output columns
        n_points          : number of grid points in the class
        share_points_pct  : [%] share of all points
        share_pop_pct     : [%] share of all people living in the points
        share_windy_area_pct : [%] share of the windy land area (GWA area above threshold)
        mean_speed_<h>m   : [m/s] average of the points' mean wind speed at each height
        weibull_k_<h>m    : [-]   average Weibull shape factor at each height
        mean_elevation_m  : [m]   average site elevation (only once ELEVATION_COLUMN exists)
        top_countries     : the five countries with the most points in the class
    """
    rows = []
    total_pop = df['population'].sum()
    total_area = df['gwa_area_ge_threshold_km2'].sum()
    for c, name in enumerate(CLASS_NAMES, start=1):
        g = df[df['wind_class'] == c]
        row = dict(wind_class=c, name=name, n_points=len(g),
                   share_points_pct=100 * len(g) / len(df),
                   share_pop_pct=100 * g['population'].sum() / total_pop,
                   share_windy_area_pct=100 * g['gwa_area_ge_threshold_km2'].sum() / total_area)
        for h in HEIGHTS:
            row[f'mean_speed_{h}m'] = g[f'h{h}_mean_speed_m_s'].mean()
            row[f'weibull_k_{h}m'] = g[f'h{h}_weibull_k'].mean()
        if ELEVATION_COLUMN in df.columns:
            row['mean_elevation_m'] = g[ELEVATION_COLUMN].mean()
        countries = g['country'].str.split('; ').explode().value_counts().head(5)
        row['top_countries'] = ', '.join(countries.index)
        rows.append(row)
    return pd.DataFrame(rows)


# =====================================================================================
# 3. WIND-SPEED DISTRIBUTION PER CLASS AND HEIGHT
# =====================================================================================
def weibull_bin_probabilities(A, k, centres=BIN_CENTRES, width=BIN_WIDTH):
    """
    Probability of each wind-speed bin for many Weibull distributions at once.

    Inputs
        A, k    : arrays (one value per point) of Weibull scale [m/s] and shape [-]
        centres : [m/s] bin centres; width: [m/s] bin width

    Output
        array (n_points, n_bins): P(lower edge <= v < upper edge) per point and bin,
        from the Weibull cumulative distribution F(v) = 1 - exp(-(v/A)^k)
    """
    A = np.asarray(A, dtype=float)[:, None]
    k = np.asarray(k, dtype=float)[:, None]
    lower = np.maximum(centres - width / 2, 0.0)[None, :]
    upper = (centres + width / 2)[None, :]
    return np.exp(-(lower / A)**k) - np.exp(-(upper / A)**k)


def class_distributions(df, weight='points'):
    """
    Wind-speed distribution of each class at each data height.

    Why: the kite model needs ONE distribution per class. Averaging the Weibull
    distributions of all points (a mixture) keeps the spread between sites; averaging A and k
    instead would make the distribution too narrow.

    Inputs
        df     : DataFrame with 'wind_class' (from assign_classes)
        weight : 'points'     every point counts equally (default)
                 'population' points count by the number of people in the cell

    Output (dict)
        {class_id: array (len(HEIGHTS), len(BIN_CENTRES))} with the bin probabilities;
        each row sums to ~1
    """
    dist = {}
    for c in sorted(df['wind_class'].dropna().unique()):
        g = df[df['wind_class'] == c]
        w = np.ones(len(g)) if weight == 'points' else g['population'].to_numpy()
        w = w / w.sum()
        table = np.zeros((len(HEIGHTS), len(BIN_CENTRES)))
        for i, h in enumerate(HEIGHTS):
            p = weibull_bin_probabilities(g[f'h{h}_weibull_A_m_s'], g[f'h{h}_weibull_k'])
            table[i] = w @ p                     # weighted average over the points
        dist[int(c)] = table
    return dist


def wind_bins(dist, class_id, z):
    """
    Wind-speed bins and probabilities of one class at altitude z (M4).

    Why: the kite flies at its own average altitude (geo['zbar']), which generally lies
    between two data heights. The probabilities are interpolated linearly in height between
    the two nearest data heights and normalised to sum to 1.

    Inputs
        dist     : output of class_distributions()
        class_id : wind class number
        z        : [m] altitude; must lie between the lowest and highest data height

    Output
        vs : [m/s] bin centres;  pj : [-] probability per bin (sums to 1)
    """
    if not HEIGHTS[0] <= z <= HEIGHTS[-1]:
        raise ValueError(f'z = {z} m is outside the data heights {HEIGHTS[0]}-{HEIGHTS[-1]} m')
    table = dist[class_id]
    pj = np.array([np.interp(z, HEIGHTS, table[:, j]) for j in range(len(BIN_CENTRES))])
    return BIN_CENTRES.copy(), pj / pj.sum()


def distributions_to_table(dist):
    """Long table (class, height, bin centre, probability) for saving to CSV."""
    rows = []
    for c, table in dist.items():
        for i, h in enumerate(HEIGHTS):
            for v, p in zip(BIN_CENTRES, table[i]):
                rows.append(dict(wind_class=c, height_m=h, v_m_s=v, probability=p))
    return pd.DataFrame(rows)


# =====================================================================================
# RUN: make the class summary and distributions and save them
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
