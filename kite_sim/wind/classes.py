"""
classes.py - Read the wind-resource data and group the points into wind classes
===============================================================================
"""

import numpy as np
import pandas as pd

from .settings import (CELL_AREA_EQUATOR_KM2, CLASS_EDGES, CLASS_HEIGHT, CLASS_NAMES, DATA_FILE,
                       ELEVATION_COLUMN, HEIGHTS)


# =====================================================================================
# 1. READ THE DATA
# =====================================================================================
def load_data(path=DATA_FILE, min_coverage=0.9, countries=None):
    """
    Read the wind-resource results and keep the usable points.

    Inputs
        path         : CSV file from the wind-resource analysis
        min_coverage : [-] minimum share of valid hours; points below are dropped
        countries    : country names to keep (target markets); None keeps all

    Output
        DataFrame, one row per grid point, plus:
            cell_area_km2 : [km2] grid cell area (shrinks with cos(latitude))
            population    : [-]   people in the cell = population density x cell area
    """
    df = pd.read_csv(path)
    df = df[df['data_coverage'] >= min_coverage].copy()   # too little valid data: drop

    # A row can list several countries ('Argentina; Chile'); keep it if any is wanted.
    if countries is not None:
        wanted = set(countries)
        keep = df['country'].str.split('; ').apply(lambda names: bool(wanted & set(names)))
        df = df[keep].copy()

    # Cell area and people, used to weight the class shares
    df['cell_area_km2'] = CELL_AREA_EQUATOR_KM2 * np.cos(np.radians(df['lat']))
    df['population'] = df['population_per_km2'] * df['cell_area_km2']
    return df


# =====================================================================================
# 2. WIND CLASSES AND THEIR SHARES
# =====================================================================================
def mean_speed_at(df, z):
    """
    Mean wind speed of every point at height z [m], linear in height between the data heights
    (the same interpolation as wind_bins(), so it is the mean of the wind the kite sees at z).
    Output: Series [m/s], one value per point.
    """
    if not HEIGHTS[0] <= z <= HEIGHTS[-1]:
        raise ValueError(f'z = {z} m is outside the data heights {HEIGHTS[0]}-{HEIGHTS[-1]} m')
    v = df[[f'h{h}_mean_speed_m_s' for h in HEIGHTS]].to_numpy()
    return pd.Series([np.interp(z, HEIGHTS, row) for row in v], index=df.index)


def assign_classes(df, height=CLASS_HEIGHT, edges=CLASS_EDGES):
    """
    Assign every point a wind class by its mean wind speed at the reference height.

    Inputs
        df     : DataFrame from load_data()
        height : [m]   reference height (within HEIGHTS; interpolated between them)
        edges  : [m/s] class edges; class i holds edges[i] <= mean speed < edges[i+1]

    Output
        copy of df with column 'wind_class' (1, 2, ... len(edges)-1)
    """
    df = df.copy()
    speed = mean_speed_at(df, height)
    df['wind_class'] = pd.cut(speed, bins=edges, right=False, labels=False) + 1
    return df


def class_summary(df):
    """
    One row per wind class: its share of the data set and its wind.

    Output columns
        n_points             :       number of grid points in the class
        share_points_pct     : [%]   share of all points
        share_pop_pct        : [%]   share of all people living in the points
        share_area_pct       : [%]   share of the total cell area (cells shrink with latitude)
        share_windy_area_pct : [%]   share of the windy land area (GWA area above threshold)
        mean_speed_<h>m      : [m/s] average of the points' mean wind speed at each height
        weibull_k_<h>m       : [-]   average Weibull shape factor at each height
        mean_elevation_m     : [m]   average site elevation (only if ELEVATION_COLUMN exists)
        top_countries        :       the five countries with the most points in the class
    """
    rows = []
    total_pop = df['population'].sum()
    total_area = df['gwa_area_ge_threshold_km2'].sum()
    for c, name in enumerate(CLASS_NAMES, start=1):
        g = df[df['wind_class'] == c]
        row = dict(wind_class=c, name=name, n_points=len(g),
                   share_points_pct=100 * len(g) / len(df),
                   share_pop_pct=100 * g['population'].sum() / total_pop,
                   share_area_pct=100 * g['cell_area_km2'].sum() / df['cell_area_km2'].sum(),
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
