"""
plotting.py - Kite design studies: wind classes -> kite sizing -> figures
=========================================================================

WHAT THIS FILE DOES
    Connects the wind classes (data_analysis.py) to the kite model (kite_model.py): for every
    case (wind class + design inputs + parameter changes) it sizes the kite for the target
    mean power, and then plots how the kite changes.

FIGURES (saved in outputs/figures/)
    1 size_per_class       kite area and mass (kite + tether) for every wind class
    2 size_vs_height       kite area, mass and tether for different flying heights
    3 downtime_per_class   hours per year out of operation and cut-in speed per class
    4 power_curves         power curve of the kite designed for each class
    5 fmax_effect          kite area and mass against the maximum tether force
    6 sensitivity          tornado chart: which model parameter moves the kite area most
    7 stability_cf         variability (stability criterion) and mean / maximum cycle power per class
    8 wind_classes         how many data points (areas) have which mean wind speed, per class
    9 size_range_ld        kite area and downtime per class for the range of measured kite L/D
   10 class_overview       kite area, airborne + tether mass and downtime per class (baseline)

HOW TO RUN
    python plotting.py                      all figures
    python plotting.py size_per_class sensitivity     only these figures
    python plotting.py --workers 4          number of parallel processes
    python plotting.py --fresh              ignore cached results and recompute

    Results of every case are cached in outputs/kite_cache.pkl, keyed by the case and by the
    contents of kite_model.py and data_analysis.py. Changing a setting here only recomputes
    the cases that changed; changing the model code recomputes everything.

RUN TIME
    One case (a full sizing) takes roughly 1-3 minutes on one core with kite_model.FAST; cases
    run in parallel. All figures together are 96 cases (about 95 min on 2 cores, 30 min on
    6 cores). The cache that comes with the repository (outputs/kite_cache.pkl) already holds
    all of them for the current code and data, so the figures are redrawn in seconds.

SITE ELEVATION
    Every case uses the average ground elevation of its wind class
    (data_analysis.class_elevations) for the air density; set USE_SITE_ELEVATION = False to
    use sea level everywhere. Without an elevation column in the data, sea level is used.
"""

import argparse
import hashlib
import json
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')                       # write files, no window needed
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory

import data_analysis as da
import kite_model as km


# =====================================================================================
# SETTINGS - change these to run other studies
# =====================================================================================
HERE = Path(__file__).parent
FIG_DIR = HERE / 'outputs' / 'figures'
CACHE_FILE = HERE / 'outputs' / 'kite_cache.pkl'

# Baseline design (every study changes one thing relative to this)
BASE = dict(zmin=100.0, zmax=200.0, Fmax=40e3, P_target=30e3)

# Largest kite area the sizing loop may use [m2]. A search bound for now; should become
# S_max from the volume budget. Classes that need more are shown as 'not feasible'.
S_MAX = 300.0

# Air density at the site's ground elevation (True) or at sea level (False)
USE_SITE_ELEVATION = True

# Average ground elevation per class [m]; filled in by main() from the data, read by make_case()
SITE_ELEVATION = {}

# Kite lift-to-drag range shown per class (C_L,o fixed, C_D,k,o varied) [-]:
# 4.0 = baseline (kite_model.P), 5.0 = measured peak of the TU Delft V3 kite in flight
# (Oehler & Schmehl 2019), 3.6 = measured traction average of van der Vlugt et al. (2019)
LD_RANGE = [3.6, 5.0]

# Classes used for the height, F_max and sensitivity studies (with the CLASS_EDGES in
# data_analysis.py: C2 5.0-5.5, C3 5.5-6.0, C4 6.0-6.5, C5 6.5-7.0, C6 7.0-7.5, C7 7.5-8.0,
# C8 8.0-8.5 m/s at 100 m)
HEIGHT_CLASSES = [2, 4, 6, 8]
FMAX_CLASSES = [3, 5, 7]
SENS_CLASS = 7

# Flying-height windows for the height study: (z_min, z_max) with the same 100 m stroke.
# Above 400 m the wind is extrapolated (data_analysis.EXTRA_HEIGHTS); those points are shaded.
HEIGHT_WINDOWS = [(50, 150), (100, 200), (150, 250), (200, 300), (250, 350), (300, 400),
                  (350, 450), (400, 500)]

# Maximum tether forces for the F_max study [N]
FMAX_VALUES = [20e3, 30e3, 40e3, 50e3, 60e3, 70e3, 80e3, 90e3, 100e3]

# Sensitivity study: parameter, low value, high value (baseline values are in kite_model.P)
SENSITIVITY = [
    ('w',      1.6,  2.4,  'Kite mass per area w [kg/m2]'),
    ('CLo',    0.8,  1.0,  'Lift coefficient reel-out C_L,o'),
    ('CDko',   0.18, 0.25, 'Kite drag C_D,k,o (L/D 5.0 / 3.6)'),
    ('vmin',   1.0,  3.0,  'Minimum reeling speed v_min [m/s]'),
    ('vmax',   8.0,  12.0, 'Maximum reeling speed v_max [m/s]'),
    ('beta_o', 20.0, 30.0, 'Reel-out elevation beta_o [deg]'),
    ('sigma',  5.6e8, 8.4e8, 'Tether allowable stress sigma_t [MPa]'),
]
# Values shown in the sensitivity chart are divided by this factor (to show MPa, not Pa)
DISPLAY_SCALE = {'sigma': 1e6}

WORKERS = max(1, (os.cpu_count() or 2) - 1)

# Colours: one accent for the kite, greys for context
C_KITE = '#2b6cb0'
C_TETHER = '#90a4b8'
C_GREY = '#9a9a9a'
C_LINES = ['#2b6cb0', '#d97706', '#2f855a', '#9b2c2c', '#6b46c1']


# =====================================================================================
# CASES AND CACHE
# =====================================================================================
def make_case(cls, zmin=None, zmax=None, Fmax=None, P_target=None, **params):
    """
    One design case: wind class + design inputs + changes to the model parameters.

    Inputs: cls = wind class number; zmin, zmax [m], Fmax [N], P_target [W] (default BASE);
    params = changes to kite_model.P, e.g. w=2.4.
    Output: dict that fully defines the case (also used as cache key). It includes the
    site elevation z_site of the class, so changing the elevation data recomputes the case.
    """
    return dict(cls=int(cls),
                z_site=float(SITE_ELEVATION.get(int(cls), 0.0)) if USE_SITE_ELEVATION else 0.0,
                zmin=float(zmin if zmin is not None else BASE['zmin']),
                zmax=float(zmax if zmax is not None else BASE['zmax']),
                Fmax=float(Fmax if Fmax is not None else BASE['Fmax']),
                P_target=float(P_target if P_target is not None else BASE['P_target']),
                params={k: float(v) for k, v in sorted(params.items())},
                S_max=S_MAX)


def _code_tag():
    """
    Fingerprint of the model code, the data code AND the wind data file, so the cache is
    reset when any of them changes (a new wind data file changes the classes without
    changing the case keys, so it must be part of the fingerprint).
    """
    h = hashlib.md5()
    for f in ('kite_model.py', 'data_analysis.py'):
        h.update((HERE / f).read_bytes())
    h.update(Path(da.DATA_FILE).read_bytes())
    h.update(json.dumps(km.NUM, sort_keys=True).encode())
    return h.hexdigest()[:12]


def _key(case):
    return json.dumps(case, sort_keys=True)


def load_cache():
    if CACHE_FILE.exists():
        with open(CACHE_FILE, 'rb') as fh:
            cache = pickle.load(fh)
        if cache.get('tag') == _code_tag():
            return cache
    return dict(tag=_code_tag(), results={})


def save_cache(cache):
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, 'wb') as fh:
        pickle.dump(cache, fh)


# =====================================================================================
# RUNNING ONE CASE (in a worker process)
# =====================================================================================
def run_case(case, dist):
    """
    Size the kite for one case and return a small summary (picklable, for the cache).

    Inputs
        case : dict from make_case()
        dist : wind-class distributions from data_analysis.class_distributions()

    Output (dict)
        feasible : bool, False if no kite up to S_max reaches the target
        no_data  : True only if the class has no data points (then nothing is computed)
        S, A, m  : [m2], [m2], [kg]  projected area, flat area, airborne mass
        mt_tot, dt, rmax, zbar : [kg], [m], [m], [m]  tether mass, diameter, length; mean altitude
        v_in, v_out, T_out, sigma_rel, CF, P_rated : metrics of block H
        v, pj, Pc, beta : power curve per wind bin (wind speed, probability, cycle power,
                          reel-out elevation)
        runtime  : [s]
    """
    t0 = time.time()
    if case['cls'] not in dist:          # class without any data points: nothing to size
        n = len(da.BIN_CENTRES)
        return dict(case=case, feasible=False, no_data=True, zbar=np.nan, dt=np.nan,
                    rmax=np.nan, mt_tot=np.nan, v=da.BIN_CENTRES.tolist(), pj=[0.0] * n,
                    S=np.nan, A=np.nan, m=np.nan, v_in=np.nan, v_out=np.nan, T_out=np.nan,
                    sigma_rel=np.nan, CF=np.nan, P_rated=np.nan, Pc=[np.nan] * n,
                    beta=[np.nan] * n, runtime=0.0)
    p = dict(km.P, **case['params'])
    geo = km.geometry(case['zmin'], case['zmax'], case['Fmax'], p, case['z_site'])
    vs, pj = da.wind_bins(dist, case['cls'], geo['zbar'])
    res = km.size_kite(case['zmin'], case['zmax'], case['Fmax'], case['P_target'],
                       vs, pj, S_hi=case['S_max'], p=p, z_site=case['z_site'])
    out = dict(case=case, feasible=res is not None, zbar=geo['zbar'], dt=geo['dt'],
               rmax=geo['rmax'], mt_tot=geo['mt_tot'], v=vs.tolist(), pj=pj.tolist())
    if res is None:
        out.update(S=np.nan, A=np.nan, m=np.nan, v_in=np.nan, v_out=np.nan, T_out=np.nan,
                   sigma_rel=np.nan, CF=np.nan, P_rated=np.nan,
                   Pc=[np.nan] * len(vs), beta=[np.nan] * len(vs))
    else:
        out.update(S=res['S'], A=res['A'], m=res['m'], v_in=res['v_in'],
                   v_out=res['v_out'] if res['v_out'] is not None else np.nan,
                   T_out=res['T_out'], sigma_rel=res['sigma_rel'], CF=res['CF'],
                   P_rated=res['P_rated'],
                   Pc=[c['Pc'] if isinstance(c, dict) else 0.0 for c in res['curve']],
                   beta=[c['beta'] if isinstance(c, dict) else np.nan for c in res['curve']])
    out['runtime'] = time.time() - t0
    return out


def run_cases(cases, dist, workers=WORKERS, fresh=False):
    """
    Run all cases that are not in the cache yet, in parallel, and return all results.

    Output: dict {cache key: summary from run_case()}
    """
    cache = dict(tag=_code_tag(), results={}) if fresh else load_cache()
    todo = {}
    for c in cases:
        if _key(c) not in cache['results']:
            todo[_key(c)] = c
    if todo:
        print(f'{len(todo)} new cases ({len(cases) - len(todo)} cached), {workers} workers')
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run_case, c, dist): k for k, c in todo.items()}
            for i, fut in enumerate(as_completed(futures), start=1):
                r = fut.result()
                cache['results'][futures[fut]] = r
                c = r['case']
                state = f"S = {r['S']:.1f} m2" if r['feasible'] else 'not feasible'
                print(f'  [{i}/{len(todo)}] class {c["cls"]}, z {c["zmin"]:.0f}-{c["zmax"]:.0f} m, '
                      f'F_max {c["Fmax"] / 1e3:.0f} kN, {c["params"] or "baseline"}: '
                      f'{state} ({r["runtime"]:.0f} s)', flush=True)
                save_cache(cache)               # save as we go: an interrupted run keeps its work
        print(f'done in {time.time() - t0:.0f} s')
    return {_key(c): cache['results'][_key(c)] for c in cases}


# =====================================================================================
# THE STUDIES: which cases each figure needs
# =====================================================================================
N_CLASSES = len(da.CLASS_NAMES)


def cases_per_class():
    return [make_case(c) for c in range(1, N_CLASSES + 1)]


def cases_ld_range():
    """Every class at the baseline and at each kite L/D in LD_RANGE (C_L,o fixed)."""
    return cases_per_class() + [make_case(c, CDko=km.P['CLo'] / LD)
                                for LD in LD_RANGE for c in range(1, N_CLASSES + 1)]


def cases_height():
    return [make_case(c, zmin=a, zmax=b) for c in HEIGHT_CLASSES for a, b in HEIGHT_WINDOWS]


def cases_fmax():
    return [make_case(c, Fmax=F) for c in FMAX_CLASSES for F in FMAX_VALUES]


def cases_sensitivity():
    cases = [make_case(SENS_CLASS)]
    for key, lo, hi, _ in SENSITIVITY:
        cases += [make_case(SENS_CLASS, **{key: lo}), make_case(SENS_CLASS, **{key: hi})]
    return cases


def short_label(cls):
    """'C3\\n4.0-4.5' style tick label for a class."""
    name = da.CLASS_NAMES[cls - 1]
    return name.split(': ')[0] + '\n' + name.split(': ')[1].replace(' m/s', '')


def _style(ax, title=None, xlabel=None, ylabel=None):
    ax.grid(axis='y', color='#e5e5e5', lw=0.8)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    if title:
        ax.set_title(title, loc='left', fontsize=11, fontweight='bold')
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def _mark_infeasible(ax, xs, rs, y_frac=0.04):
    """Write 'not feasible' above every class without a design (y in axes fraction)."""
    ax.set_xlim(min(xs) - 0.7, max(xs) + 0.7)
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    for x, r in zip(xs, rs):
        if r.get('no_data'):
            ax.text(x, y_frac, 'no data\npoints', ha='center', va='bottom',
                    fontsize=7, color=C_GREY, transform=tr)
        elif not r['feasible']:
            ax.text(x, y_frac, f'not\nfeasible\n(> {S_MAX:.0f} m²)', ha='center', va='bottom',
                    fontsize=7, color=C_GREY, transform=tr)


def _base_note():
    return (f"Baseline: heights {BASE['zmin']:.0f}-{BASE['zmax']:.0f} m, "
            f"F_max {BASE['Fmax'] / 1e3:.0f} kN, target {BASE['P_target'] / 1e3:.0f} kW mean, "
            + ("air density at each class's mean ground elevation"
               if USE_SITE_ELEVATION and any(SITE_ELEVATION.values()) else "air density at sea level"))


# =====================================================================================
# FIGURES
# =====================================================================================
def fig_size_per_class(res, df=None):
    """Kite area and mass (kite + tether) for every wind class."""
    rs = [res[_key(c)] for c in cases_per_class()]
    x = np.arange(1, N_CLASSES + 1)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    S = [r['S'] for r in rs]
    A = [r['A'] for r in rs]
    a1.bar(x - 0.2, S, 0.4, color=C_KITE, label='projected area S')
    a1.bar(x + 0.2, A, 0.4, color=C_TETHER, label='flat area A')
    _mark_infeasible(a1, x, rs)
    _style(a1, 'Kite area needed for 30 kW mean power per wind class', ylabel='area [m²]')
    a1.legend(frameon=False, loc='upper left')

    m = [r['m'] for r in rs]
    mt = [r['mt_tot'] if r['feasible'] else np.nan for r in rs]
    a2.bar(x, m, 0.6, color=C_KITE, label='kite + control unit')
    a2.bar(x, mt, 0.6, bottom=m, color=C_TETHER, label='tether (full length)')
    for xi, mi, ti in zip(x, m, mt):
        if np.isfinite(mi):
            a2.text(xi, mi + ti, f'{mi + ti:.0f}', ha='center', va='bottom', fontsize=8)
    _mark_infeasible(a2, x, rs)
    _style(a2, 'Airborne mass and tether mass', ylabel='mass [kg]')
    a2.legend(frameon=False, loc='upper left')
    a2.set_xticks(x, [short_label(c) for c in x])
    a2.set_xlabel(f'wind class (mean wind speed at {da.CLASS_HEIGHT} m, m/s)')
    fig.text(0.01, 0.005, _base_note(), fontsize=8, color=C_GREY)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return fig


def fig_size_vs_height(res, df=None):
    """Kite area, mass and tether against the average flying height."""
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2))
    for i, cls in enumerate(HEIGHT_CLASSES):
        rs = [res[_key(make_case(cls, zmin=a, zmax=b))] for a, b in HEIGHT_WINDOWS]
        z = [r['zbar'] for r in rs]
        col = C_LINES[i % len(C_LINES)]
        lab = da.CLASS_NAMES[cls - 1]
        axs[0].plot(z, [r['S'] for r in rs], 'o-', color=col, label=lab)
        axs[1].plot(z, [r['m'] + r['mt_tot'] if r['feasible'] else np.nan for r in rs],
                    'o-', color=col, label=lab)
        axs[2].plot(z, [r['T_out'] for r in rs], 'o-', color=col, label=lab)
    for ax in axs:     # wind above the highest data height is extrapolated
        ax.axvspan(da.HEIGHTS[-1], ax.get_xlim()[1], color='#f2f2f2', zorder=0)
        ax.text(da.HEIGHTS[-1] + 5, 0.98, 'wind\nextrapolated', transform=
                blended_transform_factory(ax.transData, ax.transAxes),
                va='top', fontsize=7, color=C_GREY)
    _style(axs[0], 'Kite area', 'average flying height z̄ [m]', 'projected area S [m²]')
    _style(axs[1], 'Airborne + tether mass', 'average flying height z̄ [m]', 'mass [kg]')
    _style(axs[2], 'Hours out of operation', 'average flying height z̄ [m]', 'hours per year')
    axs[0].legend(frameon=False, fontsize=8)
    fig.text(0.01, 0.01, f'Each point: heights z̄ ± 50 m, F_max {BASE["Fmax"] / 1e3:.0f} kN, '
             f'target {BASE["P_target"] / 1e3:.0f} kW. Missing points: no kite up to {S_MAX:.0f} m². '
             f'Tether mass grows from {rs[0]["mt_tot"]:.0f} kg (z̄ = {rs[0]["zbar"]:.0f} m) to '
             f'{rs[-1]["mt_tot"]:.0f} kg ({rs[-1]["zbar"]:.0f} m), the same for every class.',
             fontsize=8, color=C_GREY)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def fig_downtime_per_class(res, df=None):
    """Hours per year out of operation and cut-in wind speed for every class."""
    rs = [res[_key(c)] for c in cases_per_class()]
    x = np.arange(1, N_CLASSES + 1)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    T = [r['T_out'] for r in rs]
    ax.bar(x, T, 0.6, color=C_KITE)
    for xi, r in zip(x, rs):
        if r['feasible']:
            ax.text(xi, r['T_out'] + 80, f"{100 * r['T_out'] / 8760:.0f}%\ncut-in\n{r['v_in']:.1f} m/s",
                    ha='center', va='bottom', fontsize=7.5)
    _mark_infeasible(ax, x, rs)
    ax.set_ylim(0, 8760)
    _style(ax, 'Hours per year below cut-in or above cut-out', ylabel='hours out of operation per year')
    ax.set_xticks(x, [short_label(c) for c in x])
    ax.set_xlabel(f'wind class (mean wind speed at {da.CLASS_HEIGHT} m, m/s)')
    fig.text(0.01, 0.005, _base_note() + '. Cut-in is the wind speed at the mean flying height.',
             fontsize=8, color=C_GREY)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def fig_power_curves(res, df=None):
    """Power curve of the kite designed for each class, and the wind distribution."""
    rs = [res[_key(c)] for c in cases_per_class()]
    feas = [(i + 1, r) for i, r in enumerate(rs) if r['feasible']]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                 gridspec_kw=dict(height_ratios=[2, 1]))
    cmap = plt.get_cmap('Blues')
    for j, (cls, r) in enumerate(feas):
        col = cmap(0.35 + 0.65 * j / max(1, len(feas) - 1))
        a1.plot(r['v'], np.array(r['Pc']) / 1e3, '-', color=col,
                label=f"{da.CLASS_NAMES[cls - 1]} (S = {r['S']:.0f} m²)")
        a2.plot(r['v'], 100 * np.array(r['pj']), '-', color=col)
    _style(a1, 'Cycle power of the kite designed for each class', ylabel='cycle power P_c [kW]')
    a1.legend(frameon=False, fontsize=8)
    _style(a2, 'Wind-speed distribution at the mean flying height', 'wind speed at z̄ [m/s]',
           'probability [%]')
    a2.set_xlim(0, 30)
    fig.text(0.01, 0.005, _base_note() + '. Mechanical power at the drum.', fontsize=8, color=C_GREY)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return fig


def fig_fmax_effect(res, df=None):
    """Kite area and mass against the maximum tether force."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for i, cls in enumerate(FMAX_CLASSES):
        rs = [res[_key(make_case(cls, Fmax=F))] for F in FMAX_VALUES]
        F = np.array(FMAX_VALUES) / 1e3
        col = C_LINES[i % len(C_LINES)]
        lab = da.CLASS_NAMES[cls - 1]
        a1.plot(F, [r['S'] for r in rs], 'o-', color=col, label=lab)
        a2.plot(F, [r['m'] + r['mt_tot'] if r['feasible'] else np.nan for r in rs], 'o-', color=col, label=lab)
    _style(a1, 'Kite area against maximum tether force', 'F_max [kN]', 'projected area S [m²]')
    _style(a2, 'Total airborne + tether mass', 'F_max [kN]', 'mass [kg]')
    a1.legend(frameon=False, fontsize=8)
    fig.text(0.01, 0.01, f'Heights {BASE["zmin"]:.0f}-{BASE["zmax"]:.0f} m, target '
             f'{BASE["P_target"] / 1e3:.0f} kW. Missing points: no kite up to {S_MAX:.0f} m² '
             '(F_max sets the ceiling on mean power).', fontsize=8, color=C_GREY)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def fig_sensitivity(res, df=None):
    """Tornado chart: change in kite area when one parameter goes to its low or high value."""
    base = res[_key(make_case(SENS_CLASS))]
    rows = []
    for key, lo, hi, label in SENSITIVITY:
        r_lo = res[_key(make_case(SENS_CLASS, **{key: lo}))]
        r_hi = res[_key(make_case(SENS_CLASS, **{key: hi}))]
        d_lo = 100 * (r_lo['S'] / base['S'] - 1)
        d_hi = 100 * (r_hi['S'] / base['S'] - 1)
        scale = DISPLAY_SCALE.get(key, 1.0)
        rows.append((label, lo / scale, hi / scale, d_lo, d_hi))
    rows.sort(key=lambda t: np.nanmax(np.abs([t[3], t[4]])) if np.any(np.isfinite([t[3], t[4]])) else 1e9)
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(rows) + 1.8))
    y = np.arange(len(rows))
    ax.barh(y, [r[3] for r in rows], color=C_TETHER, label='low value')
    ax.barh(y, [r[4] for r in rows], color=C_KITE, label='high value')
    for yi, (label, lo, hi, dl, dh) in zip(y, rows):
        # One label per row, right of the bars: low and high value with their effect
        fmt = lambda d: f'{d:+.0f}%' if np.isfinite(d) else 'not feasible'
        right = np.nanmax([0, dl, dh])
        ax.text(right + 1.5, yi, f'{lo:g} \u2192 {fmt(dl)},  {hi:g} \u2192 {fmt(dh)}',
                va='center', ha='left', fontsize=8)
    ax.axvline(0, color='#555555', lw=0.8)
    ax.set_yticks(y, [r[0] for r in rows])
    _style(ax, f'Sensitivity of the kite area, {da.CLASS_NAMES[SENS_CLASS - 1]} '
               f'(baseline S = {base["S"]:.1f} m²)', 'change in kite area [%]')
    ax.grid(axis='x', color='#e5e5e5', lw=0.8)
    ax.legend(frameon=False, loc='lower right')
    lim = np.nanmax(np.abs([[r[3], r[4]] for r in rows])) if rows else 10
    ax.set_xlim(-1.2 * lim, 1.9 * lim)
    fig.text(0.01, 0.01, _base_note() + '. Changes of about 2% or less are within the numerical '
             'noise of kite_model.FAST.', fontsize=8, color=C_GREY)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def fig_stability_cf(res, df=None):
    """Variability of the power (stability criterion) and mean power as share of maximum cycle power.

    The capacity factor against the peak reel-out power (F_max * v_max) is not plotted: with a
    fixed target and fixed F_max it is the same for every class by construction.
    """
    rs = [res[_key(c)] for c in cases_per_class()]
    x = np.arange(1, N_CLASSES + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))
    a1.bar(x, [r['sigma_rel'] for r in rs], 0.6, color=C_KITE)
    _style(a1, 'Variability of the power, σ_P / mean', ylabel='σ_P / P̄ [-]')
    share = [100 * BASE['P_target'] / np.nanmax(r['Pc']) if r['feasible'] else np.nan for r in rs]
    a2.bar(x, share, 0.6, color=C_KITE)
    for xi, r, s in zip(x, rs, share):
        if r['feasible']:
            a2.text(xi, s + 0.5, f"max\n{np.nanmax(r['Pc']) / 1e3:.0f} kW", ha='center',
                    va='bottom', fontsize=7.5)
    _style(a2, 'Mean power as share of maximum cycle power', ylabel='P̄ / max P_c [%]')
    for ax in (a1, a2):
        ax.set_xticks(x, [short_label(c) for c in x], fontsize=8)
        ax.set_xlabel('wind class')
        _mark_infeasible(ax, x, rs)
    fig.text(0.01, 0.01, _base_note() + '. Variability over the wind distribution, not hourly.',
             fontsize=8, color=C_GREY)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def fig_wind_classes(res, df=None):
    """
    How many areas (ERA5 points) have which mean wind speed, and how big each class is.

    Why: shows how much of the data each kite design would serve. Top: number of points per
    0.25 m/s of mean wind speed at CLASS_HEIGHT, coloured by class. Bottom: share of each
    class in points, land area (cell area shrinks with latitude) and population.
    """
    summ = da.class_summary(df)
    v = df[f'h{da.CLASS_HEIGHT}_mean_speed_m_s']
    edges = np.arange(np.floor(v.min() * 4) / 4, np.ceil(v.max() * 4) / 4 + 0.25, 0.25)
    blues = plt.cm.Blues(np.linspace(0.3, 0.95, N_CLASSES))

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7.5), gridspec_kw=dict(height_ratios=[3, 2]))
    for c in range(1, N_CLASSES + 1):
        a1.hist(v[df['wind_class'] == c], bins=edges, color=blues[c - 1], label=None)
    for e in da.CLASS_EDGES[1:-1]:
        a1.axvline(e, color='#555555', lw=0.6, ls=':')
    top = a1.get_ylim()[1] * 1.15                  # room for the class labels above the bars
    a1.set_ylim(0, top)
    a1.set_xlim(0, np.percentile(v, 99.9) + 0.5)
    lows = [max(da.CLASS_EDGES[0], edges[0])] + da.CLASS_EDGES[1:-1]
    highs = da.CLASS_EDGES[1:-1] + [min(da.CLASS_EDGES[-1], edges[-1])]
    for c, (lo, hi) in enumerate(zip(lows, highs), start=1):
        share = summ.loc[summ['wind_class'] == c, 'share_points_pct'].iloc[0]
        a1.text(0.5 * (lo + hi) if c < N_CLASSES else lo + 0.6, top * 0.99,
                f'C{c}\n{share:.0f}%', ha='center', va='top', fontsize=8)
    _style(a1, f'Number of areas per mean wind speed at {da.CLASS_HEIGHT} m '
               f'({len(df):,} ERA5 cells)', f'mean wind speed at {da.CLASS_HEIGHT} m [m/s]',
           'number of cells')

    x = np.arange(1, N_CLASSES + 1)
    cols = [('share_points_pct', 'cells', C_KITE),
            ('share_area_pct', 'area', C_TETHER),
            ('share_pop_pct', 'population', '#d97706')]
    for i, (col, lab, colr) in enumerate(cols):
        a2.bar(x + (i - 1) * 0.27, summ[col], 0.27, color=colr, label=lab)
    a2.set_xticks(x, [short_label(c) for c in x], fontsize=8)
    _style(a2, 'Share of each wind class', f'wind class (mean wind speed at {da.CLASS_HEIGHT} m, m/s)',
           'share [%]')
    a2.legend(frameon=False, fontsize=8, ncol=3)
    fig.text(0.01, 0.005, 'Each ERA5 cell is one area of about 28 x 28 km at the equator '
             '(smaller towards the poles). Classes from data_analysis.CLASS_EDGES.',
             fontsize=8, color=C_GREY)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return fig


def fig_size_range_ld(res, df=None):
    """
    Kite area and hours out of operation per class at the baseline kite L/D (point), with the
    range for the L/D values in LD_RANGE (whisker).

    Why: the kite L/D is the most uncertain input with the largest effect on the kite size.
    The whisker shows what the kite would be with the measured peak L/D (smaller kite) and
    with the lowest measured average (larger kite).
    """
    x = np.arange(1, N_CLASSES + 1)
    LD0 = km.P['CLo'] / km.P['CDko']
    base = [res[_key(make_case(c))] for c in x]
    alt = {LD: [res[_key(make_case(c, CDko=km.P['CLo'] / LD))] for c in x] for LD in LD_RANGE}
    lo_LD, hi_LD = max(LD_RANGE), min(LD_RANGE)          # high L/D -> small kite

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
    for ax, key, unit, title in ((a1, 'S', 'projected area S [m²]', 'Kite area for 30 kW mean'),
                                 (a2, 'T_out', 'hours per year', 'Hours out of operation')):
        for xi, rb, rl, rh in zip(x, base, alt[lo_LD], alt[hi_LD]):
            y0 = rb[key]
            y_lo = rl[key] if rl['feasible'] else np.nan
            y_hi = rh[key] if rh['feasible'] else np.nan
            if np.isfinite(y_lo) or np.isfinite(y_hi) or np.isfinite(y0):
                bottom = y_lo if np.isfinite(y_lo) else y0
                top = y_hi if np.isfinite(y_hi) else (S_MAX if key == 'S' else y0)
                if np.isfinite(bottom) and np.isfinite(top):
                    ax.plot([xi, xi], [bottom, top], color=C_TETHER, lw=2.5, solid_capstyle='butt')
                    ax.plot([xi - 0.12, xi + 0.12], [bottom, bottom], color=C_TETHER, lw=1.5)
                    if np.isfinite(y_hi):
                        ax.plot([xi - 0.12, xi + 0.12], [top, top], color=C_TETHER, lw=1.5)
                    elif key == 'S':
                        ax.annotate('', (xi, top + 12), (xi, top), arrowprops=dict(arrowstyle='->', color=C_TETHER))
            if np.isfinite(y0):
                ax.plot(xi, y0, 'o', color=C_KITE, ms=7, zorder=3)
                ax.text(xi + 0.18, y0, f'{y0:.0f}', va='center', fontsize=7.5)
            elif key == 'S' and np.isfinite(y_lo):
                ax.text(xi + 0.15, S_MAX * 0.9, f'L/D {LD0:.1f}:\n> {S_MAX:.0f} m²', va='top',
                        fontsize=7, color=C_KITE)
        _style(ax, title, ylabel=unit)
        ax.set_xlim(0.4, N_CLASSES + 0.6)
    a1.set_ylim(0, S_MAX * 1.12)
    a1.axhline(S_MAX, color=C_GREY, ls=':', lw=1)
    a1.text(N_CLASSES + 0.55, S_MAX, f'search limit {S_MAX:.0f} m²', ha='right', va='bottom', fontsize=7.5,
            color=C_GREY)
    for xi, rb, rl in zip(x, base, alt[lo_LD]):
        if not rl['feasible'] and not rb['feasible']:
            a1.text(xi, 15, f'> {S_MAX:.0f} m²\neven at\nL/D {lo_LD:g}', ha='center', fontsize=7, color=C_GREY)
    from matplotlib.lines import Line2D
    a2.legend([Line2D([], [], marker='o', ls='', color=C_KITE),
               Line2D([], [], color=C_TETHER, lw=2.5)],
              [f'baseline: kite L/D {LD0:.1f}\n(measured reel-out average)',
               f'range: L/D {lo_LD:g} (measured peak)\nto {hi_LD:g} (lowest measured average);\n'
               f'arrow = above {S_MAX:.0f} m²'], frameon=False, fontsize=8, loc='upper left')
    a2.set_xticks(x, [short_label(c) for c in x])
    a2.set_xlabel(f'wind class (mean wind speed at {da.CLASS_HEIGHT} m, m/s)')
    fig.text(0.01, 0.005, _base_note() + '. Kite L/D: Oehler & Schmehl 2019; van der Vlugt et al. 2019. '
             'C_L,o fixed at 0.9, C_D,k,o varied.', fontsize=7.5, color=C_GREY)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return fig


def fig_class_overview(res, df=None):
    """
    Kite area, airborne + tether mass and hours out of operation per wind class, for the
    current model at the baseline design (one bar per class, value on top).

    Why: the three numbers per class that feed the trade-off (size, mass budget, downtime)
    side by side.
    """
    rs = [res[_key(c)] for c in cases_per_class()]
    x = np.arange(1, N_CLASSES + 1)
    panels = (
        ('Kite area for 30 kW mean', 'projected area S [m²]', [r['S'] for r in rs], '{:.0f}'),
        ('Airborne + tether mass', 'mass [kg]',
         [r['m'] + r['mt_tot'] if r['feasible'] else np.nan for r in rs], '{:.0f}'),
        ('Hours out of operation', 'hours per year', [r['T_out'] for r in rs], '{:,.0f}'),
    )
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, (title, unit, vals, fmt) in zip(axs, panels):
        ax.bar(x, vals, 0.6, color=C_KITE)
        top = np.nanmax(vals)
        for xi, v in zip(x, vals):
            if np.isfinite(v):
                ax.text(xi, v + 0.01 * top, fmt.format(v), ha='center', va='bottom', fontsize=7.5)
        ax.set_ylim(0, 1.12 * top)
        ax.set_xlim(0.4, N_CLASSES + 0.6)
        ax.set_xticks(x, [short_label(c) for c in x], fontsize=7)
        _style(ax, title, 'wind class (mean wind at 100 m, m/s)', unit)
        # One label over all classes without a design (instead of one per class)
        bad = [xi for xi, r in zip(x, rs) if not r['feasible']]
        if bad:
            ax.text(0.5 * (min(bad) + max(bad)), 0.06 * ax.get_ylim()[1],
                    f'C{min(bad)}-C{max(bad)}:\nno kite up to {S_MAX:.0f} m²\nreaches 30 kW',
                    ha='center', va='bottom', fontsize=8, color=C_GREY)
    LD = km.P['CLo'] / km.P['CDko']
    fig.text(0.01, 0.01, _base_note() + f'. Kite L/D {LD:.1f}, {km.P["w"]:.1f} kg/m². '
             f'Hours out of operation: wind below cut-in or above cut-out.', fontsize=8, color=C_GREY)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


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

    # All cases the chosen figures need, run once (shared cases are computed once)
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


if __name__ == '__main__':
    main()
