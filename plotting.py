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

HOW TO RUN
    python plotting.py                      all figures
    python plotting.py size_per_class sensitivity     only these figures
    python plotting.py --workers 4          number of parallel processes
    python plotting.py --fresh              ignore cached results and recompute

    Results of every case are cached in outputs/kite_cache.pkl, keyed by the case and by the
    contents of kite_model.py and data_analysis.py. Changing a setting here only recomputes
    the cases that changed; changing the model code recomputes everything.

RUN TIME
    One case (a full sizing) takes roughly 20-60 s on one core with kite_model.FAST; cases run
    in parallel. All figures together are about 50 cases.

NOT INCLUDED YET
    Site elevation (air density at the site). The model still assumes sea-level density.
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
S_MAX = 150.0

# Classes used for the height, F_max and sensitivity studies
HEIGHT_CLASSES = [6, 8, 10]
FMAX_CLASSES = [8, 10]
SENS_CLASS = 10

# Flying-height windows for the height study: (z_min, z_max) with the same 100 m stroke
HEIGHT_WINDOWS = [(50, 150), (100, 200), (150, 250), (200, 300), (250, 350), (300, 400)]

# Maximum tether forces for the F_max study [N]
FMAX_VALUES = [20e3, 30e3, 40e3, 50e3, 60e3]

# Sensitivity study: parameter, low value, high value (baseline values are in kite_model.P)
SENSITIVITY = [
    ('w',      1.6,  2.4,  'Kite mass per area w [kg/m2]'),
    ('CLo',    0.8,  1.0,  'Lift coefficient reel-out C_L,o'),
    ('CDko',   0.16, 0.24, 'Kite drag coefficient reel-out C_D,k,o'),
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
    Output: dict that fully defines the case (also used as cache key).
    """
    return dict(cls=int(cls),
                zmin=float(zmin if zmin is not None else BASE['zmin']),
                zmax=float(zmax if zmax is not None else BASE['zmax']),
                Fmax=float(Fmax if Fmax is not None else BASE['Fmax']),
                P_target=float(P_target if P_target is not None else BASE['P_target']),
                params={k: float(v) for k, v in sorted(params.items())},
                S_max=S_MAX)


def _code_tag():
    """Fingerprint of the model and data code, so the cache is reset when they change."""
    h = hashlib.md5()
    for f in ('kite_model.py', 'data_analysis.py'):
        h.update((HERE / f).read_bytes())
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
        S, A, m  : [m2], [m2], [kg]  projected area, flat area, airborne mass
        mt_tot, dt, rmax, zbar : [kg], [m], [m], [m]  tether mass, diameter, length; mean altitude
        v_in, v_out, T_out, sigma_rel, CF, P_rated : metrics of block H
        v, pj, Pc, beta : power curve per wind bin (wind speed, probability, cycle power,
                          reel-out elevation)
        runtime  : [s]
    """
    t0 = time.time()
    p = dict(km.P, **case['params'])
    geo = km.geometry(case['zmin'], case['zmax'], case['Fmax'], p)
    vs, pj = da.wind_bins(dist, case['cls'], geo['zbar'])
    res = km.size_kite(case['zmin'], case['zmax'], case['Fmax'], case['P_target'],
                       vs, pj, S_hi=case['S_max'], p=p)
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
        if not r['feasible']:
            ax.text(x, y_frac, f'not\nfeasible\n(> {S_MAX:.0f} m²)', ha='center', va='bottom',
                    fontsize=7, color=C_GREY, transform=tr)


def _base_note():
    return (f"Baseline: heights {BASE['zmin']:.0f}-{BASE['zmax']:.0f} m, "
            f"F_max {BASE['Fmax'] / 1e3:.0f} kN, target {BASE['P_target'] / 1e3:.0f} kW mean")


# =====================================================================================
# FIGURES
# =====================================================================================
def fig_size_per_class(res):
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


def fig_size_vs_height(res):
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


def fig_downtime_per_class(res):
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


def fig_power_curves(res):
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


def fig_fmax_effect(res):
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


def fig_sensitivity(res):
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


def fig_stability_cf(res):
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


FIGURES = {
    'size_per_class': (cases_per_class, fig_size_per_class),
    'size_vs_height': (cases_height, fig_size_vs_height),
    'downtime_per_class': (cases_per_class, fig_downtime_per_class),
    'power_curves': (cases_per_class, fig_power_curves),
    'fmax_effect': (cases_fmax, fig_fmax_effect),
    'sensitivity': (cases_sensitivity, fig_sensitivity),
    'stability_cf': (cases_per_class, fig_stability_cf),
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

    # All cases the chosen figures need, run once (shared cases are computed once)
    cases = []
    for name in args.figures:
        for c in FIGURES[name][0]():
            if c not in cases:
                cases.append(c)
    res = run_cases(cases, dist, workers=args.workers, fresh=args.fresh)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for name in args.figures:
        fig = FIGURES[name][1](res)
        path = FIG_DIR / f'{name}.png'
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f'saved {path}')


if __name__ == '__main__':
    main()
