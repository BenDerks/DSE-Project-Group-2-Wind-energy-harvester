"""
runner.py - Size the kite for each case, in parallel worker processes
=====================================================================
"""

import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from .. import model as km
from .. import wind as da
from .cases import _code_tag, _key, load_cache, save_cache
from .settings import WORKERS


# =====================================================================================
# RUN CASES
# =====================================================================================
def run_case(case, dist):
    """
    Size the kite for one case; return a small picklable summary for the cache.

    Inputs
        case : dict from make_case()
        dist : class distributions from data_analysis.class_distributions()

    Output (dict)
        feasible               : False if no kite up to S_max reaches the target
        no_data                : True only if the class has no data points (nothing computed)
        S, A, m                : [m2], [m2], [kg] projected area, flat area, airborne mass
        mt_tot, dt, rmax, zbar : [kg], [m], [m], [m] tether mass, diameter, length; mean altitude
        v_in, v_out, T_out, sigma_rel, CF, P_rated : block H metrics
        v, pj, Pc, beta        : power curve per wind bin (wind speed, probability, cycle power,
                                 reel-out elevation)
        runtime                : [s]
    """
    t0 = time.time()
    if case['cls'] not in dist:          # class without data points: nothing to size
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
    Run the cases not yet cached, in parallel; return {cache key: run_case() summary} for all.
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
