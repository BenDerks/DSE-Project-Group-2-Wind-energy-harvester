"""
sizing.py - Placeholder wind input, block G (mean power) and the sizing loop with block H
=========================================================================================

size_kite() is the function you call, once per wind class.
"""

import numpy as np
from math import gamma

from .params import P
from .numerics import NUM, solve
from .geometry import geometry
from .operating_point import operating_point


# =====================================================================================
# WIND INPUT (placeholder)
# =====================================================================================
def weibull_bins(mean, k, vs, dv):
    """
    Placeholder wind input: bin probabilities of a Weibull distribution.

    Lets the model run before the wind analysis delivers measured data; replace it with the
    measured histogram of each wind class, at geo['zbar'].

    Inputs
        mean : [m/s] mean wind speed
        k    : [-]   Weibull shape factor
        vs   : [m/s] array of bin centres
        dv   : [m/s] bin width

    Output
        pj : [-] array with the probability of each bin
    """
    c = mean / gamma(1 + 1 / k)                                  # Weibull scale factor
    cdf = lambda v: 1 - np.exp(-(np.maximum(v, 0) / c)**k)
    return np.array([cdf(v + dv / 2) - cdf(v - dv / 2) for v in vs])


# =====================================================================================
# BLOCK G - MEAN POWER OVER THE WIND CLASS
# =====================================================================================
def mean_power(S, geo, Fmax, vs, pj, p=P, p_skip=0.0):
    """
    Mean mechanical power over the wind class for one kite area (M26): the number the
    sizing loop compares with the target.

    Inputs
        S      : [m2]  projected kite area
        geo    : dict from geometry()
        Fmax   : [N]   maximum tether force
        vs, pj : [m/s], [-]  wind bins and their probabilities
        p_skip : [-]   bins less probable than this are not computed (saves time in the
                       sizing loop; their contribution to the mean is negligible)

    Output (tuple)
        P_mean : [W]  mean power = sum of pj * Pc(vj)
        curve  : list with the operating_point() result per bin (the power curve);
                 None = below cut-in, 'cutout' = above cut-out, 'skipped' = not computed;
                 all three count as Pc = 0
    """
    curve = [operating_point(v, S, geo, Fmax, p) if pr >= p_skip else 'skipped'
             for v, pr in zip(vs, pj)]
    Pc = np.array([c['Pc'] if isinstance(c, dict) else 0.0 for c in curve])
    return float(np.sum(pj * Pc)), curve


# =====================================================================================
# SIZING LOOP AND BLOCK H - size_kite()
# =====================================================================================
def size_kite(zmin, zmax, Fmax, P_target, vs, pj, S_lo=10.0, S_hi=150.0, p=P, z_site=0.0):
    """
    Find the kite area that delivers the target mean power in one wind class (call once per
    class), and compute the trade-off metrics for that design.

    Inputs
        zmin, zmax : [m]   flying heights (design inputs)
        Fmax       : [N]   maximum tether force (design input)
        P_target   : [W]   target mean power (30e3)
        vs, pj     : [m/s], [-]  wind bins and probabilities of the wind class
        S_lo, S_hi : [m2]  search bounds for the kite area, not physical limits: S_lo must
                           give less than the target; S_hi should become S_max from the
                           volume budget.
        z_site     : [m]   ground elevation of the site above sea level (air density, M3)

    Output (dict), or None if even S_hi cannot reach the target (F_max too low)
        S, A       : [m2]   projected and flat kite area                           (M7, M8)
        m          : [kg]   airborne mass                                          (M7)
        geo        : dict   tether lengths, diameter and masses                    (block A)
        P_mean     : [W]    mean power (equals the target)                         (M26)
        curve      : list   power curve, one operating_point() result per wind bin
        v_in       : [m/s]  cut-in wind speed, refined between bins to ~0.03 m/s   (M24)
        v_out      : [m/s]  cut-out wind speed; None = no cut-out within the bins
        T_out      : [h/yr] hours out of operation, below v_in or above v_out      (M27)
        sigma_rel  : [-]    variability sigma_P / P_mean                           (M28)
        P_rated    : [W]    peak reel-out power F_max * v_max                      (M25)
        CF         : [-]    capacity factor P_mean / P_rated                       (M29)

    Run time with NUM = FAST: a few seconds per mean_power() call; a sizing run needs 6-10.
    """
    geo = geometry(zmin, zmax, Fmax, p, z_site)

    # Mean power minus target vs kite area (zero at the solution). Skips bins with negligible
    # probability while searching; memoised, so no area is computed twice.
    memo = {}

    def shortfall(S):
        if S not in memo:
            memo[S] = mean_power(S, geo, Fmax, vs, pj, p, p_skip=NUM['p_skip'])[0] - P_target
        return memo[S]

    # If even the largest area falls short, no kite size works with this F_max
    if shortfall(S_hi) < 0:
        return None
    if shortfall(S_lo) > 0:
        raise ValueError(f'S_lo = {S_lo} m2 already gives more than the target; lower S_lo')

    # Root finder: narrow down the kite area until within NUM['S_tol']
    S = solve(shortfall, S_lo, S_hi, xtol=NUM['S_tol'])

    # Recompute the full power curve (all bins) at the final area and derive the metrics (block H)
    P_mean, curve = mean_power(S, geo, Fmax, vs, pj, p)
    Pc = np.array([c['Pc'] if isinstance(c, dict) else 0.0 for c in curve])
    P_rated = Fmax * p['vmax']                                           # M25

    # Cut-in and cut-out: the bins only say between which bin centres they lie (1 m/s apart,
    # and one bin can hold 10% of the year). Refine both by bisection on the wind speed.
    v_in, v_out = _refine_limits(S, geo, Fmax, vs, curve, p)

    # M27: hours out of operation. Probability below v_in and above v_out, counting the part
    # of the boundary bin that lies beyond the limit (probability spread evenly over a bin).
    dv = vs[1] - vs[0]
    below = np.clip((v_in - (vs - dv / 2)) / dv, 0, 1) if v_in is not None else np.ones_like(vs)
    above = np.clip(((vs + dv / 2) - v_out) / dv, 0, 1) if v_out is not None else np.zeros_like(vs)
    T_out = 8760 * float(np.sum(pj * np.minimum(below + above, 1)))

    return dict(
        S=S, A=S / p['area_ratio'], m=p['w'] * S, geo=geo,
        P_mean=P_mean, curve=curve,
        v_in=v_in, v_out=v_out, T_out=T_out,                             # M24, M27
        sigma_rel=float(np.sqrt(np.sum(pj * (Pc - P_mean)**2)) / P_mean), # M28
        P_rated=P_rated, CF=P_mean / P_rated,                             # M29
    )


def _refine_limits(S, geo, Fmax, vs, curve, p=P, steps=5):
    """
    Cut-in and cut-out wind speed to within (bin width / 2**steps), by bisection.

    With 1 m/s bins the cut-in jumps from bin to bin, so the hours out of operation would
    jump by several hundred hours for small design changes.

    Inputs: the kite (S, geo, Fmax), the bin centres vs and the power curve from mean_power().
    Output: (v_in, v_out) [m/s]; v_out is None if there is no cut-out within the bins,
            v_in is None if the kite never produces power.
    """
    works = [isinstance(c, dict) for c in curve]
    if not any(works):
        return None, None

    def flies(v):
        return isinstance(operating_point(v, S, geo, Fmax, p), dict)

    # Cut-in: between the last bin without power and the first bin with power
    i = works.index(True)
    if i == 0:
        v_in = float(vs[0])
    else:
        lo, hi = float(vs[i - 1]), float(vs[i])
        for _ in range(steps):
            mid = 0.5 * (lo + hi)
            lo, hi = (lo, mid) if flies(mid) else (mid, hi)
        v_in = hi

    # Cut-out: between the last bin with power and the first 'cutout' bin after it
    j = next((k for k in range(i, len(curve)) if curve[k] == 'cutout'), None)
    if j is None:
        v_out = None
    else:
        lo, hi = float(vs[j - 1]), float(vs[j])
        for _ in range(steps):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if flies(mid) else (lo, mid)
        v_out = lo
    return v_in, v_out
