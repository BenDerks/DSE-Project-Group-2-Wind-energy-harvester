"""
flight_state.py - Block C: steady flight state of the kite with gravity
=======================================================================
"""

import numpy as np
from math import sin, cos, sqrt

from .params import P
from .numerics import NUM, njit


# =====================================================================================
# BLOCK C - FLIGHT STATE WITH GRAVITY (the heart of the model)
# =====================================================================================
def state(vw, f, S, m, mt, CL, CD, theta, phi, chi, rho, p=P, trace=None, tol=None):
    """
    Steady flight state of the kite, including its weight, at one wind speed and reeling factor.

    Gives every phase its tether force: reel-out for each trial reel-out factor, reel-in in
    every time step, the transition with f = 0.

    Inputs
        vw    : [m/s]    wind speed at the average altitude
        f     : [-]      reeling factor = reeling speed / wind speed
                         (positive = reel-out, negative = reel-in, 0 = transition)
        S     : [m2]     projected kite area
        m     : [kg]     airborne mass (wing + control unit)
        mt    : [kg]     airborne tether mass
        CL,CD : [-]      lift and effective drag coefficient (from coeffs())
        theta : [rad]    polar angle (90 deg - elevation)
        phi   : [rad]    azimuth angle
        chi   : [rad]    course angle
        rho   : [kg/m3]  air density
        trace :          None, or an empty list to record every iteration step
        tol   : [-]      tolerance on the lift-to-drag ratio (default NUM['tol_state'])

    Output (dict), or None if no steady flight is possible in this state
        kappa : [-]    converged kinematic ratio (crosswind / along-tether apparent wind)
        lam   : [-]    tangential velocity factor (kite crosswind speed / wind speed)
        va    : [m/s]  apparent wind speed at the kite
        Fa    : [N]    aerodynamic force; Fr, Fth its radial and tangential part
        Ftg   : [N]    tether force at the ground station
        P     : [W]    mechanical power at the drum = Ftg * |f| * vw (always positive;
                       the caller knows whether it is produced or consumed)

    Method (van der Vlugt 2019, Eq. 34-41): start from the massless solution kappa = L/D and
    adjust kappa until the L/D implied by the force balance with gravity equals the kite's.
    The computation is in state_core() (compiled with numba); this wrapper builds the dict.
    """
    tol = NUM['tol_state'] if tol is None else tol
    hist = NO_TRACE if trace is None else np.empty((MAX_ITER, 7))
    ok, kappa, lam, va, Fa, Fr, Fth, Ftg, n = state_core(vw, f, S, m, mt, CL, CD, theta, phi,
                                                         chi, rho, p['g'], tol, hist)
    if trace is not None:
        for i in range(n):
            k, l_, v, a_, r_, t_, gi = (float(x) for x in hist[i])
            trace.append(dict(it=i + 1, kappa=k, lam=l_, va=v / vw, Fa=a_, Fr=r_, Fth=t_, Gi=gi))
    if not ok:
        return None
    # M20: power at the drum
    return dict(kappa=kappa, lam=lam, va=va, Fa=Fa, Fr=Fr, Fth=Fth, Ftg=Ftg, P=Ftg * abs(f) * vw)


MAX_ITER = 200                          # gravity iterations before a state counts as no flight
NO_TRACE = np.empty((0, 7))             # hist argument of state_core() when nothing is recorded


@njit(cache=True)
def state_core(vw, f, S, m, mt, CL, CD, theta, phi, chi, rho, g, tol, hist):
    """
    state() on plain numbers (compiled with numba). Inputs as state(), plus g [m/s2] and hist:
    an array (MAX_ITER, 7) that receives every iteration (kappa, lam, va, Fa, Fr, Fth, G_implied),
    or NO_TRACE.

    Output: (ok, kappa, lam, va, Fa, Fr, Fth, Ftg, n); ok False = no steady flight,
    n = number of iterations written to hist.
    """
    CR = sqrt(CL**2 + CD**2)            # M10: resultant force coefficient
    G_target = CL / CD                  # M10: the kite's lift-to-drag ratio G*

    # M11: wind component along the tether (b) and along the flight direction (a)
    b = sin(theta) * cos(phi)
    a = cos(theta) * cos(phi) * cos(chi) - sin(phi) * sin(chi)

    # Tether reeled out faster than the wind along it: no force
    if b - f <= 0:
        return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0

    q = 0.5 * rho * vw**2               # dynamic pressure of the real wind
    kappa = G_target                    # start: massless case, kappa = L/D (Eq. 15)
    lam = va_abs = Fa = Fr = Fth = 0.0
    record = hist.shape[0] > 0
    n = 0
    converged = False
    for _ in range(MAX_ITER):
        # M12: tangential velocity factor; negative root argument = no solution
        disc = a * a + b * b - 1 + kappa**2 * (b - f)**2
        if disc < 0:
            return False, kappa, lam, va_abs, Fa, Fr, Fth, 0.0, n
        lam = a + sqrt(disc)

        # M13: apparent wind vector [radial, theta, phi] and its size
        va_r = vw * (b - f)
        va_t = vw * (cos(theta) * cos(phi) - lam * cos(chi))
        va_p = vw * (-sin(phi) - lam * sin(chi))
        va_abs = sqrt(va_r * va_r + va_t * va_t + va_p * va_p)

        # M14: size of the aerodynamic force
        Fa = q * S * CR * (1 + kappa**2) * (b - f)**2

        # M15: tangential part needed to carry the weight (kite + half the tether)
        Fth = -(0.5 * mt + m) * g * sin(theta)
        if Fa <= abs(Fth):              # aerodynamic force cannot even carry the weight
            return False, kappa, lam, va_abs, Fa, Fr, Fth, 0.0, n

        # M16: radial part; the sideways (phi) part is zero because weight has none
        Fr = sqrt(Fa**2 - Fth**2)

        # M17: lift-to-drag ratio implied by this force and apparent-wind direction
        dot = Fr * va_r + Fth * va_t    # F_a . v_a  (= drag * |v_a|)
        if dot <= 0:                    # force points against the airflow: unphysical
            return False, kappa, lam, va_abs, Fa, Fr, Fth, 0.0, n
        G_implied = sqrt((Fa * va_abs / dot)**2 - 1)

        if record:
            hist[n, 0] = kappa
            hist[n, 1] = lam
            hist[n, 2] = va_abs
            hist[n, 3] = Fa
            hist[n, 4] = Fr
            hist[n, 5] = Fth
            hist[n, 6] = G_implied
        n += 1

        # Converged when the implied L/D equals the kite's L/D
        if abs(G_implied - G_target) < tol:
            converged = True
            break

        # M17 update (van der Vlugt 2019, Eq. 41); stop if kappa runs away
        kappa *= sqrt(G_target / G_implied)
        if not 0.05 < kappa < 60:
            return False, kappa, lam, va_abs, Fa, Fr, Fth, 0.0, n
    if not converged:                   # no convergence in MAX_ITER passes: treat as no flight
        return False, kappa, lam, va_abs, Fa, Fr, Fth, 0.0, n

    # M18: tether force at the kite, from the radial force balance
    F_radial = Fr - m * g * cos(theta)
    if F_radial <= 0:                   # tether would go slack
        return False, kappa, lam, va_abs, Fa, Fr, Fth, 0.0, n

    # M19: tether force at the ground; the tether's own weight takes a bit off
    F_tangential = 0.5 * sin(theta) * mt * g
    Ftg = sqrt((F_radial - cos(theta) * mt * g)**2 + F_tangential**2)
    return True, kappa, lam, va_abs, Fa, Fr, Fth, Ftg, n
