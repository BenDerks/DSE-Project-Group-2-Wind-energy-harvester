"""
flight_state.py - Block C: steady flight state of the kite with gravity
=======================================================================
"""

import numpy as np
from math import sin, cos, sqrt

from .params import P
from .numerics import NUM


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
    """
    g = p['g']
    tol = NUM['tol_state'] if tol is None else tol
    CR = sqrt(CL**2 + CD**2)            # M10: resultant force coefficient
    G_target = CL / CD                  # M10: the kite's lift-to-drag ratio G*

    # M11: wind component along the tether (b) and along the flight direction (a)
    b = sin(theta) * cos(phi)
    a = cos(theta) * cos(phi) * cos(chi) - sin(phi) * sin(chi)

    # Tether reeled out faster than the wind along it: no force
    if b - f <= 0:
        return None

    q = 0.5 * rho * vw**2               # dynamic pressure of the real wind
    kappa = G_target                    # start: massless case, kappa = L/D (Eq. 15)

    for iteration in range(200):
        # M12: tangential velocity factor; negative root argument = no solution
        disc = a * a + b * b - 1 + kappa**2 * (b - f)**2
        if disc < 0:
            return None
        lam = a + sqrt(disc)

        # M13: apparent wind vector [radial, theta, phi] and its size
        va = vw * np.array([b - f,
                            cos(theta) * cos(phi) - lam * cos(chi),
                            -sin(phi) - lam * sin(chi)])
        va_abs = np.linalg.norm(va)

        # M14: size of the aerodynamic force
        Fa = q * S * CR * (1 + kappa**2) * (b - f)**2

        # M15: tangential part needed to carry the weight (kite + half the tether)
        Fth = -(0.5 * mt + m) * g * sin(theta)
        if Fa <= abs(Fth):              # aerodynamic force cannot even carry the weight
            return None

        # M16: radial part; the sideways (phi) part is zero because weight has none
        Fr = sqrt(Fa**2 - Fth**2)

        # M17: lift-to-drag ratio implied by this force and apparent-wind direction
        dot = Fr * va[0] + Fth * va[1]  # F_a . v_a  (= drag * |v_a|)
        if dot <= 0:                    # force points against the airflow: unphysical
            return None
        G_implied = sqrt((Fa * va_abs / dot)**2 - 1)

        if trace is not None:
            trace.append(dict(it=iteration + 1, kappa=kappa, lam=lam, va=va_abs / vw,
                              Fa=Fa, Fr=Fr, Fth=Fth, Gi=G_implied))

        # Converged when the implied L/D equals the kite's L/D
        if abs(G_implied - G_target) < tol:
            break

        # M17 update (van der Vlugt 2019, Eq. 41); stop if kappa runs away
        kappa *= sqrt(G_target / G_implied)
        if not 0.05 < kappa < 60:
            return None
    else:
        return None                     # no convergence in 200 passes: treat as no flight

    # M18: tether force at the kite, from the radial force balance
    F_radial = Fr - m * g * cos(theta)
    if F_radial <= 0:                   # tether would go slack
        return None

    # M19: tether force at the ground; the tether's own weight takes a bit off
    F_tangential = 0.5 * sin(theta) * mt * g
    Ftg = sqrt((F_radial - cos(theta) * mt * g)**2 + F_tangential**2)

    # M20: power at the drum
    return dict(kappa=kappa, lam=lam, va=va_abs, Fa=Fa, Fr=Fr, Fth=Fth, Ftg=Ftg,
                P=Ftg * abs(f) * vw)
