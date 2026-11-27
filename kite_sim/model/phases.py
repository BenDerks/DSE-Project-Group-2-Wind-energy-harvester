"""
phases.py - Block D (reel-in, stepped in time) and block E (transition back down)
=================================================================================
"""

import numpy as np
from math import sin, radians, degrees, pi

from .params import P
from .numerics import NUM, solve
from .geometry import coeffs
from .flight_state import state


# =====================================================================================
# BLOCK D - REEL-IN, STEPPED IN TIME
# =====================================================================================
def retraction(vw, S, m, geo, Fmax, theta0, Fi, p=P, dT=None, theta_min=radians(5), record=False):
    """
    Reel-in at a constant set force while the kite climbs, stepped in time
    (van der Vlugt 2019, Eq. 42-45).

    Reel-in energy and time depend strongly on how the kite climbs while pulled in.
    Evaluating reel-in only at its start made it far too expensive, so the climb is
    followed step by step.

    Inputs
        vw        : [m/s]  wind speed at the average altitude
        S, m      : [m2], [kg]  projected kite area and airborne mass
        geo       : dict from geometry()
        Fmax      : [N]    maximum tether force
        theta0    : [rad]  polar angle at the start of reel-in (= the reel-out angle)
        Fi        : [N]    set reel-in force
        dT        : [-]    time step as a fraction of dl / vw (default NUM['dT'];
                           van der Vlugt uses 0.01, FAST uses 0.05)
        theta_min : [rad]  smallest polar angle allowed; numerical guard against passing
                           the zenith (no source; not reached in the example)
        record    : bool   store every step in 'hist'

    Output (dict), or None if reel-in is not possible within the limits
        t         : [s]    reel-in time t_i
        E         : [J]    reel-in energy E_i
        P         : [W]    average reel-in power E / t
        theta_end : [rad]  polar angle at the end of reel-in (start of the transition)
        hist      : list of dicts (t, r, beta [deg], F, v) per step, only if record=True
    """
    dT = NUM['dT'] if dT is None else dT
    r, theta = geo['rmax'], theta0
    dt_step = geo['dl'] / vw * dT       # time step (van der Vlugt 2019, Eq. 43-44)
    t = E = 0.0
    hist = []

    while r > geo['rmin']:
        # Airborne tether mass and depowered coefficients at the CURRENT tether length
        mt = pi / 4 * geo['dt']**2 * p['rho_t'] * r
        _, (CL_in, CD_in) = coeffs(S, geo['dt'], r, p)

        # Flight state for a trial reeling factor; reel-in: phi = 0, chi = 180 deg (flying up)
        def flight(f):
            return state(vw, f, S, m, mt, CL_in, CD_in, theta, 0.0, pi, geo['rho'], p)

        # M21: find the reeling factor that gives exactly the set force Fi,
        # within the speed limits (negative f = reeling in)
        f_fast, f_slow = -p['vmax'] / vw, -p['vmin'] / vw
        s_fast, s_slow = flight(f_fast), flight(f_slow)
        if s_fast is None:
            return None                                  # no steady flight even at full speed
        if s_fast['Ftg'] <= Fi:
            f, s = f_fast, s_fast                        # speed limit reached: force below Fi
        elif s_slow is not None and s_slow['Ftg'] >= Fi:
            f, s = f_slow, s_slow                        # slowest reel-in still pulls more than Fi
        else:                                            # solve force(f) = Fi
            def force_error(x):
                sx = flight(x)
                return (sx['Ftg'] if sx else 0.0) - Fi
            f = solve(force_error, f_fast, f_slow, xtol=1e-4)
            s = flight(f)
            if s is None:
                return None

        if s['Ftg'] > Fmax:
            return None                                  # not allowed: tether overloaded

        # Add energy and time; the last step is shortened to land exactly on rmin
        v = abs(f) * vw
        step = min(dt_step, (r - geo['rmin']) / v)
        E += s['Ftg'] * v * step
        t += step
        if record:
            hist.append(dict(t=t, r=r, beta=90 - degrees(theta), F=s['Ftg'], v=v))

        # Move the kite: tether shortens, kite climbs along the sphere at lam * vw
        r -= v * step
        theta = max(theta - s['lam'] * vw * step / r, theta_min)

    return dict(t=t, E=E, P=E / t, theta_end=theta, hist=hist)


# =====================================================================================
# BLOCK E - TRANSITION (flying back down)
# =====================================================================================
def transition(vw, S, m, geo, theta_from, theta_to, p=P, n=None, F_out=None):
    """
    Powered kite flies back down from the end-of-reel-in angle to the reel-out angle (M22).

    Tether length stays r_min unless the force would exceed the reel-out force F_out; then
    the tether is reeled out at F_out, which produces energy (van der Vlugt et al. 2019,
    Sect. 2.9: "the reeling velocity is increased to stay below the value F_t,o").
    This phase costs time, but in the measured cycles of van der Vlugt et al. (2019,
    Tables 2-3) it also PRODUCES power (3.4 kW and 8.5 kW): the kite diving into the wind
    window pulls harder than the reel-out set force. The first model version left this
    energy out and so underestimated the cycle power.

    Inputs
        vw, S, m, geo : as in retraction()
        theta_from    : [rad] polar angle at the end of reel-in
        theta_to      : [rad] polar angle of reel-out
        n             : [-]   number of equal steps in theta (default NUM['n_transition'])
        F_out         : [N]   reel-out force; None = never reel out (time only, no energy)

    Output (dict), or None if the kite cannot fly steadily somewhere on the way down
        t  : [s]  transition time t_x
        E  : [J]  energy produced while reeling out at F_out (0 if never needed)
        dl : [m]  tether paid out during the transition (no longer available for reel-out)
    """
    n = NUM['n_transition'] if n is None else n
    r = geo['rmin']
    mt = pi / 4 * geo['dt']**2 * p['rho_t'] * r
    (CL_out, CD_out), _ = coeffs(S, geo['dt'], r, p)

    thetas = np.linspace(theta_from, theta_to, n + 1)
    t = E = dl = 0.0
    for th_a, th_b in zip(thetas[:-1], thetas[1:]):
        th = 0.5 * (th_a + th_b)
        # Powered kite flying down (chi = 0) in the plane phi = 0, first without reeling
        def flight(f):
            return state(vw, f, S, m, mt, CL_out, CD_out, th, 0.0, 0.0, geo['rho'], p)
        s, f = flight(0.0), 0.0
        if s is None:
            return None
        if F_out is not None and s['Ftg'] > F_out:
            # Too much force: reel out so that the force equals F_out (within the speed limit)
            f_hi = min(p['vmax'] / vw, 0.999 * sin(th))
            s_hi = flight(f_hi)
            if s_hi is not None and s_hi['Ftg'] > F_out:
                f, s = f_hi, s_hi                  # speed limit: force stays above F_out
            else:
                def err(x):
                    sx = flight(x)
                    return (sx['Ftg'] if sx else 0.0) - F_out
                f = solve(err, 0.0, f_hi, xtol=1e-4)
                s = flight(f)
                if s is None:
                    return None
        # The kite moves along the sphere at lam * vw: a step of dtheta takes r*dtheta/(lam*vw)
        step = r * (th_b - th_a) / (s['lam'] * vw)
        t += step
        E += s['Ftg'] * f * vw * step
        dl += f * vw * step
    return dict(t=t, E=E, dl=dl)
