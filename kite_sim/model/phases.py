"""
phases.py - Block D (reel-in, stepped in time) and block E (transition back down)
=================================================================================
"""

import numpy as np
from math import sin, radians, degrees, pi

from .params import P
from .numerics import NUM, njit, solve
from .geometry import coeffs, tether_drag
from .flight_state import state, state_core


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

    The loop is in retraction_core() (compiled with numba); this wrapper unpacks the inputs.
    """
    dT = NUM['dT'] if dT is None else dT
    # Rows needed for the history: each step reels in at least v_min * dl / vw * dT
    rows = int(vw / (max(p['vmin'], 1e-3) * dT)) + 10 if record else 0
    hist = np.empty((rows, 5))
    ok, t, E, theta_end, n = retraction_core(
        float(vw), float(S), float(m), float(geo['rmax']), float(geo['rmin']), float(geo['dl']),
        float(geo['dt']), float(p['rho_t']), float(p['CDc']), float(p['CLi']), float(p['CDki']),
        float(geo['rho']), float(p['g']), float(p['vmin']), float(p['vmax']), float(Fmax),
        float(theta0), float(Fi), float(dT), float(theta_min), float(NUM['tol_state']), hist)
    if not ok:
        return None
    steps = [dict(t=float(h[0]), r=float(h[1]), beta=float(h[2]), F=float(h[3]), v=float(h[4]))
             for h in hist[:min(n, rows)]]
    return dict(t=t, E=E, P=E / t, theta_end=theta_end, hist=steps)


@njit(cache=True)
def retraction_core(vw, S, m, rmax, rmin, dl, dt, rho_t, CDc, CLi, CDki, rho, g, vmin, vmax,
                    Fmax, theta0, Fi, dT, theta_min, tol, hist):
    """
    retraction() on plain numbers (compiled with numba). Inputs as retraction(), with geo and p
    unpacked; hist is an array (rows, 5) that receives (t, r, beta [deg], F, v) per step.
    Output: (ok, t, E, theta_end, n); ok False = reel-in not possible, n = number of steps.
    """
    r, theta = rmax, theta0
    dt_step = dl / vw * dT              # time step (van der Vlugt 2019, Eq. 43-44)
    t = E = 0.0
    n = 0
    no_trace = np.empty((0, 7))         # state_core() records nothing
    while r > rmin:
        # Airborne tether mass and depowered coefficients at the CURRENT tether length (coeffs())
        mt = pi / 4 * dt**2 * rho_t * r
        CL_in, CD_in = CLi, CDki + tether_drag(dt, r, CDc, S)
        args = (vw, S, m, mt, CL_in, CD_in, theta, rho, g, tol, no_trace)

        # M21: find the reeling factor that gives exactly the set force Fi,
        # within the speed limits (negative f = reeling in)
        f_fast, f_slow = -vmax / vw, -vmin / vw
        ok_fast, F_fast, lam_fast = _reel_in(f_fast, *args)
        ok_slow, F_slow, lam_slow = _reel_in(f_slow, *args)
        if not ok_fast:
            return False, t, E, theta, n                 # no steady flight even at full speed
        if F_fast <= Fi:
            f, F, lam = f_fast, F_fast, lam_fast         # speed limit reached: force below Fi
        elif ok_slow and F_slow >= Fi:
            f, F, lam = f_slow, F_slow, lam_slow         # slowest reel-in still pulls more than Fi
        else:                                            # solve force(f) = Fi
            f = _solve_reel_in(f_fast, f_slow, Fi, args)
            ok, F, lam = _reel_in(f, *args)
            if not ok:
                return False, t, E, theta, n

        if F > Fmax:
            return False, t, E, theta, n                 # not allowed: tether overloaded

        # Add energy and time; the last step is shortened to land exactly on rmin
        v = abs(f) * vw
        step = min(dt_step, (r - rmin) / v)
        E += F * v * step
        t += step
        if n < hist.shape[0]:
            hist[n, 0] = t
            hist[n, 1] = r
            hist[n, 2] = 90 - degrees(theta)
            hist[n, 3] = F
            hist[n, 4] = v
        n += 1

        # Move the kite: tether shortens, kite climbs along the sphere at lam * vw
        r -= v * step
        theta = max(theta - lam * vw * step / r, theta_min)
    return True, t, E, theta, n


@njit(cache=True)
def _reel_in(f, vw, S, m, mt, CL, CD, theta, rho, g, tol, no_trace):
    """Reel-in flight state at reeling factor f (phi = 0, chi = 180 deg: flying up): (ok, Ftg, lam)."""
    s = state_core(vw, f, S, m, mt, CL, CD, theta, 0.0, pi, rho, g, tol, no_trace)
    return s[0], s[7], s[2]


@njit(cache=True)
def _solve_reel_in(a, b, Fi, args, xtol=1e-4, maxiter=60):
    """
    Reeling factor in [a, b] where the reel-in force equals Fi: solve() of numerics.py
    (Illinois method), with force(f) = Ftg (0 if no flight) - Fi.
    """
    ok, F, _ = _reel_in(a, *args)
    fa = (F if ok else 0.0) - Fi
    ok, F, _ = _reel_in(b, *args)
    fb = (F if ok else 0.0) - Fi
    if fa == 0:
        return a
    if fb == 0:
        return b
    if fa * fb > 0:
        raise ValueError('solve(): no sign change between a and b')
    side = 0
    x = a
    for _ in range(maxiter):
        x = (a * fb - b * fa) / (fb - fa)       # secant through the two bracket ends
        ok, F, _ = _reel_in(x, *args)
        fx = (F if ok else 0.0) - Fi
        if fx * fb > 0:                         # root lies between a and x
            b, fb = x, fx
            if side == -1:
                fa /= 2                         # Illinois step: avoid a stuck end
            side = -1
        else:                                   # root lies between x and b
            a, fa = x, fx
            if side == +1:
                fb /= 2
            side = +1
        if abs(b - a) < xtol or fx == 0:
            break
    return x


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
