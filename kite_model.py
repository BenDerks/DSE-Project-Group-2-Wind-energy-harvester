"""
kite_model.py - Soft-kite pumping-cycle performance model (quasi-steady, with gravity)
=====================================================================================

WHAT THIS FILE DOES
    Sizes a soft-kite ground-generation system so that it delivers a target MEAN mechanical
    power (30 kW) in one wind class, and returns the metrics for the wind-power trade-off
    (power curve, cut-in/cut-out, hours out of operation, variability, kite and tether size).

HOW TO USE IT
    1. Make the wind input for one wind class: wind-speed bins `vs` [m/s] and their
       probabilities `pj` [-], valid at the average reel-out altitude. Until the wind analysis
       delivers real data, `weibull_bins()` makes a placeholder.
    2. Call   result = size_kite(z_min, z_max, F_max, 30e3, vs, pj)
    3. Read the design and metrics from the returned dictionary (see size_kite()).
    Run this file directly (python kite_model.py) for a worked example.

MODEL BOUNDARY
    Mechanical power at the winch drum. Generator efficiency is NOT included. Reel-out and
    reel-in power are returned separately, so efficiency can be applied later as
    P_out * eta and P_in / eta.

STRUCTURE (block letters A-H and equation numbers M1-M29 refer to the model doc)
    P, FI_RATIOS         fixed parameters with their sources
    geometry()           A   tether lengths, altitude, air density, tether size (once per run)
    coeffs()             B   effective lift and drag coefficients incl. tether drag
    state()              C   steady flight state of the kite incl. gravity (used by C, D and E)
    retraction()         D   reel-in, stepped in time while the kite climbs
    transition()         E   kite flies back down to the reel-out elevation
    operating_point()    F   best way to fly at one wind speed -> one point of the power curve
    weibull_bins()           placeholder wind input
    mean_power()         G   mean power over the wind class for one kite size
    size_kite()          loop + H  finds the kite area for the target power, computes metrics

UNITS
    SI everywhere: m, s, kg, N, W, J. Angles are in DEGREES only in the parameter dictionary P
    and in outputs labelled 'beta'; everywhere inside the functions they are in RADIANS.

ANGLE CONVENTIONS (van der Vlugt et al. 2019)
    beta  elevation angle of the tether above the ground
    theta polar angle measured from the vertical: theta = 90 deg - beta (so sin(theta) = cos(beta))
    phi   azimuth angle, sideways from straight downwind
    chi   course angle (flight direction seen from the ground station): 0 = down, 90 = sideways, 180 = up

MAIN SOURCES
    van der Vlugt, Bley, Noom & Schmehl (2019), Renewable Energy 131, 83-99   (model framework)
    Schelbergen, Kalverla, Schmehl & Watson (2020), Wind Energy Science 5     (soft-kite parameters)
    Joshi, Schmehl & Kruijff (2024), Wind Energy Science 9                    (tether sizing, cycle time)
    Cayon, van Deursen & Schmehl (2026), Wind Energy Science 11              (kite mass per area)
"""

import numpy as np
from math import sin, cos, sqrt, radians, degrees, pi, exp, gamma
from scipy.optimize import brentq


# =====================================================================================
# FIXED PARAMETERS
# All fixed parameters live in this one dictionary, so a sensitivity run only changes one
# entry, e.g.   p = dict(P, vmin=1.0)   and then pass p=p to any function.
# =====================================================================================
P = dict(
    # Atmosphere
    rho0=1.225,      # [kg/m3] air density at sea level                    van der Vlugt 2019, Eq. 2
    H=8550.0,        # [m]     density scale height                         van der Vlugt 2019, Eq. 2
    g=9.81,          # [m/s2]  gravitational acceleration                   standard value

    # Representative flight state during reel-out
    beta_o=25.0,     # [deg]   reel-out elevation angle (fixed)             Schelbergen 2020, Table 5
    phi_o=13.0,      # [deg]   reel-out azimuth angle                       Schelbergen 2020, Table 4
    chi_o=100.0,     # [deg]   reel-out course angle (mostly sideways)      Schelbergen 2020, Table 4

    # Aerodynamics of the soft kite (kite only, tether drag is added in coeffs())
    CLo=0.9,         # [-]     lift coefficient, powered (reel-out)         Schelbergen 2020, Table 4
    CDko=0.2,        # [-]     kite drag coefficient, powered               Schelbergen 2020, Table 4
    CLi=0.2,         # [-]     lift coefficient, depowered (reel-in)        Schelbergen 2020, Table 4
    CDki=0.1,        # [-]     kite drag coefficient, depowered             Schelbergen 2020, Table 4

    # Kite mass and size
    w=2.0,           # [kg/m2] airborne mass (wing + control unit) per m2 of projected area
                     #         Cayon 2026, Table 1 (1.83 for 19.75 m2, 2.00 for 46.85 m2)
    area_ratio=0.78, # [-]     projected area / flat area                   V9 and V3 kite data

    # Tether
    sigma=7e8,       # [Pa]    allowable tether stress (sized on fatigue)   Joshi 2024, Table 4
    rho_t=724.0,     # [kg/m3] tether density                               van der Vlugt 2019, Table 1
    CDc=1.1,         # [-]     tether cross-flow drag coefficient           van der Vlugt 2019, Table 1

    # Winch limits
    vmin=2.0,        # [m/s]   minimum reeling speed (both phases)          Schelbergen 2020, Table 4
    vmax=10.0,       # [m/s]   maximum reeling speed (both phases)          Schelbergen 2020, Table 4
    amax=5.0,        # [m/s2]  drum acceleration                            Joshi 2024, Table 4

    # Operational limits
    Fi_min=300.0,    # [N]     lowest reel-in force (keeps the kite tensioned)  Schelbergen 2020, Table 4
    beta_max=60.0,   # [deg]   highest elevation used to shed force in strong wind  Schelbergen 2020, Table 5
)

# Reel-in forces that operating_point() tries, as a fraction of F_max (at least Fi_min).
# The bounds follow Schelbergen 2020; the six values are a numerical search grid, not physics.
FI_RATIOS = [0.03, 0.06, 0.1, 0.15, 0.25, 0.4]


# =====================================================================================
# BLOCK A - GEOMETRY, WIND ALTITUDE AND TETHER (once per run)
# =====================================================================================
def geometry(zmin, zmax, Fmax, p=P):
    """
    Everything that depends only on the design inputs, not on the kite size.

    Why: tether lengths, the altitude where the wind is read, air density and the tether
    itself are the same for every trial kite area, so they are computed once per run.

    Inputs
        zmin, zmax : [m]  lowest and highest kite altitude during reel-out (design inputs)
        Fmax       : [N]  maximum tether force (design input)
        p          : parameter dictionary (default P)

    Output (dict)
        rmin, rmax : [m]      tether length at the start and end of reel-out            (M1)
        rbar       : [m]      average airborne tether length during reel-out            (M1)
        dl         : [m]      reel-out stroke rmax - rmin                               (M1)
        zbar       : [m]      average reel-out altitude, where the wind is read         (M2)
        rho        : [kg/m3]  air density at zbar                                       (M3)
        dt         : [m]      tether diameter                                           (M5)
        mt         : [kg]     airborne tether mass at rbar (used in the force balance)  (M6)
        mt_tot     : [kg]     total tether mass at rmax (for the mass budget)           (M6)

    Note: M4 (interpolating the wind data to zbar) is not done here; the wind bins passed to
    mean_power() must already be valid at geo['zbar'].
    """
    beta = radians(p['beta_o'])

    # M1: straight tether, so altitude = tether length * sin(elevation)
    rmin = zmin / sin(beta)
    rmax = zmax / sin(beta)
    rbar = 0.5 * (rmin + rmax)

    # M2: average altitude during reel-out (van der Vlugt 2019, Eq. 49)
    zbar = 0.5 * sin(beta) * (rmin + rmax)

    # M5: tether diameter from the maximum force and the allowable (fatigue) stress
    dt = sqrt(4 * Fmax / (pi * p['sigma']))

    return dict(
        rmin=rmin, rmax=rmax, rbar=rbar, dl=rmax - rmin, zbar=zbar,
        rho=p['rho0'] * exp(-zbar / p['H']),                  # M3: barometric density
        dt=dt,
        mt=pi / 4 * dt**2 * p['rho_t'] * rbar,                # M6: airborne tether mass
        mt_tot=pi / 4 * dt**2 * p['rho_t'] * rmax,            # M6: total tether mass
    )


# =====================================================================================
# BLOCK B - EFFECTIVE AERODYNAMIC COEFFICIENTS
# =====================================================================================
def coeffs(S, dt, r, p=P):
    """
    Lift and effective drag coefficients for reel-out and reel-in, tether drag included.

    Why: the tether adds drag, and how much depends on the kite area and on how much
    tether is in the air. So this is recomputed for every trial area S, and during reel-in
    for every tether length r.

    Inputs
        S  : [m2] projected kite area
        dt : [m]  tether diameter
        r  : [m]  airborne tether length (rbar for reel-out, current r during reel-in,
                  rmin for the transition)

    Output
        ((CL_out, CD_out), (CL_in, CD_in)) : [-] coefficients for reel-out and reel-in   (M9)

    The resultant coefficient C_R and the lift-to-drag ratio G* (M10) follow inside state().
    """
    # M9: tether drag lumped at the kite; 1/4 of the tether's drag area acts as kite drag
    tether = 0.25 * dt * r * p['CDc'] / S
    return (p['CLo'], p['CDko'] + tether), (p['CLi'], p['CDki'] + tether)


# =====================================================================================
# BLOCK C - FLIGHT STATE WITH GRAVITY (the heart of the model)
# =====================================================================================
def state(vw, f, S, m, mt, CL, CD, theta, phi, chi, rho, p=P, trace=None):
    """
    Steady flight state of the kite, including its weight, for one wind speed and one
    reeling factor.

    Why: every phase needs the tether force for a given reeling speed. Reel-out calls this
    for each trial reel-out factor, reel-in calls it in every time step and the transition
    calls it with f = 0.

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
        trace : list or None. Pass an empty list to record every iteration step.

    Output (dict), or None if no steady flight is possible in this state
        kappa : [-]    converged kinematic ratio (crosswind / along-tether apparent wind)
        lam   : [-]    tangential velocity factor (kite crosswind speed / wind speed)
        va    : [m/s]  apparent wind speed at the kite
        Fa    : [N]    aerodynamic force; Fr, Fth its radial and tangential part
        Ftg   : [N]    tether force at the ground station
        P     : [W]    mechanical power at the drum = Ftg * |f| * vw (always positive;
                       the caller knows whether it is produced or consumed)

    How: start from the massless solution (kappa = L/D) and adjust kappa until the
    lift-to-drag ratio implied by the force balance with gravity equals that of the kite
    (van der Vlugt 2019, Eq. 34-41).
    """
    g = p['g']
    CR = sqrt(CL**2 + CD**2)            # M10: resultant force coefficient
    G_target = CL / CD                  # M10: the kite's lift-to-drag ratio G*

    # M11: wind component along the tether (b) and along the flight direction (a)
    b = sin(theta) * cos(phi)
    a = cos(theta) * cos(phi) * cos(chi) - sin(phi) * sin(chi)

    # If the tether runs away faster than the wind pushes along it, there is no force
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
        if abs(G_implied - G_target) < 1e-8:
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


# =====================================================================================
# BLOCK D - REEL-IN, STEPPED IN TIME
# =====================================================================================
def retraction(vw, S, m, geo, Fmax, theta0, Fi, p=P, dT=0.02, theta_min=radians(5), record=False):
    """
    Reel-in at a constant set force while the kite climbs, stepped in time
    (van der Vlugt 2019, Eq. 42-45).

    Why: reel-in costs energy and time, and both depend strongly on how the kite climbs
    while it is pulled in. Evaluating reel-in only at its starting point made it far too
    expensive, so the climb is followed step by step.

    Inputs
        vw        : [m/s]  wind speed at the average altitude
        S, m      : [m2], [kg]  projected kite area and airborne mass
        geo       : dict from geometry()
        Fmax      : [N]    maximum tether force
        theta0    : [rad]  polar angle at the start of reel-in (= the reel-out angle)
        Fi        : [N]    set reel-in force
        dT        : [-]    time step as a fraction of dl / vw
                           (van der Vlugt uses 0.01; 0.02 halves the run time)
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
            f, s = f_slow, s_slow                        # even the slowest reel-in pulls more than Fi
        else:                                            # solve force(f) = Fi exactly
            f = brentq(lambda x: (flight(x)['Ftg'] if flight(x) else 0.0) - Fi,
                       f_fast, f_slow, xtol=1e-9)
            s = flight(f)

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
def transition(vw, S, m, geo, theta_from, theta_to, p=P, n=50):
    """
    Time for the powered kite to fly back down from the end-of-reel-in angle to the
    reel-out angle, at constant tether length rmin and without reeling (M22).

    Why: this phase produces nothing but takes time, so it lowers the cycle power.

    Inputs
        vw, S, m, geo : as in retraction()
        theta_from    : [rad] polar angle at the end of reel-in
        theta_to      : [rad] polar angle of reel-out
        n             : [-]   number of equal steps in theta

    Output
        t_x [s], or None if the kite cannot fly steadily somewhere on the way down
    """
    r = geo['rmin']
    mt = pi / 4 * geo['dt']**2 * p['rho_t'] * r
    (CL_out, CD_out), _ = coeffs(S, geo['dt'], r, p)

    thetas = np.linspace(theta_from, theta_to, n + 1)
    t = 0.0
    for th_a, th_b in zip(thetas[:-1], thetas[1:]):
        # Powered kite, no reeling (f = 0), flying down (chi = 0) in the plane phi = 0
        s = state(vw, 0.0, S, m, mt, CL_out, CD_out, 0.5 * (th_a + th_b), 0.0, 0.0, geo['rho'], p)
        if s is None:
            return None
        # The kite moves along the sphere at lam * vw, so a step of dtheta takes r*dtheta/(lam*vw)
        t += r * (th_b - th_a) / (s['lam'] * vw)
    return t


# =====================================================================================
# BLOCK F - BEST OPERATING POINT AT ONE WIND SPEED
# =====================================================================================
def operating_point(vw, S, geo, Fmax, p=P, n=150, detail=False):
    """
    Choose how to fly at one wind speed: reel-out factor, reel-in force and, in strong
    wind, a higher elevation, so that the cycle power is highest within the limits (M24).

    Why: at each wind speed the kite can be flown in many ways. A real control system
    aims for the most power within the force and speed limits; this function does the same.
    Its result is one point of the power curve.

    Inputs
        vw     : [m/s] wind speed at the average altitude
        S      : [m2]  projected kite area (airborne mass follows as w * S)
        geo    : dict from geometry()
        Fmax   : [N]   maximum tether force
        n      : [-]   number of reel-out factors tried
        detail : bool  also return all candidates (used for the reel-out factor chart)

    Output
        dict with the best operating point:
            Pc [W] cycle power, Po [W] reel-out power, Pi [W] average reel-in power,
            to, ti, tx [s] phase times, Fo [N] reel-out ground force, Fi [N] reel-in force,
            vo [m/s] reel-out speed, fo [-] reel-out factor,
            beta, beta_end [deg] reel-out elevation and elevation at the end of reel-in
        None       : below cut-in (no steady flight at 25 deg, or no positive cycle power)
        'cutout'   : above cut-out (force limit cannot be met even at 60 deg)
        With detail=True: (best, list_of_all_candidates)
    """
    m = p['w'] * S                                           # M7: airborne mass
    (CL_out, CD_out), _ = coeffs(S, geo['dt'], geo['rbar'], p)

    # Normal flight is at beta_o; only if the force limit cannot be met is the elevation
    # raised (1 deg at a time) to shed force (Schelbergen 2020)
    for beta in np.arange(p['beta_o'], p['beta_max'] + 0.01, 1.0):
        theta = radians(90 - beta)
        b = sin(theta) * cos(radians(p['phi_o']))

        # Trial reel-out factors within the speed limits (and below b, where force vanishes)
        f_trials = np.linspace(p['vmin'] / vw, min(p['vmax'] / vw, 0.999 * b), n)
        reel_out = [state(vw, f, S, m, geo['mt'], CL_out, CD_out, theta,
                          radians(p['phi_o']), radians(p['chi_o']), geo['rho'], p)
                    for f in f_trials]

        # No steady flight at all at the normal elevation: below cut-in
        if beta == p['beta_o'] and all(s is None for s in reel_out):
            return None

        # Keep only reel-out states that respect the force limit
        allowed = [(f, s) for f, s in zip(f_trials, reel_out)
                   if s is not None and s['Ftg'] <= Fmax]
        if not allowed:
            continue                                         # too much force: raise elevation

        # Combine every allowed reel-out with every trial reel-in force
        candidates = []
        for ratio in FI_RATIOS:
            Fi = max(ratio * Fmax, p['Fi_min'])
            reel_in = retraction(vw, S, m, geo, Fmax, theta, Fi, p)
            if reel_in is None:
                continue
            t_x = transition(vw, S, m, geo, reel_in['theta_end'], theta, p)
            if t_x is None:
                continue
            for f, s in allowed:
                vo = f * vw
                t_o = geo['dl'] / vo + vo / p['amax']                    # M22: reel-out time
                E_o = s['P'] * geo['dl'] / vo                            # reel-out energy
                Pc = (E_o - reel_in['E']) / (t_o + reel_in['t'] + t_x)   # M23: cycle power
                candidates.append(dict(Pc=Pc, Po=s['P'], Pi=reel_in['P'],
                                       to=t_o, ti=reel_in['t'], tx=t_x,
                                       Fo=s['Ftg'], Fi=Fi, vo=vo, fo=f, beta=beta,
                                       beta_end=90 - degrees(reel_in['theta_end'])))
        if not candidates:
            continue

        best = max(candidates, key=lambda c: c['Pc'])
        if best['Pc'] <= 0:
            return None                                      # reel-in costs more than reel-out gives
        return (best, candidates) if detail else best

    return 'cutout'                                          # even 60 deg cannot shed enough force


# =====================================================================================
# WIND INPUT (placeholder)
# =====================================================================================
def weibull_bins(mean, k, vs, dv):
    """
    Placeholder wind input: bin probabilities of a Weibull distribution.

    Why: lets the model run before the wind-distribution analysis delivers measured data.
    Replace the output with the measured histogram of each wind class, at geo['zbar'].

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
def mean_power(S, geo, Fmax, vs, pj, p=P):
    """
    Mean mechanical power over the wind class for one kite area (M26).

    Why: turns the power curve into one number per kite size, which the sizing loop
    compares with the target.

    Inputs
        S      : [m2]  projected kite area
        geo    : dict from geometry()
        Fmax   : [N]   maximum tether force
        vs, pj : [m/s], [-]  wind bins and their probabilities

    Output (tuple)
        P_mean : [W]  mean power = sum of pj * Pc(vj)
        curve  : list with the operating_point() result per bin (the power curve);
                 None = below cut-in, 'cutout' = above cut-out, both count as Pc = 0
    """
    curve = [operating_point(v, S, geo, Fmax, p) for v in vs]
    Pc = np.array([c['Pc'] if isinstance(c, dict) else 0.0 for c in curve])
    return float(np.sum(pj * Pc)), curve


# =====================================================================================
# SIZING LOOP AND BLOCK H - THE FUNCTION YOU CALL
# =====================================================================================
def size_kite(zmin, zmax, Fmax, P_target, vs, pj, S_lo=10.0, S_hi=150.0, p=P):
    """
    Find the kite area that delivers the target mean power in one wind class, and compute
    the trade-off metrics for that design.

    Why: this is the entry point; call it once per wind class.

    Inputs
        zmin, zmax : [m]   flying heights (design inputs)
        Fmax       : [N]   maximum tether force (design input)
        P_target   : [W]   target mean power (30e3)
        vs, pj     : [m/s], [-]  wind bins and probabilities of the wind class
        S_lo, S_hi : [m2]  search range for the kite area. These are search bounds, not
                     physical limits: S_lo must give less than the target, and S_hi should
                     become S_max from the volume budget.

    Output (dict), or None if even S_hi cannot reach the target (F_max too low)
        S, A       : [m2]   projected and flat kite area                   (M7, M8)
        m          : [kg]   airborne mass                                  (M7)
        geo        : dict   tether lengths, diameter and masses            (block A)
        P_mean     : [W]    mean power (equals the target)                 (M26)
        curve      : list   power curve, one operating_point() result per wind bin
        v_in       : [m/s]  cut-in wind speed (lowest bin with Pc > 0)     (M24)
        v_out      : [m/s]  cut-out wind speed; None = no cut-out within the bins
        T_out      : [h/yr] hours out of operation                         (M27)
        sigma_rel  : [-]    variability sigma_P / P_mean                   (M28)
        P_rated    : [W]    peak reel-out power F_max * v_max              (M25)
        CF         : [-]    capacity factor P_mean / P_rated               (M29)

    Run time: each mean_power() call takes about 40 s; a sizing run needs 6-8 of them.
    """
    geo = geometry(zmin, zmax, Fmax, p)

    # Mean power minus target, as a function of kite area (zero at the solution)
    def shortfall(S):
        return mean_power(S, geo, Fmax, vs, pj, p)[0] - P_target

    # If even the largest area falls short, no kite size works with this F_max
    if shortfall(S_hi) < 0:
        return None

    # Brent's method: narrow down the kite area until within 0.2 m2
    S = brentq(shortfall, S_lo, S_hi, xtol=0.2)

    # Recompute the power curve at the final area and derive the metrics (block H)
    P_mean, curve = mean_power(S, geo, Fmax, vs, pj, p)
    Pc = np.array([c['Pc'] if isinstance(c, dict) else 0.0 for c in curve])
    out_of_operation = [i for i, c in enumerate(curve) if not isinstance(c, dict)]
    P_rated = Fmax * p['vmax']                                           # M25

    return dict(
        S=S, A=S / p['area_ratio'], m=p['w'] * S, geo=geo,
        P_mean=P_mean, curve=curve,
        v_in=next((v for v, c in zip(vs, curve) if isinstance(c, dict)), None),
        v_out=next((v for v, c in zip(vs, curve) if c == 'cutout'), None),
        T_out=8760 * float(sum(pj[i] for i in out_of_operation)),         # M27
        sigma_rel=float(np.sqrt(np.sum(pj * (Pc - P_mean)**2)) / P_mean), # M28
        P_rated=P_rated, CF=P_mean / P_rated,                             # M29
    )


# =====================================================================================
# WORKED EXAMPLE (runs when the file is executed directly)
# =====================================================================================
if __name__ == '__main__':
    # Placeholder wind class: Weibull, mean 8 m/s, k = 2, bins of 1 m/s from 1 to 30 m/s
    vs = np.arange(1.0, 30.01, 1.0)
    pj = weibull_bins(8.0, 2.0, vs, 1.0)

    # Example design: flying heights 100-200 m, maximum tether force 40 kN
    geo = geometry(100.0, 200.0, 40e3)

    # Mean power of the example kite found in the model doc (71.7 m2)
    P_mean, _ = mean_power(71.7, geo, 40e3, vs, pj)
    print(f'S = 71.7 m2, F_max = 40 kN: mean power {P_mean / 1e3:.1f} kW')

    # Full sizing run for one wind class (takes a few minutes):
    # result = size_kite(100.0, 200.0, 40e3, 30e3, vs, pj)
    # print(result['S'], result['v_in'], result['T_out'])
