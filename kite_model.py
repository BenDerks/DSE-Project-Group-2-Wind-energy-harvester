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
    P                    fixed parameters with their sources
    FAST, ACCURATE, NUM  numerical settings (step sizes, search grids); NUM is the active one
    solve()              root finder (so only numpy is needed)
    geometry()           A   tether lengths, altitude, air density, tether size (once per run)
    coeffs()             B   effective lift and drag coefficients incl. tether drag
    state()              C   steady flight state of the kite incl. gravity (used by C, D and E)
    retraction()         D   reel-in, stepped in time while the kite climbs
    transition()         E   kite flies back down to the reel-out elevation (may reel out)
    operating_point()    F   best way to fly at one wind speed -> one point of the power curve
    _depowered()         F   reel-out state of a partly depowered kite at a force limit
    _with_transition_energy()  F  cycle power with the energy of the transition
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

CHANGES AFTER THE VERIFICATION (November 2026, see verification/ and the overview doc)
    1. Transition energy: during the transition the tether is reeled out at the reel-out
       force when the kite pulls harder than that (van der Vlugt 2019, Sect. 2.9). Before,
       the transition only cost time. Effect: +2-4% cycle power; reproduces the measured
       transition power of van der Vlugt 2019 (3.5 vs 3.4 kW measured).
    2. Depower during reel-out at the force or generator limit (Luchsinger 2013, regimes
       II-III; reference code of Schmehl, awe_on_mars). Before, the model could only reel
       out faster at lower force, so with a generator limit the cycle power collapsed above
       rated wind.
    3. Elevation optimised in strong wind: raised while it increases the cycle power
       (Schelbergen 2020, Table 5: 25-60 deg). Before, it was only raised when no state
       was allowed at all.
    4. Optional generator limit P_gen_max (default None = no limit, as before).
    5. Site elevation z_site for the air density (geometry(), size_kite()).
    6. Kite drag C_D,k,o 0.2 -> 0.225: L/D 4.0, the measured reel-out average of the TU Delft
       V3 kite instead of 4.5 (see P).
    Options 1-3 can be switched off in P (transition_energy, depower, optimise_elevation)
    to get the first model version back.

MAIN SOURCES
    van der Vlugt, Bley, Noom & Schmehl (2019), Renewable Energy 131, 83-99   (model framework)
    Schelbergen, Kalverla, Schmehl & Watson (2020), Wind Energy Science 5     (soft-kite parameters)
    Joshi, Schmehl & Kruijff (2024), Wind Energy Science 9                    (tether sizing, cycle time)
    Cayon, van Deursen & Schmehl (2026), Wind Energy Science 11              (kite mass per area)
    Oehler & Schmehl (2019), Wind Energy Science 4, 1-21                     (measured kite L/D)
    Luchsinger (2013), Pumping cycle kite power, in: Airborne Wind Energy, Springer
                                                                          (force/power-limited control)
"""

import numpy as np
from math import sin, cos, sqrt, radians, degrees, pi, exp, gamma


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
    CDko=0.225,      # [-]     kite drag coefficient, powered. Set so that C_L/C_D = 4.0, the
                     #         AVERAGE lift-to-drag ratio measured in flight during reel-out for
                     #         this kite, turns included (Oehler & Schmehl 2019, WES 4: "about 4";
                     #         van der Vlugt et al. 2019: 3.6 and 4.0). Schelbergen 2020, Table 4
                     #         uses 0.2 (L/D 4.5), between that average and the measured peak of
                     #         about 5; the model uses one L/D for the whole reel-out, so the
                     #         average is the right value. Range for studies: L/D 3.6-5.0.
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

    # Generator (optional). None = no limit: the reel-out power is then only limited by
    # F_max * v_max. Set a value [W] to cap the mechanical reel-out power at the generator
    # rating (regime III of Luchsinger 2013; used in the verification against products).
    P_gen_max=None,

    # Model options (switches, not physical values)
    optimise_elevation=True,  # in strong wind (a limit active), raise the reel-out elevation
                              # while that increases the cycle power (Schelbergen 2020, Table 5:
                              # 25-60 deg). False = raise it only when no state is allowed.
    depower=True,             # depower the kite during reel-out to stay at the force or
                              # generator limit (Luchsinger 2013, regimes II-III), with the
                              # reel-in coefficients as the most depowered state. False = only
                              # raise the elevation to shed force (first model version).
    transition_energy=True,   # reel out at the reel-out force during the transition when the
                              # kite pulls harder than that (van der Vlugt 2019, Sect. 2.9).
                              # False = transition only costs time (first model version).
)

# =====================================================================================
# NUMERICAL SETTINGS
# Step sizes and search grids. These are numerical choices, not physics: coarser settings
# run faster but find the optimum less precisely. FAST is the default; ACCURATE is the
# setting used for the worked example in the model doc. FAST is about 7 times faster and
# gives about 1% less mean power for the example kite (so about 1.5% more kite area):
# use FAST for parameter studies and ACCURATE to check a final design.
# =====================================================================================
FAST = dict(
    n_fo=60,                          # reel-out factors tried per elevation (operating_point)
    fi_ratios=[0.03, 0.07, 0.15, 0.3],  # reel-in forces tried, fraction of F_max (bounds: Schelbergen 2020)
    beta_step=3.0,                    # [deg] elevation step when shedding force in strong wind
    dT=0.05,                          # reel-in time step as a fraction of dl / vw
    n_transition=15,                  # angle steps in the transition
    tol_state=1e-6,                   # convergence tolerance of the gravity iteration (on L/D)
    p_skip=1e-4,                      # wind bins with a smaller probability are skipped in the sizing loop
    S_tol=0.5,                        # [m2] tolerance on the kite area in the sizing loop
)
ACCURATE = dict(n_fo=150, fi_ratios=[0.03, 0.06, 0.1, 0.15, 0.25, 0.4], beta_step=1.0,
                dT=0.02, n_transition=50, tol_state=1e-8, p_skip=0.0, S_tol=0.2)
NUM = FAST


def solve(fun, a, b, xtol, maxiter=60):
    """
    Find x in [a, b] with fun(x) = 0 (Illinois variant of regula falsi; replaces scipy's brentq).

    Why: a simple, robust root finder so the model needs only numpy. fun(a) and fun(b) must
    have opposite signs. Stops when the bracket is smaller than xtol.
    """
    fa, fb = fun(a), fun(b)
    if fa == 0:
        return a
    if fb == 0:
        return b
    if fa * fb > 0:
        raise ValueError('solve(): no sign change between a and b')
    side = 0
    for _ in range(maxiter):
        x = (a * fb - b * fa) / (fb - fa)       # secant through the two bracket ends
        fx = fun(x)
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
# BLOCK A - GEOMETRY, WIND ALTITUDE AND TETHER (once per run)
# =====================================================================================
def geometry(zmin, zmax, Fmax, p=P, z_site=0.0):
    """
    Everything that depends only on the design inputs, not on the kite size.

    Why: tether lengths, the altitude where the wind is read, air density and the tether
    itself are the same for every trial kite area, so they are computed once per run.

    Inputs
        zmin, zmax : [m]  lowest and highest kite altitude during reel-out, ABOVE GROUND
                          (design inputs; the wind data is also given above ground)
        Fmax       : [N]  maximum tether force (design input)
        p          : parameter dictionary (default P)
        z_site     : [m]  ground elevation of the site above sea level (default 0 = sea level).
                          Only used for the air density: the kite flies at z_site + zbar
                          above sea level. At 1000 m elevation the density is ~11% lower.

    Output (dict)
        rmin, rmax : [m]      tether length at the start and end of reel-out            (M1)
        rbar       : [m]      average airborne tether length during reel-out            (M1)
        dl         : [m]      reel-out stroke rmax - rmin                               (M1)
        zbar       : [m]      average reel-out altitude, where the wind is read         (M2)
        rho        : [kg/m3]  air density at z_site + zbar above sea level              (M3)
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
        rho=p['rho0'] * exp(-(z_site + zbar) / p['H']),       # M3: barometric density
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
def state(vw, f, S, m, mt, CL, CD, theta, phi, chi, rho, p=P, trace=None, tol=None):
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
        tol   : [-]      convergence tolerance on the lift-to-drag ratio (default NUM['tol_state'])

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
    tol = NUM['tol_state'] if tol is None else tol
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


# =====================================================================================
# BLOCK D - REEL-IN, STEPPED IN TIME
# =====================================================================================
def retraction(vw, S, m, geo, Fmax, theta0, Fi, p=P, dT=None, theta_min=radians(5), record=False):
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
            f, s = f_slow, s_slow                        # even the slowest reel-in pulls more than Fi
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
    The powered kite flies back down from the end-of-reel-in angle to the reel-out angle
    (M22), at constant tether length r_min unless the tether force would exceed the reel-out
    force F_out: then the tether is reeled out at F_out, which produces energy
    (van der Vlugt et al. 2019, Sect. 2.9: "the reeling velocity is increased to stay below
    the value F_t,o").

    Why: this phase takes time, which lowers the cycle power, but in the measured cycles of
    van der Vlugt et al. (2019, Tables 2-3) it also PRODUCES power (3.4 kW and 8.5 kW), because
    the powered kite diving into the wind window pulls harder than the reel-out set force.
    Leaving that energy out (as the first version of this model did) underestimated the
    cycle power.

    Inputs
        vw, S, m, geo : as in retraction()
        theta_from    : [rad] polar angle at the end of reel-in
        theta_to      : [rad] polar angle of reel-out
        n             : [-]   number of equal steps in theta (default NUM['n_transition'])
        F_out         : [N]   reel-out force; None = never reel out (time only, no energy)

    Output (dict), or None if the kite cannot fly steadily somewhere on the way down
        t  : [s]  transition time t_x
        E  : [J]  energy produced while reeling out at F_out (0 if never needed)
        dl : [m]  tether paid out during the transition; this part of the reel-out stroke
                  is then no longer available for the reel-out phase
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


# =====================================================================================
# BLOCK F - BEST OPERATING POINT AT ONE WIND SPEED
# =====================================================================================
def operating_point(vw, S, geo, Fmax, p=P, n=None, detail=False):
    """
    Choose how to fly at one wind speed: reel-out factor, reel-in force and, in strong
    wind, depower and a higher elevation, so that the cycle power is highest within the
    limits (M24).

    Why: at each wind speed the kite can be flown in many ways. A real control system
    aims for the most power within the force and speed limits; this function does the same.
    Its result is one point of the power curve.

    How (per elevation, starting at beta_o):
      1. reel-out states for NUM['n_fo'] reeling factors between v_min and v_max;
      2. states that pull more than F_max (or than the generator limit P_gen_max / v_o)
         are depowered to exactly that force if the kite's depower range allows it
         (_depowered(), p['depower']); otherwise they are dropped;
      3. every allowed reel-out is combined with each trial reel-in force (NUM['fi_ratios'])
         and its reel-in (retraction) and transition time; the best cycle is kept;
      4. the transition of the best cycle is recomputed with reel-out at its reel-out force,
         which adds energy and shortens the reel-out (p['transition_energy']);
      5. if a limit was active, the next elevation (+ NUM['beta_step']) is tried and kept
         while it gives more cycle power (p['optimise_elevation']); if no state at all is
         allowed, the elevation is raised up to beta_max, beyond that it is cut-out.

    Inputs
        vw     : [m/s] wind speed at the average altitude
        S      : [m2]  projected kite area (airborne mass follows as w * S)
        geo    : dict from geometry()
        Fmax   : [N]   maximum tether force
        n      : [-]   number of reel-out factors tried (default NUM['n_fo'])
        detail : bool  also return all candidates (used for the reel-out factor chart)

    Search grids (reel-in forces, elevation step) come from NUM.

    Output
        dict with the best operating point:
            Pc [W] cycle power, Po [W] reel-out power, Pi [W] average reel-in power,
            to, ti, tx [s] phase times, Fo [N] reel-out ground force, Fi [N] reel-in force,
            vo [m/s] reel-out speed, fo [-] reel-out factor,
            Ei [J] reel-in energy, Ex [J] energy produced in the transition,
            dlx [m] tether paid out in the transition,
            beta, beta_end [deg] reel-out elevation and elevation at the end of reel-in
            depowered [-] (only if the reel-out kite is depowered) force-factor scale k
        None       : below cut-in (no steady flight at 25 deg, or no positive cycle power)
        'cutout'   : above cut-out (force limit cannot be met even at 60 deg, depowered)
        With detail=True: (best, list_of_all_candidates)
    """
    n = NUM['n_fo'] if n is None else n
    m = p['w'] * S                                           # M7: airborne mass
    (CL_out, CD_out), _ = coeffs(S, geo['dt'], geo['rbar'], p)
    P_gen = p.get('P_gen_max')

    # Normal flight is at beta_o. If the force (or generator) limit is exceeded, the kite is
    # depowered during reel-out (see below), and the elevation is raised (in steps of
    # NUM['beta_step']) as long as that gives more cycle power or is needed to shed force
    # (Schelbergen 2020)
    overall = None
    for beta in np.arange(p['beta_o'], p['beta_max'] + 0.01, NUM['beta_step']):
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

        # Keep only reel-out states that respect the force limit and, if one is set, the
        # generator power limit (regime III of Luchsinger 2013; Joshi et al. 2024).
        # Depower (M24b): a state that pulls harder than the limit is still allowed if the
        # kite can be depowered enough at that reeling speed to pull exactly the limit (see
        # _depowered()). This is the force- and power-limited control of Luchsinger (2013) and
        # of the reference code of Schmehl (awe_on_mars), and how soft kites are flown (the
        # control unit depowers the wing). Without it, the only way to respect a generator
        # limit was to reel out faster at LOWER force, which made the cycle power drop above
        # rated wind.
        allowed = []
        for f, s in zip(f_trials, reel_out):
            if s is None:
                continue
            F_lim = Fmax if P_gen is None else min(Fmax, P_gen / (f * vw))
            if s['Ftg'] <= F_lim:
                allowed.append((f, s))
            elif p.get('depower', True):
                s_dep = _depowered(vw, f, S, m, geo, theta, F_lim, s['Ftg'], p)
                if s_dep is not None:
                    allowed.append((f, s_dep))
        # Is a limit active at this elevation (some reel-out states had to be depowered or
        # were excluded)? Only then can a higher elevation be better.
        limited = (len(allowed) < sum(x is not None for x in reel_out)
                   or any(st.get('depowered') for _, st in allowed))
        if not allowed:
            continue                                         # too much force: raise elevation

        # Combine every allowed reel-out with every trial reel-in force
        candidates = []
        for ratio in NUM['fi_ratios']:
            Fi = max(ratio * Fmax, p['Fi_min'])
            reel_in = retraction(vw, S, m, geo, Fmax, theta, Fi, p)
            if reel_in is None:
                continue
            tx = transition(vw, S, m, geo, reel_in['theta_end'], theta, p)
            if tx is None:
                continue
            t_x = tx['t']
            for f, s in allowed:
                vo = f * vw
                t_o = geo['dl'] / vo + vo / p['amax']                    # M22: reel-out time
                E_o = s['P'] * geo['dl'] / vo                            # reel-out energy
                Pc = (E_o - reel_in['E']) / (t_o + reel_in['t'] + t_x)   # M23: cycle power
                candidates.append(dict(Pc=Pc, Po=s['P'], Pi=reel_in['P'],
                                       to=t_o, ti=reel_in['t'], tx=t_x, Ex=0.0, dlx=0.0,
                                       Ei=reel_in['E'],
                                       Fo=s['Ftg'], Fi=Fi, vo=vo, fo=f, beta=beta,
                                       beta_end=90 - degrees(reel_in['theta_end'])))
        if not candidates:
            continue

        best = max(candidates, key=lambda c: c['Pc'])

        # Transition with reel-out at the chosen reel-out force (van der Vlugt 2019, Sect. 2.9).
        # Done for the best candidate only (the ranking above uses the transition time, which
        # hardly changes); the tether paid out during the transition shortens the reel-out.
        if p.get('transition_energy', True):
            best = _with_transition_energy(best, vw, S, m, geo, theta, p)

        if best['Pc'] <= 0 and overall is None:
            return None                                      # reel-in costs more than reel-out gives

        # Keep the best elevation. In strong wind (a limit is active) a higher elevation can
        # give more cycle power, because reel-in then starts higher and costs less: keep
        # raising the elevation while that improves the cycle power (Schelbergen et al. 2020
        # optimise the reel-out elevation between 25 and 60 deg, Table 5).
        if overall is None or best['Pc'] > overall[0]['Pc']:
            overall = (best, candidates)
            if limited and p.get('optimise_elevation', True):
                continue
        break

    if overall is None:
        return 'cutout'                                      # even 60 deg cannot shed enough force
    if overall[0]['Pc'] <= 0:
        return None
    return overall if detail else overall[0]


def _depowered(vw, f, S, m, geo, theta, F_lim, F_powered, p=P, tol=0.01):
    """
    Reel-out state of a partly depowered kite that pulls exactly F_lim at reeling factor f.

    How depowering is modelled: the kite's lift and drag coefficients are scaled by the same
    factor k (constant kite L/D; the tether drag is not scaled). This is how the reference
    code of Schmehl (awe_on_mars, regime 3) and Luchsinger (2013) depower: they lower the
    force factor at constant lift-to-drag ratio. k is limited to the depower range of the
    kite, from 1 (powered) down to CL_i / CL_o (the reel-in lift coefficient,
    Schelbergen et al. 2020, Table 4).

    Inputs: as state(); F_lim [N] force to hold; F_powered [N] force of the powered kite.
    Output: state dict with Ftg = F_lim and P = F_lim * f * vw, or None if the kite cannot
            be depowered that far at this reeling speed (no steady flight, or out of range).
    """
    k_min = p['CLi'] / p['CLo']
    k = F_lim / F_powered                        # first guess: force proportional to k
    for _ in range(6):
        k = min(max(k, k_min), 1.0)
        (CL, CD), _ = coeffs(S, geo['dt'], geo['rbar'], dict(p, CLo=k * p['CLo'], CDko=k * p['CDko']))
        s = state(vw, f, S, m, geo['mt'], CL, CD, theta, radians(p['phi_o']), radians(p['chi_o']),
                  geo['rho'], p)
        if s is None:
            return None
        if abs(s['Ftg'] / F_lim - 1) < tol:
            return dict(s, Ftg=F_lim, P=F_lim * f * vw, depowered=k)
        if k == k_min and s['Ftg'] > F_lim:
            return None                          # fully depowered and still too much force
        k *= F_lim / s['Ftg']                    # fixed-point update (force ~ k)
    return None


def _with_transition_energy(c, vw, S, m, geo, theta, p):
    """
    Recompute the cycle power of operating point c with a transition that reels out at the
    reel-out force c['Fo'] (see transition()). Returns an updated copy of c.
    """
    tx = transition(vw, S, m, geo, radians(90 - c['beta_end']), theta, p, F_out=c['Fo'])
    if tx is None:
        return c
    dl_o = max(geo['dl'] - tx['dl'], 0.0)                   # what is left for reel-out
    t_o = dl_o / c['vo'] + c['vo'] / p['amax']               # M22
    E_o = c['Po'] * dl_o / c['vo']
    Pc = (E_o + tx['E'] - c['Ei']) / (t_o + c['ti'] + tx['t'])   # M23 with transition energy
    return dict(c, Pc=Pc, to=t_o, tx=tx['t'], Ex=tx['E'], dlx=tx['dl'])


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
def mean_power(S, geo, Fmax, vs, pj, p=P, p_skip=0.0):
    """
    Mean mechanical power over the wind class for one kite area (M26).

    Why: turns the power curve into one number per kite size, which the sizing loop
    compares with the target.

    Inputs
        S      : [m2]  projected kite area
        geo    : dict from geometry()
        Fmax   : [N]   maximum tether force
        vs, pj : [m/s], [-]  wind bins and their probabilities
        p_skip : [-]   bins with a probability below this are not computed (saves time in
                       the sizing loop; their contribution to the mean is negligible)

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
# SIZING LOOP AND BLOCK H - THE FUNCTION YOU CALL
# =====================================================================================
def size_kite(zmin, zmax, Fmax, P_target, vs, pj, S_lo=10.0, S_hi=150.0, p=P, z_site=0.0):
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
        z_site     : [m]   ground elevation of the site above sea level (air density, M3)

    Output (dict), or None if even S_hi cannot reach the target (F_max too low)
        S, A       : [m2]   projected and flat kite area                   (M7, M8)
        m          : [kg]   airborne mass                                  (M7)
        geo        : dict   tether lengths, diameter and masses            (block A)
        P_mean     : [W]    mean power (equals the target)                 (M26)
        curve      : list   power curve, one operating_point() result per wind bin
        v_in       : [m/s]  cut-in wind speed, refined between bins to ~0.03 m/s   (M24)
        v_out      : [m/s]  cut-out wind speed; None = no cut-out within the bins
        T_out      : [h/yr] hours out of operation, below v_in or above v_out      (M27)
        sigma_rel  : [-]    variability sigma_P / P_mean                   (M28)
        P_rated    : [W]    peak reel-out power F_max * v_max              (M25)
        CF         : [-]    capacity factor P_mean / P_rated               (M29)

    Run time with NUM = FAST: a few seconds per mean_power() call; a sizing run needs 6-10.
    """
    geo = geometry(zmin, zmax, Fmax, p, z_site)

    # Mean power minus target, as a function of kite area (zero at the solution).
    # Bins with negligible probability are skipped while searching; results are remembered
    # so the same area is never computed twice.
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

    Why: with 1 m/s bins the cut-in jumps from bin to bin, which makes the hours out of
    operation jump by several hundred hours for small design changes.

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

    # Full sizing run for this wind class
    result = size_kite(100.0, 200.0, 40e3, 30e3, vs, pj)
    print(f"Sized kite: S = {result['S']:.1f} m2, cut-in {result['v_in']} m/s, "
          f"{result['T_out']:.0f} h/yr out of operation")
