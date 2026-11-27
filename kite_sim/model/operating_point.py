"""
operating_point.py - Block F: best operating point at one wind speed
====================================================================
"""

import numpy as np
from math import sin, cos, radians, degrees

from .params import P
from .numerics import NUM
from .geometry import coeffs
from .flight_state import state
from .phases import retraction, transition


# =====================================================================================
# BLOCK F - BEST OPERATING POINT AT ONE WIND SPEED
# =====================================================================================
def operating_point(vw, S, geo, Fmax, p=P, n=None, detail=False):
    """
    Choose how to fly at one wind speed: reel-out factor, reel-in force and, in strong
    wind, depower and a higher elevation, so that the cycle power is highest within the
    limits (M24). Like a real control system; the result is one point of the power curve.

    How (per elevation, starting at beta_o):
      1. reel-out states for NUM['n_fo'] reeling factors between v_min and v_max;
      2. states pulling more than F_max (or the generator limit P_gen_max / v_o) are
         depowered to exactly that force if the depower range allows (_depowered(),
         p['depower']), otherwise dropped;
      3. each allowed reel-out is combined with each trial reel-in force (NUM['fi_ratios'])
         and its reel-in (retraction) and transition time; the best cycle is kept;
      4. the best cycle's transition is recomputed with reel-out at its reel-out force,
         which adds energy and shortens the reel-out (p['transition_energy']);
      5. if a limit was active, the next elevation (+ NUM['beta_step']) is tried and kept
         while it gives more cycle power (p['optimise_elevation']); if no state is allowed
         at all, the elevation is raised up to beta_max, beyond that it is cut-out.

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

    # Normal flight is at beta_o. Above the force (or generator) limit the kite is depowered
    # (below), and the elevation is raised in NUM['beta_step'] steps while that gives more
    # cycle power or is needed to shed force (Schelbergen 2020)
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

        # Keep reel-out states within the force limit and, if set, the generator power limit
        # (Luchsinger 2013, regime III; Joshi et al. 2024).
        # Depower (M24b): a state above the limit is kept if the kite can be depowered at that
        # reeling speed to pull exactly the limit (_depowered()). This is the force- and
        # power-limited control of Luchsinger (2013) and Schmehl's reference code (awe_on_mars),
        # and how soft kites are flown (the control unit depowers the wing). Without it, a
        # generator limit could only be met by reeling out faster at LOWER force, so the cycle
        # power dropped above rated wind.
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
        # Limit active here (some states depowered or excluded)? Only then can a higher
        # elevation be better.
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

        # Transition with reel-out at the chosen reel-out force (van der Vlugt 2019, Sect. 2.9),
        # for the best candidate only (the ranking above uses the transition time, which hardly
        # changes). The tether paid out in the transition shortens the reel-out.
        if p.get('transition_energy', True):
            best = _with_transition_energy(best, vw, S, m, geo, theta, p)

        if best['Pc'] <= 0 and overall is None:
            return None                                      # reel-in costs more than reel-out

        # Keep the best elevation. With a limit active, a higher elevation can give more cycle
        # power (reel-in starts higher and costs less), so keep raising it while that helps
        # (Schelbergen et al. 2020, Table 5: reel-out elevation optimised between 25-60 deg).
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

    Depower scales the kite's lift and drag coefficients by one factor k (constant kite L/D;
    tether drag not scaled), as in Luchsinger (2013) and Schmehl's reference code
    (awe_on_mars, regime 3): lower force factor at constant lift-to-drag ratio. k ranges from
    1 (powered) down to CL_i / CL_o (reel-in lift coefficient, Schelbergen et al. 2020, Table 4).

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
