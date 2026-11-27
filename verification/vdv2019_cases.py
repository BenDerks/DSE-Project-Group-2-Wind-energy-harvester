"""
vdv2019_cases.py - Re-run the two validation cycles of van der Vlugt et al. (2019)
==================================================================================

Van der Vlugt, Bley, Noom & Schmehl (2019), "Quasi-steady model of a pumping kite power
system", Renewable Energy 131, 83-99 (arXiv:1705.04133), validate their model on two measured
pumping cycles of the TU Delft 20 kW system. Table 1 gives every input; Tables 2 and 3 give
phase-averaged power and duration of their model ("gravity included") and of the measurement
("experiment"). This file re-runs both cycles with kite_model.state and compares.

    replica()             follows the paper as closely as possible: force-controlled phases,
                          log wind profile read at the kite's current altitude, time step
                          dT = 0.01. Differences with the paper's model therefore point at
                          the force balance (kite_model.state) itself.
    kite_model_as_used()  kite_model's own phase functions with ONE wind speed (at the average
                          traction altitude z_mt), as in the design studies. The difference
                          with replica() is the effect of that simplification.

INPUTS: Table 1 of the paper (CASES). L/D_k is the kite-only lift-to-drag ratio; tether drag
is added by kite_model.coeffs() as in the paper.
"""

import sys
from math import radians, degrees, sin, cos, pi, log
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import kite_model as km

# ---------------------------------------------------------------------------------------
# Table 1 of van der Vlugt et al. (2019)
# ---------------------------------------------------------------------------------------
CASES = {
    'moderate': dict(v_ref=5.9, z_ref=6.0, z0=0.07, v_mt=10.1, phi_o=10.6, beta_o=26.6,
                     chi_o=96.4, r_min=234.0, r_max=385.0, F_o=3069.0, F_i=750.0,
                     S=19.8, m=19.6, LD_o=3.6, LD_i=3.5, CL_o=0.59, CL_i=0.15,
                     CR_o=0.61, CR_i=0.20, d_t=0.004, rho_t=724.0, CD_t=1.1),
    'strong':   dict(v_ref=9.9, z_ref=6.0, z0=0.07, v_mt=18.2, phi_o=10.5, beta_o=27.0,
                     chi_o=100.9, r_min=390.0, r_max=720.0, F_o=3008.0, F_i=749.0,
                     S=10.2, m=15.0, LD_o=4.0, LD_i=3.1, CL_o=0.69, CL_i=0.17,
                     CR_o=0.71, CR_i=0.18, d_t=0.004, rho_t=724.0, CD_t=1.1),
}

# Tables 2 and 3 of the paper: phase mean power [kW] and duration [s]
PUBLISHED = {
    'moderate': {'model':      dict(P_i=-2.73, t_i=40, P_x=8.11, t_x=10, P_o=7.67, t_o=49, P_c=3.55, t_c=101),
                 'experiment': dict(P_i=-2.60, t_i=43, P_x=3.44, t_x=12, P_o=6.23, t_o=66, P_c=2.79, t_c=122)},
    'strong':   {'model':      dict(P_i=-4.03, t_i=60, P_x=23.67, t_x=7, P_o=22.57, t_o=38, P_c=7.59, t_c=106),
                 'experiment': dict(P_i=-3.64, t_i=67, P_x=8.50, t_x=9, P_o=19.12, t_o=52, P_c=6.48, t_c=128)},
}


def _cl_cd(CR, LD):
    """Lift and drag coefficient from the resultant coefficient C_R and L/D (C_R^2 = CL^2 + CD^2)."""
    CL = CR / np.sqrt(1 + 1 / LD**2)
    return CL, CL / LD


def params(c):
    """
    kite_model parameter dict for one case.

    C_L and C_D are derived from C_R and L/D_k, the quantities the paper's model uses. This
    reproduces the Table 1 C_L for three of the four phases (0.59, 0.69, 0.17). For the
    moderate retraction it gives C_L,i = 0.19, not the printed 0.15; 0.15 is inconsistent with
    C_R,i = 0.20 and L/D_k = 3.5 (it would give C_R = 0.156), so it is taken to be a typo.
    """
    CLo, CDo = _cl_cd(c['CR_o'], c['LD_o'])
    CLi, CDi = _cl_cd(c['CR_i'], c['LD_i'])
    return dict(km.P, CLo=CLo, CDko=CDo, CLi=CLi, CDki=CDi,
                rho_t=c['rho_t'], CDc=c['CD_t'],
                beta_o=c['beta_o'], phi_o=c['phi_o'], chi_o=c['chi_o'],
                vmin=0.0, vmax=50.0, amax=1e9)        # the paper's cycles are force controlled


def wind(c, z):
    """Logarithmic wind profile, van der Vlugt et al. (2019) Eq. (1)."""
    return c['v_ref'] * log(max(z, c['z0'] * 1.01) / c['z0']) / log(c['z_ref'] / c['z0'])


def _rho(z, p):
    return p['rho0'] * np.exp(-z / p['H'])


def _f_for_force(F_set, flight, f_lo, f_hi):
    """Reeling factor at which the ground tether force equals F_set (bisection on f)."""
    def err(f):
        s = flight(f)
        return (s['Ftg'] if s else 0.0) - F_set
    if err(f_lo) < 0 or err(f_hi) > 0:
        return None
    return km.solve(err, f_lo, f_hi, xtol=1e-6)


def replica(name, dT=0.01, phi_in=0.0, chi_in=pi):
    """
    The paper's cycle re-run with kite_model.state() in a time-stepped loop, as in the paper.

    Wind at the current kite altitude. Retraction at F_t,i (chi = 180 deg). Transition at
    constant length with chi = 0, reeling out only if the force would exceed F_t,o and in if it
    would drop below F_t,i. Traction at F_t,o with the representative angles of Table 1.
    Output: dict of phase mean power [kW] (P_i negative = consumed) and duration [s].
    """
    c = CASES[name]
    p = params(c)
    m, S = c['m'], c['S']
    area_t = pi / 4 * c['d_t']**2
    dt_step = dT * (c['r_max'] - c['r_min']) / c['v_mt']   # characteristic time, Eq. 43-44

    def coeffs(r):
        return km.coeffs(S, c['d_t'], r, p)

    # --- Retraction: from r_max at beta_o, force F_t,i, kite climbs -------------------
    r, theta = c['r_max'], radians(90 - c['beta_o'])
    t = E = 0.0
    while r > c['r_min']:
        z = r * cos(theta)
        vw, rho = wind(c, z), _rho(z, p)
        mt = area_t * c['rho_t'] * r
        _, (CL, CD) = coeffs(r)
        fl = lambda f: km.state(vw, f, S, m, mt, CL, CD, theta, phi_in, chi_in, rho, p)
        # Force control (Eq. 45): f may be positive (reeling out) while the kite is still low
        # and pulls more than F_t,i at zero reeling speed
        f = _f_for_force(c['F_i'], fl, -3.0, 0.95 * sin(theta))
        if f is None:
            return None
        s = fl(f)
        v = -f * vw
        step = min(dt_step, (r - c['r_min']) / v) if v > 0 else dt_step
        E -= s['Ftg'] * v * step
        t += step
        r -= v * step
        theta = max(theta - s['lam'] * vw * step / r, radians(1))
    t_i, E_i = t, E

    # --- Transition: chi = 0 (flying down) until beta_o is reached ---------------------
    t = E = 0.0
    theta_o = radians(90 - c['beta_o'])
    while theta < theta_o:
        z = r * cos(theta)
        vw, rho = wind(c, z), _rho(z, p)
        mt = area_t * c['rho_t'] * r
        (CL, CD), _ = coeffs(r)
        fl = lambda f: km.state(vw, f, S, m, mt, CL, CD, theta, 0.0, 0.0, rho, p)
        s0 = fl(0.0)
        f = 0.0
        if s0 is None or s0['Ftg'] < c['F_i']:          # too slack: reel in at F_t,i
            f = _f_for_force(c['F_i'], fl, -3.0, -1e-6) or 0.0
        elif s0['Ftg'] > c['F_o']:                       # too much force: reel out at F_t,o
            f = _f_for_force(c['F_o'], fl, 1e-6, 0.95 * sin(theta)) or 0.0
        s = fl(f)
        if s is None:
            return None
        step = dt_step
        E += s['Ftg'] * f * vw * step
        t += step
        r += f * vw * step
        theta = min(theta + s['lam'] * vw * step / r, theta_o)
        if s['lam'] * vw * step / r < 1e-6:
            break
    t_x, E_x = t, E

    # --- Traction: force F_t,o, representative angles, until r_max ---------------------
    t = E = 0.0
    theta = theta_o
    while r < c['r_max']:
        z = r * cos(theta)
        vw, rho = wind(c, z), _rho(z, p)
        mt = area_t * c['rho_t'] * r
        (CL, CD), _ = coeffs(r)
        fl = lambda f: km.state(vw, f, S, m, mt, CL, CD, theta, radians(c['phi_o']),
                                radians(c['chi_o']), rho, p)
        b = sin(theta) * cos(radians(c['phi_o']))
        f = _f_for_force(c['F_o'], fl, 1e-6, 0.999 * b)
        if f is None:
            return None
        s = fl(f)
        v = f * vw
        step = min(dt_step, (c['r_max'] - r) / v)
        E += s['Ftg'] * v * step
        t += step
        r += v * step
    t_o, E_o = t, E

    T = t_i + t_x + t_o
    return dict(P_i=E_i / t_i / 1e3, t_i=t_i, P_x=(E_x / t_x / 1e3 if t_x else 0.0), t_x=t_x,
                P_o=E_o / t_o / 1e3, t_o=t_o, P_c=(E_i + E_x + E_o) / T / 1e3, t_c=T)


def kite_model_as_used(name, transition_power=None):
    """
    The same cycle with kite_model's phase functions and ONE wind speed (at z_mt), as in the
    design studies.

    geometry() for tether and density, retraction() at F_t,i, transition(), and the traction
    state at F_t,o from state(). transition_power: None = kite_model.transition() as it is.
    """
    c = CASES[name]
    p = params(c)
    vw = c['v_mt']
    zmin = c['r_min'] * sin(radians(c['beta_o']))
    zmax = c['r_max'] * sin(radians(c['beta_o']))
    geo = km.geometry(zmin, zmax, 1e9, p)
    geo.update(dt=c['d_t'], mt=pi / 4 * c['d_t']**2 * c['rho_t'] * geo['rbar'])
    S, m = c['S'], c['m']
    theta_o = radians(90 - c['beta_o'])

    ri = km.retraction(vw, S, m, geo, 1e9, theta_o, c['F_i'], p, dT=0.01)
    F_out = c['F_o'] if p.get('transition_energy', True) else None
    tx = km.transition(vw, S, m, geo, ri['theta_end'], theta_o, p, n=50, F_out=F_out)
    if isinstance(tx, dict):
        t_x, E_x = tx['t'], tx['E']
    else:
        t_x, E_x = tx, 0.0
    (CL, CD), _ = km.coeffs(S, geo['dt'], geo['rbar'], p)
    fl = lambda f: km.state(vw, f, S, m, geo['mt'], CL, CD, theta_o, radians(c['phi_o']),
                            radians(c['chi_o']), geo['rho'], p)
    f = _f_for_force(c['F_o'], fl, 1e-6, 0.999 * sin(theta_o) * cos(radians(c['phi_o'])))
    s = fl(f)
    dl = geo['dl'] - (tx.get('dl', 0.0) if isinstance(tx, dict) else 0.0)
    t_o = dl / (f * vw)
    E_o = s['Ftg'] * dl
    T = ri['t'] + t_x + t_o
    return dict(P_i=-ri['P'] / 1e3, t_i=ri['t'], P_x=(E_x / t_x / 1e3 if t_x else 0.0), t_x=t_x,
                P_o=E_o / t_o / 1e3, t_o=t_o, P_c=(E_o + E_x - ri['E']) / T / 1e3, t_c=T)


if __name__ == '__main__':
    for name in CASES:
        print(f'\n{name} wind')
        rows = dict(PUBLISHED[name])
        rows['replica (kite_model.state)'] = replica(name)
        rows['kite_model as used'] = kite_model_as_used(name)
        keys = ['P_i', 't_i', 'P_x', 't_x', 'P_o', 't_o', 'P_c', 't_c']
        print(f'{"":28s}' + ''.join(f'{k:>8s}' for k in keys))
        for k, r in rows.items():
            print(f'{k:28s}' + ''.join(f'{r[x]:8.2f}' for x in keys))
