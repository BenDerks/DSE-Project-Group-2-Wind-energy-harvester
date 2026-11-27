"""
three_regime.py - Independent reference model: massless three-regime pumping-cycle model
========================================================================================

Numpy-only port of R. Schmehl's reference implementation (MIT licence),
https://github.com/awegroup/awe_on_mars/blob/main/powercurve.py, which follows the control
strategy of Luchsinger (2013), "Pumping cycle kite power", in: Airborne Wind Energy,
Springer, pp. 47-64, extended with a constant reel-out elevation angle.

Used ONLY to verify kite_model.py. It shares no functions with kite_model.py, so agreement
between the two is a real check.

MODEL (differences with kite_model.py kept on purpose, as in the original)
    - Kite and tether massless (no gravity); the tether only adds drag.
    - Reel-out: azimuth 0, crosswind (Loyd), constant elevation beta_o.
        tether force  F_o = q S C_R,o (1 + E_o^2) (cos(beta_o) - f_o)^2
    - Reel-in: steady flight straight towards the ground station at constant L/D E_i; the
      elevation follows from the force balance (not prescribed).
        tether force  F_i = q S C_R,i (sqrt(1 + E_i^2 (1 - f_i^2)) - f_i)^2 / (1 + E_i^2)
    - No transition phase, no acceleration time:
        P_c = (F_o v_o t_o - |F_i v_i| t_i) / (t_o + t_i),   t = stroke / |v|
    - Regime 1: f_o and f_i optimised freely (within reeling-speed limits).
      Regime 2: tether force limit -> f_o from F_o = F_max; f_i optimised.
      Regime 3: generator power limit -> reel-out speed fixed, kite depowered (force factor
                lowered) to keep F_o = F_max; f_i optimised.

The original optimises with scipy (COBYLA); here a dense grid search finds the same optimum
to within the grid spacing (checked: < 0.5% in power).
"""

import numpy as np


def _force_factor_out(CL, CD):
    """C_R (1 + E^2) with C_R = sqrt(CL^2 + CD^2), written as in the original code."""
    E2 = (CL / CD)**2
    return CL * np.sqrt(1 + 1 / E2) * (1 + E2)


def _force_factor_in(CL, CD):
    """C_R for reel-in (the (1 + E^2) term is inside the reel-in force expression)."""
    E2 = (CL / CD)**2
    return CL * np.sqrt(1 + 1 / E2)


def power_curve(v_w, rho, S, CL_out, CD_out, CL_in, CD_in, CD_tether, d_tether,
                r_min, r_max, beta_out_deg, F_max, P_max, v_in_max, v_out_max, n_grid=400):
    """
    Cycle power of the massless three-regime model at the wind speeds v_w.

    Inputs (SI units)
        v_w                 : array of wind speeds [m/s]
        rho                 : air density [kg/m3]
        S                   : kite area used for the aerodynamic force [m2]
        CL_out, CD_out      : kite lift and drag coefficient, reel-out [-]
        CL_in, CD_in        : kite lift and drag coefficient, reel-in [-] (no tether drag,
                              as in the original)
        CD_tether, d_tether : tether drag coefficient [-] and diameter [m]
        r_min, r_max        : tether length at start and end of reel-out [m]
        beta_out_deg        : reel-out elevation [deg]
        F_max               : tether force limit [N]
        P_max               : generator (reel-out) power limit [W]; np.inf for none
        v_in_max, v_out_max : maximum reel-in and reel-out speed [m/s] (positive numbers)

    Output (dict of arrays): P_c [W], P_o [W], P_i [W] (positive = consumed), f_o, f_i, F_o, F_i
    """
    r_m = 0.5 * (r_min + r_max)
    CD_o = CD_out + 0.25 * CD_tether * d_tether * r_m / S        # tether drag, as original
    E2o = (CL_out / CD_o)**2
    E2i = (CL_in / CD_in)**2
    kF_o0 = _force_factor_out(CL_out, CD_o)
    kF_i = _force_factor_in(CL_in, CD_in)
    cb = np.cos(np.radians(beta_out_deg))

    def p_norm(f_o, f_i, kF_o):
        """Cycle power normalised by kF_o * S * P_wind (Luchsinger 2013, as in the original)."""
        out = (cb - f_o)**2
        inn = (kF_i / kF_o) * (np.sqrt(1 + E2i * (1 - f_i**2)) - f_i)**2 / (1 + E2i)
        return (out - inn) * (f_i * f_o) / (f_i - f_o)

    res = {k: [] for k in ('P_c', 'P_o', 'P_i', 'f_o', 'f_i', 'F_o', 'F_i')}
    for v in np.atleast_1d(v_w):
        q = 0.5 * rho * v**2
        Pw = q * v
        fo_max = min(v_out_max / v, cb * 0.999)
        fi_min = max(-v_in_max / v, -0.999)
        fi_grid = np.linspace(fi_min, -1e-3, n_grid)
        kF_o = kF_o0

        # Regime 1: free optimisation of f_o and f_i
        fo_grid = np.linspace(1e-3, fo_max, n_grid)
        FO, FI = np.meshgrid(fo_grid, fi_grid)
        pc = p_norm(FO, FI, kF_o)
        j = np.unravel_index(np.argmax(pc), pc.shape)
        f_o, f_i = FO[j], FI[j]
        F_o = q * S * kF_o * (cb - f_o)**2

        # Regime 2: force limit -> f_o from F_o = F_max
        if F_o > F_max:
            # As in the original: the reel-out speed limit is only applied in regime 1
            f_o = cb - np.sqrt(F_max / (q * S * kF_o))
            F_o = F_max
            # Regime 3: generator limit -> reel-out speed fixed, depower to keep F_o = F_max
            if F_o * f_o * v > P_max:
                f_o = P_max / (F_max * v)
                kF_o = F_max / (q * S * (cb - f_o)**2)
            pc_i = p_norm(f_o, fi_grid, kF_o)
            f_i = fi_grid[np.argmax(pc_i)]

        p_c = p_norm(f_o, f_i, kF_o) * kF_o * S * Pw
        F_i = q * S * kF_i * (np.sqrt(1 + E2i * (1 - f_i**2)) - f_i)**2 / (1 + E2i)
        res['P_c'].append(max(p_c, 0.0))
        res['P_o'].append(F_o * f_o * v)
        res['P_i'].append(F_i * abs(f_i) * v)
        res['f_o'].append(f_o)
        res['f_i'].append(f_i)
        res['F_o'].append(F_o)
        res['F_i'].append(F_i)
    return {k: np.array(x) for k, x in res.items()}
