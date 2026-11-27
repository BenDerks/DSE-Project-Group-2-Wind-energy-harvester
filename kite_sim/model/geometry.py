"""
geometry.py - Block A (geometry, wind altitude, tether) and block B (aerodynamic coefficients)
==============================================================================================
"""

from math import sin, sqrt, radians, pi, exp

from .params import P
from .numerics import njit


# =====================================================================================
# BLOCK A - GEOMETRY, WIND ALTITUDE AND TETHER (once per run)
# =====================================================================================
def geometry(zmin, zmax, Fmax, p=P, z_site=0.0):
    """
    Everything that depends only on the design inputs, not on the kite size, so it is
    computed once per run and shared by every trial kite area.

    Inputs
        zmin, zmax : [m]  lowest and highest kite altitude during reel-out, ABOVE GROUND
                          (design inputs; the wind data is also above ground)
        Fmax       : [N]  maximum tether force (design input)
        p          :      parameter dictionary (default P)
        z_site     : [m]  site ground elevation above sea level (default 0). Only used for
                          the air density (kite at z_site + zbar above sea level); at 1000 m
                          the density is ~11% lower.

    Output (dict)
        rmin, rmax : [m]      tether length at the start and end of reel-out            (M1)
        rbar       : [m]      average airborne tether length during reel-out            (M1)
        dl         : [m]      reel-out stroke rmax - rmin                               (M1)
        zbar       : [m]      average reel-out altitude, where the wind is read         (M2)
        rho        : [kg/m3]  air density at z_site + zbar above sea level              (M3)
        dt         : [m]      tether diameter                                           (M5)
        mt         : [kg]     airborne tether mass at rbar (used in the force balance)  (M6)
        mt_tot     : [kg]     total tether mass at rmax (for the mass budget)           (M6)

    M4 (interpolating the wind data to zbar) is not done here: the wind bins passed to
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

    Tether drag depends on the kite area and the airborne tether length, so this is
    recomputed for every trial area S and, during reel-in, for every tether length r.

    Inputs
        S  : [m2] projected kite area
        dt : [m]  tether diameter
        r  : [m]  airborne tether length (rbar for reel-out, current r during reel-in,
                  rmin for the transition)

    Output
        ((CL_out, CD_out), (CL_in, CD_in)) : [-] coefficients for reel-out and reel-in   (M9)

    The resultant coefficient C_R and the lift-to-drag ratio G* (M10) are computed in state().
    """
    tether = tether_drag(dt, r, p['CDc'], S)
    return (p['CLo'], p['CDko'] + tether), (p['CLi'], p['CDki'] + tether)


@njit(cache=True)
def tether_drag(dt, r, CDc, S):
    """M9: tether drag lumped at the kite; 1/4 of the tether's drag area acts as kite drag [-]."""
    return 0.25 * dt * r * CDc / S
