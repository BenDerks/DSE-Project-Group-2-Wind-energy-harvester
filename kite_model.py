"""
kite_model.py - Soft-kite pumping-cycle performance model (quasi-steady, with gravity)
======================================================================================

WHAT IT DOES
    Sizes a soft-kite ground-generation system for a target MEAN mechanical power (30 kW) in
    one wind class, and returns the trade-off metrics: power curve, cut-in/cut-out, hours out
    of operation, variability, kite and tether size.

HOW TO USE IT
    1. Wind input for one wind class: bin centres `vs` [m/s] and probabilities `pj` [-],
       valid at the average reel-out altitude. `weibull_bins()` gives a placeholder until
       the wind analysis delivers real data.
    2. result = size_kite(z_min, z_max, F_max, 30e3, vs, pj)
    3. Read the design and metrics from the returned dict (see size_kite()).
    Run  python kite_model.py  for a worked example.

MODEL BOUNDARY
    Mechanical power at the winch drum; generator efficiency is NOT included. Reel-out and
    reel-in power are returned separately, so it can be applied later as P_out * eta and
    P_in / eta.

STRUCTURE (blocks A-H and equations M1-M29 refer to the model doc)
    The code is in kite_sim/model/, one module per block. This file re-exports all of it,
    so  import kite_model as km; km.size_kite(...)  works as before.

    params.py           P                    fixed parameters with their sources
    numerics.py         FAST, ACCURATE, NUM  numerical settings; NUM is the active one
                        solve()              root finder (so only numpy is needed)
    geometry.py         A geometry()         tether lengths, altitude, air density, tether
                                             size (once per run)
                        B coeffs()           lift and effective drag incl. tether drag
    flight_state.py     C state()            steady flight state incl. gravity (used by C-E)
    phases.py           D retraction()       reel-in, stepped in time while the kite climbs
                        E transition()       flies back down to reel-out elevation (may reel out)
    operating_point.py  F operating_point()  best way to fly at one wind speed (power-curve point)
                        F _depowered()       reel-out state of a partly depowered kite at a limit
                        F _with_transition_energy()   cycle power incl. transition energy
    sizing.py           weibull_bins()       placeholder wind input
                        G mean_power()       mean power over the wind class for one kite size
                        H size_kite()        sizing loop: kite area for the target power + metrics

UNITS
    SI everywhere: m, s, kg, N, W, J. Angles are in DEGREES only in P and in outputs labelled
    'beta'; inside the functions they are in RADIANS.

ANGLE CONVENTIONS (van der Vlugt et al. 2019)
    beta  elevation angle of the tether above the ground
    theta polar angle from the vertical: theta = 90 deg - beta (so sin(theta) = cos(beta))
    phi   azimuth angle, sideways from straight downwind
    chi   course angle (flight direction seen from the ground station):
          0 = down, 90 = sideways, 180 = up

CHANGES AFTER THE VERIFICATION (November 2026; see verification/ and the overview doc)
    1. Transition energy: in the transition the tether is reeled out at the reel-out force
       when the kite pulls harder (van der Vlugt 2019, Sect. 2.9). Before, it only cost time.
       Effect: +2-4% cycle power; transition power 3.5 kW vs 3.4 kW measured (van der Vlugt).
    2. Depower during reel-out at the force or generator limit (Luchsinger 2013, regimes
       II-III; Schmehl's reference code awe_on_mars). Before, the model could only reel out
       faster at lower force, so with a generator limit the power collapsed above rated wind.
    3. Elevation optimised in strong wind: raised while the cycle power increases
       (Schelbergen 2020, Table 5: 25-60 deg). Before, only when no state was allowed.
    4. Optional generator limit P_gen_max (default None = no limit, as before).
    5. Site elevation z_site for the air density (geometry(), size_kite()).
    6. Kite drag C_D,k,o: briefly 0.225 (L/D 4.0, measured reel-out average of the TU Delft
       V3 kite), back to 0.2 (L/D 4.5, Schelbergen 2020) on 27 Nov 2026 (see P).
    7. Kite mass per area 2.0 -> 1.32 kg/m2: the average of three flown kites (see P).
    Switch 1-3 off in P (transition_energy, depower, optimise_elevation) to get the first
    model version back.

MAIN SOURCES (WES = Wind Energy Science)
    van der Vlugt, Bley, Noom & Schmehl (2019), Renewable Energy 131, 83-99 (model framework)
    Schelbergen, Kalverla, Schmehl & Watson (2020), WES 5         (soft-kite parameters)
    Joshi, Schmehl & Kruijff (2024), WES 9                        (tether sizing, cycle time)
    Cayon, van Deursen & Schmehl (2026), WES 11                   (kite mass per area)
    Oehler & Schmehl (2019), WES 4, 1-21                          (measured kite L/D)
    Luchsinger (2013), Pumping cycle kite power, in: Airborne Wind Energy, Springer
                                                                  (force/power-limited control)
"""

import numpy as np

from kite_sim.model import *        # noqa: F401,F403  whole model API (kite_sim/model/__init__.py)
from kite_sim.model import geometry, mean_power, size_kite, weibull_bins


# =====================================================================================
# WORKED EXAMPLE (python kite_model.py)
# =====================================================================================
if __name__ == '__main__':
    # Placeholder wind class: Weibull, mean 8 m/s, k = 2, bins of 1 m/s from 1 to 30 m/s
    vs = np.arange(1.0, 30.01, 1.0)
    pj = weibull_bins(8.0, 2.0, vs, 1.0)

    # Example design: flying heights 100-200 m, maximum tether force 40 kN
    geo = geometry(100.0, 200.0, 40e3)

    # Mean power of the model doc's example kite (71.7 m2)
    P_mean, _ = mean_power(71.7, geo, 40e3, vs, pj)
    print(f'S = 71.7 m2, F_max = 40 kN: mean power {P_mean / 1e3:.1f} kW')

    # Full sizing run for this wind class
    result = size_kite(100.0, 200.0, 40e3, 30e3, vs, pj)
    print(f"Sized kite: S = {result['S']:.1f} m2, cut-in {result['v_in']} m/s, "
          f"{result['T_out']:.0f} h/yr out of operation")
