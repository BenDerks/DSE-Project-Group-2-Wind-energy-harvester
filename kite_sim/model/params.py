"""
params.py - Fixed parameters of the kite model, with their sources
==================================================================
"""


# =====================================================================================
# FIXED PARAMETERS
# One dictionary, so a sensitivity run changes one entry:  p = dict(P, vmin=1.0), then pass
# p=p to any function.
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
    CDko=0.225,      # [-]     kite drag coefficient, powered -> L/D = 4.0  Oehler & Schmehl 2019
                     #         L/D 4.0 = AVERAGE measured in flight during reel-out, turns
                     #         included (Oehler & Schmehl 2019, WES 4: "about 4"; van der Vlugt
                     #         2019: 3.6 and 4.0). Schelbergen 2020, Table 4 uses 0.2 (L/D 4.5,
                     #         between that average and the measured peak of ~5). One L/D covers
                     #         the whole reel-out, so the average fits. Study range: L/D 3.6-5.0.
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
    Fi_min=300.0,    # [N]     lowest reel-in force (keeps kite tensioned)  Schelbergen 2020, Table 4
    beta_max=60.0,   # [deg]   max elevation to shed force in strong wind   Schelbergen 2020, Table 5

    # Generator (optional) [W]. None = no limit (reel-out power only limited by F_max * v_max).
    # A value caps the mechanical reel-out power at the generator rating (Luchsinger 2013,
    # regime III; used in the verification against products).
    P_gen_max=None,

    # Model options (switches, not physical values)
    optimise_elevation=True,  # strong wind (a limit active): raise the reel-out elevation while
                              # the cycle power increases (Schelbergen 2020, Table 5: 25-60 deg).
                              # False = raise it only when no state is allowed.
    depower=True,             # depower during reel-out to stay at the force or generator limit
                              # (Luchsinger 2013, regimes II-III); most depowered = reel-in
                              # coefficients. False = shed force only by raising the elevation
                              # (first model version).
    transition_energy=True,   # in the transition, reel out at the reel-out force when the kite
                              # pulls harder (van der Vlugt 2019, Sect. 2.9).
                              # False = transition only costs time (first model version).
)
