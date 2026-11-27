"""
references.py - Published data used to verify kite_model.py
===========================================================

Every number comes from a public source (named next to it). Values NOT in the source are
marked ASSUMED, with the reason.

    FALCON     Kitepower Falcon 100 kW, product brochure and power curve
    SKYSAILS   SkySails Power PN-14 (Venyo), product data sheet
    ETA        drivetrain efficiency: model mechanical power -> electrical, for products that
               report electrical values

The other two references (code and test data) have their own files:
    three_regime.py    independent massless reference model (Luchsinger 2013; Schmehl, awe_on_mars)
    vdv2019_cases.py   two measured pumping cycles and the published model results
                       (van der Vlugt et al. 2019)
"""

import numpy as np

# ---------------------------------------------------------------------------------------
# Kitepower Falcon
# ---------------------------------------------------------------------------------------
# Sources:
#   [B] Kitepower, "New powerful ways - The Kitepower Falcon 100 kW", product brochure,
#       https://thekitepower.com/resources/product/New%20Powerful%20Ways%20-%20The%20Kitepower%20Falcon%20100kW.pdf
#   [C] Schmehl, AWE course slides, lecture 3, prototype table
#       https://awecourse.github.io/slides/awe03.html
#   [U] power curve figure supplied by the user (100 m2 kite, 350 m tether), read off by eye
#       (accuracy about +-3 kW). The brochure states that its power values are electrical.
FALCON = dict(
    curve_v=np.array([0, 1, 2, 3, 4, 5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 11, 12, 13, 14,
                      15, 17, 20, 25], dtype=float),                                   # [U]
    curve_P=np.array([-6.5, -5.5, -4, -2, 1, 5, 14, 21, 32, 50, 66, 79, 90, 97, 103, 110,
                      114, 116.5, 118, 119, 120.5, 122, 123.5]) * 1e3,                # [U] W
    curve_area_flat=100.0,          # [m2]  kite of the curve                           [U]
    area_flat_V9=(60.0, 80.0),      # [m2]  flat area of the V9 kite variants            [B]
    area_proj_V9=(47.0, 62.0),      # [m2]  projected area of the V9 variants            [B]
    mass_60=73.0,                   # [kg]  kite incl. control unit and bridle, 60 m2    [C]
    F_max=3.5e3 * 9.81,             # [N]   "Force (t): 3,5 t" in the kite table         [B]
    r_max=350.0,                    # [m]   tether length                                [B]
    r_min=200.0,                    # [m]   ASSUMED: not given; 200 m is the minimum tether
                                    #       length of the TU Delft/Kitepower 25 m2 system in
                                    #       Schelbergen et al. (2020), WES 5, Sect. 5
    P_reel_out_el=130e3,            # [W]   electrical power during reel-out at rated     [B]
    P_rated_el=100e3,               # [W]   nominal power                                [B]
    AEP=450e6 * 3600,               # [J]   yearly energy output (wind not specified)     [B]
    wind_height='not stated',       #       no reference height on the power curve        [B]
)

# ---------------------------------------------------------------------------------------
# SkySails Power PN-14 (Venyo)
# ---------------------------------------------------------------------------------------
# Source:
#   [S] SkySails Power, "Data sheet Venyo (PN-14)", REV.2, 05/2025,
#       https://skysails-power.com/wp-content/uploads/2025/05/SkySails-Power-Data-sheet-Venyo-PN-14.pdf
#   [C] Schmehl, AWE course slides, lecture 3 (projected vs flat area of the PN-14 kite)
# Data-sheet AEP conditions [S]: single system, Rayleigh (k = 2), wind at flight altitude
# 200 m, kite 170 m2, tether 800 m, air density 1.2 kg/m3, 100% availability, 0% losses.
# This matches the model's own definition (wind at flying height, power before losses), so it
# is the cleanest product comparison.
SKYSAILS = dict(
    area_flat=170.0,                # [m2]                                               [S]
    proj_over_flat=90.0 / 180.0,    # [-]   projected / flat area of the PN-14 kite      [C]
    mass=52.0 + 55.0,               # [kg]  kite (35-52 kg, largest) + control pod 55 kg [S]
    d_tether=0.014,                 # [m]                                                [S]
    tether_mass_per_m=0.245,        # [kg/m]                                             [S]
    r_max=800.0,                    # [m]                                                [S]
    z_flight=200.0,                 # [m]   flight altitude of the AEP figures           [S]
    rho=1.2,                        # [kg/m3]                                            [S]
    P_gen=380e3,                    # [W]   installed generator power                    [S]
    P_rated_cycle=200e3,            # [W]   rated cycle power ("up to")                  [S]
    v_rated=13.0,                   # [m/s] at 200 m                                     [S]
    v_cutout=25.0,                  # [m/s] at 200 m                                     [S]
    AEP_9=760e6 * 3600,             # [J]   at 9 m/s annual mean (Rayleigh)              [S]
)

# ---------------------------------------------------------------------------------------
# Drivetrain efficiency (mechanical at the drum -> electrical)
# ---------------------------------------------------------------------------------------
# Source: Joshi, Schmehl & Kruijff (2024), WES 9, 2195, Sect. 2.6.1: gearbox 95% and power
# converter 95% (assumed in the paper), generator up to ~95% near rated speed (supplier
# polynomial, Eq. 35). Reel-out: P_el = eta * P_mech; reel-in: P_el = P_mech / eta.
# Storage (another 95% plus a second converter in the paper) is left out: the products report
# ground-station output.
ETA = 0.95 * 0.95 * 0.95


def falcon_curve(v):
    """Published Falcon power curve [W] at wind speeds v (linear between read-off points)."""
    return np.interp(v, FALCON['curve_v'], FALCON['curve_P'])
