import numpy as np


def kite_power(rho, wind_speed, kite_area, lift_to_drag, eta):
    P_ideal = (
        (2 / 27)
        * rho
        * kite_area
        * lift_to_drag**2
        * wind_speed**3
    )

    P_electrical = P_ideal * eta

    return P_electrical


def required_kite_area(P_target, rho, wind_speed,
                       lift_to_drag, eta):
    area = (
        27 * P_target
        / (
            2 * rho
            * lift_to_drag**2
            * wind_speed**3
            * eta
        )
    )

    return area

