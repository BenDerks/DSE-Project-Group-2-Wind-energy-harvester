import numpy as np
from inputs.system_parameters import Kite_practical_factor
def kite_power(rho, wind_speed, kite_area, lift_to_drag, eta):

    
    #assumtion Cl
    
    Cl_kite=1
    # Theoretical Loyd power
    P_ideal = (
        (2 / 27)
        * rho * Cl_kite
        * kite_area
        * lift_to_drag**2
        * wind_speed**3
    )

    # Practical power after operational losses
    P_practical = P_ideal * Kite_practical_factor

    # Electrical power after conversion losses
    P_electrical = P_practical * eta

    return P_ideal, P_practical, P_electrical


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

