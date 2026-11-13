import numpy as np


def turbine_power(rho, wind_speed, rotor_area, Cp, eta_turbine):
    P_wind = 0.5 * rho * rotor_area * wind_speed**3

    P_mechanical = P_wind * Cp

    P_electrical = P_mechanical * eta_turbine

    return P_electrical


def required_rotor_area(P_target, rho, wind_speed, Cp, eta_turbine):
    
    area = P_target / (
        0.5 * rho * wind_speed**3 * Cp * eta_turbine)

    diameter = 2 * np.sqrt(area / np.pi)

    return area, diameter