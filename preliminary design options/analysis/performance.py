
import numpy as np

from models.turbine import turbine_power
from models.kite import kite_power

from inputs.system_parameters import (
    P_target,
    Eta_turbine,
    Eta_kite,
    Cp_HAWT,
    Cp_VAWT
)


def calculate_design_power(design, rho, wind_speed):

    architecture = design["architecture"]

    # HAWT
    if architecture == "HAWT":

        diameter = design["rotor_diameter"]
        rotor_area = np.pi * diameter**2 / 4

        power = turbine_power(
            rho,
            wind_speed,
            rotor_area,
            Cp_HAWT,
            Eta_turbine
        )

    # VAWT
    elif architecture == "VAWT":

        diameter = design["rotor_diameter"]
        height = design["rotor_height"]
        rotor_area = diameter * height

        power = turbine_power(
            rho,
            wind_speed,
            rotor_area,
            Cp_VAWT,
            Eta_turbine
        )

    # Rigid Kite
    elif architecture == "Rigid Kite":

        area = design["wing_area"]
        lift_to_drag = design["lift_to_drag"]

        power = kite_power(
            rho,
            wind_speed,
            area,
            lift_to_drag,
            Eta_kite
        )

    # Soft Kite
    elif architecture == "Soft Kite":

        area = design["kite_area"]
        lift_to_drag = design["lift_to_drag"]

        power = kite_power(
            rho,
            wind_speed,
            area,
            lift_to_drag,
            Eta_kite
        )

    result = design.copy()
    result["power_W"] = power
    result["power_kW"] = power / 1000
    result["meets_target"] = power >= P_target

    return result
