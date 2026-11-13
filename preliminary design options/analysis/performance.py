
import numpy as np

from models.turbine import turbine_power
from models.kite import kite_power

from inputs.system_parameters import (
    Eta_turbine,
    Eta_kite,
    Cp_HAWT,
    Cp_VAWT
)


def calculate_design_power(design, rho, wind_speed):

    architecture = design["architecture"]

    # Separate Kite and Non-Kite calculations
    if architecture in ["Rigid Kite", "Soft Kite"]:

        power_data = calculate_kite_power(
            design,
            rho,
            wind_speed
        )

    elif architecture in ["HAWT", "VAWT"]:

        power_data = calculate_turbine_power(
            design,
            rho,
            wind_speed
        )

    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    # Store results
    result = design.copy()
    result.update(power_data)

    result["meets_target"] = result["power_W"] >= P_target

    return result


def calculate_kite_power(design, rho, wind_speed):

    architecture = design["architecture"]

    if architecture == "Rigid Kite":
        area = design["wing_area"]

    else:  # Soft Kite
        area = design["kite_area"]

    lift_to_drag = design["lift_to_drag"]

    P_ideal, P_practical, power = kite_power(
        rho,
        wind_speed,
        area,
        lift_to_drag,
        Eta_kite
    )

    return {
        "P_ideal_W": P_ideal,
        "P_practical_W": P_practical,
        "power_W": power,
        "power_kW": power / 1000
    }


def calculate_turbine_power(design, rho, wind_speed):

    architecture = design["architecture"]

    if architecture == "HAWT":

        diameter = design["rotor_diameter"]
        rotor_area = np.pi * diameter**2 / 4
        Cp = Cp_HAWT

    else:  # VAWT

        diameter = design["rotor_diameter"]
        height = design["rotor_height"]
        rotor_area = diameter * height
        Cp = Cp_VAWT

    power = turbine_power(
        rho,
        wind_speed,
        rotor_area,
        Cp,
        Eta_turbine
    )

    return {
        "power_W": power,
        "power_kW": power / 1000
    }
