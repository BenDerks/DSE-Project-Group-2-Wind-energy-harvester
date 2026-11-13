import numpy as np
import matplotlib.pyplot as plt
from analysis.wind_interface import get_design_wind_height


def isa_atmosphere(altitude):
    # Constants
    T0 = 288.15       # K
    p0 = 101325       # Pa
    g = 9.80665       # m/s^2
    R = 287.05        # J/(kg K)
    L = 0.0065        # K/m

    if altitude < 0 or altitude > 11000:
        raise ValueError(   "Altitude must be between 0 and 11000 m")

    T = T0 - L * altitude

    p = p0 * (T / T0) ** (g / (R * L))

    rho = p / (R * T)

    return rho, T, p


def wind_conditions(altitude, wind_speed):
    
    rho, T, p = isa_atmosphere(altitude)

    return {
        "altitude": altitude,
        "wind_speed": wind_speed,
        "air_density": rho,
        "temperature": T,
        "pressure": p
    }

def Wind_scalar(altitude, wind_speed):
    
    input_altitude = 10 # m hardcoded based on input data from the wind_conditions function
    alpha = 0.2 # generalized wind shear exponent for open terrain

    v_alt = wind_speed * (altitude / input_altitude) ** alpha # 

    return v_alt

def get_design_wind_height(design):

    architecture = design["architecture"]

    # Kite systems
    if architecture in ["Rigid Kite", "Soft Kite"]:
        height = design["altitude"]

    # HAWT: use hub height
    elif architecture == "HAWT":
        height = design["hub_height"]

    # VAWT: use rotor centre height
    elif architecture == "VAWT":
        height = design["rotor_height"] / 2 + 10 #10 meter elevated

    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    return height