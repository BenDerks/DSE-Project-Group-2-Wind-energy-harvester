import numpy as np
import matplotlib.pyplot as plt


def isa_atmosphere(altitude):
    # Constants
    T0 = 288.15       # K
    p0 = 101325       # Pa
    g = 9.80665       # m/s^2
    R = 287.05        # J/(kg K)
    L = 0.0065        # K/m

    if altitude < 0 or altitude > 11000:
        raise ValueError("Altitude must be between 0 and 11000 m")

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