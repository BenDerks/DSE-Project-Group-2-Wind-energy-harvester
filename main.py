from inputs.wind_conditions import wind_conditions

from models.turbine import required_rotor_area
from models.kite import required_kite_area


import numpy as np



"System requirement"

P_target = 20000 # W


"exvironmental inputs"

altitude = 400 # m
wind_speed = 10 # m/s

conditions = wind_conditions(altitude, wind_speed)

rho = conditions["air_density"]


"turbine parameters"

Cp = 0.4
eta_turbine = 0.9
