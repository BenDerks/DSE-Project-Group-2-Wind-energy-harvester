from inputs.wind_conditions import wind_conditions

from models.turbine import required_rotor_area
from models.kite import required_kite_area




from design_options.turbine_options import HAWT_SPACE
from design_options.rigid_kite_options import RIGID_KITE_SPACE
from design_options.soft_kite_options import SOFT_KITE_SPACE
from design_options.vertical_turbine_options import VAWT_SPACE

from analysis.generate_designs import generate_designs
from analysis.performance import calculate_design_power
import numpy as np



"System requirement"

# P_target = 20000 # W


# "exvironmental inputs"

# altitude = 400 # m
# wind_speed = 10 # m/s

# conditions = wind_conditions(altitude, wind_speed)

# rho = conditions["air_density"]


# "turbine parameters"

# Cp = 0.4
# eta_turbine = 0.9

# "kite parameters"
# lift_to_drag = 5.0
# eta_kite = 0.5

# "calculations"
# turbine_area, turbine_diameter = required_rotor_area(P_target, rho, wind_speed, Cp, eta_turbine)

# kite_area = required_kite_area(P_target, rho, wind_speed, lift_to_drag, eta_kite)



hawt_designs = generate_designs("HAWT", HAWT_SPACE)
rigid_kite_designs = generate_designs("Rigid Kite", RIGID_KITE_SPACE)
soft_kite_designs = generate_designs("Soft Kite", SOFT_KITE_SPACE)
vawt_designs = generate_designs("VAWT", VAWT_SPACE)

all_designs = (
    hawt_designs
    + rigid_kite_designs
    + soft_kite_designs
    + vawt_designs
)

print("Number of HAWT designs:", len(hawt_designs))
print("Number of Rigid Kite designs:", len(rigid_kite_designs))
print("Number of Soft Kite designs:", len(soft_kite_designs))
print("Number of VAWT designs:", len(vawt_designs))
print("Total number of designs:", len(all_designs))

print("\nFirst generated design:")
print(all_designs[0])

# Reference conditions
rho = 1.225
wind_speed = 5  # m/s

# Evaluate all designs
evaluated_designs = [
    calculate_design_power(design, rho, wind_speed)
    for design in all_designs
]

# Print results
print("\nFirst evaluated design:")
print(evaluated_designs[0])

# Count designs reaching 20 kW
successful_designs = [
    design for design in evaluated_designs
    if design["meets_target"]
]

print("\nDesigns reaching 20 kW:", len(successful_designs))
