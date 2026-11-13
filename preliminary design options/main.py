from inputs.wind_conditions import wind_conditions

from models.turbine import required_rotor_area
from models.kite import required_kite_area
from design_options.turbine_options import HAWT_SPACE
from design_options.rigid_kite_options import RIGID_KITE_SPACE
from design_options.soft_kite_options import SOFT_KITE_SPACE
from design_options.vertical_turbine_options import VAWT_SPACE
from analysis.transport import Check_hawt_transport
from analysis.generate_designs import generate_designs
from analysis.performance import calculate_design_power
from analysis.geometry import Calculate_geometry
from analysis.feasibility import check_feasibility
from analysis.wind_interface import get_design_wind_height


import numpy as np
import pandas as pd


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

# ==========================================
# PERFORMANCE CALCULATIONS
# ==========================================

evaluated_designs = []

for Design in all_designs:

    Architecture = Design["architecture"]

    # Select Kite or Non-Kite
    if Architecture in ["Rigid Kite", "Soft Kite"]:
        Design_type = "Kite"

    elif Architecture in ["HAWT", "VAWT"]:
        Design_type = "Non-Kite"

    else:
        raise ValueError(f"Unknown architecture: {Architecture}")

    # Determine wind reference height
    Height = get_design_wind_height(Design)

    # Temporary wind conditions
    rho = 1.225
    wind_speed = 5

    # Calculate performance
    Result = calculate_design_power(
        Design,
        rho,
        wind_speed
    )

    Result["design_type"] = Design_type
    Result["wind_height_m"] = Height

    evaluated_designs.append(Result)





# Calculate geometry for all designs
geometric_designs = [
    Calculate_geometry(design)
    for design in all_designs
]

feasible_designs = []
rejected_designs = []
tbd_designs = []

for design in geometric_designs:
    status, reasons = check_feasibility(design)

    if status == "PASS":
        feasible_designs.append(design)
    elif status == "FAIL":
        rejected_designs.append(design)
    else:
        tbd_designs.append(design)


print("\n--- Feasibility results ---")
print("Original designs:", len(all_designs))
print("Feasible designs:", len(feasible_designs))
print("Rejected designs:", len(rejected_designs))

# Select first HAWT design
Test_design = hawt_designs[0]

Transport_result = Check_hawt_transport(Test_design)

print("\n--- HAWT transport assessment ---")

for Parameter, Value in Transport_result.items():
    print(Parameter, ":", Value)
    

# ==========================================
# GEOMETRY FEASIBILITY
# ==========================================

Geometry_results = []

for Design in geometric_designs:

    Result = Design.copy()
    Result["Geometry_status"] = check_feasibility(Design)[0]
    Result["Geometry_reasons"] = check_feasibility(Design)[1]

    Geometry_results.append(Result)

print(geometric_designs[0])


# ==========================================
# RESULTS
# ==========================================

print("\n--- Geometry feasibility ---")

for Status in ["PASS", "FAIL", "TBD"]:

    Count = sum(
        Design["Geometry_status"] == Status
        for Design in Geometry_results
    )

    print(Status, ":", Count)
    
    
# print("\n--- Successful designs per architecture ---")

# for Architecture in ["HAWT", "VAWT", "Rigid Kite", "Soft Kite"]:

#     Count = sum(
#         Design["architecture"] == Architecture
#         for Design in successful_designs
#     )

