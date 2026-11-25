import numpy as np

from design_options.turbine_options import HAWT_SPACE
from design_options.vertical_turbine_options import VAWT_SPACE
from design_options.rigid_kite_options import RIGID_KITE_SPACE
from design_options.soft_kite_options import SOFT_KITE_SPACE

from analysis.generate_designs import generate_designs
from analysis.performance import calculate_design_power
from analysis.feasibility import check_feasibility
from analysis.geometry import Calculate_geometry


P_target_min = 30000   # W
P_target_max = 60000   # W
P_target_step = 5000   # W

P_targets = np.arange(P_target_min, P_target_max + P_target_step, P_target_step)

rho = 1.225      # kg/m^3
wind_speed = 5   # m/s


design_spaces = {
    "HAWT": HAWT_SPACE,
    "VAWT": VAWT_SPACE,
    "Rigid Kite": RIGID_KITE_SPACE,
    "Soft Kite": SOFT_KITE_SPACE,
}

evaluated_designs = []

for architecture, design_space in design_spaces.items():

    for design in generate_designs(architecture, design_space):

        status, reasons = check_feasibility(Calculate_geometry(design))
        feasible = status != "FAIL"

        if feasible:
            evaluated_designs.append(
                calculate_design_power(design, rho, wind_speed)
            )


architectures = list(design_spaces.keys())

print(f"Feasible designs evaluated: {len(evaluated_designs)}")
print(f"Conditions: rho = {rho} kg/m^3, wind speed = {wind_speed} m/s\n")

header = f'{"Target [kW]":>12}' + "".join(f"{name:>12}" for name in architectures) + f'{"Total":>12}'
print(header)
print("-" * len(header))

sweep_results = []

for P_target in P_targets:

    counts = {
        architecture: sum(
            1 for design in evaluated_designs
            if design["architecture"] == architecture
            and design["power_W"] >= P_target
        )
        for architecture in architectures
    }

    sweep_results.append({"P_target": P_target, **counts})

    row = f"{P_target / 1000:>12.0f}"
    row += "".join(f"{counts[name]:>12}" for name in architectures)
    row += f"{sum(counts.values()):>12}"
    print(row)
