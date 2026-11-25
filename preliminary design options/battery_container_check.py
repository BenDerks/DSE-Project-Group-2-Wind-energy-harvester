from design_options.turbine_options import HAWT_SPACE
from design_options.vertical_turbine_options import VAWT_SPACE
from design_options.rigid_kite_options import RIGID_KITE_SPACE
from design_options.soft_kite_options import SOFT_KITE_SPACE
from design_options.battery_options import E_load_kWh

from analysis.generate_designs import generate_designs
from analysis.performance import calculate_design_power
from analysis.feasibility import check_feasibility
from analysis.geometry import Calculate_geometry
from analysis.transport import Usable_volume
from analysis.battery import battery_options, battery_autonomy
from analysis.packed_volume import Calculate_packed_volume
from analysis.solar import solar_sizing


P_output = 20000
Solar_fraction = 0.0
rho = 1.225        # kg/m^3
wind_speed = 5


solar = solar_sizing(P_output, Solar_fraction)
P_wind_required = P_output - solar["solar_average_W"]

print("--- Load split ---")
print(f"Constant load:        {P_output / 1000:.1f} kW")
print(f'Solar (average):      {solar["solar_average_W"] / 1000:.1f} kW '
      f'({solar["panels"]} panels, {solar["solar_peak_W"] / 1000:.1f} kWp, '
      f'{solar["solar_volume"]:.2f} m3)')
print(f"Wind harvester needs: {P_wind_required / 1000:.1f} kW\n")


batteries = battery_options()
autonomy_h = battery_autonomy(E_load_kWh, P_output)

print(f"Usable storage required: {E_load_kWh:.0f} kWh "
      f"({autonomy_h:.0f} h at {P_output / 1000:.0f} kW)")

print("--- Battery options ---")
print(f'{"Battery":<22}{"Nominal [kWh]":>14}{"Volume [m3]":>12}{"Mass [t]":>10}{"Left for harvester [m3]":>26}')

for battery in batteries:
    print(
        f'{battery["label"]:<22}'
        f'{battery["nominal_kWh"]:>14.0f}'
        f'{battery["volume"]:>12.2f}'
        f'{battery["mass_t"]:>10.1f}'
        f'{Usable_volume - battery["volume"] - solar["solar_volume"]:>26.2f}'
    )


design_spaces = {
    "HAWT": HAWT_SPACE,
    "VAWT": VAWT_SPACE,
    "Rigid Kite": RIGID_KITE_SPACE,
    "Soft Kite": SOFT_KITE_SPACE,
}

architectures = list(design_spaces.keys())

harvesters = []

for architecture, design_space in design_spaces.items():

    for design in generate_designs(architecture, design_space):

        result = calculate_design_power(design, rho, wind_speed)
        result.update(Calculate_packed_volume(design))

        status, reasons = check_feasibility(Calculate_geometry(design))
        feasible = status != "FAIL"
        result["feasible"] = feasible and result["sections_fit_door"]
        result["meets_output"] = result["power_W"] >= P_wind_required

        harvesters.append(result)

print("\n--- Harvester packed volume (incl. electrical equipment) ---")
print(f'{"Architecture":<14}{"Designs":>9}{"Min [m3]":>10}{"Max [m3]":>10}'
      f'{f">= {P_wind_required / 1000:.0f} kW":>10}')

for architecture in architectures:

    group = [h for h in harvesters if h["architecture"] == architecture]
    volumes = [h["harvester_volume"] for h in group]

    print(
        f"{architecture:<14}{len(group):>9}"
        f"{min(volumes):>10.1f}{max(volumes):>10.1f}"
        f'{sum(h["meets_output"] for h in group):>10}'
    )


combinations = []

for battery in batteries:

    for harvester in harvesters:

        total_volume = (
            harvester["harvester_volume"]
            + battery["volume"]
            + solar["solar_volume"]
        )
        fits = harvester["feasible"] and total_volume <= Usable_volume

        combinations.append({
            **harvester,
            "battery": battery["label"],
            "battery_volume": battery["volume"],
            "total_volume": total_volume,
            "fits_container": fits,
            "passes": fits and harvester["meets_output"],
        })


def print_table(title, key):

    print(f"\n--- {title} ---")
    print(f'{"Battery":<22}' + "".join(f"{name:>14}" for name in architectures))

    for battery in batteries:

        row = f'{battery["label"]:<22}'

        for architecture in architectures:

            group = [
                c for c in combinations
                if c["battery"] == battery["label"]
                and c["architecture"] == architecture
            ]
            cell = f"{sum(c[key] for c in group)}/{len(group)}"
            row += f"{cell:>14}"

        print(row)


print_table(
    "Designs that fit in the container with battery and solar", "fits_container"
)
print_table(
    f"Designs that fit AND deliver >= {P_wind_required / 1000:.0f} kW at {wind_speed} m/s",
    "passes",
)

passing = [c for c in combinations if c["passes"]]

print(f"\nTotal combinations: {len(combinations)}")
print(f"Passing combinations: {len(passing)}")
