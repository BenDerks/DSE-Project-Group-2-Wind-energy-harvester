
# ==========================================
# WIND ENERGY - PRELIMINARY DESIGN
# ==========================================

import pandas as pd

from inputs.wind_conditions import (
    wind_conditions,
    Wind_scalar,
    get_design_wind_height,
    isa_atmosphere
)

from design_options.turbine_options import HAWT_SPACE
from design_options.rigid_kite_options import RIGID_KITE_SPACE
from design_options.soft_kite_options import SOFT_KITE_SPACE
from design_options.vertical_turbine_options import VAWT_SPACE

from analysis.transport import Check_hawt_transport
from analysis.generate_designs import generate_designs
from analysis.performance import calculate_design_power
from analysis.geometry import Calculate_geometry
from analysis.feasibility import check_feasibility
from analysis.packed_volume import Calculate_packed_volume
from analysis.transport import Container_volume, Packing_efficiency, Usable_volume
from analysis.battery import get_battery


# ==========================================
# SYSTEM REQUIREMENTS
# ==========================================

P_target = 30_000  # W

wind_speed_10m = 6.0  # m/s

Battery_option = "Li-ion (worst)"
Battery = get_battery(Battery_option)
Battery_volume = Battery["volume"]  # m^3


# ==========================================
# DESIGN GENERATION
# ==========================================

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

print("\n==========================================")
print("DESIGN GENERATION")
print("==========================================")

print("HAWT designs:", len(hawt_designs))
print("Rigid Kite designs:", len(rigid_kite_designs))
print("Soft Kite designs:", len(soft_kite_designs))
print("VAWT designs:", len(vawt_designs))
print("Total designs:", len(all_designs))


# ==========================================
# PERFORMANCE CALCULATIONS
# ==========================================

evaluated_designs = []

for Design in all_designs:

    Architecture = Design["architecture"]

    # Determine design type
    if Architecture in ["Rigid Kite", "Soft Kite"]:
        Design_type = "Kite"

    elif Architecture in ["HAWT", "VAWT"]:
        Design_type = "Non-Kite"

    else:
        raise ValueError(f"Unknown architecture: {Architecture}")

    # Determine individual operating height
    Height = get_design_wind_height(Design)

    # Atmospheric conditions at operating height
    rho = isa_atmosphere(Height)[0]

    # Scale wind speed from 10 m to operating height
    wind_speed = Wind_scalar(Height, wind_speed_10m)
    #wind_speed=
    # Calculate performance
    Result = calculate_design_power(
        Design,
        rho,
        wind_speed
    )

    # Combine design and performance data
    Evaluated_design = Design.copy()
    Evaluated_design.update(Result)

    Evaluated_design["design_type"] = Design_type
    Evaluated_design["wind_height_m"] = Height
    Evaluated_design["wind_speed_ms"] = wind_speed
    Evaluated_design["air_density"] = rho

    evaluated_designs.append(Evaluated_design)


print("\nPerformance calculations completed:", len(evaluated_designs))


# ==========================================
# GEOMETRY CALCULATIONS
# ==========================================

geometric_designs = [
    Calculate_geometry(Design)
    for Design in all_designs
]


# ==========================================
# GEOMETRY FEASIBILITY
# ==========================================

feasible_designs = []
rejected_designs = []
tbd_designs = []

geometry_results = []

for Design in geometric_designs:

    Status, Reasons = check_feasibility(Design)

    Result = Design.copy()
    Result["Geometry_status"] = Status
    Result["Geometry_reasons"] = Reasons

    geometry_results.append(Result)

    if Status == "PASS":
        feasible_designs.append(Result)

    elif Status == "FAIL":
        rejected_designs.append(Result)

    else:
        tbd_designs.append(Result)


print("\n==========================================")
print("GEOMETRY FEASIBILITY")
print("==========================================")

print("Original designs:", len(all_designs))
print("PASS:", len(feasible_designs))
print("FAIL:", len(rejected_designs))
print("TBD:", len(tbd_designs))


# ==========================================
# HAWT TRANSPORT ASSESSMENT
# ==========================================

Test_design = hawt_designs[0]

Transport_result = Check_hawt_transport(Test_design)

print("\n==========================================")
print("HAWT TRANSPORT ASSESSMENT")
print("==========================================")

for Parameter, Value in Transport_result.items():
    print(Parameter, ":", Value)


# ==========================================
# COMBINE PERFORMANCE AND GEOMETRY
# ==========================================

combined_designs = []

for Performance, Geometry in zip(
    evaluated_designs,
    geometry_results
):

    Design = Performance.copy()

    Design["Geometry_status"] = Geometry["Geometry_status"]
    Design["Geometry_reasons"] = Geometry["Geometry_reasons"]

    Packing = Calculate_packed_volume(Performance)

    Design["harvester_volume"] = Packing["harvester_volume"]
    Design["total_volume"] = Packing["harvester_volume"] + Battery_volume
    Design["fits_container"] = (
        Geometry["Geometry_status"] != "FAIL"
        and Packing["sections_fit_door"]
        and Design["total_volume"] <= Usable_volume
    )

    combined_designs.append(Design)


# POWER OUTPUT RESULTS
# ==========================================

df = pd.DataFrame(combined_designs)

# Select power output in Watts
power_col = "power_W"

if power_col not in df.columns:
    raise ValueError(
        "power_W not found in performance results."
    )

print("\nSelected power column:", power_col)

# Create readable results table
power_table = pd.DataFrame({

    "Architecture": df["architecture"],

    "Design ID": [
        f"{architecture}-{i+1:03d}"
        for architecture, i in zip(
            df["architecture"],
            df.groupby("architecture").cumcount()
        )
    ],

    "Total volume incl. battery (m3)": df["total_volume"],

    "Harvester volume (m3)": df["harvester_volume"],

    "Rotor diameter (m)": df["rotor_diameter"],

    "Kite area (m2)": df["wing_area"].combine_first(df["kite_area"]),

    "Wind height (m)": df["wind_height_m"],

    "Wind speed (m/s)": df["wind_speed_ms"],

    "Air density (kg/m3)": df["air_density"],

    "Power output (kW)": df[power_col] / 1000,

    "Meets 30 kW": df[power_col] >= P_target,

    "Geometry status": df["Geometry_status"],

    "Fits container": df["fits_container"]

})

power_table = power_table.sort_values(
    ["Total volume incl. battery (m3)", "Power output (kW)"],
    ascending=[True, False]
)

fitting_table = power_table[power_table["Fits container"]].sort_values(
    ["Power output (kW)", "Total volume incl. battery (m3)"],
    ascending=[False, True]
)

architecture_sheets = {
    "HAWT": "HAWT",
    "Soft Kites": "Soft Kite",
    "Rigid Kites": "Rigid Kite",
}

architecture_tables = {}

for sheet_name, architecture in architecture_sheets.items():

    table = power_table[power_table["Architecture"] == architecture]

    fitting = table[table["Fits container"]].sort_values(
        ["Power output (kW)", "Total volume incl. battery (m3)"],
        ascending=[False, True]
    )

    not_fitting = table[~table["Fits container"]].sort_values(
        ["Total volume incl. battery (m3)", "Power output (kW)"],
        ascending=[True, False]
    )

    architecture_tables[sheet_name] = pd.concat([fitting, not_fitting])

print("\n==========================================")
print("CONTAINER VOLUME CHECK")
print("==========================================")

print(f"Container volume: {Container_volume} m3, "
      f"packing efficiency {Packing_efficiency:.0%} "
      f"-> usable {Usable_volume:.2f} m3")
print(f"Battery reserved: {Battery_option}, {Battery['nominal_kWh']:.0f} kWh nominal "
      f"({Battery_volume:.2f} m3, {Battery['mass_t']:.1f} t)")
print("Designs that fit:", len(fitting_table), "of", len(power_table))


# ==========================================
# ARCHITECTURE SUMMARY
# ==========================================

print("\n==========================================")
print("POWER OUTPUT SUMMARY")
print("==========================================")

summary = power_table.groupby("Architecture").agg(

    Number_of_designs=("Design ID", "count"),

    Average_power_kW=("Power output (kW)", "mean"),

    Maximum_power_kW=("Power output (kW)", "max"),

    Minimum_power_kW=("Power output (kW)", "min"),

    Designs_meeting_target=("Meets 30 kW", "sum"),

    Designs_fitting_container=("Fits container", "sum"),

    Geometry_PASS=("Geometry status",
                   lambda x: (x == "PASS").sum())

).round(2)

print(summary.to_string())


# ==========================================
# EXCEL EXPORT
# ==========================================

filename = "Power_Output_Comparison.xlsx"

with pd.ExcelWriter(
    filename,
    engine="openpyxl"
) as writer:

    power_table.to_excel(
        writer,
        sheet_name="Power Outputs",
        index=False
    )

    fitting_table.to_excel(
        writer,
        sheet_name="Fitting by Power",
        index=False
    )

    for sheet_name, table in architecture_tables.items():

        table.to_excel(
            writer,
            sheet_name=sheet_name,
            index=False
        )

    summary.to_excel(
        writer,
        sheet_name="Architecture Summary"
    )

    for sheet_name in ["Power Outputs", "Fitting by Power",
                       *architecture_tables]:

        ws = writer.sheets[sheet_name]

        # Freeze header
        ws.freeze_panes = "A2"

        # Enable filters
        ws.auto_filter.ref = ws.dimensions

        # Format header
        from openpyxl.styles import Font, PatternFill, Alignment
        from openpyxl.utils import get_column_letter

        for cell in ws[1]:

            cell.font = Font(
                bold=True,
                color="FFFFFF"
            )

            cell.fill = PatternFill(
                "solid",
                fgColor="1F4E78"
            )

            cell.alignment = Alignment(
                horizontal="center"
            )

        # Set column widths
        for column in ws.columns:

            column_letter = get_column_letter(column[0].column)

            ws.column_dimensions[column_letter].width = 23

        # Format numeric values
        headers = {
            cell.value: cell.column
            for cell in ws[1]
        }

        for row in range(2, ws.max_row + 1):

            ws.cell(
                row,
                headers["Rotor diameter (m)"]
            ).number_format = "0.00"

            ws.cell(
                row,
                headers["Kite area (m2)"]
            ).number_format = "0.00"

            ws.cell(
                row,
                headers["Power output (kW)"]
            ).number_format = "0.00"

            ws.cell(
                row,
                headers["Wind speed (m/s)"]
            ).number_format = "0.00"

            ws.cell(
                row,
                headers["Air density (kg/m3)"]
            ).number_format = "0.000"

            for column_name in ["Harvester volume (m3)",
                                "Total volume incl. battery (m3)"]:

                ws.cell(
                    row,
                    headers[column_name]
                ).number_format = "0.00"

        # Color coding
        from openpyxl.formatting.rule import ColorScaleRule, CellIsRule

        last_row = ws.max_row

        volume_scale = ("Total volume incl. battery (m3)", "63BE7B", "F8696B")
        power_scale = ("Power output (kW)", "F8696B", "63BE7B")

        if sheet_name == "Power Outputs":
            color_scales = [volume_scale]
        elif sheet_name == "Fitting by Power":
            color_scales = [power_scale]
        else:
            color_scales = [volume_scale, power_scale]

        for sort_column, low_color, high_color in color_scales:

            sort_letter = get_column_letter(headers[sort_column])

            ws.conditional_formatting.add(
                f"{sort_letter}2:{sort_letter}{last_row}",
                ColorScaleRule(
                    start_type="min", start_color=low_color,
                    mid_type="percentile", mid_value=50, mid_color="FFEB84",
                    end_type="max", end_color=high_color
                )
            )

        fit_letter = get_column_letter(headers["Fits container"])
        fit_range = f"{fit_letter}2:{fit_letter}{last_row}"

        ws.conditional_formatting.add(
            fit_range,
            CellIsRule(operator="equal", formula=["TRUE"],
                       fill=PatternFill("solid", fgColor="C6EFCE"))
        )

        ws.conditional_formatting.add(
            fit_range,
            CellIsRule(operator="equal", formula=["FALSE"],
                       fill=PatternFill("solid", fgColor="FFC7CE"))
        )

print("\n==========================================")
print("EXCEL EXPORT COMPLETED")
print("==========================================")

print("File:", filename)
print("Total designs:", len(power_table))
print("Designs meeting 30 kW:",
      power_table["Meets 30 kW"].sum())
print("Designs fitting container:", len(fitting_table))
