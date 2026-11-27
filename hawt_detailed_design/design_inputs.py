
from pathlib import Path
import sys
import pandas as pd

# ==========================================
# PROJECT PATHS
# ==========================================

PROJECT_PATH = Path(__file__).resolve().parents[1]
PRELIMINARY_PATH = PROJECT_PATH / "preliminary design options"

if not PRELIMINARY_PATH.exists():
    raise FileNotFoundError(
        f"Cannot find preliminary design folder: {PRELIMINARY_PATH}"
    )

sys.path.insert(0, str(PRELIMINARY_PATH))

# ==========================================
# IMPORT EXISTING TEAM FUNCTIONS
# ==========================================

from inputs.wind_conditions import (
    Wind_scalar,
    get_design_wind_height,
    isa_atmosphere,
)

from design_options.turbine_options import HAWT_SPACE
from analysis.generate_designs import generate_designs
from analysis.performance import calculate_design_power
from analysis.geometry import Calculate_geometry
from analysis.feasibility import check_feasibility
from analysis.packed_volume import Calculate_packed_volume
from analysis.transport import Usable_volume


# ==========================================
# INPUTS
# ==========================================

P_TARGET = 30_000          # W
WIND_SPEED_10M = 6.0        # m/s
REFERENCE_HEIGHT = 10      # m

OUTPUT_FILE = Path(__file__).resolve().parent / "HAWT_Design_Results.xlsx"


# ==========================================
# GENERATE AND EVALUATE HAWT DESIGNS
# ==========================================

hawt_results = []

designs = generate_designs("HAWT", HAWT_SPACE)

for i, design in enumerate(designs, start=1):

    # Wind conditions at design height
    height = get_design_wind_height(design)

    rho = isa_atmosphere(height)[0]

    wind_speed = Wind_scalar(height, WIND_SPEED_10M)

    # Calculate turbine power
    result = design.copy()

    result.update(
        calculate_design_power(design, rho, wind_speed)
    )

    # Geometry and transport calculations
    geometry = Calculate_geometry(design)

    status, _ = check_feasibility(geometry)

    packing = Calculate_packed_volume(result)

    # Check container feasibility
    fits_container = (
        status != "FAIL"
        and packing["sections_fit_door"]
        and packing["harvester_volume"] <= Usable_volume
    )

    # Store results
    hawt_results.append({

        "Design ID": f"HAWT-{i:03d}",

        "Rotor diameter (m)": result["rotor_diameter"],
        "Hub height (m)": design["hub_height"],

        "Wind speed at hub (m/s)": wind_speed,
        "Air density (kg/m3)": rho,

        "Power output (kW)": result["power_W"] / 1000,

        "Harvester volume (m3)": packing["harvester_volume"],

        "Geometry status": status,
        "Fits container": fits_container,

        "Meets 30 kW target": result["power_W"] >= P_TARGET,

    })


# ==========================================
# CREATE RESULTS TABLE
# ==========================================

results = pd.DataFrame(hawt_results)

if results.empty:
    raise ValueError("No HAWT designs were generated.")

results = results.sort_values(
    ["Fits container", "Power output (kW)"],
    ascending=[False, False]
)


# ==========================================
# SUMMARY
# ==========================================

print("\n========== HAWT DESIGN RESULTS ==========")

print(f"Number of designs: {len(results)}")

print(f"Target power: {P_TARGET / 1000:.1f} kW")

print(f"Reference wind speed: {WIND_SPEED_10M:.1f} m/s")

print(f"Reference wind height: {REFERENCE_HEIGHT} m")

print(f"Designs meeting power target: "
      f"{results['Meets 30 kW target'].sum()}")

print(f"Designs fitting container: "
      f"{results['Fits container'].sum()}")

print("\nBest feasible designs:")

print(
    results[results["Fits container"]]
    .head(10)
    .to_string(index=False)
)


# ==========================================
# EXPORT TO EXCEL
# ==========================================

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:

    results.to_excel(
        writer,
        sheet_name="HAWT Design Results",
        index=False
    )

    ws = writer.sheets["HAWT Design Results"]

    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions

    for column in ws.columns:

        column_letter = column[0].column_letter

        ws.column_dimensions[column_letter].width = 24

    for cell in ws[1]:

        from openpyxl.styles import Font, PatternFill, Alignment

        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill(
            "solid",
            fgColor="1F4E78"
        )

        cell.alignment = Alignment(
            horizontal="center",
            wrap_text=True
        )


print(f"\nExcel results saved to: {OUTPUT_FILE}")
print("\nHAWT evaluation completed successfully.")
