from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


from models.turbine import turbine_power
from inputs.system_parameters import (
    Cp_HAWT,
    Eta_turbine
)


# ==========================================
# INPUT
# ==========================================

Project_path = Path(__file__).resolve().parent

Data_path = Project_path / "all_designs.csv"
Plot_path = Project_path / "plots"

Plot_path.mkdir(exist_ok=True)

Data = pd.read_csv(Data_path)
HAWT = Data[Data["architecture"] == "HAWT"]

# ==========================================
# PLOT FUNCTIONS
# ==========================================

def plot_power_vs_tsr(Data, Architecture):

    Designs = Data[Data["architecture"] == Architecture]

    plt.figure(figsize=(8, 5))

    plt.scatter(
        Designs["tip_speed_ratio"],
        Designs["power_kW"],
        alpha=0.5
    )

    plt.title(f"{Architecture}: Power vs TSR")
    plt.xlabel("Tip Speed Ratio [-]")
    plt.ylabel("Electrical Power [kW]")

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        Plot_path / f"{Architecture.lower()}_power_vs_tsr.png",
        dpi=200
    )

    plt.show()
    plt.close()


def plot_power_vs_rotor_diameter(Data, Architecture):

    Designs = Data[Data["architecture"] == Architecture]

    plt.figure(figsize=(8, 5))

    plt.scatter(
        Designs["rotor_diameter"],
        Designs["power_kW"],
        alpha=0.5
    )

    plt.title(f"{Architecture}: Power vs Rotor Diameter")
    plt.xlabel("Rotor Diameter [m]")
    plt.ylabel("Electrical Power [kW]")

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        Plot_path / f"{Architecture.lower()}_power_vs_diameter.png",
        dpi=200
    )

    plt.show()
    plt.close()


def plot_hawt_power_vs_diameter(Data):

    Designs = Data[Data["architecture"] == "HAWT"]

    plt.figure(figsize=(8, 5))

    for Height, Group in Designs.groupby("hub_height"):

        plt.scatter(
            Group["rotor_diameter"],
            Group["power_kW"],
            label=f"Hub height = {Height} m",
            alpha=0.7
        )

    plt.xlabel("Rotor Diameter [m]")
    plt.ylabel("Electrical Power [kW]")
    plt.title("HAWT: Power vs Rotor Diameter")

    plt.legend(title="Hub height")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        Plot_path / "hawt_power_vs_diameter.png",
        dpi=200
    )

    plt.show()
    plt.close()





# ==========================================
# GENERATE PLOTS
# ==========================================

for Architecture in ["HAWT", "VAWT"]:

    plot_power_vs_tsr(Data, Architecture)

    plot_power_vs_rotor_diameter(Data, Architecture)
    
plot_hawt_power_vs_diameter(Data)


print("All plots generated successfully.")


