import numpy as np
from pathlib import Path



def load_airfoil_polar(filename="???"):
    #wait for name fail/zeger
    data_path= (
        Path(__file__).resolve().parents[1]
        )
    
    data = np.genfromtext(data_path, delimiter=',', names=True)
    
    polar = {
        "alpha": data["alpha_deg"],
        "Cl": data["Cl"],
        "Cd": data["Cd"],
        "Cm": data["Cm"]
    }
    
    return polar


def airfoil_coefficients(alpha_deg, polar):
    
    Cl = np.interp(alpha_deg, polar["alpha"], polar["Cl"])
    Cd = np.interp(alpha_deg, polar["alpha"], polar["Cd"])
    Cm = np.interp(alpha_deg, polar["alpha"], polar["Cm"])
    
    
    return Cl, Cd, Cm

    
    
    
    
    