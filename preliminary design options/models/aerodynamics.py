import numpy as np



def dynamic_pressure(rho, velocity):
    
    "calculate dynamic_pressure at any place"
    
    
    q = 1/2 * rho * velocity**2 
    
    return q


def Reynolds_number(rho, velocity, length, mu):
    
    "Reynold number calculation"
    
    Re = (rho * velocity * length) / mu
    
    return Re


def Lift_force(rho, velocity, area, Cl):
    
    q = dynamic_pressure(rho, velocity) 
    
    L = q * area * Cl * velocity**2
    
    return L


def Drag_force(rho, velocity, area, Cd):
    
    q = dynamic_pressure(rho, velocity)
    
    D = q * area * Cd * velocity**2
    
    
    

