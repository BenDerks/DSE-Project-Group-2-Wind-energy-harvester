import numpy as np


def Calculate_geometry(Design):

    Architecture = Design["architecture"]

    Result = Design.copy()
    
    # HAWT

    if Architecture == "HAWT":

        Rotor_diameter = Design["rotor_diameter"]
        Hub_height = Design["hub_height"]

        Rotor_radius = Rotor_diameter / 2
        Rotor_area = np.pi * Rotor_diameter**2 / 4

        Blade_length = Rotor_radius

        Ground_clearance = Hub_height - Rotor_radius
        Rotor_top_height = Hub_height + Rotor_radius

        Result.update({
            "Rotor_radius": Rotor_radius,
            "Rotor_area": Rotor_area,
            "Blade_length": Blade_length,
            "Ground_clearance": Ground_clearance,
            "Rotor_top_height": Rotor_top_height
        })


    # VAWT

    elif Architecture == "VAWT":

        Rotor_diameter = Design["rotor_diameter"]
        Rotor_height = Design["rotor_height"]

        Rotor_radius = Rotor_diameter / 2
        Rotor_area = Rotor_diameter * Rotor_height

        Rotor_aspect_ratio = Rotor_height / Rotor_diameter

        Result.update({
            "Rotor_radius": Rotor_radius,
            "Rotor_area": Rotor_area,
            "Rotor_aspect_ratio": Rotor_aspect_ratio
        })


    # RIGID KITE

    elif Architecture == "Rigid Kite":

        Wing_area = Design["wing_area"]
        Aspect_ratio = Design["aspect_ratio"]

        Wing_span = np.sqrt(Aspect_ratio * Wing_area)
        Mean_chord = Wing_area / Wing_span

        Result.update({
            "Wing_span": Wing_span,
            "Mean_chord": Mean_chord
        })


    # SOFT KITE
    elif Architecture == "Soft Kite":

        Kite_area = Design["kite_area"]
        Aspect_ratio = Design["aspect_ratio"]

        Wing_span = np.sqrt(Aspect_ratio * Kite_area)
        Mean_chord = Kite_area / Wing_span

        Result.update({
            "Wing_span": Wing_span,
            "Mean_chord": Mean_chord
        })


    return Result



