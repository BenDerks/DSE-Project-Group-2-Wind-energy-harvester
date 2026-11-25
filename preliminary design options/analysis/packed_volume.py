import numpy as np

from analysis.transport import Fits_through_door


Electrical_volume = 1.0
Drivetrain_factor = 0.0012

Tower_d0 = 0.10                    # m
Tower_d_slope = 0.01

Hawt_chord_ratio = 0.10
Hawt_thickness_ratio = 0.50

Vawt_thickness_ratio = 0.20
Vawt_struts_per_blade = 2

Wing_thickness_ratio = 0.15
Tail_fuselage_factor = 1.25

Soft_kite_packing = 0.01

Ground_station_volume = 8.0        # m^3

Max_sections = 50


def Sections_needed(Length, Width, Height):

    for Sections in range(1, Max_sections + 1):

        if Fits_through_door((Length / Sections, Width, Height)):
            return Sections

    return None


def Tower_volume(Height):

    Diameter = Tower_d0 + Tower_d_slope * Height

    return Height * Diameter**2, Sections_needed(Height, Diameter, Diameter)


def Calculate_packed_volume(Design):

    Architecture = Design["architecture"]

    Breakdown = {}
    Sections = {}

    if Architecture == "HAWT":

        Radius = Design["rotor_diameter"] / 2
        Chord = Hawt_chord_ratio * Radius
        Thickness = Hawt_thickness_ratio * Chord
        Swept_area = np.pi * Radius**2

        Breakdown["blades"] = Design["blade_count"] * Radius * Chord * Thickness
        Sections["blade"] = Sections_needed(Radius, Chord, Thickness)

        Breakdown["tower"], Sections["tower"] = Tower_volume(Design["hub_height"])
        Breakdown["drivetrain"] = Drivetrain_factor * Swept_area**1.5

    elif Architecture == "VAWT":

        Radius = Design["rotor_diameter"] / 2
        Height = Design["rotor_height"]
        Blades = Design["blade_count"]
        Chord = Design["solidity"] * Radius / Blades
        Thickness = Vawt_thickness_ratio * Chord
        Swept_area = Design["rotor_diameter"] * Height

        Breakdown["blades"] = Blades * Height * Chord * Thickness
        Sections["blade"] = Sections_needed(Height, Chord, Thickness)

        Breakdown["struts"] = (
            Vawt_struts_per_blade * Blades * Radius * Chord * Thickness
        )

        Breakdown["tower"], Sections["tower"] = Tower_volume(Height)
        Breakdown["drivetrain"] = Drivetrain_factor * Swept_area**1.5

    elif Architecture == "Rigid Kite":

        Area = Design["wing_area"]
        Span = np.sqrt(Design["aspect_ratio"] * Area)
        Chord = Area / Span
        Thickness = Wing_thickness_ratio * Chord

        Breakdown["wing"] = Tail_fuselage_factor * Span * Chord * Thickness
        Sections["wing"] = Sections_needed(Span, Chord, Thickness)

        Breakdown["ground_station"] = Ground_station_volume

    elif Architecture == "Soft Kite":

        Breakdown["kite"] = Soft_kite_packing * Design["kite_area"]
        Breakdown["ground_station"] = Ground_station_volume

    Breakdown["electrical"] = Electrical_volume

    return {
        "harvester_volume": sum(Breakdown.values()),
        "volume_breakdown": Breakdown,
        "sections": Sections,
        "sections_fit_door": None not in Sections.values(),
    }
