
from itertools import permutations
import math


# 1. Standard 20 ft shipping container

Container_length = 5.900       # m
Container_width = 2.352        # m
Container_height = 2.395       # m

Door_width = 2.340              # m
Door_height = 2.292             # m

Container_volume = 33.2         # m^3
Max_payload = 28130             # kg

#
#2. Shared system transport inventory

#assume same electrical system for all 4 different harverster systems
Shared_modules = [
    {
        "name": "ESS",
        "dimensions": (0, 0, 0),  # Length, width, height (m)
        "mass": 0                 # kg
    },
    {
        "name": "Electrical equipment",
        "dimensions": (0, 0, 0),
        "mass": 0
    },
    {
        "name": "Cables and auxiliary equipment",
        "dimensions": (0, 0, 0),
        "mass": 0
    }
]


# 3. Check individual module dimensions


def Fits_inside_container(Dimensions):

    for Orientation in permutations(Dimensions):

        Length, Width, Height = Orientation

        if (
            Length <= Container_length
            and Width <= Container_width
            and Height <= Container_height
        ):
            return True

    return False


def Fits_through_door(Dimensions):

    for Orientation in permutations(Dimensions):

        Length, Width, Height = Orientation

        if (
            Length <= Container_length
            and Width <= Door_width
            and Height <= Door_height
        ):
            return True

    return False


# 4. Check complete transport feasibility

def Check_transport_feasibility(Modules):

    Reasons = []

    Total_volume = 0
    Total_mass = 0

    for Module in Modules:

        Dimensions = Module["dimensions"]
        Mass = Module["mass"]

        Length, Width, Height = Dimensions

        Volume = Length * Width * Height

        Total_volume += Volume
        Total_mass += Mass

        if not Fits_through_door(Dimensions):

            Reasons.append(
                f'{Module["name"]} does not fit through container door'
            )

        if not Fits_inside_container(Dimensions):

            Reasons.append(
                f'{Module["name"]} does not fit inside container'
            )

    if Total_volume > Container_volume:

        Reasons.append(
            "Total packed volume exceeds container capacity"
        )

    if Total_mass > Max_payload:

        Reasons.append(
            "Total transport mass exceeds container payload"
        )

    return len(Reasons) == 0, Reasons


#5. HAWT transport assessment

Max_module_length = 5.50  # m


def Check_hawt_transport(Design):

    Rotor_diameter = Design["rotor_diameter"]
    Hub_height = Design["hub_height"]

    Blade_length = Rotor_diameter / 2
    Tower_height = Hub_height

    # Calculate required number of blade sections
    Blade_sections = math.ceil(Blade_length / Max_module_length)
    Blade_section_length = Blade_length / Blade_sections

    # Calculate required number of tower sections
    Tower_sections = math.ceil(Tower_height / Max_module_length)
    Tower_section_length = Tower_height / Tower_sections

    Result = {
        "Blade_length": Blade_length,
        "Blade_sections_per_blade": Blade_sections,
        "Blade_section_length": Blade_section_length,
        "Total_blade_sections": 3 * Blade_sections,
        "Tower_height": Tower_height,
        "Tower_sections": Tower_sections,
        "Tower_section_length": Tower_section_length,
        "Hub_nacelle_dimensions": "TBD",
        "Cross_section_check": "TBD"
    }

    return Result

