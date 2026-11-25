import math


Solar_capacity_factor = 0.15
Panel_peak_power = 400
Panel_packed_volume = 0.07


def solar_sizing(P_load, Solar_fraction):

    P_solar_average = Solar_fraction * P_load
    P_solar_peak = P_solar_average / Solar_capacity_factor

    Panels = math.ceil(P_solar_peak / Panel_peak_power)

    return {
        "solar_average_W": P_solar_average,
        "solar_peak_W": Panels * Panel_peak_power,
        "panels": Panels,
        "solar_volume": Panels * Panel_packed_volume,
    }
