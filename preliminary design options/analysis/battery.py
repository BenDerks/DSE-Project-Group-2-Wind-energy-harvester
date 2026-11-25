from design_options.battery_options import (
    E_load_kWh,
    STORAGE_TECHNOLOGIES,
    CONTAINER_SYSTEMS,
)


def nominal_capacity(E_load, DoD, eta_discharge, SOH_EOL):

    return E_load / (DoD * eta_discharge * SOH_EOL)


def battery_options(E_load=E_load_kWh):

    options = []

    for name, tech in STORAGE_TECHNOLOGIES.items():

        for case, i in [("best", 1), ("worst", 0)]:

            E_nominal = nominal_capacity(
                E_load,
                tech["DoD"][i],
                tech["eta_discharge"][i],
                tech["SOH_EOL"][i],
            )

            options.append({
                "label": f"{name} ({case})",
                "nominal_kWh": E_nominal,
                "volume": E_nominal / tech["volumetric_density_kWh_m3"][i],
                "mass_t": E_nominal / tech["specific_energy_Wh_kg"][i],
            })

    for name, system in CONTAINER_SYSTEMS.items():

        scale = E_load / 1000

        options.append({
            "label": name,
            "nominal_kWh": E_load,
            "volume": system["volume_m3"] * scale,
            "mass_t": system["mass_t"] * scale,
        })

    return options


def get_battery(label):

    for option in battery_options():
        if option["label"] == label:
            return option

    raise ValueError(f"Unknown battery option: {label}")


def battery_autonomy(E_usable_kWh, P_output):

    return E_usable_kWh / (P_output / 1000)
