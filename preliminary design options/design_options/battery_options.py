P_load_kW = 20
Autonomy_h = 50
E_load_kWh = P_load_kW * Autonomy_h


STORAGE_TECHNOLOGIES = {
    "Li-ion": {
        "volumetric_density_kWh_m3": (220, 250),
        "specific_energy_Wh_kg": (210, 325),
        "DoD": (0.80, 1.00),
        "eta_discharge": (0.95, 0.95),
        "SOH_EOL": (0.80, 0.80),
    },
    "Lead-acid": {
        "volumetric_density_kWh_m3": (50, 80),
        "specific_energy_Wh_kg": (30, 50),
        "DoD": (0.50, 0.60),
        "eta_discharge": (1.0, 1.0),
        "SOH_EOL": (1.0, 1.0),
    },
}


CONTAINER_SYSTEMS = {
    "Atlas Copco ZBC 1000": {"volume_m3": 33.2, "mass_t": 20.4},
    "Aspen Woods 3 MWh": {"volume_m3": 15.6, "mass_t": 10.4},
    "Sungrow PowerTitan": {"volume_m3": 8.6, "mass_t": 8.5},
    "Container average": {"volume_m3": 19.1, "mass_t": 13.1},
}
