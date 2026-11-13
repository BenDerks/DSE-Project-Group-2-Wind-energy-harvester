

def get_design_wind_height(design):

    architecture = design["architecture"]

    # Kite systems
    if architecture in ["Rigid Kite", "Soft Kite"]:
        height = design["altitude"]

    # HAWT: use hub height
    elif architecture == "HAWT":
        height = design["hub_height"]

    # VAWT: use rotor centre height
    elif architecture == "VAWT":
        height = design["rotor_height"] / 2 + 10 #10 meter elevated

    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    return height
