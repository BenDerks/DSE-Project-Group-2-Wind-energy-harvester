
from inputs.feasibility_requirements import Min_ground_clearance

# HAWT
def check_hawt_feasibility(Design):

    Reasons = []

    Ground_clearance = Design["Ground_clearance"]

    if Ground_clearance < Min_ground_clearance:

        Reasons.append(
            "GEO-HAWT-01: Insufficient ground clearance"
        )

    return Reasons



# VAWT
def check_vawt_feasibility(Design):

    Reasons = []

    if Design["rotor_height"] <= 0:
        Reasons.append("GEO-VAWT-01: Rotor height must be positive")

    if Design["rotor_diameter"] <= 0:
        Reasons.append("GEO-VAWT-02: Rotor diameter must be positive")

    return Reasons



# RIGID KITE

Geometry_results = []


def check_rigid_kite_feasibility(Design):

    Reasons = []

    if Design["wing_area"] <= 0 or Design["aspect_ratio"] <= 0:

        Reasons.append(
            "GEO-RK-01: Invalid wing geometry"
        )

    return Reasons



# SOFT KITE

def check_soft_kite_feasibility(Design):

    Reasons = []

    if Design["kite_area"] <= 0 or Design["aspect_ratio"] <= 0:

        Reasons.append(
            "GEO-SK-01: Invalid kite geometry"
        )

    return Reasons

# GENERAL FEASIBILITY

def check_feasibility(Design):

    Architecture = Design["architecture"]

    if Architecture == "HAWT":
        Reasons = check_hawt_feasibility(Design)

    elif Architecture == "VAWT":
        Reasons = check_vawt_feasibility(Design)

    elif Architecture == "Rigid Kite":
        Reasons = check_rigid_kite_feasibility(Design)

    elif Architecture == "Soft Kite":
        Reasons = check_soft_kite_feasibility(Design)

    else:
        Reasons = ["Unknown architecture"]

    if Reasons:
        Status = "FAIL"

    elif Architecture == "HAWT":
        Status = "PASS"

    else:
        Status = "TBD"

    return Status, Reasons
