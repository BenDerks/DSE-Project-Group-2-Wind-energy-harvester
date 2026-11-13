

def check_feasibility(design):
    
    
    architecture = design["architecture"]
    reasons = []
    
    
    #HAWT Constraints
    
    
    
    if architecture == "HAWT":
        
        rotor_radius = design["rotor_diameter"] /2
        hub_height = design["hub_height"]
        
        ground_clearance = hub_height - rotor_radius
        
        if ground_clearance < 1: #check clearification rule
            reasons.append("Insufficient ground clearance")
            
     # VAWT constraints

    elif architecture == "VAWT":
        rotor_height = design["rotor_height"]
        rotor_diameter = design["rotor_diameter"]
        
        if rotor_height <= 0 or rotor_diameter <=0:
            reasons.append("Invalid rotor dimensions for VAWT")
            
    elif architecture == "Rigid Kite":
        area = design["wing_area"]
        aspect_ratio = design["aspect_ratio"]

        if area <= 0 or aspect_ratio <= 0:
            reasons.append("Invalid wing geometry")

    # Soft Kite constraints
    elif architecture == "Soft Kite":

        area = design["kite_area"]
        aspect_ratio = design["aspect_ratio"]

        if area <= 0 or aspect_ratio <= 0:
            reasons.append("Invalid kite geometry")

    return len(reasons) == 0, reasons



def check_hawt_feasibility(design):
    pass


def check_vawt_feasibility(design):
    pass


def check_rigid_kite_feasibility(design):
    pass


def check_soft_kite_feasibility(design):
    pass


def check_system_feasibility(design):
    pass


