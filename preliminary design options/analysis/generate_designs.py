from itertools import product


def generate_designs(architecture, design_space):
    parameter_names = list(design_space.keys())
    parameter_values = list(design_space.values())

    combinations = product(*parameter_values)

    designs = []

    for combination in combinations:
        design = {"architecture": architecture}

        for name, value in zip(parameter_names, combination):
            design[name] = value

        designs.append(design)

    return designs