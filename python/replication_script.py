import json
import os
import subprocess
import sys

PARAMS_LIN = [400, 35]
PARAMS_SQRT = [460, 50]
PARAMS_CUBIC = [700, 43200, -780, 4.7]


parametrizations = [
    ("linear", 0.001, PARAMS_LIN),
    ("cubic", 1e-5, PARAMS_CUBIC),
    ("sqrt", 1e-2, PARAMS_SQRT),
]
for cost_func_name, scale, params in parametrizations:
    general_dict = json.load(open("general_specification.json", "rb"))
    general_dict["cost_func"] = cost_func_name
    general_dict["cost_scale"] = scale
    general_dict["params"] = params
    general_dict["policy_dict"] = "../solution/fixp_results_{}_{}.pkl".format(
        general_dict["sample_size"], cost_func_name
    )
    for dir in ["solution", "simulation", "validation"]:
        os.chdir(dir)
        print(dir, cost_func_name)
        spec_dict = json.load(open("specification.json", "rb"))
        for key in general_dict.keys():
            if key in spec_dict.keys():
                spec_dict[key] = general_dict[key]
        json.dump(
            spec_dict, open("specification.json", "w"), indent=2, separators=(",", ": ")
        )
        cmd = f"mpiexec -n 1 {sys.executable} run.py"
        subprocess.run(cmd, shell=True)
        os.chdir("..")

    # Varying data probability shift
    general_dict["max_omega"] = 0.95
    general_dict["num_points"] = 2
    general_dict["num_workers"] = 2
    general_dict["sample_size"] = 2223
    os.chdir("solution")
    spec_dict = json.load(open("specification.json", "rb"))
    for key in general_dict.keys():
        if key in spec_dict.keys():
            spec_dict[key] = general_dict[key]
    json.dump(
        spec_dict, open("specification.json", "w"), indent=2, separators=(",", ": ")
    )
    cmd = f"mpiexec -n 1 {sys.executable} run.py"
    subprocess.run(cmd, shell=True)
    os.chdir("..")
