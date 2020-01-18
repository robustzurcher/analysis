import json
import os
import subprocess
import sys
from zipfile import ZipFile

PARAMS_LIN = [50, 41600]
PARAMS_SQRT = [36.4, 144000]
PARAMS_QUAD = [63, -100, 3500]
SCALE = 1e-5

parametrizations = [
    ("quad", SCALE, PARAMS_QUAD),
    ("linear", SCALE, PARAMS_LIN),
    ("sqrt", SCALE, PARAMS_SQRT),
]
results = ZipFile("result.zip", "w")
for cost_func_name, scale, params in parametrizations:
    general_dict = json.load(open("general_specification.json", "rb"))
    general_dict["cost_func"] = cost_func_name
    general_dict["cost_scale"] = scale
    general_dict["params"] = params
    general_dict["policy_dict"] = "../solution/fixp_results_{}_{}.pkl".format(
        general_dict["sample_size"], cost_func_name
    )
    for directory in ["solution", "simulation", "validation"]:
        os.chdir(directory)
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

    results.write(f"solution/fixp_results_4292_{cost_func_name}.zip")
    results.write(f"simulation/simulation_results_{cost_func_name}.zip")
    results.write(f"validation/validation_results_{cost_func_name}.zip")

results.close()
# # Varying data probability shift
# general_dict["max_omega"] = 0.95
# general_dict["num_points"] = 2
# general_dict["num_workers"] = 2
# general_dict["sample_size"] = 2223
# os.chdir("solution")
# spec_dict = json.load(open("specification.json", "rb"))
# for key in general_dict.keys():
#     if key in spec_dict.keys():
#         spec_dict[key] = general_dict[key]
# json.dump(
#     spec_dict, open("specification.json", "w"), indent=2, separators=(",", ": ")
# )
# cmd = f"mpiexec -n 1 {sys.executable} run.py"
# subprocess.run(cmd, shell=True)
# os.chdir("..")
