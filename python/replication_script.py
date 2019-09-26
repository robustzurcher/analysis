import subprocess
import os
import json


general_dict = json.load(open("general_specification.json", "rb"))
general_dict["policy_dict"] = "../solution/fixp_results_5000_{}_{}_{}.pkl".format(
    general_dict["params"][0], general_dict["params"][1], general_dict["sample_size"]
)

for dir in ["solution", "simulation", "validation"]:
    os.chdir(dir)
    spec_dict = json.load(open("specification.json", "rb"))
    for key in general_dict.keys():
        if key in spec_dict.keys():
            spec_dict[key] = general_dict[key]
    json.dump(
        spec_dict, open("specification.json", "w"), indent=2, separators=(",", ": ")
    )
    subprocess.run("mpiexec -n 1 python run.py", shell=True)
    os.chdir("..")
