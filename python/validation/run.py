import json
import os
import shutil
import sys

# In this script we only have explicit use of MPI as our level of parallelism. This needs to be
# done right at the beginning of the script.
update = {
    "NUMBA_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}
os.environ.update(update)

from mpi4py import MPI
import numpy as np
from auxiliary import get_file


PARAMS_LIN = [203, 27800]
PARAMS_SQRT = [140, 152266]
PARAMS_QUAD = [266, -1000, 960]

parametrizations = [
    ("quad", PARAMS_QUAD),
    ("linear", PARAMS_LIN),
    ("sqrt", PARAMS_SQRT),
]
if __name__ == "__main__":

    spec = json.load(open("specification.json", "rb"))

    result_folder = "val_results"
    if os.path.exists(result_folder):
        shutil.rmtree(result_folder)
    os.mkdir(result_folder)

    status = MPI.Status()

    file_ = os.path.dirname(os.path.realpath(__file__)) + "/worker.py"
    comm = MPI.COMM_SELF.Spawn(
        sys.executable, args=[file_], maxprocs=spec["num_workers"]
    )

    # We now create a list of tasks.
    grid_task = []

    for cost_func_name, params in parametrizations:
        # If not unziped, unzip all pickle files
        policy_dict = "../solution/fixp_results_{}_{}.pkl".format(
            spec["sample_size"], cost_func_name
        )
        get_file(policy_dict)
        for fixp_key in spec["strategies_validation"]:
            for run in range(spec["runs_strategies_validation"]):
                task = fixp_key, run, cost_func_name, params
                grid_task.append(task)

        for run in range(spec["density_runs"]):
            task = spec["density_strategy"], run, cost_func_name, params
            grid_task.append(task)

    # We wait for everybody to be ready and then clean up the criterion function.
    check_in = np.zeros(1, dtype="float64")

    cmd = {}
    cmd["terminate"] = np.array(0, dtype="int64")
    cmd["execute"] = np.array(1, dtype="int64")

    for task in grid_task:

        comm.Recv([check_in, MPI.DOUBLE], status=status)

        comm.Send([cmd["execute"], MPI.INT], dest=status.Get_source())
        comm.send(task, dest=status.Get_source())

    for rank in range(spec["num_workers"]):
        comm.Send([cmd["terminate"], MPI.INT], dest=rank)

    comm.Disconnect()
    # Now we aggregate all the results.

    shutil.make_archive("validation_results", "zip", result_folder)
    shutil.rmtree(result_folder)
