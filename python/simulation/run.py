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

if __name__ == "__main__":

    spec = json.load(open("specification.json", "rb"))
    grid_omega = get_file(spec["policy_dict"]).keys()

    result_folder = "sim_results_{}".format(spec["cost_func"])
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

    for omega in grid_omega:

        # ML transition matrix with varying beliefs
        task = omega, 0.0
        grid_task.append(task)

        for key in spec["sim_keys"]:
            task = key, omega
            grid_task.append(task)

        # # Optimal strategy with varying omega
        # task = omega, omega
        # grid_task.append(task)

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

    shutil.make_archive(
        result_folder, "zip", "sim_results_{}".format(spec["cost_func"]),
    )
    shutil.rmtree(result_folder)
