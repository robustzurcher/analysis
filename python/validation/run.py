import json
import sys
import os
import shutil

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

if __name__ == "__main__":

    spec = json.load(open("specification.json", "rb"))
    os.makedirs("val_results", exist_ok=True)

    status = MPI.Status()

    file_ = os.path.dirname(os.path.realpath(__file__)) + "/worker.py"
    comm = MPI.COMM_SELF.Spawn(
        sys.executable, args=[file_], maxprocs=spec["num_workers"]
    )

    # We now create a list of tasks.
    grid_task = list()

    for fixp_key in spec["strategies_validation"]:
        for run in range(spec["runs_strategies_validation"]):
            task = fixp_key, run
            grid_task.append(task)

    for run in range(spec["density_runs"]):
        task = spec["density_strategy"], run
        grid_task.append(task)


    # We wait for everybody to be ready and then clean up the criterion function.
    check_in = np.zeros(1, dtype="float64")

    cmd = dict()
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
        "validation_results", "zip", "val_results"
    )
    shutil.rmtree("val_results")
