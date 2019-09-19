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

from auxiliary import get_file

if __name__ == "__main__":

    spec = json.load(open("specification.json", "rb"))

    grid_omega = get_file(
        "../../pre_processed_data/fixp_results_1000_10_10_{}.pkl".format(
            spec["sample_size"]
        )
    ).keys()

    os.makedirs("val_results", exist_ok=True)

    status = MPI.Status()

    file_ = os.path.dirname(os.path.realpath(__file__)) + "/worker.py"
    comm = MPI.COMM_SELF.Spawn(
        sys.executable, args=[file_], maxprocs=spec["num_workers"]
    )

    # We now create a list of tasks.
    grid_task = list()

    for omega in grid_omega:

        # Varying beliefs
        task = omega
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
        "validation_results_{}".format(spec["sample_size"]), "zip", "val_results"
    )
    shutil.rmtree("val_results")
