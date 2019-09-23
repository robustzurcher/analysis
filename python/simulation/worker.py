#!/usr/bin/env python
"""This script provides all capabilities for the worker processes."""
import pickle as pkl
import json
import os

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

from ruspy.simulation.simulation import simulate
from ruspy.simulation.value_zero import discount_utility
from auxiliary import get_file

comm = MPI.Comm.Get_parent()

dict_polcies = get_file(
    "../solution/fixp_results_5000_50_400_4292.pkl"
)
spec = json.load(open("specification.json", "rb"))

while True:

    comm.Send([np.zeros(1, dtype="float"), MPI.DOUBLE], dest=0)

    cmd = np.array(0, dtype="int64")
    comm.Recv([cmd, MPI.INT], source=0)

    if cmd == 0:
        comm.Disconnect()
        break

    if cmd == 1:
        fixp_key, trans_key = comm.recv(source=0)
        fname = "sim_results/result_ev_{}_mat_{}.pkl".format(
            "{:.2f}".format(fixp_key), "{:.2f}".format(trans_key)
        )
        fixp = dict_polcies[fixp_key][0]
        trans = dict_polcies[trans_key][1]

        df = simulate(spec, fixp, trans)
        repl_state = df[df["decision"] == 1]["state"].mean()
        performance = discount_utility(df, 1000, spec["beta"])
        pkl.dump((repl_state, performance), open(fname, "wb"))
