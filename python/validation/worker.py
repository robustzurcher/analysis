#!/usr/bin/env python
"""This script provides all capabilities for the worker processes."""
import pickle as pkl
import json
import os
from out_of_sample import create_asym_trans_mat

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

spec = json.load(open("specification.json", "rb"))
p_1000 = np.loadtxt("../../pre_processed_data/parameters/p_1000_4292.txt")

while True:

    comm.Send([np.zeros(1, dtype="float"), MPI.DOUBLE], dest=0)

    cmd = np.array(0, dtype="int64")
    comm.Recv([cmd, MPI.INT], source=0)

    if cmd == 0:
        comm.Disconnect()
        break

    if cmd == 1:
        fixp_key, sample_size, run = comm.recv(source=0)

        fname = "val_results/result_ev_{}_size_{}_{}.pkl".format(
            "{:.2f}".format(fixp_key), sample_size, run
        )
        dict_polcies = get_file(
            f"../../pre_processed_data/fixp_results_1000_10_10_{sample_size}.pkl")
        fixp_rob = dict_polcies[fixp_key][0]
        fixp_ml = dict_polcies[0.0][0]
        
        np.random.seed()
        obs_per_state = sample_size / 388
        trans = create_asym_trans_mat(fixp_rob.shape[0], obs_per_state, p_1000)

        df_rob = simulate(spec, fixp_rob, trans)
        performance_rob = discount_utility(df_rob, 1000, spec["beta"])[-1]

        df_ml = simulate(spec, fixp_ml, trans)
        performance_ml = discount_utility(df_ml, 1000, spec["beta"])[-1]

        pkl.dump((performance_ml, performance_rob), open(fname, "wb"))
