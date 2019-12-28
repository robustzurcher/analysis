#!/usr/bin/env python
"""This script provides all capabilities for the worker processes."""
import json
import os
import pickle as pkl

from auxiliary import select_cost_func
from ruspy.model_code.cost_functions import calc_obs_costs

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
from auxiliary import discount_utility
from auxiliary import get_file

comm = MPI.Comm.Get_parent()

spec = json.load(open("specification.json", "rb"))
dict_polcies = get_file(spec["policy_dict"])
params = np.array(spec["params"])

cost_func = select_cost_func(spec["cost_func"])
costs = calc_obs_costs(spec["num_states"], cost_func, params, spec["cost_scale"])

while True:

    comm.Send([np.zeros(1, dtype="float"), MPI.DOUBLE], dest=0)

    cmd = np.array(0, dtype="int64")
    comm.Recv([cmd, MPI.INT], source=0)

    if cmd == 0:
        comm.Disconnect()
        break

    if cmd == 1:
        fixp_key, trans_key = comm.recv(source=0)
        fname = "sim_results/result_ev_{}_mat_{}_{}.pkl".format(
            f"{fixp_key:.2f}", f"{trans_key:.2f}", spec["cost_func"]
        )
        fixp = dict_polcies[fixp_key][0]
        trans = dict_polcies[trans_key][1]

        df = simulate(spec, fixp, costs, trans)
        repl_state = df[df["decision"] == 1]["state"].mean()
        performance = discount_utility(
            df, spec["buses"], spec["periods"], 1000, spec["beta"]
        )
        pkl.dump((repl_state, performance), open(fname, "wb"))
