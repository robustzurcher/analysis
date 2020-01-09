#!/usr/bin/env python
"""This script provides all capabilities for the worker processes."""
import json
import os
import pickle as pkl

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
from auxiliary import discount_utility
from auxiliary import get_file, select_cost_func
from ruspy.model_code.cost_functions import calc_obs_costs

comm = MPI.Comm.Get_parent()

spec = json.load(open("specification.json", "rb"))
raw_trans = np.loadtxt(spec["raw_trans"])
raw_hesse_inv = np.loadtxt(spec["raw_cov"])
SCALE = 1e-5

while True:

    comm.Send([np.zeros(1, dtype="float"), MPI.DOUBLE], dest=0)

    cmd = np.array(0, dtype="int64")
    comm.Recv([cmd, MPI.INT], source=0)

    if cmd == 0:
        comm.Disconnect()
        break

    if cmd == 1:
        fixp_key, run, cost_func_name, params = comm.recv(source=0)

        fname = "val_results/result_ev_{}_run_{}_{}.pkl".format(
            f"{fixp_key:.2f}", run, cost_func_name
        )
        policy_dict = "../solution/fixp_results_{}_{}.pkl".format(
            spec["sample_size"], cost_func_name
        )
        dict_polcies = get_file(policy_dict)
        fixp_rob = dict_polcies[fixp_key][0]
        fixp_ml = dict_polcies[0.0][0]

        np.random.seed()
        trans = create_asym_trans_mat(fixp_rob.shape[0], raw_trans, raw_hesse_inv)

        cost_func = select_cost_func(cost_func_name)
        cost_sim = calc_obs_costs(fixp_ml.shape[0], cost_func, params, SCALE)

        df_rob = simulate(spec, fixp_rob, cost_sim, trans)
        performance_rob = discount_utility(
            df_rob, spec["buses"], spec["periods"], spec["periods"], spec["disc_fac"]
        )[-1]
        del df_rob

        df_ml = simulate(spec, fixp_ml, cost_sim, trans)
        performance_ml = discount_utility(
            df_ml, spec["buses"], spec["periods"], spec["periods"], spec["disc_fac"]
        )[-1]
        del df_ml

        pkl.dump((performance_ml, performance_rob), open(fname, "wb"))
