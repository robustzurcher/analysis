#!/usr/bin/env python
"""This script provides all capabilities for the worker processes."""
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

import pickle as pkl
import numpy as np
import json

from scipy.stats import chi2
from mpi4py import MPI

from auxiliary import select_cost_func
from ruspy.model_code.cost_functions import calc_obs_costs

from worst_case_policies import calc_fixp_worst


def wrapper_func(
    p_ml,
    sample_size,
    scale,
    costs,
    disc_fac,
    num_states,
    cost_func_name,
    threshold,
    omega,
):
    rho = chi2.ppf(omega, len(p_ml) - 1) / (2 * (sample_size / scale))
    result = calc_fixp_worst(num_states, p_ml, costs, disc_fac, rho, threshold)
    fname = "results_{}_{}/intermediate_{}_{}.pkl".format(
        sample_size, cost_func_name, f"{omega:.2f}", cost_func_name
    )
    pkl.dump(result, open(fname, "wb"))

    return result


spec = json.load(open("specification.json", "rb"))
p_rust = np.loadtxt(spec["trans_probs"])
params = np.array(spec["params"])

cost_func = select_cost_func(spec["cost_func"])

comm = MPI.Comm.Get_parent()

# We want to let the master know we are ready to go
costs = calc_obs_costs(spec["num_states"], cost_func, params, spec["cost_scale"])
base_args = (
    p_rust,
    spec["sample_size"],
    spec["scale"],
    costs,
    spec["disc_fac"],
    spec["num_states"],
    spec["cost_func"],
    spec["threshold"],
)

while True:

    comm.Send([np.zeros(1, dtype="float"), MPI.DOUBLE], dest=0)

    cmd = np.array(0, dtype="int64")
    comm.Recv([cmd, MPI.INT], source=0)

    if cmd == 0:
        comm.Disconnect()
        break

    if cmd == 1:
        omega = np.zeros(1, dtype="float")
        comm.Recv([omega, MPI.DOUBLE], source=0)
        wrapper_func(*base_args, omega[0])
