#!/usr/bin/env python
"""This script provides all capabilities for the worker processes."""
import os

# In this script we only have explicit use of MPI as our level of parallelism. This needs to be
# done right at the beginning of the script.
update = {'NUMBA_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
          'NUMEXPR_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
os.environ.update(update)

import pickle as pkl
import numpy as np
import json

from scipy.stats import chi2
from mpi4py import MPI

from ruspy.estimation.estimation_cost_parameters import cost_func
from ruspy.estimation.estimation_cost_parameters import lin_cost

from worst_case_policies import calc_fixp_worst


def wrapper_func(p_ml, obs_per_state, costs, beta, num_states, threshold, omega):
    if num_states > obs_per_state.shape[0]:
        raise AssertionError
    inv_omega_state = chi2.ppf(np.full(obs_per_state.shape[0], omega), len(p_ml) - 1)
    rho_state = np.divide(inv_omega_state, 2 * obs_per_state)
    result = calc_fixp_worst(num_states, p_ml, costs, beta, rho_state, threshold)
    fname = "results/intermediate_{}.pkl".format('{:3.5f}'.format(omega))
    pkl.dump(result, open(fname, "wb"))

    return result


p_rust = np.loadtxt("../../pre_processed_data/parameters/p_1000_4.txt")
params_rust = np.array([10, 10])
obs_state = np.loadtxt("../../pre_processed_data/parameters/obs_state_1000.txt")
spec = json.load(open('specification.json', 'rb'))

comm = MPI.Comm.Get_parent()

# We want to let the master know we are ready to go
costs_rust = cost_func(spec['num_states'], lin_cost, params_rust)
base_args = (p_rust, obs_state, costs_rust, spec['beta'], spec['num_states'],
             spec['threshold'])

while True:

    comm.Send([np.zeros(1, dtype='float'), MPI.DOUBLE], dest=0)

    cmd = np.array(0, dtype='int64')
    comm.Recv([cmd, MPI.INT], source=0)

    if cmd == 0:
        comm.Disconnect()
        break

    if cmd == 1:
        omega = np.zeros(1, dtype='float')
        comm.Recv([omega, MPI.DOUBLE], source=0)
        wrapper_func(*base_args, omega[0])
