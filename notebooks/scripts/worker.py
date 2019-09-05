#!/usr/bin/env python
"""This script provides all capabilities for the slave processes."""
import os

# In this script we only have explicit use of MPI as our level of parallelism. This needs to be
# done right at the beginning of the script.
update = {'NUMBA_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
          'NUMEXPR_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
os.environ.update(update)

import numpy as np
import json

p_rust = np.loadtxt("resources/rust_trans_probs.txt")
params_rust = np.loadtxt("resources/rust_cost_params.txt")
spec = json.load(open('specification.json', 'rb'))

from ruspy.estimation.estimation_cost_parameters import cost_func
from ruspy.estimation.estimation_cost_parameters import lin_cost
from mpi4py import MPI
import pickle as pkl

from scipy.stats import chi2

from worst_case_probs import calc_fixp_worst




def wrapper_func(p_ml, costs, beta, num_states, threshold, omega):

    rho = chi2.ppf(omega, len(p_ml) - 1) / (2 * (4292 / 78))
    result = 99#calc_fixp_worst(num_states, p_ml, costs, beta, rho, threshold)
    fname = "results/intermediate_{}.pkl".format('{:3.5f}'.format(omega))
    pkl.dump(result, open(fname, "wb"))

    return result


comm = MPI.Comm.Get_parent()
num_slaves, rank = comm.Get_size(), comm.Get_rank()

# We want to let the master know we are ready to go.
comm.Send([np.array(1), MPI.DOUBLE], dest=0, tag=rank)

omega = np.zeros(1, dtype='float')
comm.Recv([omega, MPI.DOUBLE], source=0)

print(num_slaves, rank, omega)

costs_rust = cost_func(spec['num_states'], lin_cost, params_rust)

args = (p_rust, costs_rust, spec['beta'], spec['num_states'], spec['threshold'], omega[0])
wrapper_func(*args)
comm.Send([np.array(1), MPI.DOUBLE], dest=0, tag=rank)

cmd = np.array(0, dtype='int64')
comm.Recv([cmd, MPI.INT], source=0)

print(cmd)
if cmd == -1:
    comm.Disconnect()
else:
    raise AssertionError