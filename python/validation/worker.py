#!/usr/bin/env python
"""This script provides all capabilities for the worker processes."""
import pickle as pkl
import json
import os
from out_of_sample import create_asym_trans_mat

# In this script we only have explicit use of MPI as our level of parallelism. This needs to be
# done right at the beginning of the script.
update = {'NUMBA_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
          'NUMEXPR_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
os.environ.update(update)

from mpi4py import MPI
import numpy as np

from ruspy.simulation.simulation import simulate
from ruspy.simulation.value_zero import discount_utility
from auxiliary import get_file

comm = MPI.Comm.Get_parent()

spec = json.load(open("specification.json", "rb"))
dict_polcies = get_file("../../pre_processed_data/fixp_results_1000_10_10_{}.pkl".format(spec["sample_size"]))
p_1000 = np.loadtxt("../../pre_processed_data/parameters/p_1000_4.txt")

while True:

    comm.Send([np.zeros(1, dtype='float'), MPI.DOUBLE], dest=0)

    cmd = np.array(0, dtype='int64')
    comm.Recv([cmd, MPI.INT], source=0)

    if cmd == 0:
        comm.Disconnect()
        break

    if cmd == 1:
        fixp_key = comm.recv(source=0)

        fname = "val_results/result_ev_{}_size_{}.pkl".format("{:.2f}".format(fixp_key),
                                                              spec["sample_size"])
        fixp = dict_polcies[fixp_key][0]
        trans = create_asym_trans_mat(fixp.shape[0], spec["sample_size"], p_1000,
                                      seed=spec["seed"])

        df = simulate(spec, fixp, trans)
        repl_state = df[df["decision"] == 1]["state"].mean()
        performance = discount_utility(df, 1000, spec["beta"])
        pkl.dump((repl_state, performance), open(fname, "wb"))
