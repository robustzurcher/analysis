#!/usr/bin/env python
""" This module contains calculates the worst case policies for a range of the
confidence parameter"""
import os

# In this script we only have explicit use of MPI as our level of parallelism. This needs to be
# done right at the beginning of the script.
update = {'NUMBA_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
          'NUMEXPR_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
os.environ.update(update)

import shutil
import json
import sys

from mpi4py import MPI
import numpy as np


NUM_WORKERS = 2

if __name__ == "__main__":

    spec = json.load(open('specification.json', 'rb'))

    if os.path.exists("results"):
        shutil.rmtree('results')
    os.mkdir('results')

    info = MPI.Info.Create()
    info.update({"wdir": os.getcwd()})

    file_ = os.path.dirname(os.path.realpath(__file__)) + '/worker.py'
    comm = MPI.COMM_SELF.Spawn(sys.executable, args=[file_], maxprocs=NUM_WORKERS, info=info)

    # We wait for everybody to be ready and then clean up the criterion function.
    check_in = np.zeros(1, dtype='float')

    grid_omega = np.linspace(0.00, 0.99, num=spec['num_points'])

    status = MPI.Status()

    for omega in grid_omega:


        comm.Recv([check_in, MPI.DOUBLE], status=status)
        comm.Send([np.array(omega), MPI.INT], dest=status.Get_source())

    for rank in range(NUM_WORKERS):

        comm.Send([np.array(-1), MPI.INT], dest=status.Get_source())
        print("received")

    #
    #comm.Disconnect()

    print("all checked in")
    # grid_omega = [0.01, 0.02]
    #

    #
    #
    #
    # costs_rust = cost_func(NUM_STATES, lin_cost, params_rust)
    #
    # p_wrapper_func = partial(wrapper_func, p_rust, costs_rust, BETA, NUM_STATES, THRESHOLD)
    # final_result = mp.Pool(NUM_WORKERS).map(p_wrapper_func, grid_omega)
    #
    # pkl.dump(final_result, open("results/final_result.pkl", "wb"))
    # [os.remove(file) for file in glob.glob("results/intermediate*")]
