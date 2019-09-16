#!/usr/bin/env python
""" This module contains calculates the worst case policies for a range of the
confidence parameter"""
import os

# In this script we only have explicit use of MPI as our level of parallelism. This needs to be
# done right at the beginning of the script.
update = {'NUMBA_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
          'NUMEXPR_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
os.environ.update(update)

import pickle as pkl
import shutil
import json
import glob
import sys

from mpi4py import MPI
import numpy as np


if __name__ == "__main__":

    spec = json.load(open('specification.json', 'rb'))

    if os.path.exists("results"):
        shutil.rmtree('results')
    os.mkdir('results')

    status = MPI.Status()

    file_ = os.path.dirname(os.path.realpath(__file__)) + '/worker.py'
    comm = MPI.COMM_SELF.Spawn(sys.executable, args=[file_], maxprocs=spec['num_workers'])

    # We wait for everybody to be ready and then clean up the criterion function.
    check_in = np.zeros(1, dtype='float64')

    cmd = dict()
    cmd['terminate'] = np.array(0, dtype='int64')
    cmd['execute'] = np.array(1, dtype='int64')

    grid_omega = np.linspace(0.00, 0.99, num=spec['num_points'])

    for omega in grid_omega:

        comm.Recv([check_in, MPI.DOUBLE], status=status)

        comm.Send([cmd['execute'], MPI.INT], dest=status.Get_source())
        comm.Send([omega, MPI.INT], dest=status.Get_source())

    for rank in range(spec['num_workers']):
        comm.Send([cmd['terminate'], MPI.INT], dest=rank)

    comm.Disconnect()

    # Now we aggregate all the intermediate results.
    rslt = dict()
    for i, fname in enumerate(sorted(glob.glob('results/intermediate_*.pkl'))):
        rslt[grid_omega[i]] = pkl.load(open(fname, 'rb'))

    pkl.dump(rslt, open("results_1000.pkl", "wb"))
    shutil.rmtree('results')
