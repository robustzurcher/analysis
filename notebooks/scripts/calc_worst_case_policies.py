#!/usr/bin/env python
""" This module contains calculates the worst case policies for a range of the
confidence parameter"""
from functools import partial
import multiprocessing as mp
import pickle as pkl
import shutil
import glob
import os

from ruspy.estimation.estimation_cost_parameters import cost_func
from ruspy.estimation.estimation_cost_parameters import lin_cost
from scipy.stats import chi2
import numpy as np

from worst_case_probs import calc_fixp_worst


NUM_WORKERS = 2
NUM_POINTS = 3
NUM_STATES = 20
BETA = 0.9999

# For debugging purposes, just set to a very large number. Then the there will only
# be a single fix point iteration.
THRESHOLD = 1e-7


def wrapper_func(p_ml, costs, beta, num_states, threshold, omega):

    rho = chi2.ppf(omega, len(p_ml) - 1) / (2 * (4292 / 78))

    result = calc_fixp_worst(num_states, p_ml, costs, beta, rho, threshold)

    fname = "results/intermediate_{}.pkl".format('{:3.5f}'.format(omega))
    pkl.dump(result, open(fname, "wb"))

    return result


if __name__ == "__main__":

    grid_omega = np.linspace(0, 0.99, NUM_POINTS)

    if os.path.exists("results"):
        shutil.rmtree('results')
    os.mkdir('results')

    P_ML = np.loadtxt("resources/rust_trans_probs.txt")
    params = np.loadtxt("resources/rust_cost_params.txt")

    COSTS = cost_func(NUM_STATES, lin_cost, params)

    p_wrapper_func = partial(wrapper_func, P_ML, COSTS, BETA, NUM_STATES, THRESHOLD)
    final_result = mp.Pool(NUM_WORKERS).map(p_wrapper_func, grid_omega)

    pkl.dump(final_result, open("results/final_result.pkl", "wb"))
    [os.remove(file) for file in glob.glob("results/intermediate*")]
