import multiprocessing as mp
import pickle as pkl
from ruspy.simulation.simulation import simulate
from functools import partial
import numpy as np
import os
from aux_03 import get_file

# Beta is set almost to one, as the agents objective is to maximize average cost.
beta = 0.9999
# 200 buses should be enough to gurantee convergence.
num_buses = 200
# Set the number of simulated periods to 80000. The first plot shows the
# convergence at this point.
num_periods = 100
params = np.array([10, 10])


init_dict = {
        'beta': beta,
        'periods': num_periods,
        'seed': 123,
        'buses': num_buses,
        'params': params
    }

NUM_WORKERS = 2


def wrapper_func_variying_ev(init, omega_ev_mat):
    pkl.dump(simulate(init, omega_ev_mat[1], omega_ev_mat[2]), open(omega_ev_mat[0], "wb"))


if __name__ == "__main__":
    os.makedirs("dfs", exist_ok=True)
    dict_polcies = get_file("../pre_processed_data/results_1000_10_10.pkl")
    ev_wrapper_func = partial(wrapper_func_variying_ev, init_dict)

    # Each of the following blocks should be run seperatly as they produce different
    # data. The simulation doesn't take that long, so we should discuss now maybe the
    # final setting of each plot!

    # ML trans_mat and varying beliefs
    omega_evs_mat_0 = []
    for omega in dict_polcies.keys():
        filename = "dfs/df_ev_{}_mat_0.pkl".format(omega)
        omega_evs_mat_0 += [[filename, dict_polcies[omega][0], dict_polcies[0.0][1]]]

    mp.Pool(NUM_WORKERS).map(ev_wrapper_func, omega_evs_mat_0)

    # NEW BLOCK

    # Nominal strategy on varying omega
    omega_ev_0_mats = []
    for omega in dict_polcies.keys():
        filename = "dfs/df_ev_0_mat_{}.pkl".format(omega)
        omega_ev_0_mats += [[filename, dict_polcies[0.0][0], dict_polcies[omega][1]]]

    mp.Pool(NUM_WORKERS).map(ev_wrapper_func, omega_ev_0_mats)

    # NEW BLOCK

    # Robust strategy with 0.5 on varying omega
    omega_ev_054_mats = []
    for omega in dict_polcies.keys():
        filename = "dfs/df_ev_054_mat_{}.pkl".format(omega)
        omega_ev_054_mats += [[filename, dict_polcies[0.54][0], dict_polcies[omega][1]]]

    mp.Pool(NUM_WORKERS).map(ev_wrapper_func, omega_ev_054_mats)

    # NEW BLOCK

    # Robust strategy with 0.95 on varying omega
    omega_ev_099_mats = []
    for omega in dict_polcies.keys():
        filename = "dfs/df_ev_095_mat_{}.pkl".format(omega)
        omega_ev_099_mats += [[filename, dict_polcies[0.99][0], dict_polcies[omega][1]]]

    mp.Pool(NUM_WORKERS).map(ev_wrapper_func, omega_ev_099_mats)
