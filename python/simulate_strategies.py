import multiprocessing as mp
import os
import pickle as pkl
from ruspy.simulation.simulation import simulate
from functools import partial
import numpy as np
from aux_03 import get_file


NUM_WORKERS = 2


def wrapper_func_variying_ev(init, omega_ev_mat):
    fname = "dfs/" + omega_ev_mat[0]
    pkl.dump(simulate(init, omega_ev_mat[1], omega_ev_mat[2]), open(fname, "wb"))


if __name__ == "__main__":
    os.mkdirs('df', exist_ok=True)
    dict_polcies = get_file("../pre_processed_data/results_1000.pkl")
    # ML trans_mat and varying beliefs
    omega_evs_mat_0 = []
    for omega in dict_polcies.keys():
        filename = "dfs/df_ev_{}_mat_0.pkl".format(omega)
        omega_evs_mat_0 += [[filename, dict_polcies[omega][0], dict_polcies[0.0][1]]]

    # Nominal strategy on varying omega
    omega_ev_0_mats = []
    for omega in dict_polcies.keys():
        filename = "dfs/df_ev_0_mat_{}.pkl".format(omega)
        omega_ev_0_mats += [[filename, dict_polcies[0.0][0], dict_polcies[omega][1]]]

    # Robust strategy with 0.5 on varying omega
    omega_ev_05_mats = []
    for omega in dict_polcies.keys():
        filename = "dfs/df_ev_05_mat_{}.pkl".format(omega)
        omega_ev_05_mats += [[filename, dict_polcies[0.5][0], dict_polcies[omega][1]]]

    # Robust strategy with 0.95 on varying omega
    omega_ev_095_mats = []
    for omega in dict_polcies.keys():
        filename = "dfs/df_ev_095_mat_{}.pkl".format(omega)
        omega_ev_095_mats += [[filename, dict_polcies[0.95][0], dict_polcies[omega][1]]]

    # Beta is set almost to one, as the agents objective is to maximize average cost.
    beta = 0.9999
    # 200 buses should be enough to gurantee convergence.
    num_buses = 200
    # Set the number of simulated periods to 80000. The first plot shows the
    # convergence at this point.
    num_periods = 70000
    params = np.array([10, 10])

    init_dict = {
        'beta': beta,
        'periods': num_periods,
        'seed': 123,
        'buses': num_buses,
        'params': params
    }
    ev_wrapper_func = partial(wrapper_func_variying_ev, init_dict)

    # ML trans_mat and varying beliefs
    mp.Pool(NUM_WORKERS).map(ev_wrapper_func, omega_evs_mat_0)

    # Nominal strategy on varying omega
    mp.Pool(NUM_WORKERS).map(ev_wrapper_func, omega_ev_0_mats)

    # Robust strategy with 0.5 on varying omega
    mp.Pool(NUM_WORKERS).map(ev_wrapper_func, omega_ev_05_mats)

    # Robust strategy with 0.95 on varying omega
    mp.Pool(NUM_WORKERS).map(ev_wrapper_func, omega_ev_095_mats)
