import multiprocessing as mp
import glob
import pickle as pkl
from ruspy.simulation.simulation import simulate
from functools import partial
import numpy as np


NUM_WORKERS = 2


def wrapper_func_variying_ev(init, trans_mat, ev_fname):
    print(ev_fname[0][0])
    fname = "dfs/df_" + ev_fname[1]
    pkl.dump(simulate(init, ev_fname[0], trans_mat), open(fname, "wb"))


if __name__ == "__main__":
    worst_evs_fname = []
    worst_trans_mats = []
    for file in sorted(glob.glob("../cg_results/*.pkl")):
        worst_evs_fname += [[pkl.load(open(file, "rb"))[0], file[14:]]]
        worst_trans_mats += [pkl.load(open(file, "rb"))[1]]

    # Beta is set almost to one, as the agents objective is to maximize average cost.
    beta = 0.9999
    # 200 buses should be enough to gurantee convergence.
    num_buses = 200
    # Set the number of simulated periods to 80000. The first plot shows the
    # convergence at this point.
    num_periods = 70000
    params = np.loadtxt("resources/rust_cost_params.txt")

    init_dict = {
        'beta': beta,
        'periods': num_periods,
        'seed': 123,
        'maint_func': 'linear',
        'buses': num_buses,
        'params': params
    }
    ev_wrapper_func = partial(wrapper_func_variying_ev, init_dict, worst_trans_mats[0])

    mp.Pool(NUM_WORKERS).map(ev_wrapper_func, worst_evs_fname)


