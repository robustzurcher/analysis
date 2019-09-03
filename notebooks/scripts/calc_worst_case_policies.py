import numpy as np
from worst_case_probs import calc_fixp_worst
from ruspy.estimation.estimation_cost_parameters import cost_func, lin_cost
import multiprocessing as mp
import pickle
from scipy.stats import chi2


workers = 100  # As there are a 100 omegas in the range
omega_range = np.arange(0, 1, 0.01)


def wrapper_func(omega):
    beta = 0.9999
    num_states = 400
    p_ml = np.loadtxt("resources/rust_trans_probs.txt")
    params = np.loadtxt("resources/rust_cost_params.txt")
    costs = cost_func(num_states, lin_cost, params)
    rho = chi2.ppf(omega, len(p_ml) - 1) / (2 * (4292 / 78))
    return calc_fixp_worst(num_states, p_ml, costs, beta, rho, threshold=1e-7)


if __name__ == "__main__":
    pool = mp.Pool(workers)
    test = pool.map(wrapper_func, omega_range)
    with open("results.pkl", "wb") as pkl_out:
        pickle.dump(test, pkl_out)
