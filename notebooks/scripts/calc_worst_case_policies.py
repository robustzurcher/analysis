
import numpy as np
from worst_case_probs import calc_fixp_worst
from ruspy.estimation.estimation_cost_parameters import (
    cost_func,
    lin_cost,
create_transition_matrix
)
import multiprocessing as mp
import pickle
from scipy.stats import chi2


workers = 2  # As there are a 100 omegas in the range
omega_range = np.arange(0, 1, 0.5)
beta = 0.9999
num_states = 20
p_ml = np.loadtxt("resources/rust_trans_probs")
params = np.loadtxt("resources/rust_cost_params")
costs = cost_func(num_states, lin_cost, params)

trans_mat = create_transition_matrix()


def wrapper_func(omega):
    rho = chi2.ppf(omega, len(p_ml) - 1) / (2 * (78 / 4292))
    return calc_fixp_worst(num_states, p_ml, costs, beta, rho)


pool = mp.Pool(workers)
test = pool.map(
    wrapper_func, omega_range
)
with open("results.pkl", "wb") as pkl_out:
    pickle.dump(test, pkl_out)
