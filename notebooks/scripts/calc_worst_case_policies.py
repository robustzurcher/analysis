import numpy as np
from worst_case_probs import calc_fixp_worst
from ruspy.estimation.estimation_cost_parameters import cost_func, lin_cost
import multiprocessing as mp
import pickle
from scipy.stats import chi2
import os

workers = 30
omega_range = np.linspace(0, 1, 26)
# omega_range = np.linspace(0, 1, 3)  # Debugging mode


def wrapper_func(omega):
    beta = 0.9999
    # num_states = 25  # Debugging
    num_states = 400
    p_ml = np.loadtxt("resources/rust_trans_probs.txt")
    params = np.loadtxt("resources/rust_cost_params.txt")
    costs = cost_func(num_states, lin_cost, params)
    rho = chi2.ppf(omega, len(p_ml) - 1) / (2 * (4292 / 78))
    result = calc_fixp_worst(num_states, p_ml, costs, beta, rho, threshold=1e-7)
    with open("results/intermediate_" + str(omega)[:4] + ".pkl", "wb") as pkl_inter:
        pickle.dump(result, pkl_inter)
    return result


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    pool = mp.Pool(workers)
    final_result = pool.map(wrapper_func, omega_range)
    with open("results/final_result.pkl", "wb") as pkl_out:
        pickle.dump(final_result, pkl_out)
    for file in os.listdir("results"):
        if file.startswith("intermediate"):
            os.remove("results/" + file)