import numpy as np
import numba
from robupy.auxiliary import get_worst_case_probs
from ruspy.estimation.estimation_cost_parameters import create_transition_matrix


@numba.jit(nopython=True)
def create_worst_trans_mat(trans_mat, v, rho):
    num_states = trans_mat.shape[0]
    worst_trans_mat = np.zeros(shape=(num_states, num_states), dtype=np.float64)
    for s in range(num_states):
        ind_non_zero = np.nonzero(trans_mat[s, :])[0]
        p_min = np.amin(ind_non_zero)
        p_max = np.amax(ind_non_zero)
        p = trans_mat[s, p_min : p_max + 1]
        v_intern = v[p_min : p_max + 1]
        worst_trans_mat[s, p_min : p_max + 1] = get_worst_case_probs(
            v_intern, p, rho, is_cost=False
        )
    return worst_trans_mat


@numba.jit(nopython=True)
def calc_fixp_worst(
    num_states, p_ml, costs, beta, rho, threshold=1e-8, max_it=1000000
):
    ev = np.zeros(num_states)
    worst_trans_mat = trans_mat = create_transition_matrix(num_states, p_ml)
    ev_new = np.dot(trans_mat, np.log(np.sum(np.exp(-costs), axis=1)))
    while (np.max(np.abs(ev_new - ev)) > threshold) & (max_it != 0):
        ev = ev_new
        maint_value = beta * ev - costs[:, 0]
        repl_value = beta * ev[0] - costs[0, 1] - costs[0, 0]

        # Select the minimal absolute value to rescale the value vector for the
        # exponential function.
        ev_min = maint_value[0]

        log_sum = ev_min + np.log(
            np.exp(maint_value - ev_min) + np.exp(repl_value - ev_min)
        )
        worst_trans_mat = create_worst_trans_mat(trans_mat, log_sum, rho)
        ev_new = np.dot(worst_trans_mat, log_sum)
        max_it -= 1
    if max_it == 0:
        print("The value function didn't converge and with absolute difference:")
        print(np.max(np.abs(ev_new - ev)))
    return ev_new, worst_trans_mat, max_it
