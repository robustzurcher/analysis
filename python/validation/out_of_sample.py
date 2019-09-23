import numpy as np


def create_asym_trans_mat(num_states, p_raw, hesse_inv_raw):
    trans_mat = np.zeros((num_states, num_states), dtype=np.float64)
    for i in range(num_states):  # Loop over all states.
        trans_prob = draw_from_raw(p_raw, hesse_inv_raw)
        for j, p in enumerate(trans_prob):  # Loop over the possible increases.
            if i + j < num_states - 1:
                trans_mat[i, i + j] = p
            elif i + j == num_states - 1:
                trans_mat[i, num_states - 1] = np.sum(trans_prob[j:])
            else:
                pass
    return trans_mat


def draw_from_raw(p_raw, hesse_inv_raw):
    draw = np.random.multivariate_normal(p_raw, hesse_inv_raw)
    return np.exp(draw) / np.sum(np.exp(draw))