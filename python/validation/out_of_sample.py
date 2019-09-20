import numpy as np


def create_asym_trans_mat(num_states, n, p_ml, seed=False):
    trans_mat = np.zeros((num_states, num_states), dtype=np.float64)
    for i in range(num_states):  # Loop over all states.
        trans_prob = draw_trans_probs_mulitvar(n, p_ml, seed=seed)
        for j, p in enumerate(trans_prob):  # Loop over the possible increases.
            if i + j < num_states - 1:
                trans_mat[i, i + j] = p
            elif i + j == num_states - 1:
                trans_mat[i, num_states - 1] = np.sum(trans_prob[j:])
            else:
                pass
    return trans_mat


def draw_trans_probs_mulitvar(n, p, seed=False):
    if seed:
        np.random.seed(seed)
    mean = p * n
    cov = calc_cov_multinomial(n, p) * (n ** 2)
    draw_array = np.random.multivariate_normal(mean, cov)
    draw_array[draw_array < 0] = 0
    return draw_array / np.sum(draw_array)


def calc_cov_multinomial(n, p):
    dim = len(p)
    cov = np.zeros(shape=(dim, dim), dtype=float)
    for i in range(dim):
        for j in range(dim):
            if i == j:
                cov[i, i] = p[i] * (1 - p[i])
            else:
                cov[i, j] = -p[i] * p[j]
    return cov / n