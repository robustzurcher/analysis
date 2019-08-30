<<<<<<< HEAD
<<<<<<< HEAD
import numpy as np
from worst_case_probs import create_worst_trans_mat
from ruspy.estimation.estimation_cost_parameters import create_transition_matrix, \
    cost_func, lin_cost

num_states = 200
p_ml = np.loadtxt("resources/rust_trans_probs")
params = np.loadtxt("resources/rust_cost_params")
costs = cost_func(num_states, lin_cost, params)

trans_mat = create_transition_matrix(num_states, p_ml)
worst_trans = create_worst_trans_mat(trans_mat, costs[0, 0:3], 0)
=======
=======
import numpy as np
>>>>>>> First lines.
from worst_case_probs import create_worst_trans_mat
from ruspy.estimation.estimation_cost_parameters import create_transition_matrix, \
    cost_func, lin_cost

num_states = 200
p_ml = np.loadtxt("resources/rust_trans_probs")
params = np.loadtxt("resources/rust_cost_params")
costs = cost_func(num_states, lin_cost, params)

<<<<<<< HEAD


trans_mat = create_transition_matrix()
create_worst_trans_mat()
>>>>>>> Rebased on add_robupy_calcs.
=======
trans_mat = create_transition_matrix(num_states, p_ml)
worst_trans = create_worst_trans_mat(trans_mat, costs[0, 0:3], 0)
>>>>>>> First lines.
