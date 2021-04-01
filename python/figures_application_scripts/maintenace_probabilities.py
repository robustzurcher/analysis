import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config import DIR_FIGURES
from global_vals_funcs import BIN_SIZE
from global_vals_funcs import COLOR_OPTS
from global_vals_funcs import COST_SCALE
from global_vals_funcs import DICT_POLICIES_4292
from global_vals_funcs import DISC_FAC
from global_vals_funcs import PARAMS
from global_vals_funcs import SPEC_DICT
from ruspy.model_code.choice_probabilities import choice_prob_gumbel
from ruspy.model_code.cost_functions import calc_obs_costs
from ruspy.model_code.cost_functions import lin_cost


keys = [0.0, 0.5, 0.95]
max_state = 30


def df_maintenance_probabilties():
    choice_ml, choices = _create_repl_prob_plot(DICT_POLICIES_4292, keys)
    states = np.arange(choice_ml.shape[0]) * BIN_SIZE
    return pd.DataFrame(
        {
            "milage_thousands": states,
            0.0: choice_ml[:, 0],
            keys[1]: choices[0][:, 0],
            keys[2]: choices[1][:, 0],
        }
    )


def get_maintenance_probabilities():

    choice_ml, choices = _create_repl_prob_plot(DICT_POLICIES_4292, keys)
    states = np.arange(max_state) * BIN_SIZE
    for color in COLOR_OPTS:
        fig, ax = plt.subplots(1, 1)

        ax.plot(
            states,
            choice_ml[:max_state, 0],
            color=SPEC_DICT[color]["colors"][0],
            ls=SPEC_DICT[color]["line"][0],
            label="as-if",
        )
        for i, choice in enumerate(choices):
            ax.plot(
                states,
                choice[:max_state, 0],
                color=SPEC_DICT[color]["colors"][i + 1],
                ls=SPEC_DICT[color]["line"][i + 1],
                label=fr"robust $(\omega = {keys[i+1]:.2f})$",
            )

        ax.set_ylabel(r"Maintenance probability")
        ax.set_xlabel(r"Mileage (in thousands)")
        ax.set_ylim([0, 1])

        plt.xticks(states[::5])
        ax.legend()

        fig.savefig(
            f"{DIR_FIGURES}/fig-application-maintenance-probabilities"
            f"{SPEC_DICT[color]['file']}"
        )


def _create_repl_prob_plot(file, keys):
    ev_ml = np.dot(DICT_POLICIES_4292[0.0][1], DICT_POLICIES_4292[0.0][0])
    num_states = ev_ml.shape[0]
    costs = calc_obs_costs(num_states, lin_cost, PARAMS, COST_SCALE)
    choice_ml = choice_prob_gumbel(ev_ml, costs, DISC_FAC)
    choices = []
    for omega in keys[1:]:
        choices += [
            choice_prob_gumbel(
                np.dot(DICT_POLICIES_4292[omega][1], DICT_POLICIES_4292[omega][0]),
                costs,
                DISC_FAC,
            )
        ]
    return choice_ml, choices
