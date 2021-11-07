import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config import DIR_FIGURES
from figures.global_vals_funcs import BIN_SIZE
from figures.global_vals_funcs import COLOR_OPTS
from figures.global_vals_funcs import COST_SCALE
from figures.global_vals_funcs import DICT_POLICIES_4292
from figures.global_vals_funcs import PARAMS
from figures.global_vals_funcs import SPEC_DICT
from ruspy.model_code.cost_functions import calc_obs_costs
from ruspy.model_code.cost_functions import lin_cost
from ruspy.simulation.simulation import simulate


def get_demonstration_df(init_dict):
    states, periods = get_demonstration_data(init_dict)
    return pd.DataFrame(
        {
            "months_ml": periods[0],
            "months_rob": periods[1],
            "opt_mileage": states[0] * BIN_SIZE,
            "rob_mileage": states[1] * BIN_SIZE,
        }
    )


def get_demonstration(df, max_period):
    states = (df["opt_mileage"], df["rob_mileage"])
    periods = (df["months_ml"], df["months_rob"])
    labels = ["as-if", r"robust ($\omega = 0.95$)"]
    for color in COLOR_OPTS:
        fig, ax = plt.subplots(1, 1)
        ax.set_xlabel(r"Months")
        ax.set_ylabel(r"Mileage (in thousands)")

        for i, state in enumerate(states):
            ax.plot(
                periods[i],
                state,
                color=SPEC_DICT[color]["colors"][i],
                ls=SPEC_DICT[color]["line"][i],
                label=labels[i],
            )
        if color == "colored":
            id = (states[1] == states[0]).idxmin()
            ax.plot(
                periods[1][:id],
                states[1][:id],
                color=SPEC_DICT[color]["colors"][0],
                ls="--",
            )
        ax.legend(loc="upper left")
        ax.set_ylim([0, 90])
        plt.xlim(left=-3, right=max_period)

        plt.xticks(range(0, max_period + 10, 10))

        fig.savefig(
            f"{DIR_FIGURES}/fig-application-demonstration{SPEC_DICT[color]['file']}"
        )


def get_demonstration_data(init_dict):

    ev_ml = np.dot(DICT_POLICIES_4292[0.0][1], DICT_POLICIES_4292[0.0][0])
    ev_95 = np.dot(DICT_POLICIES_4292[0.95][1], DICT_POLICIES_4292[0.95][0])
    trans_mat = DICT_POLICIES_4292[0.0][1]

    num_states = ev_ml.shape[0]

    costs = calc_obs_costs(num_states, lin_cost, PARAMS, COST_SCALE)

    df_ml = simulate(init_dict, ev_ml, costs, trans_mat)
    df_95 = simulate(init_dict, ev_95, costs, trans_mat)

    periods_ml = np.array(range(init_dict["periods"]), dtype=int)
    periods_95 = np.array(range(init_dict["periods"]), dtype=int)
    periods = [periods_ml, periods_95]
    states_ml = np.array(df_ml["state"], dtype=int)
    states_95 = np.array(df_95["state"], dtype=int)
    states = [states_ml, states_95]

    for i, df in enumerate([df_ml, df_95]):
        index = (
            np.array(
                df[df["decision"] == 1].index.get_level_values("period"), dtype=int
            )
            + 1
        )
        states[i] = np.insert(states[i], index, 0)
        periods[i] = np.insert(periods[i], index, index - 1)

    return states, periods
