import glob
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config import DIR_FIGURES
from global_vals_funcs import BIN_SIZE
from global_vals_funcs import COLOR_OPTS
from global_vals_funcs import OMEGA_GRID
from global_vals_funcs import SIM_RESULTS
from global_vals_funcs import SPEC_DICT

num_keys = 100


def df_thresholds():
    means_discrete = _threshold_data()
    omega_range = np.linspace(0, 0.99, num_keys)
    return pd.DataFrame({"omega": omega_range, "threshold": means_discrete})


def get_replacement_thresholds():

    means_discrete = _threshold_data() * BIN_SIZE

    means_ml = np.full(len(OMEGA_GRID), np.round(means_discrete[0])).astype(int)

    omega_sections, state_sections = _create_sections(means_discrete, OMEGA_GRID)

    y_0 = state_sections[0][0] - 2 * BIN_SIZE
    y_1 = state_sections[-1][-1] + 2 * BIN_SIZE
    for color in COLOR_OPTS:
        fig, ax = plt.subplots(1, 1)
        ax.set_ylabel(r"Mileage (in thousands)")
        ax.set_xlabel(r"$\omega$")
        ax.set_ylim([y_0, y_1])

        ax.plot(
            OMEGA_GRID,
            means_ml,
            color=SPEC_DICT[color]["colors"][1],
            ls=SPEC_DICT[color]["line"][0],
            label="optimal",
        )
        if color == "colored":
            second_color = "#ff7f0e"
        else:
            second_color = SPEC_DICT[color]["colors"][0]
        for j, i in enumerate(omega_sections[:-1]):
            ax.plot(
                i, state_sections[j], color=second_color, ls=SPEC_DICT[color]["line"][1]
            )
        ax.plot(
            omega_sections[-1],
            state_sections[-1],
            color=second_color,
            ls=SPEC_DICT[color]["line"][1],
            label="robust",
        )
        ax.legend()

        fig.savefig(
            f"{DIR_FIGURES}/fig-application-replacement-thresholds{SPEC_DICT[color]['file']}"
        )


def _threshold_data():
    file_list = sorted(glob.glob(SIM_RESULTS + "result_ev_*_mat_0.0.pkl"))
    if len(file_list) != 0:
        means_robust_strat = np.array([])
        for omega in OMEGA_GRID:
            mean = pkl.load(open(SIM_RESULTS + f"result_ev_{omega}_mat_0.0.pkl", "rb"))[
                0
            ]
            means_robust_strat = np.append(means_robust_strat, mean)
    else:
        raise AssertionError("Need to unpack simulation files")

    means_discrete = np.around(means_robust_strat).astype(int)
    return means_discrete


def _create_sections(mean_disc, om_range):
    omega_sections = []
    state_sections = []
    for j, i in enumerate(np.unique(mean_disc)):
        where = mean_disc == i
        max_ind = np.max(np.where(mean_disc == i))
        if j == 0:
            med_val = (np.max(om_range[where]) + np.min(om_range[~where])) / 2
            omega_sections += [np.append(om_range[where], med_val)]
            state_sections += [np.append(mean_disc[where], i)]
        elif j == (len(np.unique(mean_disc)) - 1):
            med_val = (np.min(om_range[where]) + np.max(om_range[~where])) / 2
            omega_sections += [np.array([med_val] + om_range[where].tolist())]
            state_sections += [np.array([i] + mean_disc[where].tolist())]
        else:
            low = (np.min(om_range[where]) + np.max(omega_sections[-1][:-1])) / 2
            high = (np.max(om_range[where]) + np.min(om_range[max_ind + 1])) / 2
            omega_sections += [np.array([low] + om_range[where].tolist() + [high])]
            state_sections += [np.array([i] + mean_disc[where].tolist() + [i])]
    return omega_sections, state_sections
