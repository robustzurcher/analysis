import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config import DIR_FIGURES
from scripts_figures.global_vals_funcs import COLOR_OPTS
from scripts_figures.global_vals_funcs import SPEC_DICT


def df_num_obs(bin_size, init_dict, num_obs_per_state):
    max_state = num_obs_per_state.index.max()
    num_steps = int(bin_size / init_dict["binsize"])
    num_bins = int(max_state / num_steps)
    hist_data = np.zeros(num_bins)
    for i in range(num_bins):
        hist_data[i] = np.sum(num_obs_per_state[i * num_steps : (i + 1) * num_steps])
    return pd.DataFrame(
        {"Num_Obs": hist_data}, index=np.arange(1, len(hist_data) + 1) * bin_size
    )


def get_number_observations(bin_size, init_dict, num_obs_per_state):

    max_state = num_obs_per_state.index.max()
    num_steps = int(bin_size / init_dict["binsize"])
    num_bins = int(max_state / num_steps)
    hist_data = np.zeros(num_bins)
    for i in range(num_bins):
        hist_data[i] = np.sum(num_obs_per_state[i * num_steps : (i + 1) * num_steps])

    scale = 10
    mileage = np.arange(num_bins) * scale
    width = 0.75 * scale
    for color in COLOR_OPTS:

        fig, ax = plt.subplots(1, 1)
        ax.set_ylabel(r"Number of observations")
        ax.set_xlabel(r"Milage (in thousands)")

        cl = SPEC_DICT[color]["colors"][0]
        ax.bar(mileage, hist_data, width, align="edge", color=cl)
        ax.set_xticks(range(0, 450, 50))
        ax.set_xticklabels(range(0, 450, 50))
        ax.set_ylim([0, 250])
        plt.xlim(right=400)

        plt.savefig(
            f"{DIR_FIGURES}/fig-introduction-observations-mileage{SPEC_DICT[color]['file']}"
        )
