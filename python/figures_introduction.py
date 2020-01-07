import numpy as np
import pandas as pd
from config import DIR_FIGURES
from figures_application import color_opts
from figures_application import spec_dict
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d


def df_num_obs(repl_data, bin_size_plot, init_dict):
    max_state = repl_data["state"].max()
    numobs_per_state = np.bincount(repl_data[repl_data["decision"] == 0]["state"])
    numobs_per_state[0] += len(repl_data[repl_data["decision"] == 1])
    num_steps = int(bin_size_plot / init_dict["binsize"])
    num_bins = int(max_state / num_steps)
    hist_data = np.zeros(num_bins)
    for i in range(num_bins):
        hist_data[i] = np.sum(numobs_per_state[i * num_steps : (i + 1) * num_steps])
    return pd.DataFrame(
        {"Num_Obs": hist_data}, index=np.arange(1, len(hist_data) + 1) * bin_size_plot
    )


def get_number_observations(repl_data, bin_size_plot, init_dict):

    max_state = repl_data["state"].max()
    numobs_per_state = np.bincount(repl_data[repl_data["decision"] == 0]["state"])
    numobs_per_state[0] += len(repl_data[repl_data["decision"] == 1])
    num_steps = int(bin_size_plot / init_dict["binsize"])
    num_bins = int(max_state / num_steps)
    hist_data = np.zeros(num_bins)
    for i in range(num_bins):
        hist_data[i] = np.sum(numobs_per_state[i * num_steps : (i + 1) * num_steps])

    scale = 10
    mileage = np.arange(num_bins) * scale
    width = 0.75 * scale
    for color in color_opts:

        fig, ax = plt.subplots(1, 1)
        ax.set_ylabel(r"Number of observations")
        ax.set_xlabel(r"Milage (in thousands)")

        cl = spec_dict[color]["colors"][0]
        ax.bar(mileage, hist_data, width, align="edge", color=cl)

        plt.savefig(
            f"{DIR_FIGURES}/fig-introduction-observations-mileage{spec_dict[color]['file']}"
        )


def get_intorduction_decision_making():

    for color in color_opts:

        fig, ax = plt.subplots(1, 1)

        x_values = [0.00, 0.33, 0.66, 1.00]
        y_values = [0.95, 0.70, 0.40, 0.00]

        f = interp1d(x_values, y_values, kind="quadratic")
        x_grid = np.linspace(0, 1, num=41, endpoint=True)

        cl = spec_dict[color]["colors"][0]
        ls = spec_dict[color]["line"][0]

        ax.plot(x_grid, f(x_grid), label="optimal", color=cl, ls=ls)

        x_values = [0.00, 0.33, 0.66, 1.00]
        y_values = [0.80, 0.70, 0.50, 0.20]

        f = interp1d(x_values, y_values, kind="quadratic")

        cl = spec_dict[color]["colors"][1]
        ls = spec_dict[color]["line"][1]

        x_grid = np.linspace(0, 1, num=41)

        ax.plot(x_grid, f(x_grid), label="robust", color=cl, ls=ls)

        ax.set_xticks((0.0, 0.2, 0.5, 0.8, 1.0))
        ax.set_xticklabels([])

        ax.yaxis.get_major_ticks()[0].set_visible(False)
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.0)
        ax.set_yticklabels([])

        ax.set_xlabel("Level of model misspecification")
        ax.set_ylabel("Performance")
        ax.legend()

        plt.savefig(
            f"{DIR_FIGURES}/fig-introduction-robust-performance{spec_dict[color]['file']}"
        )
