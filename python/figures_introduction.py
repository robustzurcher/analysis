import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

from config import DIR_FIGURES
from figures_application import color_opts, spec_dict


def get_number_observations(num_bins, init_dict, trans_results):

    numobs_per_state = trans_results["state_count"].sum(axis=1)
    hist_data = np.array([])
    for i, val in enumerate(numobs_per_state):
        hist_data = np.append(hist_data, np.full(val, i))
    hist_data = hist_data * init_dict["binsize"] / 1000

    for color in color_opts:

        fig, ax = plt.subplots(1, 1)
        ax.set_ylabel(r"Number of observations")
        ax.set_xlabel(r"Milage (in thousands)")

        cl = spec_dict[color]["colors"][0]
        ax.hist(hist_data, bins=num_bins, color=cl, rwidth=0.8)

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