import numpy as np
from config import DIR_FIGURES
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scripts_figures.global_vals_funcs import COLOR_OPTS
from scripts_figures.global_vals_funcs import SPEC_DICT


def get_introduction_decision_making():

    for color in COLOR_OPTS:

        fig, ax = plt.subplots(1, 1)

        x_values = [0.00, 0.33, 0.66, 1.00]
        y_values = [0.95, 0.70, 0.40, 0.00]

        f = interp1d(x_values, y_values, kind="quadratic")
        x_grid = np.linspace(0, 1, num=41, endpoint=True)

        cl = SPEC_DICT[color]["colors"][0]
        ls = SPEC_DICT[color]["line"][0]

        ax.plot(x_grid, f(x_grid), label="as-if", color=cl, ls=ls)

        x_values = [0.00, 0.33, 0.66, 1.00]
        y_values = [0.80, 0.70, 0.50, 0.20]

        f = interp1d(x_values, y_values, kind="quadratic")

        cl = SPEC_DICT[color]["colors"][1]
        ls = SPEC_DICT[color]["line"][1]

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
            f"{DIR_FIGURES}/fig-introduction-robust-performance{SPEC_DICT[color]['file']}"
        )
