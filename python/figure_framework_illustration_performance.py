from itertools import product

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from config import DIR_FIGURES
from global_vals_funcs import COLOR_OPTS
from global_vals_funcs import SPEC_DICT
from scipy import interpolate


GRID = np.linspace(0, 1, 1000)


# We need to make sure that the maximum for both functions is the same, but at different
# positions.
# incr = max(perf_opt_2(GRID)) - max(perf_rob_2(GRID))
INCR_1 = 0.0
INCR_2 = 0.0


def dist_func_1(grid):
    vals = stats.norm.pdf(grid, 0.5, 0.2)
    return vals / vals.sum()


def perf_rob_1(grid):
    y = [-1.0, 0.0, +0.6, -2.0]
    x = [+0.0, 0.3, +0.8, +1.0]
    return interpolate.interp1d(x, y, kind="quadratic")(grid) + INCR_1


def perf_opt_1(grid):
    # This function does have its maximum very close to 0.5, which corresponds to the
    # mean of the normal sampling distribution
    y = [-3.0, 1.2, -3.0]
    x = [+0.0, 0.5, +1.0]
    return interpolate.interp1d(x, y, kind="quadratic")(grid)


def dist_func_2(grid):
    vals = stats.lognorm.pdf(grid, 1.1, 0)
    return vals / vals.sum()


def perf_opt_2(grid):
    y = [-1.5, 2.0, +0.2, -1.5]
    x = [+0.0, 0.3, +0.7, +1.0]
    return interpolate.interp1d(x, y, kind="quadratic")(grid)


def perf_rob_2(grid):
    y = [0.5, 1.0, 0.5]
    x = [0.0, 0.5, 1.0]
    return interpolate.interp1d(x, y, kind="quadratic")(grid) + INCR_2


def create_plot_1():
    mpl.rcParams["axes.spines.right"] = True
    for color in COLOR_OPTS:
        fig, ax = plt.subplots()
        ax2 = ax.twinx()
        ax2.plot(
            GRID,
            dist_func_1(GRID) * 1000 - 4,
            label="sampling distribution",
            color=SPEC_DICT[color]["colors"][0],
        )
        ax.plot(
            GRID,
            perf_rob_1(GRID),
            label=r"robust",
            color=SPEC_DICT[color]["colors"][1],
        )
        ax.plot(
            GRID,
            perf_opt_1(GRID),
            label=r"optimal",
            color=SPEC_DICT[color]["colors"][2],
        )

        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_xticklabels([0.0, "$p_1$", 1.0])
        ax.set_xlim([0, 1])
        ax.set_xlabel(r"$\hat{p}$")

        ax.set_yticks([])
        ax.set_ylim(-3, 2)
        ax.set_ylabel("Performance")

        ax2.set_yticks([])
        ax2.set_ylim(-5, 1)
        ax2.set_ylabel("Sampling Distribution")

        ax.legend(loc="upper left", ncol=1)
        ax2.legend(loc="upper right")

        fig.savefig(
            f"{DIR_FIGURES}/fig-illustration-performance-1{SPEC_DICT[color]['file']}"
        )
    mpl.rcParams["axes.spines.right"] = False


def calculate_perf(dist_func, perf_func):
    return sum(dist_func(GRID) * perf_func(GRID))


def create_plot_2():
    mpl.rcParams["axes.spines.right"] = True
    for color in COLOR_OPTS:

        fig, ax = plt.subplots()
        ax2 = ax.twinx()
        ax2.plot(
            GRID,
            dist_func_2(GRID) * 1000 - 3,
            label="sampling distribution",
            color=SPEC_DICT[color]["colors"][0],
        )
        ax.plot(
            GRID,
            perf_rob_2(GRID),
            label="robust",
            color=SPEC_DICT[color]["colors"][1],
        )
        ax.plot(
            GRID,
            perf_opt_2(GRID),
            label="optimal",
            color=SPEC_DICT[color]["colors"][2],
        )
        ax.yaxis.set_ticklabels([])
        ax.legend()

        ax.set_xticks([0.0, GRID[perf_opt_2(GRID).argmax()], 1.0])
        ax.set_xticklabels([0.0, "$p_2$", 1.0])
        ax.set_xlim([0, 1])

        ax.set_yticks([])
        ax.set_ylabel("Performance")
        ax.set_xlabel(r"$\hat{p}$")
        ax.set_ylim(-2, 3)

        ax2.set_yticks([])
        ax2.set_ylim(-3, 1)
        ax2.set_ylabel("Sampling Distribution")

        ax.legend(loc="upper left", ncol=1)
        ax2.legend(loc="upper right")

        fig.savefig(
            f"{DIR_FIGURES}/fig-illustration-performance-2{SPEC_DICT[color]['file']}"
        )
    mpl.rcParams["axes.spines.right"] = False


def construct_regret():
    df = pd.DataFrame(None, columns=[1, 2], index=["robust", "optimal"])
    df.index.names = ["Strategy"]

    for label, perf_func in [("robust", perf_rob_1), ("optimal", perf_opt_1)]:
        df.loc[label, 1] = calculate_perf(dist_func_1, perf_func)

    for label, perf_func in [("robust", perf_rob_2), ("optimal", perf_opt_2)]:
        df.loc[label, 2] = calculate_perf(dist_func_2, perf_func)

    df_regret = df.copy()
    df_regret.loc[slice(None), :] = None

    for label, info in product(
        ["robust", "optimal"], [[1, perf_opt_1], [2, perf_opt_2]]
    ):
        pos, func = info
        df_regret.loc[label, pos] = max(func(GRID)) - df.loc[label, pos]

    return df_regret


def report_decisions():

    df = construct_regret()

    # Get decision based on maximin
    print("Maximin:", df.min(axis=1).idxmax())

    # Get decision based on minimax regret
    print("Minimax:", df.max(axis=1).idxmin())

    # Get decision based on subjective Bayes
    print("Bayes:  ", df.mean(axis=1).idxmax())


def report_decision_inputs():

    df = construct_regret()

    width = 0.35  # the width of the bars
    x = np.arange(2)

    fig, ax = plt.subplots()
    ax.bar(x - width / 2, df.loc["optimal"], width, label="Optimal")
    ax.bar(x + width / 2, df.loc["robust"], width, label="Robust")

    ax.set_ylabel("Expected performance")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["$p_1$", "$p_2$"])
    ax.legend()
    fig.savefig("fig-illustration-expected-performance-sw.png")

    fig, ax = plt.subplots()
    ax.bar(x - width / 2, df.loc["optimal"], width, label="Optimal")
    ax.bar(x + width / 2, df.loc["robust"], width, label="Robust")

    ax.set_ylabel("Expected regret")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["$p_1$", "$p_2$"])
    ax.legend()

    fig.savefig("fig-illustration-expected-regret-sw.png")


if __name__ == "__main__":

    # We need both decision rules to have the same maximum for consistency.
    INCR_1 = max(perf_opt_1(GRID)) - max(perf_rob_1(GRID))
    INCR_2 = max(perf_opt_2(GRID)) - max(perf_rob_2(GRID))

    df_results = pd.DataFrame(None, columns=[1, 2], index=["robust", "optimal"])
    df_results.index.names = ["Strategy"]

    df_results = create_plot_1()
    df_results = create_plot_2()

    report_decision_inputs()
    report_decisions()
