from itertools import product

import matplotlib.pyplot as plt
from scipy import interpolate
import scipy.stats as stats
import pandas as pd
import numpy as np

GRID = np.linspace(0, 1, 100)


def dist_func_1(grid):
    vals = stats.norm.pdf(grid, 0.5, 0.2)
    return vals / vals.sum()


def perf_rob_1(grid):
    y = [-1.0, 0.0, +0.6, -2.0]
    x = [+0.0, 0.3, +0.8, +1.0]
    return interpolate.interp1d(x, y, kind="quadratic")(grid)


def perf_opt_1(grid):
    y = [-2.0, 1.1, -3.0]
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
    return interpolate.interp1d(x, y, kind="quadratic")(grid)

    
def create_plot_1(df):
    
    fig, ax = plt.subplots()

    ax.plot(GRID, dist_func_1(GRID) * 100 - 4, label="sampling distribution")
    ax.plot(GRID, perf_rob_1(GRID), label=r"robust",)
    ax.plot(GRID, perf_opt_1(GRID), label=r"optimal")
    ax.yaxis.set_ticklabels([])

    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xticklabels([0.0, "$p_1$", 1.0])
    ax.set_xlim([0, 1])

    ax.legend()

    ax.set_ylabel("Performance")
    ax.set_xlabel("$\hat{p}$")

    fig.savefig("fig-illustration-performance-1-sw")

    for label, perf_func in [("robust", perf_rob_1), ("optimal", perf_opt_1)]:
        df.loc[label, 0] = calculate_perf(dist_func_1, perf_func)

    return df


def calculate_perf(dist_func, perf_func):
    return sum(dist_func(GRID) * perf_func(GRID))


def create_plot_2(df):

    fig, ax = plt.subplots()

    ax.plot(GRID, dist_func_2(GRID) * 100 - 3, label="sampling distribution")
    ax.plot(GRID, perf_rob_2(GRID), label="robust")
    ax.plot(GRID, perf_opt_2(GRID), label="optimal")
    ax.yaxis.set_ticklabels([])
    ax.legend()

    ax.set_xticks([0.0, 0.3, 1.0])
    ax.set_xticklabels([0.0, "$p_2$", 1.0])
    ax.set_xlim([0, 1])
    ax.legend()

    ax.set_ylabel("Performance")
    ax.set_xlabel("$\hat{p}$")

    fig.savefig("fig-illustration-performance-2-sw")

    for label, perf_func in [("robust", perf_rob_2), ("optimal", perf_opt_2)]:
        df.loc[label, 1] = calculate_perf(dist_func_2, perf_func)

    return df


def report_decisions(df):

    df_regret = df.copy()
    df_regret.loc[slice(None), :] = None

    for label, point in product(["robust", "optimal"], [0, 1]):
        df_regret.loc[label, point] = df.loc[slice(None), point].max() - df.loc[label, point]

    # Get decision based on maximin
    print("Maximin:", df.min(axis=1).idxmax())

    # Get decision based on minimax regret
    print("Minimax:", df_regret.max(axis=1).idxmin())

    # Get decision based on subjective Bayes
    print("Bayes:  ", df.mean(axis=1).idxmax())


if __name__ == '__main__':

    df_results = pd.DataFrame(None, columns=range(2), index=["robust", "optimal"])
    df_results.index.names = ["Strategy"]

    df_results = create_plot_1(df_results)
    df_results = create_plot_2(df_results)

    report_decisions(df_results)
