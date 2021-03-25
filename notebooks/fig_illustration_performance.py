from itertools import product

import matplotlib.pyplot as plt
from scipy import interpolate
import scipy.stats as stats
import pandas as pd
import numpy as np

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
    # This function does have its maximum very close to 0.5, which corresponds to the mean of
    # the normal sampling distribution
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
    y = [0.0, 1.0, 0.5]
    x = [0.0, 0.5, 1.0]
    return interpolate.interp1d(x, y, kind="quadratic")(grid) + INCR_2

    
def create_plot_1(df):
    
    fig, ax = plt.subplots()

    ax.plot(GRID, dist_func_1(GRID) * 1000 - 4, label="sampling distribution")
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
        df.loc[label, 1] = calculate_perf(dist_func_1, perf_func)

    return df


def calculate_perf(dist_func, perf_func):
    return sum(dist_func(GRID) * perf_func(GRID))


def create_plot_2(df):

    fig, ax = plt.subplots()

    ax.plot(GRID, dist_func_2(GRID) * 1000 - 3, label="sampling distribution")
    ax.plot(GRID, perf_rob_2(GRID), label="robust")
    ax.plot(GRID, perf_opt_2(GRID), label="optimal")
    ax.yaxis.set_ticklabels([])
    ax.legend()

    ax.set_xticks([0.0, GRID[perf_opt_2(GRID).argmax()], 1.0])
    ax.set_xticklabels([0.0, "$p_2$", 1.0])
    ax.set_xlim([0, 1])
    ax.legend()

    ax.set_ylabel("Performance")
    ax.set_xlabel("$\hat{p}$")

    fig.savefig("fig-illustration-performance-2-sw")

    for label, perf_func in [("robust", perf_rob_2), ("optimal", perf_opt_2)]:
        df.loc[label, 2] = calculate_perf(dist_func_2, perf_func)

    return df


def construct_regret(df):
    df_regret = df.copy()
    df_regret.loc[slice(None), :] = None

    for label, info in product(["robust", "optimal"], [[1, perf_opt_1], [2, perf_opt_2]]):
        pos, func = info
        df_regret.loc[label, pos] = max(func(GRID)) - df.loc[label, pos]

    return df_regret

def report_decisions(df):

    df_regret = construct_regret(df)

    # Get decision based on maximin
    print("Maximin:", df.min(axis=1).idxmax())

    # Get decision based on minimax regret
    print("Minimax:", df_regret.max(axis=1).idxmin())

    # Get decision based on subjective Bayes
    print("Bayes:  ", df.mean(axis=1).idxmax())


def report_decision_inputs(df):

    df_regret = construct_regret(df)

    width = 0.35  # the width of the bars
    x = np.arange(2)

    fig, ax = plt.subplots()
    ax.bar(x - width / 2, df.loc["optimal"], width, label='Optimal')
    ax.bar(x + width / 2, df.loc["robust"], width, label='Robust')

    ax.set_ylabel('Expected performance')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['$p_1$', '$p_2$'])
    ax.legend()
    fig.savefig("fig-illustration-expected-performance-sw.png")

    fig, ax = plt.subplots()
    ax.bar(x - width / 2, df_regret.loc["optimal"], width, label='Optimal')
    ax.bar(x + width / 2, df_regret.loc["robust"], width, label='Robust')

    ax.set_ylabel('Expected regret')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['$p_1$', '$p_2$'])
    ax.legend()

    fig.savefig("fig-illustration-expected-regret-sw.png")


if __name__ == '__main__':

    INCR_1 = max(perf_opt_1(GRID)) - max(perf_rob_1(GRID))
    INCR_2 = max(perf_opt_2(GRID)) - max(perf_rob_2(GRID))

    df_results = pd.DataFrame(None, columns=[1, 2], index=["robust", "optimal"])
    df_results.index.names = ["Strategy"]

    df_results = create_plot_1(df_results)
    df_results = create_plot_2(df_results)

    report_decision_inputs(df_results)
    report_decisions(df_results)

