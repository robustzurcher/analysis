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

GRIP_POINTS = 1000
GRID = np.linspace(0, 1, GRIP_POINTS)


def dist_func_1(grid):
    vals = stats.norm.pdf(grid, 0.5, 0.2)
    vals = vals - vals.min()
    return vals / vals.sum()


def perf_rob_1(grid):

    y = [-1.0, 0.0, +0.6, -2.0]
    x = [+0.0, 0.3, +0.8, +1.0]
    interpoled_result = interpolate.interp1d(x, y, kind="quadratic")(grid)
    increment = max(perf_opt_1(grid)) - max(interpoled_result)
    return interpoled_result + increment


def perf_opt_1(grid):
    # This function does have its maximum very close to 0.5, which corresponds to the
    # mean of the normal sampling distribution
    y = [-3.0, 1.2, -3.0]
    x = [+0.0, 0.5, +1.0]
    return interpolate.interp1d(x, y, kind="quadratic")(grid)


def dist_func_2(grid):
    vals = stats.lognorm.pdf(grid, 1, 0)
    return vals / vals.sum()


def perf_opt_2(grid):
    y = [-1.5, 2.0, +0.2, -1.5]
    x = [+0.0, 0.3, +0.7, +1.0]
    return interpolate.interp1d(x, y, kind="quadratic")(grid)


def perf_rob_2(grid):
    y = [-0.3, 1.0, 0.5]
    x = [0.0, 0.5, 1.0]
    interpoled_result = interpolate.interp1d(x, y, kind="quadratic")(grid)
    # We need both decision rules to have the same maximum for consistency.
    increment = max(perf_opt_2(grid)) - max(interpoled_result)
    return interpoled_result + increment


def create_plot_1():
    mpl.rcParams["axes.spines.right"] = True
    density = dist_func_1(GRID) * GRIP_POINTS / 5
    for color in COLOR_OPTS:
        fig, ax = plt.subplots()
        ax2 = ax.twinx()
        ax.set_xticks([0.0, 1.0])
        ax.set_xlim([-0.05, 1.05])
        ax.set_xlabel(r"$\hat{p}$")

        # ax.set_yticks([])
        ax.set_ylim(-3, 3)
        ax.set_ylabel("Performance")
        ax.set_yticks(range(-3, 4))
        ax.set_xticks([0.0, 1.0])
        ax.set_xticklabels([0.0, 1.0])

        # ax2.set_yticks([])
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("Sampling distribution")
        ax2.set_yticks([])

        if color == "colored":
            fig.savefig(
                f"{DIR_FIGURES}/fig-illustration-performance-1-"
                f"blank{SPEC_DICT[color]['file']}"
            )
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_xticklabels([0.0, "$p_1$", 1.0])
        if color == "colored":
            fig.savefig(
                f"{DIR_FIGURES}/fig-illustration-performance-1-"
                f"p_1{SPEC_DICT[color]['file']}"
            )

        ax2.plot(
            GRID,
            density,
            label="sampling distribution",
            color=SPEC_DICT[color]["colors"][2],
        )

        ax2.legend(loc="upper right")
        ax.set_xticklabels([0.0, "$p_1$", 1.0])

        if color == "colored":
            fig.savefig(
                f"{DIR_FIGURES}/fig-illustration-performance-1-"
                f"sampling{SPEC_DICT[color]['file']}"
            )

        ax.plot(
            GRID,
            perf_opt_1(GRID),
            label=r"as-if",
            color=SPEC_DICT[color]["colors"][0],
        )
        ax.legend(loc="upper left", ncol=1)
        ax2.legend(loc="upper right")

        if color == "colored":
            fig.savefig(
                f"{DIR_FIGURES}/fig-illustration-performance-1-as"
                f"-if{SPEC_DICT[color]['file']}"
            )
        ax.plot(
            GRID,
            perf_rob_1(GRID),
            label=r"robust",
            color=SPEC_DICT[color]["colors"][1],
        )
        ax.legend(loc="upper left", ncol=1)
        ax2.legend(loc="upper right")

        if color == "colored":
            fig.savefig(
                f"{DIR_FIGURES}/fig-illustration-performance-1-"
                f"robust{SPEC_DICT[color]['file']}"
            )

        intersect_idx = np.argwhere(
            np.diff(np.sign(perf_rob_1(GRID) - perf_opt_1(GRID)))
        )
        ax2.fill_between(
            GRID[intersect_idx[0][0] : intersect_idx[1][0]],
            density[intersect_idx[0][0] : intersect_idx[1][0]],
            color=SPEC_DICT[color]["colors"][2],
            alpha=0.4,
        )
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
    density = dist_func_2(GRID) * GRIP_POINTS / 4
    for color in COLOR_OPTS:

        fig, ax = plt.subplots()
        ax2 = ax.twinx()
        ax.plot(
            GRID,
            perf_opt_2(GRID),
            label="as-if",
            color=SPEC_DICT[color]["colors"][0],
        )

        ax.plot(
            GRID,
            perf_rob_2(GRID),
            label="robust",
            color=SPEC_DICT[color]["colors"][1],
        )

        ax2.plot(
            GRID,
            density,
            label="sampling distribution",
            color=SPEC_DICT[color]["colors"][2],
        )

        intersect_idx = np.argwhere(
            np.diff(np.sign(perf_rob_2(GRID) - perf_opt_2(GRID)))
        )
        ax2.fill_between(
            GRID[intersect_idx[0][0] : intersect_idx[1][0]],
            density[intersect_idx[0][0] : intersect_idx[1][0]],
            color=SPEC_DICT[color]["colors"][2],
            alpha=0.4,
        )

        ax.legend()

        ax.set_xticks([0.0, GRID[perf_opt_2(GRID).argmax()], 1.0])
        ax.set_xticklabels([0.0, "$p_2$", 1.0])
        ax.set_xlim([-0.05, 1.05])

        # ax.set_yticks([])
        ax.set_ylabel("Performance")
        ax.set_xlabel(r"$\hat{p}$")
        ax.set_ylim(-3, 3)
        ax.set_yticks(range(-3, 4))

        # ax2.set_yticks([])
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("Sampling distribution")
        ax2.set_yticks([])

        ax.legend(loc="upper left", ncol=1)
        ax2.legend(loc="upper right")

        fig.savefig(
            f"{DIR_FIGURES}/fig-illustration-performance-2{SPEC_DICT[color]['file']}"
        )
    mpl.rcParams["axes.spines.right"] = False


def construct_performance():
    df = pd.DataFrame(None, columns=[1, 2], index=["robust", "as-if"])
    df.index.names = ["Strategy"]

    for label, perf_func in [("robust", perf_rob_1), ("as-if", perf_opt_1)]:
        df.loc[label, 1] = calculate_perf(dist_func_1, perf_func)

    for label, perf_func in [("robust", perf_rob_2), ("as-if", perf_opt_2)]:
        df.loc[label, 2] = calculate_perf(dist_func_2, perf_func)
    return df


def construct_regret(df):

    df_regret = df.copy()
    df_regret.loc[slice(None), :] = None

    for label, info in product(["robust", "as-if"], [[1, perf_opt_1], [2, perf_opt_2]]):
        pos, func = info
        df_regret.loc[label, pos] = max(func(GRID)) - df.loc[label, pos]

    return df_regret


def report_decisions(df, df_regret, measures):

    # Get decision based on maximin
    print("Maximin:", df.min(axis=1).idxmax())

    # Get decision based on minimax regret
    print("Minimax:", df_regret.max(axis=1).idxmin())

    # Get decision based on subjective Bayes
    print("Bayes:  ", df.mean(axis=1).idxmax())

    df_rank = pd.DataFrame(
        data=[[1, 1], [1, 1], [1, 1]], index=measures, columns=["as-if", "robust"]
    )

    df_rank.loc["Subjective \n Bayes", df.mean(axis=1).idxmax()] = 0
    df_rank.loc["Minimax \n regret", df_regret.max(axis=1).idxmin()] = 0
    df_rank.loc["Maximin", df.min(axis=1).idxmax()] = 0
    return df_rank


def expected_performance(df):

    width = 0.35  # the width of the bars
    x = np.arange(2)

    for color in COLOR_OPTS:
        fig, ax = plt.subplots()
        ax.bar(
            x - width / 2,
            df.loc["as-if"],
            width,
            label="as-if",
            color=SPEC_DICT[color]["colors"][0],
        )
        ax.bar(
            x + width / 2,
            df.loc["robust"],
            width,
            label="robust",
            color=SPEC_DICT[color]["colors"][1],
        )

        ax.set_ylabel("Expected performance")
        ax.set_ylim([0, 2])
        ax.set_yticks(np.arange(0, 2.5, 0.5))
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["$p_1$", "$p_2$"])
        ax.legend()
        fig.savefig(
            f"{DIR_FIGURES}/fig-illustration-expected-performance{SPEC_DICT[color]['file']}"
        )


def expected_regret(df):
    width = 0.35  # the width of the bars
    x = np.arange(2)

    for color in COLOR_OPTS:

        fig, ax = plt.subplots()
        ax.bar(
            x - width / 2,
            df.loc["as-if"],
            width,
            label="as-if",
            color=SPEC_DICT[color]["colors"][0],
        )
        ax.bar(
            x + width / 2,
            df.loc["robust"],
            width,
            label="robust",
            color=SPEC_DICT[color]["colors"][1],
        )

        ax.set_ylabel("Expected regret")
        ax.set_ylim([0, 2])
        ax.set_yticks(np.arange(0, 2.5, 0.5))
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["$p_1$", "$p_2$"])
        ax.legend()

        fig.savefig(
            f"{DIR_FIGURES}/fig-illustration-expected-regret{SPEC_DICT[color]['file']}"
        )


def create_ranking_graph_illustrive(df):

    linestyle = ["--", "-.", "-", ":"]

    for color in COLOR_OPTS:
        fig, ax = plt.subplots()
        ax.spines["bottom"].set_color("white")
        ax.spines["left"].set_color("white")
        for i, col in enumerate(df.columns):

            ax.plot(
                df[col].index,
                df[col].values,
                marker="o",
                linestyle=linestyle[i],
                linewidth=3,
                markersize=25,
                color=SPEC_DICT[color]["colors"][i],
                label=df.columns[i],
            )

            # df[col].plot(**kwargs)
            # Flip y-axis.
            ax.axis([-0.1, 2.1, 1.2, -0.2])

            plt.yticks([0, 1], labels=["Rank 1", "Rank 2"], fontsize=14)
            plt.xticks(
                [0, 1, 2],
                labels=df.index.to_list(),
                fontsize=14,
            )
            plt.xlabel("")
            ax.tick_params(axis="both", color="white", pad=20)
            ax.legend(
                markerscale=0.3,
                labelspacing=0.8,
                handlelength=3,
                bbox_to_anchor=(0.45, 1.2),
                loc="upper center",
                ncol=4,
                fontsize=14,
            )
        fig.savefig(f"{DIR_FIGURES}/fig-illustration-ranking{SPEC_DICT[color]['file']}")
