import matplotlib.pyplot as plt
import numpy as np
from config import DIR_FIGURES
from global_vals_funcs import COLOR_OPTS
from global_vals_funcs import SPEC_DICT


def model_robust(x):
    return -((x - 0.6) ** 2) - 0.5


def model_optimal(x):
    return -((x * 2) ** 2)


def plot_performance():
    grid = np.linspace(-1, 1, 100)

    for color in COLOR_OPTS:
        fig, ax = plt.subplots()

        ax.plot(
            grid,
            model_robust(grid),
            color=SPEC_DICT[color]["colors"][0],
            ls=SPEC_DICT[color]["line"][0],
            label=r"robust",
        )
        ax.plot(
            grid,
            model_optimal(grid),
            color=SPEC_DICT[color]["colors"][1],
            ls=SPEC_DICT[color]["line"][1],
            label=r"optimal",
        )

        xticks = [-1, 0.0, 0.60, 1.0]

        ax.set_xticks(xticks)

        ax.set_xticklabels([0, r"$p_0$", r"$p^{\omega}_0$", 1.0])
        ax.set_xlim(-1, 1)

        ax.axes.yaxis.set_ticklabels([])
        ax.set_ylabel("Performance")
        ax.set_xlabel(r"$\hat{p}$")
        ax.legend()

        fig.savefig(
            f"{DIR_FIGURES}/fig-illustration-performance{SPEC_DICT[color]['file']}"
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
                labels=["Maximin", "Minimax \n regret", "Subjective \n Bayes"],
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
