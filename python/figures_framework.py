import matplotlib.pyplot as plt
import numpy as np
from config import DIR_FIGURES



def model_robust(x):
    return -(x - 0.6)**2 - 0.5

def model_optimal(x):
    return -(x*2)**2


def plot_performance():
    fig, ax = plt.subplots()
    grid = np.linspace(-1, 1, 100)

    ax.plot(grid, model_robust(grid), label=r"robust")
    ax.plot(grid, model_optimal(grid), label=r"optimal")

    xticks = [-1,  0.0, 0.60, 1.0]

    ax.set_xticks(xticks)

    ax.set_xticklabels(
            [0,  r"$p_0$", r"$p^{\omega}_0$", 1.0]
        )
    ax.set_xlim(-1, 1)

    ax.axes.yaxis.set_ticklabels([])
    ax.set_ylabel("Performance")
    ax.set_xlabel("$\hat{p}$")
    ax.legend()


    fig.savefig(f"{DIR_FIGURES}/fig-illustration-performance")


def create_ranking_graph_illustrive(df):
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["tab:blue", "tab:orange", "tab:red", "tab:purple"]
    linestyle = ["--", "-.", "-", ":"]

    ax.spines["bottom"].set_color("white")
    ax.spines["left"].set_color("white")
    for i, col in enumerate(df.columns):
        kwargs = {
            "marker": "o",
            "linestyle": linestyle[i],
            "linewidth": 3,
            "markersize": 35,
            "color": colors[i],
        }

        df[col].plot(**kwargs)
        # Flip y-axis.
        plt.axis([-0.1, 3.1, 1.2, -0.2])

        plt.yticks([0, 1], labels=["Rank 1", "Rank 2"], fontsize=14)
        plt.xticks(
            [0, 1, 2],
            labels=["Subjective \n Bayes", "Minimax \n regret", "Maximin"],
            fontsize=14,
        )
        plt.xlabel("")
        plt.tick_params(axis="both", color="white", pad=20)
        plt.legend(
            markerscale=0.3,
            labelspacing=0.8,
            handlelength=3,
            bbox_to_anchor=(0.45, 1.2),
            loc="upper center",
            ncol=4,
            fontsize=14,
        )
    fig.savefig(f"{DIR_FIGURES}/fig-illustration-ranking")
