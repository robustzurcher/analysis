import matplotlib.pyplot as plt
import numpy as np
from config import DIR_FIGURES


def model_nonlinear(x):
    return (x + 0.35 * x) ** 3


def model_linear(x):
    return 1.25 * x + 0.1


def plot_performance():
    fig, ax = plt.subplots()
    grid = np.linspace(0, 1, 100)

    ax.plot(grid, model_linear(grid), label=r"$\pi^*_{(\omega_1, \hat{p})}$")
    ax.plot(grid, model_nonlinear(grid), label=r"$\pi^*_{(\omega_2, \hat{p})}$")

    xticks = [0, 0.25, 0.50, 0.65, 0.85, 1.0]

    ax.set_xticks(xticks)

    ax.set_xticklabels(
        [0, 0.25, r"$\hat{p}$", r"$p^{\omega_1}_0$", r"$p^{\omega_2}_0$", 1.0]
    )
    ax.set_xlim(0, 1)

    ax.axes.yaxis.set_ticklabels([])
    ax.set_ylabel("Performance")
    ax.set_xlabel("$p_0$")
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
