import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config import DIR_FIGURES
from global_vals_funcs import COLOR_OPTS
from global_vals_funcs import SPEC_DICT
from scipy.stats import binom


def get_payoff(action, p):
    "Compute payoff based on p and action p'"
    return 1 - (action - p) ** 2


def take_action(r, n, rule, lambda_=None):
    "action refers here to announce a p estimate"
    if lambda_ is None:
        lambda_ = np.sqrt(n) / (1 + np.sqrt(n))

    if rule == "as-if":
        p_prime = r / n
    elif rule == "robust":
        p_prime = lambda_ * r / n + (1 - lambda_) * 0.5
    elif rule == "fixed":
        p_prime = np.tile(0.5, len(r))
    else:
        raise NotImplementedError

    return p_prime


def risk_function(n, rule, p=None, lambda_=None):
    if p is None:
        p_grid = np.linspace(0, 1, 100)

    else:
        p_grid = np.array([p])

    if lambda_ is None:
        lambda_ = np.sqrt(n) / (1 + np.sqrt(n))

    # rslt = np.tile(np.nan, len(p_grid))
    rslt = []
    for p in p_grid:
        r = np.array(range(n + 1))
        rv = binom(n, p)
        rslt.append(
            np.sum(get_payoff(take_action(r, n, rule, lambda_=lambda_), p) * rv.pmf(r))
        )
    return np.array(rslt)


def create_plot_expected_payoff_func():
    """Plot expected payoff function.

    Focusing on a single point in the state space, we combine the information on
    the sampling distribution for our action with the payoff from each action to
    determine the **expected payoff**.
    """
    matplotlib.rcParams["axes.spines.right"] = True
    for color in COLOR_OPTS:

        p = 0.50
        n = 50
        rv = binom(n, p)
        r = np.arange(binom.ppf(0.01, n, p), binom.ppf(0.99, n, p))

        fig, ax = plt.subplots(1, 1)
        ax2 = ax.twinx()
        series = pd.Series(rv.pmf(r), index=r)
        ax2.plot(
            r,
            get_payoff(take_action(r, n, "as-if"), p),
            label=r"as-if ($\lambda=1$)",
            color=SPEC_DICT[color]["colors"][0],
        )
        ax2.plot(
            r,
            get_payoff(take_action(r, n, "robust", lambda_=0.9), p),
            label=r"robust ($\lambda=0.9$)",
            color=SPEC_DICT[color]["colors"][1],
        )
        ax.bar(
            series.index,
            series,
            alpha=0.2,
            label="PMF",
            color=SPEC_DICT[color]["colors"][0],
        )
        ax.set_xticks(r)
        ax.set_ylim([0, 0.12])
        ax2.set_ylim([0.97, 1])
        ax.set_yticks([0, 0.04, 0.08, 0.12])
        ax2.set_yticks([0.97, 0.98, 0.99, 1])
        ax.set_xlabel("$r$")

        ax.legend(loc="lower left", bbox_to_anchor=(0, -0.3))
        ax2.legend(ncol=2, loc="lower right", bbox_to_anchor=(1, -0.3))

        matplotlib.rc("axes", edgecolor="k")

        ax2.set_ylabel("Payoff")
        ax.set_ylabel("Probability mass")

        fname = (
            f"{DIR_FIGURES}/fig-example-urn-expected-payoff{SPEC_DICT[color]['file']}"
        )

        fig.savefig(fname)
        matplotlib.rcParams["axes.spines.right"] = False


def create_plot_risk_functions():
    """Plot risk functions over state space.

    The **risk function** returns the expected payoff at each point
    in the state space.
    """
    n = 50
    for color in COLOR_OPTS:

        fig, ax = plt.subplots(1, 1)

        ax.plot(
            np.linspace(0, 1, 100),
            risk_function(n, "as-if"),
            label=r"as-if ($\lambda=1$)",
            color=SPEC_DICT[color]["colors"][0],
        )
        ax.plot(
            np.linspace(0, 1, 100),
            risk_function(n, "robust", lambda_=0.9),
            label=r"robust ($\lambda=0.9$)",
            color=SPEC_DICT[color]["colors"][1],
        )
        ax.set_ylabel("Expected payoff")
        ax.set_xlabel("$p$")
        ax.set_ylim(0.9945, 0.9999)
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        t = ax.yaxis.get_offset_text()
        t.set_x(-0.06)
        ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.4))

        fname = (
            f"{DIR_FIGURES}/fig-example-urn-payoff-functions{SPEC_DICT[color]['file']}"
        )

        fig.savefig(fname)


def create_exp_payoff_plot_two_points():
    """Compare expected performance and expected regret at two values of p."""
    for color in COLOR_OPTS:

        n = 50
        df = pd.DataFrame(columns=["as-if", "robust"], index=["p1", "p2"])
        df.loc["p2", "as-if"] = risk_function(n, "as-if", p=0.1)[0]
        df.loc["p2", "robust"] = risk_function(n, "robust", p=0.1, lambda_=0.9)[0]

        df.loc["p1", "as-if"] = risk_function(n, "as-if", p=0.4)[0]
        df.loc["p1", "robust"] = risk_function(n, "robust", p=0.4, lambda_=0.9)[0]

        x = np.arange(2)
        width = 0.35
        fig, ax = plt.subplots()
        ax.bar(
            x - width / 2,
            df.loc[slice(None), "as-if"],
            width,
            label=r"as-if ($\lambda=1$)",
            color=SPEC_DICT[color]["colors"][0],
        )
        ax.bar(
            x + width / 2,
            df.loc[slice(None), "robust"],
            width,
            label=r"robust ($\lambda=0.9$)",
            color=SPEC_DICT[color]["colors"][1],
        )
        ax.set_ylim([0.99, 0.999])
        ax.set_ylabel("Expected payoff")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["$p=0.1$", "$p = 0.4$"])
        ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.3))
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        t = ax.yaxis.get_offset_text()
        t.set_x(-0.06)

        fname = (
            f"{DIR_FIGURES}/fig-example-urn-exp-payoff"
            f"-two-points{SPEC_DICT[color]['file']}"
        )

        fig.savefig(fname)


def create_rank_plot_urn():
    """Create rank plot for urn example."""
    # Insert correct values in data.
    df = pd.DataFrame(
        data=[
            [1, 0],
            [1, 0],
            [0, 1],
        ],
        index=[
            "Maximin",
            "Minimax \n regret",
            "Subjective \n Bayes",
        ],
        columns=[r"as-if ($\lambda=1$)", r"robust ($\lambda=0.9$)"],
    )

    linestyle = ["--", "-."]
    for color in COLOR_OPTS:
        fig, ax = plt.subplots(1, 1)

        ax.spines["bottom"].set_color("white")
        ax.spines["left"].set_color("white")
        ax.spines["right"].set_color("white")
        ax.spines["top"].set_color("white")

        for i, col in enumerate(df.columns):

            ax.plot(
                df[col].index,
                df[col].values,
                marker="o",
                linestyle=linestyle[i],
                linewidth=6,
                markersize=30,
                color=SPEC_DICT[color]["colors"][i],
                label=df.columns[i],
            )

            # Flip y-axis.
            ax.axis([-0.1, 2.1, 1.2, -0.2])

            plt.yticks([0, 1], labels=["Rank 1", "Rank 2"])
            plt.xticks(
                [0, 1, 2],
                labels=df.index.to_list(),
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
            )

        fname = f"{DIR_FIGURES}/fig-example-urn-ranks{SPEC_DICT[color]['file']}"

        fig.savefig(fname)


def create_optimal_lambda_plot():
    """Plot mean/max of risk function for range of lambda."""

    for color in COLOR_OPTS:
        fig, ax = plt.subplots(1, 1)
        n = 50
        num_points = 20

        xmin, xmax = 0.6, 1
        ymin, ymax = 0.945, 1.0

        # Compute yvalues.
        xvals = np.linspace(xmin, xmax, num_points)
        yvals_bayes, yvals_maximin = [], []
        for lambda_ in xvals:
            yvals_bayes.append(risk_function(n, "robust", lambda_=lambda_).mean())
            yvals_maximin.append(risk_function(n, "robust", lambda_=lambda_).min())

        # Plot risk function for range of lambda.
        if color == "colored":
            color_id = 1
        else:
            color_id = 4
        ax.plot(
            xvals,
            yvals_bayes,
            label="Subjective Bayes",
            color=SPEC_DICT[color]["colors"][color_id],
        )
        ax.plot(
            xvals,
            yvals_maximin,
            label="Maximin",
            color=SPEC_DICT[color]["colors"][color_id],
            linestyle=":",
        )

        # Plot vertical lines for optimal lambda.
        lambda_bayes = pd.Series(yvals_bayes, index=xvals).idxmax()
        lambda_maximin = pd.Series(yvals_maximin, index=xvals).idxmax()
        plt.vlines(
            [lambda_bayes, lambda_maximin],
            ymin=ymin,
            ymax=[max(yvals_bayes), max(yvals_maximin)],
            linestyle="dashed",
            color="k",
        )

        # Set xticks and labels including optimal lambdas.
        ax.set_xticks([0.6, 0.7, 0.8, lambda_maximin, lambda_bayes, 1])
        ax.set_xticklabels(
            [0.6, 0.7, 0.8, r"$\lambda^*_{Maximin}$", r"$\lambda^*_{Bayes}$", 1.0]
        )

        # Axis limits
        ax.set_xlabel(r"$\lambda$")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.4))
        ax.set_ylabel("Performance measure")
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)

        fname = f"{DIR_FIGURES}/fig-example-urn-optimal{SPEC_DICT[color]['file']}"

        fig.savefig(fname)
