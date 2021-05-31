from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config import DIR_FIGURES
from global_vals_funcs import COLOR_OPTS
from global_vals_funcs import SPEC_DICT
from scipy.signal import savgol_filter
from scipy.stats import binom
from scipy.stats import chi2


def get_loss(action, p):
    "This is the loss function"
    return (action - p) ** 2


def root_kullback(q_0, rho, p_0):
    return p_0 * np.log(p_0 / q_0) + (1 - p_0) * np.log((1 - p_0) / (1 - q_0)) - rho


def get_worst_case_prob(q_0, n, gridsize, omega):
    rho = chi2.ppf(omega, 1) / (2 * n)
    root_func = partial(root_kullback, q_0, rho)
    grid = np.arange(np.finfo(float).eps, 1, gridsize)
    vals_grid = root_func(grid)
    # Get grid inside set
    set_grid = grid[vals_grid < 0]
    perf_grid = np.empty_like(set_grid)
    for i, val in enumerate(set_grid):
        perf_grid[i] = np.max((val - set_grid) ** 2)
    return set_grid[np.argmin(perf_grid)]


def risk_function(n, rule):
    p = np.linspace(0, 1, 1000)

    if rule == "as-if":
        return (p * (1 - p)) / n
    elif rule == "fixed":
        return 0.25 - p * (1 - p)
    else:
        raise NotImplementedError


def get_worsts_on_sample(n, gridsize_sets, omega):
    sample_grid = np.linspace(0, 1, n + 1)
    worst_on_sample = pd.Series(index=sample_grid, dtype=float)
    worst_on_sample[0] = sample_grid[0]
    worst_on_sample[-1] = sample_grid[-1]
    for sample in sample_grid[1:-1]:
        worst_on_sample[sample] = get_worst_case_prob(sample, n, gridsize_sets, omega)
    return worst_on_sample


def get_worst_from_samples(worst_realizations, samples):
    return pd.Series(samples).replace(worst_realizations).to_numpy()


def get_loss_plot_with_bars(n, p, worst_on_sample):
    rv = binom(n, p)
    r = np.arange(binom.ppf(0.01, n, p), binom.ppf(0.99, n, p))

    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    # pd.Series(rv.pmf(r), index=r).plot(kind="bar", ax=ax2)
    # ax2.vlines(r, 0, rv.pmf(r), colors="black", alpha=0.9)
    ax2.bar(r.astype(int), rv.pmf(r), alpha=0.1, label="sampling distribution")
    ax2.set_ylim([0, 0.2])
    ax2.set_yticks([])

    ax.plot(r.astype(int), get_loss(r / n, p), label="as-if")
    ax.plot(
        r.astype(int),
        get_loss(get_worst_from_samples(worst_on_sample, r / n), p),
        label="robust",
    )
    ax.set_ylabel("Loss")
    ax.set_xlabel("Draws")
    ax.set_xticks(np.arange(r[0], r[-1], 2).astype(int))
    ax.legend()
    fig.savefig(f"{DIR_FIGURES}/fig-illustration-expected-loss")


def expected_loss_df(n, p_1, p_2, worst_on_sample):
    rv_1 = binom(n, p_1)
    rv_2 = binom(n, p_2)

    r_1 = np.arange(binom.ppf(0.01, n, p_1), binom.ppf(0.99, n, p_1))
    r_2 = np.arange(binom.ppf(0.01, n, p_2), binom.ppf(0.99, n, p_2))

    df = pd.DataFrame(None, columns=[p_1, p_2], index=["robust", "as-if"])
    df.index.names = ["Strategy"]

    df.loc["as-if", p_1] = np.dot(rv_1.pmf(r_1), get_loss(r_1 / n, p_1))
    df.loc["as-if", p_2] = np.dot(rv_2.pmf(r_2), get_loss(r_2 / n, p_2))

    df.loc["robust", p_1] = np.dot(
        rv_1.pmf(r_1), get_loss(get_worst_from_samples(worst_on_sample, r_1 / n), p_1)
    )
    df.loc["robust", p_2] = np.dot(
        rv_2.pmf(r_2), get_loss(get_worst_from_samples(worst_on_sample, r_2 / n), p_2)
    )
    return df


def plot_expected_loss_df(df):
    x = np.arange(2)
    width = 0.35
    fig, ax = plt.subplots()
    ax.bar(
        x - width / 2,
        df.loc["as-if"],
        width,
        label="as-if",
    )
    ax.bar(
        x + width / 2,
        df.loc["robust"],
        width,
        label="robust",
    )
    ax.set_ylabel("Expected loss")
    # ax.set_ylim([0, 0.2])
    # ax.set_yticks(np.arange(0, 0.25, 0.05))
    ax.set_xticks([0, 1])
    columns = df.columns.values
    ax.set_xticklabels([f"{columns[0]}", f"{columns[1]}"])
    ax.legend()
    fig.savefig(f"{DIR_FIGURES}/fig-illustration-bars-expected-loss")


def get_losses(prob_grid, num_experiments, n, worst_on_sample):
    prob_to_consider = 1 / num_experiments
    exp_losses_robust = np.empty_like(prob_grid)
    exp_losses_as_if = np.empty_like(prob_grid)
    for i, prob in enumerate(prob_grid):
        r = np.arange(
            binom.ppf(prob_to_consider, n, prob),
            binom.ppf(1 - prob_to_consider, n, prob),
        )
        rv = binom(n, prob)
        losses_as_if = get_loss(r / n, prob)
        exp_losses_as_if[i] = np.dot(rv.pmf(r), losses_as_if)
        losses_robust = get_loss(get_worst_from_samples(worst_on_sample, r / n), prob)
        exp_losses_robust[i] = np.dot(rv.pmf(r), losses_robust)
    return exp_losses_robust, exp_losses_as_if


def plot_risk_function(prob_grid, exp_losses_robust, exp_losses_as_if):
    filter_exp_loss_as_if = savgol_filter(exp_losses_as_if, 29, 3)
    filter_exp_loss_robust = savgol_filter(exp_losses_robust, 29, 3)
    fig, ax = plt.subplots()
    ax.plot(prob_grid, filter_exp_loss_as_if, label="as-if")
    ax.plot(prob_grid, filter_exp_loss_robust, label="robust")
    # ax.plot(np.linspace(0, 1, 1000), risk_function(n, "as-if"), label="as-if")
    ax.set_ylabel("Expected loss")
    ax.set_xlabel("p")
    ax.legend()
    fig.savefig(f"{DIR_FIGURES}/fig-illustration-risk-function")


def create_ranking_df(exp_losses_robust, exp_losses_as_if):
    df = pd.DataFrame(
        columns=["as-if", "robust"],
        index=["Minimax \n regret", "Subjective \n Bayes"],
        data=np.ones((2, 2)),
    )
    if np.mean(exp_losses_robust) > np.mean(exp_losses_as_if):
        df.loc["Subjective \n Bayes", "as-if"] = 0
    else:
        df.loc["Subjective \n Bayes", "robust"] = 0

    if np.max(exp_losses_robust) > np.max(exp_losses_as_if):
        df.loc["Minimax \n regret", "as-if"] = 0
    else:
        df.loc["Minimax \n regret", "robust"] = 0

    return df


def create_ranking(df):

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
            ax.axis([-0.1, 1.1, 1.2, -0.2])

            plt.yticks([0, 1], labels=["Rank 1", "Rank 2"], fontsize=14)
            plt.xticks(
                [0, 1],
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
        fig.savefig(
            f"{DIR_FIGURES}/fig-illustration-urn-rankin" f"g{SPEC_DICT[color]['file']}"
        )
