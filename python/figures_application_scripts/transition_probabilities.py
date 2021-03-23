import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config import DIR_FIGURES
from global_vals_funcs import BIN_SIZE
from global_vals_funcs import COLOR_OPTS
from global_vals_funcs import DICT_POLICIES_2223
from global_vals_funcs import DICT_POLICIES_4292
from global_vals_funcs import SPEC_DICT


p_size = 3
width = 0.8 * BIN_SIZE
p_raw = np.loadtxt("../pre_processed_data/parameters/rust_trans_raw.txt")
hesse_inv_raw = np.loadtxt("../pre_processed_data/parameters/rust_cov_raw.txt")
x = np.arange(1, p_size + 1) * BIN_SIZE


def get_probabilities(state):
    p_ml = DICT_POLICIES_4292[0.0][1][state, state : state + p_size]

    for color in COLOR_OPTS:
        fig, ax = plt.subplots(1, 1)

        ax.bar(
            x,
            p_ml,
            width,
            color=SPEC_DICT[color]["colors"][0],
            ls=SPEC_DICT[color]["line"][0],
        )

        ax.set_ylabel(r"Transition probability")
        ax.set_xlabel(r"Mileage increase (in thousands)")
        plt.xticks(x)
        ax.set_ylim([0.00, 0.80])
        fig.savefig(
            f"{DIR_FIGURES}/fig-application-probabilities{SPEC_DICT[color]['file']}"
        )


def get_probabilities_bar(state):

    p_ml = DICT_POLICIES_4292[0.0][1][state, state : state + p_size]
    std_err = _get_standard_errors(p_ml, p_raw, hesse_inv_raw)
    capsize = 15

    for color in COLOR_OPTS:
        fig, ax = plt.subplots(1, 1)

        ax.bar(
            x,
            p_ml,
            width,
            yerr=std_err,
            capsize=capsize,
            color=SPEC_DICT[color]["colors"][0],
            ls=SPEC_DICT[color]["line"][0],
        )

        ax.set_ylabel(r"Transition probability")
        ax.set_ylim([0.00, 0.80])

        ax.set_xlabel(r"Mileage increase (in thousands)")
        plt.xticks(x)
        fig.savefig(
            f"{DIR_FIGURES}/fig-application-probabilities-bar{SPEC_DICT[color]['file']}"
        )


def df_probability_shift(state):
    return pd.DataFrame(
        {
            "0": DICT_POLICIES_2223[0.0][1][state, state : state + p_size],
            "4292_0.50": DICT_POLICIES_4292[0.5][1][state, state : state + p_size],
            "4292_0.95": DICT_POLICIES_4292[0.95][1][state, state : state + p_size],
            "2223_0.95": DICT_POLICIES_2223[0.95][1][state, state : state + p_size],
        }
    )


def get_probability_shift(state):

    width = 0.25 * BIN_SIZE

    for color in COLOR_OPTS:
        fig, ax = plt.subplots(1, 1)

        ax.bar(
            x - width,
            DICT_POLICIES_4292[0.0][1][state, state : state + p_size],
            width,
            color=SPEC_DICT[color]["colors"][0],
            hatch=SPEC_DICT[color]["hatch"][0],
            label="reference",
        )
        ax.bar(
            x,
            DICT_POLICIES_4292[0.50][1][state, state : state + p_size],
            width,
            color=SPEC_DICT[color]["colors"][1],
            hatch=SPEC_DICT[color]["hatch"][1],
            label=r"$\omega=0.50$",
        )
        ax.bar(
            x + width,
            DICT_POLICIES_4292[0.95][1][state, state : state + p_size],
            width,
            color=SPEC_DICT[color]["colors"][2],
            hatch=SPEC_DICT[color]["hatch"][2],
            label=r"$\omega=0.95$",
        )

        ax.set_ylabel(r"Transition probability")
        ax.set_xlabel(r"Mileage increase (in thousands)")
        ax.set_ylim([0.0, 0.8])
        plt.xticks(x)
        ax.legend()

        fig.savefig(
            f"{DIR_FIGURES}/fig-application-probability-shift-omega{SPEC_DICT[color]['file']}"
        )


def get_probability_shift_data(state):

    width = 0.25 * BIN_SIZE
    for color in COLOR_OPTS:
        fig, ax = plt.subplots(1, 1)

        ax.bar(
            x - width,
            DICT_POLICIES_4292[0.0][1][state, state : state + p_size],
            width,
            color=SPEC_DICT[color]["colors"][0],
            hatch=SPEC_DICT[color]["hatch"][0],
            label="reference",
        )
        ax.bar(
            x,
            DICT_POLICIES_4292[0.95][1][state, state : state + p_size],
            width,
            color=SPEC_DICT[color]["colors"][1],
            hatch=SPEC_DICT[color]["hatch"][1],
            label="$N_k = 55$",
        )
        ax.bar(
            x + width,
            DICT_POLICIES_2223[0.95][1][state, state : state + p_size],
            width,
            color=SPEC_DICT[color]["colors"][2],
            hatch=SPEC_DICT[color]["hatch"][2],
            label="$N_k = 29$",
        )

        ax.set_ylabel(r"Transition probability")
        ax.set_xlabel(r"Mileage increase (in thousands)")
        plt.xticks(x)
        ax.set_ylim([0.0, 0.8])

        ax.legend()

        fig.savefig(
            f"{DIR_FIGURES}/fig-application-probability-shift-data{SPEC_DICT[color]['file']}"
        )


def _get_standard_errors(p, p_raw, hesse_inv_raw):
    runs = 1000
    draws = np.zeros((runs, len(p_raw)), dtype=np.float)
    for i in range(runs):
        draws[i, :] = draw_from_raw(p_raw, hesse_inv_raw)
    std_err = np.zeros((2, len(p_raw)), dtype=float)
    for i in range(len(p_raw)):
        std_err[0, i] = p[i] - np.percentile(draws[:, i], 2.5)
        std_err[1, i] = np.percentile(draws[:, i], 97.5) - p[i]
    return std_err


def draw_from_raw(p_raw, hesse_inv_raw):
    draw = np.random.multivariate_normal(p_raw, hesse_inv_raw)
    return np.exp(draw) / np.sum(np.exp(draw))
