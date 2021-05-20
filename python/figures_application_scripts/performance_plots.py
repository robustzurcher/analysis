import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config import DIR_FIGURES
from global_vals_funcs import COLOR_OPTS
from global_vals_funcs import DICT_POLICIES_4292
from global_vals_funcs import NUM_PERIODS
from global_vals_funcs import OMEGA_GRID
from global_vals_funcs import SIM_RESULTS
from global_vals_funcs import SPEC_DICT
from scipy.signal import savgol_filter

GRIDSIZE = 1000
NUM_POINTS = int(NUM_PERIODS / GRIDSIZE) + 1

################################################################################
#                       Convergence plot
################################################################################


def get_decision_rule_df():
    v_exp_ml = np.full(
        NUM_POINTS, np.dot(DICT_POLICIES_4292[0.0][1], DICT_POLICIES_4292[0.0][0])[0]
    )

    v_disc_ml = pkl.load(open(SIM_RESULTS + "result_ev_0.0_mat_0.95.pkl", "rb"))[1]

    periods = np.arange(0, NUM_PERIODS + GRIDSIZE, GRIDSIZE)

    return pd.DataFrame(
        {"months": periods, "disc_strategy": v_disc_ml, "exp_value": v_exp_ml}
    )


def get_performance_decision_rules():
    print("The underlying transition matrix is the worst case given omega=0.95")

    v_exp_ml = np.full(
        NUM_POINTS, np.dot(DICT_POLICIES_4292[0.0][1], DICT_POLICIES_4292[0.0][0])[0]
    )

    v_disc_ml = pkl.load(open(SIM_RESULTS + "result_ev_0.0_mat_0.95.pkl", "rb"))[1]

    periods = np.arange(0, NUM_PERIODS + GRIDSIZE, GRIDSIZE)
    for color in COLOR_OPTS:
        fig, ax = plt.subplots(1, 1)
        ax.set_ylabel(r"Performance (in thousands)")
        ax.set_xlabel(r"Months (in thousands)")

        # 'Discounted utility of otpimal strategy'
        ax.plot(
            periods,
            v_exp_ml,
            color=SPEC_DICT[color]["colors"][0],
            ls=SPEC_DICT[color]["line"][0],
            label="long-run expectation",
        )
        ax.plot(
            periods,
            v_disc_ml,
            color=SPEC_DICT[color]["colors"][1],
            ls=SPEC_DICT[color]["line"][1],
            label="actual",
        )
        ax.set_ylim([-60_000, 0])
        ax.set_yticks(np.arange(-60_000, 10_000, 10_000))
        ax.set_yticklabels(np.arange(-60, 10, 10))
        ax.set_xticks(np.arange(0, 120_000, 20_000))
        ax.set_xticklabels(np.arange(0, 120, 20))
        plt.xlim(right=100_000)
        ax.legend()
        fig.savefig(
            f"{DIR_FIGURES}/"
            f"fig-application-performance-decision-rules{SPEC_DICT[color]['file']}"
        )


################################################################################
#                             Performance plot
################################################################################


def get_difference_df():

    nominal_costs = _performance_plot(0.0)
    robust_costs_50 = _performance_plot(0.5)
    robust_costs_95 = _performance_plot(0.95)

    diff_costs_95 = nominal_costs - robust_costs_95
    diff_costs_50 = nominal_costs - robust_costs_50

    print("The dataframe contains the difference for as-if - robust strategy.")

    return pd.DataFrame(
        {"omega": OMEGA_GRID, "robust_95": diff_costs_95, "robust_050": diff_costs_50}
    )


def get_difference_plot():

    nominal_costs = _performance_plot(0.0)
    robust_costs_50 = _performance_plot(0.5)
    robust_costs_95 = _performance_plot(0.95)

    diff_costs_95 = nominal_costs - robust_costs_95
    diff_costs_50 = nominal_costs - robust_costs_50
    filter_95 = savgol_filter(diff_costs_95, 29, 3)
    filter_50 = savgol_filter(diff_costs_50, 29, 3)

    for color in COLOR_OPTS:
        fig, ax = plt.subplots(1, 1)

        ax.axhline(
            0,
            0.05,
            1,
            color=SPEC_DICT[color]["colors"][0],
            ls=SPEC_DICT[color]["line"][2],
            label="as-if",
        )

        ax.plot(
            OMEGA_GRID,
            filter_50,
            color=SPEC_DICT[color]["colors"][1],
            label=r"robust $(\omega = 0.50)$",
            ls=SPEC_DICT[color]["line"][1],
        )

        ax.plot(
            OMEGA_GRID,
            filter_95,
            color=SPEC_DICT[color]["colors"][2],
            label=r"robust $(\omega = 0.95)$",
            ls=SPEC_DICT[color]["line"][2],
        )

        ax.set_ylim([-400, 400])
        plt.xlim(left=-0.06, right=1)
        # ax.set_ylim([diff_costs_95[0], diff_costs_95[-1]])
        ax.set_ylabel(r"$\Delta$ Performance")
        ax.set_xlabel(r"$\tilde{\omega}$")
        ax.legend()
        fig.savefig(
            f"{DIR_FIGURES}/fig-application-difference{SPEC_DICT[color]['file']}"
        )


# def get_absolute_plot():
#
#     nominal_costs = _performance_plot(0.0)
#     robust_costs_50 = _performance_plot(0.5)
#     robust_costs_95 = _performance_plot(0.95)
#
#     for color in COLOR_OPTS:
#         fig, ax = plt.subplots(1, 1)
#
#         ax.plot(
#             OMEGA_GRID,
#             nominal_costs,
#             color=SPEC_DICT[color]["colors"][0],
#             label="as-if",
#             ls=SPEC_DICT[color]["line"][0],
#         )
#
#         ax.plot(
#             OMEGA_GRID,
#             robust_costs_50,
#             color=SPEC_DICT[color]["colors"][1],
#             label=r"robust $(\omega = 0.50)$",
#             ls=SPEC_DICT[color]["line"][1],
#         )
#
#         ax.plot(
#             OMEGA_GRID,
#             robust_costs_95,
#             color=SPEC_DICT[color]["colors"][2],
#             label=r"robust $(\omega = 0.95)$",
#             ls=SPEC_DICT[color]["line"][2],
#         )
#         ax.set_ylim([-54000, -47000])
#         # ax.set_ylim([diff_costs_95[0], diff_costs_95[-1]])
#         ax.set_ylabel(r"$\Delta$ Performance")
#         ax.set_xlabel(r"$\omega$")
#         ax.legend()
#         fig.savefig(
#             f"{DIR_FIGURES}/fig-application-difference{SPEC_DICT[color]['file']}"
#         )


def _performance_plot(sim_omega):

    costs = np.zeros(len(OMEGA_GRID))
    for j, omega in enumerate(OMEGA_GRID):
        file = SIM_RESULTS + f"result_ev_{sim_omega}_mat_{omega}.pkl"
        costs[j] = pkl.load(open(file, "rb"))[1][-1]

    return costs
