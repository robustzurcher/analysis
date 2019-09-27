import pickle as pkl
import glob
import pandas as pd
from zipfile import ZipFile
import os
import shutil
from scipy.signal import savgol_filter

import matplotlib.pyplot as plt
import numpy as np
from ruspy.simulation.simulation import simulate

from auxiliary import get_file
from ruspy.estimation.estimation_cost_parameters import lin_cost, cost_func, choice_prob
from config import DIR_FIGURES

# Global variables
BETA = 0.9999
PARAMS = np.array([50, 400])
NUM_BUSES = 200
BIN_SIZE = 5  # in thousand
NUM_PERIODS = 100000
GRIDSIZE = 1000
NUM_POINTS = int(NUM_PERIODS / GRIDSIZE) + 1
FIXP_DICT_4292 = "../pre_processed_data/fixp_results_5000_50_400_4292.pkl"
FIXP_DICT_2223 = "../pre_processed_data/fixp_results_5000_50_400_2223.pkl"
SIM_RESULTS = "../pre_processed_data/sim_results/"
VAL_RESULTS = "../pre_processed_data/val_results/"
color_opts = ["colored", "black_white"]
spec_dict = {
    "colored": {"colors": [None] * 4, "line": ["-"] * 3, "hatch": [""] * 3, "file": ""},
    "black_white": {
        "colors": ["#808080", "#d3d3d3", "#d3d3d3", "#d3d3d3"],
        "line": ["-", "--", ":"],
        "hatch": ["", "OOO", "///"],
        "file": "-sw",
    },
}


def extract_zips():
    if os.path.exists(SIM_RESULTS):
        shutil.rmtree(SIM_RESULTS)
    os.makedirs("../pre_processed_data/sim_results")
    ZipFile("../pre_processed_data/simulation_results.zip").extractall(SIM_RESULTS)

    if os.path.exists(VAL_RESULTS):
        shutil.rmtree(VAL_RESULTS)
    ZipFile("../pre_processed_data/validation_results.zip").extractall(VAL_RESULTS)


################################################################################
#                           Probabilities
################################################################################

p_size = 3
x = np.arange(1, p_size + 1) * BIN_SIZE
dict_policies = get_file(FIXP_DICT_4292)
width = 0.8 * BIN_SIZE
p_raw = np.loadtxt("../pre_processed_data/parameters/rust_trans_raw.txt")
hesse_inv_raw = np.loadtxt("../pre_processed_data/parameters/rust_cov_raw.txt")


def get_probabilities(state):
    p_ml = dict_policies[0.0][1][state, state : state + p_size]

    for color in color_opts:
        fig, ax = plt.subplots(1, 1)

        ax.bar(
            x,
            p_ml,
            width,
            color=spec_dict[color]["colors"][0],
            ls=spec_dict[color]["line"][0],
        )

        ax.set_ylabel(r"Probability")
        ax.set_xlabel(r"Mileage increase (in thousands)")
        plt.xticks(x)
        ax.set_ylim([0.00, 0.80])
        fig.savefig(
            f"{DIR_FIGURES}/fig-application-probabilities{spec_dict[color]['file']}"
        )


def get_probabilities_bar(state):

    p_ml = dict_policies[0.0][1][state, state : state + p_size]
    std_err = _get_standard_errors(p_ml, p_raw, hesse_inv_raw)
    capsize = 15

    for color in color_opts:
        fig, ax = plt.subplots(1, 1)

        ax.bar(
            x,
            p_ml,
            width,
            yerr=std_err,
            capsize=capsize,
            color=spec_dict[color]["colors"][0],
            ls=spec_dict[color]["line"][0],
        )

        ax.set_ylabel(r"Probability")
        ax.set_ylim([0.00, 0.80])

        ax.set_xlabel(r"Mileage increase (in thousands)")
        plt.xticks(x)
        fig.savefig(
            f"{DIR_FIGURES}/fig-application-probabilities-bar{spec_dict[color]['file']}"
        )


def df_probability_shift(state):
    dict_policies_4292 = get_file(FIXP_DICT_4292)
    dict_policies_2223 = get_file(FIXP_DICT_2223)
    return pd.DataFrame(
        {
            "0": dict_policies_4292[0.0][1][state, state : state + p_size],
            "4292_0.50": dict_policies_4292[0.5][1][state, state : state + p_size],
            "4292_0.95": dict_policies_4292[0.95][1][state, state : state + p_size],
            "2223_0.95": dict_policies_2223[0.95][1][state, state : state + p_size],
        }
    )


def get_probability_shift(state):

    width = 0.25 * BIN_SIZE

    for color in color_opts:
        fig, ax = plt.subplots(1, 1)

        ax.bar(
            x - width,
            dict_policies[0.0][1][state, state : state + p_size],
            width,
            color=spec_dict[color]["colors"][0],
            hatch=spec_dict[color]["hatch"][0],
            label="reference",
        )
        ax.bar(
            x,
            dict_policies[0.50][1][state, state : state + p_size],
            width,
            color=spec_dict[color]["colors"][1],
            hatch=spec_dict[color]["hatch"][1],
            label="$\omega=0.50$",
        )
        ax.bar(
            x + width,
            dict_policies[0.95][1][state, state : state + p_size],
            width,
            color=spec_dict[color]["colors"][2],
            hatch=spec_dict[color]["hatch"][2],
            label="$\omega=0.95$",
        )

        ax.set_ylabel(r"Probability")
        ax.set_xlabel(r"Mileage increase (in thousands)")
        ax.set_ylim([0.0, 0.8])
        plt.xticks(x)
        ax.legend()

        fig.savefig(
            f"{DIR_FIGURES}/fig-application-probability-shift-omega{spec_dict[color]['file']}"
        )


def get_probability_shift_data(state):

    width = 0.25 * BIN_SIZE
    dict_policies_4292 = get_file(FIXP_DICT_4292)
    dict_policies_2223 = get_file(FIXP_DICT_2223)

    for color in color_opts:
        fig, ax = plt.subplots(1, 1)

        ax.bar(
            x - width,
            dict_policies_4292[0.0][1][state, state : state + p_size],
            width,
            color=spec_dict[color]["colors"][0],
            hatch=spec_dict[color]["hatch"][0],
            label="reference",
        )
        ax.bar(
            x,
            dict_policies_4292[0.95][1][state, state : state + p_size],
            width,
            color=spec_dict[color]["colors"][1],
            hatch=spec_dict[color]["hatch"][1],
            label="$N_k = 55$",
        )
        ax.bar(
            x + width,
            dict_policies_2223[0.95][1][state, state : state + p_size],
            width,
            color=spec_dict[color]["colors"][2],
            hatch=spec_dict[color]["hatch"][2],
            label="$N_k = 29$",
        )

        ax.set_ylabel(r"Probability")
        ax.set_xlabel(r"Mileage increase (in thousands)")
        plt.xticks(x)
        ax.set_ylim([0.0, 0.8])

        ax.legend()

        fig.savefig(
            f"{DIR_FIGURES}/fig-application-probability-shift-data{spec_dict[color]['file']}"
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


################################################################################
#                       Replacement/Maintenance Probabilities
################################################################################
keys = [0.0, 0.5, 0.95]
max_state = 30


def df_maintenance_probabilties():
    choice_ml, choices = _create_repl_prob_plot(FIXP_DICT_4292, keys)
    states = np.arange(choice_ml.shape[0]) * BIN_SIZE
    return pd.DataFrame(
        {
            "milage_thousands": states,
            0.0: choice_ml[:, 0],
            keys[1]: choices[0][:, 0],
            keys[2]: choices[1][:, 0],
        }
    )


def get_maintenance_probabilities():

    choice_ml, choices = _create_repl_prob_plot(FIXP_DICT_4292, keys)
    states = np.arange(max_state) * BIN_SIZE
    for color in color_opts:
        fig, ax = plt.subplots(1, 1)

        ax.plot(
            states,
            choice_ml[:max_state, 0],
            color=spec_dict[color]["colors"][0],
            ls=spec_dict[color]["line"][0],
            label="optimal",
        )
        for i, choice in enumerate(choices):
            ax.plot(
                states,
                choice[:max_state, 0],
                color=spec_dict[color]["colors"][i + 1],
                ls=spec_dict[color]["line"][i + 1],
                label=f"robust $(\omega = {keys[i+1]:.2f})$",
            )

        ax.set_ylabel(r"Maintenance probability")
        ax.set_xlabel(r"Mileage (in thousands)")
        ax.set_ylim([0, 1])

        plt.xticks(states[::5])
        ax.legend()

        fig.savefig(
            f"{DIR_FIGURES}/fig-application-maintenance-probabilities"
            f"{spec_dict[color]['file']}"
        )


def _create_repl_prob_plot(file, keys):
    dict_policies = get_file(file)
    ev_ml = dict_policies[0.0][0]
    num_states = ev_ml.shape[0]
    costs = cost_func(num_states, lin_cost, PARAMS)
    choice_ml = choice_prob(ev_ml, costs, BETA)
    choices = []
    for omega in keys[1:]:
        choices += [choice_prob(dict_policies[omega][0], costs, BETA)]
    return choice_ml, choices


################################################################################
#                       Demonstration
################################################################################


def get_demonstration_df(init_dict):
    states, periods = get_demonstration_data(init_dict)
    return pd.DataFrame({'months_ml': periods[0], 'months_rob': periods[1],
                         "opt_mileage": states[0] * BIN_SIZE,
                         "rob_mileage": states[1] * BIN_SIZE})


def get_demonstration(df, max_period):
    states = (df["opt_mileage"], df["rob_mileage"])
    periods = (df["months_ml"], df["months_rob"])
    labels = ["optimal", "robust ($\omega = 0.95$)"]
    for color in color_opts:
        fig, ax = plt.subplots(1, 1)
        ax.set_xlabel(r"Months")
        ax.set_ylabel(r"Mileage (in thousands)")

        for i, state in enumerate(states):
            ax.plot(
                periods[i][:max_period],
                state[:max_period],
                color=spec_dict[color]["colors"][i],
                ls=spec_dict[color]["line"][i],
                label=labels[i],
            )
        ax.legend(loc="upper left")
        ax.set_ylim([0, 90])

        fig.savefig(
            f"{DIR_FIGURES}/fig-application-demonstration{spec_dict[color]['file']}"
        )


def get_demonstration_data(init_dict):

    dict_policies = get_file("../pre_processed_data/fixp_results_5000_50_400_4292.pkl")
    ev_ml = dict_policies[0.0][0]
    ev_95 = dict_policies[0.95][0]
    trans_mat = dict_policies[0.0][1]

    df_ml = simulate(init_dict, ev_ml, trans_mat)
    df_95 = simulate(init_dict, ev_95, trans_mat)

    periods_ml = np.array(df_ml["period"], dtype=int)
    periods_95 = np.array(df_95["period"], dtype=int)
    periods = [periods_ml, periods_95]
    states_ml = np.array(df_ml["state"], dtype=int)
    states_95 = np.array(df_95["state"], dtype=int)
    states = [states_ml, states_95]

    for i, df in enumerate([df_ml, df_95]):
        index = np.array(df[df["decision"] == 1].index, dtype=int) + 1
        states[i] = np.insert(states[i], index, 0)
        periods[i] = np.insert(periods[i], index, index - 1)

    return states, periods




################################################################################
#                       Threshold plot
################################################################################

num_keys = 100


def df_thresholds():
    means_discrete = _threshold_data()
    omega_range = np.linspace(0, 0.99, num_keys)
    return pd.DataFrame({"omega": omega_range, "threshold": means_discrete})


def get_replacement_thresholds():

    means_discrete = _threshold_data() * BIN_SIZE

    omega_range = np.linspace(0, 0.99, num_keys)
    means_ml = np.full(len(omega_range), np.round(means_discrete[0])).astype(int)

    omega_sections, state_sections = _create_sections(means_discrete, omega_range)

    y_0 = state_sections[0][0] - 2 * BIN_SIZE
    y_1 = state_sections[-1][-1] + 2 * BIN_SIZE
    for color in color_opts:
        fig, ax = plt.subplots(1, 1)
        ax.set_ylabel(r"Mileage (in thousands)")
        ax.set_xlabel(r"$\omega$")
        ax.set_ylim([y_0, y_1])

        ax.plot(
            omega_range,
            means_ml,
            color=spec_dict[color]["colors"][0],
            ls=spec_dict[color]["line"][0],
            label="optimal",
        )
        if color == "colored":
            second_color = "#ff7f0e"
        else:
            second_color = spec_dict[color]["colors"][2]
        for j, i in enumerate(omega_sections[:-1]):
            ax.plot(
                i, state_sections[j], color=second_color, ls=spec_dict[color]["line"][1]
            )
        ax.plot(
            omega_sections[-1],
            state_sections[-1],
            color=second_color,
            ls=spec_dict[color]["line"][1],
            label="robust",
        )
        ax.legend()

        fig.savefig(
            f"{DIR_FIGURES}/fig-application-replacement-thresholds{spec_dict[color]['file']}"
        )


def _threshold_data():
    file_list = sorted(glob.glob(SIM_RESULTS + "result_ev_*_mat_0.00.pkl"))
    if len(file_list) != 0:
        means_robust_strat = np.array([])
        for file in file_list:
            mean = pkl.load(open(file, "rb"))[0]
            means_robust_strat = np.append(means_robust_strat, mean)
    else:
        raise AssertionError("Need to unpack simulation files")

    means_discrete = np.around(means_robust_strat).astype(int)
    return means_discrete


def _create_sections(mean_disc, om_range):
    omega_sections = []
    state_sections = []
    for j, i in enumerate(np.unique(mean_disc)):
        where = mean_disc == i
        max_ind = np.max(np.where(mean_disc == i))
        if j == 0:
            med_val = (np.max(om_range[where]) + np.min(om_range[~where])) / 2
            omega_sections += [np.append(om_range[where], med_val)]
            state_sections += [np.append(mean_disc[where], i)]
        elif j == (len(np.unique(mean_disc)) - 1):
            med_val = (np.min(om_range[where]) + np.max(om_range[~where])) / 2
            omega_sections += [np.array([med_val] + om_range[where].tolist())]
            state_sections += [np.array([i] + mean_disc[where].tolist())]
        else:
            low = (np.min(om_range[where]) + np.max(omega_sections[-1][:-1])) / 2
            high = (np.max(om_range[where]) + np.min(om_range[max_ind + 1])) / 2
            omega_sections += [np.array([low] + om_range[where].tolist() + [high])]
            state_sections += [np.array([i] + mean_disc[where].tolist() + [i])]
    return omega_sections, state_sections


################################################################################
#                       Convergence plot
################################################################################


def get_decision_rule_df():
    dict_policies = get_file(FIXP_DICT_4292)

    v_exp_ml = np.full(NUM_POINTS, dict_policies[0.0][0][0])

    v_disc_ml = pkl.load(open(SIM_RESULTS + "result_ev_0.00_mat_0.95.pkl", "rb"))[1]

    periods = np.arange(0, NUM_PERIODS + GRIDSIZE, GRIDSIZE)

    return pd.DataFrame(
        {"months": periods, "disc_strategy": v_disc_ml, "exp_value": v_exp_ml}
    )


def get_performance_decision_rules():
    print("The underlying transition matrix is the worst case given omega=0.95")
    dict_policies = get_file(FIXP_DICT_4292)

    v_exp_ml = np.full(NUM_POINTS, dict_policies[0.0][0][0])

    v_disc_ml = pkl.load(open(SIM_RESULTS + "result_ev_0.00_mat_0.95.pkl", "rb"))[1]

    periods = np.arange(0, NUM_PERIODS + GRIDSIZE, GRIDSIZE)
    for color in color_opts:
        fig, ax = plt.subplots(1, 1)
        ax.set_ylim([1.1 * v_disc_ml[-1], 0])
        ax.set_ylabel(r"Performance")
        ax.set_xlabel(r"Months")

        formatter = plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x)))
        ax.get_xaxis().set_major_formatter(formatter)
        ax.get_yaxis().set_major_formatter(formatter)

        # 'Discounted utility of otpimal strategy'
        ax.plot(
            periods,
            v_exp_ml,
            color=spec_dict[color]["colors"][0],
            ls=spec_dict[color]["line"][0],
            label="long-run expectation",
        )
        ax.plot(
            periods,
            v_disc_ml,
            color=spec_dict[color]["colors"][1],
            ls=spec_dict[color]["line"][1],
            label="actual",
        )
        ax.set_ylim([-60000, 0])
        ax.legend()
        fig.savefig(
            f"{DIR_FIGURES}/fig-application-performance-decision-rules{spec_dict[color]['file']}"
        )


################################################################################
#                             Performance plot
################################################################################


def get_difference_df():
    num_keys = 100

    omega_range = np.linspace(0, 0.99, num_keys)

    nominal_costs, robust_costs_95 = _performance_plot()

    file_list = sorted(glob.glob(SIM_RESULTS + "result_ev_0.50_mat_*.pkl"))
    robust_costs_50 = np.zeros(len(file_list))
    for j, file in enumerate(file_list):
        robust_costs_50[j] = pkl.load(open(file, "rb"))[1][-1]

    diff_costs_95 = robust_costs_95 - nominal_costs
    diff_costs_50 = robust_costs_50 - nominal_costs

    print("The dataframe contains the difference for robust - nominal strategy.")

    return pd.DataFrame(
        {"omega": omega_range, "robust_95": diff_costs_95, "robust_050": diff_costs_50}
    )


def get_difference_plot():

    num_keys = 100

    omega_range = np.linspace(0, 0.99, num_keys)

    nominal_costs, robust_costs_95 = _performance_plot()

    file_list = sorted(glob.glob(SIM_RESULTS + "result_ev_0.50_mat_*.pkl"))
    robust_costs_50 = np.zeros(len(file_list))
    for j, file in enumerate(file_list):
        robust_costs_50[j] = pkl.load(open(file, "rb"))[1][-1]

    diff_costs_95 = robust_costs_95 - nominal_costs
    diff_costs_50 = robust_costs_50 - nominal_costs
    filter_95 = savgol_filter(diff_costs_95, 29, 3)
    filter_50 = savgol_filter(diff_costs_50, 29, 3)

    for color in color_opts:
        fig, ax = plt.subplots(1, 1)

        ax.plot(
            omega_range,
            filter_95,
            color=spec_dict[color]["colors"][0],
            label="robust $(\omega = 0.95)$",
            ls=spec_dict[color]["line"][0],
        )

        ax.plot(
            omega_range,
            filter_50,
            color=spec_dict[color]["colors"][1],
            label="robust $(\omega = 0.50)$",
            ls=spec_dict[color]["line"][1],
        )
        if color == "colored":
            third_color = "#2ca02c"
        else:
            third_color = spec_dict[color]["colors"][2]
        ax.axhline(color=third_color, ls=spec_dict[color]["line"][2])
        ax.set_ylim([-300, 400])
        # ax.set_ylim([diff_costs_95[0], diff_costs_95[-1]])
        ax.set_ylabel(r"$\Delta$ Performance")
        ax.set_xlabel(r"$\omega$")
        ax.legend()
        fig.savefig(
            f"{DIR_FIGURES}/fig-application-difference{spec_dict[color]['file']}"
        )


def _performance_plot():

    file_list = sorted(glob.glob(SIM_RESULTS + "result_ev_0.00_mat_*.pkl"))
    nominal_costs = np.zeros(len(file_list))
    for j, file in enumerate(file_list):
        nominal_costs[j] = pkl.load(open(file, "rb"))[1][-1]

    file_list = sorted(glob.glob(SIM_RESULTS + "result_ev_0.95_mat_*.pkl"))
    robust_costs = np.zeros(len(file_list))
    for j, file in enumerate(file_list):
        robust_costs[j] = pkl.load(open(file, "rb"))[1][-1]

    # opt_costs = np.zeros(len(omega_range))
    # for j, omega in enumerate(omega_range):
    #     file = SIM_RESULTS + "result_ev_{}_mat_{}.pkl".format(
    #         "{:.2f}".format(omega), "{:.2f}".format(omega)
    #     )
    #     opt_costs[j] = pkl.load(open(file, "rb"))[1][-1]

    return nominal_costs, robust_costs


################################################################################
#                             Out of sample plot
################################################################################


def get_out_of_sample_diff(key, bins, sample_size):
    robust, nominal = _out_of_sample(key, sample_size)
    diff = robust - nominal
    hist_data = np.histogram(diff, bins=bins)
    x = np.linspace(np.min(hist_data[1]), np.max(hist_data[1]), bins)
    for color in color_opts:
        fig, ax = plt.subplots(1, 1)

        if color == "colored":
            third_color = "#ff7f0e"
        else:
            third_color = spec_dict[color]["colors"][2]

        ax.plot(
            x, hist_data[0] / sum(hist_data[0]), color=spec_dict[color]["colors"][0]
        )
        ax.axvline(color=third_color, ls=spec_dict[color]["line"][2])

        ax.set_ylabel(r"Density")
        ax.set_xlabel(r"$\Delta$ Performance")
        ax.set_ylim([0, 0.1])

        # ax.legend()
        fig.savefig(
            "{}/fig-application-validation{}".format(
                DIR_FIGURES, spec_dict[color]["file"]
            )
        )


def get_robust_performance(keys, width, sample_size):

    performance = np.zeros(len(keys), dtype=float)
    for j, key in enumerate(keys):
        robust, nominal = _out_of_sample(key, sample_size)
        diff = robust - nominal
        performance[j] = len(diff[diff >= 0]) / 1000

    for color in color_opts:
        fig, ax = plt.subplots(1, 1)

        ax.bar(
            keys,
            performance,
            width,
            color=spec_dict[color]["colors"][0],
            ls=spec_dict[color]["line"][0],
        )

        ax.set_ylabel(r"Share")
        ax.set_xlabel(r"$\omega$")
        ax.set_ylim([0., 0.3])
        plt.xticks(keys)
        fig.savefig(
            f"{DIR_FIGURES}/fig-application-validation-perfor"
            f"mance{spec_dict[color]['file']}"
        )


def _out_of_sample(key, sample_size):

    file_list = sorted(
        glob.glob(
            VAL_RESULTS
            + "result_ev_{}_size_{}_*.pkl".format("{:.2f}".format(key), sample_size)
        )
    )
    robust = np.zeros(len(file_list))
    nominal = np.zeros(len(file_list))
    for j, file in enumerate(file_list):
        res = pkl.load(open(file, "rb"))
        nominal[j] = res[0]
        robust[j] = res[1]
    return robust, nominal
