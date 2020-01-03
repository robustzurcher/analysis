import glob
import os
import pickle as pkl
import shutil
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from auxiliary import get_file
from config import DIR_FIGURES
from ruspy.model_code.choice_probabilities import choice_prob_gumbel
from ruspy.model_code.cost_functions import calc_obs_costs
from ruspy.model_code.cost_functions import cubic_costs
from ruspy.model_code.cost_functions import lin_cost
from ruspy.model_code.cost_functions import sqrt_costs
from ruspy.simulation.simulation import simulate
from scipy.signal import savgol_filter

# Global variables
BETA = 0.9999
NUM_BUSES = 200
BIN_SIZE = 5  # in thousand
NUM_PERIODS = 70000
GRIDSIZE = 1000
NUM_KEYS = 25
NUM_POINTS = int(NUM_PERIODS / GRIDSIZE) + 1
FIXP_DICT_4292_LINEAR = "../pre_processed_data/fixp_results_4292_linear.pkl"
FIXP_DICT_4292_SQRT = "../pre_processed_data/fixp_results_4292_sqrt.pkl"
FIXP_DICT_4292_CUBIC = "../pre_processed_data/fixp_results_4292_cubic.pkl"
FIXP_DICT_2223_LINEAR = "../pre_processed_data/fixp_results_2223_linear.pkl"
POLICIES_4292_LIN = get_file(FIXP_DICT_4292_LINEAR)
POLICIES_4292_SQRT = get_file(FIXP_DICT_4292_SQRT)
POLICIES_4292_CUBIC = get_file(FIXP_DICT_4292_CUBIC)
SIM_RESULTS = "../pre_processed_data/sim_results/"
VAL_RESULTS = "../pre_processed_data/val_results/"
DATA_FOLDER = "../pre_processed_data/"
color_opts = ["colored", "black_white"]
spec_dict = {
    "colored": {"colors": [None] * 4, "line": ["-"] * 3, "hatch": [""] * 3, "file": ""},
    "black_white": {
        "colors": ["#808080", "#d3d3d3", "#A9A9A9", "#C0C0C0", "k"],
        "line": ["-", "--", ":"],
        "hatch": ["", "OOO", "///"],
        "file": "-sw",
    },
}
STATES_POlICY = POLICIES_4292_LIN[0.0][0].shape[0]
PARAMS_LIN = np.array([400, 35])
PARAMS_SQRT = np.array([460, 50])
PARAMS_CUBIC = np.array([700, 43200, -780, 4.7])
SCALE_LIN = 0.001
SCALE_CUBIC = 1e-5
SCALE_SQRT = 1e-2
COSTS_LIN = calc_obs_costs(STATES_POlICY, lin_cost, PARAMS_LIN, SCALE_SQRT)
COSTS_CUBIC = calc_obs_costs(STATES_POlICY, cubic_costs, PARAMS_CUBIC, SCALE_CUBIC)
COSTS_SQRT = calc_obs_costs(STATES_POlICY, sqrt_costs, PARAMS_SQRT, SCALE_SQRT)


def extract_zips():
    if os.path.exists(SIM_RESULTS):
        shutil.rmtree(SIM_RESULTS)
    os.makedirs(SIM_RESULTS)
    sim_zip_list = glob.glob(DATA_FOLDER + "simulation_results*.zip")
    for file in sim_zip_list:
        ZipFile(file).extractall(SIM_RESULTS)

    if os.path.exists(VAL_RESULTS):
        shutil.rmtree(VAL_RESULTS)
    os.makedirs(VAL_RESULTS)

    val_zip_list = glob.glob(DATA_FOLDER + "validation_results*.zip")
    for file in val_zip_list:
        ZipFile(file).extractall(VAL_RESULTS)


################################################################################
#                           Probabilities
################################################################################

p_size = 3
x = np.arange(1, p_size + 1) * BIN_SIZE
width = 0.8 * BIN_SIZE
p_raw = np.loadtxt("../pre_processed_data/parameters/rust_trans_raw.txt")
hesse_inv_raw = np.loadtxt("../pre_processed_data/parameters/rust_cov_raw.txt")


def get_probabilities(state):
    p_ml = POLICIES_4292_LIN[0.0][1][state, state : state + p_size]

    for color in color_opts:
        fig, ax = plt.subplots(1, 1)

        ax.bar(
            x,
            p_ml,
            width,
            color=spec_dict[color]["colors"][0],
            ls=spec_dict[color]["line"][0],
        )

        ax.set_ylabel(r"Transition probability")
        ax.set_xlabel(r"Mileage increase (in thousands)")
        plt.xticks(x)
        ax.set_ylim([0.00, 0.80])
        fig.savefig(
            f"{DIR_FIGURES}/fig-application-probabilities{spec_dict[color]['file']}"
        )


def get_probabilities_bar(state):

    p_ml = POLICIES_4292_LIN[0.0][1][state, state : state + p_size]
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

        ax.set_ylabel(r"Transition probability")
        ax.set_ylim([0.00, 0.80])

        ax.set_xlabel(r"Mileage increase (in thousands)")
        plt.xticks(x)
        fig.savefig(
            f"{DIR_FIGURES}/fig-application-probabilities-bar{spec_dict[color]['file']}"
        )


def df_probability_shift(state):
    dict_policies_4292 = get_file(FIXP_DICT_4292_LINEAR)
    dict_policies_2223 = get_file(FIXP_DICT_2223_LINEAR)
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
            POLICIES_4292_LIN[0.0][1][state, state : state + p_size],
            width,
            color=spec_dict[color]["colors"][0],
            hatch=spec_dict[color]["hatch"][0],
            label="reference",
        )
        ax.bar(
            x,
            POLICIES_4292_LIN[0.50][1][state, state : state + p_size],
            width,
            color=spec_dict[color]["colors"][1],
            hatch=spec_dict[color]["hatch"][1],
            label=r"$\omega=0.50$",
        )
        ax.bar(
            x + width,
            POLICIES_4292_LIN[0.95][1][state, state : state + p_size],
            width,
            color=spec_dict[color]["colors"][2],
            hatch=spec_dict[color]["hatch"][2],
            label=r"$\omega=0.95$",
        )

        ax.set_ylabel(r"Transition probability")
        ax.set_xlabel(r"Mileage increase (in thousands)")
        ax.set_ylim([0.0, 0.8])
        plt.xticks(x)
        ax.legend()

        fig.savefig(
            f"{DIR_FIGURES}/fig-application-probability-shift-omega{spec_dict[color]['file']}"
        )


def get_probability_shift_data(state):

    width = 0.25 * BIN_SIZE
    dict_policies_4292 = get_file(FIXP_DICT_4292_LINEAR)
    dict_policies_2223 = get_file(FIXP_DICT_2223_LINEAR)

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

        ax.set_ylabel(r"Transition probability")
        ax.set_xlabel(r"Mileage increase (in thousands)")
        plt.xticks(x)
        ax.set_ylim([0.0, 0.8])

        ax.legend()

        fig.savefig(
            f"{DIR_FIGURES}/fig-application-probability-shift-data{spec_dict[color]['file']}"
        )


def get_probability_shift_models(state, omega):

    width = 0.2 * BIN_SIZE

    for color in color_opts:
        fig, ax = plt.subplots(1, 1)

        ax.bar(
            x - 1.5 * width,
            POLICIES_4292_LIN[0.0][1][state, state : state + p_size],
            width,
            color=spec_dict[color]["colors"][0],
            hatch=spec_dict[color]["hatch"][0],
            label="reference",
        )
        ax.bar(
            x - 0.5 * width,
            POLICIES_4292_LIN[omega][1][state, state : state + p_size],
            width,
            color=spec_dict[color]["colors"][3],
            hatch=spec_dict[color]["hatch"][2],
            label="linear",
        )
        ax.bar(
            x + 0.5 * width,
            POLICIES_4292_SQRT[omega][1][state, state : state + p_size],
            width,
            color=spec_dict[color]["colors"][1],
            hatch=spec_dict[color]["hatch"][1],
            label="sqrt",
        )
        ax.bar(
            x + 1.5 * width,
            POLICIES_4292_CUBIC[omega][1][state, state : state + p_size],
            width,
            color=spec_dict[color]["colors"][2],
            hatch=spec_dict[color]["hatch"][2],
            label="cubic",
        )

        ax.set_ylabel(r"Transition probability")
        ax.set_xlabel(r"Mileage increase (in thousands)")
        plt.xticks(x)
        ax.set_ylim([0.0, 0.8])

        ax.legend()

        fig.savefig(
            f"{DIR_FIGURES}/fig-application-probability-shift-models"
            f"{spec_dict[color]['file']}"
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


def df_maintenance_probabilties_lin(min_state, max_state):
    choice_ml, choices = _create_repl_prob_plot(POLICIES_4292_LIN, COSTS_LIN, keys)
    states = np.arange(choice_ml.shape[0]) * BIN_SIZE
    return pd.DataFrame(
        {
            "milage_thousands": states[min_state:max_state],
            0.0: choice_ml[min_state:max_state, 0],
            keys[1]: choices[0][min_state:max_state, 0],
            keys[2]: choices[1][min_state:max_state, 0],
        }
    )


def df_maintenance_probabilties_sqrt(min_state, max_state):
    choice_ml, choices = _create_repl_prob_plot(POLICIES_4292_SQRT, COSTS_LIN, keys)
    states = np.arange(choice_ml.shape[0]) * BIN_SIZE
    return pd.DataFrame(
        {
            "milage_thousands": states[min_state:max_state],
            0.0: choice_ml[min_state:max_state, 0],
            keys[1]: choices[0][min_state:max_state, 0],
            keys[2]: choices[1][min_state:max_state, 0],
        }
    )


def df_maintenance_probabilties_cubic(min_state, max_state):
    choice_ml, choices = _create_repl_prob_plot(POLICIES_4292_CUBIC, COSTS_LIN, keys)
    states = np.arange(choice_ml.shape[0]) * BIN_SIZE
    return pd.DataFrame(
        {
            "milage_thousands": states[min_state:max_state],
            0.0: choice_ml[min_state:max_state, 0],
            keys[1]: choices[0][min_state:max_state, 0],
            keys[2]: choices[1][min_state:max_state, 0],
        }
    )


def get_maintenance_probabilities(min_states, max_states, state_steps):

    choice_ml_lin, choices_lin = _create_repl_prob_plot(
        POLICIES_4292_LIN, COSTS_LIN, keys
    )
    choice_ml_sqrt, choices_sqrt = _create_repl_prob_plot(
        POLICIES_4292_SQRT, COSTS_SQRT, keys
    )
    choice_ml_cubic, choices_cubic = _create_repl_prob_plot(
        POLICIES_4292_CUBIC, COSTS_CUBIC, keys
    )
    choice_list = [
        (choice_ml_lin, choices_lin, "linear"),
        (choice_ml_sqrt, choices_sqrt, "sqrt"),
        (choice_ml_cubic, choices_cubic, "cubic"),
    ]

    for color in color_opts:

        for choice_ml, choices, name in choice_list:
            max_state = max_states[name]
            min_state = min_states[name]
            state_step = state_steps[name]
            states = np.arange(max_state) * BIN_SIZE
            fig, ax = plt.subplots(1, 1)
            ax.plot(
                states[min_state:max_state],
                choice_ml[min_state:max_state, 0],
                color=spec_dict[color]["colors"][0],
                ls=spec_dict[color]["line"][0],
                label="optimal",
            )
            for i, choice in enumerate(choices):
                ax.plot(
                    states[min_state:max_state],
                    choice[min_state:max_state, 0],
                    color=spec_dict[color]["colors"][i + 1],
                    ls=spec_dict[color]["line"][i + 1],
                    label=fr"robust $(\omega = {keys[i+1]:.2f})$",
                )

            ax.set_ylabel(r"Maintenance probability")
            ax.set_xlabel(r"Mileage (in thousands)")
            ax.set_ylim([0, 1])

            plt.xticks(states[min_state:max_state][::state_step])
            ax.legend()

            fig.savefig(
                f"{DIR_FIGURES}/fig-application-maintenance-probabilities-{name}"
                f"{spec_dict[color]['file']}"
            )


def _create_repl_prob_plot(dict_policies, costs, keys):
    ev_ml = dict_policies[0.0][0]
    choice_ml = choice_prob_gumbel(ev_ml, costs, BETA)
    choices = []
    for omega in keys[1:]:
        choices += [choice_prob_gumbel(dict_policies[omega][0], costs, BETA)]
    return choice_ml, choices


################################################################################
#                       Demonstration
################################################################################


def get_demonstration_df(init_dict, max_period):
    states, periods = get_demonstration_data(init_dict)
    return pd.DataFrame(
        {
            "months_ml": periods[0][:max_period],
            "months_rob": periods[1][:max_period],
            "opt_mileage": states[0][:max_period] * BIN_SIZE,
            "rob_mileage": states[1][:max_period] * BIN_SIZE,
        }
    )


def get_demonstration(df, max_period, max_mileage):
    states = (df["opt_mileage"], df["rob_mileage"])
    periods = (df["months_ml"], df["months_rob"])
    labels = ["optimal", r"robust ($\omega = 0.95$)"]
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
        ax.set_ylim([0, max_mileage])

        fig.savefig(
            f"{DIR_FIGURES}/fig-application-demonstration{spec_dict[color]['file']}"
        )


def get_demonstration_data(init_dict):

    dict_policies = POLICIES_4292_LIN.copy()
    ev_ml = dict_policies[0.0][0]
    ev_95 = dict_policies[0.95][0]
    trans_mat = dict_policies[0.0][1]
    n_states = ev_ml.shape[0]
    cost_sim = calc_obs_costs(n_states, lin_cost, PARAMS_LIN, scale=0.001)

    df_ml = simulate(init_dict, ev_ml, cost_sim, trans_mat)
    df_95 = simulate(init_dict, ev_95, cost_sim, trans_mat)

    periods_ml = np.arange(NUM_PERIODS)
    periods_95 = np.arange(NUM_PERIODS)
    periods = [periods_ml, periods_95]
    states_ml = np.array(df_ml["state"], dtype=int)
    states_95 = np.array(df_95["state"], dtype=int)
    states = [states_ml, states_95]

    for i, df in enumerate([df_ml, df_95]):
        index = (
            np.array(df[df["decision"] == 1].index.get_level_values(1), dtype=int) + 1
        )
        states[i] = np.insert(states[i], index, 0)
        periods[i] = np.insert(periods[i], index, index - 1)

    return states, periods


################################################################################
#                       Threshold plot
################################################################################


def df_thresholds(cost_func_name):
    means_discrete = _threshold_data(cost_func_name)
    omega_range = np.linspace(0, 0.99, NUM_KEYS)
    return pd.DataFrame({"omega": omega_range, "threshold": means_discrete})


def get_replacement_thresholds(cost_func_name):

    means_discrete = _threshold_data(cost_func_name) * BIN_SIZE

    omega_range = np.round(np.linspace(0, 0.99, NUM_KEYS), decimals=2)
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
            color=spec_dict[color]["colors"][1],
            ls=spec_dict[color]["line"][0],
            label="optimal",
        )
        if color == "colored":
            second_color = "#ff7f0e"
        else:
            second_color = spec_dict[color]["colors"][0]
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
            f"{DIR_FIGURES}/fig-application-replacement-thresholds"
            f"-{cost_func_name}{spec_dict[color]['file']}"
        )


def _threshold_data(cost_func_name):
    file_list = sorted(
        glob.glob(SIM_RESULTS + f"result_ev_*_mat_0.00_" f"{cost_func_name}.pkl")
    )
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


def get_decision_rule_df(func_name):
    dict_policies = select_policy_dict(func_name)

    v_exp_ml = np.full(NUM_POINTS, dict_policies[0.0][0][0])

    v_disc_ml = pkl.load(
        open(SIM_RESULTS + f"result_ev_0.00_mat_0.95_{func_name}.pkl", "rb")
    )[1]

    periods = np.arange(0, NUM_PERIODS + GRIDSIZE, GRIDSIZE)

    return pd.DataFrame(
        {"months": periods, "disc_strategy": v_disc_ml, "exp_value": v_exp_ml}
    )


def get_performance_decision_rules(func_name):
    print("The underlying transition matrix is the worst case given omega=0.95")
    policy_dict = select_policy_dict(func_name)

    v_exp_ml = np.full(NUM_POINTS, policy_dict[0.0][0][0])
    v_exp_095 = np.full(NUM_POINTS, policy_dict[0.95][0][0])

    v_disc_ml = pkl.load(
        open(SIM_RESULTS + f"result_ev_0.00_mat_0.95_{func_name}.pkl", "rb")
    )[1]

    print(v_disc_ml[-1] / v_exp_ml[-1], v_disc_ml[-1] / v_exp_095[-1])
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
            color=spec_dict[color]["colors"][0],
            ls=spec_dict[color]["line"][1],
            label="actual",
        )

        ax.plot(
            periods,
            v_exp_095,
            color=spec_dict[color]["colors"][0],
            ls=spec_dict[color]["line"][1],
            label="true",
        )

        ax.set_ylim([1.3 * v_exp_ml[0], 0])
        ax.legend()
        fig.savefig(
            f"{DIR_FIGURES}/fig-application-performance-decision-rules-{func_name}"
            f"{spec_dict[color]['file']}"
        )


################################################################################
#                             Performance plot
################################################################################


def get_difference_df(func_name):

    omega_range = np.round(np.linspace(0, 0.99, NUM_KEYS), decimals=2)

    nominal_costs, robust_costs_95 = _performance_plot(func_name)

    file_list = sorted(glob.glob(SIM_RESULTS + f"result_ev_0.50_mat_*_{func_name}.pkl"))
    robust_costs_50 = np.zeros(len(file_list))
    for j, file in enumerate(file_list):
        robust_costs_50[j] = pkl.load(open(file, "rb"))[1][-1]

    diff_costs_95 = robust_costs_95 - nominal_costs
    diff_costs_50 = robust_costs_50 - nominal_costs

    print("The dataframe contains the difference for robust - nominal strategy.")

    return pd.DataFrame(
        {"omega": omega_range, "robust_95": diff_costs_95, "robust_050": diff_costs_50}
    )


def get_difference_plot(func_name, window_length):

    omega_range = np.linspace(0, 0.99, NUM_KEYS)

    nominal_costs, robust_costs_95 = _performance_plot(func_name)

    file_list = sorted(glob.glob(SIM_RESULTS + f"result_ev_0.50_mat_*_{func_name}.pkl"))
    robust_costs_50 = np.zeros(len(file_list))
    for j, file in enumerate(file_list):
        robust_costs_50[j] = pkl.load(open(file, "rb"))[1][-1]

    diff_costs_95 = robust_costs_95 - nominal_costs
    diff_costs_50 = robust_costs_50 - nominal_costs
    filter_95 = savgol_filter(diff_costs_95, window_length, 3)
    filter_50 = savgol_filter(diff_costs_50, window_length, 3)

    for color in color_opts:
        fig, ax = plt.subplots(1, 1)

        ax.plot(
            omega_range,
            filter_95,
            color=spec_dict[color]["colors"][0],
            label=r"robust $(\omega = 0.95)$",
            ls=spec_dict[color]["line"][0],
        )

        ax.plot(
            omega_range,
            filter_50,
            color=spec_dict[color]["colors"][0],
            label=r"robust $(\omega = 0.50)$",
            ls=spec_dict[color]["line"][1],
        )
        if color == "colored":
            third_color = "#2ca02c"
        else:
            third_color = spec_dict[color]["colors"][4]
        ax.axhline(color=third_color, ls=spec_dict[color]["line"][2])
        ax.set_ylim([-300, 400])
        # ax.set_ylim([diff_costs_95[0], diff_costs_95[-1]])
        ax.set_ylabel(r"$\Delta$ Performance")
        ax.set_xlabel(r"$\omega$")
        ax.legend()
        fig.savefig(
            f"{DIR_FIGURES}/fig-application-difference-{func_name}{spec_dict[color]['file']}"
        )


def _performance_plot(func_name):

    file_list = sorted(glob.glob(SIM_RESULTS + f"result_ev_0.00_mat_*_{func_name}.pkl"))
    nominal_costs = np.zeros(len(file_list))
    for j, file in enumerate(file_list):
        nominal_costs[j] = pkl.load(open(file, "rb"))[1][-1]

    file_list = sorted(glob.glob(SIM_RESULTS + f"result_ev_0.95_mat_*_{func_name}.pkl"))
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


def get_out_of_sample_diff(key, bins, window_lenth):
    hist_filter_lin, x_lin = filtered_hist_data(key, bins, window_lenth, "linear")
    hist_filter_cubic, x_cubic = filtered_hist_data(key, bins, window_lenth, "cubic")
    hist_filter_sqrt, x_sqrt = filtered_hist_data(key, bins, window_lenth, "sqrt")
    for color in color_opts:
        fig, ax = plt.subplots(1, 1)

        if color == "colored":
            third_color = "#ff7f0e"
        else:
            third_color = spec_dict[color]["colors"][4]

        ax.plot(
            x_lin, hist_filter_lin, color=spec_dict[color]["colors"][0], label="linear"
        )
        ax.plot(
            x_sqrt, hist_filter_sqrt, color=spec_dict[color]["colors"][1], label="sqrt"
        )
        ax.plot(
            x_cubic,
            hist_filter_cubic,
            color=spec_dict[color]["colors"][2],
            label="cubic",
        )

        ax.axvline(color=third_color, ls=spec_dict[color]["line"][2])

        ax.set_ylabel(r"Density")
        ax.set_xlabel(r"$\Delta$ Performance")
        ax.set_ylim([0, None])

        ax.legend()
        fig.savefig(
            "{}/fig-application-validation{}".format(
                DIR_FIGURES, spec_dict[color]["file"]
            )
        )


def get_robust_performance(keys, width):
    performance_lin = calc_perfomance(keys, "linear")
    performance_sqrt = calc_perfomance(keys, "sqrt")
    performance_cubic = calc_perfomance(keys, "cubic")

    keys_np = np.array(keys)

    for color in color_opts:
        fig, ax = plt.subplots(1, 1)

        ax.bar(
            keys_np - width,
            performance_lin,
            width,
            color=spec_dict[color]["colors"][0],
            ls=spec_dict[color]["line"][0],
            label="linear",
        )

        ax.bar(
            keys_np,
            performance_sqrt,
            width,
            color=spec_dict[color]["colors"][1],
            ls=spec_dict[color]["line"][1],
            label="sqrt",
        )

        ax.bar(
            keys_np + width,
            performance_cubic,
            width,
            color=spec_dict[color]["colors"][2],
            ls=spec_dict[color]["line"][2],
            label="cubic",
        )

        ax.legend()
        ax.set_ylabel(r"Share")
        ax.set_xlabel(r"$\omega$")
        ax.set_ylim([0.0, 0.35])
        plt.xticks(keys)
        fig.savefig(
            f"{DIR_FIGURES}/fig-application-validation-perfor"
            f"mance{spec_dict[color]['file']}"
        )


def calc_perfomance(keys, func_name):
    performance = np.zeros(len(keys), dtype=float)
    for j, key in enumerate(keys):
        robust, nominal = _out_of_sample(key, func_name)
        diff = robust - nominal
        performance[j] = len(diff[diff >= 0]) / len(diff)
    return performance


def filtered_hist_data(key, bins, window_length, func_name):
    robust, nominal = _out_of_sample(key, func_name)
    diff = robust - nominal
    hist_data = np.histogram(diff, bins=bins)
    hist_normed = hist_data[0] / sum(hist_data[0])
    hist_filter = savgol_filter(hist_normed, window_length, 3)
    x = np.linspace(np.min(hist_data[1]), np.max(hist_data[1]), bins)
    return hist_filter, x


def _out_of_sample(key, func_name):

    file_list = sorted(
        glob.glob(
            VAL_RESULTS + "result_ev_{}_run_*_{}.pkl".format(f"{key:.2f}", func_name)
        )
    )
    robust = np.zeros(len(file_list))
    nominal = np.zeros(len(file_list))
    for j, file in enumerate(file_list):
        res = pkl.load(open(file, "rb"))
        nominal[j] = res[0]
        robust[j] = res[1]
    return robust, nominal


def select_policy_dict(func_name):
    if func_name == "linear":
        return POLICIES_4292_LIN
    elif func_name == "cubic":
        return POLICIES_4292_CUBIC
    elif func_name == "sqrt":
        return POLICIES_4292_SQRT
