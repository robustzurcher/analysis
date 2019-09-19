import pickle as pkl
import glob
import pandas as pd
from zipfile import ZipFile
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np

from auxiliary import get_file
from ruspy.estimation.estimation_cost_parameters import lin_cost, cost_func, choice_prob
from config import DIR_FIGURES

# Global variables
BETA = 0.9999
PARAMS = np.array([10, 10])
NUM_BUSES = 200
NUM_PERIODS = 70000
GRIDSIZE = 1000
NUM_POINTS = int(NUM_PERIODS / GRIDSIZE) + 1
FIXP_DICT_4292 = "../pre_processed_data/fixp_results_1000_10_10_4292.pkl"
SIM_RESULTS = "../pre_processed_data/sim_results/"
VAL_RESULTS_4292 = "../pre_processed_data/validation_results_4292/"
VAL_RESULTS_2223 = "../pre_processed_data/validation_results_2223/"


def extract_zips():
    if os.path.exists(SIM_RESULTS):
        shutil.rmtree(SIM_RESULTS)
    os.makedirs("../pre_processed_data/sim_results")
    ZipFile('../pre_processed_data/simulation_results.zip').extractall(SIM_RESULTS)

    if os.path.exists(VAL_RESULTS_4292):
        shutil.rmtree(VAL_RESULTS_4292)
    os.makedirs("../pre_processed_data/validation_results_4292")
    ZipFile('../pre_processed_data/validation_results_4292.zip').extractall(
        VAL_RESULTS_4292)

    if os.path.exists(VAL_RESULTS_2223):
        shutil.rmtree(VAL_RESULTS_2223)
    os.makedirs("../pre_processed_data/validation_results_2223")
    ZipFile('../pre_processed_data/validation_results_2223.zip').extractall(
        VAL_RESULTS_2223)


################################################################################
#                           Probabilities shift
################################################################################


def df_probability_shift():
    dict_policies = get_file(FIXP_DICT_4292)
    return pd.DataFrame(
        {
            0: dict_policies[0.0][1][0, :13],
            0.54: dict_policies[0.54][1][0, :13],
            0.99: dict_policies[0.99][1][0, :13],
        }
    )


def get_probability_shift():

    x = np.arange(13)

    dict_policies = get_file(FIXP_DICT_4292)
    width = 0.25

    fig, ax = plt.subplots(1, 1)

    ax.bar(x - width, dict_policies[0.0][1][0, :13], width, label="ML estimate")
    ax.bar(
        x, dict_policies[0.54][1][0, :13], width, label="worst case of $\omega=0.54$"
    )
    ax.bar(
        x + width,
        dict_policies[0.99][1][0, :13],
        width,
        label="worst case of $\omega=0.99$",
    )

    ax.set_ylabel(r"Probability")
    ax.set_xlabel(r"Mileage increase (in thousands)")

    plt.legend()

    fig.savefig(f"{DIR_FIGURES}/fig-application-probability-shift")


################################################################################
#                       Replacement Probabilities
################################################################################
keys = [0.0, 0.5, 0.95]


def df_replacement_probabilties():
    choice_ml, choices = _create_repl_prob_plot(FIXP_DICT_4292, keys)
    return pd.DataFrame(
        {0.0: choice_ml[:, 1], keys[1]: choices[0][:, 1], keys[2]: choices[1][:, 1]}
    )


def get_replacement_probabilities():

    choice_ml, choices = _create_repl_prob_plot(FIXP_DICT_4292, keys)
    states = range(choice_ml.shape[0])

    fig, ax = plt.subplots(1, 1)

    ax.plot(states, choice_ml[:, 1], label="Optimal")
    for i, choice in enumerate(choices):
        ax.plot(states, choice[:, 1], label=f"Robust $(\omega = {keys[i+1]})$")

    ax.set_ylabel(r"Replacement probability")
    ax.set_xlabel(r"Mileage (in thousands)")
    ax.set_ylim([0, 1])

    plt.legend()
    fig.savefig(f"{DIR_FIGURES}/fig-application-replacement-probabilities")


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
#                       Threshold plot
################################################################################

num_keys = 100


def df_thresholds():
    means_discrete = _threshold_data()
    omega_range = np.linspace(0, 0.99, num_keys)
    return pd.DataFrame({'omega': omega_range, 'threshold': means_discrete})


def get_replacement_thresholds():

    means_discrete = _threshold_data()

    omega_range = np.linspace(0, 0.99, num_keys)
    means_ml = np.full(len(omega_range), np.round(means_discrete[0])).astype(int)

    omega_sections, state_sections = _create_sections(means_discrete, omega_range)

    y_0 = state_sections[0][0] - 2
    y_1 = state_sections[-1][-1] + 2

    fig, ax = plt.subplots(1, 1)
    ax.set_ylim([y_0, y_1])
    plt.yticks(range(y_0, y_1, 2))
    ax.set_ylabel(r"Milage at replacement (in thousands)")
    ax.set_xlabel(r"$\omega$")
    ax.plot(omega_range, means_ml, label="Optimal")
    for j, i in enumerate(omega_sections[:-1]):
        ax.plot(i, state_sections[j], color="#ff7f0e")
    ax.plot(
        omega_sections[-1],
        state_sections[-1],
        color="#ff7f0e",
        label="Robust anticipating $\omega$",
    )

    plt.legend()
    fig.savefig(f"{DIR_FIGURES}/fig-application-replacement-thresholds")


def _threshold_data():
    file_list = sorted(glob.glob(SIM_RESULTS + "result_ev_*_mat_0.00.pkl"))
    if len(file_list) != 0:
        means_robust_strat = np.array([])
        for file in file_list:
            mean = pkl.load(open(file, "rb"))[0]
            means_robust_strat = np.append(
                means_robust_strat, mean)
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


def get_performance_decision_rules():
    dict_policies = get_file(FIXP_DICT_4292)

    v_exp_ml = np.full(NUM_POINTS, dict_policies[0.0][0][0])
    v_exp_worst = np.full(NUM_POINTS, dict_policies[0.95][0][0])

    v_disc_ml = pkl.load(open(SIM_RESULTS + "result_ev_0.00_mat_0.95.pkl", "rb"))[1]

    periods = np.arange(0, NUM_PERIODS + GRIDSIZE, GRIDSIZE)

    fig, ax = plt.subplots(1, 1)
    ax.set_ylim([1.1 * v_disc_ml[-1], 0])
    ax.set_ylabel(r"Performance")
    ax.set_xlabel(r"Periods")

    formatter = plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x)))
    ax.get_xaxis().set_major_formatter(formatter)
    ax.get_yaxis().set_major_formatter(formatter)

    # 'Discounted utility of otpimal strategy'
    ax.plot(periods, v_disc_ml, label="Optimal")
    # 'Expected value of nominal strategy'
    ax.plot(periods, v_exp_ml, label="Optimal (expected value)")
    # 'Expected value of robust strategy with $\omega = 0.99$'
    ax.plot(periods, v_exp_worst, label="Robust (expected value)")

    plt.legend()
    fig.savefig(f"{DIR_FIGURES}/fig-application-performance-decision-rules")



################################################################################
#                             Performance plot
################################################################################


def get_performance():

    num_keys = 100

    omega_range = np.linspace(0, 0.99, num_keys)

    nominal_costs, opt_costs, robust_5_costs = _performance_plot(omega_range)

    fig, ax = plt.subplots(1, 1)

    # ax.plot(omega_range, opt_costs, label="Discounted utilities of optimal strategy")
    ax.plot(
        omega_range, nominal_costs, label="Discounted utilities of nominal strategy"
    )
    ax.plot(
        omega_range,
        robust_5_costs,
        label="Discounted utilities of robust strategy with $\omega = 0.95$",
    )

    formatter = plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x)))
    ax.get_yaxis().set_major_formatter(formatter)

    ax.set_ylim([nominal_costs[-1], nominal_costs[0]])
    ax.set_ylabel(r"Performance")
    ax.set_xlabel(r"$\omega$")

    plt.legend()
    fig.savefig(f"{DIR_FIGURES}/fig-application-performance")


def _performance_plot(omega_range):

    file_list = sorted(glob.glob(SIM_RESULTS + "result_ev_0.00_mat_*.pkl"))
    nominal_costs = np.zeros(len(file_list))
    for j, file in enumerate(file_list):
        nominal_costs[j] =  pkl.load(open(file, "rb"))[1][-1]

    file_list = sorted(glob.glob(SIM_RESULTS + "result_ev_0.95_mat_*.pkl"))
    robust_costs = np.zeros(len(file_list))
    for j, file in enumerate(file_list):
        robust_costs[j] = pkl.load(open(file, "rb"))[1][-1]

    opt_costs = np.zeros(len(omega_range))
    for j, omega in enumerate(omega_range):
        file = SIM_RESULTS + "result_ev_{}_mat_{}.pkl".format("{:.2f}".format(
            omega), "{:.2f}".format(omega))
        opt_costs[j] = pkl.load(open(file, "rb"))[1][-1]


    return nominal_costs, opt_costs, robust_costs
