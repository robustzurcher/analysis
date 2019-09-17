from zipfile import ZipFile
from pathlib import Path
import pickle as pkl
import glob
import os

import matplotlib.pyplot as plt
import numpy as np

from ruspy.estimation.estimation_cost_parameters import lin_cost, cost_func, choice_prob
from ruspy.simulation.value_zero import discount_utility, calc_ev_0
from config import DIR_FIGURES

# Global variables
BETA = 0.9999
PARAMS = np.array([10, 10])
NUM_BUSES = 200
NUM_PERIODS = 70000
GRIDSIZE = 1000
NUM_POINTS = int(NUM_PERIODS / GRIDSIZE) + 1


def _get_file(fname):
    if not isinstance(fname, Path):
        fname = Path(fname)

    fname_zip = Path(fname).with_suffix('.zip')
    fname_pkl = Path(fname).with_suffix('.pkl')

    if not os.path.exists(fname_pkl):
        with ZipFile(fname_zip, 'r') as zipObj:
            zipObj.extractall(Path(fname).parent)

    return pkl.load(open(fname_pkl, 'rb'))


def _create_repl_prob_plot(file, keys):
    dict_policies = _get_file(file)
    ev_ml = dict_policies[0.0][0]
    num_states = ev_ml.shape[0]
    costs = cost_func(num_states, lin_cost, PARAMS)
    choice_ml = choice_prob(ev_ml, costs, BETA)
    choices = []
    for omega in keys[1:]:
        choices += [choice_prob(dict_policies[omega][0], costs, BETA)]
    return choice_ml, choices


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


def _threshold_plot(num_keys):
    threshold_folder = "../pre_processed_data/df_threshold/"
    file_list = sorted(glob.glob(threshold_folder + "df*"))
    if len(file_list) != 0:
        means_robust_strat = np.array([])
        for file in file_list:
            df = pkl.load(open(file, "rb"))
            means_robust_strat = np.append(means_robust_strat,
                                           np.mean(df[df["decision"] == 1]["state"]))
    else:
        means_robust_strat = pkl.load(open(threshold_folder + "means_robust.pkl", "rb"))

    means_discrete = np.around(means_robust_strat).astype(int)
    omega_range = np.linspace(0, 0.99, num_keys)
    means_ml = np.full(len(omega_range), np.round(means_robust_strat[0])).astype(int)
    omega_sections, state_sections = _create_sections(means_discrete, omega_range)
    return omega_range, means_ml, omega_sections, state_sections


def _convergence_plot(df_file, df_alt_file, fixp_point_dict):
    dict_policies = _get_file(fixp_point_dict)
    ev_ml = dict_policies[0.0][0]
    ev_worst = dict_policies[0.99][0]
    try:
        df_trans_99_ev_ml = pkl.load(
            open(df_file, "rb"))
        # Calculate the expected value at time zero
        v_exp_ml = np.full(NUM_POINTS, calc_ev_0(df_trans_99_ev_ml, ev_ml))
        v_exp_worst = np.full(NUM_POINTS, calc_ev_0(df_trans_99_ev_ml, ev_worst))
        # Calculate the value at time 0 by discounting the utility
        v_disc_ml = discount_utility(df_trans_99_ev_ml, GRIDSIZE, BETA)
    except:
        v_exp_ml, v_exp_worst, v_disc_ml = pkl.load(open(df_alt_file, "rb"))
    # Create a numpy array of the periods for plotting
    periods = np.arange(0, NUM_PERIODS + GRIDSIZE, GRIDSIZE)

    return v_disc_ml, v_exp_ml, v_exp_worst, periods


def _performance_plot(nominal_sub, opt_sub, rob_sub):
    strategy_folder = "../pre_processed_data/strategies/"

    file_list = sorted(
        glob.glob(strategy_folder + nominal_sub + "df*"))
    if len(file_list) != 0:
        nominal_costs = np.zeros(len(file_list))
        for j, file in enumerate(file_list):
            df = pkl.load(open(file, "rb"))
            nominal_costs[j] = discount_utility(df, GRIDSIZE, BETA)[-1]
    else:
        nominal_costs = pkl.load(open(strategy_folder + nominal_sub + "nominal_costs.pkl", "rb"))

    file_list = sorted(glob.glob(strategy_folder + opt_sub + "df*"))
    if len(file_list) != 0:
        opt_costs = np.zeros(len(file_list))
        for j, file in enumerate(file_list):
            df = pkl.load(open(file, "rb"))
            opt_costs[j] = discount_utility(df, GRIDSIZE, BETA)[-1]
    else:
        opt_costs = pkl.load(
            open(strategy_folder + opt_sub + "opt_costs.pkl", "rb"))

    file_list = sorted(glob.glob(strategy_folder + rob_sub + "df*"))
    if len(file_list) != 0:
        robust_costs = np.zeros(len(file_list))
        for j, file in enumerate(file_list):
            df = pkl.load(open(file, "rb"))
            robust_costs[j] = discount_utility(df, GRIDSIZE, BETA)[-1]
    else:
        robust_costs = pkl.load(open(glob.glob(strategy_folder + rob_sub +
                                                  "*.pkl")[0], "rb"))
    return nominal_costs, opt_costs, robust_costs


def get_probability_shift():

    x = np.arange(13)

    dict_policies = _get_file("../pre_processed_data/results_1000_10_10.pkl")
    width = 0.25

    fig, ax = plt.subplots(1, 1)

    ax.bar(x - width, dict_policies[0.0][1][0, :13], width, label="ML estimate")
    ax.bar(x, dict_policies[0.54][1][0, :13], width, label="worst case of $\omega=0.54$")
    ax.bar(x + width, dict_policies[0.99][1][0, :13], width, label="worst case of $\omega=0.99$")

    ax.set_ylabel(r"Probability")
    ax.set_xlabel(r"Mileage increase (in thousands)")

    plt.legend()

    fig.savefig(f'{DIR_FIGURES}/fig-application-probability-shift')


def get_replacement_probabilities_stylized():

    fixp_point_dict = "../pre_processed_data/results_1000_10_10.pkl"
    keys = [0.0, 0.54, 0.99]
    choice_ml, choices = _create_repl_prob_plot(fixp_point_dict, keys)

    states = range(choice_ml.shape[0])

    fig, ax = plt.subplots(1, 1)

    ax.plot(states, choice_ml[:, 1], label='Optimal')
    for i, choice in enumerate(choices):
        ax.plot(states, choice[:, 1], label=f'Robust $(\omega = {keys[i+1]})$')

    ax.set_ylabel(r"Replacement probability")
    ax.set_xlabel(r"Mileage increase (in thousands)")
    ax.set_ylim([0, 1])

    plt.legend()
    fig.savefig(f'{DIR_FIGURES}/fig-application-replacement-probabilities-stylized')


def get_replacement_thresholds():

    num_keys = 12

    omega_range, means_ml, omega_sections, state_sections = _threshold_plot(num_keys)

    y_0 = state_sections[0][0] - 2
    y_1 = state_sections[-1][-1] + 2

    fig, ax = plt.subplots(1, 1)
    ax.set_ylim([y_0, y_1])
    plt.yticks(range(y_0, y_1, 2))
    ax.set_ylabel(r"Mean state of replacement")
    ax.set_xlabel(r"$\omega$")
    ax.plot(omega_range, means_ml, label='Optimal')
    for j, i in enumerate(omega_sections[:-1]):
        ax.plot(i, state_sections[j], color='#ff7f0e')
    ax.plot(omega_sections[-1], state_sections[-1], color='#ff7f0e', label="Robust anticipating $\omega$")


    plt.legend()
    fig.savefig(f'{DIR_FIGURES}/fig-application-replacement-thresholds')
    

def get_performance_decision_rules():
    df_file = "../pre_processed_data/df_trans_99_ev_0.0.pkl"
    df_alt_file = "../pre_processed_data/evs_convergence.pkl"
    fixp_point_dict = "../pre_processed_data/results_1000_10_10.pkl"

    v_disc_ml, v_exp_ml, v_exp_worst, periods = _convergence_plot(df_file, df_alt_file, fixp_point_dict)

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
    fig.savefig(f'{DIR_FIGURES}/fig-application-performance-decision-rules')


def get_performance():

    
    num_keys = 12

    omega_range, means_ml, omega_sections, state_sections = _threshold_plot(num_keys)
    
    nominal_subfolder = "nominal_strategy/"
    rob_subfolder = "54_strategy/"
    opt_subfolder= "opt_strategy/"

    nominal_costs, opt_costs, robust_54_costs = \
    _performance_plot(nominal_subfolder, opt_subfolder, rob_subfolder)

    fig, ax = plt.subplots(1, 1)

    ax.plot(omega_range, opt_costs, label='Discounted utilities of optimal strategy')
    ax.plot(omega_range, nominal_costs, label='Discounted utilities of nominal strategy')
    ax.plot(omega_range, robust_54_costs, label='Discounted utilities of robust strategy with $\omega = 0.54$')

    formatter = plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x)))
    ax.get_yaxis().set_major_formatter(formatter)

    ax.set_ylim([nominal_costs[-1], nominal_costs[0]])
    ax.set_ylabel(r"Performance")
    ax.set_xlabel(r"$\omega$")

    plt.legend()
    fig.savefig(f'{DIR_FIGURES}/fig-application-performance')