import numpy as np
from zipfile import ZipFile
from pathlib import Path
import pickle as pkl
import os
import glob
from ruspy.estimation.estimation_cost_parameters import lin_cost, cost_func, choice_prob
from ruspy.simulation.value_zero import discount_utility, calc_ev_0

# Global variables
beta = 0.9999
params = np.array([10, 10])
num_buses = 200
num_periods = 70000
gridsize = 1000
num_points = int(num_periods/gridsize) + 1


def get_file(fname):
    if not isinstance(fname, Path):
        fname = Path(fname)

    fname_zip = Path(fname).with_suffix('.zip')
    fname_pkl = Path(fname).with_suffix('.pkl')

    if not os.path.exists(fname_pkl):
        with ZipFile(fname_zip, 'r') as zipObj:
            zipObj.extractall(Path(fname).parent)

    return pkl.load(open(fname_pkl, 'rb'))


def create_repl_prob_plot(file, keys):
    dict_policies = get_file(file)
    ev_ml = dict_policies[0.0][0]
    num_states = ev_ml.shape[0]
    costs = cost_func(num_states, lin_cost, params)
    choice_ml = choice_prob(ev_ml, costs, beta)
    choices = []
    for omega in keys[1:]:
        choices += [choice_prob(dict_policies[omega][0], costs, beta)]
    return choice_ml, choices


def create_sections(mean_disc, om_range):
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


def threshold_plot(num_keys):
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
    omega_sections, state_sections = create_sections(means_discrete, omega_range)
    return omega_range, means_ml, omega_sections, state_sections


def convergence_plot(df_file, df_alt_file, fixp_point_dict):
    dict_policies = get_file(fixp_point_dict)
    ev_ml = dict_policies[0.0][0]
    ev_worst = dict_policies[0.99][0]
    try:
        df_trans_99_ev_ml = pkl.load(
            open(df_file, "rb"))
        # Calculate the expected value at time zero
        v_exp_ml = np.full(num_points, calc_ev_0(df_trans_99_ev_ml, ev_ml))
        v_exp_worst = np.full(num_points, calc_ev_0(df_trans_99_ev_ml, ev_worst))
        # Calculate the value at time 0 by discounting the utility
        v_disc_ml = discount_utility(df_trans_99_ev_ml, gridsize, beta)
    except:
        v_exp_ml, v_exp_worst, v_disc_ml = pkl.load(open(df_alt_file, "rb"))
    # Create a numpy array of the periods for plotting
    periods = np.arange(0, num_periods + gridsize, gridsize)

    return v_disc_ml, v_exp_ml, v_exp_worst, periods


def performance_plot(nominal_sub, opt_sub, rob_sub):
    strategy_folder = "../pre_processed_data/strategies/"

    file_list = sorted(
        glob.glob(strategy_folder + nominal_sub + "df*"))
    if len(file_list) != 0:
        nominal_costs = np.zeros(len(file_list))
        for j, file in enumerate(file_list):
            df = pkl.load(open(file, "rb"))
            nominal_costs[j] = discount_utility(df, gridsize, beta)[-1]
    else:
        nominal_costs = pkl.load(open(strategy_folder + nominal_sub + "nominal_costs.pkl", "rb"))

    file_list = sorted(glob.glob(strategy_folder + opt_sub + "df*"))
    if len(file_list) != 0:
        opt_costs = np.zeros(len(file_list))
        for j, file in enumerate(file_list):
            df = pkl.load(open(file, "rb"))
            opt_costs[j] = discount_utility(df, gridsize, beta)[-1]
    else:
        opt_costs = pkl.load(
            open(strategy_folder + opt_sub + "opt_costs.pkl", "rb"))

    file_list = sorted(glob.glob(strategy_folder + rob_sub + "df*"))
    if len(file_list) != 0:
        robust_costs = np.zeros(len(file_list))
        for j, file in enumerate(file_list):
            df = pkl.load(open(file, "rb"))
            robust_costs[j] = discount_utility(df, gridsize, beta)[-1]
    else:
        robust_costs = pkl.load(open(glob.glob(strategy_folder + rob_sub +
                                                  "*.pkl")[0], "rb"))
    return nominal_costs, opt_costs, robust_costs
