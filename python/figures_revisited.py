import os
import shutil
from zipfile import ZipFile
from config import DIR_FIGURES
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import pickle as pkl
import glob

from ruspy.model_code.cost_functions import lin_cost, calc_obs_costs
from ruspy.estimation.estimation_transitions import create_transition_matrix
from ruspy.model_code.fix_point_alg import calc_fixp

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
num_states = 45
disc_fac = 0.9999
cost_scale = 1e-3
cost_params = [50, 400]
omega_strategies = np.array([0.1, 0.2, 0.3, 0.4, 0.5])


VAL_RESULTS = "../pre_processed_data/validation_results/"


def extract_zips_revisited():
    if os.path.exists(VAL_RESULTS):
        shutil.rmtree(VAL_RESULTS)
    os.makedirs("../pre_processed_data/validation_results")
    ZipFile("../pre_processed_data/validation_results_revisited.zip").extractall(VAL_RESULTS)


################################################################################
#                             Out of sample plot
################################################################################


def get_out_of_sample_diff_to_true(index, bins):
    performance = _out_of_sample()

    obs_costs = calc_obs_costs(num_states, lin_cost, cost_params, cost_scale)  #
    p = np.loadtxt("../pre_processed_data/parameters/rust_trans_probs.txt")
    trans_mat = create_transition_matrix(num_states, p)
    ev, _, _ = calc_fixp(trans_mat, obs_costs, disc_fac)

    perfomance_strategies = np.zeros_like(performance)
    for i, omega in enumerate(omega_strategies):
        perfomance_strategies[:, i] = performance[:, i] - ev[0]

    hist_data = np.histogram(perfomance_strategies[:, index], bins=bins)
    hist_normed = hist_data[0] / sum(hist_data[0])
    x = np.linspace(np.min(hist_data[1]), np.max(hist_data[1]), bins)
    hist_filter = savgol_filter(hist_normed, 29, 3)
    for color in color_opts:
        fig, ax = plt.subplots(1, 1)

        if color == "colored":
            third_color = "#ff7f0e"
        else:
            third_color = spec_dict[color]["colors"][4]

        ax.plot(
            x, hist_filter, color=spec_dict[color]["colors"][0]
        )
        ax.axvline(color=third_color, ls=spec_dict[color]["line"][2])

        ax.set_ylabel(r"Density")
        ax.set_xlabel(r"$\Delta$ Performance")
        ax.set_ylim([0, 0.12])

        # ax.legend()
        fig.savefig(
            "{}/fig-application-validation-to-true{}".format(
                DIR_FIGURES, spec_dict[color]["file"]
            )
        )


def get_out_of_sample_diff(index, bins):
    performance = _out_of_sample()

    perfomance_strategies = np.zeros((performance.shape[0], len(omega_strategies)))
    for i, omega in enumerate(omega_strategies):
        perfomance_strategies[:, i] = performance[:, 1 + i] - performance[:, 0]

    hist_data = np.histogram(perfomance_strategies[:, index], bins=bins)
    hist_normed = hist_data[0] / sum(hist_data[0])
    x = np.linspace(np.min(hist_data[1]), np.max(hist_data[1]), bins)
    hist_filter = savgol_filter(hist_normed, 29, 3)
    for color in color_opts:
        fig, ax = plt.subplots(1, 1)

        if color == "colored":
            third_color = "#ff7f0e"
        else:
            third_color = spec_dict[color]["colors"][4]

        ax.plot(
            x, hist_filter, color=spec_dict[color]["colors"][0]
        )
        ax.axvline(color=third_color, ls=spec_dict[color]["line"][2])

        ax.set_ylabel(r"Density")
        ax.set_xlabel(r"$\Delta$ Performance")
        ax.set_ylim([0, None])

        # ax.legend()
        fig.savefig(
            "{}/fig-application-validation{}".format(
                DIR_FIGURES, spec_dict[color]["file"]
            )
        )


def get_robust_performance_revisited(width):

    performance = _out_of_sample()

    perfomance_strategies = np.zeros(len(omega_strategies))
    for i, omega in enumerate(omega_strategies):
        perfomance_strategies[i] = np.mean(
            performance[:, 1 + i] > performance[:, 0])

    for color in color_opts:
        fig, ax = plt.subplots(1, 1)

        ax.bar(
            omega_strategies,
            perfomance_strategies,
            width,
            color=spec_dict[color]["colors"][0],
            ls=spec_dict[color]["line"][0],
        )

        ax.set_ylabel(r"Share")
        ax.set_xlabel(r"$\omega$")
        ax.set_ylim([0., 0.35])
        plt.xticks(omega_strategies)
        fig.savefig(
            f"{DIR_FIGURES}/fig-application-validation-bar-plot"
            f"-{spec_dict[color]['file']}"
        )


def _out_of_sample():

    file_list = glob.glob(
            VAL_RESULTS
            + "run_*.pkl")
    performance = np.zeros((len(file_list), len(omega_strategies) + 1))

    for j, file in enumerate(file_list):
        performance[j, :] = pkl.load(open(file, "rb"))
    return performance
