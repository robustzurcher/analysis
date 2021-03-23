import glob
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
from global_vals_funcs import VAL_RESULTS
from global_vals_funcs import VAL_STRATS


def create_ranking_graph(df, name):
    colors = ["tab:blue", "tab:orange", "tab:red", "tab:purple"]
    linestyle = ["--", "-.", "-", ":"]
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.spines["bottom"].set_color("white")
    ax.spines["left"].set_color("white")
    for i, col in enumerate(df.columns):
        kwargs = {
            "marker": "o",
            "linestyle": linestyle[i],
            "linewidth": 3,
            "markersize": 35,
            "color": colors[i],
        }

        df[col].plot(**kwargs)
        # Flip y-axis.
        plt.axis([-0.1, 3.1, 3.2, -0.2])

        plt.yticks(
            [0, 1, 2, 3], labels=["Rank 1", "Rank 2", "Rank 3", "Rank 4"], fontsize=14
        )
        plt.xticks(
            [0, 1, 2],
            labels=["Subjective \n Bayes", "Minimax \n regret", "Maximin"],
            fontsize=14,
        )
        plt.xlabel("")
        plt.tick_params(axis="both", color="white", pad=20)
        plt.legend(
            markerscale=0.3,
            labelspacing=0.8,
            handlelength=3,
            bbox_to_anchor=(0.45, 1.2),
            loc="upper center",
            ncol=4,
            fontsize=14,
        )
        fig.savefig(f"performance_graph_{name}.png")


def out_of_sample():

    file_list = glob.glob(VAL_RESULTS + "run_*.pkl")
    performance = np.zeros((len(file_list), len(VAL_STRATS) + 1))

    for j, file in enumerate(file_list):
        performance[j, :] = pkl.load(open(file, "rb"))
    return performance


################################################################################
#                             Out of sample plot
################################################################################
#
#
# def get_out_of_sample_diff_to_true(index, bins):
#     performance = _out_of_sample()
#
#     ev = DICT_POLICIES_4292[0.0][0]
#
#     perfomance_strategies = np.zeros_like(performance)
#     for i in range(len(VAL_STRATS)):
#         perfomance_strategies[:, i] = performance[:, i] - ev[0]
#
#     hist_data = np.histogram(perfomance_strategies[:, index], bins=bins)
#     hist_normed = hist_data[0] / sum(hist_data[0])
#     x = np.linspace(np.min(hist_data[1]), np.max(hist_data[1]), bins)
#     hist_filter = savgol_filter(hist_normed, 29, 3)
#     for color in COLOR_OPTS:
#         fig, ax = plt.subplots(1, 1)
#
#         if color == "colored":
#             third_color = "#ff7f0e"
#         else:
#             third_color = SPEC_DICT[color]["colors"][4]
#
#         ax.plot(x, hist_filter, color=SPEC_DICT[color]["colors"][0])
#         ax.axvline(color=third_color, ls=SPEC_DICT[color]["line"][2])
#
#         ax.set_ylabel(r"Density")
#         ax.set_xlabel(r"$\Delta$ Performance")
#         ax.set_ylim([0, 0.12])
#
#         # ax.legend()
#         fig.savefig(
#             "{}/fig-application-validation-to-true{}".format(
#                 DIR_FIGURES, SPEC_DICT[color]["file"]
#             )
#         )
#
#
# def get_out_of_sample_diff(index, bins):
#     performance = _out_of_sample()
#
#     perfomance_strategies = np.zeros((performance.shape[0], len(omega_strategies)))
#     for i in range(len(VAL_STRATS)):
#         perfomance_strategies[:, i] = performance[:, 1 + i] - performance[:, 0]
#
#     hist_data = np.histogram(perfomance_strategies[:, index], bins=bins)
#     hist_normed = hist_data[0] / sum(hist_data[0])
#     x = np.linspace(np.min(hist_data[1]), np.max(hist_data[1]), bins)
#     hist_filter = savgol_filter(hist_normed, 29, 3)
#     for color in COLOR_OPTS:
#         fig, ax = plt.subplots(1, 1)
#
#         if color == "colored":
#             third_color = "#ff7f0e"
#         else:
#             third_color = SPEC_DICT[color]["colors"][4]
#
#         ax.plot(x, hist_filter, color=SPEC_DICT[color]["colors"][0])
#         ax.axvline(color=third_color, ls=SPEC_DICT[color]["line"][2])
#
#         ax.set_ylabel(r"Density")
#         ax.set_xlabel(r"$\Delta$ Performance")
#         ax.set_ylim([0, None])
#
#         # ax.legend()
#         fig.savefig(
#             "{}/fig-application-validation{}".format(
#                 DIR_FIGURES, SPEC_DICT[color]["file"]
#             )
#         )
#
#
# def get_robust_performance_revisited(width):
#
#     performance = _out_of_sample()
#
#     perfomance_strategies = np.zeros(len(VAL_STRATS))
#     for i in range(len(VAL_STRATS)):
#         perfomance_strategies[i] = np.mean(performance[:, 1 + i] > performance[:, 0])
#
#     for color in COLOR_OPTS:
#         fig, ax = plt.subplots(1, 1)
#
#         ax.bar(
#             VAL_STRATS,
#             perfomance_strategies,
#             width,
#             color=SPEC_DICT[color]["colors"][0],
#             ls=SPEC_DICT[color]["line"][0],
#         )
#
#         ax.set_ylabel(r"Share")
#         ax.set_xlabel(r"$\omega$")
#         ax.set_ylim([0.0, 0.35])
#         plt.xticks(VAL_STRATS)
#         fig.savefig(
#             f"{DIR_FIGURES}/fig-application-validation-bar-plot"
#             f"-{SPEC_DICT[color]['file']}"
#         )
#
#
