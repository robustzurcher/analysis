import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config import DIR_FIGURES
from global_vals_funcs import COLOR_OPTS
from global_vals_funcs import COST_SCALE
from global_vals_funcs import DISC_FAC
from global_vals_funcs import NUM_STATES
from global_vals_funcs import PARAMS
from global_vals_funcs import SPEC_DICT
from global_vals_funcs import VAL_RESULTS
from global_vals_funcs import VAL_STRATS
from matplotlib import cm as CM
from ruspy.estimation.estimation_transitions import create_transition_matrix
from ruspy.model_code.cost_functions import calc_obs_costs
from ruspy.model_code.cost_functions import lin_cost
from ruspy.model_code.fix_point_alg import calc_fixp


GRIDSIZE_PROBS = 0.1
vectors = []
x = y = np.arange(0.01, 1, GRIDSIZE_PROBS)
for p_1 in x:
    for p_2 in y:
        if p_1 == 0 or p_2 == 0:
            pass
        elif p_1 + p_2 >= 1:
            pass
        else:
            vectors += [[p_1, p_2, 1 - p_1 - p_2]]
prob_vectors = np.array(vectors).round(2)
obs_costs = calc_obs_costs(NUM_STATES, lin_cost, PARAMS, COST_SCALE)

# Settings for dataframe
prob_colums = ["p_0", "p_1", "p_2"]
true_prob_comuns = ["p_0_true", "p_1_true", "p_2_true"]
strat_cols = [0.0] + VAL_STRATS


def create_ranking_graph(df):
    linestyle = ["--", "-.", "-", ":"]

    for color in COLOR_OPTS:
        fig, ax = plt.subplots()
        ax.spines["bottom"].set_color("white")
        ax.spines["left"].set_color("white")
        for i, col in enumerate(df.columns):
            if np.isclose(col, 0.0):
                label = "as-if"
            else:
                label = r"robust ($\omega$ = " + f"{col})"

            ax.plot(
                df[col].index,
                df[col].values,
                marker="o",
                linestyle=linestyle[i],
                linewidth=3,
                markersize=20,
                color=SPEC_DICT[color]["colors"][i],
                label=label,
            )

            # df[col].plot(**kwargs)
            # Flip y-axis.
            ax.set_xlim([-0.2, 2.1])
            ax.set_ylim([3.2, -0.8])

            plt.yticks(
                [0, 1, 2, 3],
                labels=["Rank 1", "Rank 2", "Rank 3", "Rank 4"],
            )
            plt.xticks(
                [0, 1, 2],
                labels=df.index.to_list(),
            )
            plt.xlabel("")
            ax.tick_params(axis="both", color="white", pad=20)
            ax.legend(
                markerscale=0.2,
                handlelength=1.5,
                # bbox_to_anchor=[-0.1, 1.1],
                loc="upper center",
                ncol=4,
            )
        fig.savefig(
            f"{DIR_FIGURES}/fig-application-validation-"
            f"ranking-plot{SPEC_DICT[color]['file']}.png"
        )


def print_generate_rankings(df_in, strategies, measures):
    df = df_in.loc[(slice(None), "average_three_dim"), :][strategies]

    best_result = df_in.loc[(slice(None), "average_three_dim"), "best"].to_numpy()

    subjective_bayes_order = (
        df.mean(axis=0).sort_values(ascending=False).index.to_numpy()
    )

    min_max_order = df.min().sort_values(ascending=False).index.to_numpy()

    max_min_regret = (
        (df.subtract(best_result, axis=0))
        .min()
        .sort_values(ascending=False)
        .index.to_numpy()
    )

    print("The order of subjective belief is", subjective_bayes_order)

    print("The order of min max is", min_max_order)

    print("The order of max min regret is", max_min_regret)

    df_rank = pd.DataFrame(data=None, index=measures, columns=strategies)

    for measure_name, measure_order in [
        ("Subjective \n Bayes", subjective_bayes_order),
        ("Minimax \n " "regret", max_min_regret),
        ("Maximin", min_max_order),
    ]:
        for pos, strat in enumerate(measure_order):
            df_rank.loc[measure_name, strat] = pos
    return df_rank


def performance_dataframe():
    df = pd.DataFrame(
        data=None,
        index=pd.MultiIndex.from_product(
            [range(prob_vectors.shape[0]), list(range(100)) + ["average_three_dim"]],
            names=["grid_id", "run"],
        ),
        columns=true_prob_comuns + prob_colums + strat_cols + ["best"],
    )

    for id_prob, true_prob in enumerate(prob_vectors):
        result = pkl.load(open(f"{VAL_RESULTS}result_{id_prob}.pkl", "rb"))
        i = 0
        for out_tuple in result[1]:
            drawn_prob, performances = out_tuple
            if (drawn_prob > 0).sum() == 3:
                i += 1
        performances_total = np.zeros((i, len(VAL_STRATS) + 1), dtype=float)
        i = 0
        for run, out_tuple in enumerate(result[1]):
            drawn_prob, performances = out_tuple
            df.loc[(id_prob, run), prob_colums] = drawn_prob
            df.loc[(id_prob, run), strat_cols] = performances
            if (drawn_prob > 0).sum() == 3:
                performances_total[i, :] = performances
                i += 1

        df.loc[(id_prob, "average_three_dim"), strat_cols] = np.mean(
            performances_total, axis=0
        )

        df.loc[(id_prob, slice(None)), true_prob_comuns] = true_prob

        trans_mat_org = create_transition_matrix(NUM_STATES, result[0])

        ev_ml, _, _ = calc_fixp(trans_mat_org, obs_costs, DISC_FAC)

        df.loc[(id_prob, slice(None)), "best"] = ev_ml[0]
    return df


def plot_performance_difference_matrix(val_strat):
    z = generate_plot_matrix(val_strat)
    z_2 = np.where(z == 0, 1e20, z)
    fig, ax = plt.subplots()
    ax.set_ylim([0, 10])
    ax.set_xlim([0, 10])
    t1 = plt.Polygon(np.array([[0, 10], [10, 0]]), closed=False, fill=False)
    plt.gca().add_patch(t1)
    ax.set_ylabel(r"10,000 miles")
    ax.set_xlabel(r"5,000 miles")
    # ax.spy(z, origin="lower")
    plt.xticks(np.arange(0, 12, 2), np.arange(0, 1.2, 0.2).round(2), position=(0, 0))
    plt.yticks(np.arange(0, 12, 2), np.arange(0, 1.2, 0.2).round(2))
    cmap = CM.get_cmap("Greys_r", 10)

    plt.imshow(z_2, cmap=cmap, vmax=10)
    plt.colorbar()

    ax.yaxis.get_major_ticks()[0].label1.set_visible(False)

    fig.savefig(f"{DIR_FIGURES}/fig-application-validation-contour-plot-sw")


def generate_plot_matrix(val_strat):

    id_strat = np.where(np.isclose(val_strat, VAL_STRATS))[0]
    z = np.zeros((len(x), len(y)), dtype=float)
    for id_prob, _ in enumerate(prob_vectors):
        result = pkl.load(open(f"{VAL_RESULTS}result_{id_prob}.pkl", "rb"))
        id_z_x = np.where(np.isclose(x, result[0][0]))[0][0]
        id_z_y = np.where(np.isclose(y, result[0][1]))[0][0]
        i = 0
        for out_tuple in result[1]:
            drawn_prob, performances = out_tuple
            if (drawn_prob > 0).sum() == 3:
                i += 1
        performances_total = np.zeros((i, len(VAL_STRATS) + 1), dtype=float)
        i = 0
        for out_tuple in result[1]:
            drawn_prob, performances = out_tuple
            if (drawn_prob > 0).sum() == 3:
                performances_total[i, :] = performances
                i += 1

        performances = np.mean(performances_total, axis=0)

        z[id_z_y, id_z_x] = performances[0] - performances[id_strat + 1]
    return z
