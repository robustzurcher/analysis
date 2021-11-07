import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate as interp
from config import DATA_DIR
from config import DIR_FIGURES
from figures.global_vals_funcs import COLOR_OPTS
from figures.global_vals_funcs import COST_SCALE
from figures.global_vals_funcs import DISC_FAC
from figures.global_vals_funcs import NUM_STATES
from figures.global_vals_funcs import PARAMS
from figures.global_vals_funcs import SPEC_DICT
from figures.global_vals_funcs import VAL_RESULTS
from figures.global_vals_funcs import VAL_STRATS
from matplotlib import cm as CM
from ruspy.estimation.estimation_transitions import create_transition_matrix
from ruspy.model_code.cost_functions import calc_obs_costs
from ruspy.model_code.cost_functions import lin_cost
from ruspy.model_code.fix_point_alg import calc_fixp
from scipy.signal import savgol_filter


samples = pkl.load(open(f"{DATA_DIR}/samples.pkl", "rb"))
prob_grid = pkl.load(open(f"{DATA_DIR}/grid.pkl", "rb"))
obs_costs = calc_obs_costs(NUM_STATES, lin_cost, PARAMS, COST_SCALE)

# Settings for dataframe
prob_colums = ["p_0", "p_1", "p_2"]
true_prob_comuns = ["p_0_true", "p_1_true", "p_2_true"]
strat_cols = [0.0] + VAL_STRATS


def performance_dataframe():
    df = pd.DataFrame(
        data=None,
        index=pd.MultiIndex.from_product(
            [range(prob_grid.shape[0]), range(samples.shape[1])],
            names=["id_grid", "id_sample"],
        ),
        columns=true_prob_comuns + prob_colums + list(VAL_STRATS) + ["best"],
        dtype=float,
    )

    for id_grid, gridpoint in enumerate(prob_grid):
        for id_sample, sample in enumerate(samples[id_grid, :, :]):
            for id_omega, omega in enumerate(VAL_STRATS):
                fname = f"{VAL_RESULTS}grid_{id_grid}_sample_{id_sample}_{id_omega}.pkl"
                df.loc[(id_grid, id_sample), omega] = pkl.load(open(fname, "rb"))

            df.loc[(id_grid, id_sample), prob_colums] = sample

        trans_mat_org = create_transition_matrix(NUM_STATES, gridpoint)

        ev_ml, _, _ = calc_fixp(trans_mat_org, obs_costs, DISC_FAC)

        df.loc[(id_grid, slice(None)), "best"] = ev_ml[0]

        df.loc[(id_grid, slice(None)), true_prob_comuns] = gridpoint
    return df


def print_generate_rankings(df_in, strategies, measures):
    df = df_in[strategies].mean(level=0)

    best_result = df_in.loc[(slice(None), 0), "best"].to_numpy()

    subjective_bayes_order = (
        df.mean(axis=0).sort_values(ascending=False).index.to_numpy()
    )
    print(df.mean(axis=0).sort_values(ascending=False))

    max_min_order = df.min().sort_values(ascending=False).index.to_numpy()
    print(df.min().sort_values(ascending=False))

    min_max_regret = (
        (df.subtract(best_result, axis=0) * -1)
        .max()
        .sort_values(ascending=True)
        .index.to_numpy()
    )
    print((df.subtract(best_result, axis=0) * -1).max().sort_values(ascending=True))

    print("The order of subjective belief is", subjective_bayes_order)

    print("The order of max min is", max_min_order)

    print("The order of min max regret is", min_max_regret)

    df_rank = pd.DataFrame(data=None, index=measures, columns=strategies)

    for measure_name, measure_order in [
        ("Subjective \n Bayes", subjective_bayes_order),
        ("Minimax \n " "regret", min_max_regret),
        ("Maximin", max_min_order),
    ]:
        for pos, strat in enumerate(measure_order):
            df_rank.loc[measure_name, strat] = pos
    return df_rank


def create_ranking_graph(df):
    linestyle = ["--", "-.", "-", ":"]

    for color in COLOR_OPTS:
        fig, ax = plt.subplots(figsize=(8, 4))
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


def plot_performance_difference_matrix(df, val_strat, one_dim_interpol_grid):
    z = generate_plot_matrix(df, val_strat, one_dim_interpol_grid)
    max_scale = len(one_dim_interpol_grid)
    z_2 = np.where(z > 0, 1, z)
    z_2 = np.where(z_2 < 0, -1, z_2)
    z_2 = np.where(z_2 == 0, 2, z_2)
    fig, ax = plt.subplots()
    ax.set_ylim([-0.01, max_scale])
    ax.set_xlim([-0.01, max_scale])
    t1 = plt.Polygon(
        np.array([[0, max_scale], [max_scale, 0]]), closed=False, fill=False
    )
    plt.gca().add_patch(t1)
    ax.set_ylabel(r"10,000 miles")
    ax.set_xlabel(r"5,000 miles")
    # ax.spy(z, origin="lower")
    ticks_space = max_scale / 5
    plt.xticks(
        np.arange(0, max_scale + ticks_space, ticks_space),
        np.arange(0, 1.2, 0.2).round(2),
    )
    plt.yticks(
        np.arange(0, max_scale + ticks_space, ticks_space),
        np.arange(0, 1.2, 0.2).round(2),
    )
    cmap = CM.get_cmap("Greys_r", 3)
    # print(vars(cmap))

    plt.imshow(z_2, cmap=cmap, vmax=2.1, vmin=-1.1)
    # plt.colorbar()

    ax.yaxis.get_major_ticks()[0].label1.set_visible(False)

    fig.savefig(f"{DIR_FIGURES}/fig-application-validation-contour-plot-sw")


def generate_plot_matrix(df_in, val_strat, one_dim_interpol_grid):

    eval_omega = [0.0, val_strat]
    df = df_in.loc[:, eval_omega].mean(level=0)
    interpol_grid = create_prob_grid(one_dim_interpol_grid, 3)
    interpol_res = {}
    for omega in eval_omega:
        interpol_res[omega] = interp.griddata(
            prob_grid[:, :2], df.loc[:, omega], interpol_grid[:, :2], method="linear"
        )
    z = np.zeros((len(one_dim_interpol_grid), len(one_dim_interpol_grid)), dtype=float)
    for id_prob, gridpoint in enumerate(interpol_grid):

        id_z_x = np.where(np.isclose(one_dim_interpol_grid, gridpoint[0]))[0][0]
        id_z_y = np.where(np.isclose(one_dim_interpol_grid, gridpoint[1]))[0][0]

        z[id_z_y, id_z_x] = (
            interpol_res[0.0][id_prob] - interpol_res[val_strat][id_prob]
        )
    return z


def create_prob_grid(one_dim_grid, num_dims):
    # The last column will be added in the end
    num_dims -= 1
    grid = np.array(np.meshgrid(*[one_dim_grid] * num_dims)).T.reshape(-1, num_dims)
    # Delete points which has probability larger than one
    grid = grid[np.sum(grid, axis=1) < 1]
    # Add last column
    grid = np.append(grid, (1 - np.sum(grid, axis=1)).reshape(len(grid), 1), axis=1)
    return grid


def filter_normalize(performance):
    filtered = savgol_filter(performance, 37, 3)
    moved = filtered - np.min(filtered)
    return moved / np.max(moved)


def get_optimal_omega_maximin(df):
    df_meaned = df[list(VAL_STRATS)].mean(level=0)
    omega_vals = df_meaned.loc[:, list(VAL_STRATS)].to_numpy()
    omega_eval_grid = np.arange(0, 1, 0.01)
    omega_interpol_vals = np.zeros((prob_grid.shape[0], len(omega_eval_grid)))
    for id_grid, _ in enumerate(prob_grid):
        omega_interpol_vals[id_grid, :] = interp.griddata(
            VAL_STRATS, omega_vals[id_grid, :], omega_eval_grid, method="linear"
        )

    omegas_min = np.min(omega_interpol_vals, axis=0)
    normalized_min = filter_normalize(omegas_min)
    omega_max_min = omega_eval_grid[np.argmax(normalized_min)]
    for color in COLOR_OPTS:
        fig, ax = plt.subplots(1, 1)
        ax.plot(
            omega_eval_grid,
            normalized_min,
            color=SPEC_DICT[color]["colors"][0],
            label="minimum performance",
        )
        plt.vlines(
            omega_max_min,
            ymin=0,
            ymax=np.max(normalized_min),
            linestyle="dashed",
            color="k",
        )

        ax.set_ylabel(r"Normalized performance")
        ax.set_xlabel(r"$\omega$")
        # ax.legend()
        x = np.arange(0, 1.2, 0.2).round(1)
        x_labels = [str(num) for num in list(x)]
        x_labels[2] = r"$\omega^*$"
        x[x == 0.4] = omega_max_min
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.tick_params(axis="x", which="major", pad=5)
        ax.tick_params(axis="x", which="minor", pad=0.5)
        ax.set_ylim([0, 1.01])
        ax.set_xlim(right=1)
        fig.savefig(
            f"{DIR_FIGURES}/fig-application-validation-optimal-omega"
            f"{SPEC_DICT[color]['file']}"
        )
