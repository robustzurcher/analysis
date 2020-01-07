import os
import pickle as pkl
from pathlib import Path
from zipfile import ZipFile

import numba
import numpy as np
from ruspy.model_code.cost_functions import cubic_costs
from ruspy.model_code.cost_functions import lin_cost
from ruspy.model_code.cost_functions import quadratic_costs
from ruspy.model_code.cost_functions import sqrt_costs


def get_file(fname):
    if not isinstance(fname, Path):
        fname = Path(fname)

    fname_zip = Path(fname).with_suffix(".zip")
    fname_pkl = Path(fname).with_suffix(".pkl")

    if not os.path.exists(fname_pkl):
        with ZipFile(fname_zip, "r") as zipObj:
            zipObj.extractall(Path(fname).parent)

    return pkl.load(open(fname_pkl, "rb"))


def select_cost_func(key):
    if key == "linear":
        return lin_cost
    elif key == "cubic":
        return cubic_costs
    elif key == "sqrt":
        return sqrt_costs
    elif key == "quad":
        return quadratic_costs
    else:
        raise NotImplementedError("Cost function is not implemented")


def discount_utility(df, num_buses, num_periods, gridsize, beta):
    num_points = int(num_periods / gridsize) + 1
    utilities = df["utilities"].to_numpy(np.float64).reshape(num_buses, num_periods)
    return disc_ut_loop(gridsize, num_buses, num_points, utilities, beta)


@numba.jit(nopython=True)
def disc_ut_loop(gridsize, num_buses, num_points, utilities, beta):
    v_disc = np.zeros(num_points, dtype=numba.float64)
    for point in range(num_points):
        v = 0.0
        for i in range(point * gridsize):
            v += (beta ** i) * np.sum(utilities[:, i])
        v_disc[point] = v / num_buses
    return v_disc
