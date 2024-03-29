import os
import pickle as pkl
import shutil
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from config import ROOT_DIR


OMEGA_GRID = np.arange(0, 1, 0.01).round(2)
VAL_STRATS = np.arange(0, 1.1, 0.1).round(2)
VAL_STRATS[-1] -= 0.01

NUM_STATES = 60
DISC_FAC = 0.9999
COST_SCALE = 0.001
PARAMS = np.array([50, 400])
NUM_BUSES = 200
BIN_SIZE = 5  # in thousand
NUM_PERIODS = 100_000
FIXP_DICT_4292 = f"{ROOT_DIR}/pre_processed_data/fixp_results_4292.pkl"
FIXP_DICT_2223 = f"{ROOT_DIR}/pre_processed_data/fixp_results_2223.pkl"
SIM_RESULTS = f"{ROOT_DIR}/pre_processed_data/sim_results/"
VAL_RESULTS = f"{ROOT_DIR}/pre_processed_data/val_results/"

COLOR_OPTS = ["colored", "black_white"]

# jet_color_map = [
#     "#1f77b4",
#     "#ff7f0e",
#     "#2ca02c",
#     "#d62728",
#     "#9467bd",
#     "#8c564b",
#     "#e377c2",
#     "#7f7f7f",
#     "#bcbd22",
#     "#17becf",
# ]

SPEC_DICT = {
    "colored": {
        "colors": ["tab:blue", "tab:orange", "tab:green", "tab:red"],
        "line": ["-"] * 3,
        "hatch": [""] * 3,
        "file": "",
    },
    "black_white": {
        "colors": ["#808080", "#d3d3d3", "#A9A9A9", "#C0C0C0", "k"],
        "line": ["-", "--", ":"],
        "hatch": ["", "OOO", "///"],
        "file": "-sw",
    },
}


def extract_zips():
    if os.path.exists(SIM_RESULTS):
        shutil.rmtree(SIM_RESULTS)
    os.makedirs(f"{ROOT_DIR}/pre_processed_data/sim_results")
    ZipFile(f"{ROOT_DIR}/pre_processed_data/simulation_results.zip").extractall(
        SIM_RESULTS
    )

    if os.path.exists(VAL_RESULTS):
        shutil.rmtree(VAL_RESULTS)
    ZipFile(f"{ROOT_DIR}/pre_processed_data/validation_results.zip").extractall(
        VAL_RESULTS
    )


def get_file(fname):
    if not isinstance(fname, Path):
        fname = Path(fname)

    fname_pkl = Path(fname).with_suffix(".pkl")

    if not os.path.exists(fname_pkl):
        ZipFile(f"{ROOT_DIR}/pre_processed_data/solution_results.zip").extractall(
            f"{ROOT_DIR}/pre_processed_data/"
        )

    return pkl.load(open(fname_pkl, "rb"))


DICT_POLICIES_4292 = get_file(FIXP_DICT_4292)
DICT_POLICIES_2223 = get_file(FIXP_DICT_2223)
