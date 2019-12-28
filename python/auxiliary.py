import os
import pickle as pkl
from pathlib import Path
from zipfile import ZipFile
from ruspy.model_code.cost_functions import lin_cost, cubic_costs, sqrt_costs


def get_file(fname):
    if not isinstance(fname, Path):
        fname = Path(fname)

    fname_zip = Path(fname).with_suffix('.zip')
    fname_pkl = Path(fname).with_suffix('.pkl')

    if not os.path.exists(fname_pkl):
        with ZipFile(fname_zip, 'r') as zipObj:
            zipObj.extractall(Path(fname).parent)

    return pkl.load(open(fname_pkl, 'rb'))


def select_cost_func(key):
    if key == "linear":
        return lin_cost
    elif key == "cubic":
        return cubic_costs
    elif key == "sqrt":
        return sqrt_costs
    else:
        raise NotImplementedError("Cost function is not implemented")

