import os
import pickle as pkl
from pathlib import Path
from zipfile import ZipFile


def get_file(fname):
    if not isinstance(fname, Path):
        fname = Path(fname)

    fname_zip = Path(fname).with_suffix(".zip")
    fname_pkl = Path(fname).with_suffix(".pkl")

    if not os.path.exists(fname_pkl):
        with ZipFile(fname_zip, "r") as zipObj:
            zipObj.extractall(Path(fname).parent)

    return pkl.load(open(fname_pkl, "rb"))
