#!/usr/bin/env python
"""This module executes all notebooks. It serves the main purpose to ensure that all can be
executed and work proper independently."""
import glob
import os
import subprocess as sp

os.chdir(os.environ["PROJECT_ROOT"] + "/notebooks")

for notebook in sorted(glob.glob("*.ipynb")):
    cmd = (
        f" jupyter nbconvert --to notebook --execute {notebook}"
        f"  --ExecutePreprocessor.timeout=-1"
    )
    sp.check_call(cmd, shell=True)

for convert_notebook in sorted(glob.glob("*nbconvert.ipynb")):

    cmd(f"os.remove {convert_notebook}")

    sp.check_call(cmd, shell=True)
