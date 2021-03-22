#!/usr/bin/env python
"""This module executes all notebooks. It serves the main purpose to ensure that all can be
executed and work proper independently."""
import glob
import os
import subprocess as sp

os.chdir(os.environ["PROJECT_ROOT"] + "/notebooks")

for notebook in sorted(glob.glob("*.ipynb")):

    cmd = f" jupyter nbconvert --execute {notebook}  --ExecutePreprocessor.timeout=-1"
    sp.check_call(cmd, shell=True)
