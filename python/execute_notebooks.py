#!/usr/bin/env python
"""This module executes all notebooks. It serves the main purpose to ensure that all can be
executed and work proper independently."""
import subprocess as sp
import glob
import os

os.chdir(os.environ['PROJECT_ROOT'] + '/notebooks')

for notebook in sorted(glob.glob('*.ipynb')):
    
    # TODO: The others are not yet ready for this.
    if notebook not in ['00_illustrations.ipynb', 'ivestigation.ipynb']:
        continue

    
    cmd = ' jupyter nbconvert --execute {}  --ExecutePreprocessor.timeout=-1'.format(notebook)
    sp.check_call(cmd, shell=True)

