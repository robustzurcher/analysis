from IPython import get_ipython
ipython = get_ipython()

ipython.magic('matplotlib inline')
ipython.magic('load_ext autoreload')
ipython.magic('autoreload 2')

import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from figures_application import *

DIR_FIGURES = os.environ['DIR_FIGURES']
