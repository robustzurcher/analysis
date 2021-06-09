from IPython import get_ipython

ipython = get_ipython()

ipython.magic("matplotlib inline")
ipython.magic("load_ext autoreload")
ipython.magic("autoreload 2")

import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from figures_application_scripts.transition_probabilities import *
from global_vals_funcs import *
from figures_application_scripts.maintenace_probabilities import *
from figures_application_scripts.demonstration import *
from figures_application_scripts.threshold_plot import *
from figures_application_scripts.performance_plots import *
from figures_application_scripts.validation import *
from figures_introduction import *
from urn_illustrations import *


extract_zips()


DIR_FIGURES = os.environ["DIR_FIGURES"]
