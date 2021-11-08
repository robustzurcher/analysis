from IPython import get_ipython

ipython = get_ipython()

ipython.magic("matplotlib inline")
ipython.magic("load_ext autoreload")
ipython.magic("autoreload 2")

import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from scripts_figures.ex_post.transition_probabilities import *
from scripts_figures.global_vals_funcs import *
from scripts_figures.ex_post.maintenace_probabilities import *
from scripts_figures.ex_post.demonstration import *
from scripts_figures.ex_post.threshold_plot import *
from scripts_figures.ex_post.performance_plots import *
from scripts_figures.ex_post.observations import *
from scripts_figures.ex_ante import *

from scripts_figures.introduction import *
from scripts_figures.urn_illustrations import *


extract_zips()


DIR_FIGURES = os.environ["DIR_FIGURES"]
