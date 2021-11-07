from IPython import get_ipython

ipython = get_ipython()

ipython.magic("matplotlib inline")
ipython.magic("load_ext autoreload")
ipython.magic("autoreload 2")

import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from figures.ex_post.transition_probabilities import *
from figures.global_vals_funcs import *
from figures.ex_post.maintenace_probabilities import *
from figures.ex_post.demonstration import *
from figures.ex_post.threshold_plot import *
from figures.ex_post.performance_plots import *
from figures.ex_ante import *

from figures.introduction import *
from figures.urn_illustrations import *


extract_zips()


DIR_FIGURES = os.environ["DIR_FIGURES"]
