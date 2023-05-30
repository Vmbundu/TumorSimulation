from asyncore import write
import numpy as np
import math
import copy
from scipy.ndimage import gaussian_filter
import random
from DetectBoundary import DetectBoundary
import Vess
from Bias import Bias
import writetoraw_forAndrea as write
from numpy.linalg import norm
import time
import cProfile
import pstats
from pstats import SortKey
import array as arr
import sys
#Tumor Growth Simulation Example

#Add you phantom file and result_folder for phantom_file and results_folder respective
tumorSim = TumorSim(phantom_file, results_folder, size, time_array, self.seed, [[394,750,1100,2]])
tumorSim.main()
