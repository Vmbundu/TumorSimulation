import variables
import numpy as np
import math
def Spreadhotpoint(magnitude):
# This function is performed to generate a bunch of sprouting stimulus points spread in the 3D computational domain, the density of those sprouting
# stimulus points are calculated according to the concentration of VEGF at each given grid.
    variables.hotpoint = np.zeros([1,variables.wlen])   # Reset hotpoints
    posi = magnitude*1.3**math.log(variables.TAF)
    index = np.where(np.random.rand(variables.wlen,1))
    variables.hotpoint[index] = 1
