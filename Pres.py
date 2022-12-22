import variables
import numpy as np
import math
class Pres:

    def __init__(self,cind, vind, sigma0, amplitude0, pres_bias) -> None:
        self.cind = cind
        self.vind = vind
        self.sigma0 = sigma0
        self.amplitude0 = amplitude0
        self.pres_bias = pres_bias

    
    def CTP(self):
         for tt in range(0, self.cind.size):
            s=self.cind[0,tt]

            # Density as defined by Equations (3c)
            density = np.sum(variables.celltype[0,s+variables.index_bias]>0)/len(variables.index_bias)

            # Euclidean distance in Equation (2), (4a)
            # dist = sum((variables.nod3xyz[pres_bias,0:] - np.tile(variables.nod3xyz[s, 0:],(len(pres_bias),1)))**2) ** .5
            dist = np.array((np.sum((variables.nod3xyz[s + self.pres_bias, 0:] - np.tile(variables.nod3xyz[s, 0:],(len(self.pres_bias),1))) ** 2, axis=1))**.5)

            # Alpha in Equation (4b) is assumed to be constant
            amplitude=self.amplitude0

            # Sigma as defined by Equation (4c)
            sigma = (self.sigma0*density**2)/((density**2)+(0.5**2))+0.05

            # Pressure based on Gaussian function as in Equation (4a)
            variables.pres[0:,s+self.pres_bias] = variables.pres[0:,s+self.pres_bias] + amplitude * math.e ** ((-dist**2)/(2*sigma**2))


    def VTP(self):
         for ss in range(0,self.vind.size):
            v = self.vind[0,ss]
            # Tumor cell density around vessel cell, here I assume vessel
            # only causes additional pressure inside tumor due to membrane
            # effect [Ref]

            density = np.sum(variables.celltype[0,v+variables.index_bias]>0)/len(variables.index_bias)

            # Equation (4b), alpha0 set to be 0.01
            amplitude = (0.01*density**2)/(density**2 + 0.5**2)

            # Euclidean distance in Equation (4a)
            dist = np.array((np.sum((variables.nod3xyz[v + self.pres_bias, 0:] - np.tile(variables.nod3xyz[v, 0:],(len(self.pres_bias),1))) ** 2, axis=1))**.5)

             # Equation (4c)
            sigma = np.array((self.sigma0*density**2)/((density**2)+(0.5**2))+0.05)

             # Add vessel cell induced pressure
            variables.pres[0:,v+self.pres_bias] = variables.pres[0:,v+self.pres_bias] + amplitude * math.e ** ((-dist**2)/(2*sigma**2))


        


            
